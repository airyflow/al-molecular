#!/usr/bin/env python3
"""MoLFormer embedding extraction, chunked, single-stage.

Simplest of the three backbones: MoLFormer tokenizes SMILES directly (no
RDKit graph or conformer construction at all), so there's nothing to
persist between a preprocessing stage and an embedding stage -- this
script just batches the pool and runs the tokenizer + transformer forward
pass. Measured on GPU (EnamineHTS, 2.1M molecules): 7.8 min total, i.e.
~0.22 ms/molecule -- cheapest of the three backbones by a wide margin.
CPU will be considerably slower per-molecule but this is still the
lightest of the three sharded pipelines.

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks, so no coordination/race condition is
possible during this parallel compute phase. Once every chunk finishes,
run stitch_embedding_chunks.py once (a separate, sequential job) to copy
all chunk files into the final shared (N, D) .npy that
EmbeddingFeaturizer.load() (molpal/featurizer.py) expects.

Usage
-----
python compute_molformer_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --chunk-id 0 --num-chunks 50 \\
    --chunks-dir /path/to/_molformer_chunks
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_ZOO = ROOT / "models"

from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def compute_embeddings_for_chunk(chunk_smiles: list, model_path: Path, batch_size: int = 256) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(DEVICE)
    model.eval()

    amp_dtype = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if DEVICE.type == "cuda" else dict(device_type="cpu", enabled=False)

    parts = []
    with torch.no_grad():
        for i in range(0, len(chunk_smiles), batch_size):
            smi_batch = chunk_smiles[i : i + batch_size]
            inputs = tokenizer(smi_batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.autocast(**autocast_kwargs):
                outputs = model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    emb = outputs.pooler_output.float()
                else:
                    emb = outputs.last_hidden_state.mean(dim=1).float()
            parts.append(emb.cpu().numpy())

    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent molformer_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "molformer", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)
    model_path = Path(args.model_path) if args.model_path else MODEL_ZOO / "molformer"

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(chunk_smiles, model_path, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "molformer", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
