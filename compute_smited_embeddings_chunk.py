#!/usr/bin/env python3
"""SMI-TED embedding extraction, chunked, single-stage.

Same shape as MoLFormer's extraction script: SMI-TED tokenizes SMILES
directly via its own bundled tokenizer, so there's nothing to persist
between a preprocessing stage and an embedding stage -- batch the pool,
call the model's own .encode(), done.

IBM Research's materials.smi-ted (encoder-decoder transformer, pretrained
on 91M PubChem SMILES / 4B tokens). "Light" variant: n_layer=12, n_head=12,
n_embd=768, max_len=202 tokens, 289M params. Apache 2.0.
https://huggingface.co/ibm-research/materials.smi-ted

Requires IBM's own smi-ted loader/tokenizer code, vendored at
ibm_materials/models/smi_ted/ (cloned from github.com/IBM/materials --
`load_smi_ted` is not a `transformers.AutoModel`-compatible checkpoint, so
it can't be loaded generically). The checkpoint itself is NOT vendored or
pre-downloaded by this script -- `load_smi_ted()` calls
huggingface_hub.hf_hub_download(repo_id="ibm/materials.smi-ted", ...)
internally on first use and caches to the standard HF cache dir, so the
first chunk to run will download it (~1.16GB) and every chunk after reuses
the cache.

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks. Once every chunk finishes, run
stitch_embedding_chunks.py once (a separate, sequential job) to copy all
chunk files into the final shared (N, D) .npy that EmbeddingFeaturizer.load()
(molpal/featurizer.py) expects.

Usage
-----
python compute_smited_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --chunk-id 0 --num-chunks 50 \\
    --chunks-dir /path/to/_smited_chunks
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import time
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ibm_materials" / "models" / "smi_ted"))

from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def compute_embeddings_for_chunk(chunk_smiles: list, batch_size: int = 256) -> np.ndarray:
    # load_smi_ted (vendored source: ibm_materials/models/smi_ted/smi_ted_light/,
    # cloned from github.com/IBM/materials) ignores its folder/ckpt_filename
    # params in the version checked in here -- it always fetches the vocab
    # and checkpoint via huggingface_hub.hf_hub_download(repo_id=
    # "ibm/materials.smi-ted", ...), caching to the standard HF cache dir.
    # No local checkpoint placement needed; --model-path/--ckpt-filename
    # below are accepted for interface symmetry with the other extraction
    # scripts but unused by this loader.
    from smi_ted_light.load import load_smi_ted

    model = load_smi_ted()
    model = model.to(DEVICE)
    model.eval()

    parts = []
    with torch.no_grad():
        for i in range(0, len(chunk_smiles), batch_size):
            smi_batch = chunk_smiles[i : i + batch_size]
            emb = model.encode(smi_batch, return_torch=True)
            parts.append(emb.float().cpu().numpy())

    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent smited_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "smited", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(chunk_smiles, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "smited", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
