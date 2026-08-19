#!/usr/bin/env python3
"""MHG-GED embedding extraction, chunked, single-stage.

Same shape as MoLFormer's/SMI-TED's extraction scripts: no separate
preprocessing stage, batch the pool, call the model's own .encode().

IBM Research's materials.mhg-ged (MHG-GNN: GNN encoder + a sequential
decoder constrained by Molecular Hypergraph Grammar, so decoded SMILES are
guaranteed structurally valid). Pretrained on ~1.34M PubChem molecules --
notably smaller than SMI-TED's 91M or GROVER/MoLFormer/Uni-Mol's corpora.
Apache 2.0. https://huggingface.co/ibm-research/materials.mhg-ged

embedding_dim = 1024, confirmed via smoke test (2026-08-18: encode(["CCO",
"c1ccccc1"]) -> shape (2, 1024)), not published on IBM's model card. Not
hardcoded anywhere in this script or downstream regardless
(EmbeddingFeaturizer/EnsembleFusionSurrogate both derive dims from each
backbone's actual embedding array shape at runtime -- see
predict_pool_shard_worker.py's `dims = {k: v.shape[1] ...}`).

Requires IBM's own mhg-ged package, vendored at
ibm_materials/models/mhg_model/ (cloned from github.com/IBM/materials).
The checkpoint is NOT vendored or pre-downloaded by this script --
`load()` calls huggingface_hub.hf_hub_download(repo_id=
"ibm/materials.mhg-ged", filename="pytorch_model.bin") internally on first
use and caches to the standard HF cache dir.

NOTE: PretrainedModelWrapper.encode() (ibm_materials/models/mhg_model/load.py)
returns a List[torch.tensor], one per molecule -- it loops internally via
torch_geometric's from_smiles() rather than doing real batched inference,
so --batch-size here only bounds how many molecules are converted to
list-of-tensors per Python-level chunk, not a GPU batch dimension the way
it is for MoLFormer/SMI-TED.

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks. Once every chunk finishes, run
stitch_embedding_chunks.py once (a separate, sequential job) to copy all
chunk files into the final shared (N, D) .npy that EmbeddingFeaturizer.load()
(molpal/featurizer.py) expects.

Usage
-----
python compute_mhgged_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --chunk-id 0 --num-chunks 50 \\
    --chunks-dir /path/to/_mhgged_chunks
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
sys.path.insert(0, str(ROOT / "ibm_materials" / "models"))

from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def compute_embeddings_for_chunk(chunk_smiles: list, batch_size: int = 256) -> np.ndarray:
    from mhg_model.load import load as load_mhg  # eval() already called inside __init__

    model = load_mhg()
    model = model.to(DEVICE)

    parts = []
    with torch.no_grad():
        for i in range(0, len(chunk_smiles), batch_size):
            smi_batch = chunk_smiles[i : i + batch_size]
            emb_list = model.encode(smi_batch)  # List[torch.tensor], one per molecule
            emb = torch.stack(emb_list).float().cpu().numpy()
            parts.append(emb)

    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent mhgged_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "mhgged", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(chunk_smiles, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "mhgged", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
