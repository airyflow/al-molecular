#!/usr/bin/env python3
"""Uni-Mol2 (1.1B) embedding extraction, chunked, single-stage.

Standalone PyTorch port (unimol2/ package) of the real Uni-Core-based
upstream model -- verified bit-exact against the actual upstream
implementation (installed from source, real checkpoint, same input batch:
max abs diff 0.0, cosine similarity 1.0000 on every test molecule).
See unimol2/model.py's docstring for exactly what was and wasn't ported and
why.

GPU-primary, unlike the other three backbones in this repo (which are
CPU-primary given BigRed200's scarce GPU allocation) -- measured throughput
at 1.1B params is ~1s/molecule on CPU even before batching, i.e. ~24
CPU-days serially for EnamineHTS's 2.1M molecules. This is squarely GPU
territory; falls back to CPU automatically if no GPU is available (e.g.
local smoke-testing) but is not expected to be practical at real dataset
scale there.

Conformer generation (CPU, RDKit) and the model forward pass (GPU) are NOT
overlapped in this version -- each chunk generates its whole batch's
conformers first, then runs the forward pass. Worth revisiting if profiling
shows conformer generation is a significant fraction of wall-clock time at
GPU speed.

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks, so no coordination/race condition is
possible during this parallel compute phase. Once every chunk finishes,
run stitch_embedding_chunks.py once (a separate, sequential job) to copy
all chunk files into the final shared (N, D) .npy that
EmbeddingFeaturizer.load() (molpal/featurizer.py) expects.

Usage
-----
python compute_unimol2_embeddings_chunk.py \\
    --smiles-file /path/to/smiles.txt --total-count 2141500 \\
    --chunk-id 0 --num-chunks 50 \\
    --chunks-dir /path/to/_unimol2_chunks
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

# _chunk_bounds/count_lines/read_smiles_chunk are plain chunk-math/file-reading
# utilities local to this repo (not muben) -- reused as-is. Note importing this
# module has a side effect of putting muben on sys.path (it needs muben itself,
# for an unrelated function this script never calls) -- harmless, but worth
# being precise that it's not this script's own code reaching into muben.
from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file
from unimol2 import build_model_from_checkpoint
from unimol2.data.collate import prepare_batch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def compute_embeddings_for_chunk(
    chunk_smiles: list, batch_size: int = 16, num_conformer_workers: int = 4, timeout_s: int = 30,
) -> np.ndarray:
    model = build_model_from_checkpoint().to(DEVICE)

    parts = []
    for i in range(0, len(chunk_smiles), batch_size):
        smi_batch = chunk_smiles[i : i + batch_size]
        batched_data = prepare_batch(smi_batch, num_workers=num_conformer_workers, timeout_s=timeout_s)
        batched_data = {k: v.to(DEVICE) for k, v in batched_data.items()}
        with torch.no_grad():
            emb = model(batched_data)
        parts.append(emb.float().cpu().numpy())

    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent unimol2_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-conformer-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=30)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "unimol2", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(
        chunk_smiles, batch_size=args.batch_size,
        num_conformer_workers=args.num_conformer_workers, timeout_s=args.timeout_s,
    )
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "unimol2", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
