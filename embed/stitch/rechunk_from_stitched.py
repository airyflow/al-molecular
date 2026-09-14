#!/usr/bin/env python3
"""Re-splits an already-stitched (N, D) embeddings .npy file into a
different number of chunk files, matching the on-disk chunk-file naming
convention (chunk_file_path()/write_chunk_file() -- same as the raw
compute_<backbone>_embeddings_chunk.py scripts produce).

Why this exists: some backbones (e.g. AmpC's molformer=50 chunks,
unimol=70 chunks) were originally extracted with a different --num-chunks
than the rest (grover3400/mhgged/smited=1000), because their extraction
was cheap/fast enough that fewer, larger chunks made sense at the time.
That's harmless for anything reading the final STITCHED file (nothing
about run_experiment.py/EmbeddingFeaturizer cares how many chunks were
used to build it), but it's inconvenient for any chunk-aligned tooling
that assumes every backbone shares the same chunk boundaries (e.g.
comparing/joining chunk N across backbones directly).

This does NOT re-run the model -- the stitched file's embeddings are
already final and correct; re-chunking is a pure re-slice-and-write of
already-computed rows, so it's cheap (I/O only, no GPU/model needed) even
for a 99M+-row pool.

Usage
-----
python rechunk_from_stitched.py \\
    --backbone molformer --dim 768 \\
    --embeddings-path /path/to/molformer_embeddings.npy \\
    --total-count 99459561 --num-chunks 1000 \\
    --out-chunks-dir /path/to/_molformer_chunks_1000

Verify afterward by re-stitching the new chunks into a scratch file and
diffing against the original (see the module docstring's "Verification"
note below) before deleting any original chunk directory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from smiles_chunking import _chunk_bounds
from shared_embedding_store import chunk_file_path, write_chunk_file


def rechunk(
    backbone: str, embeddings_path: str, total_count: int, num_chunks: int,
    dim: int, out_chunks_dir: str,
) -> None:
    embeddings_path = Path(embeddings_path)
    emb = np.load(embeddings_path, mmap_mode="r")
    if emb.shape != (total_count, dim):
        raise SystemExit(
            f"{embeddings_path} has shape {emb.shape}, expected "
            f"({total_count}, {dim}) -- check --total-count/--dim"
        )

    t0 = time.perf_counter()
    for chunk_id in range(num_chunks):
        out_path = chunk_file_path(out_chunks_dir, backbone, chunk_id)
        if out_path.exists():
            print(f"[skip] {out_path} exists -- chunk {chunk_id} already written")
            continue
        start, end = _chunk_bounds(total_count, chunk_id, num_chunks)
        rows = np.asarray(emb[start:end])  # one bulk sequential read of this slice
        write_chunk_file(out_chunks_dir, backbone, chunk_id, rows)
        print(f"[{chunk_id + 1}/{num_chunks}] wrote {out_path} "
              f"rows=[{start:,}, {end:,})  ({time.perf_counter() - t0:.1f}s elapsed)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--embeddings-path", required=True, help="Existing stitched (N, D) .npy file to re-slice")
    p.add_argument("--total-count", type=int, required=True)
    p.add_argument("--num-chunks", type=int, required=True, help="Target chunk count (e.g. 1000)")
    p.add_argument("--out-chunks-dir", required=True, help="Directory to write the new chunk files into -- "
                   "use a NEW directory, not the original extraction chunk dir, to avoid mixing old/new boundaries")
    args = p.parse_args()

    rechunk(args.backbone, args.embeddings_path, args.total_count, args.num_chunks, args.dim, args.out_chunks_dir)


if __name__ == "__main__":
    main()
