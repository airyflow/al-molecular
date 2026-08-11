#!/usr/bin/env python3
"""Stitches independent per-chunk embedding files (written by
compute_grover_embeddings_chunk.py / compute_molformer_embeddings_chunk.py /
compute_unimol_embeddings_chunk.py) into one final shared (N, D) .npy file.

This is a SEQUENTIAL, single-process step, run once after all of a
backbone's parallel chunk tasks finish -- deliberately not parallelized,
because that's exactly what makes it race-free: only one process ever
calls shared_embedding_store.preallocate() (no risk of two array tasks
racing to create/truncate the same file), and each chunk is copied in one
at a time, so peak memory is bounded to one chunk's worth of embeddings
(e.g. ~6GB for MoLFormer's largest AmpC chunk), never the whole array.

This is intentionally the ONLY place shared_embedding_store.preallocate()
is called for the AmpC pipeline -- the per-chunk compute scripts never
touch the shared file at all, so there's no coordination needed between
them (see their module docstrings).

Resumable: tracks stitched chunks in a sidecar `.stitch_progress` file next
to --embeddings-path, so a killed/restarted stitch run skips chunks it
already copied rather than re-copying (harmless either way, since
write_slice() would just overwrite the same rows with the same values --
but skipping avoids the wasted I/O).

Usage
-----
python stitch_embedding_chunks.py \\
    --backbone grover --dim 1600 \\
    --chunks-dir /path/to/_grover_chunks --num-chunks 150 \\
    --total-count 99459561 \\
    --embeddings-path /path/to/grover_embeddings.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from generate_unimol_conformers_chunk import _chunk_bounds
from shared_embedding_store import chunk_file_path, preallocate, write_slice


def stitch(
    backbone: str, chunks_dir: str, num_chunks: int, total_count: int,
    embeddings_path: str, dim: int, allow_partial: bool = False,
) -> None:
    chunks_dir = Path(chunks_dir)
    embeddings_path = Path(embeddings_path)

    chunk_paths = [chunk_file_path(chunks_dir, backbone, i) for i in range(num_chunks)]
    missing = [p for p in chunk_paths if not p.exists()]
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)}/{num_chunks} chunk files missing under {chunks_dir} "
            f"(e.g. {missing[0]}). Pass --allow-partial to stitch an incomplete "
            f"set anyway (any un-stitched rows stay at the zero-fill preallocate() left)."
        )

    preallocate(embeddings_path, n_molecules=total_count, dim=dim)

    progress_path = embeddings_path.with_suffix(embeddings_path.suffix + ".stitch_progress")
    done_chunks = set()
    if progress_path.exists():
        done_chunks = {int(x) for x in progress_path.read_text().splitlines() if x.strip()}
        print(f"[resume] {len(done_chunks)}/{num_chunks} chunks already stitched, skipping")

    n_stitched_now = 0
    with open(progress_path, "a") as progress_f:
        for chunk_id in range(num_chunks):
            if chunk_id in done_chunks:
                continue
            chunk_path = chunk_paths[chunk_id]
            if not chunk_path.exists():
                print(f"[skip] {chunk_path} missing (--allow-partial)")
                continue

            start, end = _chunk_bounds(total_count, chunk_id, num_chunks)
            embeddings = np.load(chunk_path)
            if embeddings.shape[0] != end - start:
                raise RuntimeError(
                    f"{chunk_path} has {embeddings.shape[0]:,} rows but chunk {chunk_id} "
                    f"should have {end - start:,} (indices [{start:,}, {end:,})) -- "
                    f"--num-chunks/--total-count mismatch between this stitch run and "
                    f"whatever produced the chunk files?"
                )

            write_slice(embeddings_path, start, end, embeddings)
            progress_f.write(f"{chunk_id}\n")
            progress_f.flush()
            n_stitched_now += 1

    print(f"[done] {n_stitched_now} chunks stitched this run ({len(done_chunks) + n_stitched_now}/{num_chunks} total) -> {embeddings_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["grover", "molformer", "unimol"])
    parser.add_argument("--dim", type=int, required=True, help="Embedding dim: grover=1600, molformer=768, unimol=512")
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--total-count", type=int, required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Stitch even if some chunk files are missing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    stitch(
        args.backbone, args.chunks_dir, args.num_chunks, args.total_count,
        args.embeddings_path, args.dim, allow_partial=args.allow_partial,
    )
    print(f"[elapsed] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
