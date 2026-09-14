#!/usr/bin/env python3
"""Applies a precomputed keep-mask (see build_dedup_mask.py) to one
backbone's stitched (N, D) embeddings.npy, writing a deduplicated copy
with only the kept rows (in original relative order).

Streams through the source file sequentially in large blocks (same
technique as run_experiment.py's _get_X_cached streaming fix) rather than
fancy-indexing the kept row indices directly against the memmap -- a
fancy-index over ~98.5M scattered-ish indices would revert to the exact
scattered-read cost that fix was built to avoid. Reading everything
sequentially once and filtering in memory is a full linear scan
regardless (there's no way to skip un-kept rows without reading them --
unlike the sparse-target case _get_X_cached handles, here we're keeping
~99% of all rows, so skipping blocks entirely almost never applies), but
a full sequential read of a huge file is still dramatically cheaper than
one scattered read touching the same total row count would be.

Usage
-----
python dedup_embeddings.py --backbone grover3400 --dim 3400 \\
    --keep-mask /path/to/ampc_dedup_keep_mask.npy \\
    --embeddings-path /path/to/embeddings_striped/grover3400_embeddings.npy \\
    --out-path /path/to/ampc_99.5M_dedup/grover3400_embeddings.npy

--resume: if --out-path already exists (e.g. a prior run was killed
partway -- preallocate() creates the file at its FULL final size
up front, zero-filled, so a killed run still looks "complete" by file
size alone), binary-searches the existing file for the last real
(nonzero) row and continues writing from there, instead of re-scanning
rows already written. Safe/idempotent to pass every time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_embedding_store import preallocate, write_slice

READ_BLOCK = 500_000  # rows/block


def _find_resume_point(out_path: str, n_keep: int, kept_orig_idx: np.ndarray) -> tuple[int, int]:
    """Returns (cursor, start_orig_row): cursor = how many dedup rows are
    already correctly written (binary-searched boundary between real,
    nonzero rows and the zero-filled tail preallocate() left); start_orig_row
    = the original file's row index to resume the source scan from."""
    out = np.load(out_path, mmap_mode="r")

    def is_zero_row(i: int) -> bool:
        return bool(np.all(np.asarray(out[i]) == 0))

    if not is_zero_row(0):
        lo = 0
    else:
        return 0, int(kept_orig_idx[0])  # nothing real written yet

    if not is_zero_row(n_keep - 1):
        return n_keep, n_keep  # already fully written

    hi = n_keep - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if is_zero_row(mid):
            hi = mid
        else:
            lo = mid
    cursor = lo + 1  # lo is the last real row; cursor is how many are done
    start_orig_row = int(kept_orig_idx[cursor]) if cursor < n_keep else n_keep
    return cursor, start_orig_row


def dedup(backbone: str, dim: int, keep_mask_path: str, embeddings_path: str, out_path: str, resume: bool) -> None:
    keep = np.load(keep_mask_path)
    n_total = len(keep)
    n_keep = int(keep.sum())

    emb = np.load(embeddings_path, mmap_mode="r")
    if emb.shape != (n_total, dim):
        raise SystemExit(f"{embeddings_path} has shape {emb.shape}, expected ({n_total}, {dim}) -- check --dim")

    cursor = 0
    scan_start = 0
    if resume and Path(out_path).exists():
        kept_orig_idx = np.where(keep)[0]
        cursor, start_orig_row = _find_resume_point(out_path, n_keep, kept_orig_idx)
        if cursor >= n_keep:
            print(f"[{backbone}] already fully written ({cursor:,}/{n_keep:,}) -- nothing to do")
            return
        scan_start = start_orig_row  # no need to block-align: emb[start:end] works for any start
        print(f"[{backbone}] resuming: {cursor:,}/{n_keep:,} rows already written, "
              f"scanning source from row {scan_start:,}")
    else:
        preallocate(out_path, n_molecules=n_keep, dim=dim)

    t0 = time.perf_counter()
    for start in range(scan_start, n_total, READ_BLOCK):
        end = min(start + READ_BLOCK, n_total)
        block = np.asarray(emb[start:end])  # one bulk sequential read
        block_keep = keep[start:end]
        kept_rows = block[block_keep]
        n = kept_rows.shape[0]
        if n:
            write_slice(out_path, cursor, cursor + n, kept_rows)
            cursor += n
        if (start // READ_BLOCK) % 20 == 0:
            print(f"[{backbone}] {end:,}/{n_total:,} rows scanned, {cursor:,}/{n_keep:,} kept "
                  f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)

    assert cursor == n_keep, f"wrote {cursor:,} rows, expected {n_keep:,}"
    print(f"[done] {backbone}: {cursor:,} rows -> {out_path} ({time.perf_counter()-t0:.1f}s total)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--keep-mask", required=True)
    p.add_argument("--embeddings-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    dedup(args.backbone, args.dim, args.keep_mask, args.embeddings_path, args.out_path, args.resume)


if __name__ == "__main__":
    main()
