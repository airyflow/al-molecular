#!/usr/bin/env python3
"""Applies a precomputed permutation (see build_shuffle_permutation.py) to
one backbone's stitched (N, D) embeddings.npy, writing a shuffled copy.

Reads the SOURCE sequentially in large blocks (same technique
dedup_embeddings.py uses, and for the same reason: a memory-mapped read
must complete synchronously before the calling code can proceed, so
touching rows in scattered order is expensive -- confirmed directly,
2026-09-28: an earlier version of this script gathered scattered SOURCE
rows per block and made literally zero progress on even the first block
after ~30 minutes). Writes are scattered instead -- each source block's
rows get written to their (scattered) shuffled destination positions in
one batched fancy-index assignment. This is the cheaper way around:
writes to a memory-mapped file are buffered by the OS page cache and
flushed lazily/asynchronously, unlike reads, which must fault the data in
before the code can use it.

Resume works differently from dedup_embeddings.py's binary-search-the-
output approach, because writes land in scattered (not prefix-sequential)
destination order here -- a separate small progress marker file
(<out_path>.shuffle_progress, one line: the next SOURCE row to scan from)
tracks resume state instead.

Usage
-----
python shuffle_embeddings.py --backbone grover --dim 3400 \\
    --permutation /path/to/shuffle_permutation.npy \\
    --embeddings-path /path/to/enhits_large/embed/grover_embeddings.npy \\
    --out-path /path/to/enhits_large_shuffled/grover_embeddings.npy

--resume: safe/idempotent to pass every time.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_embedding_store import preallocate

READ_BLOCK = 500_000  # source rows/block, sequential


def shuffle(backbone: str, dim: int, permutation_path: str, embeddings_path: str,
            out_path: str, resume: bool) -> None:
    permutation = np.load(permutation_path)
    n_total = len(permutation)
    # inverse[i] = the shuffled destination row that original row i belongs
    # at, i.e. the row j such that permutation[j] == i. Built once (cheap,
    # O(N) scatter), used to route each sequentially-read source block to
    # its scattered destination rows.
    inverse_permutation = np.empty(n_total, dtype=permutation.dtype)
    inverse_permutation[permutation] = np.arange(n_total)

    # Source reads use plain buffered file I/O (seek + read), not
    # mmap_mode="r" slicing -- confirmed directly, 2026-09-29: grover's
    # shuffle sat for 45+ minutes with AveCPU of only 1 second (sstat),
    # meaning the process was blocked on I/O almost the entire time, not
    # memory-constrained (MaxRSS was only ~160MB, ruling that out). A
    # memmap satisfies a "sequential" slice via many small page-fault-
    # triggered reads rather than one large buffered read, which performs
    # far worse than a single read() syscall (with real OS read-ahead) on
    # this network filesystem for grover's much larger per-row size.
    with open(embeddings_path, "rb") as _f:
        _version = np.lib.format.read_magic(_f)
        _shape, _, _dtype = np.lib.format._read_array_header(_f, _version)
        data_offset = _f.tell()
    if _shape != (n_total, dim):
        raise SystemExit(f"{embeddings_path} has shape {_shape}, expected ({n_total}, {dim}) -- check --dim")
    row_nbytes = dim * _dtype.itemsize
    src_file = open(embeddings_path, "rb")

    progress_path = Path(f"{out_path}.shuffle_progress")
    scan_start = 0
    if resume and Path(out_path).exists() and progress_path.exists():
        scan_start = int(progress_path.read_text().strip())
        if scan_start >= n_total:
            print(f"[{backbone}] already fully written ({n_total:,}/{n_total:,}) -- nothing to do")
            return
        print(f"[{backbone}] resuming: source already scanned up to row {scan_start:,}/{n_total:,}")
    else:
        preallocate(out_path, n_molecules=n_total, dim=dim)

    out_mm = np.lib.format.open_memmap(out_path, mode="r+")

    t0 = time.perf_counter()
    for start in range(scan_start, n_total, READ_BLOCK):
        end = min(start + READ_BLOCK, n_total)
        src_file.seek(data_offset + start * row_nbytes)
        raw = src_file.read((end - start) * row_nbytes)  # one buffered read, real OS read-ahead
        block = np.frombuffer(raw, dtype=_dtype).reshape(end - start, dim)
        dest_idx = inverse_permutation[start:end]        # scattered destination rows for this block
        out_mm[dest_idx] = block.astype(out_mm.dtype, copy=False)  # scattered write (OS-buffered)
        out_mm.flush()
        progress_path.write_text(str(end))
        print(f"[{backbone}] {end:,}/{n_total:,} source rows scanned "
              f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
        # Scattered writes touch pages across the WHOLE destination file
        # (unlike a growing contiguous prefix), so resident memory can
        # accumulate across iterations rather than staying bounded to one
        # block's size -- del + gc.collect() gives the allocator an
        # explicit chance to release the block's memory each iteration,
        # rather than relying on refcounting alone under memory pressure.
        del block, raw
        gc.collect()

    src_file.close()
    del out_mm
    progress_path.unlink(missing_ok=True)
    print(f"[done] {backbone}: {n_total:,} rows -> {out_path} ({time.perf_counter()-t0:.1f}s total)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--permutation", required=True)
    p.add_argument("--embeddings-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    shuffle(args.backbone, args.dim, args.permutation, args.embeddings_path, args.out_path, args.resume)


if __name__ == "__main__":
    main()
