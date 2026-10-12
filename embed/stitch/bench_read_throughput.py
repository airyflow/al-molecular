#!/usr/bin/env python3
"""Read-only throughput probe for a stitched (N, D) embeddings.npy.

Reads one contiguous row range in READ_BLOCK-row blocks -- the same access
pattern pca_reduce_embeddings.py uses (np.asarray on a memmap slice) or,
with --buffered, plain seek + read -- and prints MB/s per block and overall.
Writes nothing.

Run as a SLURM array: task i reads range i of --num-tasks equal ranges, so
comparing per-task MB/s between a 1-task and an 8-task submission shows
whether concurrent readers on separate nodes share the filesystem's
bandwidth (the open question for running PCA as row-range array tasks).
"""
from __future__ import annotations

import argparse
import os
import socket
import time

import numpy as np

READ_BLOCK = 500_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-tasks", type=int, required=True)
    p.add_argument("--max-gb", type=float, default=20.0, help="Stop after reading this much")
    p.add_argument("--buffered", action="store_true")
    args = p.parse_args()

    mm = np.load(args.path, mmap_mode="r")
    n, d = mm.shape
    row_bytes = d * mm.dtype.itemsize
    chunk = (n + args.num_tasks - 1) // args.num_tasks
    start, end = args.task_id * chunk, min((args.task_id + 1) * chunk, n)
    end = min(end, start + int(args.max_gb * 1e9 // row_bytes))
    with open(args.path, "rb") as f:
        v = np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, v)
        offset = f.tell()

    mode = "buffered" if args.buffered else "memmap"
    print(f"[bench] host={socket.gethostname()} task={args.task_id}/{args.num_tasks} mode={mode} "
          f"rows [{start:,}, {end:,}) of {n:,} x {d}", flush=True)
    total, t_all = 0, time.perf_counter()
    fh = open(args.path, "rb") if args.buffered else None
    for s in range(start, end, READ_BLOCK):
        e = min(s + READ_BLOCK, end)
        t = time.perf_counter()
        if args.buffered:
            fh.seek(offset + s * row_bytes)
            block = np.frombuffer(fh.read((e - s) * row_bytes), dtype=mm.dtype).reshape(e - s, d)
        else:
            block = np.asarray(mm[s:e])
        float(block[::997, 0].sum())
        dt = time.perf_counter() - t
        total += block.nbytes
        print(f"[block] {time.strftime('%H:%M:%S')} rows {s:,}-{e:,}: {block.nbytes/1e6:.0f} MB "
              f"in {dt:.1f}s = {block.nbytes/1e6/dt:.0f} MB/s", flush=True)
        del block
    el = time.perf_counter() - t_all
    print(f"[done] task={args.task_id} read {total/1e9:.2f} GB in {el:.0f}s = {total/1e6/el:.0f} MB/s", flush=True)


if __name__ == "__main__":
    main()
