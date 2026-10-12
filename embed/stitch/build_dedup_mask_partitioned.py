#!/usr/bin/env python3
"""Builds the same dedup keep-mask as build_dedup_mask.py -- keep the LAST
occurrence of each exact SMILES string -- without holding every SMILES in
one Python dict, so it scales to ~1.3B rows (docs/scaling_to_1.3B_design.md,
Fix 2).

Three stages, each runnable on its own (SLURM) or all at once (--stage all):
  scan    stream the SMILES file once; for each row write a 24-byte record
          (128-bit blake2b hash as two uint64 + row number) into one of
          --num-parts partition files, chosen by the hash's top bits.
  reduce  per partition: sort records by (hash, row) and keep the largest
          row of each equal-hash run. Array-job friendly (--part, or
          SLURM_ARRAY_TASK_ID).
  merge   set keep[row] = True for every partition's kept rows; save.

Equal strings always share a hash, so they always land in the same
partition and the same run. 128 bits makes an accidental collision between
different SMILES negligible (~2.5e-21 at 1.3B rows; 64 bits would be ~5%).

Usage
-----
python build_dedup_mask_partitioned.py --smiles-file d4_smiles.txt \\
    --work-dir /scratch/dedup_work --out-path d4_dedup_keep_mask.npy --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REC = np.dtype([("hi", "<u8"), ("lo", "<u8"), ("row", "<u8")])


def _part_path(work: Path, p: int) -> Path:
    return work / f"part_{p:04d}.bin"


def scan(smiles_file: str, work: Path, num_parts: int) -> int:
    work.mkdir(parents=True, exist_ok=True)
    shift = 64 - (num_parts.bit_length() - 1)
    assert 1 << (num_parts.bit_length() - 1) == num_parts, "--num-parts must be a power of two"
    files = [open(_part_path(work, p), "wb") for p in range(num_parts)]
    t0, row, batch = time.perf_counter(), 0, 0
    with open(smiles_file, "rb") as f:
        while True:
            batch += 1
            lines = f.readlines(64 * 1024 * 1024)
            if not lines:
                break
            m = len(lines)
            recs = np.empty(m, dtype=REC)
            hashes = b"".join(hashlib.blake2b(line.rstrip(b"\r\n"), digest_size=16).digest() for line in lines)
            h = np.frombuffer(hashes, dtype="<u8").reshape(m, 2)
            recs["hi"], recs["lo"] = h[:, 0], h[:, 1]
            recs["row"] = np.arange(row, row + m, dtype=np.uint64)
            part = (recs["hi"] >> np.uint64(shift)).astype(np.int64)
            order = np.argsort(part, kind="stable")
            recs, part = recs[order], part[order]
            bounds = np.searchsorted(part, np.arange(num_parts + 1))
            for p in range(num_parts):
                if bounds[p] < bounds[p + 1]:
                    files[p].write(recs[bounds[p]:bounds[p + 1]].tobytes())
            row += m
            if batch % 20 == 0:
                print(f"[scan] {row:,} rows ({time.perf_counter() - t0:.0f}s)", flush=True)
    for fh in files:
        fh.close()
    (work / "meta.json").write_text(json.dumps({"n_rows": row, "num_parts": num_parts,
                                                "smiles_file": str(smiles_file)}))
    print(f"[scan] done: {row:,} rows in {time.perf_counter() - t0:.0f}s -> {work}", flush=True)
    return row


def reduce_part(work: Path, p: int) -> None:
    recs = np.fromfile(_part_path(work, p), dtype=REC)
    if len(recs):
        order = np.lexsort((recs["row"], recs["lo"], recs["hi"]))
        recs = recs[order]
        last = np.ones(len(recs), dtype=bool)
        last[:-1] = (recs["hi"][1:] != recs["hi"][:-1]) | (recs["lo"][1:] != recs["lo"][:-1])
        keep_rows = recs["row"][last]
    else:
        keep_rows = np.empty(0, dtype=np.uint64)
    np.save(work / f"keep_rows_{p:04d}.npy", keep_rows)
    print(f"[reduce] part {p}: {len(recs):,} rows, {len(keep_rows):,} kept", flush=True)


def merge(work: Path, out_path: str) -> np.ndarray:
    meta = json.loads((work / "meta.json").read_text())
    keep = np.zeros(meta["n_rows"], dtype=bool)
    n_kept = 0
    for p in range(meta["num_parts"]):
        rows = np.load(work / f"keep_rows_{p:04d}.npy")
        keep[rows] = True
        n_kept += len(rows)
    assert keep.sum() == n_kept, "a row was kept by two partitions -- partitioning is broken"
    np.save(out_path, keep)
    print(f"[done] wrote keep-mask ({n_kept:,}/{len(keep):,} kept, "
          f"{len(keep) - n_kept:,} duplicate rows) -> {out_path}", flush=True)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-file")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-path")
    ap.add_argument("--stage", required=True, choices=["scan", "reduce", "merge", "all"])
    ap.add_argument("--num-parts", type=int, default=256)
    ap.add_argument("--part", type=int, default=None, help="reduce: partition id (default: SLURM_ARRAY_TASK_ID, or all)")
    args = ap.parse_args()
    work = Path(args.work_dir)

    if args.stage in ("scan", "all"):
        scan(args.smiles_file, work, args.num_parts)
    if args.stage in ("reduce", "all"):
        num_parts = json.loads((work / "meta.json").read_text())["num_parts"]
        part = args.part if args.part is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
        parts = [int(part)] if part is not None and args.stage == "reduce" else range(num_parts)
        for p in parts:
            reduce_part(work, p)
    if args.stage in ("merge", "all"):
        merge(work, args.out_path)


if __name__ == "__main__":
    main()
