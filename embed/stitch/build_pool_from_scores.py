#!/usr/bin/env python3
"""Turns a downloaded docking-results file into this project's pool format --
deduplicated, shuffled, row-aligned -- in two streaming passes, for pools too
large to shuffle in memory (1.5B rows; build_shuffle_permutation.py holds the
whole SMILES list, which is fine at 2M and impossible here).

Why shuffle at all: published docking-results files are sorted by score, best
first (verified over all 1,559,853,242 rows of the Enamine REAL / HASTEN SurA
file, and true of LSD's screens too). A score-sorted row order leaks the label
into anything that correlates with position -- contiguous extraction batches,
"first N rows" subsets, shard boundaries -- see build_shuffle_permutation.py's
docstring for the batch-statistics version of this. AmpC and D4's own score
files are in random order, so this step was never needed before.

Method (same partition-then-sort shape as build_dedup_mask_partitioned.py):
  pass 1  stream the source; skip rows the keep-mask drops; give each kept row
          a deterministic 64-bit sort key, blake2b(seed || original row), and
          append it to one of --num-buckets files chosen by the key's top bits.
  pass 2  per bucket: sort its rows by key, append them to the outputs.
Equal keys are broken by original row, so the result is reproducible from
--seed alone. Uniform keys over uniform buckets give a uniformly random order.

Outputs, all mutually row-aligned, in --out-dir:
  <prefix>_smiles.txt      SMILES only, one per line
  <prefix>_scores.csv.gz   "smiles,score" (this project's oracle format)
  <prefix>_ids.txt         supplier/ZINC id per row, for provenance
  permutation.npy          uint32: permutation[j] = ORIGINAL row index now at
                           shuffled row j -- the same convention as
                           build_shuffle_permutation.py, so shuffle_embeddings.py
                           can apply it to any file still in source order
                           (e.g. the Uni-Mol conformer chunks).

Input format is auto-detected: comma- or whitespace-separated, with or without
a header, 3 fields (SMILES, id, score). Both the LSD screens
("smiles,zincid,score" with header) and the HASTEN files (whitespace, no
header) parse without flags.

Usage
-----
python build_pool_from_scores.py \\
    --scores-file .../Enamine_REAL_lead-like_SurA_glide_HTVS_docking_scores.csv.gz \\
    --keep-mask .../enamine_dedup_keep_mask.npy \\
    --work-dir .../shuffle_work --out-dir .../dedup --prefix enamine_surA \\
    --stage all --seed 42
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

KEY = struct.Struct("<Q")


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _bucket_path(work: Path, b: int) -> Path:
    return work / f"bucket_{b:04d}.txt"


def _sniff(path: str):
    """Returns (delimiter, has_header). Reads only the first two lines."""
    with _open(path) as f:
        first = f.readline().rstrip("\n")
        second = f.readline().rstrip("\n")
    delim = "," if first.count(",") >= 2 else None  # None = any whitespace
    fields = first.split(delim) if delim else first.split()
    # A header has no parsable score in its last field.
    try:
        float(fields[-1])
        has_header = False
    except (ValueError, IndexError):
        has_header = True
    probe = (second if has_header else first).split(delim) if delim else (second if has_header else first).split()
    if len(probe) != 3:
        raise SystemExit(f"expected 3 fields (smiles, id, score), got {len(probe)}: {probe!r}")
    return delim, has_header


def scan(scores_file: str, keep_mask_path: str, work: Path, num_buckets: int, seed: int):
    delim, has_header = _sniff(scores_file)
    print(f"[scan] delimiter={'comma' if delim else 'whitespace'} header={has_header}", flush=True)
    keep = np.load(keep_mask_path, mmap_mode="r") if keep_mask_path else None
    if keep is not None:
        print(f"[scan] keep-mask: {len(keep):,} rows", flush=True)

    work.mkdir(parents=True, exist_ok=True)
    handles = [open(_bucket_path(work, b), "w") for b in range(num_buckets)]
    shift = 64 - max(1, (num_buckets - 1).bit_length())
    salt = KEY.pack(seed)

    t0 = time.perf_counter()
    n_in = n_kept = 0
    try:
        with _open(scores_file) as f:
            if has_header:
                next(f)
            for line in f:
                row = n_in
                n_in += 1
                if keep is not None and not keep[row]:
                    continue
                parts = line.split(delim) if delim else line.split()
                if len(parts) != 3:
                    continue
                smi, mol_id, score = parts[0], parts[1], parts[2].rstrip("\n")
                key = int.from_bytes(
                    hashlib.blake2b(salt + KEY.pack(row), digest_size=8).digest(), "big")
                b = key >> shift
                handles[b].write(f"{key:016x}\t{row}\t{smi}\t{mol_id}\t{score}\n")
                n_kept += 1
                if n_in % 50_000_000 == 0:
                    print(f"[scan] {n_in:,} rows read, {n_kept:,} kept "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.close()
    if keep is not None and n_in != len(keep):
        raise SystemExit(f"keep-mask is {len(keep):,} rows but file has {n_in:,}")
    (work / "meta.json").write_text(json.dumps(
        {"num_buckets": num_buckets, "n_in": n_in, "n_kept": n_kept, "seed": seed}))
    print(f"[scan] done: {n_in:,} read, {n_kept:,} kept, {n_in - n_kept:,} dropped "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


def merge(work: Path, out_dir: Path, prefix: str):
    meta = json.loads((work / "meta.json").read_text())
    num_buckets, n_kept = meta["num_buckets"], meta["n_kept"]
    out_dir.mkdir(parents=True, exist_ok=True)
    perm = np.empty(n_kept, dtype=np.uint32)
    t0 = time.perf_counter()
    written = 0

    f_smi = open(out_dir / f"{prefix}_smiles.txt", "w")
    f_ids = open(out_dir / f"{prefix}_ids.txt", "w")
    f_sc = gzip.open(out_dir / f"{prefix}_scores.csv.gz", "wt", compresslevel=6)
    try:
        f_sc.write("smiles,score\n")
        for b in range(num_buckets):
            with open(_bucket_path(work, b)) as f:
                rows = f.readlines()
            rows.sort()  # fixed-width hex key first, then row -- a plain string sort
            for line in rows:
                key, row, smi, mol_id, score = line.rstrip("\n").split("\t")
                perm[written] = int(row)
                f_smi.write(smi + "\n")
                f_ids.write(mol_id + "\n")
                f_sc.write(f"{smi},{score}\n")
                written += 1
            del rows
            if (b + 1) % 32 == 0:
                print(f"[merge] bucket {b + 1}/{num_buckets}, {written:,} rows "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        f_smi.close(); f_ids.close(); f_sc.close()

    if written != n_kept:
        raise SystemExit(f"wrote {written:,} rows, expected {n_kept:,}")
    np.save(out_dir / "permutation.npy", perm)
    assert len(np.unique(perm)) == n_kept, "permutation has duplicate rows"
    print(f"[merge] done: {written:,} rows -> {out_dir} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[merge] buckets in {work} can now be deleted", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-file", required=True, help="downloaded docking results (3 fields; .gz ok)")
    p.add_argument("--keep-mask", default=None, help="dedup keep-mask .npy; omit to keep every row")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True, help="output basename, e.g. enamine_surA")
    p.add_argument("--stage", required=True, choices=["scan", "merge", "all"])
    p.add_argument("--num-buckets", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    work, out_dir = Path(args.work_dir), Path(args.out_dir)
    if args.stage in ("scan", "all"):
        scan(args.scores_file, args.keep_mask, work, args.num_buckets, args.seed)
    if args.stage in ("merge", "all"):
        merge(work, out_dir, args.prefix)


if __name__ == "__main__":
    main()
