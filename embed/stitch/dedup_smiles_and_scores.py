#!/usr/bin/env python3
"""Applies a precomputed keep-mask (see build_dedup_mask.py) to AmpC's
row-aligned ampc_smiles.txt and ampc_scores.csv.gz, writing deduplicated
copies of both. Row-aligned means row i of the scores file describes the
exact same molecule as row i of the SMILES file (verified directly,
spot-checked against 6 rows) -- so the SAME keep-mask applies to both
files without any string re-matching.

Usage
-----
python dedup_smiles_and_scores.py \\
    --keep-mask /path/to/ampc_dedup_keep_mask.npy \\
    --smiles-file /path/to/ampc_smiles.txt \\
    --scores-file /path/to/ampc_scores.csv.gz \\
    --out-dir /path/to/ampc_99.5M_dedup
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep-mask", required=True)
    p.add_argument("--smiles-file", required=True)
    p.add_argument("--scores-file", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = np.load(args.keep_mask)
    n_keep = int(keep.sum())
    print(f"[keep-mask] {n_keep:,}/{len(keep):,} rows kept")

    t0 = time.perf_counter()
    out_smiles = out_dir / "ampc_smiles.txt"
    written = 0
    with open(args.smiles_file) as fin, open(out_smiles, "w") as fout:
        for i, line in enumerate(fin):
            if keep[i]:
                fout.write(line)
                written += 1
    assert written == n_keep, f"wrote {written:,}, expected {n_keep:,}"
    print(f"[smiles] wrote {written:,} rows -> {out_smiles} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    out_scores = out_dir / "ampc_scores.csv.gz"
    written = 0
    with gzip.open(args.scores_file, "rt") as fin, gzip.open(out_scores, "wt") as fout:
        header = fin.readline()
        fout.write(header)
        for i, line in enumerate(fin):
            if keep[i]:
                fout.write(line)
                written += 1
    assert written == n_keep, f"wrote {written:,}, expected {n_keep:,}"
    print(f"[scores] wrote {written:,} rows -> {out_scores} ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
