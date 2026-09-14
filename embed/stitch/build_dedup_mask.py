#!/usr/bin/env python3
"""Build a keep-mask for deduplicating AmpC's 99,459,561-molecule pool.

Rule: keep only the LAST occurrence of each duplicate SMILES string --
matches run_experiment.py's load_oracle()/EmbeddingMVEModel.smi2idx, both
of which build a {smiles: ...} dict via dict(zip(...))/enumerate() and are
therefore ALREADY last-write-wins for any duplicate. A duplicate's earlier
occurrence's own oracle score is consequently unreachable through any
existing SMILES-keyed lookup regardless of dedup -- confirmed directly
(row 1,791,990 vs its shard-7 duplicate row 87,029,781, same SMILES,
scores -35.9 vs -35.95: two genuinely independent DOCK3.7 runs, not a
storage artifact) -- so dropping every non-last occurrence changes no
existing behavior, it only removes rows that were already redundant.

Output: a single boolean .npy, shape (N,), True at every row to KEEP.
Reused by dedup_smiles_and_scores.py and dedup_embeddings.py so the
expensive single pass over the SMILES file only happens once.

Usage
-----
python build_dedup_mask.py --smiles-file /path/to/ampc_smiles.txt \\
    --out-path /path/to/ampc_dedup_keep_mask.npy
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def build_keep_mask(smiles_path: str) -> np.ndarray:
    t0 = time.perf_counter()
    last_occurrence: dict = {}
    with open(smiles_path) as f:
        for i, line in enumerate(f):
            last_occurrence[line.rstrip("\n")] = i
    n = i + 1
    print(f"[scan] {n:,} rows, {len(last_occurrence):,} unique SMILES "
          f"({n - len(last_occurrence):,} duplicate rows) in {time.perf_counter()-t0:.1f}s")

    keep = np.zeros(n, dtype=bool)
    keep[np.fromiter(last_occurrence.values(), dtype=np.int64, count=len(last_occurrence))] = True
    assert keep.sum() == len(last_occurrence)
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-file", required=True)
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    keep = build_keep_mask(args.smiles_file)
    np.save(args.out_path, keep)
    print(f"[done] wrote keep-mask ({keep.sum():,}/{len(keep):,} kept) -> {args.out_path}")


if __name__ == "__main__":
    main()
