#!/usr/bin/env python3
"""Builds a fixed random permutation for de-correlating a pool's row order
from any structure in the source file it was extracted from (e.g. ENHITS's
embeddings were extracted directly from EnamineHTS_scores.csv.gz, which is
sorted by docking score -- if any backbone's extraction uses batch-level
statistics (e.g. batch normalization) over contiguous file chunks, a
molecule's embedding could pick up a faint signal from its batch-neighbors'
average score, and since neighbors share similar scores when the source is
score-sorted, that's a plausible way score information could leak into
embeddings independently of a model's own genuine prediction ability --
inflating apparent recall for a reason unrelated to real method quality).

Also shuffles the (small, ~2M-line) canonical SMILES file directly, since
that's cheap enough to do in one process; shuffle_embeddings.py applies the
SAME saved permutation to each backbone's (large) embeddings.npy separately.

permutation[j] = the ORIGINAL row index that should appear at shuffled row j.
i.e. shuffled_smiles[j] = original_smiles[permutation[j]], and
shuffle_embeddings.py writes shuffled_embeddings[j] = original_embeddings[permutation[j]]
for every backbone, so all files stay mutually row-aligned after shuffling.

Usage
-----
python build_shuffle_permutation.py \\
    --smiles-file /path/to/enhits_large_smiles.txt \\
    --out-dir /path/to/enhits_large_shuffled \\
    --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with open(args.smiles_file) as f:
        smiles = [line.rstrip("\n") for line in f]
    n = len(smiles)
    print(f"[read] {n:,} SMILES from {args.smiles_file} ({time.perf_counter()-t0:.1f}s)")

    perm_path = out_dir / "shuffle_permutation.npy"
    if perm_path.exists():
        permutation = np.load(perm_path)
        if len(permutation) != n:
            raise SystemExit(
                f"{perm_path} exists with {len(permutation):,} entries, "
                f"but --smiles-file has {n:,} rows -- refusing to reuse a "
                f"permutation built for a different-sized pool."
            )
        print(f"[permutation] loaded existing {perm_path} ({n:,} entries) -- reused, not regenerated")
    else:
        permutation = np.random.default_rng(args.seed).permutation(n)
        np.save(perm_path, permutation)
        print(f"[permutation] generated and saved {perm_path} (seed={args.seed}, {n:,} entries)")

    t0 = time.perf_counter()
    out_smiles_path = out_dir / Path(args.smiles_file).name
    with open(out_smiles_path, "w") as f:
        for idx in permutation:
            f.write(smiles[idx])
            f.write("\n")
    print(f"[smiles] wrote shuffled {out_smiles_path} ({time.perf_counter()-t0:.1f}s)")

    # Sanity check: the multiset of SMILES is unchanged, only the order.
    assert sorted(smiles) == sorted(open(out_smiles_path).read().splitlines()), \
        "shuffled SMILES set differs from the original -- something is wrong"
    print("[done] permutation + shuffled SMILES ready; run shuffle_embeddings.py per backbone next")


if __name__ == "__main__":
    main()
