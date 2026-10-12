#!/usr/bin/env python3
"""Compare runs: mean +- std top-k recall per round, grouped across seeds.

Works on both adaptive_acq runs and the existing runs/ baselines (same
history.json keys). Run directories ending in _seed<N> are grouped together.

    python -m adaptive_acq.aggregate runs/adaptive/*frac0.001* runs/ltall_enhits_greedy_frac0.001_seed4* \\
        --csv adaptive_acq/results/frac0.001.csv
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    groups = defaultdict(list)
    for d in map(Path, args.run_dirs):
        h = d / "history.json"
        if not h.exists():
            print(f"[skip] {d} (no history.json -- unfinished?)")
            continue
        groups[re.sub(r"_seed\d+$", "", d.name)].append(json.loads(h.read_text()))

    rows = []
    for name, hists in sorted(groups.items()):
        n_r = min(len(h) for h in hists)
        R = np.array([[r["topk_recall"] for r in h[:n_r]] for h in hists])
        betas = [[r.get("beta_mean") for r in h[:n_r]] for h in hists]
        rows.append((name, len(hists), R.mean(0), R.std(0), betas[0]))

    width = max(len(r[0]) for r in rows)
    print(f"{'run':<{width}}  n  " + "  ".join(f"r{i + 1:<11}" for i in range(len(rows[0][2]))))
    for name, n, m, s, b in sorted(rows, key=lambda r: -r[2][-1]):
        cells = "  ".join(f"{mi:.3f}±{si:.3f}" for mi, si in zip(m, s))
        bs = "" if b[0] is None else "  beta(seed0)=" + ",".join(f"{x:.2f}" for x in b)
        print(f"{name:<{width}}  {n}  {cells}{bs}")

    if args.csv:
        import csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "n_seeds", "round", "recall_mean", "recall_std"])
            for name, n, m, s, _ in rows:
                for i, (mi, si) in enumerate(zip(m, s)):
                    w.writerow([name, n, i + 1, f"{mi:.5f}", f"{si:.5f}"])


if __name__ == "__main__":
    main()
