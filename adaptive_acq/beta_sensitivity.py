#!/usr/bin/env python3
"""Analyze the beta-sensitivity experiment (slurm/submit_beta_sensitivity.sh)
and apply its pre-registered decision rule.

    python -m adaptive_acq.beta_sensitivity [--runs runs/adaptive]

Part A: fixed beta vs greedy at each batch fraction (paired by seed).
Part B: one-round deviations (beta at round t only) vs a beta=0 control
        branched from the same greedy checkpoint (paired by seed).
Every comparison: mean paired difference in top-1000 recall at round 5 (and
in mean recall over rounds, "AUC"), with a 95% t-interval over seeds.

Decision rule (fixed before the results): a cell is a GO candidate if its
round-5 gain is >= +0.005 AND its 95% CI excludes 0. With ~20 cells, about
one false positive is expected at this level, so a candidate must then
replicate on 5 new seeds before bandits/PPO are worth building.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (42, 43, 44, 45, 46)
MIN_GAIN = 0.005


def load(d: Path):
    h = d / "history.json"
    if not h.exists():
        return None
    rec = json.loads(h.read_text())
    if len(rec) < 5:
        return None
    r = np.array([x["topk_recall"] for x in rec[:5]])
    return {"r5": r[-1], "auc": r.mean()}


def paired(arm_dirs, ctrl_dirs):
    """Paired diffs over seeds where both runs finished."""
    d5, da = [], []
    for a, c in zip(arm_dirs, ctrl_dirs):
        ra, rc = load(a), load(c)
        if ra and rc:
            d5.append(ra["r5"] - rc["r5"])
            da.append(ra["auc"] - rc["auc"])
    return np.array(d5), np.array(da)


def ci(d):
    if len(d) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def row(label, d5, da, cells):
    if len(d5) == 0:
        print(f"  {label:34s} (no finished pairs yet)")
        return
    lo, hi = ci(d5)
    go = d5.mean() >= MIN_GAIN and lo > 0
    cells.append((label, d5.mean(), lo, hi, len(d5), go))
    print(f"  {label:34s} n={len(d5)}  r5 {d5.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"AUC {da.mean():+.4f}  {'<-- GO candidate' if go else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/adaptive")
    args = p.parse_args()
    R = Path(args.runs)
    B = R / "betasens"
    cells = []

    def greedy(frac, s):
        # 0.1% greedy lives in the original Stage-1 sweep; other fractions in betasens/
        return R / f"fixed_b0_frac{frac}_seed{s}" if frac == "0.001" else B / f"fixed_b0_frac{frac}_seed{s}"

    print("Part A: fixed beta vs greedy (same batch fraction), paired by seed")
    grid = {"0.001": [("0.05", B), ("0.1", B), ("0.2", B), ("0.3", B), ("0.5", R), ("1", R), ("2", R)],
            "0.002": [("0.1", B), ("0.3", B), ("0.5", B)],
            "0.004": [("0.1", B), ("0.3", B), ("0.5", B)]}
    for frac, betas in grid.items():
        for beta, root in betas:
            arm = [root / f"fixed_b{beta}_frac{frac}_seed{s}" for s in SEEDS]
            d5, da = paired(arm, [greedy(frac, s) for s in SEEDS])
            row(f"batch {float(frac) * 100:.1f}%  beta={beta}", d5, da, cells)

    print("\nPart B: beta at round t only vs branched beta=0 control (0.1% batch)")
    for t in range(1, 6):
        for beta in ("0.2", "1.0"):
            arm = [B / f"pulse_t{t}_b{beta}_frac0.001_seed{s}" for s in SEEDS]
            ctrl = [greedy("0.001", s) if t == 1 else B / f"pulse_t{t}_b0_frac0.001_seed{s}" for s in SEEDS]
            d5, da = paired(arm, ctrl)
            row(f"round {t}  beta={beta}", d5, da, cells)

    print("\nRerun noise: branched beta=0 control vs the original greedy run (should be ~0)")
    for t in range(2, 6):
        d5, da = paired([B / f"pulse_t{t}_b0_frac0.001_seed{s}" for s in SEEDS],
                        [greedy("0.001", s) for s in SEEDS])
        if len(d5):
            lo, hi = ci(d5)
            print(f"  control t={t}  n={len(d5)}  r5 {d5.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  sd {d5.std(ddof=1) if len(d5) > 1 else 0:.4f}")

    go = [c for c in cells if c[5]]
    n_total = 7 + 3 + 3 + 10
    print(f"\nFinished cells: {len(cells)}/{n_total}.  Decision rule: gain >= +{MIN_GAIN} at round 5 with 95% CI > 0.")
    if go:
        print("GO candidates (must replicate on 5 new seeds before building controllers):")
        for label, m, lo, hi, n, _ in go:
            print(f"  {label}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})")
    elif len(cells) == n_total:
        print("NO-GO: no fixed beta and no one-round deviation beats greedy by the pre-set margin. "
              "Adapting beta has no measurable headroom on ENHITS/LT-All, so bandits/PPO over beta are not justified here.")
    else:
        print("No GO candidates among finished cells yet; wait for the rest before deciding.")


if __name__ == "__main__":
    main()
