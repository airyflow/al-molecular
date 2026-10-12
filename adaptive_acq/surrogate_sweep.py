#!/usr/bin/env python3
"""Analyze the surrogate-side recall sweep (slurm/submit_surrogate_sweep.sh).

    python -m adaptive_acq.surrogate_sweep

Each arm is paired per seed against the LT-All greedy control
(runs/adaptive/fixed_b0_frac0.001_seed*), which used the same seeded initial
draw, so the per-seed difference removes the shared init/run noise.

Decision rule (fixed before the results, same as the beta gate): a GO
candidate gains >= +0.005 top-1000 recall at round 5 with a 95% CI excluding
0, and must then replicate on 5 new seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (42, 43, 44, 45, 46)
ARMS = ("m15", "tw0.5", "tw1.0", "m15_tw1.0")
MIN_GAIN = 0.005


def curve(d: Path):
    h = d / "history.json"
    if not h.exists():
        return None
    rec = json.loads(h.read_text())
    if len(rec) < 5:
        return None
    return np.array([x["topk_recall"] for x in rec[:5]])


def ci(d):
    if len(d) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/adaptive")
    args = p.parse_args()
    R = Path(args.runs)
    S = R / "surrsweep"

    ctrl = {s: curve(R / f"fixed_b0_frac0.001_seed{s}") for s in SEEDS}
    have = [s for s in SEEDS if ctrl[s] is not None]
    if not have:
        raise SystemExit(f"no control runs under {R}/fixed_b0_frac0.001_seed* -- nothing to pair against")
    print(f"control: LT-All greedy (M=5, MSE), n={len(have)}  "
          f"r5={np.mean([ctrl[s][-1] for s in have]):.4f}\n")

    go = []
    print(f"{'arm':12s} {'n':>2}  {'r5 diff [95% CI]':28s} {'AUC diff':>9}  r5 abs")
    for arm in ARMS:
        d5, da, abs5 = [], [], []
        for s in have:
            c = curve(S / f"{arm}_frac0.001_seed{s}")
            if c is None:
                continue
            d5.append(c[-1] - ctrl[s][-1])
            da.append(c.mean() - ctrl[s].mean())
            abs5.append(c[-1])
        if not d5:
            print(f"{arm:12s}  -  (no finished runs yet)")
            continue
        d5, da = np.array(d5), np.array(da)
        lo, hi = ci(d5)
        flag = d5.mean() >= MIN_GAIN and lo > 0
        if flag:
            go.append((arm, d5.mean(), lo, hi, len(d5)))
        print(f"{arm:12s} {len(d5):>2}  {d5.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"{da.mean():+.4f}  {np.mean(abs5):.4f}  {'<-- GO candidate' if flag else ''}")

    done = sum(1 for arm in ARMS for s in have if curve(S / f"{arm}_frac0.001_seed{s}") is not None)
    total = len(ARMS) * len(have)
    print(f"\nFinished runs: {done}/{total}.  Rule: >= +{MIN_GAIN} at round 5 with 95% CI > 0.")
    if go:
        print("GO candidates (replicate on 5 new seeds before claiming):")
        for arm, m, lo, hi, n in go:
            print(f"  {arm}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})")
    elif done == total:
        print("NO-GO: neither a larger LT-All ensemble nor a top-weighted loss clears the margin.")
    else:
        print("No GO candidates among finished runs yet.")


if __name__ == "__main__":
    main()
