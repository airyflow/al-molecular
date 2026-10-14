#!/usr/bin/env python3
"""Generic paired-by-seed comparison of AL runs against a control.

Every experiment in this package compares arms to a control that used the
SAME seeded initial draw, so the per-seed difference removes the (large)
variation in starting conditions. This is the reusable version of the
analysis the experiment-specific scripts (beta_sensitivity, surrogate_sweep,
prefilter_real, d4_transfer) each hard-code.

Paths are templates containing {seed}:

    python -m adaptive_acq.paired \\
        --control "runs/adaptive/fixed_b0_frac0.001_seed{seed}" \\
        --arms "runs/adaptive/nbeta0.1_frac0.001_seed{seed}" \\
               "runs/adaptive/nbeta0.3_frac0.001_seed{seed}" \\
        --seeds 42 43 44 45 46

Reports the mean paired difference in top-k recall at the final round with a
95% t-interval, plus a sign test -- which assumes only symmetry, not
normality, and is worth reading when n is small (at n=5 its best possible
two-sided p is 0.0625, so it cannot reach 0.05 on its own).
"""

import argparse
import json
import math
from pathlib import Path


def curve(tpl: str, seed: int):
    f = Path(tpl.format(seed=seed)) / "history.json"
    if not f.exists():
        return None
    h = json.loads(f.read_text())
    return [r["topk_recall"] for r in h] if h else None


def t_crit(df: int) -> float:
    # two-sided 95%
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 19: 2.093}
    if df in table:
        return table[df]
    return min(table.items(), key=lambda kv: abs(kv[0] - df))[1]


def sign_p(k: int, n: int) -> float:
    """Two-sided sign-test p for k of n positive."""
    c = math.comb
    tail = sum(c(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True, help="path template with {seed}")
    p.add_argument("--arms", nargs="+", required=True, help="path templates with {seed}")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--round", type=int, default=0,
                   help="1-indexed round to compare; 0 = the last round present")
    p.add_argument("--label", default=None, help="optional name for the control column")
    args = p.parse_args()

    ctrl = {s: curve(args.control, s) for s in args.seeds}
    have = [s for s in args.seeds if ctrl[s]]
    if not have:
        raise SystemExit(f"no control runs matching {args.control}")
    pick = lambda c: c[args.round - 1] if args.round else c[-1]
    cm = sum(pick(ctrl[s]) for s in have) / len(have)
    print(f"control {args.label or args.control}  n={len(have)}  "
          f"r{args.round or len(ctrl[have[0]])} = {cm:.4f}\n")

    w = max(len(a) for a in args.arms)
    print(f"{'arm':{w}s} {'n':>2}  {'paired diff [95% CI]':28s} {'sign':>9}  abs")
    for tpl in args.arms:
        d, absv = [], []
        for s in have:
            c = curve(tpl, s)
            if c:
                d.append(pick(c) - pick(ctrl[s]))
                absv.append(pick(c))
        if len(d) < 2:
            print(f"{tpl:{w}s}  -  ({len(d)} finished)")
            continue
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        h = t_crit(n - 1) * sd / math.sqrt(n)
        k = sum(1 for x in d if x > 0)
        print(f"{tpl:{w}s} {n:>2}  {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}]   "
              f"{k}/{n} p={sign_p(k, n):.3f}  {sum(absv)/n:.4f}")


if __name__ == "__main__":
    main()
