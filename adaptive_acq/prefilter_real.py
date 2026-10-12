#!/usr/bin/env python3
"""Analyze the real-pool "score once, then refine" runs
(slurm/submit_prefilter_runs.sh).

    python -m adaptive_acq.prefilter_real

Each arm is paired per seed against the full-scoring LT-All greedy control
(runs/adaptive/fixed_b0_frac0.001_seed*), which used the same seeded initial
draw. Unlike the simulator gate, the scoring cost here is MEASURED: each
round's history record carries n_scored / scored_frac.

Decision rule (fixed before the results): this is a NON-INFERIORITY test, not
a search for a gain. An arm is acceptable if the 95% CI of its paired round-5
difference contains 0 and its lower bound is above -0.005 (the margin used for
the beta gate). Report as "equal recall at Nx lower scoring cost".
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (42, 43, 44, 45, 46)
MARGIN = 0.005


def load(d: Path):
    h = d / "history.json"
    if not h.exists():
        return None
    rec = json.loads(h.read_text())
    if len(rec) < 5:
        return None
    return {
        "recall": np.array([x["topk_recall"] for x in rec[:5]]),
        # scored_frac is absent in runs made before the prefilter was added;
        # those scored the whole pool, so fall back to 1.0 per round.
        "passes": float(sum(x.get("scored_frac", 1.0) for x in rec[:5])),
        "sec": float(sum(x["elapsed"] for x in rec[:5])),
    }


def ci(d):
    if len(d) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/adaptive")
    p.add_argument("--arms", nargs="+", default=None,
                   help="arm names under runs/adaptive/prefilter (default: discover them)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args()
    R, P = Path(args.runs), Path(args.runs) / "prefilter"

    ctrl = {s: load(R / f"fixed_b0_frac0.001_seed{s}") for s in args.seeds}
    have = [s for s in args.seeds if ctrl[s] is not None]
    if not have:
        raise SystemExit(f"no control runs under {R}/fixed_b0_frac0.001_seed*")
    print(f"control: LT-All greedy, full scoring, n={len(have)}  "
          f"r5={np.mean([ctrl[s]['recall'][-1] for s in have]):.4f}  "
          f"passes={np.mean([ctrl[s]['passes'] for s in have]):.2f}  "
          f"{np.mean([ctrl[s]['sec'] for s in have]):.0f}s/run\n")

    arms = args.arms
    if arms is None:
        import re
        arms = sorted({re.sub(r"_frac[\d.]+_seed\d+$", "", d.name) for d in P.glob("*_frac*_seed*")
                       if d.is_dir()},
                      key=lambda a: (len(a), a))
    if not arms:
        raise SystemExit(f"no arms found under {P}")

    print(f"{'arm':8s} {'n':>2}  {'r5 diff [95% CI]':30s} {'passes':>7} {'speedup':>8} {'sec':>7} {'verdict'}")
    ok = []
    for name in arms:
        d5, passes, secs = [], [], []
        for s in have:
            r = load(P / f"{name}_frac0.001_seed{s}")
            if r is None:
                continue
            d5.append(r["recall"][-1] - ctrl[s]["recall"][-1])
            passes.append(r["passes"])
            secs.append(r["sec"])
        if not d5:
            print(f"{name:8s}  -  (no finished runs yet)")
            continue
        d5 = np.array(d5)
        lo, hi = ci(d5)
        cp = np.mean([ctrl[s]["passes"] for s in have])
        sp = cp / np.mean(passes)
        verdict = "acceptable" if (lo <= 0 <= hi and lo > -MARGIN) else (
            "worse" if hi < 0 else "inconclusive")
        if verdict == "acceptable":
            ok.append((name, sp, d5.mean()))
        print(f"{name:8s} {len(d5):>2}  {d5.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"{np.mean(passes):>7.2f} {sp:>7.1f}x {np.mean(secs):>7.0f} {verdict}")

    print(f"\nNon-inferiority margin: CI must contain 0 with lower bound > -{MARGIN}.")
    if ok:
        best = max(ok, key=lambda t: t[1])
        print(f"Acceptable: {', '.join(n for n, _, _ in ok)}.  Cheapest: {best[0]} at "
              f"{best[1]:.1f}x fewer pool passes (paired diff {best[2]:+.4f}).")
    print("'passes' = sum of per-round scored fractions, measured from history.json.")


if __name__ == "__main__":
    main()
