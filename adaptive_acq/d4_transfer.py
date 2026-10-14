#!/usr/bin/env python3
"""Does the top-weighted loss transfer to a second target?
(slurm/submit_d4_transfer.sh)

    python -m adaptive_acq.d4_transfer

Pairs d4_tw1.0 against its own d4_ctrl per seed on the D4_SUB pool, and prints
the ENHITS result beside it so the two targets can be compared directly.

Unlike the ENHITS experiments this has no pre-existing control -- D4 is a
different target, so arm 0 of the submit script provides it.
"""

import argparse
import json
import math
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
# ENHITS reference, for context (see README): per-seed-set means of the paired
# tw1.0 - control difference at round 5.
ENHITS_REF = {"seeds 42-46": +0.0060, "seeds 47-51": +0.0152, "pooled": +0.0106}


def r5(p: Path):
    f = p / "history.json"
    if not f.exists():
        return None
    h = json.loads(f.read_text())
    return h[4]["topk_recall"] if len(h) >= 5 else None


def stat(d):
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 9: 2.262}.get(n - 1, 2.776)
    h = t * sd / math.sqrt(n)
    return m, m - h, m + h


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/adaptive/d4xfer")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args()
    R = Path(args.runs)

    pairs, ctrl_abs, arm_abs = [], [], []
    for s in args.seeds:
        c = r5(R / f"d4_ctrl_frac0.001_seed{s}")
        a = r5(R / f"d4_tw1.0_frac0.001_seed{s}")
        if c is not None and a is not None:
            pairs.append(a - c)
            ctrl_abs.append(c)
            arm_abs.append(a)

    print("ENHITS reference (tw1.0 - control, round 5):")
    for k, v in ENHITS_REF.items():
        print(f"  {k:14s} {v:+.4f}")

    print(f"\nD4_SUB (2.1M subsample of D4, same protocol):")
    if len(pairs) < 2:
        done = len(pairs)
        print(f"  only {done}/{len(args.seeds)} seed pairs finished -- wait for the rest")
        return
    m, lo, hi = stat(pairs)
    print(f"  control r5 = {sum(ctrl_abs)/len(ctrl_abs):.4f}   tw1.0 r5 = {sum(arm_abs)/len(arm_abs):.4f}")
    print(f"  paired diff = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   n={len(pairs)}   "
          f"positive in {sum(1 for x in pairs if x > 0)}/{len(pairs)}")
    print(f"  per-seed: {[round(x, 4) for x in pairs]}")

    print()
    if lo > 0:
        print("TRANSFERS: the gain is positive on a second target with a CI excluding 0. "
              "The ENHITS result is not target-specific; the full-scale D4 confirmation "
              "(~55 GPU-h) is now worth running.")
    elif hi < 0:
        print("DOES NOT TRANSFER (and hurts here): report the ENHITS gain as target-specific.")
    else:
        print("INCONCLUSIVE at this n: the CI spans 0. Either add seeds, or report the "
              "ENHITS gain as demonstrated on one target only.")


if __name__ == "__main__":
    main()
