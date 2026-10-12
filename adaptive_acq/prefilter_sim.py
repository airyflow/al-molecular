#!/usr/bin/env python3
"""How much top-k recall survives if you only score the pool ONCE?

Motivation (billion-scale). Measured on ENHITS: a round costs ~70 s, of which
only 3-6 s is surrogate training -- the rest is scoring the pool. Extrapolated
to 13B molecules that is ~120 GPU-hours *per round*, so 5 rounds of full
scoring is the dominant cost of a campaign. The oracle budget, by contrast,
does not grow with pool size at all. The beta gate showed greedy is within
0.005 of optimal *given* full scoring, so the question that matters at scale
is not "which molecules do I label" but "how few must I score".

Strategy measured here -- "score once, then refine":
  round 1 : score the whole pool, keep the top S as a shortlist
  rounds 2+: only ever consider that shortlist (minus what is already labeled)

S is set as a MULTIPLE of the total labeling budget, not as a fixed fraction
of the pool, because the two differ between simulator and real runs: the
calibrated simulator labels ~6% of its pool per campaign (batch 1% x 5 rounds
+ init) while real ENHITS labels ~0.6% (batch 0.1%). A multiple of budget is
the scale-free knob; the pass savings it implies at the real ENHITS geometry
are reported separately.

Implemented by Campaign's own prefilter_mult (which restricts the candidate
set), so the simulator and the real 2.1M runs share one code path. Do NOT mark
ineligible candidates by writing -inf into mu: that makes mu.std() nan, hence
effective_beta nan under the default normalized beta scale, which silently
randomizes selection. An earlier version of this script did exactly that and
produced identical recall for every shortlist size; effective_beta now raises
on non-finite input instead.

    python -m adaptive_acq.prefilter_sim --tasks adaptive_acq/sim_tasks/ENHITS
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_acq.controllers import FixedBeta  # noqa: E402
from adaptive_acq.sim_env import (SIM_BATCH_FRAC, SIM_TOPK_FRAC, load_tasks,  # noqa: E402
                                  run_episode)

# Real ENHITS geometry, for translating a budget multiple into pass savings.
REAL_N, REAL_BATCH, REAL_ROUNDS = 2_104_318, 2104, 5


def run(task, mult, seed, batch_frac=SIM_BATCH_FRAC, topk_frac=SIM_TOPK_FRAC, n_rounds=5):
    """One simulated campaign. mult=None means full-pool scoring every round."""
    h = run_episode(task, FixedBeta(0.0), seed=seed, batch_frac=batch_frac, topk_frac=topk_frac,
                    n_rounds=n_rounds, prefilter_mult=(0.0 if mult is None else mult))
    n = len(task)
    batch = max(1, round(batch_frac * n))
    return {"recall": [r["topk_recall"] for r in h],
            "shortlist": n if mult is None else min(n, int(round(mult * batch * (n_rounds + 1)))),
            "sim_passes": float(sum(r["scored_frac"] for r in h)),
            "scored": [r["scored_frac"] for r in h]}


def real_speedup(mult, n_rounds=REAL_ROUNDS):
    """Pass savings this budget multiple would give at the real ENHITS geometry."""
    if mult is None:
        return 1.0, float(n_rounds)
    frac = min(1.0, mult * REAL_BATCH * (n_rounds + 1) / REAL_N)
    passes = 1.0 + (n_rounds - 1) * frac
    return n_rounds / passes, passes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--mults", type=float, nargs="+", default=[1.5, 2, 5, 10],
                   help="shortlist size as a multiple of the total labeling budget")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tasks = load_tasks(args.tasks)
    settings = [None] + list(args.mults)  # None = full-pool scoring every round (control)
    rows = []
    for t in tasks:
        for mult in settings:
            for s in range(args.n_seeds):
                r = run(t, mult, s, n_rounds=args.n_rounds)
                rows.append({"task": t.name, "source": t.source, "mult": mult, "seed": s, **r})
                print(f"  {t.name[:28]:28s} mult={str(mult):>5} seed={s} "
                      f"r5={r['recall'][-1]:.3f} shortlist={r['shortlist']:,} "
                      f"sim_passes={r['sim_passes']:.2f}", flush=True)

    base = {(r["task"], r["seed"]): r for r in rows if r["mult"] is None}
    print(f"\n{'source':16s} {'mult':>5} {'r5':>7} {'paired diff':>12}   "
          f"{'real passes':>11} {'real speedup':>12}")
    for src in sorted({r["source"] for r in rows}):
        for mult in settings:
            rs = [r for r in rows if r["source"] == src and r["mult"] == mult]
            if not rs:
                continue
            r5 = np.mean([r["recall"][-1] for r in rs])
            d = np.array([r["recall"][-1] - base[(r["task"], r["seed"])]["recall"][-1]
                          for r in rs if (r["task"], r["seed"]) in base])
            sp, passes = real_speedup(mult, args.n_rounds)
            print(f"{src:16s} {str(mult):>5} {r5:>7.3f} {d.mean():>+12.4f}   "
                  f"{passes:>11.2f} {sp:>11.1f}x")
    print("\n'real passes/speedup' = what this shortlist size would cost at the real ENHITS\n"
          "geometry (2.1M pool, 2104/batch); recall is measured in the simulator.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
