#!/usr/bin/env python3
"""Fit a LinUCB prior over beta arms in the simulator (Stage 1, contextual).

The simulator knows every molecule's docking score, so at each round it can
compute the hit rate EVERY arm's batch would have had (counterfactual=True)
and update all arms -- full-information supervised learning of
E[reward | context, arm], much more sample-efficient than bandit feedback.
The trajectory itself follows the arm LinUCB actually picks, so the states
seen in training match what the controller produces at deployment.

Saves a JSON prior for `run_adaptive.py --controller linucb --controller-path`.

    python -m adaptive_acq.train_linucb --tasks sim_tasks/D4_dedup --episodes 200 \\
        --out adaptive_acq/artifacts/linucb_d4.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_acq.acquisition import DEFAULT_ARMS  # noqa: E402
from adaptive_acq.controllers import LinUCB  # noqa: E402
from adaptive_acq.sim_env import SIM_BATCH_FRAC, SIM_TOPK_FRAC, episode_return, load_tasks, run_episode  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--arms", type=float, nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--alpha", type=float, default=1.0, help="exploration bonus during training")
    p.add_argument("--surrogate", default="ridge", choices=["ridge", "mlp"])
    p.add_argument("--batch-frac", type=float, default=SIM_BATCH_FRAC)
    p.add_argument("--topk-frac", type=float, default=SIM_TOPK_FRAC)
    p.add_argument("--beta-scale", default="normalized", choices=["normalized", "raw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tasks = load_tasks(args.tasks)
    rng = np.random.default_rng(args.seed)
    ctrl = LinUCB(arms=args.arms, alpha=args.alpha)
    log = []
    for ep in range(args.episodes):
        task = tasks[rng.integers(len(tasks))]
        s = int(rng.integers(1 << 30))
        h = run_episode(task, ctrl, seed=s, surrogate=args.surrogate, batch_frac=args.batch_frac, topk_frac=args.topk_frac,
                        beta_scale=args.beta_scale, counterfactual=True)
        log.append({"episode": ep, "task": task.name, "final": episode_return(h, "final"),
                    "betas": [x["beta_mean"] for x in h]})
        if ep % 10 == 0 or ep == args.episodes - 1:
            recent = log[-10:]
            print(f"[linucb] ep {ep:4d}  final(last10)={np.mean([r['final'] for r in recent]):.3f}  "
                  f"betas={log[-1]['betas']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctrl.save(out)
    out.with_suffix(".log.json").write_text(json.dumps({"args": vars(args), "episodes": log}, indent=2))
    print(f"[linucb] prior -> {out}")


if __name__ == "__main__":
    main()
