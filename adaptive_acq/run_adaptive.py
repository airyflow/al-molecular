#!/usr/bin/env python3
"""Run one AL campaign on a real pool (e.g. ENHITS, 2.1M) with a beta
controller. Mirrors run_experiment.py's non-parallel --mode mve path; run
from the repo root:

    python -m adaptive_acq.run_adaptive --dataset ENHITS --surrogate ltall \\
        --backbones molformer grover mhgged smited unimol2 \\
        --init-size 2104 --batch-size 2104 --n-rounds 5 --topk 1000 \\
        --controller thompson --seed 42 --run-dir runs/adaptive/thompson_frac0.001_seed42 --resume

history.json has the same per-round keys as run_experiment.py
(round / n_labeled / best_score / topk_recall / elapsed), so the existing
plots/ scripts can read it, plus reward, beta_mean, per-arm plan and the
controller's context vector.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import run_experiment as exp
from adaptive_acq.cli_common import add_controller_args
from adaptive_acq.controllers import build_controller
from adaptive_acq.explorer import AdaptiveMVEExplorer

BACKBONES = ["molformer", "grover", "grover3400", "unimol", "unimol2", "smited", "mhgged"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ENHITS", choices=list(exp.DATASETS.keys()))
    p.add_argument("--surrogate", default="ltall",
                   help="any run_experiment --surrogate whose predict() returns a real sigma (ltall, single, ...)")
    p.add_argument("--backbones", nargs="+", default=["molformer", "grover", "mhgged", "smited", "unimol2"],
                   choices=BACKBONES)
    p.add_argument("--init-size", type=int, default=2104)
    p.add_argument("--batch-size", type=int, default=2104)
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pool-limit", type=int, default=None, help="smoke testing only")
    p.add_argument("--surrogate-epochs", type=int, default=None)
    p.add_argument("--surrogate-batch", type=int, default=None)
    p.add_argument("--diversity-backbone", default="molformer",
                   help="whose embeddings define similarity for --diversity-lambda (and the batch_redundancy log)")
    add_controller_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    exp.DATASET = args.dataset  # run_experiment's loaders + build_mve read this module global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"\n{'='*60}\n  {args.dataset}  surrogate={args.surrogate}  controller={args.controller}\n"
          f"  device={exp.DEVICE}  seed={args.seed}\n{'='*60}\n")

    oracle = exp.load_oracle()
    backbones = args.backbones if args.surrogate != "single" else args.backbones[:1]
    emb_dict, pool_smiles = exp.load_embeddings(backbones)
    if args.pool_limit is not None:
        pool_smiles = pool_smiles[: args.pool_limit]
        emb_dict = {bb: emb[: args.pool_limit] for bb, emb in emb_dict.items()}
    # Same pre-filter as run_experiment.py's non-parallel MVE branch, so the
    # seeded init draw matches the existing runs/ baselines exactly.
    keep_idx = [i for i, s in enumerate(pool_smiles) if s in oracle]
    emb_dict = {bb: emb[np.array(keep_idx)] for bb, emb in emb_dict.items()}
    pool_smiles = [pool_smiles[i] for i in keep_idx]
    print(f"[pool] {len(pool_smiles):,} molecules with embeddings + oracle scores")

    run_dir = Path(args.run_dir)
    resume_round, resume_scores = exp.find_resume_checkpoint(run_dir) if args.resume else (0, None)

    explorer = AdaptiveMVEExplorer(
        controller=build_controller(args), beta_scale=args.beta_scale,
        diversity_lambda=args.diversity_lambda, shortlist_mult=args.shortlist_mult,
        diversity_backbone=args.diversity_backbone,
        emb_dict=emb_dict, pool_smiles=pool_smiles, oracle=oracle,
        surrogate_type=args.surrogate, backbone=backbones[0],
        init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
        topk=args.topk, run_dir=run_dir, seed=args.seed,
        surrogate_epochs=args.surrogate_epochs, surrogate_batch=args.surrogate_batch,
        resume_scores=resume_scores, resume_round=resume_round,
    )
    history = explorer.run()

    print(f"\n{'Round':>6} {'Labeled':>10} {'Best':>8} {'Recall':>8} {'beta':>6}")
    for r in history:
        print(f"{r['round']:>6} {r['n_labeled']:>10,} {r['best_score']:>8.3f} {r['topk_recall']:>7.1%} "
              f"{r.get('beta_mean', float('nan')):>6.2f}")


if __name__ == "__main__":
    main()
