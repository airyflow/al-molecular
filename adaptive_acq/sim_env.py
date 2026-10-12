#!/usr/bin/env python3
"""Cheap simulated campaigns for training and screening beta controllers.

A full 2.1M-molecule LT-All campaign costs ~10 GPU-minutes; GRPO needs
thousands of campaigns. The simulator keeps the same Campaign loop, but on a
random subsample of a scored pool, PCA-reduced, with a cheap ensemble
surrogate that still produces a real disagreement-based sigma. Its batch and top-k
fractions are calibrated to reproduce real learning curves (see SIM_BATCH_FRAC).

Leakage rule: train controllers on tasks built from a DIFFERENT target than
the one you report (e.g. build from D4_dedup / AmpC_dedup_pca, evaluate on
ENHITS). Controllers only see 10 campaign-summary features, so they can't
memorize molecules, but tuning on the test target's oracle still counts as
test-set tuning. sim_env warns when a task's source matches --eval-dataset.

Subcommands:
  build  -- subsample (random contiguous blocks by default) + PCA a pool into
            cached task .npz files (one-time; loads the full oracle)
  eval   -- run controllers over cached tasks/seeds, write a JSON summary
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_acq.campaign import Campaign  # noqa: E402

ENHITS_TOPK_FRAC = 1000 / 2_104_318

# Simulator protocol, calibrated 2026-10-11 so that greedy's recall curve on a
# 200k D4 task matches the real 116M-molecule D4 LT-All campaign
# (sim 0.12/0.26/0.37/0.45/0.51 vs real 0.11/0.27/0.37/0.45/0.51). Matching the
# real FRACTIONS instead (0.1% batch, 1000/2.1M top-k) left the sim surrogate
# with ~200 labels per round and recall ~0.03-0.06 -- every beta looked alike
# and no controller could learn anything.
SIM_BATCH_FRAC = 0.01
SIM_TOPK_FRAC = 0.0025


# ── tasks ─────────────────────────────────────────────────────────────────────

class SimTask:
    def __init__(self, X, y, name, source):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float64)  # docking score, lower = better
        self.name, self.source = name, source

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["X"], z["y"], name=Path(path).stem, source=str(z["source"]))

    def __len__(self):
        return len(self.y)


def load_tasks(paths):
    out = []
    for p in paths:
        p = Path(p)
        out += [SimTask.load(f) for f in sorted(p.glob("*.npz"))] if p.is_dir() else [SimTask.load(p)]
    if not out:
        raise SystemExit(f"no task .npz found in {paths}")
    return out


def build_tasks(dataset, backbones, n_tasks, n_sub, pca_dim, seed, out_dir, block_size=1000):
    import run_experiment as exp
    from sklearn.decomposition import PCA

    exp.DATASET = dataset
    oracle = exp.load_oracle()
    emb_dict, smiles = exp.load_embeddings(backbones)
    scored_mask = np.fromiter((s in oracle for s in smiles), dtype=bool, count=len(smiles))
    print(f"[build] {dataset}: {scored_mask.sum():,} scored rows; backbones={backbones}; "
          f"block_size={block_size}", flush=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(smiles)
    for t in range(n_tasks):
        t0 = time.perf_counter()
        if block_size > 1:
            # Random CONTIGUOUS blocks: on the 100M-row /N/project pools a
            # scattered row read costs ~19 ms (measured on D4: 200k rows x 4
            # backbones ~ 5 h per task, 20 h for 4 -- past the 8 h job limit),
            # while 1000-row blocks read ~75x faster. Blocks are spread at
            # random over the whole pool, so the task still spans it; rows
            # within a block may be similar (pools can be ordered, e.g. ENHITS
            # by score), which lowers within-task diversity somewhat.
            starts = np.sort(rng.choice(n // block_size, -(-n_sub // block_size), replace=False)) * block_size
            blocks = [np.arange(st, st + block_size) for st in starts]
            keep = [scored_mask[r] for r in blocks]
            idx = np.concatenate([r[k] for r, k in zip(blocks, keep)])
            X = np.concatenate([
                np.concatenate([np.asarray(emb_dict[b][st:st + block_size], dtype=np.float32)[k]
                                for st, k in zip(starts, keep)])
                for b in backbones], axis=1)
        else:
            idx = np.sort(rng.choice(np.flatnonzero(scored_mask), n_sub, replace=False))
            X = np.concatenate([np.asarray(emb_dict[b][idx], dtype=np.float32) for b in backbones], axis=1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X = PCA(n_components=min(pca_dim, X.shape[1]), svd_solver="randomized", random_state=seed + t).fit_transform(X)
        y = np.array([oracle[smiles[i]] for i in idx], dtype=np.float64)
        name = f"{dataset}_{'-'.join(backbones)}_n{n_sub}_pca{pca_dim}_t{t}"
        # Write then rename, so a reader never sees a half-written task file.
        tmp = out_dir / f".{name}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, X=X.astype(np.float32), y=y, rows=idx, source=np.array(dataset))
        tmp.replace(out_dir / f"{name}.npz")
        print(f"[build] {name}  ({time.perf_counter() - t0:.0f}s)", flush=True)


# ── cheap surrogates (both return mu higher=better, sigma = member std) ─────

class BootstrapRidge:
    """M ridge regressors on bootstrap resamples of the labeled set."""

    def __init__(self, M=5, alpha=10.0, seed=0):
        self.M, self.alpha, self.rng = M, alpha, np.random.default_rng(seed)

    def __call__(self, X, labeled_idx, y_hb, cand_idx):
        n, d = len(labeled_idx), X.shape[1]
        preds = np.empty((self.M, len(cand_idx)))
        Xc = X[cand_idx]
        for m in range(self.M):
            b = self.rng.integers(0, n, n)
            Xl, yl = X[labeled_idx[b]].astype(np.float64), y_hb[b]
            xm, ym = Xl.mean(0), yl.mean()
            A = (Xl - xm).T @ (Xl - xm) + self.alpha * np.eye(d)
            w = np.linalg.solve(A, (Xl - xm).T @ (yl - ym))
            preds[m] = (Xc - xm) @ w + ym
        return preds.mean(0), preds.std(0)


class MLPEnsemble:
    """M small MLPs, fresh init each round, MSE on standardized targets --
    a scaled-down stand-in for LTAllSurrogate's ensemble sigma."""

    def __init__(self, M=5, hidden=128, steps=300, lr=1e-3, seed=0):
        self.M, self.hidden, self.steps, self.lr, self.seed = M, hidden, steps, lr, seed
        self.calls = 0

    def __call__(self, X, labeled_idx, y_hb, cand_idx):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed * 100_003 + self.calls)
        self.calls += 1
        ym, ys = y_hb.mean(), y_hb.std() + 1e-8
        Xl = torch.as_tensor(X[labeled_idx])
        yl = torch.as_tensor((y_hb - ym) / ys, dtype=torch.float32)
        Xc = torch.as_tensor(X[cand_idx])
        preds = []
        for _ in range(self.M):
            net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            for _ in range(self.steps):
                opt.zero_grad()
                loss = ((net(Xl).squeeze(-1) - yl) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                preds.append(torch.cat([net(Xc[i:i + 65536]).squeeze(-1) for i in range(0, len(Xc), 65536)]).numpy())
        P = np.stack(preds) * ys + ym
        return P.mean(0), P.std(0)


def make_surrogate(kind, seed):
    if kind == "ridge":
        return BootstrapRidge(seed=seed)
    if kind == "mlp":
        return MLPEnsemble(seed=seed)
    raise ValueError(kind)


# ── episodes ──────────────────────────────────────────────────────────────────

def run_episode(task, controller, seed, surrogate="ridge", batch_frac=SIM_BATCH_FRAC, topk_frac=SIM_TOPK_FRAC,
                n_rounds=5, beta_scale="normalized", counterfactual=False, diversity_lambda=0.0,
                shortlist_mult=10, prefilter_mult=0.0, prefilter_refresh=0):
    """One simulated campaign. Same seed -> same initial labeled set, which is
    what makes a GRPO group (several rollouts from one start) well-defined."""
    n = len(task)
    batch = max(1, round(batch_frac * n))
    k = max(1, round(topk_frac * n))
    true_top = np.zeros(n, dtype=bool)
    true_top[np.argpartition(task.y, k - 1)[:k]] = True
    init = np.random.default_rng(seed).choice(n, batch, replace=False)
    surr = make_surrogate(surrogate, seed)
    camp = Campaign(
        n_pool=n, fit_predict=lambda li, yhb, ci: surr(task.X, li, yhb, ci),
        score_fn=lambda idx: task.y[idx], init_idx=init, batch_size=batch, n_rounds=n_rounds,
        k_obs=k, beta_scale=beta_scale, true_top_mask=true_top, counterfactual=counterfactual,
        diversity_lambda=diversity_lambda, shortlist_mult=shortlist_mult, feats_fn=lambda idx: task.X[idx],
        prefilter_mult=prefilter_mult, prefilter_refresh=prefilter_refresh,
    )
    return camp.run(controller)


def episode_return(history, kind="auc"):
    """auc = mean recall over rounds (rewards finding hits early);
    final = recall after the last round."""
    r = [h["topk_recall"] for h in history]
    return float(np.mean(r)) if kind == "auc" else float(r[-1])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _eval_main(args):
    from adaptive_acq.controllers import build_controller

    tasks = load_tasks(args.tasks)
    if args.eval_dataset:
        clash = sorted({t.source for t in tasks if t.source == args.eval_dataset})
        if clash:
            print(f"[eval] WARNING: tasks built from {clash} == --eval-dataset; results are "
                  f"in-distribution, not a transfer estimate")
    rows = []
    for t in tasks:
        for s in range(args.n_seeds):
            ctrl = build_controller(argparse.Namespace(**{**vars(args), "seed": s}))
            h = run_episode(t, ctrl, seed=s, surrogate=args.surrogate, batch_frac=args.batch_frac, topk_frac=args.topk_frac,
                            n_rounds=args.n_rounds, beta_scale=args.beta_scale,
                            diversity_lambda=args.diversity_lambda, shortlist_mult=args.shortlist_mult)
            rows.append({"task": t.name, "seed": s, "auc": episode_return(h, "auc"),
                         "final": episode_return(h, "final"), "betas": [x["beta_mean"] for x in h],
                         "redundancy": [x.get("batch_redundancy") for x in h]})
            print(f"[eval] {t.name} seed={s}  final={rows[-1]['final']:.3f}  auc={rows[-1]['auc']:.3f}  "
                  f"betas={[round(b, 2) for b in rows[-1]['betas']]}", flush=True)
    summary = {
        "controller": ctrl.describe(), "surrogate": args.surrogate, "beta_scale": args.beta_scale,
        "diversity_lambda": args.diversity_lambda,
        "final_mean": float(np.mean([r["final"] for r in rows])), "final_std": float(np.std([r["final"] for r in rows])),
        "auc_mean": float(np.mean([r["auc"] for r in rows])), "auc_std": float(np.std([r["auc"] for r in rows])),
        "episodes": rows,
    }
    print(f"[eval] {summary['controller']}  final={summary['final_mean']:.3f}±{summary['final_std']:.3f}  "
          f"auc={summary['auc_mean']:.3f}±{summary['auc_std']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


def main():
    from adaptive_acq.cli_common import add_controller_args

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", required=True)
    b.add_argument("--backbones", nargs="+", required=True)
    b.add_argument("--n-tasks", type=int, default=4)
    b.add_argument("--n-sub", type=int, default=200_000)
    b.add_argument("--pca-dim", type=int, default=128)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--block-size", type=int, default=1000,
                   help="sample contiguous blocks of this many rows (fast on huge memmapped pools); 1 = scattered rows")
    e = sub.add_parser("eval")
    e.add_argument("--tasks", nargs="+", required=True, help="task .npz files or directories")
    e.add_argument("--n-seeds", type=int, default=3)
    e.add_argument("--surrogate", default="ridge", choices=["ridge", "mlp"])
    e.add_argument("--batch-frac", type=float, default=SIM_BATCH_FRAC)
    e.add_argument("--topk-frac", type=float, default=SIM_TOPK_FRAC)
    e.add_argument("--n-rounds", type=int, default=5)
    e.add_argument("--eval-dataset", default="ENHITS")
    e.add_argument("--out", default=None)
    add_controller_args(e)
    args = p.parse_args()
    if args.cmd == "build":
        build_tasks(args.dataset, args.backbones, args.n_tasks, args.n_sub, args.pca_dim, args.seed, args.out_dir,
                    block_size=args.block_size)
    else:
        _eval_main(args)


if __name__ == "__main__":
    main()
