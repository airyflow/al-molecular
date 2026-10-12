#!/usr/bin/env python3
"""Stage 2: train the beta policy with GRPO in the simulator.

Each iteration draws `--groups` (task, seed) starts. From each start, G
rollouts are sampled from the current policy -- same pool, same initial
labeled set, different beta choices (and surrogate bootstrap noise). The
episode return R (mean recall over rounds by default) is turned into a
group-relative advantage A_i = (R_i - mean_G R) / std_G R, applied to every
step of rollout i; no value network. The policy is then updated with the
PPO-clipped surrogate plus a k3 KL penalty to the initial (uniform) policy,
as in GRPO (Shao et al. 2024; used for molecular screening by GLARE,
NeurIPS 2025).

Unlike the bandits, the return is trajectory-level, so a policy CAN learn
that a high-beta round early on pays off in later rounds even though it
lowers that round's hit rate.

    python -m adaptive_acq.train_grpo --tasks sim_tasks/D4_dedup --val-tasks sim_tasks/AmpC_dedup_pca \\
        --iters 150 --group-size 8 --groups 4 --workers 16 --out adaptive_acq/artifacts/grpo_d4
"""

import argparse
import copy
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_acq.acquisition import DEFAULT_ARMS  # noqa: E402
from adaptive_acq.controllers import FixedBeta, PolicyController  # noqa: E402
from adaptive_acq.policy import BetaPolicy  # noqa: E402
from adaptive_acq.sim_env import SIM_BATCH_FRAC, SIM_TOPK_FRAC, episode_return, load_tasks, run_episode  # noqa: E402

_TASKS = {}


def _init_worker(task_paths):
    torch.set_num_threads(1)
    for t in load_tasks(task_paths):
        _TASKS[t.name] = t


def _rollout(job):
    """job = (task_name, seed, rollout_seed, policy_state, n_arms, hidden, arms, mode, cfg)"""
    name, seed, rseed, state, n_arms, hidden, arms, mode, cfg = job
    pol = BetaPolicy(n_arms, hidden)
    pol.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    pol.eval()
    ctrl = PolicyController(pol, arms=arms, mode=mode, seed=rseed)
    h = run_episode(_TASKS[name], ctrl, seed=seed, surrogate=cfg["surrogate"], batch_frac=cfg["batch_frac"],
                    topk_frac=cfg["topk_frac"], beta_scale=cfg["beta_scale"])
    return {"task": name, "seed": seed, "R": episode_return(h, cfg["return_kind"]),
            "final": episode_return(h, "final"), "traj": ctrl.trajectory,
            "betas": [x["beta_mean"] for x in h]}


def _fixed_baseline(job):
    name, seed, beta, cfg = job
    h = run_episode(_TASKS[name], FixedBeta(beta), seed=seed, surrogate=cfg["surrogate"],
                    batch_frac=cfg["batch_frac"], topk_frac=cfg["topk_frac"], beta_scale=cfg["beta_scale"])
    return {"task": name, "seed": seed, "beta": beta, "R": episode_return(h, cfg["return_kind"]),
            "final": episode_return(h, "final")}


def grpo_update(policy, ref, batch, epochs, lr_opt, clip, kl_coef, ent_coef):
    X = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    a = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    logp_old = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
    adv = torch.as_tensor([b[3] for b in batch], dtype=torch.float32)
    with torch.no_grad():
        logp_ref_all = ref(X)
    stats = {}
    for _ in range(epochs):
        logp_all = policy(X)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        ratio = (logp - logp_old).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
        lr_ref = logp_ref_all.gather(1, a[:, None]).squeeze(1) - logp
        kl = (lr_ref.exp() - lr_ref - 1).mean()  # k3 estimator of KL(pi || pi_ref)
        ent = -(logp_all.exp() * logp_all).sum(1).mean()
        loss = pg + kl_coef * kl - ent_coef * ent
        lr_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        lr_opt.step()
        stats = {"pg": float(pg), "kl": float(kl), "entropy": float(ent)}
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True, help="training task .npz files/dirs")
    p.add_argument("--val-tasks", nargs="+", default=None, help="held-out tasks (ideally another target)")
    p.add_argument("--arms", type=float, nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--groups", type=int, default=4, help="(task, seed) starts per iteration")
    p.add_argument("--group-size", type=int, default=8, help="G rollouts per start")
    p.add_argument("--epochs", type=int, default=4, help="policy epochs per iteration")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--return-kind", default="auc", choices=["auc", "final"])
    p.add_argument("--surrogate", default="ridge", choices=["ridge", "mlp"])
    p.add_argument("--batch-frac", type=float, default=SIM_BATCH_FRAC)
    p.add_argument("--topk-frac", type=float, default=SIM_TOPK_FRAC)
    p.add_argument("--beta-scale", default="normalized", choices=["normalized", "raw"])
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--val-seeds", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = {k: getattr(args, k) for k in ("surrogate", "batch_frac", "topk_frac", "beta_scale", "return_kind")}
    arms = tuple(args.arms)

    train_names = [t.name for t in load_tasks(args.tasks)]
    val_names = [t.name for t in load_tasks(args.val_tasks)] if args.val_tasks else []
    all_paths = list(args.tasks) + list(args.val_tasks or [])

    policy = BetaPolicy(len(arms), args.hidden)
    ref = copy.deepcopy(policy).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    ctx_mp = get_context("fork")
    log, best_val = [], -np.inf
    with ctx_mp.Pool(args.workers, initializer=_init_worker, initargs=(all_paths,)) as pool:
        # Fixed-beta reference on the validation starts: the bar the policy must clear.
        val_starts = [(n, 10_000 + s) for n in val_names for s in range(args.val_seeds)]
        baseline = {}
        if val_starts:
            res = pool.map(_fixed_baseline, [(n, s, b, cfg) for n, s in val_starts for b in arms])
            for b in arms:
                baseline[b] = float(np.mean([r["R"] for r in res if r["beta"] == b]))
            print(f"[grpo] val fixed-beta R: {baseline}", flush=True)

        for it in range(args.iters):
            t0 = time.perf_counter()
            state = {k: v.detach().numpy().copy() for k, v in policy.state_dict().items()}
            starts = [(train_names[rng.integers(len(train_names))], int(rng.integers(1 << 30)))
                      for _ in range(args.groups)]
            jobs = [(n, s, int(rng.integers(1 << 30)), state, len(arms), args.hidden, arms, "sample", cfg)
                    for n, s in starts for _ in range(args.group_size)]
            results = pool.map(_rollout, jobs)

            batch, Rs = [], []
            for g in range(args.groups):
                grp = results[g * args.group_size:(g + 1) * args.group_size]
                R = np.array([r["R"] for r in grp])
                Rs.append(R)
                A = (R - R.mean()) / (R.std() + 1e-6)
                for r, adv in zip(grp, A):
                    batch += [(c, a, lp, float(adv)) for c, a, lp in r["traj"]]
            stats = grpo_update(policy, ref, batch, args.epochs, opt, args.clip, args.kl_coef, args.ent_coef)

            rec = {"iter": it, "R_mean": float(np.mean(Rs)), "R_group_std": float(np.mean([r.std() for r in Rs])),
                   **stats, "sec": round(time.perf_counter() - t0, 1)}
            if val_starts and (it % args.val_every == 0 or it == args.iters - 1):
                vstate = {k: v.detach().numpy().copy() for k, v in policy.state_dict().items()}
                vres = pool.map(_rollout, [(n, s, 0, vstate, len(arms), args.hidden, arms, "greedy", cfg)
                                           for n, s in val_starts])
                rec["val_R"] = float(np.mean([r["R"] for r in vres]))
                rec["val_final"] = float(np.mean([r["final"] for r in vres]))
                rec["val_betas"] = vres[0]["betas"]
                if rec["val_R"] > best_val:
                    best_val = rec["val_R"]
                    policy.save(out / "policy_best.pt", arms, extra={"iter": it, "val_R": best_val, "cfg": cfg})
            log.append(rec)
            print(f"[grpo] it {it:4d}  R={rec['R_mean']:.3f} (grp std {rec['R_group_std']:.3f})  "
                  f"kl={stats['kl']:.4f} H={stats['entropy']:.3f}"
                  + (f"  val_R={rec['val_R']:.3f} betas={rec['val_betas']}" if "val_R" in rec else "")
                  + f"  ({rec['sec']}s)", flush=True)
            (out / "train_log.json").write_text(json.dumps({"args": vars(args), "val_fixed_beta": baseline,
                                                            "log": log}, indent=2))

    policy.save(out / "policy_last.pt", arms, extra={"iter": args.iters - 1, "cfg": cfg})
    if not val_starts:
        print("[grpo] no --val-tasks given: policy_best.pt not written; use policy_last.pt")
    print(f"[grpo] done -> {out}")


if __name__ == "__main__":
    main()
