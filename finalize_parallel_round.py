#!/usr/bin/env python3
"""Finishes one round of a ParallelMolPALExplorer run offline, on CPU, from
files already on disk -- for when every shard of round R has its .done
marker but the orchestrator was killed (e.g. its 48h SLURM limit) before it
gathered them. Resubmitting the orchestrator instead would retrain round R's
MPN from scratch (hours of GPU) only to throw that model away, since the
predictions it needs already exist.

Repeats exactly the code ParallelMolPALExplorer.run() executes after its
shard wait loop (run_experiment.py): gather round_R_shard_*_mu/var.npy, mask
out unscored and already-labeled rows, rank with the same acquisition
function, take the top --batch-size, add them to the labeled set, and
compute best score and top-k recall the same way. Starting state is
iter_(R-1)/scores.pkl, reconstructed to row indices exactly as the
orchestrator's --resume does.

Modes:
  --check   compute round R and compare against an EXISTING iter_R written by
            the orchestrator (same labeled set, same recall); writes nothing.
            Use it on an already-finished round to validate this script.
  (default) write iter_R/state.json + scores.pkl; if R is the last round,
            also history.json, all_explored_final.csv and STOP.marker, as
            the orchestrator would.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

import run_experiment as exp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(exp.DATASETS.keys()))
    p.add_argument("--coord-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--round", type=int, required=True, help="Round R to finish (1-based)")
    p.add_argument("--n-rounds", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--acq", default="greedy", choices=["greedy", "ucb"])
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    from molpal.acquirer.metrics import get_metric

    t0 = time.perf_counter()
    exp.DATASET = args.dataset
    coord_dir, run_dir, r = Path(args.coord_dir), Path(args.run_dir), args.round

    oracle = exp.load_oracle()
    pool_smiles = exp.load_library_smiles()
    n = len(pool_smiles)
    usable_mask = np.fromiter((s in oracle for s in pool_smiles), dtype=bool, count=n)
    print(f"[pool] {usable_mask.sum():,}/{n:,} molecules have oracle scores", flush=True)

    prev = run_dir / f"iter_{r - 1}" / "scores.pkl"
    with open(prev, "rb") as f:
        labeled_scores = dict(pickle.load(f))
    wanted = set(labeled_scores)
    smi2idx = {}
    for i, s in enumerate(pool_smiles):
        if s in wanted:
            smi2idx[s] = i  # last-write-wins, same as ParallelMolPALExplorer's resume
    labeled_idx = {smi2idx[s] for s in wanted if s in smi2idx}
    print(f"[start] {len(labeled_idx):,} labeled molecules from {prev}", flush=True)

    mu_parts, var_parts = [], []
    for shard_id in range(args.num_shards):
        done = coord_dir / f"round_{r}_shard_{shard_id}.done"
        if not done.exists():
            raise SystemExit(f"{done} is missing -- round {r} is not finished; nothing written")
        mu_parts.append(np.load(coord_dir / f"round_{r}_shard_{shard_id}_mu.npy"))
        var_parts.append(np.load(coord_dir / f"round_{r}_shard_{shard_id}_var.npy"))
    mu_full = np.concatenate(mu_parts)
    var_full = np.concatenate(var_parts)
    assert len(mu_full) == n, f"gathered {len(mu_full):,} predictions but pool has {n:,} molecules"

    acq_fn = get_metric(args.acq)
    needs_var = args.acq in ("ucb", "lcb", "thompson", "ts", "ei", "pi")
    mask = usable_mask.copy()
    for i in labeled_idx:
        mask[i] = False
    pool_idx = np.where(mask)[0]
    mu, var = mu_full[pool_idx], var_full[pool_idx]
    scores = acq_fn(mu, var) if needs_var else acq_fn(mu)
    top_local = np.argsort(scores)[::-1][: args.batch_size]
    selected = pool_idx[top_local]
    cutoff = scores[top_local[-1]]
    n_tied = int((scores == cutoff).sum())
    n_tied_kept = int((scores[top_local] == cutoff).sum())
    print(f"[cutoff] last selected score = {cutoff!r}; {n_tied:,} candidates share it, "
          f"{n_tied_kept:,} of them selected", flush=True)

    for i in selected:
        smi = pool_smiles[i]
        labeled_scores[smi] = oracle[smi]
        labeled_idx.add(int(i))

    true_top_k = exp.true_top_k_set(oracle, args.topk)
    recall = sum(1 for s in labeled_scores if s in true_top_k) / args.topk
    best = min(labeled_scores.values())
    elapsed = time.perf_counter() - t0
    print(f"  Round {r:02d}/{args.n_rounds}  labeled={len(labeled_idx):,}  best={best:.3f} kcal/mol  "
          f"top-{args.topk} recall={recall:.1%}  (finalized offline, {elapsed:.1f}s)", flush=True)

    if args.check:
        ref_dir = run_dir / f"iter_{r}"
        ref_state = json.loads((ref_dir / "state.json").read_text())
        with open(ref_dir / "scores.pkl", "rb") as f:
            ref_scores = dict(pickle.load(f))
        same_set = set(ref_scores) == set(labeled_scores)
        print(f"[check] orchestrator iter_{r}: n_labeled={ref_state['n_labeled']:,} "
              f"recall={ref_state['topk_recall']} best={ref_state['best_score']}")
        print(f"[check] offline:              n_labeled={len(labeled_idx):,} "
              f"recall={round(recall, 5)} best={best}")
        print(f"[check] identical labeled set: {same_set}")
        if not same_set:
            only_ref = set(ref_scores) - set(labeled_scores)
            only_off = set(labeled_scores) - set(ref_scores)
            print(f"[check] only in orchestrator: {len(only_ref):,}; only offline: {len(only_off):,}")
            if len(only_ref) + len(only_off) <= 20:
                pos = {s: i for i, s in enumerate(pool_smiles) if s in only_ref or s in only_off}
                score_all = np.full(n, np.nan, dtype=np.float64)
                score_all[pool_idx] = scores
                for tag, group in (("orchestrator-only", only_ref), ("offline-only", only_off)):
                    for s in group:
                        i = pos[s]
                        print(f"[check]   {tag}: row {i:,} acquisition score {score_all[i]!r} "
                              f"(cutoff {cutoff!r}, tie: {score_all[i] == cutoff})")
        print("[check] nothing written")
        return

    record = dict(round=r, n_labeled=len(labeled_idx), best_score=float(best),
                  topk_recall=float(recall), elapsed=round(elapsed, 2), finalized_offline=True)
    d = run_dir / f"iter_{r}"
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text(json.dumps(record, indent=2))
    with open(d / "scores.pkl", "wb") as f:
        pickle.dump(dict(labeled_scores), f)
    print(f"[write] {d}/state.json, scores.pkl")

    if r == args.n_rounds:
        history = []
        for k in range(1, r + 1):
            history.append(json.loads((run_dir / f"iter_{k}" / "state.json").read_text()))
        (coord_dir / "STOP.marker").touch()
        import pandas as pd
        pd.DataFrame(sorted(labeled_scores.items(), key=lambda x: x[1]), columns=["smiles", "score"]) \
            .to_csv(run_dir / "all_explored_final.csv", index=False)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[write] {run_dir}/history.json, all_explored_final.csv; {coord_dir}/STOP.marker")


if __name__ == "__main__":
    main()
