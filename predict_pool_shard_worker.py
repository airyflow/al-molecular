#!/usr/bin/env python3
"""Persistent single-GPU worker for chunked-prediction parallelization of the
MVE active-learning loop (see run_experiment.py::ParallelMVEExplorer).

Motivation: at AmpC scale (99.5M molecules), the per-round pool-prediction
step is the AL loop's real bottleneck, not surrogate training (the surrogate
is a small MLP trained on the labeled set only -- thousands of molecules,
trivially fast on one GPU). Splitting *prediction* across N independent
single-GPU jobs is the actual win; there is no reason to attempt real
multi-GPU distributed *training* for a model this small.

Coordination is a persistent worker pool talking over a shared-filesystem
"coord" directory with plain marker files -- the same pattern already used
throughout this repo (chunk_XXXXX.npy + .stitch_progress), not a message
queue or RPC framework. Submitted ONCE for the whole run (all --n-rounds),
not resubmitted per round, to avoid paying SLURM queue-wait latency
--n-rounds times over (h100-single has been observed busy/contended).

Each worker:
  1. Loads ONE fixed shard of the pool (SMILES + embeddings) ONCE at startup.
  2. Loops over rounds: waits for coord/round_{r}_ready.marker, loads the
     orchestrator's freshly-trained surrogate checkpoint (torch.save'd whole
     surrogate object, not a bare state_dict -- see ParallelMVEExplorer),
     swaps it into a persistent model wrapper, predicts mu/var for its ENTIRE
     shard (including already-labeled molecules -- the orchestrator masks
     those out after gathering; predicting the whole fixed shard every round
     is simpler and more robust than trying to keep each worker's notion of
     "still-unlabeled" in sync with the orchestrator's growing labeled set,
     and the wasted compute is negligible: the labeled set never exceeds
     roughly 1% of the pool even in the final AmpC round).
  3. Writes round_{r}_shard_{shard_id}_mu.npy / _var.npy, then touches
     round_{r}_shard_{shard_id}.done.
  4. Exits cleanly once coord/STOP.marker appears (written by the
     orchestrator after the last round).

Prediction reuses run_experiment.py's own _chunked_get_means_and_vars() (the
same POOL_PREDICT_CHUNK=50_000 internal sub-chunking the sequential
single-process path already uses) so a worker's output for its shard is
numerically identical to what the current in-process MVEExplorer would have
produced for that same slice of the pool -- parallelizing WHICH GPU does the
work, not changing the prediction granularity/locality that
EnsembleFusionSurrogate's within-call Borda ranking depends on.

Usage
-----
python predict_pool_shard_worker.py \\
    --dataset AmpC --backbones grover molformer unimol \\
    --shard-id 0 --num-shards 20 \\
    --coord-dir /path/to/run_dir/coord --n-rounds 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import run_experiment as exp
from generate_unimol_conformers_chunk import _chunk_bounds

# Test-only hook: lets a verification harness register a synthetic dataset
# (e.g. a tiny toy embed_dir) into a fresh subprocess's copy of
# run_experiment.DATASETS before argparse's --dataset choices are built.
# No-op in production (env var absent) -- see verify_parallel.py.
_extra_datasets = os.environ.get("AL_EXTRA_DATASETS_JSON")
if _extra_datasets:
    exp.DATASETS.update(json.loads(_extra_datasets))


def wait_for(path: Path, poll_interval: float, stop_path: Path) -> bool:
    """Polls until `path` exists. Returns False (give up) if `stop_path`
    appears first -- lets a worker exit mid-wait instead of only checking
    STOP between rounds."""
    while not path.exists():
        if stop_path.exists():
            return False
        time.sleep(poll_interval)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AmpC", choices=list(exp.DATASETS.keys()))
    p.add_argument("--backbones", nargs="+", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--coord-dir", required=True)
    p.add_argument("--n-rounds", type=int, required=True)
    p.add_argument("--poll-interval", type=float, default=5.0)
    args = p.parse_args()

    exp.DATASET = args.dataset
    coord_dir = Path(args.coord_dir)
    stop_path = coord_dir / "STOP.marker"

    print(f"[worker {args.shard_id}/{args.num_shards}] loading full pool embeddings "
          f"({args.dataset}, backbones={args.backbones}) ...", flush=True)
    emb_dict, pool_smiles = exp.load_embeddings(args.backbones)

    n_total = len(pool_smiles)
    start, end = _chunk_bounds(n_total, args.shard_id, args.num_shards)
    shard_smiles = pool_smiles[start:end]
    shard_emb = {bb: e[start:end] for bb, e in emb_dict.items()}
    print(f"[worker {args.shard_id}/{args.num_shards}] shard = indices [{start:,}, {end:,}) "
          f"({len(shard_smiles):,} molecules)", flush=True)

    # Built once with fresh (soon-to-be-overwritten) weights -- only the
    # _get_X()/wrapper plumbing matters here, not this initial surrogate.
    from molpal.models import mve as build_mve
    dims = {k: v.shape[1] for k, v in shard_emb.items()}
    # surrogate_type/backbone are placeholders; the real surrogate object is
    # loaded wholesale from the orchestrator's checkpoint each round, which
    # replaces model.surrogate entirely regardless of what was built here.
    model = build_mve(
        surrogate_type="single" if len(args.backbones) == 1 else "ensemble",
        backbone=args.backbones[0],
        emb_dict=shard_emb, pool_smiles=shard_smiles, dataset_name=args.dataset,
    )

    for r in range(1, args.n_rounds + 1):
        ready_path = coord_dir / f"round_{r}_ready.marker"
        ckpt_path = coord_dir / f"round_{r}_surrogate.pt"
        mu_path = coord_dir / f"round_{r}_shard_{args.shard_id}_mu.npy"
        var_path = coord_dir / f"round_{r}_shard_{args.shard_id}_var.npy"
        done_path = coord_dir / f"round_{r}_shard_{args.shard_id}.done"

        if done_path.exists():
            # Already completed this round (e.g. a prior attempt got this
            # far before being killed/timed out on a later round -- see
            # submit_ampc_al_predict_workers_h100single.sh's retry wrapper).
            # Skip straight to the next round instead of redoing real,
            # already-correct prediction work.
            print(f"[worker {args.shard_id}] round {r}/{args.n_rounds} already done (from a prior attempt) -- skipping", flush=True)
            continue

        if not wait_for(ready_path, args.poll_interval, stop_path):
            print(f"[worker {args.shard_id}] STOP seen while waiting for round {r} -- exiting", flush=True)
            return

        t0 = time.perf_counter()
        model.surrogate = torch.load(ckpt_path, map_location=exp.DEVICE, weights_only=False)

        mu, var = exp._chunked_get_means_and_vars(model, shard_smiles)
        np.save(mu_path, mu)
        np.save(var_path, var)
        done_path.touch()

        elapsed = time.perf_counter() - t0
        print(f"[worker {args.shard_id}] round {r}/{args.n_rounds} done in {elapsed:.1f}s "
              f"({len(shard_smiles):,} molecules)", flush=True)

    print(f"[worker {args.shard_id}] all {args.n_rounds} rounds complete -- exiting", flush=True)


if __name__ == "__main__":
    main()
