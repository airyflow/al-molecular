#!/usr/bin/env python3
"""Persistent single-GPU prediction worker for run_experiment.py's
--mode molpal --model mpn --parallel-predict (ParallelMolPALExplorer).

Same marker-file protocol as predict_pool_shard_worker.py (the LT-All
worker), but the model is MolPAL's own MPN, which featurizes SMILES
itself -- so a worker needs only its shard's SMILES, never embeddings or
the oracle (keeping per-worker memory small: ~12M SMILES for a 98.5M pool
split 8 ways).

Each worker:
  1. Reads ONLY its shard's lines of the dataset's library file (the same
     unfiltered library the orchestrator shards over -- gathered
     predictions must line up with the orchestrator's pool).
  2. Per round: waits for coord/round_{r}_ready.marker, loads the
     orchestrator's round_{r}_mpn/ checkpoint (model.pt + state.json with
     the target scaler), predicts mu/var for its whole shard in
     POOL_PREDICT_CHUNK sub-chunks, writes round_{r}_shard_{id}_mu.npy /
     _var.npy, then touches round_{r}_shard_{id}.done.
  3. Exits when coord/STOP.marker appears (or after --rounds, if given). Rounds already .done from a
     prior attempt are skipped (same retry-wrapper contract as the LT-All
     worker).

Usage
-----
python predict_pool_shard_worker_mpn.py --dataset AmpC_dedup \\
    --shard-id 0 --num-shards 8 --coord-dir /path/to/coord --n-rounds 5 \\
    [--pool-limit N] [--total-count 98489350] [--ncpu 8]

--pool-limit must match the orchestrator's (used for small-scale tests).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import run_experiment as exp
from smiles_chunking import _chunk_bounds, count_lines, read_smiles_chunk


def wait_for(path: Path, poll_interval: float, stop_path: Path, started_at: float) -> bool:
    """Waits for `path`. Gives up only on a STOP marker written after this
    worker started (minus a 60s margin for clock skew between nodes) -- a
    STOP.marker left behind by an earlier, finished run must not make a
    worker of a resumed run exit immediately."""
    while not path.exists():
        if stop_path.exists() and stop_path.stat().st_mtime >= started_at - 60:
            return False
        time.sleep(poll_interval)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AmpC_dedup", choices=list(exp.DATASETS.keys()))
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--coord-dir", required=True)
    p.add_argument("--n-rounds", type=int, required=True)
    p.add_argument("--rounds", type=int, nargs="+", default=None,
                   help="Only process these round numbers, then exit (default: all 1..n-rounds, "
                        "persistent-pool style). MPN training on the orchestrator takes hours per "
                        "round, so per-round worker jobs (submitted when a round's checkpoint is "
                        "ready) avoid holding GPUs idle in the meantime.")
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--pool-limit", type=int, default=None,
                   help="Truncate the library to its first N rows (must match the orchestrator's --pool-limit)")
    p.add_argument("--total-count", type=int, default=None,
                   help="Library line count, to skip an O(N) line-count scan at startup")
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--length", type=int, default=2048)
    args = p.parse_args()

    started_at = time.time()
    exp.DATASET = args.dataset
    coord_dir = Path(args.coord_dir)
    stop_path = coord_dir / "STOP.marker"

    lib = exp.DATASETS[args.dataset]["library"]
    if lib.suffix != ".txt":
        raise SystemExit(f"{lib} is not a plain one-SMILES-per-line .txt library; "
                         f"shard reading is only implemented for that format")
    n_total = args.total_count if args.total_count is not None else count_lines(str(lib))
    if args.pool_limit is not None:
        n_total = min(n_total, args.pool_limit)
    start, end = _chunk_bounds(n_total, args.shard_id, args.num_shards)
    shard_smiles = read_smiles_chunk(str(lib), start, end)
    if len(shard_smiles) != end - start:
        raise SystemExit(f"shard {args.shard_id}: read {len(shard_smiles):,} SMILES for "
                         f"[{start:,}, {end:,}) -- expected {end - start:,}; blank lines in {lib}? "
                         f"the orchestrator's pool indexing would no longer line up")
    print(f"[worker {args.shard_id}/{args.num_shards}] shard = indices [{start:,}, {end:,}) "
          f"({len(shard_smiles):,} molecules)", flush=True)

    model = exp.build_mpn_model(ncpu=args.ncpu, length=args.length)

    for r in (args.rounds or range(1, args.n_rounds + 1)):
        ready_path = coord_dir / f"round_{r}_ready.marker"
        state_path = coord_dir / f"round_{r}_mpn" / "state.json"
        mu_path = coord_dir / f"round_{r}_shard_{args.shard_id}_mu.npy"
        var_path = coord_dir / f"round_{r}_shard_{args.shard_id}_var.npy"
        done_path = coord_dir / f"round_{r}_shard_{args.shard_id}.done"

        if done_path.exists():
            print(f"[worker {args.shard_id}] round {r}/{args.n_rounds} already done (from a prior attempt) -- skipping", flush=True)
            continue

        if not wait_for(ready_path, args.poll_interval, stop_path, started_at):
            print(f"[worker {args.shard_id}] STOP seen while waiting for round {r} -- exiting", flush=True)
            return

        t0 = time.perf_counter()
        model.load(state_path)
        mu, var = exp._chunked_get_means_and_vars(model, shard_smiles)
        np.save(mu_path, mu)
        np.save(var_path, var)
        done_path.touch()
        print(f"[worker {args.shard_id}] round {r}/{args.n_rounds} done in {time.perf_counter() - t0:.1f}s "
              f"({len(shard_smiles):,} molecules)", flush=True)

    print(f"[worker {args.shard_id}] requested rounds complete -- exiting", flush=True)


if __name__ == "__main__":
    main()
