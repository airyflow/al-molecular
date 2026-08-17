#!/bin/bash

#SBATCH -J ampc_al_workers
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_al_worker_%A_%a.txt
#SBATCH -e logs/ampc_al_worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Persistent single-GPU worker pool for ParallelMVEExplorer's chunked-
# prediction parallelization (see predict_pool_shard_worker.py's module
# docstring for the full protocol). Submitted ONCE for the whole AL run (all
# --n-rounds), not resubmitted per round -- avoids paying SLURM queue-wait
# latency n_rounds times over, at the cost of reserving --array many GPUs
# for the run's full duration (idle between rounds while the orchestrator
# trains on the labeled set).
#
# --mem=32G (not --exclusive/--mem=0, unlike the MPN/Ray-based AL scripts):
# no Ray involved here, so none of the node-sharing port-collision reasons
# those scripts document apply. Embeddings load as memmaps and are only
# ever fancy-indexed in bounded 50,000-row chunks (see
# run_experiment.py::_chunked_get_means_and_vars), so a worker's actual
# peak RAM is a few GB at most -- --mem=0 (whole node) was tripping
# QOSMaxMemoryPerNode on first submission.
#
# Each array task is a fixed, disjoint shard of the AmpC pool (99,459,561
# molecules) -- shard i handles indices [i*ceil(N/num_shards),
# (i+1)*ceil(N/num_shards)), same _chunk_bounds() convention as every other
# sharded script in this repo. Workers wait for round_{r}_ready.marker,
# reload the orchestrator's freshly-trained surrogate checkpoint, predict
# their whole shard, write results, and loop -- until STOP.marker appears
# after the final round.
#
# --coord-dir MUST be a dedicated, otherwise-empty directory -- it's also
# where the orchestrator (submit_ampc_fusion_runs_h100single.sh) writes its
# round-ready markers and checkpoints. One coord-dir per orchestrator run;
# running "ensemble" and "learned" concurrently needs two separate worker
# pools (two separate coord-dirs), not one shared pool.
#
# --backbones MUST match the orchestrator's --backbones exactly (workers
# load embeddings independently, keyed by backbone name -- a mismatch would
# silently predict from the wrong embedding columns).
#
# Sized at 8 workers by default: 1 orchestrator + 8 workers = 9 jobs, well
# under the observed QOSMaxJobsPerUserLimit=20 on h100-single, leaving room
# to run both "ensemble" and "learned" pools concurrently (9+9=18) if
# wanted. Increase --array once a real per-shard timing number is in hand
# and the concurrency budget allows it.
#
# --num-shards is passed explicitly as $2 (NOT inferred from
# SLURM_ARRAY_TASK_COUNT) specifically so a single dead/missing shard can be
# resubmitted on its own -- e.g. sbatch --array=7 ...script.sh <coord-dir>
# 8 -- without SLURM_ARRAY_TASK_COUNT collapsing to 1 (the size of THAT
# submission's array) and silently making the resubmitted worker think it
# owns the whole pool instead of its actual 1/8th shard. This is a real
# scenario, not hypothetical: a worker task was killed independently of its
# 7 siblings mid-run once already (2026-08-14), stalling the orchestrator on
# a marker that could never arrive.
#
# Usage (submit BEFORE the matching orchestrator; workers wait patiently on
# round 1 with no timeout until it appears):
#   mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ensemble
#   sbatch --array=0-7 submit_ampc_al_predict_workers_h100single.sh \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ensemble 8
#
# Resubmitting a single dead shard (keeps the other 7 running workers and
# the in-progress coord-dir state untouched):
#   sbatch --array=7 submit_ampc_al_predict_workers_h100single.sh \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ensemble 8
#
# AUTO-RETRY-ON-TIMEOUT: shard 7 specifically has needed manual cancel+
# resubmit THREE separate times (2026-08-14/15) -- its ~192k within-shard
# duplicate SMILES mean every chunk's fetch includes a small scattered
# correction-read (see EmbeddingMVEModel._get_X()'s mostly-contiguous+patch
# path), and a scattered read against these single-OST, unstriped embedding
# files can still occasionally hang indefinitely even when it's reading only
# a handful of rows (confirmed via py-spy on a local reproduction: frozen
# syscall count, never progressing). The fix reduces how OFTEN this happens,
# it doesn't eliminate the possibility. Rather than requiring a human to
# notice and manually cycle it every round, wrap the whole worker in a
# timeout+restart loop: healthy rounds finish in ~10-13 min (confirmed
# across shards 0-6 and shard 7 once fixed), so a 45-minute timeout leaves
# ~3x margin before assuming a hang and restarting. predict_pool_shard_worker.py
# now skips rounds that already have a .done marker on restart, so a retry
# only re-does the ONE stuck round, not everything from round 1.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-N (SLURM_ARRAY_TASK_ID is unset)}"
COORD_DIR="${1:?Usage: sbatch --array=0-N submit_ampc_al_predict_workers_h100single.sh <coord-dir> <num-shards>}"
NUM_SHARDS="${2:?Usage: sbatch --array=0-N submit_ampc_al_predict_workers_h100single.sh <coord-dir> <num-shards>}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs "$COORD_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "[worker task $TASK_ID] coord-dir=$COORD_DIR num-shards=$NUM_SHARDS"

TIMEOUT_SECONDS=2700  # 45 min -- ~3x the ~10-13 min healthy completion time
attempt=1
while true; do
    echo "[worker $TASK_ID] attempt $attempt (timeout ${TIMEOUT_SECONDS}s)"
    set +e
    srun --cpu-bind=none timeout "${TIMEOUT_SECONDS}s" python3 -u predict_pool_shard_worker.py \
        --dataset AmpC --backbones grover molformer unimol \
        --shard-id "$TASK_ID" --num-shards "$NUM_SHARDS" \
        --coord-dir "$COORD_DIR" --n-rounds 5
    exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        echo "[worker $TASK_ID] completed successfully"
        break
    elif [ "$exit_code" -eq 124 ]; then
        echo "[worker $TASK_ID] TIMED OUT after ${TIMEOUT_SECONDS}s -- likely a hung scattered read, restarting "
        echo "  (already-.done rounds will be skipped, not redone)"
        attempt=$((attempt + 1))
    else
        echo "[worker $TASK_ID] exited with unexpected code $exit_code -- NOT auto-retrying (not a timeout)"
        exit "$exit_code"
    fi
done
