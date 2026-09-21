#!/bin/bash

#SBATCH -J ampc_mpn_validate
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mpn_validate_%j.txt
#SBATCH -e logs/ampc_mpn_validate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH -A r00939

# Real-GPU rehearsal of the sharded MPN pipeline (ParallelMolPALExplorer +
# predict_pool_shard_worker_mpn.py) before committing days of cluster time
# to the full AmpC run (submit_ampc_mpn_parallel_runs_h100single.sh). The
# pipeline has only been verified on a 2,000-molecule CPU toy dataset
# (protocol, checkpoint hand-off, resume); this exercises it on the real
# AmpC library/oracle, real GPU, real MPN training, at small scale.
#
# One H100 node runs everything: the orchestrator plus 2 workers as
# background processes sharing the GPU (the MPN is small), on the first
# 1,000,000 library rows (all oracle-scored, same pool as probe job
# 10544431), init=batch=4,000 (0.4% of that pool), 2 rounds, 2 shards, greedy.
#
# What to check when it finishes:
#   * completes both rounds with no traceback; history.json written;
#   * round 1 labeled=8,000, best score in the same ballpark as probe
#     10544431's 1M-pool round (best=-85.850 kcal/mol; not identical --
#     MPN init is unseeded -- but the same initial 4,000 molecules, since
#     the init draw matches the single-process explorer for an all-scored
#     pool);
#   * both workers log a round-1 and round-2 "done in ...s" (~500k
#     molecules each at ~1.7 ms/mol => ~14 min if a worker had the GPU/CPUs
#     to itself; longer here since they share the node);
#   * total time roughly: 2 x MPN training (~48 min each at this size, per
#     the probes) + prediction.
#
# Usage: sbatch slurm/al_runs/ampc/submit_ampc_mpn_parallel_validate_h100single.sh

set -euo pipefail

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"
COORD_DIR="$AMPC_ROOT/al_coord/mpn_validate_1M"
RUN_DIR="runs/ampc_mpn_validate_1M"
POOL=1000000
INIT=4000
SHARDS=2

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fi
set -a
[ -f config.env ] && source config.env
set +a
: "${CONDA_SH:=/N/slate/mengjing/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=py310}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
mkdir -p logs

# Fresh state every time -- a stale coord-dir/run-dir from an earlier
# rehearsal would otherwise be resumed/skipped instead of re-tested.
rm -rf "$COORD_DIR" "$RUN_DIR"
mkdir -p "$COORD_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    python3 -u predict_pool_shard_worker_mpn.py \
        --dataset AmpC_dedup --pool-limit "$POOL" \
        --shard-id "$i" --num-shards "$SHARDS" \
        --coord-dir "$COORD_DIR" --n-rounds 2 --ncpu 4 \
        > "logs/ampc_mpn_validate_${SLURM_JOB_ID}_worker${i}.txt" 2>&1 &
    pids+=($!)
done

set +e
python3 -u run_experiment.py --dataset AmpC_dedup \
    --mode molpal --model mpn --conf-method mve --acq greedy \
    --pool-limit "$POOL" --init-size "$INIT" --batch-size "$INIT" \
    --n-rounds 2 --topk 1000 \
    --parallel-predict --num-shards "$SHARDS" --coord-dir "$COORD_DIR" \
    --ncpu 8 --retrain-from-scratch --run-dir "$RUN_DIR"
orch_status=$?
set -e

if [ "$orch_status" -ne 0 ]; then
    echo "[validate] orchestrator failed (status $orch_status) -- stopping workers"
    kill "${pids[@]}" 2>/dev/null || true
    exit "$orch_status"
fi

# Workers exit on their own after STOP.marker; collect their exit codes.
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[validate] worker pid $pid exited non-zero"; fail=1; }
done
echo "[validate] orchestrator status=$orch_status worker_failures=$fail"
cat "$RUN_DIR/history.json"
exit $((orch_status + fail))
