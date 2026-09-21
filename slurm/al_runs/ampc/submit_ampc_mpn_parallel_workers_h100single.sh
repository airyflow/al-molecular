#!/bin/bash

#SBATCH -J ampc_mpn_par_workers
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mpn_par_worker_%A_%a.txt
#SBATCH -e logs/ampc_mpn_par_worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# Prediction workers for ONE round of the MPN parallel run (see
# submit_ampc_mpn_parallel_runs_h100single.sh for the full design).
# Each array task predicts its own shard of the 98,489,350-molecule library
# with the round's MPN checkpoint, then exits -- submit once per round, when
# that round's round_R_ready.marker exists in the coord-dir.
#
# Sizing: shard = 98.5M / NUM_SHARDS molecules at ~1.7 ms/molecule (measured
# in probe job 10544431 with --ncpu 8; prediction is largely CPU
# featurization, so more CPUs might be faster -- unmeasured). At 8 shards
# that is ~5.8h per task, hence --time=14:00:00 with margin; at 16 shards
# ~2.9h. Worker memory is small (only the shard's SMILES ~1.3GB at 8 shards
# plus the model). --time does NOT restart partial work: a task killed at
# the limit redoes its whole shard, so keep the margin generous.
#
# Idempotent: a shard whose .done marker exists is skipped, so resubmitting
# the same round only redoes missing/failed shards (submit just those array
# indices, e.g. --array=3,5). Up to 3 attempts per task for transient GPU
# faults (a bad node has crashed workers with CUDA illegal-instruction
# errors before -- the retry may land the same node, so resubmit if it
# keeps failing).
#
# Usage:
#   sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_mpn_parallel_workers_h100single.sh <coord-dir> <num-shards> <round>

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-(NUM_SHARDS-1)}"
COORD_DIR="${1:?Usage: sbatch --array=0-N slurm/al_runs/ampc/submit_ampc_mpn_parallel_workers_h100single.sh <coord-dir> <num-shards> <round>}"
NUM_SHARDS="${2:?Usage: ... <coord-dir> <num-shards> <round>}"
ROUND="${3:?Usage: ... <coord-dir> <num-shards> <round>}"

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
mkdir -p logs "$COORD_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "[worker task $TASK_ID] round=$ROUND coord-dir=$COORD_DIR num-shards=$NUM_SHARDS"

for attempt in 1 2 3; do
    echo "[worker $TASK_ID] attempt $attempt"
    set +e
    srun --cpu-bind=none python3 -u predict_pool_shard_worker_mpn.py \
        --dataset AmpC_dedup --total-count 98489350 \
        --shard-id "$TASK_ID" --num-shards "$NUM_SHARDS" \
        --coord-dir "$COORD_DIR" --n-rounds 5 --rounds "$ROUND" --ncpu 8
    exit_code=$?
    set -e
    if [ "$exit_code" -eq 0 ]; then
        echo "[worker $TASK_ID] round $ROUND completed successfully"
        exit 0
    fi
    echo "[worker $TASK_ID] attempt $attempt exited with code $exit_code"
done
echo "[worker $TASK_ID] giving up after 3 attempts -- resubmit this array index"
exit 1
