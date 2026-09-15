#!/bin/bash

#SBATCH -J ampc_dedup_ltall_ucb_workers
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_dedup_ltall_ucb_worker_%A_%a.txt
#SBATCH -e logs/ampc_dedup_ltall_ucb_worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Same as submit_ampc_dedup_ltall_predict_workers_h100single.sh -- workers
# are acquisition-function-agnostic (they only ever produce raw mu/var
# predictions; the greedy vs. ucb scoring happens in the orchestrator
# after gathering all shards' output), so this is identical except for
# its own coord-dir, kept separate to match the UCB orchestrator
# (submit_ampc_dedup_ltall_ucb_runs_h100single.sh) rather than the greedy
# one -- different acquisition function means a different sequence of
# acquired molecules from round 2 onward, so no state is shared.
#
# --coord-dir MUST be a dedicated, otherwise-empty directory, separate from
# BOTH the non-deduplicated AmpC run's coord-dir AND the greedy dedup run's
# (al_coord/dedup_ltall_frac<FRAC>, not al_coord/dedup_ltall_ucb_frac<FRAC>).
#
# Usage (submit BEFORE the matching orchestrator; workers wait patiently on
# round 1 with no timeout until it appears):
#   mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall_ucb_frac0.004
#   sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_dedup_ltall_ucb_predict_workers_h100single.sh \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall_ucb_frac0.004 8

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-N (SLURM_ARRAY_TASK_ID is unset)}"
COORD_DIR="${1:?Usage: sbatch --array=0-N slurm/al_runs/ampc/submit_ampc_dedup_ltall_ucb_predict_workers_h100single.sh <coord-dir> <num-shards>}"
NUM_SHARDS="${2:?Usage: sbatch --array=0-N slurm/al_runs/ampc/submit_ampc_dedup_ltall_ucb_predict_workers_h100single.sh <coord-dir> <num-shards>}"

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

echo "[worker task $TASK_ID] coord-dir=$COORD_DIR num-shards=$NUM_SHARDS"

TIMEOUT_SECONDS=2700  # 45 min -- same margin as the non-dedup workers
attempt=1
while true; do
    echo "[worker $TASK_ID] attempt $attempt (timeout ${TIMEOUT_SECONDS}s)"
    set +e
    srun --cpu-bind=none timeout "${TIMEOUT_SECONDS}s" python3 -u predict_pool_shard_worker.py \
        --dataset AmpC_dedup --backbones grover3400 mhgged molformer smited unimol \
        --shard-id "$TASK_ID" --num-shards "$NUM_SHARDS" \
        --coord-dir "$COORD_DIR" --n-rounds 5
    exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        echo "[worker $TASK_ID] completed successfully"
        break
    elif [ "$exit_code" -eq 124 ]; then
        echo "[worker $TASK_ID] TIMED OUT after ${TIMEOUT_SECONDS}s -- likely a hung scattered read, restarting"
        echo "  (already-.done rounds will be skipped, not redone)"
        attempt=$((attempt + 1))
    else
        echo "[worker $TASK_ID] exited with unexpected code $exit_code -- NOT auto-retrying (not a timeout)"
        exit "$exit_code"
    fi
done
