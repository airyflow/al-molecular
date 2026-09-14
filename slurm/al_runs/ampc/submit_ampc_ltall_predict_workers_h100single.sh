#!/bin/bash

#SBATCH -J ampc_ltall_workers
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_ltall_worker_%A_%a.txt
#SBATCH -e logs/ampc_ltall_worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Persistent single-GPU worker pool for the LT-All (LTAllSurrogate) AmpC
# run's ParallelMVEExplorer chunked-prediction parallelization -- same
# protocol as submit_ampc_al_predict_workers_h100single.sh (ensemble/
# learned), just a dedicated script for LT-All's own 5-backbone
# concatenation (matches the ENHITS LT-All config exactly, except unimol2
# -> unimol v1, since AmpC's own unimol2 extraction isn't finished/stitched
# yet -- see submit_ltall_enhits_seed_runs_h100single.sh for the ENHITS
# backbone set this mirrors):
#   grover3400 (3400d) + mhgged (1024d) + molformer (768d) + smited (768d)
#   + unimol (512d) = 6,472-d concatenated input.
#
# --backbones MUST match the orchestrator's (submit_ampc_ltall_runs_h100single.sh)
# --backbones exactly and in the same order -- workers load embeddings
# independently, keyed by backbone name; LTAllSurrogate additionally splits
# the concatenated embedding by cumulative dims (see surrogates.py's
# LTAllSurrogate._split), so a mismatched order would silently feed the
# wrong embedding columns into the wrong per-source sub-network.
#
# --coord-dir MUST be a dedicated, otherwise-empty directory, separate from
# any ensemble/learned coord-dir already in use (see
# submit_ampc_al_predict_workers_h100single.sh's own note on this).
#
# 8 workers (1 orchestrator + 8 = 9 jobs), same sizing as the
# ensemble/learned pools, well under the observed QOSMaxJobsPerUserLimit.
#
# --num-shards passed explicitly as $2 (not inferred from
# SLURM_ARRAY_TASK_COUNT) so a single dead/missing shard can be resubmitted
# on its own without SLURM_ARRAY_TASK_COUNT collapsing to 1 -- see
# submit_ampc_al_predict_workers_h100single.sh's own note on why this
# matters (a real, previously-hit failure mode, not hypothetical).
#
# AUTO-RETRY-ON-TIMEOUT wrapper: same reasoning as the ensemble/learned
# worker script -- a scattered read against AmpC's embedding files can
# occasionally hang indefinitely even reading only a handful of rows.
# predict_pool_shard_worker.py skips rounds with an existing .done marker
# on restart, so a retry only re-does the one stuck round.
#
# Usage (submit BEFORE the matching orchestrator; workers wait patiently on
# round 1 with no timeout until it appears):
#   mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ltall
#   sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_ltall_predict_workers_h100single.sh \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ltall 8

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-N (SLURM_ARRAY_TASK_ID is unset)}"
COORD_DIR="${1:?Usage: sbatch --array=0-N slurm/al_runs/ampc/submit_ampc_ltall_predict_workers_h100single.sh <coord-dir> <num-shards>}"
NUM_SHARDS="${2:?Usage: sbatch --array=0-N slurm/al_runs/ampc/submit_ampc_ltall_predict_workers_h100single.sh <coord-dir> <num-shards>}"

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

TIMEOUT_SECONDS=2700  # 45 min -- same margin as the ensemble/learned workers
attempt=1
while true; do
    echo "[worker $TASK_ID] attempt $attempt (timeout ${TIMEOUT_SECONDS}s)"
    set +e
    srun --cpu-bind=none timeout "${TIMEOUT_SECONDS}s" python3 -u predict_pool_shard_worker.py \
        --dataset AmpC --backbones grover3400 mhgged molformer smited unimol \
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
