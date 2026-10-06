#!/bin/bash

#SBATCH -J d4_mpn_par_workers
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_mpn_par_worker_%A_%a.txt
#SBATCH -e logs/d4_mpn_par_worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# D4 equivalent of AmpC's submit_ampc_mpn_parallel_workers_h100single.sh --
# prediction workers for ONE round of a D4 MPN parallel run (either
# batch_size, since batch_size never affects prediction -- test_batch_size
# is set independently and workers never call .train()). Each array task
# predicts its own shard of the 116,225,927-molecule deduplicated D4 pool
# with the round's MPN checkpoint, then exits -- submit once per round,
# when that round's round_R_ready.marker exists in the coord-dir.
#
# --time=14:00:00: copied directly from AmpC's own budget (measured there
# at ~1.7 ms/molecule, 8 shards). NOT independently re-measured for D4 --
# D4's deduped pool (116,225,927) is ~18% larger than AmpC's (98,489,350),
# so each shard is also ~18% larger; if this run shows shards timing out
# near the limit, raise --time on the CLI (overrides the script's own
# #SBATCH directive, no edit needed) before resubmitting those indices.
#
# Idempotent: a shard whose .done marker exists is skipped, so resubmitting
# the same round only redoes missing/failed shards (submit just those array
# indices, e.g. --array=3,5). Up to 3 attempts per task for transient GPU
# faults.
#
# Usage:
#   sbatch --array=0-7 slurm/al_runs/d4/submit_d4_mpn_parallel_workers_h100single.sh <coord-dir> <num-shards> <round>

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-(NUM_SHARDS-1)}"
COORD_DIR="${1:?Usage: sbatch --array=0-N slurm/al_runs/d4/submit_d4_mpn_parallel_workers_h100single.sh <coord-dir> <num-shards> <round>}"
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
        --dataset D4_dedup --total-count 116225927 \
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
