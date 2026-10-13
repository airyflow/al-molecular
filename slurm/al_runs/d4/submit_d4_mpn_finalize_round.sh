#!/bin/bash

#SBATCH -J d4_mpn_finalize
#SBATCH -p general
#SBATCH -o logs/d4_mpn_finalize_%j.txt
#SBATCH -e logs/d4_mpn_finalize_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=250G
#SBATCH -A r00939

# Finishes one round of the D4 MPN bs1024 run (frac=0.004) on CPU from files
# already on disk, instead of resubmitting the orchestrator, which would
# retrain that round's MPN (~11 h of GPU) only to discard it. See
# finalize_parallel_round.py for exactly what it repeats.
#
# --mem=250G: the oracle dict (SMILES -> score, 116M entries), the full SMILES
# list, and true_top_k_set()'s full sort all live in memory at once -- the
# same structures the orchestrator holds (its own request is 300G). Not
# measured for this script.
#
# Usage:
#   Validate on a round the orchestrator already finished (writes nothing):
#     sbatch slurm/al_runs/d4/submit_d4_mpn_finalize_round.sh 4 check
#   Finish the last round (only after all 8 round_5_shard_*.done exist):
#     sbatch slurm/al_runs/d4/submit_d4_mpn_finalize_round.sh 5
#   Same for the batch-size-50 run: prefix with TAG=bs50, e.g.
#     TAG=bs50 sbatch slurm/al_runs/d4/submit_d4_mpn_finalize_round.sh 5

set -euo pipefail

ROUND="${1:?Usage: sbatch slurm/al_runs/d4/submit_d4_mpn_finalize_round.sh <round> [check]}"
MODE="${2:-}"

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
export CUDA_VISIBLE_DEVICES=""

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"
FRAC=0.004
TAG="${TAG:-bs1024}"   # which run: bs1024 (default) or bs50
N=116225927
count=$(python3 -c "print(round($N * $FRAC))")

CHECK_FLAG=""
[ "$MODE" = "check" ] && CHECK_FLAG="--check"

echo "[finalize] run=$TAG round=$ROUND mode=${MODE:-write} batch-size=$count"
python3 -u finalize_parallel_round.py --dataset D4_dedup \
    --coord-dir "$D4_ROOT/al_coord/mpn_${TAG}_frac${FRAC}" \
    --run-dir "runs/d4_mpn_${TAG}_greedy_frac${FRAC}" \
    --round "$ROUND" --n-rounds 5 --num-shards 8 \
    --batch-size "$count" --topk 50000 --acq greedy $CHECK_FLAG
