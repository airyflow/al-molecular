#!/bin/bash

#SBATCH -J d4_smited
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_smited_%A_%a.txt
#SBATCH -e logs/d4_smited_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Sharded SMI-TED (dim 768) embedding extraction for D4 -- same compute
# script/reasoning as slurm/embed/ampc/submit_ampc_smited_extract.sh
# (batched .encode() on GPU, fast, no conformer stage), repointed at D4's
# own smiles file/root/total-count. Not independently re-measured for D4
# in this repo's own record -- budget a timing check on the first
# completed shard before trusting the chunk count below at scale.
#
# total-count 116241184 -- see submit_d4_mhgged_extract.sh's comment for
# why this is D4's scored-only count (from d4_enamine_style.csv /
# d4_smiles.txt), not the raw d4.csv's 138,312,677.
#
# This cluster's MaxArraySize is 1000. D4's 116,241,184 scored molecules
# are ~1.17x AmpC's 99,459,561, so 600 chunks (~194k mol each) keeps
# roughly AmpC's own per-chunk density (500 chunks/~199k mol each) --
# adjust after checking the first shard's actual throughput.
#
# Usage (one array wave):
#   sbatch --array=0-599%64 slurm/embed/d4/submit_d4_smited_extract.sh 600
#   # resubmit the same line after a partial run -- finished chunks self-skip
#
# Then stitch once, SAME NUM_CHUNKS (dim 768):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh smited 600
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# needed if NUM_CHUNKS > 1000 (a second wave for the high half). If you
# change NUM_CHUNKS after a partial run, wipe $D4_ROOT/_smited_chunks
# first (chunks are skipped by name, not row count).

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/d4/submit_d4_smited_extract.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do"
    exit 0
fi

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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

srun --cpu-bind=none python embed/compute/compute_smited_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_smited_chunks" \
    --batch-size 512
