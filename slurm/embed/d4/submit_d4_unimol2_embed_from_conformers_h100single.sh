#!/bin/bash

#SBATCH -J d4_u2_embed
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_u2_embed_%A_%a.txt
#SBATCH -e logs/d4_u2_embed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# Stage 2 of D4's split Uni-Mol2 pipeline: compute embeddings from
# pre-generated conformers (see submit_d4_unimol2_conformers_bigred.sh,
# Stage 1 -- run on BigRed200's CPU-only `general` partition). Same
# compute script as slurm/embed/ampc/submit_ampc_unimol2_embed_from_conformers_h100single.sh
# -- stays 100% GPU-forward-pass-bound, no inline RDKit conformer
# generation competing with the model for this task's CPUs.
#
# --num-chunks MUST match Stage 1's NUM_CHUNKS exactly -- chunk boundaries
# have to line up with the conformer LMDB files on disk, or
# compute_unimol2_embeddings_chunk.py's own assert will refuse to produce
# embeddings rather than silently misalign SMILES with the wrong
# conformers.
#
# --time=4:00:00 carried over from AmpC's own value -- not independently
# re-measured for D4's ~17% larger per-chunk size at the same NUM_CHUNKS.
#
# Usage (only after Stage 1's conformer chunks exist for this chunk-id):
#   sbatch --array=0-999 slurm/embed/d4/submit_d4_unimol2_embed_from_conformers_h100single.sh 1000
#
# No %N concurrency throttle -- same reasoning as the AmpC version, let
# SLURM's own scheduler/QOS decide how many run at once.
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave.
#
# After all chunks exist, stitch (dim 1536, SAME NUM_CHUNKS):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh unimol2 1000

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/d4/submit_d4_unimol2_embed_from_conformers_h100single.sh <NUM_CHUNKS> [OFFSET]}"
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

D4_ROOT="${D4_ROOT:-/N/project/SingleCell_Image/mengjing/d4_138M}"

srun --cpu-bind=none python embed/compute/compute_unimol2_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_unimol2_chunks" \
    --conformer-chunks-dir "$D4_ROOT/_unimol2_conformers/_chunks" \
    --batch-size 32
