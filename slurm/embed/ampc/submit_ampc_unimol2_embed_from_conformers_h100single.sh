#!/bin/bash

#SBATCH -J ampc_u2_embed
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_u2_embed_%A_%a.txt
#SBATCH -e logs/ampc_u2_embed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# Stage 2 of the split Uni-Mol2 pipeline: compute embeddings from
# pre-generated conformers (see submit_ampc_unimol2_conformers_bigred.sh,
# Stage 1 -- run on BigRed200's CPU-only `general` partition). This script
# stays 100% GPU-forward-pass-bound: no inline RDKit conformer generation
# competing with the model for this task's CPUs, unlike
# submit_ampc_unimol2_extract.sh's original single-stage design.
#
# Verified: two-stage mode produces bit-exact identical output to the
# original single-stage script (max abs diff 0.0 on a 3-molecule sample,
# 2026-09-08) -- this is a decoupling of where the compute happens, not a
# different computation.
#
# --num-chunks MUST match Stage 1's NUM_CHUNKS exactly -- chunk boundaries
# have to line up with the conformer LMDB files on disk, or
# compute_unimol2_embeddings_chunk.py's own assert will refuse to produce
# embeddings rather than silently misalign SMILES with the wrong
# conformers.
#
# Each task writes its own independent chunk file -- no shared state, no
# coordination, no race condition possible between parallel tasks. After
# all chunks finish, run embed/stitch/stitch_embedding_chunks.py once (see
# submit_ampc_stitch_embeddings_striped.sh unimol2 <NUM_CHUNKS>) to produce
# the final shared .npy that EmbeddingFeaturizer.load() expects.
#
# Usage (only after Stage 1's conformer chunks exist for this chunk-id):
#   sbatch --array=0-999 slurm/embed/ampc/submit_ampc_unimol2_embed_from_conformers_h100single.sh 1000
#
# No %N concurrency throttle -- the ~40-concurrent figure observed
# elsewhere this session wasn't a hard ceiling, just what happened to be
# free at the time; let SLURM's own scheduler/QOS decide how many run at
# once rather than artificially capping it.
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave.

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 submit_ampc_unimol2_embed_from_conformers_h100single.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do"
    exit 0
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # See submit_ampc_unimol2_conformers_bigred.sh's comment on why
    # SLURM_SUBMIT_DIR is preferred over the BASH_SOURCE-based git trick
    # under sbatch.
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

AMPC_ROOT="${AMPC_ROOT:-/N/project/SingleCell_Image/mengjing/ampc_99.5M}"

srun --cpu-bind=none python embed/compute/compute_unimol2_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_unimol2_chunks" \
    --conformer-chunks-dir "$AMPC_ROOT/_unimol2_conformers/_chunks" \
    --batch-size 32
