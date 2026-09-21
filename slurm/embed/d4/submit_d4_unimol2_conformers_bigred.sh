#!/bin/bash

#SBATCH -J d4_u2_conf
#SBATCH -p general
#SBATCH -o logs/d4_u2_conf_%A_%a.txt
#SBATCH -e logs/d4_u2_conf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 1 of D4's split Uni-Mol2 pipeline: conformer generation only,
# CPU-only (RDKit ETKDG is CPU-bound, no GPU benefit) -- same compute
# script as slurm/embed/ampc/submit_ampc_unimol2_conformers_bigred.sh.
# BigRed200's `general` partition, same reasoning as the AmpC version.
#
# --time=4:00:00 carried over from AmpC's own rough (never independently
# measured for unimol2) extrapolation from Uni-Mol v1's ~19ms/mol
# precedent -- run a small timed sample first (same two-point methodology
# used elsewhere this session, e.g. submit_ampc_unimol2_conformers_timing_probe_bigred.sh)
# to replace this guess with a real number before trusting it at scale.
#
# NUM_CHUNKS=1000 keeps D4's per-chunk density close to AmpC's own
# 99,459,561/1000 (~99.46k mol/chunk): D4's 116,241,184 scored molecules
# at 1000 chunks is ~116.24k mol/chunk, ~17% more per chunk -- fine given
# --time is already a rough estimate with real margin, not a tight budget.
# MaxArraySize may differ from Quartz's 1000 limit on BigRed200; check
# `scontrol show config | grep MaxArraySize` before submitting more than
# 1000 in one --array.
#
# Usage:
#   sbatch --array=0-999%100 slurm/embed/d4/submit_d4_unimol2_conformers_bigred.sh 1000
#
# %100 above is a starting guess carried over from AmpC's own submission
# -- adjust based on actual observed queue behavior on BigRed200.
#
# After all chunks finish, Stage 2 reads directly from these per-chunk
# files (no merge step) -- see
# slurm/embed/d4/submit_d4_unimol2_embed_from_conformers_h100single.sh.
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave.

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/d4/submit_d4_unimol2_conformers_bigred.sh <NUM_CHUNKS> [OFFSET]}"
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

srun --cpu-bind=none python generate_unimol2_conformers_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --out-dir "$D4_ROOT/_unimol2_conformers" \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --num-workers 16
