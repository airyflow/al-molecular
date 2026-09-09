#!/bin/bash

#SBATCH -J ampc_u2_conf
#SBATCH -p general
#SBATCH -o logs/ampc_u2_conf_%A_%a.txt
#SBATCH -e logs/ampc_u2_conf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 1 of the split Uni-Mol2 pipeline: conformer generation only, CPU-only
# (RDKit ETKDG is CPU-bound, no GPU benefit) -- see
# generate_unimol2_conformers_chunk.py's module docstring for the full
# design rationale. BigRed200's `general` partition (4-day limit, ~497
# nodes with free capacity at submission time, 2026-09-08's sinfo) --
# unlike Quartz's h100-single, which is scarce/contended and reserved for
# Stage 2's actual GPU forward pass.
#
# --time=4:00:00 is a ROUGH estimate, not yet measured for unimol2
# specifically -- extrapolated from Uni-Mol v1's own real precedent
# (al-eval-framework: ~1.3B molecules / 1000 shards / ~7h, i.e. ~19ms/mol
# for RDKit conformer generation at n_conformer=1). At NUM_CHUNKS=1000
# (~99.46k mol/chunk) that suggests ~41min/chunk -- run a small timed
# sample first (same two-point methodology used elsewhere this session)
# to replace this guess with a real number before trusting it at scale,
# the same way the GPU-side estimate for this exact backbone turned out
# to need correcting once measured for real.
#
# NUM_CHUNKS=1000 matches Stage 2's existing chunk convention (99,459,561
# total / 1000 = ~99.46k mol/chunk) -- MaxArraySize may differ from
# Quartz's 1000 limit; check `scontrol show config | grep MaxArraySize`
# on BigRed200 before submitting more than 1000 in one --array.
#
# Usage:
#   sbatch --array=0-999%100 slurm/embed/ampc/submit_ampc_unimol2_conformers_bigred.sh 1000
#
# %100 above is a starting guess given ~497 nodes showed "mix" (partially
# free) state at submission time -- adjust based on actual observed queue
# behavior (fair-share/QOS limits aren't visible from sinfo alone).
#
# After all chunks finish, Stage 2 reads directly from these per-chunk
# files (no merge step) -- see
# slurm/embed/ampc/submit_ampc_unimol2_embed_from_conformers_h100single.sh.
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave.

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 submit_ampc_unimol2_conformers_bigred.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do"
    exit 0
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Under sbatch, BASH_SOURCE[0] is not reliable -- sbatch may spool the
    # submitted script to an internal location disconnected from both its
    # original path and the submission directory (observed directly on
    # Quartz: a real job's git-based resolution below failed with "fatal:
    # not a git repository" on every one of its array tasks). SLURM always
    # sets SLURM_SUBMIT_DIR to the real directory `sbatch` was invoked
    # from -- every script in this repo is documented to be submitted
    # from the repo root, so this is exactly the repo root.
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

srun --cpu-bind=none python generate_unimol2_conformers_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --out-dir "$AMPC_ROOT/_unimol2_conformers" \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --num-workers 16
