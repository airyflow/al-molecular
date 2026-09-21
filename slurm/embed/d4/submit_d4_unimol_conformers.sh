#!/bin/bash

#SBATCH -J d4_conformers
#SBATCH -p general
#SBATCH -o logs/d4_conformers_%A_%a.txt
#SBATCH -e logs/d4_conformers_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 1 of D4's Uni-Mol v1 pipeline: conformer generation only, CPU-only
# (RDKit ETKDG is CPU-bound, no GPU benefit). Same compute script as
# slurm/embed/ampc/submit_ampc_unimol_conformers.sh
# (generate_unimol_conformers_chunk.py) -- verified byte-identical in its
# core LMDB key format/record structure to al-eval-framework's own
# generate_conformers.py, which validated this exact design at a real
# ~1.3B-molecule / 1000-chunk / ~7h-per-chunk scale, larger and more
# directly comparable to D4 (116.24M) than AmpC's own 99.46M precedent.
#
# UNLIKE the AmpC version, NUM_CHUNKS is a real positional CLI arg here
# (not hardcoded inline) -- D4's 116,241,184 scored molecules are ~1.17x
# AmpC's 99,459,561, so ~82 chunks (70*1.17, ~1.42M mol/chunk) keeps the
# same known-good per-chunk density AmpC used -- --time 12h leaves the
# same margin AmpC's own run budgeted over the ~7h/chunk precedent this
# density is based on.
#
# Usage:
#   sbatch --array=0-81 slurm/embed/d4/submit_d4_unimol_conformers.sh 82
#
# After all NUM_CHUNKS finish, Stage 2 (submit_d4_unimol_embed_h100single.sh)
# reads directly from these per-chunk files -- no merge step required.

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-N slurm/embed/d4/submit_d4_unimol_conformers.sh <NUM_CHUNKS>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-(NUM_CHUNKS-1)}"

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

# 16 conformer-generation worker processes (num_workers=16 below) each get
# their own BLAS threads capped to 4 -- same oversubscription guard proven
# necessary in al-eval-framework's conformer generation jobs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

D4_ROOT="${D4_ROOT:-/N/project/SingleCell_Image/mengjing/d4_138M}"

srun --cpu-bind=none python generate_unimol_conformers_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --out-dir "$D4_ROOT/_unimol_conformers" \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" --n-conformer 1 \
    --num-workers 16
