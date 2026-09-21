#!/bin/bash

#SBATCH -J d4_dedup_mask
#SBATCH -p general
#SBATCH -o logs/d4_dedup_mask_%j.txt
#SBATCH -e logs/d4_dedup_mask_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -A r00939

# Builds D4's dedup keep-mask (embed/stitch/build_dedup_mask.py) -- SMILES
# only, no scores needed, so this can run now regardless of the D4 scores
# blocker. Same "keep only the LAST occurrence of each duplicate SMILES"
# rule as AmpC's own mask (build_dedup_mask.py's docstring), for the same
# reason: run_experiment.py's oracle/smi2idx dicts are already
# last-write-wins, so dropping every non-last occurrence changes no
# existing lookup behavior.
#
# --mem=64G: a single Python dict over 116,241,184 SMILES strings --
# AmpC's own dict (99,459,561 keys) was not independently memory-profiled
# in this repo's record, so this is a margin-padded guess, not a measured
# number; raise if this OOMs.
#
# Usage: sbatch slurm/embed/d4/submit_d4_build_dedup_mask.sh

set -euo pipefail

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

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

python embed/stitch/build_dedup_mask.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" \
    --out-path "$D4_ROOT/d4_dedup_keep_mask.npy"
