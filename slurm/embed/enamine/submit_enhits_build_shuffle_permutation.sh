#!/bin/bash

#SBATCH -J enhits_shuf_perm
#SBATCH -p general
#SBATCH -o logs/enhits_shuf_perm_%j.txt
#SBATCH -e logs/enhits_shuf_perm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH -A r00939

# Step 1 of the ENHITS order-leakage robustness check (report Section 10):
# generates one fixed random permutation of ENHITS's 2,104,318 rows and
# shuffles the small (89MB) canonical SMILES file directly. Cheap/fast --
# the expensive part is shuffling each backbone's (large) embeddings.npy,
# see submit_enhits_shuffle_embeddings.sh, which needs this step's saved
# permutation.npy first.
#
# Usage: sbatch slurm/embed/enamine/submit_enhits_build_shuffle_permutation.sh

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

ENHITS_ROOT="/N/project/SingleCell_Image/mengjing/enhits_large"

python3 embed/stitch/build_shuffle_permutation.py \
    --smiles-file "$ENHITS_ROOT/embed/enhits_large_smiles.txt" \
    --out-dir "$ENHITS_ROOT/embed_shuffled" \
    --seed 42
