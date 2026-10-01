#!/bin/bash

#SBATCH -J d4_dedup_emb
#SBATCH -p general
#SBATCH -o logs/d4_dedup_emb_%j.txt
#SBATCH -e logs/d4_dedup_emb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# Applies D4's dedup keep-mask (built via submit_d4_build_dedup_mask.sh,
# 116,225,927/116,241,184 kept -- see run_experiment.py's D4_dedup entry)
# to one backbone's stitched (N, D) embeddings.npy
# (embed/stitch/dedup_embeddings.py), writing the deduplicated copy into
# $D4_ROOT/dedup/ that the D4_dedup dataset entry actually reads.
#
# Run once per backbone, any order, as soon as that backbone's
# embeddings_striped/<backbone>_embeddings.npy exists (no need to wait for
# all six) -- e.g. smited is already fully stitched.
#
# --time=3:00:00: smited's own stitch step (same full-pool sequential
# scan-and-write pattern, 116.24M rows) took 2,254s; dedup only drops
# 15,257/116,241,184 rows (~0.01%) so this is essentially the same cost as
# a full copy -- budgeted with margin, not independently measured for this
# script.
#
# Usage (dim must match the backbone -- same table as submit_d4_stitch_embeddings.sh):
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh smited 768
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh mhgged 1024
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh molformer 768
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh grover3400 3400
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh unimol 512
#   sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh unimol2 1536

set -euo pipefail

BACKBONE="${1:?Usage: sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh <backbone> <dim>}"
DIM="${2:?Usage: sbatch slurm/embed/d4/submit_d4_dedup_embeddings.sh <backbone> <dim>}"

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
mkdir -p "$D4_ROOT/dedup"

python embed/stitch/dedup_embeddings.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --keep-mask "$D4_ROOT/d4_dedup_keep_mask.npy" \
    --embeddings-path "$D4_ROOT/embeddings_striped/${BACKBONE}_embeddings.npy" \
    --out-path "$D4_ROOT/dedup/${BACKBONE}_embeddings.npy" \
    --resume
