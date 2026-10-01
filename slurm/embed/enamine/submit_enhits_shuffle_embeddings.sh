#!/bin/bash

#SBATCH -J enhits_shuf_emb
#SBATCH -p general
#SBATCH -o logs/enhits_shuf_emb_%A_%a.txt
#SBATCH -e logs/enhits_shuf_emb_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH -A r00939

# Step 2 of the ENHITS order-leakage robustness check: applies the
# permutation from submit_enhits_build_shuffle_permutation.sh to one
# backbone's embeddings.npy. Array task index selects the backbone (5
# total, matching ENHITS's own set) -- run once
# submit_enhits_build_shuffle_permutation.sh has finished and
# embed_shuffled/shuffle_permutation.npy exists.
#
# CPU-only, I/O-bound (sequential source reads, scattered destination
# writes) -- no GPU needed. --mem=64G (raised from an initial 16G, which
# was too tight for grover: its per-block memory footprint is ~6.8GB
# (500k rows x 3400-d x 4 bytes) vs. ~1.5-2GB for the other 4 backbones,
# and the scattered-write pattern touches pages across the WHOLE
# destination file over many iterations rather than a bounded growing
# prefix, so resident memory can accumulate well past one block's size --
# grover specifically stalled for hours under 16G (confirmed: restriping
# both source and destination changed nothing, ruling out Lustre I/O as
# the actual bottleneck) before this was identified as the real cause.
#
# Usage: sbatch --array=0-4 slurm/embed/enamine/submit_enhits_shuffle_embeddings.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-4}"

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

BACKBONES=(grover mhgged molformer smited unimol2)
DIMS=(3400 1024 768 768 768)
BACKBONE="${BACKBONES[$TASK_ID]}"
DIM="${DIMS[$TASK_ID]}"

echo "[task $TASK_ID] backbone=$BACKBONE dim=$DIM"

python3 embed/stitch/shuffle_embeddings.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --permutation "$ENHITS_ROOT/embed_shuffled/shuffle_permutation.npy" \
    --embeddings-path "$ENHITS_ROOT/embed/${BACKBONE}_embeddings.npy" \
    --out-path "$ENHITS_ROOT/embed_shuffled/${BACKBONE}_embeddings.npy" \
    --resume
