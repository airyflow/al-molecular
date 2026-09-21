#!/bin/bash

#SBATCH -J d4_stitch
#SBATCH -p general
#SBATCH -o logs/d4_stitch_%j.txt
#SBATCH -e logs/d4_stitch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# D4 equivalent of slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh
# -- re-stitches a backbone's per-chunk .npy files into a single Lustre-striped
# (N, D) .npy, same rationale (scattered fancy-indexed reads at this pool
# scale funnel through one OST otherwise -- see the AmpC script's own
# comment for the full story). Writes to embeddings_striped/ under D4_ROOT.
#
# UNLIKE the AmpC script, every backbone here takes NUM_CHUNKS as $2 (no
# hardcoded per-backbone defaults) -- D4's extraction scripts all take
# NUM_CHUNKS as a real CLI arg (see submit_d4_*_extract.sh), so there's no
# AmpC-style inline-hardcoded chunk count to fall back on; you must pass
# whatever NUM_CHUNKS you actually used for that backbone's extraction.
#
# Usage (run once per backbone, any order, no dependency between them,
# after ALL of that backbone's chunks exist):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh mhgged 1000
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh molformer 70
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh smited 700
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh grover3400 1000
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh unimol 70
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh unimol2 1000

set -euo pipefail

BACKBONE="${1:?Usage: sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh <backbone> <NUM_CHUNKS>}"
NUM_CHUNKS="${2:?Usage: sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh <backbone> <NUM_CHUNKS> -- must match the value used when extracting that backbone}"

case "$BACKBONE" in
    grover3400) DIM=3400 ;;
    mhgged)     DIM=1024 ;;
    molformer)  DIM=768  ;;
    smited)     DIM=768  ;;
    unimol)     DIM=512  ;;
    unimol2)    DIM=1536 ;;
    *) echo "Unknown backbone '$BACKBONE'" >&2; exit 1 ;;
esac

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
STRIPED_DIR="$D4_ROOT/embeddings_striped"

mkdir -p "$STRIPED_DIR"
lfs setstripe -c -1 "$STRIPED_DIR"
echo "[stripe] $STRIPED_DIR default layout: $(lfs getstripe -c "$STRIPED_DIR")-way (new files created inside inherit this)"

python embed/stitch/stitch_embedding_chunks.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --chunks-dir "$D4_ROOT/_${BACKBONE}_chunks" \
    --num-chunks "$NUM_CHUNKS" --total-count 116241184 \
    --embeddings-path "$STRIPED_DIR/${BACKBONE}_embeddings.npy"

echo "[stripe] final layout of new file: $(lfs getstripe -c "$STRIPED_DIR/${BACKBONE}_embeddings.npy")-way"
