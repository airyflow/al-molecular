#!/bin/bash

#SBATCH -J ampc_stitch
#SBATCH -p general
#SBATCH -o logs/ampc_stitch_%j.txt
#SBATCH -e logs/ampc_stitch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# Sequential merge step: copies one backbone's independent per-chunk .npy
# files (written by the parallel array jobs in submit_ampc_grover_extract.sh
# / submit_ampc_molformer_extract.sh / submit_ampc_unimol_embed.sh) into the
# single shared (N, D) .npy that EmbeddingFeaturizer.load()
# (molpal/featurizer.py) expects. See stitch_embedding_chunks.py's
# docstring: deliberately NOT an array job -- one process, one chunk copied
# into RAM at a time (bounded to ~6GB for MoLFormer's largest AmpC chunk),
# resumable via a sidecar .stitch_progress file if killed/restarted.
#
# NOT --array -- run once per backbone, after that backbone's array job has
# fully finished (all chunk files present under --chunks-dir).
#
# Usage:
#   sbatch submit_ampc_stitch_embeddings.sh grover
#   sbatch submit_ampc_stitch_embeddings.sh molformer
#   sbatch submit_ampc_stitch_embeddings.sh unimol

set -euo pipefail

BACKBONE="${1:?Usage: sbatch submit_ampc_stitch_embeddings.sh <grover|molformer|unimol>}"

case "$BACKBONE" in
    grover)    DIM=1600; NUM_CHUNKS=150 ;;
    molformer) DIM=768;  NUM_CHUNKS=50  ;;
    unimol)    DIM=512;  NUM_CHUNKS=70  ;;
    *) echo "Unknown backbone '$BACKBONE' -- expected grover, molformer, or unimol" >&2; exit 1 ;;
esac

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

python stitch_embedding_chunks.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --chunks-dir "$AMPC_ROOT/_${BACKBONE}_chunks" \
    --num-chunks "$NUM_CHUNKS" --total-count 99459561 \
    --embeddings-path "$AMPC_ROOT/embeddings/${BACKBONE}_embeddings.npy"
