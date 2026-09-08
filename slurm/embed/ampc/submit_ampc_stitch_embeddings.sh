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
# files (written by the parallel array jobs in slurm/embed/ampc/submit_ampc_grover_extract.sh
# / slurm/embed/ampc/submit_ampc_molformer_extract.sh / slurm/embed/ampc/submit_ampc_unimol_embed.sh) into the
# single shared (N, D) .npy that EmbeddingFeaturizer.load()
# (molpal/featurizer.py) expects. See embed/stitch/stitch_embedding_chunks.py's
# docstring: deliberately NOT an array job -- one process, one chunk copied
# into RAM at a time (bounded to ~6GB for MoLFormer's largest AmpC chunk),
# resumable via a sidecar .stitch_progress file if killed/restarted.
#
# NOT --array -- run once per backbone, after that backbone's array job has
# fully finished (all chunk files present under --chunks-dir).
#
# Usage:
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings.sh grover
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings.sh molformer
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings.sh unimol

set -euo pipefail

BACKBONE="${1:?Usage: sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings.sh <grover|molformer|unimol>}"

case "$BACKBONE" in
    grover)    DIM=1600; NUM_CHUNKS=150 ;;
    molformer) DIM=768;  NUM_CHUNKS=50  ;;
    unimol)    DIM=512;  NUM_CHUNKS=70  ;;
    *) echo "Unknown backbone '$BACKBONE' -- expected grover, molformer, or unimol" >&2; exit 1 ;;
esac

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Under sbatch, BASH_SOURCE[0] is not reliable -- sbatch may spool the
    # submitted script to an internal location disconnected from both its
    # original path and the submission directory (observed directly on
    # this cluster: a real job's git-based resolution below failed with
    # "fatal: not a git repository" on every one of its array tasks).
    # SLURM always sets SLURM_SUBMIT_DIR to the real directory `sbatch` was
    # invoked from -- every script in this repo is documented to be
    # submitted from the repo root, so this is exactly the repo root.
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

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

python embed/stitch/stitch_embedding_chunks.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --chunks-dir "$AMPC_ROOT/_${BACKBONE}_chunks" \
    --num-chunks "$NUM_CHUNKS" --total-count 99459561 \
    --embeddings-path "$AMPC_ROOT/embeddings/${BACKBONE}_embeddings.npy"
