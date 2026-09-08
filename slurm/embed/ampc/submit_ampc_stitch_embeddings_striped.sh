#!/bin/bash

#SBATCH -J ampc_stitch_striped
#SBATCH -p general
#SBATCH -o logs/ampc_stitch_striped_%j.txt
#SBATCH -e logs/ampc_stitch_striped_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# Re-stitches a backbone's existing per-chunk .npy files (same
# _${BACKBONE}_chunks/ directories slurm/embed/ampc/submit_ampc_stitch_embeddings.sh already
# reads) into a NEW, Lustre-striped destination file -- NOT a re-extraction,
# pure I/O. Motivation: all 3 of the current embeddings/*.npy files were
# created with the filesystem default stripe count of 1 (confirmed via
# `lfs getstripe -c`, 2026-08-15) -- every scattered fancy-indexed read (the
# access pattern both prediction and AL training-set fetches use at AmpC
# scale) funnels through exactly one OST regardless of how far apart the
# requested rows are. That's the leading suspect for this session's repeated
# multi-hour stalls (shard 7's hangs, the orchestrator's 2+ hour cold-cache
# labeled-set fetch) -- confirmed via nvidia-smi showing 0% GPU utilization
# during one such stall, i.e. it wasn't compute-bound. This filesystem has
# 30 OSTs (`lfs df`); striping across all of them (-c -1) spreads that same
# read pattern across 30 independent disks/paths instead of one.
#
# Writes to embeddings_striped/, NOT the original embeddings/ -- the old
# files are left completely untouched, so this is safe to run while the
# live AmpC AL orchestrator/workers are still reading the old ones.
#
# Individual .npy files can't be re-striped in place (Lustre fixes a file's
# layout at creation, `lfs setstripe` on a file with existing data errors
# out) -- instead we set a DEFAULT stripe policy on the destination
# DIRECTORY, so any new file created inside (via
# shared_embedding_store.preallocate()'s np.lib.format.open_memmap(...,
# mode="w+", ...) call) inherits it automatically at creation. Idempotent:
# safe to run this setstripe line every backbone invocation.
#
# Usage (run once per backbone, any order, no dependency between them):
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh grover
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh molformer
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh unimol
#
# After all 3 finish AND are verified (spot-check against the old files --
# do NOT just trust this ran without erroring), load_embeddings() in
# run_experiment.py needs to be pointed at embeddings_striped/ instead of
# embeddings/ -- not done automatically by this script.

set -euo pipefail

BACKBONE="${1:?Usage: sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh <grover|molformer|unimol|smited|mhgged|unimol2> [NUM_CHUNKS]}"

# grover/molformer/unimol were extracted at fixed chunk counts (hardcoded
# below). smited/mhgged/unimol2 take NUM_CHUNKS as $2 -- it MUST equal the
# value passed to their extract array job, or _chunk_bounds() will not line
# up and the stitch will error on a row-count mismatch.
case "$BACKBONE" in
    grover)     DIM=1600; NUM_CHUNKS=150 ;;
    molformer)  DIM=768;  NUM_CHUNKS=50  ;;
    unimol)     DIM=512;  NUM_CHUNKS=70  ;;
    smited)     DIM=768;  NUM_CHUNKS="${2:?smited needs NUM_CHUNKS as \$2 (same value as the extract job)}" ;;
    mhgged)     DIM=1024; NUM_CHUNKS="${2:?mhgged needs NUM_CHUNKS as \$2 (same value as the extract job)}" ;;
    unimol2)    DIM=1536; NUM_CHUNKS="${2:?unimol2 needs NUM_CHUNKS as \$2 (same value as the extract job)}" ;;
    grover3400) DIM=3400; NUM_CHUNKS="${2:?grover3400 needs NUM_CHUNKS as \$2 (same value as the extract job)}" ;;
    *) echo "Unknown backbone '$BACKBONE'" >&2; exit 1 ;;
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
STRIPED_DIR="$AMPC_ROOT/embeddings_striped"

mkdir -p "$STRIPED_DIR"
lfs setstripe -c -1 "$STRIPED_DIR"
echo "[stripe] $STRIPED_DIR default layout: $(lfs getstripe -c "$STRIPED_DIR")-way (new files created inside inherit this)"

python embed/stitch/stitch_embedding_chunks.py \
    --backbone "$BACKBONE" --dim "$DIM" \
    --chunks-dir "$AMPC_ROOT/_${BACKBONE}_chunks" \
    --num-chunks "$NUM_CHUNKS" --total-count 99459561 \
    --embeddings-path "$STRIPED_DIR/${BACKBONE}_embeddings.npy"

echo "[stripe] final layout of new file: $(lfs getstripe -c "$STRIPED_DIR/${BACKBONE}_embeddings.npy")-way"
