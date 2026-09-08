#!/bin/bash

#SBATCH -J ampc_unimol2
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_unimol2_%A_%a.txt
#SBATCH -e logs/ampc_unimol2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# Sharded Uni-Mol2 (1.1B-param port, unimol2/ package, dim 1536) embedding
# extraction for the AmpC 99.5M pool.
#
# Two costs per chunk: RDKit conformer generation (CPU, parallelised by
# --num-conformer-workers) then the 1.1B forward pass (GPU, batch 16).
# Measured directly on an idle A100 (2026-09-08, warm cache, two-point
# fit from 500 and 2000-molecule samples -- NOT the ~10-25 ms/mol guess
# this comment previously carried, which undershot by ~3x): fixed
# overhead ~13.8s/chunk (checkpoint load), steady-state ~71.3 ms/mol.
# At NUM_CHUNKS=1000 (~99.46k mol/chunk): ~1,975 GPU-hours total (~82
# GPU-days), ~2.0h/chunk wall-clock. A cold first invocation on a given
# node (first CUDA context, cold page cache for the ~4GB+ checkpoint)
# costs meaningfully more than this -- expect the first few chunks on any
# given node to run slower than steady-state.
#
# --cpus-per-task 16 so --num-conformer-workers can be raised; bump batch
# size only if GPU memory allows (1.1B params + attention over conformers
# is heavy). Each task writes its own independent
# _unimol2_chunks/unimol2_embeddings_chunk_NNNNN.npy; resubmit skips
# finished chunks.
#
# total-count 99459561 -- identical to every other AmpC extraction script.
#
# MaxArraySize is 1000 here -- keep NUM_CHUNKS <= 1000, one wave.
# Usage:
#   sbatch --array=0-999%40 slurm/embed/ampc/submit_ampc_unimol2_extract.sh 1000
# Then, SAME NUM_CHUNKS (dim 1536):
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh unimol2 1000
#
# %40 above matches this cluster's real observed concurrent h100-single/
# GPU-node ceiling (seen repeatedly this session across other AmpC
# extraction jobs, well below whatever %N is requested) -- at ~2.0h/chunk
# and ~40 concurrent tasks, expect roughly (1000/40)*2.0h =~ 50h wall-clock
# for the whole array, assuming steady node availability (real queue-wait
# will very likely stretch this further; check node availability at
# submission time).
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave. Changing NUM_CHUNKS after
# a partial run: wipe $AMPC_ROOT/_unimol2_chunks first (skipped by name).

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/ampc/submit_ampc_unimol2_extract.sh <NUM_CHUNKS> [OFFSET]}"
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
    # this cluster: job 10253939's git-based resolution below failed with
    # "fatal: not a git repository" on every one of its 40 array tasks).
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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python embed/compute/compute_unimol2_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_unimol2_chunks" \
    --batch-size 16 --num-conformer-workers 12 --timeout-s 30
