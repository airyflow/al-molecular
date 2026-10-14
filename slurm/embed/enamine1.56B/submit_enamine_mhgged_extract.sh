#!/bin/bash

#SBATCH -J er_mhgged
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=FAIL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/er_mhgged_%A_%a.txt
#SBATCH -e logs/er_mhgged_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# MHG-GED embeddings for the Enamine REAL lead-like pool
# (1,531,875,935 deduplicated, shuffled molecules).
#
# One of the three cheap backbones chosen to go first at this scale.
# Measured per-molecule rates from D4's own 1000-chunk runs, and what they
# cost over this pool:
#   SMI-TED     0.35 ms/mol      149 GPU-h    4.7 TB
#   Uni-Mol v1  1.76 ms/mol      749 GPU-h    3.1 TB  (conformers already done)
#   MHG-GED     2.12 ms/mol      902 GPU-h    6.3 TB
#   ---- the expensive two, deferred ----
#   Uni-Mol2   11.39 ms/mol    4,847 GPU-h    9.4 TB
#   MolFormer  16.77 ms/mol    7,136 GPU-h    4.7 TB
#   GROVER3400 16.97 ms/mol    7,221 GPU-h   20.8 TB
# All six would be 21,004 GPU-h and 49 TB; these three are 1,800 GPU-h and
# 14.1 TB -- about a day on 32 concurrent H100s, nine on 8.
#
# This one: MHG-GED at 2.12 ms/mol = 902 GPU-h, 6.3 TB at 1024 dims.
# At NUM_CHUNKS=1000 that is 1,531,875 molecules/chunk, ~54 min each,
# so --time=8:00:00 holds with wide margin (8.9x the median chunk).
# Wall time is set by the %K concurrency cap: ~14 h at %64.
#
# Reads the deduplicated, shuffled pool, so no embedding is computed for
# the 27,977,307 duplicate rows and row i of the output is row i of the
# pool -- no permutation bookkeeping. Requires submit_enamine_build_pool.sh
# to have finished.
#
# IMPORTANT: if you change NUM_CHUNKS after a partial run, wipe
# $ER_ROOT/_mhgged_chunks first -- the compute script skips an existing chunk
# file by name without checking its row count, so chunks left from a
# different NUM_CHUNKS would be silently wrong.
#
# Usage:
#   sbatch --array=0-999%64 slurm/embed/enamine1.56B/submit_enamine_mhgged_extract.sh 1000
# After ALL chunks exist, stitch with the SAME NUM_CHUNKS (dim 1024).

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999%64 $0 <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do for this task"
    exit 0
fi

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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

ER_ROOT="${ER_ROOT:-/N/project/SingleCell_Image/mengjing/enamine_real_1.56B}"
SMILES="$ER_ROOT/dedup/enamine_surA_smiles.txt"
TOTAL=1531875935

[ -f "$SMILES" ] || { echo "[mhgged] ERROR: $SMILES not found -- run submit_enamine_build_pool.sh first"; exit 1; }

# Stripe the chunk folder before the first write (6.3 TB): single-stripe
# output had to be migrated afterwards on AmpC and D4, 0.8-18.7 h per file
# (report sect. 16).
mkdir -p "$ER_ROOT/_mhgged_chunks"
lfs setstripe -c -1 "$ER_ROOT/_mhgged_chunks" 2>/dev/null || true

echo "[mhgged] chunk $TASK_ID/$NUM_CHUNKS  total=$TOTAL  dims=1024"

srun --cpu-bind=none python embed/compute/compute_mhgged_embeddings_chunk.py \
    --smiles-file "$SMILES" --total-count "$TOTAL" \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$ER_ROOT/_mhgged_chunks"
