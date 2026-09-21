#!/bin/bash

#SBATCH -J d4_grover3400
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_grover3400_%A_%a.txt
#SBATCH -e logs/d4_grover3400_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:30:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Real GROVER (tencent-ailab/grover) 3400-d "both" fingerprint extraction
# for the D4 pool (116,241,184 molecules -- see submit_d4_mhgged_extract.sh's
# comment for why this is D4's scored-only count, not the raw d4.csv's
# 138,312,677) -- same compute script/checkpoint as
# slurm/embed/ampc/submit_ampc_grover3400_extract.sh, repointed at D4.
#
# Using AmpC's OWN measured rate (idle h100-single, 2026-09-05: fixed
# overhead ~53s/chunk, steady-state ~34.5 ms/mol) as a starting estimate --
# NOT independently re-measured for D4 in this repo's own record, and this
# repo's own docs warn cross-dataset extrapolation has been off by 3x+
# before. At NUM_CHUNKS=1000 (~116.24k mol/chunk): ~53s + ~4,010s =
# ~68 min/chunk, ~1,129 GPU-hours total, ~28h wall at ~40 concurrent tasks
# (this cluster's real concurrency ceiling, well below whatever %N is
# requested -- same finding AmpC's own extraction hit). Check the first
# completed chunk's actual wall time before trusting the rest of the array.
#
# This cluster's MaxArraySize is 1000, so ONE array wave covers indices
# 0..999. Use NUM_CHUNKS=1000 and a single submission:
#
#   sbatch --array=0-999%64 slurm/embed/d4/submit_d4_grover3400_extract.sh 1000
#
# OFFSET (2nd positional arg, default 0): real chunk id = ARRAY_TASK_ID +
# OFFSET -- only needed for a deliberate NUM_CHUNKS > 1000 second wave.
#
# --checkpoint-path defaults to the same shared model.pt AmpC's extraction
# used (not AmpC-data-specific, safe to reuse as-is); override with a 3rd
# positional arg if a different checkpoint is ever needed.
#
# IMPORTANT: if you change NUM_CHUNKS after a partial run, wipe
# $D4_ROOT/_grover3400_chunks first -- the compute script skips any
# existing chunk file by name without checking its row count, so leftover
# chunks from a different NUM_CHUNKS would be silently wrong.
#
# After ALL chunks exist (pass the SAME NUM_CHUNKS, dim 3400):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh grover3400 1000

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/d4/submit_d4_grover3400_extract.sh <NUM_CHUNKS> [OFFSET] [CHECKPOINT_PATH]}"
OFFSET="${2:-0}"
CHECKPOINT_PATH="${3:-/N/project/SingleCell_Image/Yang/AI Drug/Emb output/model.pt}"
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

D4_ROOT="${D4_ROOT:-/N/project/SingleCell_Image/mengjing/d4_138M}"

srun --cpu-bind=none python embed/compute/compute_grover3400_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_grover3400_chunks" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpu 0
