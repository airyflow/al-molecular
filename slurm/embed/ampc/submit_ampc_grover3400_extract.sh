#!/bin/bash

#SBATCH -J ampc_grover3400
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_grover3400_%A_%a.txt
#SBATCH -e logs/ampc_grover3400_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Real GROVER (tencent-ailab/grover, vendored as a git submodule at grover/)
# 3400-d "both" fingerprint extraction for the AmpC 99.5M pool -- reproduces
# Yang's ENHITS grover embeddings (verified bit-exact, max abs diff 3.5e-6,
# on a 20-molecule sample against his emb_shard_000000.npy), unlike
# submit_ampc_grover_extract.sh's muben-based 1600-d version, which only
# does 2 of 4 required readout terms and never appends RDKit-2D features
# -- see compute_grover3400_embeddings_chunk.py's docstring for the full
# diagnosis.
#
# Measured on an idle h100-single node (2026-09-05): fixed overhead
# ~53s/chunk (two subprocess launches -- save_features.py then
# main.py fingerprint -- each re-importing torch/loading the checkpoint),
# steady-state ~34.5 ms/mol. At NUM_CHUNKS=1000 (~99.46k mol/chunk):
# ~58 min/chunk, ~967 GPU-hours total, ~24h wall at ~40 concurrent tasks
# (this cluster's real concurrent-job ceiling, well below the requested
# %N -- see mhgged/smited extraction notes for the same finding).
#
# This cluster's MaxArraySize is 1000, so ONE array wave covers indices
# 0..999. Use NUM_CHUNKS=1000 and a single submission:
#
#   sbatch --array=0-999%64 submit_ampc_grover3400_extract.sh 1000
#
# OFFSET (3rd positional arg, default 0): real chunk id = ARRAY_TASK_ID +
# OFFSET -- only needed for a deliberate NUM_CHUNKS > 1000 second wave.
#
# --checkpoint-path defaults to Yang's model.pt (see the compute script);
# override with a 4th positional arg if a different checkpoint is needed.
#
# IMPORTANT: if you change NUM_CHUNKS after a partial run, wipe
# $AMPC_ROOT/_grover3400_chunks first -- the compute script skips any
# existing chunk file by name without checking its row count, so leftover
# chunks from a different NUM_CHUNKS would be silently wrong.
#
# After ALL chunks exist (pass the SAME NUM_CHUNKS, dim 3400):
#   sbatch submit_ampc_stitch_embeddings_striped.sh grover3400 1000

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 submit_ampc_grover3400_extract.sh <NUM_CHUNKS> [OFFSET] [CHECKPOINT_PATH]}"
OFFSET="${2:-0}"
CHECKPOINT_PATH="${3:-/N/project/SingleCell_Image/Yang/AI Drug/Emb output/model.pt}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do for this task"
    exit 0
fi

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
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

AMPC_ROOT="${AMPC_ROOT:-/N/project/SingleCell_Image/mengjing/ampc_99.5M}"

srun --cpu-bind=none python embed/compute/compute_grover3400_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_grover3400_chunks" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpu 0
