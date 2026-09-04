#!/bin/bash

#SBATCH -J ampc_smited
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_smited_%A_%a.txt
#SBATCH -e logs/ampc_smited_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Sharded SMI-TED (ibm-research/materials.smi-ted "light", 289M params,
# dim 768) embedding extraction for the AmpC 99.5M pool.
#
# SMI-TED tokenizes SMILES directly and runs a real batched .encode() on
# GPU -- fast, no conformer/preprocessing stage. Each array task computes
# one contiguous chunk and writes its own independent
# _smited_chunks/smited_embeddings_chunk_NNNNN.npy (no shared state, no
# race). First task to run downloads the ~1.16GB checkpoint into the HF
# cache; the rest reuse it.
#
# total-count 99459561 is the value every other AmpC extraction script in
# this repo uses (matches the existing molformer/grover/unimol stitched
# files' row counts exactly) -- keep it identical so _chunk_bounds() lines
# up with the eventual stitch.
#
# This cluster's MaxArraySize is 1000, so keep NUM_CHUNKS <= 1000 and use a
# single wave. SMI-TED's batched .encode() is fast on an H100; 500 chunks
# (~199k mol each) is plenty of parallelism.
#
# Usage (one array wave):
#   sbatch --array=0-499%64 slurm/embed/ampc/submit_ampc_smited_extract.sh 500
#   # resubmit the same line after a partial run -- finished chunks self-skip
#
# Then stitch once, SAME NUM_CHUNKS (dim 768):
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh smited 500
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# needed if you deliberately set NUM_CHUNKS > 1000 and cover the rest in a
# second wave. If you change NUM_CHUNKS after a partial run, wipe
# $AMPC_ROOT/_smited_chunks first (chunks are skipped by name, not row count).

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/ampc/submit_ampc_smited_extract.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do"
    exit 0
fi

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python embed/compute/compute_smited_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_smited_chunks" \
    --batch-size 512
