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

# Sharded Uni-Mol2 (1.1B-param port, unimol2/ package, dim 768) embedding
# extraction for the AmpC 99.5M pool.
#
# Two costs per chunk: RDKit conformer generation (CPU, parallelised by
# --num-conformer-workers) then the 1.1B forward pass (GPU, batch 16).
# ~1 s/mol on CPU-only; on an H100 the conformer stage dominates. Budget
# roughly 10-25 ms/mol effective -> ~275-700 GPU-hours for the full pool.
#
#   NUM_CHUNKS=500  -> ~199k mol/chunk -> ~1-4 h/chunk
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
#   sbatch --array=0-499%48 submit_ampc_unimol2_extract.sh 500
# Then, SAME NUM_CHUNKS (dim 1536):
#   sbatch submit_ampc_stitch_embeddings_striped.sh unimol2 500
#
# OFFSET (arg 2, default 0): real chunk id = ARRAY_TASK_ID + OFFSET -- only
# for a deliberate NUM_CHUNKS > 1000 second wave. Changing NUM_CHUNKS after
# a partial run: wipe $AMPC_ROOT/_unimol2_chunks first (skipped by name).

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 submit_ampc_unimol2_extract.sh <NUM_CHUNKS> [OFFSET]}"
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

srun --cpu-bind=none python compute_unimol2_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_unimol2_chunks" \
    --batch-size 16 --num-conformer-workers 12 --timeout-s 30
