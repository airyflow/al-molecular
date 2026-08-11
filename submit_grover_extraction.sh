#!/bin/bash

#SBATCH -J grover_gen
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o logs/grover_gen_%A_%a.txt
#SBATCH -e logs/grover_gen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# NOTE: -p gpu / --gpus-per-node=1 mirror the CPU-only conformer_gen script's
# -A r00939 account, but the GPU partition name itself is a guess -- verify
# against `sinfo` / your actual BigRed200 GPU allocation and adjust if wrong.
#
# Sharded GROVER embedding extraction for the EnamineHTS 2.1M pool -- see
# generate_grover_embeddings_chunk.py's module docstring for why sharding
# is needed here (a single-process full-pool run measured ~33min just for
# RDKit graph preprocessing and grew to 260+GB RSS with the forward pass
# still not started, because torch_geometric silently falls back to slow
# pure-Python graph ops on this cluster's older glibc -- pyg_lib/torch_sparse's
# accelerated kernels fail to load). Each --mem 32G here should comfortably
# fit one ~43K-molecule chunk (2,141,500 / 50) even with that fallback path,
# vs. the 260GB+ observed for the full 2.1M pool in one process.
#
# Usage (array job, one task per chunk):
#   sbatch --array=0-49 submit_grover_extraction.sh <NUM_CHUNKS> [TOTAL_COUNT]
#
# e.g. the real run, 50 chunks of ~43K molecules each:
#   sbatch --array=0-49 submit_grover_extraction.sh 50 2141500
#
# After all chunks finish:
#   python concat_grover_chunks.py --chunks-dir results/embed/EnamineHTS/_grover_chunks \
#       --num-chunks 50 --total-count 2141500 --out-path results/embed/EnamineHTS/grover_embeddings.npz

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-N submit_grover_extraction.sh <NUM_CHUNKS> [TOTAL_COUNT]}"
TOTAL_COUNT="${2:-}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-$((NUM_CHUNKS - 1)) (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/molpal-fusion-hts
mkdir -p logs

# Same thread-oversubscription guard used in al-eval-framework's SLURM scripts.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# BigRed200 (HPE Cray) srun does not reliably inherit --cpus-per-task from
# the sbatch allocation for CPU-binding purposes without this (observed
# directly in al-eval-framework's conformer generation jobs).
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

COUNT_FLAG=()
if [[ -n "$TOTAL_COUNT" ]]; then
    COUNT_FLAG=(--total-count "$TOTAL_COUNT")
fi

srun --cpu-bind=none python generate_grover_embeddings_chunk.py \
    --smiles-file molpal/libraries/EnamineHTS.csv.gz "${COUNT_FLAG[@]}" \
    --chunk-id "$CHUNK_ID" --num-chunks "$NUM_CHUNKS" \
    --out-dir results/embed/EnamineHTS/_grover_chunks
