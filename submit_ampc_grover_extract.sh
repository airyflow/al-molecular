#!/bin/bash

#SBATCH -J ampc_grover
#SBATCH -p general
#SBATCH -o logs/ampc_grover_%A_%a.txt
#SBATCH -e logs/ampc_grover_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# GROVER embedding extraction for AmpC (99,459,561 molecules), CPU-only,
# single stage (see compute_grover_embeddings_chunk.py's docstring for why
# GROVER doesn't get a separate persisted-preprocessing stage the way
# Uni-Mol does -- graph construction is cheap enough, ~3ms/molecule
# measured, that persisting it isn't clearly worth the ~12-15TB it would
# cost). CPU-primary because GPU nodes are scarce on BigRed200 and this
# is embarrassingly parallel across the general partition's CPU nodes
# instead.
#
# --num-chunks 150 (~663K molecules/shard) is a STARTING estimate, not a
# measured one -- no empirical CPU throughput number for GROVER's model
# forward pass exists yet on this hardware. Check the FIRST completed
# shard's log line (prints ms/molecule) before assuming this shard count
# is well calibrated -- if a shard is taking much longer than the 12h
# --time budget allows, cancel and resubmit with a larger --num-chunks.
#
# Each task writes its own independent chunk file -- no shared state, no
# coordination, no race condition possible between the 150 parallel tasks.
# After all 150 finish, run stitch_embedding_chunks.py once (see
# submit_ampc_stitch_grover.sh) to produce the final shared .npy that
# EmbeddingFeaturizer.load() expects.
#
# Usage:
#   sbatch --array=0-149 submit_ampc_grover_extract.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-149 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

# No separate multiprocessing Pool competes for cores here (graph
# construction is a plain per-batch loop, not forked workers), so give
# the model's own BLAS/threading the full core count instead of the
# small fixed value used where a worker pool is also competing for CPU.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python compute_grover_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks 150 \
    --chunks-dir "$AMPC_ROOT/_grover_chunks"
