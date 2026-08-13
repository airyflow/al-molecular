#!/bin/bash

#SBATCH -J ampc_unimol_embed
#SBATCH -p general
#SBATCH -o logs/ampc_unimol_embed_%A_%a.txt
#SBATCH -e logs/ampc_unimol_embed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 2 of the AmpC Uni-Mol pipeline: compute embeddings from the
# conformer shards Stage 1 (submit_ampc_unimol_conformers.sh) already
# produced. CPU-only here (matches "hard to get GPU" -- BigRed200's GPU
# partition is far more contended than its general/CPU partition;
# compute_unimol_embeddings_chunk.py already has a CPU fallback path).
# No real CPU throughput number is in hand yet for this specific
# model/hardware combination -- check the first completed shard's log
# line (prints ms/molecule) before trusting the --time budget above.
#
# --num-chunks MUST match Stage 1's (70) -- chunk boundaries have to line
# up exactly with the LMDB shard files on disk, or the alignment check in
# compute_unimol_embeddings_chunk.py will correctly refuse to produce
# embeddings rather than silently misalign SMILES with the wrong
# conformers.
#
# Each task writes its own independent chunk file -- no shared state, no
# coordination, no race condition possible between the 70 parallel tasks.
# After all 70 finish, run stitch_embedding_chunks.py once (see
# submit_ampc_stitch_embeddings.sh) to produce the final shared .npy that
# EmbeddingFeaturizer.load() expects.
#
# Usage (only after all of Stage 1's shards exist):
#   sbatch --array=0-69 submit_ampc_unimol_embed.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-69 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

# 4 conformer/collator worker processes (num_workers=4 below) each get
# their own BLAS threads capped to 4 -- same oversubscription guard as
# Stage 1.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python compute_unimol_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --shards-dir "$AMPC_ROOT/_unimol_conformers/_shards" \
    --chunk-id "$TASK_ID" --num-chunks 70 \
    --chunks-dir "$AMPC_ROOT/_unimol_chunks" \
    --num-workers 4
