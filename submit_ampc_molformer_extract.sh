#!/bin/bash

#SBATCH -J ampc_molformer
#SBATCH -p general
#SBATCH -o logs/ampc_molformer_%A_%a.txt
#SBATCH -e logs/ampc_molformer_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# MoLFormer embedding extraction for AmpC (99,459,561 molecules), CPU-only,
# single stage (no RDKit graph/conformer construction at all -- see
# compute_molformer_embeddings_chunk.py's docstring). Cheapest of the three
# backbones by a wide margin (measured on GPU at EnamineHTS scale: ~0.22
# ms/molecule); CPU-primary here to match GROVER/UniMol and because GPU
# nodes are scarce on BigRed200.
#
# --num-chunks 50 (~1.99M molecules/shard) is a STARTING estimate -- check
# the first completed shard's log line (prints ms/molecule) before trusting
# it.
#
# Each task writes its own independent chunk file -- no shared state, no
# coordination, no race condition possible between the 50 parallel tasks.
# After all 50 finish, run stitch_embedding_chunks.py once (see
# submit_ampc_stitch_embeddings.sh) to produce the final shared .npy that
# EmbeddingFeaturizer.load() expects.
#
# Usage:
#   sbatch --array=0-49 submit_ampc_molformer_extract.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-49 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/molpal-fusion-hts
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python compute_molformer_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks 50 \
    --chunks-dir "$AMPC_ROOT/_molformer_chunks"
