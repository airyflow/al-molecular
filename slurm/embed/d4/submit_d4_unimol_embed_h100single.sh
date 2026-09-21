#!/bin/bash

#SBATCH -J d4_unimol_embed_h100
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_unimol_embed_h100_%A_%a.txt
#SBATCH -e logs/d4_unimol_embed_h100_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 2 of D4's Uni-Mol v1 pipeline: embeddings from Stage 1's conformer
# cache, GPU (h100-single). Same compute script/reasoning as
# slurm/embed/ampc/submit_ampc_unimol_embed_h100single.sh -- already
# includes both fixes applied there (conformer-duplication fix keeping
# only the first of 2 stored conformers/molecule, and a seeded
# torch.manual_seed() before model construction so the unseeded final
# embedding-projection layer is reproducible across chunk processes).
#
# --num-chunks MUST match Stage 1's (submit_d4_unimol_conformers.sh)
# NUM_CHUNKS exactly -- the compute script asserts chunk boundaries line
# up and will refuse to run on a mismatch rather than silently
# misaligning SMILES<->conformers.
#
# No array concurrency cap, same reasoning as the AmpC version -- SLURM
# only starts as many tasks as there are free h100-single nodes.
#
# --time=36:00:00 kept at AmpC's own margin (sized over its observed
# 0.12-2.32 ms/mol per-chunk variance) -- not re-measured for D4.
#
# After ALL chunks exist, stitch (dim 512, SAME NUM_CHUNKS):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh unimol 82
#
# Usage (NUM_CHUNKS must equal whatever Stage 1 used):
#   sbatch --array=0-81 slurm/embed/d4/submit_d4_unimol_embed_h100single.sh 82

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-N slurm/embed/d4/submit_d4_unimol_embed_h100single.sh <NUM_CHUNKS>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-(NUM_CHUNKS-1)}"

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

D4_ROOT="${D4_ROOT:-/N/project/SingleCell_Image/mengjing/d4_138M}"

python3 embed/compute/compute_unimol_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" \
    --total-count 116241184 \
    --conformer-chunks-dir "$D4_ROOT/_unimol_conformers/_chunks" \
    --chunk-id "$TASK_ID" \
    --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_unimol_chunks" \
    --num-workers 4
