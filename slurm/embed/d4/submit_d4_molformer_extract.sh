#!/bin/bash

#SBATCH -J d4_molformer
#SBATCH -p general
#SBATCH -o logs/d4_molformer_%A_%a.txt
#SBATCH -e logs/d4_molformer_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# MoLFormer embedding extraction for D4 (116,241,184 molecules -- see
# submit_d4_mhgged_extract.sh's comment for why this is D4's scored-only
# count, not the raw d4.csv's 138,312,677), CPU-only, single stage. Same
# compute script/reasoning as slurm/embed/ampc/submit_ampc_molformer_extract.sh
# (cheapest of the backbones by a wide margin, ~0.22 ms/mol measured on
# GPU at EnamineHTS scale; CPU-primary to avoid competing for scarce GPU
# nodes) -- not re-measured at D4/AmpC scale for this backbone in this
# repo's own record, so treat NUM_CHUNKS as a starting estimate.
#
# UNLIKE the AmpC version, NUM_CHUNKS is a real positional CLI arg here
# (not hardcoded inline) -- matches the convention grover3400/mhgged/
# smited/unimol2 already use, since D4's different scale makes a fixed
# chunk count from the AmpC copy the wrong default to just reuse silently.
#
# Usage (D4's 116,241,184 scored molecules are ~1.17x AmpC's 99,459,561 --
# start with a similar chunk density to AmpC's 50 chunks, i.e. ~58, rounded
# to 60 -- then adjust after checking the first completed shard's log line
# for actual ms/mol):
#   sbatch --array=0-59 slurm/embed/d4/submit_d4_molformer_extract.sh 60
#
# After all NUM_CHUNKS finish:
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh molformer 60

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-N slurm/embed/d4/submit_d4_molformer_extract.sh <NUM_CHUNKS>}"
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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

srun --cpu-bind=none python embed/compute/compute_molformer_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_molformer_chunks"
