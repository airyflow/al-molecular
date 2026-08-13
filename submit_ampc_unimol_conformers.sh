#!/bin/bash

#SBATCH -J ampc_conformers
#SBATCH -p general
#SBATCH -o logs/ampc_conformers_%A_%a.txt
#SBATCH -e logs/ampc_conformers_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 1 of the AmpC (99,459,561 molecules) Uni-Mol pipeline: conformer
# generation only, CPU-only (RDKit ETKDG is CPU-bound, no GPU benefit).
# See generate_unimol_conformers_chunk.py's module docstring for the full
# design rationale -- ported from al-eval-framework's generate_conformers.py,
# whose real 1000-shard / ~1.3B-molecule / ~7h run validated this design at
# a larger scale than AmpC needs.
#
# --num-chunks 70 (~1.42M molecules/shard) matches that same known-good
# per-shard size (the 1.3-1.56M/shard ratio that took ~7h/shard previously)
# -- --time 12h leaves real margin over that, not just matching it exactly.
#
# Usage:
#   sbatch --array=0-69 submit_ampc_unimol_conformers.sh
#
# After all 70 shards finish, Stage 2 (submit_ampc_unimol_embed.sh) reads
# directly from these per-chunk shards -- no merge step is required.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-69 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

# 16 conformer-generation worker processes (num_workers=16 below) each get
# their own BLAS threads capped to 4 -- same oversubscription guard proven
# necessary in al-eval-framework's conformer generation jobs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python generate_unimol_conformers_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --out-dir "$AMPC_ROOT/_unimol_conformers" \
    --chunk-id "$TASK_ID" --num-chunks 70 --n-conformer 1 \
    --num-workers 16
