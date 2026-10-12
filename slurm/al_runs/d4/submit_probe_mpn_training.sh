#!/bin/bash

#SBATCH -J probe_mpn_train
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/probe_mpn_train_%j.txt
#SBATCH -e logs/probe_mpn_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -A r00939

# Splits MPN training time into featurization (data loader alone) vs. the
# full training step, at batch sizes 50 and 1024, on the first 200,000 D4
# molecules -- same --ncpu 8 and node type as the real orchestrator runs.
# Answers why batch size 1024 sped training up only 1.2-1.4x (report sect. 16).
#
# Usage: sbatch slurm/al_runs/d4/submit_probe_mpn_training.sh

set -euo pipefail
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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

D=/N/project/SingleCell_Image/mengjing/d4_138M/dedup
python3 -u probe_mpn_training.py --smiles-file "$D/d4_smiles.txt" --scores-file "$D/d4_scores.csv.gz" \
    --n 200000 --epochs 2 --ncpu 8 --batch-sizes 50 1024
