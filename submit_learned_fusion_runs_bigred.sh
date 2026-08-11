#!/bin/bash

#SBATCH -J molpal_learned
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o logs/learned_run_%A_%a.txt
#SBATCH -e logs/learned_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# BigRed200 variant of submit_learned_fusion_runs.sh (which targets Quartz's
# hopper/H100 partition) -- use this when hopper is queue-blocked or Quartz
# is down for maintenance. --gpus-per-node=1 has no type string here:
# BigRed200's gpu partition GRES is untyped (gpu:4, no a100/h100 tag), unlike
# Quartz's hopper (gpu:h100:4) -- see submit_al_runs_bigred.sh's comments for
# the original diagnosis. GPUs are A100-PCIE-40GB, plenty for this method
# (3 small per-backbone MLPs on frozen embeddings, no full-pool re-embedding).
#
# See submit_learned_fusion_runs.sh for the full rationale: LearnedFusionSurrogate
# is 3 frozen per-backbone models combined by a RidgeCV meta-learner fit on
# held-out predictions each round -- no backbone fine-tuning, never touches
# BackboneFinetuner, cost comparable to the existing "ensemble" method
# (~6-11 min/config). Greedy-only by design: predict() returns zeros for
# sigma, so UCB would be mathematically identical to greedy.
#
# Array index <-> config:
#   index 0 = frac 0.004
#   index 1 = frac 0.002
#   index 2 = frac 0.001
#
# Usage:
#   sbatch --array=0-2 submit_learned_fusion_runs_bigred.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-2 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/molpal-fusion-hts
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=2104319

FRACS=(0.004 0.002 0.001)
frac="${FRACS[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

echo "[task $TASK_ID] surrogate=learned acq=greedy frac=$frac (init=batch=$count)"

srun --cpu-bind=none python run_experiment.py --mode mve --surrogate learned \
    --backbones grover molformer unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --run-dir "runs/learned_greedy_frac${frac}"
