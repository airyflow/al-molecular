#!/bin/bash

#SBATCH -J molpal_learned
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/learned_run_%A_%a.txt
#SBATCH -e logs/learned_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# LearnedFusionSurrogate ("learned"): 3 independent per-backbone MVE models
# (GROVER, MoLFormer, UniMol), all frozen -- no backbone ever retrains, no
# BackboneFinetuner is ever touched. Each round: 80/20 split of the labeled
# set, train the 3 backbone models on the 80%, then fit a RidgeCV linear
# meta-learner (mu = w_g*mu_g + w_m*mu_m + w_u*mu_u + bias) on the 20%
# holdout predictions vs true labels. Unlike EnsembleFusionSurrogate's fixed
# Borda rank-sum (zero learnable parameters in the combination step), the
# meta-learner's weights are genuinely fit from data each round.
#
# Cost profile matches "ensemble" (our existing frozen fusion model, ~6-11
# min/config in the saved report) -- confirmed safe to smoke-test locally
# first (unlike ft_fusion, this path never loads GROVER/UniMol's expensive
# caches), which was done before this submission: --pool-limit 3000, 3
# rounds, completed cleanly with sane RidgeCV coefficients each round.
#
# predict() returns zeros for sigma (see surrogates.py's LearnedFusionSurrogate
# docstring: "use with acq_greedy") -- UCB would be mathematically identical
# to greedy here (same behavior already confirmed for "ensemble" in the
# existing report, where greedy/UCB numbers matched exactly), so this is
# greedy-only by design, not by omission.
#
# --exclusive/--mem=0: same proven necessity as every other script in this
# repo. --time=2:00:00 is generous headroom over the ~11 min/config observed
# for the architecturally similar "ensemble" surrogate.
#
# Array index <-> config:
#   index 0 = frac 0.004
#   index 1 = frac 0.002
#   index 2 = frac 0.001
#
# Usage:
#   sbatch --array=0-2 submit_learned_fusion_runs.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-2 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
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
