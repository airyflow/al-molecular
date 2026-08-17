#!/bin/bash

#SBATCH -J molpal_learned
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
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

# Quartz H100 variant of submit_learned_fusion_runs.sh -- use this instead
# of the "hopper"-targeting original. A job submitted with -p hopper -q
# hopper queued indefinitely; `sinfo` on the actual Quartz login node
# (2026-08-13) shows NO "hopper" partition at all anymore -- only
# h100-debug, h100-multi, and h100-single (19 idle nodes at time of
# writing, g[61-79]). IU RT confirmed h100-single is the current
# single-GPU H100 partition. The original submit_learned_fusion_runs.sh's
# own comments document that "hopper" was verified as Quartz's only H100
# partition when written -- it's since been split into these three by
# job GPU-count, which is almost certainly why the old script's jobs
# queue forever now (a stale partition name, not ordinary contention).
#
# -q hopper is dropped (no equivalent QOS name confirmed for
# h100-single in this sinfo output) -- letting SLURM apply the
# partition's own default QOS instead of guessing a name that doesn't
# exist. If submission is rejected specifically over QOS, that's the
# first thing to check.
#
# --gpus-per-node h100:1 kept typed (matching the original hopper
# convention) since h100-single's nodes are explicitly H100-only by
# name -- if this specific GRES type string is rejected, retry with the
# untyped --gpus-per-node=1 (the form submit_learned_fusion_runs_bigred.sh
# already uses successfully for BigRed200's differently-configured gpu
# partition).
#
# See submit_learned_fusion_runs.sh for the full method rationale:
# LearnedFusionSurrogate is 3 frozen per-backbone models combined by a
# RidgeCV meta-learner fit on held-out predictions each round -- no
# backbone fine-tuning, cost comparable to "ensemble" (~6-11 min/config).
# Greedy-only by design: predict() returns zeros for sigma, so UCB would
# be mathematically identical to greedy.
#
# Array index <-> config:
#   index 0 = frac 0.004
#   index 1 = frac 0.002
#   index 2 = frac 0.001
#
# Usage:
#   sbatch --array=0-2 submit_learned_fusion_runs_h100single.sh

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
