#!/bin/bash

#SBATCH -J fusionfix
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/fusionfix_%A_%a.txt
#SBATCH -e logs/fusionfix_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=160G
#SBATCH -A r00939

# Rerun of method [3] ("our fusion model", EnsembleFusionSurrogate) after the
# 2026-10-10 fixes (UCB had sigma=0, ranks were per 50k chunk, bf16 ties
# favored the end of the pool -- see README.md). Same config as
# slurm/al_runs/enamine/run_all_configs.sh's fusion runs (EnamineHTS,
# grover/molformer/unimol, init == batch, 5 rounds, top-1000), now over 5
# seeds so greedy vs UCB can be compared with error bars.
#
# Array index = frac_index * 10 + acq_index * 5 + seed_index  (30 tasks):
#   frac 0.001 / 0.002 / 0.004   x   acq greedy / ucb   x   seed 42..46
#
#   sbatch --array=0-29 adaptive_acq/slurm/submit_fusionfix_runs.sh
#   sbatch --array=0-9  adaptive_acq/slurm/submit_fusionfix_runs.sh   # frac 0.001 only
#
# Writes runs/fusionfix_<acq>_frac<frac>_seed<seed>; compare against the
# pre-fix runs/fusion_<acq>_frac<frac> (seed 42).

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-29}"
FRACS=(0.001 0.002 0.004)
ACQS=(greedy ucb)
SEEDS=(42 43 44 45 46)
frac="${FRACS[$((TASK_ID / 10))]}"
acq="${ACQS[$(((TASK_ID / 5) % 2))]}"
seed="${SEEDS[$((TASK_ID % 5))]}"

N=2104319  # same N as run_all_configs.sh, so counts match the pre-fix runs exactly
count=$(python3 -c "print(round($N * $frac))")
RUN_DIR="runs/fusionfix_${acq}_frac${frac}_seed${seed}"
echo "[task $TASK_ID] ensemble acq=$acq frac=$frac (init=batch=$count) seed=$seed -> $RUN_DIR"

srun --cpu-bind=none python run_experiment.py \
    --mode mve --surrogate ensemble --backbones grover molformer unimol \
    --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume
