#!/bin/bash

#SBATCH -J molpal_ftfusion
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ftfusion_run_%A_%a.txt
#SBATCH -e logs/ftfusion_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# Quartz H100 variant of submit_ft_fusion_runs.sh -- use this instead of
# the "hopper"-targeting original, which queues indefinitely. `sinfo` on
# the actual Quartz login node (2026-08-13) shows no "hopper" partition
# at all anymore -- only h100-debug, h100-multi, and h100-single (19 idle
# nodes at time of writing). IU RT confirmed h100-single is the current
# single-GPU H100 partition; see submit_learned_fusion_runs_h100single.sh
# for the full diagnosis (same root cause applies here).
#
# -q hopper dropped (no confirmed equivalent QOS for h100-single) --
# default partition QOS applies instead. --gpus-per-node h100:1 kept
# typed; retry with untyped --gpus-per-node=1 if this specific GRES
# string is rejected.
#
# See submit_ft_fusion_runs.sh for the full method rationale: GROVER +
# MoLFormer fine-tuned jointly each round from round 3 on, UniMol kept
# frozen throughout (its ragged conformer cache can't be memory-mapped
# or cheaply rebuilt on the fly -- see surrogates.py's FTFusionSurrogate
# docstring). Greedy acquisition only, 3 batch-size fractions.
#
# Array index <-> config:
#   index 0 = frac 0.004
#   index 1 = frac 0.002
#   index 2 = frac 0.001
#
# Usage:
#   sbatch --array=0-2 submit_ft_fusion_runs_h100single.sh

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

echo "[task $TASK_ID] surrogate=ft_fusion acq=greedy frac=$frac (init=batch=$count)"

srun --cpu-bind=none python run_experiment.py --mode mve --surrogate ft_fusion \
    --backbones grover molformer unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --run-dir "runs/ftfusion_greedy_frac${frac}"
