#!/bin/bash

#SBATCH -J molpal_fig4
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o logs/fig4_run_%A_%a.txt
#SBATCH -e logs/fig4_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# BigRed200 submission for the paper's actual Figure 4 reproduction: RF and
# NN, greedy acquisition only, same 3 batch-size fractions as the other 18
# runs (MPN's greedy runs are already done -- see submit_al_runs_bigred.sh --
# so this only needs the 2 new models x 3 fractions = 6 tasks). Same
# --exclusive/--mem=0 configuration already proven to work for all 18 runs
# in submit_al_runs_bigred.sh; see that script's comments for why both flags
# are needed.
#
# RF is pure CPU (sklearn RandomForestRegressor, no CUDA usage at all) --
# still requests a GPU node here for consistency with the rest of the sweep
# and because NN (PyTorch, small MC-Dropout MLP) does use one, but RF's task
# will simply leave the GPU idle.
#
# Array index <-> config:
#   index 0-2 = rf,  greedy, frac 0.004/0.002/0.001
#   index 3-5 = nn,  greedy, frac 0.004/0.002/0.001
#
# Usage:
#   sbatch --array=0-5 submit_figure4_runs.sh
#
# After all 6 (plus the already-done mpn_greedy_frac* from the other sweep):
#   python plot_figure4.py

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-5 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=2104319

FRACS=(0.004 0.002 0.001 0.004 0.002 0.001)
MODELS=(rf rf rf nn nn nn)

frac="${FRACS[$TASK_ID]}"
model="${MODELS[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

echo "[task $TASK_ID] model=$model acq=greedy frac=$frac (init=batch=$count)"

srun --cpu-bind=none python run_experiment.py --mode molpal --model "$model" \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --run-dir "runs/${model}_greedy_frac${frac}"
