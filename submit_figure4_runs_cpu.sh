#!/bin/bash

#SBATCH -J molpal_fig4_cpu
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -o logs/fig4_cpu_run_%A_%a.txt
#SBATCH -e logs/fig4_cpu_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# CPU-only counterpart to submit_figure4_runs.sh, for a direct GPU-vs-CPU
# timing comparison (the GPU version's results/timing stay untouched under
# runs/{model}_greedy_frac{frac}/; this writes to a separate
# runs/{model}_greedy_frac{frac}_cpu/ so neither overwrites the other).
#
# No --gpus-per-node here at all -- BigRed200's "general" partition has no
# GPU GRES (confirmed via sinfo), so this targets plain CPU nodes.
#
# --exclusive/--mem=0 (CONFIRMED NECESSARY, not just defensive -- an earlier
# version of this script omitted both, reasoning that RF/NN's CPU-only
# workload didn't have the GPU-contention risk that motivated them on
# submit_figure4_runs.sh. That reasoning was wrong: job 7915929 showed both
# failure modes recurring for reasons unrelated to GPUs --
# (1) index 0 (rf_greedy_frac0.004) hit a real SLURM oom_kill under
#     --mem=32G: RandomForestRegressor.fit() routes through
#     joblib.parallel_backend("ray"), which spins up a Ray actor pool (100
#     trees) with real memory overhead beyond RF's own data, and 32G wasn't
#     a verified number, just a guess.
# (2) indices 1-2 hit "Failed to register worker to Raylet: IOError...",
#     the exact Ray-port-collision failure already diagnosed and fixed via
#     --exclusive on the GPU script -- it recurred here because multiple
#     array tasks landed on the same node and each ran its own ray.init(),
#     which is not a GPU-specific risk at all, just a node-sharing one.
#
# Array index <-> config: same as submit_figure4_runs.sh
#   index 0-2 = rf,  greedy, frac 0.004/0.002/0.001
#   index 3-5 = nn,  greedy, frac 0.004/0.002/0.001
#
# Usage:
#   sbatch --array=0-5 submit_figure4_runs_cpu.sh
#
# To compare GPU vs CPU timing once both sets are done:
#   for f in 0.004 0.002 0.001; do
#     for m in rf nn; do
#       echo "$m $f: gpu=$(python3 -c "import json;print(sum(r['elapsed'] for r in json.load(open('runs/${m}_greedy_frac${f}/history.json'))))") cpu=$(python3 -c "import json;print(sum(r['elapsed'] for r in json.load(open('runs/${m}_greedy_frac${f}_cpu/history.json'))))")"
#     done
#   done

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-5 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=""

N=2104319

FRACS=(0.004 0.002 0.001 0.004 0.002 0.001)
MODELS=(rf rf rf nn nn nn)

frac="${FRACS[$TASK_ID]}"
model="${MODELS[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

echo "[task $TASK_ID] model=$model acq=greedy frac=$frac (init=batch=$count) [CPU-only]"

srun --cpu-bind=none python run_experiment.py --mode molpal --model "$model" \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --run-dir "runs/${model}_greedy_frac${frac}_cpu"
