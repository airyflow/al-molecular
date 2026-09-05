#!/bin/bash

#SBATCH -J ltall_enhits
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ltall_enhits_%A_%a.txt
#SBATCH -e logs/ltall_enhits_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# LT-All (LTAllSurrogate) greedy active-learning on the re-embedded Enamine
# HTS pool (DATASETS["ENHITS"] -- 2,104,318 scored molecules, 5 backbones
# stitched by embed/stitch/stitch_enhits_large_shards.py). One array task per random
# seed, so the paper's "each experiment repeated five times" is 5 parallel
# single-GPU jobs.
#
# --mem=200G (NOT --exclusive/--mem=0): h100-single's QOS rejects --mem=0
# with QOSMaxMemoryPerNode -- same reason slurm/al_runs/ampc/submit_ampc_fusion_runs_h100single.sh
# uses an explicit --mem. run_experiment.py's non-parallel MVE path
# materializes all five embedding arrays in host RAM (~52 GB) once
# keep_idx-filtered, plus the oracle usable_set and concat overhead;
# 200G is comfortable headroom. Lower it if the QOS cap is below 200G.
#
# IMPORTANT: --run-dir must carry the seed. run_experiment.py's automatic
# run-dir name is mve_<surrogate>_<backbones>_<acq>_init<N> with NO seed in
# it, so without an explicit per-seed --run-dir every task would write into
# (and clobber) the same directory.
#
# Array index <-> seed:
#   0 -> 42   1 -> 43   2 -> 44   3 -> 45   4 -> 46
#
# Usage:
#   sbatch --array=0-4 slurm/al_runs/enamine/submit_ltall_enhits_seed_runs_h100single.sh
#   # or just the 4 additional seeds if 42 is already done:
#   sbatch --array=1-4 slurm/al_runs/enamine/submit_ltall_enhits_seed_runs_h100single.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-4 (SLURM_ARRAY_TASK_ID unset)}"

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
set -a
[ -f config.env ] && source config.env
set +a
: "${CONDA_SH:=/N/slate/mengjing/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=py310}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

SEEDS=(42 43 44 45 46)
seed="${SEEDS[$TASK_ID]}"

# frac 0.001 == paper's greedy-0.1% (init == batch). Change here to sweep.
N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")

RUN_DIR="runs/ltall_enhits_greedy_frac${FRAC}_seed${seed}"

echo "[task $TASK_ID] surrogate=ltall acq=greedy frac=$FRAC (init=batch=$count) seed=$seed -> $RUN_DIR"

srun --cpu-bind=none python run_experiment.py \
    --dataset ENHITS --mode mve --surrogate ltall --acq greedy \
    --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume
