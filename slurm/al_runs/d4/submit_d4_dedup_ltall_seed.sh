#!/bin/bash

#SBATCH -J d4_ltall_seed
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_ltall_seed_%j.txt
#SBATCH -e logs/d4_ltall_seed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=280G
#SBATCH -A r00939

# Same D4 LT-All orchestrator as submit_d4_dedup_ltall_runs_h100single.sh
# (5 backbones, greedy, 5 rounds, top-50,000), with FRAC and SEED as
# arguments, so extra seeds can be run beside the existing seed-42 results
# (report sect. 16 has one seed per fraction). Each (FRAC, SEED) gets its own
# coord-dir and run-dir; the seed changes the random initial batch and the
# numpy/torch seeds. Seed 42 reuses the original paths, so it never reruns.
#
# Two-step launch, per (FRAC, SEED) -- workers first, they wait for round 1:
#   C=/N/project/SingleCell_Image/mengjing/d4_138M/al_coord/dedup_ltall_frac0.004_seed43
#   mkdir -p $C
#   sbatch --array=0-7 slurm/al_runs/d4/submit_d4_dedup_ltall_predict_workers_h100single.sh $C 8
#   sbatch slurm/al_runs/d4/submit_d4_dedup_ltall_seed.sh 0.004 43

set -euo pipefail
FRAC="${1:?Usage: sbatch slurm/al_runs/d4/submit_d4_dedup_ltall_seed.sh <frac> <seed>}"
SEED="${2:?Usage: sbatch slurm/al_runs/d4/submit_d4_dedup_ltall_seed.sh <frac> <seed>}"

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"
if [ "$SEED" = "42" ]; then
    COORD_DIR="$D4_ROOT/al_coord/dedup_ltall_frac${FRAC}"
    RUN_DIR="runs/d4_dedup_ltall_greedy_frac${FRAC}"
else
    COORD_DIR="$D4_ROOT/al_coord/dedup_ltall_frac${FRAC}_seed${SEED}"
    RUN_DIR="runs/d4_dedup_ltall_greedy_frac${FRAC}_seed${SEED}"
fi

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
mkdir -p logs "$COORD_DIR"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=116225927
count=$(python3 -c "print(round($N * $FRAC))")
echo "[orchestrator] D4_dedup ltall greedy frac=$FRAC seed=$SEED (init=batch=$count) coord-dir=$COORD_DIR run-dir=$RUN_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset D4_dedup --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --seed "$SEED" --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "$RUN_DIR" --resume
