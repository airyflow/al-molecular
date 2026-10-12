#!/bin/bash

#SBATCH -J aa_surrsweep
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_surrsweep_%A_%a.txt
#SBATCH -e logs/aa_surrsweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# Surrogate-side recall sweep. The beta gate (2026-10-12) showed acquisition
# has <= +0.005 of headroom, so the remaining levers are in the surrogate.
# Two knobs, both nearly free (LT-All training is ~5% of a round's wall time;
# pool scoring is the rest):
#
#   M          ensemble members. LTAllSurrogate's docstring flags M=5 as an
#              ASSUMPTION, not a value taken from the paper. Averaging more
#              members reduces per-member init variance in the ranking.
#   top_weight training loss. The paper uses plain MSE over all labeled
#              molecules, but the metric is top-1000 recall out of 2.1M, so
#              fit on the mediocre bulk is wasted. top_weight > 0 weights each
#              molecule by exp(w * standardized score).
#
# Control: runs/adaptive/fixed_b0_frac0.001_seed{42..46} (LT-All greedy,
# M=5, plain MSE, 0.876 +/- 0.006) -- ALREADY RUN, same seeds, same seeded
# init draw, so the comparison is paired per seed and no control reruns are
# needed.
#
# Array index = arm_index * 5 + seed_index (20 tasks):
#   0 m15        M=15, MSE
#   1 tw0.5      M=5,  top_weight=0.5
#   2 tw1.0      M=5,  top_weight=1.0
#   3 m15_tw1.0  M=15, top_weight=1.0
#
# Decision rule, fixed before the results (same as the beta gate): an arm is
# a GO candidate only if it gains >= +0.005 top-1000 recall at round 5 with a
# 95% CI excluding 0, AND it then replicates on 5 new seeds. Four arms are
# tested, so expect roughly one marginal false positive.
#
#   sbatch --array=0-19%10 adaptive_acq/slurm/submit_surrogate_sweep.sh
#   python -m adaptive_acq.surrogate_sweep

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-19}"
SEEDS=(42 43 44 45 46)
seed="${SEEDS[$((TASK_ID % 5))]}"
arm=$((TASK_ID / 5))

case $arm in
  0) NAME=m15;       EXTRA=(--ltall-members 15) ;;
  1) NAME=tw0.5;     EXTRA=(--ltall-top-weight 0.5) ;;
  2) NAME=tw1.0;     EXTRA=(--ltall-top-weight 1.0) ;;
  3) NAME=m15_tw1.0; EXTRA=(--ltall-members 15 --ltall-top-weight 1.0) ;;
  *) echo "no arm $arm"; exit 1 ;;
esac

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
RUN_DIR="runs/adaptive/surrsweep/${NAME}_frac${FRAC}_seed${seed}"
echo "[task $TASK_ID] $NAME seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --controller fixed --beta 0 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${EXTRA[@]}"
