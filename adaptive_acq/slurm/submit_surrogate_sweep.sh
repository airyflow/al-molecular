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
#   4 __ctrl     plain baseline, written to runs/adaptive/fixed_b0_frac0.001_seed<S>
#                (only needed for seed sets with no control yet; skips if complete)
#
# Decision rule, fixed before the results (same as the beta gate): an arm is
# a GO candidate only if it gains >= +0.005 top-1000 recall at round 5 with a
# 95% CI excluding 0, AND it then replicates on 5 new seeds. Four arms are
# tested, so expect roughly one marginal false positive.
#
#   sbatch --array=0-19%10 adaptive_acq/slurm/submit_surrogate_sweep.sh
#
# Replication clause, e.g. for tw1.0 (arm 2) plus its control (arm 4):
#   AA_SEEDS="47 48 49 50 51" sbatch --export=ALL,AA_SEEDS --array=10-14,20-24%10 \
#       adaptive_acq/slurm/submit_surrogate_sweep.sh
#   python -m adaptive_acq.surrogate_sweep --seeds 47 48 49 50 51
#   python -m adaptive_acq.surrogate_sweep

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-19}"
# AA_SEEDS overrides the seed set, for the decision rule's replication clause:
#   AA_SEEDS="47 48 49 50 51" sbatch --export=ALL,AA_SEEDS --array=10-14 ... (tw1.0 only)
read -r -a SEEDS <<< "${AA_SEEDS:-42 43 44 45 46}"
[ "${#SEEDS[@]}" = 5 ] || { echo "AA_SEEDS must list exactly 5 seeds"; exit 1; }
seed="${SEEDS[$((TASK_ID % 5))]}"
arm=$((TASK_ID / 5))

case $arm in
  0) NAME=m15;       EXTRA=(--ltall-members 15) ;;
  1) NAME=tw0.5;     EXTRA=(--ltall-top-weight 0.5) ;;
  2) NAME=tw1.0;     EXTRA=(--ltall-top-weight 1.0) ;;
  3) NAME=m15_tw1.0; EXTRA=(--ltall-members 15 --ltall-top-weight 1.0) ;;
  4) NAME=__ctrl;    EXTRA=() ;;   # baseline at NEW seeds (see below)
  *) echo "no arm $arm"; exit 1 ;;
esac

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
if [ "$NAME" = __ctrl ]; then
    # The control is the plain fixed_b0 config, written where surrogate_sweep.py
    # looks for it. Only needed for seed sets that have no control yet (42-46
    # already do, from the Stage-1 sweep) -- skip if this one is complete.
    RUN_DIR="runs/adaptive/fixed_b0_frac${FRAC}_seed${seed}"
    if [ -f "$RUN_DIR/history.json" ] && \
       [ "$(python3 -c "import json;print(len(json.load(open('$RUN_DIR/history.json'))))")" = 5 ]; then
        echo "[task $TASK_ID] control $RUN_DIR already complete -- skipping"; exit 0
    fi
else
    RUN_DIR="runs/adaptive/surrsweep/${NAME}_frac${FRAC}_seed${seed}"
fi
echo "[task $TASK_ID] $NAME seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --controller fixed --beta 0 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${EXTRA[@]}"
