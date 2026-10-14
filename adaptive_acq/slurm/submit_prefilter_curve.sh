#!/bin/bash

#SBATCH -J aa_pfcurve
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_pfcurve_%A_%a.txt
#SBATCH -e logs/aa_pfcurve_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# Follow-up to submit_prefilter_runs.sh, which measured on the real 2.1M pool:
#   pf2  -0.0824 recall, 4.8x less scoring
#   pf5  -0.0080            4.5x
#   pf10 -0.0040            4.1x
# The loss compounds as the frozen round-1 shortlist ages (pf2 per round:
# 0.000, -0.008, -0.045, -0.070, -0.082) because the shortlist runs out of
# good unlabeled molecules. Two ways to fix that, tested here:
#
#   A. Bigger static shortlist        -- pf20, pf50
#   B. Refresh it mid-campaign       -- re-score the full pool at round 3 and
#      rebuild from the better-trained surrogate (pf10r3, pf5r3)
#
# pf50 and pf10r3 cost about the same (~2.2 full-pool passes), so they are a
# direct head-to-head: is it better to keep a big frozen list, or a smaller
# one rebuilt from a better model? That distinguishes "the shortlist was too
# small" from "the round-1 ranking was wrong", which the per-round curve
# alone cannot.
#
# Array index = arm_index * 5 + seed_index (20 tasks):
#   0 pf20    mult=20, static          (~3.4x expected)
#   1 pf50    mult=50, static          (~2.3x)
#   2 pf10r3  mult=10, refresh at r3   (~2.3x)
#   3 pf5r3   mult=5,  refresh at r3   (~2.4x)
#   4 combo   pf10r3 + tw1.0 -- do the two validated wins compose?
#             (array 20-24; expect ~+0.009 recall AND ~2.3x cheaper together)
#
# Control: runs/adaptive/fixed_b0_frac0.001_seed{42..46} (already run, same
# seeded init draws, so paired per seed). Same non-inferiority rule as before:
# acceptable if the 95% CI of the paired round-5 difference contains 0 with a
# lower bound above -0.005. Report as "equal recall at Nx lower scoring cost".
#
#   sbatch --array=0-19%10 adaptive_acq/slurm/submit_prefilter_curve.sh
#   python -m adaptive_acq.prefilter_real

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-19}"
# AA_SEEDS overrides the seed set, for the decision rule's replication clause.
# The controls it pairs against come from submit_surrogate_sweep.sh arm 4:
#   AA_SEEDS="47 48 49 50 51" sbatch --export=ALL,AA_SEEDS --array=20-24%5 \
#       adaptive_acq/slurm/submit_surrogate_sweep.sh
read -r -a SEEDS <<< "${AA_SEEDS:-42 43 44 45 46}"
[ "${#SEEDS[@]}" = 5 ] || { echo "AA_SEEDS must list exactly 5 seeds"; exit 1; }
seed="${SEEDS[$((TASK_ID % 5))]}"
arm=$((TASK_ID / 5))

case $arm in
  0) NAME=pf20;   EXTRA=(--prefilter-mult 20) ;;
  1) NAME=pf50;   EXTRA=(--prefilter-mult 50) ;;
  2) NAME=pf10r3; EXTRA=(--prefilter-mult 10 --prefilter-refresh 3) ;;
  3) NAME=pf5r3;  EXTRA=(--prefilter-mult 5  --prefilter-refresh 3) ;;
  # Do the two validated wins compose? tw1.0 (~+0.009 recall) acts on the
  # training loss, pf10r3 (2.3x cheaper) on the scoring budget -- different
  # mechanisms, so they should stack, but top-weighting sharpens exactly the
  # ranking the shortlist is built from, so an interaction is plausible.
  4) NAME=combo;  EXTRA=(--prefilter-mult 10 --prefilter-refresh 3 --ltall-top-weight 1.0) ;;
  *) echo "no arm $arm"; exit 1 ;;
esac

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
RUN_DIR="runs/adaptive/prefilter/${NAME}_frac${FRAC}_seed${seed}"
echo "[task $TASK_ID] $NAME seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --controller fixed --beta 0 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${EXTRA[@]}"
