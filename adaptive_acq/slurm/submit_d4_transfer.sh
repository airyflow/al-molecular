#!/bin/bash

#SBATCH -J aa_d4xfer
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_d4xfer_%A_%a.txt
#SBATCH -e logs/aa_d4xfer_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939
#
# Does the top-weighted loss transfer to a SECOND target?
#
# Every adaptive_acq result so far is on ENHITS alone (thymidylate kinase). A
# training-loss trick that helps on one target and not others is a benchmark
# artifact, not a result. D4 (a different target entirely) is the test.
#
# Pool: DATASETS["D4_SUB"] -- a 2,104,318-molecule subsample of D4_dedup,
# exactly ENHITS's pool size, built by adaptive_acq/build_subsample.py. Using
# the subsample rather than full D4 (116M) makes a 5-seed paired test cost
# ~1 GPU-hour instead of ~55. If top-weighting transfers here, the expensive
# full-scale confirmation is worth running; if it does not, that is learned
# cheaply.
#
# NOTE this needs its OWN control: the ENHITS controls pair against a
# different target and pool. Arm 0 below is that control.
#
# Array index = arm_index * 5 + seed_index (10 tasks):
#   0 d4_ctrl   LT-All greedy, plain MSE          (the control)
#   1 d4_tw1.0  LT-All greedy, --ltall-top-weight 1.0
#
# On ENHITS this comparison gave ~+0.009 recall (range +0.004..+0.015 by seed
# set). The pre-registered expectation here: if top-weighting is a real
# property of aligning training with a top-k metric, the D4 effect should also
# be positive. A null or negative result means the ENHITS gain is
# target-specific and should be reported as such.
#
#   sbatch --array=0-9%10 adaptive_acq/slurm/submit_d4_transfer.sh
#   python -m adaptive_acq.d4_transfer

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-9}"
read -r -a SEEDS <<< "${AA_SEEDS:-42 43 44 45 46}"
[ "${#SEEDS[@]}" = 5 ] || { echo "AA_SEEDS must list exactly 5 seeds"; exit 1; }
seed="${SEEDS[$((TASK_ID % 5))]}"
arm=$((TASK_ID / 5))

case $arm in
  0) NAME=d4_ctrl;  EXTRA=() ;;
  1) NAME=d4_tw1.0; EXTRA=(--ltall-top-weight 1.0) ;;
  *) echo "no arm $arm"; exit 1 ;;
esac

# Same protocol as the ENHITS experiments: init == batch == 0.1% of the pool,
# 5 rounds, top-1000. The pool is the same size, so the counts match exactly.
N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
RUN_DIR="runs/adaptive/d4xfer/${NAME}_frac${FRAC}_seed${seed}"
echo "[task $TASK_ID] $NAME seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset D4_SUB --surrogate ltall \
    --backbones molformer grover3400 mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --controller fixed --beta 0 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${EXTRA[@]}"
