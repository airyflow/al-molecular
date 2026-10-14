#!/bin/bash

#SBATCH -J aa_enhits
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_enhits_%A_%a.txt
#SBATCH -e logs/aa_enhits_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# Full 2.1M-molecule ENHITS campaigns (LT-All, 5 backbones, 0.1% init/batch,
# 5 rounds, top-1000) under each beta controller x 5 seeds -- same protocol
# and same seeded init draws as slurm/al_runs/enamine/submit_ltall_enhits_seed_runs_h100single.sh,
# so fixed_b0 here is the in-code-path counterpart of the existing greedy runs.
#
# Array index = config_index * 5 + seed_index. Configs:
#   0 fixed_b0   1 fixed_b0.5   2 fixed_b1   3 fixed_b2   4 raw_b2 (molpal's default ucb)
#   5 sched_decay   6 sched_alternating   7 sched_explore_exploit
#   8 thompson (original wide grid 0..4)   9 linucb (needs artifacts/linucb_d4.json)
#   10 grpo (needs artifacts/grpo_d4/policy_best.pt)   11 thompson_narrow (default grid 0..0.5)
#   12 sched_decay_narrow (linear decay 0.3 -> 0; configs 5-7 start at beta=2, which is far too high)
#   16-18 nbeta0.1 / nbeta0.3 / nbeta1.0: NEGATIVE beta, i.e. anti-exploration
#         (prefer molecules the surrogate is confident about) -- the mirror of
#         the beta gate, which only ever swept positive beta
#   13-15 div_l1 / div_l2 / div_l5: greedy (beta=0) + batch diversity (max-marginal relevance,
#         lambda = 1 / 2 / 5, MoLFormer cosine similarity) -- see acquisition.select_diverse
# 9-11 use the default (narrow) beta grid -- see acquisition.DEFAULT_ARMS.
#
#   sbatch --array=0-44 adaptive_acq/slurm/submit_enhits_controllers.sh    # Stage 1, no training needed
#   sbatch --array=45-54 adaptive_acq/slurm/submit_enhits_controllers.sh   # learned controllers
#   sbatch --array=55-64 adaptive_acq/slurm/submit_enhits_controllers.sh   # thompson + decay, narrow range
#   sbatch --array=65-79 adaptive_acq/slurm/submit_enhits_controllers.sh   # batch diversity, lambda 1/2/5
#   sbatch --array=80-94 adaptive_acq/slurm/submit_enhits_controllers.sh   # NEGATIVE beta (anti-exploration)

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array}"
SEEDS=(42 43 44 45 46)
seed="${SEEDS[$((TASK_ID % 5))]}"
cfg=$((TASK_ID / 5))
ART="${ARTIFACT_DIR:-adaptive_acq/artifacts}"

case $cfg in
  0)  NAME=fixed_b0;              CTRL=(--controller fixed --beta 0) ;;
  1)  NAME=fixed_b0.5;            CTRL=(--controller fixed --beta 0.5) ;;
  2)  NAME=fixed_b1;              CTRL=(--controller fixed --beta 1) ;;
  3)  NAME=fixed_b2;              CTRL=(--controller fixed --beta 2) ;;
  4)  NAME=raw_b2;                CTRL=(--controller fixed --beta 2 --beta-scale raw) ;;
  5)  NAME=sched_decay;           CTRL=(--controller schedule --schedule linear_decay) ;;
  6)  NAME=sched_alternating;     CTRL=(--controller schedule --schedule alternating) ;;
  7)  NAME=sched_explore_exploit; CTRL=(--controller schedule --schedule explore_then_exploit) ;;
  8)  NAME=thompson;              CTRL=(--controller thompson --arms 0 0.25 0.5 1 2 4) ;;
  9)  NAME=linucb;                CTRL=(--controller linucb --controller-path "$ART/linucb_d4.json") ;;
  10) NAME=grpo;                  CTRL=(--controller policy --controller-path "$ART/grpo_d4/policy_best.pt") ;;
  11) NAME=thompson_narrow;       CTRL=(--controller thompson) ;;
  12) NAME=sched_decay_narrow;    CTRL=(--controller schedule --schedule linear_decay --beta-hi 0.3 --beta-lo 0) ;;
  13) NAME=div_l1;                CTRL=(--controller fixed --beta 0 --diversity-lambda 1) ;;
  14) NAME=div_l2;                CTRL=(--controller fixed --beta 0 --diversity-lambda 2) ;;
  15) NAME=div_l5;                CTRL=(--controller fixed --beta 0 --diversity-lambda 5) ;;
  # ANTI-exploration. The beta gate swept 0.05..4 -- all POSITIVE -- and found
  # recall falls monotonically as beta rises. It never asked the mirror
  # question: does beta < 0 help? beta < 0 prefers molecules the surrogate is
  # CONFIDENT about (ucb = mu + beta*sigma becomes LCB), so if LT-All's
  # ensemble disagreement is noise rather than signal here, actively avoiding
  # it could beat greedy. Verified to pick lower-sigma molecules: mean sigma
  # of the batch falls 0.481 (greedy) -> 0.449 (-0.1) -> 0.201 (-1.0).
  16) NAME=nbeta0.1;              CTRL=(--controller fixed --beta -0.1) ;;
  17) NAME=nbeta0.3;              CTRL=(--controller fixed --beta -0.3) ;;
  18) NAME=nbeta1.0;              CTRL=(--controller fixed --beta -1.0) ;;
  *)  echo "no config $cfg"; exit 1 ;;
esac

# Learned controllers: refuse to run on missing or half-trained artifacts.
# train_grpo.py rewrites policy_best.pt whenever validation improves DURING
# training and writes policy_last.pt only at the very end, so policy_last.pt
# existing is the "training finished" signal.
case $cfg in
  9)  [ -f "$ART/linucb_d4.json" ] || { echo "missing $ART/linucb_d4.json -- train first (submit_train_controllers.sh)"; exit 1; } ;;
  10) [ -f "$ART/grpo_d4/policy_last.pt" ] || { echo "GRPO training not finished ($ART/grpo_d4/policy_last.pt missing) -- policy_best.pt may be a mid-training checkpoint"; exit 1; } ;;
esac

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
RUN_DIR="runs/adaptive/${NAME}_frac${FRAC}_seed${seed}"
echo "[task $TASK_ID] $NAME seed=$seed init=batch=$count -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${CTRL[@]}"
