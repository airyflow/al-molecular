#!/bin/bash

#SBATCH -J aa_prefilter
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_prefilter_%A_%a.txt
#SBATCH -e logs/aa_prefilter_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# "Score once, then refine" on the real 2.1M ENHITS pool.
#
# Why: a round costs ~70 s, of which only 3-6 s is LT-All training -- the rest
# is scoring the pool. At 13B that is ~120 GPU-hours PER ROUND, while the
# oracle budget does not grow with pool size at all. The beta gate showed
# greedy is within 0.005 of optimal GIVEN full scoring, so the question that
# matters at scale is how few molecules must be scored.
#
# Strategy: round 1 scores the full pool and keeps the top
# PREFILTER_MULT x (total labeling budget); rounds 2-5 only refine within it.
# Simulator gate (adaptive_acq/results/prefilter_sim.json, 3 targets x 3
# seeds) found a loss of <= 0.003 at mult=5 and <= 0.001 at mult=10, for ~4.5x
# and ~4.0x fewer pool passes. BUT the simulator's ENHITS tasks sit at 0.974
# (near saturation, because the simulator labels 6% of its pool vs 0.6% here),
# so D4 (0.479) and AmpC (0.858) were the informative proxies -- hence this
# real-pool confirmation.
#
# Control: runs/adaptive/fixed_b0_frac0.001_seed{42..46} (LT-All greedy, full
# scoring every round, 0.876 +/- 0.006) -- already run, same seeded init draws,
# so the comparison is paired per seed.
#
# Array index = arm_index * 5 + seed_index (15 tasks):
#   0 pf2   shortlist = 2x budget   (~25k of 2.1M)
#   1 pf5   shortlist = 5x budget   (~63k)
#   2 pf10  shortlist = 10x budget  (~126k)
#
# These runs are CHEAPER than the control: rounds 2-5 score only 1-6% of the
# pool. history.json records n_scored/scored_frac per round, so the saving is
# measured rather than extrapolated.
#
# Decision rule, fixed before the results: a mult is acceptable if its recall
# loss at round 5 is within noise of the control -- specifically if the 95% CI
# of the paired difference contains 0 AND its lower bound is above -0.005 (the
# same margin used for the beta gate, applied as a non-inferiority bound).
# Report it as "equal recall at Nx lower scoring cost", never as a gain.
#
#   sbatch --array=0-14%10 adaptive_acq/slurm/submit_prefilter_runs.sh
#   python -m adaptive_acq.prefilter_real

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-14}"
SEEDS=(42 43 44 45 46)
seed="${SEEDS[$((TASK_ID % 5))]}"
arm=$((TASK_ID / 5))
MULTS=(2 5 10)
mult="${MULTS[$arm]}"

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")
RUN_DIR="runs/adaptive/prefilter/pf${mult}_frac${FRAC}_seed${seed}"
echo "[task $TASK_ID] prefilter mult=$mult seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --controller fixed --beta 0 --prefilter-mult "$mult" \
    --seed "$seed" --run-dir "$RUN_DIR" --resume
