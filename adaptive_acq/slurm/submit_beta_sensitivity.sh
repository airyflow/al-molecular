#!/bin/bash

#SBATCH -J aa_betasens
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/aa_betasens_%A_%a.txt
#SBATCH -e logs/aa_betasens_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# beta-sensitivity experiment: is there ANY headroom for adapting beta on
# ENHITS (LT-All, 5 backbones)? Answers whether bandits/PPO are worth building.
#
# Part A (tasks 0-59): fixed-beta curve, 5 seeds each.
#   0.1% batch: beta 0.05 0.1 0.2 0.3   (0, 0.5, 1, 2 already in runs/adaptive/fixed_b*)
#   0.2% batch: beta 0 0.1 0.3 0.5
#   0.4% batch: beta 0 0.1 0.3 0.5
# Part B (tasks 60-129): one-round deviations at 0.1% batch. Branch from the
#   greedy run runs/adaptive/fixed_b0_frac0.001_seed<S> after round t-1, use
#   beta in {0.2, 1.0} at round t ONLY, then greedy again; t = 1..5. A beta=0
#   control is branched the same way for t >= 2, so a rerun's own noise
#   cancels in the paired comparison (for t = 1 the existing greedy runs are
#   the control).
#
# Decision rule (fixed before the results): adapting beta is worth building
# controllers for only if some cell beats its paired control at round 5 by
# >= +0.005 with a 95% CI excluding 0, AND that cell replicates on 5 new
# seeds. Analyze with: python -m adaptive_acq.beta_sensitivity
#
#   sbatch --array=0-129%10 adaptive_acq/slurm/submit_beta_sensitivity.sh

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-129}"
SEEDS=(42 43 44 45 46)
N=2104318
OUT=runs/adaptive/betasens

# Build the task table: "kind frac beta t seed" per line.
TABLE=()
for fb in "0.001 0.05" "0.001 0.1" "0.001 0.2" "0.001 0.3" \
          "0.002 0" "0.002 0.1" "0.002 0.3" "0.002 0.5" \
          "0.004 0" "0.004 0.1" "0.004 0.3" "0.004 0.5"; do
    for s in "${SEEDS[@]}"; do TABLE+=("fixed $fb 0 $s"); done
done
for t in 1 2 3 4 5; do
    for b in 0 0.2 1.0; do
        [ "$t" = 1 ] && [ "$b" = 0 ] && continue
        for s in "${SEEDS[@]}"; do TABLE+=("pulse 0.001 $b $t $s"); done
    done
done
[ "${#TABLE[@]}" = 130 ] || { echo "task table has ${#TABLE[@]} rows, expected 130"; exit 1; }
read -r kind frac beta t seed <<< "${TABLE[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

if [ "$kind" = fixed ]; then
    RUN_DIR="$OUT/fixed_b${beta}_frac${frac}_seed${seed}"
    CTRL=(--controller fixed --beta "$beta")
else
    RUN_DIR="$OUT/pulse_t${t}_b${beta}_frac${frac}_seed${seed}"
    CTRL=(--controller schedule --schedule pulse --beta-hi "$beta" --beta-lo 0 --pulse-round "$t")
    BASE="runs/adaptive/fixed_b0_frac${frac}_seed${seed}"
    if [ "$t" -gt 1 ] && [ ! -d "$RUN_DIR" ]; then
        # Branch: copy the greedy run's state after rounds 1..t-1; --resume
        # then continues from round t. (Skipped on a requeue, when RUN_DIR
        # already holds this branch's own progress.)
        mkdir -p "$RUN_DIR"
        for r in $(seq 1 $((t - 1))); do
            [ -f "$BASE/iter_$r/scores.pkl" ] || { echo "missing $BASE/iter_$r -- run fixed_b0 first"; exit 1; }
            cp -r "$BASE/iter_$r" "$RUN_DIR/"
        done
    fi
fi
echo "[task $TASK_ID] $kind frac=$frac beta=$beta t=$t seed=$seed (init=batch=$count) -> $RUN_DIR"

srun --cpu-bind=none python -m adaptive_acq.run_adaptive \
    --dataset ENHITS --surrogate ltall --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume "${CTRL[@]}"
