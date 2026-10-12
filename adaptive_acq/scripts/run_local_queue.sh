#!/bin/bash
# Run a small set of full-pool campaigns sequentially on ONE local GPU (for
# shared interactive nodes without SLURM access). One process at a time,
# niced, --resume so an interrupted queue can just be restarted. The full
# 5-seed sweeps belong on SLURM: adaptive_acq/slurm/submit_enhits_controllers.sh.
#
#   GPU=1 nohup adaptive_acq/scripts/run_local_queue.sh > logs/aa_local_queue.txt 2>&1 &

set -uo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "${CONDA_SH:-/N/slate/mengjing/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-py310}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p logs runs/adaptive
SEED=42
COUNT=2104   # 0.1% init/batch -- same as runs/fusion_*_frac0.001 and ltall_enhits_*_frac0.001

run() {  # name, command...
    local name=$1; shift
    if [ -f "$name/history.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('$name/history.json'))))")" = 5 ]; then
        echo "[skip] $name already complete"; return
    fi
    echo "=== $(date '+%F %T')  $name ==="
    nice -n 10 "$@" --run-dir "$name" --resume || echo "[FAILED] $name"
}

# 1. Fusion fix: original method [3] config (EnamineHTS, grover/molformer/unimol,
#    seed 42), identical except for the surrogate fix -- compare with
#    runs/fusion_{greedy,ucb}_frac0.001.
for acq in greedy ucb; do
    run "runs/fusionfix_${acq}_frac0.001" python run_experiment.py --mode mve --surrogate ensemble \
        --backbones grover molformer unimol --acq "$acq" --init-size $COUNT --batch-size $COUNT \
        --n-rounds 5 --topk 1000 --seed $SEED
done

# 2. Stage 1 adaptive beta on ENHITS (LT-All, 5 backbones), one seed.
AA=(python -m adaptive_acq.run_adaptive --dataset ENHITS --surrogate ltall
    --backbones molformer grover mhgged smited unimol2
    --init-size $COUNT --batch-size $COUNT --n-rounds 5 --topk 1000 --seed $SEED)
run runs/adaptive/fixed_b0_frac0.001_seed$SEED    "${AA[@]}" --controller fixed --beta 0
run runs/adaptive/thompson_frac0.001_seed$SEED    "${AA[@]}" --controller thompson
run runs/adaptive/fixed_b1_frac0.001_seed$SEED    "${AA[@]}" --controller fixed --beta 1
run runs/adaptive/sched_decay_frac0.001_seed$SEED "${AA[@]}" --controller schedule --schedule linear_decay
run runs/adaptive/raw_b2_frac0.001_seed$SEED      "${AA[@]}" --controller fixed --beta 2 --beta-scale raw
run runs/adaptive/fixed_b0.5_frac0.001_seed$SEED  "${AA[@]}" --controller fixed --beta 0.5
echo "=== $(date '+%F %T')  queue done ==="
