#!/bin/bash

#SBATCH -J ltall_enhits_shuf
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -o logs/ltall_enhits_shuf_%A_%a.txt
#SBATCH -e logs/ltall_enhits_shuf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# ENHITS order-leakage robustness check (report Section 10): the exact same
# protocol as slurm/al_runs/enamine/submit_ltall_enhits_seed_runs_h100single.sh
# (LTAllSurrogate, 5-member ensemble, greedy, 0.1% init/batch, 5 rounds,
# seeds 42-46), but against DATASETS["ENHITS_shuffled"] -- embeddings/SMILES
# rows permuted by a fixed random shuffle, de-correlating pool position from
# EnamineHTS_scores.csv.gz's own score-sorted order that ENHITS's embeddings
# were extracted directly from. If recall matches the unshuffled ENHITS runs
# within seed-to-seed noise, that rules out order-leakage as an explanation
# for Section 10's near-exact match to the paper's 87.54%; if shuffled
# recall is meaningfully worse, that confirms it.
#
# BigRed200 CPU (`general`), not h100-single -- surrogates.py's DEVICE
# already falls back to CPU cleanly (torch.cuda.is_available() gate on
# every .to(DEVICE)/AMP call), and LT-All trains a small MLP ensemble over
# already-computed embeddings, not a GPU-bound per-molecule pass like
# extraction was.
#
# --time=24:00:00: the original GPU version's own 4h budget times an
# UNVERIFIED margin for CPU-vs-GPU MLP training slowdown -- not measured.
# Check the first completed task's actual wall time before trusting the
# rest of the array; shrink or grow accordingly for a resubmit.
#
# --mem=200G: same reasoning as the original script (all 5 embedding
# arrays materialized in host RAM once keep_idx-filtered, ~52GB, plus
# overhead) -- algorithm/memory footprint unchanged by CPU vs GPU.
#
# PREREQUISITE: both submit_enhits_build_shuffle_permutation.sh and
# submit_enhits_shuffle_embeddings.sh (all 5 backbones) must have finished
# first -- this reads DATASETS["ENHITS_shuffled"]'s embed_dir directly.
#
# Array index <-> seed: 0->42  1->43  2->44  3->45  4->46
#
# Usage:
#   sbatch --array=0-4 slurm/al_runs/enamine/submit_ltall_enhits_shuffled_seed_runs_bigred_cpu.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-4 (SLURM_ARRAY_TASK_ID unset)}"

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
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=""

SEEDS=(42 43 44 45 46)
seed="${SEEDS[$TASK_ID]}"

N=2104318
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")

RUN_DIR="runs/ltall_enhits_shuffled_greedy_frac${FRAC}_seed${seed}"

echo "[task $TASK_ID] surrogate=ltall acq=greedy frac=$FRAC (init=batch=$count) seed=$seed dataset=ENHITS_shuffled -> $RUN_DIR"

srun --cpu-bind=none python run_experiment.py \
    --dataset ENHITS_shuffled --mode mve --surrogate ltall --acq greedy \
    --backbones molformer grover mhgged smited unimol2 \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --seed "$seed" --run-dir "$RUN_DIR" --resume
