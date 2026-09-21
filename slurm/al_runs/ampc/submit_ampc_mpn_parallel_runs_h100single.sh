#!/bin/bash

#SBATCH -J ampc_mpn_par
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mpn_par_%j.txt
#SBATCH -e logs/ampc_mpn_par_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH -A r00939

# Orchestrator for the paper's MolPAL MPN surrogate (--mode molpal --model
# mpn --conf-method mve) on the deduplicated AmpC pool, greedy acquisition,
# at frac=0.004 -- the paper's own AmpC experiment, comparable to the
# paper's 89.3% greedy result and to sections 11-13 of the report.
# ParallelMolPALExplorer (run_experiment.py) trains their MPN here (one
# GPU, retrained FROM SCRATCH every round, as the paper describes) and
# hands per-round MPN checkpoints to prediction workers
# (predict_pool_shard_worker_mpn.py) over the coord-dir marker protocol.
#
# Why per-round worker jobs, not a persistent pool: measured MPN training
# time (probe jobs 10544431/10545879) grows from ~1h to roughly 15h per
# round as the labeled set grows to ~2M, while prediction is ~1.7 ms/mol
# (~47h per full 98.5M pass on ONE GPU). Persistent workers would hold
# their GPUs idle through every multi-hour training phase.
#
# Expected timeline (extrapolated from the probes, NOT measured at this
# scale -- the largest training set actually timed was 200k molecules):
#   training per round ~4h, 6.5h, 9h, 12h, 14.5h  (~46h total)
#   prediction per round ~47h / NUM_SHARDS (~6h at 8 workers, ~3h at 16)
# => well over one 48h job. --resume makes the orchestrator restartable:
# it restores the labeled set from the latest complete iter_N checkpoint
# and continues (MPN weights are not part of resume state, but rounds
# retrain from scratch anyway, so nothing is lost beyond a round's
# in-progress training). Resubmit this same script until history.json
# shows all 5 rounds.
#
# --mem=300G: oracle dict (~24GB) + 98.5M-entry SMILES list + usable mask +
# the true-top-k sort, plus MPN training memory on up to ~2M molecules
# (NOT measured at that size -- raise if a late round OOMs).
#
# LAUNCH (order matters -- workers only for a round whose checkpoint is
# ready; the orchestrator prints "[MPN] trained on N molecules" and then
# waits on shards for that round):
#   1. mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/mpn_frac0.004
#   2. sbatch this script.
#   3. When round R's checkpoint appears (coord-dir round_R_ready.marker),
#      submit that round's workers:
#        sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_mpn_parallel_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/mpn_frac0.004 8 R
#      (--array size must equal NUM_SHARDS; 16 shards also works, pass 16.)

set -euo pipefail

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

FRAC=0.004
NUM_SHARDS=8
COORD_DIR="$AMPC_ROOT/al_coord/mpn_frac${FRAC}"

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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=98489350
count=$(python3 -c "print(round($N * $FRAC))")

echo "[orchestrator] dataset=AmpC_dedup mode=molpal model=mpn acq=greedy frac=$FRAC (init=batch=$count) shards=$NUM_SHARDS coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup \
    --mode molpal --model mpn --conf-method mve --acq greedy \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards "$NUM_SHARDS" --coord-dir "$COORD_DIR" \
    --ncpu 8 --retrain-from-scratch \
    --run-dir "runs/ampc_mpn_greedy_frac${FRAC}" \
    --resume
