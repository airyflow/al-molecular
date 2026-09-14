#!/bin/bash

#SBATCH -J ampc_ltall
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_ltall_%j.txt
#SBATCH -e logs/ampc_ltall_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH -A r00939

# AmpC (99.5M pool) LT-All (LTAllSurrogate) run -- dedicated orchestrator,
# mirroring submit_ampc_fusion_runs_h100single.sh's (ensemble/learned)
# --parallel-predict protocol exactly, but for LT-All's own backbone set:
#   grover3400 (3400d) + mhgged (1024d) + molformer (768d) + smited (768d)
#   + unimol (512d) = 6,472-d concatenated input.
# This is the ENHITS LT-All config (submit_ltall_enhits_seed_runs_h100single.sh:
# molformer grover mhgged smited unimol2) with unimol2 -> unimol v1 --
# AmpC's own unimol2 extraction isn't finished/stitched yet, so this run
# uses the 5 backbones that ARE already fully extracted and stitched at
# AmpC scale (all verified present in embeddings_striped/, 2026-09-10).
#
# --mem=160G: same reasoning and same measured peak as the ensemble/learned
# orchestrator (load_oracle() alone measured ~24GB RSS for AmpC's 99.46M-row
# scores.csv.gz, plus embedding memmaps + oracle-keys usable_set, plus the
# _emb_cache growth that OOM-killed that orchestrator at 96G) -- LT-All's
# per-round embedding footprint is comparably sized (5 backbones here vs. 3
# for ensemble/learned, so if anything this needs AT LEAST as much headroom,
# not less), so starting directly at the already-hard-won 160G rather than
# re-discovering the same OOM the ensemble run did at 96G.
#
# TWO-STEP LAUNCH, in order (same protocol as ensemble/learned):
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ltall
#        sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_ltall_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/ltall 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/ampc/submit_ampc_ltall_runs_h100single.sh
#
# 1 orchestrator + 8 workers = 9 jobs -- well under the observed
# QOSMaxJobsPerUserLimit, same sizing as the ensemble/learned pools. Running
# ltall concurrently with an ensemble/learned pool needs its own separate
# coord-dir (already the case here: al_coord/ltall, not al_coord/ensemble or
# al_coord/learned) -- ltall's own 9 jobs stack on top of whatever else is
# already running, so check total job count against the QOS limit first if
# another pool is active.
#
# --resume: safe to leave on permanently -- a fresh run_dir with no iter_N
# checkpoint falls back to a normal random init automatically, same as the
# ensemble/learned orchestrator's own --resume.
#
# EXCLUDE_SHARD_IDS (optional 1st arg, comma-separated, e.g. "7"):
# permanently drops shard(s) from candidate selection -- see
# ParallelMVEExplorer's exclude_shard_ids docstring in run_experiment.py.
# Same shard-7 hang history as the ensemble/learned runs may apply here too
# (same underlying embedding-file access pattern), so this is wired through
# from day one rather than added reactively.

set -euo pipefail

EXCLUDE_SHARD_IDS="${1:-}"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"
COORD_DIR="$AMPC_ROOT/al_coord/ltall"

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

N=99459561
FRAC=0.001
count=$(python3 -c "print(round($N * $FRAC))")

EXCLUDE_ARGS=()
if [ -n "$EXCLUDE_SHARD_IDS" ]; then
    EXCLUDE_ARGS=(--exclude-shard-ids "$EXCLUDE_SHARD_IDS")
fi

echo "[orchestrator] dataset=AmpC surrogate=ltall acq=greedy frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR exclude-shard-ids=${EXCLUDE_SHARD_IDS:-none}"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/ampc_ltall_greedy_frac${FRAC}" \
    --resume "${EXCLUDE_ARGS[@]}"
