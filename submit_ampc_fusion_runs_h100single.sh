#!/bin/bash

#SBATCH -J ampc_fusion_probe
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_fusion_probe_%j.txt
#SBATCH -e logs/ampc_fusion_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH -A r00939

# AmpC (99.5M pool, Figure 5) timing PROBE for the fusion/learned_fusion
# comparison on Quartz's h100-single partition (17 idle nodes at time of
# writing, g[63-79]; 2-day/job cap confirmed via `sinfo`/`sbatch --help`).
# Single fraction (0.001 -> init=batch=99,460/round) chosen deliberately
# as a first probe, not a full sweep -- no empirical AL-round timing
# exists at AmpC scale yet.
#
# PREREQUISITE -- SATISFIED: the corrected AmpC UniMol embeddings
# (unimol_embeddings.npy) were re-stitched and verified 2026-08-14 (correct
# shape (99459561, 512), 0 NaN/Inf in a 200k-row sample, sane chunk-boundary
# values). Both bugs behind the original corruption (doubled conformer
# rows, an unseeded per-chunk random final projection) are fixed in
# compute_unimol_embeddings_chunk.py.
#
# Uses --parallel-predict (ParallelMVEExplorer): per-round pool prediction
# over the 99.5M-molecule pool is delegated to a persistent pool of
# single-GPU workers instead of predicting in-process -- surrogate training
# itself stays in-process (the MVE heads are tiny, trained on the labeled
# set only, no reason to distribute that). See predict_pool_shard_worker.py
# and run_experiment.py::ParallelMVEExplorer for the full protocol; verified
# locally to reproduce the sequential (non-parallel) path's results exactly
# before this was wired in here.
#
# --mem=96G: the orchestrator (unlike the workers) calls load_oracle(), which
# alone measured ~24GB RSS for AmpC's 99.46M-row scores.csv.gz (a pandas
# DataFrame of long SMILES strings is genuinely that heavy) -- plus ~5GB for
# opening the 3 embedding memmaps and ~6GB for the oracle-keys usable_set,
# measured peak 35.4GB total (2026-08-14, on colo27). 96G leaves real margin
# above that measured peak; workers never call load_oracle() and stayed well
# under 32G unaided, so only this script's --mem was raised.
#
# TWO-STEP LAUNCH, in order:
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p <AMPC_ROOT>/al_coord/<surrogate>
#        sbatch --array=0-7 submit_ampc_al_predict_workers_h100single.sh \
#            <AMPC_ROOT>/al_coord/<surrogate>
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch submit_ampc_fusion_runs_h100single.sh <ensemble|learned>
#
# One coord-dir per surrogate -- running both "ensemble" and "learned"
# needs two separate worker pools (two separate coord-dirs, two separate
# `sbatch --array` submissions), not one shared pool. At 8 workers each,
# 1 orchestrator + 8 workers = 9 jobs/surrogate, so both together (18) still
# fit under the observed QOSMaxJobsPerUserLimit=20.
#
# --time=48:00:00 uses the full h100-single per-job cap as a safety
# margin for this unknown-duration probe; the wall time observed here sets
# the budget for any larger-fraction follow-up.
#
# --mem bumped 96G -> 160G: the orchestrator OOM-killed partway into round 3
# even at 96G (2026-08-15, node g40) -- root cause not yet fully diagnosed
# (candidates: _emb_cache's growing labeled-embedding cache in
# ParallelMVEExplorer, ~3-4GB/round and climbing; several ~100MB-800MB
# transient arrays in the per-round mask/argsort/acquisition step). Bumped
# with real headroom rather than precisely diagnosed, given another OOM
# would cost more lost compute than a conservative --mem bump.
#
# --resume: the "ensemble" orchestrator was OOM-killed partway into round 3
# (2026-08-15, node g40 -- cause not yet root-caused, unlike the earlier,
# already-fixed load_oracle()/eager-materialization OOMs). --resume makes
# main() look for the latest runs/<run-dir>/iter_N/{state.json,scores.pkl}
# checkpoint (find_resume_checkpoint() in run_experiment.py) and continue
# from there instead of a fresh random init -- recovers the already-labeled
# set exactly, at the cost of NOT preserving the surrogate's trained weights
# (only labeled_scores is checkpointed, not model state_dict, so training
# resumes from a fresh init on the recovered data rather than warm-started --
# verified locally this still produces a valid, correctly-trained model each
# round, just not bit-identical to an uninterrupted run). Safe to leave on
# permanently: a fresh run_dir with no iter_N checkpoint falls back to a
# normal random init automatically.
#
# EXCLUDE_SHARD_IDS (optional 2nd arg, comma-separated, e.g. "7"): permanently
# drops shard(s) from candidate selection -- see ParallelMVEExplorer's
# exclude_shard_ids docstring in run_experiment.py. Built for shard 7's
# repeated hangs on AmpC (single-OST/unstriped embedding files + ~192k
# within-shard duplicate SMILES breaking the contiguous-fetch optimization --
# confirmed via py-spy on a local repro, not just slow but genuinely
# stalling). Combining this with --resume on an ALREADY-RUNNING stuck
# orchestrator is safe: a resumed round's retrain overwrites
# round_R_surrogate.pt on disk, but any shard that already finished round R
# under the OLD checkpoint has already written its round_R_shard_I.done +
# _mu/_var.npy files, and the gather loop (_wait_for_shard) is a pure
# existence poll with no freshness check -- it picks those up immediately
# rather than waiting for new ones, so no already-completed shard work is
# discarded or redone. Do NOT resubmit a worker for an excluded shard --
# it will never be waited on again.

set -euo pipefail

SURROGATE="${1:?Usage: sbatch submit_ampc_fusion_runs_h100single.sh <ensemble|learned> [exclude-shard-ids]}"
EXCLUDE_SHARD_IDS="${2:-}"
case "$SURROGATE" in
    ensemble|learned) ;;
    *) echo "Unknown surrogate '$SURROGATE' -- expected ensemble or learned" >&2; exit 1 ;;
esac

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"
COORD_DIR="$AMPC_ROOT/al_coord/$SURROGATE"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
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

echo "[orchestrator] dataset=AmpC surrogate=$SURROGATE acq=greedy frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR exclude-shard-ids=${EXCLUDE_SHARD_IDS:-none}"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC --mode mve --surrogate "$SURROGATE" \
    --backbones grover molformer unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/ampc_${SURROGATE}_greedy_frac${FRAC}" \
    --resume "${EXCLUDE_ARGS[@]}"
