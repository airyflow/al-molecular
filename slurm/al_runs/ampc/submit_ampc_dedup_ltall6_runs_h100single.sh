#!/bin/bash

#SBATCH -J ampc_dedup_ltall6
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_dedup_ltall6_%j.txt
#SBATCH -e logs/ampc_dedup_ltall6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=280G
#SBATCH -A r00939

# --mem=280G (bumped from 160G, 2026-09-14): the labeled-set embedding
# cache (self._emb_cache_idx/_emb_cache_X in run_experiment.py) merges
# old+new rows into a freshly-allocated array every round -- during that
# merge, the OLD cache array, the newly-fetched rows, and the new MERGED
# array are all resident simultaneously (a real, unavoidable ~2x transient
# peak over the merged size itself, not a bug in the merge logic). At
# frac=0.004 this OOM-killed a real run at 160G during round 4's merge
# (~1.97M cumulative rows, ~51GB merged array -> ~102GB transient peak
# for the merge alone, plus ~24GB oracle dict, plus baseline overhead).
# 280G gives comfortable headroom through round 5 (~2.36M rows, ~61GB
# merged, ~122GB transient peak). Higher frac or more rounds would need
# this raised further, or a capacity-doubling append-based cache instead
# of a full re-merge every round (not yet implemented).

# Same as submit_ampc_ltall_runs_h100single.sh, but against the
# DEDUPLICATED AmpC pool (--dataset AmpC_dedup, run_experiment.py's
# DATASETS["AmpC_dedup"]): 98,489,350 molecules instead of 99,459,561 --
# 970,211 duplicate-SMILES rows removed, ~957,702 of which (98.7% of all
# duplicates in the pool) were concentrated in shard 7 alone (7.7% of its
# rows), which is what was actually causing shard 7's chronic worker hangs
# across multiple different runs (a real logic bug in
# molpal/models/mvemodels.py's EmbeddingMVEModel._get_X(), separately
# fixed -- not a node/hardware issue). This run no longer needs
# --exclude-shard-ids at all.
#
# Adds a 6th backbone (unimol2) on top of the 5-backbone LT-All run
# (grover3400 mhgged molformer smited unimol). unimol2's raw stitched
# embeddings (embeddings_striped/unimol2_embeddings.npy, 99,459,561 rows)
# had to be deduplicated first via embed/stitch/dedup_embeddings.py against
# the same keep_mask.npy used for the other 5, to bring it down to the
# AmpC_dedup pool's 98,489,350 rows -- see dedup/unimol2_embeddings.npy.
#
# Own, separate coord-dir/run-dir from BOTH the non-deduplicated AmpC run
# AND the 5-backbone dedup runs (al_coord/dedup_ltall6_frac<FRAC> /
# runs/ampc_dedup_ltall6_greedy_frac<FRAC>, not al_coord/dedup_ltall_frac<FRAC>
# or al_coord/ltall) -- a different backbone set changes the per-round
# embedding cache's total dim and the concatenation order, so no state
# from the 5-backbone or non-dedup runs is reusable or should be mixed in.
# Both paths are also keyed by FRAC itself (see below) -- a different frac
# needs its own clean coord-dir too, since init/batch size differs.
#
# TWO-STEP LAUNCH, in order (edit FRAC below FIRST if you want a value
# other than the current one, then match it in the workers command's
# coord-dir path):
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall6_frac0.004
#        sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_dedup_ltall6_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall6_frac0.004 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/ampc/submit_ampc_dedup_ltall6_runs_h100single.sh

set -euo pipefail

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

# FRAC is baked into BOTH coord-dir and run-dir names -- a run at a
# different frac has a different init/batch size, so its round_N_* marker
# files, surrogate checkpoints, and shard predictions are NOT compatible
# with a previous frac's leftovers. Keying both paths by frac means
# switching frac never requires manually wiping the old coord-dir first
# (confirmed necessary: frac=0.001's completed 5-round run left
# round_1..5_*/STOP.marker sitting in al_coord/dedup_ltall, which a plain
# frac=0.002 rerun against that SAME directory would have collided with).
FRAC=0.004
COORD_DIR="$AMPC_ROOT/al_coord/dedup_ltall6_frac${FRAC}"

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

echo "[orchestrator] dataset=AmpC_dedup surrogate=ltall(6bb) acq=greedy frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol unimol2 \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/ampc_dedup_ltall6_greedy_frac${FRAC}" \
    --resume
