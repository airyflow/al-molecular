#!/bin/bash

#SBATCH -J ampc_dedup_ltall6pca_ucb
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_dedup_ltall6pca_ucb_%j.txt
#SBATCH -e logs/ampc_dedup_ltall6pca_ucb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -A r00939

# --dataset AmpC_dedup_pca (run_experiment.py's DATASETS["AmpC_dedup_pca"])
# -- identical pool/library/oracle to AmpC_dedup (same 98,489,350
# molecules, same row order, same docking scores), but every backbone's
# embeddings are a per-backbone PCA reduction of AmpC_dedup's own
# embeddings (embed/stitch/pca_reduce_embeddings.py: fit on a 1M-row
# random sample, 95% variance retained, transform the full pool -- see
# that script and slurm/embed/ampc/submit_ampc_pca_reduce_embeddings.sh).
# Purpose: isolate the effect of the dimensionality reduction itself by
# comparing directly against AmpC_dedup's own LT-All results at the same
# frac/acquisition (89.3% greedy / 93.6% UCB at 6 backbones, frac=0.004).
#
# --mem=128G (down from the 6-backbone raw-embedding run's 280G): the
# labeled-set embedding cache's size scales with total concatenated
# width, and PCA cuts every backbone's width substantially (e.g. mhgged
# 1024->88, molformer 768->271, smited 768->148, unimol 512->114) -- total
# post-PCA width is well under half of the raw 8,008-d, so the ~2x
# transient merge peak (see the raw-embedding script's own comment for why
# that 2x exists) should need well under half of 280G. 128G is a
# provisional, still-generous estimate: unimol2 and grover3400's reduced
# widths weren't final when this was written (see logs/ampc_pca_reduce_*)
# -- revisit downward once the true total width is known, if this proves
# oversized in practice.
#
# Own, separate coord-dir/run-dir from every other AmpC run (a different
# embed_dir means different embedding VALUES even where backbone names and
# row count match, so no state from any raw-embedding run is reusable or
# should be mixed in).
#
# TWO-STEP LAUNCH, in order (edit FRAC below FIRST if you want a value
# other than the current one, then match it in the workers command's
# coord-dir path):
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall6pca_ucb_frac0.004
#        sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_dedup_ltall6pca_ucb_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall6pca_ucb_frac0.004 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/ampc/submit_ampc_dedup_ltall6pca_ucb_runs_h100single.sh

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
COORD_DIR="$AMPC_ROOT/al_coord/dedup_ltall6pca_ucb_frac${FRAC}"

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

echo "[orchestrator] dataset=AmpC_dedup_pca surrogate=ltall(6bb,pca) acq=ucb frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup_pca --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol unimol2 \
    --acq ucb --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/ampc_dedup_ltall6pca_ucb_frac${FRAC}" \
    --resume
