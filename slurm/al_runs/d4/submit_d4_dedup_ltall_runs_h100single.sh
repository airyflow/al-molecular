#!/bin/bash

#SBATCH -J d4_dedup_ltall
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_dedup_ltall_%j.txt
#SBATCH -e logs/d4_dedup_ltall_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=280G
#SBATCH -A r00939

# D4 equivalent of AmpC's submit_ampc_dedup_ltall_runs_h100single.sh --
# against the DEDUPLICATED D4 pool (--dataset D4_dedup): 116,225,927
# molecules (116,241,184 raw, 15,257 duplicate-SMILES rows removed; all 6
# backbones' dedup files verified row-count- and content-aligned against
# the shared keep-mask -- see chat history, not yet written up in the
# report).
#
# --mem=280G: same budget as AmpC's own dedup_ltall run, carried over
# without independent re-measurement. D4's deduped pool (116,225,927) is
# ~18% larger than AmpC's (98,489,350), but this run uses the SAME 5
# backbones/dims as AmpC's (grover3400+mhgged+molformer+smited+unimol,
# fused width 6,472) and the same FRAC, so the final round's cumulative
# labeled count (~2.79M vs AmpC's 2.36M) and merged-cache transient peak
# scale by roughly that same ~18% -- AmpC's 280G had ~130% headroom over
# its own measured ~122GB peak, so this should still comfortably cover
# D4's ~144GB extrapolated peak. Watch round 4-5's merge specifically if
# this is ever wrong; raise if an OOM actually occurs (unverified, not
# AmpC's confirmed number).
#
# --topk 50000: copied directly from AmpC's own value (its own paper
# citation: "top-50,000 (ca. top-0.05%)" on a 98.5M-scale pool). NOT
# independently derived for D4's pool size or checked against any
# published D4 topk convention -- if the paper this D4 pool is drawn from
# states its own top-k definition, use that number instead of this one.
#
# Own, separate coord-dir/run-dir, keyed by FRAC (same reasoning as AmpC's
# script: a different frac has a different init/batch size, so its
# round_N_* state isn't compatible with a previous frac's leftovers).
#
# TWO-STEP LAUNCH, in order (edit FRAC below FIRST if you want a value
# other than the current one, then match it in the workers command's
# coord-dir path):
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/dedup_ltall_frac0.004
#        sbatch --array=0-7 slurm/al_runs/d4/submit_d4_dedup_ltall_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/dedup_ltall_frac0.004 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/d4/submit_d4_dedup_ltall_runs_h100single.sh

set -euo pipefail

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

FRAC=0.004
COORD_DIR="$D4_ROOT/al_coord/dedup_ltall_frac${FRAC}"

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

N=116225927
count=$(python3 -c "print(round($N * $FRAC))")

echo "[orchestrator] dataset=D4_dedup surrogate=ltall acq=greedy frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset D4_dedup --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/d4_dedup_ltall_greedy_frac${FRAC}" \
    --resume
