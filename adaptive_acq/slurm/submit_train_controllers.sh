#!/bin/bash

#SBATCH -J aa_train
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -o logs/aa_train_%A_%a.txt
#SBATCH -e logs/aa_train_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:30:00
#SBATCH --mem=64G
#SBATCH -A r00939

# Train the learned controllers in the simulator (CPU only).
#   array 0: LinUCB prior     (Stage 1, contextual bandit)
#   array 1: GRPO beta policy (Stage 2)
# Train on D4, validate (GRPO checkpoint selection) on AmpC, evaluate on
# ENHITS with submit_enhits_controllers.sh -- three different targets, so
# the ENHITS number is a genuine transfer estimate.
#
#   sbatch --array=0-1 adaptive_acq/slurm/submit_train_controllers.sh

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SIM="${SIM_TASK_DIR:-adaptive_acq/sim_tasks}"
ART="${ARTIFACT_DIR:-adaptive_acq/artifacts}"
mkdir -p "$ART"

# Wait (up to WAIT_MIN minutes) until every task dir this job reads holds
# N_TASKS .npz files and the newest is >= 2 min old -- so it can be submitted
# while a build is still running (SLURM's afterok can't see a local build).
N_TASKS="${N_TASKS:-4}"
WAIT_MIN="${WAIT_MIN:-180}"
for i in $(seq 1 "$WAIT_MIN"); do
    ready=1
    for d in "$SIM/D4_dedup" "$SIM/AmpC_dedup_pca"; do
        # `|| true`: with set -euo pipefail a failing ls/stat (no .npz yet)
        # would otherwise abort the whole job silently.
        n=$(ls "$d"/*.npz 2>/dev/null | wc -l || true)
        newest=$(stat -c %Y "$d"/*.npz 2>/dev/null | sort -n | tail -1 || true)
        if [ "$n" -lt "$N_TASKS" ] || [ $(( $(date +%s) - ${newest:-0} )) -lt 120 ]; then ready=0; fi
    done
    [ $ready = 1 ] && break
    if [ $((i % 10)) = 1 ]; then
        echo "[wait] sim tasks not ready in $SIM ($(ls "$SIM"/D4_dedup/*.npz 2>/dev/null | wc -l || true) D4," \
             "$(ls "$SIM"/AmpC_dedup_pca/*.npz 2>/dev/null | wc -l || true) AmpC of $N_TASKS)"
    fi
    sleep 60
done
[ $ready = 1 ] || { echo "sim tasks still not ready after $WAIT_MIN min"; exit 1; }

case "${SLURM_ARRAY_TASK_ID:?submit with --array}" in
  0) python -m adaptive_acq.train_linucb --tasks "$SIM/D4_dedup" --episodes 400 \
         --out "$ART/linucb_d4.json" ;;
  1) python -m adaptive_acq.train_grpo --tasks "$SIM/D4_dedup" --val-tasks "$SIM/AmpC_dedup_pca" \
         --iters 200 --groups 4 --group-size 8 --workers "$SLURM_CPUS_PER_TASK" \
         --out "$ART/grpo_d4" ;;
esac
