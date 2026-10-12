#!/bin/bash
# Submit the whole adaptive_acq + fusion-fix pipeline with dependencies.
# Run from the repo root on a SLURM login node:
#
#   bash adaptive_acq/slurm/submit_all.sh            # everything
#   bash adaptive_acq/slurm/submit_all.sh --dry-run  # print the sbatch commands only
#
# Jobs (GPU = h100-single, CPU = general):
#   A  fusionfix      30 GPU tasks  fixed EnsembleFusion, 3 fracs x greedy/ucb x 5 seeds   (no deps)
#   B  stage1         45 GPU tasks  9 non-learned beta controllers x 5 seeds on ENHITS      (no deps)
#   C  build_sim       3 CPU tasks  simulator tasks from D4 / AmpC / ENHITS                 (no deps)
#   D  train           2 CPU tasks  LinUCB prior + GRPO policy                              (after C)
#   E  learned        10 GPU tasks  LinUCB + GRPO controllers x 5 seeds on ENHITS           (after D)
#
# GPU arrays are throttled (%MAX_GPU concurrent tasks each) to stay polite on
# h100-single; raise MAX_GPU if the queue is empty.

set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p logs
MAX_GPU="${MAX_GPU:-10}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

sub() {  # prints the job id (or a placeholder when dry-running)
    if [ $DRY = 1 ]; then
        echo "sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

S=adaptive_acq/slurm
A=$(sub --array=0-29%"$MAX_GPU" $S/submit_fusionfix_runs.sh)
B=$(sub --array=0-44%"$MAX_GPU" $S/submit_enhits_controllers.sh)
C=$(sub --array=0-2 $S/submit_build_sim_tasks.sh)
D=$(sub --array=0-1 --dependency=afterok:"$C" $S/submit_train_controllers.sh)
E=$(sub --array=45-54%"$MAX_GPU" --dependency=afterok:"$D" $S/submit_enhits_controllers.sh)

cat <<MSG
submitted:
  A fusionfix   $A
  B stage1      $B
  C build_sim   $C
  D train       $D   (after C)
  E learned     $E   (after D)

monitor:   squeue -u \$USER
results:   python -m adaptive_acq.aggregate runs/fusionfix_*_seed* runs/fusion_greedy_frac* runs/fusion_ucb_frac*
           python -m adaptive_acq.aggregate runs/adaptive/*_seed* runs/ltall_enhits_greedy_frac0.001_seed4*
MSG
