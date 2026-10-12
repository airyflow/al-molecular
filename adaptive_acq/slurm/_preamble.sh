# Sourced by every adaptive_acq/slurm/*.sh: cd to repo root, load config.env,
# activate the conda env. Same logic as slurm/al_runs/enamine/*.sh (see the
# SLURM_SUBMIT_DIR comment there for why BASH_SOURCE isn't trusted under sbatch).
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
