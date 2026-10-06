#!/bin/bash

#SBATCH -J singlenode_multigpu_test
#SBATCH -p h100-multi
#SBATCH --gpus-per-node h100:4
#SBATCH -o logs/singlenode_multigpu_test_%j.txt
#SBATCH -e logs/singlenode_multigpu_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Easier-to-schedule alternative to slurm/test_multinode_gpu.sh (which
# requests 2 separate nodes on h100-multi and has sat PENDING for a long
# while -- only 12 nodes total in that partition per `sinfo`, evidently
# contended). This asks for 4 GPUs on a SINGLE node instead -- if
# h100-multi's nodes are themselves multi-GPU boxes (plausible given the
# partition's name, distinct from h100-single's one-GPU-per-node nodes),
# this should be much easier for the scheduler to grant and would make
# single-node multi-GPU DDP training straightforward (no cross-node NCCL
# needed at all).
#
# If SLURM grants fewer than 4 GPUs (or this also sits pending a long
# time), that's useful negative information too -- try --gpus-per-node
# h100:2 instead, or check actual per-node GPU count directly via
# `scontrol show node <nodename>` (look for the Gres= line).
#
# Usage: sbatch slurm/test_singlenode_multigpu.sh

set -euo pipefail

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

echo "[job] host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'torch sees {n} GPU(s):')
for i in range(n):
    print(f'  [{i}] {torch.cuda.get_device_name(i)}')
"
