#!/bin/bash
# Submits the three stages of row-range parallel PCA for one backbone,
# chained so each starts only if the previous one succeeded:
#   gather    (array of NUM_TASKS)  sampled rows of each row range
#   fit       (1 task)              fit on the joined sample, save params, preallocate output
#   transform (array of NUM_TASKS)  reduce each row range
# Output is byte-identical to the single-process --stage all run (tested
# 2026-10-12 on a 1.2M-row file, including a killed-and-resumed transform task).
#
# Run with bash, not sbatch:
#   bash slurm/embed/submit_pca_parallel.sh <backbone> <dim> <embeddings-path> <out-path> <num-tasks>
# e.g.
#   bash slurm/embed/submit_pca_parallel.sh molformer 768 \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/dedup/molformer_embeddings.npy \
#       /N/project/SingleCell_Image/mengjing/ampc_99.5M/dedup_pca_par/molformer_embeddings.npy 8

set -euo pipefail
BACKBONE="${1:?backbone}"; DIM="${2:?dim}"; SRC="${3:?embeddings-path}"; OUT="${4:?out-path}"; T="${5:?num-tasks}"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p "$(dirname "$OUT")"
lfs setstripe -c -1 "$(dirname "$OUT")" 2>/dev/null || true
LAST=$((T - 1))
ARGS=("$BACKBONE" "$DIM" "$SRC" "$OUT" "$T")

g=$(sbatch --parsable -J "pca_gather_$BACKBONE" --array=0-$LAST slurm/embed/pca_stage.sh gather "${ARGS[@]}")
f=$(sbatch --parsable -J "pca_fit_$BACKBONE" --dependency=afterok:$g slurm/embed/pca_stage.sh fit "${ARGS[@]}")
t=$(sbatch --parsable -J "pca_transform_$BACKBONE" --dependency=afterok:$f --array=0-$LAST slurm/embed/pca_stage.sh transform "${ARGS[@]}")
echo "gather=$g fit=$f transform=$t ($BACKBONE, $T row ranges)"
