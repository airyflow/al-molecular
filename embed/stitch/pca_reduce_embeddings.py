#!/usr/bin/env python3
"""Fits a per-backbone PCA on a random sample of the pool, then transforms
every row and writes a reduced-width (N, k) embeddings.npy compatible with
EmbeddingFeaturizer.load() (molpal/featurizer.py) -- same format as the raw
stitched/deduplicated files, just narrower.

Two passes over the source file:
  1. Fit pass: uniformly samples --n-sample row indices (no replacement,
     seeded), reads them via one sequential block scan (same technique as
     dedup_embeddings.py -- a boolean mask over each block, not scattered
     fancy-indexing against the memmap), and fits sklearn PCA on the
     resulting (n_sample, dim) matrix held in memory. Cached to
     <out-path>.pca_params.npz so a resumed run never re-fits.
  2. Transform pass: streams the FULL pool sequentially in blocks, applies
     (block - mean_) @ components_.T, and writes each block's reduced rows
     via shared_embedding_store.write_slice. Resumable (binary-searches the
     output file for the last non-zero row, same as dedup_embeddings.py),
     since this pass is the expensive, full-pool one.

Sizing --n-sample: memory cost of the fit pass is roughly
n_sample * dim * 4 bytes (float32). E.g. 1,000,000 rows at dim=3400
(GROVER3400) is ~13.6GB; at dim=1536 (Uni-Mol2) ~6.1GB. Size down for wider
backbones if running on a memory-constrained node.

Usage
-----
python pca_reduce_embeddings.py --backbone grover3400 --dim 3400 --n-components 500 \\
    --embeddings-path /path/to/dedup/grover3400_embeddings.npy \\
    --out-path /path/to/dedup_pca/grover3400_embeddings.npy \\
    --n-sample 1000000 --seed 42

--resume: safe/idempotent to pass every time (mirrors dedup_embeddings.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_embedding_store import preallocate, write_slice

READ_BLOCK = 500_000  # rows/block, same as dedup_embeddings.py


def _sample_rows(emb: np.memmap, n_total: int, dim: int, n_sample: int, seed: int) -> np.ndarray:
    """Reads n_sample uniformly-random rows via one sequential block scan."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    sample_idx.sort()  # so the block scan below can consume it in order
    is_sampled = np.zeros(n_total, dtype=bool)
    is_sampled[sample_idx] = True

    out = np.empty((len(sample_idx), dim), dtype=np.float32)
    cursor = 0
    t0 = time.perf_counter()
    for start in range(0, n_total, READ_BLOCK):
        end = min(start + READ_BLOCK, n_total)
        block_mask = is_sampled[start:end]
        n_hit = int(block_mask.sum())
        if n_hit:
            block = np.asarray(emb[start:end])
            out[cursor:cursor + n_hit] = block[block_mask]
            cursor += n_hit
        if (start // READ_BLOCK) % 40 == 0:
            print(f"  [sample] {end:,}/{n_total:,} scanned, {cursor:,}/{len(sample_idx):,} "
                  f"sampled rows found ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
    assert cursor == len(sample_idx), f"found {cursor:,} of {len(sample_idx):,} sampled rows"
    return out


def _fit_or_load_pca(backbone: str, dim: int, n_components, embeddings_path: str,
                      out_path: str, n_sample: int, seed: int) -> PCA:
    params_path = Path(f"{out_path}.pca_params.npz")
    if params_path.exists():
        data = np.load(params_path)
        pca = PCA(n_components=data["components"].shape[0])
        pca.components_ = data["components"]
        pca.mean_ = data["mean"]
        pca.explained_variance_ratio_ = data["explained_variance_ratio"]
        pca.n_components_ = data["components"].shape[0]
        print(f"[{backbone}] loaded cached PCA fit from {params_path} "
              f"({pca.n_components_} components, "
              f"{data['explained_variance_ratio'].sum():.4f} variance retained)")
        return pca

    emb = np.load(embeddings_path, mmap_mode="r")
    n_total = emb.shape[0]
    if emb.shape[1] != dim:
        raise SystemExit(f"{embeddings_path} has shape {emb.shape}, expected (*, {dim}) -- check --dim")

    print(f"[{backbone}] sampling {min(n_sample, n_total):,}/{n_total:,} rows for PCA fit (seed={seed})")
    sample = _sample_rows(emb, n_total, dim, n_sample, seed)

    print(f"[{backbone}] fitting PCA (n_components={n_components}) on sample shape {sample.shape}")
    t0 = time.perf_counter()
    svd_solver = "randomized" if isinstance(n_components, int) else "full"
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=seed)
    pca.fit(sample)
    print(f"[{backbone}] fit done in {time.perf_counter()-t0:.1f}s, "
          f"{pca.n_components_} components, "
          f"{pca.explained_variance_ratio_.sum():.4f} variance retained")
    del sample

    params_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(params_path, components=pca.components_, mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    print(f"[{backbone}] saved fit params -> {params_path}")
    return pca


def _find_resume_point(out_path: str, n_total: int) -> int:
    """Binary-searches out_path for the last written (non-zero) row.
    Mirrors dedup_embeddings.py's _find_resume_point."""
    out = np.load(out_path, mmap_mode="r")

    def is_zero_row(i: int) -> bool:
        return bool(np.all(np.asarray(out[i]) == 0))

    if not is_zero_row(0):
        lo = 0
    else:
        return 0
    if not is_zero_row(n_total - 1):
        return n_total
    hi = n_total - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if is_zero_row(mid):
            hi = mid
        else:
            lo = mid
    return lo + 1


def reduce_embeddings(backbone: str, dim: int, n_components, embeddings_path: str,
                       out_path: str, n_sample: int, seed: int, resume: bool) -> None:
    pca = _fit_or_load_pca(backbone, dim, n_components, embeddings_path, out_path, n_sample, seed)
    k = pca.components_.shape[0]

    emb = np.load(embeddings_path, mmap_mode="r")
    n_total = emb.shape[0]

    scan_start = 0
    if resume and Path(out_path).exists():
        scan_start = _find_resume_point(out_path, n_total)
        if scan_start >= n_total:
            print(f"[{backbone}] already fully written ({n_total:,}/{n_total:,}) -- nothing to do")
            return
        print(f"[{backbone}] resuming transform pass from row {scan_start:,}/{n_total:,}")
    else:
        preallocate(out_path, n_molecules=n_total, dim=k)

    # mean_/components_ pulled out of the sklearn object once, so the hot
    # loop below is plain numpy, not attribute lookups on `pca` per block.
    mean_ = pca.mean_
    components_T = pca.components_.T  # (dim, k)

    t0 = time.perf_counter()
    for start in range(scan_start, n_total, READ_BLOCK):
        end = min(start + READ_BLOCK, n_total)
        block = np.asarray(emb[start:end])
        reduced = (block - mean_) @ components_T
        write_slice(out_path, start, end, reduced.astype(np.float32, copy=False))
        if (start // READ_BLOCK) % 20 == 0:
            print(f"[{backbone}] transformed {end:,}/{n_total:,} rows "
                  f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)

    print(f"[done] {backbone}: {n_total:,} rows -> {out_path} "
          f"(dim {dim}->{k}) ({time.perf_counter()-t0:.1f}s total)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dim", type=int, required=True, help="source embedding width")
    p.add_argument("--n-components", type=float, required=True,
                   help="target width (int, e.g. 500) or variance-to-retain fraction (float in (0,1), e.g. 0.95)")
    p.add_argument("--embeddings-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--n-sample", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    n_components = int(args.n_components) if args.n_components >= 1 else args.n_components
    reduce_embeddings(args.backbone, args.dim, n_components, args.embeddings_path,
                       args.out_path, args.n_sample, args.seed, args.resume)


if __name__ == "__main__":
    main()
