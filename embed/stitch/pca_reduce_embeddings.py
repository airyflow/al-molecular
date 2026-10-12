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
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_embedding_store import preallocate, write_slice
from smiles_chunking import _chunk_bounds

READ_BLOCK = 500_000  # rows/block, same as dedup_embeddings.py


class _BlockReader:
    """Reads row ranges of a .npy with plain seek + read instead of
    np.asarray(memmap slice). Same bytes, so results are bit-identical;
    on /N/project, 0.5 GB test reads (2026-10-12) ran ~240 MB/s median
    buffered vs ~115 MB/s via memmap, which also had stalls down to 10 MB/s."""

    def __init__(self, path: str):
        mm = np.load(path, mmap_mode="r")
        self.shape, self.dtype = mm.shape, mm.dtype
        self.row_bytes = mm.shape[1] * mm.dtype.itemsize
        del mm
        self._f = open(path, "rb")
        version = np.lib.format.read_magic(self._f)
        np.lib.format._read_array_header(self._f, version)
        self._offset = self._f.tell()

    def read(self, start: int, end: int) -> np.ndarray:
        self._f.seek(self._offset + start * self.row_bytes)
        raw = self._f.read((end - start) * self.row_bytes)
        return np.frombuffer(raw, dtype=self.dtype).reshape(end - start, self.shape[1])


def _effective_n_sample(n_sample: int, dim: int, backbone: str) -> int:
    # LAPACK (via scipy's full SVD) indexes with 32-bit ints here: a sample
    # matrix over 2**31 ~ 2.15B elements raises "Indexing a matrix of N
    # elements would incur an integer overflow in LAPACK" -- grover3400's
    # 1,000,000 x 3,400 sample crashed this way on 2026-09-17, unimol2's
    # 1,000,000 x 1,536 fit fine. Capped at a conservative 1.5B elements.
    MAX_SAMPLE_ELEMENTS = 1_500_000_000
    effective = min(n_sample, MAX_SAMPLE_ELEMENTS // dim)
    if effective < n_sample:
        print(f"[{backbone}] capping sample to {effective:,} rows (from {n_sample:,}) -- "
              f"dim={dim} would otherwise risk a LAPACK integer-overflow crash during the full-SVD fit")
    return effective


def _sample_rows(emb: _BlockReader, n_total: int, dim: int, n_sample: int, seed: int,
                 start: int = 0, end: int | None = None) -> np.ndarray:
    """Reads the sampled rows that fall in [start, end) via one sequential
    block scan of that range. The sample itself is always drawn over the
    whole pool from the same seed, so the parts gathered by separate range
    tasks concatenate (in range order) to exactly the single-process sample."""
    end = n_total if end is None else end
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    sample_idx.sort()  # so the block scan below can consume it in order
    is_sampled = np.zeros(n_total, dtype=bool)
    is_sampled[sample_idx] = True
    n_wanted = int(is_sampled[start:end].sum())

    out = np.empty((n_wanted, dim), dtype=np.float32)
    cursor = 0
    t0 = time.perf_counter()
    for b, s in enumerate(range(start, end, READ_BLOCK)):
        e = min(s + READ_BLOCK, end)
        block_mask = is_sampled[s:e]
        n_hit = int(block_mask.sum())
        if n_hit:
            block = emb.read(s, e)
            out[cursor:cursor + n_hit] = block[block_mask]
            cursor += n_hit
        if b % 40 == 0:
            print(f"  [sample] {e:,}/{end:,} scanned, {cursor:,}/{n_wanted:,} "
                  f"sampled rows found ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
    assert cursor == n_wanted, f"found {cursor:,} of {n_wanted:,} sampled rows"
    print(f"  [sample] rows [{start:,}, {end:,}) scanned in {time.perf_counter()-t0:.1f}s", flush=True)
    return out


def _load_params(params_path: Path) -> PCA:
    data = np.load(params_path)
    pca = PCA(n_components=data["components"].shape[0])
    pca.components_ = data["components"]
    pca.mean_ = data["mean"]
    pca.explained_variance_ratio_ = data["explained_variance_ratio"]
    pca.n_components_ = data["components"].shape[0]
    return pca


def _fit(backbone: str, n_components, sample: np.ndarray, seed: int, params_path: Path) -> PCA:
    print(f"[{backbone}] fitting PCA (n_components={n_components}) on sample shape {sample.shape}")
    t0 = time.perf_counter()
    svd_solver = "randomized" if isinstance(n_components, int) else "full"
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=seed)
    pca.fit(sample)
    print(f"[{backbone}] fit done in {time.perf_counter()-t0:.1f}s, "
          f"{pca.n_components_} components, "
          f"{pca.explained_variance_ratio_.sum():.4f} variance retained")
    params_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(params_path, components=pca.components_, mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    print(f"[{backbone}] saved fit params -> {params_path}")
    return pca


def _fit_or_load_pca(backbone: str, dim: int, n_components, embeddings_path: str,
                      out_path: str, n_sample: int, seed: int) -> PCA:
    params_path = Path(f"{out_path}.pca_params.npz")
    if params_path.exists():
        pca = _load_params(params_path)
        print(f"[{backbone}] loaded cached PCA fit from {params_path} "
              f"({pca.n_components_} components, "
              f"{pca.explained_variance_ratio_.sum():.4f} variance retained)")
        return pca

    emb = _BlockReader(embeddings_path)
    n_total = emb.shape[0]
    if emb.shape[1] != dim:
        raise SystemExit(f"{embeddings_path} has shape {emb.shape}, expected (*, {dim}) -- check --dim")
    effective_n_sample = _effective_n_sample(n_sample, dim, backbone)
    print(f"[{backbone}] sampling {min(effective_n_sample, n_total):,}/{n_total:,} rows for PCA fit (seed={seed})")
    sample = _sample_rows(emb, n_total, dim, effective_n_sample, seed)
    return _fit(backbone, n_components, sample, seed, params_path)


def _find_resume_point(out_path: str, start: int, end: int) -> int:
    """Binary-searches rows [start, end) of out_path for the last written
    (non-zero) row; returns the first row still to write. Valid because rows
    within one range are written strictly in order."""
    out = np.load(out_path, mmap_mode="r")

    def is_zero_row(i: int) -> bool:
        return bool(np.all(np.asarray(out[i]) == 0))

    if is_zero_row(start):
        return start
    if not is_zero_row(end - 1):
        return end
    lo, hi = start, end - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if is_zero_row(mid):
            hi = mid
        else:
            lo = mid
    return lo + 1


def _transform_range(backbone: str, pca: PCA, emb: _BlockReader, out_path: str,
                     start: int, end: int, resume: bool) -> None:
    k = pca.components_.shape[0]
    scan_start = start
    if resume:
        scan_start = _find_resume_point(out_path, start, end)
        if scan_start >= end:
            print(f"[{backbone}] rows [{start:,}, {end:,}) already written -- nothing to do")
            return
        if scan_start > start:
            print(f"[{backbone}] resuming transform at row {scan_start:,} of [{start:,}, {end:,})")

    # mean_/components_ pulled out of the sklearn object once, so the hot
    # loop below is plain numpy, not attribute lookups on `pca` per block.
    mean_ = pca.mean_
    components_T = pca.components_.T  # (dim, k)

    t0 = time.perf_counter()
    for b, s in enumerate(range(scan_start, end, READ_BLOCK)):
        e = min(s + READ_BLOCK, end)
        block = emb.read(s, e)
        reduced = (block - mean_) @ components_T
        write_slice(out_path, s, e, reduced.astype(np.float32, copy=False))
        if b % 20 == 0:
            print(f"[{backbone}] transformed {e:,}/{end:,} rows "
                  f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)
    print(f"[done] {backbone}: rows [{start:,}, {end:,}) -> {out_path} (dim {emb.shape[1]}->{k}) "
          f"(transform only: {time.perf_counter()-t0:.1f}s)")


def reduce_embeddings(backbone: str, dim: int, n_components, embeddings_path: str,
                       out_path: str, n_sample: int, seed: int, resume: bool) -> None:
    """Single-process path: sample, fit (or load cached fit), transform all rows."""
    pca = _fit_or_load_pca(backbone, dim, n_components, embeddings_path, out_path, n_sample, seed)
    emb = _BlockReader(embeddings_path)
    n_total = emb.shape[0]
    if not (resume and Path(out_path).exists()):
        preallocate(out_path, n_molecules=n_total, dim=pca.components_.shape[0])
    _transform_range(backbone, pca, emb, out_path, 0, n_total, resume and Path(out_path).exists())


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
    p.add_argument("--stage", default="all", choices=["all", "gather", "fit", "transform"],
                   help="all = single process (default). For row-range parallelism run "
                        "gather (array of --num-tasks), then fit (one task), then transform (array).")
    p.add_argument("--task-id", type=int, default=None, help="gather/transform: default SLURM_ARRAY_TASK_ID")
    p.add_argument("--num-tasks", type=int, default=None, help="gather/transform: number of row ranges")
    args = p.parse_args()

    n_components = int(args.n_components) if args.n_components >= 1 else args.n_components
    if args.stage == "all":
        reduce_embeddings(args.backbone, args.dim, n_components, args.embeddings_path,
                          args.out_path, args.n_sample, args.seed, args.resume)
        return

    emb = _BlockReader(args.embeddings_path)
    n_total = emb.shape[0]
    if emb.shape[1] != args.dim:
        raise SystemExit(f"{args.embeddings_path} has shape {emb.shape}, expected (*, {args.dim})")
    params_path = Path(f"{args.out_path}.pca_params.npz")
    parts_dir = Path(f"{args.out_path}.sample_parts")
    if args.stage in ("gather", "transform"):
        if args.num_tasks is None:
            raise SystemExit(f"--stage {args.stage} needs --num-tasks")
        task = args.task_id if args.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        start, end = _chunk_bounds(n_total, task, args.num_tasks)

    if args.stage == "gather":
        part_path = parts_dir / f"part_{task:04d}.npy"
        if args.resume and part_path.exists():
            print(f"[{args.backbone}] {part_path} exists -- nothing to do")
            return
        n_eff = _effective_n_sample(args.n_sample, args.dim, args.backbone)
        part = _sample_rows(emb, n_total, args.dim, n_eff, args.seed, start, end)
        parts_dir.mkdir(parents=True, exist_ok=True)
        tmp = parts_dir / f"part_{task:04d}.tmp.npy"
        np.save(tmp, part)
        tmp.rename(part_path)  # a killed task never leaves a half-written part behind
        print(f"[{args.backbone}] task {task}: {len(part):,} sampled rows -> {part_path}")
    elif args.stage == "fit":
        if args.resume and params_path.exists():
            pca = _load_params(params_path)
            print(f"[{args.backbone}] {params_path} exists -- not refitting")
        else:
            parts = sorted(parts_dir.glob("part_[0-9][0-9][0-9][0-9].npy"))
            n_eff = _effective_n_sample(args.n_sample, args.dim, args.backbone)
            sample = np.concatenate([np.load(f) for f in parts])
            if len(sample) != min(n_eff, n_total):
                raise SystemExit(f"gathered {len(sample):,} sample rows from {len(parts)} parts, expected "
                                 f"{min(n_eff, n_total):,} -- a gather task is missing or used another --num-tasks")
            pca = _fit(args.backbone, n_components, sample, args.seed, params_path)
        preallocate(args.out_path, n_molecules=n_total, dim=pca.components_.shape[0])
    else:  # transform
        if not params_path.exists() or not Path(args.out_path).exists():
            raise SystemExit("run --stage fit first (it saves the params and preallocates the output)")
        _transform_range(args.backbone, _load_params(params_path), emb, args.out_path, start, end, True)


if __name__ == "__main__":
    main()
