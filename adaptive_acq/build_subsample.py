#!/usr/bin/env python3
"""Build a fixed-size subsample of a huge pool as a standalone dataset, in the
same on-disk format as DATASETS["ENHITS"].

Why: every adaptive_acq result so far is on ENHITS alone (thymidylate kinase,
2.1M). A training-loss trick that helps on one target and not others is a
benchmark artifact, not a result -- but a full-scale D4 campaign is ~7.6 h per
run (116M molecules), so a 5-seed paired test there costs ~55 GPU-hours. This
builds a D4 subsample at exactly the ENHITS pool size so the transfer question
can be answered at ENHITS cost (~6 min/run) first, and the expensive
full-scale test only run if it transfers.

Writes <out-dir>/{backbone}_embeddings.npy plus scores.csv.gz (smiles,score,
row-aligned), which is what EmbeddingFeaturizer + load_oracle expect.

Sampling is by random CONTIGUOUS blocks (default 1000 rows), not scattered
rows: a scattered read of the 100M-row /N/project pools costs ~19 ms/row, so
2.1M scattered rows would take ~11 h per backbone. Blocks are spread at random
over the whole pool so it still spans it, but rows within a block may be
correlated if the source pool is ordered -- the same caveat as sim_env's task
builder. D4's rows are not score-sorted (unlike ENHITS), so this is mild here.

    python -m adaptive_acq.build_subsample --dataset D4_dedup \\
        --backbones molformer grover3400 mhgged smited unimol2 \\
        --n 2104318 --out-dir /N/slate/mengjing/d4_sub
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_complete(path: Path, n_rows: int, probe: int = 2000) -> bool:
    """True if `path` looks fully written. np.lib.format.open_memmap
    PREALLOCATES the file at full size, so a build killed partway leaves a
    file of exactly the right length whose tail is still zeros -- it looks
    finished to any size check. Probing the last rows catches that; a run of
    `probe` genuinely all-zero embedding rows is vanishingly unlikely."""
    if not path.exists():
        return False
    try:
        a = np.load(path, mmap_mode="r")
    except Exception:
        return False
    if a.shape[0] != n_rows:
        return False
    tail = np.asarray(a[-min(probe, a.shape[0]):])
    return bool((np.abs(tail).sum(axis=1) != 0).all())


def write_backbones(args, out: Path, emb_dict, idx):
    """Write one <backbone>_embeddings.npy per backbone, streamed block by
    block so peak RAM is one backbone's slice rather than all of them.

    Derived from `idx` alone (sorted source row indices) so the fresh and
    --resume paths share it. Each backbone is skipped if it already passes
    is_complete(), and verified after writing -- a build killed mid-write
    otherwise leaves a correctly-sized file with a zero tail.
    """
    n, bs = len(idx), args.block_size
    # group the sorted source rows by the contiguous block they came from
    groups = np.split(np.arange(n), np.flatnonzero(np.diff(idx // bs)) + 1)
    for b in args.backbones:
        path = out / f"{b}_embeddings.npy"
        src = emb_dict[b]
        if is_complete(path, n):
            print(f"[build] {b}: already complete -- skipping", flush=True)
            continue
        t0 = time.perf_counter()
        dst = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, src.shape[1]))
        for g in groups:
            rows = idx[g]
            st = int(rows[0] // bs) * bs
            block = np.asarray(src[st:st + bs], dtype=np.float32)
            dst[g[0]:g[0] + len(g)] = block[rows - st]
        dst.flush()
        del dst
        if not is_complete(path, n):
            raise SystemExit(f"{b}: written but failed the integrity probe")
        print(f"[build] {b}: ({n:,}, {src.shape[1]}) in {time.perf_counter() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="source dataset key in run_experiment.DATASETS")
    p.add_argument("--backbones", nargs="+", required=True)
    p.add_argument("--n", type=int, default=2_104_318, help="target pool size (default = ENHITS's)")
    p.add_argument("--block-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true",
                   help="reuse an existing scores.csv.gz/rows.npy and write only the backbones whose "
                        ".npy is missing or incomplete, skipping the (slow) oracle load")
    args = p.parse_args()

    import run_experiment as exp
    exp.DATASET = args.dataset
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows_path, scores_path = out / "rows.npy", out / "scores.csv.gz"
    if args.resume:
        if not scores_path.exists():
            raise SystemExit(f"--resume needs an existing {scores_path}")
        emb_dict, smiles = exp.load_embeddings(args.backbones)
        sub = pd.read_csv(scores_path)
        if rows_path.exists():
            idx = np.load(rows_path)
        else:
            # Older builds did not save rows.npy; recover the source rows from
            # the saved SMILES instead. Still far cheaper than load_oracle().
            print("[build] rows.npy absent -- recovering source rows from scores.csv.gz", flush=True)
            pos = {s: i for i, s in enumerate(smiles)}
            idx = np.array([pos[s] for s in sub.smiles], dtype=np.int64)
            np.save(rows_path, idx)
        if len(idx) != len(sub):
            raise SystemExit(f"rows.npy has {len(idx):,} entries but scores.csv.gz has {len(sub):,}")
        print(f"[build] resume: {len(idx):,} rows", flush=True)
        write_backbones(args, out, emb_dict, idx)
        return

    oracle = exp.load_oracle()
    emb_dict, smiles = exp.load_embeddings(args.backbones)
    n_src = len(smiles)
    scored = np.fromiter((s in oracle for s in smiles), dtype=bool, count=n_src)
    print(f"[build] {args.dataset}: {n_src:,} rows, {scored.sum():,} scored; target {args.n:,}", flush=True)

    # Draw blocks until enough SCORED rows are collected (unscored rows inside a
    # block are dropped, so draw with headroom and trim).
    rng = np.random.default_rng(args.seed)
    need = int(args.n / max(scored.mean(), 1e-9) / args.block_size * 1.15) + 1
    n_blocks = min(need, n_src // args.block_size)
    starts = np.sort(rng.choice(n_src // args.block_size, n_blocks, replace=False)) * args.block_size
    keep = [np.arange(st, st + args.block_size)[scored[st:st + args.block_size]] for st in starts]
    idx = np.concatenate(keep)
    if len(idx) < args.n:
        raise SystemExit(f"only {len(idx):,} scored rows from {n_blocks:,} blocks; raise --block-size or --n lower")
    idx = np.sort(idx[:args.n])
    print(f"[build] selected {len(idx):,} scored rows from {n_blocks:,} blocks", flush=True)

    # scores.csv.gz must be row-aligned with the embeddings written below.
    np.save(rows_path, idx)
    sub_smiles = [smiles[i] for i in idx]
    pd.DataFrame({"smiles": sub_smiles, "score": [oracle[s] for s in sub_smiles]}) \
        .to_csv(out / "scores.csv.gz", index=False)
    print(f"[build] wrote scores.csv.gz ({len(idx):,} rows)", flush=True)

    write_backbones(args, out, emb_dict, idx)

    print(f"[build] done -> {out}")


if __name__ == "__main__":
    main()
