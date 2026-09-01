#!/usr/bin/env python3
"""Stitch Yang's pre-computed ENHITS-large per-shard embeddings into the
one-file-per-backbone .npy layout run_experiment.py / EmbeddingFeaturizer
expects.

Source layout (read-only, on project storage):

    <SRC>/shard_manifest.csv                     # 211 shards, 10k rows each
    <SRC>/shards/shard_000000.csv ...            # embed_idx,source_row_idx,smiles
    <SRC>/embedding_outputs/
        molformer_grover/molformer/shards/emb_shard_XXXXXX.npy   ->  molformer  (768)
        molformer_grover/grover/shards/emb_shard_XXXXXX.npy      ->  grover     (3400)
        fm4m/MHG-GED/shards/emb_shard_XXXXXX.npy                 ->  mhgged     (1024)
        fm4m/SMI-TED/shards/emb_shard_XXXXXX.npy                 ->  smited     (768)
        unimol/unimolv2/shards/emb_shard_XXXXXX.npy             ->  unimol2    (768)

All five backbones share the same shard split (shard_manifest.csv) and are
row-aligned by `source_row_idx`; the shard .npy rows are in embed_idx order,
which is a plain 0..N-1 range identical to the row order of the source
scores CSV (data/EnamineHTS_scores.csv.gz). So stitching is just an
in-order concatenation of emb_shard_000000..000210 per backbone.

Output layout (what this writes):

    <OUT>/molformer_embeddings.npy      (N, 768)   float32, mmap-ready
    <OUT>/grover_embeddings.npy         (N, 3400)
    <OUT>/mhgged_embeddings.npy         (N, 1024)
    <OUT>/smited_embeddings.npy         (N, 768)
    <OUT>/unimol2_embeddings.npy        (N, 768)
    <OUT>/enhits_large_smiles.txt       N lines, embedding row order

EmbeddingFeaturizer then loads each <bb>_embeddings.npy memory-mapped and
recovers row-order SMILES from the smiles_source file (no per-backbone .npz
sibling needed). Peak RAM is one shard (~140 MB for grover), never a whole
backbone array.

Usage
-----
    python stitch_enhits_large_shards.py \
        --src "/N/project/SingleCell_Image/Yang/AI Drug/Emb output/EnamineHITS_large_embedding_shards" \
        --out /N/project/SingleCell_Image/mengjing/enhits_large/embed \
        --oracle data/EnamineHTS_scores.csv.gz          # optional cross-check

Resumable: a <bb>_embeddings.npy.stitch_progress sidecar records which
shards were copied, so a re-run skips finished shards (and skips whole
backbones whose output already has the right shape and full progress).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# backbone name (run_experiment --backbones choice)  ->  sub-path under
# <src>/embedding_outputs/ that holds that backbone's shards/ dir
BACKBONE_SUBDIR = {
    "molformer": "molformer_grover/molformer",
    "grover":    "molformer_grover/grover",
    "mhgged":    "fm4m/MHG-GED",
    "smited":    "fm4m/SMI-TED",
    "unimol2":   "unimol/unimolv2",
}


def read_manifest(src: Path):
    """Return [(shard_id, start, end)] from shard_manifest.csv, ordered."""
    import csv

    rows = []
    with open(src / "shard_manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["shard_id"]), int(r["start_embed_idx"]), int(r["end_embed_idx"])))
    rows.sort()
    ids = [i for i, _, _ in rows]
    if ids != list(range(len(ids))):
        raise SystemExit(f"shard_manifest.csv shard_ids are not a clean 0..{len(ids)-1} range")
    if rows[0][1] != 0 or any(rows[k][2] != rows[k + 1][1] for k in range(len(rows) - 1)):
        raise SystemExit("shard_manifest.csv rows are not contiguous / do not start at 0")
    return rows


def load_smiles_from_shard_csvs(src: Path, manifest) -> list[str]:
    import csv

    smiles: list[str] = []
    for shard_id, start, end in manifest:
        p = src / "shards" / f"shard_{shard_id:06d}.csv"
        with open(p, newline="") as f:
            rdr = csv.DictReader(f)
            got = [row["smiles"] for row in rdr]
        if len(got) != end - start:
            raise SystemExit(f"{p}: {len(got)} rows, manifest says {end - start}")
        smiles.extend(got)
    return smiles


def stitch_backbone(bb: str, src: Path, out: Path, manifest, total: int) -> None:
    shard_dir = src / "embedding_outputs" / BACKBONE_SUBDIR[bb] / "shards"
    shard_paths = [shard_dir / f"emb_shard_{sid:06d}.npy" for sid, _, _ in manifest]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit(f"[{bb}] {len(missing)} shard files missing, e.g. {missing[0]}")

    dim = int(np.load(shard_paths[0], mmap_mode="r").shape[1])
    out_path = out / f"{bb}_embeddings.npy"
    progress_path = out_path.with_suffix(".npy.stitch_progress")

    done: set[int] = set()
    if out_path.exists():
        mm = np.lib.format.open_memmap(out_path, mode="r")
        if mm.shape != (total, dim):
            raise SystemExit(
                f"[{bb}] {out_path} exists with shape {mm.shape}, expected {(total, dim)} "
                f"-- remove it explicitly to re-stitch"
            )
        del mm
        if progress_path.exists():
            done = {int(x) for x in progress_path.read_text().split() if x.strip()}
        if len(done) == len(manifest):
            print(f"[{bb}] already complete ({out_path}, {(total, dim)}) -- skipping")
            return
    else:
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(total, dim))
        del mm
        print(f"[{bb}] preallocated {out_path}  shape={(total, dim)}")

    out_mm = np.lib.format.open_memmap(out_path, mode="r+")
    n_now = 0
    with open(progress_path, "a") as pf:
        for (sid, start, end), sp in zip(manifest, shard_paths):
            if sid in done:
                continue
            arr = np.load(sp)
            if arr.shape != (end - start, dim):
                raise SystemExit(f"[{bb}] {sp} shape {arr.shape}, expected {(end - start, dim)}")
            out_mm[start:end] = arr.astype(np.float32, copy=False)
            pf.write(f"{sid}\n")
            pf.flush()
            n_now += 1
            if n_now % 25 == 0:
                print(f"[{bb}] {len(done) + n_now}/{len(manifest)} shards", flush=True)
    out_mm.flush()
    del out_mm
    print(f"[{bb}] done: {n_now} shards this run ({len(done) + n_now}/{len(manifest)}) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/N/project/SingleCell_Image/Yang/AI Drug/Emb output/EnamineHITS_large_embedding_shards",
        help="root holding shard_manifest.csv, shards/, embedding_outputs/",
    )
    ap.add_argument("--out", required=True, help="output dir for <bb>_embeddings.npy + smiles txt")
    ap.add_argument(
        "--backbones", nargs="+", default=list(BACKBONE_SUBDIR),
        choices=list(BACKBONE_SUBDIR),
    )
    ap.add_argument(
        "--oracle", default=None,
        help="optional data/EnamineHTS_scores.csv.gz to cross-check SMILES row alignment",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(src)
    total = manifest[-1][2]
    print(f"[manifest] {len(manifest)} shards, {total:,} molecules total")

    smi_path = out / "enhits_large_smiles.txt"
    if smi_path.exists() and sum(1 for _ in open(smi_path)) == total:
        print(f"[smiles] {smi_path} already has {total:,} lines -- keeping")
        smiles = [l.rstrip("\n") for l in open(smi_path)]
    else:
        print("[smiles] concatenating shard CSV smiles columns ...")
        smiles = load_smiles_from_shard_csvs(src, manifest)
        assert len(smiles) == total
        tmp = smi_path.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(smiles) + "\n")
        tmp.rename(smi_path)
        print(f"[smiles] wrote {smi_path}  ({total:,} lines)")

    if args.oracle:
        import pandas as pd

        df = pd.read_csv(args.oracle)
        df.columns = df.columns.str.strip().str.lower()
        col = next(c for c in df.columns if "smiles" in c)
        o: list[str] = df[col].dropna().tolist()
        n_match = sum(1 for a, b in zip(o, smiles) if a == b)
        print(
            f"[oracle-check] {args.oracle}: {len(o):,} rows, "
            f"{n_match:,}/{min(len(o), len(smiles)):,} positionally identical to shard order"
        )
        if len(o) == total and n_match == total:
            print("[oracle-check] PASS -- shard embedding row order == oracle CSV row order")
        else:
            print(
                "[oracle-check] NOTE: not a perfect positional match. Matching in "
                "run_experiment.py is by SMILES string, not row position, so this is "
                "only fatal if many shard SMILES are absent from the oracle."
            )
            missing = set(smiles) - set(o)
            print(f"[oracle-check] shard SMILES not present in oracle at all: {len(missing):,}")

    for bb in args.backbones:
        stitch_backbone(bb, src, out, manifest, total)

    print("\n[all done] point run_experiment.py's DATASETS[...]['embed_dir'] at:")
    print(f"    {out}")
    print(f"  and ['library'] / ['oracle'] at your EnamineHTS_scores.csv.gz")


if __name__ == "__main__":
    sys.exit(main())
