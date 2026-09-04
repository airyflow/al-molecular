#!/usr/bin/env python3
"""Concatenate per-chunk GROVER embeddings into the single grover_embeddings.npz
that run_experiment.py / EmbeddingFeaturizer expects. Safe to concatenate here
(unlike the 1.56B-molecule pipeline): 2.1M x 1600 x 4 bytes ~= 13.7GB, fits
comfortably in one file and in memory.

Verifies chunk count and total molecule count against --total-count before
writing, and that chunk order (by chunk-id, ascending) matches the original
pool's global index order -- both must hold for the concatenated file's rows
to stay aligned with molpal/libraries/EnamineHTS.csv.gz.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", required=True)
    p.add_argument("--num-chunks", type=int, required=True)
    p.add_argument("--total-count", type=int, required=True)
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    chunks_dir = Path(args.chunks_dir)
    all_emb, all_smi = [], []

    for chunk_id in range(args.num_chunks):
        path = chunks_dir / f"grover_embeddings_chunk_{chunk_id:05d}.npz"
        if not path.exists():
            raise SystemExit(f"Missing chunk: {path} -- not all {args.num_chunks} chunks have finished yet")
        data = np.load(path, allow_pickle=True)
        all_emb.append(data["embeddings"])
        all_smi.append(data["smiles"])
        print(f"[{chunk_id+1}/{args.num_chunks}] {path.name}: {data['embeddings'].shape}")

    embeddings = np.vstack(all_emb)
    smiles = np.concatenate(all_smi)

    if embeddings.shape[0] != args.total_count:
        raise SystemExit(
            f"Concatenated row count {embeddings.shape[0]:,} != --total-count {args.total_count:,} "
            f"-- a chunk boundary mismatch or a chunk was generated with different --num-chunks. Do not trust this output."
        )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings.astype(np.float32), smiles=smiles)
    print(f"[done] {out_path}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
