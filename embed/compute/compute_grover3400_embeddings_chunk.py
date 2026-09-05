#!/usr/bin/env python3
"""Real GROVER (tencent-ailab/grover) 3400-d "both" fingerprint extraction,
chunked, single-stage. Reproduces Yang's ENHITS grover embeddings, not
muben's simplified 1600-d one (see compute_grover_embeddings_chunk.py) --
verified bit-exact (max abs diff 3.5e-6, cosine sim > 0.9999999999999 on a
20-molecule sample against Yang's own emb_shard_000000.npy) before this
script was written.

Why subprocess, not a direct Python import (unlike every other
compute_*_embeddings_chunk.py in this repo): the official grover package's
`generate_fingerprints()` (grover/task/fingerprint.py) expects a fully
argparse-populated Namespace built by grover/util/parsing.py's
parse_args()/modify_fingerprint_args() -- reconstructing every attribute
that machinery sets by hand risks silently missing one and diverging from
the validated path. Invoking `python main.py fingerprint` and
`python scripts/save_features.py` as subprocesses exactly reuses the
literal CLI path already verified bit-exact, at the cost of a small
subprocess-per-chunk overhead (model load happens once per chunk either
way, same as every other backbone's script).

Two-step recipe (see grover/grover/model/models.py:GroverFpGeneration.forward,
--fingerprint_source both): concat(
    mean_readout(atom_from_atom, a_scope),   # 800
    mean_readout(atom_from_bond, a_scope),   # 800
    mean_readout(bond_from_atom, b_scope),   # 800
    mean_readout(bond_from_bond, b_scope),   # 800
    rdkit_2d_normalized_features(smiles),    # 200
) = 3400

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks. Once every chunk finishes, run
stitch_embedding_chunks.py once (backbone "grover3400", dim 3400) to
produce the final shared .npy that EmbeddingFeaturizer.load() expects.

Usage
-----
python compute_grover3400_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --chunk-id 0 --num-chunks 1000 \\
    --chunks-dir /path/to/_grover3400_chunks
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (this file lives in embed/compute/)
GROVER_ROOT = ROOT / "grover"

sys.path.insert(0, str(ROOT))
from smiles_chunking import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

DEFAULT_CHECKPOINT = "/N/project/SingleCell_Image/Yang/AI Drug/Emb output/model.pt"


def compute_embeddings_for_chunk(
    chunk_smiles: list, checkpoint_path: str, batch_size: int = 32, gpu: int | None = 0,
) -> np.ndarray:
    tmp_dir = Path(tempfile.mkdtemp(prefix="grover3400_"))
    try:
        smiles_csv = tmp_dir / "smiles.csv"
        with open(smiles_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smiles"])
            for s in chunk_smiles:
                w.writerow([s])

        features_npz = tmp_dir / "features.npz"
        subprocess.run(
            [
                sys.executable, str(GROVER_ROOT / "scripts" / "save_features.py"),
                "--data_path", str(smiles_csv),
                "--features_generator", "rdkit_2d_normalized",
                "--save_path", str(features_npz),
                "--sequential",
            ],
            cwd=str(GROVER_ROOT), check=True,
        )

        fp_npz = tmp_dir / "fp.npz"
        gpu_args = ["--gpu", str(gpu)] if gpu is not None else ["--no_cuda"]
        subprocess.run(
            [
                sys.executable, str(GROVER_ROOT / "main.py"), "fingerprint",
                "--data_path", str(smiles_csv),
                "--checkpoint_path", checkpoint_path,
                "--features_path", str(features_npz),
                "--fingerprint_source", "both",
                "--output_path", str(fp_npz),
                "--batch_size", str(batch_size),
                *gpu_args,
            ],
            cwd=str(GROVER_ROOT), check=True,
        )

        fps = np.load(fp_npz, allow_pickle=True)["fps"]
        return np.asarray(fps, dtype=np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent grover3400_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT, help="GROVER checkpoint (args/state_dict/data_scaler .pt) -- default is Yang's model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, or omit --gpu and pass --no-cuda for CPU")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "grover3400", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)
    gpu = None if args.no_cuda else args.gpu

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  gpu={gpu}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(chunk_smiles, args.checkpoint_path, batch_size=args.batch_size, gpu=gpu)
    elapsed = time.perf_counter() - t0

    if matrix.shape[0] != len(chunk_smiles):
        raise RuntimeError(
            f"grover fingerprint returned {matrix.shape[0]:,} rows for {len(chunk_smiles):,} input SMILES "
            f"-- refusing to write a misaligned chunk (a malformed SMILES may have been dropped instead of NaN-filled)."
        )

    write_chunk_file(args.chunks_dir, "grover3400", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
