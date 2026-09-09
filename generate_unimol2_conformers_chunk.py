#!/usr/bin/env python3
"""Stage 1 of the split Uni-Mol2 pipeline: conformer generation only.

Mirrors generate_unimol_conformers_chunk.py's design (Uni-Mol v1's own
Stage 1) -- separates CPU-bound RDKit conformer generation from the GPU-
bound 1.1B-param forward pass so each can run on the hardware it actually
needs: conformer generation is embarrassingly parallel, cheap CPU work
(this repo's own precedent: al-eval-framework's real 1000-shard/~1.3B-
molecule/~7h run, see that script's docstring), while Uni-Mol2's forward
pass needs a GPU (see compute_unimol2_embeddings_chunk.py's docstring on
why CPU-only inference isn't practical for a 1.1B-param model at this
scale). Running them in one process per chunk (the original single-stage
design) means every GPU task also pays the CPU conformer-generation cost
serially inline, competing for that chunk's own CPUs with the very workers
generating the batches the GPU is waiting on.

Unlike Uni-Mol v1's Stage 1 (which stores multiple raw conformers +
an always-appended 2D fallback per molecule, and Stage 2 has to know to
keep only index 0), Uni-Mol2's own generate_conformers() (unimol2/data/
conformer.py) already reduces to exactly one (atoms_with_h, coords) pair
per molecule internally -- so each LMDB record here is already exactly
what Stage 2 needs, no fallback-duplicate bookkeeping required.

Output: one LMDB chunk file per chunk-id, same directory/key/naming
convention as Uni-Mol v1's Stage 1 (_chunks/chunk_NNNNN.lmdb, global
zero-padded index keys, byte-lexicographic order == positional order).

Usage
-----
python generate_unimol2_conformers_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --out-dir /path/to/_unimol2_conformers \\
    --chunk-id 0 --num-chunks 1000 --num-workers 16
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np

from smiles_chunking import _chunk_bounds, count_lines, read_smiles_chunk

_KEY_WIDTH = 13  # zero-padded decimal width; comfortably covers > 1.3B indices


def _global_index_key(idx: int) -> bytes:
    return f"{idx:0{_KEY_WIDTH}d}".encode()


def generate_conformers_for_chunk(
    smiles: list, chunk_path: Path, start: int, map_size_gb: float,
    num_workers: int = 4, timeout_s: int = 30, commit_every: int = 2000,
) -> int:
    """Same incremental-commit, resumable design as Uni-Mol v1's Stage 1
    (generate_unimol_conformers_chunk.py::generate_conformers_for_chunk) --
    but calling unimol2's own generate_conformers() (batched internally
    via its own multiprocessing Pool) rather than a single-molecule
    function, since one Pool per whole batch is cheaper than one Pool per
    molecule. commit_every splits `smiles` into sub-batches so a kill
    mid-chunk only loses one sub-batch's worth of work, not the whole
    chunk."""
    from unimol2.data.conformer import generate_conformers

    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(chunk_path), subdir=False, map_size=int(map_size_gb * (1024 ** 3)))

    with env.begin() as txn:
        done_keys = set(txn.cursor().iternext(values=False))

    todo = [(i, smi) for i, smi in enumerate(smiles) if _global_index_key(start + i) not in done_keys]
    n_done_already = len(smiles) - len(todo)
    if n_done_already:
        print(f"[resume] {n_done_already:,}/{len(smiles):,} molecules already present in {chunk_path} -- skipping")

    if not todo:
        env.close()
        return 0

    n_written = 0
    for sub_start in range(0, len(todo), commit_every):
        sub = todo[sub_start:sub_start + commit_every]
        sub_smiles = [smi for _, smi in sub]
        conformers = generate_conformers(sub_smiles, n_conformer=1, num_workers=num_workers, timeout_s=timeout_s)

        with env.begin(write=True) as txn:
            for (i, _), (atoms, coords) in zip(sub, conformers):
                key = _global_index_key(start + i)
                value = pickle.dumps({"atoms": atoms, "coordinates": np.asarray(coords, dtype=np.float32)})
                txn.put(key, value)

        n_written += len(sub)
        print(f"[progress] {n_written:,}/{len(todo):,} molecules generated this run", flush=True)

    env.close()
    return len(todo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--out-dir", required=True, help="Directory to hold _chunks/chunk_NNNNN.lmdb")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--commit-every", type=int, default=2000)
    parser.add_argument("--map-size-gb", type=float, default=30.0, help="LMDB map_size for a single chunk file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    chunks_dir = out_dir / "_chunks"

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)
    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))")

    chunk_path = chunks_dir / f"chunk_{args.chunk_id:05d}.lmdb"

    t0 = time.perf_counter()
    n_written = generate_conformers_for_chunk(
        chunk_smiles, chunk_path, start, args.map_size_gb,
        num_workers=args.num_workers, timeout_s=args.timeout_s, commit_every=args.commit_every,
    )
    elapsed = time.perf_counter() - t0

    print(f"[done] chunk {args.chunk_id}: {n_written:,} molecules written in {elapsed:.1f}s -> {chunk_path}")


if __name__ == "__main__":
    main()
