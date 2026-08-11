#!/usr/bin/env python3
"""Stage 1 of the split Uni-Mol pipeline: conformer generation only.

Ported from al-eval-framework's src/representations/generate_conformers.py
(that repo's own real 1000-shard / ~1.3B-molecule / ~7h run validated this
exact design) -- simplified for molpal-fusion-hts's use case: dropped
DOCKSTRING/--target support (not needed here) and the file-metadata cache
-identity hashing (not needed either -- molpal-fusion-hts's DATASETS
registry in run_experiment.py already gives each dataset one explicit,
unambiguous embed_dir, so there's no risk of two different pools
colliding on the same auto-derived cache path the way a generic --target
-keyed tool has to guard against).

Conformer generation (RDKit ETKDG, CPU-bound) and embedding computation
(Uni-Mol transformer forward pass) are kept as separate stages because they
have very different resource shapes: conformer generation is embarrassingly
parallel per-molecule CPU work (no GPU benefit at all), while embedding
computation is a GPU-accelerable (or CPU-fallback) forward pass. Running
them as one continuous process (molpal-fusion-hts's older extract_embeddings.py
did this) doesn't scale to AmpC's 99.5M molecules -- conformer generation
alone would dominate a single process for weeks.

This script runs *only* conformer generation, over one chunk of the pool,
writing results to an LMDB shard. No merge step is needed afterward:
compute_unimol_embeddings_chunk.py reads directly from each chunk's shard
(same chunk boundaries, same chunk-id) -- see that script's docstring.

Storage note (measured on real Enamine REAL molecules in al-eval-framework):
a conformer record is ~9.15KB pickled at n_conformer=10, ~4.01KB at
n_conformer=1. Use n_conformer=1 for frozen feature extraction specifically:
the embedding-computation stage always discards all but one *randomly
chosen* conformer before its single forward pass (muben's process_training()
-> conformer_sampling()), so generating 10 and discarding 9 is equivalent
in expectation to generating 1 directly -- and ~9x faster, measured. At
AmpC's 99,459,561 molecules that's ~399GB at n_conformer=1 (~910GB at the
muben default of 10) -- budget storage accordingly.

Usage
-----
python generate_unimol_conformers_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --out-dir /path/to/_unimol_conformers \\
    --chunk-id 0 --num-chunks 70 --n-conformer 1

# After all chunks finish (only needed if you want one consolidated file
# instead of reading per-chunk shards directly -- compute_unimol_embeddings_chunk.py
# does NOT require this):
python generate_unimol_conformers_chunk.py \\
    --out-dir /path/to/_unimol_conformers --num-chunks 70 --merge
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import time
from functools import partial
from multiprocessing import get_context
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import lmdb

import sys
ROOT = Path(__file__).resolve().parent
_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))

_KEY_WIDTH = 13  # zero-padded decimal width; comfortably covers > 1.3B indices


def _global_index_key(idx: int) -> bytes:
    """LMDB iterates keys in byte-lexicographic order, and DatasetUniMol's
    non-random_split load path (create_features()) trusts that order to
    match the SMILES list positionally. Zero-padding guarantees correct
    numeric ordering regardless of index magnitude."""
    return f"{idx:0{_KEY_WIDTH}d}".encode()


def _chunk_bounds(n: int, chunk_id: int, num_chunks: int) -> tuple[int, int]:
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, n)
    return start, end


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def read_smiles_chunk(path: str, start: int, end: int) -> list:
    """Reads only lines [start, end) of a SMILES file. Uses islice over a
    lazily-iterated file handle rather than loading the whole file into a
    list first -- memory stays bounded to one chunk even for a billion-line
    pool (the O(start) time to skip preceding lines is accepted here)."""
    with open(path) as f:
        lines = list(itertools.islice(f, start, end))
    return [line.strip() for line in lines if line.strip()]


def generate_conformers_for_chunk(
    smiles: list, shard_path: Path, start: int, map_size_gb: float,
    n_conformer: int = 1, num_workers: int = 4, timeout_s: int = 30,
    commit_every: int = 2000,
) -> int:
    """Same timeout-protected generation as
    muben/muben/dataset/dataset_unimol/dataset.py's create_features() --
    apply_async with a per-molecule timeout, falling back to 2D
    coordinates on a hang, since plain pool.imap has no per-item timeout
    and one bad molecule can block an entire chunk forever.

    Writes results to the LMDB shard incrementally (every `commit_every`
    molecules) instead of accumulating a chunk's full results in memory
    and writing once at the end -- bounds how much work a kill (SLURM
    walltime, preemption, crash) can lose. Also resumable: molecules whose
    global index already has an entry in `shard_path` (e.g. from a prior
    killed attempt) are skipped rather than regenerated.
    """
    from muben.utils.chem import smiles_to_coords, smiles_to_2d_coords
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from tqdm.auto import tqdm

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(shard_path), subdir=False, map_size=int(map_size_gb * (1024 ** 3)))

    with env.begin() as txn:
        done_keys = set(txn.cursor().iternext(values=False))

    todo = [(i, smi) for i, smi in enumerate(smiles) if _global_index_key(start + i) not in done_keys]
    n_done_already = len(smiles) - len(todo)
    if n_done_already:
        print(f"[resume] {n_done_already:,}/{len(smiles):,} molecules already present in {shard_path} -- skipping")

    if not todo:
        env.close()
        return 0

    s2c = partial(smiles_to_coords, n_conformer=n_conformer)
    buffer = []

    def flush():
        if not buffer:
            return
        with env.begin(write=True) as txn:
            for key, value in buffer:
                txn.put(key, value)
        buffer.clear()

    with get_context("fork").Pool(num_workers) as pool:
        pending = [(i, smi, pool.apply_async(s2c, (smi,))) for i, smi in todo]
        for i, smi, async_result in tqdm(pending, total=len(pending)):
            try:
                atoms, coordinates = async_result.get(timeout=timeout_s)
            except MPTimeoutError:
                print(f"[timeout] {smi!r} -- falling back to 2D coordinates")
                mol = Chem.MolFromSmiles(smi)
                coordinates = [smiles_to_2d_coords(smi)] * (n_conformer + 1)
                mol = AllChem.AddHs(mol)
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            buffer.append((_global_index_key(start + i), pickle.dumps({"atoms": atoms, "coordinates": coordinates})))
            if len(buffer) >= commit_every:
                flush()

        flush()

    env.close()
    return len(todo)


def merge_shards(shards_dir: Path, out_path: Path, expected_num_chunks: int, map_size_gb: float, allow_partial: bool) -> None:
    """Commits one write transaction per shard (not one giant transaction
    for the whole merge) and tracks completed shards in a sidecar
    `.merge_progress` file, so a merge across many shards / several TB can
    be killed and resumed without losing everything and restarting."""
    shard_paths = sorted(shards_dir.glob("chunk_*.lmdb"))
    if not allow_partial and len(shard_paths) != expected_num_chunks:
        raise SystemExit(
            f"Expected {expected_num_chunks} shards in {shards_dir}, found {len(shard_paths)}. "
            f"Pass --allow-partial to merge an incomplete set anyway."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(out_path.suffix + ".merge_progress")
    done_shards = set()
    if progress_path.exists():
        done_shards = set(progress_path.read_text().splitlines())
        print(f"[resume] {len(done_shards)}/{len(shard_paths)} shards already merged, skipping")

    out_env = lmdb.open(str(out_path), subdir=False, map_size=int(map_size_gb * (1024 ** 3)))

    n_written = 0
    n_merged_now = 0
    with open(progress_path, "a") as progress_f:
        for shard_path in shard_paths:
            if shard_path.name in done_shards:
                continue

            shard_env = lmdb.open(
                str(shard_path), subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256,
            )
            with out_env.begin(write=True) as out_txn:
                with shard_env.begin() as txn:
                    cursor = txn.cursor()
                    for key, value in cursor.iternext(keys=True, values=True):
                        out_txn.put(key, value)
                        n_written += 1
            shard_env.close()

            progress_f.write(shard_path.name + "\n")
            progress_f.flush()
            n_merged_now += 1

    out_env.close()

    print(f"[merge] {n_merged_now} shards merged this run ({len(shard_paths)} total) -> {n_written:,} new records -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True, help="Plain text file, one SMILES per line")
    parser.add_argument("--total-count", type=int, default=None, help="Total pool size, to skip an O(N) line-count scan (strongly recommended for large array jobs)")
    parser.add_argument("--out-dir", required=True, help="Directory to hold _shards/chunk_NNNNN.lmdb (and, after --merge, train.lmdb)")
    parser.add_argument("--chunk-id", type=int, default=None, help="Required unless --merge")
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--n-conformer", type=int, default=1, help="1 is recommended here (see module docstring) -- the muben default of 10 is for a different use case")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--commit-every", type=int, default=2000)
    parser.add_argument("--partition", default="train", help="Must match the partition compute_unimol_embeddings_chunk.py's model expects (always 'train' today)")
    parser.add_argument("--map-size-gb", type=float, default=30.0, help="LMDB map_size for a single shard. At ~4.01KB/molecule (n_conformer=1, measured), a ~1.4M-molecule shard (70-way split of AmpC's 99.5M pool) needs ~5.6GB; 30GB leaves >5x headroom for larger-than-average molecules.")
    parser.add_argument("--merge", action="store_true", help="Consolidate all chunk shards into one final cache instead of generating one (optional -- Stage 2 can read per-chunk shards directly)")
    parser.add_argument("--merge-map-size-gb", type=float, default=500.0)
    parser.add_argument("--allow-partial", action="store_true", help="Merge even if fewer than --num-chunks shards are present")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "_shards"
    final_path = out_dir / f"{args.partition}.lmdb"

    if args.merge:
        merge_shards(shards_dir, final_path, args.num_chunks, args.merge_map_size_gb, args.allow_partial)
        return

    if args.chunk_id is None:
        raise SystemExit("--chunk-id is required unless --merge is set")

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)
    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))")

    shard_path = shards_dir / f"chunk_{args.chunk_id:05d}.lmdb"

    t0 = time.perf_counter()
    n_written = generate_conformers_for_chunk(
        chunk_smiles, shard_path, start, args.map_size_gb,
        n_conformer=args.n_conformer, num_workers=args.num_workers, timeout_s=args.timeout_s,
        commit_every=args.commit_every,
    )
    elapsed = time.perf_counter() - t0

    print(f"[done] chunk {args.chunk_id}: {n_written:,} molecules written in {elapsed:.1f}s -> {shard_path}")


if __name__ == "__main__":
    main()
