"""Embedding output for sharded extraction pipelines: per-chunk files during
the parallel compute phase, one shared memmap after a sequential stitch.

Replaces the original "each shard writes its own <backbone>_embeddings_chunk_
NNNNN.npz, then a separate merge script np.vstack()s all of them" pattern --
at AmpC scale (99.5M molecules) that merge would need ~200-640GB of RAM
depending on backbone (measured: 636.5GB for GROVER's 1600-d embeddings
alone), infeasible on any node we'd realistically get.

Two-phase design:
  1. Parallel compute phase (many SLURM array tasks): each shard task calls
     write_chunk_file() to write its own INDEPENDENT .npy file. No shared
     state, no coordination, no race condition possible -- every task owns
     a distinct path.
  2. Stitch phase (one sequential job, after all chunks finish -- see
     stitch_embedding_chunks.py): preallocate() creates the final (N, D)
     shared .npy file (header-only, near-zero-RAM), then write_slice() is
     called once per chunk, in order, copying each chunk file's rows into
     the corresponding disjoint slice. Single process, so no race is
     possible here either -- and it's memory-bounded to one chunk at a
     time (e.g. ~6GB for MoLFormer's largest AmpC chunk), never the whole
     array.

The final stitched file is in the exact mmap-ready .npy format
EmbeddingFeaturizer.load() (molpal/featurizer.py) expects, and no further
merge is ever needed after the stitch completes.

Usage
-----
    from shared_embedding_store import write_chunk_file, preallocate, write_slice

    # inside each parallel shard task, after computing this shard's embeddings:
    write_chunk_file(chunks_dir, "grover", chunk_id, embeddings)

    # once, in the sequential stitch job, after all chunk files exist:
    preallocate("/path/to/grover_embeddings.npy", n_molecules=99_459_561, dim=1600)
    write_slice("/path/to/grover_embeddings.npy", start, end, embeddings)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def chunk_file_path(chunks_dir: str | Path, backbone: str, chunk_id: int) -> Path:
    return Path(chunks_dir) / f"{backbone}_embeddings_chunk_{chunk_id:05d}.npy"


def write_chunk_file(chunks_dir: str | Path, backbone: str, chunk_id: int, embeddings: np.ndarray) -> Path:
    """Write one shard's embeddings to its own independent .npy file.

    Written atomically (to a temp path, then renamed into place) so a job
    killed mid-write never leaves a corrupt or truncated file that a
    resumed run, or the stitch phase, might mistake for a completed chunk
    -- os.rename is atomic on POSIX filesystems including Lustre, so
    readers only ever see either the old state (file absent) or the
    fully-written new one, never a partial file.

    The temp path must itself end in ".npy" -- np.save() silently appends
    ".npy" to any path that doesn't already end with it (verified
    directly: np.save("x.npy.tmp", arr) actually writes "x.npy.tmp.npy",
    not "x.npy.tmp"), which would otherwise break the rename below with a
    FileNotFoundError against the path we thought we'd written.
    """
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    final_path = chunk_file_path(chunks_dir, backbone, chunk_id)
    tmp_path = final_path.parent / f"{final_path.stem}.tmp.npy"
    np.save(tmp_path, embeddings)
    tmp_path.rename(final_path)
    return final_path


def preallocate(path: str | Path, n_molecules: int, dim: int, dtype=np.float32) -> None:
    """Create a (n_molecules, dim) .npy file on disk, header-only cost.

    Idempotent guard: if a correctly-shaped file already exists, does
    nothing (safe to re-run this step without clobbering in-progress or
    completed shard writes). Raises if a file exists with the WRONG shape
    -- that almost always means a stale file from a previous, differently
    -sized run; refuses to silently overwrite it.
    """
    path = Path(path)
    if path.exists():
        existing = np.lib.format.open_memmap(path, mode="r")
        if existing.shape != (n_molecules, dim):
            raise FileExistsError(
                f"{path} already exists with shape {existing.shape}, expected "
                f"{(n_molecules, dim)}. Remove it explicitly first if this is "
                f"intentional (e.g. dimension changed) -- refusing to silently "
                f"overwrite what may be a completed or in-progress run."
            )
        print(f"[preallocate] {path} already exists with correct shape {existing.shape}, leaving as-is")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_molecules, dim))
    del mm  # flush the header; no row data has been written, so this is cheap
    print(f"[preallocate] created {path}  shape=({n_molecules:,}, {dim})  dtype={dtype}")


def write_slice(path: str | Path, start: int, end: int, embeddings: np.ndarray) -> None:
    """Write embeddings into rows [start, end) of the shared .npy file.

    embeddings.shape must be (end - start, dim). Opens in "r+" (the file
    must already exist -- see preallocate()) and writes only this shard's
    slice; never touches or loads any other shard's rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- call preallocate() once before any "
            f"shard task runs write_slice()."
        )
    n = end - start
    if embeddings.shape[0] != n:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]:,} rows but the requested "
            f"slice [{start:,}, {end:,}) is {n:,} rows wide -- refusing to "
            f"write a misaligned slice."
        )

    mm = np.lib.format.open_memmap(path, mode="r+")
    if embeddings.shape[1] != mm.shape[1]:
        raise ValueError(
            f"embeddings has dim {embeddings.shape[1]} but {path} was "
            f"preallocated with dim {mm.shape[1]}."
        )
    mm[start:end] = embeddings.astype(mm.dtype, copy=False)
    mm.flush()
    del mm
    print(f"[write_slice] {path.name}: wrote rows [{start:,}, {end:,})")


def slice_status(path: str | Path, start: int, end: int, atol: float = 0.0) -> bool:
    """Best-effort check for whether rows [start, end) look already written
    (non-all-zero), so a resumed array job can skip completed shards. Not a
    substitute for a real completion marker -- an all-zero embedding row is
    astronomically unlikely for a trained model's output but not provably
    impossible, so callers doing exact resume-safety should track completion
    with a separate marker file instead. Provided as a cheap convenience for
    interactive use.
    """
    path = Path(path)
    if not path.exists():
        return False
    mm = np.lib.format.open_memmap(path, mode="r")
    chunk = mm[start:end]
    return bool(np.any(chunk != 0))
