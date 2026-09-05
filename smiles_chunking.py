#!/usr/bin/env python3
"""Backbone-agnostic chunk-math and SMILES-file-reading utilities, shared by
every compute_*_embeddings_chunk.py, stitch_embedding_chunks.py,
generate_unimol_conformers_chunk.py, run_experiment.py, and
predict_pool_shard_worker.py.

Split out of generate_unimol_conformers_chunk.py (where these lived by
historical accident -- that module is Stage 1 of the Uni-Mol-v1-specific
conformer pipeline and, unlike these three functions, genuinely needs muben
on sys.path). Importing generate_unimol_conformers_chunk.py just for these
had the side effect of putting muben on sys.path for every consumer, even
ones (e.g. compute_unimol2_embeddings_chunk.py) that never touch muben.
Importing this module instead has no such side effect.
"""
from __future__ import annotations

import itertools


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
