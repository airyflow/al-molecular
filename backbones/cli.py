"""Shared CLI scaffold + extraction driver for compute_*_embeddings_chunk.py.

Collapses what was near-identical boilerplate across all 4 scripts: the
--smiles-file/--total-count/--chunk-id/--num-chunks/--chunks-dir arg shape,
and the chunk-bounds -> skip-if-exists -> read_smiles_chunk ->
compute_embeddings_for_chunk -> write_chunk_file -> timed-logging sequence
in every main().
"""
from __future__ import annotations

import argparse
import time
from typing import Callable, List, Optional

import numpy as np
import torch

from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

# NOTE: _chunk_bounds/count_lines/read_smiles_chunk currently live inside
# generate_unimol_conformers_chunk.py by historical accident (they're
# fully backbone-agnostic). Relocating them to a proper smiles_chunking.py
# is a separate, later cleanup step (see the approved plan) -- not bundled
# into this initial scaffold so each migration step stays isolated and
# independently verifiable.

from .base import Backbone


def make_base_parser(chunks_dir_help: str = "Directory to write this chunk's independent embeddings_chunk_NNNNN.npy into") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help=chunks_dir_help)
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the backbone's own default_batch_size if unset")
    parser.add_argument("--device", default=None, help="Defaults to the backbone's own default_device (cuda if available, else cpu)")
    return parser


def compute_embeddings_for_chunk(
    backbone: Backbone, chunk_smiles: List[str], batch_size: int,
) -> np.ndarray:
    """Shared batch loop: featurize -> collate -> forward_and_pool, for
    every backbone. Molecules that fail to featurize (featurize() returns
    None) are dropped, not crashed on -- callers relying on strict
    row-count alignment with chunk_smiles should check the returned
    array's row count against len(chunk_smiles) themselves (this is a
    real, backbone-specific concern -- e.g. UniMol's _verify_alignment
    check -- not papered over here)."""
    parts = []
    with torch.no_grad():
        for i in range(0, len(chunk_smiles), batch_size):
            smi_batch = chunk_smiles[i : i + batch_size]
            feats = [f for f in (backbone.featurize(s) for s in smi_batch) if f is not None]
            if not feats:
                continue
            batch = backbone.collate(feats).to(backbone.device)
            emb = backbone.forward_and_pool(batch)
            parts.append(emb.float().cpu().numpy())
    return np.vstack(parts)


def run_extraction(
    backbone_name: str,
    args: argparse.Namespace,
    load_backbone: Callable[[Optional[str]], Backbone],
) -> None:
    """The shared main()-body driver. `load_backbone(device)` is a thin
    per-script closure (usually just `lambda device: backbones.load(backbone_name, checkpoint_path=..., device=device)`)
    so backbone-specific extra CLI flags (e.g. UniMol's --shards-dir) can
    be threaded through without this function needing to know about them."""
    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, backbone_name, args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    backbone = load_backbone(args.device)
    backbone_module = getattr(backbone, "model", None)
    if backbone_module is not None:
        backbone_module.eval()
    batch_size = args.batch_size or backbone.default_batch_size

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules "
          f"(indices [{start}, {end}))  device={backbone.device}  batch_size={batch_size}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(backbone, chunk_smiles, batch_size)
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, backbone_name, args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")
