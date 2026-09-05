"""Structural correctness checks for the full model (requires the
checkpoint on disk; skips, not fails, if absent).

Run directly: `python3 -m unimol1.tests.test_model_invariants`
"""
from __future__ import annotations

import numpy as np
import torch

from unimol1.data.collate import CollatorUniMol, build_instance
from unimol1.data.conformer import smiles_to_coords
from unimol1.data.process import ProcessingPipeline
from unimol1.dictionary import load_production_dictionary

from ._skip import require_checkpoint

_MODEL = None
_PIPELINE = None


def _get_model():
    global _MODEL, _PIPELINE
    if _MODEL is None:
        require_checkpoint()
        from unimol1 import build_model_from_checkpoint

        dictionary = load_production_dictionary()
        _MODEL = build_model_from_checkpoint(dictionary=dictionary)
        _PIPELINE = ProcessingPipeline(dictionary=dictionary, max_atoms=64, max_seq_len=80, remove_hydrogen_flag=True, remove_polar_hydrogen_flag=False)
    return _MODEL, _PIPELINE


def _batch_for(smiles: str) -> torch.Tensor:
    model, pipeline = _get_model()
    atoms, coordinates = smiles_to_coords(smiles, n_conformer=1)
    # smiles_to_coords returns n_conformer 3D conformers + one appended 2D
    # fallback -- keep only the first real conformer, matching production's
    # own "index 0 is the primary one" convention (see
    # compute_unimol_embeddings_chunk.py's docstring on this exact point).
    feat = build_instance(atoms, coordinates[0], pipeline)
    collator = CollatorUniMol(atom_pad_idx=load_production_dictionary().pad())
    return collator([feat])


def test_no_nan_or_inf() -> None:
    model, _ = _get_model()
    for smi in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]:
        batch = _batch_for(smi)
        emb = model(batch)
        assert not torch.isnan(emb).any(), f"NaN in output embedding for {smi}"
        assert not torch.isinf(emb).any(), f"Inf in output embedding for {smi}"
        assert emb.shape == (1, 512), emb.shape


def test_determinism() -> None:
    model, _ = _get_model()
    batch = _batch_for("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    emb1 = model(batch)
    emb2 = model(batch)
    assert torch.equal(emb1, emb2), (emb1 - emb2).abs().max()


def _run_all() -> None:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    import unittest
    n_skipped = 0
    for t in tests:
        print(f"running {t.__name__} ...", end=" ")
        try:
            t()
        except unittest.SkipTest as e:
            print(f"SKIP ({e})")
            n_skipped += 1
            continue
        print("PASS")
    print(f"\n{len(tests) - n_skipped}/{len(tests)} tests passed ({n_skipped} skipped).")


if __name__ == "__main__":
    _run_all()
