"""Structural correctness checks for the full model (requires the
checkpoint on disk; skips, not fails, if absent). These don't need a
reference implementation to be meaningful -- they test real mathematical
invariants of the architecture that a wrong indexing/padding/masking bug
would very likely break.

Run directly: `python3 -m unimol2.tests.test_model_invariants`
"""
from __future__ import annotations

import numpy as np
import torch

from unimol2.data.collate import build_molecule_features, collate_batch, prepare_batch
from unimol2.data.conformer import generate_conformers

from ._skip import require_checkpoint

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        require_checkpoint()
        from unimol2 import build_model_from_checkpoint
        _MODEL = build_model_from_checkpoint()
    return _MODEL


def test_no_nan_or_inf() -> None:
    model = _get_model()
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    batch = prepare_batch(smiles, num_workers=len(smiles), timeout_s=30)
    with torch.no_grad():
        emb = model(batch)
    assert not torch.isnan(emb).any(), "NaN in output embedding"
    assert not torch.isinf(emb).any(), "Inf in output embedding"
    assert emb.shape == (len(smiles), 1536), emb.shape


def test_determinism() -> None:
    model = _get_model()
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    atoms_h, coords_h = generate_conformers([smi], n_conformer=1, num_workers=1, timeout_s=30)[0]
    feat = build_molecule_features(smi, atoms_h, coords_h, seed=0)
    batch = collate_batch([feat])
    with torch.no_grad():
        emb1 = model(batch)
        emb2 = model(batch)
    assert torch.equal(emb1, emb2), (emb1 - emb2).abs().max()


def test_permutation_invariance() -> None:
    """Shuffling atom order (and correspondingly permuting every per-atom
    and per-pair feature) must leave the pooled (virtual-node) embedding
    unchanged -- a real structural property of this architecture (attention
    + the pair-track outer-product/triangle-multiplication updates are
    permutation-equivariant; pooling reads a fixed virtual-node slot that
    doesn't depend on real-atom ordering at all). This is the strongest
    correctness signal available without an external reference
    implementation, and is what actually caught this port's early
    plumbing bugs during development."""
    model = _get_model()
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin, 13 heavy atoms
    atoms_h, coords_h = generate_conformers([smi], n_conformer=1, num_workers=1, timeout_s=30)[0]
    feat = build_molecule_features(smi, atoms_h, coords_h, seed=0)

    N = feat["atom_mask"].shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)

    feat_perm = dict(feat)
    feat_perm["src_token"] = feat["src_token"][perm]
    feat_perm["src_pos"] = feat["src_pos"][perm]
    feat_perm["atom_feat"] = feat["atom_feat"][perm]
    feat_perm["atom_mask"] = feat["atom_mask"][perm]
    feat_perm["degree"] = feat["degree"][perm]
    feat_perm["edge_feat"] = feat["edge_feat"][perm][:, perm]
    feat_perm["shortest_path"] = feat["shortest_path"][perm][:, perm]
    feat_perm["pair_type"] = feat["pair_type"][perm][:, perm]
    attn_bias_perm = feat["attn_bias"].clone()
    attn_bias_perm[1:, 1:] = feat["attn_bias"][1:, 1:][perm][:, perm]
    feat_perm["attn_bias"] = attn_bias_perm

    batch = collate_batch([feat])
    batch_perm = collate_batch([feat_perm])
    with torch.no_grad():
        emb = model(batch)
        emb_perm = model(batch_perm)

    assert torch.allclose(emb, emb_perm, atol=1e-4), (emb - emb_perm).abs().max()


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
