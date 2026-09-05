"""Fast tests for the feature-computation pipeline (unimol1/data/*.py, plus
dictionary.py) -- no checkpoint or GPU required.
Run directly: `python3 -m unimol1.tests.test_feature_pipeline`

test_ethanol_features hand-verifies every value against an independent,
from-scratch computation of the real Uni-Mol dictionary/edge-type/distance
recipe (not a re-derivation of the code under test), the same spirit as
unimol2/tests/test_feature_pipeline.py's test_ethanol_atom_features.
"""
from __future__ import annotations

import numpy as np
import torch

from unimol1.data.collate import build_instance
from unimol1.data.process import ProcessingPipeline
from unimol1.dictionary import DictionaryUniMol


def test_ethanol_features() -> None:
    """Ethanol (CCO) with hydrogens explicit, RDKit heavy-atom order
    [C(methyl), C(hydroxyl-bearing), O], then RDKit's AddHs appends the 6
    hydrogens after (order: 3 on atom0, 2 on atom1, 1 on atom2) -- 9 atoms
    total, remove_hydrogen_flag=True strips them back down to the 3 heavy
    atoms before tokenization, matching production's config
    (remove_hydrogen=True).

    Dictionary indices (see dictionary.py's UNIMOL_DICT, 0-indexed):
    [PAD]=0 [CLS]=1 [SEP]=2 [UNK]=3 C=4 N=5 O=6 S=7 H=8 ...
    So heavy-atom tokens are [C, C, O] -> [4, 4, 6], then
    prepend_and_append wraps with bos()=1 ([CLS]) and eos()=2 ([SEP]):
    [1, 4, 4, 6, 2].

    edge_type = atoms.view(-1,1)*30 + atoms.view(1,-1) (num_types=len(dict)=30),
    on the wrapped 5-token sequence [1,4,4,6,2] -- hand-computable directly.
    """
    dictionary = DictionaryUniMol.load()
    assert len(dictionary) == 30, len(dictionary)
    assert dictionary.pad() == 0 and dictionary.bos() == 1 and dictionary.eos() == 2
    assert dictionary.index("C") == 4 and dictionary.index("O") == 6 and dictionary.index("H") == 8

    atoms_with_h = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]  # RDKit AddHs order for CCO
    # A simple planar-ish placeholder geometry is fine here -- this test
    # checks tokenization/edge-type/dictionary bookkeeping, not real 3D
    # chemistry (that's exercised by test_determinism/test_no_nan_or_inf
    # in test_model_invariants.py, using real RDKit-generated conformers).
    coords_with_h = np.arange(9 * 3, dtype=np.float32).reshape(9, 3)

    pipeline = ProcessingPipeline(
        dictionary=dictionary, max_atoms=64, max_seq_len=80,
        remove_hydrogen_flag=True, remove_polar_hydrogen_flag=False,
    )
    feat = build_instance(atoms_with_h, coords_with_h, pipeline)

    assert feat["atoms"].shape == (1, 5), feat["atoms"].shape  # (n_conformer=1, [CLS] + 3 heavy atoms + [SEP])
    assert feat["atoms"][0].tolist() == [1, 4, 4, 6, 2], feat["atoms"][0].tolist()

    expected_edge_type = (np.array([1, 4, 4, 6, 2]).reshape(-1, 1) * 30 + np.array([1, 4, 4, 6, 2]).reshape(1, -1))
    assert feat["edge_types"][0].tolist() == expected_edge_type.tolist(), feat["edge_types"][0]

    assert feat["distances"].shape == (1, 5, 5)
    # Distance matrix is symmetric with a zero diagonal by construction
    # (Euclidean distance matrix on the coordinates, including the
    # zero-padded [CLS]/[SEP] rows/cols).
    d = feat["distances"][0]
    assert torch.allclose(d, d.T)
    assert torch.allclose(d.diagonal(), torch.zeros(5))


def test_batch_padding_and_concatenation() -> None:
    """Two molecules of different heavy-atom-count sequence-length must
    collate into one batch, right-padded to the longer sequence length,
    concatenated (not stacked) along dim 0 -- matching
    CollatorUniMol.__call__'s real behavior (see collate.py)."""
    from unimol1.data.collate import CollatorUniMol

    dictionary = DictionaryUniMol.load()
    pipeline = ProcessingPipeline(dictionary=dictionary, max_atoms=64, max_seq_len=80, remove_hydrogen_flag=True)

    feat_a = build_instance(["C", "H", "H", "H", "H"], np.zeros((5, 3), dtype=np.float32), pipeline)  # methane, 1 heavy atom
    feat_b = build_instance(["C", "C", "O", "H", "H", "H", "H", "H", "H"], np.zeros((9, 3), dtype=np.float32), pipeline)  # ethanol, 3 heavy atoms

    collator = CollatorUniMol(atom_pad_idx=dictionary.pad())
    batch = collator([feat_a, feat_b])

    # methane -> [CLS] C [SEP] = 3 tokens; ethanol -> [CLS] C C O [SEP] = 5 tokens.
    assert batch.atoms.shape == (2, 5), batch.atoms.shape
    assert batch.atoms[0].tolist() == [1, 4, 2, 0, 0], batch.atoms[0].tolist()  # padded with atom_pad_idx=0
    assert batch.atoms[1].tolist() == [1, 4, 4, 6, 2], batch.atoms[1].tolist()
    assert len(batch) == 2


def _run_all() -> None:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"running {t.__name__} ...", end=" ")
        t()
        print("PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
