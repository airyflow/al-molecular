"""Fast tests for the feature-computation pipeline (unimol2/data/*.py) --
no checkpoint or GPU required. Run directly: `python3 -m unimol2.tests.test_feature_pipeline`
(or any pytest-compatible runner, if one is ever added to this repo).

test_ethanol_atom_features hand-verifies every value against an independent,
from-scratch manual computation of RDKit's OGB-style atom features (not a
re-derivation of the code under test) -- documented inline so the expected
values can be checked by a human, not just trusted. This is the single
most important test in this file: it's what caught (well, would have
caught) the "pair_type derived from atom_feat[...,0], not atomic number"
detail described in unimol2/data/graph_features.py's docstring.
"""
from __future__ import annotations

import numpy as np

from unimol2.data.collate import build_molecule_features
from unimol2.data.conformer import generate_conformers
from unimol2.data.graph_features import _floyd_warshall


def test_floyd_warshall_matches_reference() -> None:
    """The vectorized _floyd_warshall (graph_features.py) replaces the
    original numba triple-loop implementation upstream uses -- verify they
    agree exactly (they should: row/column k are provably invariant during
    round k of Floyd-Warshall, since M[k,k]=0 makes that round's own update
    to row/column k a no-op, so the vectorized "compute all of round k from
    the pre-round-k snapshot" is mathematically identical to the
    sequential in-place triple loop, not just an approximation)."""

    def reference(adj: np.ndarray) -> np.ndarray:
        M = adj.copy()
        n = M.shape[0]
        for i in range(n):
            for j in range(n):
                if M[i, j] == 0:
                    M[i, j] = 509
        for i in range(n):
            M[i, i] = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    cost = M[i, k] + M[k, j]
                    if M[i, j] > cost:
                        M[i, j] = cost
        for i in range(n):
            for j in range(n):
                if M[i, j] >= 509:
                    M[i, j] = 509
        return M

    rng = np.random.default_rng(0)
    for trial in range(200):
        n = rng.integers(2, 25)
        p = rng.uniform(0.05, 0.6)
        adj = (rng.random((n, n)) < p).astype(np.int64)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)  # symmetric, like real bond adjacency
        assert np.array_equal(reference(adj), _floyd_warshall(adj)), f"mismatch at trial {trial}, n={n}"


def test_ethanol_atom_features() -> None:
    """Ethanol (CCO), RDKit atom order [C(methyl), C(hydroxyl-bearing), O].
    Every value below was computed by hand from RDKit's own atom properties
    (GetTotalDegree/GetFormalCharge/GetTotalNumHs/GetHybridization/etc.),
    independently of unimol2/data/graph_features.py -- see the module
    docstring there for the exact offset recipe being checked here.

    atom0 (CH3): TotalDegree=4 (3H+1C), FormalCharge=0, NumHs=3, SP3, not
                 aromatic, not in ring, no radical, no chirality.
      raw 8-dim (chirality,degree,charge,numH,radical,hybrid,aromatic,ring)
        = [0, 4, 5, 3, 0, 2, 0, 0]  (indices into their allowable-value lists)
      offset (sizes=[16]*8, offset starts at 1): [1, 21, 38, 52, 65, 83, 97, 113]
      +2 (drop_feat=False branch): [3, 23, 40, 54, 67, 85, 99, 115]

    atom1 (CH2, bonded to atom0-C, atom2-O, 2H): TotalDegree=4, NumHs=2, else same as atom0.
      raw = [0, 4, 5, 2, 0, 2, 0, 0] -> +offset -> [1,21,38,51,65,83,97,113] -> +2 -> [3,23,40,53,67,85,99,115]

    atom2 (OH, bonded to atom1-C, 1H): TotalDegree=2, NumHs=1, else same pattern.
      raw = [0, 2, 5, 1, 0, 2, 0, 0] -> +offset -> [1,19,38,50,65,83,97,113] -> +2 -> [3,21,40,52,67,85,99,115]

    degree field (heavy-atom-graph-only, NOT GetTotalDegree): atom0 has 1
    heavy-atom bond (to atom1), atom1 has 2 (atom0, atom2), atom2 has 1
    (atom1). Raw=[1,2,1], +2 offset -> [3,4,3].

    shortest_path (heavy-atom graph distance, +1 offset): d(0,1)=1, d(0,2)=2,
    d(1,2)=1 -> +1 -> [[1,2,3],[2,1,2],[3,2,1]].

    pair_type: atoms = atom_feat[...,0] = [3,3,3] (all identical -- none of
    the 3 atoms have chirality set). Every (i,j) pair -> [3,3] ->
    convert_to_single_emb([128,128]) -> dim0: 3+1=4, dim1: 3+129=132 ->
    every entry is [4, 132].
    """
    smi = "CCO"
    atoms_h, coords_h = generate_conformers([smi], n_conformer=1, num_workers=1, timeout_s=30)[0]
    feat = build_molecule_features(smi, atoms_h, coords_h, seed=0)

    assert feat["src_token"].tolist() == [6, 6, 8], feat["src_token"]

    expected_atom_feat = [
        [3, 23, 40, 54, 67, 85, 99, 115],
        [3, 23, 40, 53, 67, 85, 99, 115],
        [3, 21, 40, 52, 67, 85, 99, 115],
    ]
    assert feat["atom_feat"].tolist() == expected_atom_feat, feat["atom_feat"]

    assert feat["degree"].tolist() == [3, 4, 3], feat["degree"]

    expected_shortest_path = [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
    assert feat["shortest_path"].tolist() == expected_shortest_path, feat["shortest_path"]

    expected_pair_type = [[[4, 132]] * 3 for _ in range(3)]
    assert feat["pair_type"].tolist() == expected_pair_type, feat["pair_type"]

    assert feat["attn_bias"].shape == (4, 4)  # N+1 for the virtual node
    assert (feat["attn_bias"] == 0).all()


def test_coordinates_are_centered() -> None:
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    atoms_h, coords_h = generate_conformers([smi], n_conformer=1, num_workers=1, timeout_s=30)[0]
    feat = build_molecule_features(smi, atoms_h, coords_h, seed=0)
    centroid = feat["src_pos"].mean(dim=0)
    assert centroid.abs().max().item() < 1e-4, centroid


def _run_all() -> None:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"running {t.__name__} ...", end=" ")
        t()
        print("PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
