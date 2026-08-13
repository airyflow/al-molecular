"""Ported from unimol2/unimol2/data/index_atom_dataset.py::IndexAtomDataset
(deepmodeling/Uni-Mol). No vocabulary/dict.txt file exists anywhere under
unimol2/ (unlike the original Uni-Mol) -- tokens are literal RDKit atomic
numbers."""
from __future__ import annotations

from typing import List

import numpy as np
from rdkit.Chem import AllChem

_PERIODIC_TABLE = AllChem.GetPeriodicTable()


def atoms_to_tokens(atom_symbols: List[str]) -> np.ndarray:
    return np.array([_PERIODIC_TABLE.GetAtomicNumber(s) for s in atom_symbols], dtype=np.int64)
