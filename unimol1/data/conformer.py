"""3D conformer generation for Uni-Mol (v1) embedding extraction.

`smiles_to_2d_coords`/`smiles_to_3d_coords`/`smiles_to_coords` below are
ported verbatim from muben's `muben/muben/utils/chem.py` (RDKit ETKDG
embedding + MMFF optimization, falling back to 2D coordinates on failure --
already exercised successfully across all 99,459,561 AmpC molecules).
Pulled in directly rather than imported from muben so this package has zero
muben dependency -- identical reasoning, and identical code, to
`unimol2/data/conformer.py`'s own copy of this same utility (conformer
generation is generic RDKit logic with no Uni-Mol-version-specific content,
so both ports carry their own copy rather than depending on each other or
on muben).
"""
from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")

logger = logging.getLogger(__name__)


def smiles_to_2d_coords(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {smiles}"
    return coordinates


def smiles_to_3d_coords(smiles: str, n_conformer: int) -> List[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(n_conformer):
        coordinates = list()
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smiles)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)
        except Exception as e:
            logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
            coordinates = smiles_to_2d_coords(smiles)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {smiles}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smiles_to_coords(smiles: str, n_conformer: int = 10) -> Tuple[List[str], List[np.ndarray]]:
    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > 400:
        coordinates = [smiles_to_2d_coords(smiles)] * (n_conformer + 1)
        logger.warning(f"atom num > 400, use 2D coords {smiles}")
    else:
        coordinates = smiles_to_3d_coords(smiles, n_conformer)
        coordinates.append(smiles_to_2d_coords(smiles).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atoms, coordinates
