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
from functools import partial
from multiprocessing import get_context
from multiprocessing import TimeoutError as MPTimeoutError
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


def generate_conformers(
    smiles_list: List[str], n_conformer: int = 1, num_workers: int = 4, timeout_s: int = 30,
) -> List[Tuple[List[str], np.ndarray]]:
    """Bulk, in-memory, timeout-protected conformer generation for a whole
    SMILES list -- for single-process/whole-pool callers (e.g.
    embed/extract_embeddings.py) that don't need Stage 1's separate
    LMDB-persisted-chunk design (see generate_unimol_conformers_chunk.py,
    which duplicates this same worker-pool/timeout pattern for its own
    LMDB-writing use case rather than calling this). Returns one
    (atoms_with_h, single_coords) pair per input SMILES, in the same order
    -- `single_coords` is already reduced to the first real conformer
    (index 0), matching production's own "keep only conformer index 0"
    convention (see compute_unimol_embeddings_chunk.py's docstring)."""
    s2c = partial(smiles_to_coords, n_conformer=n_conformer)
    results: List[Tuple[List[str], np.ndarray]] = [None] * len(smiles_list)

    with get_context("fork").Pool(num_workers) as pool:
        pending = [(i, smi, pool.apply_async(s2c, (smi,))) for i, smi in enumerate(smiles_list)]
        for i, smi, async_result in pending:
            try:
                atoms, coordinates = async_result.get(timeout=timeout_s)
            except MPTimeoutError:
                print(f"[timeout] {smi!r} -- falling back to 2D coordinates")
                mol = Chem.MolFromSmiles(smi)
                coordinates = [smiles_to_2d_coords(smi)] * (n_conformer + 1)
                mol = AllChem.AddHs(mol)
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            results[i] = (list(atoms), np.asarray(coordinates[0], dtype=np.float32))

    return results
