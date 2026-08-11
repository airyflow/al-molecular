"""A featurizer transforms input representations into uncompressed feature representations for use
with clustering and model training/prediction."""
from dataclasses import dataclass
from itertools import chain
import math
from typing import List, Optional

import numpy as np
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

try:
    import ray
    from p_tqdm import p_map
except ImportError:
    ray = None
    p_map = None

try:
    from map4 import map4
except ImportError:
    pass

from molpal.utils import batches


@dataclass
class Featurizer:
    fingerprint: str = "pair"
    radius: int = 2
    length: int = 2048

    def __post_init__(self):
        if self.fingerprint == "maccs":
            self.radius = 0
            self.length = 167

    def __len__(self):
        return self.length

    def __call__(self, smi: str) -> Optional[np.ndarray]:
        return featurize(smi, self.fingerprint, self.radius, self.length)


def featurize(smi, fingerprint, radius, length) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == "morgan":
        fp = rdmd.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, useChirality=True)
    elif fingerprint == "pair":
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1 + radius, nBits=length
        )
    elif fingerprint == "rdkit":
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=1 + radius, fpSize=length)
    elif fingerprint == "maccs":
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    elif fingerprint == "map4":
        fp = map4.MAP4Calculator(dimensions=length, radius=radius, is_folded=True).calculate(mol)
    else:
        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    X = np.empty(len(fp))
    ConvertToNumpyArray(fp, X)
    return X


def featurize_batch(smis, fingerprint, radius, length) -> List[np.ndarray]:
    if p_map is not None:
        return p_map(featurize, smis, fingerprint, radius, length)
    return [featurize(s, fingerprint, radius, length) for s in smis]


# @ray.remote
# def featurize_batch(smis, fingerprint, radius, length) -> List[np.ndarray]:
#     return [featurize(smi, fingerprint, radius, length) for smi in smis]


def feature_matrix(smis, featurizer, disable: bool = False) -> List[np.ndarray]:
    fingerprint = featurizer.fingerprint
    radius = featurizer.radius
    length = len(featurizer)
    feats = featurize_batch(smis, fingerprint, radius, length)
    return list(feats)
    # chunksize = int(math.sqrt(ray.cluster_resources()["CPU"]) * 1024)
    # refs = [
    #     featurize_batch.remote(smis, fingerprint, radius, length)
    #     for smis in batches(smis, chunksize)
    # ]
    # fps_chunks = [
    #     ray.get(r) for r in tqdm(refs, "Featurizing", leave=False, disable=disable, unit="smi")
    # ]

    # return list(chain(*fps_chunks))


# ── Embedding-based featurizer ──────────────────────────────────────────────

def _load_plain_smiles(path) -> List[str]:
    """Read an ordered SMILES list from a plain .txt (one SMILES/line) or a
    .csv[.gz] with a smiles column -- same dispatch-on-suffix convention as
    run_experiment.py's load_library_smiles(), duplicated narrowly here
    rather than imported to avoid a molpal/ -> top-level-script dependency.
    Used as the AmpC-scale fallback: AmpC's sharded extraction pipeline
    (al-eval-framework) writes pure (N, D) .npy embedding files with no
    per-backbone smiles side-channel, since row order is already exactly
    the canonical ampc_smiles.txt order by construction (every shard's
    boundaries come from the same _chunk_bounds() function) -- no need to
    duplicate a 99.5M-entry smiles array into every backbone's output file."""
    import pathlib
    path = pathlib.Path(path)
    if path.suffix == ".txt":
        with open(path) as f:
            return [line.rstrip("\n") for line in f]
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    smi_col = next(c for c in df.columns if "smiles" in c)
    return df[smi_col].dropna().tolist()


@dataclass
class EmbeddingFeaturizer:
    """Load pre-extracted backbone embeddings from .npz/.npy files.

    Each .npz file must contain:
      - 'embeddings': float32 array of shape (N, D)
      - 'smiles':     string array of length N

    A bare .npy file (no per-backbone smiles) is also supported, for pools
    too large to duplicate a smiles array into every backbone's output --
    see smiles_source below.

    Parameters
    ----------
    embed_dir : str
        Path to the directory containing <backbone>_embeddings.npz/.npy files.
    backbones : list[str]
        Backbone names to load (e.g. ["molformer", "grover", "unimol"]).
    smiles_source : str, optional
        Fallback ordered-SMILES file (.txt or .csv[.gz]) to use when a
        backbone's embeddings are a bare .npy with no npz sibling to read
        smiles from. Must be in the exact same row order the embeddings
        were extracted in (true by construction for AmpC: both the
        extraction pipeline and this fallback read the same
        DATASETS["AmpC"]["library"] file). Not needed for EnamineHTS,
        where each backbone's npz already carries its own smiles array.
    """

    embed_dir: str
    backbones: List[str]
    smiles_source: Optional[str] = None

    def load(self) -> tuple:
        """Load embeddings and return (emb_dict, pool_smiles).

        Returns
        -------
        emb_dict : dict[str, np.ndarray]
            {backbone: (N, D) float32}
        pool_smiles : list[str]
            Ordered SMILES corresponding to rows in each embedding matrix.
        """
        import pathlib

        embed_path = pathlib.Path(self.embed_dir)
        pool_smiles = None
        emb_dict: dict = {}
        _fallback_smiles = None  # lazily loaded once, shared across backbones

        for bb in self.backbones:
            npz_path = embed_path / f"{bb}_embeddings.npz"
            npy_path = embed_path / f"{bb}_embeddings.npy"

            if npy_path.exists():
                # Memory-mapped path: O(rows actually indexed) resident memory
                # instead of loading the full (N, D) array up front.
                embeddings = np.load(npy_path, mmap_mode="r")
                if npz_path.exists():
                    smiles = np.load(npz_path, allow_pickle=True)["smiles"].tolist()
                elif self.smiles_source is not None:
                    if _fallback_smiles is None:
                        _fallback_smiles = _load_plain_smiles(self.smiles_source)
                        if len(_fallback_smiles) != embeddings.shape[0]:
                            raise ValueError(
                                f"{self.smiles_source} has {len(_fallback_smiles):,} SMILES "
                                f"but {npy_path} has {embeddings.shape[0]:,} rows -- refusing "
                                f"to pair a mismatched smiles/embedding count."
                            )
                    smiles = _fallback_smiles
                else:
                    raise FileNotFoundError(
                        f"{npy_path} exists but sibling {npz_path} (needed for "
                        f"the smiles array) is missing, and no smiles_source "
                        f"fallback was given."
                    )
            elif npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                embeddings = data["embeddings"].astype(np.float32)
                smiles = data["smiles"].tolist()
            else:
                raise FileNotFoundError(
                    f"Embedding file not found: {npz_path}\n"
                    f"Run the embedding extraction script first."
                )

            if pool_smiles is None:
                pool_smiles = smiles
            elif pool_smiles != smiles:
                raise ValueError(
                    f"SMILES order mismatch between backbones. "
                    f"Ensure all embeddings were extracted from the same library."
                )

            emb_dict[bb] = embeddings
            print(f"[EmbeddingFeaturizer] {bb}: {embeddings.shape}")

        return emb_dict, pool_smiles
