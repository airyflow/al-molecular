"""Standalone port of muben's `muben/muben/dataset/dataset_unimol/dictionary.py`
(itself ported from dptech-corp/Uni-Mol). Ported verbatim -- no muben
dependency, no RNG interaction (pure Python list/dict bookkeeping), so
there's zero fidelity risk in copying this class directly.

The fixed 30-symbol vocabulary below (`UNIMOL_DICT`) is the real released
Uni-Mol dictionary -- not a placeholder. `len(DictionaryUniMol.load())`
must equal 30, and `.pad()` must equal 0 (`[PAD]` is index 0), since
`unimol1/model.py`'s `embed_tokens`/`gbf` sizes and `padding_idx` are
derived directly from this.
"""
from __future__ import annotations

import numpy as np

UNIMOL_DICT = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[UNK]",
    "C",
    "N",
    "O",
    "S",
    "H",
    "Cl",
    "F",
    "Br",
    "I",
    "Si",
    "P",
    "B",
    "Na",
    "K",
    "Al",
    "Ca",
    "Sn",
    "As",
    "Hg",
    "Fe",
    "Zn",
    "Cr",
    "Se",
    "Gd",
    "Au",
    "Li",
]


class DictionaryUniMol:
    """A mapping from symbols to consecutive integers."""

    def __init__(self, *, bos="[CLS]", pad="[PAD]", eos="[SEP]", unk="[UNK]"):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        return self.index(self.bos_word)

    def pad(self):
        return self.index(self.pad_word)

    def eos(self):
        return self.index(self.eos_word)

    def unk(self):
        return self.index(self.unk_word)

    @classmethod
    def load(cls):
        d = cls()
        d.add_from_macro(UNIMOL_DICT)
        return d

    def add_from_macro(self, token_list):
        for line_idx, token in enumerate(token_list):
            count = len(token_list) - line_idx
            self.add_symbol(token, n=count, overwrite=False)


def load_production_dictionary() -> DictionaryUniMol:
    """The exact dictionary construction production actually uses (see
    `compute_unimol_embeddings_chunk.py`, `embed/extract_embeddings.py`,
    `backbone_finetuner.py` -- all three call `DictionaryUniMol.load()`
    THEN `.add_symbol("[MASK]", is_special=True)`), NOT `.load()` alone.

    The real checkpoint's `embed_tokens`/`gbf.mul`/`gbf.bias` tables were
    built against this 31-symbol vocabulary (30 from UNIMOL_DICT + 1
    appended [MASK]) even though no real SMILES ever tokenizes to [MASK] --
    caught directly by `build_model_from_checkpoint()`'s strict shape
    check when this function wasn't yet used here (embed_tokens came back
    (31, 512) from the checkpoint vs. (30, 512) from a bare `.load()`)."""
    dictionary = DictionaryUniMol.load()
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary
