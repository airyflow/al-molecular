"""Loads muben's real Uni-Mol v1 checkpoint (mol_pre_all_h_220816.pt) into
a standalone UniMolModel.

Unlike unimol2 (checkpoint covers every weight the port uses, strict=True),
this checkpoint does NOT cover `hidden_layer` -- confirmed directly:
production (`compute_unimol_embeddings_chunk.py`) loads it with
`strict=False` specifically because of this gap. `hidden_layer` is
therefore left at whatever PyTorch's default random init produces, made
reproducible only by calling `torch.manual_seed(config.construction_seed)`
immediately before constructing the model -- see model.py's docstring for
why the *exact* module-construction sequence up to that point has to match
production bit-for-bit for this to actually reproduce the same value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import UniMolConfig
from .dictionary import DictionaryUniMol, load_production_dictionary
from .model import UniMolModel

# hidden_layer: the one submodule this checkpoint never covers (see above).
# output_layer/pair2coord_proj/dist_head weights DO exist in the real
# checkpoint file (muben builds and checkpoint-loads output_layer; the
# other two are gated off in production's own config) but this port never
# builds those modules at all (see model.py's top docstring) -- so their
# checkpoint keys are simply never requested, same net effect as muben's
# own `strict=False` silently ignoring them.
_EXPECTED_UNCOVERED_PREFIXES = ("hidden_layer.",)


def load_raw_state_dict(checkpoint_path) -> dict:
    with open(checkpoint_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    if "model" not in state:
        raise ValueError(f"{checkpoint_path}: expected a top-level 'model' key, got {list(state.keys())}")
    return state["model"]


def build_model_from_checkpoint(config: Optional[UniMolConfig] = None, dictionary: Optional[DictionaryUniMol] = None) -> UniMolModel:
    config = config or UniMolConfig()
    dictionary = dictionary or load_production_dictionary()

    # Must happen immediately before UniMolModel(...) -- see model.py's
    # docstring on why every RNG-consuming construction between this call
    # and hidden_layer's construction has to match production exactly.
    torch.manual_seed(config.construction_seed)
    model = UniMolModel(config, dictionary)

    raw_state = load_raw_state_dict(config.checkpoint_path)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(raw_state.keys())

    missing = model_keys - checkpoint_keys
    unaccounted_missing = [k for k in missing if not k.startswith(_EXPECTED_UNCOVERED_PREFIXES)]
    if unaccounted_missing:
        raise RuntimeError(
            f"{len(unaccounted_missing)} keys required by the model are missing from the checkpoint "
            f"(e.g. {sorted(unaccounted_missing)[:5]}) -- config/architecture mismatch."
        )

    state_to_load = {k: v for k, v in raw_state.items() if k in model_keys}
    result = model.load_state_dict(state_to_load, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    assert all(k.startswith(_EXPECTED_UNCOVERED_PREFIXES) for k in result.missing_keys), result.missing_keys

    model.eval()
    return model
