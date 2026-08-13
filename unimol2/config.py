"""Config for the standalone Uni-Mol2 port.

Every field here is sourced from one of two places, verified directly against
upstream source (deepmodeling/Uni-Mol, unimol2/unimol2/models/unimol2.py) --
never an invented "seems reasonable" default:

  1. The `unimol2_1100M` architecture registration (`register_model_architecture`
     calls in unimol2.py) for the 1.1B checkpoint specifically.
  2. Constants hardcoded directly in `UniMol2Model.__init__` (not args-driven
     upstream either -- e.g. num_atom=512, token_num=128) -- these are the same
     regardless of model size (84M through 1.1B all share them).

All of these were additionally cross-validated against the real downloaded
checkpoint's state_dict tensor shapes (e.g. embed_tokens.weight is (128, 1536),
matching token_num=128 and encoder_embed_dim=1536) before being trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ZOO = ROOT / "models"


@dataclass
class UniMol2Config:
    # Architecture -- from unimol2_1100M registration.
    encoder_layers: int = 64
    encoder_embed_dim: int = 1536
    encoder_attention_heads: int = 96
    encoder_ffn_embed_dim: int = 1536  # 1x expansion here, NOT the usual 4x -- verified
    pair_embed_dim: int = 512
    pair_hidden_dim: int = 64
    activation_fn: str = "gelu"
    pooler_activation_fn: str = "tanh"

    # Dropout -- eval-mode override. Upstream training defaults (dropout=0.1 etc.)
    # are irrelevant at inference (nn.Dropout/F.dropout are no-ops when
    # module.training=False), set to 0.0 here for clarity, not correctness.
    dropout: float = 0.0
    emb_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0

    max_seq_len: int = 512

    # Hardcoded constants inside UniMol2Model.__init__ upstream -- NOT args-driven,
    # identical across all model sizes (84M through 1.1B).
    num_atom: int = 512
    num_degree: int = 128
    num_edge: int = 64
    num_pair: int = 512
    num_spatial: int = 512
    gaussian_kernel_k: int = 128
    token_num: int = 128
    padding_idx: int = 0
    mask_idx: int = 127

    # Gaussian kernel range for the 3D distance featurization (SE3InvariantKernel).
    # Not exposed by any register_model_architecture override -- inherited from
    # base_architecture's own defaults, which every size (including 1100M) uses.
    gaussian_std_width: float = 1.0
    gaussian_mean_start: float = 0.0
    gaussian_mean_stop: float = 9.0

    checkpoint_path: str = str(MODEL_ZOO / "unimol2" / "1.1B" / "checkpoint.pt")
