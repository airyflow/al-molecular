"""Standalone replacements for the Uni-Core plumbing Uni-Mol2's model code
imports. Each one is verified (by reading Uni-Core's actual source,
dptech-corp/Uni-Core) to be either pure registration/init-only plumbing, or a
fused-CUDA-kernel wrapper whose own CPU fallback is already ordinary PyTorch --
never something that hides real model math. See module docstrings below for
the exact upstream file each is ported from.

`unicore.modules.LayerNorm` is NOT reimplemented here at all: reading
unicore/modules/layer_norm.py directly shows it's `torch.nn.functional.layer_norm`
under the hood whenever no fused CUDA kernel is installed (same registered
`weight`/`bias` parameter names as plain `torch.nn.LayerNorm`), so this port
uses `torch.nn.LayerNorm` everywhere upstream uses `unicore.modules.LayerNorm` --
state_dict keys match exactly, zero behavior difference.

`init_bert_params` (unicore.modules) is also not ported: it's a training-time
weight-init helper (`self.apply(init_bert_params)` at the end of
UniMol2Model.__init__ upstream), fully overwritten the moment a pretrained
checkpoint's state_dict is loaded afterward -- irrelevant to this port, which
always loads pretrained weights immediately after construction.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


def get_activation_fn(activation: str) -> Callable:
    """Ported verbatim from unicore/utils.py::get_activation_fn -- a plain
    dict-style dispatcher, no math of its own."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"activation_fn {activation!r} not supported")


def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    """Ported verbatim from unicore/utils.py::permute_final_dims."""
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def softmax_dropout(
    input: torch.Tensor,
    dropout_prob: float,
    is_training: bool = True,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CPU fallback path of unicore/modules/softmax_dropout.py::softmax_dropout,
    ported verbatim (the fused-CUDA-kernel branch, taken only when
    `input.is_cuda`, is skipped entirely -- this repo runs on CPU-primary
    BigRed200 nodes, so that branch is never exercised regardless)."""
    input = input.contiguous()
    if mask is not None:
        input = input + mask
    if bias is not None:
        input = input + bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)
