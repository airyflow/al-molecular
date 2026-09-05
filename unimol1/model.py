"""Standalone port of muben's `muben/muben/model/unimol/unimol.py` (itself
"Modified from https://github.com/dptech-corp/Uni-Mol/tree/main/unimol" --
this is the real Uni-Mol v1 architecture, not a reimplementation from
scratch) -- ported to remove the muben dependency entirely, faithfully
enough to reproduce muben's own frozen-embedding output bit-exact (see
unimol1/tests/ and the one-off verification script run during development).

Only the frozen-embedding-extraction path is ported: `output_layer`
(muben's `OutputLayer`, with its BBP/MC-Dropout/ensemble uncertainty-method
branching) and `pair2coord_proj`/`dist_head` (muben's masked-coordinate/
masked-distance pretraining auxiliary heads) are all dropped, matching
unimol2's precedent of dropping post-embedding heads -- verified safe here
by directly reading production's own extraction code
(`compute_unimol_embeddings_chunk.py`'s monkey-patched `_get_embeddings`),
which returns `self.hidden_layer(encoder_rep[:, 0, :])` and never calls
`self.output_layer` or reads `pair2coord_proj`/`dist_head` at all. Also,
`compute_unimol_embeddings_chunk.py`'s own `_UniMolConfig` sets
`masked_coord_loss = masked_dist_loss = 0.0`, so muben's own `if
config.masked_coord_loss > 0` gates mean `pair2coord_proj`/`dist_head`
are never even built in production -- dropping them here changes nothing.

CRITICAL fidelity constraint, unlike unimol2: production's `hidden_layer`
(the actual embedding projection: Dropout -> Linear -> activation ->
Dropout) is NOT covered by the pretrained checkpoint (confirmed: loaded
with `strict=False` specifically because of this) -- it is left at
whatever random init falls out of construction, made reproducible only by
calling `torch.manual_seed(20260813)` immediately before constructing the
model (see checkpoint.py). That random value depends on the *exact
sequence* of every RNG-consuming module constructed BEFORE hidden_layer --
every `nn.Linear`/`nn.Embedding` in the encoder and the Gaussian
pair-distance features, in the exact order muben builds them, since
PyTorch's global RNG stream advances by the number of values drawn, not by
which module ends up covered by the checkpoint later. So: every class
below mirrors muben's attribute-assignment order EXACTLY, including
submodules whose values get fully overwritten by the checkpoint moments
later (their construction still consumes RNG draws that shift where
hidden_layer's own draw lands). Do not reorder, rename-and-reorder, or
"simplify" any `self.x = ...` sequence in `UniMolModel.__init__` --
faithfulness there is the entire point.
"""
from __future__ import annotations

import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cached_property

from .config import UniMolConfig


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/layers/layer_norm.py -- verbatim (deterministic
# ones_/zeros_ init, never RNG-consuming regardless of position).
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "tanh":
        return torch.tanh
    raise RuntimeError(f"activation_fn {activation!r} not supported")


def _softmax_dropout(input, dropout_prob, is_training=True, bias=None, inplace=True):
    """muben/muben/model/unimol/layers/softmax_dropout.py, verbatim."""
    input = input.contiguous()
    if not inplace:
        input = input.clone()
    if bias is not None:
        input += bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/layers/multihead_attention.py -- only
# SelfMultiheadAttention (CrossMultiheadAttention is never instantiated by
# UniMol, dropped).
# ---------------------------------------------------------------------------
class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key_padding_mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None, return_attn: bool = False):
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim) * self.scaling
        k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = _softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)
        else:
            attn_weights = attn_weights + attn_bias
            attn = _softmax_dropout(attn_weights, self.dropout, self.training, inplace=False)

        o = torch.bmm(attn, v)
        o = o.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.out_proj(o)
        if not return_attn:
            return o
        return o, attn_weights, attn


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/layers/transformer_encoder_layer.py -- verbatim.
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, ffn_embed_dim=3072, attention_heads=8, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, activation_fn="gelu", post_ln=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = _get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(self.embed_dim, attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(self, x, attn_bias=None, padding_mask=None, return_attn=False):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(query=x, key_padding_mask=padding_mask, attn_bias=attn_bias, return_attn=return_attn)
        attn_weights = attn_probs = None
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        return x, attn_weights, attn_probs


def init_bert_params(module):
    """muben/muben/model/unimol/layers/transformer_encoder.py, verbatim."""

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/encoder.py -- TransformerEncoderWithPair, verbatim.
# ---------------------------------------------------------------------------
class TransformerEncoderWithPair(nn.Module):
    def __init__(self, encoder_layers=6, embed_dim=768, ffn_embed_dim=3072, attention_heads=8, emb_dropout=0.1, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, max_seq_len=256, activation_fn="gelu", post_ln=False, no_final_head_layer_norm=False):
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = None if post_ln else LayerNorm(self.embed_dim)
        self.final_head_layer_norm = None if no_final_head_layer_norm else LayerNorm(attention_heads)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=self.embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads,
                dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout,
                activation_fn=activation_fn, post_ln=post_ln,
            )
            for _ in range(encoder_layers)
        ])

    def forward(self, emb, attn_mask=None, padding_mask=None):
        """Only returns the pooled token representation `x` -- upstream's
        `forward()` also returns attn_mask/delta_pair_repr/x_norm/
        delta_pair_repr_norm, all training-time auxiliary-loss diagnostics
        derived from comparing the pair-bias representation before/after
        the layer stack. None of them feed back into `x`, and nothing in
        the extraction path this port serves reads them (see this class's
        callers), so they're dropped rather than computed and discarded.
        The layer-to-layer pair-bias update itself (`attn_mask` carried
        forward as each layer's `attn_bias` input) is real architecture,
        not a diagnostic, and is preserved below."""
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), fill_val)
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for layer in self.layers:
            x, attn_mask, _ = layer(x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/module.py -- only NonLinearHead and GaussianLayer
# (DistanceHead is only used by dist_head, never built -- see this file's
# top docstring).
# ---------------------------------------------------------------------------
class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = _get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class GaussianLayer(nn.Module):
    def __init__(self, k=128, edge_types=1024):
        super().__init__()
        self.K = k
        self.means = nn.Embedding(1, k)
        self.stds = nn.Embedding(1, k)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    @cached_property
    def half_log_2pi(self):
        return 0.9189385  # float32 precision, matches muben exactly

    def gaussian_prob(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - self.half_log_2pi)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5
        return self.gaussian_prob(x, mean, std)


# ---------------------------------------------------------------------------
# muben/muben/model/unimol/unimol.py's UniMol -- ported as UniMolModel.
# Attribute-assignment order below is IDENTICAL to muben's, through
# hidden_layer -- see this file's top docstring for why that matters.
# ---------------------------------------------------------------------------
class UniMolModel(nn.Module):
    def __init__(self, config: UniMolConfig, dictionary):
        super().__init__()
        self.config = config
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), config.encoder_embed_dim, self.padding_idx)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.activation_fn,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
        )

        k = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(k, config.encoder_attention_heads, config.activation_fn)
        self.gbf = GaussianLayer(k, n_edge_type)

        # config.masked_coord_loss / masked_dist_loss are both 0.0 in
        # production -- muben's own `if > 0` gates mean pair2coord_proj /
        # dist_head are never built there either, so they're omitted here
        # entirely rather than gated (see this file's top docstring).

        self.apply(init_bert_params)

        self.hidden_layer = nn.Sequential(
            nn.Dropout(config.pooler_dropout),
            nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim),
            getattr(nn, config.pooler_activation_fn)(),
            nn.Dropout(config.pooler_dropout),
        )
        # muben's `self.output_layer = OutputLayer(...)` is constructed
        # here next, AFTER hidden_layer -- so it cannot affect
        # hidden_layer's RNG-drawn weights regardless of whether this port
        # builds it, and it's never called by the extraction path this
        # port exists for (see top docstring). Omitted.

    @torch.no_grad()
    def forward(self, batch) -> torch.Tensor:
        """batch: an object with `.atoms` [B,N], `.distances` [B,N,N],
        `.edge_types` [B,N,N] attributes (see unimol1/data/collate.py's
        Batch). Returns the pooled per-molecule embedding, [B, encoder_embed_dim]
        -- identical computation to production's monkey-patched
        `_get_embeddings` in compute_unimol_embeddings_chunk.py."""
        src_tokens, src_distance, src_edge_type = batch.atoms, batch.distances, batch.edge_types

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        n_node = src_distance.size(-1)
        gbf_feat = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feat)
        attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        encoder_rep = self.encoder(x, attn_mask=attn_bias, padding_mask=padding_mask)

        return self.hidden_layer(encoder_rep[:, 0, :])
