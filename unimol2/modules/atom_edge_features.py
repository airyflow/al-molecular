"""Ported from unimol2/unimol2/models/layers.py::AtomFeature, EdgeFeature
(deepmodeling/Uni-Mol). Verbatim forward-pass logic; the custom `Linear`/
`Embedding` init-scheme wrapper classes upstream are replaced with plain
`nn.Linear`/`nn.Embedding` since their only difference is weight
initialization, which is irrelevant once a pretrained checkpoint's
state_dict is loaded (same parameter names/shapes either way)."""
from __future__ import annotations

import torch
import torch.nn as nn


class AtomFeature(nn.Module):
    """Combines per-atom token embedding, per-atom RDKit/OGB-style feature
    embedding, and node degree embedding into one atom-track representation,
    then prepends a learned "virtual node" row (mechanically identical to a
    BERT [CLS] token) that becomes the pooled molecule embedding after the
    encoder runs."""

    def __init__(self, num_atom: int, num_degree: int, hidden_dim: int):
        super().__init__()
        self.atom_encoder = nn.Embedding(num_atom, hidden_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder = nn.Embedding(1, hidden_dim)

    def forward(self, batched_data: dict, token_feat: torch.Tensor) -> torch.Tensor:
        x, degree = batched_data["atom_feat"], batched_data["degree"]
        n_graph = x.size(0)

        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, hidden]
        dtype = node_feature.dtype
        degree_feature = self.degree_encoder(degree)
        node_feature = node_feature + degree_feature + token_feat

        graph_token_feature = self.vnode_encoder.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature.type(dtype)


class EdgeFeature(nn.Module):
    """Builds the initial pair-track representation (attention bias) from
    bond features and shortest-path graph distance, including the
    virtual-node row/column entries."""

    def __init__(self, pair_dim: int, num_edge: int, num_spatial: int):
        super().__init__()
        self.pair_dim = pair_dim
        self.edge_encoder = nn.Embedding(num_edge, pair_dim, padding_idx=0)
        self.shorest_path_encoder = nn.Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = nn.Embedding(1, pair_dim)

    def forward(self, batched_data: dict, graph_attn_bias: torch.Tensor) -> torch.Tensor:
        shortest_path = batched_data["shortest_path"]
        edge_input = batched_data["edge_feat"]

        graph_attn_bias[:, 1:, 1:, :] = self.shorest_path_encoder(shortest_path)

        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, 1:, 0, :] = t
        graph_attn_bias[:, 0, :, :] = t

        edge_input = self.edge_encoder(edge_input).mean(-2)
        graph_attn_bias[:, 1:, 1:, :] = graph_attn_bias[:, 1:, 1:, :] + edge_input
        return graph_attn_bias
