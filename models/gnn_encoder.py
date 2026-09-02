"""
GNN encoder: encodes one GraphSnapshot (a network state S_t) into a
fixed-size latent vector, regardless of how many hosts/flows were active in
that window.

Architecture choice: stacked `TransformerConv` layers (graph attention with
native edge-feature conditioning). Chosen over a plain GCN/GraphSAGE for two
reasons:
  1. Edge features matter a lot here (byte/packet counts, flag bitmask,
     port-scan score, ...) -- TransformerConv incorporates `edge_attr`
     directly into its attention-weighted message computation, rather than
     requiring edges to be reduced to scalar weights first.
  2. It exposes per-edge attention weights (`return_attention_weights=True`),
     which Phase 5's explainability layer will use directly ("which flows
     drove this prediction") instead of needing to bolt on SHAP separately.
     Not wired up yet (Phase 3 doesn't need it), but the encoder is built so
     Phase 5 can request them without changing this module's core logic.

Zero-node graphs (a fully quiet time window -- see graph/snapshot.py's
`_empty_snapshot`) need no special-casing: they contribute zero rows to the
batched node/edge tensors, and PyTorch Geometric's `global_mean_pool` /
`global_max_pool` correctly return a zero vector for a graph index with no
assigned nodes (verified: pooling with `size=batch.num_graphs` never drops or
NaNs an empty graph's slot). A subsequent Linear layer's bias term gives the
model a learnable "nothing happened" representation for free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim

        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            # concat=False averages the per-head outputs rather than
            # concatenating them, keeping the layer's output width fixed at
            # hidden_dim regardless of `heads` -- simpler to stack.
            self.convs.append(
                TransformerConv(in_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False, dropout=dropout)
            )
            in_dim = hidden_dim

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Graph-level readout: mean+max pool of final node embeddings,
        # concatenated with a mean pool of the (normalized) raw edge
        # features -- node pooling alone can wash out edge-only signal
        # (e.g. flag counts on a pair) if it didn't strongly shape node
        # embeddings through message passing.
        pooled_dim = hidden_dim * 2 + edge_dim
        self.project = nn.Sequential(
            nn.Linear(pooled_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, batch: Batch, return_attention: bool = False):
        """batch: a PyG Batch of N graphs (possibly including zero-node
        graphs). Returns (N, latent_dim), or (embeddings, edge_attention) if
        return_attention=True, where edge_attention is (E,) -- one weight per
        edge in `batch.edge_index`, averaged across attention heads and
        across the stacked TransformerConv layers (see explain/attention.py
        for why: it's a single, simple, per-edge importance score rather
        than per-layer-per-head detail, which is what Phase 5's
        human-readable "top contributing flows" output wants).

        return_attention has no effect on the returned embeddings themselves
        (same computation either way) -- it only additionally captures what
        TransformerConv already computes internally via
        `return_attention_weights=True`, at zero cost to model behavior or
        trained weights (no new parameters).
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        num_graphs = batch.num_graphs

        h = x
        layer_attentions = []
        for conv in self.convs:
            if return_attention:
                h, (_, attn) = conv(h, edge_index, edge_attr, return_attention_weights=True)
                # attn: (E, heads) -> mean over heads for a single scalar per edge.
                layer_attentions.append(attn.mean(dim=-1))
            else:
                h = conv(h, edge_index, edge_attr)
            h = self.dropout(self.activation(h))

        node_mean = global_mean_pool(h, batch.batch, size=num_graphs)
        node_max = global_max_pool(h, batch.batch, size=num_graphs)

        if edge_attr.shape[0] > 0:
            edge_batch = batch.batch[edge_index[0]]
            edge_mean = global_mean_pool(edge_attr, edge_batch, size=num_graphs)
        else:
            edge_mean = torch.zeros(num_graphs, self.edge_dim, device=x.device, dtype=x.dtype)

        pooled = torch.cat([node_mean, node_max, edge_mean], dim=-1)
        embeddings = self.project(pooled)

        if not return_attention:
            return embeddings

        if layer_attentions:
            edge_attention = torch.stack(layer_attentions, dim=0).mean(dim=0)  # (E,)
        else:
            edge_attention = torch.zeros(edge_index.shape[1], device=x.device, dtype=x.dtype)
        return embeddings, edge_attention
