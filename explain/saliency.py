"""
Gradient-based feature attribution: "which specific flags/ports/flow
statistics are driving this prediction", computed as Input x Gradient
(d(attack_logit)/d(input) times the input's own value) -- a standard,
lightweight, model-agnostic attribution method.

Why this complements attention (explain/attention.py)
--------------------------------------------------------
GNN attention tells you WHICH EDGE (flow, i.e. a specific host pair)
mattered most for a prediction -- a structural, per-flow signal. It doesn't
by itself say WHICH FEATURE of that edge (byte count? port diversity?
retransmissions?) drove the weighting, and it isn't available at all for the
temporal core when using LSTM (LSTMs have no attention mechanism -- only the
optional Transformer temporal core would have one, and this project's
trained checkpoint uses LSTM). Input x Gradient works identically regardless
of which TemporalCore is in use, and gives a genuine per-named-feature
ranking (e.g. "dst_port_nunique" contributed most), directly answering the
PS's "which specific flags/ports/flow statistics" requirement at the feature
level rather than the flow level.

Why Input x Gradient and not SHAP
------------------------------------
Exact SHAP (or even sampling-based approximations like KernelSHAP) requires
many forward passes over perturbed/masked feature coalitions per
explanation, and its usual baseline-masking definition doesn't map cleanly
onto a variable-size graph batched into one autograd computation the way it
does onto a fixed-length tabular input. A single backward pass giving exact
gradients w.r.t. every input feature is cheap, well-established
(Simonyan et al. 2013 / the "gradient x input" variant used across
saliency-map literature), and -- combined with attention above -- satisfies
the PS's "no black-box outputs" requirement without the added complexity
SHAP would need here. Documented as a deliberate, reasoned choice, not a
shortcut.

Scope: only the LAST context snapshot's features are reported at the
per-feature-name granularity (the "current, most proximate observed state"
-- consistent with Phase 4's explainability stub only being able to explain
real, observed windows, not predicted future ones, which exist only as
latent embeddings). `per_window_saliency` additionally reports how much
TOTAL gradient-weighted signal came from each window in the context,
answering "which past window mattered most" for any temporal core (a
signal genuine attention can't give for an LSTM core at all).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from models.dataset import snapshot_to_pyg_data
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel


def _predict_next_logit(model: WorldModel, z_context: torch.Tensor) -> torch.Tensor:
    """Mirrors rollout/engine.py's first rollout step: run the temporal core
    once over the full observed context and predict the attack logit for the
    window immediately after it. Kept in sync deliberately -- this explains
    exactly the prediction rollout/engine.py reports as step 1."""
    if hasattr(model.temporal, "lstm"):
        hidden_seq, _ = model.temporal.lstm(z_context)
    else:
        hidden_seq = model.temporal(z_context)
    h_t = hidden_seq[:, -1, :]
    z_hat = model.dynamics_head(h_t)
    combined = torch.cat([h_t, z_hat], dim=-1)
    return model.attack_head(combined).squeeze(-1)


def compute_feature_saliency(
    model: WorldModel,
    context_snapshots: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    device: torch.device,
    top_k: int = 10,
) -> dict:
    """Returns:
      {
        "top_node_features": [(name, score), ...],   # last window, sorted desc
        "top_edge_features": [(name, score), ...],   # last window, sorted desc
        "per_window_saliency": [float, ...],           # length len(context_snapshots)
      }
    All scores are non-negative (|input * gradient|, summed over nodes/edges).
    """
    model.eval()
    last = context_snapshots[-1]

    data_list = [snapshot_to_pyg_data(s, normalizer) for s in context_snapshots]
    for d in data_list:
        d.x.requires_grad_(True)
        d.edge_attr.requires_grad_(True)
    batch = Batch.from_data_list(data_list).to(device)
    # Batch.from_data_list concatenates each Data's x/edge_attr into a NEW
    # tensor (batch.x is the output of torch.cat, not the leaf tensors we
    # just set requires_grad_ on) -- it inherits requires_grad=True from
    # those leaves, but being non-leaf, autograd won't populate its .grad
    # during backward() unless we explicitly ask it to retain one.
    batch.x.retain_grad()
    batch.edge_attr.retain_grad()

    z_context = model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(context_snapshots))
    logit = _predict_next_logit(model, z_context)
    model.zero_grad(set_to_none=True)
    logit.backward()

    node_grad = batch.x.grad
    edge_grad = batch.edge_attr.grad
    node_attr = (batch.x * node_grad).abs()
    edge_attr_score = (batch.edge_attr * edge_grad).abs()

    # Per-window totals, using the same node/edge counts used to build the
    # batch (Batch.from_data_list concatenates in order, so slicing by
    # cumulative offsets recovers each snapshot's rows).
    per_window_saliency = []
    node_offset, edge_offset = 0, 0
    last_node_slice = last_edge_slice = None
    for i, snap in enumerate(context_snapshots):
        n, e = snap.num_nodes, snap.num_edges
        window_total = float(node_attr[node_offset:node_offset + n].sum().item()
                              + edge_attr_score[edge_offset:edge_offset + e].sum().item())
        per_window_saliency.append(window_total)
        if i == len(context_snapshots) - 1:
            last_node_slice = slice(node_offset, node_offset + n)
            last_edge_slice = slice(edge_offset, edge_offset + e)
        node_offset += n
        edge_offset += e

    node_feature_scores = node_attr[last_node_slice].sum(dim=0).detach().cpu().numpy() if last.num_nodes else np.zeros(len(NODE_FEATURE_NAMES))
    edge_feature_scores = edge_attr_score[last_edge_slice].sum(dim=0).detach().cpu().numpy() if last.num_edges else np.zeros(len(EDGE_FEATURE_NAMES))

    top_node = sorted(zip(NODE_FEATURE_NAMES, node_feature_scores), key=lambda kv: kv[1], reverse=True)[:top_k]
    top_edge = sorted(zip(EDGE_FEATURE_NAMES, edge_feature_scores), key=lambda kv: kv[1], reverse=True)[:top_k]

    return {
        "top_node_features": [(name, float(score)) for name, score in top_node],
        "top_edge_features": [(name, float(score)) for name, score in top_edge],
        "per_window_saliency": per_window_saliency,
    }
