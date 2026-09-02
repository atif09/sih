"""
Attention-based explainability: "which flows drove this prediction", read
directly from the trained GNN encoder's own TransformerConv attention
weights -- not a hand-picked heuristic score (contrast with Phase 4's
rollout/explain_stub.py, which is an explicitly-labeled placeholder).

Why attention instead of SHAP
------------------------------
The PS allows either ("SHAP values or attention-weight extraction ... use
[attention] if available"). GNNEncoder was built in Phase 3 specifically so
this would be available for free: TransformerConv natively exposes per-edge
attention via `return_attention_weights=True` (see models/gnn_encoder.py).
SHAP would require treating the model as a black-box function over a
choice of feature coalitions and running many perturbed forward passes per
explanation; attention extraction is a single extra forward pass and comes
from the model's actual internal computation, not a post-hoc approximation.

IMPORTANT caveat, stated plainly rather than glossed over
------------------------------------------------------------
TransformerConv's attention is a softmax **per target node**, over that
node's incoming edges only. An edge with attention weight 0.9 to a host with
only 2 incoming edges is not "more important overall" than an edge with
weight 0.4 to a host with 5 incoming edges -- the two numbers are normalized
over different competitor sets and are not globally comparable. This is a
known, general property of attention weights (not specific to this model),
and the output below is grouped/labeled by target host precisely so it's
read correctly: "among flows into this host, this one got the most weight",
not "this is the single most important flow in the whole network state".
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot
from models.dataset import snapshot_to_pyg_data
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel

_BYTES_IDX = EDGE_FEATURE_NAMES.index("bytes_sum")
_FLOW_COUNT_IDX = EDGE_FEATURE_NAMES.index("flow_count")
_PORT_IDX = EDGE_FEATURE_NAMES.index("dst_port_nunique")
_FLAG_IDX = EDGE_FEATURE_NAMES.index("flag_bitmask_or")
_RETRANS_IDX = EDGE_FEATURE_NAMES.index("retransmission_sum")


@torch.no_grad()
def compute_edge_attention(
    model: WorldModel,
    snapshot: GraphSnapshot,
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> list[dict]:
    """Runs the GNN encoder on a single snapshot with attention extraction
    enabled, and returns every edge with its real attention weight plus
    human-readable flow stats, grouped and sorted by target host (see module
    docstring for why: attention is only comparable within one target's
    incoming edges, not globally).

    Returns a list of dicts: {src_ip, dst_ip, attention_weight,
    attention_rank_within_target, bytes_sum, flow_count, dst_port_nunique,
    flag_bitmask_or, retransmission_sum, is_attack_edge}, sorted by
    (dst_ip, -attention_weight).
    """
    if snapshot.num_edges == 0:
        return []

    data = snapshot_to_pyg_data(snapshot, normalizer)
    batch = Batch.from_data_list([data]).to(device)

    model.eval()
    _embeddings, edge_attention = model.encoder(batch, return_attention=True)
    edge_attention = edge_attention.cpu().numpy()

    records = []
    for i in range(snapshot.num_edges):
        src_idx, dst_idx = snapshot.edge_index[0, i], snapshot.edge_index[1, i]
        records.append({
            "src_ip": snapshot.node_ids[src_idx],
            "dst_ip": snapshot.node_ids[dst_idx],
            "attention_weight": float(edge_attention[i]),
            "bytes_sum": float(snapshot.edge_features[i, _BYTES_IDX]),
            "flow_count": float(snapshot.edge_features[i, _FLOW_COUNT_IDX]),
            "dst_port_nunique": float(snapshot.edge_features[i, _PORT_IDX]),
            "flag_bitmask_or": int(snapshot.edge_features[i, _FLAG_IDX]),
            "retransmission_sum": float(snapshot.edge_features[i, _RETRANS_IDX]),
            "is_attack_edge": bool(snapshot.edge_labels[i]),
        })

    # Group by target, rank within each group by attention descending.
    records.sort(key=lambda r: r["dst_ip"])
    by_target: dict[str, list[dict]] = {}
    for r in records:
        by_target.setdefault(r["dst_ip"], []).append(r)
    for group in by_target.values():
        group.sort(key=lambda r: r["attention_weight"], reverse=True)
        n_competitors = len(group)
        for rank, r in enumerate(group, start=1):
            r["attention_rank_within_target"] = rank
            # How many times more attention this edge got than an
            # uninformative uniform split across its real competitors would
            # give (attention_weight / (1/n_competitors)). A target with
            # only one incoming edge trivially gets attention_weight=1.0
            # with ZERO competition -- concentration=1.0, same as "no
            # information". A genuinely-selected edge among many
            # competitors (e.g. 0.625 out of 7) scores much higher
            # (0.625*7=4.4) and is correctly surfaced as more informative by
            # top_edges_overall below, whereas ranking by raw
            # attention_weight alone would bury it under every trivial
            # single-choice target. Verified against real data: the true
            # DDoS attack edge in Friday-Afternoon-DDoS window 40 scored
            # 0.625 attention among 7 competitors (concentration=4.4) and
            # was otherwise invisible in a raw-weight ranking dominated by
            # single-edge targets' trivial 1.0 scores.
            r["attention_concentration"] = r["attention_weight"] * n_competitors

    return [r for group in by_target.values() for r in group]


def top_edges_overall(edge_records: list[dict], top_k: int = 5) -> list[dict]:
    """A pragmatic, clearly-labeled global shortlist: the top-ranked
    (attention_rank_within_target == 1) edge for each target host, sorted by
    `attention_concentration` (NOT raw attention_weight -- see the
    computation above for why that would systematically favor
    zero-competition targets over genuinely-selected ones). This does NOT
    claim cross-target comparability of the raw weights themselves (see
    module docstring) -- it's "the most surprisingly-favored incoming flow
    for each host", a defensible way to produce one global top-K list
    without pretending the raw softmax outputs are calibrated across
    different-sized competitor groups."""
    top_per_target = [r for r in edge_records if r["attention_rank_within_target"] == 1]
    top_per_target.sort(key=lambda r: r["attention_concentration"], reverse=True)
    return top_per_target[:top_k]
