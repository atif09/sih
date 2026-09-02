"""
Simple, non-model-based "top contributing flows" stub for Phase 4.

The PS explicitly allows this: "Output top contributing flags/ports/flow
patterns per prediction (stub/simple version fine for now — full
explainability comes in Phase 5)". This module ranks edges in a
GraphSnapshot by a documented, hand-picked composite score over raw
(unnormalized) edge features known to correlate with attack behavior --
NOT model attribution (no gradients, no attention weights). It exists so
every Phase 4 prediction ships with *something* interpretable attached.

Phase 5 replaces this with real attention-weight extraction (the GNN
encoder's TransformerConv layers already expose per-edge attention -- see
models/gnn_encoder.py's docstring) or SHAP values computed against the
trained model itself.

Only the last REAL observed context snapshot can be explained this way --
predicted future windows exist only as latent embeddings (see
models/world_model.py's docstring on why), so there are no raw edge features
to rank for steps 2..K of a rollout. rollout/engine.py attaches this stub's
output once, at the context level, rather than fabricating a per-future-step
version of it.
"""

from __future__ import annotations

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot

# Columns used for the composite anomaly score, and why each was picked:
#   dst_port_nunique       -- port-scan signature (many ports touched between one pair)
#   retransmission_sum     -- unusual for ordinary traffic; correlates with scanning/flooding
#   syn_cnt_sum            -- SYN-heavy edges are typical of scans/floods (vs. established sessions)
#   bytes_sum              -- raw volume; large transfers stand out for DoS/exfiltration-shaped edges
_SCORE_COLUMNS = ("dst_port_nunique", "retransmission_sum", "syn_cnt_sum", "bytes_sum")
_SCORE_INDICES = [EDGE_FEATURE_NAMES.index(c) for c in _SCORE_COLUMNS]


def top_contributing_edges(snapshot: GraphSnapshot, top_k: int = 5) -> list[dict]:
    """Returns up to top_k edges from `snapshot`, ranked by a simple
    magnitude-based composite score, each as a human-readable dict:
    {src_ip, dst_ip, score, flow_count, bytes_sum, dst_port_nunique,
     retransmission_sum, flag_bitmask_or, is_attack_edge}.
    """
    if snapshot.num_edges == 0:
        return []

    raw = snapshot.edge_features[:, _SCORE_INDICES]
    # Per-column min-max normalize within this snapshot before summing --
    # otherwise bytes_sum (often in the thousands) would completely dominate
    # a raw sum against e.g. dst_port_nunique (usually single digits).
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range < 1e-8] = 1.0
    normalized = (raw - col_min) / col_range
    scores = normalized.sum(axis=1)

    order = scores.argsort()[::-1][:top_k]

    flag_idx = EDGE_FEATURE_NAMES.index("flag_bitmask_or")
    bytes_idx = EDGE_FEATURE_NAMES.index("bytes_sum")
    flow_count_idx = EDGE_FEATURE_NAMES.index("flow_count")
    port_idx = EDGE_FEATURE_NAMES.index("dst_port_nunique")
    retrans_idx = EDGE_FEATURE_NAMES.index("retransmission_sum")

    results = []
    for i in order:
        src_idx, dst_idx = snapshot.edge_index[0, i], snapshot.edge_index[1, i]
        results.append({
            "src_ip": snapshot.node_ids[src_idx],
            "dst_ip": snapshot.node_ids[dst_idx],
            "score": float(scores[i]),
            "flow_count": float(snapshot.edge_features[i, flow_count_idx]),
            "bytes_sum": float(snapshot.edge_features[i, bytes_idx]),
            "dst_port_nunique": float(snapshot.edge_features[i, port_idx]),
            "retransmission_sum": float(snapshot.edge_features[i, retrans_idx]),
            "flag_bitmask_or": int(snapshot.edge_features[i, flag_idx]),
            "is_attack_edge": bool(snapshot.edge_labels[i]),
        })
    return results
