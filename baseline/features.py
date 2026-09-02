"""
Flattened, fixed-size feature vector per window for the Phase 6 logistic
regression baseline -- "the same flattened features" the PS asks for, with
no learned transformation and no temporal context.

Design: mirrors GNNEncoder's own pooling structure (models/gnn_encoder.py:
mean-pool + max-pool over node features, mean-pool over edge features)
applied to the SAME normalized raw features the world model consumes, but
with fixed (not learned) aggregation and no message passing. This makes the
comparison in benchmark/run_benchmark.py about "does the GNN + temporal
dynamics learning help", not "does the baseline see worse features" -- both
models start from identical per-window information.

Task framing: exactly the world model's forecasting task minus the temporal
context. Given window t's flattened vector, predict whether window t+1 has
attack present. The baseline never sees more than one window at a time (no
sequence, no LSTM) -- that is precisely the capability the world model adds,
and precisely what this benchmark is meant to isolate and measure.

Same-day safety: `build_pairs_for_day` only ever pairs consecutive windows
within ONE day's snapshot list (mirroring graph/transitions.py's invariant).
`build_pairs_for_days` takes a {day: snapshots} dict (as produced by
models.dataset.split_day_snapshots) specifically so callers can't
accidentally flatten across days before pairing.
"""

from __future__ import annotations

import numpy as np

from graph.snapshot import GraphSnapshot
from models.normalization import FeatureNormalizer

# Vector layout: [node_mean(37), node_max(37), edge_mean(24), scalar_counts(3)]
# scalar_counts = [num_nodes, num_edges, num_flows], log1p-scaled (raw counts
# span orders of magnitude -- e.g. num_flows from 0 to tens of thousands --
# so a simple log1p keeps them on a comparable scale to the already
# z-scored node/edge features without needing a separately-fit scaler).


def flatten_snapshot(snapshot: GraphSnapshot, normalizer: FeatureNormalizer) -> np.ndarray:
    node_dim = snapshot.node_features.shape[1]
    edge_dim = snapshot.edge_features.shape[1]

    if snapshot.num_nodes > 0:
        norm_nodes = normalizer.transform_node(snapshot.node_features)
        node_mean = norm_nodes.mean(axis=0)
        node_max = norm_nodes.max(axis=0)
    else:
        node_mean = np.zeros(node_dim)
        node_max = np.zeros(node_dim)

    if snapshot.num_edges > 0:
        norm_edges = normalizer.transform_edge(snapshot.edge_features)
        edge_mean = norm_edges.mean(axis=0)
    else:
        edge_mean = np.zeros(edge_dim)

    scalar_counts = np.log1p([snapshot.num_nodes, snapshot.num_edges, snapshot.num_flows])

    return np.concatenate([node_mean, node_max, edge_mean, scalar_counts])


def flattened_feature_dim(node_dim: int, edge_dim: int) -> int:
    return node_dim * 2 + edge_dim + 3


def build_pairs_for_day(
    snapshots: list[GraphSnapshot], normalizer: FeatureNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for every consecutive (S_t, S_t+1) pair within one day.
    Pairs where S_t has zero observed flows are dropped -- the baseline's
    only input for that pair would be an all-zero vector, carrying no real
    information to predict from (mirrors models/dataset.py's analogous
    filter for fully-empty world-model training sequences)."""
    X, y = [], []
    for s_t, s_t1 in zip(snapshots, snapshots[1:]):
        if s_t.num_flows == 0:
            continue
        X.append(flatten_snapshot(s_t, normalizer))
        y.append(int(s_t1.graph_attack_present))

    if not X:
        node_dim = snapshots[0].node_features.shape[1] if snapshots else 0
        edge_dim = snapshots[0].edge_features.shape[1] if snapshots else 0
        return np.zeros((0, flattened_feature_dim(node_dim, edge_dim))), np.zeros((0,), dtype=np.int64)
    return np.stack(X), np.array(y, dtype=np.int64)


def build_pairs_for_days(
    day_snapshots: dict[str, list[GraphSnapshot]], normalizer: FeatureNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as build_pairs_for_day, pooled across multiple days (each day's
    pairs built independently first, so no pair ever spans a day boundary)."""
    Xs, ys = [], []
    for snapshots in day_snapshots.values():
        X, y = build_pairs_for_day(snapshots, normalizer)
        if len(X):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return np.zeros((0, 0)), np.zeros((0,), dtype=np.int64)
    return np.concatenate(Xs), np.concatenate(ys)
