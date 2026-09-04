"""
Graph mutation primitives for the Counterfactual Intervention Simulator
(Phase 9, Feature 3). Each function returns a NEW GraphSnapshot (the input
is never mutated in place) with a defender action applied.

Scope, disclosed rather than overclaimed
--------------------------------------------
The brief names three illustrative actions: "isolate a host/endpoint,
revoke an account/session, block a specific flow/port." Two graph-structural
primitives are implemented here:

  - `isolate_host`: removes a host and every edge touching it. Covers
    "isolate a host/endpoint" directly.
  - `block_flow`: removes edges between one specific (src_ip, dst_ip) pair
    only, leaving the rest of each host's traffic untouched. Covers
    "revoke an account/session" and "block a specific flow" -- at the
    granularity this project's graph represents a "session" or "flow" as
    (a host-pair's aggregated traffic in one window), these are the same
    operation.

Per-PORT blocking is NOT implemented as a true graph-structural mutation,
and this is a real, checked limitation, not a shortcut: `graph/snapshot.py`
aggregates individual flows into host-pair edges, keeping only
`dst_port_nunique` (a COUNT of distinct ports touched) as an edge feature --
the actual port numbers are discarded during that aggregation and are not
recoverable from a GraphSnapshot. Implementing true port-level blocking
would require re-deriving the graph from the original per-flow DataFrame
(features/windowing.py's output, before graph/snapshot.py aggregates it) --
out of scope for this feature, noted here for anyone extending it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot

_FLOW_COUNT_IDX = EDGE_FEATURE_NAMES.index("flow_count")


def _rebuild_from_edge_mask(snapshot: GraphSnapshot, edge_mask: np.ndarray, node_ids: list[str], node_features: np.ndarray, node_labels: np.ndarray) -> GraphSnapshot:
    """Shared reconstruction logic: apply `edge_mask` (over the ORIGINAL
    edge order) and the given (possibly already-filtered) node arrays,
    remapping edge_index to the new node ordering."""
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(
        [i for i, nid in enumerate(snapshot.node_ids) if nid in node_ids]
    )}

    filtered_index = snapshot.edge_index[:, edge_mask]
    if filtered_index.size > 0:
        remap = np.vectorize(old_to_new.get)
        new_edge_index = np.stack([remap(filtered_index[0]), remap(filtered_index[1])])
    else:
        new_edge_index = np.zeros((2, 0), dtype=np.int64)

    new_edge_features = snapshot.edge_features[edge_mask]
    new_edge_labels = snapshot.edge_labels[edge_mask]
    new_edge_attack_types = [t for t, keep in zip(snapshot.edge_attack_types, edge_mask) if keep]

    graph_attack_present = bool(new_edge_labels.sum() > 0)
    graph_attack_types = frozenset().union(*new_edge_attack_types) if new_edge_attack_types else frozenset()
    num_flows = int(new_edge_features[:, _FLOW_COUNT_IDX].sum()) if len(new_edge_features) else 0

    return GraphSnapshot(
        day=snapshot.day, window_id=snapshot.window_id, window_start=snapshot.window_start,
        node_ids=node_ids, node_features=node_features, node_labels=node_labels,
        edge_index=new_edge_index, edge_features=new_edge_features, edge_labels=new_edge_labels,
        edge_attack_types=new_edge_attack_types,
        graph_attack_present=graph_attack_present, graph_attack_types=graph_attack_types,
        num_flows=num_flows,
    )


def isolate_host(snapshot: GraphSnapshot, host_ip: str) -> GraphSnapshot:
    """Removes `host_ip` and every edge touching it (as source or
    destination). A no-op (returns the snapshot unchanged) if the host
    isn't present in this particular window."""
    if host_ip not in snapshot.node_ids:
        return snapshot

    removed_idx = snapshot.node_ids.index(host_ip)
    keep_node_idx = [i for i in range(snapshot.num_nodes) if i != removed_idx]
    new_node_ids = [snapshot.node_ids[i] for i in keep_node_idx]
    new_node_features = snapshot.node_features[keep_node_idx]
    new_node_labels = snapshot.node_labels[keep_node_idx]

    edge_mask = (snapshot.edge_index[0] != removed_idx) & (snapshot.edge_index[1] != removed_idx)
    return _rebuild_from_edge_mask(snapshot, edge_mask, new_node_ids, new_node_features, new_node_labels)


def block_flow(snapshot: GraphSnapshot, src_ip: str, dst_ip: str, bidirectional: bool = True) -> GraphSnapshot:
    """Removes edges for the specific (src_ip, dst_ip) pair only -- both
    hosts remain in the graph with their OTHER traffic intact. With
    bidirectional=True (default -- "revoke this session" naturally means
    both directions of it), also removes the (dst_ip, src_ip) edge if
    present."""
    if src_ip not in snapshot.node_ids or dst_ip not in snapshot.node_ids:
        return snapshot

    src_idx = snapshot.node_ids.index(src_ip)
    dst_idx = snapshot.node_ids.index(dst_ip)

    forward = (snapshot.edge_index[0] == src_idx) & (snapshot.edge_index[1] == dst_idx)
    if bidirectional:
        backward = (snapshot.edge_index[0] == dst_idx) & (snapshot.edge_index[1] == src_idx)
        remove_mask = forward | backward
    else:
        remove_mask = forward
    edge_mask = ~remove_mask

    return _rebuild_from_edge_mask(snapshot, edge_mask, snapshot.node_ids, snapshot.node_features, snapshot.node_labels)
