import numpy as np

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from rollout.explain_stub import top_contributing_edges

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _snapshot_with_edges(edge_specs: list[dict]) -> GraphSnapshot:
    n_nodes = len(edge_specs) + 1
    node_ids = [f"10.0.0.{i}" for i in range(n_nodes)]
    edge_index = np.array([[i for i in range(len(edge_specs))], [n_nodes - 1] * len(edge_specs)])
    edge_features = np.zeros((len(edge_specs), EDGE_DIM))
    for i, spec in enumerate(edge_specs):
        for col, val in spec.items():
            edge_features[i, EDGE_FEATURE_NAMES.index(col)] = val

    return GraphSnapshot(
        day="Day", window_id=0, window_start=0.0,
        node_ids=node_ids, node_features=np.zeros((n_nodes, NODE_DIM)),
        node_labels=np.zeros(n_nodes, dtype=np.int64),
        edge_index=edge_index, edge_features=edge_features,
        edge_labels=np.zeros(len(edge_specs), dtype=np.int64),
        edge_attack_types=[frozenset() for _ in edge_specs],
        graph_attack_present=False,
    )


def test_empty_snapshot_returns_empty_list():
    snap = _snapshot_with_edges([])
    assert top_contributing_edges(snap) == []


def test_ranks_by_composite_score_descending():
    snap = _snapshot_with_edges([
        {"dst_port_nunique": 1, "bytes_sum": 100, "syn_cnt_sum": 1, "retransmission_sum": 0},
        {"dst_port_nunique": 30, "bytes_sum": 50, "syn_cnt_sum": 20, "retransmission_sum": 5},  # scan-shaped, should rank first
        {"dst_port_nunique": 1, "bytes_sum": 10, "syn_cnt_sum": 0, "retransmission_sum": 0},
    ])
    result = top_contributing_edges(snap, top_k=3)
    assert len(result) == 3
    assert result[0]["dst_port_nunique"] == 30
    # scores strictly descending
    assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]


def test_respects_top_k():
    snap = _snapshot_with_edges([{"bytes_sum": i} for i in range(10)])
    result = top_contributing_edges(snap, top_k=3)
    assert len(result) == 3


def test_includes_human_readable_fields():
    snap = _snapshot_with_edges([{"bytes_sum": 500, "dst_port_nunique": 5}])
    result = top_contributing_edges(snap, top_k=1)
    row = result[0]
    assert row["src_ip"] == "10.0.0.0"
    assert row["dst_ip"] == snap.node_ids[-1]
    assert "is_attack_edge" in row
