import numpy as np
import pytest

from counterfactual.graph_ops import block_flow, isolate_host
from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)
FLOW_COUNT_IDX = EDGE_FEATURE_NAMES.index("flow_count")


def _snapshot(node_ids, edges, edge_labels=None, edge_attack_types=None, flow_counts=None):
    """edges: list of (src_ip, dst_ip) pairs -> builds edge_index by looking
    up indices in node_ids."""
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = np.array([[idx[s] for s, d in edges], [idx[d] for s, d in edges]], dtype=np.int64) \
        if edges else np.zeros((2, 0), dtype=np.int64)
    num_edges = len(edges)
    edge_features = np.zeros((num_edges, EDGE_DIM))
    if flow_counts is not None:
        for i, fc in enumerate(flow_counts):
            edge_features[i, FLOW_COUNT_IDX] = fc
    else:
        edge_features[:, FLOW_COUNT_IDX] = 1.0
    if edge_labels is None:
        edge_labels = np.zeros((num_edges,), dtype=np.int64)
    else:
        edge_labels = np.array(edge_labels, dtype=np.int64)
    if edge_attack_types is None:
        edge_attack_types = [frozenset() for _ in edges]

    return GraphSnapshot(
        day="Day", window_id=0, window_start=0.0,
        node_ids=list(node_ids),
        node_features=np.zeros((n, NODE_DIM)),
        node_labels=np.zeros((n,), dtype=np.int64),
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_attack_types=edge_attack_types,
        graph_attack_present=bool(np.any(edge_labels)) if num_edges else False,
        graph_attack_types=frozenset().union(*edge_attack_types) if edge_attack_types else frozenset(),
    )


class TestIsolateHost:
    def test_removes_host_and_its_edges(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C"), ("A", "C")],
        )
        result = isolate_host(snap, "B")
        assert result.node_ids == ["A", "C"]
        assert result.num_edges == 1  # only A-C survives
        # edge_index must be reindexed into the new 2-node space (0,1)
        assert result.edge_index.max() <= 1
        assert result.edge_index.min() >= 0

    def test_no_op_when_host_absent(self):
        snap = _snapshot(["A", "B"], [("A", "B")])
        result = isolate_host(snap, "Z")
        assert result is snap

    def test_recomputes_num_flows_from_remaining_edges(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C"), ("A", "C")],
            flow_counts=[5, 7, 3],
        )
        result = isolate_host(snap, "B")
        # only A-C (flow_count=3) remains
        assert result.num_flows == 3

    def test_recomputes_graph_attack_present(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
            edge_labels=[1, 0],  # attack edge touches B
            edge_attack_types=[frozenset({"PortScan"}), frozenset()],
        )
        result = isolate_host(snap, "B")
        assert result.num_edges == 0
        assert result.graph_attack_present is False
        assert result.graph_attack_types == frozenset()

    def test_preserves_attack_edge_not_touching_isolated_host(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("A", "C")],
            edge_labels=[0, 1],
            edge_attack_types=[frozenset(), frozenset({"DDoS"})],
        )
        result = isolate_host(snap, "B")
        assert result.num_edges == 1
        assert result.graph_attack_present is True
        assert result.graph_attack_types == frozenset({"DDoS"})

    def test_isolating_all_but_one_host_leaves_no_edges(self):
        snap = _snapshot(["A", "B"], [("A", "B")])
        result = isolate_host(snap, "A")
        assert result.node_ids == ["B"]
        assert result.num_edges == 0

    def test_does_not_mutate_original_snapshot(self):
        snap = _snapshot(["A", "B", "C"], [("A", "B"), ("B", "C")])
        original_node_ids = list(snap.node_ids)
        original_edge_index = snap.edge_index.copy()
        isolate_host(snap, "B")
        assert snap.node_ids == original_node_ids
        assert np.array_equal(snap.edge_index, original_edge_index)


class TestBlockFlow:
    def test_removes_only_targeted_pair_bidirectional(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("B", "A"), ("A", "C")],
        )
        result = block_flow(snap, "A", "B", bidirectional=True)
        assert result.num_edges == 1
        assert result.node_ids == ["A", "B", "C"]  # nodes untouched

    def test_unidirectional_leaves_reverse_edge(self):
        snap = _snapshot(
            ["A", "B"],
            [("A", "B"), ("B", "A")],
        )
        result = block_flow(snap, "A", "B", bidirectional=False)
        assert result.num_edges == 1
        # the surviving edge should be B->A
        assert list(result.edge_index[0]) == [1]
        assert list(result.edge_index[1]) == [0]

    def test_no_op_when_either_host_absent(self):
        snap = _snapshot(["A", "B"], [("A", "B")])
        result = block_flow(snap, "A", "Z")
        assert result is snap

    def test_leaves_other_traffic_on_same_hosts_intact(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("A", "C"), ("C", "B")],
        )
        result = block_flow(snap, "A", "B")
        assert result.num_edges == 2
        assert result.node_ids == ["A", "B", "C"]

    def test_recomputes_num_flows(self):
        snap = _snapshot(
            ["A", "B", "C"],
            [("A", "B"), ("A", "C")],
            flow_counts=[10, 4],
        )
        result = block_flow(snap, "A", "B")
        assert result.num_flows == 4

    def test_does_not_mutate_original_snapshot(self):
        snap = _snapshot(["A", "B", "C"], [("A", "B"), ("A", "C")])
        original_edge_index = snap.edge_index.copy()
        block_flow(snap, "A", "B")
        assert np.array_equal(snap.edge_index, original_edge_index)
