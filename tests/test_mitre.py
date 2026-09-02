import numpy as np

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from rollout.mitre import (
    STAGE_BENIGN,
    STAGE_COMMAND_AND_CONTROL,
    STAGE_DOS_IMPACT,
    STAGE_EXFILTRATION,
    STAGE_RECONNAISSANCE,
    predict_stage,
)

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _snapshot_with_node_features(overrides: dict[str, float], n_nodes: int = 1) -> GraphSnapshot:
    feats = np.zeros((n_nodes, NODE_DIM))
    for name, value in overrides.items():
        feats[:, NODE_FEATURE_NAMES.index(name)] = value
    return GraphSnapshot(
        day="Day", window_id=0, window_start=0.0,
        node_ids=[f"h{i}" for i in range(n_nodes)], node_features=feats,
        node_labels=np.zeros(n_nodes, dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64), edge_features=np.zeros((0, EDGE_DIM)),
        edge_labels=np.zeros((0,), dtype=np.int64), edge_attack_types=[],
        graph_attack_present=False,
    )


def test_low_infiltration_probability_is_benign():
    snap = _snapshot_with_node_features({})
    result = predict_stage(0.1, {"PortScan": 0.9}, snap)
    assert result.stage == STAGE_BENIGN
    assert result.basis == "benign"


def test_confident_attack_type_maps_via_table():
    snap = _snapshot_with_node_features({})
    result = predict_stage(0.9, {"Bot": 0.8, "PortScan": 0.1}, snap)
    assert result.stage == STAGE_COMMAND_AND_CONTROL
    assert result.basis == "attack_type"
    assert result.predicted_attack_type == "Bot"


def test_dos_family_maps_to_explicit_sixth_label_not_one_of_five():
    snap = _snapshot_with_node_features({})
    result = predict_stage(0.9, {"DoS Hulk": 0.7}, snap)
    assert result.stage == STAGE_DOS_IMPACT


def test_low_confidence_type_falls_back_to_heuristic():
    snap = _snapshot_with_node_features({"out_port_scan_score_max": 0.9})
    # No attack type crosses TYPE_CONFIDENCE_THRESHOLD (0.5).
    result = predict_stage(0.9, {"Bot": 0.2, "PortScan": 0.3}, snap)
    assert result.basis == "heuristic_fallback"
    assert result.stage == STAGE_RECONNAISSANCE


def test_no_attack_type_head_uses_heuristic_fallback():
    snap = _snapshot_with_node_features({"out_port_scan_score_max": 0.5})
    result = predict_stage(0.9, None, snap)
    assert result.basis == "heuristic_fallback"
    assert result.stage == STAGE_RECONNAISSANCE


def test_heuristic_fallback_detects_exfiltration_shaped_traffic():
    snap = _snapshot_with_node_features({"out_bytes_sum": 100000.0, "in_bytes_sum": 100.0})
    result = predict_stage(0.9, {}, snap)
    assert result.stage == STAGE_EXFILTRATION


def test_heuristic_fallback_on_empty_context_is_benign():
    snap = _snapshot_with_node_features({}, n_nodes=0)
    result = predict_stage(0.9, {}, snap)
    assert result.stage == STAGE_BENIGN
