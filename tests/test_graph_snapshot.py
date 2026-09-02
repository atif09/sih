"""
Unit tests for graph/snapshot.py, built against small hand-crafted "already
windowed" flow tables (the shape produced by
features.windowing.merge_flow_and_packet_features), rather than real data.
"""

import numpy as np
import pandas as pd
import pytest

from graph.snapshot import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    NODE_LABEL_ATTACK_BOTH,
    NODE_LABEL_ATTACK_SOURCE,
    NODE_LABEL_ATTACK_TARGET,
    NODE_LABEL_BENIGN,
    build_snapshots_for_day,
)

REQUIRED_COLS = [
    "window_id", "window_start", "src_ip", "src_port", "dst_ip", "dst_port",
    "tot_bytes", "tot_pkts", "duration_us", "flow_iat_mean",
    "syn_flag_cnt", "ack_flag_cnt", "fin_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "urg_flag_cnt", "flag_bitmask", "down_up_ratio",
    "fwd_bwd_byte_ratio", "ttl_mean", "ttl_var", "tcp_window_mean",
    "ip_frag_ratio", "payload_size_mean", "retransmission_count",
    "packet_features_available", "port_scan_score", "label", "label_binary",
]


def _row(window_id, window_start, src_ip, dst_ip, dst_port=80, label="BENIGN", **overrides):
    row = {c: 0 for c in REQUIRED_COLS}
    row.update({
        "window_id": window_id, "window_start": window_start,
        "src_ip": src_ip, "src_port": 40000, "dst_ip": dst_ip, "dst_port": dst_port,
        "tot_bytes": 100, "tot_pkts": 2, "duration_us": 1000, "flow_iat_mean": 500.0,
        "flag_bitmask": 0b000001, "down_up_ratio": 1.0, "fwd_bwd_byte_ratio": 1.0,
        "packet_features_available": False,
        "label": label, "label_binary": 0 if label.upper() == "BENIGN" else 1,
    })
    row.update(overrides)
    return row


def test_single_window_two_hosts_one_edge():
    df = pd.DataFrame([
        _row(0, 0.0, "10.0.0.1", "10.0.0.2"),
        _row(0, 0.0, "10.0.0.1", "10.0.0.2", dst_port=443),
    ])
    snapshots = build_snapshots_for_day(df, day="TestDay")

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.window_id == 0
    assert set(snap.node_ids) == {"10.0.0.1", "10.0.0.2"}
    assert snap.num_edges == 1  # both flows collapse into one (src,dst) edge

    edge_flow_count_idx = EDGE_FEATURE_NAMES.index("flow_count")
    assert snap.edge_features[0, edge_flow_count_idx] == 2

    edge_bytes_idx = EDGE_FEATURE_NAMES.index("bytes_sum")
    assert snap.edge_features[0, edge_bytes_idx] == 200  # 100 + 100

    assert snap.graph_attack_present is False
    assert (snap.node_labels == NODE_LABEL_BENIGN).all()


def test_directed_edge_and_attack_labeling():
    df = pd.DataFrame([
        _row(0, 0.0, "172.16.0.99", "10.0.0.5", dst_port=22, label="FTP-Patator"),
        _row(0, 0.0, "10.0.0.5", "172.16.0.99", dst_port=50000),  # return traffic, opposite direction
    ])
    snapshots = build_snapshots_for_day(df, day="TestDay")
    snap = snapshots[0]

    assert snap.num_edges == 2  # directed: (attacker->victim) and (victim->attacker) are distinct edges
    assert snap.graph_attack_present is True
    assert "FTP-Patator" in snap.graph_attack_types

    node_index = {ip: i for i, ip in enumerate(snap.node_ids)}
    assert snap.node_labels[node_index["172.16.0.99"]] == NODE_LABEL_ATTACK_SOURCE
    assert snap.node_labels[node_index["10.0.0.5"]] == NODE_LABEL_ATTACK_TARGET


def test_host_both_source_and_target_of_attack():
    df = pd.DataFrame([
        _row(0, 0.0, "10.0.0.9", "10.0.0.1", label="PortScan"),  # 10.0.0.9 attacks 10.0.0.1
        _row(0, 0.0, "10.0.0.2", "10.0.0.9", label="DDoS"),      # 10.0.0.2 attacks 10.0.0.9
    ])
    snapshots = build_snapshots_for_day(df, day="TestDay")
    snap = snapshots[0]
    node_index = {ip: i for i, ip in enumerate(snap.node_ids)}
    assert snap.node_labels[node_index["10.0.0.9"]] == NODE_LABEL_ATTACK_BOTH


def test_port_scan_signature_max_aggregated_onto_node():
    df = pd.DataFrame([
        _row(0, 0.0, "172.16.0.99", "10.0.0.5", dst_port=1000, port_scan_score=0.4),
        _row(0, 0.0, "172.16.0.99", "10.0.0.5", dst_port=1001, port_scan_score=0.8),
    ])
    snapshots = build_snapshots_for_day(df, day="TestDay")
    snap = snapshots[0]
    node_index = {ip: i for i, ip in enumerate(snap.node_ids)}
    score_idx = NODE_FEATURE_NAMES.index("out_port_scan_score_max")
    assert snap.node_features[node_index["172.16.0.99"], score_idx] == pytest.approx(0.8)

    # And the edge should record 2 distinct destination ports touched.
    port_idx = EDGE_FEATURE_NAMES.index("dst_port_nunique")
    assert snap.edge_features[0, port_idx] == 2


def test_gap_filling_produces_empty_windows():
    # Only windows 0 and 3 have traffic -- windows 1, 2 must still appear,
    # empty, so a fixed-Δt sequence model sees uniform time steps.
    df = pd.DataFrame([
        _row(0, 0.0, "10.0.0.1", "10.0.0.2"),
        _row(3, 180.0, "10.0.0.1", "10.0.0.2"),
    ])
    snapshots = build_snapshots_for_day(df, day="TestDay")

    assert [s.window_id for s in snapshots] == [0, 1, 2, 3]
    assert snapshots[1].num_nodes == 0
    assert snapshots[1].num_edges == 0
    assert snapshots[1].graph_attack_present is False
    # Interpolated window_start for the empty windows should follow the
    # inferred 60s step (window 0 -> 0.0, window 3 -> 180.0 => step 60).
    assert snapshots[1].window_start == pytest.approx(60.0)
    assert snapshots[2].window_start == pytest.approx(120.0)


def test_empty_dataframe_returns_no_snapshots():
    df = pd.DataFrame(columns=REQUIRED_COLS)
    assert build_snapshots_for_day(df, day="TestDay") == []
