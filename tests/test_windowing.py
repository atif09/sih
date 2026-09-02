import numpy as np
import pandas as pd
import pytest

from features.windowing import (
    assign_windows,
    merge_flow_and_packet_features,
    normalize_feature_matrix,
)


def test_assign_windows_buckets_correctly():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2018-03-02 09:00:00", "2018-03-02 09:00:30",
            "2018-03-02 09:01:05", "2018-03-02 09:02:00",
        ]),
        "value": [1, 2, 3, 4],
    })
    out = assign_windows(df, "timestamp", window_size_seconds=60.0)
    # First two rows fall in window 0 (0-60s from anchor), third in window 1
    # (65s in), fourth in window 2 (120s in).
    assert list(out["window_id"]) == [0, 0, 1, 2]


def test_assign_windows_drops_unparsable_timestamps():
    df = pd.DataFrame({
        "timestamp": ["2018-03-02 09:00:00", "not-a-date", "2018-03-02 09:01:00"],
        "value": [1, 2, 3],
    })
    out = assign_windows(df, "timestamp", window_size_seconds=60.0)
    assert len(out) == 2
    assert list(out["value"]) == [1, 3]


def _flow_row(src_ip, sport, dst_ip, dport, proto, ts):
    return {
        "src_ip": src_ip, "src_port": sport, "dst_ip": dst_ip, "dst_port": dport,
        "protocol": proto, "timestamp": pd.Timestamp(ts),
        "tot_pkts": 5, "label": "Benign", "label_binary": 0,
    }


def test_merge_uses_shared_anchor_across_sources():
    """Regression test: flow_df and packet_df must be windowed against the
    same anchor, otherwise window_id=0 means different absolute times in each
    source and the merge silently misaligns."""
    flow_df = pd.DataFrame([
        _flow_row("10.0.0.1", 1000, "10.0.0.2", 80, 6, "2018-03-02 09:00:10"),
    ])
    # Packet capture started 5 seconds *before* the flow's timestamp -- if
    # each source picked its own min-as-anchor, the flow row would land in
    # window 0 of its own frame but the packet row (5s earlier) would need to
    # land in window 0 of *its* frame too, which is a different absolute time.
    packet_df = pd.DataFrame([{
        "src_ip": "10.0.0.1", "src_port": 1000, "dst_ip": "10.0.0.2", "dst_port": 80,
        "protocol": 6, "start_ts": pd.Timestamp("2018-03-02 09:00:05").timestamp(),
        "ttl_mean": 64.0, "ttl_var": 0.0, "ttl_min": 64.0, "ttl_max": 64.0,
        "tcp_window_mean": 8192.0, "tcp_window_var": 0.0,
        "ip_frag_count": 0, "ip_frag_ratio": 0.0,
        "payload_size_mean": 50.0, "payload_size_std": 0.0,
        "payload_size_min": 50.0, "payload_size_max": 50.0,
        "retransmission_count": 0,
    }])

    merged = merge_flow_and_packet_features(flow_df, packet_df, window_size_seconds=60.0)

    assert len(merged) == 1
    assert merged.loc[0, "packet_features_available"] == True  # noqa: E712
    assert merged.loc[0, "ttl_mean"] == pytest.approx(64.0)


def test_merge_without_packet_data_fills_defaults():
    flow_df = pd.DataFrame([
        _flow_row("10.0.0.1", 1000, "10.0.0.2", 80, 6, "2018-03-02 09:00:10"),
    ])
    merged = merge_flow_and_packet_features(flow_df, packet_df=None, window_size_seconds=60.0)

    assert merged.loc[0, "packet_features_available"] == False  # noqa: E712
    assert merged.loc[0, "ttl_mean"] == 0.0
    assert merged.loc[0, "is_sequential_scan"] == False  # noqa: E712


def test_normalize_feature_matrix_excludes_identifiers():
    df = pd.DataFrame({
        "src_ip": ["10.0.0.1", "10.0.0.2"],
        "dst_port": [80, 443],
        "tot_pkts": [10.0, 1000.0],
        "label_binary": [0, 1],
    })
    normalized, scalers = normalize_feature_matrix(
        df, method="standard", exclude=("src_ip", "dst_port", "label_binary")
    )

    assert "tot_pkts" in scalers
    assert "src_ip" not in scalers
    assert "label_binary" not in scalers
    # standard-scaled 2-point column should be +/-1 (population std)
    assert normalized["tot_pkts"].abs().sum() == pytest.approx(2.0, rel=1e-3)
    # excluded columns unchanged
    assert list(normalized["dst_port"]) == [80, 443]
