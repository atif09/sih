"""
Unit tests for packet_features.py against SYNTHETICALLY CRAFTED packets
(built in-memory with Scapy, written to a temp pcap). This validates the
extraction logic itself; it is not a substitute for running against real
captured traffic, which the pipeline has not yet been tested on (see module
docstring in features/packet_features.py).
"""

import pytest

scapy = pytest.importorskip("scapy.all")

from features.packet_features import (
    detect_port_scan_signatures,
    extract_packet_features,
)


def _tcp_packet(src, sport, dst, dport, ttl, window, seq, payload_len, time, flags="S", frag=0, mf=False):
    ip_flags = "MF" if mf else 0
    pkt = (
        scapy.IP(src=src, dst=dst, ttl=ttl, flags=ip_flags, frag=frag)
        / scapy.TCP(sport=sport, dport=dport, seq=seq, window=window, flags=flags)
        / scapy.Raw(load=b"A" * payload_len)
    )
    pkt.time = time
    return pkt


def test_extract_packet_features_basic_flow(tmp_path):
    pcap_path = tmp_path / "sample.pcap"
    packets = [
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1000, payload_len=50, time=0.0, flags="S"),
        _tcp_packet("10.0.0.2", 80, "10.0.0.1", 40000, ttl=60, window=4096, seq=5000, payload_len=100, time=0.1, flags="SA"),
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1050, payload_len=20, time=0.2, flags="A"),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    df = extract_packet_features(pcap_path)
    assert len(df) == 1  # all three packets belong to one bidirectional flow

    row = df.iloc[0]
    assert row["packet_count"] == 3
    # TTLs seen: 64, 60, 64 -> mean 62.67, variance > 0
    assert row["ttl_mean"] == pytest.approx((64 + 60 + 64) / 3, rel=1e-3)
    assert row["ttl_var"] > 0
    assert row["tcp_window_mean"] > 0
    assert row["payload_size_mean"] > 0
    assert row["retransmission_count"] == 0


def test_retransmission_detection(tmp_path):
    pcap_path = tmp_path / "retrans.pcap"
    packets = [
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1000, payload_len=50, time=0.0, flags="PA"),
        # Same seq + same direction + nonzero payload => retransmission.
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1000, payload_len=50, time=0.5, flags="PA"),
        _tcp_packet("10.0.0.2", 80, "10.0.0.1", 40000, ttl=60, window=4096, seq=1, payload_len=0, time=0.6, flags="A"),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    df = extract_packet_features(pcap_path)
    assert len(df) == 1
    assert df.iloc[0]["retransmission_count"] == 1


def test_ip_fragmentation_flagged(tmp_path):
    pcap_path = tmp_path / "frag.pcap"
    packets = [
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1, payload_len=50, time=0.0, flags="S", mf=True),
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=100, payload_len=50, time=0.1, flags="A", frag=185),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    df = extract_packet_features(pcap_path)
    assert df.iloc[0]["ip_frag_count"] == 2
    assert df.iloc[0]["ip_frag_ratio"] == pytest.approx(1.0)


def test_flow_timeout_splits_flow_into_two_instances(tmp_path):
    pcap_path = tmp_path / "timeout.pcap"
    packets = [
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=1, payload_len=10, time=0.0, flags="S"),
        # Gap of 200s > default 120s timeout -> new flow instance.
        _tcp_packet("10.0.0.1", 40000, "10.0.0.2", 80, ttl=64, window=8192, seq=2, payload_len=10, time=200.0, flags="S"),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    df = extract_packet_features(pcap_path, flow_timeout_seconds=120.0)
    assert len(df) == 2
    assert all(df["packet_count"] == 1)


def test_no_ip_packets_returns_empty_dataframe(tmp_path):
    pcap_path = tmp_path / "empty.pcap"
    arp = scapy.Ether() / scapy.ARP()
    arp.time = 0.0
    scapy.wrpcap(str(pcap_path), [arp])

    df = extract_packet_features(pcap_path)
    assert df.empty


def test_detect_port_scan_signatures_sequential():
    import pandas as pd

    rows = [
        {"src_ip": "172.16.0.99", "dst_ip": "10.0.0.5", "dst_port": p,
         "start_ts": 100.0 + i, "packet_count": 1}
        for i, p in enumerate(range(1000, 1010))
    ]
    df = pd.DataFrame(rows)
    result = detect_port_scan_signatures(df, window_size_seconds=60.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["unique_dst_ports"] == 10
    assert row["is_sequential_scan"]
    assert not row["is_randomized_scan"]
    assert row["port_scan_score"] > 0


def test_detect_port_scan_signatures_randomized():
    import pandas as pd

    ports = [1, 4432, 22, 8080, 53, 9999, 3389, 21]
    rows = [
        {"src_ip": "172.16.0.50", "dst_ip": "10.0.0.5", "dst_port": p,
         "start_ts": 100.0, "packet_count": 1}
        for p in ports
    ]
    df = pd.DataFrame(rows)
    result = detect_port_scan_signatures(df, window_size_seconds=60.0)

    row = result.iloc[0]
    assert row["is_randomized_scan"]
    assert not row["is_sequential_scan"]


def test_detect_port_scan_signatures_ignores_normal_traffic():
    import pandas as pd

    # A single client hitting one server on one port repeatedly is not scan-shaped.
    rows = [
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "dst_port": 443,
         "start_ts": 100.0 + i, "packet_count": 5}
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    result = detect_port_scan_signatures(df, window_size_seconds=60.0)

    row = result.iloc[0]
    assert not row["is_sequential_scan"]
    assert not row["is_randomized_scan"]
    assert row["port_scan_score"] == 0.0
