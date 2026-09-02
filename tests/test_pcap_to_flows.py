"""
Unit tests for features/pcap_to_flows.py, against a synthetically crafted
PCAP (same style as tests/test_packet_features.py). Exercises the real
cicflowmeter dependency end-to-end, including the two upstream-bug
workarounds documented in that module.
"""

import pytest

scapy = pytest.importorskip("scapy.all")
pytest.importorskip("cicflowmeter")

from features.pcap_to_flows import extract_flow_features_from_pcap, pcap_to_flow_csv


def _tcp_packet(src, sport, dst, dport, seq, payload_len, time, flags="S"):
    pkt = (
        scapy.IP(src=src, dst=dst, ttl=64)
        / scapy.TCP(sport=sport, dport=dport, seq=seq, window=8192, flags=flags)
        / scapy.Raw(load=b"A" * payload_len)
    )
    pkt.time = time
    return pkt


def _write_sample_pcap(path):
    packets = [
        _tcp_packet("192.168.1.10", 44000, "10.0.0.5", 443, seq=0, payload_len=100, time=0.0, flags="PA"),
        _tcp_packet("10.0.0.5", 443, "192.168.1.10", 44000, seq=0, payload_len=50, time=0.05, flags="PA"),
        _tcp_packet("172.16.0.99", 50000, "10.0.0.5", 2000, seq=0, payload_len=0, time=1.0, flags="S"),
        _tcp_packet("172.16.0.99", 50001, "10.0.0.5", 2001, seq=0, payload_len=0, time=2.0, flags="S"),
    ]
    scapy.wrpcap(str(path), packets)


def test_pcap_to_flow_csv_produces_nonempty_file(tmp_path):
    pcap_path = tmp_path / "sample.pcap"
    _write_sample_pcap(pcap_path)

    csv_path = pcap_to_flow_csv(pcap_path)
    try:
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) >= 2  # at least the two distinct flows
        assert "src_ip" in df.columns
        assert "tot_fwd_pkts" in df.columns
    finally:
        csv_path.unlink(missing_ok=True)


def test_extract_flow_features_from_pcap_matches_canonical_schema(tmp_path):
    pcap_path = tmp_path / "sample.pcap"
    _write_sample_pcap(pcap_path)

    feats = extract_flow_features_from_pcap(pcap_path)
    assert not feats.empty
    assert set(["src_ip", "dst_ip", "dst_port", "tot_pkts", "flag_bitmask"]).issubset(feats.columns)
    # No Label column in a raw capture -- must be handled gracefully (see
    # graph/snapshot.py's NaN-label fix), not crash.
    assert feats["label_binary"].isna().all()


def test_raises_clear_error_on_nonexistent_pcap(tmp_path):
    with pytest.raises(FileNotFoundError):
        pcap_to_flow_csv(tmp_path / "does_not_exist.pcap")
