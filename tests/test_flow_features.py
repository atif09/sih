import numpy as np
import pandas as pd
import pytest

from features.flow_features import extract_flow_features
from features.schema import normalize_columns


def _make_raw_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Src IP": "10.0.0.1", "Src Port": 40001, "Dst IP": "10.0.0.2", "Dst Port": 80,
            "Protocol": 6, "Timestamp": "02/03/2018 09:00:00",
            "Flow Duration": 0,  # deliberately 0 to trigger inf in rate columns
            "Tot Fwd Pkts": 1, "Tot Bwd Pkts": 0,
            "TotLen Fwd Pkts": 60, "TotLen Bwd Pkts": 0,
            "Flow Byts/s": np.inf, "Flow Pkts/s": np.inf,
            "Flow IAT Mean": 0, "Flow IAT Std": 0, "Flow IAT Max": 0,
            "SYN Flag Cnt": 1, "ACK Flag Cnt": 0, "FIN Flag Cnt": 0,
            "RST Flag Cnt": 0, "PSH Flag Cnt": 0, "URG Flag Cnt": 0,
            "Label": "PortScan",
        },
        {
            "Src IP": "10.0.0.3", "Src Port": 50000, "Dst IP": "10.0.0.4", "Dst Port": 443,
            "Protocol": 6, "Timestamp": "02/03/2018 09:00:05",
            "Flow Duration": 500000,
            "Tot Fwd Pkts": 5, "Tot Bwd Pkts": 5,
            "TotLen Fwd Pkts": 1200, "TotLen Bwd Pkts": 3000,
            "Flow Byts/s": 8400.0, "Flow Pkts/s": 20.0,
            "Flow IAT Mean": 1000, "Flow IAT Std": 200, "Flow IAT Max": 3000,
            "SYN Flag Cnt": 1, "ACK Flag Cnt": 1, "FIN Flag Cnt": 1,
            "RST Flag Cnt": 0, "PSH Flag Cnt": 1, "URG Flag Cnt": 0,
            "Label": "Benign",
        },
    ])


def test_normalize_columns_maps_known_aliases():
    raw_cols = ["Src IP", "Dst Port", "SYN Flag Cnt", "Flow Byts/s", "TotallyUnknownColumn"]
    mapping = normalize_columns(raw_cols)
    assert mapping["Src IP"] == "src_ip"
    assert mapping["Dst Port"] == "dst_port"
    assert mapping["SYN Flag Cnt"] == "syn_flag_cnt"
    assert mapping["Flow Byts/s"] == "flow_byts_s"
    assert "TotallyUnknownColumn" not in mapping


def test_normalize_columns_strict_raises_on_unknown():
    with pytest.raises(ValueError, match="TotallyUnknownColumn"):
        normalize_columns(["Src IP", "TotallyUnknownColumn"], strict=True)


def test_extract_flow_features_shape_and_identifiers():
    raw = _make_raw_df()
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)

    assert len(out) == 2
    assert set(["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]).issubset(out.columns)
    assert out.loc[0, "src_ip"] == "10.0.0.1"
    assert out.loc[0, "dst_port"] == 80


def test_inf_rate_columns_are_sanitized():
    raw = _make_raw_df()
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)

    assert np.isfinite(out["flow_byts_s"]).all()
    assert np.isfinite(out["flow_pkts_s"]).all()
    # The inf value should have been replaced by the max *finite* value in
    # that column (8400.0 / 20.0 from row 2), not by 0 or an arbitrary sentinel.
    assert out.loc[0, "flow_byts_s"] == pytest.approx(8400.0)
    assert out.loc[0, "flow_pkts_s"] == pytest.approx(20.0)


def test_flag_bitmask_matches_bit_order():
    raw = _make_raw_df()
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)

    # Row 0: SYN only -> bit 0 set -> bitmask == 1
    assert out.loc[0, "flag_bitmask"] == 0b000001
    # Row 1: SYN, ACK, FIN, PSH -> bits 0,1,2,4 -> 1+2+4+16 = 23
    assert out.loc[1, "flag_bitmask"] == 0b010111


def test_label_binary_derivation():
    raw = _make_raw_df()
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)

    assert out.loc[0, "label"] == "PortScan"
    assert out.loc[0, "label_binary"] == 1
    assert out.loc[1, "label"] == "Benign"
    assert out.loc[1, "label_binary"] == 0


def test_timestamp_parsed_as_dayfirst():
    """Regression test: CIC-IDS2018 timestamps are DD/MM/YYYY (confirmed
    against real downloaded files' capture-day filenames, e.g.
    Thursday-01-03-2018 => March 1st). "01/03/2018" must parse as March 1,
    not January 3 (pandas' default MM/DD interpretation for ambiguous dates)."""
    raw = pd.DataFrame([{
        "Src IP": "10.0.0.1", "Src Port": 1, "Dst IP": "10.0.0.2", "Dst Port": 80,
        "Protocol": 6, "Timestamp": "01/03/2018 08:17:11",
        "Flow Duration": 100, "Tot Fwd Pkts": 1, "Tot Bwd Pkts": 1,
        "TotLen Fwd Pkts": 10, "TotLen Bwd Pkts": 10,
        "Label": "Benign",
    }])
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)
    assert out.loc[0, "timestamp"] == pd.Timestamp("2018-03-01 08:17:11")


def test_bidirectional_ratios_handle_zero_denominator():
    raw = _make_raw_df()
    raw = raw.rename(columns=normalize_columns(list(raw.columns)))
    out = extract_flow_features(raw)

    # Row 0 has 0 backward packets/bytes -- ratio must be 0, not inf/NaN.
    assert out.loc[0, "fwd_bwd_pkt_ratio"] == 0.0
    assert out.loc[0, "fwd_bwd_byte_ratio"] == 0.0
    assert np.isfinite(out["fwd_bwd_pkt_ratio"]).all()
