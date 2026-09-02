"""
Time-windowed merge of flow-level and packet-level features into a single
normalized feature matrix.

Granularity note
------------------
Phase 2 will build one graph snapshot S_t per time window, with hosts as nodes
and flows as edges. This module produces the tabular precursor to that: every
row is one (flow, window) observation, tagged with a `window_id`/`window_start`
so Phase 2 can group rows by window to build each snapshot, and by
(src_ip, dst_ip) within a window to build edges.

Packet-level features are optional: when no packet_df is supplied (the common
case until real PCAPs are available, see packet_features.py), the merged
matrix still gets the full packet-level column set, filled with 0 and flagged
via `packet_features_available=False`, so the schema is stable regardless of
whether packet-level data was present for a given input file. Downstream code
(models, baseline) can therefore be written once against a fixed schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# Packet-level columns as produced by packet_features.extract_packet_features,
# minus identifiers/timestamps (those are used for the join key, not carried
# as standalone feature columns in the merged matrix).
PACKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "ttl_mean", "ttl_var", "ttl_min", "ttl_max",
    "tcp_window_mean", "tcp_window_var",
    "ip_frag_count", "ip_frag_ratio",
    "payload_size_mean", "payload_size_std", "payload_size_min", "payload_size_max",
    "retransmission_count",
)

# Port-scan signature columns (keyed by src_ip + window, not by flow -- see
# packet_features.detect_port_scan_signatures).
PORT_SCAN_COLUMNS: tuple[str, ...] = (
    "unique_dst_ports", "unique_dst_ips", "port_sequential_ratio",
    "is_sequential_scan", "is_randomized_scan", "port_scan_score",
)


def load_config(config_path: str | Path = "configs/pipeline.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _to_epoch_seconds(ts: pd.Series) -> pd.Series:
    """Convert a datetime64 Series to float epoch seconds.

    Deliberately NOT `ts.astype("int64") / 1e9`: pandas' datetime64 storage
    resolution is not guaranteed to be nanoseconds (it changed from a fixed
    'ns' to a per-Series resolution -- e.g. 'us' -- as of pandas 3.0), so that
    cast silently returns the wrong unit depending on which pandas version and
    parse path produced the Series. Subtracting the epoch as a Timestamp and
    dividing by a 1-second Timedelta is resolution-agnostic.
    """
    return (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


def assign_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    window_size_seconds: float,
    align_to_clock: bool = False,
) -> pd.DataFrame:
    """Add `window_start` (float epoch seconds, start of the window) and
    `window_id` (0-indexed, sequential) columns. Rows with a null timestamp are
    dropped (with a warning) since they cannot be windowed.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    n_null = ts.isna().sum()
    if n_null:
        logger.warning(
            "assign_windows: dropping %d/%d rows with unparsable/missing %s",
            n_null, len(df), timestamp_col,
        )
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]

    epoch = _to_epoch_seconds(ts)
    anchor = 0.0 if align_to_clock else float(epoch.min())
    window_index = np.floor((epoch - anchor) / window_size_seconds).astype(np.int64)

    df["window_id"] = window_index
    df["window_start"] = anchor + window_index * window_size_seconds
    return df


def merge_flow_and_packet_features(
    flow_df: pd.DataFrame,
    packet_df: pd.DataFrame | None = None,
    port_scan_df: pd.DataFrame | None = None,
    window_size_seconds: float = 60.0,
    align_to_clock: bool = False,
) -> pd.DataFrame:
    """Join flow-level features (from flow_features.extract_flow_features) with
    packet-level features (from packet_features.extract_packet_features /
    detect_port_scan_signatures), and assign both to time windows.

    Join key for packet-level flow features: the 5-tuple (src_ip, src_port,
    dst_ip, dst_port, protocol) plus window -- packet-level flows and
    CICFlowMeter flows are reconstructed independently and won't have identical
    boundaries, so we join at (5-tuple, window) granularity rather than
    assuming a 1:1 flow correspondence.

    Join key for port-scan signatures: (src_ip, window) -- these are
    source-centric, not per-flow.

    IMPORTANT: flow_df, packet_df, and port_scan_df are windowed against one
    shared anchor (computed once, below) rather than each being windowed
    independently against its own earliest timestamp. If each source picked
    its own anchor, `window_id=0` could refer to a different absolute time in
    each source (e.g. if the pcap capture started a few seconds before the
    CICFlowMeter CSV's first flow), silently misaligning the join.
    """
    flow_ts = pd.to_datetime(flow_df["timestamp"], errors="coerce")
    flow_epoch = _to_epoch_seconds(flow_ts.dropna())
    candidate_mins = [float(flow_epoch.min())] if len(flow_epoch) else []
    if packet_df is not None and not packet_df.empty:
        candidate_mins.append(float(packet_df["start_ts"].min()))
    global_min = min(candidate_mins) if candidate_mins else 0.0
    anchor = 0.0 if align_to_clock else global_min

    def _window_id(epoch_seconds: pd.Series) -> pd.Series:
        return np.floor((epoch_seconds - anchor) / window_size_seconds).astype(np.int64)

    merged = flow_df.copy()
    n_null = flow_ts.isna().sum()
    if n_null:
        logger.warning(
            "merge_flow_and_packet_features: dropping %d/%d flow rows with "
            "unparsable/missing timestamp", n_null, len(flow_df),
        )
    merged = merged[flow_ts.notna()].copy()
    merged["window_id"] = _window_id(_to_epoch_seconds(flow_ts[flow_ts.notna()]))
    merged["window_start"] = anchor + merged["window_id"] * window_size_seconds

    join_keys = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "window_id"]

    if packet_df is not None and not packet_df.empty:
        # start_ts is epoch seconds (float), not a datetime string, so we
        # window it directly here (using the shared anchor above) rather than
        # routing through pd.to_datetime.
        pkt = packet_df.copy()
        pkt["window_id"] = _window_id(pkt["start_ts"].astype(np.float64))

        agg = pkt.groupby(["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "window_id"], as_index=False)[
            list(PACKET_FEATURE_COLUMNS)
        ].mean()

        merged = merged.merge(agg, on=join_keys, how="left")
        merged["packet_features_available"] = merged[list(PACKET_FEATURE_COLUMNS)].notna().any(axis=1)
        merged[list(PACKET_FEATURE_COLUMNS)] = merged[list(PACKET_FEATURE_COLUMNS)].fillna(0.0)
    else:
        for col in PACKET_FEATURE_COLUMNS:
            merged[col] = 0.0
        merged["packet_features_available"] = False

    if port_scan_df is not None and not port_scan_df.empty:
        # port_scan_df's window_start was computed independently inside
        # packet_features.detect_port_scan_signatures (anchored at 0, i.e.
        # absolute epoch time) -- recover the raw epoch seconds and rewindow
        # with the same shared anchor used here, rather than trusting its
        # window_id to already match ours.
        scan = port_scan_df.copy()
        scan["window_id"] = _window_id(scan["window_start"].astype(np.float64))
        scan = scan.drop(columns=["window_start"])
        merged = merged.merge(scan, on=["src_ip", "window_id"], how="left")
        for col in PORT_SCAN_COLUMNS:
            if col in merged.columns:
                if merged[col].dtype == bool or col.startswith("is_"):
                    merged[col] = merged[col].fillna(False)
                else:
                    merged[col] = merged[col].fillna(0.0)
    else:
        for col in PORT_SCAN_COLUMNS:
            merged[col] = False if col.startswith("is_") else 0.0

    return merged


def normalize_feature_matrix(
    df: pd.DataFrame,
    method: str = "standard",
    exclude: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    """Scale numeric feature columns (everything except `exclude` and any
    non-numeric column). Returns (scaled_df, fitted_scalers) where
    fitted_scalers maps column -> fitted sklearn scaler, so the exact same
    transform can be applied to held-out/inference data later (critical for
    Phase 3's train/val/test split and Phase 7's inference path).
    """
    df = df.copy()
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool
    ]

    scalers = {}
    scaler_cls = StandardScaler if method == "standard" else MinMaxScaler
    for col in numeric_cols:
        scaler = scaler_cls()
        values = df[[col]].to_numpy(dtype=np.float64)
        df[col] = scaler.fit_transform(values).ravel()
        scalers[col] = scaler

    return df, scalers
