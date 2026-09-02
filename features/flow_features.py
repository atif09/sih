"""
Flow-level feature extraction from CIC-IDS2018 processed CSVs (CICFlowMeter output).

Ingests a raw CSV, normalizes its columns against the canonical schema in
schema.py, and produces a DataFrame of engineered flow-level features:
    - identifiers: src_ip:src_port, dst_ip:dst_port, protocol, timestamp
    - TCP flag bitmask (SYN/ACK/FIN/RST/PSH/URG) + raw per-flag counts
    - bytes/packets per flow (total, forward, backward)
    - flow duration
    - inter-arrival time stats: mean/variance/max (flow-level, fwd, bwd)
    - bidirectional flow ratios (down/up ratio, fwd:bwd byte and packet ratios)
    - label (raw, plus a derived binary benign/attack column)

This module intentionally keeps a 1:1 mapping to "one row in, one row out" (flow
granularity) -- windowing/aggregation into network *state* snapshots happens in
windowing.py, and graph construction happens in Phase 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.schema import (
    FLAG_BIT_ORDER,
    IDENTIFIER_COLUMNS,
    INF_PRONE_COLUMNS,
    normalize_columns,
)

logger = logging.getLogger(__name__)

# Known benign label strings across CIC-IDS2018 capture days (case varies:
# "Benign", "BENIGN"). Anything not matching this set is treated as an attack.
BENIGN_LABELS = {"benign"}


def load_cicflowmeter_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CICFlowMeter CSV and return it with normalized (canonical) column
    names. Handles the known "duplicated header row" quirk in some CIC-IDS2018
    files by dropping any data row whose Label-equivalent cell is the literal
    string "Label" (i.e. a repeated header row read in as data).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Place CICFlowMeter CSVs under data/ "
            f"(see configs/pipeline.yaml -> paths, and features/datasets.py "
            f"for the CIC-IDS2017 file registry)."
        )

    # low_memory=False avoids dtype-guessing warnings on these wide (80+ col) files.
    # Not every day's file is valid UTF-8 -- e.g. CIC-IDS2017's Thursday
    # web-attacks file has a stray 0x96 byte in its Label column. That byte is
    # a Windows-1252 EN DASH (from "Web Attack – Brute Force" being saved
    # by a Windows tool), not arbitrary binary -- decoding it as Latin-1 would
    # "succeed" but produce the wrong (unprintable) character, silently
    # corrupting the label text. Try UTF-8, then cp1252 (correctly handles
    # this and other Windows-authored smart-quote/dash bytes), and only fall
    # back to latin-1 as a last resort (it never raises -- every byte maps to
    # *some* code point -- but for exactly that reason it's the least likely
    # to be the *right* decoding, so it's tried last).
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, nrows=nrows, low_memory=False, skip_blank_lines=True, encoding=encoding)
            if encoding != "utf-8":
                logger.warning("%s: not valid UTF-8, decoded with %s instead", path.name, encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Unreachable in practice -- latin-1 maps every byte to some code
        # point and cannot raise -- but keeps this loop honest if that ever
        # changes upstream.
        raise ValueError(f"{path}: failed to decode with utf-8, cp1252, or latin-1")

    col_map = normalize_columns(list(df.columns), strict=False)
    unmapped = [c for c in df.columns if c not in col_map]
    if unmapped:
        logger.warning(
            "%s: %d column(s) not in canonical schema and will be dropped: %s",
            path.name, len(unmapped), unmapped,
        )
    df = df.rename(columns=col_map)
    df = df[[c for c in df.columns if c in col_map.values()]]

    # Drop duplicated in-file header rows (label column literally contains "Label").
    if "label" in df.columns:
        before = len(df)
        df = df[df["label"].astype(str).str.strip().str.lower() != "label"]
        dropped = before - len(df)
        if dropped:
            logger.warning("%s: dropped %d duplicated header row(s)", path.name, dropped)

    return df.reset_index(drop=True)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """In-place: force listed columns to numeric, turning unparsable values into
    NaN rather than raising (some CSVs have stray non-numeric junk rows)."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _compute_flag_bitmask(df: pd.DataFrame) -> pd.Series:
    """Build a per-flow TCP flag bitmask from CICFlowMeter's per-flag *counts*.

    CICFlowMeter reports how many packets in the flow carried each flag (e.g.
    `syn_flag_cnt`), not a single combined per-packet flag byte -- a flow can
    legitimately have both SYN and FIN set (SYN at open, FIN at close). We
    therefore define the bitmask as "was this flag present anywhere in the
    flow" (count > 0), which is the flow-level analogue of a packet's flag
    byte. Bit order matches the PS: SYN=bit0, ACK=bit1, FIN=bit2, RST=bit3,
    PSH=bit4, URG=bit5.
    """
    bitmask = np.zeros(len(df), dtype=np.int64)
    for bit, col in enumerate(FLAG_BIT_ORDER):
        if col in df.columns:
            present = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).to_numpy()
            bitmask = bitmask | (present.astype(np.int64) << bit)
    return pd.Series(bitmask, index=df.index)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator/denominator, returning 0 where denominator is 0
    (avoids inf/NaN from divide-by-zero, which is common for single-direction
    flows e.g. a SYN scan that gets no reply)."""
    denom = denominator.replace(0, np.nan)
    ratio = numerator / denom
    return ratio.fillna(0.0)


def extract_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a normalized CICFlowMeter DataFrame (see load_cicflowmeter_csv)
    into the engineered flow-level feature set described in the module
    docstring. Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()

    numeric_candidates = [c for c in df.columns if c not in IDENTIFIER_COLUMNS + ("label",)]
    _coerce_numeric(df, numeric_candidates)

    # Flow Byts/s and Flow Pkts/s divide by duration, which is 0 for flows that
    # complete within one CICFlowMeter timing tick. Replace resulting +/-inf
    # with the max finite value observed in that column (a large-but-finite
    # sentinel), which keeps "this was a very fast flow" as a *large* value
    # instead of corrupting downstream normalization/training with inf/NaN.
    for col in INF_PRONE_COLUMNS:
        if col in df.columns:
            finite = df[col].replace([np.inf, -np.inf], np.nan)
            fill_value = finite.max()
            if pd.isna(fill_value):
                fill_value = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], fill_value)

    out = pd.DataFrame(index=df.index)

    # --- identifiers ---
    for col in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol"):
        out[col] = df[col] if col in df.columns else np.nan
    if "timestamp" in df.columns:
        # CIC-IDS2018's Timestamp column is day-first (DD/MM/YYYY HH:MM:SS,
        # e.g. "01/03/2018 08:17:11" for March 1st -- confirmed against the
        # capture-day filenames, which use DD-MM-YYYY, e.g.
        # "Thursday-01-03-2018"). Without dayfirst=True, pandas' default
        # MM/DD interpretation silently misparses any ambiguous date (day
        # <= 12), e.g. reading March 1st as January 3rd.
        out["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", format="mixed", dayfirst=True
        )
    else:
        logger.warning(
            "No 'Timestamp' column found -- this CSV variant may have stripped "
            "it for anonymization. Windowing (windowing.py) requires timestamps; "
            "flag this file to the user before using it for Phase 2."
        )
        out["timestamp"] = pd.NaT

    # --- duration ---
    out["duration_us"] = df.get("flow_duration", np.nan)

    # --- packet/byte totals ---
    out["fwd_pkts"] = df.get("tot_fwd_pkts", 0.0)
    out["bwd_pkts"] = df.get("tot_bwd_pkts", 0.0)
    out["fwd_bytes"] = df.get("totlen_fwd_pkts", 0.0)
    out["bwd_bytes"] = df.get("totlen_bwd_pkts", 0.0)
    out["tot_pkts"] = out["fwd_pkts"] + out["bwd_pkts"]
    out["tot_bytes"] = out["fwd_bytes"] + out["bwd_bytes"]

    # --- IAT stats (flow / fwd / bwd) ---
    for prefix in ("flow", "fwd", "bwd"):
        for stat in ("mean", "std", "max"):
            col = f"{prefix}_iat_{stat}"
            out[col] = df.get(col, 0.0)

    # --- bidirectional ratios ---
    out["fwd_bwd_pkt_ratio"] = _safe_ratio(out["fwd_pkts"], out["bwd_pkts"])
    out["fwd_bwd_byte_ratio"] = _safe_ratio(out["fwd_bytes"], out["bwd_bytes"])
    out["down_up_ratio"] = df.get("down_up_ratio", np.nan)
    if out["down_up_ratio"].isna().all():
        # Not present in this file's columns -- derive it ourselves.
        out["down_up_ratio"] = _safe_ratio(out["bwd_pkts"], out["fwd_pkts"])

    # --- TCP flag bitmask + raw counts (kept for explainability in Phase 5) ---
    out["flag_bitmask"] = _compute_flag_bitmask(df)
    for col in FLAG_BIT_ORDER:
        out[col] = df.get(col, 0)

    # --- rates (already computed by CICFlowMeter, inf-sanitized above) ---
    out["flow_byts_s"] = df.get("flow_byts_s", 0.0)
    out["flow_pkts_s"] = df.get("flow_pkts_s", 0.0)

    # --- label ---
    if "label" in df.columns:
        # A few CIC-IDS2017 label strings (e.g. "Web Attack – Brute
        # Force") were authored with a Windows-style en/em dash rather than a
        # plain hyphen. Normalize to "-" so label strings are consistent and
        # match cleanly against anything downstream (Phase 4's MITRE stage
        # mapping will key off these strings) regardless of which day's file
        # they came from.
        out["label"] = (
            df["label"].astype(str).str.strip()
            .str.replace("–", "-", regex=False)  # en dash
            .str.replace("—", "-", regex=False)  # em dash
            .str.replace(r"\s*-\s*", " - ", regex=True)
        )
        out["label_binary"] = (~out["label"].str.lower().isin(BENIGN_LABELS)).astype(np.int64)
    else:
        logger.warning("No 'Label' column found -- label/label_binary will be all-NaN.")
        out["label"] = np.nan
        out["label_binary"] = np.nan

    # Any remaining NaNs in numeric feature columns become 0.0 -- this is a
    # deliberate, documented choice (missing packet-timing stat == "no signal",
    # not "unknown"), distinct from missing identifiers/timestamps/labels which
    # are left as NaN so callers can detect and handle them explicitly.
    numeric_out_cols = [
        c for c in out.columns
        if c not in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                     "timestamp", "label", "label_binary")
    ]
    out[numeric_out_cols] = out[numeric_out_cols].fillna(0.0)

    return out


def extract_from_file(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Convenience wrapper: load + extract in one call."""
    raw = load_cicflowmeter_csv(path, nrows=nrows)
    return extract_flow_features(raw)
