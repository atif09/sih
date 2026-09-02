"""
Phase 1 sanity check: run the flow-level extractor (and, if a --pcap is given,
the packet-level extractor + merge) against a sample and report shape, value
ranges, missing-data handling, and label distribution.

Usage:
    # Deep-dive on one CIC-IDS2017 day/attack-type file
    python scripts/sanity_check.py --csv "data/cicids2017/TrafficLabelling /Monday-WorkingHours.pcap_ISCX.csv"

    # Compact per-file summary across all 8 CIC-IDS2017 files
    python scripts/sanity_check.py --all-days

    # Include packet-level features once a PCAP is available
    python scripts/sanity_check.py --csv <csv> --pcap <pcap>

If --csv and --all-days are both omitted, falls back to a synthetic fixture
(generating it first if it doesn't exist) so this script is always runnable
without real data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.datasets import CICIDS2017_DIR, CICIDS2017_FILES, list_cicids2017_files
from features.flow_features import extract_flow_features, load_cicflowmeter_csv
from features.windowing import (
    load_config,
    merge_flow_and_packet_features,
    normalize_feature_matrix,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe(df: pd.DataFrame, name: str) -> None:
    _section(f"{name}: shape")
    print(df.shape)

    _section(f"{name}: dtypes")
    print(df.dtypes)

    _section(f"{name}: missing values (count, % of rows)")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("(none)")
    else:
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"missing_count": missing, "missing_pct": pct}))

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        _section(f"{name}: numeric value ranges")
        print(numeric_df.describe().T[["min", "max", "mean", "std"]])

    inf_cols = {
        c: int(((numeric_df[c] == float("inf")) | (numeric_df[c] == float("-inf"))).sum())
        for c in numeric_df.columns
    }
    inf_cols = {c: n for c, n in inf_cols.items() if n > 0}
    _section(f"{name}: +/-inf sanity check")
    print(inf_cols if inf_cols else "(none -- inf-prone columns were sanitized)")

    if "label" in df.columns:
        _section(f"{name}: label distribution")
        print(df["label"].value_counts(dropna=False))
    if "label_binary" in df.columns:
        _section(f"{name}: binary label distribution")
        print(df["label_binary"].value_counts(dropna=False))


def summarize_all_cicids2017_days(nrows: int | None) -> None:
    """Compact per-file pass over all 8 CIC-IDS2017 day/attack-type CSVs:
    shape, label distribution, and whether Source IP/Destination IP came
    through populated with real values (not just present-but-null)."""
    try:
        paths = list_cicids2017_files()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Reading up to {nrows or 'ALL'} rows per file from {CICIDS2017_DIR!r} ...\n")
    for meta, path in zip(CICIDS2017_FILES, paths):
        raw = load_cicflowmeter_csv(path, nrows=nrows)
        feats = extract_flow_features(raw)

        n = len(feats)
        ip_populated = feats["src_ip"].notna().mean() * 100 if n else 0.0
        label_counts = feats["label"].value_counts(dropna=False).to_dict()

        _section(f"{meta.day}: {path.name}")
        print(f"  documented attack categories : {', '.join(meta.attack_categories)}")
        print(f"  shape                         : {feats.shape}")
        print(f"  src_ip/dst_ip populated       : {ip_populated:.1f}% of rows")
        if ip_populated > 0:
            print(f"  sample src_ip -> dst_ip       : "
                  f"{feats['src_ip'].iloc[0]}:{feats['src_port'].iloc[0]} -> "
                  f"{feats['dst_ip'].iloc[0]}:{feats['dst_port'].iloc[0]}")
        print(f"  label distribution            : {label_counts}")
        numeric_df = feats.select_dtypes(include="number")
        n_inf = int(((numeric_df == float("inf")) | (numeric_df == float("-inf"))).sum().sum())
        print(f"  +/-inf remaining after sanitize: {n_inf}")

    print("\nAll-days summary complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None, help="Path to a single CICFlowMeter CSV to deep-dive on")
    parser.add_argument("--all-days", action="store_true", help="Compact summary across all 8 CIC-IDS2017 files")
    parser.add_argument("--pcap", type=str, default=None, help="Optional path to a PCAP for packet-level features")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--rows", type=int, default=None, help="Override sample_rows_for_sanity_check")
    args = parser.parse_args()

    config = load_config(args.config)
    fe_cfg = config["feature_extraction"]
    norm_cfg = config["normalization"]
    nrows = args.rows if args.rows is not None else fe_cfg["sample_rows_for_sanity_check"]

    if args.all_days:
        summarize_all_cicids2017_days(nrows)
        return

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"ERROR: {csv_path} does not exist.")
            sys.exit(1)
    else:
        csv_path = Path("data/cicids2018/_synthetic_sample.csv")
        if not csv_path.exists():
            print(f"No --csv/--all-days given and {csv_path} doesn't exist yet "
                  f"-- generating synthetic fixture...")
            from scripts.generate_synthetic_fixture import build_fixture
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            build_fixture().to_csv(csv_path, index=False)
            print(f"Wrote synthetic fixture to {csv_path}\n"
                  f"NOTE: this is SYNTHETIC data for pipeline smoke-testing only, "
                  f"not real traffic. Pass --csv or --all-days to use real data.")

    print(f"Loading {csv_path} (nrows={nrows or 'all'}) ...")
    raw = load_cicflowmeter_csv(csv_path, nrows=nrows)
    print(f"Raw columns matched to canonical schema: {len(raw.columns)}")

    flow_feats = extract_flow_features(raw)
    describe(flow_feats, "Flow-level features")

    packet_df = None
    port_scan_df = None
    if args.pcap:
        from features.packet_features import detect_port_scan_signatures, extract_packet_features
        print(f"\nExtracting packet-level features from {args.pcap} ...")
        packet_df = extract_packet_features(args.pcap)
        describe(packet_df, "Packet-level features")
        port_scan_df = detect_port_scan_signatures(packet_df, fe_cfg["window_size_seconds"])
        describe(port_scan_df, "Port-scan signatures")
    else:
        print(
            "\nNo --pcap given: packet-level columns in the merged matrix will "
            "be zero-filled with packet_features_available=False (expected "
            "until real PCAPs are provided -- see PHASE 1 notes)."
        )

    merged = merge_flow_and_packet_features(
        flow_feats, packet_df, port_scan_df,
        window_size_seconds=fe_cfg["window_size_seconds"],
        align_to_clock=fe_cfg["align_windows_to_clock"],
    )
    describe(merged, "Merged, windowed feature matrix (pre-normalization)")

    normalized, scalers = normalize_feature_matrix(
        merged, method=norm_cfg["method"], exclude=tuple(norm_cfg["exclude_from_scaling"])
    )
    _section("Normalized feature matrix: numeric range check (should be ~0-mean/unit-var or [0,1])")
    numeric_df = normalized.select_dtypes(include="number")
    print(numeric_df.describe().T[["min", "max", "mean", "std"]].head(15))
    print(f"\n{len(scalers)} column(s) scaled with method='{norm_cfg['method']}'.")

    _section("Window summary")
    print(f"Number of distinct windows: {merged['window_id'].nunique()}")
    print(f"Window size: {fe_cfg['window_size_seconds']}s")
    print(merged.groupby("window_id").size().rename("flows_per_window").describe())

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
