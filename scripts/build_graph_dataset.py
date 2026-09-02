"""
Phase 2 build script: runs Phase 1 extraction + windowing, then Phase 2 graph
construction + transition extraction, for each CIC-IDS2017 day/attack-type
file, and persists the result.

Usage:
    # All 8 days, full files (no row cap) -- the real run
    python scripts/build_graph_dataset.py

    # Quick smoke test on a row-capped sample
    python scripts/build_graph_dataset.py --rows 50000

    # One specific day
    python scripts/build_graph_dataset.py --day Monday
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.datasets import CICIDS2017_FILES, cicids2017_path
from features.flow_features import extract_flow_features, load_cicflowmeter_csv
from features.windowing import load_config, merge_flow_and_packet_features
from graph.io import save_day, summarize_day
from graph.snapshot import build_snapshots_for_day
from graph.transitions import build_transitions


def build_one_day(csv_path: Path, day: str, window_size_seconds: float, align_to_clock: bool, nrows: int | None):
    raw = load_cicflowmeter_csv(csv_path, nrows=nrows)
    flow_feats = extract_flow_features(raw)
    merged = merge_flow_and_packet_features(
        flow_feats, packet_df=None, port_scan_df=None,
        window_size_seconds=window_size_seconds, align_to_clock=align_to_clock,
    )
    snapshots = build_snapshots_for_day(merged, day=day, window_size_seconds=window_size_seconds)
    transitions = build_transitions(snapshots)
    return snapshots, transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--rows", type=int, default=None, help="Row cap per file, for smoke-testing (default: full file)")
    parser.add_argument("--day", type=str, default=None, help="Only build this day (substring match, e.g. 'Monday')")
    parser.add_argument("--out-dir", type=str, default="data/processed/cicids2017")
    args = parser.parse_args()

    config = load_config(args.config)
    fe_cfg = config["feature_extraction"]

    files = CICIDS2017_FILES
    if args.day:
        files = [f for f in files if args.day.lower() in f.day.lower()]
        if not files:
            print(f"ERROR: no CIC-IDS2017 file matches day filter {args.day!r}")
            sys.exit(1)

    out_dir = Path(args.out_dir)
    overall_start = time.time()

    for meta in files:
        path = cicids2017_path(meta)
        if not path.exists():
            print(f"SKIP {meta.day}: {path} not found")
            continue

        print(f"\n{'=' * 70}\n{meta.day}: {path.name}\n{'=' * 70}")
        t0 = time.time()
        snapshots, transitions = build_one_day(
            path, meta.day,
            window_size_seconds=fe_cfg["window_size_seconds"],
            align_to_clock=fe_cfg["align_windows_to_clock"],
            nrows=args.rows,
        )
        elapsed = time.time() - t0

        pkl_path = save_day(meta.day, snapshots, transitions, out_dir)
        summary = summarize_day(meta.day, snapshots, transitions)

        print(f"  windows: {summary['num_windows']} "
              f"({summary['num_empty_windows']} empty) | "
              f"transitions: {summary['num_transitions']} | "
              f"flows: {summary['total_flows']} | "
              f"elapsed: {elapsed:.1f}s")
        print(f"  avg nodes/window: {summary['avg_nodes_per_nonempty_window']:.1f} | "
              f"avg edges/window: {summary['avg_edges_per_nonempty_window']:.1f} | "
              f"max nodes: {summary['max_nodes_in_a_window']} | "
              f"max edges: {summary['max_edges_in_a_window']}")
        print(f"  transition types: {summary['transition_type_counts']}")
        print(f"  saved -> {pkl_path}")

    print(f"\nTotal elapsed: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
