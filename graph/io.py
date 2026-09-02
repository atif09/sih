"""
Persistence for the Phase 2 graph-sequence dataset.

These pickle files are a regenerable CACHE, not source data: they're fully
reproducible from the raw CIC-IDS2017 CSVs + this codebase (see
scripts/build_graph_dataset.py), so they're git-ignored (see .gitignore)
rather than committed. A small JSON summary is written alongside each pickle
for quick human inspection without unpickling.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from graph.snapshot import GraphSnapshot
from graph.transitions import Transition


def save_day(day: str, snapshots: list[GraphSnapshot], transitions: list[Transition], out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = day.lower().replace(" ", "_")

    pkl_path = out_dir / f"{slug}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"day": day, "snapshots": snapshots, "transitions": transitions}, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = summarize_day(day, snapshots, transitions)
    with open(out_dir / f"{slug}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return pkl_path


def load_day(day: str, out_dir: Path | str) -> tuple[list[GraphSnapshot], list[Transition]]:
    out_dir = Path(out_dir)
    slug = day.lower().replace(" ", "_")
    with open(out_dir / f"{slug}.pkl", "rb") as f:
        data = pickle.load(f)
    return data["snapshots"], data["transitions"]


def summarize_day(day: str, snapshots: list[GraphSnapshot], transitions: list[Transition]) -> dict:
    non_empty = [s for s in snapshots if s.num_nodes > 0]
    transition_type_counts = Counter(t.transition_type for t in transitions)
    attack_label_counts = Counter(t.attack_label_next for t in transitions)
    all_attack_types: Counter = Counter()
    for s in snapshots:
        all_attack_types.update(s.graph_attack_types)

    return {
        "day": day,
        "num_windows": len(snapshots),
        "num_empty_windows": len(snapshots) - len(non_empty),
        "num_transitions": len(transitions),
        "avg_nodes_per_nonempty_window": (
            sum(s.num_nodes for s in non_empty) / len(non_empty) if non_empty else 0.0
        ),
        "avg_edges_per_nonempty_window": (
            sum(s.num_edges for s in non_empty) / len(non_empty) if non_empty else 0.0
        ),
        "max_nodes_in_a_window": max((s.num_nodes for s in snapshots), default=0),
        "max_edges_in_a_window": max((s.num_edges for s in snapshots), default=0),
        "total_flows": sum(s.num_flows for s in snapshots),
        "windows_with_attack_present": sum(1 for s in snapshots if s.graph_attack_present),
        "transition_type_counts": dict(transition_type_counts),
        "attack_label_next_counts": {str(k): v for k, v in attack_label_counts.items()},
        "attack_types_observed": dict(all_attack_types),
    }
