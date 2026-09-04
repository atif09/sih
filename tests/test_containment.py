import numpy as np
import pytest

from containment.estimator import (
    ACTION_TABLE,
    build_stage_duration_stats,
    estimate_containment_window,
)
from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from rollout.mitre import STAGE_BENIGN

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _snap(window_id, attack_present=False, attack_types=frozenset()):
    return GraphSnapshot(
        day="Day", window_id=window_id, window_start=float(window_id * 60),
        node_ids=[], node_features=np.zeros((0, NODE_DIM)), node_labels=np.zeros((0,), dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64), edge_features=np.zeros((0, EDGE_DIM)),
        edge_labels=np.zeros((0,), dtype=np.int64), edge_attack_types=[],
        graph_attack_present=attack_present, graph_attack_types=attack_types,
    )


class TestBuildStageDurationStats:
    def test_single_episode_duration(self):
        # 3 benign, 4 PortScan (Reconnaissance), 3 benign -> one 4-window episode
        snaps = (
            [_snap(i) for i in range(3)]
            + [_snap(3 + i, attack_present=True, attack_types=frozenset({"PortScan"})) for i in range(4)]
            + [_snap(7 + i) for i in range(3)]
        )
        stats = build_stage_duration_stats({"Day": snaps}, window_size_seconds=60.0)
        assert "Reconnaissance" in stats
        assert stats["Reconnaissance"].n_episodes == 1
        assert stats["Reconnaissance"].durations_minutes == [4.0]  # 4 windows * 60s / 60 = 4 min

    def test_multiple_episodes_pooled_across_days(self):
        day_a = [_snap(0), _snap(1, True, frozenset({"PortScan"})), _snap(2)]
        day_b = [_snap(0), _snap(1, True, frozenset({"PortScan"})), _snap(2, True, frozenset({"PortScan"})), _snap(3)]
        stats = build_stage_duration_stats({"A": day_a, "B": day_b}, window_size_seconds=60.0)
        assert stats["Reconnaissance"].n_episodes == 2
        assert sorted(stats["Reconnaissance"].durations_minutes) == [1.0, 2.0]

    def test_excludes_specified_days(self):
        day_a = [_snap(0, True, frozenset({"PortScan"}))]
        day_b = [_snap(0, True, frozenset({"PortScan"}))]
        stats = build_stage_duration_stats({"A": day_a, "B": day_b}, window_size_seconds=60.0, exclude_days={"A"})
        assert stats["Reconnaissance"].n_episodes == 1

    def test_benign_only_day_produces_no_stages(self):
        snaps = [_snap(i) for i in range(5)]
        stats = build_stage_duration_stats({"Day": snaps}, window_size_seconds=60.0)
        assert stats == {}

    def test_respects_window_size_seconds(self):
        snaps = [_snap(0, True, frozenset({"PortScan"})), _snap(1, True, frozenset({"PortScan"}))]
        stats = build_stage_duration_stats({"Day": snaps}, window_size_seconds=30.0)
        assert stats["Reconnaissance"].durations_minutes == [1.0]  # 2 windows * 30s / 60 = 1 min


class TestEstimateContainmentWindow:
    def test_benign_stage_needs_no_action(self):
        result = estimate_containment_window(STAGE_BENIGN, {})
        assert "No action needed" in result.recommended_action
        assert result.likely_range_minutes is None

    def test_stage_with_no_historical_data(self):
        result = estimate_containment_window("Exfiltration", {})
        assert result.n_historical_episodes == 0
        assert result.likely_range_minutes is None
        assert result.recommended_action == ACTION_TABLE["Exfiltration"]

    def test_stage_with_sufficient_data_gives_range(self):
        stats = build_stage_duration_stats(
            {"Day": (
                [_snap(0)]
                + [_snap(1 + i, True, frozenset({"PortScan"})) for i in range(5)]
                + [_snap(6)]
                + [_snap(7 + i, True, frozenset({"PortScan"})) for i in range(2)]
                + [_snap(9)]
                + [_snap(10 + i, True, frozenset({"PortScan"})) for i in range(8)]
            )},
            window_size_seconds=60.0,
        )
        result = estimate_containment_window("Reconnaissance", stats)
        assert result.has_sufficient_data  # 3 episodes >= MIN_EPISODES_FOR_CONFIDENCE
        assert result.likely_range_minutes[0] <= result.likely_range_minutes[1]
        assert result.n_historical_episodes == 3

    def test_insufficient_data_flagged_not_hidden(self):
        stats = build_stage_duration_stats(
            {"Day": [_snap(0, True, frozenset({"PortScan"}))]},
            window_size_seconds=60.0,
        )
        result = estimate_containment_window("Reconnaissance", stats)
        assert not result.has_sufficient_data
        assert "fewer than" in result.rationale

    def test_feature_qualifier_added_when_recognizable_feature_present(self):
        result = estimate_containment_window("Reconnaissance", {}, top_feature_names=["out_unique_ports", "bytes_sum"])
        assert "scan-like spread" in result.recommended_action

    def test_no_qualifier_when_no_recognizable_feature(self):
        result = estimate_containment_window("Reconnaissance", {}, top_feature_names=["duration_mean"])
        assert result.recommended_action == ACTION_TABLE["Reconnaissance"]

    def test_unknown_stage_treated_like_benign(self):
        result = estimate_containment_window("Some Unknown Stage", {})
        assert "No action needed" in result.recommended_action
