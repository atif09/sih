import numpy as np
import pytest

from futures.parallel_futures import FutureOutcome, ParallelFuturesResult
from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from momentum.score import compute_momentum, _newly_affected_host_ratio, _slope, _stage_progression_rate
from rollout.engine import RolloutResult

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _snap(window_id, node_ids):
    n = len(node_ids)
    return GraphSnapshot(
        day="Day", window_id=window_id, window_start=float(window_id * 60),
        node_ids=node_ids, node_features=np.zeros((n, NODE_DIM)),
        node_labels=np.zeros(n, dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64), edge_features=np.zeros((0, EDGE_DIM)),
        edge_labels=np.zeros((0,), dtype=np.int64), edge_attack_types=[],
        graph_attack_present=False,
    )


def _rollout(probs):
    return RolloutResult(
        day="Day", context_end_window_id=0, k=len(probs),
        infiltration_probability=probs,
        attack_type_probs=[{} for _ in probs],
        stage_predictions=[],
    )


def _futures(top_probability: float, n_outcomes: int = 2, n_samples: int = 40):
    """A minimal ParallelFuturesResult stub -- top outcome at
    `top_probability`, remaining mass split evenly across the rest."""
    remaining = 1.0 - top_probability
    outcomes = [FutureOutcome(stage="Reconnaissance", probability=top_probability, n_samples=int(top_probability * n_samples), mean_infiltration_probability=0.5)]
    for i in range(n_outcomes - 1):
        p = remaining / (n_outcomes - 1)
        outcomes.append(FutureOutcome(stage=f"Stage{i}", probability=p, n_samples=int(p * n_samples), mean_infiltration_probability=0.5))
    return ParallelFuturesResult(n_samples=n_samples, outcomes=outcomes, deterministic_stage="Reconnaissance")


class TestSlope:
    def test_slope_of_rising_series(self):
        assert _slope([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.1, abs=1e-9)

    def test_slope_of_flat_series(self):
        assert _slope([0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)

    def test_slope_needs_two_points(self):
        assert _slope([0.5]) == 0.0
        assert _slope([]) == 0.0


class TestNewlyAffectedHosts:
    def test_all_new_hosts_gives_ratio_one(self):
        snaps = [_snap(0, ["a", "b"]), _snap(1, ["c", "d"])]
        assert _newly_affected_host_ratio(snaps) == pytest.approx(1.0)

    def test_no_new_hosts_gives_ratio_zero(self):
        snaps = [_snap(0, ["a", "b"]), _snap(1, ["a", "b"])]
        assert _newly_affected_host_ratio(snaps) == pytest.approx(0.0)

    def test_partial_overlap(self):
        snaps = [_snap(0, ["a", "b"]), _snap(1, ["a", "c"])]
        assert _newly_affected_host_ratio(snaps) == pytest.approx(0.5)

    def test_empty_latest_snapshot_is_zero_not_divide_error(self):
        snaps = [_snap(0, ["a"]), _snap(1, [])]
        assert _newly_affected_host_ratio(snaps) == 0.0

    def test_insufficient_history_is_zero(self):
        assert _newly_affected_host_ratio([_snap(0, ["a"])]) == 0.0


class TestStageProgressionRate:
    def test_constant_stage_is_zero(self):
        assert _stage_progression_rate(["Benign", "Benign", "Benign"]) == 0.0

    def test_every_window_changes(self):
        assert _stage_progression_rate(["A", "B", "C"]) == pytest.approx(1.0)

    def test_half_changes(self):
        assert _stage_progression_rate(["A", "A", "B", "B"]) == pytest.approx(1 / 3)


class TestComputeMomentum:
    def test_rising_probability_gives_high_momentum(self):
        snaps = [_snap(i, [f"h{i}"]) for i in range(6)]  # every window has a brand-new host
        result = compute_momentum(
            recent_probabilities=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            recent_stages=["Benign", "Benign", "Reconnaissance", "Reconnaissance", "Lateral Movement", "Exfiltration"],
            recent_snapshots=snaps,
        )
        assert result.score > 60
        assert 0 <= result.score <= 100

    def test_flat_benign_gives_low_momentum(self):
        snaps = [_snap(i, ["a", "b"]) for i in range(6)]  # same hosts every window
        result = compute_momentum(
            recent_probabilities=[0.4, 0.4, 0.41, 0.4, 0.39, 0.4],
            recent_stages=["Benign"] * 6,
            recent_snapshots=snaps,
        )
        assert result.score < 20

    def test_score_bounded_0_100(self):
        snaps = [_snap(i, [f"h{i}", f"h{i}b"]) for i in range(6)]
        result = compute_momentum(
            recent_probabilities=[0.1, 0.99],
            recent_stages=["Benign", "Exfiltration"],
            recent_snapshots=snaps,
            rollout_result=_rollout([0.99, 0.99, 0.99]),
        )
        assert 0.0 <= result.score <= 100.0

    def test_missing_rollout_excludes_forecast_trend_not_zero_penalizes(self):
        snaps = [_snap(i, ["a"]) for i in range(4)]
        with_rollout = compute_momentum([0.5, 0.6], ["Benign", "Benign"], snaps, rollout_result=_rollout([0.9, 0.95]))
        without_rollout = compute_momentum([0.5, 0.6], ["Benign", "Benign"], snaps, rollout_result=None)
        assert not without_rollout.signals.attack_path_expansion_available
        # Excluding a strongly-contributing signal via reweighting must not
        # silently zero it out and tank the score -- it should reweight the
        # remaining signals, not just drop mass.
        assert without_rollout.score > 0

    def test_explanation_names_top_driver(self):
        snaps = [_snap(i, [f"h{i}"]) for i in range(4)]  # all new hosts every window -> host ratio dominates
        result = compute_momentum(
            recent_probabilities=[0.5, 0.5, 0.5, 0.5],  # flat, no probability signal
            recent_stages=["Benign"] * 4,  # flat, no stage signal
            recent_snapshots=snaps,
        )
        assert result.top_driver == "newly_affected_host_ratio"
        assert "newly active hosts" in result.explanation

    def test_insufficient_history_does_not_crash(self):
        result = compute_momentum([], [], [])
        assert result.score == 0.0


class TestPathExpansionSignal:
    def test_confident_single_outcome_gives_zero_path_expansion(self):
        snaps = [_snap(i, ["a"]) for i in range(4)]
        result = compute_momentum(
            [0.5, 0.5], ["Benign", "Benign"], snaps,
            futures_result=_futures(top_probability=1.0),
        )
        assert result.signals.path_expansion == pytest.approx(0.0)
        assert result.signals.attack_path_expansion_available is True

    def test_spread_out_outcomes_give_high_path_expansion(self):
        snaps = [_snap(i, ["a"]) for i in range(4)]
        result = compute_momentum(
            [0.5, 0.5], ["Benign", "Benign"], snaps,
            futures_result=_futures(top_probability=0.3, n_outcomes=4),
        )
        assert result.signals.path_expansion == pytest.approx(0.7)

    def test_missing_futures_result_excludes_path_expansion_not_zero_penalizes(self):
        snaps = [_snap(i, ["a"]) for i in range(4)]
        with_futures = compute_momentum([0.5, 0.5], ["Benign", "Benign"], snaps, futures_result=_futures(0.2, n_outcomes=5))
        without_futures = compute_momentum([0.5, 0.5], ["Benign", "Benign"], snaps, futures_result=None)
        assert without_futures.signals.attack_path_expansion_available is False
        assert without_futures.signals.path_expansion == 0.0
        # A highly-spread futures ensemble should score higher momentum than
        # the same inputs with that signal simply absent, not identical --
        # confirms path_expansion is actually wired into the weighted sum.
        assert with_futures.score > without_futures.score

    def test_path_expansion_can_be_top_driver(self):
        snaps = [_snap(i, ["a", "b"]) for i in range(4)]  # flat host/prob/stage signals
        result = compute_momentum(
            [0.5, 0.5, 0.5, 0.5], ["Benign"] * 4, snaps,
            futures_result=_futures(top_probability=0.1, n_outcomes=6),
        )
        assert result.top_driver == "path_expansion"
        assert "plausible futures" in result.explanation

    def test_zero_outcomes_handled_gracefully(self):
        # Defensive: an empty outcomes list (shouldn't happen in practice,
        # MC-Dropout always samples at least once) must not crash.
        snaps = [_snap(i, ["a"]) for i in range(4)]
        empty_futures = ParallelFuturesResult(n_samples=0, outcomes=[], deterministic_stage="Benign / No Stage")
        result = compute_momentum([0.5, 0.5], ["Benign", "Benign"], snaps, futures_result=empty_futures)
        assert result.signals.path_expansion == 0.0
