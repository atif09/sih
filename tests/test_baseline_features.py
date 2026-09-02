import numpy as np
import pytest

from baseline.features import (
    build_pairs_for_day,
    build_pairs_for_days,
    flatten_snapshot,
    flattened_feature_dim,
)
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.normalization import FeatureNormalizer
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _fit_normalizer(snapshots):
    return FeatureNormalizer().fit(snapshots)


def test_flatten_snapshot_shape():
    snapshots = [_snap("Day", i, num_flows=1) for i in range(5)]
    normalizer = _fit_normalizer(snapshots)
    vec = flatten_snapshot(snapshots[0], normalizer)
    assert vec.shape == (flattened_feature_dim(NODE_DIM, EDGE_DIM),)
    assert vec.shape == (NODE_DIM * 2 + EDGE_DIM + 3,)


def test_flatten_empty_snapshot_is_all_zero():
    snapshots = [_snap("Day", i, num_flows=1) for i in range(5)]
    normalizer = _fit_normalizer(snapshots)
    empty = _snap("Day", 99, num_flows=0, n_nodes=0)
    vec = flatten_snapshot(empty, normalizer)
    assert np.allclose(vec, 0.0)


def test_build_pairs_for_day_basic():
    snapshots = [_snap("Day", i, num_flows=1, attack_present=(i % 2 == 0)) for i in range(6)]
    normalizer = _fit_normalizer(snapshots)
    X, y = build_pairs_for_day(snapshots, normalizer)
    assert len(X) == 5  # 6 snapshots -> 5 consecutive pairs
    assert len(y) == 5
    # y[i] should equal snapshots[i+1].graph_attack_present
    expected = [int(snapshots[i + 1].graph_attack_present) for i in range(5)]
    assert list(y) == expected


def test_build_pairs_for_day_drops_zero_flow_source_windows():
    snapshots = [
        _snap("Day", 0, num_flows=0, n_nodes=0),
        _snap("Day", 1, num_flows=1, attack_present=True),
        _snap("Day", 2, num_flows=1, attack_present=False),
    ]
    normalizer = _fit_normalizer([snapshots[1], snapshots[2]])
    X, y = build_pairs_for_day(snapshots, normalizer)
    # Pair (0,1) dropped because snapshots[0].num_flows == 0; only (1,2) kept.
    assert len(X) == 1
    assert y[0] == int(snapshots[2].graph_attack_present)


def test_build_pairs_for_days_never_crosses_day_boundary():
    day_a = [_snap("A", i, num_flows=1) for i in range(4)]
    day_b = [_snap("B", i, num_flows=1) for i in range(4)]
    normalizer = _fit_normalizer(day_a + day_b)
    X, y = build_pairs_for_days({"A": day_a, "B": day_b}, normalizer)
    # 3 pairs per day (4 snapshots -> 3 pairs), 2 days -> 6 total, never 7
    # (which would happen if day_a's last snapshot paired with day_b's first).
    assert len(X) == 6


def test_build_pairs_for_day_empty_input_returns_empty_arrays():
    normalizer = _fit_normalizer([_snap("Day", 0, num_flows=1)])
    X, y = build_pairs_for_day([], normalizer)
    assert len(X) == 0
    assert len(y) == 0
