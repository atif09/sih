import numpy as np
import pytest

from graph.snapshot import GraphSnapshot
from graph.transitions import (
    TRANSITION_ATTACK_TO_ATTACK,
    TRANSITION_ATTACK_TO_BENIGN,
    TRANSITION_BENIGN_TO_ATTACK,
    TRANSITION_BENIGN_TO_BENIGN,
    build_transitions,
)


def _snap(day, window_id, attack_present, attack_types=frozenset()):
    return GraphSnapshot(
        day=day, window_id=window_id, window_start=float(window_id * 60),
        node_ids=[], node_features=np.zeros((0, 1)), node_labels=np.zeros((0,), dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64), edge_features=np.zeros((0, 1)),
        edge_labels=np.zeros((0,), dtype=np.int64), edge_attack_types=[],
        graph_attack_present=attack_present, graph_attack_types=attack_types,
    )


def test_transition_types_cover_all_four_combinations():
    snapshots = [
        _snap("Day1", 0, False),
        _snap("Day1", 1, False),
        _snap("Day1", 2, True, frozenset({"DDoS"})),
        _snap("Day1", 3, True, frozenset({"DDoS"})),
        _snap("Day1", 4, False),
    ]
    transitions = build_transitions(snapshots)

    assert len(transitions) == 4
    assert transitions[0].transition_type == TRANSITION_BENIGN_TO_BENIGN
    assert transitions[1].transition_type == TRANSITION_BENIGN_TO_ATTACK
    assert transitions[2].transition_type == TRANSITION_ATTACK_TO_ATTACK
    assert transitions[3].transition_type == TRANSITION_ATTACK_TO_BENIGN

    assert transitions[1].attack_label_next == 1
    assert transitions[1].attack_types_next == frozenset({"DDoS"})
    assert transitions[0].attack_label_next == 0


def test_transitions_only_within_same_day():
    day1 = [_snap("Day1", 0, False), _snap("Day1", 1, False)]
    day2 = [_snap("Day2", 0, False), _snap("Day2", 1, True)]

    t1 = build_transitions(day1)
    t2 = build_transitions(day2)
    assert len(t1) == 1
    assert len(t2) == 1
    # Concatenating snapshot lists across days and building transitions in
    # one pass must NOT create a cross-day transition (window_id wouldn't be
    # consecutive here, but this documents the intended usage: call
    # build_transitions() once per day, not once globally).
    combined = build_transitions(day1 + day2)
    assert len(combined) == 2  # not 3 -- no Day1[-1] -> Day2[0] transition


def test_non_consecutive_window_ids_are_skipped():
    snapshots = [_snap("Day1", 0, False), _snap("Day1", 5, True)]
    transitions = build_transitions(snapshots)
    assert transitions == []
