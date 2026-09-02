import numpy as np
import pytest

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES
from models.dataset import (
    build_attack_type_vocab,
    build_day_sequences,
    build_split,
)
from models.normalization import FeatureNormalizer

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _snap(day, window_id, num_flows=1, attack_present=False, attack_types=frozenset(), n_nodes=2):
    if n_nodes > 0:
        node_features = np.random.RandomState(window_id).randn(n_nodes, NODE_DIM)
        edge_index = np.array([[0], [1]]) if n_nodes >= 2 else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.random.RandomState(window_id).randn(1 if n_nodes >= 2 else 0, EDGE_DIM)
        edge_labels = np.array([int(attack_present)]) if n_nodes >= 2 else np.zeros((0,), dtype=np.int64)
    else:
        node_features = np.zeros((0, NODE_DIM))
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_DIM))
        edge_labels = np.zeros((0,), dtype=np.int64)

    return GraphSnapshot(
        day=day, window_id=window_id, window_start=float(window_id * 60),
        node_ids=[f"h{i}" for i in range(n_nodes)], node_features=node_features,
        node_labels=np.zeros(n_nodes, dtype=np.int64),
        edge_index=edge_index, edge_features=edge_features, edge_labels=edge_labels,
        edge_attack_types=[attack_types] if n_nodes >= 2 else [],
        graph_attack_present=attack_present, graph_attack_types=attack_types,
        num_flows=num_flows,
    )


def test_build_day_sequences_basic():
    snapshots = [_snap("Day", i) for i in range(10)]
    seqs = build_day_sequences(snapshots, seq_len=3, seq_stride=2)
    # window = seq_len+1 = 4; starts at 0,2,4,6 (6+4=10 fits); next start 8+4=12 > 10 stops
    assert len(seqs) == 4
    assert all(len(s) == 4 for s in seqs)
    assert [s.window_id for s in seqs[0]] == [0, 1, 2, 3]
    assert [s.window_id for s in seqs[1]] == [2, 3, 4, 5]


def test_build_day_sequences_drops_fully_empty_chunks():
    quiet = [_snap("Day", i, num_flows=0, n_nodes=0) for i in range(4)]
    busy = [_snap("Day", i, num_flows=5) for i in range(4, 8)]
    seqs = build_day_sequences(quiet + busy, seq_len=3, seq_stride=4)
    assert len(seqs) == 1  # only the busy chunk survives
    assert [s.window_id for s in seqs[0]] == [4, 5, 6, 7]


def test_build_split_holds_out_full_day_and_splits_chronologically():
    day_snapshots = {
        "A": [_snap("A", i, num_flows=1) for i in range(10)],
        "B": [_snap("B", i, num_flows=1) for i in range(10)],
        "HeldOut": [_snap("HeldOut", i, num_flows=1, attack_present=True, attack_types=frozenset({"DDoS"})) for i in range(10)],
    }
    train, val, test = build_split(day_snapshots, seq_len=2, seq_stride=1, held_out_day="HeldOut", val_fraction=0.3)

    # Test set is exclusively from the held-out day.
    assert len(test) > 0
    for seq in test:
        assert all(s.day == "HeldOut" for s in seq)

    # Train/val sequences never come from the held-out day.
    for seq in train + val:
        assert all(s.day != "HeldOut" for s in seq)

    # Chronological split: no train sequence's window_ids overlap into the
    # region reserved for val, for a given day. split_point = int(10*0.7)=7.
    for seq in train:
        assert max(s.window_id for s in seq) < 7
    for seq in val:
        assert min(s.window_id for s in seq) >= 7


def test_attack_type_vocab_excludes_held_out_day():
    day_snapshots = {
        "A": [_snap("A", 0, attack_present=True, attack_types=frozenset({"PortScan"}))],
        "HeldOut": [_snap("HeldOut", 0, attack_present=True, attack_types=frozenset({"DDoS"}))],
    }
    vocab = build_attack_type_vocab(day_snapshots, held_out_day="HeldOut")
    assert vocab == ["PortScan"]
    assert "DDoS" not in vocab


def test_collate_fn_produces_expected_shapes():
    from models.dataset import GraphSequenceDataset

    snapshots = [_snap("Day", i, num_flows=1, attack_present=(i % 3 == 0)) for i in range(6)]
    sequences = build_day_sequences(snapshots, seq_len=3, seq_stride=1)
    assert len(sequences) >= 1

    normalizer = FeatureNormalizer().fit(snapshots)
    dataset = GraphSequenceDataset(sequences, normalizer, attack_type_vocab=[])

    batch = [dataset[0], dataset[1]] if len(dataset) > 1 else [dataset[0], dataset[0]]
    out = dataset.collate_fn(batch)

    assert out["batch_size"] == 2
    assert out["seq_len_plus_1"] == 4
    assert out["attack_label"].shape == (2, 3)
    assert out["pyg_batch"].num_graphs == 2 * 4
