"""
Sequence dataset + day-based train/val/test split for world model training.

Split design (the PS requires holding out a full attack type/day to prove
generalization to unseen attack patterns)
---------------------------------------------------------------------------
- **Test = one entire day, fully excluded from train/val** (`held_out_day`,
  default "Friday-Afternoon-DDoS" -- a distinct volumetric attack type,
  behaviorally unlike anything else in the remaining 7 days). This is the
  sole source of the "generalizes to an unseen attack" claim.
- **Train/val on the remaining 7 days is split chronologically PER DAY**,
  not randomly across all windows: the first `1 - val_fraction` of each
  day's windows (by window_id, i.e. by time) go to train, the last
  `val_fraction` go to val. Consecutive windows are highly autocorrelated
  (a burst that starts in one window usually continues into the next), so a
  random shuffle across windows would let val sequences overlap in time with
  train sequences a few windows away -- leaking information and giving an
  optimistic val score. A chronological split also matches how the model
  will actually be used (trained on the past, evaluated on more recent time
  it hasn't seen).
- Sequences never straddle the train/val boundary within a day: they are
  built independently on the train-region and val-region snapshot lists, so
  no sequence's input window spans across the split point.

Sequences that contain zero flows across every one of their timesteps
(fully-quiet stretches, e.g. overnight/pre-capture gaps) are dropped -- they
carry no dynamics signal and would just dilute training with trivial
all-zero examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from features.datasets import CICIDS2017_FILES
from graph.io import load_day
from graph.snapshot import GraphSnapshot
from models.normalization import FeatureNormalizer

DEFAULT_HELD_OUT_DAY = "Friday-Afternoon-DDoS"


def load_all_days(processed_dir: str | Path) -> dict[str, list[GraphSnapshot]]:
    """Loads every day's cached GraphSnapshot list (see
    scripts/build_graph_dataset.py). Raises FileNotFoundError naming which
    day's cache is missing, rather than silently skipping it."""
    result = {}
    for meta in CICIDS2017_FILES:
        try:
            snapshots, _transitions = load_day(meta.day, processed_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing graph cache for {meta.day!r} in {processed_dir}. "
                f"Run scripts/build_graph_dataset.py first."
            ) from exc
        result[meta.day] = snapshots
    return result


def build_day_sequences(
    snapshots: list[GraphSnapshot], seq_len: int, seq_stride: int,
) -> list[list[GraphSnapshot]]:
    """Sliding-window sequences of length seq_len+1 (context of seq_len
    states + the one target state needed for the last supervised position).
    Drops sequences with zero total flows across every timestep."""
    sequences = []
    window = seq_len + 1
    for start in range(0, len(snapshots) - window + 1, seq_stride):
        chunk = snapshots[start : start + window]
        if sum(s.num_flows for s in chunk) == 0:
            continue
        sequences.append(chunk)
    return sequences


def build_attack_type_vocab(day_snapshots: dict[str, list[GraphSnapshot]], held_out_day: str) -> list[str]:
    """Fixed vocabulary for the auxiliary attack-type head, built ONLY from
    non-held-out days. Deliberately excludes the held-out day's attack
    type(s) -- see world_model.py's docstring for why: giving the classifier
    a dedicated output unit for the held-out category would undermine the
    "generalizes to an unseen attack pattern" claim that head #2 (binary
    attack-forecast) is meant to demonstrate."""
    types: set[str] = set()
    for day, snapshots in day_snapshots.items():
        if day == held_out_day:
            continue
        for s in snapshots:
            types.update(s.graph_attack_types)
    return sorted(types)


def get_train_region_snapshots(
    day_snapshots: dict[str, list[GraphSnapshot]],
    held_out_day: str,
    val_fraction: float,
) -> list[GraphSnapshot]:
    """The raw (pre-sequencing) snapshots belonging to the train region of
    each non-held-out day, pooled together. Used to fit FeatureNormalizer on
    an unbiased sample of individual windows -- fitting on `build_split`'s
    train *sequences* instead would over-weight windows that appear in
    multiple overlapping sequences (when seq_stride < seq_len)."""
    pooled: list[GraphSnapshot] = []
    for day, snapshots in day_snapshots.items():
        if day == held_out_day:
            continue
        split_point = int(len(snapshots) * (1.0 - val_fraction))
        pooled.extend(snapshots[:split_point])
    return pooled


def build_split(
    day_snapshots: dict[str, list[GraphSnapshot]],
    seq_len: int,
    seq_stride: int,
    held_out_day: str = DEFAULT_HELD_OUT_DAY,
    val_fraction: float = 0.2,
) -> tuple[list[list[GraphSnapshot]], list[list[GraphSnapshot]], list[list[GraphSnapshot]]]:
    if held_out_day not in day_snapshots:
        raise ValueError(f"held_out_day={held_out_day!r} not found among days: {list(day_snapshots)}")

    train_seqs: list[list[GraphSnapshot]] = []
    val_seqs: list[list[GraphSnapshot]] = []
    test_seqs: list[list[GraphSnapshot]] = []

    for day, snapshots in day_snapshots.items():
        if day == held_out_day:
            test_seqs.extend(build_day_sequences(snapshots, seq_len, seq_stride))
            continue
        split_point = int(len(snapshots) * (1.0 - val_fraction))
        train_region = snapshots[:split_point]
        val_region = snapshots[split_point:]
        train_seqs.extend(build_day_sequences(train_region, seq_len, seq_stride))
        val_seqs.extend(build_day_sequences(val_region, seq_len, seq_stride))

    return train_seqs, val_seqs, test_seqs


class GraphSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[GraphSnapshot]],
        normalizer: FeatureNormalizer,
        attack_type_vocab: list[str],
    ):
        self.sequences = sequences
        self.normalizer = normalizer
        self.attack_type_vocab = attack_type_vocab
        self.vocab_index = {t: i for i, t in enumerate(attack_type_vocab)}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list[GraphSnapshot]:
        return self.sequences[idx]

    def _snapshot_to_data(self, snap: GraphSnapshot) -> Data:
        x = torch.as_tensor(self.normalizer.transform_node(snap.node_features), dtype=torch.float32)
        edge_attr = torch.as_tensor(self.normalizer.transform_edge(snap.edge_features), dtype=torch.float32)
        edge_index = torch.as_tensor(snap.edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def collate_fn(self, batch: list[list[GraphSnapshot]]):
        batch_size = len(batch)
        seq_len_plus_1 = len(batch[0])
        assert all(len(seq) == seq_len_plus_1 for seq in batch), "all sequences in a batch must share length"

        flat_data = []
        for seq in batch:
            for snap in seq:
                flat_data.append(self._snapshot_to_data(snap))
        pyg_batch = Batch.from_data_list(flat_data)

        seq_len = seq_len_plus_1 - 1
        attack_label = torch.zeros(batch_size, seq_len, dtype=torch.float32)
        attack_type_target = torch.zeros(batch_size, seq_len, max(len(self.attack_type_vocab), 1), dtype=torch.float32)

        for b, seq in enumerate(batch):
            for j, snap in enumerate(seq[1:]):  # snap = target state at position j (predicts transition j -> j+1)
                attack_label[b, j] = float(snap.graph_attack_present)
                for t in snap.graph_attack_types:
                    if t in self.vocab_index:
                        attack_type_target[b, j, self.vocab_index[t]] = 1.0
                    # else: attack type not in training vocabulary (expected
                    # for the held-out day) -- silently no-op, the attack_type
                    # head is not expected to classify unseen categories.

        return {
            "pyg_batch": pyg_batch,
            "batch_size": batch_size,
            "seq_len_plus_1": seq_len_plus_1,
            "attack_label": attack_label,
            "attack_type_target": attack_type_target,
        }
