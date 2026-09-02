"""
Feature normalization for the graph-sequence dataset.

Fit on the TRAIN split ONLY, then applied uniformly to val/test. Phase 2
deliberately left node/edge features unnormalized (see graph/snapshot.py's
docstring) specifically so this could happen after the train/val/test split
-- Phase 3 holds out a full day for generalization testing, so fitting a
scaler before the split (e.g. across all 8 days) would leak the held-out
day's feature distribution into training.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from graph.snapshot import GraphSnapshot


class FeatureNormalizer:
    """Per-column z-score normalization (mean 0, std 1), independently for
    node features and edge features. Columns with ~zero variance in the
    training data (e.g. a flag that never fires on any training day) get
    std=1 instead of a near-zero divisor, so they pass through as
    (x - mean) rather than being blown up numerically."""

    def __init__(self):
        self.node_mean: np.ndarray | None = None
        self.node_std: np.ndarray | None = None
        self.edge_mean: np.ndarray | None = None
        self.edge_std: np.ndarray | None = None

    def fit(self, snapshots: list[GraphSnapshot]) -> "FeatureNormalizer":
        node_rows = [s.node_features for s in snapshots if s.num_nodes > 0]
        edge_rows = [s.edge_features for s in snapshots if s.num_edges > 0]

        if not node_rows:
            raise ValueError("FeatureNormalizer.fit: no non-empty graphs to fit on")

        node_stack = np.concatenate(node_rows, axis=0)
        self.node_mean = node_stack.mean(axis=0)
        self.node_std = node_stack.std(axis=0)
        self.node_std[self.node_std < 1e-8] = 1.0

        if edge_rows:
            edge_stack = np.concatenate(edge_rows, axis=0)
            self.edge_mean = edge_stack.mean(axis=0)
            self.edge_std = edge_stack.std(axis=0)
            self.edge_std[self.edge_std < 1e-8] = 1.0
        else:
            # Degenerate but handled: a training split with literally no
            # edges anywhere. Identity-normalize rather than crash.
            n_edge_feats = snapshots[0].edge_features.shape[1]
            self.edge_mean = np.zeros(n_edge_feats)
            self.edge_std = np.ones(n_edge_feats)

        return self

    def transform_node(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        return (x - self.node_mean) / self.node_std

    def transform_edge(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        return (x - self.edge_mean) / self.edge_std

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "node_mean": self.node_mean, "node_std": self.node_std,
                "edge_mean": self.edge_mean, "edge_std": self.edge_std,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        norm = cls()
        norm.node_mean, norm.node_std = data["node_mean"], data["node_std"]
        norm.edge_mean, norm.edge_std = data["edge_mean"], data["edge_std"]
        return norm
