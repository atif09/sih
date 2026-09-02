import numpy as np
import pytest
import torch

from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.dataset import GraphSequenceDataset, build_day_sequences
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _make_batch(seq_len=3, batch_size=2, attack_vocab=()):
    snapshots = [_snap("Day", i, num_flows=1, attack_present=(i % 2 == 0), attack_types=frozenset({"PortScan"}) if i % 2 == 0 else frozenset()) for i in range(10)]
    sequences = build_day_sequences(snapshots, seq_len=seq_len, seq_stride=1)
    normalizer = FeatureNormalizer().fit(snapshots)
    dataset = GraphSequenceDataset(sequences, normalizer, attack_type_vocab=list(attack_vocab))
    batch = [dataset[i] for i in range(batch_size)]
    return dataset.collate_fn(batch)


@pytest.mark.parametrize("temporal_type", ["lstm", "transformer"])
def test_world_model_forward_shapes(temporal_type):
    batch = _make_batch(seq_len=3, batch_size=2, attack_vocab=["PortScan"])
    config = WorldModelConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=16,
        gnn_hidden_dim=8, gnn_num_layers=2, gnn_heads=2,
        temporal_type=temporal_type, temporal_hidden_dim=24, temporal_num_layers=2,
        attack_type_vocab=["PortScan"],
    )
    model = WorldModel(config)
    out = model(batch["pyg_batch"], batch["batch_size"], batch["seq_len_plus_1"])

    seq_len = batch["seq_len_plus_1"] - 1
    assert out["z_seq"].shape == (2, seq_len + 1, 16)
    assert out["z_hat_next"].shape == (2, seq_len, 16)
    assert out["h_seq"].shape == (2, seq_len, 24)
    assert out["attack_logits"].shape == (2, seq_len)
    assert out["attack_type_logits"].shape == (2, seq_len, 1)
    assert not torch.isnan(out["attack_logits"]).any()


def test_dynamics_target_is_detached():
    batch = _make_batch()
    config = WorldModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=8, gnn_hidden_dim=8, temporal_hidden_dim=8)
    model = WorldModel(config)
    out = model(batch["pyg_batch"], batch["batch_size"], batch["seq_len_plus_1"])
    assert out["z_target"].requires_grad is False
    assert out["z_hat_next"].requires_grad is True


def test_no_attack_type_head_when_vocab_empty():
    config = WorldModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, attack_type_vocab=[])
    model = WorldModel(config)
    assert model.attack_type_head is None


def test_training_step_reduces_loss():
    """A minimal end-to-end training loop for a handful of steps on a tiny
    synthetic batch should reduce combined loss -- a basic 'does this even
    learn' smoke test, not a convergence guarantee.

    Seeds torch explicitly: model init uses torch's global RNG, which
    otherwise depends on how many random draws earlier tests in the same
    session happened to make -- an unseeded run was observed to occasionally
    land on an unlucky initialization and fail this assertion depending on
    test execution order.
    """
    torch.manual_seed(0)
    batch = _make_batch(seq_len=4, batch_size=4, attack_vocab=["PortScan"])
    config = WorldModelConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=16, gnn_hidden_dim=16,
        temporal_hidden_dim=16, attack_type_vocab=["PortScan"], dropout=0.0,
    )
    model = WorldModel(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = model(batch["pyg_batch"], batch["batch_size"], batch["seq_len_plus_1"])
        loss = mse(out["z_hat_next"], out["z_target"]) + bce(out["attack_logits"], batch["attack_label"])
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Compare average of the first few vs. last few steps rather than a
    # single before/after pair, so one noisy step doesn't flip the result.
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
