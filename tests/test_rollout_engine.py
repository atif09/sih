import pytest
import torch

from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.dataset import build_day_sequences
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from rollout.engine import run_rollout
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _make_context_and_model(temporal_type: str, attack_vocab=("PortScan",)):
    snapshots = [_snap("Day", i, num_flows=1) for i in range(8)]
    normalizer = FeatureNormalizer().fit(snapshots)
    context = snapshots[:5]

    config = WorldModelConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=16, gnn_hidden_dim=8,
        gnn_num_layers=2, gnn_heads=2, temporal_type=temporal_type, temporal_hidden_dim=16,
        temporal_num_layers=1, attack_type_vocab=list(attack_vocab),
    )
    model = WorldModel(config)
    return model, context, normalizer


@pytest.mark.parametrize("temporal_type", ["lstm", "transformer"])
def test_rollout_produces_k_steps(temporal_type):
    model, context, normalizer = _make_context_and_model(temporal_type)
    result = run_rollout(model, context, normalizer, k=5, device=torch.device("cpu"))

    assert result.k == 5
    assert len(result.infiltration_probability) == 5
    assert len(result.attack_type_probs) == 5
    assert len(result.stage_predictions) == 5
    assert all(0.0 <= p <= 1.0 for p in result.infiltration_probability)
    assert result.day == "Day"
    assert result.context_end_window_id == context[-1].window_id


def test_rollout_attack_type_probs_populated_when_vocab_present():
    model, context, normalizer = _make_context_and_model("lstm", attack_vocab=("PortScan", "Bot"))
    result = run_rollout(model, context, normalizer, k=3, device=torch.device("cpu"))
    assert all(set(p.keys()) == {"PortScan", "Bot"} for p in result.attack_type_probs)


def test_rollout_no_attack_type_head_gives_empty_dicts():
    model, context, normalizer = _make_context_and_model("lstm", attack_vocab=())
    result = run_rollout(model, context, normalizer, k=3, device=torch.device("cpu"))
    assert all(p == {} for p in result.attack_type_probs)


def test_rollout_rejects_mixed_day_context():
    model, context, normalizer = _make_context_and_model("lstm")
    mixed = context[:2] + [_snap("OtherDay", 0, num_flows=1)]
    with pytest.raises(ValueError, match="same day"):
        run_rollout(model, mixed, normalizer, k=2, device=torch.device("cpu"))


def test_rollout_driving_context_flows_present_when_edges_exist():
    model, context, normalizer = _make_context_and_model("lstm")
    result = run_rollout(model, context, normalizer, k=2, device=torch.device("cpu"))
    # The synthetic _snap helper gives 2-node/1-edge snapshots by default.
    assert len(result.driving_context_flows) >= 1


def test_lstm_and_transformer_rollout_are_both_deterministic_in_eval_mode():
    for temporal_type in ["lstm", "transformer"]:
        model, context, normalizer = _make_context_and_model(temporal_type)
        model.eval()
        r1 = run_rollout(model, context, normalizer, k=4, device=torch.device("cpu"))
        r2 = run_rollout(model, context, normalizer, k=4, device=torch.device("cpu"))
        assert r1.infiltration_probability == pytest.approx(r2.infiltration_probability)
