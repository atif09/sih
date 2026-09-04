import pytest
import torch

from counterfactual.simulator import (
    InterventionAction,
    apply_action,
    simulate_intervention,
)
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _make_context_and_model(attack_vocab=("PortScan",)):
    # _snap's fixed node_ids ("h0", "h1") and one h0->h1 edge per window make
    # it a convenient stand-in for a real host pair here.
    snapshots = [_snap("Day", i, num_flows=1) for i in range(8)]
    normalizer = FeatureNormalizer().fit(snapshots)
    context = snapshots[:5]

    config = WorldModelConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=16, gnn_hidden_dim=8,
        gnn_num_layers=2, gnn_heads=2, temporal_type="lstm", temporal_hidden_dim=16,
        temporal_num_layers=1, attack_type_vocab=list(attack_vocab),
    )
    model = WorldModel(config)
    model.eval()
    return model, context, normalizer


class TestApplyAction:
    def test_isolate_host_applies_to_every_context_snapshot(self):
        _, context, _ = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="h0")
        modified = apply_action(context, action)
        assert len(modified) == len(context)
        assert all("h0" not in s.node_ids for s in modified)
        assert all(s.num_edges == 0 for s in modified)  # h0's only edge is gone

    def test_block_flow_applies_to_every_context_snapshot(self):
        _, context, _ = _make_context_and_model()
        action = InterventionAction(kind="block_flow", src_ip="h0", dst_ip="h1")
        modified = apply_action(context, action)
        assert all(s.num_edges == 0 for s in modified)
        assert all("h0" in s.node_ids and "h1" in s.node_ids for s in modified)

    def test_isolate_host_missing_host_ip_raises(self):
        _, context, _ = _make_context_and_model()
        with pytest.raises(ValueError, match="host_ip"):
            apply_action(context, InterventionAction(kind="isolate_host"))

    def test_block_flow_missing_ips_raises(self):
        _, context, _ = _make_context_and_model()
        with pytest.raises(ValueError, match="src_ip"):
            apply_action(context, InterventionAction(kind="block_flow"))

    def test_unknown_kind_raises(self):
        _, context, _ = _make_context_and_model()
        with pytest.raises(ValueError, match="Unknown action kind"):
            apply_action(context, InterventionAction(kind="nonsense"))


class TestInterventionActionLabel:
    def test_isolate_host_label(self):
        assert InterventionAction(kind="isolate_host", host_ip="10.0.0.1").label == "Isolate host 10.0.0.1"

    def test_block_flow_bidirectional_label(self):
        action = InterventionAction(kind="block_flow", src_ip="a", dst_ip="b", bidirectional=True)
        assert "<->" in action.label

    def test_block_flow_unidirectional_label(self):
        action = InterventionAction(kind="block_flow", src_ip="a", dst_ip="b", bidirectional=False)
        assert "->" in action.label and "<->" not in action.label


class TestSimulateIntervention:
    def test_returns_matching_length_probability_series(self):
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="h0")
        result = simulate_intervention(model, context, normalizer, action, k=3, device=torch.device("cpu"))
        assert len(result.original_probability) == 3
        assert len(result.modified_probability) == 3
        assert len(result.probability_delta) == 3

    def test_delta_is_modified_minus_original(self):
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="h0")
        result = simulate_intervention(model, context, normalizer, action, k=3, device=torch.device("cpu"))
        for delta, orig, mod in zip(result.probability_delta, result.original_probability, result.modified_probability):
            assert delta == pytest.approx(mod - orig)

    def test_isolating_absent_host_gives_zero_delta(self):
        # Isolating a host that isn't in the context at all is a no-op graph
        # mutation, so the modified rollout must be byte-identical to the
        # original -- delta should be exactly zero, not just "close".
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="not-present")
        result = simulate_intervention(model, context, normalizer, action, k=3, device=torch.device("cpu"))
        assert result.probability_delta == pytest.approx([0.0, 0.0, 0.0])
        assert result.original_stage == result.modified_stage

    def test_side_effects_populated_for_isolate_host_with_benign_traffic(self):
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="h0")
        result = simulate_intervention(model, context, normalizer, action, k=2, device=torch.device("cpu"))
        assert len(result.side_effects) == 1
        assert "h0" in result.side_effects[0]

    def test_side_effects_for_absent_host_notes_no_effect(self):
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="isolate_host", host_ip="ghost")
        result = simulate_intervention(model, context, normalizer, action, k=2, device=torch.device("cpu"))
        assert "not active" in result.side_effects[0]

    def test_side_effects_for_block_flow_notes_pivot_risk(self):
        model, context, normalizer = _make_context_and_model()
        action = InterventionAction(kind="block_flow", src_ip="h0", dst_ip="h1")
        result = simulate_intervention(model, context, normalizer, action, k=2, device=torch.device("cpu"))
        assert len(result.side_effects) == 1
        assert "pivot" in result.side_effects[0]

    def test_does_not_mutate_input_context(self):
        model, context, normalizer = _make_context_and_model()
        original_node_ids = [list(s.node_ids) for s in context]
        action = InterventionAction(kind="isolate_host", host_ip="h0")
        simulate_intervention(model, context, normalizer, action, k=2, device=torch.device("cpu"))
        assert [list(s.node_ids) for s in context] == original_node_ids
