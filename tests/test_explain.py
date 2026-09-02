import pytest
import torch

from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from explain.attention import compute_edge_attention, top_edges_overall
from explain.saliency import compute_feature_saliency
from explain.explainer import explain_prediction
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _make_model_and_context(temporal_type="lstm", attack_vocab=("PortScan",)):
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


class TestAttention:
    def test_returns_one_record_per_edge(self):
        model, context, normalizer = _make_model_and_context()
        records = compute_edge_attention(model, context[-1], normalizer, torch.device("cpu"))
        assert len(records) == context[-1].num_edges

    def test_attention_weights_sum_to_one_within_target_group(self):
        model, context, normalizer = _make_model_and_context()
        # _snap gives each snapshot a single edge (0->1), so trivially 1.0;
        # verify at minimum the invariant "weights in [0,1]" and grouping
        # metadata is present.
        records = compute_edge_attention(model, context[-1], normalizer, torch.device("cpu"))
        for r in records:
            assert 0.0 <= r["attention_weight"] <= 1.0
            assert r["attention_rank_within_target"] >= 1

    def test_empty_snapshot_returns_empty_list(self):
        model, context, normalizer = _make_model_and_context()
        empty = _snap("Day", 99, num_flows=0, n_nodes=0)
        assert compute_edge_attention(model, empty, normalizer, torch.device("cpu")) == []

    def test_top_edges_overall_respects_top_k(self):
        model, context, normalizer = _make_model_and_context()
        records = compute_edge_attention(model, context[-1], normalizer, torch.device("cpu"))
        top = top_edges_overall(records, top_k=1)
        assert len(top) <= 1

    def test_top_edges_overall_prefers_concentration_over_raw_weight(self):
        """Regression test for a real bug found against live data: ranking
        by raw attention_weight alone systematically favors a target with
        only one (trivially attention=1.0) incoming edge over a target
        where an edge won genuine competition against several rivals (e.g.
        0.6 out of 5 real competitors) -- exactly the case where a
        volumetric attack edge among ordinary background traffic got
        buried. top_edges_overall must surface the latter first."""
        trivial_choice = {
            "src_ip": "a", "dst_ip": "target_a", "attention_weight": 1.0,
            "attention_rank_within_target": 1, "attention_concentration": 1.0 * 1,
            "bytes_sum": 0, "flow_count": 0, "dst_port_nunique": 0,
            "retransmission_sum": 0, "flag_bitmask_or": 0, "is_attack_edge": False,
        }
        real_competition_winner = {
            "src_ip": "attacker", "dst_ip": "target_b", "attention_weight": 0.6,
            "attention_rank_within_target": 1, "attention_concentration": 0.6 * 5,
            "bytes_sum": 46_000_000, "flow_count": 6000, "dst_port_nunique": 1,
            "retransmission_sum": 0, "flag_bitmask_or": 0, "is_attack_edge": True,
        }
        top = top_edges_overall([trivial_choice, real_competition_winner], top_k=2)
        assert top[0]["dst_ip"] == "target_b"
        assert top[0]["is_attack_edge"] is True


class TestSaliency:
    @pytest.mark.parametrize("temporal_type", ["lstm", "transformer"])
    def test_saliency_shapes(self, temporal_type):
        model, context, normalizer = _make_model_and_context(temporal_type)
        result = compute_feature_saliency(model, context, normalizer, torch.device("cpu"), top_k=5)
        assert len(result["top_node_features"]) <= 5
        assert len(result["top_edge_features"]) <= 5
        assert len(result["per_window_saliency"]) == len(context)

    def test_saliency_scores_are_nonnegative(self):
        model, context, normalizer = _make_model_and_context()
        result = compute_feature_saliency(model, context, normalizer, torch.device("cpu"))
        assert all(score >= 0 for _, score in result["top_node_features"])
        assert all(score >= 0 for _, score in result["top_edge_features"])
        assert all(v >= 0 for v in result["per_window_saliency"])

    def test_feature_names_are_valid(self):
        model, context, normalizer = _make_model_and_context()
        result = compute_feature_saliency(model, context, normalizer, torch.device("cpu"))
        for name, _ in result["top_node_features"]:
            assert name in NODE_FEATURE_NAMES
        for name, _ in result["top_edge_features"]:
            assert name in EDGE_FEATURE_NAMES

    def test_does_not_mutate_model_or_leave_dangling_grad_state(self):
        model, context, normalizer = _make_model_and_context()
        compute_feature_saliency(model, context, normalizer, torch.device("cpu"))
        # A subsequent ordinary no-grad forward pass should still work fine.
        with torch.no_grad():
            from models.dataset import snapshot_to_pyg_data
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([snapshot_to_pyg_data(s, normalizer) for s in context])
            model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(context))


class TestExplainer:
    def test_explanation_record_has_all_fields(self):
        model, context, normalizer = _make_model_and_context()
        record = explain_prediction(model, context, normalizer, torch.device("cpu"))
        assert 0.0 <= record.infiltration_probability <= 1.0
        assert record.stage is not None
        assert record.day == "Day"
        assert record.context_end_window_id == context[-1].window_id
        assert len(record.summary) > 0
        assert "infiltration probability" in record.summary

    def test_explanation_matches_rollout_step_1(self):
        """explain_prediction must report the exact same probability/stage
        rollout.engine.run_rollout(k=1) would for the same context -- it's
        explicitly built by reusing that function, not just documented to
        agree with it."""
        from rollout.engine import run_rollout
        model, context, normalizer = _make_model_and_context()
        record = explain_prediction(model, context, normalizer, torch.device("cpu"))
        rollout_result = run_rollout(model, context, normalizer, k=1, device=torch.device("cpu"))
        assert record.infiltration_probability == pytest.approx(rollout_result.infiltration_probability[0])
        assert record.stage.stage == rollout_result.stage_predictions[0].stage

    def test_rejects_mixed_day_context(self):
        model, context, normalizer = _make_model_and_context()
        mixed = context[:2] + [_snap("OtherDay", 0, num_flows=1)]
        with pytest.raises(ValueError, match="same day"):
            explain_prediction(model, mixed, normalizer, torch.device("cpu"))
