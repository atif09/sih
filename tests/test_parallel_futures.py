import pytest
import torch

from futures.parallel_futures import sample_parallel_futures
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from tests.test_model_dataset import _snap

NODE_DIM = len(NODE_FEATURE_NAMES)
EDGE_DIM = len(EDGE_FEATURE_NAMES)


def _make_context_and_model(dropout: float = 0.3, attack_vocab=("PortScan", "Bot")):
    snapshots = [_snap("Day", i, num_flows=1) for i in range(8)]
    normalizer = FeatureNormalizer().fit(snapshots)
    context = snapshots[:5]

    config = WorldModelConfig(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, latent_dim=16, gnn_hidden_dim=8,
        gnn_num_layers=2, gnn_heads=2, temporal_type="lstm", temporal_hidden_dim=16,
        temporal_num_layers=1, attack_type_vocab=list(attack_vocab), dropout=dropout,
    )
    model = WorldModel(config)
    model.eval()
    return model, context, normalizer


class TestSampleParallelFutures:
    def test_probabilities_sum_to_one(self):
        model, context, normalizer = _make_context_and_model()
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=20)
        assert sum(o.probability for o in result.outcomes) == pytest.approx(1.0)

    def test_sample_counts_sum_to_n_samples(self):
        model, context, normalizer = _make_context_and_model()
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=25)
        assert sum(o.n_samples for o in result.outcomes) == 25
        assert result.n_samples == 25

    def test_outcomes_sorted_descending_by_probability(self):
        model, context, normalizer = _make_context_and_model()
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=30)
        probs = [o.probability for o in result.outcomes]
        assert probs == sorted(probs, reverse=True)

    def test_mean_infiltration_probability_in_valid_range(self):
        model, context, normalizer = _make_context_and_model()
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=20)
        for outcome in result.outcomes:
            assert 0.0 <= outcome.mean_infiltration_probability <= 1.0

    def test_deterministic_stage_matches_eval_mode_rollout(self):
        from rollout.engine import run_rollout
        model, context, normalizer = _make_context_and_model()
        expected = run_rollout(model, context, normalizer, k=3, device=torch.device("cpu"), compute_explanations=False)
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=5)
        assert result.deterministic_stage == expected.stage_predictions[-1].stage

    def test_model_restored_to_eval_mode_after_sampling(self):
        model, context, normalizer = _make_context_and_model()
        model.eval()
        sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=10)
        assert model.training is False

    def test_model_restored_to_eval_mode_even_if_rollout_raises(self, monkeypatch):
        model, context, normalizer = _make_context_and_model()
        model.eval()

        calls = {"n": 0}
        import futures.parallel_futures as mod
        real_run_rollout = mod.run_rollout

        def _flaky_run_rollout(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("simulated failure mid-sampling")
            return real_run_rollout(*args, **kwargs)

        monkeypatch.setattr(mod, "run_rollout", _flaky_run_rollout)
        with pytest.raises(RuntimeError, match="simulated failure"):
            sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=10)
        assert model.training is False

    def test_zero_dropout_collapses_to_single_deterministic_outcome(self):
        # With dropout=0.0, every MC sample is identical to the eval-mode
        # rollout -- a real sanity check that variability comes from
        # dropout noise, not some other hidden randomness.
        model, context, normalizer = _make_context_and_model(dropout=0.0)
        result = sample_parallel_futures(model, context, normalizer, k=3, device=torch.device("cpu"), n_samples=15)
        assert len(result.outcomes) == 1
        assert result.outcomes[0].probability == pytest.approx(1.0)
        assert result.outcomes[0].stage == result.deterministic_stage
