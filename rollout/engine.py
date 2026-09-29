"""
K-step forward-simulation engine: given a real observed context window
(S_{t-L+1}..S_t), roll the trained world model forward K steps and produce a
time-series infiltration probability, a predicted MITRE stage per step, and
real, context-level explainability (attention weights + gradient-based
feature attribution, added in Phase 5 -- see explain/attention.py and
explain/saliency.py; not the Phase 4 heuristic stub, which is also still
attached for comparison as `driving_context_flows`).

How the rollout works, per temporal core (see models/temporal.py)
-------------------------------------------------------------------
- **LSTM**: the context is run through `model.temporal.lstm` once to get its
  final (h_n, c_n) state -- an O(1) summary of the whole context. Each
  further step calls `model.rollout_step`, which feeds the model's own
  previous latent prediction ẑ back in and advances the LSTM state by one
  step. O(K) total.
- **Transformer**: there is no fixed-size state to carry -- causal attention
  needs the whole sequence. Each step re-runs `model.temporal(...)` (still
  causally masked) over the context extended with every ẑ predicted so far,
  and reads off the last position's hidden state. O(K^2) total, fine for the
  K values this is meant for (tens of steps, not thousands).

Both paths use the SAME model, the same two heads, and produce results in
the same `RolloutResult` shape -- this is the concrete meaning of "structure
the code so the temporal component is swappable" carried through to
inference, not just training.

At every step k, infiltration_probability[k] and attack_type_probs[k] come
from the attack-forecast/attack-type heads applied to [h_k ; ẑ_{t+k+1}] --
i.e. from the model's own predicted dynamics, exactly as at training time
(see models/world_model.py's docstring). There is no ground truth available
during rollout by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Batch

from graph.snapshot import GraphSnapshot
from models.dataset import snapshot_to_pyg_data
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel
from rollout.explain_stub import top_contributing_edges
from rollout.mitre import StagePrediction, predict_stage

# Imported lazily-safe here (neither module imports rollout.engine, so no
# circular import) -- explain/explainer.py, by contrast, imports
# run_rollout() FROM this module, so this module must never import
# explain.explainer back.
from explain.attention import compute_edge_attention, top_edges_overall
from explain.saliency import compute_feature_saliency


@dataclass
class RolloutResult:
    day: str
    context_end_window_id: int
    k: int
    infiltration_probability: list[float]
    attack_type_probs: list[dict[str, float]]  # length k, empty dict if no attack-type head
    stage_predictions: list[StagePrediction]
    driving_context_flows: list[dict] = field(default_factory=list)  # Phase 4 heuristic stub, see explain_stub.py
    attention_flows: list[dict] = field(default_factory=list)  # Phase 5: real GNN attention, see explain/attention.py
    feature_saliency: dict = field(default_factory=dict)  # Phase 5: real gradient attribution, see explain/saliency.py
    predicted_latents: list[list[float]] = field(default_factory=list)  # length k; ẑ_hat at each step, for Network Forecast View's decoder (models/forecast_decoder.py) -- was always computed internally, just discarded before this field existed


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def run_rollout(
    model: WorldModel,
    context_snapshots: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    k: int,
    device: torch.device,
    infiltration_threshold: float = 0.5,
    compute_explanations: bool = True,
    use_dropout: bool = False,
) -> RolloutResult:
    """compute_explanations=True (default) additionally computes real
    attention-weight and gradient-based feature attribution for the context
    (Phase 5) -- set False to skip that extra forward/backward pass when only
    the probability time series is needed (e.g. inside a training loop or a
    benchmark sweep over many contexts).

    use_dropout=False (default) puts the model in eval() mode before running
    -- deterministic, dropout disabled, correct for every normal caller. Set
    True ONLY for MC-Dropout sampling (futures/parallel_futures.py), which
    needs dropout genuinely active across a whole loop of these calls.

    This parameter exists because of a real, previously undiscovered bug: an
    earlier version of this function unconditionally called model.eval() as
    its first line, silently overriding a caller's own model.train() call on
    every single invocation -- so Multiple Parallel Futures' MC-Dropout
    sampling was NEVER actually active. Verified: 0.00000 standard deviation
    across 40 "independent" samples at every rollout step, before this fix.
    An implicit "caller sets the mode, this function trusts it" contract was
    tried first and immediately broke a real test (a freshly constructed
    model defaults to train() mode in plain PyTorch) -- this explicit flag
    is the safer fix: every call site's intent is visible at the call, and
    nothing depends on remembering ambient mutable model state."""
    if len(context_snapshots) < 1:
        raise ValueError("run_rollout requires at least one context snapshot")
    if len({s.day for s in context_snapshots}) > 1:
        raise ValueError("context_snapshots must all come from the same day (see graph/transitions.py's same-day invariant)")

    if not use_dropout:
        model.eval()
    day = context_snapshots[0].day
    context_end = context_snapshots[-1].window_id
    last_context_snapshot = context_snapshots[-1]

    # The K-step rollout loop itself never needs gradients (it's pure
    # inference) -- explicitly scoped rather than a function-level
    # @torch.no_grad() decorator, because the explanation step below (when
    # requested) needs gradients enabled and must run in the same function.
    with torch.no_grad():
        data_list = [snapshot_to_pyg_data(s, normalizer) for s in context_snapshots]
        batch = Batch.from_data_list(data_list).to(device)
        z_context = model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(context_snapshots))  # (1, L, D)

        infiltration_probs: list[float] = []
        type_probs: list[dict[str, float]] = []
        predicted_latents: list[list[float]] = []
        vocab = model.config.attack_type_vocab

        def _record(attack_logit: torch.Tensor, attack_type_logit: torch.Tensor | None) -> None:
            prob = float(_sigmoid(attack_logit).item())
            infiltration_probs.append(prob)
            if attack_type_logit is not None and vocab:
                probs = _sigmoid(attack_type_logit).squeeze(0).cpu().tolist()
                type_probs.append(dict(zip(vocab, probs)))
            else:
                type_probs.append({})

        if hasattr(model.temporal, "lstm"):
            hidden_seq, lstm_state = model.temporal.lstm(z_context)
            h_t = hidden_seq[:, -1, :]
            z_hat = model.dynamics_head(h_t)
            combined = torch.cat([h_t, z_hat], dim=-1)
            attack_logit = model.attack_head(combined).squeeze(-1)
            attack_type_logit = model.attack_type_head(combined) if model.attack_type_head is not None else None
            _record(attack_logit, attack_type_logit)
            predicted_latents.append(z_hat.squeeze(0).cpu().tolist())

            for _ in range(k - 1):
                step = model.rollout_step(z_hat, lstm_state)
                z_hat, lstm_state = step["z_hat_next"], step["lstm_state"]
                _record(step["attack_logit"], step["attack_type_logit"])
                predicted_latents.append(z_hat.squeeze(0).cpu().tolist())
        else:
            # Transformer (or any future TemporalCore): re-run the causal
            # encoder over the growing latent sequence each step.
            z_running = z_context
            for _ in range(k):
                h_seq = model.temporal(z_running)
                h_t = h_seq[:, -1, :]
                z_hat = model.dynamics_head(h_t)
                combined = torch.cat([h_t, z_hat], dim=-1)
                attack_logit = model.attack_head(combined).squeeze(-1)
                attack_type_logit = model.attack_type_head(combined) if model.attack_type_head is not None else None
                _record(attack_logit, attack_type_logit)
                predicted_latents.append(z_hat.squeeze(0).cpu().tolist())
                z_running = torch.cat([z_running, z_hat.unsqueeze(1)], dim=1)

        stage_predictions = [
            predict_stage(prob, types, last_context_snapshot, infiltration_threshold)
            for prob, types in zip(infiltration_probs, type_probs)
        ]

    attention_flows: list[dict] = []
    feature_saliency: dict = {}
    if compute_explanations:
        edge_records = compute_edge_attention(model, last_context_snapshot, normalizer, device)
        attention_flows = top_edges_overall(edge_records, top_k=5)
        feature_saliency = compute_feature_saliency(model, context_snapshots, normalizer, device)

    return RolloutResult(
        day=day, context_end_window_id=context_end, k=k,
        infiltration_probability=infiltration_probs,
        attack_type_probs=type_probs,
        stage_predictions=stage_predictions,
        driving_context_flows=top_contributing_edges(last_context_snapshot),
        attention_flows=attention_flows,
        feature_saliency=feature_saliency,
        predicted_latents=predicted_latents,
    )
