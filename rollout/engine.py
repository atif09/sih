"""
K-step forward-simulation engine: given a real observed context window
(S_{t-L+1}..S_t), roll the trained world model forward K steps and produce a
time-series infiltration probability, a predicted MITRE stage per step, and
a (context-level) explainability stub.

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


@dataclass
class RolloutResult:
    day: str
    context_end_window_id: int
    k: int
    infiltration_probability: list[float]
    attack_type_probs: list[dict[str, float]]  # length k, empty dict if no attack-type head
    stage_predictions: list[StagePrediction]
    driving_context_flows: list[dict] = field(default_factory=list)  # see explain_stub.py


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@torch.no_grad()
def run_rollout(
    model: WorldModel,
    context_snapshots: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    k: int,
    device: torch.device,
    infiltration_threshold: float = 0.5,
) -> RolloutResult:
    if len(context_snapshots) < 1:
        raise ValueError("run_rollout requires at least one context snapshot")
    if len({s.day for s in context_snapshots}) > 1:
        raise ValueError("context_snapshots must all come from the same day (see graph/transitions.py's same-day invariant)")

    model.eval()
    day = context_snapshots[0].day
    context_end = context_snapshots[-1].window_id

    data_list = [snapshot_to_pyg_data(s, normalizer) for s in context_snapshots]
    batch = Batch.from_data_list(data_list).to(device)
    z_context = model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(context_snapshots))  # (1, L, D)

    infiltration_probs: list[float] = []
    type_probs: list[dict[str, float]] = []
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

        for _ in range(k - 1):
            step = model.rollout_step(z_hat, lstm_state)
            z_hat, lstm_state = step["z_hat_next"], step["lstm_state"]
            _record(step["attack_logit"], step["attack_type_logit"])
    else:
        # Transformer (or any future TemporalCore): re-run the causal encoder
        # over the growing latent sequence each step.
        z_running = z_context
        for _ in range(k):
            h_seq = model.temporal(z_running)
            h_t = h_seq[:, -1, :]
            z_hat = model.dynamics_head(h_t)
            combined = torch.cat([h_t, z_hat], dim=-1)
            attack_logit = model.attack_head(combined).squeeze(-1)
            attack_type_logit = model.attack_type_head(combined) if model.attack_type_head is not None else None
            _record(attack_logit, attack_type_logit)
            z_running = torch.cat([z_running, z_hat.unsqueeze(1)], dim=1)

    last_context_snapshot = context_snapshots[-1]
    stage_predictions = [
        predict_stage(prob, types, last_context_snapshot, infiltration_threshold)
        for prob, types in zip(infiltration_probs, type_probs)
    ]

    return RolloutResult(
        day=day, context_end_window_id=context_end, k=k,
        infiltration_probability=infiltration_probs,
        attack_type_probs=type_probs,
        stage_predictions=stage_predictions,
        driving_context_flows=top_contributing_edges(last_context_snapshot),
    )
