"""
The world model: GNNEncoder + TemporalCore + two prediction heads, trained to
learn P(S_t+1 | S_t) via self-predictive latent dynamics -- not a static
classifier over individual windows.

Training signal, per sequence position t (t = 1..L, given L+1 encoded states
z_1..z_{L+1}):
  1. **Dynamics loss** (the actual "world model" objective): predict the
     NEXT window's latent embedding, z_hat_{t+1} = dynamics_head(h_t), and
     compare it (MSE) against the true z_{t+1} produced by the same encoder.
     This is exactly P(S_t+1 | S_t) learned in latent space -- the model is
     never told the future window's raw features, only trained to predict
     their encoded representation from causal history h_t = f(z_1..z_t).

     The target z_{t+1} is **detached** (stop-gradient) before computing this
     loss. Without that, the dynamics loss alone could be trivially driven to
     ~0 by collapsing every embedding to a constant (both sides of the loss
     come from the same encoder) -- a well-known failure mode in
     self-predictive latent objectives (see BYOL/SimSiam/JEPA-style
     literature). The stop-gradient removes that degenerate shortcut; the
     classification loss below (using real ground-truth labels, not a
     self-referential target) is what actually anchors the embedding space to
     meaningful structure.

  2. **Attack-forecast loss**: predict attack_label_next (binary) from
     [h_t ; z_hat_{t+1}] -- i.e. from the *learned dynamics prediction*, not
     from the ground-truth next state. This is what makes the forecast a
     genuine use of the world model rather than a bypass around it, and it's
     exactly what Phase 4's K-step rollout will chain forward at inference
     time (no ground truth available then either).

  3. **Attack-type loss** (auxiliary, optional): multi-label classification
     over a FIXED vocabulary of attack-type strings built from the TRAINING
     days only (see models/dataset.py). This is deliberate: the held-out
     day's attack type must not have a dedicated output unit, or the
     "generalizes to an unseen attack pattern" claim would be undermined by
     the architecture already knowing that category exists. Expect (and
     report, not hide) this head performing poorly on the held-out day's
     actual type -- the primary generalization claim rests on head #2, the
     binary attack-forecast head, which has no such vocabulary constraint.

At inference / K-step rollout (Phase 4), there is no ground-truth z_{t+1} to
feed the temporal core at the next step -- z_hat_{t+1} (this model's own
prediction) is fed back in autoregressively instead. `rollout_step()` below
implements exactly that one-step recurrence; Phase 4 will call it K times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.gnn_encoder import GNNEncoder
from models.temporal import TemporalCore, build_temporal_core


@dataclass
class WorldModelConfig:
    node_dim: int
    edge_dim: int
    latent_dim: int = 128
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_heads: int = 4
    temporal_type: str = "lstm"
    temporal_hidden_dim: int = 128
    temporal_num_layers: int = 2
    temporal_num_heads: int = 4
    dropout: float = 0.1
    attack_type_vocab: list[str] = field(default_factory=list)


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.encoder = GNNEncoder(
            node_dim=config.node_dim, edge_dim=config.edge_dim,
            latent_dim=config.latent_dim, hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_num_layers, heads=config.gnn_heads, dropout=config.dropout,
        )
        self.temporal: TemporalCore = build_temporal_core(
            config.temporal_type, input_dim=config.latent_dim, hidden_dim=config.temporal_hidden_dim,
            num_layers=config.temporal_num_layers, dropout=config.dropout, num_heads=config.temporal_num_heads,
        )

        self.dynamics_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.temporal_hidden_dim, config.latent_dim),
        )

        combined_dim = config.temporal_hidden_dim + config.latent_dim
        self.attack_head = nn.Sequential(
            nn.Linear(combined_dim, config.temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.temporal_hidden_dim, 1),
        )

        self.num_attack_types = len(config.attack_type_vocab)
        if self.num_attack_types > 0:
            self.attack_type_head = nn.Sequential(
                nn.Linear(combined_dim, config.temporal_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.temporal_hidden_dim, self.num_attack_types),
            )
        else:
            self.attack_type_head = None

    def encode_sequence(self, pyg_batch: Batch, batch_size: int, seq_len_plus_1: int) -> torch.Tensor:
        """pyg_batch must contain exactly batch_size * seq_len_plus_1 graphs,
        ordered as [example_0_t0, example_0_t1, ..., example_0_tL,
        example_1_t0, ...] (i.e. sequence-major, matching how
        models/dataset.py's collate_fn builds it). Returns z_seq of shape
        (batch_size, seq_len_plus_1, latent_dim)."""
        z_flat = self.encoder(pyg_batch)  # (batch_size * seq_len_plus_1, latent_dim)
        return z_flat.view(batch_size, seq_len_plus_1, self.config.latent_dim)

    def forward(self, pyg_batch: Batch, batch_size: int, seq_len_plus_1: int) -> dict:
        """Full training forward pass. Returns a dict of everything needed to
        compute the three losses at every one of the L supervised positions."""
        z_seq = self.encode_sequence(pyg_batch, batch_size, seq_len_plus_1)  # (B, L+1, D)

        context = z_seq[:, :-1, :]  # (B, L, D) -- z_1..z_L
        targets = z_seq[:, 1:, :]  # (B, L, D) -- z_2..z_{L+1}, i.e. the "next state" at each position

        h_seq = self.temporal(context)  # (B, L, H), causal
        z_hat_next = self.dynamics_head(h_seq)  # (B, L, D)

        combined = torch.cat([h_seq, z_hat_next], dim=-1)  # (B, L, H+D)
        attack_logits = self.attack_head(combined).squeeze(-1)  # (B, L)

        attack_type_logits = self.attack_type_head(combined) if self.attack_type_head is not None else None

        return {
            "z_seq": z_seq,
            "z_target": targets.detach(),  # stop-gradient, see module docstring
            "z_hat_next": z_hat_next,
            "h_seq": h_seq,
            "attack_logits": attack_logits,
            "attack_type_logits": attack_type_logits,
        }

    @torch.no_grad()
    def rollout_step(self, h_prev: torch.Tensor, z_hat_prev: torch.Tensor, lstm_state=None):
        """One autoregressive step for Phase 4's K-step rollout: given the
        current hidden state and the model's own previous latent prediction
        (no ground truth available at inference), predict the next latent,
        attack probability, and advance the temporal core by one step.

        Only meaningful for the LSTM temporal core (which has an explicit,
        cheap single-step recurrence via `nn.LSTM`'s hidden state tuple).
        The Transformer core's rollout instead re-runs over the growing
        sequence each step, which Phase 4 will implement directly against
        `forward()`/`encode_sequence()` rather than this method.
        """
        if not isinstance(self.temporal, torch.nn.Module) or not hasattr(self.temporal, "lstm"):
            raise NotImplementedError(
                "rollout_step is only implemented for the LSTM temporal core; "
                "Phase 4 will add a Transformer-compatible rollout path."
            )
        z_in = z_hat_prev.unsqueeze(1)  # (B, 1, D)
        h_out, lstm_state = self.temporal.lstm(z_in, lstm_state)
        h_t = h_out[:, 0, :]
        z_hat_next = self.dynamics_head(h_t)
        combined = torch.cat([h_t, z_hat_next], dim=-1)
        attack_logit = self.attack_head(combined).squeeze(-1)
        return {
            "h_t": h_t, "z_hat_next": z_hat_next,
            "attack_logit": attack_logit, "lstm_state": lstm_state,
        }
