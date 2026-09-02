"""
Temporal component: consumes a sequence of per-window latent embeddings
z_1..z_L (from GNNEncoder) and produces causal hidden states h_1..h_L, where
h_t depends only on z_1..z_t (never future windows) -- this is what makes the
per-position "predict the next state" training in world_model.py valid.

Built as a swappable interface (`TemporalCore`) per the architecture
decision: two concrete implementations are provided now (LSTM and a
causally-masked Transformer encoder), selected via config
(`model.temporal.type: lstm | transformer`). A Recurrent State-Space Model
(RSSM) is a documented future upgrade -- it would slot in here as a third
`TemporalCore` subclass without touching GNNEncoder or WorldModel, which only
depend on this class's `forward(z_seq) -> hidden_seq` contract.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TemporalCore(nn.Module, ABC):
    hidden_dim: int

    @abstractmethod
    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (B, L, D) -> hidden_seq: (B, L, hidden_dim), causal."""
        raise NotImplementedError


class LSTMTemporalCore(TemporalCore):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # nn.LSTM is causal by construction: output[:, t] only ever depends
        # on inputs[:, :t+1].
        hidden_seq, _ = self.lstm(z_seq)
        return hidden_seq


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class TransformerTemporalCore(TemporalCore):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 2,
        num_heads: int = 4, dropout: float = 0.1, max_len: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = _SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        seq_len = z_seq.shape[1]
        # Causal mask: position i may only attend to positions <= i.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=z_seq.device), diagonal=1,
        )
        h = self.pos_encoding(self.input_proj(z_seq))
        return self.encoder(h, mask=causal_mask, is_causal=True)


def build_temporal_core(temporal_type: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_heads: int = 4) -> TemporalCore:
    if temporal_type == "lstm":
        return LSTMTemporalCore(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
    if temporal_type == "transformer":
        return TransformerTemporalCore(input_dim, hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    raise ValueError(f"Unknown temporal_type: {temporal_type!r} (expected 'lstm' or 'transformer')")
