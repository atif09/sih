"""
Network Forecast View (differentiator): decodes the world model's predicted
NEXT-window latent (z_hat_{t+1}, from the existing, unchanged dynamics_head)
back into human-readable aggregate network statistics -- flow count, SYN
share, distinct ports touched, bytes out/in, new-host count -- so an analyst
sees what the network is expected to look like, not just a risk score.

Why this doesn't touch the trained checkpoint
--------------------------------------------------
The base world model (checkpoints/phase3/best_model.pt) is loaded FROZEN
(no_grad, eval mode) and used only to produce ẑ_t+1 exactly as every other
feature in this project already does. This decoder is a SEPARATE, small
MLP trained on top of that frozen output -- a standard "probe on frozen
embeddings" setup, not a retrain. The base model's own weights, and every
evaluation number already reported (0.989 AUC etc.), are completely
unaffected; scripts/train_forecast_decoder.py never touches best_model.pt.

Targets are real ground truth, never the model's own guess
----------------------------------------------------------------
Training targets are computed directly from the REAL next-window
GraphSnapshot (see `compute_target_stats` below) -- graph-level aggregates,
not per-node reconstruction (which would need per-node decoding and much
more validation data to trust). This keeps the decoder's job tractable
enough to actually validate (see the held-out-day R^2/MAE script), rather
than a high-dimensional reconstruction nobody checked.

Honest limitation (disclosed, not hidden)
--------------------------------------------
This predicts GRAPH-LEVEL aggregates for the whole window, not per-host
values -- "distinct ports touched" here is the single most concentrated
host-pair's port count in the window (max, not global unique), a real,
disclosed approximation, not literal global port diversity (which would
need cross-host deduplication this decoder doesn't attempt). Only as
reliable as its own validated R^2 -- see the training script's printed
held-out metrics before trusting any specific number this produces.

Two stats were tried and DROPPED after real held-out-day validation showed
negative R^2 (worse than predicting the mean): `syn_share` and
`new_host_count`. new_host_count has a real architectural reason, not just
insufficient training: the world model's readout (see
models/world_model.py) pools node features via mean+max, which is
permutation-invariant over nodes BY DESIGN -- it structurally discards
exactly the host-identity information needed to answer "which specific
hosts are new," so no amount of decoder training on top of z_hat could fix
this without changing the base model's own architecture. syn_share's
failure is more likely a training-signal issue (a noisy ratio the base
model's own loss never needed to preserve precisely) rather than an
architectural impossibility, but it was dropped on the same real-evidence
basis: negative held-out R^2, not a hunch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES

STAT_NAMES: tuple[str, ...] = (
    "flow_count",
    "max_dst_ports_touched",
    "bytes_out",
    "bytes_in",
)

_OUT_BYTES_IDX = NODE_FEATURE_NAMES.index("out_bytes_sum")
_IN_BYTES_IDX = NODE_FEATURE_NAMES.index("in_bytes_sum")
_DST_PORT_IDX = EDGE_FEATURE_NAMES.index("dst_port_nunique")


def compute_target_stats(current_snapshot: GraphSnapshot, next_snapshot: GraphSnapshot) -> np.ndarray:
    """Real ground-truth graph-level aggregates for `next_snapshot`, in the
    same order as STAT_NAMES. `current_snapshot` is accepted (unused by the
    stats that survived validation) to keep this function's signature
    stable if a future stat needs a same-vs-previous-window delta again."""
    nf = next_snapshot.node_features
    ef = next_snapshot.edge_features

    flow_count = float(next_snapshot.num_flows)
    max_dst_ports = float(ef[:, _DST_PORT_IDX].max()) if ef.shape[0] else 0.0
    bytes_out = float(nf[:, _OUT_BYTES_IDX].sum()) if nf.shape[0] else 0.0
    bytes_in = float(nf[:, _IN_BYTES_IDX].sum()) if nf.shape[0] else 0.0

    return np.array([flow_count, max_dst_ports, bytes_out, bytes_in], dtype=np.float32)


class ForecastDecoder(nn.Module):
    """Small MLP: predicted next-window latent (latent_dim) -> real
    human-readable graph-level stats (len(STAT_NAMES)). Trained on
    log1p-transformed targets (flow counts and byte sums are heavily
    right-skewed) -- see TargetScaler below; the model itself only ever
    sees/produces the transformed scale."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(STAT_NAMES)),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


def transform_targets(raw: np.ndarray) -> np.ndarray:
    """log1p on every stat -- all six are non-negative counts/sums with
    heavy right skew (a handful of high-volume windows dominate raw-scale
    MSE otherwise, exactly the kind of distortion models/normalization.py's
    own feature normalizer avoids elsewhere in this project)."""
    return np.log1p(np.clip(raw, 0, None))


def inverse_transform_targets(transformed: np.ndarray) -> np.ndarray:
    return np.expm1(transformed)
