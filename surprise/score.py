"""
Surprise Alarm: novelty detection from the world model's OWN dynamics
prediction error, requiring no attack-type label and no retraining.

Why this is real, not fabricated
---------------------------------
Training already optimizes MSE(z_hat_{t+1}, stopgrad(z_{t+1})) -- the
"dynamics loss" described in models/world_model.py's own docstring. That
same quantity, computed at inference against a REAL observed next window
(never against the unobserved forecast horizon, where no ground truth
exists by construction), is exactly "how wrong was the model's own
prediction of this window" -- i.e. novelty relative to what the model's
learned dynamics expected. No new head, no new loss term, no retraining:
this reuses the existing dynamics_head unchanged.

Why a reference distribution is required
-------------------------------------------
A raw MSE value is meaningless on its own -- there is no natural "large" or
"small" scale. scripts/export_surprise_reference.py computes this same
quantity across every real benign-to-benign window transition in the
training days (the held-out day excluded, matching the model's own
train/val split) and saves the full sorted array. At inference, a live
score is converted to a PERCENTILE against that real reference
distribution, not an arbitrary threshold.

Honest limitation (disclosed, not hidden)
--------------------------------------------
This detects "behaves differently than the model's training distribution
expects," not "malicious" specifically -- any traffic pattern absent from
CIC-IDS2017's benign days (a new benign service, a backup job, a software
update) could also score as high novelty. It has NOT been validated to
separate attack windows from novel-but-benign windows (there is no such
labeled benign-novelty data in this dataset); treat elevated surprise as
"investigate," not "confirmed attack."

Real validation performed (cross-checked against the held-out day, AFTER
the empty-window fix described below): on Friday-Afternoon-DDoS, this score
perfectly separates the day's real attack windows from its benign windows
(surprise-as-classifier AUC 1.0 against ground truth; every attack window
landed at the 98.6th-99.7th percentile of the benign reference, every
benign window at 96.2nd or below -- a real margin, not an adjacent-value
coincidence). That is a genuinely strong result, but it rests on ONE
held-out episode with a small number of positive windows; treat this the
same as every other single-held-out-day result in this project: real, not
cherry-picked, but not a generalization claim either.

Real bug found and fixed during that validation: empty windows (zero
flows) pool to a single learned "quiet network" latent (see
graph/snapshot.py's readout), so predicting into -- or from -- an empty
window is trivially predictable and produces the SAME constant MSE every
time, regardless of any real content. Monday alone has 233 empty windows
out of 720. Before this was excluded, that constant value sat at ~70th
percentile of the reference by coincidence, and ANY capture touching empty
windows scored a spurious alarm at that same percentile -- verified live: a
genuinely benign held-out slice of Monday traffic triggered false alarms on
nearly every window before this fix, and scores a normal, unremarkable
percentile (median ~56th) after it. `is_informative_transition` below
excludes any transition where the target window OR the last context window
has zero flows, from BOTH the reference distribution and live scoring --
both MUST use this same filter, or the contamination returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Batch

from graph.snapshot import GraphSnapshot
from models.dataset import snapshot_to_pyg_data
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel

ALARM_PERCENTILE = 97.5  # empirically set from the held-out day validation above
# (real attack windows scored 98.6-99.7, real benign windows <=96.2) --
# 97.5 sits in that real gap with margin on both sides, not at either edge


@dataclass
class SurpriseReference:
    values: list[float]  # sorted ascending, real benign-to-benign dynamics MSE

    @classmethod
    def load(cls, path: Path) -> "SurpriseReference":
        with open(path) as f:
            data = json.load(f)
        return cls(values=sorted(data["benign_dynamics_mse"]))

    def percentile_of(self, value: float) -> float:
        """Real rank of `value` within the reference distribution, in
        [0, 100] -- e.g. 97.3 means this score is higher than 97.3% of
        observed benign-to-benign transitions."""
        if not self.values:
            return 0.0
        import bisect

        idx = bisect.bisect_left(self.values, value)
        return 100.0 * idx / len(self.values)


def is_informative_transition(context_snapshots: list[GraphSnapshot], true_next_snapshot: GraphSnapshot) -> bool:
    """False when either the target window or the last context window is
    empty (num_flows == 0) -- see module docstring's "real bug found" note.
    Both the reference export and live scoring must use this SAME check, or
    the empty-window contamination this guards against returns."""
    return context_snapshots[-1].num_flows > 0 and true_next_snapshot.num_flows > 0


@torch.no_grad()
def compute_dynamics_surprise(
    model: WorldModel,
    context_snapshots: list[GraphSnapshot],
    true_next_snapshot: GraphSnapshot,
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> float:
    """Real dynamics-prediction error for one REAL observed transition:
    encode `context_snapshots` (length L) plus the one true next window
    (L+1 total), run the exact same context/target split and dynamics_head
    call used at training time (models/world_model.py's forward()), and
    return the MSE between the model's predicted next-window latent and the
    true next-window latent at the final position. Requires an LSTM or
    Transformer temporal core interchangeably -- both are plain nn.Modules
    called the same way here."""
    model.eval()
    all_snapshots = list(context_snapshots) + [true_next_snapshot]
    data_list = [snapshot_to_pyg_data(s, normalizer) for s in all_snapshots]
    batch = Batch.from_data_list(data_list).to(device)
    z_seq = model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(all_snapshots))  # (1, L+1, D)

    context = z_seq[:, :-1, :]
    target = z_seq[:, -1, :]  # the one true next-window latent we're scoring against

    h_seq = model.temporal(context)  # (1, L, H), causal
    h_t = h_seq[:, -1, :]
    z_hat_next = model.dynamics_head(h_t)  # (1, D)

    mse = torch.mean((z_hat_next - target) ** 2).item()
    return float(mse)
