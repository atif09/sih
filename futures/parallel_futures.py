"""
Multiple Parallel Futures (differentiator feature, priority #2 per the
reference brief): instead of a single deterministic forecast, show several
plausible attack outcomes ranked by likelihood.

How this is real, not fabricated
------------------------------------
The existing K-step rollout (rollout/engine.py) is DETERMINISTIC: the same
context always produces the same latent trajectory, because the model runs
in eval() mode (dropout disabled, as it should be for a single point
forecast). To get genuinely different candidate futures without training a
new model or hand-authoring branch probabilities, this module uses
**MC-Dropout** (Gal & Ghahramani, 2016, "Dropout as a Bayesian
Approximation"): re-enable the model's own already-trained dropout layers
at inference time, and run the SAME rollout N independent times. Each run's
dropout noise perturbs the latent trajectory slightly differently, so the
rollout can genuinely land on a different predicted stage. Tallying the
resulting stage outcomes across N samples gives a real, if approximate,
probability distribution over "possible futures" -- grounded in the model's
own learned uncertainty, not invented percentages.

Checked before building: this architecture has NO BatchNorm anywhere (GNN
encoder, LSTM/Transformer temporal core, output heads all use only
nn.Dropout) -- see models/gnn_encoder.py, models/temporal.py,
models/world_model.py. That matters because MC-Dropout requires flipping
the module to train() mode for the dropout layers to actually sample; if
BatchNorm were present, train() mode would also start updating its running
statistics from single-sample batches, corrupting the model. Since there's
no BatchNorm, calling model.train() here only activates dropout sampling,
exactly what's wanted, with no other side effect. The model is always
restored to eval() before returning (try/finally), so this can never leak
train-mode into any other caller.

What "outcome" means here: the predicted MITRE stage at the FINAL rollout
step (step k), decided by the exact same rollout.mitre.predict_stage logic
every other feature in this project uses -- not a separate scoring rule.
Two rollouts that land on the same final stage are the same "future" for
this feature's purposes, even if their step-by-step paths differed.

Honest limitations, disclosed rather than hidden
------------------------------------------------------
- This resamples the SAME encoded context via the model's own uncertainty,
  not literally different observed starting conditions -- it answers "how
  confident/consistent is the model's own forecast", not "what would a
  different attacker choice look like" (that's Counterfactual Intervention
  Simulation, a different feature).
- Percentages are a Monte Carlo ESTIMATE: re-running with a different
  random seed will shift them slightly. N_SAMPLES is chosen high enough
  (see validation script) that the top-ranked outcome is stable in
  practice, but exact percentages are not bit-reproducible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from graph.snapshot import GraphSnapshot
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel
from rollout.engine import run_rollout

N_SAMPLES_DEFAULT = 40


@dataclass
class FutureOutcome:
    stage: str
    probability: float  # fraction of MC-Dropout samples that landed here, in [0, 1]
    n_samples: int  # raw count backing `probability`, for anyone who wants to judge confidence
    mean_infiltration_probability: float  # mean final-step infiltration probability among samples with this outcome


@dataclass
class ParallelFuturesResult:
    n_samples: int
    outcomes: list[FutureOutcome]  # sorted descending by probability
    deterministic_stage: str  # the single eval()-mode (no dropout) outcome, for cross-reference


@torch.no_grad()
def run_mc_dropout_samples(
    model: WorldModel,
    context: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    k: int,
    device: torch.device,
    n_samples: int = N_SAMPLES_DEFAULT,
) -> list:
    """Real MC-Dropout sampling, extracted so callers needing MORE than the
    final-step stage tally (e.g. futures/time_to_compromise.py, which needs
    the FULL per-step probability trajectory of every sample) don't have to
    pay for a second, separate round of n_samples rollouts. Returns the raw
    list of RolloutResult objects; see module docstring for the method and
    its honest limitations, and rollout/engine.py's run_rollout docstring
    for why use_dropout=True is required here (a real bug, now fixed, meant
    this was previously a silent no-op)."""
    model.train()  # activates dropout sampling only -- see module docstring (no BatchNorm exists)
    try:
        return [
            run_rollout(model, context, normalizer, k=k, device=device, compute_explanations=False, use_dropout=True)
            for _ in range(n_samples)
        ]
    finally:
        model.eval()  # always restore -- every other caller in this project assumes eval mode


@torch.no_grad()
def sample_parallel_futures(
    model: WorldModel,
    context: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    k: int,
    device: torch.device,
    n_samples: int = N_SAMPLES_DEFAULT,
    samples: list | None = None,
) -> ParallelFuturesResult:
    """Runs `n_samples` independent MC-Dropout rollouts from the same
    context and tallies the resulting final-step stage outcomes. See
    module docstring for the full method and its honest limitations. Pass
    `samples` (from a prior run_mc_dropout_samples call) to reuse an
    already-sampled ensemble instead of sampling a fresh one -- api/pipeline.py
    does this so /analyze pays for MC-Dropout sampling once, not twice, when
    both this and Time-to-Compromise are computed for the same request."""
    # The one deterministic (eval-mode) forecast, for reference -- this is
    # exactly what /analyze's own `forecast` field already shows.
    deterministic = run_rollout(model, context, normalizer, k=k, device=device, compute_explanations=False)
    deterministic_stage = deterministic.stage_predictions[-1].stage

    if samples is None:
        samples = run_mc_dropout_samples(model, context, normalizer, k, device, n_samples)
    n_samples = len(samples)

    outcome_counts: Counter[str] = Counter()
    outcome_prob_sums: dict[str, float] = {}
    for sample in samples:
        stage = sample.stage_predictions[-1].stage
        prob = sample.infiltration_probability[-1]
        outcome_counts[stage] += 1
        outcome_prob_sums[stage] = outcome_prob_sums.get(stage, 0.0) + prob

    outcomes = [
        FutureOutcome(
            stage=stage,
            probability=count / n_samples,
            n_samples=count,
            mean_infiltration_probability=outcome_prob_sums[stage] / count,
        )
        for stage, count in outcome_counts.items()
    ]
    outcomes.sort(key=lambda o: o.probability, reverse=True)

    return ParallelFuturesResult(n_samples=n_samples, outcomes=outcomes, deterministic_stage=deterministic_stage)
