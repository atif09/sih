"""
Cyber Threat Momentum Score (Phase 9, Feature 1): measures how FAST a threat
is escalating, not just how severe it is right now -- two incidents at the
same point-in-time risk level can be in very different situations depending
on whether that risk just spiked or has been flat for an hour.

Deliberately a pure post-processing/derived metric -- no new model, no
training. Every signal below is computed from things Phases 1-5 already
produce: the walk-forward infiltration-probability history (Phase 3/4), raw
graph-snapshot structure (Phase 2), MITRE stage history (Phase 4), and the
K-step rollout's own trajectory (Phase 4).

Signal design note: every input is either the model's OWN prediction or a
directly-observable graph statistic -- never a ground-truth label. This
metric must work on a real, unlabeled upload the same way it works on
labeled validation data; using ground truth to compute it would make it
unusable in the one place it matters (a live analysis), even though ground
truth IS used to validate it against the held-out day (see
scripts/validate_momentum.py) -- that's an evaluation-time check, not an
input to the metric.

Signals used (see MomentumSignals):
  1. probability_slope: rate of change of the model's own predicted
     infiltration probability over recent observed windows -- "is risk
     visibly rising right now, or flat."
  2. newly_affected_host_ratio: fraction of hosts active in the most recent
     window that were NOT active in the preceding lookback window(s) -- a
     directly observable graph-structure signal (no labels needed) for
     "is the footprint of this activity expanding."
  3. stage_progression_rate: how often the predicted MITRE stage has
     CHANGED across recent windows -- a network sitting in one stage for
     20 windows has near-zero momentum even if that stage is severe; one
     that's changed stage 3 times in 5 windows is moving fast.
  4. forecast_trend: slope of the rollout engine's OWN K-step forecast
     trajectory -- "does the model's own forward simulation expect this to
     keep climbing."

NOT included, and why (one of two features explicitly requested in the
brief -- the other, "expansion of predicted attack paths," USED to be
missing too, but is now wired to the real Multiple Parallel Futures
feature (futures/parallel_futures.py) once that shipped -- see
`path_expansion` below):
  - "Changes in privilege level" -- this project's feature set is derived
    from NetFlow/CICFlowMeter-style flow records, which contain no
    authentication or privilege information at all (no user/role/auth
    fields anywhere in the schema). Not approximated with a proxy;
    genuinely omitted, and reported as such.

Path expansion (added once Multiple Parallel Futures existed)
-------------------------------------------------------------------
`path_expansion = 1 - probability(most likely outcome)`, from a REAL
MC-Dropout ensemble over the forecast horizon (see
futures/parallel_futures.py) -- not the single deterministic rollout path
this signal fell back to before that feature existed. A confident,
single-outcome forecast (top probability near 1.0) scores 0 here (no
expansion); a forecast where plausible outcomes have genuinely spread
across several stages scores higher. Already bounded to [0, 1] by
construction, so it needs no saturation constant like the other raw
signals below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from futures.parallel_futures import ParallelFuturesResult
from graph.snapshot import GraphSnapshot
from rollout.engine import RolloutResult

# Clip ranges used to normalize each raw signal into [0, 1] before
# weighting. These are documented, reasoned constants, not fit to data:
# a probability swing of +-0.05 per window is already a large move given
# Phase 3's observed probabilities cluster narrowly around 0.45-0.56 (see
# README's Phase 4 section), so +-0.05/window is treated as saturating.
PROBABILITY_SLOPE_SATURATION = 0.05
STAGE_HISTORY_LOOKBACK = 6
HOST_LOOKBACK_WINDOWS = 3

# Weights, sum to 1.0. probability_slope carries the most weight since it's
# the most direct, least noisy signal (aggregates the whole network state,
# not just host membership); documented rather than tuned against this
# project's single held-out day (a genuine hyperparameter search would need
# far more held-out incidents than this project has). Rebalanced (not just
# appended to) when path_expansion was added, so the set still sums to 1.0.
WEIGHTS = {
    "probability_slope": 0.35,
    "newly_affected_host_ratio": 0.20,
    "stage_progression_rate": 0.15,
    "forecast_trend": 0.15,
    "path_expansion": 0.15,
}


@dataclass
class MomentumSignals:
    probability_slope: float  # raw units: probability change per window
    newly_affected_host_ratio: float  # in [0, 1]
    stage_progression_rate: float  # in [0, 1]
    forecast_trend: float  # raw units: probability change per rollout step
    path_expansion: float = 0.0  # in [0, 1]; see module docstring
    attack_path_expansion_available: bool = False  # True once a real futures_result was provided


@dataclass
class MomentumResult:
    score: float  # 0-100
    signals: MomentumSignals
    top_driver: str
    explanation: str


def _slope(values: list[float]) -> float:
    """Least-squares slope of values against their position index. Returns
    0.0 for fewer than 2 points (no trend is computable)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    # np.polyfit degree-1 gives [slope, intercept]; a constant series (zero
    # variance in x, which can't happen here since x is always 0..n-1) is
    # not a concern, but a constant y is fine and correctly yields slope 0.
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _normalize_slope(slope: float, saturation: float) -> float:
    """Maps a raw slope to [0, 1], saturating at +-`saturation`. A negative
    slope (risk falling) maps to 0, not a negative number -- momentum is
    "how fast is this escalating," and a de-escalating trend has zero
    escalation momentum, not negative momentum."""
    return float(np.clip(slope / saturation, 0.0, 1.0))


def _newly_affected_host_ratio(recent_snapshots: list[GraphSnapshot]) -> float:
    """Fraction of the LAST snapshot's active hosts that were not active in
    any of the preceding lookback snapshots. 0.0 if the last snapshot is
    empty (nothing to be "newly affected") or there's no prior history."""
    if len(recent_snapshots) < 2:
        return 0.0
    *history, latest = recent_snapshots[-(HOST_LOOKBACK_WINDOWS + 1):]
    if latest.num_nodes == 0:
        return 0.0
    seen_before: set[str] = set()
    for snap in history:
        seen_before.update(snap.node_ids)
    latest_hosts = set(latest.node_ids)
    new_hosts = latest_hosts - seen_before
    return len(new_hosts) / len(latest_hosts)


def _stage_progression_rate(recent_stages: list[str]) -> float:
    """Fraction of consecutive window-pairs in recent history where the
    predicted MITRE stage changed. 0.0 if fewer than 2 stages available."""
    if len(recent_stages) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(recent_stages, recent_stages[1:]) if a != b)
    return changes / (len(recent_stages) - 1)


def compute_momentum(
    recent_probabilities: list[float],
    recent_stages: list[str],
    recent_snapshots: list[GraphSnapshot],
    rollout_result: RolloutResult | None = None,
    futures_result: ParallelFuturesResult | None = None,
) -> MomentumResult:
    """
    recent_probabilities: walk-forward infiltration_probability values for
        the last N *observed* windows, oldest -> newest (e.g. from the
        API's `timeline`). At least 2 needed for a meaningful slope.
    recent_stages: the corresponding predicted MITRE stage string per
        window, same length/order as recent_probabilities.
    recent_snapshots: the GraphSnapshot objects underlying those same
        windows (at least HOST_LOOKBACK_WINDOWS + 1 for a meaningful
        newly-affected-host signal; degrades gracefully with fewer).
    rollout_result: optional K-step forecast (Phase 4) for the
        forward-looking `forecast_trend` signal. If omitted, that signal
        is 0 and excluded from the explanation (not silently
        mis-weighted -- see the renormalization below).
    futures_result: optional Multiple Parallel Futures ensemble (Phase 10)
        for the `path_expansion` signal. If omitted, that signal is
        excluded the same way rollout_result's is, and
        `attack_path_expansion_available` reports False.
    """
    prob_history = recent_probabilities[-STAGE_HISTORY_LOOKBACK:]
    stage_history = recent_stages[-STAGE_HISTORY_LOOKBACK:]

    raw_prob_slope = _slope(prob_history)
    host_ratio = _newly_affected_host_ratio(recent_snapshots)
    stage_rate = _stage_progression_rate(stage_history)
    raw_forecast_slope = _slope(rollout_result.infiltration_probability) if rollout_result else 0.0
    path_expansion = (
        float(np.clip(1.0 - futures_result.outcomes[0].probability, 0.0, 1.0))
        if futures_result and futures_result.outcomes else 0.0
    )

    signals = MomentumSignals(
        probability_slope=raw_prob_slope,
        newly_affected_host_ratio=host_ratio,
        stage_progression_rate=stage_rate,
        forecast_trend=raw_forecast_slope,
        path_expansion=path_expansion,
        attack_path_expansion_available=futures_result is not None,
    )

    normalized = {
        "probability_slope": _normalize_slope(raw_prob_slope, PROBABILITY_SLOPE_SATURATION),
        "newly_affected_host_ratio": host_ratio,
        "stage_progression_rate": stage_rate,
        "forecast_trend": _normalize_slope(raw_forecast_slope, PROBABILITY_SLOPE_SATURATION),
        "path_expansion": path_expansion,
    }

    # If a signal's real input wasn't provided, it's structurally
    # unavailable (not just zero) -- exclude its weight from the
    # denominator rather than let it silently drag the score down.
    active_weights = dict(WEIGHTS)
    if rollout_result is None:
        active_weights.pop("forecast_trend")
    if futures_result is None:
        active_weights.pop("path_expansion")
    weight_sum = sum(active_weights.values())

    contributions = {k: normalized[k] * w for k, w in active_weights.items()}
    score = 100.0 * sum(contributions.values()) / weight_sum

    top_driver = max(contributions, key=contributions.get) if contributions else "none"
    explanation = _build_explanation(signals, top_driver, active_weights)

    return MomentumResult(score=round(score, 1), signals=signals, top_driver=top_driver, explanation=explanation)


_SIGNAL_LABELS = {
    "probability_slope": "rising infiltration probability",
    "newly_affected_host_ratio": "newly active hosts",
    "stage_progression_rate": "frequent MITRE stage changes",
    "forecast_trend": "a rollout forecast that keeps climbing",
    "path_expansion": "predicted outcomes spreading across multiple plausible futures",
}


def _build_explanation(signals: MomentumSignals, top_driver: str, active_weights: dict) -> str:
    parts = [f"Momentum driven primarily by {_SIGNAL_LABELS[top_driver]}"]
    detail = {
        "probability_slope": f"{signals.probability_slope:+.4f}/window",
        "newly_affected_host_ratio": f"{signals.newly_affected_host_ratio:.0%} of active hosts are new",
        "stage_progression_rate": f"stage changed in {signals.stage_progression_rate:.0%} of recent windows",
        "forecast_trend": f"{signals.forecast_trend:+.4f}/step",
        "path_expansion": f"top predicted outcome only {(1 - signals.path_expansion):.0%} likely",
    }
    parts.append(f"({detail[top_driver]})")
    if not signals.attack_path_expansion_available:
        parts.append("[note: path-expansion signal unavailable this run -- Multiple Parallel Futures wasn't provided]")
    return " ".join(parts)
