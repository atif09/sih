"""
Pydantic request/response schemas for the /analyze endpoint.

Maps directly to the PS's requested output: "infiltration probability
timeline, flagged flows, MITRE ATT&CK stage predictions, and explainability
data (feature attributions per prediction)".
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MitreStage(BaseModel):
    stage: str = Field(description="One of the PS's five stages, the 6th 'Denial of Service / Impact' label, or 'Benign / No Stage'")
    confidence: float
    basis: str = Field(description="'attack_type' (model's own prediction), 'heuristic_fallback' (raw-feature heuristic), or 'benign'")
    predicted_attack_type: str | None = Field(default=None, description="Only set when basis == 'attack_type'")


class FlaggedFlow(BaseModel):
    src_ip: str
    dst_ip: str
    attention_weight: float = Field(description="Real GNN attention weight (Phase 5) -- see note on cross-target comparability below")
    attention_concentration: float = Field(description="attention_weight * num_competitors; the field this list is actually ranked by")
    flow_count: float
    bytes_sum: float
    dst_port_nunique: float
    retransmission_sum: float
    is_attack_edge: bool = Field(description="Only meaningful for labeled reference data (e.g. CIC-IDS2017); always False for a real unlabeled upload -- not a prediction")


class TimelinePoint(BaseModel):
    window_id: int
    window_start: float = Field(description="Epoch seconds, start of this time window")
    infiltration_probability: float = Field(description="Model's predicted probability that the NEXT window (window_id + 1) contains attack activity")
    mitre_stage: MitreStage
    top_flows: list[FlaggedFlow] = Field(
        default_factory=list,
        description="Attack Future Replay: real GNN attention over THIS window's own observed graph (cheap forward-pass-only attention, not the full Phase 5 gradient explanation, which stays attached to the final prediction only). Lets a replay scrubber show genuine per-window structure for every OBSERVED window.",
    )


class ForecastPoint(BaseModel):
    step: int = Field(description="1-indexed step beyond the last observed window (K-step rollout, Phase 4)")
    infiltration_probability: float
    mitre_stage: MitreStage


class FeatureAttribution(BaseModel):
    feature_name: str
    score: float = Field(description="Input x Gradient attribution score (Phase 5), non-negative")


class Explanation(BaseModel):
    summary: str = Field(description="Human-readable synthesis of the explanation below")
    top_edges_by_attention: list[FlaggedFlow]
    top_node_features: list[FeatureAttribution]
    top_edge_features: list[FeatureAttribution]
    per_window_saliency: list[float] = Field(description="Which window in the context contributed most gradient signal, oldest -> newest")


class MomentumScore(BaseModel):
    """Phase 9, Feature 1 -- Cyber Threat Momentum Score. A rate-of-escalation
    metric (0-100), not a point-in-time severity score -- see momentum/score.py."""

    score: float = Field(description="0-100. How fast risk is escalating right now, not how severe it already is")
    top_driver: str = Field(description="Which underlying signal is driving the score (see `explanation`)")
    explanation: str = Field(description="Human-readable summary naming the top driver and its raw value")
    attack_path_expansion_available: bool = Field(description="True whenever a real Multiple Parallel Futures ensemble (Phase 10) fed the 'expansion of predicted attack paths' signal; False only if that ensemble wasn't provided to this computation")


class ContainmentEstimate(BaseModel):
    """Phase 9, Feature 2 -- Containment Window Estimator. Estimates
    time-to-containment for the CURRENT predicted stage from real historical
    episode durations (see containment/estimator.py's reframing note: this
    dataset has no genuine stage-to-different-stage transitions to measure)."""

    current_stage: str
    likely_range_minutes: list[float] | None = Field(default=None, description="[p25, p90] minutes -- the 'typical' band. None if the current stage is benign/unrecognized or has no historical data")
    full_observed_range_minutes: list[float] | None = Field(default=None, description="[min, max] minutes observed historically for this stage")
    n_historical_episodes: int = Field(description="How many historical episodes of this stage the range is built from")
    has_sufficient_data: bool = Field(description="False if fewer than 3 historical episodes -- range should be treated with caution, not hidden")
    recommended_action: str = Field(description="Simple, rules-based recommended action for this stage, not a model prediction")
    rationale: str


class FutureOutcome(BaseModel):
    stage: str
    probability: float = Field(description="Fraction of MC-Dropout samples landing on this stage at the forecast horizon, in [0, 1]")
    n_samples: int = Field(description="Raw sample count backing `probability` -- a low count on a long-tail outcome is a real signal to treat that probability with caution")
    mean_infiltration_probability: float = Field(description="Mean final-step infiltration probability among samples with this outcome")


class MultipleFutures(BaseModel):
    """Phase 10 differentiator -- Multiple Parallel Futures. Built via
    MC-Dropout resampling of the SAME context (see futures/parallel_futures.py
    for the full method and its honest limitations), not a claim of
    literally different observed starting conditions."""

    n_samples: int
    outcomes: list[FutureOutcome] = Field(description="Sorted descending by probability")
    deterministic_stage: str = Field(description="The single no-dropout (eval-mode) forecast outcome -- same as forecast[-1].mitre_stage.stage, given here for cross-reference")


class StageFingerprintEntry(BaseModel):
    stage: str
    score: float = Field(description="Max attack-type probability mapping to this stage, in [0, 1] -- see fingerprint/attack_fingerprint.py for why max, not sum")
    similarity: str = Field(description="'High' | 'Medium' | 'Low'")
    driving_attack_type: str | None = Field(default=None, description="Which attack type produced `score`; null if no attack type maps to this stage at all (always true for Exfiltration -- see module docstring)")


class FingerprintComparisonEntry(BaseModel):
    stage: str
    current_score: float
    current_similarity: str
    predicted_score: float
    predicted_similarity: str
    trend: str = Field(description="'rising' | 'falling' | 'stable'")


class AttackFingerprint(BaseModel):
    """Phase 10 differentiator -- Explainable Attack Fingerprint. Uses this
    model's real 6 MITRE stages as its dimensions, not the brief's
    illustrative ATT&CK tactic list (Credential Access, Persistence, ...)
    -- see fingerprint/attack_fingerprint.py's module docstring for why."""

    current: list[StageFingerprintEntry] = Field(description="Sorted descending by score")
    predicted: list[StageFingerprintEntry] = Field(description="Sorted descending by score; mean attack-type probabilities across the whole K-step forecast")
    comparison: list[FingerprintComparisonEntry] = Field(description="Sorted by |predicted - current| descending")
    top_stage: str
    summary: str


class AnalyzeResponse(BaseModel):
    filename: str
    file_type: str = Field(description="'csv', 'pcap', or 'csv+pcap'")
    window_size_seconds: float
    num_windows: int = Field(description="Total time windows extracted from the uploaded capture")
    context_len: int = Field(description="Number of windows actually used as model context (min(configured seq_len, num_windows))")

    timeline: list[TimelinePoint] = Field(description="Walk-forward: one entry per observed window with enough preceding context, predicting the window right after it")
    forecast: list[ForecastPoint] = Field(description="K-step-ahead rollout beyond the last observed window -- genuine forward simulation, no ground truth available")

    flagged_flows: list[FlaggedFlow] = Field(description="Real GNN-attention-ranked flows from the final (most recent) context window")
    explanation: Explanation = Field(description="Full Phase 5 explanation attached to the final prediction -- the PS's 'explainability data (feature attributions per prediction)'")

    momentum: MomentumScore = Field(description="Phase 9, Feature 1")
    containment: ContainmentEstimate = Field(description="Phase 9, Feature 2")
    multiple_futures: MultipleFutures = Field(description="Phase 10 differentiator")
    attack_fingerprint: AttackFingerprint = Field(description="Phase 10 differentiator")

    warnings: list[str] = Field(default_factory=list, description="Non-fatal notices, e.g. packet-level features unavailable, short capture")


class ErrorResponse(BaseModel):
    detail: str


class BenchmarkRow(BaseModel):
    """One model's metrics on one evaluation region -- see
    benchmark/run_benchmark.py's `compute_metrics`. Field set matches that
    function's real output exactly; nothing here is computed by the API,
    only re-served from benchmark/report.json."""

    model: str = Field(description="'World Model' or 'Baseline (LogReg)'")
    n: int = Field(description="Number of examples this row's metrics were computed over")
    positive_rate: float = Field(description="Fraction of these examples that were true attack windows -- class balance context for reading the metrics below")
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float = Field(description="False positive rate")
    roc_auc: float


class ModelValidationResponse(BaseModel):
    """Phase 6 benchmark (world model vs. logistic-regression baseline),
    served live from the console's Model Validation panel rather than only
    existing as static prose on the marketing page. This is a MODEL-level
    property, computed once by benchmark/run_benchmark.py against the fixed
    val/held-out regions -- not recomputed per request, since it has nothing
    to do with whatever capture a user uploads to /analyze. See
    benchmark/report.md for the full narrative version of these same
    numbers."""

    held_out_day: str = Field(description="The day excluded entirely from training -- its attack type is genuinely unseen by the model. This is the PS's 'held-out generalization test', evaluated on real CIC-IDS2017 data. CTU-13 (mentioned in CLAUDE.md as a SECOND, separate held-out generalization test) is a different dataset/schema and is not yet integrated -- see `ctu13_status`.")
    seq_len: int = Field(description="World model's configured temporal context length, in windows -- the baseline sees exactly 1 window per prediction, by design, so this is the capability the benchmark exists to measure")
    validation: list[BenchmarkRow] = Field(description="In-distribution: later chronological chunk of each non-held-out day, seen attack types")
    held_out: list[BenchmarkRow] = Field(description="The held-out day above: unseen attack type, the PS's core generalization claim")
    single_class_val_days: list[str] = Field(description="Non-held-out days whose validation chunk contains NO attack windows at all (their campaigns ended before capture end) -- disclosed because it explains why pooled validation AUC reads lower than the held-out day's")
    multi_class_val_days: list[str] = Field(description="The only validation days with real class variation to measure ranking quality on")
    ctu13_status: str = Field(default="CTU-13 (secondary held-out generalization test, different schema) is not yet integrated into this pipeline.", description="Honest status of the CLAUDE.md-listed secondary dataset -- stated plainly rather than displaying a fabricated result for it")


class SimulateRequest(BaseModel):
    """Phase 9, Feature 3 -- Counterfactual Intervention Simulator. Acts on
    the context window from the MOST RECENT /analyze call in this server
    process (there is no separate upload here -- this is an offline,
    single-session demo, not a multi-tenant API). Call /analyze first."""

    action: str = Field(description="'isolate_host' or 'block_flow'")
    host_ip: str | None = Field(default=None, description="Required for action='isolate_host'")
    src_ip: str | None = Field(default=None, description="Required for action='block_flow'")
    dst_ip: str | None = Field(default=None, description="Required for action='block_flow'")
    bidirectional: bool = Field(default=True, description="For action='block_flow': also remove the reverse-direction edge")
    k: int | None = Field(default=None, description="Rollout horizon; defaults to the model bundle's default_k")


class SimulateResponse(BaseModel):
    action_label: str = Field(description="Human-readable description of the applied action")
    original_probability: list[float] = Field(description="K-step infiltration probability with NO intervention (unmodified graph)")
    modified_probability: list[float] = Field(description="K-step infiltration probability AFTER the action is applied")
    probability_delta: list[float] = Field(description="modified - original, per step; negative means the action reduced predicted risk")
    original_stage: str = Field(description="Predicted MITRE stage one step ahead, unmodified graph")
    modified_stage: str = Field(description="Predicted MITRE stage one step ahead, modified graph")
    side_effects: list[str] = Field(description="Simple, rules-based trade-off notes about the action -- not a model prediction")
