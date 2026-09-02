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

    warnings: list[str] = Field(default_factory=list, description="Non-fatal notices, e.g. packet-level features unavailable, short capture")


class ErrorResponse(BaseModel):
    detail: str
