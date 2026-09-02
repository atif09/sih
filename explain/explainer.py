"""
Unified explanation record: combines attention-based flow attribution
(explain/attention.py), gradient-based feature attribution
(explain/saliency.py), and the MITRE stage mapping (already computed by
rollout/engine.py) into one structured, human-readable object.

This is the "no black-box outputs" entry point for Phase 5: every prediction
that goes through `explain_prediction()` ships with genuine model-internal
attribution attached (not the Phase 4 heuristic stub), covering both
"which flow" (attention) and "which feature of that flow" (saliency).

Deliberately reuses `rollout.engine.run_rollout(..., k=1)` for the
probability/attack-type/stage computation rather than reimplementing that
forward pass a third time -- this explains exactly the prediction
rollout/engine.py reports as step 1, by construction, not just "in sync by
convention".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from explain.attention import compute_edge_attention, top_edges_overall
from explain.saliency import compute_feature_saliency
from graph.snapshot import GraphSnapshot
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel
from rollout.engine import run_rollout
from rollout.mitre import StagePrediction


@dataclass
class ExplanationRecord:
    day: str
    context_end_window_id: int
    infiltration_probability: float
    stage: StagePrediction
    top_edges_by_attention: list[dict]
    top_node_features: list[tuple[str, float]]
    top_edge_features: list[tuple[str, float]]
    per_window_saliency: list[float]
    summary: str = field(default="")

    def __post_init__(self):
        if not self.summary:
            self.summary = _build_summary(self)


def _build_summary(record: "ExplanationRecord") -> str:
    parts = [
        f"Predicted {record.infiltration_probability:.0%} infiltration probability "
        f"for the window after {record.day} #{record.context_end_window_id}, "
        f"mapped to MITRE stage '{record.stage.stage}' "
        f"(basis: {record.stage.basis}"
        + (f", predicted type: {record.stage.predicted_attack_type}" if record.stage.predicted_attack_type else "")
        + ")."
    ]
    if record.top_edges_by_attention:
        top = record.top_edges_by_attention[0]
        parts.append(
            f"Most-attended flow: {top['src_ip']} -> {top['dst_ip']} "
            f"(attention={top['attention_weight']:.2f}, {int(top['flow_count'])} flows, "
            f"{top['bytes_sum']:.0f} bytes, {int(top['dst_port_nunique'])} distinct dst ports)."
        )
    if record.top_node_features:
        names = ", ".join(name for name, _ in record.top_node_features[:3])
        parts.append(f"Top contributing host-level features: {names}.")
    if record.top_edge_features:
        names = ", ".join(name for name, _ in record.top_edge_features[:3])
        parts.append(f"Top contributing flow-level features: {names}.")
    return " ".join(parts)


def explain_prediction(
    model: WorldModel,
    context_snapshots: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    device: torch.device,
    top_k_edges: int = 5,
    top_k_features: int = 10,
    infiltration_threshold: float = 0.5,
) -> ExplanationRecord:
    """Explains the model's prediction for the window immediately after
    context_snapshots (rollout/engine.py's step 1 for this same context)."""
    if len({s.day for s in context_snapshots}) > 1:
        raise ValueError("context_snapshots must all come from the same day")

    last = context_snapshots[-1]

    rollout_1step = run_rollout(model, context_snapshots, normalizer, k=1, device=device,
                                 infiltration_threshold=infiltration_threshold, compute_explanations=False)
    saliency = compute_feature_saliency(model, context_snapshots, normalizer, device, top_k=top_k_features)
    edge_records = compute_edge_attention(model, last, normalizer, device)
    top_edges = top_edges_overall(edge_records, top_k=top_k_edges)

    return ExplanationRecord(
        day=last.day, context_end_window_id=last.window_id,
        infiltration_probability=rollout_1step.infiltration_probability[0],
        stage=rollout_1step.stage_predictions[0],
        top_edges_by_attention=top_edges,
        top_node_features=saliency["top_node_features"],
        top_edge_features=saliency["top_edge_features"],
        per_window_saliency=saliency["per_window_saliency"],
    )
