"""
Full inference pipeline orchestration: feature extraction -> graph
construction -> world model inference -> K-step rollout -> MITRE stage
mapping -> explainability, wrapped as one function the FastAPI endpoint
calls. Every step reuses existing Phase 1-5 code unchanged -- this module
does not reimplement any pipeline logic, only sequences it.

Input handling (see the module docstring in features/pcap_to_flows.py for
the PCAP-only design decision):
  - CSV only: used directly as flow-level features (Phase 1).
  - PCAP only: converted to a flow-level CSV via cicflowmeter first
    (features/pcap_to_flows.py), THEN run through the identical flow-level
    extraction path -- the model never knows the difference. The same raw
    PCAP is also run through features/packet_features.py for packet-level
    augmentation (TTL, retransmissions, port-scan signature).
  - Both: the user-supplied CSV is used for flow-level features (assumed
    authoritative/already-processed); the PCAP is still used for
    packet-level augmentation of the same traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from explain.explainer import explain_prediction
from features.flow_features import extract_flow_features, load_cicflowmeter_csv
from features.packet_features import detect_port_scan_signatures, extract_packet_features
from features.pcap_to_flows import extract_flow_features_from_pcap
from features.windowing import merge_flow_and_packet_features
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, build_snapshots_for_day
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from rollout.engine import run_rollout


class InsufficientDataError(ValueError):
    """Raised when the uploaded file doesn't produce enough time windows to
    run inference (need at least 2: one to serve as context, one to predict).
    A 4xx-worthy error, not a 500 -- the pipeline itself is working correctly,
    the input just isn't large/long enough."""


@dataclass
class ModelBundle:
    model: WorldModel
    normalizer: FeatureNormalizer
    device: torch.device
    window_size_seconds: float
    seq_len: int
    default_k: int

    @classmethod
    def load(cls, checkpoint_dir: str | Path, pipeline_config_path: str | Path, device: torch.device) -> "ModelBundle":
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "run_config.yaml") as f:
            run_config = yaml.safe_load(f)
        with open(checkpoint_dir / "attack_type_vocab.json") as f:
            attack_type_vocab = json.load(f)
        with open(pipeline_config_path) as f:
            pipeline_config = yaml.safe_load(f)

        model_cfg = run_config["model"]
        wm_config = WorldModelConfig(
            node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
            latent_dim=model_cfg["latent_dim"], gnn_hidden_dim=model_cfg["gnn_hidden_dim"],
            gnn_num_layers=model_cfg["gnn_num_layers"], gnn_heads=model_cfg["gnn_heads"],
            temporal_type=model_cfg["temporal_type"], temporal_hidden_dim=model_cfg["temporal_hidden_dim"],
            temporal_num_layers=model_cfg["temporal_num_layers"], temporal_num_heads=model_cfg["temporal_num_heads"],
            dropout=model_cfg["dropout"], attack_type_vocab=attack_type_vocab,
        )
        model = WorldModel(wm_config).to(device)
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
        model.eval()

        normalizer = FeatureNormalizer.load(checkpoint_dir / "normalizer.pkl")

        return cls(
            model=model, normalizer=normalizer, device=device,
            window_size_seconds=pipeline_config["feature_extraction"]["window_size_seconds"],
            seq_len=run_config["data"]["seq_len"],
            default_k=10,
        )


def _load_flow_features(csv_path: Path | None, pcap_path: Path | None) -> tuple:
    """Returns (flow_features_df, file_type_str, warnings)."""
    warnings: list[str] = []
    if csv_path is not None:
        raw = load_cicflowmeter_csv(csv_path)
        flow_feats = extract_flow_features(raw)
        file_type = "csv+pcap" if pcap_path is not None else "csv"
    elif pcap_path is not None:
        flow_feats = extract_flow_features_from_pcap(pcap_path)
        file_type = "pcap"
        warnings.append(
            "Flow-level features were derived from the PCAP via cicflowmeter "
            "(no CSV was supplied)."
        )
    else:
        raise ValueError("At least one of csv_path or pcap_path must be provided")
    return flow_feats, file_type, warnings


def run_pipeline(
    bundle: ModelBundle,
    csv_path: Path | None,
    pcap_path: Path | None,
    k: int | None = None,
) -> dict:
    flow_feats, file_type, warnings = _load_flow_features(csv_path, pcap_path)

    packet_df = port_scan_df = None
    if pcap_path is not None:
        packet_df = extract_packet_features(pcap_path)
        port_scan_df = detect_port_scan_signatures(packet_df, bundle.window_size_seconds)
    else:
        warnings.append("No PCAP provided -- packet-level features (TTL, retransmissions, port-scan signature) are zero-filled.")

    merged = merge_flow_and_packet_features(
        flow_feats, packet_df, port_scan_df, window_size_seconds=bundle.window_size_seconds,
    )
    snapshots = build_snapshots_for_day(merged, day="upload", window_size_seconds=bundle.window_size_seconds)

    if len(snapshots) < 2:
        raise InsufficientDataError(
            f"Uploaded file produced only {len(snapshots)} time window(s); need at least 2 "
            f"(window_size_seconds={bundle.window_size_seconds}). Upload a longer capture or "
            f"reduce the window size."
        )

    context_len = min(bundle.seq_len, len(snapshots))
    if context_len < bundle.seq_len:
        warnings.append(
            f"Uploaded file has only {len(snapshots)} window(s), fewer than the model's "
            f"configured context length (seq_len={bundle.seq_len}). Using all {context_len} "
            f"available windows as context -- predictions may be less reliable than with a "
            f"full-length context."
        )

    k = k if k is not None else bundle.default_k

    def _stage_to_dict(stage) -> dict:
        # rollout.mitre.predict_stage returns a StagePrediction dataclass,
        # not a plain dict -- Pydantic v2 doesn't auto-coerce an arbitrary
        # dataclass into a BaseModel field without from_attributes, so this
        # conversion is required, not cosmetic (found via a genuine 500 in
        # testing: FastAPI raised a validation error trying to serialize a
        # StagePrediction directly into the MitreStage response field).
        return {
            "stage": stage.stage, "confidence": stage.confidence,
            "basis": stage.basis, "predicted_attack_type": stage.predicted_attack_type,
        }

    # Walk-forward timeline: for every valid end position, predict the window
    # right after it using exactly context_len windows of history (no
    # explanation computed per-step -- that's attached once, below, to the
    # single most decision-relevant prediction, both for response size and
    # because computing a fresh gradient/attention pass per window doesn't
    # scale to long captures).
    timeline = []
    for i in range(context_len - 1, len(snapshots) - 1):
        context = snapshots[i - context_len + 1: i + 1]
        result = run_rollout(bundle.model, context, bundle.normalizer, k=1, device=bundle.device, compute_explanations=False)
        timeline.append({
            "window_id": snapshots[i + 1].window_id,
            "window_start": snapshots[i + 1].window_start,
            "infiltration_probability": result.infiltration_probability[0],
            "mitre_stage": _stage_to_dict(result.stage_predictions[0]),
        })

    # Final K-step forecast + full explanation, from the very end of the
    # uploaded capture -- genuine forward simulation into the unobserved
    # future, with real attention/saliency attached (Phase 4 + Phase 5).
    final_context = snapshots[-context_len:]
    forecast_result = run_rollout(bundle.model, final_context, bundle.normalizer, k=k, device=bundle.device, compute_explanations=True)
    explanation = explain_prediction(bundle.model, final_context, bundle.normalizer, bundle.device)

    forecast = [
        {"step": step, "infiltration_probability": prob, "mitre_stage": _stage_to_dict(stage)}
        for step, (prob, stage) in enumerate(
            zip(forecast_result.infiltration_probability, forecast_result.stage_predictions), start=1,
        )
    ]

    return {
        "file_type": file_type,
        "window_size_seconds": bundle.window_size_seconds,
        "num_windows": len(snapshots),
        "context_len": context_len,
        "timeline": timeline,
        "forecast": forecast,
        "flagged_flows": forecast_result.attention_flows,
        "explanation": {
            "summary": explanation.summary,
            "top_edges_by_attention": explanation.top_edges_by_attention,
            "top_node_features": [{"feature_name": n, "score": s} for n, s in explanation.top_node_features],
            "top_edge_features": [{"feature_name": n, "score": s} for n, s in explanation.top_edge_features],
            "per_window_saliency": explanation.per_window_saliency,
        },
        "warnings": warnings,
    }
