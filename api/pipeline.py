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

from containment.estimator import StageDurationStats, build_stage_duration_stats, estimate_containment_window
from counterfactual.planner import plan_response
from counterfactual.simulator import InterventionAction, simulate_intervention
from explain.attention import compute_edge_attention, top_edges_overall
from explain.explainer import explain_prediction
from features.flow_features import extract_flow_features, load_cicflowmeter_csv
from features.packet_features import detect_port_scan_signatures, extract_packet_features
from features.pcap_to_flows import extract_flow_features_from_pcap
from features.windowing import merge_flow_and_packet_features
from fingerprint.attack_fingerprint import compute_attack_fingerprint
from futures.parallel_futures import run_mc_dropout_samples, sample_parallel_futures
from futures.time_to_compromise import compute_time_to_compromise
from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot, NODE_FEATURE_NAMES, build_snapshots_for_day
from models.dataset import load_all_days
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig
from models.forecast_decoder import STAT_NAMES, ForecastDecoder, compute_target_stats, inverse_transform_targets
from momentum.score import compute_momentum
from rollout.engine import run_rollout
from surprise.score import ALARM_PERCENTILE, SurpriseReference, compute_dynamics_surprise, is_informative_transition


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
    stage_duration_stats: dict[str, StageDurationStats]
    surprise_reference: SurpriseReference | None
    forecast_decoder: ForecastDecoder | None
    forecast_decoder_r2: dict[str, float] | None

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

        # Containment Window Estimator (Phase 9, Feature 2) statistics, built
        # ONCE at startup like the model itself -- these are ground-truth
        # episode-duration statistics pooled from every cached training day,
        # not something recomputed per request. Deliberately uses ALL 8 days
        # here (including the day held out from model *training*): this is
        # the production bundle, not a validation run -- the held-out split
        # only matters for measuring the world model's own generalization
        # (see scripts/validate_containment.py for that controlled
        # leave-one-out/held-out comparison); the containment estimator's
        # statistics benefit from every real episode available.
        window_size_seconds = pipeline_config["feature_extraction"]["window_size_seconds"]
        # Deployment ships a precomputed copy (checkpoint_dir/stage_duration_stats.json,
        # ~1KB, generated by scripts/export_stage_duration_stats.py from the
        # exact computation below) instead of the 214MB of processed day
        # caches. Local dev with the caches present recomputes from them.
        stats_path = checkpoint_dir / "stage_duration_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                raw_stats = json.load(f)
            stage_duration_stats = {
                stage: StageDurationStats(stage=stage, n_episodes=len(durs), durations_minutes=durs)
                for stage, durs in raw_stats.items()
            }
        else:
            day_snapshots = load_all_days(run_config["data"]["processed_dir"])
            stage_duration_stats = build_stage_duration_stats(day_snapshots, window_size_seconds)

        # Surprise Alarm reference distribution (see surprise/score.py) --
        # same precomputed-JSON-if-present pattern as stage_duration_stats
        # above. None (not an empty reference) when absent, so callers can
        # tell "no reference available" apart from "reference is empty."
        surprise_reference_path = checkpoint_dir / "surprise_reference.json"
        surprise_reference = SurpriseReference.load(surprise_reference_path) if surprise_reference_path.exists() else None

        # Network Forecast View decoder (see models/forecast_decoder.py) --
        # a small MLP trained SEPARATELY on top of this frozen model
        # (scripts/train_forecast_decoder.py), never touching best_model.pt.
        # Loaded only if both the weights and their real held-out validation
        # numbers exist -- forecast_decoder_r2 is surfaced in every response
        # so the frontend can show per-stat reliability instead of a bare
        # number with no honesty signal attached.
        decoder_path = checkpoint_dir / "forecast_decoder.pt"
        decoder_r2_path = checkpoint_dir / "forecast_decoder_validation.json"
        if decoder_path.exists() and decoder_r2_path.exists():
            forecast_decoder = ForecastDecoder(latent_dim=wm_config.latent_dim).to(device)
            forecast_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            forecast_decoder.eval()
            forecast_decoder_r2 = json.loads(decoder_r2_path.read_text())["held_out_r2"]
        else:
            forecast_decoder = None
            forecast_decoder_r2 = None

        return cls(
            model=model, normalizer=normalizer, device=device,
            window_size_seconds=window_size_seconds,
            seq_len=run_config["data"]["seq_len"],
            default_k=10,
            stage_duration_stats=stage_duration_stats,
            surprise_reference=surprise_reference,
            forecast_decoder=forecast_decoder,
            forecast_decoder_r2=forecast_decoder_r2,
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
) -> tuple[dict, list[GraphSnapshot]]:
    """Returns (response_dict, final_context) -- final_context is the exact
    context window used for the forecast/explanation above, returned
    alongside the response so the caller (api/main.py) can cache it for a
    later /simulate call (Phase 9, Feature 3) without re-running the whole
    feature-extraction pipeline. Not part of AnalyzeResponse itself."""
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
    # right after it using exactly context_len windows of history. The full
    # Phase 5 explanation (gradient saliency, which needs a backward pass
    # over the whole context) stays attached to the single final prediction
    # only -- that doesn't scale to every window. But attention is cheap:
    # compute_edge_attention is a single forward pass over ONE snapshot (no
    # backward, no temporal rollout), so computing it per window is
    # affordable and gives Attack Future Replay (the doc's #1 priority
    # differentiator) a REAL per-window graph view for every observed
    # window, not just the last one.
    timeline = []
    for i in range(context_len - 1, len(snapshots) - 1):
        context = snapshots[i - context_len + 1: i + 1]
        result = run_rollout(bundle.model, context, bundle.normalizer, k=1, device=bundle.device, compute_explanations=False)
        window_edges = compute_edge_attention(bundle.model, snapshots[i], bundle.normalizer, bundle.device)

        # Surprise Alarm (differentiator): real dynamics-prediction error for
        # this OBSERVED transition (snapshots[i] -> snapshots[i+1], the true
        # next window, already available here) -- see surprise/score.py.
        # None when the reference is missing, OR when this transition
        # touches an empty window (is_informative_transition) -- an empty
        # window's MSE is a constant "quiet network" artifact, not a real
        # novelty signal (see that function's module-docstring note on the
        # real false-alarm bug this excludes).
        if bundle.surprise_reference is not None and is_informative_transition(context, snapshots[i + 1]):
            surprise_mse = compute_dynamics_surprise(bundle.model, context, snapshots[i + 1], bundle.normalizer, bundle.device)
            surprise_percentile = bundle.surprise_reference.percentile_of(surprise_mse)
        else:
            surprise_percentile = None

        timeline.append({
            "window_id": snapshots[i + 1].window_id,
            "window_start": snapshots[i + 1].window_start,
            "infiltration_probability": result.infiltration_probability[0],
            "mitre_stage": _stage_to_dict(result.stage_predictions[0]),
            "top_flows": top_edges_overall(window_edges, top_k=3),
            "surprise_percentile": surprise_percentile,
            "surprise_alarm": surprise_percentile is not None and surprise_percentile >= ALARM_PERCENTILE,
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

    # Network Forecast View (differentiator): decode a real predicted
    # future latent back into human-readable stats. Step 5 (the brief's own
    # "now vs +5 min" framing) sits past this project's own disclosed
    # "trust steps 1-3" reliability window -- surfaced with its real
    # held-out R^2 per stat so the frontend can show that honestly, not
    # hide it. "now" stats are real ground truth from the actual current
    # window (final_context[-1]), no model involved.
    network_forecast = None
    if bundle.forecast_decoder is not None and forecast_result.predicted_latents:
        forecast_step_index = min(4, len(forecast_result.predicted_latents) - 1)  # step 5, 1-indexed -> index 4
        z_hat = torch.tensor(forecast_result.predicted_latents[forecast_step_index], device=bundle.device).unsqueeze(0)
        with torch.no_grad():
            decoded = inverse_transform_targets(bundle.forecast_decoder(z_hat).squeeze(0).cpu().numpy())
        now_stats_raw = compute_target_stats(final_context[-1], final_context[-1])
        network_forecast = {
            "step": forecast_step_index + 1,
            "now_stats": dict(zip(STAT_NAMES, [float(x) for x in now_stats_raw])),
            "predicted_stats": dict(zip(STAT_NAMES, [float(x) for x in decoded])),
            "held_out_r2": bundle.forecast_decoder_r2,
        }

    current_stage = forecast_result.stage_predictions[0].stage

    # Multiple Parallel Futures (Phase 10): MC-Dropout resampling of the
    # SAME final context -- reuses forecast_result's own k, no separate
    # rollout config. Computed before Momentum below so its real
    # path_expansion signal can feed in (see momentum/score.py).
    mc_dropout_samples = run_mc_dropout_samples(bundle.model, final_context, bundle.normalizer, k=k, device=bundle.device)
    futures_result = sample_parallel_futures(bundle.model, final_context, bundle.normalizer, k=k, device=bundle.device, samples=mc_dropout_samples)

    # Time-to-Compromise (differentiator): pure post-processing of the SAME
    # MC-Dropout ensemble above -- no extra model calls. See
    # futures/time_to_compromise.py's module docstring for the real bug fix
    # (MC-Dropout sampling was previously a silent no-op) this depends on.
    ttc_result = compute_time_to_compromise(mc_dropout_samples, k=k)

    # Momentum Score (Phase 9, Feature 1): the "current" moment is defined as
    # the freshest available prediction -- the walk-forward timeline PLUS the
    # forecast's own first step (one window more current than the timeline's
    # last entry, which stops one window short of the upload's true end).
    # compute_momentum() internally trims to its own lookback windows, so the
    # full timeline/snapshots lists can be passed as-is.
    momentum_result = compute_momentum(
        recent_probabilities=[t["infiltration_probability"] for t in timeline] + [forecast_result.infiltration_probability[0]],
        recent_stages=[t["mitre_stage"]["stage"] for t in timeline] + [current_stage],
        recent_snapshots=snapshots,
        rollout_result=forecast_result,
        futures_result=futures_result,
    )

    # Containment Window Estimator (Phase 9, Feature 2): keyed on that same
    # current stage, using explainability's own top feature names (already
    # computed above) for the recommended-action qualifier -- no separate
    # model, no separate explanation pass.
    top_feature_names = [n for n, _ in explanation.top_node_features] + [n for n, _ in explanation.top_edge_features]
    containment_estimate = estimate_containment_window(current_stage, bundle.stage_duration_stats, top_feature_names=top_feature_names)

    # Explainable Attack Fingerprint (Phase 10): "current" = the forecast's
    # own first-step attack_type_probs (same freshest-prediction convention
    # momentum/containment already use); "predicted" = mean across the
    # whole forecast horizon (computed inside compute_attack_fingerprint).
    fingerprint_result = compute_attack_fingerprint(
        current_attack_type_probs=forecast_result.attack_type_probs[0] if forecast_result.attack_type_probs else {},
        forecast_attack_type_probs=forecast_result.attack_type_probs,
    )

    # Surprise Alarm summary: "current" = the most recent observed timeline
    # entry (the forecast horizon has no ground truth to score surprise
    # against, so it stops at the same point the walk-forward timeline
    # does -- see surprise/score.py's module docstring).
    scored_entries = [t for t in timeline if t["surprise_percentile"] is not None]
    if timeline and scored_entries:
        # "Current" = the most recent SCORED entry, not necessarily
        # timeline[-1] -- the freshest window may itself be an empty,
        # unscored transition while earlier windows in the same capture
        # were scored.
        current_surprise = scored_entries[-1]["surprise_percentile"]
        max_surprise = max(t["surprise_percentile"] for t in scored_entries)
        alarm_raised = scored_entries[-1]["surprise_alarm"]
        if bundle.surprise_reference is None:
            surprise_explanation = "No surprise reference distribution shipped with this checkpoint -- run scripts/export_surprise_reference.py."
        elif alarm_raised:
            surprise_explanation = f"Most recent observed window scored at the {current_surprise:.1f}th percentile of real benign-to-benign behavior -- the model's own dynamics prediction was more wrong here than {current_surprise:.0f}% of normal transitions in the training data."
        else:
            surprise_explanation = f"Most recent observed window scored at the {current_surprise:.1f}th percentile of real benign-to-benign behavior -- within the range the model's dynamics normally expect."
        surprise_summary = {
            "current_percentile": current_surprise,
            "alarm_raised": alarm_raised,
            "max_percentile_in_window": max_surprise,
            "explanation": surprise_explanation,
        }
    elif not timeline:
        surprise_summary = {
            "current_percentile": None,
            "alarm_raised": False,
            "max_percentile_in_window": None,
            "explanation": "No observed timeline entries to score (upload has exactly context_len windows, none left over after context) -- Surprise Alarm needs at least one real observed transition.",
        }
    else:
        surprise_summary = {
            "current_percentile": None,
            "alarm_raised": False,
            "max_percentile_in_window": None,
            "explanation": "Every observed transition in this capture touched an empty (zero-flow) window -- Surprise Alarm excludes those as uninformative (see surprise/score.py).",
        }

    result = {
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
        "momentum": {
            "score": momentum_result.score,
            "top_driver": momentum_result.top_driver,
            "explanation": momentum_result.explanation,
            "attack_path_expansion_available": momentum_result.signals.attack_path_expansion_available,
        },
        "containment": {
            "current_stage": containment_estimate.current_stage,
            "likely_range_minutes": list(containment_estimate.likely_range_minutes) if containment_estimate.likely_range_minutes else None,
            "full_observed_range_minutes": list(containment_estimate.full_observed_range_minutes) if containment_estimate.full_observed_range_minutes else None,
            "n_historical_episodes": containment_estimate.n_historical_episodes,
            "has_sufficient_data": containment_estimate.has_sufficient_data,
            "recommended_action": containment_estimate.recommended_action,
            "rationale": containment_estimate.rationale,
        },
        "multiple_futures": {
            "n_samples": futures_result.n_samples,
            "outcomes": [
                {
                    "stage": o.stage, "probability": o.probability,
                    "n_samples": o.n_samples, "mean_infiltration_probability": o.mean_infiltration_probability,
                }
                for o in futures_result.outcomes
            ],
            "deterministic_stage": futures_result.deterministic_stage,
        },
        "attack_fingerprint": {
            "current": [
                {"stage": e.stage, "score": e.score, "similarity": e.similarity, "driving_attack_type": e.driving_attack_type}
                for e in fingerprint_result.current
            ],
            "predicted": [
                {"stage": e.stage, "score": e.score, "similarity": e.similarity, "driving_attack_type": e.driving_attack_type}
                for e in fingerprint_result.predicted
            ],
            "comparison": [
                {
                    "stage": e.stage, "current_score": e.current_score, "current_similarity": e.current_similarity,
                    "predicted_score": e.predicted_score, "predicted_similarity": e.predicted_similarity, "trend": e.trend,
                }
                for e in fingerprint_result.comparison
            ],
            "top_stage": fingerprint_result.top_stage,
            "summary": fingerprint_result.summary,
        },
        "surprise_alarm": surprise_summary,
        "time_to_compromise": {
            "n_samples": ttc_result.n_samples,
            "threshold": ttc_result.threshold,
            "cumulative_fraction": ttc_result.cumulative_fraction,
            "median_crossing_step": ttc_result.median_crossing_step,
            "never_crossed_fraction": ttc_result.never_crossed_fraction,
        },
        "network_forecast": network_forecast,
        "warnings": warnings,
    }
    return result, final_context


def run_simulation(
    bundle: ModelBundle,
    context: list[GraphSnapshot],
    action_kind: str,
    host_ip: str | None,
    src_ip: str | None,
    dst_ip: str | None,
    bidirectional: bool,
    k: int | None,
) -> dict:
    """Phase 9, Feature 3: apply a defender action to `context` (the cached
    context from the most recent /analyze call) and re-run the rollout on
    the modified graph. No new model, no re-extraction -- reuses the exact
    same context snapshots and the same run_rollout entry point as /analyze."""
    action = InterventionAction(
        kind=action_kind, host_ip=host_ip, src_ip=src_ip, dst_ip=dst_ip, bidirectional=bidirectional,
    )
    k = k if k is not None else bundle.default_k
    sim = simulate_intervention(bundle.model, context, bundle.normalizer, action, k=k, device=bundle.device)
    return {
        "action_label": sim.action.label,
        "original_probability": sim.original_probability,
        "modified_probability": sim.modified_probability,
        "probability_delta": sim.probability_delta,
        "original_stage": sim.original_stage,
        "modified_stage": sim.modified_stage,
        "side_effects": sim.side_effects,
    }


def run_response_plan(bundle: ModelBundle, context: list[GraphSnapshot], k: int | None) -> dict:
    """Response Planner (differentiator): candidates are built from the SAME
    real attention-ranked flows /analyze itself surfaces as `flagged_flows`
    (recomputed fresh here from `context[-1]`, rather than requiring the
    caller to have cached them separately -- compute_edge_attention is a
    cheap single forward pass, the same cost /analyze already pays for this
    exact computation)."""
    k = k if k is not None else bundle.default_k
    edge_records = compute_edge_attention(bundle.model, context[-1], bundle.normalizer, bundle.device)
    flagged_flows = top_edges_overall(edge_records, top_k=5)

    if not flagged_flows:
        return {"actions": [], "warning": "No flagged flows in the current context to build candidate actions from."}

    planned = plan_response(bundle.model, context, bundle.normalizer, flagged_flows, k=k, device=bundle.device)
    return {
        "actions": [
            {
                "action_label": p.action.label,
                "risk_reduction": p.risk_reduction,
                "normal_flows_disrupted": p.normal_flows_disrupted,
                "original_stage": p.original_stage,
                "modified_stage": p.modified_stage,
                "side_effects": p.side_effects,
            }
            for p in planned
        ],
        "warning": (
            "This model was never trained on intervened graphs -- real measured effect sizes are typically "
            "hundredths of a probability point. Read this ranking as 'which action had any measurable effect,' "
            "not as a statistically validated recommendation (see counterfactual/planner.py's module docstring)."
        ),
    }
