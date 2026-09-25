/**
 * Types mirroring api/schemas.py exactly. Field names and shapes must match
 * the real, tested Phase 7 response -- no speculative fields for
 * not-yet-built backend features (multiple parallel futures, per-stage
 * confidence distributions, attack fingerprints are Phase 9, not this).
 */

export interface MitreStage {
  stage: string
  confidence: number
  basis: "attack_type" | "heuristic_fallback" | "benign"
  predicted_attack_type: string | null
}

export interface FlaggedFlow {
  src_ip: string
  dst_ip: string
  attention_weight: number
  attention_concentration: number
  flow_count: number
  bytes_sum: number
  dst_port_nunique: number
  retransmission_sum: number
  is_attack_edge: boolean
}

export interface TimelinePoint {
  window_id: number
  window_start: number
  infiltration_probability: number
  mitre_stage: MitreStage
  // Attack Future Replay (Phase 10): real per-window GNN attention, cheap
  // enough to compute for every observed window (unlike the full Phase 5
  // explanation, which stays attached to the final prediction only).
  top_flows: FlaggedFlow[]
}

export interface ForecastPoint {
  step: number
  infiltration_probability: number
  mitre_stage: MitreStage
}

export interface FeatureAttribution {
  feature_name: string
  score: number
}

export interface Explanation {
  summary: string
  top_edges_by_attention: FlaggedFlow[]
  top_node_features: FeatureAttribution[]
  top_edge_features: FeatureAttribution[]
  per_window_saliency: number[]
}

// Phase 9, Feature 1 -- see momentum/score.py. A rate-of-escalation metric
// (0-100), not a point-in-time severity score.
export interface MomentumScore {
  score: number
  top_driver: string
  explanation: string
  attack_path_expansion_available: boolean
}

// Phase 9, Feature 2 -- see containment/estimator.py. Ranges are null
// exactly when there's no historical data to estimate from (e.g. the
// current stage is benign, or genuinely unrecognized).
export interface ContainmentEstimate {
  current_stage: string
  likely_range_minutes: [number, number] | null
  full_observed_range_minutes: [number, number] | null
  n_historical_episodes: number
  has_sufficient_data: boolean
  recommended_action: string
  rationale: string
}

// Phase 10 differentiator -- Multiple Parallel Futures. See
// futures/parallel_futures.py: built via MC-Dropout resampling of the same
// context, not literally different observed starting conditions.
export interface FutureOutcome {
  stage: string
  probability: number
  n_samples: number
  mean_infiltration_probability: number
}

export interface MultipleFutures {
  n_samples: number
  outcomes: FutureOutcome[] // sorted descending by probability
  deterministic_stage: string
}

// Phase 10 differentiator -- Explainable Attack Fingerprint. Uses this
// model's real 6 MITRE stages as its dimensions, not the brief's
// illustrative ATT&CK tactic list -- see fingerprint/attack_fingerprint.py.
export interface StageFingerprintEntry {
  stage: string
  score: number
  similarity: "High" | "Medium" | "Low"
  driving_attack_type: string | null
}

export interface FingerprintComparisonEntry {
  stage: string
  current_score: number
  current_similarity: "High" | "Medium" | "Low"
  predicted_score: number
  predicted_similarity: "High" | "Medium" | "Low"
  trend: "rising" | "falling" | "stable"
}

export interface AttackFingerprint {
  current: StageFingerprintEntry[]
  predicted: StageFingerprintEntry[]
  comparison: FingerprintComparisonEntry[]
  top_stage: string
  summary: string
}

export interface AnalyzeResponse {
  filename: string
  file_type: "csv" | "pcap" | "csv+pcap"
  window_size_seconds: number
  num_windows: number
  context_len: number
  timeline: TimelinePoint[]
  forecast: ForecastPoint[]
  flagged_flows: FlaggedFlow[]
  explanation: Explanation
  momentum: MomentumScore
  containment: ContainmentEstimate
  multiple_futures: MultipleFutures
  attack_fingerprint: AttackFingerprint
  warnings: string[]
}

// Phase 9, Feature 3 -- see counterfactual/simulator.py + api/schemas.py's
// SimulateRequest/SimulateResponse. Acts on the context cached server-side
// from the most recent /analyze call.
export type InterventionActionKind = "isolate_host" | "block_flow"

export interface SimulateRequest {
  action: InterventionActionKind
  host_ip?: string
  src_ip?: string
  dst_ip?: string
  bidirectional?: boolean
  k?: number
}

export interface SimulateResponse {
  action_label: string
  original_probability: number[]
  modified_probability: number[]
  probability_delta: number[]
  original_stage: string
  modified_stage: string
  side_effects: string[]
}

// The five PS-mandated MITRE stages, in kill-chain order. Matches
// rollout/mitre.py's ATTACK_TYPE_TO_STAGE value set exactly.
export const MITRE_FIVE_STAGES = [
  "Reconnaissance",
  "Initial Access",
  "Lateral Movement",
  "Command and Control",
  "Exfiltration",
] as const

export type MitreFiveStage = (typeof MITRE_FIVE_STAGES)[number]

// The 5 canonical stages plus rollout/mitre.py's honest 6th label for
// traffic that doesn't fit the kill chain (STAGE_DOS_IMPACT) -- exact
// string match required, this is real backend vocabulary, not a UI label.
export const MITRE_SIX_STAGES = [...MITRE_FIVE_STAGES, "Denial of Service / Impact"] as const
export type MitreSixStage = (typeof MITRE_SIX_STAGES)[number]

export const STAGE_BENIGN = "Benign / No Stage"

// Model Performance section (Mode 1's pre-upload state) -- served from
// /model-performance/roc-pr, generated once by benchmark/run_benchmark.py.
// Model-level data (fixed at benchmark time), not tied to any upload.
export interface ROCCurve {
  fpr: number[]
  tpr: number[]
  thresholds: number[]
}

export interface PRCurve {
  precision: number[]
  recall: number[]
  thresholds: number[]
}

export interface ModelCurves {
  auc: number // ROC-AUC -- the ROC chart's summary metric only, NOT the PR chart's (use average_precision there)
  average_precision: number // PR chart's own summary metric, computed separately from auc
  roc: ROCCurve | null
  pr: PRCurve | null
}

export interface ROCPRCurvesResponse {
  held_out_day: string
  world_model: ModelCurves
  baseline: ModelCurves
}

// Confidence Calibration panel -- reliability diagram, world model only.
// Served from /model-performance/calibration.
export interface CalibrationCurve {
  predicted_confidence: number[]
  actual_frequency: number[]
  bin_counts: number[]
}

export interface CalibrationResponse {
  held_out_day: string
  n_bins: number
  world_model: CalibrationCurve | null
}
