"""
Explainable Attack Fingerprint (differentiator feature, last priority per
the reference brief): summarize the attack's behavioral pattern as a
per-stage "similarity" score, and compare the CURRENT fingerprint against
the PREDICTED FUTURE one.

A disclosed reframing of the brief's illustrative categories
------------------------------------------------------------------
The brief's example uses ATT&CK *tactic* names (Credential Access,
Persistence, Data Exfiltration...) that don't all correspond to something
this model actually predicts -- its real, trained vocabulary is the 6
MITRE STAGE labels already used everywhere else in this project
(rollout/mitre.py), not the fuller ATT&CK tactic taxonomy. Inventing
separate "Persistence" or "Credential Access" scores this model has no
signal for would be fabrication. This fingerprint uses the model's own 6
real stages as its dimensions instead -- the same honest substitution
Phase 9's other features already made when the brief's illustrative
categories didn't match this project's real output classes.

Where the numbers come from
-------------------------------
`model.attack_type_head` already outputs one independent probability per
CIC-IDS2017 attack type (a real multi-label sigmoid head, trained
end-to-end -- see rollout/engine.py's `attack_type_probs`). Each attack
type maps to one MITRE stage via the SAME `ATTACK_TYPE_TO_STAGE` table
every other feature in this project uses. A stage's fingerprint score is
the MAX probability among attack types that map to it (not a sum -- these
are independent per-class sigmoid outputs, not a normalized distribution,
so summing could exceed 1 and misrepresent the signal; MAX asks "does any
real evidence point to this stage" without that distortion).

Exfiltration has NO attack type mapped to it anywhere in
ATTACK_TYPE_TO_STAGE (documented in rollout/mitre.py: CIC-IDS2017 has no
clean exfiltration category, reachable only through the raw-feature
heuristic fallback). Its fingerprint score is therefore always 0.0 here,
with that reason stated explicitly rather than silently showing a
misleadingly-confident zero.

"Current" vs "predicted future" (brief's own comparison, not a fabricated
extra): current = the freshest available prediction's attack_type_probs
(the K-step forecast's own first step, same "now" convention Momentum and
Containment already use). Predicted future = the FINAL K-step forecast
step's attack_type_probs -- "where this ends up by the end of the
forecast horizon," the same convention Multiple Parallel Futures already
uses for "the predicted outcome" (futures/parallel_futures.py's
`deterministic_stage` is likewise the LAST rollout step's stage, not an
average across steps). Originally this used the mean across the whole
horizon instead; changed to the endpoint for consistency with that other
feature, and because averaging in the current step diluted an already
small real signal (see the threshold note below).

Trend threshold, checked against real numbers rather than picked blind
------------------------------------------------------------------------------
Real attack-type probabilities on the held-out day were directly inspected
before choosing this constant: e.g. one active stage's driving attack type
moved from 0.0506 at step 1 to 0.0463 at step 10 across a real k=10
rollout -- a genuine, consistent decay (mirroring the well-documented
infiltration-probability decay toward its own fixed point), but only
~0.004 of absolute movement, because this model's attack-type head is
already low-confidence on this unseen-attack-type day (see the baseline
investigation in README's Phase 6 section). The original threshold (0.05)
was picked by analogy to `rollout/mitre.py`'s unrelated
TYPE_CONFIDENCE_THRESHOLD without checking this, and made "stable" the
answer for every stage on every real capture tried, regardless of real
(if small) directional movement -- a threshold that never fires isn't
telling the analyst anything. Lowered to a value real deltas of this
model's actual scale can cross.
"""

from __future__ import annotations

from dataclasses import dataclass

from rollout.mitre import ATTACK_TYPE_TO_STAGE, STAGE_EXFILTRATION

HIGH_THRESHOLD = 0.5  # matches rollout/mitre.py's own TYPE_CONFIDENCE_THRESHOLD
MEDIUM_THRESHOLD = 0.2
STABLE_DELTA_THRESHOLD = 0.003

STAGES = sorted(set(ATTACK_TYPE_TO_STAGE.values()) | {STAGE_EXFILTRATION})


def _similarity_label(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


@dataclass
class StageFingerprintEntry:
    stage: str
    score: float  # in [0, 1]; max attack-type probability mapping to this stage
    similarity: str  # "High" | "Medium" | "Low"
    driving_attack_type: str | None  # which attack type produced `score`, or None if no signal exists


@dataclass
class FingerprintComparisonEntry:
    stage: str
    current_score: float
    current_similarity: str
    predicted_score: float
    predicted_similarity: str
    trend: str  # "rising" | "falling" | "stable"


@dataclass
class AttackFingerprintResult:
    current: list[StageFingerprintEntry]
    predicted: list[StageFingerprintEntry]
    comparison: list[FingerprintComparisonEntry]  # sorted by |delta| descending
    top_stage: str  # the current fingerprint's highest-scoring stage
    summary: str


def _stage_fingerprint(attack_type_probs: dict[str, float]) -> list[StageFingerprintEntry]:
    entries = []
    for stage in STAGES:
        types_for_stage = [t for t, s in ATTACK_TYPE_TO_STAGE.items() if s == stage]
        candidates = {t: attack_type_probs.get(t, 0.0) for t in types_for_stage if t in attack_type_probs}
        if not candidates:
            entries.append(StageFingerprintEntry(stage=stage, score=0.0, similarity=_similarity_label(0.0), driving_attack_type=None))
            continue
        driver, score = max(candidates.items(), key=lambda kv: kv[1])
        entries.append(StageFingerprintEntry(stage=stage, score=score, similarity=_similarity_label(score), driving_attack_type=driver))
    entries.sort(key=lambda e: e.score, reverse=True)
    return entries


def compute_attack_fingerprint(
    current_attack_type_probs: dict[str, float],
    forecast_attack_type_probs: list[dict[str, float]],
) -> AttackFingerprintResult:
    """current_attack_type_probs: the freshest single-step prediction (K-step
    forecast's own step 1). forecast_attack_type_probs: the attack_type_probs
    list across the WHOLE K-step forecast -- only the LAST step is used as
    "predicted future" (see module docstring for why: consistency with
    Multiple Parallel Futures' own endpoint convention, and to avoid
    diluting an already-small real signal by averaging in the current
    step)."""
    current = _stage_fingerprint(current_attack_type_probs)
    predicted_probs = forecast_attack_type_probs[-1] if forecast_attack_type_probs else {}
    predicted = _stage_fingerprint(predicted_probs)

    current_by_stage = {e.stage: e for e in current}
    predicted_by_stage = {e.stage: e for e in predicted}

    comparison = []
    for stage in STAGES:
        c, p = current_by_stage[stage], predicted_by_stage[stage]
        delta = p.score - c.score
        trend = "stable" if abs(delta) < STABLE_DELTA_THRESHOLD else ("rising" if delta > 0 else "falling")
        comparison.append(FingerprintComparisonEntry(
            stage=stage, current_score=c.score, current_similarity=c.similarity,
            predicted_score=p.score, predicted_similarity=p.similarity, trend=trend,
        ))
    comparison.sort(key=lambda e: abs(e.predicted_score - e.current_score), reverse=True)

    top_stage = current[0].stage if current else STAGE_EXFILTRATION
    top_entry = current_by_stage[top_stage]
    if top_entry.driving_attack_type:
        summary = (
            f"Current behavior most resembles '{top_stage}' ({top_entry.similarity.lower()} similarity, "
            f"driven by the model's '{top_entry.driving_attack_type}' signal)."
        )
    else:
        summary = f"Current behavior shows no strong resemblance to any stage (top: '{top_stage}', {top_entry.similarity.lower()})."

    return AttackFingerprintResult(current=current, predicted=predicted, comparison=comparison, top_stage=top_stage, summary=summary)
