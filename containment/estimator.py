"""
Containment Window Estimator (Phase 9, Feature 2): given the current
predicted stage, estimate how much time remains before the attack episode
is likely to end (contained, or the attacker moves on) -- surfaced as a
time RANGE grounded in real historical transition data, not a fixed
heuristic.

A real, disclosed reframing of the original brief
-----------------------------------------------------
The brief asks for "a time range to reach each plausible next stage" (i.e.
stage-to-stage transition times, e.g. Reconnaissance -> Lateral Movement).
Checked this against the real ground truth first: every one of the 8
CIC-IDS2017 capture days contains exactly ONE attack category for its
entire duration (confirmed by mapping each window's real Label-derived
attack type through `rollout.mitre.ATTACK_TYPE_TO_STAGE` and looking for
stage-to-different-stage transitions -- there are none, anywhere, in this
dataset). What DOES exist, richly, is repeated Benign <-> [stage] flips
within a day (e.g. Wednesday's DoS traffic oscillates on and off 21 times).
So "time to the next stage" isn't a question this dataset can answer
honestly -- but "how long does an episode of stage X typically last before
it ends" is, and has real sample sizes (Wednesday alone contributes 21
episodes). That's what this module estimates: not kill-chain progression
time, but time-to-containment for the CURRENT stage. This is disclosed
here rather than silently substituted for the original ask.

Ground truth, not model predictions
--------------------------------------
`build_stage_duration_stats` uses each window's REAL attack-type label
(`GraphSnapshot.graph_attack_types`, mapped through the same
ATTACK_TYPE_TO_STAGE table Phase 4 uses for predictions) -- this is
training-time ground truth statistics, computed once and reused at
inference, not a live signal. At inference (`estimate_containment_window`),
only the currently PREDICTED stage (a string) is needed as input -- no
ground truth required to use the estimator on a real, unlabeled upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graph.snapshot import GraphSnapshot
from rollout.mitre import ATTACK_TYPE_TO_STAGE, STAGE_BENIGN

MIN_EPISODES_FOR_CONFIDENCE = 3


def ground_truth_stage(snapshot: GraphSnapshot) -> str:
    """Real (label-derived) stage for one window -- independent of any
    model prediction. Mirrors rollout.mitre's stage vocabulary."""
    if not snapshot.graph_attack_present:
        return STAGE_BENIGN
    mapped = {ATTACK_TYPE_TO_STAGE.get(t) for t in snapshot.graph_attack_types}
    mapped.discard(None)
    if not mapped:
        return STAGE_BENIGN
    return "+".join(sorted(mapped))  # a window can rarely map to >1 stage if multiple attack types co-occur


@dataclass
class StageDurationStats:
    stage: str
    n_episodes: int
    durations_minutes: list[float] = field(repr=False)

    @property
    def median_minutes(self) -> float:
        return float(np.median(self.durations_minutes))

    @property
    def likely_range_minutes(self) -> tuple[float, float]:
        """p25-p90 -- the "typical" band, per requirement 3 (a range, not a
        point). Not mean +- std: episode durations here are heavily
        right-skewed (many 1-window blips, a few long episodes), so a
        percentile band is a more honest summary than a symmetric one."""
        return (
            float(np.percentile(self.durations_minutes, 25)),
            float(np.percentile(self.durations_minutes, 90)),
        )

    @property
    def full_observed_range_minutes(self) -> tuple[float, float]:
        return (float(np.min(self.durations_minutes)), float(np.max(self.durations_minutes)))

    @property
    def has_sufficient_data(self) -> bool:
        return self.n_episodes >= MIN_EPISODES_FOR_CONFIDENCE


def build_stage_duration_stats(
    day_snapshots: dict[str, list[GraphSnapshot]],
    window_size_seconds: float,
    exclude_days: set[str] = frozenset(),
) -> dict[str, StageDurationStats]:
    """Ground-truth episode-duration statistics per stage, pooled across
    every non-excluded day. An "episode" is a maximal run of consecutive
    windows sharing the same non-benign ground-truth stage."""
    durations_by_stage: dict[str, list[float]] = {}

    for day, snapshots in day_snapshots.items():
        if day in exclude_days:
            continue
        stages = [ground_truth_stage(s) for s in snapshots]
        i = 0
        while i < len(stages):
            if stages[i] == STAGE_BENIGN:
                i += 1
                continue
            j = i
            while j < len(stages) and stages[j] == stages[i]:
                j += 1
            duration_minutes = (j - i) * window_size_seconds / 60.0
            durations_by_stage.setdefault(stages[i], []).append(duration_minutes)
            i = j

    return {
        stage: StageDurationStats(stage=stage, n_episodes=len(durs), durations_minutes=durs)
        for stage, durs in durations_by_stage.items()
    }


# Simple, rules-based recommended action per stage -- deliberately NOT a
# model (per requirement 5). Feature-name substrings are matched against
# the top explainability features (Phase 5) to add a one-line qualifier
# when a recognizable signal is present, without attempting real NLG.
ACTION_TABLE: dict[str, str] = {
    "Reconnaissance": "Rate-limit or block the scanning source; increase monitoring on the scanned hosts.",
    "Initial Access": "Reset/revoke the targeted credentials; block the source IP at the perimeter.",
    "Lateral Movement": "Segment or isolate the affected hosts; revoke session tokens for the pivoting account.",
    "Command and Control": "Isolate the suspected beacon host; block the destination IP/port at the firewall.",
    "Exfiltration": "Block outbound traffic to the destination; isolate the host immediately.",
    "Denial of Service / Impact": "Engage rate-limiting/DDoS mitigation; consider isolating the targeted host.",
}

_FEATURE_QUALIFIERS: list[tuple[str, str]] = [
    ("port_nunique", "driven by a scan-like spread across many destination ports"),
    ("unique_ports", "driven by a scan-like spread across many destination ports"),
    ("bytes_sum", "driven by an unusually large data volume"),
    ("retransmission", "driven by elevated retransmissions"),
    ("syn_cnt", "driven by a SYN-heavy flag pattern"),
]


@dataclass
class ContainmentEstimate:
    current_stage: str
    likely_range_minutes: tuple[float, float] | None
    full_observed_range_minutes: tuple[float, float] | None
    n_historical_episodes: int
    has_sufficient_data: bool
    recommended_action: str
    rationale: str


def estimate_containment_window(
    current_stage: str,
    stats: dict[str, StageDurationStats],
    top_feature_names: list[str] | None = None,
) -> ContainmentEstimate:
    if current_stage == STAGE_BENIGN or current_stage not in ACTION_TABLE:
        return ContainmentEstimate(
            current_stage=current_stage, likely_range_minutes=None, full_observed_range_minutes=None,
            n_historical_episodes=0, has_sufficient_data=False,
            recommended_action="No action needed -- current state is not an active attack stage.",
            rationale="Containment windows only apply once a specific attack stage has been predicted.",
        )

    stage_stats = stats.get(current_stage)
    action = ACTION_TABLE[current_stage]
    qualifier = ""
    if top_feature_names:
        for name in top_feature_names:
            for substr, text in _FEATURE_QUALIFIERS:
                if substr in name:
                    qualifier = f" ({text})"
                    break
            if qualifier:
                break

    if stage_stats is None:
        return ContainmentEstimate(
            current_stage=current_stage, likely_range_minutes=None, full_observed_range_minutes=None,
            n_historical_episodes=0, has_sufficient_data=False,
            recommended_action=action + qualifier,
            rationale=f"No historical '{current_stage}' episodes in the training data to estimate a window from.",
        )

    rationale = (
        f"Based on {stage_stats.n_episodes} historical '{current_stage}' episode(s): "
        f"typical duration {stage_stats.likely_range_minutes[0]:.1f}-{stage_stats.likely_range_minutes[1]:.1f} min "
        f"(observed range {stage_stats.full_observed_range_minutes[0]:.1f}-{stage_stats.full_observed_range_minutes[1]:.1f} min)."
    )
    if not stage_stats.has_sufficient_data:
        rationale += f" NOTE: fewer than {MIN_EPISODES_FOR_CONFIDENCE} historical episodes -- treat this range with caution."

    return ContainmentEstimate(
        current_stage=current_stage,
        likely_range_minutes=stage_stats.likely_range_minutes,
        full_observed_range_minutes=stage_stats.full_observed_range_minutes,
        n_historical_episodes=stage_stats.n_episodes,
        has_sufficient_data=stage_stats.has_sufficient_data,
        recommended_action=action + qualifier,
        rationale=rationale,
    )
