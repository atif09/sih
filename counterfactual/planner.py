"""
Response Planner (differentiator): automates what the Counterfactual
Intervention Simulator already does one action at a time. Generates every
sensible candidate action from the flows the model's own attention actually
flagged, runs simulate_intervention() on each, and ranks the results by
real risk reduced vs. real normal traffic disrupted. No new model, no new
rollout logic -- this loops the existing, already-tested simulator.

Honest limitation (disclosed, not hidden)
--------------------------------------------
This project's own counterfactual simulator has already shown, on real
captures, that single-host/single-flow interventions typically move the
forecast probability by hundredths of a point (see CounterfactualPanel's
own isNearNoop handling on the frontend) -- the model was never trained on
intervened graphs, so its sensitivity to a removed host or flow is whatever
falls out of the GNN's learned representation, not something calibrated to
be meaningful at this granularity. A ranked list built on such small deltas
should be read as "which of these had *any* measurable effect," not as a
confident, statistically validated recommendation -- the ranking has NOT
been checked for stability against repeated runs or minor perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from counterfactual.simulator import InterventionAction, SimulationResult, count_normal_flows_disrupted, simulate_intervention
from graph.snapshot import GraphSnapshot
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel


@dataclass
class PlannedAction:
    action: InterventionAction
    risk_reduction: float  # mean(original) - mean(modified) across the k-step rollout; positive = action helped
    normal_flows_disrupted: int
    original_stage: str
    modified_stage: str
    side_effects: list[str]


def generate_candidate_actions(flagged_flows: list[dict]) -> list[InterventionAction]:
    """Every sensible action derivable from the flows the model's own
    attention surfaced (api/pipeline.py's flagged_flows) -- not every host
    in the graph, which would include hosts the model gave no reason to
    suspect. One isolate_host per unique host, one block_flow per unique
    (src, dst) pair, deduplicated in order of first appearance."""
    seen_hosts: list[str] = []
    seen_pairs: list[tuple[str, str]] = []
    for f in flagged_flows:
        src, dst = f["src_ip"], f["dst_ip"]
        if src not in seen_hosts:
            seen_hosts.append(src)
        if dst not in seen_hosts:
            seen_hosts.append(dst)
        pair = (src, dst)
        if pair not in seen_pairs:
            seen_pairs.append(pair)

    actions: list[InterventionAction] = [InterventionAction(kind="isolate_host", host_ip=h) for h in seen_hosts]
    actions += [InterventionAction(kind="block_flow", src_ip=s, dst_ip=d, bidirectional=True) for s, d in seen_pairs]
    return actions


@torch.no_grad()
def plan_response(
    model: WorldModel,
    context: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    flagged_flows: list[dict],
    k: int,
    device: torch.device,
) -> list[PlannedAction]:
    """Runs every candidate action (see generate_candidate_actions) through
    the existing simulator and ranks by risk_reduction descending, ties
    broken by fewer normal flows disrupted. Returns [] if there are no
    flagged flows to build candidates from (e.g. a benign capture)."""
    actions = generate_candidate_actions(flagged_flows)
    current_snapshot = context[-1]

    planned: list[PlannedAction] = []
    for action in actions:
        result: SimulationResult = simulate_intervention(model, context, normalizer, action, k=k, device=device)
        mean_original = sum(result.original_probability) / len(result.original_probability)
        mean_modified = sum(result.modified_probability) / len(result.modified_probability)
        planned.append(
            PlannedAction(
                action=action,
                risk_reduction=mean_original - mean_modified,
                normal_flows_disrupted=count_normal_flows_disrupted(action, current_snapshot),
                original_stage=result.original_stage,
                modified_stage=result.modified_stage,
                side_effects=result.side_effects,
            )
        )

    planned.sort(key=lambda p: (-p.risk_reduction, p.normal_flows_disrupted))
    return planned
