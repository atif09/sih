"""
Counterfactual Intervention Simulator (Phase 9, Feature 3): apply a
defender action to the CURRENT graph state, re-run the existing K-step
rollout engine on the modified graph, and compare against the unmodified
("do nothing") outcome. No new model, no retraining -- reuses the trained
world model and rollout.engine.run_rollout exactly as Phase 4 built them,
only the INPUT graph changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from counterfactual.graph_ops import block_flow, isolate_host
from graph.snapshot import EDGE_FEATURE_NAMES, GraphSnapshot
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel
from rollout.engine import run_rollout

_FLOW_COUNT_IDX = EDGE_FEATURE_NAMES.index("flow_count")


@dataclass
class InterventionAction:
    kind: str  # "isolate_host" | "block_flow"
    host_ip: str | None = None  # for isolate_host
    src_ip: str | None = None  # for block_flow
    dst_ip: str | None = None  # for block_flow
    bidirectional: bool = True  # for block_flow

    @property
    def label(self) -> str:
        if self.kind == "isolate_host":
            return f"Isolate host {self.host_ip}"
        if self.kind == "block_flow":
            return f"Block flow {self.src_ip} <-> {self.dst_ip}" if self.bidirectional else f"Block flow {self.src_ip} -> {self.dst_ip}"
        return self.kind


@dataclass
class SimulationResult:
    action: InterventionAction
    original_probability: list[float]
    modified_probability: list[float]
    probability_delta: list[float]  # modified - original, per step (negative = action helped)
    original_stage: str
    modified_stage: str
    side_effects: list[str] = field(default_factory=list)


def apply_action(context: list[GraphSnapshot], action: InterventionAction) -> list[GraphSnapshot]:
    if action.kind == "isolate_host":
        if not action.host_ip:
            raise ValueError("isolate_host requires host_ip")
        return [isolate_host(s, action.host_ip) for s in context]
    if action.kind == "block_flow":
        if not action.src_ip or not action.dst_ip:
            raise ValueError("block_flow requires src_ip and dst_ip")
        return [block_flow(s, action.src_ip, action.dst_ip, action.bidirectional) for s in context]
    raise ValueError(f"Unknown action kind: {action.kind!r} (expected 'isolate_host' or 'block_flow')")


def count_normal_flows_disrupted(action: InterventionAction, current_snapshot: GraphSnapshot) -> int:
    """Real count of non-attack-labeled edges this action would cut in the
    CURRENT snapshot -- the same computation _side_effects' text warning is
    built from, exposed as a plain int so the Response Planner (see
    counterfactual/planner.py) can rank actions by it without re-parsing a
    sentence. isolate_host: every OTHER (non-flagged) edge touching the
    host. block_flow: this specific edge, 0 if ground truth marks it
    attack traffic (nothing "normal" is being cut)."""
    if action.kind == "isolate_host":
        host = action.host_ip
        if host not in current_snapshot.node_ids:
            return 0
        idx = current_snapshot.node_ids.index(host)
        touching = (current_snapshot.edge_index[0] == idx) | (current_snapshot.edge_index[1] == idx)
        total_touching = int(touching.sum())
        flagged_touching = int((touching & (current_snapshot.edge_labels == 1)).sum())
        return total_touching - flagged_touching

    if action.kind == "block_flow":
        src_idx = current_snapshot.node_ids.index(action.src_ip) if action.src_ip in current_snapshot.node_ids else None
        dst_idx = current_snapshot.node_ids.index(action.dst_ip) if action.dst_ip in current_snapshot.node_ids else None
        if src_idx is None or dst_idx is None:
            return 0
        forward = (current_snapshot.edge_index[0] == src_idx) & (current_snapshot.edge_index[1] == dst_idx)
        match = forward | ((current_snapshot.edge_index[0] == dst_idx) & (current_snapshot.edge_index[1] == src_idx)) if action.bidirectional else forward
        non_attack = match & (current_snapshot.edge_labels == 0)
        return int(non_attack.sum())

    return 0


def _side_effects(action: InterventionAction, current_snapshot: GraphSnapshot) -> list[str]:
    """Simple, rules-based trade-off notes -- not a model. Counts real
    structure in the current snapshot (e.g. how many OTHER, non-flagged
    edges a host has) rather than a generic disclaimer, so the warning is
    specific to this actual state."""
    effects: list[str] = []

    if action.kind == "isolate_host":
        host = action.host_ip
        if host in current_snapshot.node_ids:
            benign_touching = count_normal_flows_disrupted(action, current_snapshot)
            if benign_touching > 0:
                effects.append(
                    f"Isolating {host} would also cut {benign_touching} other flow(s) on this host "
                    f"not flagged as attack traffic -- verify these aren't legitimate services before acting."
                )
            else:
                effects.append(f"No other (non-flagged) traffic observed on {host} in the current window.")
        else:
            effects.append(f"{host} is not active in the current window -- isolating it has no effect right now.")

    elif action.kind == "block_flow":
        effects.append(
            f"Only traffic between {action.src_ip} and {action.dst_ip} is blocked; "
            f"either host may still reach the network through other paths, and an attacker "
            f"controlling {action.src_ip} could pivot to a different target."
        )

    return effects


@torch.no_grad()
def simulate_intervention(
    model: WorldModel,
    context: list[GraphSnapshot],
    normalizer: FeatureNormalizer,
    action: InterventionAction,
    k: int,
    device: torch.device,
) -> SimulationResult:
    original = run_rollout(model, context, normalizer, k=k, device=device, compute_explanations=False)
    modified_context = apply_action(context, action)
    modified = run_rollout(model, modified_context, normalizer, k=k, device=device, compute_explanations=False)

    delta = [m - o for m, o in zip(modified.infiltration_probability, original.infiltration_probability)]
    side_effects = _side_effects(action, context[-1])

    return SimulationResult(
        action=action,
        original_probability=original.infiltration_probability,
        modified_probability=modified.infiltration_probability,
        probability_delta=delta,
        original_stage=original.stage_predictions[0].stage,
        modified_stage=modified.stage_predictions[0].stage,
        side_effects=side_effects,
    )
