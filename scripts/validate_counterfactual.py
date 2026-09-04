"""
Validates the Counterfactual Intervention Simulator (Phase 9, Feature 3)
against the real held-out day (Friday-Afternoon-DDoS): takes a real context
window that contains the known DDoS attack edge
(172.16.0.1 -> 192.168.10.50, confirmed by direct inspection of the day's
ground-truth edge labels), applies each defender action to it, re-runs the
existing K-step rollout on the modified graph, and reports the before/after
infiltration-probability comparison plus side effects.

This is a QUALITATIVE validation (there's no ground truth for "what would
the network have done if isolated" -- that counterfactual was never
observed), so it inspects whether the simulator behaves sensibly: does
removing the attacking host's edges measurably reduce predicted
infiltration probability relative to doing nothing, and do side effects
reflect the real structure of that window.

Usage:
    python scripts/validate_counterfactual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from counterfactual.simulator import InterventionAction, simulate_intervention
from graph.io import load_day
from models.train import resolve_device
from scripts.run_rollout import load_trained_model

ATTACKER_IP = "172.16.0.1"
VICTIM_IP = "192.168.10.50"


def _print_result(title: str, result) -> None:
    print(f"\n--- {title} ---")
    print(f"Action: {result.action.label}")
    print(f"Original probability (k steps):  {[f'{p:.3f}' for p in result.original_probability]}")
    print(f"Modified probability (k steps):  {[f'{p:.3f}' for p in result.modified_probability]}")
    print(f"Delta (modified - original):     {[f'{d:+.3f}' for d in result.probability_delta]}")
    mean_orig = sum(result.original_probability) / len(result.original_probability)
    mean_mod = sum(result.modified_probability) / len(result.modified_probability)
    diff = mean_mod - mean_orig
    # Note: removing ANY node changes the GNN's graph-level aggregation by a
    # small amount, even a totally uninvolved one -- so a exactly-zero delta
    # is not expected even for a "no-op" action. What matters is magnitude:
    # a real causal action should move probability by orders of magnitude
    # more than an unrelated one. Reported as a signed value, not collapsed
    # to a reduced/increased/unchanged label, so that comparison is visible.
    print(f"Mean probability: {mean_orig:.5f} -> {mean_mod:.5f} (delta {diff:+.5f})")
    print(f"Stage: {result.original_stage!r} -> {result.modified_stage!r}")
    print("Side effects:")
    for effect in result.side_effects:
        print(f"  - {effect}")


def main() -> None:
    device = resolve_device("auto")
    model, normalizer = load_trained_model(Path("checkpoints/phase3"), device)

    day = "Friday-Afternoon-DDoS"
    snapshots, _ = load_day(day, "data/processed/cicids2017")
    context_len = 20  # matches training seq_len

    # Find the first window at/after the known attack onset (window 26, per
    # scripts/validate_momentum.py's summary) with enough context behind it.
    attack_onset_idx = next(i for i, s in enumerate(snapshots) if s.graph_attack_present)
    ctx_end = max(attack_onset_idx + 3, context_len - 1)  # a few windows into the attack
    context = snapshots[ctx_end - context_len + 1 : ctx_end + 1]
    print(f"Context: windows {context[0].window_id}-{context[-1].window_id} of {day!r} "
          f"(attack onset at window {snapshots[attack_onset_idx].window_id})")
    print(f"Last context window contains attacker edge "
          f"{ATTACKER_IP in context[-1].node_ids and VICTIM_IP in context[-1].node_ids}")

    k = 5

    isolate_result = simulate_intervention(
        model, context, normalizer,
        InterventionAction(kind="isolate_host", host_ip=ATTACKER_IP),
        k=k, device=device,
    )
    _print_result(f"Isolate attacker host ({ATTACKER_IP})", isolate_result)

    block_result = simulate_intervention(
        model, context, normalizer,
        InterventionAction(kind="block_flow", src_ip=ATTACKER_IP, dst_ip=VICTIM_IP),
        k=k, device=device,
    )
    _print_result(f"Block flow ({ATTACKER_IP} <-> {VICTIM_IP})", block_result)

    # Negative control: isolating an unrelated, uninvolved host should have
    # ~no effect on predicted probability (sanity check that the simulator
    # isn't just always predicting "safer" regardless of which host is named).
    other_hosts = [h for h in context[-1].node_ids if h not in (ATTACKER_IP, VICTIM_IP)]
    if other_hosts:
        control_result = simulate_intervention(
            model, context, normalizer,
            InterventionAction(kind="isolate_host", host_ip=other_hosts[0]),
            k=k, device=device,
        )
        _print_result(f"Negative control: isolate unrelated host ({other_hosts[0]})", control_result)


if __name__ == "__main__":
    main()
