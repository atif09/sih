"""
Validates the Momentum Score (Phase 9, Feature 1) against the real held-out
day: computes momentum walk-forward across the ENTIRE day using only
model-predicted/graph-observable signals (never ground truth -- see
momentum/score.py's docstring), then compares the resulting trend against
the day's real ground-truth attack labels purely for evaluation.

Usage:
    python scripts/validate_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.io import load_day
from momentum.score import STAGE_HISTORY_LOOKBACK, compute_momentum
from models.train import resolve_device
from rollout.engine import run_rollout
from scripts.run_rollout import load_trained_model


def main() -> None:
    device = resolve_device("auto")
    model, normalizer = load_trained_model(Path("checkpoints/phase3"), device)

    day = "Friday-Afternoon-DDoS"
    snapshots, _ = load_day(day, "data/processed/cicids2017")
    context_len = model.config.attack_type_vocab and 20 or 20  # matches training seq_len (20)

    # Walk-forward: probability/stage for every window with enough context.
    prob_history: list[float] = []
    stage_history: list[str] = []
    window_ids: list[int] = []
    with torch.no_grad():
        for i in range(context_len - 1, len(snapshots) - 1):
            context = snapshots[i - context_len + 1 : i + 1]
            result = run_rollout(model, context, normalizer, k=1, device=device, compute_explanations=False)
            prob_history.append(result.infiltration_probability[0])
            stage_history.append(result.stage_predictions[0].stage)
            window_ids.append(snapshots[i + 1].window_id)

    print(f"{'window':>7} {'truth':>6} {'prob':>7} {'stage':>22} {'momentum':>9} {'top_driver':>26}")
    momentum_scores = []
    for idx, wid in enumerate(window_ids):
        lo = max(0, idx - STAGE_HISTORY_LOOKBACK + 1)
        recent_probs = prob_history[lo : idx + 1]
        recent_stages = stage_history[lo : idx + 1]
        # Snapshot window matching the recent_probs/stages range (offset by
        # context_len-1 since prob_history[0] corresponds to snapshots[context_len-1]).
        snap_lo = context_len - 1 + lo
        recent_snaps = snapshots[snap_lo : context_len + idx]

        # Also compute a short forward rollout from "now" for the forecast_trend signal.
        ctx_end = context_len - 1 + idx
        context = snapshots[ctx_end - context_len + 1 : ctx_end + 1]
        rollout_result = run_rollout(model, context, normalizer, k=5, device=device, compute_explanations=False)

        momentum = compute_momentum(recent_probs, recent_stages, recent_snaps, rollout_result)
        momentum_scores.append(momentum.score)

        truth = "ATTACK" if snapshots[ctx_end + 1].graph_attack_present else "-"
        if idx % 2 == 0 or truth == "ATTACK":  # print every other row + all attack rows, for a readable console table
            print(f"{wid:>7} {truth:>6} {prob_history[idx]:>7.3f} {stage_history[idx]:>22} {momentum.score:>9.1f} {momentum.top_driver:>26}")

    # Summary: momentum in the run-up to/during the known attack (window 26
    # onward per Phase 2's transition analysis) vs. the benign run before it.
    attack_start_idx = next((i for i, wid in enumerate(window_ids) if snapshots[context_len + i].graph_attack_present), None)
    if attack_start_idx is not None:
        pre = momentum_scores[max(0, attack_start_idx - 6):attack_start_idx]
        during = momentum_scores[attack_start_idx: attack_start_idx + 6]
        print(f"\nMean momentum, 6 windows BEFORE attack onset: {sum(pre)/len(pre):.1f}" if pre else "\n(no pre-attack window)")
        print(f"Mean momentum, 6 windows DURING attack onset: {sum(during)/len(during):.1f}" if during else "")

    benign_scores = [
        s for i, s in enumerate(momentum_scores)
        if not snapshots[context_len + i].graph_attack_present
    ]
    print(f"Mean momentum, all benign windows: {sum(benign_scores)/len(benign_scores):.1f} (n={len(benign_scores)})")
    attack_scores = [
        s for i, s in enumerate(momentum_scores)
        if snapshots[context_len + i].graph_attack_present
    ]
    if attack_scores:
        print(f"Mean momentum, all attack windows: {sum(attack_scores)/len(attack_scores):.1f} (n={len(attack_scores)})")


if __name__ == "__main__":
    main()
