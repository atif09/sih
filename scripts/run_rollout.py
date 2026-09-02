"""
Phase 4 CLI: load a trained world model checkpoint (from models/train.py) and
run a K-step forward-simulation rollout from a chosen context window,
printing the infiltration probability time series, predicted MITRE stages,
and the context-level explainability stub.

Usage:
    # Roll out from the end of the held-out day's context (the interesting
    # case: genuinely unseen attack type, not memorized)
    python scripts/run_rollout.py --day Friday-Afternoon-DDoS --start-window 20 --context-len 10 --k 10

    # Any day/window works, including training days
    python scripts/run_rollout.py --day Tuesday --start-window 100 --context-len 20 --k 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.io import load_day
from models.normalization import FeatureNormalizer
from models.train import resolve_device
from models.world_model import WorldModel, WorldModelConfig
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from rollout.engine import run_rollout


def load_trained_model(checkpoint_dir: Path, device: torch.device) -> tuple[WorldModel, FeatureNormalizer]:
    with open(checkpoint_dir / "run_config.yaml") as f:
        run_config = yaml.safe_load(f)
    with open(checkpoint_dir / "attack_type_vocab.json") as f:
        attack_type_vocab = json.load(f)

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
    return model, normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--processed-dir", type=str, default="data/processed/cicids2017")
    parser.add_argument("--day", type=str, required=True, help="Day name, e.g. 'Friday-Afternoon-DDoS'")
    parser.add_argument("--start-window", type=int, required=True, help="window_id of the LAST context window")
    parser.add_argument("--context-len", type=int, default=20, help="Number of windows of real context to feed in")
    parser.add_argument("--k", type=int, default=10, help="Number of future steps to roll out")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to also save the result as JSON")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"Loading model from {checkpoint_dir} (device={device}) ...")
    model, normalizer = load_trained_model(checkpoint_dir, device)

    print(f"Loading {args.day} graph cache from {args.processed_dir} ...")
    snapshots, _transitions = load_day(args.day, args.processed_dir)
    by_window = {s.window_id: s for s in snapshots}

    context_ids = list(range(args.start_window - args.context_len + 1, args.start_window + 1))
    missing = [w for w in context_ids if w not in by_window]
    if missing:
        print(f"ERROR: window_id(s) {missing} not found for {args.day} "
              f"(day has windows {min(by_window)}..{max(by_window)})")
        sys.exit(1)
    context = [by_window[w] for w in context_ids]

    result = run_rollout(model, context, normalizer, k=args.k, device=device)

    print(f"\n{'=' * 70}\nRollout: {result.day}, context ends at window {result.context_end_window_id}, "
          f"K={result.k}\n{'=' * 70}")
    print(f"{'step':>5} {'infil_prob':>11} {'stage':>28} {'basis':>18} {'top_attack_type':>22}")
    for i, (prob, stage) in enumerate(zip(result.infiltration_probability, result.stage_predictions), start=1):
        print(f"{i:>5} {prob:>11.3f} {stage.stage:>28} {stage.basis:>18} {str(stage.predicted_attack_type):>22}")

    print(f"\nDriving flows from the last observed real window (window {result.context_end_window_id}) "
          f"-- simple magnitude-based stub, see rollout/explain_stub.py; real attention/SHAP attribution in Phase 5:")
    for flow in result.driving_context_flows:
        print(f"  {flow['src_ip']:>15} -> {flow['dst_ip']:<15} "
              f"score={flow['score']:.2f} flows={int(flow['flow_count'])} "
              f"bytes={flow['bytes_sum']:.0f} dst_ports={int(flow['dst_port_nunique'])} "
              f"retrans={int(flow['retransmission_sum'])} attack={flow['is_attack_edge']}")

    if args.json_out:
        payload = {
            "day": result.day, "context_end_window_id": result.context_end_window_id, "k": result.k,
            "infiltration_probability": result.infiltration_probability,
            "attack_type_probs": result.attack_type_probs,
            "stage_predictions": [
                {"stage": s.stage, "confidence": s.confidence, "basis": s.basis, "predicted_attack_type": s.predicted_attack_type}
                for s in result.stage_predictions
            ],
            "driving_context_flows": result.driving_context_flows,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {args.json_out}")


if __name__ == "__main__":
    main()
