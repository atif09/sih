"""
Phase 5 CLI: explain a single prediction end-to-end -- infiltration
probability, MITRE stage, real GNN attention over contributing flows, and
real gradient-based attribution over contributing flags/ports/flow
statistics -- as one structured, human-readable record. This is the direct
answer to the PS's "for each prediction, output which specific flags/ports/
flow statistics are driving it, in a structured, human-readable format."

Usage:
    python scripts/explain_prediction.py --day Friday-Afternoon-DDoS --start-window 40 --context-len 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.io import load_day
from explain.explainer import explain_prediction
from models.train import resolve_device
from scripts.run_rollout import load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--processed-dir", type=str, default="data/processed/cicids2017")
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--start-window", type=int, required=True, help="window_id of the LAST context window")
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, normalizer = load_trained_model(Path(args.checkpoint_dir), device)

    snapshots, _ = load_day(args.day, args.processed_dir)
    by_window = {s.window_id: s for s in snapshots}
    context_ids = list(range(args.start_window - args.context_len + 1, args.start_window + 1))
    missing = [w for w in context_ids if w not in by_window]
    if missing:
        print(f"ERROR: window_id(s) {missing} not found for {args.day}")
        sys.exit(1)
    context = [by_window[w] for w in context_ids]

    record = explain_prediction(model, context, normalizer, device)

    print(f"\n{'=' * 70}\nExplanation: {record.day} #{record.context_end_window_id}\n{'=' * 70}")
    print(record.summary)
    print(f"\nTop edges by attention:")
    for e in record.top_edges_by_attention:
        print(f"  {e['src_ip']:>15} -> {e['dst_ip']:<15} attention={e['attention_weight']:.2f} "
              f"(concentration={e['attention_concentration']:.2f}) attack={e['is_attack_edge']}")
    print(f"\nTop node-level features (gradient attribution): {record.top_node_features}")
    print(f"Top flow-level features (gradient attribution): {record.top_edge_features}")
    print(f"Per-window saliency (oldest -> newest): {[round(v, 3) for v in record.per_window_saliency]}")

    if args.json_out:
        payload = {
            "day": record.day, "context_end_window_id": record.context_end_window_id,
            "infiltration_probability": record.infiltration_probability,
            "stage": record.stage.stage, "stage_basis": record.stage.basis,
            "predicted_attack_type": record.stage.predicted_attack_type,
            "top_edges_by_attention": record.top_edges_by_attention,
            "top_node_features": record.top_node_features,
            "top_edge_features": record.top_edge_features,
            "per_window_saliency": record.per_window_saliency,
            "summary": record.summary,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {args.json_out}")


if __name__ == "__main__":
    main()
