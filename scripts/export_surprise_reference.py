"""Builds the Surprise Alarm reference distribution: real dynamics-prediction
error (MSE(z_hat_{t+1}, z_{t+1})) for every benign-to-benign window
transition in the TRAINING days (the held-out day excluded, matching the
model's own train/val split -- see models/dataset.py's
DEFAULT_HELD_OUT_DAY). Saves the full sorted array to
checkpoints/phase3/surprise_reference.json so deployment can compute a real
percentile rank without needing the 214MB processed day caches at runtime
(same pattern as export_stage_duration_stats.py).

Run locally (where data/processed exists):
    python scripts/export_surprise_reference.py
"""

import json
from pathlib import Path

import torch
import yaml

from containment.estimator import ground_truth_stage
from models.dataset import DEFAULT_HELD_OUT_DAY, load_all_days
from models.normalization import FeatureNormalizer
from models.train import resolve_device
from models.world_model import WorldModel, WorldModelConfig
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from rollout.mitre import STAGE_BENIGN
from surprise.score import compute_dynamics_surprise, is_informative_transition

CHECKPOINT_DIR = Path("checkpoints/phase3")


def main() -> None:
    with open(CHECKPOINT_DIR / "run_config.yaml") as f:
        run_config = yaml.safe_load(f)
    with open(CHECKPOINT_DIR / "attack_type_vocab.json") as f:
        attack_type_vocab = json.load(f)

    device = resolve_device("auto")
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
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device))
    model.eval()
    normalizer = FeatureNormalizer.load(CHECKPOINT_DIR / "normalizer.pkl")

    seq_len = run_config["data"]["seq_len"]
    day_snapshots = load_all_days(run_config["data"]["processed_dir"])

    values: list[float] = []
    for day, snapshots in day_snapshots.items():
        if day == DEFAULT_HELD_OUT_DAY:
            continue
        for i in range(seq_len - 1, len(snapshots) - 1):
            # Only benign-to-benign transitions -- this is a reference for
            # "what normal looks like," not a mix of attack transitions.
            if not (ground_truth_stage(snapshots[i]) == STAGE_BENIGN and ground_truth_stage(snapshots[i + 1]) == STAGE_BENIGN):
                continue
            context = snapshots[i - seq_len + 1: i + 1]
            if not is_informative_transition(context, snapshots[i + 1]):
                continue  # empty-window transition -- see surprise/score.py's module docstring
            mse = compute_dynamics_surprise(model, context, snapshots[i + 1], normalizer, device)
            values.append(mse)
        print(f"{day}: {len(values)} cumulative reference points")

    out_path = CHECKPOINT_DIR / "surprise_reference.json"
    out_path.write_text(json.dumps({"benign_dynamics_mse": values}))
    print(f"Saved {len(values)} reference points to {out_path}")
    import statistics
    print("mean", statistics.mean(values), "p50", statistics.median(values),
          "p95", sorted(values)[int(0.95 * len(values))], "p99", sorted(values)[int(0.99 * len(values))])


if __name__ == "__main__":
    main()
