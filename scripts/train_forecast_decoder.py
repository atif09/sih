"""Trains the Network Forecast View decoder (models/forecast_decoder.py) on
top of the FROZEN, already-trained world model -- does not touch
best_model.pt or any existing evaluation number. Targets are real
ground-truth graph-level stats of the true next window (see
compute_target_stats); inputs are the frozen model's own ẑ_t+1 prediction
for the same transition, computed the identical way every other feature in
this project already does (model.dynamics_head on the causal hidden state).

Training days only (held-out day excluded, matching the base model's own
split); reports held-out-day R^2 per stat before saving, so the decoder's
real reliability is checked, not assumed. Run locally (where
data/processed exists):

    python scripts/train_forecast_decoder.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from containment.estimator import ground_truth_stage
from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.dataset import DEFAULT_HELD_OUT_DAY, load_all_days, snapshot_to_pyg_data
from models.forecast_decoder import STAT_NAMES, ForecastDecoder, compute_target_stats, transform_targets
from models.normalization import FeatureNormalizer
from models.train import resolve_device
from models.world_model import WorldModel, WorldModelConfig
from rollout.mitre import STAGE_BENIGN
from surprise.score import is_informative_transition
from torch_geometric.data import Batch

CHECKPOINT_DIR = Path("checkpoints/phase3")


def build_examples(model, normalizer, device, day_snapshots, seq_len, days):
    xs, ys = [], []
    for day in days:
        snapshots = day_snapshots[day]
        for i in range(seq_len - 1, len(snapshots) - 1):
            context = snapshots[i - seq_len + 1: i + 1]
            target_snap = snapshots[i + 1]
            if not is_informative_transition(context, target_snap):
                continue
            data_list = [snapshot_to_pyg_data(s, normalizer) for s in context]
            batch = Batch.from_data_list(data_list).to(device)
            with torch.no_grad():
                z_seq = model.encode_sequence(batch, batch_size=1, seq_len_plus_1=len(context))
                h_seq = model.temporal(z_seq)
                z_hat = model.dynamics_head(h_seq[:, -1, :]).squeeze(0).cpu().numpy()
            xs.append(z_hat)
            ys.append(transform_targets(compute_target_stats(context[-1], target_snap)))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    with open(CHECKPOINT_DIR / "run_config.yaml") as f:
        run_config = yaml.safe_load(f)
    with open(CHECKPOINT_DIR / "attack_type_vocab.json") as f:
        attack_type_vocab = json.load(f)

    device = resolve_device("auto")
    mc = run_config["model"]
    cfg = WorldModelConfig(
        node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
        latent_dim=mc["latent_dim"], gnn_hidden_dim=mc["gnn_hidden_dim"],
        gnn_num_layers=mc["gnn_num_layers"], gnn_heads=mc["gnn_heads"],
        temporal_type=mc["temporal_type"], temporal_hidden_dim=mc["temporal_hidden_dim"],
        temporal_num_layers=mc["temporal_num_layers"], temporal_num_heads=mc["temporal_num_heads"],
        dropout=mc["dropout"], attack_type_vocab=attack_type_vocab,
    )
    model = WorldModel(cfg).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device))
    model.eval()  # FROZEN -- never trained further, see module docstring
    normalizer = FeatureNormalizer.load(CHECKPOINT_DIR / "normalizer.pkl")
    seq_len = run_config["data"]["seq_len"]

    day_snapshots = load_all_days(run_config["data"]["processed_dir"])
    train_days = [d for d in day_snapshots if d != DEFAULT_HELD_OUT_DAY]

    print("Building training examples...")
    X_train, Y_train = build_examples(model, normalizer, device, day_snapshots, seq_len, train_days)
    print(f"{len(X_train)} training examples")

    print("Building held-out validation examples...")
    X_val, Y_val = build_examples(model, normalizer, device, day_snapshots, seq_len, [DEFAULT_HELD_OUT_DAY])
    print(f"{len(X_val)} held-out examples")

    decoder = ForecastDecoder(latent_dim=cfg.latent_dim).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    Y_train_t = torch.from_numpy(Y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    n = len(X_train_t)
    batch_size = 64
    best_val_loss = float("inf")
    best_state = None
    patience, patience_ctr = 15, 0

    for epoch in range(200):
        decoder.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            opt.zero_grad()
            pred = decoder(X_train_t[idx])
            loss = loss_fn(pred, Y_train_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        decoder.eval()
        with torch.no_grad():
            val_pred = decoder(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    decoder.load_state_dict(best_state)
    decoder.eval()

    # Real held-out-day R^2 per stat -- the check that decides whether this
    # decoder's numbers are trustworthy, not assumed.
    with torch.no_grad():
        val_pred = decoder(X_val_t).cpu().numpy()
    val_true = Y_val

    print("\nHeld-out day (Friday-Afternoon-DDoS) R^2 per stat (log1p scale):")
    r2_per_stat = {}
    for i, name in enumerate(STAT_NAMES):
        ss_res = np.sum((val_true[:, i] - val_pred[:, i]) ** 2)
        ss_tot = np.sum((val_true[:, i] - val_true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_per_stat[name] = float(r2)
        print(f"  {name}: R^2={r2:.3f}")

    torch.save(decoder.state_dict(), CHECKPOINT_DIR / "forecast_decoder.pt")
    (CHECKPOINT_DIR / "forecast_decoder_validation.json").write_text(json.dumps({"held_out_r2": r2_per_stat, "n_val": len(X_val)}))
    print(f"\nSaved decoder to {CHECKPOINT_DIR / 'forecast_decoder.pt'}")


if __name__ == "__main__":
    main()
