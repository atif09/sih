"""
Phase 3 training script: trains the world model (GNN encoder + temporal core
+ dynamics/attack-forecast/attack-type heads) via supervised dynamics
learning on the Phase 2 graph-sequence dataset.

Usage:
    python models/train.py --config configs/training.yaml

Produces (under output.checkpoint_dir):
    best_model.pt        -- model state_dict for the best val-loss epoch
    normalizer.pkl        -- FeatureNormalizer fit on the train split only
    run_config.yaml        -- exact config used (seed, hyperparams, split)
    attack_type_vocab.json  -- the fixed attack-type vocabulary (train days only)
    training_log.json      -- per-epoch train/val metrics
    test_report.json        -- final evaluation on the held-out day
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.snapshot import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from models.dataset import (
    GraphSequenceDataset,
    build_attack_type_vocab,
    build_split,
    get_train_region_snapshots,
    load_all_days,
)
from models.normalization import FeatureNormalizer
from models.world_model import WorldModel, WorldModelConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(out: dict, batch: dict, weights: dict, pos_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    mse = nn.functional.mse_loss(out["z_hat_next"], out["z_target"])
    bce = nn.functional.binary_cross_entropy_with_logits(out["attack_logits"], batch["attack_label"], pos_weight=pos_weight)
    loss = weights["dynamics_loss_weight"] * mse + weights["attack_loss_weight"] * bce
    parts = {"dynamics_loss": mse.item(), "attack_loss": bce.item()}

    if out["attack_type_logits"] is not None and batch["attack_type_target"].shape[-1] > 0:
        type_bce = nn.functional.binary_cross_entropy_with_logits(
            out["attack_type_logits"], batch["attack_type_target"]
        )
        loss = loss + weights["attack_type_loss_weight"] * type_bce
        parts["attack_type_loss"] = type_bce.item()

    parts["total_loss"] = loss.item()
    return loss, parts


def _flatten_progression_mask(batch: dict) -> np.ndarray:
    """True at positions where the state just BEFORE the target was benign,
    i.e. this position specifically tests catching a benign->attack
    progression (not just recognizing an already-ongoing attack)."""
    # attack_label at position j is the label of the target (t+1); we need
    # to know whether t itself was already under attack, which isn't passed
    # through collate_fn directly -- approximate using the previous
    # position's label within the same sequence (position 0 has no "before"
    # information here, so it's excluded from this specific diagnostic).
    label = batch["attack_label"].numpy()
    prev = np.zeros_like(label)
    prev[:, 1:] = label[:, :-1]
    mask = (prev == 0)
    mask[:, 0] = False
    return mask


@torch.no_grad()
def evaluate(
    model: WorldModel, loader: DataLoader, device: torch.device, weights: dict,
    pos_weight: torch.Tensor | None = None, return_raw: bool = False,
):
    """Returns the metrics dict as before. If return_raw=True, returns
    (metrics, labels, probs) instead -- the flattened ground-truth labels
    and predicted probabilities across every evaluated position, used by
    benchmark/run_benchmark.py (Phase 6) to compute metrics (e.g. false
    positive rate) not already in the metrics dict, on the exact same raw
    predictions rather than re-running the model."""
    model.eval()
    all_logits, all_labels, all_progression_mask = [], [], []
    loss_accum: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        out = model(pyg_batch, batch["batch_size"], batch["seq_len_plus_1"])
        attack_label = batch["attack_label"].to(device)
        attack_type_target = batch["attack_type_target"].to(device)
        _, parts = compute_loss(out, {"attack_label": attack_label, "attack_type_target": attack_type_target}, weights, pos_weight)
        for k, v in parts.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v
        n_batches += 1

        all_logits.append(out["attack_logits"].cpu().numpy().ravel())
        all_labels.append(batch["attack_label"].numpy().ravel())
        all_progression_mask.append(_flatten_progression_mask(batch).ravel())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    progression_mask = np.concatenate(all_progression_mask)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    if len(np.unique(labels)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        # AUC is threshold-independent and robust to class imbalance, so it's
        # used for model selection (see main()). F1 at a fixed 0.5 threshold
        # is sensitive to calibration -- also report the best achievable F1
        # over a threshold sweep, as a ranking-quality sanity check distinct
        # from "did we happen to pick a good decision threshold".
        thresholds = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1s))
        metrics["f1_best_threshold"] = float(f1s[best_idx])
        metrics["best_threshold"] = float(thresholds[best_idx])
    else:
        metrics["precision"] = metrics["recall"] = metrics["f1"] = metrics["roc_auc"] = float("nan")
        metrics["f1_best_threshold"] = metrics["best_threshold"] = float("nan")

    # The specific "catch the progression early" diagnostic: recall
    # restricted to positions transitioning FROM benign TO attack.
    if progression_mask.any() and labels[progression_mask].sum() > 0:
        metrics["benign_to_attack_recall"] = float(
            recall_score_safe(labels[progression_mask], preds[progression_mask])
        )
    else:
        metrics["benign_to_attack_recall"] = float("nan")

    if return_raw:
        return metrics, labels, probs
    return metrics


def recall_score_safe(labels: np.ndarray, preds: np.ndarray) -> float:
    positives = labels == 1
    if not positives.any():
        return float("nan")
    return float((preds[positives] == 1).mean())


def train_one_epoch(model: WorldModel, loader: DataLoader, optimizer, device: torch.device, weights: dict, pos_weight: torch.Tensor | None = None) -> dict:
    model.train()
    loss_accum: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        attack_label = batch["attack_label"].to(device)
        attack_type_target = batch["attack_type_target"].to(device)

        optimizer.zero_grad()
        out = model(pyg_batch, batch["batch_size"], batch["seq_len_plus_1"])
        loss, parts = compute_loss(out, {"attack_label": attack_label, "attack_type_target": attack_type_target}, weights, pos_weight)
        loss.backward()
        optimizer.step()

        for k, v in parts.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = resolve_device(config["train"]["device"])
    print(f"Device: {device}")

    data_cfg = config["data"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph-sequence cache from {data_cfg['processed_dir']} ...")
    day_snapshots = load_all_days(data_cfg["processed_dir"])

    train_seqs, val_seqs, test_seqs = build_split(
        day_snapshots, seq_len=data_cfg["seq_len"], seq_stride=data_cfg["seq_stride"],
        held_out_day=data_cfg["held_out_day"], val_fraction=data_cfg["val_fraction"],
    )
    print(f"Sequences -- train: {len(train_seqs)}, val: {len(val_seqs)}, "
          f"test (held-out day {data_cfg['held_out_day']!r}): {len(test_seqs)}")

    attack_type_vocab = build_attack_type_vocab(day_snapshots, held_out_day=data_cfg["held_out_day"])
    print(f"Attack-type vocabulary (train days only, {len(attack_type_vocab)} types): {attack_type_vocab}")

    train_region_snapshots = get_train_region_snapshots(
        day_snapshots, held_out_day=data_cfg["held_out_day"], val_fraction=data_cfg["val_fraction"],
    )
    normalizer = FeatureNormalizer().fit(train_region_snapshots)
    normalizer.save(checkpoint_dir / "normalizer.pkl")

    train_ds = GraphSequenceDataset(train_seqs, normalizer, attack_type_vocab)
    val_ds = GraphSequenceDataset(val_seqs, normalizer, attack_type_vocab)
    test_ds = GraphSequenceDataset(test_seqs, normalizer, attack_type_vocab)

    train_cfg = config["train"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=test_ds.collate_fn)

    model_cfg = config["model"]
    wm_config = WorldModelConfig(
        node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
        latent_dim=model_cfg["latent_dim"], gnn_hidden_dim=model_cfg["gnn_hidden_dim"],
        gnn_num_layers=model_cfg["gnn_num_layers"], gnn_heads=model_cfg["gnn_heads"],
        temporal_type=model_cfg["temporal_type"], temporal_hidden_dim=model_cfg["temporal_hidden_dim"],
        temporal_num_layers=model_cfg["temporal_num_layers"], temporal_num_heads=model_cfg["temporal_num_heads"],
        dropout=model_cfg["dropout"], attack_type_vocab=attack_type_vocab,
    )
    model = WorldModel(wm_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    loss_weights = {
        "dynamics_loss_weight": train_cfg["dynamics_loss_weight"],
        "attack_loss_weight": train_cfg["attack_loss_weight"],
        "attack_type_loss_weight": train_cfg["attack_type_loss_weight"],
    }

    # Class balance in the attack-forecast target is not 50/50 (attack
    # windows are the minority class) -- weight the positive class in BCE so
    # the model isn't rewarded for just always predicting "benign". Computed
    # from the TRAIN split only.
    train_pos_rate = np.mean([float(s.graph_attack_present) for seq in train_seqs for s in seq[1:]])
    pos_weight_value = (1.0 - train_pos_rate) / max(train_pos_rate, 1e-6)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    print(f"Train attack-window rate: {train_pos_rate:.3f} -> BCE pos_weight={pos_weight_value:.2f}")

    # Model selection uses val ROC-AUC (maximize), not combined total_loss.
    # total_loss mixes the dynamics-regression term (MSE) with the
    # attack-forecast term (BCE) at very different, easily-dominated scales
    # -- early training was observed to pick a checkpoint where MSE had
    # collapsed to a trivial near-zero value while the classifier had not
    # yet learned to separate attack from benign at all (val F1 == 0),
    # simply because that made total_loss numerically smallest. AUC is
    # threshold-independent and measures exactly the capability we care
    # about (ranking attack windows above benign ones), so it doesn't have
    # this failure mode.
    best_val_auc = -1.0
    epochs_without_improvement = 0
    training_log = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_weights, pos_weight)
        val_metrics = evaluate(model, val_loader, device, loss_weights, pos_weight)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"train_loss={train_metrics['total_loss']:.4f} | "
              f"val_loss={val_metrics['total_loss']:.4f} val_f1={val_metrics['f1']:.3f} "
              f"val_f1@best_thresh={val_metrics['f1_best_threshold']:.3f} "
              f"val_auc={val_metrics['roc_auc']:.3f} | {elapsed:.1f}s")

        training_log.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "elapsed_s": elapsed})

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no val AUC improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs)")
                break

    with open(checkpoint_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    with open(checkpoint_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    with open(checkpoint_dir / "attack_type_vocab.json", "w") as f:
        json.dump(attack_type_vocab, f, indent=2)

    print(f"\nLoading best checkpoint (val_auc={best_val_auc:.4f}) for final held-out-day evaluation ...")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, loss_weights, pos_weight)

    print(f"\n{'=' * 70}\nHeld-out day ({data_cfg['held_out_day']}) evaluation\n{'=' * 70}")
    print(f"  Binary attack-forecast head (primary generalization result):")
    print(f"    accuracy={test_metrics['accuracy']:.3f} precision={test_metrics['precision']:.3f} "
          f"recall={test_metrics['recall']:.3f} f1={test_metrics['f1']:.3f} auc={test_metrics['roc_auc']:.3f}")
    print(f"    f1_best_threshold={test_metrics['f1_best_threshold']:.3f} at threshold={test_metrics['best_threshold']:.2f} "
          f"(ranking-quality check independent of the fixed 0.5 decision threshold)")
    print(f"    benign->attack recall (catches the progression early): "
          f"{test_metrics['benign_to_attack_recall']:.3f}")
    print(f"  NOTE: the attack-type head was NOT trained on this day's attack "
          f"type ({data_cfg['held_out_day']!r} excluded from the vocabulary by "
          f"design -- see models/dataset.py). Its output on this day is "
          f"expected to be unreliable; only the binary head above is the valid "
          f"generalization signal.")

    with open(checkpoint_dir / "test_report.json", "w") as f:
        json.dump({"held_out_day": data_cfg["held_out_day"], "metrics": test_metrics}, f, indent=2)

    print(f"\nArtifacts saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
