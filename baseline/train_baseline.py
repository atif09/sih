"""
Phase 6 baseline: logistic regression trained on flattened, non-temporal
per-window features, for a fair comparison against the Phase 3 world model
(see baseline/features.py's docstring for the exact task framing and why the
feature vector mirrors GNNEncoder's own pooling).

Reuses Phase 3's exact split (held_out_day, val_fraction, read from
checkpoints/phase3/run_config.yaml) and exact fitted FeatureNormalizer (from
checkpoints/phase3/normalizer.pkl) -- both models must see identically
scaled inputs and be evaluated against identically defined train/val/test
regions for benchmark/run_benchmark.py's comparison to mean anything.

Usage:
    python baseline/train_baseline.py --phase3-checkpoint-dir checkpoints/phase3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.features import build_pairs_for_day, build_pairs_for_days
from models.dataset import load_all_days, split_day_snapshots
from models.normalization import FeatureNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-checkpoint-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--processed-dir", type=str, default="data/processed/cicids2017")
    parser.add_argument("--output-dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    phase3_dir = Path(args.phase3_checkpoint_dir)
    with open(phase3_dir / "run_config.yaml") as f:
        phase3_config = yaml.safe_load(f)
    held_out_day = phase3_config["data"]["held_out_day"]
    val_fraction = phase3_config["data"]["val_fraction"]
    print(f"Reusing Phase 3 split: held_out_day={held_out_day!r}, val_fraction={val_fraction}")

    normalizer = FeatureNormalizer.load(phase3_dir / "normalizer.pkl")
    print(f"Reusing Phase 3's fitted normalizer from {phase3_dir / 'normalizer.pkl'}")

    print(f"Loading graph-sequence cache from {args.processed_dir} ...")
    day_snapshots = load_all_days(args.processed_dir)
    train_by_day, val_by_day, test_snapshots = split_day_snapshots(day_snapshots, held_out_day, val_fraction)

    X_train, y_train = build_pairs_for_days(train_by_day, normalizer)
    X_val, y_val = build_pairs_for_days(val_by_day, normalizer)
    X_test, y_test = build_pairs_for_day(test_snapshots, normalizer)
    print(f"Pairs -- train: {len(X_train)}, val: {len(X_val)}, test (held-out day): {len(X_test)}")
    print(f"Train attack-window rate: {y_train.mean():.3f}")

    # Small, documented hyperparameter search over regularization strength,
    # selected by val ROC-AUC -- consistent with how the world model's
    # checkpoint was selected (models/train.py), for a like-for-like
    # "properly tuned baseline" rather than a single arbitrary default.
    best_auc, best_C, best_model = -1.0, None, None
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=args.seed)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else float("nan")
        print(f"  C={C:<6} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_C, best_model = auc, C, clf

    print(f"Selected C={best_C} (val_auc={best_auc:.4f})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "held_out_day": held_out_day, "val_fraction": val_fraction,
            "selected_C": best_C, "val_auc": best_auc,
            "train_pairs": len(X_train), "val_pairs": len(X_val), "test_pairs": len(X_test),
            "train_attack_rate": float(y_train.mean()),
            "seed": args.seed,
        }, f, indent=2)

    print(f"Saved baseline model + config to {output_dir}/")


if __name__ == "__main__":
    main()
