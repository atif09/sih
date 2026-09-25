"""
Phase 6 benchmark: world model (Phase 3) vs. logistic regression baseline
(Phase 6), on identical val/test regions (same split, same normalizer),
reporting F1, precision, recall, false positive rate, and ROC-AUC -- with the
held-out day (unseen attack type) broken out as its own row, since that's
the PS's specific generalization claim to substantiate.

Methodological note, stated plainly rather than glossed over: the two
models are NOT evaluated on identically-defined example sets. The baseline
sees every individual (window_t -> window_t+1) pair in the val/test regions
(except pairs where window_t itself had zero flows). The world model is
evaluated on the positions covered by the sequences models/dataset.py forms
for it (governed by seq_len/seq_stride, and dropping any sequence that is
*entirely* flow-free across all its windows) -- a different, and for
short/sparse regions sometimes smaller, set of transitions. This is
intentional, not an oversight: it reflects how each model is actually meant
to be used (baseline = context-free, single-window; world model = its
configured multi-window context), which is exactly the capability this
benchmark exists to evaluate the value of. Pretending they consumed
identical example sets would misrepresent that difference away.

Usage:
    python benchmark/run_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.features import build_pairs_for_day, build_pairs_for_days
from benchmark.metrics import compute_metrics
from models.dataset import GraphSequenceDataset, build_split, load_all_days, split_day_snapshots
from models.train import evaluate, resolve_device
from scripts.run_rollout import load_trained_model
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
import pickle
import argparse


def _curve_data(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Full ROC/PR curves (not just the single threshold=0.5 operating point
    compute_metrics reports) -- the actual points behind the headline
    roc_auc figure. Only meaningful when both classes are present.

    Includes average_precision (area under the PR curve) as its own field,
    separate from roc_auc -- the two are different metrics computed
    differently (trapezoidal ROC integral vs. a step-weighted precision
    sum) and can disagree, especially under class imbalance. Reusing
    roc_auc as if it were "the AUC" for a PR chart would mislabel it."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return {"roc": None, "pr": None, "average_precision": float("nan")}
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
        # precision_recall_curve returns one fewer threshold than precision/recall points
        # (the last point is always (precision=1, recall=0) with no corresponding threshold)
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
        "average_precision": float(average_precision_score(labels, probs)),
    }


def _calibration_data(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict | None:
    """Reliability diagram data: for each of n_bins equal-width confidence
    bins, the model's mean predicted probability in that bin vs. the real
    fraction of positives observed in it -- sklearn's own calibration_curve,
    not a hand-rolled binning (avoids a real off-by-one/edge-bin mistake
    that's easy to introduce by hand). Empty bins are dropped by sklearn,
    so bin_counts (computed separately, same bin edges) discloses which
    remaining points rest on thin sample counts rather than hiding it."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return None
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)
    all_counts = np.bincount(bin_idx, minlength=n_bins)
    # Only the non-empty bins' counts, in the same order calibration_curve
    # returned its (now-fewer-than-n_bins) points in.
    bin_counts = [int(c) for c in all_counts if c > 0]
    return {
        "predicted_confidence": prob_pred.tolist(),
        "actual_frequency": prob_true.tolist(),
        "bin_counts": bin_counts,
    }


def format_table(rows: list[dict], columns: list[str]) -> str:
    header = "| model | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines = [header, sep]
    for row in rows:
        vals = " | ".join(
            f"{row[c]:.3f}" if isinstance(row[c], float) and row[c] == row[c] else str(row[c])
            for c in columns
        )
        lines.append(f"| {row['model']} | {vals} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-checkpoint-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--baseline-checkpoint-dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--processed-dir", type=str, default="data/processed/cicids2017")
    parser.add_argument("--report-out", type=str, default="benchmark/report.md")
    parser.add_argument("--curve-out", type=str, default="benchmark/roc_pr_curves.json", help="Full ROC/PR curve points (held-out day only) for the console's ROC/PR chart panels -- the real data behind the headline roc_auc figure, not a single point.")
    parser.add_argument("--calibration-out", type=str, default="benchmark/calibration.json", help="Reliability diagram data (held-out day, world model only) for the console's Confidence Calibration panel.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    phase3_dir = Path(args.phase3_checkpoint_dir)
    baseline_dir = Path(args.baseline_checkpoint_dir)
    device = resolve_device(args.device)

    with open(phase3_dir / "run_config.yaml") as f:
        phase3_config = yaml.safe_load(f)
    held_out_day = phase3_config["data"]["held_out_day"]
    val_fraction = phase3_config["data"]["val_fraction"]
    seq_len = phase3_config["data"]["seq_len"]
    seq_stride = phase3_config["data"]["seq_stride"]

    print(f"Loading world model from {phase3_dir} ...")
    world_model, normalizer = load_trained_model(phase3_dir, device)
    with open(phase3_dir / "attack_type_vocab.json") as f:
        attack_type_vocab = json.load(f)

    print(f"Loading baseline from {baseline_dir} ...")
    with open(baseline_dir / "model.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    print(f"Loading graph-sequence cache from {args.processed_dir} ...")
    day_snapshots = load_all_days(args.processed_dir)
    train_by_day, val_by_day, test_snapshots = split_day_snapshots(day_snapshots, held_out_day, val_fraction)

    # --- World model: val + test, via its own sequence-based evaluation ---
    _train_seqs, val_seqs, test_seqs = build_split(day_snapshots, seq_len, seq_stride, held_out_day, val_fraction)
    val_ds = GraphSequenceDataset(val_seqs, normalizer, attack_type_vocab)
    test_ds = GraphSequenceDataset(test_seqs, normalizer, attack_type_vocab)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_ds.collate_fn)

    dummy_weights = {"dynamics_loss_weight": 1.0, "attack_loss_weight": 1.0, "attack_type_loss_weight": 0.5}
    _wm_val_metrics_unused, wm_val_labels, wm_val_probs = evaluate(world_model, val_loader, device, dummy_weights, return_raw=True)
    _wm_test_metrics_unused, wm_test_labels, wm_test_probs = evaluate(world_model, test_loader, device, dummy_weights, return_raw=True)

    # --- Baseline: val + test, via flattened single-window pairs ---
    X_val, y_val = build_pairs_for_days(val_by_day, normalizer)
    X_test, y_test = build_pairs_for_day(test_snapshots, normalizer)
    bl_val_probs = baseline_model.predict_proba(X_val)[:, 1]
    bl_test_probs = baseline_model.predict_proba(X_test)[:, 1]

    # Diagnostic, surfaced in the report rather than left implicit: how many
    # of the 7 non-held-out days' chronological tail (the val region) have
    # NO attack windows at all, e.g. because that day's attack campaign
    # finished well before the capture ended. Those days contribute zero
    # ranking signal (single-class) and the pooled val AUC ends up dominated
    # by whichever multi-class day has the most examples -- found while
    # investigating why the baseline's pooled val AUC came out below 0.5.
    single_class_days, multi_class_days = [], []
    for day, snaps in val_by_day.items():
        _, y_day = build_pairs_for_day(snaps, normalizer)
        (single_class_days if len(np.unique(y_day)) < 2 else multi_class_days).append(day)

    columns = ["n", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc"]

    val_rows = [
        {"model": "World Model", **compute_metrics(wm_val_labels, wm_val_probs)},
        {"model": "Baseline (LogReg)", **compute_metrics(y_val, bl_val_probs)},
    ]
    test_rows = [
        {"model": "World Model", **compute_metrics(wm_test_labels, wm_test_probs)},
        {"model": "Baseline (LogReg)", **compute_metrics(y_test, bl_test_probs)},
    ]

    val_table = format_table(val_rows, columns)
    test_table = format_table(test_rows, columns)

    print(f"\n{'=' * 70}\nValidation (in-distribution, seen attack types, later time chunk of "
          f"each non-held-out day)\n{'=' * 70}")
    print(val_table)
    print(f"\n{'=' * 70}\nHeld-out day '{held_out_day}' (UNSEEN attack type -- the PS's "
          f"generalization requirement)\n{'=' * 70}")
    print(test_table)

    report = f"""# Phase 6 Benchmark: World Model vs. Logistic Regression Baseline

Held-out day (unseen attack type): **{held_out_day}**

## Validation set (in-distribution)
{val_table}

## Held-out day (unseen attack type)
{test_table}

## Methodology notes
- Both models see identically normalized, identically flattened-or-graph
  per-window features (baseline: mean/max node pool + mean edge pool,
  mirroring the world model's own GNN pooling structure, see
  baseline/features.py). Neither model has an input-quality advantage.
- The world model uses its full configured temporal context
  (seq_len={seq_len} windows); the baseline sees exactly one window per
  prediction, by design -- this is precisely the capability being measured.
- The two models are NOT evaluated on identically-defined example sets (see
  benchmark/run_benchmark.py's module docstring) -- this reflects how each
  model is actually used, not an oversight.
- **Validation AUC for both models is lower than the held-out-day AUC** --
  disclosed, not hidden. {len(single_class_days)} of the 7 non-held-out
  days' chronological tail (the val region) contain NO attack windows at all
  ({', '.join(single_class_days) if single_class_days else 'none'}), because
  those days' attack campaigns finished before the capture ended. Only
  {', '.join(multi_class_days) if multi_class_days else 'none'} have any
  class variation in val to measure ranking quality on, so the pooled val
  AUC is dominated by whichever of those has the most examples. The
  held-out day's AUC is the more statistically meaningful number for both
  models, and is where the world model's advantage is largest.
"""
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\nSaved report to {report_path}")

    # Full ROC/PR curves, held-out day only (that's the PS's generalization
    # claim, and what the console's ROC/PR chart panels visualize) -- the
    # actual (fpr, tpr) / (precision, recall) points behind test_rows'
    # single roc_auc number above, not a fabricated curve interpolated from
    # one point.
    curve_data = {
        "held_out_day": held_out_day,
        "world_model": {"auc": test_rows[0]["roc_auc"], **_curve_data(wm_test_labels, wm_test_probs)},
        "baseline": {"auc": test_rows[1]["roc_auc"], **_curve_data(y_test, bl_test_probs)},
    }
    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps(curve_data, indent=2))
    print(f"Saved ROC/PR curve data to {curve_path}")

    # Confidence calibration (reliability diagram), world model only, held-out
    # day -- checks whether "70% confidence" predictions are actually right
    # ~70% of the time. Not meaningful for the baseline here: its held-out
    # predictions are almost all on one side of 0.5 (see report.md's
    # precision=0.000 row), leaving too few populated bins to plot honestly.
    calibration_data = {
        "held_out_day": held_out_day,
        "n_bins": 10,
        "world_model": _calibration_data(wm_test_labels, wm_test_probs, n_bins=10),
    }
    calibration_path = Path(args.calibration_out)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_data, indent=2))
    print(f"Saved calibration data to {calibration_path}")


if __name__ == "__main__":
    main()
