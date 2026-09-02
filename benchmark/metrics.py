"""
Shared metric computation for the Phase 6 benchmark -- one function, used
identically for both the world model's and the baseline's raw (labels,
probs) arrays, so neither side gets a methodological advantage in how a
metric is computed.

Includes false positive rate (FPR = FP / (FP + TN)), which the PS asks for
explicitly but neither models/train.py's evaluate() nor scikit-learn expose
directly under that name.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    metrics: dict = {"n": int(len(labels)), "positive_rate": float(labels.mean()) if len(labels) else float("nan")}

    if len(np.unique(labels)) < 2:
        metrics.update({k: float("nan") for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr")})
        return metrics

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    metrics["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics
