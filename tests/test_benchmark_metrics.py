import numpy as np
import pytest

from benchmark.metrics import compute_metrics


def test_compute_metrics_known_confusion_matrix():
    # 4 true positives, 2 false positives, 3 true negatives, 1 false negative.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    # preds @0.5: [1,1,1,1,1,1,0,0,0,0]
    # labels:     [1,1,1,1,0,0,0,0,0,1]
    # TP=4 (idx0-3), FP=2 (idx4,5), TN=3 (idx6-8), FN=1 (idx9)
    metrics = compute_metrics(labels, probs)
    assert metrics["n"] == 10
    assert metrics["accuracy"] == pytest.approx((4 + 3) / 10)
    assert metrics["precision"] == pytest.approx(4 / (4 + 2))
    assert metrics["recall"] == pytest.approx(4 / (4 + 1))
    assert metrics["fpr"] == pytest.approx(2 / (2 + 3))


def test_compute_metrics_perfect_classifier():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = compute_metrics(labels, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_single_class_returns_nan():
    labels = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics(labels, probs)
    assert metrics["fpr"] != metrics["fpr"]  # NaN
    assert metrics["roc_auc"] != metrics["roc_auc"]


def test_compute_metrics_respects_custom_threshold():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.6, 0.6, 0.6, 0.6])
    low = compute_metrics(labels, probs, threshold=0.5)
    high = compute_metrics(labels, probs, threshold=0.7)
    assert low["accuracy"] == 0.5  # all predicted positive
    assert high["accuracy"] == 0.5  # all predicted negative
    assert low["fpr"] == 1.0
    assert high["fpr"] == 0.0
