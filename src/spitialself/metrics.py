"""Metric helpers for communication prediction tables."""

import numpy as np


def compute_confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def compute_metrics_from_confusion(tp, fp, tn, fn):
    eps = 1e-12
    positives = tp + fn
    negatives = tn + fp

    precision = tp / (tp + fp + eps)
    recall = tp / (positives + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    tnr = tn / (negatives + eps)
    fpr = fp / (negatives + eps)
    balanced_acc = 0.5 * (recall + tnr)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr,
        "fpr": fpr,
        "balanced_acc": balanced_acc,
    }


def find_best_thresholds(y_true, probs, n_bins=200):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)

    best_f1 = -1.0
    best_f1_thr = 0.5
    best_f1_metrics = None

    best_j = -1.0
    best_j_thr = 0.5
    best_j_metrics = None

    for thr in np.linspace(0.0, 1.0, n_bins):
        y_pred = (probs >= thr).astype(int)
        metrics = compute_metrics_from_confusion(*compute_confusion(y_true, y_pred))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_f1_thr = thr
            best_f1_metrics = metrics

        youden_j = metrics["recall"] + metrics["tnr"] - 1.0
        if youden_j > best_j:
            best_j = youden_j
            best_j_thr = thr
            best_j_metrics = metrics

    return {
        "thr_maxF1": best_f1_thr,
        "metrics_maxF1": best_f1_metrics,
        "thr_maxJ": best_j_thr,
        "metrics_maxJ": best_j_metrics,
    }
