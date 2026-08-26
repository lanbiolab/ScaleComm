"""Evaluation helpers for ground-truth CCC benchmark predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def evaluate_ground_truth_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    top_k: int = 100,
    threshold_quantile: float = 0.95,
    score_column: str = "score",
) -> dict:
    """Evaluate candidate-level CCC predictions against binary ground truth.

    Expected prediction schema: `candidate_id`, `method`, `score`.
    Expected truth schema: `candidate_id`, `ground_truth`, `weak_signal`,
    `false_positive_high_expression`, `mechanism`. When available, the
    evaluator also reports selected-call rates for wrong-range and wrong-niche
    negative controls; those columns are never required for real-data use.
    """

    required_prediction = {"candidate_id", score_column}
    missing_prediction = required_prediction - set(predictions.columns)
    if missing_prediction:
        raise ValueError(f"Prediction table missing columns: {sorted(missing_prediction)}")

    required_truth = {
        "candidate_id",
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
        "mechanism",
    }
    missing_truth = required_truth - set(truth.columns)
    if missing_truth:
        raise ValueError(f"Truth table missing columns: {sorted(missing_truth)}")

    merged = truth.merge(predictions, on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between truth and predictions")

    y_true = merged["ground_truth"].astype(int).to_numpy()
    scores = _minmax_normalize(merged[score_column].astype(float).to_numpy())
    predicted_positive = scores >= np.quantile(scores, threshold_quantile)

    top_k = min(int(top_k), len(merged))
    top_idx = np.argsort(-scores)[:top_k]

    weak_mask = merged["weak_signal"].astype(bool).to_numpy()
    false_positive_mask = merged["false_positive_high_expression"].astype(bool).to_numpy()
    wrong_range_mask = (
        merged["wrong_range_false_positive"].fillna(False).astype(bool).to_numpy()
        if "wrong_range_false_positive" in merged
        else np.zeros(len(merged), dtype=bool)
    )
    wrong_niche_mask = (
        (~merged["niche_rule_satisfied"].fillna(True).astype(bool).to_numpy()) & (y_true == 0)
        if "niche_rule_satisfied" in merged
        else np.zeros(len(merged), dtype=bool)
    )

    metrics = {
        "n_candidates": int(len(merged)),
        "n_positive": int(y_true.sum()),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "auroc": _auroc(y_true, scores),
        "aupr": _aupr(y_true, scores),
        "top_k": int(top_k),
        "top_k_precision": float(y_true[top_idx].mean()) if top_k else 0.0,
        "recall_at_threshold": _safe_divide(
            int((predicted_positive & (y_true == 1)).sum()), int(y_true.sum())
        ),
        "weak_signal_recall_at_threshold": _safe_divide(
            int((predicted_positive & (y_true == 1) & weak_mask).sum()),
            int(((y_true == 1) & weak_mask).sum()),
        ),
        "empirical_fdr_at_threshold": _safe_divide(
            int((predicted_positive & (y_true == 0)).sum()), int(predicted_positive.sum())
        ),
        "false_positive_high_expression_rate_at_threshold": _safe_divide(
            int((predicted_positive & false_positive_mask).sum()), int(false_positive_mask.sum())
        ),
        "wrong_range_false_positive_rate_at_threshold": _safe_divide(
            int((predicted_positive & wrong_range_mask).sum()), int(wrong_range_mask.sum())
        ),
        "wrong_niche_false_positive_rate_at_threshold": _safe_divide(
            int((predicted_positive & wrong_niche_mask).sum()), int(wrong_niche_mask.sum())
        ),
        "calibration_error_10bin": _expected_calibration_error(y_true, scores, n_bins=10),
        "threshold_quantile": float(threshold_quantile),
        "threshold_score": float(np.quantile(scores, threshold_quantile)),
    }
    true_positive = int((predicted_positive & (y_true == 1)).sum())
    false_positive = int((predicted_positive & (y_true == 0)).sum())
    false_negative = int((~predicted_positive & (y_true == 1)).sum())
    true_negative = int((~predicted_positive & (y_true == 0)).sum())
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    metrics["precision_at_threshold"] = precision
    metrics["f1_at_threshold"] = _safe_divide(2.0 * precision * recall, precision + recall)
    metrics["specificity_at_threshold"] = _safe_divide(true_negative, true_negative + false_positive)

    for mechanism, subset in merged.groupby("mechanism"):
        mechanism_true = subset["ground_truth"].astype(int).to_numpy()
        mechanism_scores = _minmax_normalize(subset[score_column].astype(float).to_numpy())
        metrics[f"auroc__{mechanism}"] = _auroc(mechanism_true, mechanism_scores)
        metrics[f"aupr__{mechanism}"] = _aupr(mechanism_true, mechanism_scores)
        metrics[f"positive_count__{mechanism}"] = int(mechanism_true.sum())

    return metrics


def evaluate_predictions_auto(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    top_k: int = 100,
    threshold_quantile: float = 0.95,
    score_column: str = "score",
) -> dict:
    """Evaluate predictions when ground truth exists, otherwise summarize real-data scores."""

    required_truth = {
        "candidate_id",
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
        "mechanism",
    }
    if required_truth.issubset(candidates.columns):
        metrics = evaluate_ground_truth_predictions(
            candidates,
            predictions,
            top_k=top_k,
            threshold_quantile=threshold_quantile,
            score_column=score_column,
        )
        metrics["evaluation_mode"] = "ground_truth"
        metrics["truth_metrics_available"] = True
        return metrics
    metrics = summarize_real_data_predictions(
        candidates,
        predictions,
        top_k=top_k,
        score_column=score_column,
    )
    metrics["truth_metrics_available"] = False
    metrics["truth_metrics_note"] = "candidate table has no complete ground-truth columns"
    return metrics


def summarize_real_data_predictions(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    top_k: int = 100,
    score_column: str = "score",
) -> dict:
    """Summarize candidate scores for real data without native CCC ground truth."""

    required_prediction = {"candidate_id", score_column}
    missing_prediction = required_prediction - set(predictions.columns)
    if missing_prediction:
        raise ValueError(f"Prediction table missing columns: {sorted(missing_prediction)}")
    if "candidate_id" not in candidates.columns:
        raise ValueError("Candidate table missing candidate_id column")

    merged = candidates.merge(predictions, on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions")

    scores = merged[score_column].astype(float)
    top_k = min(int(top_k), len(merged))
    top = merged.sort_values(score_column, ascending=False).head(top_k)
    lr_summary = _lr_score_summary(merged, score_column=score_column)
    top_lr = lr_summary.iloc[0].to_dict() if not lr_summary.empty else {}

    return {
        "n_candidates": int(len(merged)),
        "n_positive": float("nan"),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "auroc": float("nan"),
        "aupr": float("nan"),
        "top_k": int(top_k),
        "top_k_precision": float("nan"),
        "recall_at_threshold": float("nan"),
        "weak_signal_recall_at_threshold": float("nan"),
        "empirical_fdr_at_threshold": float("nan"),
        "false_positive_high_expression_rate_at_threshold": float("nan"),
        "wrong_range_false_positive_rate_at_threshold": float("nan"),
        "wrong_niche_false_positive_rate_at_threshold": float("nan"),
        "calibration_error_10bin": float("nan"),
        "threshold_quantile": float("nan"),
        "threshold_score": float("nan"),
        "evaluation_mode": "real_data_no_ground_truth",
        "n_lr_pairs_scored": int(lr_summary.shape[0]),
        "top_lr_by_mean_score": str(top_lr.get("lr_id", "")),
        "top_lr_mean_score": float(top_lr.get("mean_score", float("nan"))),
        "top100_lr_counts": ";".join(f"{lr}:{count}" for lr, count in top["lr_id"].value_counts().items())
        if "lr_id" in top
        else "",
    }


def _lr_score_summary(merged: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    group_cols = [column for column in ("lr_id", "ligand", "receptor") if column in merged.columns]
    if not group_cols:
        return pd.DataFrame()
    return (
        merged.groupby(group_cols, dropna=False)
        .agg(
            mean_score=(score_column, "mean"),
            max_score=(score_column, "max"),
            n_candidates=("candidate_id", "size"),
        )
        .reset_index()
        .sort_values(["mean_score", "max_score"], ascending=False)
    )


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    sum_pos_ranks = float(ranks[positives].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    # Average precision evaluates tied values as one score threshold, avoiding
    # arbitrary candidate-row ordering in coarse or projected baseline scores.
    return float(average_precision_score(y_true, scores))


def _expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        if end == 1.0:
            mask = (scores >= start) & (scores <= end)
        else:
            mask = (scores >= start) & (scores < end)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += float(mask.mean()) * abs(confidence - accuracy)
    return float(error)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        # Ranks are 1-based. Ties receive the average rank.
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)
