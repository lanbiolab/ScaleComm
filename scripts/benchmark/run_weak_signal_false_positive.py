#!/usr/bin/env python
"""Summarize weak-signal recovery and false-positive burden."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_ground_truth_predictions


def summarize_stress(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    score_column: str = "score",
    threshold_quantiles: tuple[float, ...] = (0.9, 0.95, 0.99),
    top_k_values: tuple[int, ...] = (50, 100, 500),
) -> dict:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    merged = _merge(candidates, predictions, score_column)
    method = _method_name(predictions, prediction_path)
    replicate = candidate_path.parent.name

    output_dir.mkdir(parents=True, exist_ok=True)
    base_metrics = evaluate_ground_truth_predictions(candidates, predictions, score_column=score_column)
    base_metrics.update(
        {
            "replicate": replicate,
            "method": method,
            "candidate_path": str(candidate_path),
            "prediction_path": str(prediction_path),
        }
    )

    threshold_table = _threshold_summary(merged, method, replicate, score_column, threshold_quantiles)
    topk_table = _topk_summary(merged, method, replicate, score_column, top_k_values)
    lr_table = _lr_summary(merged, method, replicate, score_column)
    mechanism_table = _mechanism_summary(merged, method, replicate, score_column)

    prefix = f"{replicate}__{method}"
    threshold_table.to_csv(output_dir / f"{prefix}__threshold_stress.csv", index=False)
    topk_table.to_csv(output_dir / f"{prefix}__topk_stress.csv", index=False)
    lr_table.to_csv(output_dir / f"{prefix}__lr_stress.csv", index=False)
    mechanism_table.to_csv(output_dir / f"{prefix}__mechanism_stress.csv", index=False)
    with (output_dir / f"{prefix}__stress_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(base_metrics, handle, indent=2, allow_nan=True)
    return base_metrics


def _merge(candidates: pd.DataFrame, predictions: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {
        "candidate_id",
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
        "mechanism",
        "lr_id",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    if "candidate_id" not in predictions.columns or score_column not in predictions.columns:
        raise ValueError("Prediction table must contain candidate_id and score column")
    merged = candidates.merge(
        predictions[["candidate_id", score_column] + (["method"] if "method" in predictions.columns else [])],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    merged[score_column] = _minmax(merged[score_column].to_numpy(dtype=float))
    return merged


def _method_name(predictions: pd.DataFrame, prediction_path: Path) -> str:
    if "method" in predictions.columns and predictions["method"].notna().any():
        return str(predictions["method"].dropna().iloc[0])
    name = prediction_path.stem
    for suffix in ("__predictions", "_predictions"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _threshold_summary(
    table: pd.DataFrame,
    method: str,
    replicate: str,
    score_column: str,
    threshold_quantiles: tuple[float, ...],
) -> pd.DataFrame:
    y_true = table["ground_truth"].astype(int).to_numpy()
    weak = table["weak_signal"].astype(bool).to_numpy()
    fp_high = table["false_positive_high_expression"].astype(bool).to_numpy()
    scores = table[score_column].to_numpy(dtype=float)
    rows = []
    for quantile in threshold_quantiles:
        threshold = float(np.quantile(scores, quantile))
        pred = scores >= threshold
        rows.append(
            {
                "replicate": replicate,
                "method": method,
                "threshold_quantile": float(quantile),
                "threshold_score": threshold,
                "n_predicted_positive": int(pred.sum()),
                "true_positive": int((pred & (y_true == 1)).sum()),
                "false_positive": int((pred & (y_true == 0)).sum()),
                "recall": _safe_divide(int((pred & (y_true == 1)).sum()), int(y_true.sum())),
                "weak_signal_recall": _safe_divide(
                    int((pred & (y_true == 1) & weak).sum()),
                    int(((y_true == 1) & weak).sum()),
                ),
                "empirical_fdr": _safe_divide(
                    int((pred & (y_true == 0)).sum()), int(pred.sum())
                ),
                "false_positive_high_expression_call_rate": _safe_divide(
                    int((pred & fp_high).sum()), int(fp_high.sum())
                ),
                "false_positive_high_expression_fraction_among_calls": _safe_divide(
                    int((pred & fp_high).sum()), int(pred.sum())
                ),
            }
        )
    return pd.DataFrame(rows)


def _topk_summary(
    table: pd.DataFrame,
    method: str,
    replicate: str,
    score_column: str,
    top_k_values: tuple[int, ...],
) -> pd.DataFrame:
    ordered = table.sort_values(score_column, ascending=False)
    rows = []
    for top_k in top_k_values:
        subset = ordered.head(min(int(top_k), len(ordered)))
        y_true = subset["ground_truth"].astype(int)
        fp_high = subset["false_positive_high_expression"].astype(bool)
        weak_true = subset["weak_signal"].astype(bool) & (y_true == 1)
        rows.append(
            {
                "replicate": replicate,
                "method": method,
                "top_k": int(len(subset)),
                "precision": float(y_true.mean()) if len(subset) else float("nan"),
                "weak_true_count": int(weak_true.sum()),
                "false_positive_high_expression_count": int(fp_high.sum()),
                "false_positive_high_expression_fraction": _safe_divide(
                    int(fp_high.sum()), int(len(subset))
                ),
                "mechanism_counts": json.dumps(
                    subset["mechanism"].value_counts().to_dict(), ensure_ascii=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _lr_summary(
    table: pd.DataFrame,
    method: str,
    replicate: str,
    score_column: str,
) -> pd.DataFrame:
    rows = []
    for lr_id, group in table.groupby("lr_id", sort=False):
        truth = group["ground_truth"].astype(int)
        scores = group[score_column].to_numpy(dtype=float)
        rows.append(
            {
                "replicate": replicate,
                "method": method,
                "lr_id": lr_id,
                "ligand": group["ligand"].iloc[0] if "ligand" in group.columns else "",
                "receptor": group["receptor"].iloc[0] if "receptor" in group.columns else "",
                "mechanism": group["mechanism"].iloc[0],
                "weak_signal": bool(group["weak_signal"].astype(bool).any()),
                "false_positive_high_expression": bool(
                    group["false_positive_high_expression"].astype(bool).any()
                ),
                "n_candidates": int(len(group)),
                "n_positive": int(truth.sum()),
                "mean_score": float(np.mean(scores)),
                "q95_score": float(np.quantile(scores, 0.95)),
                "auroc": _binary_auc(truth.to_numpy(dtype=int), scores),
            }
        )
    return pd.DataFrame(rows).sort_values(["q95_score", "mean_score"], ascending=False)


def _mechanism_summary(
    table: pd.DataFrame,
    method: str,
    replicate: str,
    score_column: str,
) -> pd.DataFrame:
    rows = []
    for mechanism, group in table.groupby("mechanism", sort=False):
        truth = group["ground_truth"].astype(int).to_numpy()
        scores = group[score_column].to_numpy(dtype=float)
        rows.append(
            {
                "replicate": replicate,
                "method": method,
                "mechanism": mechanism,
                "n_candidates": int(len(group)),
                "n_positive": int(truth.sum()),
                "weak_positive": int(
                    ((group["weak_signal"].astype(bool).to_numpy()) & (truth == 1)).sum()
                ),
                "false_positive_high_expression_candidates": int(
                    group["false_positive_high_expression"].astype(bool).sum()
                ),
                "mean_score": float(np.mean(scores)),
                "auroc": _binary_auc(truth, scores),
            }
        )
    return pd.DataFrame(rows)


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--threshold-quantile", action="append", type=float, dest="threshold_quantiles")
    parser.add_argument("--top-k", action="append", type=int, dest="top_k_values")
    args = parser.parse_args()

    metrics = summarize_stress(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.output_dir),
        score_column=args.score_column,
        threshold_quantiles=tuple(args.threshold_quantiles or (0.9, 0.95, 0.99)),
        top_k_values=tuple(args.top_k_values or (50, 100, 500)),
    )
    print(
        pd.DataFrame([metrics])[
            [
                "replicate",
                "method",
                "auroc",
                "aupr",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
                "false_positive_high_expression_rate_at_threshold",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote weak-signal/false-positive stress outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
