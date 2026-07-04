#!/usr/bin/env python
"""Evaluate whether CCC methods recover LR-specific spatial ranges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_ground_truth_predictions


def evaluate_range_mechanism(
    replicate_dir: Path,
    output_dir: Path,
    *,
    predictions: dict[str, Path],
    top_k: int = 100,
    threshold_quantile: float = 0.95,
) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = []
    mechanism_rows = []
    lr_rows = []
    distance_rows = []
    confusion_rows = []

    for method, prediction_path in predictions.items():
        prediction = pd.read_csv(prediction_path)
        merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner")
        if merged.empty:
            raise ValueError(f"No overlapping candidate_id values for method={method}: {prediction_path}")
        merged["score_normalized"] = _minmax(merged["score"].to_numpy(dtype=float))
        merged = merged.merge(
            lr_pairs[["lr_id", "range_multiplier", "active_regions", "true_lr"]],
            on="lr_id",
            how="left",
            validate="many_to_one",
        )
        merged["true_range_class"] = merged["mechanism"].map(_range_class)
        merged["predicted_positive"] = merged["score_normalized"] >= float(
            np.quantile(merged["score_normalized"], threshold_quantile)
        )

        base = evaluate_ground_truth_predictions(
            candidates,
            prediction,
            top_k=top_k,
            threshold_quantile=threshold_quantile,
        )
        base.update(_range_distance_error(merged, subset_name="all"))
        base["method"] = method
        method_rows.append(base)

        for mechanism, group in merged.groupby("mechanism", sort=False):
            mechanism_rows.append(
                {
                    "method": method,
                    "mechanism": mechanism,
                    "range_class": _range_class(mechanism),
                    **_binary_metrics(group),
                    **_range_distance_error(group, subset_name=str(mechanism)),
                }
            )

        for lr_id, group in merged.groupby("lr_id", sort=False):
            lr_rows.append(
                {
                    "method": method,
                    "lr_id": lr_id,
                    "ligand": group["ligand"].iloc[0],
                    "receptor": group["receptor"].iloc[0],
                    "mechanism": group["mechanism"].iloc[0],
                    "range_class": _range_class(group["mechanism"].iloc[0]),
                    **_binary_metrics(group),
                    **_range_distance_error(group, subset_name=str(lr_id)),
                }
            )
            distance_rows.extend(_distance_bin_rows(method, lr_id, group))

        confusion_rows.extend(_range_confusion_rows(method, merged))

    tables = {
        "method_range_summary": pd.DataFrame(method_rows),
        "mechanism_range_summary": pd.DataFrame(mechanism_rows),
        "lr_range_summary": pd.DataFrame(lr_rows),
        "distance_bin_summary": pd.DataFrame(distance_rows),
        "range_confusion": pd.DataFrame(confusion_rows),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{replicate_dir.name}__{name}.csv", index=False)
    return tables


def _range_class(mechanism: str) -> str:
    mechanism = str(mechanism)
    if mechanism == "juxtacrine_contact":
        return "contact"
    if mechanism == "short_range_paracrine":
        return "short"
    if mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        return "medium"
    if mechanism == "long_range_chemokine_niche":
        return "long_niche"
    if mechanism == "false_positive_high_expression":
        return "negative_control"
    return "other"


def _binary_metrics(group: pd.DataFrame) -> dict:
    y = group["ground_truth"].astype(int).to_numpy()
    score = group["score_normalized"].to_numpy(dtype=float)
    predicted = group["predicted_positive"].astype(bool).to_numpy()
    top = group.sort_values("score_normalized", ascending=False).head(min(100, len(group)))
    wrong_range = (
        group["wrong_range_false_positive"].astype(bool).to_numpy()
        if "wrong_range_false_positive" in group
        else np.zeros(len(group), dtype=bool)
    )
    too_close_long = (
        group["range_violation_type"].astype(str).eq("too_close_for_long_range_lr").to_numpy()
        if "range_violation_type" in group
        else np.zeros(len(group), dtype=bool)
    )
    return {
        "n_candidates": int(len(group)),
        "n_positive": int(y.sum()),
        "n_wrong_range_false_positive": int(wrong_range.sum()),
        "n_too_close_long_range_false_positive": int((wrong_range & too_close_long).sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "auroc": _auroc(y, score),
        "aupr": _aupr(y, score),
        "top100_precision": float(top["ground_truth"].astype(int).mean()) if len(top) else np.nan,
        "recall_top5pct": _safe_divide(int((predicted & (y == 1)).sum()), int((y == 1).sum())),
        "fdr_top5pct": _safe_divide(int((predicted & (y == 0)).sum()), int(predicted.sum())),
        "wrong_range_rate_top5pct": _safe_divide(int((predicted & wrong_range).sum()), int(predicted.sum())),
        "too_close_long_range_rate_top5pct": _safe_divide(
            int((predicted & wrong_range & too_close_long).sum()),
            int(predicted.sum()),
        ),
    }


def _range_distance_error(group: pd.DataFrame, *, subset_name: str) -> dict:
    positives = group[group["ground_truth"].astype(int) == 1]
    if positives.empty:
        return {
            "subset": subset_name,
            "true_positive_median_distance": np.nan,
            "predicted_top_median_distance": np.nan,
            "distance_median_error": np.nan,
            "distance_q75_error": np.nan,
            "distance_profile_emd": np.nan,
        }
    top_n = max(int(len(positives)), 1)
    predicted_top = group.sort_values("score_normalized", ascending=False).head(top_n)
    true_distance = positives["distance"].to_numpy(dtype=float)
    pred_distance = predicted_top["distance"].to_numpy(dtype=float)
    return {
        "subset": subset_name,
        "true_positive_median_distance": float(np.median(true_distance)),
        "predicted_top_median_distance": float(np.median(pred_distance)),
        "distance_median_error": float(abs(np.median(pred_distance) - np.median(true_distance))),
        "distance_q75_error": float(abs(np.quantile(pred_distance, 0.75) - np.quantile(true_distance, 0.75))),
        "distance_profile_emd": _histogram_emd(true_distance, pred_distance),
    }


def _distance_bin_rows(method: str, lr_id: str, group: pd.DataFrame) -> list[dict]:
    if group.empty:
        return []
    distance = group["distance"].to_numpy(dtype=float)
    if np.nanmax(distance) <= np.nanmin(distance):
        bins = np.array([np.nanmin(distance), np.nanmax(distance) + 1e-6])
    else:
        bins = np.quantile(distance, np.linspace(0.0, 1.0, 7))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([np.nanmin(distance), np.nanmax(distance) + 1e-6])
    rows = []
    labels = pd.cut(group["distance"], bins=bins, include_lowest=True, duplicates="drop")
    for bin_label, subset in group.groupby(labels, observed=True):
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                "mechanism": subset["mechanism"].iloc[0],
                "range_class": _range_class(subset["mechanism"].iloc[0]),
                "distance_bin": str(bin_label),
                "n_candidates": int(len(subset)),
                "truth_rate": float(subset["ground_truth"].astype(int).mean()),
                "mean_score": float(subset["score_normalized"].mean()),
                "median_score": float(subset["score_normalized"].median()),
                "median_distance": float(subset["distance"].median()),
            }
        )
    return rows


def _range_confusion_rows(method: str, merged: pd.DataFrame) -> list[dict]:
    rows = []
    selected = merged[merged["predicted_positive"].astype(bool)].copy()
    if selected.empty:
        return rows
    for true_class, group in selected.groupby("true_range_class", sort=False):
        rows.append(
            {
                "method": method,
                "true_range_class": true_class,
                "n_predicted_positive": int(len(group)),
                "truth_rate": float(group["ground_truth"].astype(int).mean()),
                "median_distance": float(group["distance"].median()),
                "lr_counts": ";".join(
                    group["lr_id"].value_counts().astype(str).pipe(lambda s: s.index + "=" + s).tolist()
                ),
            }
        )
    return rows


def _histogram_emd(a: np.ndarray, b: np.ndarray, n_bins: int = 20) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    low = float(np.nanmin(np.r_[a, b]))
    high = float(np.nanmax(np.r_[a, b]))
    if high <= low:
        return 0.0
    bins = np.linspace(low, high, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=False)
    hb, _ = np.histogram(b, bins=bins, density=False)
    ha = ha / max(float(ha.sum()), 1.0)
    hb = hb / max(float(hb.sum()), 1.0)
    return float(np.mean(np.abs(np.cumsum(ha) - np.cumsum(hb))))


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


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    sum_pos = float(ranks[y_true == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.trapz(np.r_[1.0, precision], np.r_[0.0, recall]))


def _parse_prediction(values: list[str] | None) -> dict[str, Path]:
    predictions = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Prediction must be METHOD=PATH, got: {value}")
        method, path = value.split("=", 1)
        predictions[method] = Path(path)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replicate_dir")
    parser.add_argument("--output-dir", default="outputs/fair_benchmark_v2/range_mechanism_eval")
    parser.add_argument("--prediction", action="append", help="Prediction file as METHOD=PATH. Repeatable.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()

    predictions = _parse_prediction(args.prediction)
    if not predictions:
        raise ValueError("At least one --prediction METHOD=PATH is required.")
    tables = evaluate_range_mechanism(
        Path(args.replicate_dir),
        Path(args.output_dir),
        predictions=predictions,
        top_k=args.top_k,
        threshold_quantile=args.threshold_quantile,
    )
    print(tables["method_range_summary"][[
        "method",
        "auroc",
        "aupr",
        "top_k_precision",
        "distance_median_error",
        "distance_profile_emd",
    ]].to_string(index=False))
    print(f"Wrote range mechanism evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
