#!/usr/bin/env python
"""Diagnose fair benchmark v2 predictions and SpatialSelf pseudo-labels.

This script is intentionally read-only with respect to model outputs. It helps
separate benchmark fairness issues from model/training issues by summarizing:

* method performance by communication mechanism;
* top-ranked candidate composition and high-expression false positives;
* LR-level ranking behavior;
* pseudo-label precision for clean SpatialSelf training strategies.
"""

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
from scripts.benchmark.run_spatialself import _pseudo_label_indices, _pseudo_label_score


def diagnose_fair_benchmark(
    replicate_dir: Path,
    output_dir: Path,
    *,
    predictions: dict[str, Path],
    pseudo_label_strategies: tuple[str, ...],
    top_k: int,
    threshold_quantile: float,
    positive_quantile: float,
    negative_quantile: float,
) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    mechanism_rows = []
    top_rows = []
    lr_rows = []

    for method, prediction_path in predictions.items():
        pred = pd.read_csv(prediction_path)
        merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        if merged.empty:
            raise ValueError(f"No overlapping candidate_id values for method={method}: {prediction_path}")
        score = _minmax(merged["score"].to_numpy(dtype=float))
        merged = merged.assign(score_normalized=score)

        metrics = evaluate_ground_truth_predictions(
            candidates,
            pred,
            top_k=top_k,
            threshold_quantile=threshold_quantile,
        )
        metric_rows.append(
            {
                "method": method,
                "n_candidates": metrics["n_candidates"],
                "n_positive": metrics["n_positive"],
                "auroc": metrics["auroc"],
                "aupr": metrics["aupr"],
                "top_k_precision": metrics["top_k_precision"],
                "recall_at_threshold": metrics["recall_at_threshold"],
                "weak_signal_recall_at_threshold": metrics["weak_signal_recall_at_threshold"],
                "empirical_fdr_at_threshold": metrics["empirical_fdr_at_threshold"],
                "false_positive_high_expression_rate_at_threshold": metrics[
                    "false_positive_high_expression_rate_at_threshold"
                ],
                "calibration_error_10bin": metrics["calibration_error_10bin"],
            }
        )
        mechanism_rows.extend(_summarize_by_mechanism(method, merged, top_k=top_k))
        top_rows.extend(_summarize_top_candidates(method, merged, top_k=top_k, threshold_quantile=threshold_quantile))
        lr_rows.extend(_summarize_lr_ranks(method, merged))

    pseudo_rows, pseudo_lr_rows = _summarize_pseudo_labels(
        candidates,
        pseudo_label_strategies=pseudo_label_strategies,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
    )

    tables = {
        "method_summary": pd.DataFrame(metric_rows).sort_values("aupr", ascending=False),
        "mechanism_summary": pd.DataFrame(mechanism_rows),
        "top_candidate_summary": pd.DataFrame(top_rows),
        "lr_rank_summary": pd.DataFrame(lr_rows),
        "pseudo_label_summary": pd.DataFrame(pseudo_rows),
        "pseudo_label_lr_summary": pd.DataFrame(pseudo_lr_rows),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{replicate_dir.name}__{name}.csv", index=False)
    return tables


def _summarize_by_mechanism(method: str, merged: pd.DataFrame, *, top_k: int) -> list[dict]:
    rows = []
    for mechanism, group in merged.groupby("mechanism", sort=False):
        y = group["ground_truth"].astype(int).to_numpy()
        score = group["score_normalized"].to_numpy(dtype=float)
        top = group.sort_values("score_normalized", ascending=False).head(min(top_k, len(group)))
        threshold = float(np.quantile(score, 0.95)) if len(group) else float("nan")
        selected = group["score_normalized"] >= threshold
        rows.append(
            {
                "method": method,
                "mechanism": mechanism,
                "n_candidates": int(len(group)),
                "n_positive": int(y.sum()),
                "positive_rate": _safe_mean(y),
                "auroc": _auroc(y, score),
                "aupr": _aupr(y, score),
                "top_k_precision_within_mechanism": float(top["ground_truth"].astype(int).mean()) if len(top) else np.nan,
                "fdr_top5pct_within_mechanism": _safe_divide(
                    int((selected & (group["ground_truth"].astype(int) == 0)).sum()),
                    int(selected.sum()),
                ),
                "weak_recall_top5pct_within_mechanism": _safe_divide(
                    int((selected & (group["ground_truth"].astype(int) == 1) & group["weak_signal"].astype(bool)).sum()),
                    int(((group["ground_truth"].astype(int) == 1) & group["weak_signal"].astype(bool)).sum()),
                ),
                "high_expr_fp_rate_top5pct_within_mechanism": _safe_divide(
                    int((selected & group["false_positive_high_expression"].astype(bool)).sum()),
                    int(group["false_positive_high_expression"].astype(bool).sum()),
                ),
            }
        )
    return rows


def _summarize_top_candidates(
    method: str,
    merged: pd.DataFrame,
    *,
    top_k: int,
    threshold_quantile: float,
) -> list[dict]:
    rows = []
    top = merged.sort_values("score_normalized", ascending=False).head(min(top_k, len(merged)))
    rows.append(_composition_row(method, "top_k", top))
    selected = merged[merged["score_normalized"] >= float(np.quantile(merged["score_normalized"], threshold_quantile))]
    rows.append(_composition_row(method, f"top_{int((1.0 - threshold_quantile) * 100)}pct", selected))
    for lr_id, group in top.groupby("lr_id", sort=False):
        rows.append(
            {
                "method": method,
                "subset": "top_k_by_lr",
                "lr_id": lr_id,
                "mechanism": group["mechanism"].iloc[0],
                "n": int(len(group)),
                "truth_rate": float(group["ground_truth"].astype(int).mean()),
                "weak_truth_rate": float(
                    ((group["ground_truth"].astype(int) == 1) & group["weak_signal"].astype(bool)).mean()
                ),
                "false_positive_high_expression_rate": float(group["false_positive_high_expression"].astype(bool).mean()),
                "mean_score": float(group["score_normalized"].mean()),
            }
        )
    return rows


def _composition_row(method: str, subset_name: str, table: pd.DataFrame) -> dict:
    truth = table["ground_truth"].astype(int)
    weak_truth = (truth == 1) & table["weak_signal"].astype(bool)
    fp_high = table["false_positive_high_expression"].astype(bool)
    return {
        "method": method,
        "subset": subset_name,
        "lr_id": "ALL",
        "mechanism": "ALL",
        "n": int(len(table)),
        "truth_rate": float(truth.mean()) if len(table) else np.nan,
        "weak_truth_rate": float(weak_truth.mean()) if len(table) else np.nan,
        "false_positive_high_expression_rate": float(fp_high.mean()) if len(table) else np.nan,
        "mean_score": float(table["score_normalized"].mean()) if len(table) else np.nan,
        "mechanism_counts": ";".join(table["mechanism"].value_counts().astype(str).pipe(lambda s: s.index + "=" + s).tolist())
        if len(table)
        else "",
        "lr_counts": ";".join(table["lr_id"].value_counts().astype(str).pipe(lambda s: s.index + "=" + s).tolist())
        if len(table)
        else "",
    }


def _summarize_lr_ranks(method: str, merged: pd.DataFrame) -> list[dict]:
    rows = []
    lr_summary = (
        merged.groupby(["lr_id", "ligand", "receptor", "mechanism"], dropna=False)
        .agg(
            n_candidates=("candidate_id", "size"),
            n_positive=("ground_truth", "sum"),
            truth_rate=("ground_truth", "mean"),
            weak_positive_count=("weak_signal", lambda x: int(((merged.loc[x.index, "ground_truth"].astype(int) == 1) & x.astype(bool)).sum())),
            false_positive_high_expression_count=("false_positive_high_expression", "sum"),
            mean_score=("score_normalized", "mean"),
            max_score=("score_normalized", "max"),
            top20_mean_score=("score_normalized", lambda x: float(np.mean(np.sort(x.to_numpy(dtype=float))[-min(20, len(x)) :]))),
        )
        .reset_index()
    )
    lr_summary["rank_by_mean"] = lr_summary["mean_score"].rank(method="min", ascending=False).astype(int)
    lr_summary["rank_by_top20_mean"] = lr_summary["top20_mean_score"].rank(method="min", ascending=False).astype(int)
    lr_summary["method"] = method
    rows.extend(lr_summary.sort_values("rank_by_top20_mean").to_dict("records"))
    return rows


def _summarize_pseudo_labels(
    candidates: pd.DataFrame,
    *,
    pseudo_label_strategies: tuple[str, ...],
    positive_quantile: float,
    negative_quantile: float,
) -> tuple[list[dict], list[dict]]:
    rows = []
    lr_rows = []
    for strategy in pseudo_label_strategies:
        score = _pseudo_label_score(candidates, strategy=strategy)
        pos_idx, neg_idx = _pseudo_label_indices(
            candidates,
            score,
            strategy=strategy,
            positive_quantile=positive_quantile,
            negative_quantile=negative_quantile,
        )
        for subset_name, idx in (("pseudo_positive", pos_idx), ("pseudo_negative", neg_idx)):
            subset = candidates.iloc[idx]
            rows.append(_pseudo_composition_row(strategy, subset_name, subset))
        for lr_id, group in candidates.iloc[pos_idx].groupby("lr_id", sort=False):
            lr_rows.append(_pseudo_composition_row(strategy, "pseudo_positive_by_lr", group) | {"lr_id": lr_id})
    return rows, lr_rows


def _pseudo_composition_row(strategy: str, subset_name: str, subset: pd.DataFrame) -> dict:
    truth = subset["ground_truth"].astype(int)
    weak_truth = (truth == 1) & subset["weak_signal"].astype(bool)
    fp_high = subset["false_positive_high_expression"].astype(bool)
    return {
        "strategy": strategy,
        "subset": subset_name,
        "n": int(len(subset)),
        "truth_count": int(truth.sum()),
        "truth_rate": float(truth.mean()) if len(subset) else np.nan,
        "weak_truth_count": int(weak_truth.sum()),
        "weak_truth_rate": float(weak_truth.mean()) if len(subset) else np.nan,
        "false_positive_high_expression_count": int(fp_high.sum()),
        "false_positive_high_expression_rate": float(fp_high.mean()) if len(subset) else np.nan,
        "mechanism_counts": ";".join(subset["mechanism"].value_counts().astype(str).pipe(lambda s: s.index + "=" + s).tolist())
        if len(subset)
        else "",
    }


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


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


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
    parser.add_argument("--output-dir", default="outputs/fair_benchmark_v2/diagnostics")
    parser.add_argument(
        "--prediction",
        action="append",
        help="Prediction file as METHOD=PATH. Repeatable.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--positive-quantile", type=float, default=0.9)
    parser.add_argument("--negative-quantile", type=float, default=0.2)
    parser.add_argument(
        "--pseudo-label-strategy",
        action="append",
        dest="pseudo_label_strategies",
        default=[],
    )
    args = parser.parse_args()

    replicate_dir = Path(args.replicate_dir)
    predictions = _parse_prediction(args.prediction)
    if not predictions:
        raise ValueError("At least one --prediction METHOD=PATH is required.")
    pseudo_label_strategies = tuple(
        args.pseudo_label_strategies
        or [
            "expression_distance",
            "lr_balanced_expression_distance",
            "lr_capped_expression_distance",
            "expression_plus_distance",
            "mechanism_expression_distance",
            "mechanism_niche_expression_distance",
            "lr_capped_mechanism_niche_expression_distance",
        ]
    )

    tables = diagnose_fair_benchmark(
        replicate_dir,
        Path(args.output_dir),
        predictions=predictions,
        pseudo_label_strategies=pseudo_label_strategies,
        top_k=args.top_k,
        threshold_quantile=args.threshold_quantile,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
    )
    print(tables["method_summary"].to_string(index=False))
    print("\nPseudo-label summary:")
    print(tables["pseudo_label_summary"].to_string(index=False))
    print(f"\nWrote fair benchmark diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
