#!/usr/bin/env python
"""Calculate extended metrics for the scDesign3 200-LR benchmark.

The added metrics align the benchmark with common SRT CCC benchmarking panels:
precision, recall/sensitivity, F1, specificity, detected interaction counts,
and method-overlap Jaccard, while keeping the existing ranking/FDR/range metrics.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = ROOT / "extended_metrics"

THRESHOLD_QUANTILE = 0.95
TOP_FRACTION = 1.0 - THRESHOLD_QUANTILE
TOP_K = 100

METHODS = [
    ("SpatialSelf", "ScaleComm", "spatialself_range_context"),
    ("COMMOT", "COMMOT", "commot"),
    ("SpaOTsc", "SpaOTsc", "spaotsc"),
    ("CellPhoneDB", "CellPhoneDB", "cellphonedb"),
    ("CellChat", "CellChat", "cellchat"),
    ("NICHES", "NICHES", "niches"),
    ("SpaTalk", "SpaTalk", "spatalk"),
    ("CellNEST", "CellNEST", "cellnest"),
    ("DeepTalk", "DeepTalk", "deeptalk"),
    ("CytoSignal", "CytoSignal", "cytosignal"),
]

DATASETS = [
    {
        "dataset_id": "lymph_scdesign3_200lr",
        "reference": "LN reference",
        "tissue": "lymph_node",
        "noise": "clean",
        "candidate_path": ROOT
        / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "method_runs/lymph_scdesign3_200lr",
        "prediction_prefix": "reference_lymph_node_scdesign3_200lr_replicate_00",
    },
    {
        "dataset_id": "melanoma_scdesign3_200lr",
        "reference": "Melanoma reference",
        "tissue": "melanoma",
        "noise": "clean",
        "candidate_path": ROOT
        / "datasets/reference_melanoma_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "method_runs/melanoma_scdesign3_200lr",
        "prediction_prefix": "reference_melanoma_scdesign3_200lr_replicate_00",
    },
    {
        "dataset_id": "lymph_scdesign3_200lr_low_noise",
        "reference": "LN reference",
        "tissue": "lymph_node",
        "noise": "low",
        "candidate_path": ROOT
        / "noisy_datasets/lymph_scdesign3_200lr_low_noise/reference_lymph_node_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "noisy_method_runs/lymph_scdesign3_200lr_low_noise",
        "prediction_prefix": "reference_lymph_node_scdesign3_200lr_replicate_00",
    },
    {
        "dataset_id": "lymph_scdesign3_200lr_high_noise",
        "reference": "LN reference",
        "tissue": "lymph_node",
        "noise": "high",
        "candidate_path": ROOT
        / "noisy_datasets/lymph_scdesign3_200lr_high_noise/reference_lymph_node_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "noisy_method_runs/lymph_scdesign3_200lr_high_noise",
        "prediction_prefix": "reference_lymph_node_scdesign3_200lr_replicate_00",
    },
    {
        "dataset_id": "melanoma_scdesign3_200lr_low_noise",
        "reference": "Melanoma reference",
        "tissue": "melanoma",
        "noise": "low",
        "candidate_path": ROOT
        / "noisy_datasets/melanoma_scdesign3_200lr_low_noise/reference_melanoma_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "noisy_method_runs/melanoma_scdesign3_200lr_low_noise",
        "prediction_prefix": "reference_melanoma_scdesign3_200lr_replicate_00",
    },
    {
        "dataset_id": "melanoma_scdesign3_200lr_high_noise",
        "reference": "Melanoma reference",
        "tissue": "melanoma",
        "noise": "high",
        "candidate_path": ROOT
        / "noisy_datasets/melanoma_scdesign3_200lr_high_noise/reference_melanoma_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "run_root": ROOT / "noisy_method_runs/melanoma_scdesign3_200lr_high_noise",
        "prediction_prefix": "reference_melanoma_scdesign3_200lr_replicate_00",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    selected_long_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        candidates = pd.read_csv(dataset["candidate_path"])
        selected_by_method: dict[str, dict[str, set[str]]] = {}
        for method, method_label, method_dir in METHODS:
            pred_path = (
                dataset["run_root"]
                / method_dir
                / f"{dataset['prediction_prefix']}__{method_dir}_predictions.csv"
            )
            if method == "SpatialSelf":
                pred_path = (
                    dataset["run_root"]
                    / method_dir
                    / f"{dataset['prediction_prefix']}__spatialself_range_context_predictions.csv"
                )
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction for {dataset['dataset_id']} {method}: {pred_path}")
            pred = pd.read_csv(pred_path)
            merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
            if merged.empty:
                raise ValueError(f"No overlapping candidates for {dataset['dataset_id']} {method}")

            scored = _with_detection(merged, threshold_quantile=THRESHOLD_QUANTILE)
            selected = scored[scored["detected"]].copy()
            selected_by_method[method] = {
                "candidate": set(selected["candidate_id"].astype(str)),
                "lr_pair": set(selected["lr_id"].astype(str)),
            }

            metric_row = {
                **_dataset_meta(dataset),
                "method": method,
                "method_label": method_label,
                **_binary_metrics(scored),
                **_ranking_metrics(scored),
                **_range_metrics(scored),
            }
            rows.append(metric_row)
            selected_long_rows.extend(_selected_interaction_rows(dataset, method, method_label, scored))

            for (mechanism, range_class), group in scored.groupby(["mechanism", "range_class"], dropna=False, sort=False):
                mechanism_rows.append(
                    {
                        **_dataset_meta(dataset),
                        "method": method,
                        "method_label": method_label,
                        "mechanism": mechanism,
                        "range_class": range_class,
                        **_binary_metrics(group),
                        **_ranking_metrics(group),
                        **_range_metrics(group),
                    }
                )

        overlap_rows.extend(_overlap_rows(dataset, selected_by_method))

    metrics = pd.DataFrame(rows)
    mechanism = pd.DataFrame(mechanism_rows)
    overlap = pd.DataFrame(overlap_rows)
    selected_long = pd.DataFrame(selected_long_rows)
    overlap_summary = _summarize_overlap(overlap)
    summary = _summarize_methods(metrics).merge(
        _overlap_wide(overlap), on=["method", "method_label"], how="left"
    )
    clean_summary = _summarize_methods(metrics[metrics["noise"].eq("clean")]).merge(
        _overlap_wide(overlap[overlap["noise"].eq("clean")]), on=["method", "method_label"], how="left"
    )

    metrics.to_csv(OUT / "scdesign3_200lr_extended_method_metrics.csv", index=False)
    summary.to_csv(OUT / "scdesign3_200lr_extended_method_summary.csv", index=False)
    clean_summary.to_csv(OUT / "scdesign3_200lr_extended_clean_method_summary.csv", index=False)
    mechanism.to_csv(OUT / "scdesign3_200lr_extended_by_mechanism.csv", index=False)
    overlap.to_csv(OUT / "scdesign3_200lr_method_overlap_pairwise.csv", index=False)
    overlap_summary.to_csv(OUT / "scdesign3_200lr_method_overlap_summary.csv", index=False)
    selected_long.to_csv(OUT / "scdesign3_200lr_detected_interactions_long.csv", index=False)
    _write_manifest(metrics, mechanism, overlap)

    print("Wrote extended benchmark metrics to", OUT)
    display_cols = [
        "method_label",
        "aupr_mean",
        "auroc_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "f1_mean",
        "empirical_fdr_mean",
        "wrong_range_rate_mean",
        "detected_candidate_count_mean",
        "candidate_jaccard_mean",
    ]
    print(summary[display_cols].round(4).sort_values("aupr_mean", ascending=False).to_string(index=False))


def _dataset_meta(dataset: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_id": dataset["dataset_id"],
        "reference": dataset["reference"],
        "tissue": dataset["tissue"],
        "noise": dataset["noise"],
    }


def _range_class(mechanism: object) -> str:
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


def _with_detection(df: pd.DataFrame, *, threshold_quantile: float) -> pd.DataFrame:
    out = df.copy()
    out["score_normalized"] = _minmax(out["score"].to_numpy(dtype=float))
    top_n = max(1, int(np.ceil(len(out) * (1.0 - threshold_quantile))))
    ranked = out.sort_values(["score_normalized", "candidate_id"], ascending=[False, True], kind="mergesort")
    selected_idx = ranked.index[:top_n]
    threshold = float(out.loc[selected_idx, "score_normalized"].min())
    out["threshold_quantile"] = float(threshold_quantile)
    out["threshold_score"] = threshold
    out["detected"] = False
    out.loc[selected_idx, "detected"] = True
    out["detection_rule"] = f"exact_top_{top_n}_of_{len(out)}"
    out["range_class"] = out["mechanism"].map(_range_class)
    return out


def _binary_metrics(group: pd.DataFrame) -> dict[str, object]:
    y = group["ground_truth"].astype(int).to_numpy()
    pred = group["detected"].astype(bool).to_numpy()
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if np.isfinite(precision + recall) else np.nan
    weak = group["weak_signal"].astype(bool).to_numpy() if "weak_signal" in group else np.zeros(len(group), dtype=bool)
    high_expr_fp = (
        group["false_positive_high_expression"].astype(bool).to_numpy()
        if "false_positive_high_expression" in group
        else np.zeros(len(group), dtype=bool)
    )
    return {
        "n_candidates": int(len(group)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": _safe_divide(fp, fp + tn),
        "empirical_fdr": _safe_divide(fp, tp + fp),
        "detected_candidate_count": int(pred.sum()),
        "detected_candidate_fraction": float(pred.mean()) if len(pred) else np.nan,
        "detected_lr_pair_count": int(group.loc[pred, "lr_id"].astype(str).nunique()) if "lr_id" in group else np.nan,
        "weak_signal_recall": _safe_divide(int(((y == 1) & pred & weak).sum()), int(((y == 1) & weak).sum())),
        "high_expression_fp_rate": _safe_divide(int((pred & high_expr_fp).sum()), int(high_expr_fp.sum())),
        "threshold_quantile": float(group["threshold_quantile"].iloc[0]) if "threshold_quantile" in group and len(group) else np.nan,
        "threshold_score": float(group["threshold_score"].iloc[0]) if "threshold_score" in group and len(group) else np.nan,
    }


def _ranking_metrics(group: pd.DataFrame) -> dict[str, object]:
    y = group["ground_truth"].astype(int).to_numpy()
    scores = group["score_normalized"].to_numpy(dtype=float)
    top_k = min(TOP_K, len(group))
    order = np.argsort(-scores)
    top = order[:top_k]
    return {
        "auroc": _auroc(y, scores),
        "aupr": _aupr(y, scores),
        "top_k": int(top_k),
        "top_k_precision": float(y[top].mean()) if len(top) else np.nan,
        "calibration_error_10bin": _expected_calibration_error(y, scores, 10),
    }


def _range_metrics(group: pd.DataFrame) -> dict[str, object]:
    pred = group["detected"].astype(bool).to_numpy()
    wrong = (
        group["wrong_range_false_positive"].astype(bool).to_numpy()
        if "wrong_range_false_positive" in group
        else np.zeros(len(group), dtype=bool)
    )
    too_close = (
        group["range_violation_type"].astype(str).eq("too_close_for_long_range_lr").to_numpy()
        if "range_violation_type" in group
        else np.zeros(len(group), dtype=bool)
    )
    positives = group[group["ground_truth"].astype(int).eq(1)]
    selected = group[group["detected"]]
    if positives.empty or selected.empty:
        distance_metrics = {
            "true_positive_median_distance": np.nan,
            "detected_median_distance": np.nan,
            "distance_median_error": np.nan,
            "distance_profile_emd": np.nan,
        }
    else:
        true_dist = positives["distance"].to_numpy(dtype=float)
        selected_dist = selected["distance"].to_numpy(dtype=float)
        distance_metrics = {
            "true_positive_median_distance": float(np.median(true_dist)),
            "detected_median_distance": float(np.median(selected_dist)),
            "distance_median_error": float(abs(np.median(selected_dist) - np.median(true_dist))),
            "distance_profile_emd": _histogram_emd(true_dist, selected_dist),
        }
    return {
        "wrong_range_fp_count": int((pred & wrong).sum()),
        "wrong_range_rate": _safe_divide(int((pred & wrong).sum()), int(pred.sum())),
        "too_close_long_range_fp_count": int((pred & wrong & too_close).sum()),
        "too_close_long_range_rate": _safe_divide(int((pred & wrong & too_close).sum()), int(pred.sum())),
        **distance_metrics,
    }


def _selected_interaction_rows(
    dataset: dict[str, object], method: str, method_label: str, scored: pd.DataFrame
) -> list[dict[str, object]]:
    selected = scored[scored["detected"]]
    rows = []
    for lr_id, sub in selected.groupby("lr_id", sort=False):
        rows.append(
            {
                **_dataset_meta(dataset),
                "method": method,
                "method_label": method_label,
                "lr_id": lr_id,
                "ligand": sub["ligand"].iloc[0] if "ligand" in sub else "",
                "receptor": sub["receptor"].iloc[0] if "receptor" in sub else "",
                "mechanism": sub["mechanism"].iloc[0] if "mechanism" in sub else "",
                "range_class": sub["range_class"].iloc[0] if "range_class" in sub else "",
                "detected_candidate_count": int(len(sub)),
                "true_positive_count": int(sub["ground_truth"].astype(int).sum()),
                "wrong_range_fp_count": int(sub.get("wrong_range_false_positive", pd.Series(False, index=sub.index)).astype(bool).sum()),
                "mean_score_normalized": float(sub["score_normalized"].mean()),
                "median_distance": float(sub["distance"].median()) if "distance" in sub else np.nan,
            }
        )
    return rows


def _overlap_rows(dataset: dict[str, object], selected_by_method: dict[str, dict[str, set[str]]]) -> list[dict[str, object]]:
    rows = []
    for level in ["candidate", "lr_pair"]:
        for method_a, method_b in combinations([method for method, _, _ in METHODS], 2):
            a = selected_by_method[method_a][level]
            b = selected_by_method[method_b][level]
            rows.append(
                {
                    **_dataset_meta(dataset),
                    "overlap_level": level,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_a": len(a),
                    "n_b": len(b),
                    "intersection": len(a & b),
                    "union": len(a | b),
                    "jaccard": _safe_divide(len(a & b), len(a | b)),
                }
            )
    return rows


def _summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "auroc",
        "aupr",
        "top_k_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "empirical_fdr",
        "weak_signal_recall",
        "high_expression_fp_rate",
        "calibration_error_10bin",
        "wrong_range_rate",
        "too_close_long_range_rate",
        "distance_median_error",
        "distance_profile_emd",
        "detected_candidate_count",
        "detected_candidate_fraction",
        "detected_lr_pair_count",
    ]
    summary = (
        metrics.groupby(["method", "method_label"], sort=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    return summary


def _summarize_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    if overlap.empty:
        return pd.DataFrame()
    rows = []
    for level, sub in overlap.groupby("overlap_level", sort=False):
        for method, method_label, _method_dir in METHODS:
            mask = sub["method_a"].eq(method) | sub["method_b"].eq(method)
            rows.append(
                {
                    "overlap_level": level,
                    "method": method,
                    "method_label": method_label,
                    "jaccard_mean": float(sub.loc[mask, "jaccard"].mean()),
                    "jaccard_std": float(sub.loc[mask, "jaccard"].std()),
                    "intersection_mean": float(sub.loc[mask, "intersection"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    wide = summary.pivot(index=["method", "method_label"], columns="overlap_level", values="jaccard_mean").reset_index()
    wide = wide.rename(columns={"candidate": "candidate_jaccard_mean", "lr_pair": "lr_pair_jaccard_mean"})
    return summary.merge(wide, on=["method", "method_label"], how="left")


def _overlap_wide(overlap: pd.DataFrame) -> pd.DataFrame:
    if overlap.empty:
        return pd.DataFrame(columns=["method", "method_label", "candidate_jaccard_mean", "lr_pair_jaccard_mean"])
    summary = _summarize_overlap(overlap)
    return summary[
        ["method", "method_label", "candidate_jaccard_mean", "lr_pair_jaccard_mean"]
    ].drop_duplicates()


def _write_manifest(metrics: pd.DataFrame, mechanism: pd.DataFrame, overlap: pd.DataFrame) -> None:
    manifest = {
        "benchmark": "scDesign3 200-LR mechanism benchmark",
        "threshold_definition": (
            f"detected = exact top {TOP_FRACTION:.0%} ranked candidates per dataset/method; "
            "ties are ordered deterministically by candidate_id"
        ),
        "top_k": TOP_K,
        "n_methods": int(metrics["method"].nunique()),
        "n_baselines": int(metrics["method"].nunique() - 1),
        "methods": metrics["method_label"].drop_duplicates().tolist(),
        "datasets": metrics["dataset_id"].drop_duplicates().tolist(),
        "metric_tables": {
            "method_metrics": "scdesign3_200lr_extended_method_metrics.csv",
            "method_summary": "scdesign3_200lr_extended_method_summary.csv",
            "clean_method_summary": "scdesign3_200lr_extended_clean_method_summary.csv",
            "by_mechanism": "scdesign3_200lr_extended_by_mechanism.csv",
            "overlap_pairwise": "scdesign3_200lr_method_overlap_pairwise.csv",
            "overlap_summary": "scdesign3_200lr_method_overlap_summary.csv",
            "detected_interactions_long": "scdesign3_200lr_detected_interactions_long.csv",
        },
        "n_rows": {
            "method_metrics": int(len(metrics)),
            "by_mechanism": int(len(mechanism)),
            "overlap_pairwise": int(len(overlap)),
        },
    }
    (OUT / "scdesign3_200lr_extended_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
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
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.trapezoid(precision, recall))


def _expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (scores >= start) & (scores <= end) if end == 1.0 else (scores >= start) & (scores < end)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += float(mask.mean()) * abs(confidence - accuracy)
    return float(error)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
