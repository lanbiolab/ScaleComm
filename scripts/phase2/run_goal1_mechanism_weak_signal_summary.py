#!/usr/bin/env python
"""Summarize Goal 1 weak-signal recovery by CCC mechanism."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ANALYSES = (
    {
        "analysis_label": "controlled_five_replicate_initial",
        "dataset_root": "outputs/phase1/exp01_controlled_benchmark/datasets",
        "prediction_root": "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial",
        "output_subdir": "controlled_five_replicate_mechanism_summary",
        "threshold_quantile": 0.95,
    },
    {
        "analysis_label": "full_slide_sparse_initial",
        "dataset_root": "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets",
        "prediction_root": "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial",
        "output_subdir": "full_slide_sparse_mechanism_summary",
        "threshold_quantile": 0.95,
    },
    {
        "analysis_label": "full_slide_sparse_operating_q94",
        "dataset_root": "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets",
        "prediction_root": "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial",
        "output_subdir": "full_slide_sparse_operating_q94_mechanism_summary",
        "threshold_quantile": 0.94,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/phase2/goal1_weak_signal_recovery")
    parser.add_argument("--analysis-label", action="append", dest="analysis_labels")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / args.output_root
    selected = set(args.analysis_labels or [])
    summaries = []
    for config in DEFAULT_ANALYSES:
        if selected and config["analysis_label"] not in selected:
            continue
        result = _run_analysis(config, output_root)
        summaries.append(result)
    combined = pd.concat([item["summary"] for item in summaries], ignore_index=True) if summaries else pd.DataFrame()
    if not combined.empty:
        combined_path = output_root / "goal1_mechanism_weak_signal_combined_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined mechanism summary: {combined_path}")


def _run_analysis(config: dict[str, object], output_root: Path) -> dict[str, pd.DataFrame]:
    dataset_root = PROJECT_ROOT / str(config["dataset_root"])
    prediction_root = PROJECT_ROOT / str(config["prediction_root"])
    output_dir = output_root / str(config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_quantile = float(config["threshold_quantile"])
    replicate_dirs = _replicate_dirs(dataset_root)
    rows = []
    for ablation_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        for replicate_dir in replicate_dirs:
            prediction_path = _prediction_path(ablation_dir, replicate_dir)
            if not prediction_path.exists():
                continue
            candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
            prediction = pd.read_csv(prediction_path)
            rows.extend(
                _mechanism_rows(
                    candidates,
                    prediction,
                    analysis_label=str(config["analysis_label"]),
                    ablation=ablation_dir.name,
                    replicate=replicate_dir.name,
                    threshold_quantile=threshold_quantile,
                )
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError(f"No mechanism rows generated for {config['analysis_label']}")
    summary = _summary(detail)
    overview = _overview(summary, config)
    detail_path = output_dir / "mechanism_weak_signal_detail.csv"
    summary_path = output_dir / "mechanism_weak_signal_summary.csv"
    overview_path = output_dir / "mechanism_weak_signal_overview.csv"
    manifest_path = output_dir / "mechanism_weak_signal_manifest.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([overview]).to_csv(overview_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "goal_id": "1",
                "analysis_label": config["analysis_label"],
                "evidence_status": "mechanism_weak_signal_initial",
                "dataset_root": config["dataset_root"],
                "prediction_root": config["prediction_root"],
                "threshold_quantile": threshold_quantile,
                "detail": str(detail_path.relative_to(PROJECT_ROOT)),
                "summary": str(summary_path.relative_to(PROJECT_ROOT)),
                "overview": str(overview_path.relative_to(PROJECT_ROOT)),
                "limitations": [
                    "Mechanism-specific weak recall is post-hoc from existing predictions.",
                    "Full-slide sparse benchmark currently has no juxtacrine/contact positives.",
                    "Use as diagnostic evidence for Goal 1 failure modes, not as a new trained model.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{config['analysis_label']} summary: {summary_path}")
    return {"detail": detail, "summary": summary}


def _mechanism_rows(
    candidates: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    analysis_label: str,
    ablation: str,
    replicate: str,
    threshold_quantile: float,
) -> list[dict[str, object]]:
    merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    scores = _minmax(merged["score"].to_numpy(dtype=float))
    threshold = float(np.quantile(scores, threshold_quantile))
    predicted = scores >= threshold
    merged = merged.copy()
    merged["_score_norm"] = scores
    merged["_predicted_positive"] = predicted
    rows = []
    for mechanism, subset in merged.groupby("mechanism", sort=True):
        truth = subset["ground_truth"].astype(bool).to_numpy()
        weak = subset["weak_signal"].astype(bool).to_numpy()
        fp_high = subset["false_positive_high_expression"].astype(bool).to_numpy()
        pred = subset["_predicted_positive"].astype(bool).to_numpy()
        weak_pos = truth & weak
        positives = truth
        rows.append(
            {
                "analysis_label": analysis_label,
                "ablation": ablation,
                "replicate": replicate,
                "mechanism": mechanism,
                "threshold_quantile": threshold_quantile,
                "n_candidates": int(len(subset)),
                "n_positive": int(positives.sum()),
                "n_weak_positive": int(weak_pos.sum()),
                "n_false_positive_high_expression": int(fp_high.sum()),
                "n_predicted_positive": int(pred.sum()),
                "recall_at_threshold": _safe_divide(int((pred & positives).sum()), int(positives.sum())),
                "weak_signal_recall_at_threshold": _safe_divide(int((pred & weak_pos).sum()), int(weak_pos.sum())),
                "empirical_fdr_at_threshold": _safe_divide(int((pred & ~positives).sum()), int(pred.sum())),
                "false_positive_high_expression_rate_at_threshold": _safe_divide(
                    int((pred & fp_high).sum()),
                    int(fp_high.sum()),
                ),
                "auroc": _auroc(truth.astype(int), subset["_score_norm"].to_numpy(dtype=float)),
                "aupr": _aupr(truth.astype(int), subset["_score_norm"].to_numpy(dtype=float)),
            }
        )
    return rows


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_candidates",
        "n_positive",
        "n_weak_positive",
        "n_false_positive_high_expression",
        "n_predicted_positive",
        "recall_at_threshold",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "auroc",
        "aupr",
    ]
    grouped = detail.groupby(["analysis_label", "ablation", "mechanism", "threshold_quantile"], sort=False)[metric_columns].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _overview(summary: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    edge = summary[summary["ablation"] == "wavelet_mechanism_niche_edge_prior_ranking"]
    focal = summary[summary["ablation"] == "phase2_focal_weak_mining_edge_prior_wavelet"]
    best = summary.sort_values("weak_signal_recall_at_threshold_mean", ascending=False).iloc[0]
    mechanisms = sorted(summary["mechanism"].unique())
    edge_map = {row.mechanism: row.weak_signal_recall_at_threshold_mean for row in edge.itertuples(index=False)}
    focal_map = {row.mechanism: row.weak_signal_recall_at_threshold_mean for row in focal.itertuples(index=False)}
    return {
        "goal_id": "1",
        "analysis_label": config["analysis_label"],
        "evidence_status": "mechanism_weak_signal_initial",
        "threshold_quantile": float(config["threshold_quantile"]),
        "n_ablations": int(summary["ablation"].nunique()),
        "n_mechanisms": int(len(mechanisms)),
        "mechanisms": ";".join(mechanisms),
        "best_ablation": best["ablation"],
        "best_mechanism": best["mechanism"],
        "best_mechanism_weak_recall": float(best["weak_signal_recall_at_threshold_mean"]),
        "edge_prior_mechanism_weak_recall": ";".join(
            f"{mechanism}:{edge_map.get(mechanism, math.nan):.4f}" for mechanism in mechanisms
        ),
        "focal_weak_mining_mechanism_weak_recall": ";".join(
            f"{mechanism}:{focal_map.get(mechanism, math.nan):.4f}" for mechanism in mechanisms
        ),
        "interpretation": (
            "Post-hoc mechanism-specific weak-signal diagnostic from existing predictions; "
            "used to identify whether contact, short/medium, or long-range signals drive Goal 1 behavior."
        ),
    }


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _prediction_path(ablation_dir: Path, replicate_dir: Path) -> Path:
    paths = sorted(ablation_dir.glob(f"{replicate_dir.name}__*_predictions.csv"))
    return paths[0] if paths else ablation_dir / f"{replicate_dir.name}__spatialself_wavelet_predictions.csv"


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
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
        return math.nan
    ranks = _average_ranks(scores)
    sum_pos = float(ranks[positives].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.trapz(precision, recall))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
