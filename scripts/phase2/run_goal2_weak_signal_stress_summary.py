#!/usr/bin/env python
"""Build formal weak-signal stress summaries from matched benchmark predictions."""

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


PREDICTION_ROOTS = {
    "spatialself_wavelet_edge_prior": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "commot": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "cellchat": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "spatialself_wavelet": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "spatalk": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "cellnest": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
}

DATASETS = {
    "controlled_v2_formal": {
        "dataset_root": "outputs/phase2_revised/goal3_simulator_v2/formal_datasets",
        "prediction_dataset": "controlled_v2_formal",
    },
    "real_background_v2_formal": {
        "dataset_root": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets",
        "prediction_dataset": "real_background_v2_formal",
    },
}

METHODS = (
    "spatialself_wavelet_edge_prior",
    "spatialself_wavelet",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal2_weak_signal_stress_formal")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    status_rows: list[dict] = []
    for dataset, dataset_cfg in DATASETS.items():
        dataset_root = PROJECT_ROOT / dataset_cfg["dataset_root"]
        for replicate_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            candidates_path = replicate_dir / "candidate_interactions.csv"
            if not candidates_path.exists():
                continue
            candidates = pd.read_csv(candidates_path)
            for method in METHODS:
                prediction_path = _prediction_path(
                    PROJECT_ROOT / PREDICTION_ROOTS[method],
                    dataset_cfg["prediction_dataset"],
                    method,
                    replicate_dir.name,
                )
                if not prediction_path.exists():
                    status_rows.append(
                        {
                            "dataset": dataset,
                            "replicate": replicate_dir.name,
                            "method": method,
                            "status": "missing_prediction",
                            "detail": str(prediction_path.relative_to(PROJECT_ROOT)),
                        }
                    )
                    continue
                try:
                    prediction = pd.read_csv(prediction_path)
                    annotated = _merge(candidates, prediction)
                    metric_rows.extend(
                        _metric_rows(
                            annotated,
                            dataset=dataset,
                            replicate=replicate_dir.name,
                            method=method,
                            prediction_path=prediction_path,
                            threshold_quantile=args.threshold_quantile,
                            top_k=args.top_k,
                        )
                    )
                    status_rows.append(
                        {
                            "dataset": dataset,
                            "replicate": replicate_dir.name,
                            "method": method,
                            "status": "completed",
                            "detail": "",
                            "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    status_rows.append(
                        {
                            "dataset": dataset,
                            "replicate": replicate_dir.name,
                            "method": method,
                            "status": "failed",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    status = pd.DataFrame(status_rows)
    summary = _summary(metrics)
    claims = _claims(summary, status)

    metrics.to_csv(output_dir / "goal2_weak_signal_stress_metrics.csv", index=False)
    summary.to_csv(output_dir / "goal2_weak_signal_stress_summary.csv", index=False)
    claims.to_csv(output_dir / "goal2_weak_signal_stress_claims.csv", index=False)
    status.to_csv(output_dir / "goal2_weak_signal_stress_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_weak_signal_stress_formal",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": sorted(DATASETS),
        "methods": list(METHODS),
        "threshold_quantile": float(args.threshold_quantile),
        "top_k": int(args.top_k),
        "claim_level": "formal_weak_signal_stress_not_final_tuned_baseline",
        "outputs": {
            "metrics": str((output_dir / "goal2_weak_signal_stress_metrics.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_weak_signal_stress_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_weak_signal_stress_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_weak_signal_stress_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This is a derived stress summary from formal adapter-aware predictions, not a newly tuned baseline run.",
            "Weak-signal superiority is interpreted through FP/FDR guardrails, not maximum weak recall alone.",
            "SpaTalk and CellNEST remain adapter-light where their formal predictions are projected to candidate edges.",
        ],
    }
    (output_dir / "goal2_weak_signal_stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not claims.empty:
        print(claims.to_string(index=False))
    print(f"Wrote formal weak-signal stress outputs to {output_dir}")


def _prediction_path(prediction_root: Path, dataset: str, method: str, replicate_name: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        filename = f"{replicate_name}__spatialself_wavelet_predictions.csv"
    else:
        filename = f"{replicate_name}__{method}_predictions.csv"
    return prediction_root / dataset / method / filename


def _merge(candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "score"}
    missing = required - set(prediction.columns)
    if missing:
        raise ValueError(f"Prediction missing columns: {sorted(missing)}")
    merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    if len(merged) != len(candidates):
        raise ValueError(f"Prediction/candidate row mismatch: {len(merged)} != {len(candidates)}")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _metric_rows(
    annotated: pd.DataFrame,
    *,
    dataset: str,
    replicate: str,
    method: str,
    prediction_path: Path,
    threshold_quantile: float,
    top_k: int,
) -> list[dict]:
    rows = []
    rows.append(
        _one_metric(
            annotated,
            target_mechanism=None,
            dataset=dataset,
            replicate=replicate,
            method=method,
            mechanism="all_mechanisms",
            prediction_path=prediction_path,
            threshold_quantile=threshold_quantile,
            top_k=top_k,
        )
    )
    for mechanism in sorted(str(value) for value in annotated["mechanism"].dropna().unique()):
        if mechanism == "false_positive_high_expression":
            continue
        rows.append(
            _one_metric(
                annotated,
                target_mechanism=mechanism,
                dataset=dataset,
                replicate=replicate,
                method=method,
                mechanism=str(mechanism),
                prediction_path=prediction_path,
                threshold_quantile=threshold_quantile,
                top_k=top_k,
            )
        )
    return rows


def _one_metric(
    table: pd.DataFrame,
    *,
    target_mechanism: str | None,
    dataset: str,
    replicate: str,
    method: str,
    mechanism: str,
    prediction_path: Path,
    threshold_quantile: float,
    top_k: int,
) -> dict:
    truth = table["ground_truth"].astype(int).to_numpy()
    ground_truth = table["ground_truth"].astype(bool).to_numpy()
    if target_mechanism is None:
        target_positive = ground_truth
        weak_truth = (table["ground_truth"].astype(bool) & table["weak_signal"].astype(bool)).to_numpy()
    else:
        target_mask = table["mechanism"].astype(str).to_numpy() == target_mechanism
        target_positive = ground_truth & target_mask
        weak_truth = target_positive & table["weak_signal"].astype(bool).to_numpy()
    score = table["score"].to_numpy(dtype=float)
    threshold = float(np.quantile(score, threshold_quantile))
    called = score >= threshold
    top_mask = np.zeros(len(table), dtype=bool)
    if len(table):
        top_idx = np.argsort(-score, kind="mergesort")[: min(top_k, len(table))]
        top_mask[top_idx] = True
    fp_high = table["false_positive_high_expression"].astype(bool).to_numpy()
    nonweak_positive = target_positive & (~weak_truth)
    weak_or_negative = weak_truth | (truth == 0)
    return {
        "dataset": dataset,
        "replicate": replicate,
        "method": method,
        "mechanism": mechanism,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "n_candidates": int(len(table)),
        "n_positives": int(target_positive.sum()),
        "n_weak_positives": int(weak_truth.sum()),
        "n_high_expression_false_positive_candidates": int(fp_high.sum()),
        "threshold_quantile": float(threshold_quantile),
        "threshold_score": threshold,
        "called_edges": int(called.sum()),
        "weak_recall_at_threshold": _safe_divide(int((called & weak_truth).sum()), int(weak_truth.sum())),
        "all_positive_recall_at_threshold": _safe_divide(int((called & target_positive).sum()), int(target_positive.sum())),
        "empirical_fdr_at_threshold": _safe_divide(int((called & (truth == 0)).sum()), int(called.sum())),
        "high_expression_false_positive_rate_at_threshold": _safe_divide(int((called & fp_high).sum()), int(fp_high.sum())),
        "top_k": int(top_k),
        "top_k_weak_precision": _safe_divide(int((top_mask & weak_truth).sum()), int(top_mask.sum())),
        "top_k_all_positive_precision": _safe_divide(int((top_mask & target_positive).sum()), int(top_mask.sum())),
        "top_k_high_expression_false_positive_rate": _safe_divide(int((top_mask & fp_high).sum()), int(top_mask.sum())),
        "weak_only_auroc": _binary_auroc(weak_truth[weak_or_negative].astype(int), score[weak_or_negative]),
        "weak_only_aupr": _binary_aupr(weak_truth[weak_or_negative].astype(int), score[weak_or_negative]),
        "nonweak_positive_leakage_at_threshold": _safe_divide(int((called & nonweak_positive).sum()), int(nonweak_positive.sum())),
    }


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "weak_recall_at_threshold",
        "all_positive_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "high_expression_false_positive_rate_at_threshold",
        "top_k_weak_precision",
        "top_k_all_positive_precision",
        "top_k_high_expression_false_positive_rate",
        "weak_only_auroc",
        "weak_only_aupr",
        "nonweak_positive_leakage_at_threshold",
    ]
    summary = metrics.groupby(["dataset", "mechanism", "method"], sort=False)[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _claims(summary: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (dataset, mechanism), group in summary.groupby(["dataset", "mechanism"], sort=False):
        scored = group.copy()
        scored["guardrail_pass"] = (
            scored["high_expression_false_positive_rate_at_threshold_mean"].fillna(1.0).le(0.05)
            & scored["empirical_fdr_at_threshold_mean"].fillna(1.0).le(0.65)
        )
        constrained = scored[scored["guardrail_pass"]].copy()
        if constrained.empty:
            constrained = scored.copy()
            status_label = "no_method_met_fp_fdr_guardrail"
        else:
            status_label = "fp_fdr_guardrail_met"
        constrained["tradeoff_score"] = (
            constrained["weak_recall_at_threshold_mean"].fillna(0.0)
            + 0.5 * constrained["weak_only_aupr_mean"].fillna(0.0)
            + 0.25 * constrained["top_k_weak_precision_mean"].fillna(0.0)
            - 0.5 * constrained["empirical_fdr_at_threshold_mean"].fillna(0.0)
            - constrained["high_expression_false_positive_rate_at_threshold_mean"].fillna(1.0)
        )
        scored["unconstrained_tradeoff_score"] = (
            scored["weak_recall_at_threshold_mean"].fillna(0.0)
            + 0.5 * scored["weak_only_aupr_mean"].fillna(0.0)
            - 0.5 * scored["empirical_fdr_at_threshold_mean"].fillna(0.0)
            - 0.5 * scored["high_expression_false_positive_rate_at_threshold_mean"].fillna(0.0)
        )
        best_tradeoff = constrained.sort_values("tradeoff_score", ascending=False).iloc[0]
        best_weak = scored.sort_values("weak_recall_at_threshold_mean", ascending=False).iloc[0]
        best_fp = scored.sort_values("high_expression_false_positive_rate_at_threshold_mean", ascending=True).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "mechanism": mechanism,
                "claim_level": "formal_weak_signal_stress_not_final_tuned_baseline",
                "best_guardrailed_tradeoff_method": best_tradeoff["method"],
                "best_guardrailed_tradeoff_score": best_tradeoff["tradeoff_score"],
                "best_guardrailed_tradeoff_status": status_label,
                "highest_weak_recall_method": best_weak["method"],
                "highest_weak_recall": best_weak["weak_recall_at_threshold_mean"],
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["high_expression_false_positive_rate_at_threshold_mean"],
                "spatialself_best_guardrailed_tradeoff": str(best_tradeoff["method"]).startswith("spatialself"),
                "spatialself_highest_weak_recall": str(best_weak["method"]).startswith("spatialself"),
                "spatialself_lowest_high_expression_fp": str(best_fp["method"]).startswith("spatialself"),
                "metric_methods": int(scored["method"].nunique()),
                "completed_status_rows": int(
                    ((status["dataset"] == dataset) & (status["status"].astype(str) == "completed")).sum()
                )
                if not status.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return float((rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives))


def _binary_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_truth = y_true[order]
    tp = np.cumsum(sorted_truth)
    precision = tp / np.arange(1, len(sorted_truth) + 1)
    recall = tp / positives
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapz(precision, recall))


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
