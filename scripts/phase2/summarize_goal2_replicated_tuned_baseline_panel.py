#!/usr/bin/env python
"""Summarize completed replicated tuned baseline method batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PANEL = "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL)
    args = parser.parse_args()

    panel_dir = PROJECT_ROOT / args.panel_dir
    metrics = _collect_metrics(panel_dir / "method_batches")
    status = _collect_status(panel_dir)
    metrics_path = panel_dir / "goal2_replicated_tuned_metric_rows.csv"
    summary_path = panel_dir / "goal2_replicated_tuned_metric_summary.csv"
    tissue_summary_path = panel_dir / "goal2_replicated_tuned_tissue_metric_summary.csv"
    replicate_path = panel_dir / "goal2_replicated_tuned_replicate_summary.csv"
    claim_path = panel_dir / "goal2_replicated_tuned_claims.csv"
    manifest_path = panel_dir / "goal2_replicated_tuned_panel_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    _metric_summary(metrics).to_csv(summary_path, index=False)
    _tissue_metric_summary(metrics).to_csv(tissue_summary_path, index=False)
    _replicate_summary(metrics, status).to_csv(replicate_path, index=False)
    _claims(metrics, status).to_csv(claim_path, index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_replicated_tuned_baseline_panel_summary",
        "panel_dir": str(panel_dir.relative_to(PROJECT_ROOT)),
        "metric_rows": int(len(metrics)),
        "metric_methods": sorted(metrics["method"].astype(str).unique()) if not metrics.empty else [],
        "matched_metric_complete_replicates": int(_matched_metric_complete_replicates(metrics)),
        "expected_matched_replicates": 18,
        "evidence_status": "replicated_panel_complete" if _matched_metric_complete_replicates(metrics) >= 18 else "replicated_panel_in_progress",
        "outputs": {
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "tissue_summary": str(tissue_summary_path.relative_to(PROJECT_ROOT)),
            "replicate_summary": str(replicate_path.relative_to(PROJECT_ROOT)),
            "claims": str(claim_path.relative_to(PROJECT_ROOT)),
        },
        "claim_scope": "replicated tuned panel summary; conclusions must preserve metric-specific tradeoffs",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if metrics.empty:
        print("No completed metric rows found.")
    else:
        print(_replicate_summary(metrics, status).to_string(index=False))
        tissue_summary = _tissue_metric_summary(metrics)
        if not tissue_summary.empty:
            print(tissue_summary.to_string(index=False))
        print(_metric_summary(metrics).to_string(index=False))
    print(f"Wrote replicated tuned baseline summary to {panel_dir}")


def _collect_metrics(method_batches: Path) -> pd.DataFrame:
    rows = []
    if not method_batches.exists():
        return pd.DataFrame()
    for path in sorted(method_batches.glob("*/*/goal2_formal_tuned_small_metrics.csv")):
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if table.empty or "aupr" not in table.columns:
            continue
        table = table.copy()
        table["method_batch_dir"] = str(path.parent.relative_to(PROJECT_ROOT))
        table["dataset_key"] = _dataset_key_from_replicate(str(table["replicate"].iloc[0]) if "replicate" in table else path.parent.parent.name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_status(panel_dir: Path) -> pd.DataFrame:
    status_path = panel_dir / "goal2_replicated_tuned_job_status.csv"
    if status_path.exists():
        return pd.read_csv(status_path)
    return pd.DataFrame()


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metric_columns if column in metrics.columns]
    summary = metrics.groupby("method", sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _tissue_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "dataset_key" not in metrics.columns:
        return pd.DataFrame()
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metric_columns if column in metrics.columns]
    summary = metrics.groupby(["dataset_key", "method"], sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _replicate_summary(metrics: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    method_count = metrics.groupby("replicate")["method"].nunique().reset_index(name="metric_methods_completed")
    row_count = metrics.groupby("replicate").size().reset_index(name="metric_rows")
    table = method_count.merge(row_count, on="replicate", how="left")
    if not status.empty:
        status_counts = (
            status.groupby(["replicate_name", "method_type", "status"])
            .size()
            .reset_index(name="n")
            .pivot_table(index="replicate_name", columns=["method_type", "status"], values="n", fill_value=0, aggfunc="sum")
        )
        status_counts.columns = [f"{a}_{b}" for a, b in status_counts.columns]
        status_counts = status_counts.reset_index().rename(columns={"replicate_name": "replicate"})
        table = table.merge(status_counts, on="replicate", how="left")
    table["matched_metric_complete"] = table["metric_methods_completed"] >= 6
    return table.sort_values("replicate").reset_index(drop=True)


def _claims(metrics: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            [
                {
                    "claim_id": "replicated_tuned_panel_no_metric_results",
                    "claim_level": "progress_summary_no_result_claim",
                    "supported": True,
                    "summary": "No completed metric rows were found.",
                }
            ]
        )
    matched = _matched_metric_complete_replicates(metrics)
    panel_complete = matched >= 18
    rows = [
        {
            "claim_id": "replicated_tuned_panel_completed" if panel_complete else "replicated_tuned_panel_progress",
            "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "progress_summary_not_final_replicated_claim",
            "supported": True,
            "summary": f"Collected {len(metrics)} metric rows across {metrics['replicate'].nunique()} replicates; {matched} replicates have all six metric methods completed.",
        }
    ]
    if matched > 0:
        complete_reps = (
            metrics.groupby("replicate")["method"]
            .nunique()
            .loc[lambda series: series >= 6]
            .index
        )
        complete = metrics[metrics["replicate"].isin(complete_reps)].copy()
        complete["guardrailed_tradeoff_score"] = (
            complete["weak_signal_recall_at_threshold"].fillna(0.0)
            + 0.25 * complete["aupr"].fillna(0.0)
            - 0.25 * complete["empirical_fdr_at_threshold"].fillna(1.0)
            - complete["false_positive_high_expression_rate_at_threshold"].fillna(1.0)
        )
        method_mean = complete.groupby("method", as_index=False)[
            [
                "aupr",
                "top_k_precision",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
                "false_positive_high_expression_rate_at_threshold",
                "guardrailed_tradeoff_score",
            ]
        ].mean(numeric_only=True)
        tissue_winners = _tissue_winners(complete)
        best_fp_method = method_mean.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
        best_weak_method = method_mean.sort_values("weak_signal_recall_at_threshold", ascending=False).iloc[0]
        best_tradeoff_method = method_mean.sort_values("guardrailed_tradeoff_score", ascending=False).iloc[0]
        rows.extend(
            [
                {
                    "claim_id": "replicated_tuned_method_mean_lowest_fp",
                    "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "single_or_partial_replicate_snapshot",
                    "supported": True,
                    "summary": f"Lowest method-mean high-expression FP: {best_fp_method['method']} ({best_fp_method['false_positive_high_expression_rate_at_threshold']:.6g}).",
                },
                {
                    "claim_id": "replicated_tuned_method_mean_highest_weak_recall",
                    "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "single_or_partial_replicate_snapshot",
                    "supported": True,
                    "summary": f"Highest method-mean weak recall: {best_weak_method['method']} ({best_weak_method['weak_signal_recall_at_threshold']:.6g}).",
                },
                {
                    "claim_id": "replicated_tuned_method_mean_tradeoff",
                    "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "single_or_partial_replicate_snapshot",
                    "supported": True,
                    "summary": f"Best method-mean guardrailed weak/FP tradeoff: {best_tradeoff_method['method']} ({best_tradeoff_method['guardrailed_tradeoff_score']:.6g}).",
                },
                {
                    "claim_id": "replicated_tuned_tissue_winners",
                    "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "single_or_partial_replicate_snapshot",
                    "supported": True,
                    "summary": "; ".join(tissue_winners),
                },
            ]
        )
    rows.append(
        {
            "claim_id": "replicated_tuned_panel_tradeoff_not_global_dominance",
            "claim_level": "replicated_panel_complete_tradeoff_claim" if panel_complete else "progress_summary_not_final_replicated_claim",
            "supported": True,
            "summary": "Do not claim global dominance: the completed replicated panel supports metric- and tissue-specific tradeoff interpretation, especially weak-signal/false-positive balance.",
        }
    )
    return pd.DataFrame(rows)


def _matched_metric_complete_replicates(metrics: pd.DataFrame) -> int:
    if metrics.empty or "replicate" not in metrics or "method" not in metrics:
        return 0
    return int((metrics.groupby("replicate")["method"].nunique() >= 6).sum())


def _tissue_winners(metrics: pd.DataFrame) -> list[str]:
    if "dataset_key" not in metrics.columns:
        return []
    rows: list[str] = []
    for dataset_key, tissue in metrics.groupby("dataset_key", sort=True):
        mean = tissue.groupby("method", as_index=False)[
            [
                "aupr",
                "top_k_precision",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
                "false_positive_high_expression_rate_at_threshold",
                "guardrailed_tradeoff_score",
            ]
        ].mean(numeric_only=True)
        best_aupr = mean.sort_values("aupr", ascending=False).iloc[0]
        best_tradeoff = mean.sort_values("guardrailed_tradeoff_score", ascending=False).iloc[0]
        lowest_fp = mean.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
        rows.append(
            f"{dataset_key}: best AUPR={best_aupr['method']}, best tradeoff={best_tradeoff['method']}, lowest FP={lowest_fp['method']}"
        )
    return rows


def _dataset_key_from_replicate(replicate: str) -> str:
    if "lymph_node" in replicate:
        return "lymph_node"
    if "tonsil" in replicate:
        return "tonsil"
    if "melanoma" in replicate:
        return "melanoma"
    return "unknown"


if __name__ == "__main__":
    main()
