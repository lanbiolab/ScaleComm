#!/usr/bin/env python
"""Merge revised Goal 2 full-baseline panels across datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUTS = (
    "outputs/phase2_revised/goal2_full_baseline_panel",
    "outputs/phase2_revised/goal2_full_baseline_panel_lymph_melanoma",
    "outputs/phase2_revised/goal2_full_baseline_panel_melanoma",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", dest="input_dirs", help="Input panel directory. Repeatable.")
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal2_full_baseline_panel_all")
    args = parser.parse_args()

    input_dirs = [PROJECT_ROOT / path for path in (args.input_dirs or DEFAULT_INPUTS)]
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_many(input_dirs, "goal2_full_baseline_metrics.csv")
    status = _read_many(input_dirs, "goal2_full_baseline_status.csv")
    dataset_runs = _read_many(input_dirs, "goal2_full_baseline_dataset_runs.csv")
    if metrics.empty:
        raise ValueError(f"No metrics rows found in {input_dirs}")
    metrics.to_csv(output_dir / "goal2_full_baseline_metrics.csv", index=False)
    status.to_csv(output_dir / "goal2_full_baseline_status.csv", index=False)
    dataset_runs.to_csv(output_dir / "goal2_full_baseline_dataset_runs.csv", index=False)
    _summary(metrics).to_csv(output_dir / "goal2_full_baseline_summary.csv", index=False)
    _claims(metrics, status).to_csv(output_dir / "goal2_full_baseline_claims.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_full_baseline_panel_all",
        "evidence_status": "three_real_background_adapter_aware_smoke_not_final_tuned_baseline",
        "input_dirs": [str(path.relative_to(PROJECT_ROOT)) for path in input_dirs],
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": sorted(metrics["dataset"].astype(str).unique()),
        "methods_with_metrics": sorted(metrics["method"].astype(str).unique()),
        "methods_with_status": sorted(status["method"].astype(str).unique()) if not status.empty else [],
        "outputs": {
            "metrics": str((output_dir / "goal2_full_baseline_metrics.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_full_baseline_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_full_baseline_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_full_baseline_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "CellChat, SpaTalk, and CellNEST rows are adapter-light smoke evidence.",
            "DeepTalk remains projection/model-required until LR-specific predict_ccc outputs are available.",
            "NICHES remains dependency-blocked in the local R environment.",
        ],
    }
    (output_dir / "goal2_full_baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cols = [
        "dataset",
        "method",
        "aupr",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
    ]
    print(metrics[[c for c in cols if c in metrics.columns]].to_string(index=False))
    print(f"Wrote merged full-baseline panel to {output_dir}")


def _read_many(input_dirs: list[Path], filename: str) -> pd.DataFrame:
    parts = []
    for directory in input_dirs:
        path = directory / filename
        if path.exists():
            part = pd.read_csv(path)
            part["source_panel_dir"] = str(directory.relative_to(PROJECT_ROOT))
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in columns if column in metrics]
    summary = metrics.groupby(["dataset", "method"], sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _claims(metrics: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in metrics.groupby("dataset", sort=False):
        runnable = group.dropna(subset=["weak_signal_recall_at_threshold", "false_positive_high_expression_rate_at_threshold"], how="all")
        if runnable.empty:
            continue
        scored = runnable.copy()
        scored["unconstrained_tradeoff_score"] = (
            scored["weak_signal_recall_at_threshold"].fillna(0.0)
            + 0.25 * scored["aupr"].fillna(0.0)
            - 0.5 * scored["empirical_fdr_at_threshold"].fillna(0.0)
            - 0.5 * scored["false_positive_high_expression_rate_at_threshold"].fillna(0.0)
        )
        constrained = scored[
            (scored["false_positive_high_expression_rate_at_threshold"].fillna(1.0) <= 0.05)
            & (scored["empirical_fdr_at_threshold"].fillna(1.0) <= 0.5)
        ].copy()
        if constrained.empty:
            constrained = scored.copy()
            constrained["constrained_tradeoff_status"] = "no_method_met_fp_fdr_guardrail"
        else:
            constrained["constrained_tradeoff_status"] = "fp_fdr_guardrail_met"
        constrained["constrained_tradeoff_score"] = (
            constrained["weak_signal_recall_at_threshold"].fillna(0.0)
            + 0.25 * constrained["aupr"].fillna(0.0)
            - 0.25 * constrained["empirical_fdr_at_threshold"].fillna(0.0)
            - constrained["false_positive_high_expression_rate_at_threshold"].fillna(1.0)
        )
        best_tradeoff = constrained.sort_values("constrained_tradeoff_score", ascending=False).iloc[0]
        best_unconstrained = scored.sort_values("unconstrained_tradeoff_score", ascending=False).iloc[0]
        best_fp = scored.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
        best_weak = scored.sort_values("weak_signal_recall_at_threshold", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "three_background_adapter_aware_smoke_not_final_tuned_baseline",
                "best_tradeoff_method": best_tradeoff["method"],
                "best_tradeoff_score": best_tradeoff["constrained_tradeoff_score"],
                "best_tradeoff_status": best_tradeoff["constrained_tradeoff_status"],
                "best_unconstrained_tradeoff_method": best_unconstrained["method"],
                "best_unconstrained_tradeoff_score": best_unconstrained["unconstrained_tradeoff_score"],
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["false_positive_high_expression_rate_at_threshold"],
                "highest_weak_recall_method": best_weak["method"],
                "highest_weak_recall": best_weak["weak_signal_recall_at_threshold"],
                "metric_methods": int(scored["method"].nunique()),
                "status_methods": int(status.loc[status["dataset"] == dataset, "method"].nunique()) if not status.empty else 0,
                "spatialself_best_tradeoff": str(best_tradeoff["method"]).startswith("spatialself"),
                "spatialself_lowest_fp": str(best_fp["method"]).startswith("spatialself"),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
