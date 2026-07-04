#!/usr/bin/env python
"""Analyze guardrail sensitivity for the formal tuned-small baseline panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_METRICS = "outputs/phase2_revised/goal2_formal_tuned_small_panel/goal2_formal_tuned_small_metrics.csv"
DEFAULT_OUTPUT_DIR = "outputs/phase2_revised/goal2_tuned_guardrail_sensitivity"
FP_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10)
FDR_THRESHOLDS = (0.40, 0.50, 0.60, 0.65, 0.75, 0.85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics_path = PROJECT_ROOT / args.metrics
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    metric_methods = metrics.dropna(subset=["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold"], how="all")
    sensitivity = _sensitivity(metric_methods)
    summary = _summary(sensitivity)
    claims = _claims(summary)
    status = pd.DataFrame(
        [
            {
                "analysis": "goal2_tuned_guardrail_sensitivity",
                "status": "completed",
                "metrics_source": str(metrics_path.relative_to(PROJECT_ROOT)),
                "fp_thresholds": ";".join(str(value) for value in FP_THRESHOLDS),
                "fdr_thresholds": ";".join(str(value) for value in FDR_THRESHOLDS),
                "claim_level": "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline",
            }
        ]
    )

    sensitivity.to_csv(output_dir / "goal2_tuned_guardrail_sensitivity.csv", index=False)
    summary.to_csv(output_dir / "goal2_tuned_guardrail_sensitivity_summary.csv", index=False)
    claims.to_csv(output_dir / "goal2_tuned_guardrail_sensitivity_claims.csv", index=False)
    status.to_csv(output_dir / "goal2_tuned_guardrail_sensitivity_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_tuned_guardrail_sensitivity",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "metrics_source": str(metrics_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_status": "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline",
        "fp_thresholds": FP_THRESHOLDS,
        "fdr_thresholds": FDR_THRESHOLDS,
        "outputs": {
            "sensitivity": str((output_dir / "goal2_tuned_guardrail_sensitivity.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_tuned_guardrail_sensitivity_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_tuned_guardrail_sensitivity_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_tuned_guardrail_sensitivity_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This is a sensitivity analysis of one tuned-small real-background panel, not a replicated final baseline.",
            "Winner changes are interpreted as guardrail dependence, not biological superiority.",
            "SpaTalk cell-type/LR scores remain projected to candidate edges.",
        ],
    }
    (output_dir / "goal2_tuned_guardrail_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote tuned guardrail sensitivity to {output_dir}")


def _sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in _scope_groups(metrics):
        scored = _mean_by_method(group)
        for max_fp in FP_THRESHOLDS:
            for max_fdr in FDR_THRESHOLDS:
                passing = scored[
                    (scored["false_positive_high_expression_rate_at_threshold"].fillna(1.0) <= max_fp)
                    & (scored["empirical_fdr_at_threshold"].fillna(1.0) <= max_fdr)
                ].copy()
                if passing.empty:
                    passing = scored.copy()
                    status = "no_method_met_guardrail"
                else:
                    status = "guardrail_met"
                passing["guardrail_score"] = (
                    passing["weak_signal_recall_at_threshold"].fillna(0.0)
                    + 0.25 * passing["aupr"].fillna(0.0)
                    - 0.25 * passing["empirical_fdr_at_threshold"].fillna(1.0)
                    - passing["false_positive_high_expression_rate_at_threshold"].fillna(1.0)
                )
                best = passing.sort_values("guardrail_score", ascending=False).iloc[0]
                rows.append(
                    {
                        "scope": scope,
                        "max_high_expression_fp_rate": max_fp,
                        "max_empirical_fdr": max_fdr,
                        "guardrail_status": status,
                        "winner_method": best["method"],
                        "winner_score": best["guardrail_score"],
                        "winner_weak_recall": best["weak_signal_recall_at_threshold"],
                        "winner_aupr": best["aupr"],
                        "winner_empirical_fdr": best["empirical_fdr_at_threshold"],
                        "winner_high_expression_fp_rate": best["false_positive_high_expression_rate_at_threshold"],
                        "spatialself_winner": str(best["method"]).startswith("spatialself"),
                    }
                )
    return pd.DataFrame(rows)


def _scope_groups(metrics: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("aggregate_real_background_v2_formal", metrics)]
    groups.extend((str(replicate), group.copy()) for replicate, group in metrics.groupby("replicate", sort=False))
    return groups


def _mean_by_method(group: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in columns if column in group]
    return group.groupby("method", as_index=False)[available].mean(numeric_only=True)


def _summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in sensitivity.groupby("scope", sort=False):
        met = group[group["guardrail_status"] == "guardrail_met"]
        table = met if not met.empty else group
        winner_counts = table["winner_method"].value_counts().to_dict()
        rows.append(
            {
                "scope": scope,
                "n_guardrails_total": int(len(group)),
                "n_guardrails_met_by_at_least_one_method": int(len(met)),
                "n_spatialself_wins": int(table["spatialself_winner"].astype(bool).sum()),
                "spatialself_win_fraction": float(table["spatialself_winner"].astype(bool).mean()) if len(table) else 0.0,
                "dominant_winner_method": max(winner_counts, key=winner_counts.get) if winner_counts else "",
                "winner_counts_json": json.dumps(winner_counts, sort_keys=True),
                "claim_level": "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline",
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    aggregate = summary[summary["scope"] == "aggregate_real_background_v2_formal"]
    tissue = summary[summary["scope"] != "aggregate_real_background_v2_formal"]
    aggregate_fraction = float(aggregate["spatialself_win_fraction"].iloc[0]) if len(aggregate) else 0.0
    tissue_fraction = float(tissue["spatialself_win_fraction"].mean()) if len(tissue) else 0.0
    rows = [
        {
            "claim_id": "tuned_guardrail_sensitivity_aggregate",
            "claim_level": "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline",
            "supported": bool(aggregate_fraction >= 0.50),
            "summary": "SpatialSelf wins at least half of tested aggregate FP/FDR guardrails."
            if aggregate_fraction >= 0.50
            else "SpatialSelf does not dominate aggregate FP/FDR guardrail sensitivity.",
            "spatialself_win_fraction": aggregate_fraction,
        },
        {
            "claim_id": "tuned_guardrail_sensitivity_tissue_level",
            "claim_level": "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline",
            "supported": bool(tissue_fraction >= 0.50),
            "summary": "SpatialSelf wins at least half of tested tissue-level FP/FDR guardrails on average."
            if tissue_fraction >= 0.50
            else "Tissue-level tuned-small winner is guardrail-dependent and does not support SpatialSelf tissue-level dominance.",
            "spatialself_win_fraction": tissue_fraction,
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
