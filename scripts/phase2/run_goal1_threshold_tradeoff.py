#!/usr/bin/env python
"""Evaluate weak-recall/FDR operating points for Phase 2 Goal 1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_ground_truth_predictions


DEFAULT_QUANTILES = (0.99, 0.975, 0.95, 0.945, 0.94, 0.93, 0.92, 0.90, 0.85, 0.80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prediction", action="append", dest="predictions", required=True)
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--output-dir", default="outputs/phase2/goal1_weak_signal_recovery/threshold_tradeoff")
    parser.add_argument("--quantile", action="append", type=float, dest="quantiles")
    parser.add_argument("--weak-recall-reference", type=float, default=0.040)
    parser.add_argument("--fdr-increase-max", type=float, default=0.02)
    parser.add_argument("--fp-rate-max", type=float, default=0.005)
    args = parser.parse_args()

    labels = args.labels or [Path(path).parent.name for path in args.predictions]
    if len(labels) != len(args.predictions):
        raise SystemExit("--label count must match --prediction count")
    quantiles = tuple(args.quantiles) if args.quantiles else DEFAULT_QUANTILES
    candidates = pd.read_csv(args.candidates)
    rows = []
    for label, prediction_path in zip(labels, args.predictions):
        predictions = pd.read_csv(prediction_path)
        baseline_95 = evaluate_ground_truth_predictions(candidates, predictions, threshold_quantile=0.95)
        baseline_fdr = float(baseline_95["empirical_fdr_at_threshold"])
        for quantile in quantiles:
            metrics = evaluate_ground_truth_predictions(candidates, predictions, threshold_quantile=float(quantile))
            rows.append(
                {
                    "label": label,
                    "prediction_path": prediction_path,
                    "threshold_quantile": float(quantile),
                    "weak_signal_recall_at_threshold": metrics["weak_signal_recall_at_threshold"],
                    "recall_at_threshold": metrics["recall_at_threshold"],
                    "empirical_fdr_at_threshold": metrics["empirical_fdr_at_threshold"],
                    "false_positive_high_expression_rate_at_threshold": metrics[
                        "false_positive_high_expression_rate_at_threshold"
                    ],
                    "auroc": metrics["auroc"],
                    "aupr": metrics["aupr"],
                    "threshold_score": metrics["threshold_score"],
                    "baseline_95_fdr": baseline_fdr,
                    "fdr_delta_vs_95": metrics["empirical_fdr_at_threshold"] - baseline_fdr,
                    "weak_recall_reference": float(args.weak_recall_reference),
                    "candidate_pass": (
                        metrics["weak_signal_recall_at_threshold"] > float(args.weak_recall_reference)
                        and metrics["empirical_fdr_at_threshold"] - baseline_fdr <= float(args.fdr_increase_max)
                        and metrics["false_positive_high_expression_rate_at_threshold"] <= float(args.fp_rate_max)
                    ),
                }
            )

    table = pd.DataFrame(rows)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "threshold_tradeoff_metrics.csv"
    summary_path = output_dir / "threshold_tradeoff_summary.csv"
    manifest_path = output_dir / "threshold_tradeoff_manifest.json"
    table.to_csv(table_path, index=False)
    _summarize(table).to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "candidates": args.candidates,
                "predictions": args.predictions,
                "labels": labels,
                "quantiles": list(quantiles),
                "weak_recall_reference": args.weak_recall_reference,
                "fdr_increase_max": args.fdr_increase_max,
                "fp_rate_max": args.fp_rate_max,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(_summarize(table).to_string(index=False))
    print(f"Wrote threshold tradeoff outputs to {output_dir}")


def _summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in table.groupby("label", sort=False):
        passing = group[group["candidate_pass"].astype(bool)]
        if passing.empty:
            best = group.sort_values(
                ["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold"],
                ascending=[False, True],
            ).iloc[0]
            status = "no_passing_operating_point"
        else:
            best = passing.sort_values(
                ["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold"],
                ascending=[False, True],
            ).iloc[0]
            status = "passing_operating_point"
        rows.append(
            {
                "label": label,
                "status": status,
                "best_quantile": best["threshold_quantile"],
                "best_weak_signal_recall": best["weak_signal_recall_at_threshold"],
                "best_recall": best["recall_at_threshold"],
                "best_fdr": best["empirical_fdr_at_threshold"],
                "best_fdr_delta_vs_95": best["fdr_delta_vs_95"],
                "best_false_positive_high_expression_rate": best[
                    "false_positive_high_expression_rate_at_threshold"
                ],
                "auroc": best["auroc"],
                "aupr": best["aupr"],
                "n_passing_quantiles": int(len(passing)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
