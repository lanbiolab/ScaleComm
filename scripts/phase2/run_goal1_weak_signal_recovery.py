#!/usr/bin/env python
"""Run Phase 2 Goal 1 weak-signal recovery candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself_ablation import ABLATIONS, run_ablation


PHASE1_CONTROLLED_WEAK_RECALL_REFERENCE = 0.155
PHASE1_FULL_SLIDE_WEAK_RECALL_REFERENCE = 0.040

DEFAULT_ABLATIONS = (
    "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
    "wavelet_mechanism_niche_edge_prior_ranking",
    "phase2_weak_mining_wavelet",
    "phase2_weak_mining_edge_prior_wavelet",
    "phase2_focal_weak_mining_wavelet",
    "phase2_focal_weak_mining_edge_prior_wavelet",
    "phase2_lr_balanced_focal_wavelet",
    "gat_hard_expression_distance",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/phase1/exp01_controlled_benchmark/smoke_datasets",
        help="Benchmark dataset root containing replicate directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal1_weak_signal_recovery/smoke_controlled",
    )
    parser.add_argument("--ablation", action="append", dest="ablations")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument(
        "--success-reference",
        choices=("controlled", "full_slide"),
        default="controlled",
        help="Weak-recall reference used for phase2_goal1_candidate_pass.",
    )
    parser.add_argument(
        "--reference-controlled-weak-recall",
        type=float,
        default=PHASE1_CONTROLLED_WEAK_RECALL_REFERENCE,
    )
    parser.add_argument(
        "--reference-full-slide-weak-recall",
        type=float,
        default=PHASE1_FULL_SLIDE_WEAK_RECALL_REFERENCE,
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations = tuple(args.ablations) if args.ablations else DEFAULT_ABLATIONS
    unknown = [name for name in ablations if name not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {unknown}")

    metrics = run_ablation(
        PROJECT_ROOT / args.dataset_root,
        output_dir,
        ablations=ablations,
        epochs=args.epochs,
        k_nn=args.k_nn,
        max_train_samples=args.max_train_samples,
    )
    summary = _summarize_phase2_goal1(
        metrics,
        controlled_reference=args.reference_controlled_weak_recall,
        full_slide_reference=args.reference_full_slide_weak_recall,
        success_reference=args.success_reference,
    )
    metrics_path = output_dir / "weak_signal_recovery_metrics.csv"
    summary_path = output_dir / "weak_signal_recovery_summary.csv"
    status_path = output_dir / "weak_signal_recovery_status.csv"
    manifest_path = output_dir / "weak_signal_recovery_manifest.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(
        [
            {
                "ablation": name,
                "status": "completed" if name in set(metrics.get("ablation", [])) else "missing",
                "detail": json.dumps(ABLATIONS.get(name, {}), sort_keys=True),
            }
            for name in ablations
        ]
    ).to_csv(status_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": args.dataset_root,
                "output_dir": args.output_dir,
                "ablations": list(ablations),
                "epochs": args.epochs,
                "k_nn": args.k_nn,
                "max_train_samples": args.max_train_samples,
                "reference_controlled_weak_recall": args.reference_controlled_weak_recall,
                "reference_full_slide_weak_recall": args.reference_full_slide_weak_recall,
                "success_reference": args.success_reference,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote Phase 2 Goal 1 outputs to {output_dir}")


def _summarize_phase2_goal1(
    metrics: pd.DataFrame,
    *,
    controlled_reference: float,
    full_slide_reference: float,
    success_reference: str,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "auroc",
        "aupr",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
        "n_pseudo_positive",
        "n_pseudo_negative",
        "n_soft_weak_positive",
        "n_ranking_pairs",
    ]
    available = [column for column in metric_columns if column in metrics.columns]
    summary = metrics.groupby("ablation", sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["controlled_weak_recall_reference"] = float(controlled_reference)
    summary["full_slide_weak_recall_reference"] = float(full_slide_reference)
    summary["controlled_weak_recall_delta"] = (
        summary["weak_signal_recall_at_threshold_mean"] - float(controlled_reference)
    )
    summary["full_slide_weak_recall_delta_proxy"] = (
        summary["weak_signal_recall_at_threshold_mean"] - float(full_slide_reference)
    )
    phase1_default = summary.loc[
        summary["ablation"] == "wavelet_mechanism_niche_lr_prior_lrwise_ranking"
    ]
    if not phase1_default.empty:
        default_fdr = float(phase1_default["empirical_fdr_at_threshold_mean"].iloc[0])
        default_fp = float(phase1_default["false_positive_high_expression_rate_at_threshold_mean"].iloc[0])
    else:
        default_fdr = float("nan")
        default_fp = float("nan")
    summary["fdr_delta_vs_phase1_default"] = summary["empirical_fdr_at_threshold_mean"] - default_fdr
    summary["false_positive_delta_vs_phase1_default"] = (
        summary["false_positive_high_expression_rate_at_threshold_mean"] - default_fp
    )
    if success_reference == "full_slide":
        weak_reference = float(full_slide_reference)
    elif success_reference == "controlled":
        weak_reference = float(controlled_reference)
    else:
        raise ValueError(f"Unknown success_reference: {success_reference}")
    summary["success_weak_recall_reference"] = weak_reference
    summary["weak_recall_delta_vs_success_reference"] = (
        summary["weak_signal_recall_at_threshold_mean"] - weak_reference
    )
    summary["phase2_goal1_candidate_pass"] = (
        (summary["weak_signal_recall_at_threshold_mean"] > weak_reference)
        & (summary["fdr_delta_vs_phase1_default"] <= 0.05)
        & (summary["false_positive_delta_vs_phase1_default"] <= 0.02)
    )
    return summary


if __name__ == "__main__":
    main()
