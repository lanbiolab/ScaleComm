#!/usr/bin/env python
"""Run focused SpatialSelf ablations on benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


ABLATIONS = {
    "gat_hard_expression_distance": {
        "model_name": "spatialself_gat",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
    },
    "wavelet_hard_expression_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
    },
    "wavelet_adaptive_hard_expression_distance": {
        "model_name": "spatialself_wavelet_adaptive",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
    },
    "wavelet_no_lr_embedding": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_lr_embedding": False,
    },
    "wavelet_fixed_random_lr_embedding": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "lr_encoding_mode": "fixed_random",
    },
    "wavelet_one_hot_lr_identity": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "lr_encoding_mode": "one_hot",
    },
    "wavelet_no_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_distance": False,
    },
    "wavelet_no_spatial_graph": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_spatial_graph": False,
    },
    "wavelet_no_expression_features": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_expression_features": False,
    },
    "wavelet_expression_distance_no_lr_identity": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_lr_embedding": False,
    },
    "wavelet_expression_only": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_only",
        "soft_labels": False,
    },
    "wavelet_distance_only": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "distance_only",
        "soft_labels": False,
    },
    "wavelet_expression_plus_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "expression_plus_distance",
        "soft_labels": False,
    },
    "wavelet_soft_expression_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "soft_pseudo_label",
        "soft_labels": True,
    },
    "gcn_hard_expression_distance": {
        "model_name": "spatialself_gcn",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
    },
    "mlp_hard_expression_distance": {
        "model_name": "spatialself_mlp",
        "pseudo_label_strategy": "expression_distance",
        "soft_labels": False,
        "use_spatial_graph": False,
    },
    "wavelet_lr_balanced_expression_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_balanced_expression_distance",
        "soft_labels": False,
    },
    "wavelet_lr_balanced_soft_expression_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_balanced_soft_pseudo_label",
        "soft_labels": True,
    },
    "wavelet_lr_capped_expression_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_capped_expression_distance",
        "soft_labels": False,
    },
    "wavelet_lr_capped_expression_plus_distance": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_capped_expression_plus_distance",
        "soft_labels": False,
    },
    "wavelet_lr_capped_wide": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_capped_expression_distance",
        "soft_labels": False,
        "lr_cap_positive": 250,
        "lr_cap_negative": 500,
    },
    "wavelet_lr_prior_hard_negative": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_prior_lr_capped_expression_distance",
        "soft_labels": False,
    },
    "wavelet_mechanism_niche_lr_capped": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
    },
    "wavelet_mechanism_niche_lr_prior": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
    },
    "wavelet_mechanism_niche_lr_prior_ranking": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
        "ranking_loss_weight": 0.25,
        "ranking_margin": 0.75,
    },
    "wavelet_mechanism_niche_lr_prior_lrwise_ranking": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
        "ranking_loss_weight": 0.35,
        "ranking_margin": 0.5,
    },
    "wavelet_mechanism_niche_edge_prior": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
        "use_edge_prior_features": True,
    },
    "wavelet_mechanism_niche_edge_prior_ranking": {
        "model_name": "spatialself_wavelet",
        "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
        "soft_labels": False,
        "ranking_loss_weight": 0.35,
        "ranking_margin": 0.5,
        "use_edge_prior_features": True,
    },
}


def run_ablation(
    dataset_root: Path,
    output_dir: Path,
    *,
    ablations: tuple[str, ...],
    epochs: int,
    k_nn: int,
    max_train_samples: int,
) -> pd.DataFrame:
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")

    rows = []
    status = []
    for ablation_name in ablations:
        if ablation_name not in ABLATIONS:
            status.append(
                {
                    "ablation": ablation_name,
                    "status": "unknown_ablation",
                    "detail": "not registered",
                }
            )
            continue
        params = ABLATIONS[ablation_name]
        ablation_dir = output_dir / ablation_name
        for replicate_dir in replicate_dirs:
            run_params = {
                "epochs": epochs,
                "k_nn": k_nn,
                "max_train_samples": max_train_samples,
                **params,
            }
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                ablation_dir,
                **run_params,
            )
            metrics["ablation"] = ablation_name
            rows.append(metrics)
        status.append(
            {
                "ablation": ablation_name,
                "status": "completed",
                "detail": json.dumps(params, sort_keys=True),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(output_dir / "spatialself_ablation_metrics.csv", index=False)
    summary_table = _summarize_metrics(metrics_table)
    summary_table.to_csv(output_dir / "spatialself_ablation_summary.csv", index=False)
    pd.DataFrame(status).to_csv(output_dir / "spatialself_ablation_status.csv", index=False)
    return metrics_table


def _summarize_metrics(metrics_table: pd.DataFrame) -> pd.DataFrame:
    if metrics_table.empty or "ablation" not in metrics_table:
        return pd.DataFrame()
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
        "n_pseudo_positive",
        "n_pseudo_negative",
        "n_soft_weak_positive",
        "n_ranking_pairs",
    ]
    available = [column for column in metric_columns if column in metrics_table.columns]
    if not available:
        return pd.DataFrame()

    summary = metrics_table.groupby("ablation", sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp04_model_ablation/spatialself",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        dest="ablations",
        help="Ablation to run. Repeatable. Defaults to all registered ablations.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    args = parser.parse_args()

    ablations = tuple(args.ablations) if args.ablations else tuple(ABLATIONS)
    metrics = run_ablation(
        Path(args.dataset_root),
        Path(args.output_dir),
        ablations=ablations,
        epochs=args.epochs,
        k_nn=args.k_nn,
        max_train_samples=args.max_train_samples,
    )
    if metrics.empty:
        print("No ablations ran.")
    else:
        print(
            metrics[
                [
                    "ablation",
                    "replicate",
                    "method",
                    "auroc",
                    "aupr",
                    "weak_signal_recall_at_threshold",
                    "empirical_fdr_at_threshold",
                ]
            ].to_string(index=False)
        )
    print(f"Wrote SpatialSelf ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
