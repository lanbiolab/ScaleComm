#!/usr/bin/env python
"""Tune SpatialSelf variants for contact and long-range niche mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.evaluate_range_mechanism_benchmark import evaluate_range_mechanism
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


DATASETS = {
    "lymph_scdesign3": {
        "tissue": "lymph_node",
        "replicate_dir": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/lymph/reference_lymph_node_scdesign3_replicate_00",
    },
    "melanoma_scdesign3": {
        "tissue": "melanoma",
        "replicate_dir": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/melanoma/reference_melanoma_scdesign3_replicate_00",
    },
}


CONFIGS = [
    {
        "run_id": "range_context_mechanism_prior_fixed",
        "model_name": "spatialself_range_context",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 40,
        "k_nn": 12,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.0,
        "spatial_null_contrastive_loss_weight": 0.0,
        "scale_contrastive_loss_weight": 0.0,
        "wavelet_scale_alignment_loss_weight": 0.0,
    },
    {
        "run_id": "range_context_guard",
        "model_name": "spatialself_range_context",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 50,
        "k_nn": 12,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "spatial_null_contrastive_loss_weight": 0.0,
        "scale_contrastive_loss_weight": 0.0,
        "wavelet_scale_alignment_loss_weight": 0.0,
    },
    {
        "run_id": "range_context_spatial_null",
        "model_name": "spatialself_range_context",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 50,
        "k_nn": 12,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "spatial_null_contrastive_loss_weight": 0.05,
        "spatial_null_negative_mode": "range_violation_receiver",
        "scale_contrastive_loss_weight": 0.0,
        "wavelet_scale_alignment_loss_weight": 0.0,
    },
    {
        "run_id": "adaptive_wavelet_scale",
        "model_name": "spatialself_wavelet_adaptive",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 50,
        "k_nn": 12,
        "use_spatial_graph": True,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "spatial_null_contrastive_loss_weight": 0.03,
        "spatial_null_negative_mode": "range_violation_receiver",
        "scale_contrastive_loss_weight": 0.03,
        "scale_contrastive_margin": 0.5,
        "wavelet_scale_alignment_loss_weight": 0.01,
    },
    {
        "run_id": "adaptive_wavelet_scale_contact",
        "model_name": "spatialself_wavelet_adaptive",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 60,
        "k_nn": 8,
        "use_spatial_graph": True,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.08,
        "mechanism_guard_loss_weight": 0.04,
        "spatial_null_contrastive_loss_weight": 0.04,
        "spatial_null_negative_mode": "range_violation_receiver",
        "scale_contrastive_loss_weight": 0.04,
        "scale_contrastive_margin": 0.5,
        "wavelet_scale_alignment_loss_weight": 0.015,
        "train_sampling_strategy": "balanced_per_lr",
    },
    {
        "run_id": "range_context_contact_kernel",
        "model_name": "spatialself_range_context",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 50,
        "k_nn": 12,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.04,
        "mechanism_guard_loss_weight": 0.02,
        "spatial_null_contrastive_loss_weight": 0.0,
        "scale_contrastive_loss_weight": 0.0,
        "wavelet_scale_alignment_loss_weight": 0.0,
        "train_sampling_strategy": "balanced_per_lr",
    },
    {
        "run_id": "adaptive_contact_kernel",
        "model_name": "spatialself_wavelet_adaptive",
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "epochs": 50,
        "k_nn": 10,
        "use_spatial_graph": True,
        "use_edge_prior_features": True,
        "hard_negative_loss_weight": 0.04,
        "mechanism_guard_loss_weight": 0.02,
        "spatial_null_contrastive_loss_weight": 0.02,
        "spatial_null_negative_mode": "range_violation_receiver",
        "scale_contrastive_loss_weight": 0.02,
        "scale_contrastive_margin": 0.5,
        "wavelet_scale_alignment_loss_weight": 0.01,
        "train_sampling_strategy": "balanced_per_lr",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/spatialself_mechanism_tuning",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    mechanism_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    for config in CONFIGS:
        for dataset_id, dataset in DATASETS.items():
            replicate_dir = PROJECT_ROOT / dataset["replicate_dir"]
            run_dir = output_dir / "runs" / config["run_id"] / dataset_id
            try:
                metrics = run_spatialself_on_replicate(
                    replicate_dir,
                    run_dir,
                    model_name=config["model_name"],
                    epochs=int(config.get("epochs", 50)),
                    k_nn=int(config.get("k_nn", 12)),
                    max_train_samples=int(args.max_train_samples),
                    pseudo_label_strategy=str(config["pseudo_label_strategy"]),
                    hard_negative_loss_weight=float(config.get("hard_negative_loss_weight", 0.0)),
                    mechanism_guard_loss_weight=float(config.get("mechanism_guard_loss_weight", 0.0)),
                    spatial_null_contrastive_loss_weight=float(
                        config.get("spatial_null_contrastive_loss_weight", 0.0)
                    ),
                    spatial_null_negative_mode=str(config.get("spatial_null_negative_mode", "random_receiver")),
                    scale_contrastive_loss_weight=float(config.get("scale_contrastive_loss_weight", 0.0)),
                    scale_contrastive_margin=float(config.get("scale_contrastive_margin", 0.5)),
                    wavelet_scale_alignment_loss_weight=float(
                        config.get("wavelet_scale_alignment_loss_weight", 0.0)
                    ),
                    train_sampling_strategy=str(config.get("train_sampling_strategy", "balanced_global")),
                    use_spatial_graph=bool(config.get("use_spatial_graph", True)),
                    use_edge_prior_features=bool(config.get("use_edge_prior_features", False)),
                    seed=int(args.seed),
                )
                metrics["run_id"] = config["run_id"]
                metrics["dataset_id"] = dataset_id
                metrics["tissue"] = dataset["tissue"]
                metric_rows.append(metrics)

                prediction_path = Path(metrics["prediction_path"])
                tables = evaluate_range_mechanism(
                    replicate_dir,
                    output_dir / "range_mechanism" / config["run_id"] / dataset_id,
                    predictions={config["run_id"]: prediction_path},
                )
                mechanism = tables["mechanism_range_summary"].copy()
                mechanism["run_id"] = config["run_id"]
                mechanism["dataset_id"] = dataset_id
                mechanism["tissue"] = dataset["tissue"]
                mechanism_rows.append(mechanism)
                status_rows.append(
                    {
                        "run_id": config["run_id"],
                        "dataset_id": dataset_id,
                        "status": "completed",
                        "detail": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "run_id": config["run_id"],
                        "dataset_id": dataset_id,
                        "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    mechanism = pd.concat(mechanism_rows, ignore_index=True) if mechanism_rows else pd.DataFrame()
    summary = _build_summary(metrics, mechanism)
    status = pd.DataFrame(status_rows)

    metrics.to_csv(output_dir / "spatialself_tuning_detail.csv", index=False)
    mechanism.to_csv(output_dir / "spatialself_tuning_by_mechanism.csv", index=False)
    summary.to_csv(output_dir / "spatialself_tuning_summary.csv", index=False)
    status.to_csv(output_dir / "spatialself_tuning_status.csv", index=False)
    manifest = {
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": DATASETS,
        "configs": CONFIGS,
        "outputs": {
            "detail": "spatialself_tuning_detail.csv",
            "summary": "spatialself_tuning_summary.csv",
            "by_mechanism": "spatialself_tuning_by_mechanism.csv",
            "status": "spatialself_tuning_status.csv",
        },
    }
    (output_dir / "spatialself_tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nWrote SpatialSelf tuning outputs to {output_dir}")


def _build_summary(metrics: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    base_cols = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
    ]
    summary = metrics.groupby("run_id", as_index=False)[base_cols].mean(numeric_only=True)
    if not mechanism.empty:
        mech = (
            mechanism.groupby(["run_id", "range_class"], as_index=False)[
                ["aupr", "top100_precision", "fdr_top5pct", "wrong_range_rate_top5pct", "too_close_long_range_rate_top5pct"]
            ]
            .mean(numeric_only=True)
        )
        wide = mech.pivot(index="run_id", columns="range_class")
        wide.columns = [f"{metric}__{range_class}" for metric, range_class in wide.columns]
        wide = wide.reset_index()
        summary = summary.merge(wide, on="run_id", how="left")
    summary["selection_score"] = (
        summary["aupr"].fillna(0.0)
        + 0.30 * summary.get("aupr__contact", 0.0).fillna(0.0)
        + 0.30 * summary.get("aupr__long_niche", 0.0).fillna(0.0)
        - 0.20 * summary["empirical_fdr_at_threshold"].fillna(1.0)
        - 0.15 * summary.get("too_close_long_range_rate_top5pct__long_niche", 0.0).fillna(0.0)
    )
    return summary.sort_values("selection_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()
