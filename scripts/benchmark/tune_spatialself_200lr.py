#!/usr/bin/env python
"""Tune SpatialSelf on the fixed 200-LR scDesign3 mechanism benchmark."""

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


CONFIGS = [
    {
        "run_id": "current_no_edge_prior",
        "model_name": "spatialself_range_context",
        "epochs": 50,
        "k_nn": 12,
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "use_spatial_graph": False,
        "use_edge_prior_features": False,
        "train_sampling_strategy": "balanced_global",
    },
    {
        "run_id": "edge_prior_guard",
        "model_name": "spatialself_range_context",
        "epochs": 50,
        "k_nn": 12,
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "train_sampling_strategy": "balanced_global",
    },
    {
        "run_id": "edge_prior_guard_per_lr",
        "model_name": "spatialself_range_context",
        "epochs": 50,
        "k_nn": 12,
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "hard_negative_loss_weight": 0.05,
        "mechanism_guard_loss_weight": 0.03,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "train_sampling_strategy": "balanced_per_lr",
    },
    {
        "run_id": "edge_prior_stronger_guard",
        "model_name": "spatialself_range_context",
        "epochs": 60,
        "k_nn": 12,
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "hard_negative_loss_weight": 0.08,
        "mechanism_guard_loss_weight": 0.05,
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "train_sampling_strategy": "balanced_global",
    },
    {
        "run_id": "edge_prior_spatial_null",
        "model_name": "spatialself_range_context",
        "epochs": 60,
        "k_nn": 12,
        "pseudo_label_strategy": "mechanism_niche_expression_distance",
        "hard_negative_loss_weight": 0.06,
        "mechanism_guard_loss_weight": 0.04,
        "spatial_null_contrastive_loss_weight": 0.03,
        "spatial_null_negative_mode": "range_violation_receiver",
        "use_spatial_graph": False,
        "use_edge_prior_features": True,
        "train_sampling_strategy": "balanced_global",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_200lr/datasets/scdesign3_200lr_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_200lr/spatialself_tuning",
    )
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", action="append", dest="configs")
    args = parser.parse_args()

    manifest_path = _project_path(args.manifest)
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    configs = [cfg for cfg in CONFIGS if not args.configs or cfg["run_id"] in set(args.configs)]

    rows = []
    status_rows = []
    for config in configs:
        for row in manifest.itertuples(index=False):
            dataset_id = str(row.dataset_id)
            replicate_dir = _project_path(str(row.replicate_dir))
            run_dir = output_dir / "runs" / config["run_id"] / dataset_id
            try:
                metrics = run_spatialself_on_replicate(
                    replicate_dir,
                    run_dir,
                    model_name=str(config["model_name"]),
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
                    train_sampling_strategy=str(config.get("train_sampling_strategy", "balanced_global")),
                    use_spatial_graph=bool(config.get("use_spatial_graph", True)),
                    use_edge_prior_features=bool(config.get("use_edge_prior_features", False)),
                    seed=int(args.seed),
                )
                metrics["run_id"] = config["run_id"]
                metrics["dataset_id"] = dataset_id
                metrics["tissue"] = str(getattr(row, "tissue", ""))
                rows.append(metrics)
                status_rows.append(
                    {
                        "run_id": config["run_id"],
                        "dataset_id": dataset_id,
                        "status": "completed",
                        "detail": str(Path(metrics["prediction_path"]).relative_to(PROJECT_ROOT)),
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

    detail = pd.DataFrame(rows)
    summary = _summary(detail)
    status = pd.DataFrame(status_rows)
    detail.to_csv(output_dir / "spatialself_200lr_tuning_detail.csv", index=False)
    summary.to_csv(output_dir / "spatialself_200lr_tuning_summary.csv", index=False)
    status.to_csv(output_dir / "spatialself_200lr_tuning_status.csv", index=False)
    (output_dir / "spatialself_200lr_tuning_manifest.json").write_text(
        json.dumps({"manifest": args.manifest, "configs": configs}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote SpatialSelf 200-LR tuning outputs to {output_dir}")


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metrics = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "calibration_error_10bin",
    ]
    summary = detail.groupby("run_id", as_index=False)[metrics].mean(numeric_only=True)
    summary["selection_score"] = (
        summary["aupr"].fillna(0.0)
        + 0.10 * summary["auroc"].fillna(0.0)
        + 0.08 * summary["top_k_precision"].fillna(0.0)
        - 0.18 * summary["empirical_fdr_at_threshold"].fillna(1.0)
    )
    return summary.sort_values("selection_score", ascending=False).reset_index(drop=True)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
