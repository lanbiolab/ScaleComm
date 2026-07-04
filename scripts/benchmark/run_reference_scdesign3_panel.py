#!/usr/bin/env python
"""Run and summarize the retained scDesign3 reference benchmark panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.evaluate_range_mechanism_benchmark import evaluate_range_mechanism
from scripts.benchmark.run_baseline_comparison import run_baselines_on_dataset


DEFAULT_CONFIG = "configs/experiments/reference_scdesign3_panel.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / (
        args.output_dir or config.get("output_dir", "outputs/reference_based_benchmark/panel_scdesign3_reference_baseline")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = tuple(config["methods"]["metric_panel"])
    spatialself_cfg = dict(config["spatialself_reference_baseline"])
    spatialself_display = str(spatialself_cfg.pop("display_name"))
    model_name = str(spatialself_cfg.pop("model_name"))
    commot_cfg = config.get("commot", {})
    range_cfg = config.get("range_evaluation", {})

    all_metrics = []
    all_status = []
    all_range = []
    run_manifest = []

    for dataset_cfg in config["datasets"]:
        dataset_id = str(dataset_cfg["dataset_id"])
        dataset_root = PROJECT_ROOT / dataset_cfg["dataset_root"]
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]
        dataset_out = output_dir / dataset_id
        dataset_out.mkdir(parents=True, exist_ok=True)

        metrics = run_baselines_on_dataset(
            dataset_root,
            dataset_out / "method_runs",
            methods=methods,
            cot_nitermax=int(commot_cfg.get("cot_nitermax", 500)),
            cellchat_nboot=20,
            spatialself_epochs=int(spatialself_cfg["epochs"]),
            spatialself_k_nn=int(spatialself_cfg["k_nn"]),
            spatialself_max_train_samples=int(spatialself_cfg["max_train_samples"]),
            spatialself_pseudo_label_strategy=str(spatialself_cfg["pseudo_label_strategy"]),
            spatialself_ranking_loss_weight=float(spatialself_cfg["ranking_loss_weight"]),
            spatialself_ranking_margin=float(spatialself_cfg["ranking_margin"]),
            spatialself_feasibility_loss_weight=float(spatialself_cfg["feasibility_loss_weight"]),
            spatialself_hard_negative_loss_weight=float(spatialself_cfg["hard_negative_loss_weight"]),
            spatialself_context_contrastive_loss_weight=float(spatialself_cfg["context_contrastive_loss_weight"]),
            spatialself_context_contrastive_margin=float(spatialself_cfg["context_contrastive_margin"]),
            spatialself_context_contrastive_negative_mode=str(spatialself_cfg["context_contrastive_negative_mode"]),
            spatialself_null_candidate_loss_weight=float(spatialself_cfg["null_candidate_loss_weight"]),
            spatialself_mechanism_guard_loss_weight=float(spatialself_cfg.get("mechanism_guard_loss_weight", 0.0)),
            spatialself_lr_cap_positive=int(spatialself_cfg["lr_cap_positive"]),
            spatialself_lr_cap_negative=int(spatialself_cfg["lr_cap_negative"]),
            spatialself_no_lr_embedding=bool(spatialself_cfg["no_lr_embedding"]),
            spatialself_no_distance=bool(spatialself_cfg["no_distance"]),
            spatialself_no_spatial_graph=bool(spatialself_cfg["no_spatial_graph"]),
            spatialself_no_expression_features=bool(spatialself_cfg["no_expression_features"]),
            cellnest_num_epoch=2,
            cellnest_hidden=32,
            deeptalk_input_root=None,
            spatalk_per_num=100,
            spatalk_min_pairs=1,
            spatalk_use_synthetic_pathways=False,
        )
        status = _read_status(dataset_out / "method_runs" / "baseline_status.csv")

        if not metrics.empty:
            metrics = metrics.copy()
            metrics["dataset_id"] = dataset_id
            metrics["tissue"] = str(dataset_cfg.get("tissue", ""))
            metrics["replicate_dir"] = str(replicate_dir.relative_to(PROJECT_ROOT))
            metrics["method_display"] = metrics["method"].map(
                {
                    model_name: spatialself_display,
                    "commot": "commot",
                }
            ).fillna(metrics["method"].astype(str))
            all_metrics.append(metrics)

        if not status.empty:
            status = status.copy()
            status["dataset_id"] = dataset_id
            status["tissue"] = str(dataset_cfg.get("tissue", ""))
            status["method_display"] = status["method"].map(
                {
                    model_name: spatialself_display,
                    "commot": "commot",
                }
            ).fillna(status["method"].astype(str))
            all_status.append(status)

        prediction_paths = {
            spatialself_display: dataset_out / "method_runs" / model_name / f"{replicate_dir.name}__{model_name}_predictions.csv",
            "commot": dataset_out / "method_runs" / "commot" / f"{replicate_dir.name}__commot_predictions.csv",
        }
        if all(path.exists() for path in prediction_paths.values()):
            range_tables = evaluate_range_mechanism(
                replicate_dir,
                dataset_out / "range_eval",
                predictions=prediction_paths,
                top_k=int(range_cfg.get("top_k", 100)),
                threshold_quantile=float(range_cfg.get("threshold_quantile", 0.95)),
            )
            method_range = range_tables["method_range_summary"].copy()
            method_range["dataset_id"] = dataset_id
            method_range["tissue"] = str(dataset_cfg.get("tissue", ""))
            all_range.append(method_range)

        run_manifest.append(
            {
                "dataset_id": dataset_id,
                "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT)),
                "replicate_dir": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "output_dir": str(dataset_out.relative_to(PROJECT_ROOT)),
            }
        )

    metrics_table = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    status_table = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    range_table = pd.concat(all_range, ignore_index=True) if all_range else pd.DataFrame()

    metrics_path = output_dir / "reference_scdesign3_metrics.csv"
    status_path = output_dir / "reference_scdesign3_status.csv"
    summary_path = output_dir / "reference_scdesign3_summary.csv"
    range_path = output_dir / "reference_scdesign3_range_summary.csv"
    parameter_path = output_dir / "reference_scdesign3_parameters.csv"
    manifest_path = output_dir / "reference_scdesign3_manifest.json"

    metrics_table.to_csv(metrics_path, index=False)
    status_table.to_csv(status_path, index=False)
    _build_summary(metrics_table).to_csv(summary_path, index=False)
    range_table.to_csv(range_path, index=False)
    _parameter_table(spatialself_display, spatialself_cfg, commot_cfg, config).to_csv(parameter_path, index=False)

    manifest = {
        "experiment_id": config.get("experiment_id", "reference_scdesign3_panel"),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "methods": list(methods),
        "spatialself_display_name": spatialself_display,
        "spatialself_model_name": model_name,
        "datasets": run_manifest,
        "outputs": {
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "status": str(status_path.relative_to(PROJECT_ROOT)),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "range_summary": str(range_path.relative_to(PROJECT_ROOT)),
            "parameters": str(parameter_path.relative_to(PROJECT_ROOT)),
        },
        "claim_scope": config.get("claim_scope", ""),
        "limitations": config.get("limitations", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not metrics_table.empty:
        cols = [
            "dataset_id",
            "method_display",
            "aupr",
            "top_k_precision",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
        ]
        print(metrics_table[[c for c in cols if c in metrics_table.columns]].to_string(index=False))
    if not range_table.empty:
        cols = [
            "dataset_id",
            "method",
            "aupr",
            "top_k_precision",
            "distance_profile_emd",
            "distance_median_error",
        ]
        print()
        print(range_table[[c for c in cols if c in range_table.columns]].to_string(index=False))
    print(f"Wrote scDesign3 reference panel to {output_dir}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_status(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "distance_profile_emd",
        "distance_median_error",
    ]
    available = [column for column in columns if column in metrics.columns]
    if not available:
        return metrics[["dataset_id", "method_display"]].drop_duplicates().reset_index(drop=True)
    summary = metrics.groupby(["dataset_id", "tissue", "method_display"], sort=False)[available].mean().reset_index()
    return summary


def _parameter_table(spatialself_display: str, spatialself_cfg: dict, commot_cfg: dict, config: dict) -> pd.DataFrame:
    rows = []
    for key, value in spatialself_cfg.items():
        rows.append({"method_display": spatialself_display, "parameter": key, "value": str(value)})
    for key, value in commot_cfg.items():
        rows.append({"method_display": "commot", "parameter": key, "value": str(value)})
    for key, value in config.get("range_evaluation", {}).items():
        rows.append({"method_display": "shared_range_evaluation", "parameter": key, "value": str(value)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
