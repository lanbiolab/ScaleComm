#!/usr/bin/env python
"""Run revised Phase 2 adapter-aware baseline panel.

This runner is intentionally status-heavy. It executes methods when their local
adapter is available and records non-runnable methods with explicit reasons
instead of silently dropping them from the benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_baseline_comparison import PENDING_METHODS, run_baselines_on_dataset


DEFAULT_METHODS = (
    "spatialself_wavelet_edge_prior",
    "spatialself_wavelet",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
    "deeptalk",
    "niches",
)


@dataclass
class DatasetRun:
    dataset: str
    dataset_root: str
    output_dir: str
    requested_methods: str
    completed_metric_methods: int
    metrics_rows: int
    status_rows: int
    failed_or_blocked_rows: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal2_weak_signal_tradeoff.yaml")
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal2_full_baseline_panel")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset key from config. Repeatable.")
    parser.add_argument("--method", action="append", dest="methods", help="Method to run/record. Repeatable.")
    parser.add_argument("--adapter-light", action="store_true", help="Use low-cost adapter settings for R/deep baselines.")
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = tuple(args.datasets or ("tonsil_real_background_smoke",))
    selected_methods = tuple(args.methods or DEFAULT_METHODS)
    dataset_rows: list[DatasetRun] = []
    all_metrics: list[pd.DataFrame] = []
    all_status: list[pd.DataFrame] = []
    for dataset in selected_datasets:
        dataset_cfg = config["datasets"].get(dataset)
        if not dataset_cfg:
            all_status.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "method": "dataset",
                            "status": "missing_dataset_config",
                            "detail": f"Dataset key not in {args.config}",
                        }
                    ]
                )
            )
            continue
        dataset_root = PROJECT_ROOT / dataset_cfg["dataset_root"]
        dataset_out = output_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        kwargs = _runner_kwargs(config, adapter_light=args.adapter_light, spatalk_use_synthetic_pathways=args.spatalk_use_synthetic_pathways)
        metrics, status = _run_methods_isolated(dataset_root, dataset_out, selected_methods, kwargs)
        status_path = dataset_out / "baseline_status.csv"
        if not metrics.empty:
            metrics = metrics.copy()
            metrics["dataset"] = dataset
            metrics["baseline_evidence_level"] = metrics["method"].map(_evidence_level).fillna("candidate_edge_metric")
            metrics["interpretation_note"] = metrics["method"].map(_interpretation_note).fillna("")
            metrics.to_csv(dataset_out / "baseline_metrics.csv", index=False)
            _summarize_by_dataset(metrics).to_csv(dataset_out / "baseline_metrics_summary.csv", index=False)
            all_metrics.append(metrics)
        if not status.empty:
            status = status.copy()
            status["dataset"] = dataset
            status["baseline_evidence_level"] = status["method"].map(_evidence_level).fillna("status_only")
            status["interpretation_note"] = status["method"].map(_interpretation_note).fillna("")
            status.to_csv(status_path, index=False)
            all_status.append(status)
        failed_or_blocked = 0
        if not status.empty:
            failed_or_blocked = int(
                status["status"].astype(str).isin(
                    ["failed", "optional_blocked", "model_required", "completed_or_model_required"]
                ).sum()
            )
        dataset_rows.append(
            DatasetRun(
                dataset=dataset,
                dataset_root=str(dataset_root.relative_to(PROJECT_ROOT)),
                output_dir=str(dataset_out.relative_to(PROJECT_ROOT)),
                requested_methods=";".join(selected_methods),
                completed_metric_methods=int(metrics["method"].nunique()) if not metrics.empty and "method" in metrics else 0,
                metrics_rows=int(len(metrics)),
                status_rows=int(len(status)),
                failed_or_blocked_rows=failed_or_blocked,
            )
        )

    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined_status = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    if not combined_metrics.empty:
        combined_metrics.to_csv(output_dir / "goal2_full_baseline_metrics.csv", index=False)
        _summarize_by_dataset(combined_metrics).to_csv(output_dir / "goal2_full_baseline_summary.csv", index=False)
        _claim_table(combined_metrics, combined_status).to_csv(output_dir / "goal2_full_baseline_claims.csv", index=False)
    combined_status.to_csv(output_dir / "goal2_full_baseline_status.csv", index=False)
    pd.DataFrame([asdict(row) for row in dataset_rows]).to_csv(output_dir / "goal2_full_baseline_dataset_runs.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_full_baseline_panel",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": selected_datasets,
        "methods": selected_methods,
        "adapter_light": bool(args.adapter_light),
        "status_scope": "adapter-aware smoke/status panel; not final tuned baseline comparison",
        "pending_methods": PENDING_METHODS,
        "outputs": {
            "metrics": str((output_dir / "goal2_full_baseline_metrics.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_full_baseline_summary.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_full_baseline_status.csv").relative_to(PROJECT_ROOT)),
            "dataset_runs": str((output_dir / "goal2_full_baseline_dataset_runs.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (output_dir / "goal2_full_baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if combined_metrics.empty:
        print("No metric-producing baseline rows were generated.")
    else:
        cols = [
            "dataset",
            "replicate",
            "method",
            "auroc",
            "aupr",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
            "false_positive_high_expression_rate_at_threshold",
            "baseline_evidence_level",
        ]
        print(combined_metrics[[c for c in cols if c in combined_metrics.columns]].to_string(index=False))
    print(f"Wrote revised full baseline panel to {output_dir}")


def _run_methods_isolated(
    dataset_root: Path,
    dataset_out: Path,
    selected_methods: tuple[str, ...],
    kwargs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_tables: list[pd.DataFrame] = []
    status_tables: list[pd.DataFrame] = []
    for method in selected_methods:
        try:
            metrics = run_baselines_on_dataset(dataset_root, dataset_out, methods=(method,), **kwargs)
            if not metrics.empty:
                metric_tables.append(metrics)
            method_status_path = dataset_out / "baseline_status.csv"
            if method_status_path.exists():
                status_tables.append(pd.read_csv(method_status_path))
            else:
                status_tables.append(
                    pd.DataFrame(
                        [
                            {
                                "method": method,
                                "status": "missing_status",
                                "detail": "run_baselines_on_dataset returned without baseline_status.csv",
                            }
                        ]
                    )
                )
        except Exception as exc:  # noqa: BLE001 - per-method failure is evidence.
            status_tables.append(
                pd.DataFrame(
                    [
                        {
                            "method": method,
                            "status": "failed",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    ]
                )
            )
    metrics = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    status = pd.concat(status_tables, ignore_index=True) if status_tables else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(dataset_out / "baseline_metrics.csv", index=False)
        _summarize_by_dataset(metrics.assign(dataset="dataset_placeholder")).drop(columns=["dataset"], errors="ignore").to_csv(
            dataset_out / "baseline_metrics_summary.csv",
            index=False,
        )
    status.to_csv(dataset_out / "baseline_status.csv", index=False)
    return metrics, status


def _runner_kwargs(config: dict, *, adapter_light: bool, spatalk_use_synthetic_pathways: bool) -> dict:
    spatialself = config.get("spatialself", {})
    commot = config.get("commot", {})
    return {
        "cot_nitermax": int(commot.get("cot_nitermax", 200 if adapter_light else 500)),
        "cellchat_nboot": 5 if adapter_light else 20,
        "spatialself_epochs": int(spatialself.get("epochs", 8)),
        "spatialself_k_nn": int(spatialself.get("k_nn", 12)),
        "spatialself_max_train_samples": int(spatialself.get("max_train_samples", 6000)),
        "spatialself_pseudo_label_strategy": str(
            spatialself.get("pseudo_label_strategy", "lr_prior_lr_capped_mechanism_niche_expression_distance")
        ),
        "spatialself_weak_positive_fraction": float(spatialself.get("weak_positive_fraction", 0.0)),
        "spatialself_ranking_loss_weight": float(spatialself.get("ranking_loss_weight", 0.15)),
        "spatialself_ranking_margin": float(spatialself.get("ranking_margin", 1.0)),
        "spatialself_lr_cap_positive": int(spatialself.get("lr_cap_positive", 80)),
        "spatialself_lr_cap_negative": int(spatialself.get("lr_cap_negative", 240)),
        "spatialself_no_lr_embedding": False,
        "spatialself_no_distance": False,
        "spatialself_no_spatial_graph": False,
        "spatialself_no_expression_features": False,
        "cellnest_num_epoch": 1 if adapter_light else 2,
        "cellnest_hidden": 16 if adapter_light else 32,
        "deeptalk_input_root": None,
        "spatalk_per_num": 20 if adapter_light else 100,
        "spatalk_min_pairs": 1,
        "spatalk_use_synthetic_pathways": bool(spatalk_use_synthetic_pathways),
    }


def _summarize_by_dataset(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in columns if column in metrics.columns]
    if not available:
        return pd.DataFrame()
    summary = metrics.groupby(["dataset", "method"], sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["baseline_evidence_level"] = summary["method"].map(_evidence_level).fillna("candidate_edge_metric")
    summary["interpretation_note"] = summary["method"].map(_interpretation_note).fillna("")
    return summary


def _claim_table(metrics: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in metrics.groupby("dataset", sort=False):
        runnable = group.dropna(subset=["weak_signal_recall_at_threshold", "false_positive_high_expression_rate_at_threshold"], how="all")
        if runnable.empty:
            continue
        best_fp = runnable.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
        best_weak = runnable.sort_values("weak_signal_recall_at_threshold", ascending=False).iloc[0]
        group = runnable.copy()
        group["tradeoff_score"] = (
            group["weak_signal_recall_at_threshold"].fillna(0.0)
            + 0.25 * group["aupr"].fillna(0.0)
            - 0.5 * group["empirical_fdr_at_threshold"].fillna(0.0)
            - 0.5 * group["false_positive_high_expression_rate_at_threshold"].fillna(0.0)
        )
        best_tradeoff = group.sort_values("tradeoff_score", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "adapter_aware_smoke_not_final_tuned_baseline",
                "best_tradeoff_method": best_tradeoff["method"],
                "best_tradeoff_score": best_tradeoff["tradeoff_score"],
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["false_positive_high_expression_rate_at_threshold"],
                "highest_weak_recall_method": best_weak["method"],
                "highest_weak_recall": best_weak["weak_signal_recall_at_threshold"],
                "metric_methods": int(group["method"].nunique()),
                "status_methods": int(status.loc[status["dataset"] == dataset, "method"].nunique()) if not status.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _evidence_level(method: str) -> str:
    mapping = {
        "spatialself_wavelet_edge_prior": "candidate_edge_metric",
        "spatialself_wavelet": "candidate_edge_metric",
        "spatialself_gat": "candidate_edge_metric",
        "commot": "candidate_edge_metric",
        "cellchat": "cell_type_pair_projected_adapter_light",
        "spatalk": "cell_type_pair_projected_adapter_light",
        "cellnest": "candidate_edge_adapter_light",
        "deeptalk": "projection_only_model_required_unless_predict_ccc_exists",
        "niches": "optional_blocked_dependency_unreproducible",
        "nichenet": "downstream_validation_only",
    }
    return mapping.get(str(method), "status_only")


def _interpretation_note(method: str) -> str:
    notes = {
        "cellchat": "Cell-type-pair CellChat scores are projected to candidate edges; treat as adapter-light unless fully tuned.",
        "spatalk": "SpaTalk scores are projected to candidate edges; synthetic pathway option must be reported if used.",
        "cellnest": "Short CellNEST training is adapter-light; final manuscript needs tuned runtime/seed panel.",
        "deeptalk": "DeepTalk requires LR-specific predict_ccc outputs; missing model outputs are status evidence, not a failed score.",
        "niches": "NICHES is recorded as optional blocked in the local R dependency chain.",
    }
    return notes.get(str(method), "")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
