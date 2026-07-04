#!/usr/bin/env python
"""Run representative CCC methods on noisy scDesign3 mechanism benchmarks."""

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
from scripts.benchmark.run_baseline_comparison import run_baselines_on_dataset
from scripts.benchmark.run_extended_baselines import run_spaotsc_on_replicate


DEFAULT_METHODS = (
    "spatialself_range_context",
    "commot",
    "cellchat",
    "cellnest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/datasets/noisy_scdesign3_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/method_runs",
    )
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--skip-spaotsc", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild summaries from existing metric/prediction files without rerunning methods.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(args.methods) if args.methods else DEFAULT_METHODS

    metric_rows = []
    status_rows = []
    mechanism_rows = []

    for row in manifest.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        noise_level = str(row.noise_level)
        replicate_dir = PROJECT_ROOT / str(row.replicate_dir)
        run_id = f"{dataset_id}_{noise_level}"
        run_out = output_dir / run_id
        if args.reuse_existing:
            recovered = _recover_existing_metrics(run_out, dataset_id=dataset_id, noise_level=noise_level)
            if not recovered.empty:
                metric_rows.append(recovered)
            status_rows.append(
                {
                    "dataset_id": dataset_id,
                    "noise_level": noise_level,
                    "method_group": "existing",
                    "status": "reused_existing_outputs",
                    "detail": str(run_out.relative_to(PROJECT_ROOT)),
                }
            )
            prediction_paths = _prediction_paths(run_out, replicate_dir.name)
            existing = {method: path for method, path in prediction_paths.items() if path.exists()}
            if existing:
                tables = evaluate_range_mechanism(
                    replicate_dir,
                    output_dir / "range_mechanism" / run_id,
                    predictions=existing,
                )
                mechanism = tables["mechanism_range_summary"].copy()
                mechanism["dataset_id"] = dataset_id
                mechanism["noise_level"] = noise_level
                mechanism_rows.append(mechanism)
            continue
        try:
            metrics = run_baselines_on_dataset(
                replicate_dir,
                run_out,
                methods=methods,
                cot_nitermax=500,
                cellchat_nboot=10,
                spatialself_epochs=50,
                spatialself_k_nn=12,
                spatialself_max_train_samples=12000,
                spatialself_pseudo_label_strategy="mechanism_niche_expression_distance",
                spatialself_ranking_loss_weight=0.0,
                spatialself_ranking_margin=1.0,
                spatialself_feasibility_loss_weight=0.0,
                spatialself_hard_negative_loss_weight=0.05,
                spatialself_context_contrastive_loss_weight=0.0,
                spatialself_context_contrastive_margin=0.5,
                spatialself_context_contrastive_negative_mode="random_receiver",
                spatialself_null_candidate_loss_weight=0.0,
                spatialself_mechanism_guard_loss_weight=0.03,
                spatialself_lr_cap_positive=50,
                spatialself_lr_cap_negative=200,
                spatialself_no_lr_embedding=False,
                spatialself_no_distance=False,
                spatialself_no_spatial_graph=True,
                spatialself_no_expression_features=False,
                cellnest_num_epoch=80,
                cellnest_hidden=32,
                deeptalk_input_root=None,
                spatalk_per_num=100,
                spatalk_min_pairs=1,
                spatalk_use_synthetic_pathways=False,
            )
            if not metrics.empty:
                metrics = metrics.copy()
                metrics["dataset_id"] = dataset_id
                metrics["noise_level"] = noise_level
                metric_rows.append(metrics)
            status_rows.append({"dataset_id": dataset_id, "noise_level": noise_level, "method_group": "core", "status": "completed", "detail": ""})
        except Exception as exc:
            recovered = _recover_existing_metrics(run_out, dataset_id=dataset_id, noise_level=noise_level)
            if not recovered.empty:
                metric_rows.append(recovered)
            status_rows.append(
                {
                    "dataset_id": dataset_id,
                    "noise_level": noise_level,
                    "method_group": "core",
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

        if not args.skip_spaotsc:
            try:
                spa_out = run_out / "spaotsc"
                spa_metrics = run_spaotsc_on_replicate(replicate_dir, spa_out)
                spa_metrics["dataset_id"] = dataset_id
                spa_metrics["noise_level"] = noise_level
                metric_rows.append(pd.DataFrame([spa_metrics]))
                status_rows.append({"dataset_id": dataset_id, "noise_level": noise_level, "method_group": "spaotsc", "status": "completed", "detail": ""})
            except Exception as exc:
                status_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "noise_level": noise_level,
                        "method_group": "spaotsc",
                        "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )

        prediction_paths = _prediction_paths(run_out, replicate_dir.name)
        existing = {method: path for method, path in prediction_paths.items() if path.exists()}
        if existing:
            tables = evaluate_range_mechanism(
                replicate_dir,
                output_dir / "range_mechanism" / run_id,
                predictions=existing,
            )
            mechanism = tables["mechanism_range_summary"].copy()
            mechanism["dataset_id"] = dataset_id
            mechanism["noise_level"] = noise_level
            mechanism_rows.append(mechanism)

    detail = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    mechanism_detail = pd.concat(mechanism_rows, ignore_index=True) if mechanism_rows else pd.DataFrame()
    summary = _summary(detail)
    mechanism_summary = _mechanism_summary(mechanism_detail)
    status = pd.DataFrame(status_rows)

    detail.to_csv(output_dir / "noisy_method_detail.csv", index=False)
    summary.to_csv(output_dir / "noisy_method_summary.csv", index=False)
    mechanism_detail.to_csv(output_dir / "noisy_method_by_mechanism.csv", index=False)
    mechanism_summary.to_csv(output_dir / "noisy_method_mechanism_summary.csv", index=False)
    status.to_csv(output_dir / "noisy_method_status.csv", index=False)
    manifest_out = {
        "manifest": args.manifest,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "methods": list(methods) + ([] if args.skip_spaotsc else ["spaotsc"]),
        "outputs": {
            "detail": "noisy_method_detail.csv",
            "summary": "noisy_method_summary.csv",
            "by_mechanism": "noisy_method_by_mechanism.csv",
            "mechanism_summary": "noisy_method_mechanism_summary.csv",
            "status": "noisy_method_status.csv",
        },
    }
    (output_dir / "noisy_method_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote noisy benchmark method outputs to {output_dir}")


def _prediction_paths(run_out: Path, replicate_name: str) -> dict[str, Path]:
    return {
        "SpatialSelf": run_out / "spatialself_range_context" / f"{replicate_name}__spatialself_range_context_predictions.csv",
        "COMMOT": run_out / "commot" / f"{replicate_name}__commot_predictions.csv",
        "CellChat": run_out / "cellchat" / f"{replicate_name}__cellchat_predictions.csv",
        "CellNEST": run_out / "cellnest" / f"{replicate_name}__cellnest_predictions.csv",
        "SpaOTsc": run_out / "spaotsc" / f"{replicate_name}__spaotsc_predictions.csv",
    }


def _recover_existing_metrics(run_out: Path, *, dataset_id: str, noise_level: str) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(run_out.glob("*/*_metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        except Exception:
            continue
        row["dataset_id"] = dataset_id
        row["noise_level"] = noise_level
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metrics = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
    ]
    return (
        detail.groupby(["noise_level", "method"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["noise_level", "aupr"], ascending=[True, False])
    )


def _mechanism_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()
    return (
        mechanism.groupby(["noise_level", "method", "range_class"], as_index=False)[
            ["aupr", "top100_precision", "fdr_top5pct", "wrong_range_rate_top5pct", "too_close_long_range_rate_top5pct"]
        ]
        .mean(numeric_only=True)
        .sort_values(["noise_level", "range_class", "aupr"], ascending=[True, True, False])
    )


if __name__ == "__main__":
    main()
