#!/usr/bin/env python
"""Run and assemble an expanded noisy scDesign3 mechanism benchmark.

This script keeps the original noisy method_runs directory intact and writes a
complete comparison bundle under method_runs_complete. It reuses existing core
outputs when available, uses the official-like CellNEST noisy rerun, and runs
the extended projection adapters for methods that were missing from the first
noisy benchmark pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.evaluate_range_mechanism_benchmark import evaluate_range_mechanism
from scripts.benchmark.run_baseline_comparison import run_baselines_on_dataset
from scripts.benchmark.run_extended_baselines import (
    run_cellphonedb_on_replicate,
    run_cytosignal_on_replicate,
    run_niches_on_replicate,
    run_spaotsc_on_replicate,
)


DISPLAY_NAME = {
    "spatialself_range_context": "SpatialSelf",
    "commot": "COMMOT",
    "spaotsc": "SpaOTsc",
    "cellchat": "CellChat",
    "cellphonedb": "CellPhoneDB",
    "niches": "NICHES",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
    "cytosignal": "CytoSignal",
    "spatalk": "SpaTalk",
}

METHOD_ORDER = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellChat",
    "CellPhoneDB",
    "NICHES",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
    "SpaTalk",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/datasets/noisy_scdesign3_manifest.csv",
    )
    parser.add_argument(
        "--existing-dir",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/method_runs",
    )
    parser.add_argument(
        "--cellnest-official-dir",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/cellnest_official_like",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/method_runs_complete",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    existing_dir = PROJECT_ROOT / args.existing_dir
    cellnest_official_dir = PROJECT_ROOT / args.cellnest_official_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[pd.DataFrame] = []
    mechanism_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    for row in manifest.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        noise_level = str(row.noise_level)
        replicate_dir = PROJECT_ROOT / str(row.replicate_dir)
        run_id = f"{dataset_id}_{noise_level}"
        run_out = output_dir / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        if not args.reuse_existing:
            _copy_existing_core_predictions(existing_dir / run_id, run_out, replicate_dir.name, status_rows, dataset_id, noise_level)
            _copy_cellnest_official(cellnest_official_dir / run_id, run_out, replicate_dir.name, status_rows, dataset_id, noise_level)
            _run_missing_core(replicate_dir, run_out, status_rows, dataset_id, noise_level)
            _run_extended(replicate_dir, run_out, status_rows, dataset_id, noise_level)

        recovered = _recover_metrics(run_out, dataset_id=dataset_id, noise_level=noise_level)
        if not recovered.empty:
            metric_rows.append(recovered)

        predictions = _prediction_paths(run_out, replicate_dir.name)
        existing = {method: path for method, path in predictions.items() if path.exists()}
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
    (output_dir / "noisy_method_manifest.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "existing_dir": args.existing_dir,
                "cellnest_official_dir": args.cellnest_official_dir,
                "methods": METHOD_ORDER,
                "outputs": {
                    "detail": "noisy_method_detail.csv",
                    "summary": "noisy_method_summary.csv",
                    "by_mechanism": "noisy_method_by_mechanism.csv",
                    "mechanism_summary": "noisy_method_mechanism_summary.csv",
                    "status": "noisy_method_status.csv",
                },
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote complete noisy benchmark outputs to {output_dir}")


def _copy_existing_core_predictions(source_run: Path, run_out: Path, replicate_name: str, status_rows: list[dict], dataset_id: str, noise_level: str) -> None:
    for method in ["spatialself_range_context", "commot", "cellchat", "spaotsc"]:
        src = source_run / method / f"{replicate_name}__{method}_predictions.csv"
        dst_dir = run_out / method
        dst = dst_dir / src.name
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            metrics_src = source_run / method / f"{replicate_name}__{method}_metrics.json"
            if metrics_src.exists():
                shutil.copy2(metrics_src, dst_dir / metrics_src.name)
            status_rows.append(_status(dataset_id, noise_level, method, "reused_existing", str(src.relative_to(PROJECT_ROOT))))
        else:
            status_rows.append(_status(dataset_id, noise_level, method, "missing_existing", str(src.relative_to(PROJECT_ROOT))))


def _copy_cellnest_official(source_run: Path, run_out: Path, replicate_name: str, status_rows: list[dict], dataset_id: str, noise_level: str) -> None:
    method = "cellnest"
    src = source_run / f"{replicate_name}__cellnest_predictions.csv"
    metrics_src = source_run / f"{replicate_name}__cellnest_metrics.json"
    dst_dir = run_out / method
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        if metrics_src.exists():
            shutil.copy2(metrics_src, dst_dir / metrics_src.name)
        status_rows.append(_status(dataset_id, noise_level, method, "reused_official_like_th60_k50_epoch500", str(src.relative_to(PROJECT_ROOT))))
    else:
        status_rows.append(_status(dataset_id, noise_level, method, "missing_official_like", str(src.relative_to(PROJECT_ROOT))))


def _run_missing_core(replicate_dir: Path, run_out: Path, status_rows: list[dict], dataset_id: str, noise_level: str) -> None:
    for method in ["spatalk", "deeptalk"]:
        pred = run_out / method / f"{replicate_dir.name}__{method}_predictions.csv"
        if pred.exists():
            status_rows.append(_status(dataset_id, noise_level, method, "existing", str(pred.relative_to(PROJECT_ROOT))))
            continue
        try:
            run_baselines_on_dataset(
                replicate_dir,
                run_out,
                methods=(method,),
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
            status_rows.append(_status(dataset_id, noise_level, method, "completed", str(pred.relative_to(PROJECT_ROOT))))
        except Exception as exc:
            status_rows.append(_status(dataset_id, noise_level, method, "failed", f"{type(exc).__name__}: {exc}"))


def _run_extended(replicate_dir: Path, run_out: Path, status_rows: list[dict], dataset_id: str, noise_level: str) -> None:
    runners = {
        "cellphonedb": run_cellphonedb_on_replicate,
        "niches": run_niches_on_replicate,
        "cytosignal": run_cytosignal_on_replicate,
    }
    for method, runner in runners.items():
        pred = run_out / method / f"{replicate_dir.name}__{method}_predictions.csv"
        if pred.exists():
            status_rows.append(_status(dataset_id, noise_level, method, "existing", str(pred.relative_to(PROJECT_ROOT))))
            continue
        try:
            runner(replicate_dir, run_out / method)
            status_rows.append(_status(dataset_id, noise_level, method, "completed", str(pred.relative_to(PROJECT_ROOT))))
        except Exception as exc:
            status_rows.append(_status(dataset_id, noise_level, method, "failed", f"{type(exc).__name__}: {exc}"))


def _prediction_paths(run_out: Path, replicate_name: str) -> dict[str, Path]:
    return {
        "SpatialSelf": run_out / "spatialself_range_context" / f"{replicate_name}__spatialself_range_context_predictions.csv",
        "COMMOT": run_out / "commot" / f"{replicate_name}__commot_predictions.csv",
        "SpaOTsc": run_out / "spaotsc" / f"{replicate_name}__spaotsc_predictions.csv",
        "CellChat": run_out / "cellchat" / f"{replicate_name}__cellchat_predictions.csv",
        "CellPhoneDB": run_out / "cellphonedb" / f"{replicate_name}__cellphonedb_predictions.csv",
        "NICHES": run_out / "niches" / f"{replicate_name}__niches_predictions.csv",
        "CellNEST": run_out / "cellnest" / f"{replicate_name}__cellnest_predictions.csv",
        "DeepTalk": run_out / "deeptalk" / f"{replicate_name}__deeptalk_predictions.csv",
        "CytoSignal": run_out / "cytosignal" / f"{replicate_name}__cytosignal_predictions.csv",
        "SpaTalk": run_out / "spatalk" / f"{replicate_name}__spatalk_predictions.csv",
    }


def _recover_metrics(run_out: Path, *, dataset_id: str, noise_level: str) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(run_out.glob("*/*_metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        except Exception:
            continue
        method = str(row.get("method", ""))
        row["method"] = DISPLAY_NAME.get(method, method)
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
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metrics if column in detail.columns]
    summary = detail.groupby(["noise_level", "method"], as_index=False)[available].mean(numeric_only=True)
    summary["_method_order"] = summary["method"].map({method: i for i, method in enumerate(METHOD_ORDER)}).fillna(999)
    return summary.sort_values(["noise_level", "_method_order"]).drop(columns=["_method_order"])


def _mechanism_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()
    summary = (
        mechanism.groupby(["noise_level", "method", "range_class"], as_index=False)[
            ["aupr", "top100_precision", "fdr_top5pct", "wrong_range_rate_top5pct", "too_close_long_range_rate_top5pct"]
        ]
        .mean(numeric_only=True)
    )
    summary["_method_order"] = summary["method"].map({method: i for i, method in enumerate(METHOD_ORDER)}).fillna(999)
    return summary.sort_values(["noise_level", "_method_order", "range_class"]).drop(columns=["_method_order"])


def _status(dataset_id: str, noise_level: str, method: str, status: str, detail: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "noise_level": noise_level,
        "method": DISPLAY_NAME.get(method, method),
        "status": status,
        "detail": detail,
    }


if __name__ == "__main__":
    main()
