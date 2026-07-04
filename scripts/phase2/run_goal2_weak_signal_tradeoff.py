#!/usr/bin/env python
"""Run revised Phase 2 weak-signal/false-positive tradeoff smoke panels."""

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

from scripts.benchmark.run_commot import run_commot_on_replicate
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


@dataclass
class MethodStatus:
    dataset: str
    replicate: str
    method: str
    status: str
    detail: str = ""
    prediction_path: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal2_weak_signal_tradeoff.yaml")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset key from the config. Repeatable.")
    parser.add_argument("--method", action="append", dest="methods", help="Method to run. Repeatable.")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal2_weak_signal_tradeoff"))
    selected_datasets = tuple(args.datasets or ("lymph_node_real_background_smoke", "tonsil_real_background_smoke", "melanoma_real_background_smoke"))
    selected_methods = tuple(args.methods or config["methods"]["default_smoke"])

    rows: list[dict] = []
    statuses: list[MethodStatus] = []
    for dataset_key in selected_datasets:
        dataset_root = PROJECT_ROOT / config["datasets"][dataset_key]["dataset_root"]
        replicate_dirs = _replicate_dirs(dataset_root)
        if not replicate_dirs:
            statuses.append(MethodStatus(dataset_key, "", "dataset", "missing", f"No replicate dirs under {dataset_root}"))
            continue
        for replicate_dir in replicate_dirs:
            for method in selected_methods:
                method_dir = output_dir / dataset_key / method
                try:
                    metrics = _run_method(method, replicate_dir, method_dir, config)
                    metrics["dataset"] = dataset_key
                    metrics["tradeoff_score"] = _tradeoff_score(metrics)
                    rows.append(metrics)
                    statuses.append(
                        MethodStatus(
                            dataset_key,
                            replicate_dir.name,
                            method,
                            "completed",
                            prediction_path=str(metrics.get("prediction_path", "")),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - per-method status is part of the benchmark.
                    statuses.append(MethodStatus(dataset_key, replicate_dir.name, method, "failed", str(exc)))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table = pd.DataFrame(rows)
    status_table = pd.DataFrame([asdict(status) for status in statuses])
    if not metrics_table.empty:
        metrics_table.to_csv(output_dir / "goal2_weak_signal_tradeoff_metrics.csv", index=False)
        _summarize(metrics_table).to_csv(output_dir / "goal2_weak_signal_tradeoff_summary.csv", index=False)
        _claim_table(metrics_table, status_table).to_csv(output_dir / "goal2_weak_signal_tradeoff_claims.csv", index=False)
    status_table.to_csv(output_dir / "goal2_weak_signal_tradeoff_status.csv", index=False)
    manifest = {
        "config": args.config,
        "datasets": selected_datasets,
        "methods": selected_methods,
        "output_dir": str(output_dir),
        "metrics_rows": int(len(metrics_table)),
        "completed": int((status_table["status"] == "completed").sum()) if not status_table.empty else 0,
        "failed": int((status_table["status"] == "failed").sum()) if not status_table.empty else 0,
    }
    (output_dir / "goal2_weak_signal_tradeoff_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if metrics_table.empty:
        print("No Goal 2 metrics generated.")
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
            "tradeoff_score",
        ]
        print(metrics_table[cols].to_string(index=False))
    print(f"Wrote revised Goal 2 outputs to {output_dir}")


def _run_method(method: str, replicate_dir: Path, output_dir: Path, config: dict) -> dict:
    if method in {"spatialself_wavelet", "spatialself_wavelet_edge_prior"}:
        spatialself = config.get("spatialself", {})
        is_edge_prior = method == "spatialself_wavelet_edge_prior"
        metrics = run_spatialself_on_replicate(
            replicate_dir,
            output_dir,
            model_name="spatialself_wavelet",
            epochs=int(spatialself.get("epochs", 8)),
            k_nn=int(spatialself.get("k_nn", 12)),
            max_train_samples=int(spatialself.get("max_train_samples", 6000)),
            pseudo_label_strategy=str(spatialself.get("pseudo_label_strategy", "lr_prior_lr_capped_mechanism_niche_expression_distance")),
            weak_positive_fraction=float(spatialself.get("weak_positive_fraction", 0.0)),
            ranking_loss_weight=float(spatialself.get("ranking_loss_weight", 0.15)),
            ranking_margin=float(spatialself.get("ranking_margin", 1.0)),
            lr_cap_positive=int(spatialself.get("lr_cap_positive", 80)),
            lr_cap_negative=int(spatialself.get("lr_cap_negative", 240)),
            use_edge_prior_features=is_edge_prior,
        )
        if is_edge_prior:
            metrics["method"] = method
            _rewrite_prediction_method(Path(metrics["prediction_path"]), method)
        return metrics
    if method == "commot":
        commot = config.get("commot", {})
        return run_commot_on_replicate(
            replicate_dir,
            output_dir,
            cot_nitermax=int(commot.get("cot_nitermax", 200)),
        )
    raise ValueError(f"Method {method} is not enabled in revised Goal 2 smoke runner.")


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "tradeoff_score",
    ]
    available = [column for column in columns if column in metrics.columns]
    summary = metrics.groupby(["dataset", "method"], sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _claim_table(metrics: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in metrics.groupby("dataset", sort=False):
        best_tradeoff = group.sort_values("tradeoff_score", ascending=False).iloc[0]
        best_fp = group.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
        best_weak = group.sort_values("weak_signal_recall_at_threshold", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "smoke_tradeoff_evidence_not_manuscript_ready",
                "best_tradeoff_method": best_tradeoff["method"],
                "best_tradeoff_score": best_tradeoff["tradeoff_score"],
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["false_positive_high_expression_rate_at_threshold"],
                "highest_weak_recall_method": best_weak["method"],
                "highest_weak_recall": best_weak["weak_signal_recall_at_threshold"],
                "completed_methods": int(((status["dataset"] == dataset) & (status["status"] == "completed")).sum()),
                "failed_methods": int(((status["dataset"] == dataset) & (status["status"] == "failed")).sum()),
            }
        )
    return pd.DataFrame(rows)


def _tradeoff_score(metrics: dict) -> float:
    weak = _as_float(metrics.get("weak_signal_recall_at_threshold"))
    fdr = _as_float(metrics.get("empirical_fdr_at_threshold"))
    fp = _as_float(metrics.get("false_positive_high_expression_rate_at_threshold"))
    aupr = _as_float(metrics.get("aupr"))
    return float(weak + 0.25 * aupr - 0.5 * fdr - 0.5 * fp)


def _as_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(value):
        return 0.0
    return value


def _rewrite_prediction_method(prediction_path: Path, method: str) -> None:
    prediction = pd.read_csv(prediction_path)
    prediction["method"] = method
    prediction.to_csv(prediction_path, index=False)


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
