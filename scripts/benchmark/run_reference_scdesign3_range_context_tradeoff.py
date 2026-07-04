#!/usr/bin/env python
"""Run a compact tradeoff matrix for the scDesign3 range-context candidate."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


DEFAULT_CONFIG = "configs/experiments/reference_scdesign3_range_context_tradeoff.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / (
        args.output_dir or config.get("output_dir", "outputs/reference_based_benchmark/range_context_tradeoff")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base = config["base_model"]
    grid = config["grid"]
    rows = []

    for dataset_cfg in config["datasets"]:
        dataset_id = str(dataset_cfg["dataset_id"])
        tissue = str(dataset_cfg.get("tissue", ""))
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]

        for mechanism_guard_loss_weight, positive_quantile, negative_quantile in itertools.product(
            grid["mechanism_guard_loss_weight"],
            grid["positive_quantile"],
            grid["negative_quantile"],
        ):
            run_id = (
                f"{dataset_id}__guard{str(mechanism_guard_loss_weight).replace('.', 'p')}"
                f"__pos{str(positive_quantile).replace('.', 'p')}"
                f"__neg{str(negative_quantile).replace('.', 'p')}"
            )
            run_out = output_dir / "runs" / run_id
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                run_out,
                model_name=str(base["model_name"]),
                epochs=int(base["epochs"]),
                k_nn=int(base["k_nn"]),
                max_train_samples=int(base["max_train_samples"]),
                pseudo_label_strategy=str(base["pseudo_label_strategy"]),
                hard_negative_loss_weight=float(base["hard_negative_loss_weight"]),
                ranking_loss_weight=float(base.get("ranking_loss_weight", 0.0)),
                ranking_margin=float(base.get("ranking_margin", 1.0)),
                mechanism_guard_loss_weight=float(mechanism_guard_loss_weight),
                positive_quantile=float(positive_quantile),
                negative_quantile=float(negative_quantile),
            )
            metrics["dataset_id"] = dataset_id
            metrics["tissue"] = tissue
            metrics["run_id"] = run_id
            metrics["mechanism_guard_loss_weight"] = float(mechanism_guard_loss_weight)
            metrics["positive_quantile"] = float(positive_quantile)
            metrics["negative_quantile"] = float(negative_quantile)
            rows.append(metrics)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "tradeoff_metrics.csv", index=False)
    _best_per_dataset(table, config).to_csv(output_dir / "tradeoff_best_by_dataset.csv", index=False)
    manifest = {
        "experiment_id": config.get("experiment_id", "reference_scdesign3_range_context_tradeoff"),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": [row["dataset_id"] for row in config["datasets"]],
        "grid": grid,
        "outputs": {
            "metrics": str((output_dir / "tradeoff_metrics.csv").relative_to(PROJECT_ROOT)),
            "best_by_dataset": str((output_dir / "tradeoff_best_by_dataset.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (output_dir / "tradeoff_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cols = [
        "dataset_id",
        "run_id",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
    ]
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print(f"Wrote scDesign3 range-context tradeoff matrix to {output_dir}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _best_per_dataset(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    priority = list(config.get("selection_priority", []))
    ascending = []
    for column in priority:
        if column in {"empirical_fdr_at_threshold"}:
            ascending.append(True)
        else:
            ascending.append(False)
    ranked = []
    for dataset_id, group in table.groupby("dataset_id", sort=False):
        local = group.sort_values(priority, ascending=ascending, kind="mergesort").head(1).copy()
        local["selection_rank"] = 1
        ranked.append(local)
    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


if __name__ == "__main__":
    main()
