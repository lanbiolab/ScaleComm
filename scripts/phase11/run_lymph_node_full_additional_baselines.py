#!/usr/bin/env python
"""Run additional baselines on the native full-transcriptome lymph node subset.

This script deliberately writes extended status/metric files instead of
overwriting the original Phase 11 SpatialSelf/COMMOT baseline outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellchat import run_cellchat_on_replicate
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.benchmark.run_spatalk import run_spatalk_on_replicate


OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
BASELINE_DIR = OUT_DIR / "baselines"
RESULT_DIR = OUT_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        choices=("cellchat", "spatalk", "cellnest"),
        help="Method to run. Repeatable. Defaults to all additional baselines.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if prediction output already exists.")
    parser.add_argument("--cellchat-nboot", type=int, default=10)
    parser.add_argument("--spatalk-per-num", type=int, default=50)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    parser.add_argument("--cellnest-num-epoch", type=int, default=2)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    methods = args.method if args.method else ["cellchat", "spatalk", "cellnest"]
    rows = []
    metrics_rows = []
    for method in methods:
        method_dir = BASELINE_DIR / method
        prediction_path = method_dir / f"{DATASET_DIR.name}__{method}_predictions.csv"
        if method == "cellnest":
            prediction_path = method_dir / f"{DATASET_DIR.name}__cellnest_predictions.csv"
        if prediction_path.exists() and not args.force:
            rows.append(
                {
                    "method": method,
                    "status": "skipped_existing_prediction",
                    "prediction_path": rel(prediction_path),
                    "detail": "Existing prediction file kept; pass --force to rerun.",
                }
            )
            metric_path = method_dir / f"{DATASET_DIR.name}__{method}_metrics.json"
            if metric_path.exists():
                metrics_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
            continue

        try:
            if method == "cellchat":
                metrics = run_cellchat_on_replicate(
                    DATASET_DIR,
                    method_dir,
                    nboot=args.cellchat_nboot,
                )
                detail = f"CellChat completed with nboot={args.cellchat_nboot}."
            elif method == "spatalk":
                metrics = run_spatalk_on_replicate(
                    DATASET_DIR,
                    method_dir,
                    per_num=args.spatalk_per_num,
                    min_pairs=args.spatalk_min_pairs,
                    use_synthetic_pathways=args.spatalk_use_synthetic_pathways,
                )
                detail = (
                    f"SpaTalk completed with per_num={args.spatalk_per_num}, "
                    f"min_pairs={args.spatalk_min_pairs}, "
                    f"use_synthetic_pathways={args.spatalk_use_synthetic_pathways}."
                )
            elif method == "cellnest":
                metrics = run_cellnest_on_replicate(
                    DATASET_DIR,
                    method_dir,
                    num_epoch=args.cellnest_num_epoch,
                    hidden=args.cellnest_hidden,
                )
                detail = f"CellNEST completed with num_epoch={args.cellnest_num_epoch}, hidden={args.cellnest_hidden}."
            else:
                raise ValueError(f"Unsupported method: {method}")

            metrics_rows.append(metrics)
            rows.append(
                {
                    "method": method,
                    "status": "completed",
                    "prediction_path": rel(Path(metrics.get("prediction_path", prediction_path))),
                    "detail": detail,
                }
            )
        except Exception as exc:  # noqa: BLE001 - failures are part of the baseline status panel.
            failure_log = RESULT_DIR / f"lymph_node_full_{method}_failure.log"
            failure_log.write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            rows.append(
                {
                    "method": method,
                    "status": "failed",
                    "prediction_path": rel(prediction_path),
                    "detail": f"{type(exc).__name__}: {exc}",
                    "failure_log": rel(failure_log),
                }
            )

    status = pd.DataFrame(rows)
    metrics_table = pd.DataFrame(metrics_rows)
    status.to_csv(RESULT_DIR / "lymph_node_full_additional_baseline_status.csv", index=False)
    metrics_table.to_csv(RESULT_DIR / "lymph_node_full_additional_baseline_metrics.csv", index=False)
    write_extended_baseline_tables(status, metrics_table)

    print(status.to_string(index=False))
    if not metrics_table.empty:
        columns = [column for column in ["replicate", "method", "evaluation_mode", "n_candidates", "top_lr_by_mean_score", "top_lr_mean_score"] if column in metrics_table.columns]
        print()
        print(metrics_table[columns].to_string(index=False))


def write_extended_baseline_tables(additional_status: pd.DataFrame, additional_metrics: pd.DataFrame) -> None:
    base_status_path = BASELINE_DIR / "baseline_status.csv"
    if base_status_path.exists():
        base_status = pd.read_csv(base_status_path)
    else:
        base_status = pd.DataFrame(columns=["method", "status", "detail"])
    status = pd.concat([base_status, additional_status], ignore_index=True, sort=False)
    status = status.drop_duplicates(subset=["method"], keep="last")
    status.to_csv(BASELINE_DIR / "baseline_status_extended.csv", index=False)

    base_metrics_path = BASELINE_DIR / "baseline_metrics.csv"
    if base_metrics_path.exists():
        base_metrics = pd.read_csv(base_metrics_path)
    else:
        base_metrics = pd.DataFrame()
    metrics = pd.concat([base_metrics, additional_metrics], ignore_index=True, sort=False)
    if not metrics.empty and "method" in metrics.columns:
        metrics = metrics.drop_duplicates(subset=["method", "replicate"], keep="last")
    metrics.to_csv(BASELINE_DIR / "baseline_metrics_extended.csv", index=False)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
