#!/usr/bin/env python
"""Run CellChat on generated CCC benchmark replicate directories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_predictions_auto


R_RUNNER = PROJECT_ROOT / "scripts/benchmark/run_cellchat_replicate.R"


def run_cellchat_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    conda_env: str = "cellchat_env",
    nboot: int = 20,
) -> dict:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{replicate_dir.name}__cellchat_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__cellchat_metrics.json"

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "Rscript",
        str(R_RUNNER),
        str(replicate_dir / "cells.csv"),
        str(replicate_dir / "expression.csv"),
        str(replicate_dir / "lr_pairs.csv"),
        str(replicate_dir / "candidate_interactions.csv"),
        str(prediction_path),
        str(int(nboot)),
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "CellChat failed for "
            f"{replicate_dir}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    predictions = pd.read_csv(prediction_path)
    metrics = evaluate_predictions_auto(candidates, predictions)
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": "cellchat",
            "prediction_path": str(prediction_path),
            "nboot": int(nboot),
            "output_level": "cell_type_pair_projected_to_candidate_edges",
        }
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/cellchat",
        help="Directory for prediction and metric outputs.",
    )
    parser.add_argument("--conda-env", default="cellchat_env")
    parser.add_argument("--nboot", type=int, default=20)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    output_dir = Path(args.output_dir)
    metrics = [
        run_cellchat_on_replicate(
            replicate_dir,
            output_dir,
            conda_env=args.conda_env,
            nboot=args.nboot,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / "cellchat_metrics.csv", index=False)
    print(
        metrics_table[
            [
                "replicate",
                "method",
                "auroc",
                "aupr",
                "top_k_precision",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote CellChat outputs to {output_dir}")


if __name__ == "__main__":
    main()
