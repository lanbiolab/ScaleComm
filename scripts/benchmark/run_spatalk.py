#!/usr/bin/env python
"""Run SpaTalk on generated CCC benchmark replicate directories."""

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


R_RUNNER = PROJECT_ROOT / "scripts/benchmark/run_spatalk_replicate.R"


def run_spatalk_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    conda_env: str = "spatalk_env",
    per_num: int = 100,
    min_pairs: int = 1,
    use_synthetic_pathways: bool = False,
) -> dict:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{replicate_dir.name}__spatalk_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__spatalk_metrics.json"

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
        str(int(per_num)),
        str(int(min_pairs)),
        "TRUE" if use_synthetic_pathways else "FALSE",
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "SpaTalk failed for "
            f"{replicate_dir}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    predictions = pd.read_csv(prediction_path)
    metrics = evaluate_predictions_auto(candidates, predictions)
    status = "completed"
    if "spatalk_status" in predictions.columns:
        unique_status = sorted(set(predictions["spatalk_status"].dropna().astype(str)))
        if unique_status:
            status = "; ".join(unique_status[:3])
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": "spatalk",
            "prediction_path": str(prediction_path),
            "output_level": "cell_type_pair_projected_to_candidate_edges",
            "per_num": int(per_num),
            "min_pairs": int(min_pairs),
            "use_synthetic_pathways": bool(use_synthetic_pathways),
            "spatalk_status": status,
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
        default="outputs/phase1/exp03_baseline_comparison/spatalk",
        help="Directory for prediction and metric outputs.",
    )
    parser.add_argument("--conda-env", default="spatalk_env")
    parser.add_argument("--per-num", type=int, default=100)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument(
        "--use-synthetic-pathways",
        action="store_true",
        help=(
            "Add minimal receptor-to-target pathway rows for synthetic benchmark genes. "
            "Use for smoke tests and report this setting separately from native SpaTalk."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    output_dir = Path(args.output_dir)
    metrics = [
        run_spatalk_on_replicate(
            replicate_dir,
            output_dir,
            conda_env=args.conda_env,
            per_num=args.per_num,
            min_pairs=args.min_pairs,
            use_synthetic_pathways=args.use_synthetic_pathways,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / "spatalk_metrics.csv", index=False)
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
                "spatalk_status",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote SpaTalk outputs to {output_dir}")


if __name__ == "__main__":
    main()
