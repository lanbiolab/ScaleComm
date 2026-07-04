#!/usr/bin/env python
"""Run COMMOT on generated CCC benchmark replicate directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.baselines import score_commot
from src.spitialself.evaluation import evaluate_predictions_auto


def run_commot_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    dis_thr: float | None = None,
    cot_eps_p: float = 0.1,
    cot_nitermax: int = 2000,
) -> dict:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")

    predictions = score_commot(
        cells,
        expression,
        lr_pairs,
        candidates,
        dis_thr=dis_thr,
        cot_eps_p=cot_eps_p,
        cot_nitermax=cot_nitermax,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{replicate_dir.name}__commot_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__commot_metrics.json"

    predictions.to_csv(prediction_path, index=False)
    metrics = evaluate_predictions_auto(candidates, predictions)
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": "commot",
            "prediction_path": str(prediction_path),
            "dis_thr": float(dis_thr) if dis_thr is not None else float(candidates["distance"].max()),
            "cot_eps_p": float(cot_eps_p),
            "cot_nitermax": int(cot_nitermax),
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
        default="outputs/phase1/exp03_baseline_comparison/commot",
        help="Directory for prediction and metric outputs.",
    )
    parser.add_argument("--dis-thr", type=float, default=None)
    parser.add_argument("--cot-eps-p", type=float, default=0.1)
    parser.add_argument("--cot-nitermax", type=int, default=2000)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    output_dir = Path(args.output_dir)
    metrics = [
        run_commot_on_replicate(
            replicate_dir,
            output_dir,
            dis_thr=args.dis_thr,
            cot_eps_p=args.cot_eps_p,
            cot_nitermax=args.cot_nitermax,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / "commot_metrics.csv", index=False)
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
    print(f"Wrote COMMOT outputs to {output_dir}")


if __name__ == "__main__":
    main()
