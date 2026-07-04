#!/usr/bin/env python
"""Run lightweight methods and summarize weak-signal/false-positive stress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
from scripts.benchmark.run_weak_signal_false_positive import summarize_stress


def run_stress_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    spatialself_epochs: int,
    spatialself_max_train_samples: int,
    spatialself_k_nn: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_weak_positive_fraction: float,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    spatialself_lr_cap_positive: int,
    spatialself_lr_cap_negative: int,
) -> pd.DataFrame:
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    status = []
    for method in methods:
        method_dir = output_dir / method
        for replicate_dir in replicate_dirs:
            try:
                prediction_path = _run_or_find_predictions(
                    method,
                    replicate_dir,
                    method_dir,
                    spatialself_epochs=spatialself_epochs,
                    spatialself_max_train_samples=spatialself_max_train_samples,
                    spatialself_k_nn=spatialself_k_nn,
                    spatialself_pseudo_label_strategy=spatialself_pseudo_label_strategy,
                    spatialself_weak_positive_fraction=spatialself_weak_positive_fraction,
                    spatialself_ranking_loss_weight=spatialself_ranking_loss_weight,
                    spatialself_ranking_margin=spatialself_ranking_margin,
                    spatialself_lr_cap_positive=spatialself_lr_cap_positive,
                    spatialself_lr_cap_negative=spatialself_lr_cap_negative,
                )
                metrics = summarize_stress(
                    replicate_dir / "candidate_interactions.csv",
                    prediction_path,
                    output_dir / "stress_tables",
                )
                rows.append(metrics)
                status.append(
                    {
                        "replicate": replicate_dir.name,
                        "method": method,
                        "status": "completed",
                        "prediction_path": str(prediction_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - status table should capture per-method failures.
                status.append(
                    {
                        "replicate": replicate_dir.name,
                        "method": method,
                        "status": "failed",
                        "detail": str(exc),
                    }
                )

    metrics_table = pd.DataFrame(rows)
    if not metrics_table.empty:
        metrics_table.to_csv(output_dir / "weak_signal_false_positive_metrics.csv", index=False)
    pd.DataFrame(status).to_csv(output_dir / "weak_signal_false_positive_status.csv", index=False)
    return metrics_table


def _run_or_find_predictions(
    method: str,
    replicate_dir: Path,
    output_dir: Path,
    *,
    spatialself_epochs: int,
    spatialself_max_train_samples: int,
    spatialself_k_nn: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_weak_positive_fraction: float,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    spatialself_lr_cap_positive: int,
    spatialself_lr_cap_negative: int,
) -> Path:
    if method in {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}:
        is_edge_prior_candidate = method == "spatialself_wavelet_edge_prior"
        model_name = "spatialself_wavelet" if is_edge_prior_candidate else method
        metrics = run_spatialself_on_replicate(
            replicate_dir,
            output_dir,
            model_name=model_name,
            epochs=spatialself_epochs,
            max_train_samples=spatialself_max_train_samples,
            k_nn=spatialself_k_nn,
            pseudo_label_strategy=spatialself_pseudo_label_strategy,
            weak_positive_fraction=spatialself_weak_positive_fraction,
            ranking_loss_weight=spatialself_ranking_loss_weight,
            ranking_margin=spatialself_ranking_margin,
            lr_cap_positive=spatialself_lr_cap_positive,
            lr_cap_negative=spatialself_lr_cap_negative,
            use_edge_prior_features=is_edge_prior_candidate,
        )
        prediction_path = Path(metrics["prediction_path"])
        if is_edge_prior_candidate:
            prediction = pd.read_csv(prediction_path)
            prediction["method"] = method
            prediction.to_csv(prediction_path, index=False)
        return prediction_path
    prediction_path = output_dir / f"{replicate_dir.name}__{method}_predictions.csv"
    if prediction_path.exists():
        return prediction_path
    raise ValueError(f"Method {method} is not runnable here and no prediction file exists at {prediction_path}")


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp05_weak_signal_false_positive",
    )
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--spatialself-epochs", type=int, default=5)
    parser.add_argument("--spatialself-max-train-samples", type=int, default=4000)
    parser.add_argument("--spatialself-k-nn", type=int, default=12)
    parser.add_argument("--spatialself-pseudo-label-strategy", default="expression_distance")
    parser.add_argument("--spatialself-weak-positive-fraction", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-margin", type=float, default=1.0)
    parser.add_argument("--spatialself-lr-cap-positive", type=int, default=50)
    parser.add_argument("--spatialself-lr-cap-negative", type=int, default=200)
    args = parser.parse_args()

    methods = tuple(args.methods or ("spatialself_wavelet", "spatialself_gat"))
    metrics = run_stress_dataset(
        Path(args.dataset_root),
        Path(args.output_dir),
        methods=methods,
        spatialself_epochs=args.spatialself_epochs,
        spatialself_max_train_samples=args.spatialself_max_train_samples,
        spatialself_k_nn=args.spatialself_k_nn,
        spatialself_pseudo_label_strategy=args.spatialself_pseudo_label_strategy,
        spatialself_weak_positive_fraction=args.spatialself_weak_positive_fraction,
        spatialself_ranking_loss_weight=args.spatialself_ranking_loss_weight,
        spatialself_ranking_margin=args.spatialself_ranking_margin,
        spatialself_lr_cap_positive=args.spatialself_lr_cap_positive,
        spatialself_lr_cap_negative=args.spatialself_lr_cap_negative,
    )
    if metrics.empty:
        print("No stress metrics were generated.")
    else:
        print(
            metrics[
                [
                    "replicate",
                    "method",
                    "auroc",
                    "aupr",
                    "weak_signal_recall_at_threshold",
                    "empirical_fdr_at_threshold",
                    "false_positive_high_expression_rate_at_threshold",
                ]
            ].to_string(index=False)
        )
    print(f"Wrote weak-signal/false-positive stress outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
