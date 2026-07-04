#!/usr/bin/env python
"""Project DeepTalk prediction outputs to benchmark candidate edges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.prepare_deeptalk_inputs import prepare_replicate
from src.spitialself.evaluation import evaluate_ground_truth_predictions


def run_deeptalk_projection_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    deeptalk_input_root: Path | None = None,
) -> dict:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = deeptalk_input_root or (output_dir / "deeptalk_inputs")
    replicate_input_dir = input_root / replicate_dir.name
    if not replicate_input_dir.exists():
        prepare_replicate(replicate_dir, input_root)

    predictions, status = project_deeptalk_outputs_to_candidates(candidates, replicate_input_dir)
    prediction_path = output_dir / f"{replicate_dir.name}__deeptalk_predictions.csv"
    status_path = output_dir / f"{replicate_dir.name}__deeptalk_status.csv"
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(status).to_csv(status_path, index=False)

    metrics = {
        "replicate": replicate_dir.name,
        "method": "deeptalk",
        "prediction_path": str(prediction_path),
        "status_path": str(status_path),
        "n_projected_candidates": int((predictions["score"] > 0).sum()),
        "n_candidates": int(len(predictions)),
    }
    if metrics["n_projected_candidates"] > 0:
        metrics.update(evaluate_ground_truth_predictions(candidates, predictions))
        metrics["status"] = "completed_from_existing_predict_ccc"
    else:
        metrics.update(
            {
                "status": "model_required",
                "auroc": np.nan,
                "aupr": np.nan,
                "top_k_precision": np.nan,
                "weak_signal_recall_at_threshold": np.nan,
                "empirical_fdr_at_threshold": np.nan,
            }
        )

    metrics_path = output_dir / f"{replicate_dir.name}__deeptalk_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def project_deeptalk_outputs_to_candidates(
    candidates: pd.DataFrame,
    replicate_input_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    scores_by_candidate: dict[str, float] = {}
    status = []
    data_dir = replicate_input_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"DeepTalk data directory not found: {data_dir}")

    for lr_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        map_path = lr_dir / "candidate_map.csv"
        predict_path = lr_dir / "predict_ccc.txt"
        if not map_path.exists():
            status.append({"lr_dir": str(lr_dir), "status": "missing_candidate_map"})
            continue
        candidate_map = pd.read_csv(map_path)
        if not predict_path.exists():
            status.append(
                {
                    "lr_dir": str(lr_dir),
                    "lr_id": str(candidate_map["lr_id"].iloc[0]) if len(candidate_map) else "",
                    "status": "model_required_missing_predict_ccc",
                    "n_candidates": int(len(candidate_map)),
                }
            )
            continue

        predict = _read_deeptalk_predict(predict_path)
        joined = candidate_map.merge(
            predict,
            on=["line_index"],
            how="left",
            validate="one_to_one",
        )
        joined["score"] = joined["score"].fillna(0.0).astype(float)
        for row in joined.itertuples(index=False):
            scores_by_candidate[str(row.candidate_id)] = float(row.score)
        status.append(
            {
                "lr_dir": str(lr_dir),
                "lr_id": str(candidate_map["lr_id"].iloc[0]) if len(candidate_map) else "",
                "status": "projected",
                "n_candidates": int(len(candidate_map)),
                "n_predicted_rows": int(len(predict)),
            }
        )

    score = candidates["candidate_id"].astype(str).map(scores_by_candidate).fillna(0.0).to_numpy(dtype=float)
    predictions = pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "method": "deeptalk",
            "score": score,
        }
    )
    return predictions, status


def _read_deeptalk_predict(predict_path: Path) -> pd.DataFrame:
    rows = []
    with predict_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            try:
                score = float(tokens[3])
            except ValueError:
                score = 0.0
            rows.append({"line_index": line_index, "score": score})
    return pd.DataFrame(rows)


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/deeptalk",
    )
    parser.add_argument(
        "--deeptalk-input-root",
        default="",
        help="Existing DeepTalk input root containing replicate/data/* directories.",
    )
    args = parser.parse_args()

    replicate_dirs = _replicate_dirs(Path(args.dataset_root))
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {args.dataset_root}")
    input_root = Path(args.deeptalk_input_root) if args.deeptalk_input_root else None

    metrics = [
        run_deeptalk_projection_on_replicate(
            replicate_dir,
            Path(args.output_dir),
            deeptalk_input_root=input_root,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(Path(args.output_dir) / "deeptalk_metrics.csv", index=False)
    print(
        metrics_table[
            [
                "replicate",
                "method",
                "status",
                "n_projected_candidates",
                "n_candidates",
                "auroc",
                "aupr",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote DeepTalk outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
