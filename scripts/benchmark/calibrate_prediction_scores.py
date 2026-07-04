#!/usr/bin/env python
"""Apply LR-wise score calibration to candidate-level prediction tables."""

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

from src.spitialself.evaluation import evaluate_ground_truth_predictions


TRANSFORMS = (
    "raw",
    "lr_minmax",
    "lr_rank",
    "lr_zscore_minmax",
    "mechanism_rank",
    "lr_mechanism_distance_rank",
    "lr_mechanism_prior_rank",
)


def calibrate_prediction_scores(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    transforms: tuple[str, ...],
    score_column: str = "score",
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "candidate_id" not in predictions.columns or score_column not in predictions.columns:
        raise ValueError(f"Prediction table must contain candidate_id and {score_column}")
    if "candidate_id" not in candidates.columns or "lr_id" not in candidates.columns:
        raise ValueError("Candidate table must contain candidate_id and lr_id")

    base_method = (
        str(predictions["method"].iloc[0])
        if "method" in predictions.columns and len(predictions)
        else prediction_path.stem
    )
    candidate_columns = ["candidate_id", "lr_id"]
    for column in (
        "mechanism",
        "distance",
        "sender_niche",
        "receiver_niche",
        "type_rule_satisfied",
        "range_rule_satisfied",
        "niche_rule_satisfied",
    ):
        if column in candidates.columns:
            candidate_columns.append(column)
    merged = candidates[candidate_columns].merge(
        predictions[["candidate_id", score_column]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    score_by_candidate = merged.set_index("candidate_id")[score_column].astype(float)

    rows = []
    for transform in transforms:
        calibrated = _calibrated_scores(merged, score_column=score_column, transform=transform)
        out = pd.DataFrame(
            {
                "candidate_id": merged["candidate_id"].to_numpy(),
                "method": f"{base_method}__{transform}",
                "score": calibrated,
            }
        )
        # Preserve candidates that were in the original prediction but absent from
        # the candidate merge only if that ever happens; normal benchmark tables
        # should be one-to-one.
        if len(out) != len(predictions):
            missing = predictions.loc[~predictions["candidate_id"].isin(out["candidate_id"])]
            if len(missing):
                fallback = pd.DataFrame(
                    {
                        "candidate_id": missing["candidate_id"].to_numpy(),
                        "method": f"{base_method}__{transform}",
                        "score": missing["candidate_id"].map(score_by_candidate).fillna(0.0).to_numpy(),
                    }
                )
                out = pd.concat([out, fallback], ignore_index=True)
        prediction_out = output_dir / f"{prediction_path.stem}__{transform}_predictions.csv"
        metrics_out = output_dir / f"{prediction_path.stem}__{transform}_metrics.json"
        out.to_csv(prediction_out, index=False)
        if "ground_truth" in candidates.columns:
            metrics = evaluate_ground_truth_predictions(candidates, out)
        else:
            metrics = {
                "auroc": float("nan"),
                "aupr": float("nan"),
                "top_k_precision": float("nan"),
                "weak_signal_recall_at_threshold": float("nan"),
                "empirical_fdr_at_threshold": float("nan"),
                "false_positive_high_expression_rate_at_threshold": float("nan"),
                "calibration_error_10bin": float("nan"),
                "truth_diagnostics_note": "ground_truth_column_absent",
            }
        metrics.update(
            {
                "method": f"{base_method}__{transform}",
                "base_method": base_method,
                "score_transform": transform,
                "candidate_path": str(candidate_path),
                "input_prediction_path": str(prediction_path),
                "prediction_path": str(prediction_out),
            }
        )
        with metrics_out.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=True)
        rows.append(metrics)

    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(output_dir / f"{prediction_path.stem}__calibration_metrics.csv", index=False)
    return metrics_table


def _calibrated_scores(
    table: pd.DataFrame,
    *,
    score_column: str,
    transform: str,
) -> np.ndarray:
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown score transform {transform}. Known: {TRANSFORMS}")
    scores = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0).astype(float)
    if transform == "raw":
        return scores.to_numpy(dtype=float)

    out = np.zeros(len(table), dtype=float)
    if transform in {"mechanism_rank", "lr_mechanism_distance_rank", "lr_mechanism_prior_rank"}:
        group_column = "mechanism" if transform == "mechanism_rank" else "lr_id"
        if group_column not in table.columns:
            group_column = "lr_id"
        for _, idx in table.groupby(group_column, sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            values = scores.iloc[idx].to_numpy(dtype=float)
            ranked = _percentile_rank(values)
            if transform == "lr_mechanism_distance_rank":
                ranked = ranked * _mechanism_distance_prior(table.iloc[idx], use_oracle_rules=False)
            elif transform == "lr_mechanism_prior_rank":
                ranked = ranked * _mechanism_distance_prior(table.iloc[idx], use_oracle_rules=True)
            out[idx] = ranked
        return out

    for _, idx in table.groupby("lr_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        values = scores.iloc[idx].to_numpy(dtype=float)
        if transform == "lr_minmax":
            out[idx] = _minmax(values)
        elif transform == "lr_rank":
            out[idx] = _percentile_rank(values)
        elif transform == "lr_zscore_minmax":
            std = float(np.std(values))
            if std <= 1e-12:
                out[idx] = 0.0
            else:
                out[idx] = _minmax((values - float(np.mean(values))) / std)
    return out


def _mechanism_distance_prior(table: pd.DataFrame, *, use_oracle_rules: bool) -> np.ndarray:
    prior = np.ones(len(table), dtype=float)
    if use_oracle_rules:
        for column in ("type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied"):
            if column in table.columns:
                prior *= np.where(table[column].astype(bool).to_numpy(), 1.0, 0.25)
    if "mechanism" in table.columns and "distance" in table.columns:
        distance = table["distance"].to_numpy(dtype=float)
        mechanism = table["mechanism"].astype(str).to_numpy()
        for value in np.unique(mechanism):
            mask = mechanism == value
            if not mask.any():
                continue
            scale = max(float(np.quantile(distance[mask], 0.75)), 1e-6)
            if value == "long_range_chemokine_niche":
                ranked_distance = _percentile_rank(distance[mask])
                prior[mask] *= 0.45 + 0.55 * ranked_distance
            else:
                prior[mask] *= np.exp(-distance[mask] / scale)
    if not use_oracle_rules and {"sender_niche", "receiver_niche"}.issubset(table.columns):
        same_niche = (
            table["sender_niche"].astype(str).to_numpy()
            == table["receiver_niche"].astype(str).to_numpy()
        )
        prior *= np.where(same_niche, 1.0, 0.75)
    return np.clip(prior, 0.0, 1.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks / max(len(values) - 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--transform",
        action="append",
        dest="transforms",
        choices=TRANSFORMS,
        help="Score transform to apply. Repeatable. Defaults to all transforms.",
    )
    parser.add_argument("--score-column", default="score")
    args = parser.parse_args()

    transforms = tuple(args.transforms or TRANSFORMS)
    metrics = calibrate_prediction_scores(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.output_dir),
        transforms=transforms,
        score_column=args.score_column,
    )
    columns = [
        "method",
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
    ]
    print(metrics[columns].to_string(index=False))
    print(f"Wrote calibrated predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
