#!/usr/bin/env python
"""Re-score existing model-rerun predictions with spatial-background calibration.

This diagnostic reuses observed and null SpatialSelf predictions from an
existing model-rerun FDR run. It does not retrain the model. The goal is to test
whether post-hoc calibration against edge/spatial background reduces generic
high-expression leakage while preserving LR-specific context signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase7.run_goal5_model_rerun_fdr import (  # noqa: E402
    DATASET_SPECS,
    GOAL5_DIR,
    NULL_MODELS,
    _audit,
    _bh_adjust,
    _control_summary,
    _lookup_expression,
    _model_rerun_fdr,
    _niche_rule,
    _percentile_rank,
    _refresh_candidate_metadata,
    _rel,
    _top_tail_mean,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="slide_tags_tonsil_n1200")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--observed-prediction", default="")
    parser.add_argument("--source-rerun-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--null-model", action="append", choices=NULL_MODELS)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--calibration",
        choices=(
            "lr_rank",
            "edge_background_residual_rank",
            "edge_distance_background_residual_rank",
            "edge_niche_background_residual_rank",
            "edge_background_residual",
            "edge_distance_background_residual",
            "edge_background_product",
            "edge_distance_background_product",
        ),
        default="edge_background_residual_rank",
    )
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--n-distance-bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(spec["dataset_dir"])
    observed_prediction = (
        Path(args.observed_prediction) if args.observed_prediction else Path(spec["observed_prediction"])
    )
    source_rerun_dir = Path(args.source_rerun_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    null_models = tuple(args.null_model) if args.null_model else NULL_MODELS

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    observed_predictions = pd.read_csv(observed_prediction)
    status = _load_status(source_rerun_dir, args.dataset)

    observed_stats = _lr_stats_calibrated(
        _merge_scores(candidates, observed_predictions),
        lr_pairs,
        top_fraction=float(args.top_fraction),
        calibration=str(args.calibration),
        n_distance_bins=int(args.n_distance_bins),
        background_weight=float(args.background_weight),
    )

    null_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for null_model in null_models:
        model_status = status[
            status["null_model"].astype(str).eq(null_model)
            & status["status"].astype(str).eq("completed")
        ].copy()
        model_status = model_status.sort_values("permutation").head(int(args.n_permutations))
        for _, row in model_status.iterrows():
            perm_idx = int(row["permutation"])
            seed = int(row["seed"])
            pred_path = (
                source_rerun_dir
                / "null_predictions"
                / null_model
                / f"perm_{perm_idx:04d}"
                / f"perm_{perm_idx:04d}__spatialself_wavelet_predictions.csv"
            )
            try:
                if not pred_path.exists():
                    raise FileNotFoundError(pred_path)
                null_candidates = _make_null_candidates(
                    cells,
                    expression,
                    lr_pairs,
                    candidates,
                    null_model=null_model,
                    seed=seed,
                )
                null_predictions = pd.read_csv(pred_path)
                stats = _lr_stats_calibrated(
                    _merge_scores(null_candidates, null_predictions),
                    lr_pairs,
                    top_fraction=float(args.top_fraction),
                    calibration=str(args.calibration),
                    n_distance_bins=int(args.n_distance_bins),
                    background_weight=float(args.background_weight),
                )
                for stat in stats.to_dict("records"):
                    stat.update(
                        {
                            "null_model": null_model,
                            "permutation": perm_idx,
                            "seed": seed,
                            "prediction_path": _rel(pred_path),
                        }
                    )
                    null_rows.append(stat)
                status_rows.append(
                    {
                        "null_model": null_model,
                        "permutation": perm_idx,
                        "seed": seed,
                        "status": "completed",
                        "detail": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - long-run status capture.
                status_rows.append(
                    {
                        "null_model": null_model,
                        "permutation": perm_idx,
                        "seed": seed,
                        "status": "failed",
                        "detail": repr(exc),
                    }
                )

    null_stats = pd.DataFrame(null_rows)
    calibrated_status = pd.DataFrame(status_rows)
    fdr = _model_rerun_fdr(
        observed_stats,
        null_stats,
        lr_pairs,
        null_models=null_models,
        top_fraction=float(args.top_fraction),
    )
    control = _control_summary(fdr, alpha=float(args.alpha))
    audit = _audit(control, calibrated_status, null_models=null_models, alpha=float(args.alpha))
    diagnostic = _diagnostic_summary(control, audit, fdr, alpha=float(args.alpha))

    prefix = f"{args.dataset}_spatial_background_calibrated"
    observed_path = output_dir / f"{prefix}_observed_lr_stats.csv"
    null_path = output_dir / f"{prefix}_null_lr_stats.csv"
    fdr_path = output_dir / f"{prefix}_lr_pair_fdr.csv"
    control_path = output_dir / f"{prefix}_control_fdr_summary.csv"
    status_path = output_dir / f"{prefix}_status.csv"
    audit_path = output_dir / f"{prefix}_fdr_audit.csv"
    diagnostic_path = output_dir / f"{prefix}_diagnostic_summary.csv"
    manifest_path = output_dir / f"{prefix}_manifest.json"

    observed_stats.to_csv(observed_path, index=False)
    null_stats.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    calibrated_status.to_csv(status_path, index=False)
    audit.to_csv(audit_path, index=False)
    diagnostic.to_csv(diagnostic_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "dataset_dir": _rel(dataset_dir),
        "observed_prediction": _rel(observed_prediction),
        "source_rerun_dir": _rel(source_rerun_dir),
        "output_dir": _rel(output_dir),
        "null_models": list(null_models),
        "n_permutations_requested_per_null": int(args.n_permutations),
        "n_permutations_completed": int(calibrated_status["status"].eq("completed").sum())
        if not calibrated_status.empty
        else 0,
        "top_fraction": float(args.top_fraction),
        "calibration": str(args.calibration),
        "n_distance_bins": int(args.n_distance_bins),
        "background_weight": float(args.background_weight),
        "alpha": float(args.alpha),
        "claim_boundary": (
            "This is a post-hoc calibration diagnostic on existing model-rerun predictions, not a new model "
            "training run. Passing results should be confirmed with higher-permutation model-rerun FDR."
        ),
        "p_value_resolution_note": (
            "With 20 completed permutations, the minimum empirical p-value is 1/21=0.0476. "
            "BH FDR over 45 LR tests cannot strictly pass alpha=0.05 while all three negative controls are "
            "suppressed unless additional permutations or a smaller pre-specified test family is used."
        ),
        "outputs": {
            "observed_lr_stats": _rel(observed_path),
            "null_lr_stats": _rel(null_path),
            "lr_pair_fdr": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "status": _rel(status_path),
            "audit": _rel(audit_path),
            "diagnostic_summary": _rel(diagnostic_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(diagnostic.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _load_status(source_rerun_dir: Path, dataset: str) -> pd.DataFrame:
    path = source_rerun_dir / f"{dataset}_model_rerun_status.csv"
    if not path.exists():
        matches = sorted(source_rerun_dir.glob("*_model_rerun_status.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not find unique model-rerun status CSV in {source_rerun_dir}")
        path = matches[0]
    return pd.read_csv(path)


def _make_null_candidates(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    null_model: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    null_cells = cells.copy()
    null_candidates = candidates.copy()
    if null_model == "spatial_coordinate_permutation":
        perm = rng.permutation(len(null_cells))
        null_cells[["x", "y"]] = null_cells[["x", "y"]].to_numpy(dtype=float)[perm]
    elif null_model == "receiver_permutation":
        null_candidates["receiver_id"] = rng.permutation(null_candidates["receiver_id"].to_numpy())
    elif null_model == "lr_identity_permutation":
        null_candidates["lr_id"] = rng.permutation(null_candidates["lr_id"].to_numpy())
    else:
        raise ValueError(f"Unsupported null model: {null_model}")
    return _refresh_candidate_metadata(null_candidates, null_cells, expression, lr_pairs)


def _merge_scores(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions.")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _lr_stats_calibrated(
    merged: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    top_fraction: float,
    calibration: str,
    n_distance_bins: int,
    background_weight: float,
) -> pd.DataFrame:
    calibrated = _apply_spatial_background_calibration(
        merged,
        calibration=calibration,
        n_distance_bins=n_distance_bins,
        background_weight=background_weight,
    )
    rows = []
    for lr_id, group in calibrated.groupby("lr_id", sort=False):
        scores = group["score"].to_numpy(dtype=float)
        rows.append(
            {
                "lr_id": lr_id,
                "observed_or_null_statistic": _top_tail_mean(scores, top_fraction=top_fraction),
                "mean_score": float(np.mean(scores)) if len(scores) else np.nan,
                "max_score": float(np.max(scores)) if len(scores) else np.nan,
                "n_candidate_edges": int(len(scores)),
                "top_fraction": float(top_fraction),
            }
        )
    stats = pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")
    return stats.sort_values("observed_or_null_statistic", ascending=False)


def _apply_spatial_background_calibration(
    merged: pd.DataFrame,
    *,
    calibration: str,
    n_distance_bins: int,
    background_weight: float,
) -> pd.DataFrame:
    out = merged.copy()
    score = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lr_rank = np.zeros(len(out), dtype=float)
    for _, idx_values in out.groupby("lr_id", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        lr_rank[idx] = _percentile_rank(score[idx])
    if calibration == "lr_rank":
        out["score"] = lr_rank
        return out

    edge_key = out["sender_id"].astype(str) + "||" + out["receiver_id"].astype(str)
    background = pd.Series(lr_rank, index=out.index).groupby(edge_key, sort=False).transform("median").to_numpy(
        dtype=float
    )

    if calibration in {
        "edge_distance_background_residual_rank",
        "edge_niche_background_residual_rank",
        "edge_distance_background_residual",
        "edge_distance_background_product",
    }:
        distance_bin = _distance_bins(out["distance"], n_bins=n_distance_bins)
        distance_background = (
            pd.Series(lr_rank, index=out.index)
            .groupby([out["lr_id"].astype(str), distance_bin], sort=False)
            .transform("median")
            .to_numpy(dtype=float)
        )
        background = 0.5 * background + 0.5 * distance_background

    if calibration == "edge_niche_background_residual_rank":
        niche_key = (
            out.get("sender_niche", pd.Series("", index=out.index)).astype(str)
            + "||"
            + out.get("receiver_niche", pd.Series("", index=out.index)).astype(str)
        )
        niche_background = (
            pd.Series(lr_rank, index=out.index)
            .groupby([out["lr_id"].astype(str), niche_key], sort=False)
            .transform("median")
            .to_numpy(dtype=float)
        )
        background = (background + niche_background) / 2.0

    weight = float(np.clip(background_weight, 0.0, 1.0))
    residual = lr_rank - weight * background
    if calibration in {"edge_background_residual", "edge_distance_background_residual"}:
        out["score"] = np.maximum(residual, 0.0)
        return out
    if calibration in {"edge_background_product", "edge_distance_background_product"}:
        out["score"] = lr_rank * np.clip(1.0 - weight * background, 0.0, 1.0)
        return out

    calibrated = np.zeros(len(out), dtype=float)
    for _, idx_values in out.groupby("lr_id", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        positive_residual = np.maximum(residual[idx], 0.0)
        calibrated[idx] = _percentile_rank(positive_residual)
    out["score"] = calibrated
    return out


def _distance_bins(distance: pd.Series, *, n_bins: int) -> pd.Series:
    values = pd.to_numeric(distance, errors="coerce").fillna(0.0)
    try:
        return pd.qcut(values, q=max(int(n_bins), 2), labels=False, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["0"] * len(values), index=distance.index)


def _diagnostic_summary(
    control: pd.DataFrame,
    audit: pd.DataFrame,
    fdr: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for null_model, group in fdr.groupby("null_model", sort=False):
        positives = group[_positive_control_mask(group)]
        negatives = group[group.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_p_value": _best_value(positives, "model_rerun_p_value"),
                "best_positive_bh_fdr": _best_value(positives, "model_rerun_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_p_value": _best_value(negatives, "model_rerun_p_value"),
                "best_negative_bh_fdr": _best_value(negatives, "model_rerun_bh_fdr"),
                "n_positive_at_min_p": int(
                    pd.to_numeric(positives.get("model_rerun_p_value", pd.Series(dtype=float)), errors="coerce")
                    .eq(_best_value(positives, "model_rerun_p_value"))
                    .sum()
                )
                if not positives.empty
                else 0,
                "n_negative_at_min_p": int(
                    pd.to_numeric(negatives.get("model_rerun_p_value", pd.Series(dtype=float)), errors="coerce")
                    .eq(_best_value(negatives, "model_rerun_p_value"))
                    .sum()
                )
                if not negatives.empty
                else 0,
                "alpha": float(alpha),
            }
        )
    return pd.DataFrame(rows)


def _positive_control_mask(frame: pd.DataFrame) -> pd.Series:
    if "true_lr" in frame:
        return frame["true_lr"].fillna(False).astype(bool) & ~frame.get(
            "false_positive_high_expression",
            pd.Series(False, index=frame.index),
        ).fillna(False).astype(bool)
    if "control_class" in frame:
        return ~frame["control_class"].astype(str).str.contains("negative", case=False, na=False)
    return pd.Series(False, index=frame.index)


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    ordered = frame.sort_values(
        ["model_rerun_bh_fdr", "model_rerun_p_value", "model_rerun_group_statistic"],
        ascending=[True, True, False],
    )
    row = ordered.iloc[0]
    return str(row.get("control_name", row.get("lr_id", "")))


def _best_value(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    return float(pd.to_numeric(frame[column], errors="coerce").min())


if __name__ == "__main__":
    main()
