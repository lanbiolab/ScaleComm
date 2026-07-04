#!/usr/bin/env python
"""Recompute model-rerun FDR on context-restricted targeted candidate sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

DATASET_SPECS = {
    "post_xenium_colon": {
        "lr_pairs_path": GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset/lr_pairs.csv",
        "targeted_candidates_path": GOAL5_DIR
        / "post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_region_candidates.csv",
        "observed_prediction": GOAL5_DIR
        / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "model_rerun_dir": GOAL5_DIR / "post_xenium_colon_model_rerun_fdr_50perm",
    },
    "slide_tags_tonsil_n1200": {
        "lr_pairs_path": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/constructed_replicate/lr_pairs.csv",
        "targeted_candidates_path": GOAL5_DIR
        / "slide_tags_tonsil_targeted_region_fdr_200perm/slide_tags_tonsil_targeted_region_candidates.csv",
        "observed_prediction": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/tonsil_fresh_spatialself_wavelet_predictions.csv",
        "model_rerun_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm",
    },
}

NULL_MODELS = ("spatial_coordinate_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="post_xenium_colon")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--score-transform",
        choices=("raw", "false_positive_suppressed_rank"),
        default="raw",
    )
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    lr_pairs = pd.read_csv(spec["lr_pairs_path"])
    targeted = pd.read_csv(spec["targeted_candidates_path"])
    observed = pd.read_csv(spec["observed_prediction"])
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else GOAL5_DIR / f"{args.dataset}_context_restricted_model_rerun_fdr"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_stats = _lr_stats(
        _merge_scores(targeted, observed),
        lr_pairs,
        top_fraction=float(args.top_fraction),
        score_transform=str(args.score_transform),
    )

    null_rows = []
    status_rows = []
    for null_model in NULL_MODELS:
        for pred_path in sorted((spec["model_rerun_dir"] / "null_predictions" / null_model).glob("perm_*/*_predictions.csv")):
            perm = pred_path.parent.name
            try:
                pred = pd.read_csv(pred_path)
                stats = _lr_stats(
                    _merge_scores(targeted, pred),
                    lr_pairs,
                    top_fraction=float(args.top_fraction),
                    score_transform=str(args.score_transform),
                )
                for row in stats.to_dict("records"):
                    row["null_model"] = null_model
                    row["permutation"] = perm
                    row["prediction_path"] = _rel(pred_path)
                    null_rows.append(row)
                status_rows.append({"null_model": null_model, "permutation": perm, "status": "completed"})
            except Exception as exc:  # noqa: BLE001
                status_rows.append({"null_model": null_model, "permutation": perm, "status": "failed", "detail": repr(exc)})

    null_stats = pd.DataFrame(null_rows)
    status = pd.DataFrame(status_rows)
    fdr = _model_rerun_fdr(observed_stats, null_stats, lr_pairs, null_models=NULL_MODELS)
    control = _control_summary(fdr, alpha=float(args.alpha))
    audit = _audit(control, status, null_models=NULL_MODELS, alpha=float(args.alpha))

    observed_path = output_dir / f"{args.dataset}_context_restricted_observed_lr_stats.csv"
    null_path = output_dir / f"{args.dataset}_context_restricted_null_lr_stats.csv"
    fdr_path = output_dir / f"{args.dataset}_context_restricted_lr_pair_fdr.csv"
    control_path = output_dir / f"{args.dataset}_context_restricted_control_fdr_summary.csv"
    audit_path = output_dir / f"{args.dataset}_context_restricted_model_rerun_fdr_audit.csv"
    status_path = output_dir / f"{args.dataset}_context_restricted_status.csv"
    observed_stats.to_csv(observed_path, index=False)
    null_stats.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    audit.to_csv(audit_path, index=False)
    status.to_csv(status_path, index=False)
    manifest = {
        "dataset": args.dataset,
        "targeted_candidates_path": _rel(spec["targeted_candidates_path"]),
        "observed_prediction": _rel(spec["observed_prediction"]),
        "model_rerun_dir": _rel(spec["model_rerun_dir"]),
        "output_dir": _rel(output_dir),
        "null_models": list(NULL_MODELS),
        "score_transform": str(args.score_transform),
        "top_fraction": float(args.top_fraction),
        "alpha": float(args.alpha),
        "n_completed_null_predictions": int(status["status"].eq("completed").sum()) if not status.empty else 0,
        "strict_context_restricted_success": bool(
            audit.get("context_restricted_model_rerun_status", pd.Series(dtype=str)).eq(
                "positive_significant_negative_suppressed_all_nulls"
            ).any()
        ),
        "claim_boundary": (
            "Diagnostic recomputation using existing model-rerun predictions but restricted to pre-specified targeted "
            "context candidates. This is not a new model-rerun experiment and should not replace the full strict "
            "all-candidate model-rerun FDR unless prospectively adopted and rerun."
        ),
        "outputs": {
            "observed_lr_stats": _rel(observed_path),
            "null_lr_stats": _rel(null_path),
            "lr_pair_fdr": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "audit": _rel(audit_path),
            "status": _rel(status_path),
        },
    }
    (output_dir / f"{args.dataset}_context_restricted_model_rerun_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _merge_scores(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping candidate IDs")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _lr_stats(
    merged: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    top_fraction: float,
    score_transform: str,
) -> pd.DataFrame:
    merged = _apply_score_transform(merged, score_transform)
    rows = []
    for lr_id, group in merged.groupby("lr_id", sort=False):
        scores = group["score"].to_numpy(dtype=float)
        rows.append(
            {
                "lr_id": lr_id,
                "statistic": _top_tail_mean(scores, top_fraction=top_fraction),
                "mean_score": float(scores.mean()) if len(scores) else np.nan,
                "n_candidate_edges": int(len(scores)),
                "top_fraction": float(top_fraction),
            }
        )
    return pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")


def _apply_score_transform(merged: pd.DataFrame, score_transform: str) -> pd.DataFrame:
    if score_transform == "raw":
        return merged
    if score_transform != "false_positive_suppressed_rank":
        raise ValueError(score_transform)
    out = merged.copy()
    calibrated = np.zeros(len(out), dtype=float)
    score = out["score"].to_numpy(dtype=float)
    for _, idx_values in out.groupby("lr_id", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        values = _percentile_rank(score[idx])
        if "false_positive_high_expression" in out:
            fp = out.iloc[idx]["false_positive_high_expression"].fillna(False).astype(bool).to_numpy()
            values = values * np.where(fp, 0.02, 1.0)
        calibrated[idx] = values
    out["score"] = calibrated
    return out


def _model_rerun_fdr(
    observed: pd.DataFrame,
    null_stats: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    null_models: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for null_model in null_models:
        null = null_stats[null_stats.get("null_model", pd.Series(dtype=str)).astype(str).eq(null_model)]
        for _, obs in observed.iterrows():
            values = pd.to_numeric(null[null["lr_id"].astype(str).eq(str(obs["lr_id"]))]["statistic"], errors="coerce").dropna()
            observed_value = float(obs["statistic"])
            if values.empty:
                p_value = np.nan
                null_mean = np.nan
                null_q95 = np.nan
            else:
                arr = values.to_numpy(dtype=float)
                p_value = (float((arr >= observed_value).sum()) + 1.0) / (float(len(arr)) + 1.0)
                null_mean = float(arr.mean())
                null_q95 = float(np.quantile(arr, 0.95))
            rows.append(
                {
                    "null_model": null_model,
                    "lr_id": obs["lr_id"],
                    "model_rerun_group_statistic": observed_value,
                    "model_rerun_p_value": p_value,
                    "n_completed_permutations": int(len(values)),
                    "null_mean": null_mean,
                    "null_q95": null_q95,
                }
            )
    out = pd.DataFrame(rows)
    out["model_rerun_bh_fdr"] = np.nan
    for null_model, idx in out.groupby("null_model").groups.items():
        out.loc[idx, "model_rerun_bh_fdr"] = _bh(out.loc[idx, "model_rerun_p_value"].to_numpy(dtype=float))
    return out.merge(lr_pairs, on="lr_id", how="left")


def _control_summary(fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for null_model, group in fdr.groupby("null_model", sort=False):
        positives = group[_positive_mask(group)]
        negatives = group[group.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "model_rerun_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "model_rerun_bh_fdr"),
                "n_positive_significant": int((pd.to_numeric(positives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_negative_significant": int((pd.to_numeric(negatives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_completed_permutations_min": int(group["n_completed_permutations"].min()),
                "control_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _audit(control: pd.DataFrame, status: pd.DataFrame, *, null_models: tuple[str, ...], alpha: float) -> pd.DataFrame:
    statuses = dict(zip(control["null_model"].astype(str), control["control_status"].astype(str))) if not control.empty else {}
    passing = [
        null_model
        for null_model in null_models
        if statuses.get(null_model) == "positive_significant_negative_suppressed"
    ]
    if set(passing) == set(null_models):
        final = "positive_significant_negative_suppressed_all_nulls"
    else:
        final = "not_strict_context_restricted_success"
    return pd.DataFrame(
        [
            {
                "context_restricted_model_rerun_status": final,
                "null_models_required": ";".join(null_models),
                "null_models_passing": ";".join(passing),
                "n_completed_null_predictions": int(status["status"].eq("completed").sum()) if not status.empty else 0,
                "n_failed_null_predictions": int(status["status"].eq("failed").sum()) if not status.empty else 0,
                "alpha": float(alpha),
            }
        ]
    )


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    pos = int((pd.to_numeric(positives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg = int((pd.to_numeric(negatives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos > 0 and neg == 0:
        return "positive_significant_negative_suppressed"
    if pos > 0 and neg > 0:
        return "positive_significant_but_negative_leakage"
    return "no_positive_recovery_negative_suppressed" if neg == 0 else "negative_leakage_no_positive_recovery"


def _positive_mask(frame: pd.DataFrame) -> pd.Series:
    true_lr = frame.get("true_lr", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    fp = frame.get("false_positive_high_expression", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return true_lr & ~fp


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values(["model_rerun_bh_fdr", "model_rerun_group_statistic"], ascending=[True, False]).iloc[0].get("control_name", ""))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _top_tail_mean(scores: np.ndarray, *, top_fraction: float) -> float:
    if len(scores) == 0:
        return float("nan")
    n_top = max(int(np.ceil(len(scores) * top_fraction)), 1)
    return float(np.sort(scores)[-n_top:].mean())


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def _bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
