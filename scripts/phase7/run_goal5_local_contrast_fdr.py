#!/usr/bin/env python
"""Run local-contrast targeted FDR for Phase 7 Goal 5 datasets.

The standard group FDR uses raw top-tail scores. On some real-platform runs,
SpatialSelf scores are saturated near one for both observed and structure-
preserving nulls. This diagnostic instead tests whether a positive LR exceeds
local high-expression negative-control background on matched sender-receiver
edges.
"""

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

from src.spitialself.fdr import bh_adjust  # noqa: E402
from src.spitialself.fdr import _null_groups, _permute_scores_by_group  # noqa: E402


GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

DATASETS = {
    "post_xenium_colon": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset",
        "prediction_path": GOAL_DIR
        / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "output_prefix": "post_xenium_colon",
        "null_models": ("within_region_permutation", "cell_type_preserving_permutation", "lr_identity_permutation"),
    },
    "spamosaic_tonsil_A1": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/spamosaic_tonsil_A1_subset",
        "prediction_path": GOAL_DIR
        / "spamosaic_tonsil_A1_spatialself/spatialself_wavelet_edge_prior/"
        "replicate_00__spatialself_wavelet_predictions.csv",
        "output_prefix": "spamosaic_tonsil_A1",
        "null_models": ("spatial_permutation", "receiver_permutation", "lr_identity_permutation"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="post_xenium_colon")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--prediction-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--null-model", action="append")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=202613)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(spec["dataset_dir"])
    prediction_path = Path(args.prediction_path) if args.prediction_path else Path(spec["prediction_path"])
    null_models = tuple(args.null_model) if args.null_model else tuple(spec["null_models"])
    output_prefix = str(spec["output_prefix"])
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else GOAL_DIR / f"{output_prefix}_local_contrast_fdr_{int(args.n_permutations)}perm"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    predictions = pd.read_csv(prediction_path)
    merged = _merge(candidates, predictions)
    targeted = _targeted_candidates(merged)
    targeted_path = output_dir / f"{output_prefix}_local_contrast_candidates.csv"
    targeted.to_csv(targeted_path, index=False)

    rows = []
    for null_model in null_models:
        table = _local_contrast_fdr(
            targeted,
            lr_pairs,
            null_model=null_model,
            n_permutations=int(args.n_permutations),
            top_fraction=float(args.top_fraction),
            seed=int(args.random_seed),
        )
        rows.append(table)
        table.to_csv(output_dir / f"{output_prefix}__{null_model}__local_contrast_fdr.csv", index=False)
    fdr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    control = _control_summary(fdr, alpha=float(args.alpha))
    audit = _audit(fdr, alpha=float(args.alpha), null_models=null_models)

    fdr_path = output_dir / f"{output_prefix}_local_contrast_fdr_all.csv"
    control_path = output_dir / f"{output_prefix}_local_contrast_control_summary.csv"
    audit_path = output_dir / f"{output_prefix}_local_contrast_audit.csv"
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    audit.to_csv(audit_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "dataset_dir": _rel(dataset_dir),
        "prediction_path": _rel(prediction_path),
        "output_dir": _rel(output_dir),
        "candidate_path": _rel(targeted_path),
        "null_models": list(null_models),
        "n_permutations": int(args.n_permutations),
        "top_fraction": float(args.top_fraction),
        "alpha": float(args.alpha),
        "strict_local_contrast_success": bool(
            audit.get("local_contrast_all_null_status", pd.Series(dtype=str)).eq(
                "positive_significant_negative_suppressed_all_nulls"
            ).any()
        ),
        "claim_boundary": (
            "Diagnostic local-contrast FDR. The statistic is positive LR top-tail score minus matched "
            "high-expression negative-control background on the same sender-receiver candidate edges. "
            "This can reduce score saturation, but it is still post-hoc association evidence, not native "
            "perturbation ground truth."
        ),
        "outputs": {
            "fdr_all": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "audit": _rel(audit_path),
        },
    }
    manifest_path = output_dir / f"{output_prefix}_local_contrast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _merge(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
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


def _targeted_candidates(merged: pd.DataFrame) -> pd.DataFrame:
    required = {"type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied", "false_positive_high_expression"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"candidate table missing required targeted columns: {sorted(missing)}")
    positive_rule = (
        merged["type_rule_satisfied"].fillna(False).astype(bool)
        & merged["range_rule_satisfied"].fillna(False).astype(bool)
        & merged["niche_rule_satisfied"].fillna(False).astype(bool)
        & ~merged["false_positive_high_expression"].fillna(False).astype(bool)
    )
    negative = merged["false_positive_high_expression"].fillna(False).astype(bool)
    out = merged[positive_rule | negative].copy()
    out["targeted_region_reason"] = "positive_rule_satisfied"
    out.loc[negative.loc[out.index].to_numpy(dtype=bool), "targeted_region_reason"] = "negative_control_all_edges"
    return out.reset_index(drop=True)


def _local_contrast_fdr(
    targeted: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    null_model: str,
    n_permutations: int,
    top_fraction: float,
    seed: int,
) -> pd.DataFrame:
    positive_pairs = lr_pairs[
        lr_pairs.get("true_lr", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool)
        & ~lr_pairs.get("false_positive_high_expression", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool)
    ].copy()
    negative_lr_ids = set(
        lr_pairs[
            lr_pairs.get("false_positive_high_expression", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool)
        ]["lr_id"].astype(str)
    )
    if positive_pairs.empty or not negative_lr_ids:
        raise ValueError("Local contrast FDR requires positive LR controls and high-expression negative controls.")

    scores = targeted["score"].to_numpy(dtype=float)
    observed = _local_contrast_stats(targeted, scores, positive_pairs, negative_lr_ids, top_fraction=top_fraction)
    groups = _null_groups(targeted, null_model)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(observed), dtype=np.int32)
    null_sum = np.zeros(len(observed), dtype=float)
    null_sum_sq = np.zeros(len(observed), dtype=float)
    for _ in range(int(n_permutations)):
        null_scores = _permute_scores_by_group(scores, groups, rng)
        null_stats = _local_contrast_stats(
            targeted,
            null_scores,
            positive_pairs,
            negative_lr_ids,
            top_fraction=top_fraction,
        )["local_contrast_statistic"].to_numpy(dtype=float)
        obs = observed["local_contrast_statistic"].to_numpy(dtype=float)
        exceed += null_stats >= obs
        null_sum += null_stats
        null_sum_sq += null_stats * null_stats

    p_values = (exceed + 1.0) / (float(n_permutations) + 1.0)
    result = observed.copy()
    result["null_model"] = null_model
    result["local_contrast_p_value"] = p_values
    result["local_contrast_bh_fdr"] = bh_adjust(p_values)
    result["null_mean"] = null_sum / max(int(n_permutations), 1)
    null_var = np.maximum(null_sum_sq / max(int(n_permutations), 1) - result["null_mean"].to_numpy() ** 2, 0.0)
    result["null_std"] = np.sqrt(null_var)
    result["n_permutations"] = int(n_permutations)
    result["top_fraction"] = float(top_fraction)
    return result.sort_values(
        ["local_contrast_bh_fdr", "local_contrast_p_value", "local_contrast_statistic"],
        ascending=[True, True, False],
    )


def _local_contrast_stats(
    targeted: pd.DataFrame,
    scores: np.ndarray,
    positive_pairs: pd.DataFrame,
    negative_lr_ids: set[str],
    *,
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    lr_values = targeted["lr_id"].astype(str).to_numpy()
    edge_key = targeted["sender_id"].astype(str).to_numpy() + "||" + targeted["receiver_id"].astype(str).to_numpy()
    negative_mask = np.array([lr_id in negative_lr_ids for lr_id in lr_values], dtype=bool)
    negative_by_edge: dict[str, list[int]] = {}
    for idx, key in enumerate(edge_key):
        if negative_mask[idx]:
            negative_by_edge.setdefault(key, []).append(idx)

    for _, lr in positive_pairs.iterrows():
        lr_id = str(lr["lr_id"])
        pos_idx = np.flatnonzero(lr_values == lr_id)
        if len(pos_idx) == 0:
            continue
        matched_neg = []
        for idx in pos_idx:
            matched_neg.extend(negative_by_edge.get(edge_key[idx], []))
        matched_neg = np.unique(np.asarray(matched_neg, dtype=int)) if matched_neg else np.array([], dtype=int)
        positive_stat = _top_tail_mean(scores[pos_idx], top_fraction=top_fraction)
        negative_stat = _top_tail_mean(scores[matched_neg], top_fraction=top_fraction) if len(matched_neg) else 0.0
        rows.append(
            {
                "lr_id": lr_id,
                "ligand": lr.get("ligand", ""),
                "receptor": lr.get("receptor", ""),
                "mechanism": lr.get("mechanism", ""),
                "control_name": lr.get("control_name", f"{lr.get('ligand', '')}-{lr.get('receptor', '')}"),
                "control_class": lr.get("control_class", ""),
                "false_positive_high_expression": False,
                "true_lr": True,
                "n_positive_edges": int(len(pos_idx)),
                "n_matched_negative_edges": int(len(matched_neg)),
                "positive_top_tail": positive_stat,
                "matched_negative_top_tail": negative_stat,
                "local_contrast_statistic": positive_stat - negative_stat,
            }
        )
    for lr_id in sorted(negative_lr_ids):
        neg_idx = np.flatnonzero(lr_values == lr_id)
        if len(neg_idx) == 0:
            continue
        meta = positive_pairs.iloc[0].copy()
        lr_meta = targeted[targeted["lr_id"].astype(str).eq(lr_id)].iloc[0]
        neg_stat = _top_tail_mean(scores[neg_idx], top_fraction=top_fraction)
        rows.append(
            {
                "lr_id": lr_id,
                "ligand": lr_meta.get("ligand", ""),
                "receptor": lr_meta.get("receptor", ""),
                "mechanism": lr_meta.get("mechanism", ""),
                "control_name": lr_meta.get("control_name", f"{lr_meta.get('ligand', '')}-{lr_meta.get('receptor', '')}"),
                "control_class": lr_meta.get("control_class", ""),
                "false_positive_high_expression": True,
                "true_lr": False,
                "n_positive_edges": int(len(neg_idx)),
                "n_matched_negative_edges": int(len(neg_idx)),
                "positive_top_tail": neg_stat,
                "matched_negative_top_tail": neg_stat,
                "local_contrast_statistic": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _top_tail_mean(values: np.ndarray, *, top_fraction: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(values.size * float(top_fraction))))
    if k >= values.size:
        return float(np.mean(values))
    return float(np.mean(np.partition(values, values.size - k)[-k:]))


def _control_summary(fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for null_model, group in fdr.groupby("null_model", sort=False):
        positives = group[group["true_lr"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_local_contrast_fdr": _best_value(positives, "local_contrast_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_local_contrast_fdr": _best_value(negatives, "local_contrast_bh_fdr"),
                "n_positive_significant": int(
                    (pd.to_numeric(positives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()
                ),
                "n_negative_significant": int(
                    (pd.to_numeric(negatives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()
                ),
                "control_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _audit(fdr: pd.DataFrame, *, alpha: float, null_models: tuple[str, ...]) -> pd.DataFrame:
    passing = []
    for null_model in null_models:
        group = fdr[fdr["null_model"].astype(str).eq(str(null_model))]
        positives = group[group["true_lr"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        pos = int((pd.to_numeric(positives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
        neg = int((pd.to_numeric(negatives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
        passing.append(pos > 0 and neg == 0)
    return pd.DataFrame(
        [
            {
                "local_contrast_all_null_status": (
                    "positive_significant_negative_suppressed_all_nulls"
                    if passing and all(passing)
                    else "not_strict_local_contrast_success"
                ),
                "alpha": float(alpha),
                "null_models_required": ";".join(null_models),
                "null_models_passing": ";".join(
                    null_model for null_model, ok in zip(null_models, passing, strict=False) if ok
                ),
                "claim_boundary": (
                    "Strict local-contrast success requires at least one positive LR significant and no "
                    "high-expression negative-control LR significant under every requested null model."
                ),
            }
        ]
    )


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    pos = int((pd.to_numeric(positives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg = int((pd.to_numeric(negatives.get("local_contrast_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos > 0 and neg == 0:
        return "positive_significant_negative_suppressed"
    if pos > 0 and neg > 0:
        return "positive_significant_but_negative_leakage"
    if pos == 0 and neg == 0:
        return "no_positive_recovery_negative_suppressed"
    return "negative_control_leakage"


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(
        frame.sort_values(
            ["local_contrast_bh_fdr", "local_contrast_p_value", "local_contrast_statistic"],
            ascending=[True, True, False],
        ).iloc[0].get("control_name", "")
    )


def _best_value(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
