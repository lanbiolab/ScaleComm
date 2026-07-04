#!/usr/bin/env python
"""Run non-degenerate competitive LR-pair FDR for targeted real-data panels.

The existing LR-level spatial/receiver score-permutation nulls can be
degenerate when the group statistic is computed within the same LR group: all
permutations preserve the same set of scores and therefore the same top-tail
mean. This supplemental audit compares each LR's top-tail statistic against
random LR-sized edge sets sampled from a matched targeted candidate pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"


DATASET_SPECS = {
    "slide_tags_tonsil": {
        "candidate_path": GOAL_DIR
        / "slide_tags_tonsil_targeted_region_fdr_200perm/slide_tags_tonsil_targeted_region_candidates.csv",
        "prediction_path": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/predictions/"
        "spatialself_wavelet_calibrated/constructed_replicate__spatialself_wavelet_calibrated_predictions.csv",
        "lr_pairs_path": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/constructed_replicate/lr_pairs.csv",
    },
    "post_xenium_colon": {
        "candidate_path": GOAL_DIR
        / "post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_region_candidates.csv",
        "prediction_path": GOAL_DIR
        / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "lr_pairs_path": GOAL_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset/lr_pairs.csv",
    },
    "visium_hd_crc": {
        "candidate_path": GOAL_DIR
        / "visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/visium_hd_crc_p2crc_region_aware_targeted_candidates.csv",
        "prediction_path": GOAL_DIR
        / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
        "lr_pairs_path": GOAL_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset/lr_pairs.csv",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--candidate-path", default="")
    parser.add_argument("--prediction-path", default="")
    parser.add_argument("--lr-pairs-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=202620)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pool-mode", choices=("all_targeted", "same_reason"), default="all_targeted")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    candidate_path = Path(args.candidate_path) if args.candidate_path else Path(spec["candidate_path"])
    prediction_path = Path(args.prediction_path) if args.prediction_path else Path(spec["prediction_path"])
    lr_pairs_path = Path(args.lr_pairs_path) if args.lr_pairs_path else Path(spec["lr_pairs_path"])
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else GOAL_DIR / f"{args.dataset}_competitive_lr_fdr_{int(args.n_permutations)}perm"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    lr_pairs = pd.read_csv(lr_pairs_path)
    merged = _merge(candidates, predictions)

    fdr = _competitive_lr_fdr(
        merged,
        lr_pairs,
        n_permutations=int(args.n_permutations),
        top_fraction=float(args.top_fraction),
        random_seed=int(args.random_seed),
        pool_mode=str(args.pool_mode),
    )
    control = _control_summary(fdr, alpha=float(args.alpha))
    degeneracy = _degeneracy_diagnostics(merged)
    audit = _closure_audit(fdr, control, alpha=float(args.alpha))

    fdr_path = output_dir / f"{args.dataset}_competitive_lr_pair_fdr.csv"
    control_path = output_dir / f"{args.dataset}_competitive_control_fdr_summary.csv"
    degeneracy_path = output_dir / f"{args.dataset}_existing_null_degeneracy_diagnostics.csv"
    audit_path = output_dir / f"{args.dataset}_competitive_fdr_audit.csv"
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    degeneracy.to_csv(degeneracy_path, index=False)
    audit.to_csv(audit_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "candidate_path": _rel(candidate_path),
        "prediction_path": _rel(prediction_path),
        "lr_pairs_path": _rel(lr_pairs_path),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "top_fraction": float(args.top_fraction),
        "pool_mode": str(args.pool_mode),
        "alpha": float(args.alpha),
        "strict_competitive_success": bool(audit.get("competitive_fdr_status", pd.Series(dtype=str)).eq("positive_significant_negative_suppressed").any()),
        "claim_boundary": (
            "Supplemental non-degenerate competitive LR-pair FDR. It compares each LR's top-tail score to random "
            "LR-sized edge sets from the targeted candidate pool. It is not a model-rerun spatial permutation and "
            "should be reported separately from strict spatial/receiver/LR-identity all-null FDR."
        ),
        "outputs": {
            "lr_pair_fdr": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "degeneracy_diagnostics": _rel(degeneracy_path),
            "audit": _rel(audit_path),
        },
    }
    manifest_path = output_dir / f"{args.dataset}_competitive_lr_fdr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _merge(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" not in candidates or "candidate_id" not in predictions:
        raise ValueError("candidate_id is required in candidates and predictions")
    if "score" not in predictions:
        raise ValueError("prediction table requires score column")
    columns = ["candidate_id", "score"]
    if "method" in predictions:
        columns.append("method")
    merged = candidates.merge(predictions[columns], on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping candidate_id values")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _competitive_lr_fdr(
    merged: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    n_permutations: int,
    top_fraction: float,
    random_seed: int,
    pool_mode: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    score_pool_all = merged["score"].to_numpy(dtype=float)
    for lr_id, group in merged.groupby("lr_id", sort=False):
        scores = group["score"].to_numpy(dtype=float)
        observed = _top_tail_mean(scores, top_fraction=top_fraction)
        n_edges = int(len(scores))
        pool = _pool_scores(merged, group, pool_mode=pool_mode)
        if len(pool) == 0:
            pool = score_pool_all
        replace = len(pool) < n_edges
        null = np.empty(int(n_permutations), dtype=float)
        for i in range(int(n_permutations)):
            sample = rng.choice(pool, size=n_edges, replace=replace)
            null[i] = _top_tail_mean(sample, top_fraction=top_fraction)
        p_value = (float((null >= observed).sum()) + 1.0) / (float(n_permutations) + 1.0)
        rows.append(
            {
                "lr_id": lr_id,
                "competitive_null_model": f"candidate_pool_{pool_mode}",
                "group_statistic": observed,
                "group_p_value": p_value,
                "n_candidate_edges": n_edges,
                "null_mean": float(null.mean()),
                "null_std": float(null.std()),
                "null_q95": float(np.quantile(null, 0.95)),
                "top_fraction": float(top_fraction),
                "n_permutations": int(n_permutations),
                "pool_size": int(len(pool)),
            }
        )
    result = pd.DataFrame(rows)
    result["group_bh_fdr"] = _bh_adjust(result["group_p_value"].to_numpy(dtype=float))
    result = result.merge(lr_pairs, on="lr_id", how="left")
    return result.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False])


def _pool_scores(merged: pd.DataFrame, group: pd.DataFrame, *, pool_mode: str) -> np.ndarray:
    if pool_mode == "all_targeted" or "targeted_region_reason" not in merged:
        pool = merged
    elif pool_mode == "same_reason":
        reasons = set(group["targeted_region_reason"].astype(str))
        pool = merged[merged["targeted_region_reason"].astype(str).isin(reasons)]
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode}")
    return pool["score"].to_numpy(dtype=float)


def _control_summary(fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    positives = fdr[_positive_control_mask(fdr)]
    negatives = fdr[fdr.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    return pd.DataFrame(
        [
            {
                "competitive_null_model": fdr["competitive_null_model"].iloc[0] if not fdr.empty else "",
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "group_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "group_bh_fdr"),
                "n_positive_significant": int((pd.to_numeric(positives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_negative_significant": int((pd.to_numeric(negatives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "control_status": _control_status(positives, negatives, alpha),
            }
        ]
    )


def _closure_audit(fdr: pd.DataFrame, control: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for _, row in fdr.iterrows():
        is_negative = bool(row.get("false_positive_high_expression", False))
        significant = float(row.get("group_bh_fdr", 1.0)) <= float(alpha)
        if is_negative and significant:
            status = "negative_control_leakage"
        elif (not is_negative) and significant:
            status = "positive_significant_negative_suppressed"
        else:
            status = "not_competitive_fdr_significant"
        rows.append(
            {
                "lr_id": row.get("lr_id", ""),
                "control_name": row.get("control_name", ""),
                "control_class": row.get("control_class", ""),
                "false_positive_high_expression": is_negative,
                "competitive_group_bh_fdr": row.get("group_bh_fdr", ""),
                "competitive_group_p_value": row.get("group_p_value", ""),
                "competitive_group_statistic": row.get("group_statistic", ""),
                "competitive_null_mean": row.get("null_mean", ""),
                "competitive_fdr_status": status,
            }
        )
    audit = pd.DataFrame(rows)
    if not control.empty and str(control.iloc[0].get("control_status", "")) != "positive_significant_negative_suppressed":
        audit.loc[
            audit["competitive_fdr_status"].eq("positive_significant_negative_suppressed"),
            "competitive_fdr_status",
        ] = "positive_significant_but_negative_control_risk"
    return audit


def _positive_control_mask(frame: pd.DataFrame) -> pd.Series:
    true_lr = frame.get("true_lr", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    negative = frame.get("false_positive_high_expression", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    control_class = frame.get("control_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    named_control = frame.get("control_name", pd.Series("", index=frame.index)).fillna("").astype(str).str.len() > 0
    return true_lr & ~negative & named_control & ~control_class.eq("background_lr")


def _degeneracy_diagnostics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for null_model in ("spatial_permutation", "receiver_permutation"):
        if null_model == "spatial_permutation":
            strata = ["lr_id"]
        else:
            strata = ["lr_id", "sender_cell_type"] if "sender_cell_type" in merged else ["lr_id"]
        group_counts = merged.groupby(strata, dropna=False).size()
        rows.append(
            {
                "null_model": null_model,
                "diagnostic": "lr_group_statistic_degeneracy_expected",
                "strata": ",".join(strata),
                "n_strata": int(len(group_counts)),
                "n_singleton_strata": int((group_counts <= 1).sum()),
                "explanation": (
                    "This null permutes scores within strata that include lr_id. For an LR-pair group statistic, "
                    "the score multiset inside each LR is preserved, so top-tail LR statistics can be unchanged."
                ),
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
        return float(values.mean())
    return float(np.partition(values, values.size - k)[values.size - k :].mean())


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False]).iloc[0]
    return str(row.get("control_name", f"{row.get('ligand', '')}-{row.get('receptor', '')}"))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    pos = int((pd.to_numeric(positives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg = int((pd.to_numeric(negatives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos > 0 and neg == 0:
        return "positive_significant_negative_suppressed"
    if pos > 0 and neg > 0:
        return "positive_and_negative_controls_significant"
    if pos == 0 and neg > 0:
        return "negative_controls_significant_without_positive"
    return "no_control_significant"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
