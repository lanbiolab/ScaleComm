#!/usr/bin/env python
"""Matched spatial-null diagnostic for Phase 12 expanded atlas FDR.

The pooled spatial-distance null in the expanded atlas audit compares each LR
against every permuted LR score. That is intentionally conservative, but it can
also be too broad because high-support LR pairs create a hard null for weak,
mechanism-specific signals.

This script keeps the original conservative audit intact and adds one diagnostic
variant: each LR is tested against spatial-distance null scores from LR pairs
matched by mechanism and expression/support features. Receiver and LR-identity
nulls are still taken from the original audit. This is a post-hoc diagnostic,
not an end-to-end model-rerun FDR procedure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PERM_DIR = EXPANDED_DIR / "permutation_audit"
OUT_DIR = PHASE12_DIR / "matched_spatial_atlas_fdr"
N_PERMUTATIONS = perm.N_PERMUTATIONS
N_MATCHED_LR = 120
MIN_MATCHED_LR = 30
RNG_SEED = 12122026 + 777


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    original = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")

    lr_rows = []
    match_rows = []
    null_summary_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        dataset_ranking = ranking[ranking["dataset_key"].eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        dataset_original = original[original["dataset_key"].eq(dataset_key)].copy()
        result = run_dataset(dataset_key, dataset_ranking, dataset_original)
        lr_rows.append(result["lr"])
        match_rows.append(result["matches"])
        null_summary_rows.append(result["null_summary"])

    lr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    matches = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    null_summary = pd.concat(null_summary_rows, ignore_index=True) if null_summary_rows else pd.DataFrame()
    topk = summarize_topk(lr)
    dataset = summarize_dataset(lr, topk)
    story = summarize_story(dataset)

    lr.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_lr.csv", index=False)
    matches.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_matches.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_dataset.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_null_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_matched_spatial_atlas_fdr_story.csv", index=False)
    write_status(dataset, topk, lr, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(lr.groupby("dataset_key").head(15).to_string(index=False))
    print()
    print(story.to_string(index=False))


def run_dataset(dataset_key: str, ranking: pd.DataFrame, original: pd.DataFrame) -> dict[str, pd.DataFrame]:
    replicate_dir = EXPANDED_DIR / "replicates" / dataset_key
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = perm.prepare_base_table(ranking, lr_pairs)
    arrays = perm.prepare_candidate_arrays(candidates, base)

    rng = np.random.default_rng(RNG_SEED + perm.stable_dataset_offset(dataset_key))
    spatial_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    strict_spatial_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    for perm_idx in range(N_PERMUTATIONS):
        null_table = perm.build_null_table(base, arrays, "spatial_distance_permutation", rng)
        spatial_scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
        strict_spatial_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)

    lr, matches = build_lr_table(dataset_key, base, original, spatial_scores, strict_spatial_scores)
    flat = spatial_scores.ravel()
    strict_flat = strict_spatial_scores.ravel()
    null_summary = pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "n_permutations": N_PERMUTATIONS,
                "n_lr": int(len(base)),
                "n_null_scores": int(len(flat)),
                "pooled_spatial_null_mean": float(np.nanmean(flat)),
                "pooled_spatial_null_q95": float(np.nanquantile(flat, 0.95)),
                "pooled_spatial_null_q99": float(np.nanquantile(flat, 0.99)),
                "matched_null_median_n_lr": float(lr["matched_null_n_lr"].median()),
                "matched_null_min_n_lr": int(lr["matched_null_n_lr"].min()),
                "matched_null_median_n_scores": float(lr["matched_null_n_scores"].median()),
                "strict_pooled_spatial_null_mean": float(np.nanmean(strict_flat)),
                "strict_pooled_spatial_null_q95": float(np.nanquantile(strict_flat, 0.95)),
                "strict_pooled_spatial_null_q99": float(np.nanquantile(strict_flat, 0.99)),
            }
        ]
    )
    return {"lr": lr, "matches": matches, "null_summary": null_summary}


def build_lr_table(
    dataset_key: str,
    base: pd.DataFrame,
    original: pd.DataFrame,
    spatial_scores: np.ndarray,
    strict_spatial_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_sets, match_rows = build_match_sets(dataset_key, base)
    rows = []
    observed = base["calibrated_score"].to_numpy(dtype=float)
    strict_observed = base["strict_discovery_score"].to_numpy(dtype=float)
    original_by_lr = original.set_index("lr_id")
    matched_spatial_p = np.full(len(base), np.nan, dtype=float)
    strict_matched_spatial_p = np.full(len(base), np.nan, dtype=float)

    for idx, row in enumerate(base.itertuples(index=False)):
        match_idx = np.asarray(match_sets[idx], dtype=int)
        null_values = spatial_scores[:, match_idx].ravel() if len(match_idx) else np.asarray([], dtype=float)
        strict_null_values = (
            strict_spatial_scores[:, match_idx].ravel() if len(match_idx) else np.asarray([], dtype=float)
        )
        matched_spatial_p[idx] = empirical_upper_p(observed[idx], null_values)
        strict_matched_spatial_p[idx] = empirical_upper_p(strict_observed[idx], strict_null_values)

    matched_spatial_q = bh_by_dataset(base, matched_spatial_p)
    strict_matched_spatial_q = bh_by_dataset(base, strict_matched_spatial_p)
    matched_all_null_p = np.full(len(base), np.nan, dtype=float)
    strict_matched_all_null_p = np.full(len(base), np.nan, dtype=float)

    for idx, row in enumerate(base.itertuples(index=False)):
        orig = original_by_lr.loc[row.lr_id] if row.lr_id in original_by_lr.index else pd.Series(dtype=float)
        receiver_p = to_float(orig.get("receiver_permutation_global_empirical_p", np.nan))
        identity_p = to_float(orig.get("lr_identity_permutation_global_empirical_p", np.nan))
        strict_receiver_p = to_float(orig.get("strict_receiver_permutation_global_empirical_p", np.nan))
        strict_identity_p = to_float(orig.get("strict_lr_identity_permutation_global_empirical_p", np.nan))
        matched_all_null_p[idx] = np.nanmax([receiver_p, identity_p, matched_spatial_p[idx]])
        strict_matched_all_null_p[idx] = np.nanmax(
            [strict_receiver_p, strict_identity_p, strict_matched_spatial_p[idx]]
        )

    matched_all_null_q = bh_by_dataset(base, matched_all_null_p)
    strict_matched_all_null_q = bh_by_dataset(base, strict_matched_all_null_p)

    for idx, row in enumerate(base.itertuples(index=False)):
        orig = original_by_lr.loc[row.lr_id] if row.lr_id in original_by_lr.index else pd.Series(dtype=float)
        match_idx = np.asarray(match_sets[idx], dtype=int)
        rows.append(
            {
                "dataset_key": dataset_key,
                "rank": int(row.rank),
                "strict_rank": int(row.strict_rank),
                "lr_id": row.lr_id,
                "lr_pair": row.lr_pair,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "biological_category": row.biological_category,
                "mechanism": row.mechanism,
                "selected_reason": row.selected_reason,
                "true_lr": bool(row.true_lr),
                "false_positive_high_expression": bool(row.false_positive_high_expression),
                "calibrated_score": float(row.calibrated_score),
                "strict_discovery_score": float(row.strict_discovery_score),
                "matched_null_scope": match_scope(base, idx, match_idx),
                "matched_null_n_lr": int(len(match_idx)),
                "matched_null_n_scores": int(len(match_idx) * N_PERMUTATIONS),
                "matched_spatial_empirical_p": float(matched_spatial_p[idx]),
                "matched_spatial_bh_fdr": float(matched_spatial_q[idx]),
                "strict_matched_spatial_empirical_p": float(strict_matched_spatial_p[idx]),
                "strict_matched_spatial_bh_fdr": float(strict_matched_spatial_q[idx]),
                "receiver_global_empirical_p": to_float(orig.get("receiver_permutation_global_empirical_p", np.nan)),
                "lr_identity_global_empirical_p": to_float(orig.get("lr_identity_permutation_global_empirical_p", np.nan)),
                "pooled_spatial_global_empirical_p": to_float(
                    orig.get("spatial_distance_permutation_global_empirical_p", np.nan)
                ),
                "pooled_spatial_global_bh_fdr": to_float(orig.get("spatial_distance_permutation_global_bh_fdr", np.nan)),
                "original_conservative_bh_fdr": to_float(orig.get("conservative_global_bh_fdr", np.nan)),
                "matched_all_null_empirical_p": float(matched_all_null_p[idx]),
                "matched_all_null_bh_fdr": float(matched_all_null_q[idx]),
                "strict_receiver_global_empirical_p": to_float(
                    orig.get("strict_receiver_permutation_global_empirical_p", np.nan)
                ),
                "strict_lr_identity_global_empirical_p": to_float(
                    orig.get("strict_lr_identity_permutation_global_empirical_p", np.nan)
                ),
                "strict_pooled_spatial_global_empirical_p": to_float(
                    orig.get("strict_spatial_distance_permutation_global_empirical_p", np.nan)
                ),
                "strict_pooled_spatial_global_bh_fdr": to_float(
                    orig.get("strict_spatial_distance_permutation_global_bh_fdr", np.nan)
                ),
                "strict_original_conservative_bh_fdr": to_float(
                    orig.get("strict_conservative_global_bh_fdr", np.nan)
                ),
                "strict_matched_all_null_empirical_p": float(strict_matched_all_null_p[idx]),
                "strict_matched_all_null_bh_fdr": float(strict_matched_all_null_q[idx]),
                "matched_spatial_call": classify_lr(
                    matched_spatial_q[idx],
                    matched_all_null_q[idx],
                    strict_matched_spatial_q[idx],
                    strict_matched_all_null_q[idx],
                ),
            }
        )
    lr = pd.DataFrame(rows).sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)
    return lr, pd.DataFrame(match_rows)


def build_match_sets(dataset_key: str, base: pd.DataFrame) -> tuple[dict[int, list[int]], list[dict[str, object]]]:
    features = [
        "n_nonzero_expr",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "source_support",
        "raw_mean_score",
        "raw_q95_score",
    ]
    work = base.copy()
    if "source_support" not in work.columns:
        source_count = pd.to_numeric(work.get("source_count", pd.Series(1.0, index=work.index)), errors="coerce").fillna(1.0)
        work["source_support"] = (np.log1p(source_count) / np.log1p(12)).clip(0, 1)
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    matrix = work[features].to_numpy(dtype=float)
    matrix[:, 0] = np.log1p(matrix[:, 0])
    scale = np.nanstd(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    z = (matrix - np.nanmean(matrix, axis=0)) / scale

    match_sets: dict[int, list[int]] = {}
    match_rows: list[dict[str, object]] = []
    for idx, row in enumerate(work.itertuples(index=False)):
        scopes = [
            ("same_mechanism_same_category", work.index[
                work["mechanism"].astype(str).eq(str(row.mechanism))
                & work["biological_category"].astype(str).eq(str(row.biological_category))
                & (work.index != idx)
            ].to_numpy(dtype=int)),
            ("same_mechanism", work.index[
                work["mechanism"].astype(str).eq(str(row.mechanism)) & (work.index != idx)
            ].to_numpy(dtype=int)),
            ("same_category", work.index[
                work["biological_category"].astype(str).eq(str(row.biological_category)) & (work.index != idx)
            ].to_numpy(dtype=int)),
            ("all_lr_except_self", work.index[work.index != idx].to_numpy(dtype=int)),
        ]
        selected_scope = "all_lr_except_self"
        pool = scopes[-1][1]
        for scope, candidate_pool in scopes:
            if len(candidate_pool) >= MIN_MATCHED_LR:
                selected_scope = scope
                pool = candidate_pool
                break
        distance = np.sqrt(np.sum((z[pool, :] - z[idx, :]) ** 2, axis=1))
        order = np.lexsort((work.iloc[pool]["rank"].to_numpy(dtype=float), distance))
        selected = pool[order[: min(N_MATCHED_LR, len(order))]]
        match_sets[idx] = selected.astype(int).tolist()
        match_rows.append(
            {
                "dataset_key": dataset_key,
                "lr_id": row.lr_id,
                "lr_pair": row.lr_pair,
                "rank": int(row.rank),
                "true_lr": bool(row.true_lr),
                "matched_null_scope": selected_scope,
                "matched_null_n_lr": int(len(selected)),
                "matched_true_lr": int(work.iloc[selected]["true_lr"].astype(bool).sum()) if len(selected) else 0,
                "matched_high_expression_negative": int(
                    work.iloc[selected]["false_positive_high_expression"].astype(bool).sum()
                )
                if len(selected)
                else 0,
                "matched_best_rank": int(work.iloc[selected]["rank"].min()) if len(selected) else 0,
                "matched_best_pair": str(work.iloc[selected].sort_values("rank").iloc[0]["lr_pair"])
                if len(selected)
                else "",
            }
        )
    return match_sets, match_rows


def match_scope(base: pd.DataFrame, idx: int, match_idx: np.ndarray) -> str:
    if len(match_idx) == 0:
        return "none"
    row = base.iloc[idx]
    matched = base.iloc[match_idx]
    if (
        matched["mechanism"].astype(str).eq(str(row["mechanism"])).all()
        and matched["biological_category"].astype(str).eq(str(row["biological_category"])).all()
    ):
        return "same_mechanism_same_category"
    if matched["mechanism"].astype(str).eq(str(row["mechanism"])).all():
        return "same_mechanism"
    if matched["biological_category"].astype(str).eq(str(row["biological_category"])).all():
        return "same_category"
    return "all_lr_except_self"


def summarize_topk(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in (10, 20, 50, 100):
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_matched_spatial_q_0_1": int(top["matched_spatial_bh_fdr"].le(0.1).sum()),
                    "n_matched_all_null_q_0_1": int(top["matched_all_null_bh_fdr"].le(0.1).sum()),
                    "n_strict_matched_spatial_q_0_1": int(top["strict_matched_spatial_bh_fdr"].le(0.1).sum()),
                    "n_strict_matched_all_null_q_0_1": int(top["strict_matched_all_null_bh_fdr"].le(0.1).sum()),
                    "median_matched_all_null_q": float(top["matched_all_null_bh_fdr"].median()),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(lr: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        true = sub["true_lr"].astype(bool)
        top10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))]
        top10_row = top10.iloc[0].to_dict() if len(top10) else {}
        true_sub = sub[true].copy()
        best_true = true_sub.sort_values(["matched_all_null_bh_fdr", "rank"]).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "n_permutations": N_PERMUTATIONS,
                "top10_true_lr": int(top10_row.get("n_true_lr", 0)),
                "top10_matched_spatial_q_0_1": int(top10_row.get("n_matched_spatial_q_0_1", 0)),
                "top10_matched_all_null_q_0_1": int(top10_row.get("n_matched_all_null_q_0_1", 0)),
                "all_lr_matched_spatial_q_0_1": int(sub["matched_spatial_bh_fdr"].le(0.1).sum()),
                "all_true_lr_matched_spatial_q_0_1": int((true & sub["matched_spatial_bh_fdr"].le(0.1)).sum()),
                "all_lr_matched_all_null_q_0_1": int(sub["matched_all_null_bh_fdr"].le(0.1).sum()),
                "all_true_lr_matched_all_null_q_0_1": int((true & sub["matched_all_null_bh_fdr"].le(0.1)).sum()),
                "all_lr_strict_matched_spatial_q_0_1": int(sub["strict_matched_spatial_bh_fdr"].le(0.1).sum()),
                "all_true_lr_strict_matched_spatial_q_0_1": int(
                    (true & sub["strict_matched_spatial_bh_fdr"].le(0.1)).sum()
                ),
                "all_lr_strict_matched_all_null_q_0_1": int(sub["strict_matched_all_null_bh_fdr"].le(0.1).sum()),
                "all_true_lr_strict_matched_all_null_q_0_1": int(
                    (true & sub["strict_matched_all_null_bh_fdr"].le(0.1)).sum()
                ),
                "best_true_lr": str(best_true.iloc[0]["lr_pair"]) if len(best_true) else "",
                "best_true_rank": int(best_true.iloc[0]["rank"]) if len(best_true) else 0,
                "best_true_matched_all_null_q": float(best_true.iloc[0]["matched_all_null_bh_fdr"])
                if len(best_true)
                else np.nan,
                "best_true_strict_matched_all_null_q": float(best_true.iloc[0]["strict_matched_all_null_bh_fdr"])
                if len(best_true)
                else np.nan,
                "matched_spatial_audit_call": classify_dataset(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    all_true_hits = int(dataset["all_true_lr_matched_all_null_q_0_1"].sum())
    strict_true_hits = int(dataset["all_true_lr_strict_matched_all_null_q_0_1"].sum())
    spatial_true_hits = int(dataset["all_true_lr_matched_spatial_q_0_1"].sum())
    if all_true_hits > 0:
        call = "matched_spatial_all_null_supportive_with_caveat"
    elif spatial_true_hits > 0:
        call = "matched_spatial_only_supportive_pairing_identity_bottleneck"
    else:
        call = "matched_spatial_null_not_supportive"
    return pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset)),
                "n_true_matched_spatial_q_0_1": spatial_true_hits,
                "n_true_matched_all_null_q_0_1": all_true_hits,
                "n_true_strict_matched_all_null_q_0_1": strict_true_hits,
                "overall_matched_spatial_call": call,
                "claim_boundary": "Matched spatial FDR is a post-hoc diagnostic using matched LR null pools. It can explain whether pooled spatial nulls are too broad, but it does not replace end-to-end model-rerun whole-atlas FDR.",
            }
        ]
    )


def classify_lr(spatial_q: float, all_q: float, strict_spatial_q: float, strict_all_q: float) -> str:
    if np.isfinite(strict_all_q) and strict_all_q <= 0.1:
        return "strict_matched_all_null_supported"
    if np.isfinite(all_q) and all_q <= 0.1:
        return "matched_all_null_supported"
    if np.isfinite(strict_spatial_q) and strict_spatial_q <= 0.1:
        return "strict_matched_spatial_only"
    if np.isfinite(spatial_q) and spatial_q <= 0.1:
        return "matched_spatial_only"
    return "not_matched_spatial_supported"


def classify_dataset(sub: pd.DataFrame) -> str:
    true = sub["true_lr"].astype(bool)
    true_all = int((true & sub["matched_all_null_bh_fdr"].le(0.1)).sum())
    strict_true_all = int((true & sub["strict_matched_all_null_bh_fdr"].le(0.1)).sum())
    true_spatial = int((true & sub["matched_spatial_bh_fdr"].le(0.1)).sum())
    top10 = sub.sort_values("rank").head(10)
    top10_true = int(top10["true_lr"].astype(bool).sum())
    top10_all = int(top10["matched_all_null_bh_fdr"].le(0.1).sum())
    if strict_true_all > 0:
        return "strict_matched_all_null_supportive"
    if true_all > 0 and top10_true >= 3 and top10_all > 0:
        return "matched_all_null_supportive_with_topk_context"
    if true_all > 0:
        return "matched_all_null_supportive_with_caveat"
    if true_spatial > 0:
        return "matched_spatial_only_pairing_identity_bottleneck"
    return "matched_spatial_not_supportive"


def empirical_upper_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(observed):
        return np.nan
    return float((1.0 + np.sum(null_values >= observed)) / (1.0 + len(null_values)))


def bh_by_dataset(base: pd.DataFrame, pvalues: np.ndarray) -> np.ndarray:
    out = np.full(len(base), np.nan, dtype=float)
    for _, idx in base.groupby("dataset_key", sort=False).groups.items() if "dataset_key" in base else [("", base.index)]:
        idx_array = np.asarray(list(idx), dtype=int)
        out[idx_array] = bh_adjust(np.asarray(pvalues[idx_array], dtype=float))
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, lr: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Matched Spatial Atlas FDR Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic replaces the pooled spatial-distance null with an LR-matched spatial null while keeping receiver-pairing and LR-identity nulls from the original expanded-atlas permutation audit.",
        "It is post-hoc and does not retrain SpatialSelf under the matched null.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.matched_spatial_audit_call}`: "
            f"LR={int(row.n_lr)}, true LR={int(row.n_true_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, "
            f"matched spatial true q<=0.1={int(row.all_true_lr_matched_spatial_q_0_1)}, "
            f"matched all-null true q<=0.1={int(row.all_true_lr_matched_all_null_q_0_1)}, "
            f"strict matched all-null true q<=0.1={int(row.all_true_lr_strict_matched_all_null_q_0_1)}, "
            f"best true={row.best_true_lr} q={fmt(row.best_true_matched_all_null_q)}."
        )
    lines += ["", "## Top LR Diagnostics", ""]
    for row in lr.sort_values(["dataset_key", "rank"]).groupby("dataset_key").head(10).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` rank {int(row.rank)} `{row.lr_pair}`: "
            f"{row.matched_spatial_call}; matched spatial q={fmt(row.matched_spatial_bh_fdr)}, "
            f"matched all-null q={fmt(row.matched_all_null_bh_fdr)}, "
            f"original conservative q={fmt(row.original_conservative_bh_fdr)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        row = story.iloc[0]
        lines.append(
            f"- `{row.overall_matched_spatial_call}`: true matched-spatial q<=0.1="
            f"{int(row.n_true_matched_spatial_q_0_1)}, true matched all-null q<=0.1="
            f"{int(row.n_true_matched_all_null_q_0_1)}, strict true matched all-null q<=0.1="
            f"{int(row.n_true_strict_matched_all_null_q_0_1)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this to diagnose whether the pooled spatial null is too broad.",
        "- Do not use it as native real-tissue ground truth or as final end-to-end whole-atlas FDR.",
        "- If matched all-null FDR remains weak, the next step is model/null redesign rather than more post-hoc calibration.",
    ]
    (OUT_DIR / "phase12_matched_spatial_atlas_fdr_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_matched_spatial_atlas_fdr",
        "n_permutations": N_PERMUTATIONS,
        "n_matched_lr": N_MATCHED_LR,
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv",
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_matches.csv",
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_topk.csv",
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv",
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv",
        ],
        "claim_boundary": "Post-hoc matched spatial-null diagnostic; not end-to-end model-rerun whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_matched_spatial_atlas_fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
