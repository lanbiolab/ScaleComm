#!/usr/bin/env python
"""Exact-score permutation audit for spatial-specificity calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm  # noqa: E402
from scripts.phase12.calibrate_phase12_spatial_specificity import (  # noqa: E402
    add_spatial_specificity_calibration,
)
from scripts.phase12.calibrate_phase12_expanded_atlas_ranking import pct  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "spatial_specificity_permutation_audit"
TOP_K_VALUES = (10, 20, 50, 100)
N_PERMUTATIONS = 80
RNG_SEED = 12122026 + 202
NULL_TYPES = ("receiver_permutation", "spatial_distance_permutation", "lr_identity_permutation")

FEATURE_COLS = [
    "n_candidates",
    "n_nonzero_expr",
    "paired_expr_fraction",
    "ligand_positive_fraction",
    "receptor_positive_fraction",
    "mean_pair_expr",
    "q95_pair_expr",
    "mean_coexpression_distance",
    "q95_coexpression_distance",
    "rule_satisfied_fraction",
    "range_satisfied_fraction",
    "niche_satisfied_fraction",
    "in_window_expr_mean",
    "off_window_expr_mean",
    "in_window_coexpr_mean",
    "off_window_coexpr_mean",
    "spatial_expr_contrast",
    "spatial_coexpr_contrast",
    "spatial_lift",
    "spatial_specificity_score",
]
IDENTITY_SCORE_COLS = FEATURE_COLS + [
    "expr_gate",
    "source_support",
    "category_weight",
    "hard_negative_weight",
    "zero_expression_weight",
    "low_support_weight",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    status = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_status.csv")
    lr_rows = []
    null_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        sub = ranking[ranking["dataset_key"].eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        result = run_dataset(dataset_key, sub)
        lr_rows.append(result["lr_fdr"])
        null_rows.append(result["null_summary"])

    lr_fdr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null_summary = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    topk = summarize_topk(lr_fdr)
    dataset = summarize_dataset(lr_fdr, topk, null_summary, status)
    story = summarize_story(dataset)

    lr_fdr.to_csv(OUT_DIR / "phase12_spatial_specificity_permutation_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_spatial_specificity_permutation_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_spatial_specificity_permutation_null_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_specificity_permutation_story.csv", index=False)
    write_status(dataset, topk, null_summary, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))
    print()
    print(lr_fdr.groupby("dataset_key").head(15).to_string(index=False))


def run_dataset(dataset_key: str, ranking: pd.DataFrame) -> dict[str, pd.DataFrame]:
    replicate_dir = EXPANDED_DIR / "replicates" / dataset_key
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = prepare_base_table(ranking, lr_pairs)
    arrays = prepare_arrays(candidates, base, lr_pairs)

    observed_features = compute_spatial_features(arrays)
    observed_table = score_table(base, observed_features)
    observed = observed_table["calibrated_score"].to_numpy(dtype=float)
    strict_observed = observed_table["strict_discovery_score"].to_numpy(dtype=float)

    rng = np.random.default_rng(RNG_SEED + perm.stable_dataset_offset(dataset_key))
    null_scores_by_type: dict[str, np.ndarray] = {}
    strict_null_scores_by_type: dict[str, np.ndarray] = {}
    null_summary_rows = []
    for null_type in NULL_TYPES:
        scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        strict_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            null_table = build_null_score_table(base, arrays, null_type, rng)
            scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
            strict_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)
        null_scores_by_type[null_type] = scores
        strict_null_scores_by_type[null_type] = strict_scores
        flat = scores.ravel()
        strict_flat = strict_scores.ravel()
        null_summary_rows.append(
            {
                "dataset_key": dataset_key,
                "null_type": null_type,
                "n_permutations": N_PERMUTATIONS,
                "n_null_scores": int(len(flat)),
                "null_score_mean": float(np.mean(flat)),
                "null_score_q95": float(np.quantile(flat, 0.95)),
                "null_score_q99": float(np.quantile(flat, 0.99)),
                "null_score_max": float(np.max(flat)),
                "observed_top_score": float(np.max(observed)),
                "observed_top_exceeds_null_q99": bool(np.max(observed) > np.quantile(flat, 0.99)),
                "observed_top_exceeds_null_max": bool(np.max(observed) > np.max(flat)),
                "strict_null_score_mean": float(np.mean(strict_flat)),
                "strict_null_score_q95": float(np.quantile(strict_flat, 0.95)),
                "strict_null_score_q99": float(np.quantile(strict_flat, 0.99)),
                "strict_null_score_max": float(np.max(strict_flat)),
                "observed_top_strict_score": float(np.max(strict_observed)),
                "observed_top_strict_exceeds_null_q99": bool(
                    np.max(strict_observed) > np.quantile(strict_flat, 0.99)
                ),
                "observed_top_strict_exceeds_null_max": bool(np.max(strict_observed) > np.max(strict_flat)),
            }
        )

    lr_fdr = build_lr_fdr_table(dataset_key, observed_table, null_scores_by_type, strict_null_scores_by_type)
    return {"lr_fdr": lr_fdr, "null_summary": pd.DataFrame(null_summary_rows)}


def prepare_base_table(ranking: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    source = lr_pairs[["lr_id", "source_count"]].copy()
    base = ranking.merge(source, on="lr_id", how="left")
    base["source_count"] = pd.to_numeric(base["source_count"], errors="coerce").fillna(1)
    return base.reset_index(drop=True)


def prepare_arrays(candidates: pd.DataFrame, base: pd.DataFrame, lr_pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    lr_to_code = {lr_id: idx for idx, lr_id in enumerate(base["lr_id"].astype(str))}
    lr_code = candidates["lr_id"].astype(str).map(lr_to_code)
    valid = lr_code.notna()
    if not bool(valid.all()):
        candidates = candidates.loc[valid].copy()
        lr_code = candidates["lr_id"].astype(str).map(lr_to_code)
    lr_code_arr = lr_code.to_numpy(dtype=int)
    order = np.argsort(lr_code_arr)
    n_lr = len(base)
    counts = np.bincount(lr_code_arr, minlength=n_lr).astype(float)

    range_map = lr_pairs.set_index("lr_id")["spatial_range"].to_dict() if "spatial_range" in lr_pairs.columns else {}
    spatial_range_by_lr = []
    for row in base.itertuples(index=False):
        spatial_range_by_lr.append(float(range_map.get(row.lr_id, range_from_mechanism(row.mechanism))))
    spatial_range_by_lr = np.asarray(spatial_range_by_lr, dtype=float)
    return {
        "lr_code": lr_code_arr,
        "order": order,
        "counts": counts,
        "n_lr": np.asarray([n_lr], dtype=int),
        "ligand_expr": candidates["ligand_expr"].to_numpy(dtype=float),
        "receptor_expr": candidates["receptor_expr"].to_numpy(dtype=float),
        "distance": candidates["distance"].to_numpy(dtype=float),
        "type_rule": candidates["type_rule_satisfied"].astype(bool).to_numpy(),
        "niche_rule": (
            candidates["niche_rule_satisfied"].astype(bool).to_numpy()
            if "niche_rule_satisfied" in candidates.columns
            else np.ones(len(candidates), dtype=bool)
        ),
        "spatial_range_by_candidate": spatial_range_by_lr[lr_code_arr],
    }


def build_null_score_table(
    base: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    null_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if null_type == "receiver_permutation":
        features = compute_spatial_features(arrays, receptor_expr=rng.permutation(arrays["receptor_expr"]))
        return score_table(base, features)
    if null_type == "spatial_distance_permutation":
        features = compute_spatial_features(arrays, distance=rng.permutation(arrays["distance"]))
        return score_table(base, features)
    if null_type == "lr_identity_permutation":
        observed = score_table(base, compute_spatial_features(arrays))
        null = observed.copy()
        perm_idx = rng.permutation(len(null))
        for col in IDENTITY_SCORE_COLS:
            if col in null.columns:
                null[col] = null[col].to_numpy()[perm_idx]
        null = add_spatial_specificity_calibration(null)
        null["rank"] = null["calibrated_score"].rank(ascending=False, method="min").astype(int)
        null["strict_rank"] = null["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
        return null
    raise ValueError(f"Unknown null type: {null_type}")


def compute_spatial_features(
    arrays: dict[str, np.ndarray],
    receptor_expr: np.ndarray | None = None,
    distance: np.ndarray | None = None,
) -> pd.DataFrame:
    lr_code = arrays["lr_code"].astype(int)
    order = arrays["order"].astype(int)
    counts = arrays["counts"].astype(float)
    n_lr = int(arrays["n_lr"][0])
    ligand = arrays["ligand_expr"].astype(float)
    receptor = receptor_expr.astype(float) if receptor_expr is not None else arrays["receptor_expr"].astype(float)
    dist = distance.astype(float) if distance is not None else arrays["distance"].astype(float)
    spatial_range = arrays["spatial_range_by_candidate"].astype(float)
    type_rule = arrays["type_rule"].astype(bool)
    niche_rule = arrays["niche_rule"].astype(bool)

    pair_expr = np.log1p(ligand) * np.log1p(receptor)
    distance_weight = np.exp(-dist / np.maximum(spatial_range, 1.0))
    coexpr_distance = pair_expr * distance_weight
    range_rule = dist <= spatial_range
    in_window = range_rule & type_rule & niche_rule
    off_window = ~in_window

    n_nonzero = np.bincount(lr_code, weights=(pair_expr > 0).astype(float), minlength=n_lr)
    ligand_pos = np.bincount(lr_code, weights=(ligand > 0).astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    receptor_pos = np.bincount(lr_code, weights=(receptor > 0).astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    paired_expr_fraction = n_nonzero / np.maximum(counts, 1.0)
    mean_pair_expr = np.bincount(lr_code, weights=pair_expr, minlength=n_lr) / np.maximum(counts, 1.0)
    mean_coexpr = np.bincount(lr_code, weights=coexpr_distance, minlength=n_lr) / np.maximum(counts, 1.0)
    q95_pair_expr = perm.grouped_quantile(pair_expr, lr_code, order, n_lr, 0.95)
    q95_coexpr = perm.grouped_quantile(coexpr_distance, lr_code, order, n_lr, 0.95)
    type_fraction = np.bincount(lr_code, weights=type_rule.astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    range_fraction = np.bincount(lr_code, weights=in_window.astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    niche_fraction = np.bincount(lr_code, weights=niche_rule.astype(float), minlength=n_lr) / np.maximum(counts, 1.0)

    in_count = np.bincount(lr_code, weights=in_window.astype(float), minlength=n_lr)
    off_count = np.bincount(lr_code, weights=off_window.astype(float), minlength=n_lr)
    in_expr = np.bincount(lr_code, weights=pair_expr * in_window, minlength=n_lr) / np.maximum(in_count, 1.0)
    off_expr = np.bincount(lr_code, weights=pair_expr * off_window, minlength=n_lr) / np.maximum(off_count, 1.0)
    in_coexpr = np.bincount(lr_code, weights=coexpr_distance * in_window, minlength=n_lr) / np.maximum(in_count, 1.0)
    off_coexpr = np.bincount(lr_code, weights=coexpr_distance * off_window, minlength=n_lr) / np.maximum(off_count, 1.0)
    expr_contrast = (in_expr + 1e-6) / (in_expr + off_expr + 2e-6)
    coexpr_contrast = (in_coexpr + 1e-6) / (in_coexpr + off_coexpr + 2e-6)
    spatial_lift = np.log1p(in_expr) - np.log1p(off_expr)
    specificity = (
        0.35 * np.clip(expr_contrast, 0, 1)
        + 0.35 * np.clip(coexpr_contrast, 0, 1)
        + 0.20 * pct(pd.Series(spatial_lift)).to_numpy(dtype=float)
        + 0.10 * np.clip(range_fraction, 0, 1)
    )
    return pd.DataFrame(
        {
            "n_candidates": counts.astype(int),
            "n_nonzero_expr": n_nonzero.astype(int),
            "paired_expr_fraction": paired_expr_fraction,
            "ligand_positive_fraction": ligand_pos,
            "receptor_positive_fraction": receptor_pos,
            "mean_pair_expr": mean_pair_expr,
            "q95_pair_expr": q95_pair_expr,
            "mean_coexpression_distance": mean_coexpr,
            "q95_coexpression_distance": q95_coexpr,
            "rule_satisfied_fraction": type_fraction,
            "range_satisfied_fraction": range_fraction,
            "niche_satisfied_fraction": niche_fraction,
            "in_window_expr_mean": in_expr,
            "off_window_expr_mean": off_expr,
            "in_window_coexpr_mean": in_coexpr,
            "off_window_coexpr_mean": off_coexpr,
            "spatial_expr_contrast": expr_contrast,
            "spatial_coexpr_contrast": coexpr_contrast,
            "spatial_lift": spatial_lift,
            "spatial_specificity_score": specificity,
        }
    )


def score_table(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for col in FEATURE_COLS:
        out[col] = features[col].to_numpy()
    out = add_spatial_specificity_calibration(out)
    out["method"] = "spatialself_wavelet_edge_prior_spatial_specificity_permutation_audit"
    out["rank"] = out["calibrated_score"].rank(ascending=False, method="min").astype(int)
    out["strict_rank"] = out["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    out["rank_q95"] = out["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    return out


def build_lr_fdr_table(
    dataset_key: str,
    observed_table: pd.DataFrame,
    null_scores_by_type: dict[str, np.ndarray],
    strict_null_scores_by_type: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = observed_table.copy()
    observed = out["calibrated_score"].to_numpy(dtype=float)
    strict_observed = out["strict_discovery_score"].to_numpy(dtype=float)
    conservative_global_p = np.zeros(len(out), dtype=float)
    conservative_within_p = np.zeros(len(out), dtype=float)
    for null_type, scores in null_scores_by_type.items():
        flat = scores.ravel()
        global_p = np.asarray([(1.0 + np.sum(flat >= score)) / (1.0 + len(flat)) for score in observed], dtype=float)
        within_p = np.asarray(
            [(1.0 + np.sum(scores[:, idx] >= observed[idx])) / (1.0 + scores.shape[0]) for idx in range(len(out))],
            dtype=float,
        )
        out[f"{null_type}_global_empirical_p"] = global_p
        out[f"{null_type}_within_lr_empirical_p"] = within_p
        out[f"{null_type}_global_bh_fdr"] = perm.bh_adjust(global_p)
        conservative_global_p = np.maximum(conservative_global_p, global_p)
        conservative_within_p = np.maximum(conservative_within_p, within_p)
    out["conservative_global_empirical_p"] = conservative_global_p
    out["conservative_global_bh_fdr"] = perm.bh_adjust(conservative_global_p)
    out["conservative_within_lr_empirical_p"] = conservative_within_p

    strict_conservative_global_p = np.zeros(len(out), dtype=float)
    strict_conservative_within_p = np.zeros(len(out), dtype=float)
    for null_type, scores in strict_null_scores_by_type.items():
        flat = scores.ravel()
        global_p = np.asarray(
            [(1.0 + np.sum(flat >= score)) / (1.0 + len(flat)) for score in strict_observed],
            dtype=float,
        )
        within_p = np.asarray(
            [(1.0 + np.sum(scores[:, idx] >= strict_observed[idx])) / (1.0 + scores.shape[0]) for idx in range(len(out))],
            dtype=float,
        )
        out[f"strict_{null_type}_global_empirical_p"] = global_p
        out[f"strict_{null_type}_within_lr_empirical_p"] = within_p
        out[f"strict_{null_type}_global_bh_fdr"] = perm.bh_adjust(global_p)
        strict_conservative_global_p = np.maximum(strict_conservative_global_p, global_p)
        strict_conservative_within_p = np.maximum(strict_conservative_within_p, within_p)
    out["strict_conservative_global_empirical_p"] = strict_conservative_global_p
    out["strict_conservative_global_bh_fdr"] = perm.bh_adjust(strict_conservative_global_p)
    out["strict_conservative_within_lr_empirical_p"] = strict_conservative_within_p
    out["dataset_key"] = dataset_key
    out["permutation_support_class"] = [perm.classify_lr(row) for row in out.itertuples(index=False)]
    out["strict_permutation_support_class"] = [perm.classify_strict_lr(row) for row in out.itertuples(index=False)]

    keep = [
        "dataset_key",
        "method",
        "rank",
        "strict_rank",
        "rank_q95",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "calibrated_score",
        "strict_discovery_score",
        "spatial_specificity_score",
        "spatial_expr_contrast",
        "spatial_lift",
        "conservative_global_empirical_p",
        "conservative_global_bh_fdr",
        "conservative_within_lr_empirical_p",
        "strict_conservative_global_empirical_p",
        "strict_conservative_global_bh_fdr",
        "strict_conservative_within_lr_empirical_p",
        "permutation_support_class",
        "strict_permutation_support_class",
    ]
    for null_type in NULL_TYPES:
        keep += [
            f"{null_type}_global_empirical_p",
            f"{null_type}_global_bh_fdr",
            f"{null_type}_within_lr_empirical_p",
            f"strict_{null_type}_global_empirical_p",
            f"strict_{null_type}_global_bh_fdr",
            f"strict_{null_type}_within_lr_empirical_p",
        ]
    return out[keep].sort_values(["rank", "lr_pair"]).reset_index(drop=True)


def summarize_topk(lr_fdr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annotated = perm.annotate_risk(lr_fdr)
    for dataset_key, sub in annotated.groupby("dataset_key"):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": perm.safe_div(top["negative_or_risk"].sum(), len(top)),
                    "n_conservative_fdr_0_1": int(top["conservative_global_bh_fdr"].le(0.1).sum()),
                    "median_conservative_fdr": float(top["conservative_global_bh_fdr"].median()),
                    "median_spatial_specificity": float(top["spatial_specificity_score"].median()),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(
    lr_fdr: pd.DataFrame,
    topk: pd.DataFrame,
    null_summary: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    annotated = perm.annotate_risk(lr_fdr)
    for dataset_key, sub in annotated.groupby("dataset_key"):
        t10 = perm.one_row(topk, dataset_key, 10)
        t50 = perm.one_row(topk, dataset_key, 50)
        stat = status[status["dataset_key"].eq(dataset_key)].head(1)
        true = sub[sub["true_lr"].astype(bool)].copy()
        best_true = true.sort_values("conservative_global_bh_fdr").head(1)
        best_spatial = true.sort_values("spatial_distance_permutation_global_bh_fdr").head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "n_permutations_per_null": N_PERMUTATIONS,
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_negative_or_risk_fraction": float(t10.get("negative_or_risk_fraction", np.nan)),
                "top10_conservative_fdr_0_1": int(t10.get("n_conservative_fdr_0_1", 0)),
                "top50_conservative_fdr_0_1": int(t50.get("n_conservative_fdr_0_1", 0)),
                "all_lr_conservative_fdr_0_1": int(sub["conservative_global_bh_fdr"].le(0.1).sum()),
                "all_true_lr_conservative_fdr_0_1": int(
                    (sub["true_lr"].astype(bool) & sub["conservative_global_bh_fdr"].le(0.1)).sum()
                ),
                "best_true_lr": str(best_true["lr_pair"].iloc[0]) if len(best_true) else "NA",
                "best_true_lr_fdr": float(best_true["conservative_global_bh_fdr"].iloc[0]) if len(best_true) else np.nan,
                "best_true_spatial_lr": str(best_spatial["lr_pair"].iloc[0]) if len(best_spatial) else "NA",
                "best_true_spatial_fdr": float(best_spatial["spatial_distance_permutation_global_bh_fdr"].iloc[0]) if len(best_spatial) else np.nan,
                "runtime_seconds": float(stat["runtime_seconds"].iloc[0]) if len(stat) else np.nan,
                "spatial_specificity_permutation_call": classify_dataset(sub, t10),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n = len(dataset)
    total_true_fdr = int(pd.to_numeric(dataset["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
    clean_top10 = int(pd.to_numeric(dataset["top10_negative_or_risk_fraction"], errors="coerce").fillna(1).le(0.0).sum())
    if total_true_fdr > 0:
        call = "spatial_specificity_fdr_candidate"
    elif clean_top10 == n:
        call = "spatial_specificity_ranking_clean_fdr_not_closed"
    else:
        call = "spatial_specificity_not_fdr_supportive"
    return pd.DataFrame(
        [
            {
                "topic": "spatial_specificity_permutation",
                "overall_call": call,
                "n_datasets": n,
                "total_true_q_0_1": total_true_fdr,
                "n_clean_top10_risk": clean_top10,
                "best_true_q_min": float(pd.to_numeric(dataset["best_true_lr_fdr"], errors="coerce").min()),
                "claim_boundary": "Exact-score post-hoc audit; do not claim end-to-end discovery FDR unless true q<=0.1 is clean.",
            }
        ]
    )


def classify_dataset(sub: pd.DataFrame, top10: dict[str, object]) -> str:
    true_fdr = int((sub["true_lr"].astype(bool) & sub["conservative_global_bh_fdr"].le(0.1)).sum())
    top10_true = int(top10.get("n_true_lr", 0))
    top10_risk = float(top10.get("negative_or_risk_fraction", np.nan))
    if true_fdr > 0 and np.isfinite(top10_risk) and top10_risk <= 0.1:
        return "spatial_specificity_fdr_candidate"
    if top10_true >= 5 and np.isfinite(top10_risk) and top10_risk <= 0.1:
        return "spatial_specificity_strong_ranking_fdr_not_closed"
    if top10_true >= 3:
        return "spatial_specificity_ranking_support_fdr_not_closed"
    return "spatial_specificity_mixed_or_negative"


def range_from_mechanism(mechanism: str) -> float:
    mechanism = str(mechanism)
    if mechanism == "long_range_chemokine_niche":
        return 700.0
    if mechanism == "juxtacrine_contact":
        return 220.0
    if mechanism == "short_medium_paracrine":
        return 420.0
    return 420.0


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, null_summary: pd.DataFrame, story: pd.DataFrame) -> None:
    first = story.iloc[0] if not story.empty else None
    lines = [
        "# Phase 12 Spatial-Specificity Permutation Audit",
        "",
        f"- Overall: `{first.overall_call if first is not None else 'NA'}`.",
        f"- Total true q<=0.1: {int(first.total_true_q_0_1) if first is not None else 0}.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.spatial_specificity_permutation_call}`: "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={perm.fmt(row.top10_negative_or_risk_fraction)}, "
            f"true q<=0.1={int(row.all_true_lr_conservative_fdr_0_1)}, "
            f"best true={row.best_true_lr} q={perm.fmt(row.best_true_lr_fdr)}, "
            f"best spatial={row.best_true_spatial_lr} spatial-q={perm.fmt(row.best_true_spatial_fdr)}."
        )
    lines += ["", "## Null Summary", ""]
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.null_type}`: null q99={perm.fmt(row.null_score_q99)}, "
            f"null max={perm.fmt(row.null_score_max)}, observed top={perm.fmt(row.observed_top_score)}, "
            f"top>q99={bool(row.observed_top_exceeds_null_q99)}, top>max={bool(row.observed_top_exceeds_null_max)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is a post-hoc exact-score audit for a spatial-specificity calibration layer.",
        "- It is not an end-to-end retrained SpatialSelf FDR procedure.",
    ]
    (OUT_DIR / "phase12_spatial_specificity_permutation_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_specificity_permutation_audit",
        "n_permutations_per_null": N_PERMUTATIONS,
        "null_types": list(NULL_TYPES),
        "inputs": [
            rel(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv"),
            rel(EXPANDED_DIR / "replicates/*/candidate_interactions.csv"),
            rel(EXPANDED_DIR / "replicates/*/lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_lr_fdr.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_topk_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_null_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_story.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_permutation_status.md"),
        ],
        "claim_boundary": "Post-hoc exact-score spatial-specificity audit, not end-to-end discovery FDR.",
    }
    (OUT_DIR / "phase12_spatial_specificity_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
