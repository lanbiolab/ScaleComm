#!/usr/bin/env python
"""Permutation audit for calibrated expanded Phase 12 atlas ranking.

This script adds a post-hoc null audit for the expanded calibrated atlas
ranking. It does not retrain SpatialSelf under each permutation. Instead, it
keeps the trained model scores fixed and recomputes the atlas calibration layer
under three nulls:

1. receiver permutation: break sender-receiver expression pairing;
2. spatial-distance permutation: break observed distance/range coupling;
3. LR-identity permutation: break alignment between model score and LR support.

For each null, the script reports both within-LR empirical p-values and a global
whole-atlas empirical p-value using all permuted LR scores as the null
distribution. The global p-values are BH-adjusted per dataset.
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

from scripts.phase12.calibrate_phase12_expanded_atlas_ranking import add_calibration


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = EXPANDED_DIR / "permutation_audit"
TOP_K_VALUES = (10, 20, 50, 100)
N_PERMUTATIONS = 80
RNG_SEED = 12122026
NULL_TYPES = ("receiver_permutation", "spatial_distance_permutation", "lr_identity_permutation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    status = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_status.csv")
    lr_rows = []
    null_summary_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        dataset_ranking = ranking[ranking["dataset_key"].eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        replicate_dir = EXPANDED_DIR / "replicates" / dataset_key
        result = run_dataset(dataset_key, dataset_ranking, replicate_dir)
        lr_rows.append(result["lr_fdr"])
        null_summary_rows.append(result["null_summary"])

    lr_fdr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null_summary = pd.concat(null_summary_rows, ignore_index=True) if null_summary_rows else pd.DataFrame()
    topk = summarize_topk(lr_fdr)
    dataset = summarize_dataset(lr_fdr, topk, null_summary, status)

    lr_fdr.to_csv(OUT_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_expanded_atlas_permutation_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_expanded_atlas_permutation_null_summary.csv", index=False)
    write_status(dataset, topk, null_summary)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr_fdr.groupby("dataset_key").head(15).to_string(index=False))


def run_dataset(dataset_key: str, ranking: pd.DataFrame, replicate_dir: Path) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = prepare_base_table(ranking, lr_pairs)
    candidate_arrays = prepare_candidate_arrays(candidates, base)
    observed_score = base["calibrated_score"].to_numpy(dtype=float)
    observed_strict_score = base["strict_discovery_score"].to_numpy(dtype=float)

    null_scores_by_type: dict[str, np.ndarray] = {}
    strict_null_scores_by_type: dict[str, np.ndarray] = {}
    null_summary_rows = []
    rng = np.random.default_rng(RNG_SEED + stable_dataset_offset(dataset_key))
    for null_type in NULL_TYPES:
        scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        strict_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            null_table = build_null_table(base, candidate_arrays, null_type, rng)
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
                "observed_top_score": float(np.max(observed_score)),
                "observed_top_exceeds_null_q99": bool(np.max(observed_score) > np.quantile(flat, 0.99)),
                "observed_top_exceeds_null_max": bool(np.max(observed_score) > np.max(flat)),
                "strict_null_score_mean": float(np.mean(strict_flat)),
                "strict_null_score_q95": float(np.quantile(strict_flat, 0.95)),
                "strict_null_score_q99": float(np.quantile(strict_flat, 0.99)),
                "strict_null_score_max": float(np.max(strict_flat)),
                "observed_top_strict_score": float(np.max(observed_strict_score)),
                "observed_top_strict_exceeds_null_q99": bool(
                    np.max(observed_strict_score) > np.quantile(strict_flat, 0.99)
                ),
                "observed_top_strict_exceeds_null_max": bool(np.max(observed_strict_score) > np.max(strict_flat)),
            }
        )

    lr_fdr = build_lr_fdr_table(dataset_key, base, null_scores_by_type, strict_null_scores_by_type)
    return {"lr_fdr": lr_fdr, "null_summary": pd.DataFrame(null_summary_rows)}


def prepare_base_table(ranking: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    source = lr_pairs[["lr_id", "source_count"]].copy()
    base = ranking.merge(source, on="lr_id", how="left")
    # add_calibration expects these names and uses them to recompute the active
    # score under permuted support features.
    base = base.rename(
        columns={
            "raw_mean_score": "raw_mean_score",
            "raw_q95_score": "raw_q95_score",
        }
    )
    required = [
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "raw_mean_score",
        "raw_q95_score",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "n_nonzero_expr",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "source_count",
        "rank",
        "strict_rank",
        "rank_q95",
        "calibrated_score",
        "strict_discovery_score",
    ]
    missing = [col for col in required if col not in base.columns]
    if missing:
        raise ValueError(f"Calibrated ranking is missing required columns: {missing}")
    base["source_count"] = pd.to_numeric(base["source_count"], errors="coerce").fillna(1)
    return base[required].reset_index(drop=True)


def prepare_candidate_arrays(candidates: pd.DataFrame, base: pd.DataFrame) -> dict[str, np.ndarray | list[np.ndarray]]:
    lr_to_code = {lr_id: idx for idx, lr_id in enumerate(base["lr_id"].astype(str))}
    lr_code = candidates["lr_id"].astype(str).map(lr_to_code).to_numpy()
    valid = pd.notna(lr_code)
    if not bool(np.all(valid)):
        candidates = candidates.loc[valid].copy()
        lr_code = candidates["lr_id"].astype(str).map(lr_to_code).to_numpy()
    lr_code = lr_code.astype(int)
    order = np.argsort(lr_code)
    counts = np.bincount(lr_code, minlength=len(base)).astype(float)
    spatial_range_by_lr = base["mechanism"].map(range_from_mechanism).to_numpy(dtype=float)
    return {
        "lr_code": lr_code,
        "order": order,
        "counts": counts,
        "ligand_expr": candidates["ligand_expr"].to_numpy(dtype=float),
        "receptor_expr": candidates["receptor_expr"].to_numpy(dtype=float),
        "distance": candidates["distance"].to_numpy(dtype=float),
        "type_rule": candidates["type_rule_satisfied"].astype(bool).to_numpy(),
        "spatial_range_by_candidate": spatial_range_by_lr[lr_code],
    }


def build_null_table(
    base: pd.DataFrame,
    arrays: dict[str, np.ndarray | list[np.ndarray]],
    null_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if null_type == "lr_identity_permutation":
        return build_lr_identity_null(base, rng)

    lr_code = arrays["lr_code"]  # type: ignore[assignment]
    order = arrays["order"]  # type: ignore[assignment]
    counts = arrays["counts"]  # type: ignore[assignment]
    ligand = arrays["ligand_expr"]  # type: ignore[assignment]
    receptor = arrays["receptor_expr"]  # type: ignore[assignment]
    distance = arrays["distance"]  # type: ignore[assignment]
    type_rule = arrays["type_rule"]  # type: ignore[assignment]
    spatial_range = arrays["spatial_range_by_candidate"]  # type: ignore[assignment]

    ligand = np.asarray(ligand, dtype=float)
    receptor = np.asarray(receptor, dtype=float)
    distance = np.asarray(distance, dtype=float)
    lr_code = np.asarray(lr_code, dtype=int)
    counts = np.asarray(counts, dtype=float)
    order = np.asarray(order, dtype=int)
    type_rule = np.asarray(type_rule, dtype=bool)
    spatial_range = np.asarray(spatial_range, dtype=float)

    if null_type == "receiver_permutation":
        receptor_use = rng.permutation(receptor)
        distance_use = distance
        range_rule = distance <= spatial_range
    elif null_type == "spatial_distance_permutation":
        receptor_use = receptor
        distance_use = rng.permutation(distance)
        range_rule = distance_use <= spatial_range
    else:
        raise ValueError(f"Unknown null type: {null_type}")

    coexpr = np.log1p(ligand) * np.log1p(receptor_use) / (1.0 + distance_use)
    n_lr = len(base)
    n_nonzero = np.bincount(lr_code, weights=(coexpr > 0).astype(float), minlength=n_lr)
    mean_coexpr = np.bincount(lr_code, weights=coexpr, minlength=n_lr) / np.maximum(counts, 1.0)
    q95_coexpr = grouped_quantile(coexpr, lr_code, order, n_lr, 0.95)
    ligand_pos = np.bincount(lr_code, weights=(ligand > 0).astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    receptor_pos = np.bincount(lr_code, weights=(receptor_use > 0).astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    type_fraction = np.bincount(lr_code, weights=type_rule.astype(float), minlength=n_lr) / np.maximum(counts, 1.0)
    range_fraction = np.bincount(lr_code, weights=range_rule.astype(float), minlength=n_lr) / np.maximum(counts, 1.0)

    null = base.copy()
    null["n_nonzero_expr"] = n_nonzero.astype(int)
    null["ligand_positive_fraction"] = ligand_pos
    null["receptor_positive_fraction"] = receptor_pos
    null["mean_coexpression_distance"] = mean_coexpr
    null["q95_coexpression_distance"] = q95_coexpr
    null["rule_satisfied_fraction"] = type_fraction
    null["range_satisfied_fraction"] = range_fraction
    null = add_calibration(null)
    return null


def build_lr_identity_null(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    null = base.copy()
    perm = rng.permutation(len(null))
    support_cols = [
        "n_nonzero_expr",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "source_count",
        "biological_category",
        "false_positive_high_expression",
    ]
    for col in support_cols:
        null[col] = null[col].to_numpy()[perm]
    null = add_calibration(null)
    return null


def build_lr_fdr_table(
    dataset_key: str,
    base: pd.DataFrame,
    null_scores_by_type: dict[str, np.ndarray],
    strict_null_scores_by_type: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = base.copy()
    observed = out["calibrated_score"].to_numpy(dtype=float)
    strict_observed = out["strict_discovery_score"].to_numpy(dtype=float)
    conservative_global_p = np.zeros(len(out), dtype=float)
    conservative_within_p = np.zeros(len(out), dtype=float)
    for null_type, scores in null_scores_by_type.items():
        global_null = scores.ravel()
        global_p = np.asarray(
            [(1.0 + np.sum(global_null >= score)) / (1.0 + len(global_null)) for score in observed],
            dtype=float,
        )
        within_p = np.asarray(
            [(1.0 + np.sum(scores[:, idx] >= observed[idx])) / (1.0 + scores.shape[0]) for idx in range(len(out))],
            dtype=float,
        )
        out[f"{null_type}_global_empirical_p"] = global_p
        out[f"{null_type}_within_lr_empirical_p"] = within_p
        out[f"{null_type}_global_bh_fdr"] = bh_adjust(global_p)
        conservative_global_p = np.maximum(conservative_global_p, global_p)
        conservative_within_p = np.maximum(conservative_within_p, within_p)
    out["conservative_global_empirical_p"] = conservative_global_p
    out["conservative_global_bh_fdr"] = bh_adjust(conservative_global_p)
    out["conservative_within_lr_empirical_p"] = conservative_within_p
    strict_conservative_global_p = np.zeros(len(out), dtype=float)
    strict_conservative_within_p = np.zeros(len(out), dtype=float)
    for null_type, scores in strict_null_scores_by_type.items():
        global_null = scores.ravel()
        global_p = np.asarray(
            [(1.0 + np.sum(global_null >= score)) / (1.0 + len(global_null)) for score in strict_observed],
            dtype=float,
        )
        within_p = np.asarray(
            [(1.0 + np.sum(scores[:, idx] >= strict_observed[idx])) / (1.0 + scores.shape[0]) for idx in range(len(out))],
            dtype=float,
        )
        out[f"strict_{null_type}_global_empirical_p"] = global_p
        out[f"strict_{null_type}_within_lr_empirical_p"] = within_p
        out[f"strict_{null_type}_global_bh_fdr"] = bh_adjust(global_p)
        strict_conservative_global_p = np.maximum(strict_conservative_global_p, global_p)
        strict_conservative_within_p = np.maximum(strict_conservative_within_p, within_p)
    out["strict_conservative_global_empirical_p"] = strict_conservative_global_p
    out["strict_conservative_global_bh_fdr"] = bh_adjust(strict_conservative_global_p)
    out["strict_conservative_within_lr_empirical_p"] = strict_conservative_within_p
    out["permutation_support_class"] = [
        classify_lr(row) for row in out.itertuples(index=False)
    ]
    out["strict_permutation_support_class"] = [
        classify_strict_lr(row) for row in out.itertuples(index=False)
    ]
    out["dataset_key"] = dataset_key
    out["method"] = "spatialself_wavelet_edge_prior_calibrated_permutation_audit"
    columns = [
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
        columns += [
            f"{null_type}_global_empirical_p",
            f"{null_type}_global_bh_fdr",
            f"{null_type}_within_lr_empirical_p",
            f"strict_{null_type}_global_empirical_p",
            f"strict_{null_type}_global_bh_fdr",
            f"strict_{null_type}_within_lr_empirical_p",
        ]
    return out[columns].sort_values(["rank", "lr_pair"]).reset_index(drop=True)


def summarize_topk(lr_fdr: pd.DataFrame) -> pd.DataFrame:
    if lr_fdr.empty:
        return pd.DataFrame()
    rows = []
    annotated = annotate_risk(lr_fdr)
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
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
                    "n_conservative_fdr_0_05": int(top["conservative_global_bh_fdr"].le(0.05).sum()),
                    "n_conservative_fdr_0_1": int(top["conservative_global_bh_fdr"].le(0.1).sum()),
                    "n_conservative_fdr_0_25": int(top["conservative_global_bh_fdr"].le(0.25).sum()),
                    "median_conservative_fdr": float(top["conservative_global_bh_fdr"].median()),
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
    annotated = annotate_risk(lr_fdr)
    for dataset_key, sub in annotated.groupby("dataset_key"):
        t10 = one_row(topk, dataset_key, 10)
        t50 = one_row(topk, dataset_key, 50)
        stat = status[status["dataset_key"].eq(dataset_key)].head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "n_permutations_per_null": N_PERMUTATIONS,
                "n_null_types": len(NULL_TYPES),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_negative_or_risk_fraction": float(t10.get("negative_or_risk_fraction", np.nan)),
                "top10_conservative_fdr_0_05": int(t10.get("n_conservative_fdr_0_05", 0)),
                "top10_conservative_fdr_0_1": int(t10.get("n_conservative_fdr_0_1", 0)),
                "top10_median_conservative_fdr": float(t10.get("median_conservative_fdr", np.nan)),
                "top50_conservative_fdr_0_1": int(t50.get("n_conservative_fdr_0_1", 0)),
                "all_lr_conservative_fdr_0_1": int(sub["conservative_global_bh_fdr"].le(0.1).sum()),
                "all_true_lr_conservative_fdr_0_1": int(
                    (sub["true_lr"].astype(bool) & sub["conservative_global_bh_fdr"].le(0.1)).sum()
                ),
                "best_true_lr": best_true_lr(sub),
                "best_true_lr_fdr": best_true_lr_fdr(sub),
                "runtime_seconds": float(stat["runtime_seconds"].iloc[0]) if len(stat) else np.nan,
                "n_candidates": int(stat["n_candidates"].iloc[0]) if len(stat) else 0,
                "permutation_audit_call": classify_dataset(t10, sub),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def annotate_risk(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["true_lr"] = out["true_lr"].astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].astype(bool)
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for dataset_key, sub in out.groupby("dataset_key"):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors) & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands) & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = out["false_positive_high_expression"] | out["receptor_shared_risk"] | out["ligand_shared_risk"]
    return out


def classify_lr(row: object) -> str:
    fdr = to_float(getattr(row, "conservative_global_bh_fdr", np.nan))
    within = to_float(getattr(row, "conservative_within_lr_empirical_p", np.nan))
    if np.isfinite(fdr) and fdr <= 0.1 and np.isfinite(within) and within <= 0.25:
        return "global_fdr_and_within_lr_supported"
    if np.isfinite(fdr) and fdr <= 0.1:
        return "global_fdr_supported_with_within_lr_caveat"
    if np.isfinite(fdr) and fdr <= 0.25:
        return "suggestive_global_fdr"
    return "not_permutation_supported"


def classify_strict_lr(row: object) -> str:
    fdr = to_float(getattr(row, "strict_conservative_global_bh_fdr", np.nan))
    within = to_float(getattr(row, "strict_conservative_within_lr_empirical_p", np.nan))
    if np.isfinite(fdr) and fdr <= 0.1 and np.isfinite(within) and within <= 0.25:
        return "strict_global_fdr_and_within_lr_supported"
    if np.isfinite(fdr) and fdr <= 0.1:
        return "strict_global_fdr_supported_with_within_lr_caveat"
    if np.isfinite(fdr) and fdr <= 0.25:
        return "strict_suggestive_global_fdr"
    return "strict_not_permutation_supported"


def classify_dataset(top10: dict[str, object], sub: pd.DataFrame) -> str:
    n_top10_fdr = int(top10.get("n_conservative_fdr_0_1", 0))
    top10_true = int(top10.get("n_true_lr", 0))
    top10_risk = float(top10.get("negative_or_risk_fraction", np.nan))
    n_all_fdr = int(sub["conservative_global_bh_fdr"].le(0.1).sum())
    if n_top10_fdr >= 3 and top10_true >= 3 and top10_risk <= 0.25:
        return "expanded_permutation_rank_support"
    if n_all_fdr > 0 and top10_risk <= 0.35:
        return "expanded_permutation_partial_support"
    return "expanded_permutation_not_yet_supportive"


def grouped_quantile(values: np.ndarray, codes: np.ndarray, order: np.ndarray, n_groups: int, q: float) -> np.ndarray:
    sorted_codes = codes[order]
    sorted_values = values[order]
    out = np.zeros(n_groups, dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], len(sorted_codes)]
    for start, end in zip(starts, ends):
        code = int(sorted_codes[start])
        out[code] = float(np.quantile(sorted_values[start:end], q))
    return out


def range_from_mechanism(mechanism: str) -> float:
    mechanism = str(mechanism)
    if mechanism == "long_range_chemokine_niche":
        return 700.0
    if mechanism == "juxtacrine_contact":
        return 220.0
    if mechanism == "short_medium_paracrine":
        return 420.0
    return 420.0


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


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, null_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Atlas Permutation Audit",
        "",
        "## Scope",
        "",
        "This is a post-hoc permutation audit for the calibrated expanded atlas ranking. It recomputes the calibration layer under receiver, spatial-distance and LR-identity nulls without retraining SpatialSelf.",
        "Global empirical p-values compare observed LR scores with all permuted LR scores in the same dataset/null; BH FDR is adjusted across the expanded LR universe per dataset.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.permutation_audit_call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 q<=0.1={int(row.top10_conservative_fdr_0_1)}, "
            f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
            f"best true LR={row.best_true_lr} q={fmt(row.best_true_lr_fdr)}, "
            f"top10 risk={fmt(row.top10_negative_or_risk_fraction)}"
        )
    lines += ["", "## Top-K Summary", ""]
    for row in topk[topk["top_k"].isin([10, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"risk={fmt(row.negative_or_risk_fraction)}, q<=0.1={int(row.n_conservative_fdr_0_1)}, "
            f"median q={fmt(row.median_conservative_fdr)}"
        )
    lines += ["", "## Null Summary", ""]
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.null_type}`: null q99={fmt(row.null_score_q99)}, "
            f"null max={fmt(row.null_score_max)}, observed top={fmt(row.observed_top_score)}, "
            f"top>q99={bool(row.observed_top_exceeds_null_q99)}, top>max={bool(row.observed_top_exceeds_null_max)}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This strengthens the calibrated expanded ranking with explicit permutation nulls.",
        "- It is still post-hoc: the SpatialSelf model itself was not retrained under each null.",
        "- Do not call this causal validation or final end-to-end permutation FDR.",
    ]
    (OUT_DIR / "phase12_expanded_atlas_permutation_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_atlas_permutation_audit",
        "n_permutations_per_null": N_PERMUTATIONS,
        "null_types": list(NULL_TYPES),
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            "permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "permutation_audit/phase12_expanded_atlas_permutation_topk_summary.csv",
            "permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "permutation_audit/phase12_expanded_atlas_permutation_null_summary.csv",
            "permutation_audit/phase12_expanded_atlas_permutation_audit_status.md",
        ],
        "claim_boundary": "Post-hoc permutation audit for calibrated expanded ranking; not end-to-end retraining permutation FDR.",
    }
    (OUT_DIR / "phase12_expanded_atlas_permutation_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values(["conservative_global_bh_fdr", "rank"])
    return str(true.iloc[0]["lr_pair"]) if len(true) else ""


def best_true_lr_fdr(sub: pd.DataFrame) -> float:
    true = sub[sub["true_lr"].astype(bool)].sort_values(["conservative_global_bh_fdr", "rank"])
    return float(true.iloc[0]["conservative_global_bh_fdr"]) if len(true) else np.nan


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    if table.empty:
        return {}
    sub = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def stable_dataset_offset(dataset_key: str) -> int:
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(dataset_key)) % 100_000)


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if float(denom) else float("nan")


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


if __name__ == "__main__":
    main()
