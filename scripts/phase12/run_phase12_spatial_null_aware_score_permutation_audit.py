#!/usr/bin/env python
"""Exact-score permutation audit for the spatial-null-aware score prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm_audit


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "spatial_null_aware_score_permutation_audit"
N_PERMUTATIONS = 80
NULL_TYPES = ("receiver_permutation", "spatial_distance_permutation", "lr_identity_permutation")
RNG_SEED = 12122026 + 991


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    rows = []
    null_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        result = run_dataset(dataset_key, ranking[ranking["dataset_key"].eq(dataset_key)].copy())
        rows.append(result["lr_fdr"])
        null_rows.append(result["null_summary"])
    lr_fdr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    null_summary = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    dataset = summarize_dataset(lr_fdr)
    story = summarize_story(dataset)

    lr_fdr.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_permutation_lr_fdr.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_permutation_null_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_permutation_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_permutation_story.csv", index=False)
    write_status(dataset, story)
    manifest = {
        "task": "phase12_spatial_null_aware_score_permutation_audit",
        "n_permutations": N_PERMUTATIONS,
        "null_types": list(NULL_TYPES),
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_null_aware_score_permutation_lr_fdr.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_permutation_null_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_permutation_dataset.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_permutation_story.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_permutation_status.md"),
        ],
        "claim_boundary": "Exact-score null audit for an exploratory score-level prototype; not a retrained model.",
    }
    (OUT_DIR / "phase12_spatial_null_aware_score_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(dataset.to_string(index=False))
    print(story.to_string(index=False))


def run_dataset(dataset_key: str, ranking: pd.DataFrame) -> dict[str, pd.DataFrame]:
    replicate_dir = EXPANDED_DIR / "replicates" / dataset_key
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = perm_audit.prepare_base_table(ranking.sort_values("rank").reset_index(drop=True), lr_pairs)
    base = add_score_features(base)
    arrays = perm_audit.prepare_candidate_arrays(candidates, base)
    observed = spatial_null_aware_score(base)

    rng = np.random.default_rng(RNG_SEED + perm_audit.stable_dataset_offset(dataset_key))
    null_scores_by_type: dict[str, np.ndarray] = {}
    null_summary_rows = []
    for null_type in NULL_TYPES:
        scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            null_table = perm_audit.build_null_table(base, arrays, null_type, rng)
            null_table = add_score_features(null_table)
            scores[perm_idx, :] = spatial_null_aware_score(null_table)
        null_scores_by_type[null_type] = scores
        flat = scores.ravel()
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
            }
        )

    lr_fdr = build_fdr_table(dataset_key, base, observed, null_scores_by_type)
    return {"lr_fdr": lr_fdr, "null_summary": pd.DataFrame(null_summary_rows)}


def add_score_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in [
        "calibrated_score",
        "strict_discovery_score",
        "n_nonzero_expr",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "source_count",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    source_support = (np.log1p(out["source_count"].clip(lower=0.0)) / np.log1p(12)).clip(0, 1)
    expr_support = np.sqrt(
        out["ligand_positive_fraction"].clip(0, 1)
        * out["receptor_positive_fraction"].clip(0, 1)
    )
    observed_pair_support = (np.log1p(out["n_nonzero_expr"].clip(lower=0.0)) / np.log1p(60)).clip(0, 1)
    out["prototype_source_support"] = source_support
    out["prototype_expr_support"] = (0.45 * expr_support + 0.55 * observed_pair_support).clip(0, 1)
    out["prototype_broad_other"] = (
        out["selected_reason"].astype(str).eq("category_balanced")
        & out["biological_category"].astype(str).eq("other")
    )
    out["prototype_category_balanced"] = out["selected_reason"].astype(str).eq("category_balanced")
    return out


def spatial_null_aware_score(table: pd.DataFrame) -> np.ndarray:
    source = table["prototype_source_support"].to_numpy(dtype=float)
    expr = table["prototype_expr_support"].to_numpy(dtype=float)
    spatial_rule = (
        0.65 * table["rule_satisfied_fraction"].to_numpy(dtype=float)
        + 0.35 * table["range_satisfied_fraction"].to_numpy(dtype=float)
    ).clip(0, 1)
    coexpr = table["mean_coexpression_distance"].to_numpy(dtype=float)
    q95 = table["q95_coexpression_distance"].to_numpy(dtype=float)
    base = (
        0.38 * pct(table["strict_discovery_score"].to_numpy(dtype=float))
        + 0.18 * pct(table["calibrated_score"].to_numpy(dtype=float))
        + 0.18 * pct(coexpr)
        + 0.08 * pct(q95)
        + 0.10 * spatial_rule
        + 0.08 * source
    )
    source_gate = np.power(np.clip(0.12 + 0.88 * source, 0, 1), 1.8)
    expr_gate = np.power(np.clip(0.08 + 0.92 * expr, 0, 1), 1.35)
    broad_penalty = np.where(table["prototype_broad_other"].to_numpy(dtype=bool), 0.04, 1.0)
    category_penalty = np.where(
        table["prototype_category_balanced"].to_numpy(dtype=bool)
        & ~table["prototype_broad_other"].to_numpy(dtype=bool),
        np.where(source < 0.45, 0.20, 0.48),
        1.0,
    )
    hard_penalty = np.where(table["false_positive_high_expression"].astype(bool).to_numpy(), 0.03, 1.0)
    low_expr_penalty = np.where(table["n_nonzero_expr"].to_numpy(dtype=float) <= 0, 0.02, 1.0)
    return base * source_gate * expr_gate * broad_penalty * category_penalty * hard_penalty * low_expr_penalty


def build_fdr_table(
    dataset_key: str,
    base: pd.DataFrame,
    observed: np.ndarray,
    null_scores_by_type: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = base.copy()
    out["spatial_null_aware_score"] = observed
    conservative_p = np.zeros(len(out), dtype=float)
    within_p = np.zeros(len(out), dtype=float)
    for null_type, scores in null_scores_by_type.items():
        flat = scores.ravel()
        global_p = np.asarray([(1 + np.sum(flat >= score)) / (1 + len(flat)) for score in observed])
        per_lr_p = np.asarray([(1 + np.sum(scores[:, idx] >= observed[idx])) / (1 + scores.shape[0]) for idx in range(len(out))])
        out[f"{null_type}_global_empirical_p"] = global_p
        out[f"{null_type}_within_lr_empirical_p"] = per_lr_p
        out[f"{null_type}_global_bh_fdr"] = bh(global_p)
        conservative_p = np.maximum(conservative_p, global_p)
        within_p = np.maximum(within_p, per_lr_p)
    out["conservative_global_empirical_p"] = conservative_p
    out["conservative_global_bh_fdr"] = bh(conservative_p)
    out["conservative_within_lr_empirical_p"] = within_p
    out["spatial_null_aware_rank"] = out["spatial_null_aware_score"].rank(ascending=False, method="min").astype(int)
    out["dataset_key"] = dataset_key
    out["prototype_support_class"] = [
        classify_lr(row) for row in out.itertuples(index=False)
    ]
    keep = [
        "dataset_key",
        "spatial_null_aware_rank",
        "rank",
        "strict_rank",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "spatial_null_aware_score",
        "conservative_global_empirical_p",
        "conservative_global_bh_fdr",
        "conservative_within_lr_empirical_p",
        "prototype_support_class",
    ]
    for null_type in NULL_TYPES:
        keep.extend(
            [
                f"{null_type}_global_empirical_p",
                f"{null_type}_global_bh_fdr",
                f"{null_type}_within_lr_empirical_p",
            ]
        )
    return out[keep].sort_values(["spatial_null_aware_rank", "lr_pair"]).reset_index(drop=True)


def summarize_dataset(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annotated = annotate_risk(table)
    for dataset_key, group in annotated.groupby("dataset_key"):
        ranked = group.sort_values("spatial_null_aware_rank")
        top10 = ranked.head(10)
        supported = group[group["conservative_global_bh_fdr"].le(0.1)]
        true_supported = supported[supported["true_lr"].astype(bool)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": len(group),
                "top10_true_lr": int(top10["true_lr"].astype(bool).sum()),
                "top10_negative_or_risk": int(top10["negative_or_risk"].sum()),
                "top10_broad_other": int(
                    (
                        top10["selected_reason"].astype(str).eq("category_balanced")
                        & top10["biological_category"].astype(str).eq("other")
                    ).sum()
                ),
                "all_lr_q_0_1": len(supported),
                "true_lr_q_0_1": int(true_supported.shape[0]),
                "negative_or_risk_q_0_1": int(supported["negative_or_risk"].sum()),
                "best_true_lr": best_true_lr(ranked),
                "best_true_q": best_true_q(ranked),
                "dataset_call": dataset_call(supported, top10),
                "top_lr_pairs": ";".join(ranked["lr_pair"].astype(str).head(12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(summary: pd.DataFrame) -> pd.DataFrame:
    total_true = int(summary["true_lr_q_0_1"].sum()) if not summary.empty else 0
    total_risk = int(summary["negative_or_risk_q_0_1"].sum()) if not summary.empty else 0
    if total_true > 0 and total_risk == 0:
        call = "prototype_exact_score_null_partially_supportive"
    else:
        call = "prototype_exact_score_null_not_closed"
    return pd.DataFrame(
        [
            {
                "topic": "spatial_null_aware_exact_score_permutation",
                "overall_call": call,
                "n_datasets": len(summary),
                "n_datasets_with_true_q_0_1": int((summary["true_lr_q_0_1"] > 0).sum()) if not summary.empty else 0,
                "total_true_lr_q_0_1": total_true,
                "total_negative_or_risk_q_0_1": total_risk,
                "claim_boundary": (
                    "Exact-score null audit for an exploratory score-level prototype. "
                    "It can inform next-model design, but it is not a retrained SpatialSelf model."
                ),
            }
        ]
    )


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
            out.loc[mask, "receptor"].astype(str).isin(true_receptors)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = (
        out["false_positive_high_expression"]
        | out["receptor_shared_risk"]
        | out["ligand_shared_risk"]
    )
    return out


def dataset_call(supported: pd.DataFrame, top10: pd.DataFrame) -> str:
    if len(supported) and supported["true_lr"].astype(bool).sum() and not supported["negative_or_risk"].sum():
        return "exact_score_true_support_without_risk"
    if len(supported):
        return "exact_score_support_contaminated_or_background"
    if top10["true_lr"].astype(bool).sum() >= 5 and not top10["negative_or_risk"].sum():
        return "rank_improved_no_fdr_closure"
    return "not_closed"


def classify_lr(row: object) -> str:
    q = float(getattr(row, "conservative_global_bh_fdr", np.nan))
    within = float(getattr(row, "conservative_within_lr_empirical_p", np.nan))
    if np.isfinite(q) and q <= 0.1 and np.isfinite(within) and within <= 0.25:
        return "global_and_within_supported"
    if np.isfinite(q) and q <= 0.1:
        return "global_supported_with_within_caveat"
    return "not_supported"


def best_true_lr(ranked: pd.DataFrame) -> str:
    true_rows = ranked[ranked["true_lr"].astype(bool)]
    if true_rows.empty:
        return "none"
    return str(true_rows.iloc[0]["lr_pair"])


def best_true_q(ranked: pd.DataFrame) -> float:
    true_rows = ranked[ranked["true_lr"].astype(bool)]
    if true_rows.empty:
        return np.nan
    return float(true_rows.iloc[0]["conservative_global_bh_fdr"])


def pct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks / max(len(values), 1)


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = p[order[i]] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q


def write_status(summary: pd.DataFrame, story: pd.DataFrame) -> None:
    overall = story.iloc[0]
    lines = [
        "# Phase 12 Spatial-Null-Aware Exact-Score Permutation Audit",
        "",
        "## Overall",
        "",
        f"- Overall call: `{overall.overall_call}`.",
        f"- Datasets with true q<=0.1: {int(overall.n_datasets_with_true_q_0_1)}/{int(overall.n_datasets)}.",
        f"- Total true LR q<=0.1: {int(overall.total_true_lr_q_0_1)}.",
        f"- Total negative/risk q<=0.1: {int(overall.total_negative_or_risk_q_0_1)}.",
        "",
        "## Dataset Calls",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_call}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 risk={int(row.top10_negative_or_risk)}, all q<=0.1={int(row.all_lr_q_0_1)}, "
            f"true q<=0.1={int(row.true_lr_q_0_1)}, risk q<=0.1={int(row.negative_or_risk_q_0_1)}, "
            f"best true={row.best_true_lr} q={fmt_num(row.best_true_q)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: the exact-score prototype audit tests a concrete next-model scoring direction against the same null classes.",
        "- Unsafe wording: this is the final SpatialSelf model or a completed broad whole-atlas FDR result.",
    ]
    (OUT_DIR / "phase12_spatial_null_aware_score_permutation_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def fmt_num(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
