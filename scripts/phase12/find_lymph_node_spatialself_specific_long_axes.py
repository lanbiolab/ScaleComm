#!/usr/bin/env python
"""Find SpatialSelf-prioritized long-range axes in native human lymph node."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/phase12/lymph_node_spatialself_specific_long_axes"

SPATIALSELF_RANKING = (
    ROOT
    / "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv"
)
OFFICIAL_BASELINES = (
    ROOT
    / "outputs/phase12/atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_lr_ranking.csv"
)
DISCOVERY_FDR = (
    ROOT / "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_all_lr.csv"
)
STRICT_EVIDENCE = ROOT / "outputs/phase12/strict_story_candidates/phase12_strict_story_candidate_evidence.csv"
DOWNSTREAM_AUDIT = (
    ROOT / "outputs/phase12/downstream_target_background_audit/phase12_downstream_target_background_lr_audit.csv"
)

DATASET_KEY = "human_lymph_node_visium_raw_full"
OFFICIAL_METHOD_ORDER = ["commot", "cellchat", "spatalk", "cellnest"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = _load_spatialself_candidates()
    candidates = _attach_official_baselines(candidates)
    candidates = _attach_discovery_fdr(candidates)
    candidates = _attach_downstream(candidates)
    candidates = _attach_strict_story_evidence(candidates)
    candidates = _score_candidates(candidates)

    long_table = candidates[
        candidates["mechanism"].eq("long_range_chemokine_niche")
        | candidates["biological_category"].eq("chemokine_recruitment")
    ].copy()
    long_table = long_table.sort_values(
        ["recommendation_tier", "spatialself_rank", "non_spatalk_best_baseline_rank"],
        ascending=[True, True, False],
    )

    candidates.to_csv(OUT / "lymph_node_spatialself_specific_all_candidate_axes.csv", index=False)
    long_table.to_csv(OUT / "lymph_node_spatialself_specific_long_axes.csv", index=False)
    _write_summary(long_table, OUT / "lymph_node_spatialself_specific_long_axes_summary.md")

    print(f"Wrote {OUT / 'lymph_node_spatialself_specific_long_axes.csv'}")
    print(f"Wrote {OUT / 'lymph_node_spatialself_specific_long_axes_summary.md'}")


def _load_spatialself_candidates() -> pd.DataFrame:
    df = pd.read_csv(SPATIALSELF_RANKING)
    df = df[df["dataset_key"].eq(DATASET_KEY)].copy()
    df = df.rename(
        columns={
            "rank": "spatialself_rank",
            "rank_q95": "spatialself_rank_q95",
            "calibrated_score": "spatialself_score",
            "calibrated_q95_score": "spatialself_q95_score",
        }
    )
    keep = [
        "dataset_key",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "spatialself_rank",
        "strict_rank",
        "spatialself_rank_q95",
        "raw_rank",
        "raw_rank_q95",
        "spatialself_score",
        "strict_discovery_score",
        "spatialself_q95_score",
        "n_candidates",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "range_satisfied_fraction",
        "expr_gate",
        "source_support",
        "calibration_note",
    ]
    return df[[c for c in keep if c in df.columns]]


def _attach_official_baselines(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(OFFICIAL_BASELINES)
    base = base[base["dataset_key"].eq(DATASET_KEY)].copy()
    rank_pivot = base.pivot_table(index="lr_pair", columns="method", values="rank", aggfunc="min")
    score_pivot = base.pivot_table(index="lr_pair", columns="method", values="score", aggfunc="max")
    rank_pivot = rank_pivot.rename(columns={m: f"{m}_rank" for m in rank_pivot.columns})
    score_pivot = score_pivot.rename(columns={m: f"{m}_score" for m in score_pivot.columns})
    out = df.merge(rank_pivot.reset_index(), on="lr_pair", how="left")
    out = out.merge(score_pivot.reset_index(), on="lr_pair", how="left")
    for method in OFFICIAL_METHOD_ORDER:
        col = f"{method}_rank"
        if col not in out:
            out[col] = np.nan
    rank_cols = [f"{m}_rank" for m in OFFICIAL_METHOD_ORDER]
    non_spatalk = [f"{m}_rank" for m in OFFICIAL_METHOD_ORDER if m != "spatalk"]
    out["all_official_best_baseline_rank"] = out[rank_cols].min(axis=1, skipna=True)
    out["non_spatalk_best_baseline_rank"] = out[non_spatalk].min(axis=1, skipna=True)
    out["n_official_baselines_top10"] = (out[rank_cols] <= 10).sum(axis=1)
    out["n_non_spatalk_baselines_top10"] = (out[non_spatalk] <= 10).sum(axis=1)
    out["spatalk_only_top10"] = (
        out["spatalk_rank"].le(10)
        & out["n_non_spatalk_baselines_top10"].eq(0)
    )
    return out


def _attach_discovery_fdr(df: pd.DataFrame) -> pd.DataFrame:
    if not DISCOVERY_FDR.exists():
        return df
    fdr = pd.read_csv(DISCOVERY_FDR)
    fdr = fdr[fdr["dataset_key"].eq(DATASET_KEY)].copy()
    cols = [
        "lr_pair",
        "discovery_eligible",
        "discovery_eligibility_reason",
        "eligible_pooled_all_null_bh_fdr",
        "eligible_strict_pooled_all_null_bh_fdr",
        "eligible_matched_all_null_bh_fdr",
        "eligible_strict_matched_all_null_bh_fdr",
        "eligible_matched_spatial_bh_fdr",
        "eligible_fdr_call",
    ]
    cols = [c for c in cols if c in fdr.columns]
    return df.merge(fdr[cols], on="lr_pair", how="left")


def _attach_downstream(df: pd.DataFrame) -> pd.DataFrame:
    if not DOWNSTREAM_AUDIT.exists():
        return df
    ds = pd.read_csv(DOWNSTREAM_AUDIT)
    ds = ds[ds["dataset_key"].eq(DATASET_KEY)].copy()
    cols = [
        "lr_pair",
        "n_targets_present",
        "target_top300_enrichment",
        "target_top300_enrichment_bh_fdr",
        "target_enrichment_minus_best_negative",
        "has_curated_target_set",
        "target_support_score",
        "target_background_call",
        "top300_mean_spearman",
        "best_target_gene",
        "best_target_gene_rank",
    ]
    cols = [c for c in cols if c in ds.columns]
    return df.merge(ds[cols], on="lr_pair", how="left")


def _attach_strict_story_evidence(df: pd.DataFrame) -> pd.DataFrame:
    if not STRICT_EVIDENCE.exists():
        return df
    ev = pd.read_csv(STRICT_EVIDENCE)
    ev = ev[ev["dataset_key"].eq(DATASET_KEY)].copy()
    cols = [
        "lr_pair",
        "coordinate_bh_fdr",
        "targeted_validation_call",
        "targeted_panel_bh_q",
        "matched_control_bh_q",
        "hard_negative_bh_q",
        "direction_receiver_pairing_q",
        "direction_vector_q",
        "direction_edge_call",
        "program_specificity_class",
        "best_specific_term",
        "best_specific_term_fdr",
        "evidence_score",
        "strict_story_class",
        "claim_boundary",
    ]
    cols = [c for c in cols if c in ev.columns]
    return df.merge(ev[cols], on="lr_pair", how="left")


def _score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spatialself_top10"] = out["spatialself_rank"].le(10)
    out["non_spatalk_baselines_after10"] = out["n_non_spatalk_baselines_top10"].eq(0)
    out["eligible_fdr_supported"] = out.get("eligible_fdr_call", "").fillna("").str.contains("supported")
    out["target_supported"] = out.get("target_background_call", "").fillna("").str.contains("support")
    out["direction_supported"] = out.get("direction_edge_call", "").fillna("").str.contains("supported")

    def tier(row: pd.Series) -> int:
        if (
            row["spatialself_top10"]
            and row["non_spatalk_baselines_after10"]
            and row["eligible_fdr_supported"]
            and (row["target_supported"] or row["direction_supported"])
        ):
            return 1
        if (
            row["spatialself_top10"]
            and row["non_spatalk_baselines_after10"]
            and row["eligible_fdr_supported"]
        ):
            return 2
        if row["spatialself_top10"] and row["non_spatalk_baselines_after10"]:
            return 3
        if row["spatialself_rank"] <= 30:
            return 4
        return 5

    out["recommendation_tier"] = out.apply(tier, axis=1)
    return out


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    top = df.head(20).copy()
    lines = [
        "# Human lymph node SpatialSelf-specific long-axis screen",
        "",
        "Criteria: native human lymph node, long-range/chemokine class, SpatialSelf calibrated ranking, official adapter baseline ranks, discovery FDR, downstream/edge evidence where available.",
        "",
        "Important caveat: SpaTalk ranks many candidate LR pairs at rank 1 in this atlas; therefore the table reports both all-baseline and non-SpaTalk baseline ranks.",
        "",
        "## Top candidates",
        "",
    ]
    cols = [
        "lr_pair",
        "recommendation_tier",
        "spatialself_rank",
        "commot_rank",
        "cellchat_rank",
        "spatalk_rank",
        "cellnest_rank",
        "non_spatalk_best_baseline_rank",
        "spatialself_score",
        "eligible_matched_spatial_bh_fdr",
        "eligible_fdr_call",
        "target_background_call",
        "direction_edge_call",
        "strict_story_class",
    ]
    cols = [c for c in cols if c in top.columns]
    lines.extend(_markdown_table(top[cols]))
    lines.append("")
    path.write_text("\n".join(lines))


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No candidates passed the current screen."]
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]):
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(safe.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in safe.to_numpy(dtype=str)]
    return [header, sep, *rows]


if __name__ == "__main__":
    main()
