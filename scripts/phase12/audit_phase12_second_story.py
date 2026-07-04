#!/usr/bin/env python
"""Audit candidate second biological stories for Phase 12.

This script does not rerun models. It consolidates existing Phase 11/12
evidence for lymph-node candidates into a story-readiness table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase12/second_story"
LN_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = LN_DIR / "dataset"
RESULTS_DIR = LN_DIR / "results"
BASELINE_DIR = LN_DIR / "baselines"
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"

METHOD_LABELS = {
    "spatialself_wavelet_edge_prior": "spatialself",
    "spatialself": "spatialself",
    "commot": "commot",
    "cellchat": "cellchat",
    "cellnest": "cellnest",
    "spatalk": "spatalk",
}

CANDIDATE_LR = ["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "CXCL13-CXCR5"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank_table = build_lr_score_ranks()
    direction = read_csv(RESULTS_DIR / "lymph_node_full_directionality_lr_summary.csv")
    direction_bench = read_csv(PHASE12_DIR / "directionality_benchmark/phase12_directionality_lr_metrics.csv")
    target_rank = read_csv(RESULTS_DIR / "lymph_node_full_transcriptome_target_rank_summary.csv")
    impact = read_csv(RESULTS_DIR / "lymph_node_full_lr_level_impact_summary.csv")
    if not impact.empty and "lr_pair" not in impact.columns and {"ligand", "receptor"}.issubset(impact.columns):
        impact = impact.assign(lr_pair=impact["ligand"].astype(str) + "-" + impact["receptor"].astype(str))
    panel = read_csv(RESULTS_DIR / "lymph_node_full_five_method_panel_summary.csv")

    audit = build_candidate_audit(rank_table, direction, direction_bench, target_rank, impact, panel)
    evidence = build_evidence_matrix(audit)

    audit.to_csv(OUT_DIR / "phase12_second_story_candidate_audit.csv", index=False)
    evidence.to_csv(OUT_DIR / "phase12_second_story_evidence_matrix.csv", index=False)
    write_status(audit, evidence)
    manifest = {
        "task": "phase12_second_story_audit",
        "dataset": "human_lymph_node_visium_full_subset",
        "inputs": [
            rel(DATASET_DIR / "candidate_interactions.csv"),
            rel(DATASET_DIR / "lr_pairs.csv"),
            rel(RESULTS_DIR / "lymph_node_full_directionality_lr_summary.csv"),
            rel(RESULTS_DIR / "lymph_node_full_transcriptome_target_rank_summary.csv"),
            rel(RESULTS_DIR / "lymph_node_full_lr_level_impact_summary.csv"),
            rel(RESULTS_DIR / "lymph_node_full_five_method_panel_summary.csv"),
            rel(PHASE12_DIR / "directionality_benchmark/phase12_directionality_lr_metrics.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_second_story_candidate_audit.csv"),
            rel(OUT_DIR / "phase12_second_story_evidence_matrix.csv"),
            rel(OUT_DIR / "phase12_second_story_status.md"),
        ],
        "claim_boundary": "Second-story audit only. Lymph-node CCR7 evidence supports biological consistency, not a SpatialSelf-exclusive discovery.",
    }
    (OUT_DIR / "phase12_second_story_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))


def build_lr_score_ranks() -> pd.DataFrame:
    candidates = read_csv(DATASET_DIR / "candidate_interactions.csv")
    lr_pairs = read_csv(DATASET_DIR / "lr_pairs.csv")
    lr_pairs = lr_pairs.assign(lr_pair=lr_pairs["ligand"].astype(str) + "-" + lr_pairs["receptor"].astype(str))
    lr_meta_cols = [
        "lr_id",
        "lr_pair",
        "true_lr",
        "false_positive_high_expression",
        "control_class",
        "mechanism",
    ]
    lr_meta = lr_pairs[lr_meta_cols].drop_duplicates("lr_id")
    candidate_lr = candidates[["candidate_id", "lr_id"]].merge(lr_meta, on="lr_id", how="left")

    rows = []
    for method_dir in sorted(BASELINE_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        pred_paths = sorted(method_dir.glob("*predictions.csv"))
        if not pred_paths:
            continue
        pred = pd.read_csv(pred_paths[0], usecols=lambda c: c in {"candidate_id", "score", "method"})
        method = METHOD_LABELS.get(str(pred["method"].iloc[0]) if "method" in pred.columns and len(pred) else method_dir.name, method_dir.name)
        merged = candidate_lr.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        if merged.empty:
            continue
        summary = (
            merged.groupby(["lr_id", "lr_pair", "true_lr", "false_positive_high_expression", "control_class", "mechanism"], as_index=False)
            .agg(
                mean_score=("score", "mean"),
                median_score=("score", "median"),
                q95_score=("score", lambda x: float(np.nanquantile(x, 0.95))),
                max_score=("score", "max"),
                n_candidate_edges=("score", "size"),
            )
        )
        summary["method"] = method
        summary["mean_score_rank"] = summary["mean_score"].rank(method="min", ascending=False).astype(int)
        summary["q95_score_rank"] = summary["q95_score"].rank(method="min", ascending=False).astype(int)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_candidate_audit(
    rank_table: pd.DataFrame,
    direction: pd.DataFrame,
    direction_bench: pd.DataFrame,
    target_rank: pd.DataFrame,
    impact: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for lr_pair in CANDIDATE_LR:
        for method in ["spatialself", "commot", "cellchat", "cellnest", "spatalk"]:
            rank_row = pick(rank_table, method=method, lr_pair=lr_pair)
            dir_row = pick(direction, method=method, lr_pair=lr_pair)
            bench_row = pick(direction_bench, dataset_key="lymph_node_full", method=method, lr_pair=lr_pair)
            target_row = pick(target_rank, method=method, lr_pair=lr_pair)
            impact_row = pick(impact, method=method, lr_pair=lr_pair)
            panel_row = pick(panel, method=method)
            rows.append(
                {
                    "dataset_key": "lymph_node_full",
                    "dataset": "Native human lymph node Visium",
                    "lr_pair": lr_pair,
                    "method": method,
                    "mean_score_rank": get(rank_row, "mean_score_rank"),
                    "q95_score_rank": get(rank_row, "q95_score_rank"),
                    "mean_score": get(rank_row, "mean_score"),
                    "q95_score": get(rank_row, "q95_score"),
                    "true_lr": get(rank_row, "true_lr"),
                    "weak_signal": lr_pair in {"CCL21-CCR7", "CD48-CD244"},
                    "expected_flow_fraction": first_non_null(
                        get(bench_row, "expected_flow_score_fraction"),
                        get(dir_row, "expected_flow_score_fraction"),
                    ),
                    "expected_flow_bh_fdr": get(bench_row, "expected_flow_bh_fdr"),
                    "directionality_support_class": get(bench_row, "directionality_support_class"),
                    "top_niche_flow_expected": first_non_null(
                        get(bench_row, "top_niche_flow_expected"),
                        get(dir_row, "best_flow_expected"),
                    ),
                    "target_top500_enrichment": get(target_row, "target_top500_enrichment"),
                    "n_targets_top500": get(target_row, "n_targets_top500"),
                    "best_target_rank": get(target_row, "best_target_rank"),
                    "top500_enrichment_minus_best_negative": get(target_row, "top500_enrichment_minus_best_negative"),
                    "mean_delta_r2": get(impact_row, "mean_delta_r2"),
                    "n_impact_fdr_0_1": get(impact_row, "n_impact_fdr_0_1"),
                    "best_gene": get(impact_row, "best_gene"),
                    "best_gene_fdr": get(impact_row, "best_gene_fdr"),
                    "best_negative_mean_delta_r2": get(impact_row, "best_negative_mean_delta_r2"),
                    "mean_delta_specificity_vs_best_negative": get(impact_row, "mean_delta_specificity_vs_best_negative"),
                    "method_mean_priority_delta_r2": get(panel_row, "mean_priority_delta_r2"),
                    "method_max_negative_delta_r2": get(panel_row, "max_negative_delta_r2"),
                    "method_priority_minus_max_negative": get(panel_row, "priority_minus_max_negative"),
                    "baseline_top_lr_by_mean_score": get(panel_row, "baseline_top_lr_by_mean_score"),
                    "baseline_top_lr_is_high_expression_negative": get(panel_row, "baseline_top_lr_is_high_expression_negative"),
                }
            )
    audit = pd.DataFrame(rows)
    audit["story_score"] = audit.apply(score_story_readiness, axis=1)
    audit["negative_warning"] = audit.apply(negative_warning, axis=1)
    audit["story_suitability_class"] = audit.apply(story_class, axis=1)
    audit["interpretation"] = audit.apply(interpret_row, axis=1)
    return audit.sort_values(["lr_pair", "method"]).reset_index(drop=True)


def build_evidence_matrix(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lr_pair, sub in audit.groupby("lr_pair"):
        ss = sub[sub["method"].eq("spatialself")].head(1)
        commot = sub[sub["method"].eq("commot")].head(1)
        if ss.empty:
            continue
        ssr = ss.iloc[0]
        cr = commot.iloc[0] if len(commot) else pd.Series(dtype=object)
        rows.append(
            {
                "dataset_key": "lymph_node_full",
                "candidate_story": lr_pair,
                "spatialself_rank": ssr.get("mean_score_rank"),
                "commot_rank": cr.get("mean_score_rank", np.nan),
                "rank_advantage_vs_commot": safe_float(cr.get("mean_score_rank", np.nan)) - safe_float(ssr.get("mean_score_rank", np.nan)),
                "spatialself_expected_flow": ssr.get("expected_flow_fraction"),
                "commot_expected_flow": cr.get("expected_flow_fraction", np.nan),
                "spatialself_target_enrichment": ssr.get("target_top500_enrichment"),
                "commot_target_enrichment": cr.get("target_top500_enrichment", np.nan),
                "spatialself_mean_delta_r2": ssr.get("mean_delta_r2"),
                "commot_mean_delta_r2": cr.get("mean_delta_r2", np.nan),
                "spatialself_specificity_vs_negative": ssr.get("mean_delta_specificity_vs_best_negative"),
                "spatialself_story_class": ssr.get("story_suitability_class"),
                "recommended_use": recommended_use(ssr),
            }
        )
    return pd.DataFrame(rows).sort_values(["spatialself_story_class", "candidate_story"]).reset_index(drop=True)


def score_story_readiness(row: pd.Series) -> float:
    score = 0.0
    rank = safe_float(row.get("mean_score_rank"))
    if np.isfinite(rank) and rank <= 5:
        score += 1.0
    if safe_float(row.get("expected_flow_fraction")) >= 0.8:
        score += 1.0
    if safe_float(row.get("expected_flow_bh_fdr")) <= 0.1:
        score += 1.0
    if safe_float(row.get("target_top500_enrichment")) >= 10:
        score += 1.0
    if safe_float(row.get("n_impact_fdr_0_1")) >= 5:
        score += 1.0
    if safe_float(row.get("mean_delta_specificity_vs_best_negative")) > 0:
        score += 1.0
    if bool(row.get("baseline_top_lr_is_high_expression_negative")):
        score -= 0.5
    return score


def negative_warning(row: pd.Series) -> str:
    warnings = []
    if safe_float(row.get("mean_delta_specificity_vs_best_negative")) <= 0:
        warnings.append("best_negative_downstream_competitive")
    if safe_float(row.get("top500_enrichment_minus_best_negative")) <= 0:
        warnings.append("best_negative_target_rank_competitive")
    if bool(row.get("baseline_top_lr_is_high_expression_negative")):
        warnings.append("method_top_rank_high_expression_negative")
    return ";".join(warnings) if warnings else "no_major_negative_warning"


def story_class(row: pd.Series) -> str:
    if row.get("method") != "spatialself":
        return "baseline_context"
    flow = safe_float(row.get("expected_flow_fraction"))
    target = safe_float(row.get("target_top500_enrichment"))
    impact_n = safe_float(row.get("n_impact_fdr_0_1"))
    specificity = safe_float(row.get("mean_delta_specificity_vs_best_negative"))
    if flow >= 0.8 and target >= 10 and impact_n >= 5 and specificity > 0:
        return "strong_second_story_candidate"
    if flow >= 0.8 and target >= 10 and impact_n >= 5:
        return "strong_biological_support_negative_competitive"
    if flow >= 0.8 or target >= 10:
        return "supportive_but_incomplete"
    return "insufficient_for_main_story"


def interpret_row(row: pd.Series) -> str:
    if row.get("method") != "spatialself":
        return "Baseline comparator row."
    lr = row.get("lr_pair")
    cls = row.get("story_suitability_class")
    if cls == "strong_second_story_candidate":
        return f"{lr} has directionality, target response, and better specificity than the hardest negative."
    if cls == "strong_biological_support_negative_competitive":
        return f"{lr} supports lymph-node biology, but a receptor-shared/high-expression negative remains competitive."
    if cls == "supportive_but_incomplete":
        return f"{lr} has partial biological support but is not a clean SpatialSelf-specific main-story case."
    return f"{lr} is not currently strong enough for a main biological story."


def recommended_use(row: pd.Series) -> str:
    cls = row.get("story_suitability_class")
    lr = row.get("lr_pair")
    if cls == "strong_second_story_candidate":
        return f"Use {lr} as a second main biological story after SPP1-CD44."
    if cls == "strong_biological_support_negative_competitive":
        return f"Use {lr} as biological consistency validation; disclose negative-control caveat."
    if cls == "supportive_but_incomplete":
        return f"Use {lr} as supplementary support, not the main second story."
    return f"Do not center the Phase 12 story on {lr} without more evidence."


def write_status(audit: pd.DataFrame, evidence: pd.DataFrame) -> None:
    ss = audit[audit["method"].eq("spatialself")].sort_values("story_score", ascending=False)
    best = ss.iloc[0] if len(ss) else pd.Series(dtype=object)
    lines = [
        "# Phase 12 Second-Story Audit",
        "",
        "## Scope",
        "",
        "- Dataset: native human lymph node Visium subset from Phase 11.",
        "- Candidates audited: CCL19-CCR7, CCL21-CCR7, CD48-CD244, CXCL13-CXCR5.",
        "- Evidence consolidated: method LR rank, source-sink directionality, whole-transcriptome target rank, receiver downstream impact, and hard-negative specificity.",
        "",
        "## Main Call",
        "",
    ]
    if len(best):
        lines += [
            f"- Best current second-story candidate: **{best.lr_pair}**.",
            f"- SpatialSelf story class: `{best.story_suitability_class}`.",
            f"- Expected-flow fraction: {fmt(best.expected_flow_fraction)}; directionality FDR: {fmt(best.expected_flow_bh_fdr)}.",
            f"- Target top500 enrichment: {fmt(best.target_top500_enrichment)}; downstream genes at FDR <= 0.1: {fmt(best.n_impact_fdr_0_1)}.",
            f"- Negative warning: `{best.negative_warning}`.",
            f"- Interpretation: {best.interpretation}",
        ]
    lines += [
        "",
        "## Evidence Matrix",
        "",
    ]
    for row in evidence.itertuples(index=False):
        lines.append(
            f"- **{row.candidate_story}**: SpatialSelf rank {fmt(row.spatialself_rank)}, COMMOT rank {fmt(row.commot_rank)}, "
            f"SpatialSelf flow {fmt(row.spatialself_expected_flow)}, target enrichment {fmt(row.spatialself_target_enrichment)}, "
            f"delta R2 {fmt(row.spatialself_mean_delta_r2)}, specificity vs negative {fmt(row.spatialself_specificity_vs_negative)}. "
            f"{row.recommended_use}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Lymph-node CCR7-axis results support immune-zone chemokine communication and receiver-side transcriptional consistency.",
        "- They should not be framed as a clean SpatialSelf-only discovery: COMMOT is also strong on directionality, and EEF1A1-CCR7 remains a competitive receptor-shared/high-expression negative.",
        "- The strongest cross-platform Phase 12 story remains post-Xenium colon SPP1-CD44; lymph-node CCR7 is currently a second biological-consistency story.",
    ]
    (OUT_DIR / "phase12_second_story_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick(df: pd.DataFrame, **kwargs: object) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    sub = df
    for col, val in kwargs.items():
        if col not in sub.columns:
            return pd.Series(dtype=object)
        sub = sub[sub[col].astype(str).eq(str(val))]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def get(row: pd.Series, col: str) -> object:
    if row is None or row.empty or col not in row.index:
        return np.nan
    return row[col]


def first_non_null(*values: object) -> object:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return np.nan


def safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    val = safe_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
