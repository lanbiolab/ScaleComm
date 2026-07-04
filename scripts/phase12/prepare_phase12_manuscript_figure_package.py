#!/usr/bin/env python
"""Prepare source-data package and panel map for Phase 12 manuscript figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
PHASE7_REAL = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
OUT_DIR = PHASE12_DIR / "manuscript_figure_package"


SOURCES = {
    "expanded_calibrated_dataset": PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv",
    "expanded_calibrated_topk": PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv",
    "expanded_calibrated_ranking": PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
    "expanded_permutation_dataset": PHASE12_DIR
    / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
    "expanded_permutation_topk": PHASE12_DIR
    / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_topk_summary.csv",
    "expanded_target_decoy_dataset": PHASE12_DIR
    / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv",
    "expanded_target_decoy_topk": PHASE12_DIR
    / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_topk_summary.csv",
    "expanded_same_universe_method": PHASE12_DIR
    / "atlas_model_expanded/same_universe_baselines/phase12_expanded_same_universe_baseline_method_summary.csv",
    "expanded_same_universe_dataset": PHASE12_DIR
    / "atlas_model_expanded/same_universe_baselines/phase12_expanded_same_universe_baseline_dataset_summary.csv",
    "expanded_official_baseline_status": PHASE12_DIR
    / "atlas_model_expanded/same_universe_baselines/phase12_expanded_official_baseline_status.csv",
    "expanded_official_adapter_method": PHASE12_DIR
    / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_method_summary.csv",
    "expanded_official_adapter_dataset": PHASE12_DIR
    / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_dataset_summary.csv",
    "expanded_official_adapter_topk": PHASE12_DIR
    / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_topk_summary.csv",
    "targeted_matched_dataset": PHASE12_DIR
    / "targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv",
    "targeted_matched_lr": PHASE12_DIR
    / "targeted_matched_null/phase12_targeted_matched_null_lr_audit.csv",
    "targeted_matched_method": PHASE12_DIR
    / "targeted_matched_null/phase12_targeted_matched_null_method_comparison.csv",
    "atlas_dataset": PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_dataset_summary.csv",
    "atlas_topk": PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_topk_summary.csv",
    "atlas_category": PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_category_summary.csv",
    "atlas_positive": PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_positive_control_summary.csv",
    "atlas_ranking": PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_lr_ranking.csv",
    "atlas_modules": PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_modules.csv",
    "downstream_lr": PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv",
    "gene_specificity": PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_specificity_summary.csv",
    "commot_impact": PHASE12_DIR / "gene_program_enrichment/phase12_commot_impact_comparison_summary.csv",
    "hard_dataset": PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv",
    "hard_positive": PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_positive_summary.csv",
    "hard_comparisons": PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_comparisons.csv",
    "edge_hard_dataset": PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_dataset_summary.csv",
    "edge_hard_class": PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_class_summary.csv",
    "edge_hard_lr": PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_lr_summary.csv",
    "pathway_specific_dataset": PHASE12_DIR
    / "pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv",
    "pathway_specific_candidates": PHASE12_DIR
    / "pathway_specific_programs/phase12_pathway_specific_candidate_programs.csv",
    "pathway_specific_terms": PHASE12_DIR / "pathway_specific_programs/phase12_pathway_specific_terms.csv",
    "direction_lr": PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_lr_summary.csv",
    "direction_dataset": PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_dataset_summary.csv",
    "direction_comparison": PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_method_comparison.csv",
    "direction_readiness": PHASE12_DIR / "directionality_statistics/phase12_directionality_permutation_readiness.csv",
    "direction_permutation_lr": PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_lr_summary.csv",
    "direction_permutation_dataset": PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_dataset_summary.csv",
    "direction_permutation_comparison": PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_method_comparison.csv",
    "direction_edge_lr": PHASE12_DIR
    / "directionality_edge_permutation/phase12_directionality_edge_permutation_lr_summary.csv",
    "direction_edge_dataset": PHASE12_DIR
    / "directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv",
    "direction_edge_comparison": PHASE12_DIR
    / "directionality_edge_permutation/phase12_directionality_edge_permutation_method_comparison.csv",
    "coordinate_control_guardrail_dataset": PHASE12_DIR
    / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv",
    "coordinate_control_guardrail_lr": PHASE12_DIR
    / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_lr_summary.csv",
    "strict_story_candidate_evidence": PHASE12_DIR
    / "strict_story_candidates/phase12_strict_story_candidate_evidence.csv",
    "strict_story_candidate_dataset": PHASE12_DIR
    / "strict_story_candidates/phase12_strict_story_candidate_dataset_summary.csv",
    "second_story": PHASE12_DIR / "second_story/phase12_second_story_evidence_matrix.csv",
    "cross_platform": PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv",
    "claim_matrix": PHASE12_DIR / "claim_matrix/phase12_claim_matrix.csv",
    "post_xenium_method_rank": PHASE7_REAL / "post_xenium_human_colon_cancer_method_lr_rank_table.csv",
    "post_xenium_local_contrast": PHASE7_REAL
    / "post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_fdr_all.csv",
    "post_xenium_local_contrast_audit": PHASE7_REAL
    / "post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_audit.csv",
    "post_xenium_targeted_region": PHASE7_REAL
    / "post_xenium_colon_region_aware_targeted_fdr_200perm/post_xenium_colon_region_aware_targeted_candidate_summary.csv",
    "tonsil_a1_method_rank": PHASE7_REAL / "spamosaic_tonsil_A1_method_lr_rank_table.csv",
    "tonsil_a1_local_contrast": PHASE7_REAL
    / "spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_fdr_all.csv",
    "tonsil_a1_local_contrast_audit": PHASE7_REAL
    / "spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_audit.csv",
    "tonsil_a1_targeted_region": PHASE7_REAL
    / "spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_candidate_summary.csv",
    "tonsil_d1_method_rank": PHASE7_REAL / "spamosaic_tonsil_D1_method_lr_rank_table.csv",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {name: read_csv(path) for name, path in SOURCES.items()}
    cache["coordinate_model_rerun_smoke"] = read_coordinate_model_rerun_smoke()
    outputs: list[dict[str, str]] = []
    outputs.extend(write_figure1(cache))
    outputs.extend(write_figure2(cache))
    outputs.extend(write_figure3(cache))
    outputs.extend(write_figure4(cache))
    panel_index = pd.DataFrame(panel_rows())
    panel_index.to_csv(OUT_DIR / "phase12_manuscript_figure_panel_index.csv", index=False)
    source_manifest = pd.DataFrame(outputs)
    source_manifest.to_csv(OUT_DIR / "phase12_manuscript_figure_source_manifest.csv", index=False)
    write_contract(panel_index, source_manifest)
    write_manifest(source_manifest)
    print(panel_index.to_string(index=False))


def write_figure1(cache: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rows = []
    rows.append(
        write_table(
            cache["expanded_calibrated_dataset"],
            "figure1_expanded_calibrated_dataset_summary.csv",
            "Figure 1",
            "expanded whole-LR atlas calibration summary",
        )
    )
    rows.append(
        write_table(
            cache["expanded_calibrated_topk"],
            "figure1_expanded_calibrated_topk_summary.csv",
            "Figure 1",
            "expanded atlas calibrated top-k contamination and control-panel FDR",
        )
    )
    ranking = cache["expanded_calibrated_ranking"]
    if not ranking.empty:
        ranking = ranking[pd.to_numeric(ranking["rank"], errors="coerce").le(25)].copy()
    rows.append(
        write_table(
            ranking,
            "figure1_expanded_calibrated_lr_ranking_top25.csv",
            "Figure 1",
            "top-25 LR ranks per expanded atlas dataset after calibration",
        )
    )
    rows.append(
        write_table(
            cache["expanded_permutation_dataset"],
            "figure1_expanded_permutation_dataset_summary.csv",
            "Figure 1",
            "expanded atlas post-hoc permutation FDR audit",
        )
    )
    rows.append(
        write_table(
            cache["expanded_permutation_topk"],
            "figure1_expanded_permutation_topk_summary.csv",
            "Figure 1",
            "expanded atlas top-k conservative permutation FDR summary",
        )
    )
    rows.append(
        write_table(
            cache["expanded_target_decoy_dataset"],
            "figure1_expanded_target_decoy_dataset_summary.csv",
            "Figure 1",
            "expanded atlas whole-ranking target-decoy/control-FDR summary",
        )
    )
    rows.append(
        write_table(
            cache["expanded_target_decoy_topk"],
            "figure1_expanded_target_decoy_topk_summary.csv",
            "Figure 1",
            "expanded atlas top-k hard-control FDR summary",
        )
    )
    rows.append(
        write_table(
            cache["expanded_same_universe_method"],
            "figure1_expanded_same_universe_baseline_method_summary.csv",
            "Figure 1",
            "same-universe simple baseline comparison across expanded atlas datasets",
        )
    )
    rows.append(
        write_table(
            cache["expanded_same_universe_dataset"],
            "figure1_expanded_same_universe_baseline_dataset_summary.csv",
            "Figure 1",
            "dataset-level same-universe simple baseline comparison",
        )
    )
    rows.append(
        write_table(
            cache["expanded_official_baseline_status"],
            "figure1_expanded_official_baseline_status.csv",
            "Figure 1",
            "official expanded baseline availability/status guardrail",
        )
    )
    rows.append(
        write_table(
            cache["expanded_official_adapter_method"],
            "figure1_expanded_official_adapter_method_summary.csv",
            "Figure 1",
            "same-universe official adapter-light baseline method summary",
        )
    )
    rows.append(
        write_table(
            cache["expanded_official_adapter_dataset"],
            "figure1_expanded_official_adapter_dataset_summary.csv",
            "Figure 1",
            "same-universe official adapter-light baseline dataset summary",
        )
    )
    rows.append(
        write_table(
            cache["targeted_matched_dataset"],
            "figure1_targeted_matched_null_dataset_summary.csv",
            "Figure 1",
            "targeted anchor-panel and matched-control validation summary",
        )
    )
    targeted_lr = cache["targeted_matched_lr"]
    if not targeted_lr.empty:
        targeted_lr = targeted_lr[
            targeted_lr["targeted_validation_call"].astype(str).isin(
                ["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"]
            )
        ].copy()
    rows.append(
        write_table(
            targeted_lr,
            "figure1_targeted_matched_null_supported_lr.csv",
            "Figure 1",
            "supported targeted anchor LR with panel/matched null q-values",
        )
    )
    rows.append(
        write_table(
            cache["targeted_matched_method"],
            "figure1_targeted_matched_null_method_comparison.csv",
            "Figure 1",
            "adapter-light baseline ranks for targeted supported anchors",
        )
    )
    return rows


def write_figure2(cache: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rows = []
    focus = {"SPP1-CD44", "CXCL12-CXCR4", "TGFB1-TGFBR2", "VEGFA-KDR", "ACTB-KRT8", "COL1A1-EPCAM"}
    rank = filter_lr(cache["post_xenium_method_rank"], focus)
    if not rank.empty and "method" in rank.columns:
        rank = rank[~rank["method"].astype(str).eq("spatialself_default")].copy()
    rows.append(
        write_table(
            rank,
            "figure2_crc_method_lr_rank_table.csv",
            "Figure 2",
            "SPP1-CD44 and controls across calibrated SpatialSelf/COMMOT/baselines in post-Xenium colon",
        )
    )
    rows.append(
        write_table(
            filter_lr(cache["post_xenium_local_contrast"], focus),
            "figure2_crc_local_contrast_fdr.csv",
            "Figure 2",
            "SPP1-CD44 local-contrast FDR and negative-control suppression",
        )
    )
    rows.append(
        write_table(
            cache["post_xenium_local_contrast_audit"],
            "figure2_crc_local_contrast_audit.csv",
            "Figure 2",
            "local-contrast all-null audit status",
        )
    )
    rows.append(
        write_table(
            filter_lr(cache["downstream_lr"][cache["downstream_lr"]["dataset_key"].eq("post_xenium_colon_full")], focus),
            "figure2_crc_whole_transcriptome_downstream.csv",
            "Figure 2",
            "whole-transcriptome receiver target enrichment for CRC story LRs",
        )
    )
    pathway_crc = cache["pathway_specific_candidates"]
    if not pathway_crc.empty:
        pathway_crc = pathway_crc[
            pathway_crc["dataset_key"].eq("post_xenium_colon_full")
            & pathway_crc["program_label"].astype(str).str.contains("|".join(sorted(focus)), regex=True, na=False)
        ].copy()
    rows.append(
        write_table(
            pathway_crc,
            "figure2_crc_pathway_specific_programs.csv",
            "Figure 2",
            "pathway-specific receiver-program audit for CRC candidate LRs",
        )
    )
    spec = cache["gene_specificity"]
    spec = spec[spec["dataset_key"].eq("post_xenium_colon_full") & spec["lr_pair"].astype(str).isin(focus.union({"HGF-CD44", "IL21-IL21R"}))]
    rows.append(write_table(spec, "figure2_crc_gene_program_specificity.csv", "Figure 2", "ECM/immune program specificity for CRC story LRs"))
    rows.append(
        write_table(
            cache["commot_impact"][cache["commot_impact"]["dataset_key"].eq("post_xenium_colon_full")],
            "figure2_crc_commot_style_impact.csv",
            "Figure 2",
            "COMMOT-style whole-transcriptome impact comparison",
        )
    )
    rows.append(
        write_table(
            filter_lr(cache["post_xenium_targeted_region"], focus),
            "figure2_crc_targeted_region_candidates.csv",
            "Figure 2",
            "targeted region/niche candidate definitions",
        )
    )
    return rows


def write_figure3(cache: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rows = []
    lymph_focus = {"CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "CXCL13-CXCR5"}
    tonsil_focus = {"CXCL13-CXCR5", "CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "MS4A1-CD3E", "ACTB-EGFR"}
    rows.append(write_table(cache["second_story"], "figure3_lymph_node_second_story_evidence.csv", "Figure 3", "lymph node candidate story matrix"))
    rows.append(write_table(cache["strict_story_candidate_evidence"], "figure3_strict_story_candidate_evidence.csv", "Figure 3", "strict story-candidate evidence table"))
    rows.append(write_table(cache["strict_story_candidate_dataset"], "figure3_strict_story_candidate_dataset_summary.csv", "Figure 3", "strict story-candidate dataset summary"))
    direction_lr = cache["direction_lr"]
    direction_lr = direction_lr[direction_lr["dataset_key"].isin(["lymph_node_full", "lymph_node_expanded"])]
    rows.append(write_table(direction_lr, "figure3_lymph_node_directionality_lr_statistics.csv", "Figure 3", "lymph node source-sink directionality statistics"))
    direction_perm = cache["direction_permutation_lr"]
    direction_perm = direction_perm[direction_perm["dataset_key"].isin(["lymph_node_full", "lymph_node_expanded"])] if not direction_perm.empty else direction_perm
    rows.append(write_table(direction_perm, "figure3_lymph_node_directionality_label_permutation.csv", "Figure 3", "lymph node aggregated directionality label-permutation audit"))
    direction_edge = cache["direction_edge_lr"]
    direction_edge = direction_edge[
        direction_edge["dataset_key"].isin(["lymph_node_full", "lymph_node_expanded"])
    ] if not direction_edge.empty else direction_edge
    rows.append(write_table(direction_edge, "figure3_lymph_node_directionality_edge_permutation.csv", "Figure 3", "lymph node edge-level directionality permutation audit"))
    rows.append(
        write_table(
            filter_lr(cache["atlas_positive"][cache["atlas_positive"]["dataset_key"].eq("human_lymph_node_visium_raw_full")], lymph_focus),
            "figure3_lymph_node_atlas_positive_controls.csv",
            "Figure 3",
            "lymph node atlas rank and hard-negative stress for chemokine/checkpoint candidates",
        )
    )
    rows.append(
        write_table(
            filter_lr(cache["gene_specificity"][cache["gene_specificity"]["dataset_key"].eq("human_lymph_node_visium_raw_full")], lymph_focus),
            "figure3_lymph_node_program_specificity.csv",
            "Figure 3",
            "lymph node program specificity with broad-program caveat",
        )
    )
    cross = cache["cross_platform"]
    rows.append(
        write_table(
            cross[cross["dataset_key"].isin(["spamosaic_tonsil_A1", "spamosaic_tonsil_D1", "xenium_human_lymph_node"])],
            "figure3_cross_platform_second_tissue_audit.csv",
            "Figure 3",
            "second tissue/platform story audit",
        )
    )
    rows.append(write_table(filter_lr(cache["tonsil_a1_method_rank"], tonsil_focus), "figure3_tonsil_A1_method_lr_rank.csv", "Figure 3", "SpaMosaic tonsil A1 LR ranking and controls"))
    rows.append(write_table(filter_lr(cache["tonsil_a1_local_contrast"], tonsil_focus), "figure3_tonsil_A1_local_contrast_fdr.csv", "Figure 3", "SpaMosaic tonsil A1 local-contrast FDR"))
    rows.append(write_table(filter_lr(cache["tonsil_a1_targeted_region"], tonsil_focus), "figure3_tonsil_A1_targeted_region_candidates.csv", "Figure 3", "SpaMosaic tonsil A1 targeted region candidates"))
    rows.append(write_table(filter_lr(cache["tonsil_d1_method_rank"], tonsil_focus), "figure3_tonsil_D1_method_lr_rank.csv", "Figure 3", "SpaMosaic tonsil D1 LR ranking and controls"))
    return rows


def write_figure4(cache: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rows = []
    rows.append(write_table(cache["hard_dataset"], "figure4_hard_negative_dataset_summary.csv", "Figure 4", "hard-negative stress by dataset"))
    rows.append(write_table(cache["hard_positive"], "figure4_hard_negative_positive_summary.csv", "Figure 4", "positive LR hard-negative comparisons"))
    rows.append(write_table(cache["hard_comparisons"], "figure4_hard_negative_comparisons.csv", "Figure 4", "pairwise hard-negative comparison table"))
    rows.append(write_table(cache["edge_hard_dataset"], "figure4_edge_hard_negative_dataset_summary.csv", "Figure 4", "edge-level hard-negative leakage before and after diagnostic guardrail"))
    rows.append(write_table(cache["edge_hard_class"], "figure4_edge_hard_negative_class_summary.csv", "Figure 4", "edge-level hard-negative leakage by class"))
    rows.append(write_table(cache["edge_hard_lr"], "figure4_edge_hard_negative_lr_summary.csv", "Figure 4", "edge-level hard-negative leakage by LR"))
    rows.append(write_table(cache["atlas_modules"], "figure4_communication_modules.csv", "Figure 4", "LR modules and negative-heavy module flags"))
    rows.append(write_table(cache["gene_specificity"], "figure4_program_specificity_all.csv", "Figure 4", "program specificity and negative-specific leakage"))
    rows.append(write_table(cache["pathway_specific_dataset"], "figure4_pathway_specific_dataset_summary.csv", "Figure 4", "pathway-specific receiver-program support and leakage by dataset"))
    rows.append(write_table(cache["pathway_specific_candidates"], "figure4_pathway_specific_candidate_programs.csv", "Figure 4", "pathway-specific receiver-program candidate calls"))
    rows.append(write_table(cache["pathway_specific_terms"], "figure4_pathway_specific_terms.csv", "Figure 4", "pathway-specific receiver-program term-level enrichment"))
    rows.append(write_table(cache["direction_readiness"], "figure4_directionality_permutation_readiness.csv", "Figure 4", "directionality permutation-null gap"))
    rows.append(write_table(cache["direction_permutation_dataset"], "figure4_directionality_label_permutation_dataset_summary.csv", "Figure 4", "aggregated directionality label-permutation dataset summary"))
    rows.append(write_table(cache["direction_permutation_comparison"], "figure4_directionality_label_permutation_method_comparison.csv", "Figure 4", "aggregated directionality label-permutation method comparison"))
    rows.append(write_table(cache["direction_edge_dataset"], "figure4_directionality_edge_permutation_dataset_summary.csv", "Figure 4", "edge-level directionality permutation dataset summary"))
    rows.append(write_table(cache["direction_edge_comparison"], "figure4_directionality_edge_permutation_method_comparison.csv", "Figure 4", "edge-level directionality permutation method comparison"))
    rows.append(write_table(cache["coordinate_model_rerun_smoke"], "figure4_coordinate_model_rerun_smoke_summary.csv", "Figure 4", "end-to-end coordinate-permutation model-rerun smoke summary"))
    rows.append(write_table(cache["coordinate_control_guardrail_dataset"], "figure4_coordinate_control_guardrail_dataset_summary.csv", "Figure 4", "coordinate-rerun high-expression-control guardrail dataset summary"))
    rows.append(write_table(cache["coordinate_control_guardrail_lr"], "figure4_coordinate_control_guardrail_lr_summary.csv", "Figure 4", "coordinate-rerun high-expression-control guardrail LR summary"))
    rows.append(write_table(cache["targeted_matched_lr"], "figure4_targeted_matched_null_lr_audit.csv", "Figure 4", "targeted anchor-panel matched-null audit with claim boundaries"))
    rows.append(write_table(cache["claim_matrix"], "figure4_claim_matrix_guardrails.csv", "Figure 4", "supported claims and forbidden extensions"))
    return rows


def panel_rows() -> list[dict[str, str]]:
    return [
        panel("Figure 1", "a", "Expanded atlas calibration", "figure1_expanded_calibrated_dataset_summary.csv", "SpatialSelf can score 886/957 LR atlases and produce calibrated whole-LR rankings across three real-background datasets.", "This is calibrated rank support, not final end-to-end permutation FDR."),
        panel("Figure 1", "b", "Calibrated top-k contamination", "figure1_expanded_calibrated_topk_summary.csv", "Calibration suppresses zero-expression and high-expression hard negatives in top-k LR lists.", "Other/background LR are not all ground-truth negatives."),
        panel("Figure 1", "c", "Permutation, control and targeted nulls", "figure1_expanded_permutation_dataset_summary.csv;figure1_expanded_permutation_topk_summary.csv;figure1_expanded_target_decoy_dataset_summary.csv;figure1_expanded_target_decoy_topk_summary.csv;figure1_targeted_matched_null_dataset_summary.csv;figure1_targeted_matched_null_supported_lr.csv", "Whole-atlas BH FDR remains strict; target-decoy/control FDR quantifies hard-control contamination; pre-specified anchor LR can be validated with targeted permutation-panel and matched-control nulls.", "Do not conflate target-decoy/control FDR or targeted story validation with unbiased whole-atlas discovery FDR."),
        panel("Figure 1", "d", "Same-universe baseline audit", "figure1_expanded_same_universe_baseline_method_summary.csv;figure1_expanded_official_adapter_method_summary.csv;figure1_expanded_official_baseline_status.csv", "Calibrated SpatialSelf is cleaner than simple heuristics and official adapter-light baselines on top-k contamination.", "Adapter-light/projected baselines are not fully tuned official end-to-end runs."),
        panel("Figure 2", "a", "SPP1-CD44 ranking and controls", "figure2_crc_method_lr_rank_table.csv", "SPP1-CD44 is a high-ranking CRC/post-Xenium communication candidate with controls shown.", "SpatialSelf is represented by the calibrated score only; do not reintroduce the old default variant."),
        panel("Figure 2", "b", "Local-contrast FDR", "figure2_crc_local_contrast_fdr.csv;figure2_crc_local_contrast_audit.csv", "SPP1-CD44 passes local contrast with negative controls suppressed across nulls.", "Local contrast validates spatial concentration, not causal signaling."),
        panel("Figure 2", "c", "Receiver transcriptome support", "figure2_crc_whole_transcriptome_downstream.csv;figure2_crc_gene_program_specificity.csv;figure2_crc_pathway_specific_programs.csv", "SPP1-CD44 and related CRC LRs show selected receiver target/program support; pathway-specific closure remains leakage-caveated in the expanded audit.", "Observational downstream association, not perturbation."),
        panel("Figure 2", "d", "COMMOT-style impact", "figure2_crc_commot_style_impact.csv", "Whole-transcriptome impact summaries give an independent functional layer.", "Existing impact table is reused from Phase 10."),
        panel("Figure 3", "a", "Second-story candidate triage", "figure3_lymph_node_second_story_evidence.csv;figure3_strict_story_candidate_evidence.csv;figure3_strict_story_candidate_dataset_summary.csv", "Strict story triage keeps SPP1-CD44 as the only current A-class main story and classifies lymph-node chemokines as B-class biological-consistency candidates.", "Not SpatialSelf-exclusive; lymph-node candidates have broad-program/downstream caveats and some baselines remain competitive."),
        panel("Figure 3", "b", "Lymph node directionality", "figure3_lymph_node_directionality_lr_statistics.csv;figure3_lymph_node_directionality_label_permutation.csv;figure3_lymph_node_directionality_edge_permutation.csv", "Lymph node source-sink consistency is quantifiable and now has aggregated label-permutation plus edge-level receiver-pairing nulls.", "Edge-level null is fixed-score and lacks vector-null support; it does not prove causal transport."),
        panel("Figure 3", "c", "Lymph node program specificity", "figure3_lymph_node_program_specificity.csv", "Immune/interferon terms exist but broad translation/ribosome terms dominate many programs.", "Use as caveated biological consistency, not clean downstream closure."),
        panel("Figure 3", "d", "Cross-platform caveat audit", "figure3_cross_platform_second_tissue_audit.csv", "Tonsil A1/D1 provide cross-platform ranking support but retain targeted negative leakage; Xenium lymph node remains mixed.", "Use these as supportive platform validation, not a clean second main story."),
        panel("Figure 4", "a", "Hard-negative stress", "figure4_hard_negative_dataset_summary.csv;figure4_hard_negative_positive_summary.csv", "Hard negatives are explicitly quantified rather than hidden.", "Some positive controls do not pass empirical FDR<=0.1."),
        panel("Figure 4", "b", "Edge-level hard-negative guardrail", "figure4_edge_hard_negative_dataset_summary.csv;figure4_edge_hard_negative_class_summary.csv;figure4_edge_hard_negative_lr_summary.csv", "A diagnostic edge-level guardrail reduces aggregate hard-negative leakage, especially in lymph node.", "This guardrail is post-hoc and class-specific; it is not a retrained model or final FDR solution."),
        panel("Figure 4", "c", "Pathway-specific program support", "figure4_pathway_specific_dataset_summary.csv;figure4_pathway_specific_candidate_programs.csv;figure4_pathway_specific_terms.csv", "The expanded pathway-specific receiver-program audit is leakage-confounded across post-Xenium colon, lymph node and Visium HD CRC, so it is used as a guardrail rather than clean closure.", "Program terms are observational enrichment summaries, not perturbational validation."),
        panel("Figure 4", "d", "Claim guardrails", "figure4_directionality_permutation_readiness.csv;figure4_directionality_label_permutation_dataset_summary.csv;figure4_directionality_edge_permutation_dataset_summary.csv;figure4_coordinate_model_rerun_smoke_summary.csv;figure4_coordinate_control_guardrail_dataset_summary.csv;figure4_targeted_matched_null_lr_audit.csv;figure4_claim_matrix_guardrails.csv", "The manuscript can separate whole-atlas FDR, targeted biological-story validation, edge-level directionality nulls, coordinate model-rerun smoke, high-expression-control guardrails, and broad-background leakage.", "Aggregated label permutation, edge-level fixed-score permutation, targeted matched-null support, coordinate-rerun smoke, control guardrails, and adapter-light baselines are available, but final whole-atlas FDR remains incomplete and broad/background LR leakage remains high."),
    ]


def write_contract(panel_index: pd.DataFrame, source_manifest: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Manuscript Figure Package",
        "",
        "## Core Conclusion",
        "",
        "SpatialSelf extends beyond selected LR panels into expanded 886/957-LR real-background atlases, where calibration improves top-k cleanliness, target-decoy/control FDR quantifies hard-control contamination, and same-universe heuristic plus official adapter-light baselines expose high-expression contamination; the strongest current biological case remains post-Xenium colon SPP1-CD44, while final whole-atlas permutation discovery FDR remains caveated.",
        "",
        "## Figure Archetype",
        "",
        "- Figure 1: quantitative grid.",
        "- Figure 2: asymmetric mixed-modality figure.",
        "- Figure 3: asymmetric mixed-modality figure.",
        "- Figure 4: quantitative grid / mechanism audit.",
        "",
        "## Panel Map",
        "",
    ]
    for row in panel_index.itertuples(index=False):
        lines.append(f"- **{row.figure_id}{row.panel} {row.panel_title}**: {row.primary_message}")
        lines.append(f"  - source: `{row.source_data}`")
        lines.append(f"  - review risk: {row.review_risk}")
    lines += [
        "",
        "## Evidence Rules",
        "",
        "- Do not claim final whole-atlas superiority from calibrated expanded atlas ranking alone.",
        "- Do not call target-decoy/control FDR a permutation-calibrated whole-atlas discovery FDR.",
        "- Do not call targeted anchor-panel matched-null validation unbiased whole-atlas discovery FDR.",
        "- Describe COMMOT/CellChat/SpaTalk/CellNEST expanded comparisons as adapter-light/projected unless fully tuned end-to-end runs are added.",
        "- Do not claim global directionality superiority over COMMOT.",
        "- Do not call aggregated niche-flow label permutation a coordinate-level directionality rerun.",
        "- Do not call edge-level fixed-score directionality permutation model retraining under coordinate permutation.",
        "- Do not call the 20-per-dataset / 40-total coordinate model-rerun smoke a formal coordinate-null FDR analysis.",
        "- Do not present lymph-node program enrichment as clean pathway-specific downstream validation when broad ribosome/translation terms dominate.",
        "- Use post-Xenium colon as the main positive case; use lymph node and tonsil as caveated support.",
        "",
        "## Source Data Count",
        "",
        f"- {len(source_manifest)} source-data tables prepared.",
    ]
    (OUT_DIR / "phase12_manuscript_figure_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(source_manifest: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_manuscript_figure_package",
        "backend_for_future_rendering": "python",
        "outputs": [
            "phase12_manuscript_figure_panel_index.csv",
            "phase12_manuscript_figure_source_manifest.csv",
            "phase12_manuscript_figure_contract.md",
            *source_manifest["output_file"].astype(str).tolist(),
        ],
        "claim_boundary": "Source-data and panel-map package only; no final rendered manuscript figure has been generated in this step.",
    }
    (OUT_DIR / "phase12_manuscript_figure_package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def panel(figure_id: str, panel: str, title: str, source: str, message: str, risk: str) -> dict[str, str]:
    return {
        "figure_id": figure_id,
        "panel": panel,
        "panel_title": title,
        "source_data": source,
        "primary_message": message,
        "review_risk": risk,
    }


def filter_lr(table: pd.DataFrame, lr_pairs: set[str]) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    if "lr_pair" in out.columns:
        key = out["lr_pair"].astype(str)
    elif "control_name" in out.columns:
        key = out["control_name"].astype(str)
    else:
        ligand = out["ligand"].astype(str) if "ligand" in out.columns else ""
        receptor = out["receptor"].astype(str) if "receptor" in out.columns else ""
        key = ligand + "-" + receptor
    return out[key.isin(lr_pairs)].copy()


def write_table(table: pd.DataFrame, filename: str, figure: str, description: str) -> dict[str, str]:
    path = OUT_DIR / filename
    table.to_csv(path, index=False)
    return {
        "figure_id": figure,
        "output_file": filename,
        "n_rows": str(len(table)),
        "n_columns": str(len(table.columns)),
        "description": description,
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_coordinate_model_rerun_smoke() -> pd.DataFrame:
    root = PHASE12_DIR / "coordinate_model_rerun_smoke"
    frames = []
    for path in sorted(root.glob("*/*_coordinate_model_rerun_summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()
