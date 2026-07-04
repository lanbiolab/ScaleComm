#!/usr/bin/env python
"""Requirement-level completion audit for the Phase 12 objective.

This is not another result-generating analysis. It maps the user-defined Phase
12 requirements to current files and marks whether the evidence proves the
requirement, only partially supports it, contradicts it, or is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "completion_audit"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(goal1_rows())
    rows.extend(goal2_rows())
    rows.extend(goal3_rows())
    rows.extend(goal4_rows())
    rows.extend(goal5_rows())
    rows.extend(goal6_rows())
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "phase12_requirement_completion_audit.csv", index=False)
    summary = summarize(audit)
    summary.to_csv(OUT_DIR / "phase12_goal_completion_summary.csv", index=False)
    write_status(audit, summary)
    write_manifest(audit, summary)
    print(summary.to_string(index=False))
    print()
    print(audit.to_string(index=False))


def goal1_rows() -> list[dict[str, object]]:
    rows = []
    cal = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv")
    rank = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    perm = read_csv(PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv")
    target_decoy = read_csv(PHASE12_DIR / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv")
    fdr_closure = read_csv(PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_dataset.csv")
    fdr_story = read_csv(PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_story.csv")
    strict_discovery = read_csv(PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv")
    strict_story = read_csv(PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_story.csv")
    null_specific = read_csv(PHASE12_DIR / "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv")
    null_specific_story = read_csv(PHASE12_DIR / "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_story.csv")
    spatial_dependency = read_csv(PHASE12_DIR / "spatial_dependency_audit/phase12_spatial_dependency_dataset.csv")
    spatial_dependency_story = read_csv(PHASE12_DIR / "spatial_dependency_audit/phase12_spatial_dependency_story.csv")
    stratified_fdr = read_csv(PHASE12_DIR / "stratified_atlas_fdr/phase12_stratified_atlas_fdr_dataset.csv")
    stratified_fdr_story = read_csv(PHASE12_DIR / "stratified_atlas_fdr/phase12_stratified_atlas_fdr_story.csv")
    matched_spatial_fdr = read_csv(PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv")
    matched_spatial_story = read_csv(PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv")
    guarded_matched_fdr = read_csv(PHASE12_DIR / "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_dataset.csv")
    guarded_matched_story = read_csv(PHASE12_DIR / "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_story.csv")
    spatial_prototype_fdr = read_csv(PHASE12_DIR / "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_dataset.csv")
    spatial_prototype_story = read_csv(PHASE12_DIR / "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_story.csv")
    spatial_training_smoke = read_csv(PHASE12_DIR / "spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv")
    spatial_trained_variant = read_csv(PHASE12_DIR / "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv")
    spatial_anchor_variant = read_csv(PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv")
    spatial_anchor_permutation = read_csv(
        PHASE12_DIR
        / "spatial_null_anchor_preserving_permutation_comparison/"
        / "phase12_spatial_null_anchor_preserving_permutation_story.csv"
    )
    anchor_background_permutation = read_csv(
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_comparison/"
        / "phase12_anchor_background_permutation_story.csv"
    )
    range_weighted_fdr = read_csv(PHASE12_DIR / "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_dataset.csv")
    range_weighted_story = read_csv(PHASE12_DIR / "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_story.csv")
    lr_standardized_fdr = read_csv(
        PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_method_summary.csv"
    )
    lr_standardized_story = read_csv(
        PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_story.csv"
    )
    null_margin_guarded = read_csv(
        PHASE12_DIR
        / "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_dataset_comparison.csv"
    )
    null_margin_guarded_story = read_csv(
        PHASE12_DIR / "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_story.csv"
    )
    fdr_margin_gap_story = read_csv(PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_story.csv")
    spatial_specificity_story = read_csv(
        PHASE12_DIR / "spatial_specificity_comparison/phase12_spatial_specificity_story.csv"
    )
    guarded_spatial_specificity_story = read_csv(
        PHASE12_DIR / "guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_story.csv"
    )
    integrated_guarded_spatial = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv"
    )
    integrated_guarded_spatial_story = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_story.csv"
    )
    integrated_highres_stability_story = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_highres_stability/"
        / "phase12_integrated_guarded_spatial_highres_stability_story.csv"
    )
    integrated_coordinate_smoke = best_integrated_coordinate_smoke()
    discovery_eligible_fdr = read_csv(PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv")
    discovery_eligible_story = read_csv(PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv")
    fdr_bottleneck = read_csv(PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv")
    fdr_bottleneck_story = read_csv(PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_story.csv")
    cat = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_category_summary.csv")
    topk = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv")
    modules = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_modules.csv")
    hard = read_csv(PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_comparisons.csv")
    receptor_atlas = read_csv(PHASE12_DIR / "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv")
    coord_guard = read_csv(PHASE12_DIR / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv")
    supplementary_panel = read_csv(
        PHASE12_DIR
        / "supplementary_cross_platform_expanded_panel_atlas"
        / "phase12_supplementary_expanded_panel_dataset_summary.csv"
    )

    n_large = n_datasets(cal)
    min_lr = int(pd.to_numeric(cal.get("n_lr", pd.Series(dtype=float)), errors="coerce").min()) if not cal.empty else 0
    supplementary_text = ""
    if not supplementary_panel.empty:
        supplementary_text = (
            " Supplementary 2011-gene expanded-panel atlas also covers "
            + ", ".join(str(x) for x in supplementary_panel["dataset_key"].tolist())
            + "."
        )
    rows.append(row("Goal 1", "At least two large-LR real-background atlas runs", "complete" if n_large >= 2 and min_lr >= 800 else "partial", "direct", f"{n_large} calibrated strict full-transcriptome expanded datasets; min LR={min_lr}.{supplementary_text}", "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv; supplementary_cross_platform_expanded_panel_atlas/phase12_supplementary_expanded_panel_dataset_summary.csv", "Strict full-transcriptome evidence covers post-Xenium colon, human lymph-node Visium and Visium HD CRC. Tonsil/melanoma currently add cross-platform support only as 2011-gene expanded-panel constructed real-background atlases, not strict full-transcriptome FDR closure.", "Keep the strict full-transcriptome atlas and supplementary expanded-panel atlas separate in manuscript claims."))

    rows.append(row("Goal 1", "Whole-LR ranking table", "complete" if not rank.empty and rank["dataset_key"].nunique() >= 2 else "missing", "direct" if not rank.empty else "missing", f"Ranking rows={len(rank)}, datasets={rank['dataset_key'].nunique() if not rank.empty else 0}.", "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv", "Ranking is calibrated score output, not discovery FDR.", "Keep raw and calibrated rankings linked in source data."))

    q_hits = int(pd.to_numeric(perm.get("n_all_lr_q_le_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not perm.empty else 0
    control_fdr_text = ""
    if not target_decoy.empty:
        control_fdr_text = "; target-decoy/control FDR exists: " + "; ".join(
            f"{r.dataset_key} top10={float(r.top10_primary_control_prefix_fdr):.3g}, top100={float(r.top100_primary_control_prefix_fdr):.3g}"
            for r in target_decoy.itertuples(index=False)
        )
    closure_text = ""
    if not fdr_closure.empty:
        closure_text = "; FDR closure: " + "; ".join(
            f"{r.dataset_key}={r.dataset_fdr_closure_call}"
            for r in fdr_closure.itertuples(index=False)
        )
        if not fdr_story.empty:
            closure_text += f"; overall={fdr_story.iloc[0].overall_atlas_fdr_closure_call}"
    strict_text = ""
    if not strict_discovery.empty:
        strict_q_hits = int(pd.to_numeric(strict_discovery.get("all_lr_strict_permutation_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_text = "; strict discovery audit: " + "; ".join(
            f"{r.dataset_key} top10FDR={float(r.top10_primary_control_prefix_fdr):.3g}, strict q<=0.1={int(r.all_lr_strict_permutation_q_0_1)}"
            for r in strict_discovery.itertuples(index=False)
        )
        if not strict_story.empty:
            strict_text += f"; strict overall={strict_story.iloc[0].overall_strict_atlas_call}"
    else:
        strict_q_hits = 0
    null_specific_text = ""
    if not null_specific.empty:
        two_null_true = int(pd.to_numeric(null_specific.get("two_null_pairing_identity_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        spatial_true = int(pd.to_numeric(null_specific.get("spatial_distance_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        null_specific_text = f"; null-specific diagnostic: pairing+identity true q<=0.1={two_null_true}, spatial-distance true q<=0.1={spatial_true}"
        if not null_specific_story.empty:
            null_specific_text += f"; null-specific overall={null_specific_story.iloc[0].overall_null_specific_call}"
    spatial_dependency_text = ""
    if not spatial_dependency.empty:
        true_delta = int(pd.to_numeric(spatial_dependency.get("n_true_spatial_delta_positive", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        true_spatial_q = int(pd.to_numeric(spatial_dependency.get("n_true_spatial_within_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        spatial_dependency_text = f"; spatial-dependency audit: true positive spatial delta={true_delta}, true within spatial q<=0.1={true_spatial_q}"
        if not spatial_dependency_story.empty:
            spatial_dependency_text += f"; spatial-dependency overall={spatial_dependency_story.iloc[0].overall_spatial_dependency_call}"
    stratified_fdr_text = ""
    if not stratified_fdr.empty:
        true_mech = int(pd.to_numeric(stratified_fdr.get("true_mechanism_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        true_cat = int(pd.to_numeric(stratified_fdr.get("true_category_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        stratified_fdr_text = f"; stratified all-null FDR: true mechanism q<=0.1={true_mech}, true category q<=0.1={true_cat}"
        if not stratified_fdr_story.empty:
            stratified_fdr_text += f"; stratified overall={stratified_fdr_story.iloc[0].overall_stratified_atlas_fdr_call}"
    matched_spatial_text = ""
    if not matched_spatial_fdr.empty:
        matched_spatial_true = int(pd.to_numeric(matched_spatial_fdr.get("all_true_lr_matched_spatial_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        matched_all_true = int(pd.to_numeric(matched_spatial_fdr.get("all_true_lr_matched_all_null_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_matched_all_true = int(pd.to_numeric(matched_spatial_fdr.get("all_true_lr_strict_matched_all_null_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        matched_spatial_text = f"; matched spatial diagnostic: true matched-spatial q<=0.1={matched_spatial_true}, true matched all-null q<=0.1={matched_all_true}, strict true matched all-null q<=0.1={strict_matched_all_true}"
        if not matched_spatial_story.empty:
            matched_spatial_text += f"; matched spatial overall={matched_spatial_story.iloc[0].overall_matched_spatial_call}"
    guarded_matched_text = ""
    if not guarded_matched_fdr.empty:
        guarded_true = int(pd.to_numeric(guarded_matched_fdr.get("guarded_matched_all_null_n_true_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guarded_controls = int(pd.to_numeric(guarded_matched_fdr.get("guarded_matched_all_null_n_primary_control_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guarded_receptor = int(pd.to_numeric(guarded_matched_fdr.get("guarded_matched_all_null_n_receptor_shared_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guarded_background = int(pd.to_numeric(guarded_matched_fdr.get("guarded_matched_all_null_n_background_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guarded_matched_text = f"; guarded matched full-LR FDR: true supported={guarded_true}, primary controls={guarded_controls}, receptor-shared={guarded_receptor}, background={guarded_background}"
        if not guarded_matched_story.empty:
            guarded_matched_text += f"; guarded matched overall={guarded_matched_story.iloc[0].overall_call}"
    spatial_prototype_text = ""
    if not spatial_prototype_fdr.empty:
        prototype_true = int(pd.to_numeric(spatial_prototype_fdr.get("true_lr_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        prototype_risk = int(pd.to_numeric(spatial_prototype_fdr.get("negative_or_risk_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        prototype_top10_true = int(pd.to_numeric(spatial_prototype_fdr.get("top10_true_lr", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
        spatial_prototype_text = f"; spatial-null-aware exact-score prototype: true q<=0.1={prototype_true}, risk q<=0.1={prototype_risk}, best dataset top10 true={prototype_top10_true}"
        if not spatial_prototype_story.empty:
            spatial_prototype_text += f"; prototype overall={spatial_prototype_story.iloc[0].overall_call}"
    training_smoke_text = ""
    if not spatial_training_smoke.empty:
        training_smoke_text = (
            "; spatial-null contrastive training smoke: "
            f"status={spatial_training_smoke.iloc[0].status}, "
            f"spatial-null pairs={int(spatial_training_smoke.iloc[0].n_spatial_null_contrastive_train)}"
        )
    trained_variant_text = ""
    if not spatial_trained_variant.empty:
        improved = int(spatial_trained_variant["comparison_call"].astype(str).eq("ranking_improved_without_top10_risk").sum())
        worse = int(spatial_trained_variant["comparison_call"].astype(str).eq("tradeoff_or_worse").sum())
        delta_true = int(pd.to_numeric(spatial_trained_variant.get("delta_top10_true_lr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        delta_risk = float(pd.to_numeric(spatial_trained_variant.get("delta_top10_risk_fraction", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        trained_variant_text = f"; spatial-null-trained expanded variant: improved datasets={improved}, tradeoff/worse={worse}, sum delta top10 true={delta_true}, sum delta top10 risk={delta_risk:.3g}"
    anchor_variant_text = ""
    if not spatial_anchor_variant.empty:
        calls = spatial_anchor_variant["comparison_call"].astype(str)
        improved = int(calls.eq("anchor_variant_improves_main").sum())
        preserved = int(calls.eq("anchor_variant_preserves_main").sum())
        reduced = int(calls.eq("anchor_variant_reduces_true_without_risk").sum())
        delta_true = int(pd.to_numeric(spatial_anchor_variant.get("delta_top10_true_lr_vs_main", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        delta_risk = float(pd.to_numeric(spatial_anchor_variant.get("delta_top10_risk_vs_main", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        anchor_variant_text = f"; anchor-preserving spatial-null variant: improved={improved}, preserved={preserved}, reduced-true-without-risk={reduced}, sum delta top10 true={delta_true}, sum delta top10 risk={delta_risk:.3g}"
    if not spatial_anchor_permutation.empty:
        first = spatial_anchor_permutation.iloc[0]
        anchor_variant_text += (
            "; anchor-preserving exact-score permutation: "
            f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}, "
            f"top10 improved={int(first.n_top10_improved_without_risk)}, "
            f"top10 preserved={int(first.n_top10_preserved_without_risk)}, "
            f"best true q improved={int(first.n_best_true_q_improved)}, "
            f"overall={first.overall_call}"
        )
    if not anchor_background_permutation.empty:
        first = anchor_background_permutation.iloc[0]
        anchor_variant_text += (
            "; anchor+background exact-score permutation: "
            f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}, "
            f"top10 improved vs base={int(first.n_top10_improved_without_risk_vs_base)}, "
            f"best true q improved={int(first.n_best_true_q_improved_vs_base)}, "
            f"CRC top10 recovered={int(first.n_crc_top10_recovered_vs_base)}, "
            f"overall={first.overall_call}"
        )
    range_weighted_text = ""
    if not range_weighted_fdr.empty:
        weighted_true = int(pd.to_numeric(range_weighted_fdr.get("weighted_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        weighted_controls = int(pd.to_numeric(range_weighted_fdr.get("weighted_control_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_weighted_true = int(pd.to_numeric(range_weighted_fdr.get("strict_weighted_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_weighted_controls = int(pd.to_numeric(range_weighted_fdr.get("strict_weighted_control_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        range_weighted_text = f"; range-weighted diagnostic: weighted true q<=0.1={weighted_true}, weighted controls q<=0.1={weighted_controls}, strict weighted true q<=0.1={strict_weighted_true}, strict weighted controls q<=0.1={strict_weighted_controls}"
        if not range_weighted_story.empty:
            range_weighted_text += f"; range-weighted overall={range_weighted_story.iloc[0].overall_range_weighted_call}"
    lr_standardized_text = ""
    if not lr_standardized_fdr.empty:
        standardized_true = int(pd.to_numeric(lr_standardized_fdr.get("standardized_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        standardized_controls = int(pd.to_numeric(lr_standardized_fdr.get("standardized_control_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_standardized_true = int(pd.to_numeric(lr_standardized_fdr.get("strict_standardized_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        strict_standardized_controls = int(pd.to_numeric(lr_standardized_fdr.get("strict_standardized_control_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        lr_standardized_text = f"; LR-standardized all-null diagnostic: true q<=0.1={standardized_true}, controls q<=0.1={standardized_controls}, strict true q<=0.1={strict_standardized_true}, strict controls q<=0.1={strict_standardized_controls}"
        if not lr_standardized_story.empty:
            lr_standardized_text += f"; LR-standardized overall={lr_standardized_story.iloc[0].overall_call}"
    null_margin_guarded_text = ""
    if not null_margin_guarded.empty:
        nm_top10_improved = int(pd.to_numeric(null_margin_guarded.get("delta_top10_true_lr_vs_main", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum())
        nm_hard_clean = int(pd.to_numeric(null_margin_guarded.get("hard_negative_best_comparison_beats", pd.Series(dtype=float)), errors="coerce").fillna(999).eq(0).sum())
        nm_true_q = int(pd.to_numeric(null_margin_guarded.get("all_true_lr_conservative_fdr_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        null_margin_guarded_text = f"; null-margin guarded training: top10 improved datasets={nm_top10_improved}, hard-negative clean datasets={nm_hard_clean}, true q<=0.1={nm_true_q}"
        if not null_margin_guarded_story.empty:
            null_margin_guarded_text += f"; null-margin overall={null_margin_guarded_story.iloc[0].overall_call}"
    fdr_margin_gap_text = ""
    if not fdr_margin_gap_story.empty:
        first = fdr_margin_gap_story.iloc[0]
        fdr_margin_gap_text = (
            f"; FDR margin gap: best variant={first.best_variant_by_min_q_gap}, "
            f"min q-gap={float(first.best_variant_min_q_gap_to_0_1):.3g}x, "
            f"median q-gap={float(first.best_variant_median_q_gap_to_0_1):.3g}x, "
            f"dominant nulls={first.best_variant_dominant_null_counts}"
        )
    spatial_specificity_text = ""
    if not spatial_specificity_story.empty:
        first = spatial_specificity_story.iloc[0]
        spatial_specificity_text = (
            f"; spatial-specificity calibration: true q<=0.1={int(first.n_true_q_0_1)}, "
            f"best true q improved vs main={int(first.n_best_true_q_improved_vs_main)}/{int(first.n_datasets)}, "
            f"best q={float(first.best_true_q_min):.3g}, "
            f"overall={first.overall_call}"
        )
    guarded_spatial_specificity_text = ""
    if not guarded_spatial_specificity_story.empty:
        first = guarded_spatial_specificity_story.iloc[0]
        guarded_spatial_specificity_text = (
            f"; guarded spatial-specificity calibration: true q<=0.1={int(first.n_true_q_0_1)}, "
            f"best true q improved vs main={int(first.n_best_true_q_improved_vs_main)}/{int(first.n_datasets)}, "
            f"risk improved vs unguarded={int(first.n_top10_risk_improved_vs_spatial_specificity)}/{int(first.n_datasets)}, "
            f"best q={float(first.best_true_q_min):.3g}, "
            f"overall={first.overall_call}"
        )
    integrated_guarded_spatial_text = ""
    if not integrated_guarded_spatial.empty:
        integrated_true = int(
            pd.to_numeric(
                integrated_guarded_spatial.get("all_true_lr_conservative_fdr_0_1", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        integrated_all = int(
            pd.to_numeric(
                integrated_guarded_spatial.get("all_lr_conservative_fdr_0_1", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        integrated_best_q = float(
            pd.to_numeric(integrated_guarded_spatial.get("best_true_lr_fdr", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .min()
        )
        integrated_best_lr = ""
        if "best_true_lr_fdr" in integrated_guarded_spatial.columns:
            best = integrated_guarded_spatial.sort_values("best_true_lr_fdr").head(1)
            if len(best):
                integrated_best_lr = f"{best.iloc[0].dataset_key}:{best.iloc[0].best_true_lr}"
        integrated_guarded_spatial_text = (
            f"; integrated guarded spatial training candidate: true q<=0.1={integrated_true}, "
            f"all LR q<=0.1={integrated_all}, best true={integrated_best_lr} q={integrated_best_q:.3g}"
        )
        if not integrated_guarded_spatial_story.empty:
            integrated_guarded_spatial_text += (
                f"; integrated overall={integrated_guarded_spatial_story.iloc[0].overall_call}"
            )
        if not integrated_highres_stability_story.empty:
            first = integrated_highres_stability_story.iloc[0]
            integrated_guarded_spatial_text += (
                f"; highres stability={first.overall_call}, "
                f"stable q<=0.1 hits={int(first.n_stable_q_0_1_hits)}, "
                f"best highres={first.best_highres_true} q={float(first.best_highres_true_q):.3g}"
            )
        if not integrated_coordinate_smoke.empty:
            first = integrated_coordinate_smoke.iloc[0]
            integrated_guarded_spatial_text += (
                f"; end-to-end coordinate smoke target={first.target_lr}, "
                f"reruns={int(first.n_completed_permutations)}, "
                f"score={float(first.target_score):.3g} vs null q95={float(first.target_null_q95):.3g}, "
                f"p={float(first.target_p):.3g}, BH={float(first.target_bh_fdr):.3g}, "
                f"best negative={first.best_negative_lr} p={float(first.best_negative_p):.3g}"
            )
    discovery_eligible_text = ""
    if not discovery_eligible_fdr.empty:
        eligible_matched_true = int(pd.to_numeric(discovery_eligible_fdr.get("eligible_matched_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        eligible_strict_true = int(pd.to_numeric(discovery_eligible_fdr.get("eligible_strict_matched_true_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        eligible_controls = int(pd.to_numeric(discovery_eligible_fdr.get("eligible_matched_control_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        discovery_eligible_text = f"; discovery-eligible FDR: matched true q<=0.1={eligible_matched_true}, strict matched true q<=0.1={eligible_strict_true}, matched controls q<=0.1={eligible_controls}"
        if not discovery_eligible_story.empty:
            discovery_eligible_text += f"; discovery-eligible overall={discovery_eligible_story.iloc[0].overall_discovery_eligible_call}"
    bottleneck_text = ""
    if not fdr_bottleneck.empty:
        spatial_bottleneck = int(fdr_bottleneck["dominant_bottleneck_null"].astype(str).str.contains("spatial_distance").sum())
        target_decoy_clean = int(pd.to_numeric(fdr_bottleneck.get("target_decoy_top100_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).sum())
        bottleneck_text = f"; Goal1 bottleneck: spatial-distance bottleneck datasets={spatial_bottleneck}, target-decoy top100 clean datasets={target_decoy_clean}"
        if not fdr_bottleneck_story.empty:
            bottleneck_text += f"; bottleneck overall={fdr_bottleneck_story.iloc[0].overall_state}"
    fdr_status = "missing"
    if not perm.empty and not target_decoy.empty:
        fdr_status = "partial"
    elif not perm.empty and q_hits == 0:
        fdr_status = "not_supported"
    elif not perm.empty:
        fdr_status = "partial"
    rows.append(row("Goal 1", "Global whole-atlas FDR", fdr_status, "direct" if not perm.empty else "missing", f"Post-hoc permutation audit exists; all-LR q<=0.1 discoveries={q_hits}; strict all-LR q<=0.1 discoveries={strict_q_hits}{control_fdr_text}{closure_text}{strict_text}{null_specific_text}{spatial_dependency_text}{stratified_fdr_text}{matched_spatial_text}{guarded_matched_text}{spatial_prototype_text}{training_smoke_text}{trained_variant_text}{anchor_variant_text}{range_weighted_text}{lr_standardized_text}{null_margin_guarded_text}{fdr_margin_gap_text}{spatial_specificity_text}{guarded_spatial_specificity_text}{integrated_guarded_spatial_text}{discovery_eligible_text}{bottleneck_text}.", "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv; atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv; atlas_fdr_closure/phase12_atlas_fdr_closure_dataset.csv; strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv; null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv; spatial_dependency_audit/phase12_spatial_dependency_dataset.csv; stratified_atlas_fdr/phase12_stratified_atlas_fdr_dataset.csv; matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv; guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_dataset.csv; spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_dataset.csv; spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv; atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv; atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv; spatial_null_anchor_preserving_permutation_comparison/phase12_spatial_null_anchor_preserving_permutation_story.csv; anchor_background_suppressed_permutation_comparison/phase12_anchor_background_permutation_story.csv; range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_dataset.csv; lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_method_summary.csv; null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_dataset_comparison.csv; fdr_margin_gap/phase12_fdr_margin_gap_story.csv; spatial_specificity_comparison/phase12_spatial_specificity_story.csv; guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_story.csv; integrated_guarded_spatial_training_permutation_audit/phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv; integrated_guarded_spatial_training_highres_stability/phase12_integrated_guarded_spatial_highres_stability_story.csv; integrated_guarded_spatial_coordinate_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_integrated_coordinate_summary.csv; discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv; goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv", "Permutation whole-atlas BH FDR did not support pooled all-null discoveries for the original reporting baseline. Null-specific diagnostics show receiver-pairing and LR-identity support, but spatial-distance remains the bottleneck; spatial-dependency effect sizes are present but do not pass within-LR BH FDR. Mechanism/category stratification also does not recover true discoveries. Matched spatial, guarded matched, range-weighted, and discovery-eligible diagnostics add useful story/universe-level support, including a clean selected post-Xenium SPP1-CD44 guarded full-LR matched-null result. LR-standardized all-null FDR is a negative diagnostic: standardizing each LR against its own null baseline still yields zero true q<=0.1 hits. Stronger null-margin guarded training improves top-k recovery and hard-negative suppression but exact-score broad FDR still has zero true q<=0.1 discoveries. FDR margin-gap diagnosis shows the dominant blocker remains spatial-distance null across tested variants. The spatial-specificity and guarded spatial-specificity calibration layers shrink the q gap, with guarded spatial-specificity bringing post-Xenium SPP1-CD44 close to q<=0.1, but they remain post-hoc and do not close broad FDR. The integrated guarded spatial training candidate is the first tested candidate to produce a conservative broad true LR q<=0.1 hit, post-Xenium TGFB1-TGFBR2; the hit is stable or stronger after increasing exact-score permutations from 80 to 240, q=0.025 to q=0.00833. A low-permutation coordinate-rerun smoke keeps TGFB1-TGFBR2 rank 1 and above the coordinate-null q95, but p=0.25/BH≈0.96 and high-expression negatives also reach the minimum p. Therefore Goal 1 is improved but remains partial rather than globally closed.", "Treat integrated guarded spatial training as the next-model candidate. The higher-permutation exact-score hit is reproduced and coordinate smoke is directionally supportive; next scale end-to-end nulls, repair CRC/background leakage, and compare against current baseline before making a broad positive whole-atlas discovery-FDR claim."))

    rows.append(row("Goal 1", "Top LR biological category summary", "complete" if not cat.empty else "missing", "direct" if not cat.empty else "missing", f"Category rows={len(cat)} across {cat['dataset_key'].nunique() if not cat.empty else 0} datasets.", "atlas_model_expanded/phase12_expanded_atlas_category_summary.csv", "Other/background category remains large and must not be overinterpreted.", "Use category summary as descriptive context, not ground truth."))

    high_neg_top10 = int(pd.to_numeric(topk.get("top10_high_expression_negative", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not topk.empty else 0
    broad_leak = int(pd.to_numeric(coord_guard.get("n_control_guardrail_broad_background", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not coord_guard.empty else 0
    rows.append(row("Goal 1", "High-expression negative contamination", "complete" if not topk.empty and not coord_guard.empty else "partial", "direct", f"Calibrated top10 high-expression negatives={high_neg_top10}; coordinate guardrail broad/background leakage={broad_leak}.", "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv; coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv", "Top10 high-expression controls are suppressed, but broad/background coordinate leakage remains high.", "Keep high-expression and broad-background leakage separate in claims."))

    receptor_rows = int(hard.get("is_receptor_shared", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not hard.empty else 0
    receptor_beats = int((hard.get("is_receptor_shared", pd.Series(dtype=bool)).fillna(False).astype(bool) & hard.get("negative_beats_positive", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not hard.empty else 0
    if not receptor_atlas.empty:
        atlas_decoys = int(pd.to_numeric(receptor_atlas.get("n_receptor_shared_decoy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        atlas_top10 = int(pd.to_numeric(receptor_atlas.get("top10_receptor_shared_decoy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        atlas_positive_beaten = int(pd.to_numeric(receptor_atlas.get("n_positive_anchors_beaten_by_receptor_shared_decoy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        atlas_pairwise_beats = int(pd.to_numeric(receptor_atlas.get("total_receptor_shared_decoys_beating_positive", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        rows.append(row("Goal 1", "Receptor-shared negative contamination", "complete", "direct", f"Atlas-level receptor-shared decoys={atlas_decoys}; top10 decoys={atlas_top10}; positive anchors beaten={atlas_positive_beaten}; pairwise beats={atlas_pairwise_beats}; stress comparisons={receptor_rows}, stress beats={receptor_beats}.", "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv; receptor_shared_atlas_audit/phase12_receptor_shared_atlas_positive_pair_audit.csv; hard_negative_stress/phase12_hard_negative_stress_comparisons.csv", "Top10 receptor-shared contamination is suppressed, but pairwise leakage remains in lower-ranked anchors; this is a control-contamination audit, not native whole-atlas discovery FDR.", "Use atlas-level receptor-shared audit in the manuscript and keep pairwise leakage as a limitation/calibration target."))
    else:
        rows.append(row("Goal 1", "Receptor-shared negative contamination", "partial" if receptor_rows else "missing", "direct" if receptor_rows else "missing", f"Receptor-shared hard-negative comparisons={receptor_rows}; beats positive={receptor_beats}.", "hard_negative_stress/phase12_hard_negative_stress_comparisons.csv", "Measured in hard-negative stress, not as formal global atlas FDR.", "Promote receptor-shared controls into formal atlas-level calibration if making a global false-positive claim."))

    rows.append(row("Goal 1", "Communication modules", "complete" if not modules.empty else "missing", "direct" if not modules.empty else "missing", f"Expanded atlas modules={len(modules)} across {modules['dataset_key'].nunique() if not modules.empty else 0} datasets.", "atlas_model_expanded/phase12_expanded_atlas_modules.csv", "Modules are descriptive clustering and not independent validation.", "Use with downstream program enrichment to support module biology."))
    return rows


def goal2_rows() -> list[dict[str, object]]:
    expanded_downstream = read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv")
    expanded_genes = read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_gene_table.csv")
    expanded_dataset = read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_dataset_summary.csv")
    expanded_modules = read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_module_summary.csv")
    pilot_downstream = read_csv(PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv")
    pilot_genes = read_csv(PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_gene_table.csv")
    downstream = expanded_downstream if not expanded_downstream.empty else pilot_downstream
    genes = expanded_genes if not expanded_genes.empty else pilot_genes
    enrich = read_csv(PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_enrichment_summary.csv")
    top_genes = read_csv(PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_top_genes.csv")
    commot = read_csv(PHASE12_DIR / "gene_program_enrichment/phase12_commot_impact_comparison_summary.csv")
    expanded_commot_dataset = read_csv(PHASE12_DIR / "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_dataset_summary.csv")
    expanded_commot_comparison = read_csv(PHASE12_DIR / "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_method_comparison.csv")
    pathway = read_csv(PHASE12_DIR / "pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv")
    target_background = read_csv(PHASE12_DIR / "downstream_target_background_audit/phase12_downstream_target_background_dataset_audit.csv")

    datasets = downstream["dataset_key"].nunique() if not downstream.empty else 0
    sig_lr = int(pd.to_numeric(downstream.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).sum()) if not downstream.empty else 0
    module_rep_lr = int(pd.to_numeric(expanded_dataset.get("n_module_representative_lr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not expanded_dataset.empty else 0
    modules_with_support = int(pd.to_numeric(expanded_dataset.get("n_modules_with_target_enriched_rep", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not expanded_dataset.empty else 0
    downstream_file = "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv" if not expanded_downstream.empty else "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv"
    gene_file = "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_gene_table.csv" if not expanded_genes.empty else "atlas_downstream_scan/phase12_atlas_downstream_gene_table.csv"
    downstream_status = "complete" if (not expanded_downstream.empty and datasets >= 2 and module_rep_lr > 0) else ("partial" if datasets >= 2 else "missing")
    downstream_proof = f"Expanded downstream LR rows={len(downstream)}, datasets={datasets}, target-enriched LR q<=0.1={sig_lr}, module-representative LR={module_rep_lr}, modules with enriched representative={modules_with_support}." if not expanded_downstream.empty else f"Downstream LR rows={len(downstream)}, datasets={datasets}, target-enriched LR q<=0.1={sig_lr}."
    downstream_caveat = "Covers calibrated top LR, priority anchors and module representatives, but not every expanded atlas LR." if not expanded_downstream.empty else "Systematic for selected top/anchor LR, but not every expanded atlas LR/module."
    downstream_next = "Use as Phase 12 functional-coverage source data; expand to all 886/957 LR only if claiming whole-atlas target support." if not expanded_downstream.empty else "Expand to all top LR/modules selected in final Figure 1 and Figure 4."
    expanded_commot_datasets = expanded_commot_dataset["dataset_key"].nunique() if not expanded_commot_dataset.empty else 0
    if not expanded_commot_dataset.empty:
        ss_hits = int(expanded_commot_dataset[
            expanded_commot_dataset["method"].astype(str).eq("spatialself_calibrated")
        ]["n_target_enriched_q_0_1"].sum())
        cm_hits = int(expanded_commot_dataset[
            expanded_commot_dataset["method"].astype(str).eq("commot_adapter")
        ]["n_target_enriched_q_0_1"].sum())
        call_counts = expanded_commot_comparison["comparison_call"].value_counts().to_dict() if not expanded_commot_comparison.empty else {}
        call_text = "; ".join(f"{k}={int(v)}" for k, v in call_counts.items())
        commot_status = "complete" if expanded_commot_datasets >= 2 else "partial"
        commot_strength = "direct"
        commot_proof = f"Expanded COMMOT-style comparison datasets={expanded_commot_datasets}; SpatialSelf target-enriched LR={ss_hits}; COMMOT adapter target-enriched LR={cm_hits}; calls: {call_text}."
        commot_file = "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_method_comparison.csv; expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_dataset_summary.csv"
        commot_caveat = "Same-universe adapter/downstream proxy, not fully tuned official COMMOT communication_impact. Lymph-node CCL19/CCL21-CCR7 is stronger for COMMOT, while CXCL12-CXCR4 and post-Xenium anchors favor SpatialSelf."
        commot_next = "Use as final Phase 12 COMMOT-style impact comparison unless official COMMOT communication_impact is rerun end-to-end."
    else:
        commot_status = "partial" if not commot.empty else "missing"
        commot_strength = "direct" if not commot.empty else "missing"
        commot_proof = f"COMMOT-style impact rows={len(commot)}, datasets={commot['dataset_key'].nunique() if not commot.empty else 0}."
        commot_file = "gene_program_enrichment/phase12_commot_impact_comparison_summary.csv"
        commot_caveat = "Reused Phase 10 impact summaries; not newly computed for every Phase 12 dataset."
        commot_next = "Recompute or clearly label reused COMMOT-style impact for final datasets."
    if not target_background.empty:
        tb_datasets = int(target_background["dataset_key"].nunique())
        tb_scanned = int(pd.to_numeric(target_background.get("n_lr_scanned", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        tb_with_targets = int(pd.to_numeric(target_background.get("n_lr_with_curated_targets", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        tb_supported = int(pd.to_numeric(target_background.get("n_target_supported_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        tb_true_supported = int(pd.to_numeric(target_background.get("n_true_lr_target_supported_q_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        tb_leakage = int(pd.to_numeric(target_background.get("n_high_expression_negative_target_leakage", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        target_rank_status = "complete" if tb_datasets >= 2 and tb_scanned > 100 and tb_supported > 0 else "partial"
        target_rank_strength = "direct"
        target_rank_proof = f"Target-background audit datasets={tb_datasets}, scanned LR={tb_scanned}, LR with curated targets={tb_with_targets}, target-supported q<=0.1={tb_supported}, true-supported={tb_true_supported}, high-expression target leakage={tb_leakage}."
        target_rank_file = "downstream_target_background_audit/phase12_downstream_target_background_dataset_audit.csv; downstream_target_background_audit/phase12_downstream_target_background_lr_audit.csv"
        target_rank_caveat = "This completes the selected top/priority/module LR target-background audit, but support is concentrated in a subset and many scanned LR lack curated target sets. It is not pan-atlas functional validation for every LR."
        target_rank_next = "Use the audit as manuscript source data; expand curated target resources only if claiming broader target coverage."
    else:
        target_rank_status = "partial" if sig_lr > 0 else "not_supported"
        target_rank_strength = "direct" if not downstream.empty else "missing"
        target_rank_proof = f"Target top300 enrichment q<=0.1 LR={sig_lr}; pathway clean support datasets={int(pd.to_numeric(pathway.get('n_clean_specific_support', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not pathway.empty else 0}; expanded modules with enriched representatives={modules_with_support}."
        target_rank_file = f"{downstream_file}; expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_module_summary.csv; pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv"
        target_rank_caveat = "Target evidence is concentrated in a subset of LR and modules; this supports selected communication programs, not pan-atlas functional validation."
        target_rank_next = "Do not claim every predicted LR has downstream target support."

    rows = [
        row("Goal 2", "Whole-transcriptome downstream association", downstream_status, "direct" if datasets else "missing", downstream_proof, downstream_file, downstream_caveat, downstream_next),
        row("Goal 2", "Communication-dependent gene table", "complete" if not genes.empty or not top_genes.empty else "missing", "direct" if not genes.empty or not top_genes.empty else "missing", f"Gene-level downstream rows={len(genes)}; program top-gene rows={len(top_genes)}.", f"{gene_file}; gene_program_enrichment/phase12_gene_program_top_genes.csv", "Gene associations are observational and residualized, not perturbational.", "Keep source-data gene tables for final plotted gene programs."),
        row("Goal 2", "GO/Reactome/KEGG/NicheNet-style enrichment", "complete" if not enrich.empty else "missing", "direct" if not enrich.empty else "missing", f"Enrichment rows={len(enrich)}, libraries={','.join(sorted(enrich['library'].astype(str).unique())) if not enrich.empty else 'none'}.", "gene_program_enrichment/phase12_gene_program_enrichment_summary.csv", "NicheNet target ranking exists through target-enrichment layer; exact library coverage depends on available resources.", "Separate broad housekeeping terms from specific terms in final figures."),
        row("Goal 2", "COMMOT communication-impact comparison", commot_status, commot_strength, commot_proof, commot_file, commot_caveat, commot_next),
        row("Goal 2", "Target genes ranked above background", target_rank_status, target_rank_strength, target_rank_proof, target_rank_file, target_rank_caveat, target_rank_next)
    ]
    return rows


def goal3_rows() -> list[dict[str, object]]:
    templates = read_csv(PHASE12_DIR / "atlas_readiness/phase12_hard_negative_templates.csv")
    hard = read_csv(PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_comparisons.csv")
    edge = read_csv(PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_class_summary.csv")
    receiver_shuffle = read_csv(PHASE12_DIR / "receiver_shuffle_hard_negative/phase12_receiver_shuffle_dataset_summary.csv")
    required_classes = {
        "same receptor": "housekeeping_ligand_same_receptor",
        "same ligand": "ligand_shared",
        "housekeeping ligand + real receptor": "housekeeping_ligand_same_receptor",
        "high-expression false LR": "high_expression_false_lr_edges",
        "wrong cell type": "same_lr_wrong_type",
        "wrong niche": "same_lr_wrong_niche",
        "shuffled LR identity": "same_spatial_edge_wrong_lr_identity",
        "shuffled receiver": "receiver_shuffle",
        "spatially shifted sender": "same_lr_spatially_shifted_sender_proxy",
    }
    available = set(templates.get("negative_type", pd.Series(dtype=str)).astype(str).unique()) | set(edge.get("negative_class", pd.Series(dtype=str)).astype(str).unique())
    if not receiver_shuffle.empty:
        available.add("receiver_shuffle")
    class_notes = []
    missing = []
    for label, token in required_classes.items():
        found = token in available or (label == "same ligand" and hard.get("is_ligand_shared", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
        class_notes.append(f"{label}={'yes' if found else 'no'}")
        if not found:
            missing.append(label)
    rows = [
        row("Goal 3", "Hard-negative class coverage", "partial" if missing else "complete", "direct", "; ".join(class_notes), "atlas_readiness/phase12_hard_negative_templates.csv; edge_hard_negative_audit/phase12_edge_hard_negative_class_summary.csv", f"Missing or indirect classes: {', '.join(missing) if missing else 'none'}", "Add explicit shuffled-receiver controls if required for final methods."),
        row("Goal 3", "Hard-negative metrics: rank/score/FDR/leakage", "complete" if not hard.empty and not edge.empty and not receiver_shuffle.empty else "partial", "direct" if not hard.empty and not edge.empty else "missing", f"Pairwise hard-negative comparisons={len(hard)}; edge class rows={len(edge)}; receiver-shuffle datasets={receiver_shuffle['dataset_key'].nunique() if not receiver_shuffle.empty else 0}.", "hard_negative_stress/phase12_hard_negative_stress_comparisons.csv; edge_hard_negative_audit/phase12_edge_hard_negative_class_summary.csv; receiver_shuffle_hard_negative/phase12_receiver_shuffle_dataset_summary.csv", "Target-overlap/deltaR2 are available mainly through downstream/COMMOT-style layers, not every hard-negative class. Receiver shuffle is fixed-score, not model retraining.", "Unify target-overlap/deltaR2 with hard-negative classes if making a single stress-test metric."),
    ]
    return rows


def goal4_rows() -> list[dict[str, object]]:
    strict = read_csv(PHASE12_DIR / "strict_story_candidates/phase12_strict_story_candidate_evidence.csv")
    cross = read_csv(PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv")
    closure = read_csv(PHASE12_DIR / "second_story_closure/phase12_second_story_closure_summary.csv")
    cross_closure = read_csv(PHASE12_DIR / "cross_platform_closure_matrix/phase12_cross_platform_closure_dataset_summary.csv")
    cross_story = read_csv(PHASE12_DIR / "cross_platform_closure_matrix/phase12_cross_platform_closure_story_summary.csv")
    n_a = int(strict["strict_story_class"].astype(str).eq("A_main_story_ready").sum()) if not strict.empty else 0
    n_b = int(strict["strict_story_class"].astype(str).eq("B_biological_consistency_with_stat_support").sum()) if not strict.empty else 0
    n_closure_a = int(pd.to_numeric(closure.get("n_A_candidates", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not closure.empty else 0
    best_closure = ""
    if not closure.empty:
        top = closure.sort_values(["n_A_candidates", "best_closure_score"], ascending=[False, False]).head(1)
        best_closure = f"; second-story closure best={top.iloc[0].best_candidate} ({top.iloc[0].best_closure_class})"
    if not cross_closure.empty:
        n_cross_datasets = int(cross_closure["dataset_key"].nunique())
        n_supportive = int(pd.to_numeric(cross_closure.get("n_supportive_or_better", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum())
        n_cross_a = int(pd.to_numeric(cross_closure.get("n_A_candidates", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        overall = str(cross_story["overall_cross_platform_closure_call"].iloc[0]) if not cross_story.empty else ""
        cross_status = "complete" if n_cross_datasets >= 4 and n_supportive >= 2 and n_cross_a >= 1 else "partial"
        cross_proof = f"Cross-platform closure datasets={n_cross_datasets}; supportive-or-better datasets={n_supportive}; A candidates={n_cross_a}; overall={overall}."
        cross_file = "cross_platform_closure_matrix/phase12_cross_platform_closure_dataset_summary.csv; cross_platform_closure_matrix/phase12_cross_platform_closure_candidate_matrix.csv"
        cross_caveat = "This completes a conservative cross-platform closure audit, not a claim that every platform is clean. Tonsil is supportive with downstream/negative-leakage caveats; melanoma remains not clean."
        cross_next = "Use lymph-node CXCL12-CXCR4 as conservative second story and tonsil as supportive platform validation; do not use melanoma as success under current controls."
    else:
        cross_status = "partial" if not cross.empty else "missing"
        cross_proof = f"Cross-platform audit datasets={cross['dataset_key'].nunique() if not cross.empty else 0}."
        cross_file = "cross_platform_story/phase12_cross_platform_story_audit.csv"
        cross_caveat = "Tonsil supports ranking/local FDR but has targeted negative leakage; melanoma is not clean."
        cross_next = "Choose one tonsil/melanoma candidate for deeper downstream/FDR closure."
    rows = [
        row("Goal 4", "Second strong biological story", "complete" if n_closure_a > 0 else ("partial" if n_b > 0 else "missing"), "direct" if not strict.empty or not closure.empty else "missing", f"A-class main stories={n_a}; B-class biological-consistency candidates={n_b}; closure A-candidates={n_closure_a}{best_closure}.", "strict_story_candidates/phase12_strict_story_candidate_evidence.csv; second_story_closure/phase12_second_story_closure_summary.csv", "The current A-class non-CRC closure candidate is conservative and observational; it should not be framed as perturbational causality or unbiased whole-atlas discovery.", "Use CXCL12-CXCR4 as a conservative lymph-node second story, while keeping CCL19/CCL21-CCR7 as supportive chemokine biology."),
        row("Goal 4", "Cross-platform validation beyond CRC", cross_status, "direct" if not cross.empty or not cross_closure.empty else "missing", cross_proof, cross_file, cross_caveat, cross_next),
    ]
    return rows


def goal5_rows() -> list[dict[str, object]]:
    stat = read_csv(PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_lr_summary.csv")
    guardrail = read_csv(PHASE12_DIR / "directionality_claim_guardrail/phase12_directionality_claim_guardrail_dataset.csv")
    perm = read_csv(PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_dataset_summary.csv")
    edge = read_csv(PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv")
    consensus = read_csv(PHASE12_DIR / "directionality_consensus/phase12_directionality_consensus_dataset_summary.csv")
    coord = read_csv(PHASE12_DIR / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv")
    coord_closure = read_csv(PHASE12_DIR / "coordinate_null_closure/phase12_coordinate_null_closure_dataset.csv")
    coord_story = read_csv(PHASE12_DIR / "coordinate_null_closure/phase12_coordinate_null_closure_story.csv")
    receiver_hits = int(pd.to_numeric(edge.get("n_receiver_pairing_fdr_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not edge.empty else 0
    vector_hits = int(pd.to_numeric(edge.get("n_edge_vector_fdr_0_1", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not edge.empty else 0
    consensus_supported = int(pd.to_numeric(consensus.get("n_consensus_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not consensus.empty else 0
    strict_vector_consensus = int(pd.to_numeric(consensus.get("n_strict_edge_vector_consensus", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not consensus.empty else 0
    coord_reruns = 0
    for path in (PHASE12_DIR / "coordinate_model_rerun_smoke").glob("*/*_coordinate_model_rerun_summary.csv"):
        df = read_csv(path)
        if not df.empty and "n_completed_permutations" in df:
            coord_reruns += int(pd.to_numeric(df["n_completed_permutations"], errors="coerce").fillna(0).max())
    if not coord_closure.empty:
        closure_datasets = int(coord_closure["dataset_key"].nunique())
        story_supported = int(pd.to_numeric(coord_closure.get("n_story_coordinate_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        true_story_supported = int(pd.to_numeric(coord_closure.get("n_true_story_coordinate_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        high_story = int(pd.to_numeric(coord_closure.get("n_high_expression_story_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        broad_leak = int(pd.to_numeric(coord_closure.get("n_broad_background_leakage", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        overall = str(coord_story["overall_coordinate_null_closure_call"].iloc[0]) if not coord_story.empty else ""
        coord_status = "complete" if closure_datasets >= 2 and coord_reruns >= 40 and story_supported > 0 and high_story == 0 else "partial"
        coord_proof = f"Coordinate model reruns completed={coord_reruns}; coordinate-null closure datasets={closure_datasets}; story-supported LR={story_supported}; true story-supported={true_story_supported}; high-expression story-supported={high_story}; broad-background leakage={broad_leak}; overall={overall}."
        coord_file = "coordinate_model_rerun_smoke/*/*_coordinate_model_rerun_summary.csv; coordinate_null_closure/phase12_coordinate_null_closure_dataset.csv; coordinate_null_closure/phase12_coordinate_null_closure_story.csv"
        coord_caveat = "End-to-end coordinate reruns support selected story LR after guardrails, but broad-background leakage blocks any whole-atlas coordinate-null discovery claim."
        coord_next = "Use as story-level coordinate-null support only; scale reruns or improve broad-background calibration before claiming whole-atlas coordinate FDR."
    else:
        coord_status = "partial" if coord_reruns >= 40 else ("partial" if coord_reruns else "missing")
        coord_proof = f"Coordinate model reruns completed={coord_reruns}; coordinate story guardrail datasets={len(coord)}."
        coord_file = "coordinate_model_rerun_smoke/*/*_coordinate_model_rerun_summary.csv; coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv"
        coord_caveat = "Reruns are useful but low-resolution; nominal discoveries are contaminated by high-expression/broad-background controls."
        coord_next = "Scale or replace with stricter formal null if claiming coordinate-level FDR."
    return [
        row("Goal 5", "Quantified source-sink/vector/boundary metrics", "complete" if not stat.empty and not guardrail.empty else ("partial" if not stat.empty else "missing"), "direct" if not stat.empty else "missing", f"Directionality LR rows={len(stat)}; guardrail dataset rows={len(guardrail)}.", "directionality_statistics/phase12_directionality_statistical_lr_summary.csv; directionality_claim_guardrail/phase12_directionality_claim_guardrail_dataset.csv", "Metrics are descriptive/fixed-score and the guardrail explicitly blocks causal-transport or global-superiority language.", "Use guardrail claim calls in manuscript wording."),
        row("Goal 5", "Permutation p-values for directionality", "complete" if not perm.empty and not edge.empty and not consensus.empty else ("partial" if not perm.empty or not edge.empty else "missing"), "direct" if not perm.empty or not edge.empty else "missing", f"Aggregated label-permutation datasets={len(perm)}; edge receiver q<=0.1 hits={receiver_hits}; vector q<=0.1 hits={vector_hits}; consensus-supported LR={consensus_supported}; strict vector consensus={strict_vector_consensus}.", "directionality_permutation/phase12_directionality_permutation_dataset_summary.csv; directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv; directionality_consensus/phase12_directionality_consensus_dataset_summary.csv", "Permutation p-values and consensus calls are available, but vector-null support is weak and fixed-score edge null is not model retraining.", "Avoid claiming global directionality superiority or causal ligand transport."),
        row("Goal 5", "End-to-end coordinate-rerun null", coord_status, "direct" if coord_reruns else "missing", coord_proof, coord_file, coord_caveat, coord_next)
    ]


def goal6_rows() -> list[dict[str, object]]:
    fig_status = read_csv(PHASE12_DIR / "figures/phase12_rendered_figure_status.csv")
    panel = read_csv(PHASE12_DIR / "manuscript_figure_package/phase12_manuscript_figure_panel_index.csv")
    claim = read_csv(PHASE12_DIR / "claim_matrix/phase12_claim_matrix.csv")
    all_figs = bool(not fig_status.empty and fig_status.get("all_expected_outputs_exist", pd.Series(dtype=bool)).fillna(False).astype(bool).all())
    return [
        row("Goal 6", "Four manuscript-grade draft figures", "complete" if all_figs and len(panel) == 16 else "partial", "direct" if not fig_status.empty else "missing", f"Rendered figures={len(fig_status)}, all outputs={all_figs}, panels={len(panel)}.", "figures/phase12_rendered_figure_status.csv; manuscript_figure_package/phase12_manuscript_figure_panel_index.csv", "Figures are draft manuscript art, not final journal submission.", "Finalize after FDR/baseline claims are settled."),
        row("Goal 6", "Claim/story consolidation", "complete" if not claim.empty else "missing", "direct" if not claim.empty else "missing", f"Claim matrix rows={len(claim)}.", "claim_matrix/phase12_claim_matrix.csv", "Claim matrix explicitly blocks whole-atlas superiority and global directionality superiority.", "Use claim matrix as writing guardrail.")
    ]


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    order = ["complete", "partial", "not_supported", "missing"]
    rows = []
    for goal, sub in audit.groupby("goal", sort=False):
        counts = sub["completion_status"].value_counts()
        n_total = len(sub)
        n_complete = int(counts.get("complete", 0))
        blockers = int(counts.get("not_supported", 0) + counts.get("missing", 0))
        if n_complete == n_total:
            overall = "complete"
        elif blockers:
            overall = "incomplete_with_hard_gaps"
        else:
            overall = "partial"
        rows.append({
            "goal": goal,
            "overall_status": overall,
            "n_requirements": n_total,
            "n_complete": n_complete,
            "n_partial": int(counts.get("partial", 0)),
            "n_not_supported": int(counts.get("not_supported", 0)),
            "n_missing": int(counts.get("missing", 0)),
            "blocking_requirements": "; ".join(sub[sub["completion_status"].isin(["not_supported", "missing"])]["requirement"].tolist()),
        })
    return pd.DataFrame(rows)


def row(goal: str, requirement: str, status: str, evidence_strength: str, proof: str, evidence_file: str, caveat: str, next_action: str) -> dict[str, object]:
    return {
        "goal": goal,
        "requirement": requirement,
        "completion_status": status,
        "evidence_strength": evidence_strength,
        "proof": proof,
        "evidence_file": rels(evidence_file),
        "caveat_or_contradiction": caveat,
        "next_action": next_action,
    }


def write_status(audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Requirement-Level Completion Audit",
        "",
        "## Scope",
        "",
        "This audit maps the user-defined Phase 12 goals to current evidence. A green-looking figure or status table is not treated as completion unless the evidence directly covers the requirement.",
        "",
        "## Goal Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- **{r.goal}**: `{r.overall_status}`; complete={int(r.n_complete)}/{int(r.n_requirements)}, "
            f"partial={int(r.n_partial)}, not-supported={int(r.n_not_supported)}, missing={int(r.n_missing)}"
        )
        if str(r.blocking_requirements):
            lines.append(f"  - hard gaps: {r.blocking_requirements}")
    lines.extend(["", "## Main Claim Boundary", ""])
    lines.append(
        "Phase 12 currently supports an expanded-atlas, hard-negative-aware, receiver-program-supported manuscript story, but it does not yet prove whole-atlas FDR-controlled discovery or fully tuned official-baseline superiority."
    )
    (OUT_DIR / "phase12_completion_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_requirement_completion_audit",
        "n_requirements": int(len(audit)),
        "n_complete": int(audit["completion_status"].eq("complete").sum()),
        "outputs": {
            "requirement_audit": rel(OUT_DIR / "phase12_requirement_completion_audit.csv"),
            "goal_summary": rel(OUT_DIR / "phase12_goal_completion_summary.csv"),
            "status": rel(OUT_DIR / "phase12_completion_audit_status.md"),
        },
        "claim_boundary": "Completion audit only; incomplete requirements keep Phase 12 active.",
    }
    (OUT_DIR / "phase12_completion_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_integrated_coordinate_smoke() -> pd.DataFrame:
    paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke").glob(
            "*/*_integrated_coordinate_summary.csv"
        )
    )
    frames = []
    for path in paths:
        frame = read_csv(path)
        if frame.empty:
            continue
        frame["supporting_file"] = rel(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["n_completed_permutations"] = pd.to_numeric(
        table["n_completed_permutations"],
        errors="coerce",
    ).fillna(0)
    return table.sort_values(["dataset_key", "n_completed_permutations"]).groupby(
        "dataset_key",
        as_index=False,
    ).tail(1)


def n_datasets(df: pd.DataFrame) -> int:
    return int(df["dataset_key"].nunique()) if not df.empty and "dataset_key" in df else 0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def rels(paths: str) -> str:
    out = []
    for item in str(paths).split(";"):
        item = item.strip()
        if not item:
            continue
        out.append(rel(PHASE12_DIR / item) if not item.startswith("outputs/") else item)
    return "; ".join(out)


if __name__ == "__main__":
    main()
