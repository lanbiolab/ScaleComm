#!/usr/bin/env python
"""Summarize Phase 12 supported claims, limitations, and next-model requirements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "final_limitations_next_model"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(build_rows())
    table.to_csv(OUT_DIR / "phase12_final_limitations_next_model.csv", index=False)
    write_status(table)
    manifest = {
        "task": "phase12_final_limitations_next_model",
        "outputs": [
            rel(OUT_DIR / "phase12_final_limitations_next_model.csv"),
            rel(OUT_DIR / "phase12_final_limitations_next_model_status.md"),
        ],
        "claim_boundary": (
            "Phase 12 supports ranking/control-panel/story-level validation, but broad whole-atlas "
            "discovery FDR remains incomplete."
        ),
    }
    (OUT_DIR / "phase12_final_limitations_next_model_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


def build_rows() -> list[dict[str, object]]:
    fdr = read_csv(PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv")
    fdr_interpretation = read_csv(PHASE12_DIR / "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_story.csv")
    closure_contract = read_csv(PHASE12_DIR / "goal1_closure_contract/phase12_goal1_closure_contract_status.csv")
    bottleneck = read_csv(PHASE12_DIR / "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_story.csv")
    variants = read_csv(PHASE12_DIR / "goal1_model_variant_scoreboard/phase12_goal1_model_variant_summary.csv")
    background = read_csv(
        PHASE12_DIR
        / "background_suppressed_permutation_comparison/phase12_background_suppressed_permutation_story.csv"
    )
    guarded = read_csv(PHASE12_DIR / "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_story.csv")
    spatial_prototype = read_csv(
        PHASE12_DIR
        / "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_story.csv"
    )
    training_smoke = read_csv(PHASE12_DIR / "spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv")
    trained_variant = read_csv(
        PHASE12_DIR / "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv"
    )
    anchor_variant = read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv"
    )
    anchor_permutation = read_csv(
        PHASE12_DIR
        / "spatial_null_anchor_preserving_permutation_comparison/"
        / "phase12_spatial_null_anchor_preserving_permutation_story.csv"
    )
    anchor_background = read_csv(
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_comparison/"
        / "phase12_anchor_background_permutation_story.csv"
    )
    anchor_background_guarded_training = read_csv(
        PHASE12_DIR
        / "anchor_background_guarded_training_comparison/"
        / "phase12_anchor_background_guarded_training_story.csv"
    )
    anchor_background_range_guarded_training = read_csv(
        PHASE12_DIR
        / "anchor_background_range_guarded_training_comparison/"
        / "phase12_anchor_background_range_guarded_training_story.csv"
    )
    variant_matched_spatial = read_csv(
        PHASE12_DIR / "variant_matched_spatial_fdr/phase12_variant_matched_spatial_story.csv"
    )
    lr_standardized_null = read_csv(
        PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_story.csv"
    )
    null_margin_guarded = read_csv(
        PHASE12_DIR / "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_story.csv"
    )
    fdr_margin_gap = read_csv(PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_story.csv")
    spatial_specificity = read_csv(
        PHASE12_DIR / "spatial_specificity_comparison/phase12_spatial_specificity_story.csv"
    )
    guarded_spatial_specificity = read_csv(
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
    integrated_highres_stability = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_highres_stability/"
        / "phase12_integrated_guarded_spatial_highres_stability_story.csv"
    )
    integrated_coordinate_smoke = best_integrated_coordinate_smoke()
    integrated_coordinate_formal_smoke = best_integrated_coordinate_formal_parameter_smoke()
    integrated_coordinate_summary_only = best_integrated_coordinate_summary_only_smoke()
    coordinate_runner_concordance = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_runner_concordance/"
        / "phase12_coordinate_runner_concordance_summary.csv"
    )
    integrated_coordinate_leakage = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_leakage/"
        / "phase12_integrated_coordinate_leakage_summary.csv"
    )
    integrated_coordinate_eligible = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_eligible_audit/"
        / "phase12_integrated_coordinate_eligible_dataset.csv"
    )
    integrated_coordinate_magnitude_gate = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_magnitude_gate/"
        / "phase12_integrated_coordinate_magnitude_gate_best.csv"
    )
    integrated_coordinate_background_targets = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_background_targets/"
        / "phase12_integrated_coordinate_background_target_summary.csv"
    )
    integrated_background_guard_targets = read_csv(
        PHASE12_DIR
        / "integrated_background_guard_training_targets/"
        / "phase12_integrated_background_guard_training_spec.csv"
    )
    integrated_background_target_guarded_smoke = read_csv(
        PHASE12_DIR
        / "integrated_background_target_guarded_training_smoke/"
        / "phase12_integrated_background_target_guarded_dataset_summary.csv"
    )
    integrated_background_target_guarded_permutation_smoke = read_csv(
        PHASE12_DIR
        / "integrated_background_target_guarded_permutation_smoke/"
        / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv"
    )
    integrated_background_target_guarded_coordinate_smoke = read_csv(
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke/"
        / "post_xenium_colon_full/"
        / "post_xenium_colon_full_target_guarded_coordinate_summary.csv"
    )
    disk_cleanup = read_csv(PHASE12_DIR / "disk_cleanup_audit/phase12_disk_cleanup_candidates.csv")
    hardlink_dedup_dry_run = read_csv(
        PHASE12_DIR / "disk_cleanup_audit/phase12_replicate_hardlink_dedup_dry_run.csv"
    )
    hardlink_dedup_executed = read_csv(
        PHASE12_DIR / "disk_cleanup_audit/phase12_replicate_hardlink_dedup_executed.csv"
    )
    tiff_cleanup_executed = read_csv(
        PHASE12_DIR / "disk_cleanup_audit/phase12_regenerable_tiff_cleanup_executed.csv"
    )
    null_component = read_csv(PHASE12_DIR / "null_component_diagnostic/phase12_null_component_story.csv")
    background_guarded_training = read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_background_guarded_training/"
        / "phase12_background_guarded_training_comparison.csv"
    )
    background_guarded_permutation = read_csv(
        PHASE12_DIR
        / "background_guarded_training_permutation_audit/"
        / "phase12_background_guarded_training_permutation_dataset_summary.csv"
    )
    range_violation_training = read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_range_violation_guarded_training/"
        / "phase12_range_violation_guarded_training_comparison.csv"
    )
    range_violation_permutation = read_csv(
        PHASE12_DIR
        / "range_violation_guarded_training_permutation_audit/"
        / "phase12_range_violation_guarded_training_permutation_dataset_summary.csv"
    )
    supplementary_panel = read_csv(
        PHASE12_DIR
        / "supplementary_cross_platform_expanded_panel_atlas"
        / "phase12_supplementary_expanded_panel_dataset_summary.csv"
    )
    completion = read_csv(PHASE12_DIR / "completion_audit/phase12_goal_completion_summary.csv")
    rows = [
        row(
            "Current reporting-baseline whole-atlas discovery FDR",
            "not_closed_for_reporting_baseline",
            f"{int(fdr['broad_true_lr_q_0_1'].sum()) if len(fdr) else 0} broad true LR discoveries at q<=0.1 for the current reporting baseline across expanded atlases.",
            "Do not claim global whole-atlas FDR control or unbiased atlas-scale discovery for the current reporting baseline.",
            "Use the integrated guarded spatial candidate as the follow-up model direction, then rerun end-to-end permutation audits before baseline replacement.",
            "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv",
        ),
        row(
            "Goal 1 FDR interpretation framework",
            "claim_boundary_framework_available",
            fdr_interpretation_summary(fdr_interpretation),
            "This framework collates existing audits and does not create a new FDR procedure or close broad all-covered-LR FDR.",
            "Use the layer table as the manuscript guardrail and as the model-design target: preserve selected guarded/eligible signals while fixing pooled spatial-null and broad-background leakage.",
            "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_story.csv",
        ),
        row(
            "Goal 1 closure contract",
            "not_ready_for_goal1_closure",
            goal1_closure_contract_summary(closure_contract),
            "This is a requirement contract, not new discovery evidence. It should prevent diagnostic gates from being written as final FDR closure.",
            "A future baseline replacement must satisfy the contract: strict broad FDR, clean hard controls, and end-to-end coordinate/null support.",
            "goal1_closure_contract/phase12_goal1_closure_contract_status.csv",
        ),
        row(
            "Spatial-distance null",
            "primary_bottleneck",
            bottleneck_summary(bottleneck),
            "Matched/range diagnostics are useful but cannot replace conservative pooled spatial FDR.",
            "Add a model objective or calibration layer that preserves range-matched anchors while suppressing pooled spatial-null leakage.",
            "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_story.csv",
        ),
        row(
            "Guarded matched full-LR FDR",
            "selected_support_not_broad_closure",
            guarded_summary(guarded),
            "This does not replace conservative pooled spatial-distance FDR and does not support lymph node or CRC as guarded full-LR discoveries.",
            "Use this as selected post-Xenium SPP1-CD44 statistical support; next model still needs pooled-spatial/background-null training.",
            "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_story.csv",
        ),
        row(
            "Broad/background LR leakage",
            "quantified_not_solved",
            "Post-Xenium and Visium HD CRC still show broad/background LR contamination in top-ranked regions.",
            "Do not present CRC/native melanoma broad-atlas outputs as clean discovery stories under current controls.",
            "Use category-balanced hard negatives and broad-expression suppression inside training, not only post-hoc score adjustment.",
            "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_dataset.csv",
        ),
        row(
            "Supplementary expanded-panel cross-platform atlas",
            "supportive_not_strict_fdr_closure",
            supplementary_panel_summary(supplementary_panel),
            "This is 2011-gene expanded-panel support on constructed real-background replicates, not strict full-transcriptome whole-atlas FDR closure for tonsil or melanoma.",
            "Use it as cross-platform support and keep hard-negative leakage visible; do not merge it with the strict 886/957-LR full-transcriptome atlas claim.",
            "supplementary_cross_platform_expanded_panel_atlas/phase12_supplementary_expanded_panel_dataset_summary.csv",
        ),
        row(
            "Main calibrated ranking",
            "current_reporting_baseline",
            variant_summary(variants, "main_calibrated"),
            "This supports calibrated ranking/control-panel reporting, not broad discovery FDR.",
            "Keep as current baseline until a new model variant passes exact-score and control audits.",
            "goal1_model_variant_scoreboard/phase12_goal1_model_variant_summary.csv",
        ),
        row(
            "Background-suppressed variant",
            "candidate_component_not_replacement",
            background_summary(background, variants),
            "Do not replace the main model with this variant yet; exact-score permutation still has zero q<=0.1 discoveries.",
            "Move background suppression into the training/calibration design, then rerun the full null audit.",
            "background_suppressed_permutation_comparison/phase12_background_suppressed_permutation_story.csv",
        ),
        row(
            "Spatial-null-aware score prototype",
            "ranking_direction_not_fdr_closure",
            spatial_prototype_summary(spatial_prototype),
            "This is an exact-score audit of a score-level prototype, not a retrained model and not broad whole-atlas FDR closure.",
            "Use it to define the next training loss: suppress broad/background LR and pooled spatial-null leakage while preserving true anchors.",
            "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_story.csv",
        ),
        row(
            "Spatial-null contrastive training",
            "training_path_smoke_passed",
            training_smoke_summary(training_smoke),
            "This is only a one-dataset, one-epoch smoke test; it does not prove FDR improvement.",
            "Run full expanded-atlas training with this loss, then rerun exact-score and end-to-end permutation audits.",
            "spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv",
        ),
        row(
            "Spatial-null-trained expanded atlas variant",
            "mixed_not_reporting_baseline",
            trained_variant_summary(trained_variant),
            "The variant improves post-Xenium but worsens lymph node/CRC top10 true recovery, so it should not replace the current main calibrated ranking.",
            "Try dataset-adaptive spatial-null weighting, loss annealing, or anchor-preserving regularization before rerunning exact permutation FDR.",
            "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv",
        ),
        row(
            "Anchor-preserving spatial-null variant",
            "best_candidate_exact_audit_not_closed",
            f"{anchor_variant_summary(anchor_variant)} {anchor_permutation_summary(anchor_permutation)}",
            "This preserves/improves selected top-k diagnostics but still underperforms main in CRC and exact-score whole-atlas FDR remains unclosed.",
            "Keep it as the next model candidate; add CRC/anchor recovery safeguards or dataset-adaptive spatial-null weighting, then rerun exact-score and end-to-end null audits before baseline replacement.",
            "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv; spatial_null_anchor_preserving_permutation_comparison/phase12_spatial_null_anchor_preserving_permutation_story.csv",
        ),
        row(
            "Anchor plus background suppression",
            "calibration_direction_not_fdr_closure",
            anchor_background_summary(anchor_background),
            "This is a post-hoc calibration diagnostic. It improves top-k recovery but does not close broad whole-atlas FDR and leaves CRC high-expression leakage.",
            "Move the useful pieces into training: dataset-adaptive spatial-null loss plus broad/background suppression with explicit high-expression guards.",
            "anchor_background_suppressed_permutation_comparison/phase12_anchor_background_permutation_story.csv",
        ),
        row(
            "Anchor/background guarded training",
            "training_integration_not_fdr_closure",
            anchor_background_guarded_training_summary(anchor_background_guarded_training),
            "This training-integrated version improves top-k recovery and hard-negative suppression, but exact-score broad whole-atlas FDR remains unclosed and post-Xenium risk can be worse than the post-hoc anchor/background score.",
            "Keep the training integration idea, but add stronger spatial-distance null separation and a risk-regression guard against the post-hoc anchor/background score before replacing the reporting baseline.",
            "anchor_background_guarded_training_comparison/phase12_anchor_background_guarded_training_story.csv",
        ),
        row(
            "Anchor/background range-guarded training",
            "range_null_training_not_fdr_closure",
            anchor_background_range_guarded_training_summary(anchor_background_range_guarded_training),
            "This low-weight range-violation spatial-null version preserves top-k anchors and improves selected best true q-values, but exact-score broad whole-atlas FDR remains unclosed.",
            "Treat simple range-violation receiver sampling as insufficient. Next step should be a null-specific calibration/reporting layer or dataset-adaptive loss that directly reduces spatial-distance global BH q-values without increasing post-hoc control risk.",
            "anchor_background_range_guarded_training_comparison/phase12_anchor_background_range_guarded_training_story.csv",
        ),
        row(
            "Null-component FDR bottleneck",
            "diagnostic_explains_not_closes_fdr",
            null_component_summary(null_component),
            "Component support cannot be written as broad FDR control. The diagnostic shows why broad FDR fails, not that it has passed.",
            "Develop a spatial-distance-specific calibration or dataset-adaptive null weighting target; report non-spatial component support separately from broad FDR in the manuscript.",
            "null_component_diagnostic/phase12_null_component_story.csv",
        ),
        row(
            "Variant matched-spatial calibration diagnostic",
            "supportive_direction_not_fdr_closure",
            variant_matched_spatial_summary(variant_matched_spatial),
            "This is post-hoc matched-spatial/range-weighted evidence, so it cannot be used as broad whole-atlas discovery FDR or as a final model replacement claim.",
            "Use anchor/background guarded training as the leading next-model component, then train or calibrate directly against matched/range-aware spatial nulls and rerun exact-score plus end-to-end audits.",
            "variant_matched_spatial_fdr/phase12_variant_matched_spatial_story.csv",
        ),
        row(
            "LR-standardized all-null FDR",
            "negative_diagnostic_model_score_gap_remains",
            lr_standardized_null_summary(lr_standardized_null),
            "This diagnostic does not support a positive whole-atlas discovery claim. It shows that simply standardizing each LR by its own null baseline is not enough.",
            "Change the model objective so observed true LR scores separate from receiver/spatial/LR-identity nulls with an explicit margin; do not rely on post-hoc FDR re-scaling alone.",
            "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_story.csv",
        ),
        row(
            "Null-margin guarded training",
            "ranking_improved_fdr_not_closed",
            null_margin_guarded_summary(null_margin_guarded),
            "This stronger null-margin training improves top-k ranking and hard-negative suppression, but exact-score broad whole-atlas FDR remains unclosed.",
            "Do not simply increase all null-loss weights further. Next model should redesign the score distribution or use end-to-end null-aware training/calibration that directly optimizes conservative global q-values while monitoring CRC/post-Xenium control risk.",
            "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_story.csv",
        ),
        row(
            "FDR margin gap",
            "spatial_null_gap_quantified",
            fdr_margin_gap_summary(fdr_margin_gap),
            "This is not a discovery result and does not change Goal 1 completion. It quantifies how far the best current variants remain from broad q<=0.1.",
            "Make the next model specifically reduce the spatial-distance null gap. Generic extra null-margin pressure already improved ranking but did not reduce the best q-gap.",
            "fdr_margin_gap/phase12_fdr_margin_gap_story.csv",
        ),
        row(
            "Spatial-specificity calibration",
            "q_gap_shrunk_not_fdr_closure",
            spatial_specificity_summary(spatial_specificity),
            "This post-hoc calibration still has zero broad true q<=0.1 discoveries and raises top10 risk in post-Xenium/CRC, so it cannot replace the reporting baseline.",
            "Use spatial-window concentration as a next-model feature/objective, but couple it to stronger background suppression and risk regression before rerunning exact-score and end-to-end null audits.",
            "spatial_specificity_comparison/phase12_spatial_specificity_story.csv",
        ),
        row(
            "Guarded spatial-specificity calibration",
            "near_fdr_not_closed",
            guarded_spatial_specificity_summary(guarded_spatial_specificity),
            "This post-hoc guarded score is close in post-Xenium but still has zero true q<=0.1 discoveries, so it cannot be written as broad FDR control or adopted as final baseline.",
            "Promote the useful ingredients into the model objective: spatial-window concentration, shared-anchor/background guards, and zero-expression suppression, then rerun exact-score and end-to-end null audits.",
            "guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_story.csv",
        ),
        row(
            "Integrated guarded spatial training candidate",
            "first_broad_true_hit_not_reporting_baseline",
            integrated_guarded_spatial_summary(
                integrated_guarded_spatial,
                integrated_guarded_spatial_story,
                integrated_highres_stability,
                integrated_coordinate_smoke,
            ),
            "This is the first candidate with a conservative broad q<=0.1 true LR hit, but it is not formal end-to-end retraining FDR, not the current reporting baseline, and CRC still has top10 risk. The coordinate-rerun smoke is directionally supportive but low-resolution and broad/background leakage remains.",
            "Scale end-to-end coordinate/null reruns, repair CRC/background leakage, and compare against the current baseline before considering baseline replacement.",
            "integrated_guarded_spatial_training_permutation_audit/phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv; integrated_guarded_spatial_training_highres_stability/phase12_integrated_guarded_spatial_highres_stability_story.csv; integrated_guarded_spatial_coordinate_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_integrated_coordinate_summary.csv",
        ),
        row(
            "Summary-only coordinate rerun infrastructure",
            "runner_smoke_ready_not_scientific_evidence",
            integrated_coordinate_summary_only_summary(integrated_coordinate_summary_only),
            "This is a runner/infrastructure check. The low-permutation smoke has no useful p-value resolution and should not be cited as FDR evidence.",
            "Use this no-cache path to scale end-to-end coordinate/null reruns without retaining bulky per-permutation prediction directories.",
            "integrated_guarded_spatial_coordinate_rerun_summary_only_smoke/*/*_integrated_coordinate_summary.csv",
        ),
        row(
            "Formal-parameter coordinate rerun smoke",
            "formal_parameter_smoke_rank_stable_not_fdr",
            integrated_coordinate_formal_parameter_summary(integrated_coordinate_formal_smoke),
            "This uses manuscript-like training parameters but remains a low-permutation coordinate smoke, so it supports rank/score stability and high-expression negative suppression, not formal FDR.",
            "Scale this same formal-parameter no-cache path to enough coordinate reruns for useful empirical p-value resolution before changing manuscript claims.",
            "integrated_guarded_spatial_coordinate_rerun_smoke/*formal*/*_integrated_coordinate_summary.csv",
        ),
        row(
            "Coordinate runner concordance",
            "nocache_runner_validated_against_cached_smoke",
            coordinate_runner_concordance_summary(coordinate_runner_concordance),
            "This validates the execution path only; it does not add independent FDR evidence or change Goal 1 completion.",
            "Use the no-cache/resume runner for future scaled coordinate/null reruns, while preserving the same claim boundary as the cached smoke.",
            "integrated_guarded_spatial_coordinate_runner_concordance/phase12_coordinate_runner_concordance_summary.csv",
        ),
        row(
            "Integrated coordinate broad/background leakage",
            "high_expression_clean_broad_background_not_solved",
            integrated_coordinate_leakage_summary(integrated_coordinate_leakage),
            "The low-permutation coordinate smoke suppresses high-expression negatives at the minimum p-value, but most minimum-p LR are broad/background candidates, so BH FDR remains far above the discovery threshold.",
            "Add broad/background-aware nulls or eligibility filters to the integrated candidate before scaling end-to-end FDR; do not treat the coordinate smoke as whole-atlas closure.",
            "integrated_guarded_spatial_coordinate_leakage/phase12_integrated_coordinate_leakage_summary.csv",
        ),
        row(
            "Integrated coordinate eligible-universe audit",
            "eligible_filter_not_sufficient_for_coordinate_fdr",
            integrated_coordinate_eligible_summary(integrated_coordinate_eligible),
            "A simple functionally annotated/expression-supported eligible universe still leaves background and ligand-shared controls tied with true LR at the minimum coordinate p-value, so eligible-coordinate BH FDR remains above 0.1.",
            "Do not solve broad/background leakage with a weak eligibility filter alone; the next model needs stronger background-aware training/calibration or a pre-specified anchor/story universe.",
            "integrated_guarded_spatial_coordinate_eligible_audit/phase12_integrated_coordinate_eligible_dataset.csv",
        ),
        row(
            "Integrated coordinate magnitude-gate diagnostic",
            "clean_selected_gate_not_reporting_rule",
            integrated_coordinate_magnitude_gate_summary(integrated_coordinate_magnitude_gate),
            "This gate is chosen diagnostically after inspecting the score distribution, so it cannot be used as the final unbiased whole-atlas FDR procedure.",
            "Convert the useful signal into a prespecified calibration/model component: require strong absolute communication magnitude while retaining broad/background and high-expression control audits, then rerun exact-score and end-to-end coordinate FDR.",
            "integrated_guarded_spatial_coordinate_magnitude_gate/phase12_integrated_coordinate_magnitude_gate_best.csv",
        ),
        row(
            "Integrated coordinate background target list",
            "background_targets_defined_not_solved",
            integrated_coordinate_background_target_summary(integrated_coordinate_background_targets),
            "This target list explains the leakage mode and score separation, but it is still a diagnostic derived from the current run.",
            "Use these LR as broad/background hard negatives or calibration checks in the next integrated model, then rerun coordinate permutations with a prespecified decision rule.",
            "integrated_guarded_spatial_coordinate_background_targets/phase12_integrated_coordinate_background_target_summary.csv",
        ),
        row(
            "Integrated background-guard training target panel",
            "next_model_target_panel_ready_not_evidence",
            integrated_background_guard_training_target_summary(integrated_background_guard_targets),
            "This is a concrete next-model target specification derived from the current diagnostic run. It does not add discovery evidence and does not close whole-atlas FDR.",
            "Promote these broad/background LR into training negatives or prespecified calibration checks, then rerun exact-score and end-to-end coordinate/null FDR before baseline replacement.",
            "integrated_background_guard_training_targets/phase12_integrated_background_guard_training_spec.csv",
        ),
        row(
            "Integrated background-target guarded training smoke",
            "target_aware_training_path_smoke_not_evidence",
            integrated_background_target_guarded_smoke_summary(integrated_background_target_guarded_smoke),
            "This is a low-budget one-dataset training smoke. It shows the target-aware path runs and can be evaluated, but it is not a reporting baseline and does not establish FDR improvement.",
            "Scale this target-aware training path to the full expanded atlas and rerun exact-score plus coordinate-null FDR before considering baseline replacement.",
            "integrated_background_target_guarded_training_smoke/phase12_integrated_background_target_guarded_dataset_summary.csv",
        ),
        row(
            "Integrated background-target guarded exact-score smoke",
            "target_aware_exact_score_smoke_promising_not_fdr",
            integrated_background_target_guarded_permutation_smoke_summary(
                integrated_background_target_guarded_permutation_smoke
            ),
            "This is a low-permutation exact-score smoke on one dataset. It is promising directionally but cannot establish formal FDR or replace end-to-end coordinate reruns.",
            "If this path is kept, rerun it on the full expanded atlas with formal exact-score permutations and then repeat coordinate-null model-rerun FDR.",
            "integrated_background_target_guarded_permutation_smoke/phase12_integrated_background_target_guarded_permutation_dataset_summary.csv",
        ),
        row(
            "Integrated background-target guarded coordinate-rerun smoke",
            "target_aware_coordinate_runner_smoke_not_fdr",
            integrated_background_target_guarded_coordinate_smoke_summary(
                integrated_background_target_guarded_coordinate_smoke
            ),
            "This is a low-permutation execution smoke with very coarse p-values. It supports target-aware end-to-end coordinate-null feasibility and score stability, not formal FDR.",
            "Scale the same no-cache coordinate-rerun path to enough permutations before using it for Goal 1 closure or baseline replacement.",
            "integrated_background_target_guarded_coordinate_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_target_guarded_coordinate_summary.csv",
        ),
        row(
            "Coordinate-null scaling disk guardrail",
            "cleanup_candidates_identified_not_executed",
            disk_cleanup_summary(disk_cleanup),
            "No cleanup action has been executed by this audit. Do not assume enough disk is available for 20/40/80 coordinate-null reruns until space is actually freed.",
            "If scaling coordinate-null reruns, first hardlink-deduplicate immutable duplicate replicate inputs or archive regenerable rendered figures with explicit approval.",
            "disk_cleanup_audit/phase12_disk_cleanup_candidates.csv",
        ),
        row(
            "Replicate hardlink dedup dry-run",
            "dedup_candidates_verified_not_executed",
            hardlink_dedup_dry_run_summary(hardlink_dedup_dry_run),
            "Dry-run only: duplicate files were compared but not hardlinked. Current free disk remains too low for large coordinate-null scaling.",
            "If replicate inputs are immutable, run the dedup tool with --execute to reclaim the verified duplicate storage before scaling reruns.",
            "disk_cleanup_audit/phase12_replicate_hardlink_dedup_dry_run.csv",
        ),
        row(
            "Replicate hardlink dedup execution",
            "dedup_executed_space_reclaimed",
            hardlink_dedup_executed_summary(hardlink_dedup_executed),
            "This is an infrastructure cleanup, not scientific evidence. Hardlinked replicate inputs must be treated as immutable because both paths share storage.",
            "Continue coordinate-null scaling only while preserving no-cache/drop-null-predictions behavior and monitoring free disk.",
            "disk_cleanup_audit/phase12_replicate_hardlink_dedup_executed.csv",
        ),
        row(
            "Regenerable TIFF artifact cleanup",
            "tiff_artifacts_removed_counterparts_preserved",
            tiff_cleanup_executed_summary(tiff_cleanup_executed),
            "TIFF submission artifacts were removed to recover disk, but same-stem SVG/PDF/PNG outputs were preserved. Regenerate TIFF before final submission if needed.",
            "Keep figure source tables and vector/PDF/PNG outputs under versioned outputs; regenerate TIFFs only at final packaging time.",
            "disk_cleanup_audit/phase12_regenerable_tiff_cleanup_executed.csv",
        ),
        row(
            "Background-guarded training variant",
            "training_direction_not_reporting_baseline",
            background_guarded_training_summary(background_guarded_training, background_guarded_permutation),
            "This version improves lymph-node top-k recovery but does not close exact-score broad whole-atlas FDR and remains weak on CRC.",
            "Use it as evidence that broad-background guards should live inside training; next iteration needs stronger CRC/background suppression and spatial-null separation.",
            "atlas_model_expanded_background_guarded_training/phase12_background_guarded_training_comparison.csv; background_guarded_training_permutation_audit/phase12_background_guarded_training_permutation_dataset_summary.csv",
        ),
        row(
            "Range-violation spatial-null training",
            "diagnostic_negative_result",
            range_violation_training_summary(range_violation_training, range_violation_permutation),
            "This version reduces selected CRC control risk but sacrifices lymph-node anchors and still does not close exact-score broad whole-atlas FDR.",
            "Do not simply increase spatial-null pressure; next model needs anchor-preserving, dataset-adaptive spatial-null loss plus background suppression.",
            "atlas_model_expanded_range_violation_guarded_training/phase12_range_violation_guarded_training_comparison.csv; range_violation_guarded_training_permutation_audit/phase12_range_violation_guarded_training_permutation_dataset_summary.csv",
        ),
        row(
            "Range-null calibrated variant",
            "hold_out_exploratory",
            variant_summary(variants, "range_null_calibrated"),
            "It improves some top10 recovery but increases broader control risk, so it should not be adopted globally.",
            "Redesign as range-matched contrastive training rather than a post-hoc replacement score.",
            "goal1_model_variant_scoreboard/phase12_goal1_model_variant_summary.csv",
        ),
        row(
            "Downstream validation",
            "selected_support",
            "Expanded whole-transcriptome target scans support selected LR/modules, especially post-Xenium SPP1-CD44 and selected lymph-node anchors.",
            "This is observational receiver-side association, not perturbation or proof that every top LR has functional targets.",
            "Expand curated target resources and keep negative-target/background audits paired with every story LR.",
            "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv",
        ),
        row(
            "Hard-negative controls",
            "strong_audit_layer",
            "High-expression, receptor-shared, receiver-shuffle, and edge-level hard negatives are now explicitly audited.",
            "These audits expose and reduce false-positive modes but are still scoped diagnostics, not native FDR.",
            "Use the hardest decoys as training negatives and make same-receptor/wrong-niche controls mandatory for story selection.",
            "hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv",
        ),
        row(
            "Directionality",
            "selected_source_sink_support",
            "Directionality supports selected lymph-node source-sink/receiver-pairing consistency but not global SpatialSelf-over-COMMOT superiority.",
            "Do not call this causal ligand transport or universal directionality advantage.",
            "If needed, scale end-to-end coordinate reruns and add stronger vector-null support before main-text superiority claims.",
            "directionality_consensus/phase12_directionality_consensus_dataset_summary.csv",
        ),
        row(
            "Phase 12 completion",
            "active_partial",
            completion_summary(completion),
            "Phase 12 should not be marked complete while Goal 1 broad whole-atlas FDR remains partial.",
            "Treat Phase 12 as a mature evidence audit plus next-model specification, not final model closure.",
            "completion_audit/phase12_goal_completion_summary.csv",
        ),
    ]
    return rows


def row(
    topic: str,
    status: str,
    supported: str,
    limitation: str,
    next_requirement: str,
    evidence_file: str,
) -> dict[str, object]:
    return {
        "topic": topic,
        "status": status,
        "supported_evidence": supported,
        "limitation_or_forbidden_claim": limitation,
        "next_model_or_next_analysis_requirement": next_requirement,
        "evidence_file": f"outputs/phase12/{evidence_file}",
    }


def fdr_interpretation_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Goal 1 FDR interpretation framework story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; "
        f"strict datasets={int(first['n_strict_full_transcriptome_datasets'])}; "
        f"supplementary panel datasets={int(first['n_supplementary_panel_datasets'])}; "
        f"broad-supported datasets={int(first['n_datasets_with_broad_fdr_support'])}; "
        f"spatial-null blockers={int(first['n_datasets_with_spatial_null_blocker'])}; "
        f"selected guarded support datasets={int(first['n_datasets_with_selected_guarded_support'])}; "
        f"eligible clean support datasets={int(first['n_datasets_with_eligible_clean_support'])}."
    )


def goal1_closure_contract_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Goal 1 closure contract status file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_status']}; items={int(first['n_contract_items'])}; "
        f"pass={int(first['n_pass'])}; candidate/partial={int(first['n_candidate_or_partial'])}; "
        f"fail/not-ready={int(first['n_fail_or_not_ready'])}; "
        f"main blocker={first['main_blocker']}."
    )


def bottleneck_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Spatial-null bottleneck story file not found."
    first = table.iloc[0]
    return (
        f"{int(first.get('n_true_spatial_specific_blocker', 0))} true LR blocked by pooled spatial-distance null; "
        f"{int(first.get('n_true_matched_spatial_supported', 0))} true LR supported by matched spatial diagnostics."
    )


def variant_summary(table: pd.DataFrame, variant: str) -> str:
    if table.empty:
        return f"{variant} summary file not found."
    selected = table[table["variant"].astype(str) == variant]
    if selected.empty:
        return f"{variant} not found in variant summary."
    first = selected.iloc[0]
    return (
        f"mean top10 true LR={float(first['mean_top10_true_lr']):.2f}; "
        f"true LR q<=0.1={int(first['total_true_lr_q_0_1'])}; "
        f"decision={first['variant_decision']}."
    )


def background_summary(story: pd.DataFrame, variants: pd.DataFrame) -> str:
    variant_text = variant_summary(variants, "background_suppressed")
    if story.empty:
        return variant_text
    first = story.iloc[0]
    return (
        f"{variant_text} Exact-score audit: FDR-closed datasets={int(first['n_fdr_closed_datasets'])}, "
        f"top10 improved without added risk={int(first['n_top10_improved_without_risk'])}."
    )


def guarded_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Guarded matched full-LR FDR story file not found."
    first = table.iloc[0]
    return (
        f"{int(first['total_true_supported'])} true LR supported across "
        f"{int(first['n_supported_datasets'])}/{int(first['n_datasets'])} datasets; "
        f"primary-control supported={int(first['total_primary_control_supported'])}; "
        f"receptor-shared supported={int(first['total_receptor_shared_supported'])}; "
        f"call={first['overall_call']}."
    )


def spatial_prototype_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Spatial-null-aware exact-score prototype story file not found."
    first = table.iloc[0]
    return (
        f"datasets with true q<=0.1={int(first['n_datasets_with_true_q_0_1'])}/{int(first['n_datasets'])}; "
        f"total true q<=0.1={int(first['total_true_lr_q_0_1'])}; "
        f"total risk q<=0.1={int(first['total_negative_or_risk_q_0_1'])}; "
        f"call={first['overall_call']}."
    )


def training_smoke_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Spatial-null training smoke summary file not found."
    first = table.iloc[0]
    return (
        f"status={first['status']}; spatial-null pairs={int(first['n_spatial_null_contrastive_train'])}; "
        f"context pairs={int(first['n_context_contrastive_train'])}; "
        f"hard-negative pairs={int(first['n_hard_negative_train'])}; "
        f"runtime={float(first['runtime_seconds']):.1f}s."
    )


def trained_variant_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Spatial-null-trained expanded atlas comparison file not found."
    improved = int(table["comparison_call"].astype(str).eq("ranking_improved_without_top10_risk").sum())
    worse = int(table["comparison_call"].astype(str).eq("tradeoff_or_worse").sum())
    delta_true = int(pd.to_numeric(table["delta_top10_true_lr"], errors="coerce").fillna(0).sum())
    delta_risk = float(pd.to_numeric(table["delta_top10_risk_fraction"], errors="coerce").fillna(0).sum())
    return (
        f"datasets improved={improved}, tradeoff/worse={worse}; "
        f"sum delta top10 true={delta_true}; sum delta top10 risk={delta_risk:.3g}."
    )


def anchor_variant_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Anchor-preserving spatial-null comparison file not found."
    calls = table["comparison_call"].astype(str)
    improved = int(calls.eq("anchor_variant_improves_main").sum())
    preserved = int(calls.eq("anchor_variant_preserves_main").sum())
    reduced = int(calls.eq("anchor_variant_reduces_true_without_risk").sum())
    delta_true = int(pd.to_numeric(table["delta_top10_true_lr_vs_main"], errors="coerce").fillna(0).sum())
    delta_risk = float(pd.to_numeric(table["delta_top10_risk_vs_main"], errors="coerce").fillna(0).sum())
    return (
        f"improved={improved}, preserved={preserved}, reduced-true-without-risk={reduced}; "
        f"sum delta top10 true={delta_true}; sum delta top10 risk={delta_risk:.3g}."
    )


def anchor_permutation_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Anchor-preserving exact-score permutation story file not found."
    first = table.iloc[0]
    return (
        f"Exact-score audit: FDR-closed datasets={int(first['n_fdr_closed_datasets'])}/{int(first['n_datasets'])}; "
        f"top10 improved without risk={int(first['n_top10_improved_without_risk'])}; "
        f"top10 preserved without risk={int(first['n_top10_preserved_without_risk'])}; "
        f"best true q improved={int(first['n_best_true_q_improved'])}; "
        f"call={first['overall_call']}."
    )


def anchor_background_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Anchor plus background permutation story file not found."
    first = table.iloc[0]
    return (
        f"Exact-score audit: FDR-closed datasets={int(first['n_fdr_closed_datasets'])}/{int(first['n_datasets'])}; "
        f"top10 improved without risk vs base={int(first['n_top10_improved_without_risk_vs_base'])}; "
        f"best true q improved vs base={int(first['n_best_true_q_improved_vs_base'])}; "
        f"CRC top10 recovered={int(first['n_crc_top10_recovered_vs_base'])}; "
        f"call={first['overall_call']}."
    )


def anchor_background_guarded_training_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Anchor/background guarded training story file not found."
    first = table.iloc[0]
    return (
        f"Exact-score audit: FDR-closed datasets={int(first['n_fdr_closed_datasets'])}/{int(first['n_datasets'])}; "
        f"top10 improved without risk vs base={int(first['n_top10_improved_without_risk_vs_base'])}; "
        f"worse top10 risk than post-hoc anchor/background={int(first['n_worse_top10_risk_than_posthoc_anchor_background'])}; "
        f"hard-negative best-beat clean datasets={int(first['n_hard_negative_best_beat_clean'])}; "
        f"call={first['overall_call']}."
    )


def anchor_background_range_guarded_training_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Anchor/background range-guarded training story file not found."
    first = table.iloc[0]
    return (
        f"Exact-score audit: FDR-closed datasets={int(first['n_fdr_closed_datasets'])}/{int(first['n_datasets'])}; "
        f"top10 improved without risk vs base={int(first['n_top10_improved_without_risk_vs_base'])}; "
        f"worse top10 risk than post-hoc anchor/background={int(first['n_worse_top10_risk_than_posthoc_anchor_background'])}; "
        f"hard-negative best-beat clean datasets={int(first['n_hard_negative_best_beat_clean'])}; "
        f"call={first['overall_call']}."
    )


def null_component_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Null-component diagnostic story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; methods={int(first['n_methods'])}; "
        f"best non-spatial method={first['best_non_spatial_method']} with true supported={int(first['best_non_spatial_true_supported'])}; "
        f"best spatial-q method={first['best_spatial_q_method']} best spatial q={float(first['best_true_spatial_q']):.3g}; "
        f"broad true q<=0.1 total={int(first['broad_fdr_supported_true_total'])}."
    )


def variant_matched_spatial_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Variant matched-spatial diagnostic story file not found."
    first = table.iloc[0]
    return (
        f"Best method={first['best_method']}; strict matched true q<=0.1="
        f"{int(first['best_method_strict_matched_true_q_0_1'])}, matched true="
        f"{int(first['best_method_matched_true_q_0_1'])}, weighted true="
        f"{int(first['best_method_weighted_true_q_0_1'])}, weighted controls="
        f"{int(first['best_method_weighted_control_q_0_1'])}."
    )


def lr_standardized_null_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "LR-standardized null FDR story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; best method={first['best_method']}; "
        f"strict true q<=0.1={int(first['best_method_strict_true_q_0_1'])}, "
        f"strict controls={int(first['best_method_strict_control_q_0_1'])}, "
        f"true q<=0.1={int(first['best_method_true_q_0_1'])}, "
        f"controls={int(first['best_method_control_q_0_1'])}."
    )


def null_margin_guarded_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Null-margin guarded training story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; top10 improved vs main="
        f"{int(first['n_top10_true_improved_vs_main'])}/{int(first['n_datasets'])}, "
        f"hard-negative clean={int(first['n_hard_negative_best_beat_clean'])}/{int(first['n_datasets'])}, "
        f"FDR-closed datasets={int(first['n_fdr_closed_datasets'])}, "
        f"total true q<=0.1={int(first['total_true_q_0_1'])}."
    )


def fdr_margin_gap_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "FDR margin-gap story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; best variant={first['best_variant_by_min_q_gap']}; "
        f"min q-gap={float(first['best_variant_min_q_gap_to_0_1']):.3g}x, "
        f"median q-gap={float(first['best_variant_median_q_gap_to_0_1']):.3g}x; "
        f"dominant nulls={first['best_variant_dominant_null_counts']}."
    )


def spatial_specificity_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Spatial-specificity comparison story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; true q<=0.1={int(first['n_true_q_0_1'])}, "
        f"best true q improved vs main={int(first['n_best_true_q_improved_vs_main'])}/{int(first['n_datasets'])}, "
        f"top10 true improved vs main={int(first['n_top10_true_improved_vs_main'])}/{int(first['n_datasets'])}, "
        f"best true q={float(first['best_true_q_min']):.3g} "
        f"({float(first['best_true_q_gap_to_0_1']):.3g}x over q<=0.1)."
    )


def guarded_spatial_specificity_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Guarded spatial-specificity comparison story file not found."
    first = table.iloc[0]
    return (
        f"overall={first['overall_call']}; true q<=0.1={int(first['n_true_q_0_1'])}, "
        f"best true q improved vs main={int(first['n_best_true_q_improved_vs_main'])}/{int(first['n_datasets'])}, "
        f"risk improved vs unguarded spatial-specificity="
        f"{int(first['n_top10_risk_improved_vs_spatial_specificity'])}/{int(first['n_datasets'])}, "
        f"best true q={float(first['best_true_q_min']):.3g} "
        f"({float(first['best_true_q_gap_to_0_1']):.3g}x over q<=0.1)."
    )


def integrated_guarded_spatial_summary(
    dataset: pd.DataFrame,
    story: pd.DataFrame,
    highres: pd.DataFrame,
    coordinate_smoke: pd.DataFrame,
) -> str:
    if dataset.empty:
        return "Integrated guarded spatial training permutation file not found."
    true_hits = int(pd.to_numeric(dataset["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
    all_hits = int(pd.to_numeric(dataset["all_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
    best = dataset.sort_values("best_true_lr_fdr").head(1)
    best_text = "NA"
    if len(best):
        first = best.iloc[0]
        best_text = f"{first.dataset_key}:{first.best_true_lr} q={float(first.best_true_lr_fdr):.3g}"
    clean_top10 = int(pd.to_numeric(dataset["top10_negative_or_risk_fraction"], errors="coerce").fillna(1).le(0).sum())
    overall = story.iloc[0]["overall_call"] if not story.empty else "NA"
    highres_text = ""
    if not highres.empty:
        first = highres.iloc[0]
        highres_text = (
            f" Highres: overall={first['overall_call']}, stable q<=0.1 hits="
            f"{int(first['n_stable_q_0_1_hits'])}, best={first['best_highres_true']} "
            f"q={float(first['best_highres_true_q']):.3g}."
        )
    coordinate_text = ""
    if not coordinate_smoke.empty:
        first = coordinate_smoke.iloc[0]
        coordinate_text = (
            f" Coordinate smoke: {first['target_lr']} rank={int(first['target_rank'])}, "
            f"score={float(first['target_score']):.3g} vs null q95={float(first['target_null_q95']):.3g}, "
            f"p={float(first['target_p']):.3g}, BH={float(first['target_bh_fdr']):.3g}, "
            f"best negative={first['best_negative_lr']} p={float(first['best_negative_p']):.3g}."
        )
    return (
        f"overall={overall}; true q<=0.1={true_hits}, all LR q<=0.1={all_hits}, "
        f"best true={best_text}, clean top10 datasets={clean_top10}/{len(dataset)}."
        f"{highres_text}{coordinate_text}"
    )


def integrated_coordinate_leakage_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated coordinate leakage audit file not found."
    first = table.iloc[0]
    return (
        f"call={first['leakage_call']}; min-p LR={int(first['n_min_p_lr'])}, "
        f"true={int(first['n_min_p_true_lr'])}, "
        f"high-expression negative={int(first['n_min_p_high_expression_negative'])}, "
        f"broad/background={int(first['n_min_p_broad_background'])}; "
        f"best true={first['best_true_lr']} p={float(first['best_true_p']):.3g}, "
        f"BH={float(first['best_true_bh_fdr']):.3g}; "
        f"best high-expression negative={first['best_high_expression_negative_lr']} "
        f"p={float(first['best_high_expression_negative_p']):.3g}; "
        f"best broad/background={first['best_broad_background_lr']} "
        f"p={float(first['best_broad_background_p']):.3g}."
    )


def integrated_coordinate_summary_only_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Summary-only coordinate runner smoke file not found."
    first = table.iloc[0]
    return (
        f"dataset={first['dataset_key']}; status={first['status']}; "
        f"completed reruns={int(first['n_completed_permutations'])}; "
        f"target={first['target_lr']} rank={int(first['target_rank'])}; "
        f"score={float(first['target_score']):.3g}; "
        f"null q95={float(first['target_null_q95']):.3g}; "
        f"p={float(first['target_p']):.3g}; BH={float(first['target_bh_fdr']):.3g}."
    )


def integrated_coordinate_formal_parameter_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Formal-parameter coordinate runner smoke file not found."
    first = table.iloc[0]
    epochs = first.get("epochs", "NA")
    max_train_samples = first.get("max_train_samples", "NA")
    return (
        f"dataset={first['dataset_key']}; status={first['status']}; "
        f"completed reruns={int(first['n_completed_permutations'])}; "
        f"epochs={epochs}; max_train_samples={max_train_samples}; "
        f"target={first['target_lr']} rank={int(first['target_rank'])}; "
        f"score={float(first['target_score']):.3g}; "
        f"null q95={float(first['target_null_q95']):.3g}; "
        f"p={float(first['target_p']):.3g}; BH={float(first['target_bh_fdr']):.3g}; "
        f"best high-expression negative={first['best_negative_lr']} "
        f"p={float(first['best_negative_p']):.3g}; "
        f"min-p true={int(first['n_positive_p_min'])}; "
        f"min-p high-expression negative={int(first['n_negative_p_min'])}."
    )


def coordinate_runner_concordance_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Coordinate runner concordance file not found."
    first = table.iloc[0]
    return (
        f"call={first['runner_concordance_call']}; common LR={int(first['n_common_lr'])}; "
        f"target={first['target_lr']}; p ref/no-cache="
        f"{float(first['reference_target_p']):.3g}/{float(first['nocache_target_p']):.3g}; "
        f"BH ref/no-cache={float(first['reference_target_bh_fdr']):.3g}/"
        f"{float(first['nocache_target_bh_fdr']):.3g}; "
        f"null q95 delta={float(first['target_null_q95_abs_delta']):.3g}; "
        f"best negative same={bool(first['best_negative_same'])}; "
        f"min-p true ref/no-cache={int(first['reference_min_p_true'])}/"
        f"{int(first['nocache_min_p_true'])}; "
        f"min-p high-expression negative ref/no-cache="
        f"{int(first['reference_min_p_high_expression_negative'])}/"
        f"{int(first['nocache_min_p_high_expression_negative'])}."
    )


def integrated_coordinate_eligible_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated coordinate eligible audit file not found."
    first = table.iloc[0]
    return (
        f"call={first['eligible_coordinate_call']}; eligible LR={int(first['n_eligible_lr'])}/"
        f"{int(first['n_total_lr'])}; q<=0.1 total={int(first['eligible_coordinate_q_0_1'])}, "
        f"true={int(first['eligible_coordinate_true_q_0_1'])}, "
        f"controls={int(first['eligible_coordinate_control_q_0_1'])}, "
        f"background={int(first['eligible_coordinate_background_q_0_1'])}; "
        f"best true={first['best_true_lr']} p={float(first['best_true_coordinate_p']):.3g}, "
        f"eligible q={float(first['best_true_eligible_coordinate_q']):.3g}; "
        f"best control={first['best_control_lr']} "
        f"eligible q={float(first['best_control_eligible_coordinate_q']):.3g}."
    )


def integrated_coordinate_magnitude_gate_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated coordinate magnitude-gate audit file not found."
    first = table.iloc[0]
    return (
        f"call={first['gate_call']}; threshold={float(first['score_threshold']):.3g}; "
        f"retained LR={int(first['n_lr'])}, true={int(first['n_true_lr'])}, "
        f"high-expression negative={int(first['n_high_expression_negative'])}, "
        f"broad/background={int(first['n_broad_background'])}; "
        f"q<=0.1 total={int(first['n_q_0_1'])}, true={int(first['n_true_q_0_1'])}, "
        f"high-expression negative={int(first['n_high_expression_negative_q_0_1'])}, "
        f"broad/background={int(first['n_broad_background_q_0_1'])}; "
        f"best true q={float(first['best_true_q']):.3g}; retained={first['top_lr_pairs']}."
    )


def integrated_coordinate_background_target_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated coordinate background target audit file not found."
    first = table.iloc[0]
    return (
        f"call={first['target_call']}; min-p LR={int(first['n_min_p_lr'])}, "
        f"true={int(first['n_min_p_true_lr'])}, "
        f"high-expression negative={int(first['n_min_p_high_expression_negative'])}, "
        f"broad/background={int(first['n_min_p_broad_background'])}; "
        f"broad/background top10={int(first['n_broad_background_top10'])}, "
        f"top50={int(first['n_broad_background_top50'])}; "
        f"retained true floor={float(first['retained_true_score_floor']):.3g}, "
        f"max broad/background score={float(first['max_broad_background_score']):.3g}, "
        f"gap={float(first['retained_true_to_max_broad_score_gap']):.3g}x; "
        f"broad/background above gate={int(first['n_broad_background_above_gate'])}."
    )


def integrated_background_guard_training_target_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated background-guard training target specification file not found."
    first = table.iloc[0]
    return (
        f"dataset={first['dataset_key']}; coordinate reruns={int(first['n_completed_coordinate_reruns'])}; "
        f"background targets={int(first['n_background_targets'])}, "
        f"primary targets={int(first['n_primary_background_targets'])}; "
        f"min-p LR={int(first['n_min_p_lr'])}, true={int(first['n_min_p_true_lr'])}, "
        f"high-expression negative={int(first['n_min_p_high_expression_negative'])}, "
        f"broad/background={int(first['n_min_p_broad_background'])}; "
        f"gate={float(first['magnitude_gate_threshold']):.3g}; "
        f"background above gate={int(first['n_background_above_gate'])}; "
        f"success criterion={first['success_criterion_before_baseline_replacement']}"
    )


def integrated_background_target_guarded_smoke_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated background-target guarded training smoke file not found."
    first = table.iloc[0]
    return (
        f"dataset={first['dataset_key']}; call={first['target_guard_smoke_call']}; "
        f"epochs={int(first['epochs'])}; max_train_samples={int(first['max_train_samples'])}; "
        f"target guard train={int(first['n_background_guard_target_train'])}/"
        f"{int(first['n_background_guard_train'])}; "
        f"top10 true={int(first['top10_true_lr'])} vs reference={int(first['reference_top10_true_lr'])}; "
        f"top10 background targets={int(first['top10_background_targets'])} vs reference="
        f"{int(first['reference_top10_background_targets'])}; "
        f"targets above gate={int(first['n_targets_above_magnitude_gate'])}; "
        f"anchor min delta={float(first['anchor_min_delta_score']):.3g}."
    )


def integrated_background_target_guarded_permutation_smoke_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated background-target guarded exact-score smoke file not found."
    first = table.iloc[0]
    return (
        f"dataset={first['dataset_key']}; call={first['target_guarded_permutation_smoke_call']}; "
        f"permutations/null={int(first['n_permutations_per_null'])}; "
        f"top10 true={int(first['top10_true_lr'])}; "
        f"top10 risk={float(first['top10_negative_or_risk_fraction']):.3g}; "
        f"all LR q<=0.1={int(first['all_lr_conservative_fdr_0_1'])}; "
        f"true LR q<=0.1={int(first['all_true_lr_conservative_fdr_0_1'])}; "
        f"best true={first['best_true_lr']} q={float(first['best_true_lr_fdr']):.3g}."
    )


def integrated_background_target_guarded_coordinate_smoke_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Integrated background-target guarded coordinate-rerun smoke file not found."
    first = table.iloc[0]
    gap_detail = integrated_background_target_guarded_coordinate_score_gap(
        str(first["target_lr"]),
        str(first["best_negative_lr"]),
    )
    low_disk_detail = integrated_background_target_guarded_coordinate_low_disk_detail()
    return (
        f"dataset={first['dataset_key']}; status={first['status']}; "
        f"coordinate reruns={int(first['n_completed_permutations'])}; "
        f"target={first['target_lr']} rank={int(first['target_rank'])}; "
        f"score={float(first['target_score']):.3g} vs null q95={float(first['target_null_q95']):.3g}; "
        f"p={float(first['target_p']):.3g}; BH={float(first['target_bh_fdr']):.3g}; "
        f"best high-expression negative={first['best_negative_lr']} p={float(first['best_negative_p']):.3g}; "
        f"min-p true={int(first['n_positive_p_min'])}; min-p high-expression negative={int(first['n_negative_p_min'])}"
        f"{gap_detail}{low_disk_detail}."
    )


def integrated_background_target_guarded_coordinate_score_gap(target_lr: str, negative_lr: str) -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_lr_fdr.csv"
    )
    table = read_csv(path)
    if table.empty or "lr_pair" not in table:
        return ""
    target = table[table["lr_pair"].astype(str).eq(str(target_lr))]
    negative = table[table["lr_pair"].astype(str).eq(str(negative_lr))]
    if target.empty or negative.empty:
        return ""
    target_score = float(pd.to_numeric(target.iloc[0].get("calibrated_score"), errors="coerce"))
    negative_score = float(pd.to_numeric(negative.iloc[0].get("calibrated_score"), errors="coerce"))
    negative_rank = pd.to_numeric(negative.iloc[0].get("rank"), errors="coerce")
    if not np.isfinite(target_score) or not np.isfinite(negative_score):
        return ""
    ratio = target_score / max(negative_score, 1e-12)
    rank_text = int(negative_rank) if pd.notna(negative_rank) else "NA"
    return f"; negative rank={rank_text}; negative score={negative_score:.3g}; target/negative score ratio={ratio:.3g}x"


def integrated_background_target_guarded_coordinate_low_disk_detail() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_status.csv"
    )
    table = read_csv(path)
    if table.empty or "status" not in table:
        return ""
    skipped = table[table["status"].astype(str).eq("skipped_low_disk")]
    if skipped.empty:
        skipped_text = ""
    else:
        detail = str(skipped.iloc[-1].get("detail", ""))
        skipped_text = f"; low-disk skips={len(skipped)} ({detail})"
    failed = table[table["status"].astype(str).eq("failed")]
    failed_text = ""
    if not failed.empty:
        detail = str(failed.iloc[-1].get("detail", ""))
        failed_text = f"; failed attempts={len(failed)} ({detail})"
    return f"{skipped_text}{failed_text}"


def disk_cleanup_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Disk cleanup audit has not been generated."
    total = float(pd.to_numeric(table["estimated_savings_mib"], errors="coerce").fillna(0).sum())
    primary = table.sort_values("estimated_savings_mib", ascending=False).iloc[0]
    approval = int(table["requires_user_approval"].fillna(False).astype(bool).sum())
    return (
        f"cleanup candidates={len(table)}; potential savings={total:.1f} MiB; "
        f"largest={primary['candidate']} ({float(primary['estimated_savings_mib']):.1f} MiB); "
        f"user-approval-required candidates={approval}; audit is read-only."
    )


def hardlink_dedup_dry_run_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Replicate hardlink dedup dry-run has not been generated."
    candidates = table[table["planned_action"].astype(str).eq("candidate_hardlink")].copy()
    not_identical = table[table["planned_action"].astype(str).eq("not_identical")].copy()
    savings = float(pd.to_numeric(candidates["size_mib"], errors="coerce").fillna(0).sum()) if len(candidates) else 0.0
    largest = candidates.sort_values("size_mib", ascending=False).iloc[0] if len(candidates) else None
    largest_text = (
        f"; largest={largest['relative_path']} ({float(largest['size_mib']):.1f} MiB)"
        if largest is not None
        else ""
    )
    return (
        f"audited files={len(table)}; hardlink candidates={len(candidates)}; "
        f"not-identical files={len(not_identical)}; estimated savings={savings:.1f} MiB"
        f"{largest_text}; execution not performed."
    )


def hardlink_dedup_executed_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Replicate hardlink dedup execution log has not been generated."
    hardlinked = table[table["executed_action"].astype(str).eq("hardlinked_to_base")].copy()
    failures = table[table["executed_action"].astype(str).eq("failed")].copy()
    skipped = table[table["planned_action"].astype(str).eq("not_identical")].copy()
    savings = float(pd.to_numeric(hardlinked["size_mib"], errors="coerce").fillna(0).sum()) if len(hardlinked) else 0.0
    return (
        f"hardlinked files={len(hardlinked)}; estimated storage reclaimed={savings:.1f} MiB; "
        f"failures={len(failures)}; non-identical skipped={len(skipped)}."
    )


def tiff_cleanup_executed_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Regenerable TIFF cleanup execution log has not been generated."
    deleted = table[table["executed_action"].astype(str).eq("deleted_tiff_artifact")].copy()
    failures = table[table["executed_action"].astype(str).eq("failed")].copy()
    missing_counterparts = table[~table["eligible_for_cleanup"].fillna(False).astype(bool)].copy()
    savings = float(pd.to_numeric(deleted["size_mib"], errors="coerce").fillna(0).sum()) if len(deleted) else 0.0
    return (
        f"deleted TIFF artifacts={len(deleted)}; reclaimed={savings:.1f} MiB; "
        f"failures={len(failures)}; missing-counterpart skips={len(missing_counterparts)}; "
        "same-stem SVG/PDF/PNG counterparts were present for deleted files."
    )


def background_guarded_training_summary(comparison: pd.DataFrame, permutation: pd.DataFrame) -> str:
    if comparison.empty and permutation.empty:
        return "Background-guarded training files not found."
    parts = []
    if not comparison.empty:
        calls = comparison["comparison_call"].astype(str)
        improved = int(calls.eq("guarded_training_improves_main_top10").sum())
        conservative = int(calls.eq("guarded_training_conservative_tradeoff").sum())
        delta_true = int(pd.to_numeric(comparison["delta_top10_true_lr_vs_main"], errors="coerce").fillna(0).sum())
        delta_risk = float(pd.to_numeric(comparison["delta_top10_risk_vs_main"], errors="coerce").fillna(0).sum())
        parts.append(
            f"top-k comparison: improved datasets={improved}, conservative tradeoff datasets={conservative}, "
            f"sum delta top10 true={delta_true}, sum delta top10 risk={delta_risk:.3g}"
        )
    if not permutation.empty:
        true_q = int(pd.to_numeric(permutation["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
        all_q = int(pd.to_numeric(permutation["all_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
        best_q = float(pd.to_numeric(permutation["best_true_lr_fdr"], errors="coerce").min())
        parts.append(f"exact-score audit: true q<=0.1={true_q}, all-LR q<=0.1={all_q}, best true q={best_q:.3g}")
    return "; ".join(parts)


def range_violation_training_summary(comparison: pd.DataFrame, permutation: pd.DataFrame) -> str:
    if comparison.empty and permutation.empty:
        return "Range-violation training files not found."
    parts = []
    if not comparison.empty:
        calls = comparison["comparison_call"].astype(str)
        reduced_risk = int(calls.eq("range_violation_reduces_main_risk").sum())
        tradeoff = int(calls.eq("range_violation_conservative_tradeoff").sum())
        delta_true = int(pd.to_numeric(comparison["delta_top10_true_lr_vs_main"], errors="coerce").fillna(0).sum())
        delta_risk = float(pd.to_numeric(comparison["delta_top10_risk_vs_main"], errors="coerce").fillna(0).sum())
        parts.append(
            f"top-k comparison: risk-reduced datasets={reduced_risk}, conservative tradeoff datasets={tradeoff}, "
            f"sum delta top10 true={delta_true}, sum delta top10 risk={delta_risk:.3g}"
        )
    if not permutation.empty:
        true_q = int(pd.to_numeric(permutation["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
        all_q = int(pd.to_numeric(permutation["all_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum())
        best_q = float(pd.to_numeric(permutation["best_true_lr_fdr"], errors="coerce").min())
        parts.append(f"exact-score audit: true q<=0.1={true_q}, all-LR q<=0.1={all_q}, best true q={best_q:.3g}")
    return "; ".join(parts)


def supplementary_panel_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Supplementary expanded-panel atlas summary file not found."
    parts = []
    for row in table.itertuples(index=False):
        parts.append(
            f"{row.dataset_key}: LR={int(row.n_lr)}, top10 true={int(row.top10_true_lr)}, "
            f"top10 risk={float(row.top10_negative_or_risk_fraction):.3g}, "
            f"hard-negative best-beat={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}, "
            f"call={row.supplementary_call}"
        )
    return "; ".join(parts)


def completion_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Completion audit file not found."
    parts = []
    for entry in table.itertuples(index=False):
        parts.append(
            f"{entry.goal}={entry.overall_status} "
            f"({int(entry.n_complete)}/{int(entry.n_requirements)} complete)"
        )
    return "; ".join(parts)


def write_status(table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Final Limitations and Next-Model Requirements",
        "",
        "## Bottom Line",
        "",
        "- Phase 12 is strong as an evidence audit: whole-LR ranking, hard negatives, selected downstream support, second-story triage, and directionality guardrails are in place.",
        "- Phase 12 is not closed as a broad discovery-FDR result: whole-atlas q<=0.1 discoveries remain absent under conservative nulls.",
        "- The next model task is clear: train/calibrate against spatial-distance and broad-background nulls instead of relying on post-hoc score variants.",
        "",
        "## Decision Table",
        "",
    ]
    for entry in table.itertuples(index=False):
        lines.append(f"- **{entry.topic}** (`{entry.status}`): {entry.supported_evidence}")
        lines.append(f"  Limitation: {entry.limitation_or_forbidden_claim}")
        lines.append(f"  Next: {entry.next_model_or_next_analysis_requirement}")
    (OUT_DIR / "phase12_final_limitations_next_model_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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


def best_integrated_coordinate_formal_parameter_smoke() -> pd.DataFrame:
    paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke").glob(
            "*formal*/*_integrated_coordinate_summary.csv"
        )
    )
    frames = []
    for path in paths:
        frame = read_csv(path)
        if frame.empty:
            continue
        manifest_path = path.with_name(path.name.replace("_summary.csv", "_manifest.json"))
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        else:
            manifest = {}
        frame["epochs"] = manifest.get("epochs", np.nan)
        frame["max_train_samples"] = manifest.get("max_train_samples", np.nan)
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


def best_integrated_coordinate_summary_only_smoke() -> pd.DataFrame:
    paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_summary_only_smoke").glob(
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
