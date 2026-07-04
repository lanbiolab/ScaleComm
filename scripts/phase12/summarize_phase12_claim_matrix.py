#!/usr/bin/env python
"""Build a conservative claim matrix for Phase 12 manuscript planning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "claim_matrix"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(core_claims())
    rows.extend(dataset_claims())
    rows.extend(method_claims())
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "phase12_claim_matrix.csv", index=False)
    write_status(table)
    manifest = {
        "task": "phase12_claim_matrix",
        "outputs": [
            rel(OUT_DIR / "phase12_claim_matrix.csv"),
            rel(OUT_DIR / "phase12_claim_matrix_status.md"),
        ],
        "claim_boundary": "Conservative manuscript planning. This file separates supported claims from claims that still need stronger evidence.",
    }
    (OUT_DIR / "phase12_claim_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


def core_claims() -> list[dict[str, object]]:
    return [
        claim(
            "Benchmark realism/readiness",
            "supported_as_input_readiness",
            "We have audited LR coverage across seven datasets and can run full-transcriptome-background pilots.",
            "atlas_readiness/phase12_dataset_lr_coverage.csv",
            "Do not say this proves whole-atlas performance; it proves data/model plumbing and coverage.",
            "Goal 1",
        ),
        claim(
            "Naive expression-distance failure mode",
            "supported",
            "Expression-distance scoring is strongly contaminated by high-expression hard negatives.",
            "atlas_smoke/phase12_atlas_smoke_hard_negative_leakage.csv",
            "This supports the need for hard-negative controls, not SpatialSelf superiority by itself.",
            "Goal 3",
        ),
        claim(
            "Edge-level hard-negative leakage modes",
            "failure_modes_exposed_not_solved",
            "Phase 12 now quantifies edge-level hard negatives including wrong range, wrong type, zero-expression edges, spatially shifted sender proxies, same spatial edge with wrong LR identity, and high-expression false LR edges, and includes a diagnostic edge-guardrail calibration.",
            "edge_hard_negative_audit/phase12_edge_hard_negative_dataset_summary.csv",
            "The guardrail reduces aggregate leakage in the two edge-audited expanded datasets, especially lymph node, but it is post-hoc and mixed by class; use it as a calibration target, not as evidence that false positives are fully controlled.",
            "Goal 3",
        ),
        claim(
            "Receiver-context hard-negative stress",
            "receiver_shuffle_support_with_dataset_caveat",
            "Phase 12 now includes an explicit receiver-shuffle audit: lymph node shows strong receiver-context dependence for most anchor LR, while post-Xenium supports the top SPP1-CD44/TGFB1/CXCL12 anchors but still has leakage among lower-ranked anchors.",
            "receiver_shuffle_hard_negative/phase12_receiver_shuffle_dataset_summary.csv",
            "This is a fixed-score receiver-context stress test, not a model-rerun null or unbiased whole-atlas FDR. Use it to support story-level context dependence and to disclose residual leakage.",
            "Goal 3",
        ),
        claim(
            "Atlas-level receptor-shared controls",
            "top10_clean_pairwise_leakage_quantified",
            "Phase 12 now audits same-receptor decoys directly on the full 886/957-LR expanded rankings. The calibrated top10 has zero receptor-shared decoys across the expanded atlas audit, while lower-ranked anchors still expose limited pairwise same-receptor leakage.",
            "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv",
            "This is a receptor-shared control-contamination audit, not unbiased native whole-atlas FDR. Do not imply all same-receptor artifacts are solved; use the pairwise leakage rows as calibration targets.",
            "Goal 1/3",
        ),
        claim(
            "Whole-transcriptome downstream support",
            "expanded_top_module_support_with_subset_caveat",
            "Expanded-atlas receiver-side target scans now cover calibrated top LR, priority anchors and module representatives across post-Xenium colon, human lymph node and Visium HD CRC; target-enriched support is present but concentrated in a subset of LR/modules.",
            "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv",
            "This is observational association after residualization, not perturbation or causal proof. Do not claim every predicted LR has downstream target support.",
            "Goal 2",
        ),
        claim(
            "Target-background whole-transcriptome audit",
            "selected_lr_target_background_supported",
            "A target-background audit now tests whether curated target genes rank above all expressed genes for scanned top/priority/module LR. Post-Xenium shows strong selected-LR support, lymph node supports CXCL12-CXCR4, and high-expression target leakage is not observed in this audit layer.",
            "downstream_target_background_audit/phase12_downstream_target_background_dataset_audit.csv",
            "This is selected-LR/module functional support, not pan-atlas validation for every LR. Many scanned LR lack curated target sets, and the associations remain observational.",
            "Goal 2",
        ),
        claim(
            "Gene-program enrichment",
            "program_enrichment_added_with_specificity_caveat",
            "Phase 12 now tests LR-level and module-level receiver gene programs against GO, Reactome, KEGG, CellChatDB, and OmniPath resources.",
            "gene_program_enrichment/phase12_gene_program_enrichment_summary.csv",
            "The target-rank layer has selected support, but pathway/gene-program enrichment is confounded by broad programs and negative-specific leakage in the expanded rerun; do not sell it as clean pathway-specific downstream closure.",
            "Goal 2",
        ),
        claim(
            "Pathway-specific receiver programs",
            "specific_signal_confounded_by_negative_leakage",
            "Phase 12 now separates clean pathway-specific receiver programs from broad housekeeping-dominated programs and negative-specific leakage; in the expanded rerun all three datasets are leakage-confounded at this layer, so pathway-specific programs are a guardrail rather than a positive clean-closure claim.",
            "pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv",
            "This is still observational receiver-side association, not perturbational validation. Use the clean/mixed/leakage labels to choose stories and avoid overstating lymph-node chemokine downstream closure.",
            "Goal 2",
        ),
        claim(
            "Program specificity audit",
            "specificity_audit_added",
            "Phase 12 now separates broad housekeeping-like enrichment from immune, ECM/stroma, migration/boundary, and signaling pathway support, and flags negative-specific leakage.",
            "gene_program_enrichment/phase12_gene_program_specificity_summary.csv",
            "This is a conservative annotation layer, not a biological ontology gold standard; use it to prevent overclaiming and to choose cleaner story candidates.",
            "Goal 2/3",
        ),
        claim(
            "COMMOT-style communication impact",
            "expanded_adapter_downstream_comparison_available",
            "Phase 12 now compares SpatialSelf calibrated and COMMOT adapter scores on the same expanded-atlas LR selection using receiver-side whole-transcriptome target scans. Post-Xenium anchors favor SpatialSelf, while lymph-node chemokine CCR7 pairs favor COMMOT and lymph-node CXCL12-CXCR4 favors SpatialSelf.",
            "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_method_comparison.csv",
            "This is a same-universe COMMOT-style adapter/downstream proxy, not a fully tuned official COMMOT communication_impact rerun. Do not claim SpatialSelf globally beats COMMOT on downstream response.",
            "Goal 2",
        ),
        claim(
            "Global directionality superiority",
            "not_supported",
            "Current directionality metrics do not show SpatialSelf globally beating COMMOT.",
            "directionality_benchmark/phase12_directionality_method_comparison.csv",
            "Can claim quantified source-sink consistency in selected settings, especially lymph node, but not global directionality superiority.",
            "Goal 5",
        ),
        claim(
            "Directionality statistical audit",
            "empirical_expected_flow_fdr_available",
            "Phase 12 now separates empirical expected-flow FDR and SpatialSelf-vs-COMMOT directionality calls.",
            "directionality_statistics/phase12_directionality_statistical_dataset_summary.csv",
            "Current FDR uses non-expected niche flows as the empirical null; use the separate label-permutation audit for stronger null checks.",
            "Goal 5",
        ),
        claim(
            "Directionality label-permutation audit",
            "aggregate_label_permutation_available_not_globally_supportive",
            "Phase 12 now includes LR-label and niche-label permutation p-values on aggregated niche-flow scores; lymph-node signals show niche-label support, but no LR passes both nulls at BH FDR <= 0.1.",
            "directionality_permutation/phase12_directionality_permutation_dataset_summary.csv",
            "This is stronger than the previous empirical expected-flow null, but it is not a coordinate-level edge rerun and does not support global SpatialSelf directionality superiority over COMMOT.",
            "Goal 5",
        ),
        claim(
            "Directionality edge-level permutation audit",
            "edge_level_null_available_with_limited_support",
            "Phase 12 now includes fixed-score edge-level directionality nulls that shuffle receiver niche/coordinate pairing and edge vectors; lymph-node full has receiver-pairing support for SpatialSelf CCL19-CCR7, CCL21-CCR7 and CD48-CD244, but vector-null support is absent.",
            "directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv",
            "This is not end-to-end model retraining under coordinate permutation and does not prove causal ligand transport or global SpatialSelf directionality superiority over COMMOT.",
            "Goal 5",
        ),
        claim(
            "Directionality claim guardrail",
            "selected_source_sink_support_only",
            "A unified directionality guardrail now maps source-sink alignment, vector coherence, local-retained versus long-range flow, label/edge permutation support, hard-negative leakage and COMMOT comparison into explicit claim calls. The supported SpatialSelf directionality claim is selected lymph-node source-sink/receiver-pairing support, not vector-coherent causal transport.",
            "directionality_claim_guardrail/phase12_directionality_claim_guardrail_dataset.csv",
            "Do not claim global SpatialSelf directionality superiority over COMMOT. Visium HD CRC directionality is not supported under this guardrail, and vector-null support remains limited.",
            "Goal 5",
        ),
        claim(
            "Directionality consensus",
            "selected_lymph_node_source_sink_consensus",
            "A consensus audit now combines expected-flow FDR, label permutation, edge receiver-pairing/vector nulls, high-expression negative flags and SpatialSelf-vs-COMMOT calls. It supports selected lymph-node source-sink/receiver-pairing claims, including SpatialSelf-specific CD48-CD244 in native lymph node, while showing COMMOT is stronger for several expanded lymph-node chemokine flows.",
            "directionality_consensus/phase12_directionality_consensus_dataset_summary.csv",
            "No dataset currently has strict edge-vector consensus, and this remains fixed-score directionality evidence rather than causal ligand transport or end-to-end coordinate-null proof.",
            "Goal 5",
        ),
        claim(
            "Coordinate model-rerun smoke",
            "end_to_end_coordinate_rerun_smoke_available",
            "Phase 12 now has end-to-end coordinate-permutation reruns in post-Xenium colon and human lymph node that shuffle cell coordinates, refresh candidate distances, retrain SpatialSelf, and rescan expanded-atlas LR statistics.",
            "coordinate_model_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_coordinate_model_rerun_summary.csv; coordinate_model_rerun_smoke/human_lymph_node_visium_raw_full/human_lymph_node_visium_raw_full_coordinate_model_rerun_summary.csv",
            "Both post-Xenium and human lymph node now have 20 coordinate reruns, but nominal coordinate discoveries are contaminated by high-expression controls; this does not solve whole-atlas FDR, directionality causality, or final coordinate-permutation significance.",
            "Goal 5",
        ),
        claim(
            "Coordinate control guardrail",
            "posthoc_control_guardrail_supports_story_candidates",
            "A post-hoc high-expression-control guardrail suppresses known high-expression negative controls in coordinate-rerun outputs while retaining conservative story-guardrail candidates in both post-Xenium colon and human lymph node.",
            "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv",
            "This is a control-panel calibration layer, not unbiased whole-atlas FDR. It should be used to choose conservative story candidates; lymph-node candidates remain biological-consistency support rather than a clean SpatialSelf-exclusive story.",
            "Goal 1/3/5",
        ),
        claim(
            "Coordinate-null story closure",
            "selected_story_coordinate_null_supported",
            "End-to-end coordinate-shuffle model reruns now support selected story LR in post-Xenium colon and human lymph node after high-expression-control guardrails.",
            "coordinate_null_closure/phase12_coordinate_null_closure_story.csv",
            "This is story-level coordinate-null support. Broad-background leakage remains high, so do not claim whole-atlas coordinate-null discovery FDR or causal transport.",
            "Goal 5",
        ),
        claim(
            "Coordinate broad-background leakage",
            "failure_mode_quantified_not_solved",
            "The coordinate high-expression-control guardrail now explicitly quantifies broad/background LR leakage: the story guardrail retains 15 pre-specified anchors with high-expression negatives suppressed, but most control-guardrail calls are broad/background LR.",
            "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv",
            "This is a reason not to claim unbiased whole-atlas discovery from coordinate reruns. Use the strict story guardrail for pre-specified biological anchors and treat broad/background leakage as a remaining calibration problem.",
            "Goal 1/3/5",
        ),
        claim(
            "Expanded whole-LR atlas scoring",
            "calibrated_expanded_rank_support",
            "SpatialSelf has been run on three expanded covered-LR atlases: post-Xenium colon, human lymph node Visium, and Visium HD CRC. The calibrated ranking removes zero-expression and high-expression-negative LR from the top10 in all three datasets.",
            "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv",
            "Post-hoc permutation audit is now available but does not support conservative q<=0.1 discoveries. Post-Xenium and Visium HD CRC still have hard-negative/control caveats, so this is ranking support, not broad discovery FDR.",
            "Goal 1",
        ),
        claim(
            "Goal 1 discovery-tier guardrail",
            "scoped_evidence_available_broad_fdr_blocked",
            "A dedicated discovery-tier audit now separates broad all-covered-LR FDR, target-decoy control FDR, matched-spatial diagnostics, discovery-eligible universe FDR, targeted-anchor support, and adapter-light baseline comparison across the three expanded atlases.",
            "goal1_discovery_tiers/phase12_goal1_discovery_tier_story.csv",
            "This table is the writing guardrail for Goal 1: SpatialSelf supports calibrated whole-LR ranking plus scoped FDR diagnostics, but broad all-covered-LR whole-atlas discovery FDR remains incomplete.",
            "Goal 1",
        ),
        claim(
            "Goal 1 closure contract",
            "goal1_not_ready_for_closure",
            "A formal closure contract now separates passed evidence from candidate diagnostics and blocking failures. Current status is not ready for Goal 1 closure: large-LR atlas runs and ranking pass, while strict broad FDR and coordinate broad/background leakage remain unresolved.",
            "goal1_closure_contract/phase12_goal1_closure_contract.csv; goal1_closure_contract/phase12_goal1_closure_contract_status.csv",
            "Use this as the decision gate for baseline replacement and manuscript wording. It does not add new discovery evidence and does not change Goal 1 from partial to complete.",
            "Goal 1",
        ),
        claim(
            "Goal 1 FDR bottleneck",
            "spatial_distance_null_bottleneck_blocks_broad_fdr",
            "A dedicated bottleneck summary shows all three expanded datasets have zero broad all-LR q<=0.1 discoveries, while target-decoy top100 control FDR is clean and selected eligible/matched diagnostics retain true-anchor support.",
            "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_story.csv",
            "The remaining blocker is the spatial-distance permutation layer. Do not convert eligible-universe or matched-spatial support into a broad all-covered-LR discovery-FDR claim; the next model work should train or calibrate against range-matched spatial-distance nulls.",
            "Goal 1",
        ),
        claim(
            "Goal 1 FDR interpretation framework",
            "broad_fdr_not_closed_scoped_support_available",
            "A layer-by-layer FDR interpretation framework now separates broad all-covered-LR FDR, pooled spatial-distance nulls, target-decoy contamination audits, guarded matched-null support, discovery-eligible universe support, targeted anchors, and supplementary expanded-panel evidence. It shows zero broad-FDR-supported datasets, spatial-null blockers in the three strict full-transcriptome atlases, selected guarded support in post-Xenium, eligible clean support in lymph node, and supplementary panel support in tonsil/melanoma/lymph node.",
            "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_story.csv",
            "This is an interpretation and writing guardrail, not a new FDR procedure. It should prevent scoped diagnostics from being written as broad whole-atlas discovery FDR.",
            "Goal 1",
        ),
        claim(
            "Spatial-null bottleneck map",
            "calibration_targets_identified_not_solved",
            "A spatial-null bottleneck map now identifies why broad whole-atlas FDR fails: true anchors can have receiver/LR-identity or matched-spatial support while pooled spatial-distance BH remains blocked, and broad category-balanced background LR dominate some top ranks.",
            "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_story.csv",
            "This is a diagnostic map, not FDR closure. Use it to justify the next model task: preserve range-matched true signals while suppressing broad/background category-balanced leakage.",
            "Goal 1",
        ),
        claim(
            "Guarded matched full-LR FDR",
            "selected_full_lr_matched_null_support",
            "A full-LR guarded matched-null audit now requires receiver-null, LR-identity-null and matched-spatial-null support simultaneously. Under this layer, post-Xenium SPP1-CD44 is supported at q<=0.1 without primary-control or receptor-shared contamination; lymph node and Visium HD CRC remain unsupported.",
            "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_story.csv",
            "This is a full-LR matched-null guarded discovery layer, not the conservative pooled spatial-distance whole-atlas FDR. Use it as selected SPP1-CD44 statistical support and keep broad pooled FDR caveats intact.",
            "Goal 1",
        ),
        claim(
            "Background-suppressed ranking variant",
            "exploratory_component_not_global_replacement",
            "An exploratory background-suppressed ranking improves lymph-node top10 recovery and is a post-Xenium candidate, supporting the calibration direction suggested by the spatial-null bottleneck map.",
            "background_suppression_attempt/phase12_background_suppression_story.csv",
            "Do not use this as the main model or a whole-atlas FDR result. CRC and broader top-k control diagnostics remain caveated, and exact-score permutation/control audits would be required before adoption.",
            "Goal 1",
        ),
        claim(
            "Background-suppressed exact-score permutation",
            "exact_score_audit_not_fdr_closed",
            "The background-suppressed score was audited under the same receiver, spatial-distance and LR-identity nulls. It improves selected top-k or best-true-q diagnostics, but no dataset gains whole-atlas q<=0.1 discoveries.",
            "background_suppressed_permutation_comparison/phase12_background_suppressed_permutation_story.csv",
            "This is useful calibration evidence and a negative control on overclaiming. It does not solve broad whole-atlas FDR and should not replace the main model without stronger null/FDR support.",
            "Goal 1",
        ),
        claim(
            "Spatial-null-aware exact-score prototype",
            "prototype_ranking_improves_but_fdr_not_closed",
            "A score-level next-model prototype explicitly penalizing broad/background and spatial-null risk was audited by recomputing the same score under receiver, spatial-distance and LR-identity nulls. It produces a cleaner lymph-node top10 ranking but still yields zero whole-atlas q<=0.1 true LR discoveries.",
            "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_story.csv",
            "This is not a retrained model and does not close broad whole-atlas FDR. Use it to justify the next training objective, not as a manuscript discovery result.",
            "Goal 1",
        ),
        claim(
            "Spatial-null contrastive training path",
            "next_model_training_smoke_available",
            "SpatialSelf now has a runnable spatial-null contrastive loss path. A full-LR lymph-node smoke run completed with 726 spatial-null contrastive pairs, alongside hard-negative, context, LR-identity and control contrastive losses.",
            "spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv",
            "This is a training smoke, not a full atlas rerun or FDR improvement result. Use it as the concrete next-model implementation direction.",
            "Goal 1",
        ),
        claim(
            "Spatial-null-trained expanded atlas variant",
            "mixed_variant_not_replacement",
            "A full expanded-atlas training variant with spatial-null contrastive loss completed on three datasets. It improves post-Xenium top10 true recovery and lowers top10 risk, but reduces top10 true recovery in lymph node and Visium HD CRC.",
            "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv",
            "Do not adopt this as the reporting baseline or claim FDR closure. Use it as evidence that the spatial-null loss direction needs dataset-adaptive weighting or annealing before full replacement.",
            "Goal 1",
        ),
        claim(
            "Anchor-preserving spatial-null variant",
            "best_next_model_candidate_exact_audit_not_closed",
            "A conservative anchor-preserving spatial-null variant preserves the lymph-node main top10 true count and improves post-Xenium top10 true recovery/risk. Its exact-score permutation audit preserves or improves selected top-k diagnostics, but still produces zero whole-atlas q<=0.1 discoveries.",
            "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv; spatial_null_anchor_preserving_permutation_comparison/phase12_spatial_null_anchor_preserving_permutation_story.csv",
            "This is the strongest next-model candidate so far, not a final reporting baseline. CRC remains weaker than the main model and exact-score/end-to-end whole-atlas FDR is not closed.",
            "Goal 1",
        ),
        claim(
            "Anchor plus background suppression",
            "calibration_direction_not_fdr_closure",
            "Combining anchor-preserving trained scores with background suppression improves top10 true-anchor recovery without increasing top10 risk versus the main calibrated ranking in all three expanded atlases, and partly recovers CRC. Exact-score permutation still has zero whole-atlas q<=0.1 discoveries.",
            "atlas_model_expanded_anchor_background_suppressed/phase12_anchor_background_suppressed_dataset_summary.csv; anchor_background_suppressed_permutation_comparison/phase12_anchor_background_permutation_story.csv",
            "Use as next-model/calibration direction only. It is post-hoc, worsens or fails to improve best-true-q in lymph node/post-Xenium, has CRC high-expression leakage, and does not close broad whole-atlas FDR.",
            "Goal 1",
        ),
        claim(
            "Anchor/background guarded training",
            "training_integration_not_fdr_closure",
            "A training-integrated anchor/background guarded variant improves top10 true recovery versus the main calibrated ranking in all three expanded atlases and removes best hard-negative beats, but exact-score permutation still has zero whole-atlas q<=0.1 discoveries.",
            "atlas_model_expanded_anchor_background_guarded_training/phase12_anchor_background_guarded_training_dataset_summary.csv; anchor_background_guarded_training_comparison/phase12_anchor_background_guarded_training_story.csv",
            "Use as next-model implementation evidence only. It can be riskier than the post-hoc anchor/background score in post-Xenium and still does not close broad whole-atlas FDR.",
            "Goal 1",
        ),
        claim(
            "Anchor/background range-guarded training",
            "range_null_training_not_fdr_closure",
            "A low-weight range-violation spatial-null training variant preserves the top10 true recovery of the anchor/background training path, improves post-Xenium top10 risk versus the random-receiver training variant, and improves selected best true q-values, but still has zero whole-atlas q<=0.1 discoveries.",
            "atlas_model_expanded_anchor_background_range_guarded_training/phase12_anchor_background_range_guarded_training_dataset_summary.csv; anchor_background_range_guarded_training_comparison/phase12_anchor_background_range_guarded_training_story.csv",
            "Use as evidence that simple range-violation negatives are not enough to close broad spatial-distance FDR. Do not present it as solved whole-atlas FDR.",
            "Goal 1",
        ),
        claim(
            "Null-component FDR bottleneck",
            "spatial_global_null_is_primary_blocker",
            "A component-level null diagnostic shows that receiver and LR-identity nulls support several true LR, especially in anchor/background guarded variants, but pooled spatial-distance global BH remains the dominant blocker; broad true q<=0.1 remains zero.",
            "null_component_diagnostic/phase12_null_component_story.csv; null_component_diagnostic/phase12_null_component_method_summary.csv",
            "This is a bottleneck explanation and calibration target, not a replacement FDR procedure. Do not claim broad whole-atlas FDR from non-spatial component support.",
            "Goal 1",
        ),
        claim(
            "Goal 1 model-variant decision",
            "variants_inform_next_model_not_fdr_closure",
            "A model-variant scoreboard now compares eight variants including main calibrated, post-hoc background suppression, anchor/background suppression, background-guarded training, range-violation training, training-integrated anchor/background guard, and low-weight range-violation anchor/background guard. Main calibrated remains the reporting baseline; anchor/background directions are next-model components; no tested variant closes whole-atlas FDR.",
            "goal1_model_variant_scoreboard/phase12_goal1_model_variant_story.csv",
            "Do not present model variants as final performance improvements unless exact-score or end-to-end FDR support is added. Use the scoreboard as the decision guardrail for the next model iteration.",
            "Goal 1",
        ),
        claim(
            "Range-weighted atlas FDR diagnostic",
            "range_weighted_diagnostic_supportive_with_caveat",
            "A label-independent weighted BH diagnostic over range-matched all-null p-values recovers true anchors in all three expanded datasets with zero weighted primary-control discoveries at q<=0.1.",
            "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_story.csv",
            "This is a diagnostic calibration experiment using expression/source/category support weights, not a replacement for the original conservative pooled whole-atlas FDR guardrail or an end-to-end model-rerun null.",
            "Goal 1",
        ),
        claim(
            "Variant matched-spatial FDR diagnostic",
            "variant_matched_spatial_supportive_with_caveat",
            "Matched-spatial and range-weighted diagnostics across four model variants support the anchor/background guarded training path most strongly: it has five strict matched true LR at q<=0.1 and zero matched/weighted control discoveries.",
            "variant_matched_spatial_fdr/phase12_variant_matched_spatial_story.csv; variant_matched_spatial_fdr/phase12_variant_matched_spatial_method_summary.csv",
            "This is still post-hoc matched-spatial/range-weighted evidence. It helps choose the next calibration direction, but it does not close broad pooled whole-atlas FDR or replace the main reporting baseline.",
            "Goal 1",
        ),
        claim(
            "LR-standardized all-null FDR diagnostic",
            "negative_diagnostic_model_score_gap_remains",
            "A stricter diagnostic standardized each LR against its own receiver, spatial-distance and LR-identity null distributions before whole-atlas BH. Across main, anchor/background guarded training, and anchor/background range-guarded training, no true LR passed q<=0.1.",
            "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_story.csv; lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_method_summary.csv",
            "This argues the remaining blocker is not only raw pooled-null scale mismatch. Do not use LR-standardized FDR as a positive discovery claim; use it to justify changing the model objective so true LR separate from nulls more strongly.",
            "Goal 1",
        ),
        claim(
            "Null-margin guarded training variant",
            "ranking_improved_fdr_not_closed",
            "A stronger null-margin guarded training variant improves top10 true recovery versus the main model in all three expanded atlases and removes hard-negative best-beat failures, with lymph node reaching 10/10 true LR in the top10.",
            "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_story.csv; null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_dataset_comparison.csv",
            "Exact-score broad whole-atlas FDR remains unclosed: zero true LR pass q<=0.1. Use this as next-model evidence for ranking/null separation, not as a discovery-FDR result or a reporting-baseline replacement.",
            "Goal 1",
        ),
        claim(
            "FDR margin-gap diagnosis",
            "spatial_null_gap_quantified",
            "A cross-variant margin-gap audit shows all tested variants are still blocked by the spatial-distance null. The closest broad-FDR case is main calibrated post-Xenium SPP1-CD44, still about 1.87x above the q<=0.1 target; null-margin training improves ranking but does not shrink the best q-gap.",
            "fdr_margin_gap/phase12_fdr_margin_gap_story.csv; fdr_margin_gap/phase12_fdr_margin_gap_dataset.csv",
            "This is a design-target diagnostic, not a new discovery result. It argues the next model must reshape spatial-null score separation directly, not merely increase generic null loss weights.",
            "Goal 1",
        ),
        claim(
            "Spatial-specificity calibration diagnostic",
            "q_gap_shrunk_not_fdr_closure",
            "A post-hoc spatial-specificity calibration recomputes window-in/window-out expression concentration under receiver, spatial-distance and LR-identity nulls. It improves best true q versus main in all three expanded datasets and brings post-Xenium SPP1-CD44 to q=0.112, but zero true LR pass q<=0.1.",
            "spatial_specificity_comparison/phase12_spatial_specificity_story.csv; spatial_specificity_permutation_audit/phase12_spatial_specificity_permutation_dataset_summary.csv",
            "This is not a baseline replacement or broad discovery-FDR result. It is evidence that spatial-window specificity helps the right failure mode, but it also increases top10 risk in post-Xenium/CRC and remains above the FDR threshold.",
            "Goal 1",
        ),
        claim(
            "Guarded spatial-specificity diagnostic",
            "near_fdr_not_closed",
            "Adding background, shared-anchor and zero-expression guards to spatial-specificity reduces the unguarded top10 risk in post-Xenium and CRC while preserving q-gap gains. Post-Xenium SPP1-CD44 reaches q=0.106 with clean top10 risk, but no true LR crosses q<=0.1.",
            "guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_story.csv; guarded_spatial_specificity_permutation_audit/phase12_guarded_spatial_specificity_permutation_dataset_summary.csv",
            "This is still a post-hoc exact-score diagnostic, not broad whole-atlas FDR closure or a final baseline. The result is useful because it identifies the next design: spatial-window concentration plus explicit background/risk guards.",
            "Goal 1",
        ),
        claim(
            "Integrated guarded spatial training candidate",
            "first_candidate_true_broad_fdr_hit_not_baseline",
            "Promoting guarded spatial-window specificity into a training/ranking candidate produces the first conservative broad q<=0.1 true LR hit among tested Phase 12 model directions: post-Xenium TGFB1-TGFBR2. The hit is stable or stronger when increasing exact-score permutations from 80 to 240, q=0.025 to q=0.00833. A low-permutation end-to-end coordinate-rerun smoke keeps TGFB1-TGFBR2 rank 1 with observed score above the coordinate-null q95, but does not pass BH FDR.",
            "atlas_model_expanded_integrated_guarded_spatial_training/phase12_integrated_guarded_spatial_training_dataset_summary.csv; integrated_guarded_spatial_training_permutation_audit/phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv; integrated_guarded_spatial_training_highres_stability/phase12_integrated_guarded_spatial_highres_stability_story.csv; integrated_guarded_spatial_coordinate_rerun_smoke/post_xenium_colon_full_10perm/post_xenium_colon_full_integrated_coordinate_summary.csv",
            "This is a candidate exact-score audit plus low-permutation coordinate-rerun smoke, not the current reporting baseline or formal end-to-end FDR. CRC still has top10 risk and coordinate-rerun broad/background leakage remains, so Goal 1 is improved but not globally closed.",
            "Goal 1",
        ),
        claim(
            "Integrated coordinate leakage decomposition",
            "high_expression_suppressed_broad_background_leakage",
            integrated_coordinate_leakage_claim(),
            "integrated_guarded_spatial_coordinate_leakage/phase12_integrated_coordinate_leakage_summary.csv; integrated_guarded_spatial_coordinate_leakage/phase12_integrated_coordinate_leakage_low_p_lr.csv",
            "Use this to say high-expression artifacts are better controlled in the candidate, but do not claim whole-atlas FDR because broad/background LR still share the strongest coordinate-null p-values.",
            "Goal 1",
        ),
        claim(
            "Integrated coordinate eligible-universe audit",
            "eligible_filter_not_sufficient",
            "Applying the discovery-eligible universe idea to the integrated coordinate-rerun result keeps true LR at the minimum p-value, but background LR and a ligand-shared control remain tied, so eligible-coordinate BH FDR does not pass q<=0.1.",
            "integrated_guarded_spatial_coordinate_eligible_audit/phase12_integrated_coordinate_eligible_dataset.csv; integrated_guarded_spatial_coordinate_eligible_audit/phase12_integrated_coordinate_eligible_low_p_lr.csv",
            "This is a negative/scoping diagnostic: simple eligibility filtering is not enough to close coordinate FDR for the integrated candidate.",
            "Goal 1",
        ),
        claim(
            "Integrated coordinate magnitude-gate diagnostic",
            "clean_selected_gate_not_final_fdr_rule",
            integrated_coordinate_magnitude_claim(),
            "integrated_guarded_spatial_coordinate_magnitude_gate/phase12_integrated_coordinate_magnitude_gate_best.csv; integrated_guarded_spatial_coordinate_magnitude_gate/phase12_integrated_coordinate_magnitude_gate_scan.csv",
            "This is diagnostic evidence that absolute score magnitude helps separate selected true anchors from broad/background leakage. It is not a prespecified final FDR rule and should guide the next training/calibration layer rather than be presented as whole-atlas closure.",
            "Goal 1",
        ),
        claim(
            "Integrated coordinate background target audit",
            "background_leakage_targets_defined",
            integrated_coordinate_background_claim(),
            "integrated_guarded_spatial_coordinate_background_targets/phase12_integrated_coordinate_background_target_summary.csv; integrated_guarded_spatial_coordinate_background_targets/phase12_integrated_coordinate_background_target_lr.csv",
            "This explains why low-permutation coordinate p-values alone are too coarse and identifies training/calibration targets. It does not close whole-atlas FDR until the guard is prespecified and rerun end-to-end.",
            "Goal 1",
        ),
        claim(
            "Integrated background-guard training target panel",
            "next_model_targets_specified",
            integrated_background_guard_target_claim(),
            "integrated_background_guard_training_targets/phase12_integrated_background_guard_training_spec.csv; integrated_background_guard_training_targets/phase12_integrated_background_guard_target_panel.csv",
            "This is a concrete next-model target panel, not new discovery evidence. It should be used to train or prespecify the next guarded model before repeating exact-score and coordinate-null FDR.",
            "Goal 1",
        ),
        claim(
            "Integrated background-target guarded training smoke",
            "next_model_training_path_runs",
            integrated_background_target_guarded_smoke_claim(),
            "integrated_background_target_guarded_training_smoke/phase12_integrated_background_target_guarded_dataset_summary.csv; integrated_background_target_guarded_training_smoke/phase12_integrated_background_target_guarded_target_summary.csv",
            "This is a low-budget one-dataset smoke. It supports the feasibility of the target-aware training path, not FDR improvement or baseline replacement.",
            "Goal 1",
        ),
        claim(
            "Integrated background-target guarded exact-score smoke",
            "next_model_exact_score_smoke_promising",
            integrated_background_target_guarded_permutation_smoke_claim(),
            "integrated_background_target_guarded_permutation_smoke/phase12_integrated_background_target_guarded_permutation_dataset_summary.csv; integrated_background_target_guarded_permutation_smoke/phase12_integrated_background_target_guarded_permutation_lr_fdr.csv",
            integrated_background_target_guarded_permutation_smoke_caveat(),
            "Goal 1",
        ),
        claim(
            "Integrated background-target guarded coordinate-rerun smoke",
            "next_model_coordinate_path_runs",
            integrated_background_target_guarded_coordinate_smoke_claim(),
            "integrated_background_target_guarded_coordinate_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_target_guarded_coordinate_summary.csv; integrated_background_target_guarded_coordinate_rerun_smoke/post_xenium_colon_full/post_xenium_colon_full_target_guarded_coordinate_lr_fdr.csv",
            "This is a low-permutation coordinate-null execution smoke. It supports feasibility and score stability, not formal FDR or baseline replacement.",
            "Goal 1",
        ),
        claim(
            "Range-null calibration variant",
            "not_adopted_exploratory_variant",
            "An exploratory range-null-aware ranking variant was tested, but it is not adopted because the true-LR gains are not consistent with improved control-risk diagnostics across datasets.",
            "range_null_calibration_attempt/phase12_range_null_calibration_story.csv",
            "Do not use this variant as the main Phase 12 model. It is useful only as a documented calibration attempt and as evidence that broad-background/control leakage remains the key modeling target.",
            "Goal 1",
        ),
        claim(
            "Expanded atlas permutation FDR",
            "posthoc_permutation_audit_not_significant",
            "Receiver, spatial-distance and LR-identity null audits show calibrated top LR exceed null q99, but no expanded-atlas LR reaches conservative BH FDR <= 0.1.",
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "Use this as an honest null audit and calibration target. It is post-hoc and does not retrain SpatialSelf under each null, so it is not final end-to-end permutation FDR control.",
            "Goal 1/3",
        ),
        claim(
            "Expanded atlas target-decoy/control FDR",
            "control_fdr_available_with_permutation_caveat",
            "A whole-ranking target-decoy audit now quantifies hard-control contamination across all 886/957 LR pairs in three expanded atlases: lymph node has clean top10 hard-control FDR, while post-Xenium and Visium HD CRC are caveated in top10 but improve by top100.",
            "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv",
            "This is not native ground-truth discovery FDR. It should be reported as hard-control contamination/FDR and paired with the stricter permutation audit, which still has no all-LR BH q<=0.1 discoveries.",
            "Goal 1/3",
        ),
        claim(
            "Strict atlas discovery calibration",
            "strict_topk_improved_but_global_fdr_blocked",
            "A stricter discovery score improves priority-anchor concentration, especially in lymph node where 9/10 strict top10 LR are pre-specified positives and hard-control FDR is clean, but post-Xenium and Visium HD CRC still have top10 ligand/background contamination and no expanded dataset has strict all-LR permutation q<=0.1 discoveries.",
            "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv",
            "Use this as evidence that we audited and stress-tested the ranking layer. Do not claim whole-atlas discovery FDR control from this result.",
            "Goal 1/3",
        ),
        claim(
            "Null-specific atlas FDR diagnostic",
            "pairing_identity_supported_spatial_null_bottleneck",
            "The expanded atlas permutation failure is now decomposed by null type across three expanded atlases. Receiver-pairing and LR-identity nulls support some true LR discoveries, but the spatial-distance null blocks conservative all-null whole-atlas FDR.",
            "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_story.csv",
            "This diagnostic explains the blocker but does not replace the conservative all-null FDR. Do not report pairing+identity q-values as final global discovery FDR.",
            "Goal 1",
        ),
        claim(
            "Spatial-distance dependency effect audit",
            "spatial_effects_detected_without_discovery_fdr",
            "A spatial-distance null effect audit shows that many true LR have observed scores above their shuffled-distance null mean across the three expanded atlases, but no true LR passes within-LR BH q<=0.1 under the spatial-distance null.",
            "spatial_dependency_audit/phase12_spatial_dependency_story.csv",
            "Use this to explain that the spatial null shows weak/moderate effect sizes rather than clean discovery FDR. Do not claim spatial-distance FDR control from these effect sizes.",
            "Goal 1",
        ),
        claim(
            "Mechanism-stratified atlas FDR",
            "stratified_all_null_fdr_not_supported",
            "A mechanism/category-stratified all-null sensitivity audit does not recover q<=0.1 true LR discoveries in the expanded atlases. This rules out simple heterogeneity across communication mechanisms as the only reason the pooled all-null FDR fails.",
            "stratified_atlas_fdr/phase12_stratified_atlas_fdr_story.csv",
            "Do not claim stratified whole-atlas FDR support. Use this as a negative diagnostic that motivates changing the model/null design rather than only re-stratifying the current scores.",
            "Goal 1",
        ),
        claim(
            "Matched spatial atlas FDR diagnostic",
            "story_level_support_with_global_caveat",
            "A matched spatial-null diagnostic recovers one post-Xenium colon matched all-null true LR hit, SPP1-CD44. Lymph node remains spatial-supported but all-null blocked, and Visium HD CRC is mainly a stress case.",
            "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv",
            "This is post-hoc matched-null evidence. It supports a conservative SPP1-CD44 story-level statement, not pooled/end-to-end whole-atlas discovery FDR or global SpatialSelf superiority.",
            "Goal 1/4",
        ),
        claim(
            "Discovery-eligible atlas FDR",
            "eligible_universe_support_with_control_caveat",
            "Within a pre-defined functionally annotated and expression-supported LR universe, SpatialSelf recovers FDR-supported true anchors across the expanded atlases, with clean matched controls in lymph node, a control caveat in post-Xenium, and Visium HD CRC acting as a control-contaminated stress case.",
            "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv",
            "This is not the same as FDR over every covered LR pair. It should be framed as eligible-universe discovery support, while the broader all-covered-LR pooled/end-to-end FDR remains unresolved.",
            "Goal 1",
        ),
        claim(
            "Discovery-eligible baseline comparison",
            "spatialself_eligible_universe_topk_advantage",
            "On the same discovery-eligible LR universe, SpatialSelf has higher top-10 true-anchor recall and fewer hard controls than COMMOT, CellChat, SpaTalk and CellNEST adapter-light baselines.",
            "discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv",
            "This is an adapter-light/projected same-universe ranking comparison. It does not replace fully tuned official end-to-end baselines or prove global whole-atlas superiority.",
            "Goal 1",
        ),
        claim(
            "Targeted matched-null anchor validation",
            "supported_as_prespecified_story_validation",
            "Pre-specified Phase 12 anchor LR pairs now have targeted permutation-panel q-values and matched-control/hard-negative audits; lymph node has strong targeted matched support, while post-Xenium colon supports SPP1-CD44/TGFB1-TGFBR2/CXCL12-CXCR4 with a hard-negative caveat.",
            "targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv",
            "This is not unbiased whole-atlas discovery FDR. Use it to validate pre-specified biological stories and keep whole-atlas BH FDR as the stricter discovery criterion.",
            "Goal 1/4",
        ),
        claim(
            "Strict biological story triage",
            "story_triage_available",
            "Phase 12 now integrates coordinate reruns, high-expression control suppression, targeted/matched nulls, downstream target enrichment, pathway-specific receiver programs, adapter-light baseline ranks and directionality into a strict story-candidate table. Under this audit, post-Xenium SPP1-CD44 is the only current A-class main story, while lymph-node chemokines are B-class biological-consistency candidates.",
            "strict_story_candidates/phase12_strict_story_candidate_evidence.csv",
            "This is story triage, not unbiased whole-atlas discovery. B-class lymph-node candidates should be framed as biological consistency with statistical support, not as clean SpatialSelf-exclusive discoveries.",
            "Goal 4/6",
        ),
        claim(
            "Non-CRC second-story closure",
            "conservative_lymph_node_A_candidate_available",
            "A candidate-level closure audit identifies lymph-node CXCL12-CXCR4 as the current strongest non-CRC A-class candidate when framed conservatively, while CCL19/CCL21-CCR7, tonsil IL21-IL21R, and melanoma PDCD1-CD274/SPP1-CD44 remain supportive but blocked by missing target support or downstream negative leakage.",
            "second_story_closure/phase12_second_story_closure_candidate_evidence.csv",
            "This does not turn real tissue into ground truth and does not prove causality. Tonsil and melanoma should remain supportive cross-platform evidence until downstream negative leakage is resolved.",
            "Goal 4",
        ),
        claim(
            "Cross-platform closure matrix",
            "conservative_cross_platform_validation_supported",
            "The cross-platform closure matrix supports lymph-node CXCL12-CXCR4 as the conservative A-class non-CRC second story and separates SpaMosaic tonsil A1/D1 as supportive platform validation, while native melanoma remains not clean under current controls.",
            "cross_platform_closure_matrix/phase12_cross_platform_closure_candidate_matrix.csv",
            "Do not claim every platform is clean. Tonsil has downstream/negative-leakage caveats, post-Xenium remains the main CRC case, and melanoma should not be used as a success story without additional closure.",
            "Goal 4",
        ),
        claim(
            "Expanded same-universe simple baselines",
            "supported_against_simple_heuristic_baselines",
            "On the exact same 886/957 LR candidate universe, calibrated SpatialSelf has cleaner top10 rankings than expression, coexpression-distance, distance-rule, and high-expression naive baselines.",
            "atlas_model_expanded/same_universe_baselines/phase12_expanded_same_universe_baseline_method_summary.csv",
            "This is not an official COMMOT/CellChat/SpaTalk/CellNEST expanded benchmark; it only shows simple same-universe heuristics are contaminated or low-recall.",
            "Goal 1/3",
        ),
        claim(
            "Official expanded baseline comparison",
            "adapter_light_same_universe_completed",
            "COMMOT, CellChat, SpaTalk, and CellNEST adapter-light baselines now run on the same Phase 12 886/957 LR candidate universe; COMMOT/CellChat/CellNEST show high-expression top10 contamination and SpaTalk is clean but low-recall.",
            "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_method_summary.csv",
            "Describe these as adapter-light/projected same-universe baselines, not fully tuned official end-to-end runs. Whole-atlas superiority still requires stronger permutation FDR.",
            "Goal 1",
        ),
        claim(
            "Supplementary expanded-panel cross-platform atlas",
            "supportive_panel_atlas_with_calibration_caveat",
            "A supplementary 2011-gene expanded-panel atlas now runs SpatialSelf on tonsil, melanoma, and lymph-node real-background constructed replicates. Tonsil recovers CXCL13-CXCR5 and CD40LG-CD40 in the top10, and melanoma recovers PDCD1-CD274, SPP1-CD44, CXCL10-CXCR3, and CXCL9-CXCR3 in the top10.",
            "supplementary_cross_platform_expanded_panel_atlas/phase12_supplementary_expanded_panel_dataset_summary.csv",
            "This is cross-platform support, not strict full-transcriptome whole-atlas FDR closure. Hard-negative leakage remains visible, especially in lower-ranked anchors and in the lymph-node expanded-panel run.",
            "Goal 1/4",
        ),
        claim(
            "Whole-atlas superiority",
            "not_yet_supported",
            "Calibrated expanded covered-LR ranking and adapter-light official baseline comparisons are now available, but conservative whole-atlas permutation FDR is still not significant.",
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "Need stronger permutation-calibrated FDR and ideally fully tuned official end-to-end baselines before claiming whole-atlas superiority.",
            "Goal 1",
        ),
        claim(
            "Atlas FDR/contamination audit",
            "control_panel_and_posthoc_permutation_available",
            "Phase 12 now summarizes top-k contamination, hard-negative leakage, and post-hoc permutation FDR for atlas runs.",
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "The calibrated top10 is cleaner, but conservative post-hoc permutation FDR is not significant at q<=0.1; do not claim final global FDR control.",
            "Goal 1/3",
        ),
        claim(
            "Atlas FDR closure",
            "audit_complete_discovery_claim_blocked",
            "A whole-atlas FDR closure audit now separates strict permutation FDR from target-decoy/control FDR. Lymph node has clean top10 control FDR and both datasets improve by top100, but strict all-LR permutation FDR has no q<=0.1 discoveries.",
            "atlas_fdr_closure/phase12_atlas_fdr_closure_story.csv",
            "Do not claim positive whole-atlas discovery FDR or whole-atlas superiority. Use the audit to report control contamination and to justify the remaining calibration/model-improvement need.",
            "Goal 1",
        ),
        claim(
            "Manuscript figure source-data package",
            "draft_figures_rendered",
            "Phase 12 now has four rendered manuscript draft figures with SVG/PDF/PNG outputs and editable SVG text; regenerable TIFF artifacts were removed after preserving same-stem counterparts.",
            "figures/phase12_rendered_figure_status.csv",
            "These are manuscript drafts, not final journal-submission art; biological claims still depend on the underlying evidence caveats.",
            "Goal 6",
        ),
        claim(
            "Requirement-level completion audit",
            "phase12_active_not_complete",
            "A requirement-level audit now maps the six Phase 12 goals to current evidence. Goal 2, Goal 3, Goal 4, Goal 5 and Goal 6 are complete under the current conservative definitions, while Goal 1 remains partial.",
            "completion_audit/phase12_requirement_completion_audit.csv",
            "Do not mark Phase 12 complete until the audit has no not-supported or missing hard requirements and the partial requirements are either completed or explicitly descoped by the user.",
            "Goal 1-6",
        ),
    ]


def dataset_claims() -> list[dict[str, object]]:
    rows = []
    cross = read_csv(PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv")
    second = read_csv(PHASE12_DIR / "second_story/phase12_second_story_evidence_matrix.csv")
    if not cross.empty:
        for row in cross.itertuples(index=False):
            if row.dataset_key == "post_xenium_colon":
                status = "main_case_supported_with_downstream_caveat"
                supported = "Post-Xenium colon is the strongest current main case: positive LR ranks ahead of hard negatives and local-contrast FDR suppresses negatives."
                caveat = "Clean pathway-specific downstream file remains conservative; use whole-transcriptome target-enrichment support instead of overstating pathway-specific proof."
            elif row.dataset_key == "spamosaic_tonsil_A1":
                status = "supportive_second_platform_candidate"
                supported = "SpaMosaic tonsil A1 supports cross-platform generalization: SpatialSelf ranks CXCL13-CXCR5 first and local-contrast FDR suppresses negatives."
                caveat = "Targeted cell-type FDR still shows negative leakage and pathway-specific downstream is not clean."
            elif row.dataset_key == "spamosaic_tonsil_D1":
                status = "supportive_but_incomplete"
                supported = "SpaMosaic tonsil D1 supports positive-over-negative ranking for CD40LG-CD40."
                caveat = "Local-contrast FDR summary is missing and targeted FDR has negative leakage."
            elif row.dataset_key == "xenium_human_lymph_node":
                status = "platform_support_mixed"
                supported = "Xenium lymph node has graphclust-residual/FDR support for positive controls."
                caveat = "Raw ranking has a negative top hit and downstream is mixed."
            elif row.dataset_key == "native_human_melanoma_multiome":
                status = "do_not_use_as_success_story"
                supported = "Native melanoma currently identifies plausible positives, but controls are not clean."
                caveat = "Claim audit explicitly says not to claim native real-sample functional/FDR success."
            else:
                status = row.story_class
                supported = f"{row.dataset_key} audited for cross-platform story readiness."
                caveat = row.recommended_use
            rows.append(
                claim(
                    f"Dataset story: {row.dataset_key}",
                    status,
                    supported,
                    "cross_platform_story/phase12_cross_platform_story_audit.csv",
                    caveat,
                    "Goal 4",
                )
            )
    if not second.empty:
        for row in second.itertuples(index=False):
            if row.candidate_story in {"CCL19-CCR7", "CCL21-CCR7"}:
                status = "biological_consistency_story"
            elif row.candidate_story == "CD48-CD244":
                status = "supportive_weak_context_story"
            else:
                status = "not_main_story"
            rows.append(
                claim(
                    f"Lymph-node candidate: {row.candidate_story}",
                    status,
                    f"{row.candidate_story} has SpatialSelf class {row.spatialself_story_class}; SpatialSelf flow={row.spatialself_expected_flow}, target enrichment={row.spatialself_target_enrichment}.",
                    "second_story/phase12_second_story_evidence_matrix.csv",
                    "Use as biological consistency validation, not a SpatialSelf-exclusive discovery, because COMMOT and hard negatives remain competitive.",
                    "Goal 4/5",
                )
            )
    return rows


def method_claims() -> list[dict[str, object]]:
    hard = read_csv(PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv")
    rows = []
    if not hard.empty:
        for row in hard.itertuples(index=False):
            if row.dataset_key == "post_xenium_colon_full":
                status = "partial_hard_negative_control"
                supported = "Post-Xenium colon has lower hard-negative leakage than Visium HD CRC in the pilot."
                caveat = "Rank-score empirical pass is still 0 in the pilot; do not claim complete FDR control from this table alone."
            elif row.dataset_key == "human_lymph_node_visium_raw_full":
                status = "lymph_node_atlas_rank_support_downstream_gap"
                supported = (
                    "Human lymph node raw full-transcriptome atlas pilot ranks chemokine anchors highly "
                    "and has some rank-score hard-negative passes."
                )
                caveat = (
                    f"Downstream pass is {int(row.n_positive_downstream_pass)} and receptor-shared beats are "
                    f"{int(row.n_receptor_shared_beats)}; use this as atlas ranking/module support, not downstream validation."
                )
            else:
                status = "failure_mode_documented"
                supported = "Visium HD CRC exposes hard-negative leakage and weak downstream support."
                caveat = "Use this as a stress-test/failure-mode result, not as a positive biological case."
            rows.append(
                claim(
                    f"Hard-negative stress: {row.dataset_key}",
                    status,
                    supported,
                    "hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv",
                    caveat,
                    "Goal 3",
                )
            )
    return rows


def integrated_coordinate_leakage_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_leakage"
        / "phase12_integrated_coordinate_leakage_summary.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "The integrated coordinate-rerun audit separates false-positive modes, but the leakage summary file is missing."
    row = table.iloc[0]
    return (
        f"The {int(row.n_completed_permutations)}-permutation integrated coordinate-rerun audit separates "
        f"false-positive modes: at the minimum p-value, true LR are present "
        f"({int(row.n_min_p_true_lr)}) and high-expression negatives are absent "
        f"({int(row.n_min_p_high_expression_negative)}), but broad/background LR dominate the minimum-p set "
        f"({int(row.n_min_p_broad_background)} of {int(row.n_min_p_lr)} minimum-p LR)."
    )


def integrated_coordinate_magnitude_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_magnitude_gate"
        / "phase12_integrated_coordinate_magnitude_gate_best.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "A score-magnitude gate diagnostic was planned, but the magnitude-gate summary file is missing."
    row = table.iloc[0]
    retained = str(row.top_lr_pairs).split(";")
    retained = [x for x in retained if x]
    retained_text = ", ".join(retained[: int(row.n_true_q_0_1)]) if retained else "the retained true anchors"
    return (
        f"A score-magnitude gate on the {int(row.n_lr)} retained-LR integrated coordinate-rerun family "
        f"uses threshold {float(row.score_threshold):.3g} and keeps {int(row.n_true_q_0_1)} true q<=0.1 "
        f"anchors ({retained_text}) while removing high-expression negatives "
        f"({int(row.n_high_expression_negative_q_0_1)}) and broad/background LR "
        f"({int(row.n_broad_background_q_0_1)}) at the selected threshold."
    )


def integrated_coordinate_background_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_background_targets"
        / "phase12_integrated_coordinate_background_target_summary.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "The broad/background coordinate leakage target list was planned, but the background-target summary file is missing."
    row = table.iloc[0]
    return (
        "The broad/background LR that block coordinate-rerun BH FDR are now converted into a concrete target list. "
        f"In {row.dataset_key}, {int(row.n_min_p_broad_background)} broad/background LR share the minimum "
        f"coordinate p-value among {int(row.n_min_p_lr)} minimum-p LR, but none exceed the diagnostic magnitude "
        f"gate and the retained true-anchor score floor is about "
        f"{float(row.retained_true_to_max_broad_score_gap):.0f}-fold above the maximum broad/background score."
    )


def integrated_background_guard_target_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_guard_training_targets"
        / "phase12_integrated_background_guard_training_spec.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "A broad/background guard target panel was planned, but the training specification file is missing."
    row = table.iloc[0]
    return (
        f"The formal coordinate leakage has been converted into a next-model target panel: "
        f"{int(row.n_background_targets)} broad/background LR targets, including "
        f"{int(row.n_primary_background_targets)} primary hard negatives, with a prespecified success criterion "
        "requiring true-anchor retention while high-expression and broad/background targets fail the guarded "
        "q<=0.1 criterion."
    )


def integrated_background_target_guarded_smoke_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_training_smoke"
        / "phase12_integrated_background_target_guarded_dataset_summary.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "The target-aware background-guard training smoke was planned, but its dataset summary file is missing."
    row = table.iloc[0]
    return (
        f"The target-aware background-guard training path now runs end-to-end on {row.dataset_key}: "
        f"{int(row.n_background_guard_target_train)}/{int(row.n_background_guard_train)} background-guard "
        f"training samples came from the target panel, top10 background targets changed from "
        f"{int(row.reference_top10_background_targets)} to {int(row.top10_background_targets)}, and top10 true LR "
        f"changed from {int(row.reference_top10_true_lr)} to {int(row.top10_true_lr)} in the low-budget smoke."
    )


def integrated_background_target_guarded_permutation_smoke_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_permutation_smoke"
        / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "The target-aware exact-score smoke was planned, but its dataset summary file is missing."
    row = table.iloc[0]
    n_perm = int(row.n_permutations_per_null)
    article = "An" if str(n_perm)[0] in {"8", "11", "18"} else "A"
    return (
        f"{article} {n_perm}-permutation exact-score smoke for the target-aware path gives "
        f"{int(row.all_true_lr_conservative_fdr_0_1)} "
        f"true LR at q<=0.1 in {row.dataset_key}, with best true {row.best_true_lr} q="
        f"{float(row.best_true_lr_fdr):.3g} and top10 risk {float(row.top10_negative_or_risk_fraction):.3g}."
    )


def integrated_background_target_guarded_permutation_smoke_caveat() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_permutation_smoke"
        / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv"
    )
    table = read_csv(path)
    n_perm = int(table.iloc[0]["n_permutations_per_null"]) if not table.empty else "low"
    article = "an" if str(n_perm)[0] in {"8", "11", "18"} else "a"
    return (
        f"This is only {article} {n_perm}-permutation one-dataset exact-score smoke. It should guide whether to "
        "scale the target-aware model, not be written as formal FDR."
    )


def integrated_background_target_guarded_coordinate_smoke_claim() -> str:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_summary.csv"
    )
    table = read_csv(path)
    if table.empty:
        return "The target-aware coordinate-rerun smoke was planned, but its summary file is missing."
    row = table.iloc[0]
    gap_detail = integrated_background_target_guarded_coordinate_score_gap(str(row.target_lr), str(row.best_negative_lr))
    return (
        f"The target-aware background-guard model now runs through an end-to-end coordinate-null smoke on "
        f"{row.dataset_key}: {int(row.n_completed_permutations)} coordinate reruns completed, "
        f"{row.target_lr} remains rank {int(row.target_rank)} with score {float(row.target_score):.3g} "
        f"versus null q95 {float(row.target_null_q95):.3g}, and the best high-expression negative is "
        f"{row.best_negative_lr} with p={float(row.best_negative_p):.3g}{gap_detail}."
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
    target_score = pd.to_numeric(pd.Series([target.iloc[0].get("calibrated_score")]), errors="coerce").iloc[0]
    negative_score = pd.to_numeric(pd.Series([negative.iloc[0].get("calibrated_score")]), errors="coerce").iloc[0]
    negative_rank = pd.to_numeric(pd.Series([negative.iloc[0].get("rank")]), errors="coerce").iloc[0]
    if pd.isna(target_score) or pd.isna(negative_score):
        return ""
    ratio = float(target_score) / max(float(negative_score), 1e-12)
    rank_text = int(negative_rank) if pd.notna(negative_rank) else "NA"
    return f"; negative rank={rank_text}, negative score={float(negative_score):.3g}, target/negative score ratio={ratio:.3g}x"


def claim(topic: str, support_status: str, supported_statement: str, evidence_file: str, caveat: str, phase12_goal: str) -> dict[str, object]:
    return {
        "topic": topic,
        "support_status": support_status,
        "supported_statement": supported_statement,
        "evidence_file": f"outputs/phase12/{evidence_file}",
        "caveat_or_forbidden_extension": caveat,
        "phase12_goal": phase12_goal,
    }


def write_status(table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Claim Matrix",
        "",
        "## Can Say",
        "",
    ]
    can = table[~table["support_status"].astype(str).str.contains("not_supported|not_yet|do_not", na=False)]
    for row in can.itertuples(index=False):
        lines.append(f"- **{row.topic}** (`{row.support_status}`): {row.supported_statement}")
        lines.append(f"  Caveat: {row.caveat_or_forbidden_extension}")
    lines += ["", "## Cannot Yet Say", ""]
    cannot = table[table["support_status"].astype(str).str.contains("not_supported|not_yet|do_not", na=False)]
    for row in cannot.itertuples(index=False):
        lines.append(f"- **{row.topic}** (`{row.support_status}`): {row.caveat_or_forbidden_extension}")
    lines += [
        "",
        "## Manuscript Framing",
        "",
        "- Main positive story: post-Xenium colon SPP1-CD44 / CRC communication, with ranking, local-contrast FDR, and whole-transcriptome downstream support.",
        "- Second biological-consistency story: lymph-node CXCL12-CXCR4 as the conservative SpatialSelf-supported candidate, with CCL19/CCL21-CCR7 kept as supportive chemokine biology where COMMOT is also strong.",
        "- Cross-platform support: SpaMosaic tonsil A1, emphasizing positive-over-negative ranking and local-contrast FDR but not claiming clean downstream proof.",
        "- Stress-test honesty: Visium HD CRC and native melanoma reveal failure modes under hard negatives; keep them as transparency/limitations unless improved.",
    ]
    (OUT_DIR / "phase12_claim_matrix_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
