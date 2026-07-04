#!/usr/bin/env python
"""Combine Phase 12 readiness, smoke and model-pilot evidence into one status file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR


def main() -> None:
    rows = []
    rows.extend(readiness_rows())
    rows.extend(completion_audit_rows())
    rows.extend(goal1_discovery_tier_rows())
    rows.extend(goal1_fdr_bottleneck_rows())
    rows.extend(goal1_fdr_interpretation_rows())
    rows.extend(goal1_closure_contract_rows())
    rows.extend(smoke_rows())
    rows.extend(model_pilot_rows())
    rows.extend(expanded_atlas_rows())
    rows.extend(supplementary_expanded_panel_rows())
    rows.extend(targeted_matched_null_rows())
    rows.extend(atlas_quality_rows())
    rows.extend(atlas_fdr_rows())
    rows.extend(atlas_fdr_closure_rows())
    rows.extend(strict_atlas_discovery_rows())
    rows.extend(null_specific_atlas_fdr_rows())
    rows.extend(spatial_dependency_rows())
    rows.extend(stratified_atlas_fdr_rows())
    rows.extend(matched_spatial_atlas_fdr_rows())
    rows.extend(guarded_matched_atlas_fdr_rows())
    rows.extend(range_weighted_atlas_fdr_rows())
    rows.extend(variant_matched_spatial_fdr_rows())
    rows.extend(lr_standardized_null_fdr_rows())
    rows.extend(null_margin_guarded_training_rows())
    rows.extend(fdr_margin_gap_rows())
    rows.extend(spatial_specificity_comparison_rows())
    rows.extend(guarded_spatial_specificity_comparison_rows())
    rows.extend(spatial_null_bottleneck_map_rows())
    rows.extend(background_suppression_attempt_rows())
    rows.extend(background_suppressed_permutation_comparison_rows())
    rows.extend(spatial_null_aware_score_permutation_rows())
    rows.extend(spatial_null_training_smoke_rows())
    rows.extend(spatial_null_trained_atlas_rows())
    rows.extend(spatial_null_anchor_preserving_rows())
    rows.extend(spatial_null_anchor_preserving_permutation_rows())
    rows.extend(anchor_background_suppressed_rows())
    rows.extend(anchor_background_guarded_training_rows())
    rows.extend(anchor_background_range_guarded_training_rows())
    rows.extend(integrated_guarded_spatial_training_rows())
    rows.extend(integrated_guarded_spatial_highres_stability_rows())
    rows.extend(null_component_diagnostic_rows())
    rows.extend(goal1_model_variant_scoreboard_rows())
    rows.extend(range_null_calibration_attempt_rows())
    rows.extend(discovery_eligible_atlas_fdr_rows())
    rows.extend(discovery_eligible_method_comparison_rows())
    rows.extend(receptor_shared_atlas_rows())
    rows.extend(downstream_rows())
    rows.extend(downstream_target_background_rows())
    rows.extend(gene_program_rows())
    rows.extend(hard_negative_stress_rows())
    rows.extend(directionality_rows())
    rows.extend(directionality_statistical_rows())
    rows.extend(directionality_edge_permutation_rows())
    rows.extend(directionality_claim_guardrail_rows())
    rows.extend(directionality_consensus_rows())
    rows.extend(coordinate_model_rerun_smoke_rows())
    rows.extend(integrated_guarded_spatial_coordinate_rerun_smoke_rows())
    rows.extend(integrated_guarded_spatial_coordinate_formal_parameter_smoke_rows())
    rows.extend(integrated_guarded_spatial_coordinate_summary_only_smoke_rows())
    rows.extend(coordinate_runner_concordance_rows())
    rows.extend(integrated_guarded_spatial_coordinate_leakage_rows())
    rows.extend(integrated_guarded_spatial_coordinate_eligible_rows())
    rows.extend(integrated_guarded_spatial_coordinate_magnitude_gate_rows())
    rows.extend(integrated_guarded_spatial_coordinate_background_target_rows())
    rows.extend(integrated_background_guard_training_target_rows())
    rows.extend(integrated_background_guard_interface_smoke_rows())
    rows.extend(integrated_background_target_guarded_training_smoke_rows())
    rows.extend(integrated_background_target_guarded_permutation_smoke_rows())
    rows.extend(integrated_background_target_guarded_coordinate_smoke_rows())
    rows.extend(coordinate_control_guardrail_rows())
    rows.extend(coordinate_null_closure_rows())
    rows.extend(strict_story_candidate_rows())
    rows.extend(second_story_rows())
    rows.extend(second_story_closure_rows())
    rows.extend(cross_platform_closure_rows())
    rows.extend(cross_platform_story_rows())
    rows.extend(figure_package_rows())
    rows.extend(claim_matrix_rows())
    rows.extend(final_limitations_next_model_rows())
    rows.extend(disk_cleanup_audit_rows())
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "phase12_current_status_table.csv", index=False)
    write_markdown(table)
    manifest = {
        "task": "phase12_current_status_summary",
        "outputs": [
            "phase12_current_status_table.csv",
            "phase12_current_status_summary.md",
        ],
        "claim_boundary": "Phase 12 remains active. Current evidence proves atlas input readiness and pilot model plumbing, not final whole-atlas FDR/downstream validation.",
    }
    (OUT_DIR / "phase12_current_status_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


def readiness_rows() -> list[dict[str, object]]:
    out = []
    coverage_path = PHASE12_DIR / "atlas_readiness/phase12_dataset_lr_coverage.csv"
    audit_path = PHASE12_DIR / "atlas_readiness/phase12_goal_audit.csv"
    hard_path = PHASE12_DIR / "atlas_readiness/phase12_hard_negative_templates.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        full = coverage[coverage["kind"].astype(str).str.contains("full", case=False, na=False)]
        out.append(
            {
                "phase12_component": "Goal 1 input readiness",
                "status": "ready_for_model_runs",
                "key_result": f"{len(coverage)} datasets audited; full-transcriptome LR coverage {int(full['n_lr_both_genes_present'].min()) if len(full) else 'NA'}-{int(full['n_lr_both_genes_present'].max()) if len(full) else 'NA'} LR pairs",
                "supporting_file": rel(coverage_path),
                "claim_level": "input coverage only",
            }
        )
    if hard_path.exists():
        hard = pd.read_csv(hard_path)
        out.append(
            {
                "phase12_component": "Goal 3 hard-negative templates",
                "status": "template_ready",
                "key_result": f"{len(hard)} hard-negative template rows across {hard['dataset_key'].nunique() if len(hard) else 0} datasets",
                "supporting_file": rel(hard_path),
                "claim_level": "template only",
            }
        )
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        out.append(
            {
                "phase12_component": "Overall goal audit",
                "status": "partial",
                "key_result": "; ".join(f"{r.goal}: {r.status}" for r in audit.itertuples(index=False)),
                "supporting_file": rel(audit_path),
                "claim_level": "roadmap status",
            }
        )
    return out


def completion_audit_rows() -> list[dict[str, object]]:
    summary_path = PHASE12_DIR / "completion_audit/phase12_goal_completion_summary.csv"
    audit_path = PHASE12_DIR / "completion_audit/phase12_requirement_completion_audit.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Requirement-level completion audit: {row.goal}",
                "status": row.overall_status,
                "key_result": (
                    f"complete={int(row.n_complete)}/{int(row.n_requirements)}, "
                    f"partial={int(row.n_partial)}, "
                    f"not_supported={int(row.n_not_supported)}, missing={int(row.n_missing)}; "
                    f"blocking={row.blocking_requirements if pd.notna(row.blocking_requirements) and str(row.blocking_requirements) else 'none'}"
                ),
                "supporting_file": rel(audit_path),
                "claim_level": "requirement-level completion audit; supersedes coarse roadmap status for deciding whether Phase 12 is complete",
            }
        )
    return rows


def goal1_discovery_tier_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "goal1_discovery_tiers/phase12_goal1_discovery_tier_dataset.csv"
    story_path = PHASE12_DIR / "goal1_discovery_tiers/phase12_goal1_discovery_tier_story.csv"
    if not dataset_path.exists() or not story_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 discovery-tier dataset: {row.dataset_key}",
                "status": row.dataset_support_tier,
                "key_result": row.dataset_claim_boundary,
                "supporting_file": rel(dataset_path),
                "claim_level": "Goal 1 scoped discovery-tier audit; separates broad whole-atlas FDR from eligible/targeted/control diagnostics",
            }
        )
    for row in story.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 discovery-tier claim: {row.topic}",
                "status": row.support_status,
                "key_result": row.evidence_summary,
                "supporting_file": rel(story_path),
                "claim_level": row.claim_boundary,
            }
        )
    return rows


def goal1_fdr_bottleneck_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv"
    story_path = PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 FDR bottleneck: {row.dataset_key}",
                "status": row.goal1_fdr_state,
                "key_result": (
                    f"broad true q<=0.1={int(row.broad_true_lr_q_0_1)}, "
                    f"spatial true q<=0.1={int(row.spatial_distance_true_q_0_1)}, "
                    f"matched spatial true q<=0.1={int(row.matched_spatial_true_q_0_1)}, "
                    f"eligible matched true q<=0.1={int(row.eligible_matched_true_q_0_1)}, "
                    f"target-decoy top100 FDR={fmt_num(row.target_decoy_top100_fdr)}, "
                    f"bottleneck={row.dominant_bottleneck_null}; next={row.next_model_or_null_action}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "Goal 1 blocker audit; explains why scoped FDR diagnostics do not close broad whole-atlas FDR",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 FDR bottleneck story: {row.topic}",
                    "status": row.overall_state,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, broad-supported={int(row.n_broad_supported)}, "
                        f"spatial-distance bottleneck={int(row.n_spatial_distance_bottleneck)}, "
                        f"eligible-universe true-hit datasets={int(row.n_eligible_universe_with_true_hits)}, "
                        f"target-decoy top100 clean datasets={int(row.n_target_decoy_top100_clean)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.manuscript_boundary,
                }
            )
    return rows


def goal1_closure_contract_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "goal1_closure_contract/phase12_goal1_closure_contract_status.csv"
    if not path.exists():
        return []
    status = pd.read_csv(path)
    rows = []
    for row in status.itertuples(index=False):
        rows.append(
            {
                "phase12_component": "Goal 1 closure contract",
                "status": row.overall_status,
                "key_result": (
                    f"contract items={int(row.n_contract_items)}, pass={int(row.n_pass)}, "
                    f"candidate/partial={int(row.n_candidate_or_partial)}, "
                    f"fail/not-ready={int(row.n_fail_or_not_ready)}; "
                    f"main blocker={row.main_blocker}; next={row.next_required_action}"
                ),
                "supporting_file": rel(path),
                "claim_level": "requirement contract for Goal 1 closure; not new discovery evidence",
            }
        )
    return rows


def goal1_fdr_interpretation_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_dataset_summary.csv"
    story_path = PHASE12_DIR / "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_story.csv"
    layer_path = PHASE12_DIR / "goal1_fdr_interpretation_framework/phase12_goal1_fdr_interpretation_layers.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 FDR interpretation framework: {row.dataset_key}",
                "status": row.dataset_fdr_interpretation,
                "key_result": (
                    f"layers={int(row.n_layers)}, broad-supported={int(row.n_broad_fdr_supported_layers)}, "
                    f"blockers={int(row.n_blocking_layers)}, control-supported={int(row.n_control_contamination_supported_layers)}, "
                    f"scoped/support layers={int(row.n_scoped_or_supplementary_support_layers)}; "
                    f"{row.allowed_dataset_claim}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "layer-by-layer FDR interpretation framework; separates broad discovery FDR from scoped/control/supplementary support",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 FDR interpretation story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"strict full-transcriptome datasets={int(row.n_strict_full_transcriptome_datasets)}, "
                        f"supplementary panel datasets={int(row.n_supplementary_panel_datasets)}, "
                        f"broad-supported={int(row.n_datasets_with_broad_fdr_support)}, "
                        f"spatial-null blockers={int(row.n_datasets_with_spatial_null_blocker)}, "
                        f"selected guarded support={int(row.n_datasets_with_selected_guarded_support)}, "
                        f"eligible clean support={int(row.n_datasets_with_eligible_clean_support)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.manuscript_claim,
                }
            )
    if layer_path.exists():
        layers = pd.read_csv(layer_path)
        counts = layers["claim_tier"].value_counts().to_dict() if "claim_tier" in layers else {}
        rows.append(
            {
                "phase12_component": "Goal 1 FDR interpretation layer inventory",
                "status": "completed_layer_inventory",
                "key_result": "; ".join(f"{key}={int(value)}" for key, value in sorted(counts.items())),
                "supporting_file": rel(layer_path),
                "claim_level": "inventory of broad, spatial-null, target-decoy, guarded, eligible, targeted and supplementary evidence layers",
            }
        )
    return rows


def smoke_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "atlas_smoke/phase12_atlas_smoke_hard_negative_leakage.csv"
    rank_path = PHASE12_DIR / "atlas_smoke/phase12_atlas_smoke_lr_ranking.csv"
    if not path.exists() or not rank_path.exists():
        return []
    hard = pd.read_csv(path)
    rank = pd.read_csv(rank_path)
    rows = []
    for dataset_key, sub in hard.groupby("dataset_key"):
        n = len(sub)
        beats = int(sub["negative_beats_positive"].sum()) if n else 0
        top_lr = "; ".join(
            rank[rank["dataset_key"].eq(dataset_key)].sort_values("rank")["lr_pair"].head(5).astype(str).tolist()
        )
        rows.append(
            {
                "phase12_component": f"Goal 1/3 expression-distance smoke: {dataset_key}",
                "status": "completed_baseline_smoke",
                "key_result": f"top LR: {top_lr}; hard negatives beating positives: {beats}/{n}",
                "supporting_file": rel(rank_path),
                "claim_level": "naive expression-distance failure mode",
            }
        )
    return rows


def model_pilot_rows() -> list[dict[str, object]]:
    status_path = PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_status.csv"
    rank_path = PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_lr_ranking.csv"
    hard_path = PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_hard_negative_leakage.csv"
    if not status_path.exists() or not rank_path.exists():
        return []
    status = pd.read_csv(status_path)
    rank = pd.read_csv(rank_path)
    hard = pd.read_csv(hard_path) if hard_path.exists() else pd.DataFrame()
    rows = []
    for stat in status.itertuples(index=False):
        dataset_key = stat.dataset_key
        sub = rank[rank["dataset_key"].eq(dataset_key)].sort_values("rank")
        top = "; ".join(
            f"{r.lr_pair}#{int(r.rank)}{' NEG' if bool(r.false_positive_high_expression) else ''}"
            for r in sub.head(6).itertuples(index=False)
        )
        hard_sub = hard[hard["dataset_key"].eq(dataset_key)] if len(hard) else pd.DataFrame()
        n = len(hard_sub)
        beats = int(hard_sub["negative_beats_positive"].sum()) if n else 0
        priority_top10 = int(sub[sub["true_lr"].astype(bool)]["rank"].le(10).sum()) if len(sub) else 0
        rows.append(
            {
                "phase12_component": f"Goal 1/3 SpatialSelf atlas model pilot: {dataset_key}",
                "status": stat.status,
                "key_result": f"{int(stat.n_lr_pairs)} LR, {int(stat.n_candidates)} candidates; top: {top}; priority anchors in top10={priority_top10}; hard negatives beating positives={beats}/{n}",
                "supporting_file": rel(rank_path),
                "claim_level": "pilot model evidence, not final atlas FDR",
            }
        )
    return rows


def expanded_atlas_rows() -> list[dict[str, object]]:
    status_path = PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_status.csv"
    dataset_path = PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_dataset_summary.csv"
    topk_path = PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_topk_summary.csv"
    module_path = PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_modules.csv"
    calibrated_dataset_path = PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv"
    calibrated_topk_path = PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv"
    permutation_dataset_path = PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv"
    permutation_topk_path = PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_topk_summary.csv"
    target_decoy_dataset_path = PHASE12_DIR / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv"
    target_decoy_topk_path = PHASE12_DIR / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_topk_summary.csv"
    same_universe_method_path = PHASE12_DIR / "atlas_model_expanded/same_universe_baselines/phase12_expanded_same_universe_baseline_method_summary.csv"
    same_universe_dataset_path = PHASE12_DIR / "atlas_model_expanded/same_universe_baselines/phase12_expanded_same_universe_baseline_dataset_summary.csv"
    official_baseline_status_path = PHASE12_DIR / "atlas_model_expanded/same_universe_baselines/phase12_expanded_official_baseline_status.csv"
    official_adapter_method_path = PHASE12_DIR / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_method_summary.csv"
    official_adapter_dataset_path = PHASE12_DIR / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_dataset_summary.csv"
    if not status_path.exists() or not dataset_path.exists():
        return []
    status = pd.read_csv(status_path)
    dataset = pd.read_csv(dataset_path)
    topk = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    modules = pd.read_csv(module_path) if module_path.exists() else pd.DataFrame()
    calibrated_dataset = pd.read_csv(calibrated_dataset_path) if calibrated_dataset_path.exists() else pd.DataFrame()
    calibrated_topk = pd.read_csv(calibrated_topk_path) if calibrated_topk_path.exists() else pd.DataFrame()
    permutation_dataset = pd.read_csv(permutation_dataset_path) if permutation_dataset_path.exists() else pd.DataFrame()
    permutation_topk = pd.read_csv(permutation_topk_path) if permutation_topk_path.exists() else pd.DataFrame()
    target_decoy_dataset = pd.read_csv(target_decoy_dataset_path) if target_decoy_dataset_path.exists() else pd.DataFrame()
    target_decoy_topk = pd.read_csv(target_decoy_topk_path) if target_decoy_topk_path.exists() else pd.DataFrame()
    same_universe_method = pd.read_csv(same_universe_method_path) if same_universe_method_path.exists() else pd.DataFrame()
    same_universe_dataset = pd.read_csv(same_universe_dataset_path) if same_universe_dataset_path.exists() else pd.DataFrame()
    official_baseline_status = pd.read_csv(official_baseline_status_path) if official_baseline_status_path.exists() else pd.DataFrame()
    official_adapter_method = pd.read_csv(official_adapter_method_path) if official_adapter_method_path.exists() else pd.DataFrame()
    official_adapter_dataset = pd.read_csv(official_adapter_dataset_path) if official_adapter_dataset_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        stat = status[status["dataset_key"].eq(row.dataset_key)].head(1)
        runtime = float(stat["runtime_seconds"].iloc[0]) if len(stat) else np.nan
        rows.append(
            {
                "phase12_component": f"Goal 1 expanded SpatialSelf atlas model: {row.dataset_key}",
                "status": row.expanded_audit_call,
                "key_result": (
                    f"{int(row.n_lr)} LR, {int(row.n_candidates)} candidates, runtime={fmt_num(runtime)}s; "
                    f"top10 true LR={int(row.top10_true_lr)}, "
                    f"top10 risk fraction={fmt_num(row.top10_negative_or_risk_fraction)}, "
                    f"top10 zero-expression fraction={fmt_num(row.top10_zero_expr_fraction)}, "
                    f"best hard-negative comparisons beating positives={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "expanded covered-LR model run; exposes calibration need, not final whole-atlas superiority",
            }
        )
    if not calibrated_dataset.empty:
        for row in calibrated_dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 calibrated expanded SpatialSelf atlas ranking: {row.dataset_key}",
                    "status": row.calibrated_audit_call,
                    "key_result": (
                        f"{int(row.n_lr)} LR; top10 true LR={int(row.top10_true_lr)}, "
                        f"top10 high-expression negatives={int(row.top10_high_expression_negative)}, "
                        f"top10 zero-expression LR={int(row.top10_zero_expr_lr)}, "
                        f"top10 risk fraction={fmt_num(row.top10_negative_or_risk_fraction)}, "
                        f"best hard-negative comparisons beating positives={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}"
                    ),
                    "supporting_file": rel(calibrated_dataset_path),
                    "claim_level": "calibrated expanded ranking; not permutation-calibrated FDR or whole-atlas baseline superiority",
                }
            )
    if not calibrated_topk.empty:
        top10_cal = calibrated_topk[calibrated_topk["top_k"].eq(10)]
        rows.append(
            {
                "phase12_component": "Goal 1 calibrated expanded atlas top10 contamination",
                "status": "completed_calibrated_topk_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}: true={int(r.n_true_lr)}, highNeg={int(r.n_high_expression_negative)}, zeroExpr={int(r.n_zero_expr_lr)}, risk={fmt_num(r.negative_or_risk_fraction)}"
                    for r in top10_cal.itertuples(index=False)
                ),
                "supporting_file": rel(calibrated_topk_path),
                "claim_level": "calibrated top-k contamination audit; not permutation-calibrated FDR",
            }
        )
    if not permutation_dataset.empty:
        for row in permutation_dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 expanded atlas permutation audit: {row.dataset_key}",
                    "status": row.permutation_audit_call,
                    "key_result": (
                        f"{int(row.n_lr)} LR, nulls={int(row.n_null_types)}x{int(row.n_permutations_per_null)}; "
                        f"top10 true LR={int(row.top10_true_lr)}, top10 q<=0.1={int(row.top10_conservative_fdr_0_1)}, "
                        f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
                        f"best true LR={row.best_true_lr} q={fmt_num(row.best_true_lr_fdr)}, "
                        f"top10 risk fraction={fmt_num(row.top10_negative_or_risk_fraction)}"
                    ),
                    "supporting_file": rel(permutation_dataset_path),
                    "claim_level": "post-hoc expanded atlas permutation audit; model not retrained under nulls",
                }
            )
    if not permutation_topk.empty:
        top10_perm = permutation_topk[permutation_topk["top_k"].eq(10)]
        rows.append(
            {
                "phase12_component": "Goal 1 expanded atlas permutation top10 FDR",
                "status": "completed_posthoc_permutation_topk_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}: true={int(r.n_true_lr)}, risk={fmt_num(r.negative_or_risk_fraction)}, q<=0.1={int(r.n_conservative_fdr_0_1)}, medianQ={fmt_num(r.median_conservative_fdr)}"
                    for r in top10_perm.itertuples(index=False)
                ),
                "supporting_file": rel(permutation_topk_path),
                "claim_level": "post-hoc permutation FDR summary; currently not significant at q<=0.1",
            }
        )
    if not target_decoy_dataset.empty:
        for row in target_decoy_dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 expanded atlas target-decoy/control FDR: {row.dataset_key}",
                    "status": row.control_fdr_call,
                    "key_result": (
                        f"{int(row.n_lr)} LR; primary controls={int(row.n_primary_control_decoy)}, "
                        f"strict controls={int(row.n_strict_control_decoy)}, "
                        f"top10 primary FDR={fmt_num(row.top10_primary_control_prefix_fdr)}, "
                        f"top10 strict FDR={fmt_num(row.top10_strict_control_prefix_fdr)}, "
                        f"top100 primary FDR={fmt_num(row.top100_primary_control_prefix_fdr)}, "
                        f"passing top-k primary<=0.1={fmt_text(row.reported_topk_primary_fdr_le_0_1)}"
                    ),
                    "supporting_file": rel(target_decoy_dataset_path),
                    "claim_level": "whole-ranking hard-control contamination audit; not unbiased real-tissue discovery FDR",
                }
            )
    if not target_decoy_topk.empty:
        selected = target_decoy_topk[target_decoy_topk["top_k"].isin([10, 100])]
        rows.append(
            {
                "phase12_component": "Goal 1 expanded atlas target-decoy top-k control FDR",
                "status": "completed_control_fdr_topk_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key} top{int(r.top_k)}: primary={fmt_num(r.primary_control_prefix_fdr)}, strict={fmt_num(r.strict_control_prefix_fdr)}, controls={int(r.n_primary_control_decoy)}/{int(r.n_lr)}"
                    for r in selected.itertuples(index=False)
                ),
                "supporting_file": rel(target_decoy_topk_path),
                "claim_level": "top-k control FDR over hard decoys; report alongside permutation FDR",
            }
        )
    if not same_universe_method.empty:
        top_methods = same_universe_method.sort_values(
            ["n_clean_top10_datasets", "mean_top10_true_lr", "mean_top10_negative_or_risk_fraction"],
            ascending=[False, False, True],
        ).head(6)
        rows.append(
            {
                "phase12_component": "Goal 1 expanded same-universe heuristic baselines",
                "status": "completed_same_universe_heuristic_baseline_audit",
                "key_result": "; ".join(
                    f"{r.method}: clean={int(r.n_clean_top10_datasets)}/{int(r.n_datasets)}, top10true={fmt_num(r.mean_top10_true_lr)}, highNeg={fmt_num(r.mean_top10_high_expression_negative)}, risk={fmt_num(r.mean_top10_negative_or_risk_fraction)}"
                    for r in top_methods.itertuples(index=False)
                ),
                "supporting_file": rel(same_universe_method_path),
                "claim_level": "same expanded LR/candidate universe heuristic baselines; not official COMMOT/CellChat/CellNEST/SpaTalk full-universe benchmark",
            }
        )
    if not same_universe_dataset.empty:
        ss = same_universe_dataset[same_universe_dataset["method"].eq("spatialself_calibrated")]
        rows.append(
            {
                "phase12_component": "Goal 1 calibrated SpatialSelf vs simple same-universe baselines",
                "status": "spatialself_cleaner_than_simple_same_universe_baselines",
                "key_result": "; ".join(
                    f"{r.dataset_key}: SS top10 true={int(r.top10_true_lr)}, highNeg={int(r.top10_high_expression_negative)}, risk={fmt_num(r.top10_negative_or_risk_fraction)}, hardBeat={int(r.hard_negative_best_comparison_beats)}/{int(r.hard_negative_best_comparisons)}"
                    for r in ss.itertuples(index=False)
                ),
                "supporting_file": rel(same_universe_dataset_path),
                "claim_level": "supports advantage over simple expression/distance/rule baselines only",
            }
        )
    if not official_baseline_status.empty:
        all_available = bool(official_baseline_status["same_universe_phase12_available"].astype(bool).all()) if "same_universe_phase12_available" in official_baseline_status else False
        rows.append(
            {
                "phase12_component": "Goal 1 official expanded baseline status",
                "status": "official_adapter_light_same_universe_completed" if all_available else "official_same_universe_expanded_baselines_incomplete",
                "key_result": "; ".join(
                    f"{r.method}:{r.status}/sameUniverse={bool(r.same_universe_phase12_available)}"
                    for r in official_baseline_status.itertuples(index=False)
                ),
                "supporting_file": rel(official_baseline_status_path),
                "claim_level": "same-universe official adapter-light status; projected/tuned-method caveats still apply",
            }
        )
    if not official_adapter_method.empty:
        rows.append(
            {
                "phase12_component": "Goal 1 official adapter-light expanded baseline comparison",
                "status": "completed_official_adapter_light_benchmark",
                "key_result": "; ".join(
                    f"{r.method}: completed={int(r.n_completed_datasets)}/{int(r.n_datasets)}, top10true={fmt_num(r.mean_top10_true_lr)}, highNeg={fmt_num(r.mean_top10_high_expression_negative)}, risk={fmt_num(r.mean_top10_negative_or_risk_fraction)}, call={r.summary_call}"
                    for r in official_adapter_method.itertuples(index=False)
                ),
                "supporting_file": rel(official_adapter_method_path),
                "claim_level": "same 886/957 LR universe adapter-light benchmark; not fully tuned official end-to-end superiority",
            }
        )
    if not official_adapter_dataset.empty:
        best = official_adapter_dataset.sort_values(["dataset_key", "top10_true_lr"], ascending=[True, False])
        rows.append(
            {
                "phase12_component": "Goal 1 official adapter-light dataset details",
                "status": "completed_dataset_level_adapter_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}/{r.method}: top10true={int(r.top10_true_lr)}, highNeg={int(r.top10_high_expression_negative)}, risk={fmt_num(r.top10_negative_or_risk_fraction)}, best={r.best_true_lr}#{int(r.best_true_lr_rank)}"
                    for r in best.itertuples(index=False)
                ),
                "supporting_file": rel(official_adapter_dataset_path),
                "claim_level": "dataset-level official adapter-light ranking and contamination audit",
            }
        )
    if not topk.empty:
        top10 = topk[topk["top_k"].eq(10)]
        rows.append(
            {
                "phase12_component": "Goal 1 expanded atlas top10 contamination comparison",
                "status": "completed_expanded_topk_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}: true={int(r.n_true_lr)}, risk={fmt_num(r.negative_or_risk_fraction)}, zeroExpr={fmt_num(r.zero_expr_lr_fraction)}, category={r.top_biological_category}"
                    for r in top10.itertuples(index=False)
                ),
                "supporting_file": rel(topk_path),
                "claim_level": "expanded top-k contamination; not permutation-calibrated FDR",
            }
        )
    if not modules.empty:
        rows.append(
            {
                "phase12_component": "Goal 1 expanded atlas communication modules",
                "status": "completed_expanded_module_audit",
                "key_result": f"modules={len(modules)} across {modules['dataset_key'].nunique()} datasets; negative-heavy modules={int(modules['module_warning'].astype(str).str.contains('negative', case=False, na=False).sum())}; priority-anchor modules={int(modules['module_warning'].astype(str).str.contains('priority', case=False, na=False).sum())}",
                "supporting_file": rel(module_path),
                "claim_level": "expanded module summary; descriptive clustering",
            }
        )
    return rows


def supplementary_expanded_panel_rows() -> list[dict[str, object]]:
    base = PHASE12_DIR / "supplementary_cross_platform_expanded_panel_atlas"
    dataset_path = base / "phase12_supplementary_expanded_panel_dataset_summary.csv"
    topk_path = base / "phase12_supplementary_expanded_panel_topk_summary.csv"
    hard_path = base / "phase12_supplementary_expanded_panel_hard_negative_leakage.csv"
    status_path = base / "phase12_supplementary_expanded_panel_status.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    topk = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    hard = pd.read_csv(hard_path) if hard_path.exists() else pd.DataFrame()
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    rows = []
    rows.append(
        {
            "phase12_component": "Goal 1 supplementary expanded-panel cross-platform atlas",
            "status": "completed_supplementary_cross_platform_panel",
            "key_result": "; ".join(
                f"{r.dataset_key}: LR={int(r.n_lr)}, top10 true={int(r.top10_true_lr)}, "
                f"top10 risk={fmt_num(r.top10_negative_or_risk_fraction)}, "
                f"hardBeat={int(r.hard_negative_best_comparison_beats)}/{int(r.hard_negative_best_comparisons)}, "
                f"call={r.supplementary_call}"
                for r in dataset.itertuples(index=False)
            ),
            "supporting_file": rel(dataset_path),
            "claim_level": "supplementary 2011-gene expanded-panel atlas; cross-platform support only, not strict full-transcriptome FDR closure",
        }
    )
    if not topk.empty:
        top10 = topk[topk["top_k"].eq(10)]
        rows.append(
            {
                "phase12_component": "Goal 1 supplementary expanded-panel top10 contamination",
                "status": "completed_supplementary_topk_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}: true={int(r.n_true_lr)}, highNeg={int(r.n_high_expression_negative)}, "
                    f"zeroExpr={int(r.n_zero_expr_lr)}, risk={fmt_num(r.negative_or_risk_fraction)}"
                    for r in top10.itertuples(index=False)
                ),
                "supporting_file": rel(topk_path),
                "claim_level": "top-k contamination audit on 2011-gene constructed real-background replicates",
            }
        )
    if not hard.empty:
        grouped = hard.groupby("dataset_key")["negative_beats_positive"].agg(["sum", "count"])
        rows.append(
            {
                "phase12_component": "Goal 1 supplementary expanded-panel hard-negative leakage",
                "status": "completed_supplementary_hard_negative_audit",
                "key_result": "; ".join(
                    f"{dataset_key}: beats={int(row['sum'])}/{int(row['count'])}"
                    for dataset_key, row in grouped.iterrows()
                ),
                "supporting_file": rel(hard_path),
                "claim_level": "hard-negative leakage audit; exposes residual calibration risk in panel atlas",
            }
        )
    if not status.empty:
        rows.append(
            {
                "phase12_component": "Goal 1 supplementary expanded-panel run status",
                "status": "completed",
                "key_result": "; ".join(
                    f"{r.dataset_key}: source={r.source_replicate}, cells={int(r.n_cells)}, panelGenes={int(r.n_panel_genes)}, runtime={fmt_num(r.runtime_seconds)}s"
                    for r in status.itertuples(index=False)
                    if str(r.status) == "completed"
                ),
                "supporting_file": rel(status_path),
                "claim_level": "run provenance for supplementary expanded-panel atlas",
            }
        )
    return rows


def atlas_quality_rows() -> list[dict[str, object]]:
    quality_path = PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_quality.csv"
    module_path = PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_modules.csv"
    if not quality_path.exists():
        return []
    quality = pd.read_csv(quality_path)
    modules = pd.read_csv(module_path) if module_path.exists() else pd.DataFrame()
    rows = []
    for dataset_key, sub in quality.groupby("dataset_key"):
        top10 = sub[sub["top_k"].eq(10)].head(1)
        if top10.empty:
            continue
        row = top10.iloc[0]
        mod = modules[modules["dataset_key"].eq(dataset_key)] if not modules.empty else pd.DataFrame()
        n_negative_modules = int(mod["module_warning"].astype(str).str.contains("negative", case=False, na=False).sum()) if len(mod) else 0
        rows.append(
            {
                "phase12_component": f"Goal 1 atlas contamination/modules: {dataset_key}",
                "status": "completed_pilot_summary",
                "key_result": f"top10 priority anchors={int(row.n_priority_anchor)}, high-expression negatives={int(row.n_high_expression_negative)}, receptor-risk LR={int(row.n_receptor_shared_risk)}; modules={len(mod)}, negative-heavy modules={n_negative_modules}",
                "supporting_file": rel(quality_path),
                "claim_level": "pilot atlas quality summary",
            }
        )
    return rows


def targeted_matched_null_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv"
    lr_path = PHASE12_DIR / "targeted_matched_null/phase12_targeted_matched_null_lr_audit.csv"
    method_path = PHASE12_DIR / "targeted_matched_null/phase12_targeted_matched_null_method_comparison.csv"
    if not dataset_path.exists() or not lr_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    lr = pd.read_csv(lr_path)
    method = pd.read_csv(method_path) if method_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        sub = lr[lr["dataset_key"].eq(row.dataset_key)]
        supported = sub[
            sub["targeted_validation_call"].isin(
                ["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"]
            )
        ].sort_values("spatialself_rank")
        supported_text = "; ".join(
            f"{r.lr_pair}#{int(r.spatialself_rank)} panelQ={fmt_num(r.targeted_panel_bh_q)} matchedQ={fmt_num(r.matched_control_bh_q)}"
            for r in supported.head(5).itertuples(index=False)
        )
        rows.append(
            {
                "phase12_component": f"Goal 1/4 targeted matched-null anchor audit: {row.dataset_key}",
                "status": row.dataset_call,
                "key_result": (
                    f"anchors={int(row.n_anchor_lr)}, top10 anchors={int(row.n_rank_top10)}, "
                    f"whole-atlas q<=0.1={int(row.n_whole_atlas_fdr_0_1)}, "
                    f"targeted panel q<=0.1={int(row.n_targeted_panel_q_0_1)}, "
                    f"matched q<=0.1={int(row.n_matched_control_q_0_1)}, "
                    f"hard-neg q<=0.1={int(row.n_hard_negative_q_0_1)}; "
                    f"supported: {supported_text if supported_text else 'none'}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "pre-specified anchor-panel and matched-control validation; not whole-atlas discovery FDR",
            }
        )
    if not method.empty:
        rows.append(
            {
                "phase12_component": "Goal 1/4 targeted anchors vs adapter-light baselines",
                "status": "completed_targeted_baseline_rank_audit",
                "key_result": "; ".join(
                    f"{r.method}: supported={int(r.n_supported_anchor)}, SS top10 but baseline>10={int(r.n_supported_spatialself_top10_baseline_after10)}, medianRank={fmt_num(r.median_rank_on_supported_anchor)}"
                    for r in method.itertuples(index=False)
                ),
                "supporting_file": rel(method_path),
                "claim_level": "adapter-light rank comparison for targeted supported anchors; not fully tuned official end-to-end superiority",
            }
        )
    return rows


def atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_dataset_summary.csv"
    topk_path = PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_topk_summary.csv"
    positive_path = PHASE12_DIR / "atlas_fdr_audit/phase12_atlas_fdr_positive_control_summary.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 atlas FDR/contamination audit: {row.dataset_key}",
                "status": row.atlas_audit_call,
                "key_result": (
                    f"top10 true LR={int(row.top10_true_lr)}, "
                    f"top10 risk fraction={fmt_num(row.top10_negative_or_risk_fraction)}, "
                    f"top10 control-panel empirical FDR={fmt_num(row.top10_control_panel_empirical_fdr)}, "
                    f"top20 true LR={int(row.top20_true_lr)}, "
                    f"negative beat fraction={fmt_num(row.dataset_negative_beat_fraction)}, "
                    f"receptor-shared beats={int(row.dataset_receptor_shared_beats)}, "
                    f"positive empirical FDR<=0.1={int(row.n_positive_empirical_fdr_0_1)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "pilot control-panel empirical FDR; not final whole-atlas permutation FDR",
            }
        )
    if topk_path.exists():
        topk = pd.read_csv(topk_path)
        best = topk[topk["top_k"].eq(10)].copy()
        if not best.empty:
            text = "; ".join(
                f"{r.dataset_key}: risk={fmt_num(r.negative_or_risk_fraction)}, FDR={fmt_num(r.control_panel_empirical_fdr)}, true={int(r.n_true_lr)}"
                for r in best.itertuples(index=False)
            )
            rows.append(
                {
                    "phase12_component": "Goal 1 top10 atlas contamination comparison",
                    "status": "completed_control_panel_comparison",
                    "key_result": text,
                    "supporting_file": rel(topk_path),
                    "claim_level": "top-k control-panel contamination comparison",
                }
            )
    if positive_path.exists():
        positive = pd.read_csv(positive_path)
        if not positive.empty:
            passing = positive[pd.to_numeric(positive["empirical_bh_fdr_score"], errors="coerce").le(0.1)]
            rows.append(
                {
                    "phase12_component": "Goal 1/3 positive-control empirical FDR summary",
                    "status": "completed_positive_control_fdr_audit",
                    "key_result": f"positive controls={len(positive)}, empirical FDR<=0.1={len(passing)}, rank-score pass={int(positive['stress_pass_rank_score'].sum())}, downstream pass={int(positive['stress_pass_downstream'].sum())}",
                    "supporting_file": rel(positive_path),
                    "claim_level": "positive-control hard-negative FDR audit",
                }
            )
    return rows


def atlas_fdr_closure_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_dataset.csv"
    top_path = PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_top_lr.csv"
    story_path = PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    top = pd.read_csv(top_path) if top_path.exists() else pd.DataFrame()
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 atlas FDR closure: {row.dataset_key}",
                "status": row.dataset_fdr_closure_call,
                "key_result": (
                    f"LR={int(row.n_lr)}, top10 true={int(row.top10_true_lr)}, "
                    f"top10 highNeg={int(row.top10_high_expression_negative)}, "
                    f"top10 control FDR={fmt_num(row.top10_primary_control_fdr)}, "
                    f"top100 control FDR={fmt_num(row.top100_primary_control_fdr)}, "
                    f"all-LR permutation q<=0.1={int(row.all_lr_permutation_q_0_1)}, "
                    f"top10 background={int(row.top10_unlabeled_background)}, best true={row.best_true_lr} q={fmt_num(row.best_true_lr_permutation_fdr)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "global FDR closure audit; output complete but positive whole-atlas discovery claim blocked",
            }
        )
    if not top.empty:
        blockers = top[
            top["top_lr_fdr_call"].astype(str).isin(["hard_control_decoy_in_top_rank", "unlabeled_background_top_rank"])
        ]
        if len(blockers):
            rows.append(
                {
                    "phase12_component": "Goal 1 top-rank FDR blockers",
                    "status": "completed_failure_mode_breakdown",
                    "key_result": "; ".join(
                        f"{r.dataset_key}#{int(r.rank)} {r.lr_pair}:{r.top_lr_fdr_call}"
                        for r in blockers.head(10).itertuples(index=False)
                    ),
                    "supporting_file": rel(top_path),
                    "claim_level": "top-rank blocker audit for calibration/model improvement",
                }
            )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 atlas FDR closure overall",
                "status": s.overall_atlas_fdr_closure_call,
                "key_result": (
                    f"datasets={int(s.n_datasets)}, "
                    f"top10-control-clean={int(s.n_datasets_top10_control_fdr_le_0_1)}, "
                    f"top100-control-clean={int(s.n_datasets_top100_control_fdr_le_0_1)}, "
                    f"all-LR permutation discoveries={int(s.n_all_lr_permutation_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "whole-atlas FDR claim boundary",
            }
        )
    return rows


def strict_atlas_discovery_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv"
    topk_path = PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_topk.csv"
    story_path = PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    topk = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 strict atlas discovery audit: {row.dataset_key}",
                "status": row.strict_atlas_discovery_call,
                "key_result": (
                    f"LR={int(row.n_lr)}, top10 true={int(row.top10_true_lr)}, "
                    f"top10 background={int(row.top10_background_lr)}, "
                    f"top10 primary decoy={int(row.top10_primary_control_decoy)}, "
                    f"top10 primary FDR={fmt_num(row.top10_primary_control_prefix_fdr)}, "
                    f"top100 primary FDR={fmt_num(row.top100_primary_control_prefix_fdr)}, "
                    f"strict all-LR q<=0.1={int(row.all_lr_strict_permutation_q_0_1)}, "
                    f"best true={row.best_true_lr} q={fmt_num(row.best_true_lr_strict_fdr)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "strict discovery ranking audit; top-k control evidence only unless strict permutation q-values pass",
            }
        )
    if not topk.empty:
        top10 = topk[topk["top_k"].eq(10)]
        if not top10.empty:
            rows.append(
                {
                    "phase12_component": "Goal 1 strict atlas top10 summary",
                    "status": "strict_top10_summary_available",
                    "key_result": "; ".join(
                        f"{r.dataset_key}: true={int(r.n_true_lr)}, background={int(r.n_background_lr)}, "
                        f"primary decoy={int(r.n_primary_control_decoy)}, FDR={fmt_num(r.primary_control_prefix_fdr)}"
                        for r in top10.itertuples(index=False)
                    ),
                    "supporting_file": rel(topk_path),
                    "claim_level": "strict top-k control contamination summary",
                }
            )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 strict atlas discovery overall",
                "status": s.overall_strict_atlas_call,
                "key_result": (
                    f"datasets={int(s.n_datasets)}, "
                    f"top10-control-clean={int(s.n_datasets_top10_primary_control_fdr_le_0_1)}, "
                    f"top100-control-clean={int(s.n_datasets_top100_primary_control_fdr_le_0_1)}, "
                    f"strict all-LR discoveries={int(s.n_all_lr_strict_permutation_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "strict whole-atlas discovery claim boundary",
            }
        )
    return rows


def null_specific_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv"
    story_path = PHASE12_DIR / "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 null-specific atlas FDR diagnostic: {row.dataset_key}",
                "status": row.null_specific_audit_call,
                "key_result": (
                    f"receiver true q<=0.1={int(row.receiver_true_q_0_1)}, "
                    f"LR-identity true q<=0.1={int(row.lr_identity_true_q_0_1)}, "
                    f"pairing+identity true q<=0.1={int(row.two_null_pairing_identity_true_q_0_1)}, "
                    f"spatial-distance true q<=0.1={int(row.spatial_distance_true_q_0_1)}, "
                    f"conservative q<=0.1={int(row.conservative_q_0_1)}, "
                    f"bottleneck={row.dominant_bottleneck_null}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "diagnostic null decomposition; not a replacement for conservative all-null FDR",
            }
        )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 null-specific atlas FDR overall",
                "status": s.overall_null_specific_call,
                "key_result": (
                    f"pairing+identity discoveries={int(s.n_two_null_pairing_identity_q_0_1)}, "
                    f"pairing+identity true discoveries={int(s.n_two_null_pairing_identity_true_q_0_1)}, "
                    f"spatial-distance discoveries={int(s.n_spatial_distance_q_0_1)}, "
                    f"conservative discoveries={int(s.n_conservative_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "null-specific FDR claim boundary",
            }
        )
    return rows


def spatial_dependency_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "spatial_dependency_audit/phase12_spatial_dependency_dataset.csv"
    story_path = PHASE12_DIR / "spatial_dependency_audit/phase12_spatial_dependency_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 spatial-dependency audit: {row.dataset_key}",
                "status": row.spatial_dependency_audit_call,
                "key_result": (
                    f"true positive spatial delta={int(row.n_true_spatial_delta_positive)}/{int(row.n_true_lr)}, "
                    f"true spatial within q<=0.1={int(row.n_true_spatial_within_q_0_1)}, "
                    f"strict true spatial within q<=0.1={int(row.n_true_strict_spatial_within_q_0_1)}, "
                    f"top10 true positive delta={int(row.top10_true_spatial_delta_positive)}, "
                    f"best true delta={row.best_true_spatial_dependency_lr}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "spatial-distance null effect-size diagnostic; not global discovery FDR",
            }
        )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 spatial-dependency overall",
                "status": s.overall_spatial_dependency_call,
                "key_result": (
                    f"true LR with positive spatial delta={int(s.n_true_lr_with_positive_spatial_delta)}, "
                    f"true within q<=0.1={int(s.n_true_lr_spatial_within_q_0_1)}, "
                    f"strict true within q<=0.1={int(s.n_true_lr_strict_spatial_within_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "spatial-dependency diagnostic claim boundary",
            }
        )
    return rows


def stratified_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "stratified_atlas_fdr/phase12_stratified_atlas_fdr_dataset.csv"
    story_path = PHASE12_DIR / "stratified_atlas_fdr/phase12_stratified_atlas_fdr_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 stratified atlas FDR audit: {row.dataset_key}",
                "status": row.stratified_atlas_fdr_call,
                "key_result": (
                    f"true mechanism q<=0.1={int(row.true_mechanism_q_0_1)}, "
                    f"true category q<=0.1={int(row.true_category_q_0_1)}, "
                    f"strict true mechanism q<=0.1={int(row.true_strict_mechanism_q_0_1)}, "
                    f"best true mechanism q={row.best_true_mechanism_q_lr}, "
                    f"best true category q={row.best_true_category_q_lr}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "mechanism/category-stratified all-null FDR sensitivity audit",
            }
        )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 stratified atlas FDR overall",
                "status": s.overall_stratified_atlas_fdr_call,
                "key_result": (
                    f"true mechanism q<=0.1={int(s.n_true_mechanism_stratified_q_0_1)}, "
                    f"true category q<=0.1={int(s.n_true_category_stratified_q_0_1)}, "
                    f"strict true mechanism q<=0.1={int(s.n_true_strict_mechanism_stratified_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "stratified FDR claim boundary",
            }
        )
    return rows


def matched_spatial_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv"
    story_path = PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 matched spatial atlas FDR diagnostic: {row.dataset_key}",
                "status": row.matched_spatial_audit_call,
                "key_result": (
                    f"top10 true={int(row.top10_true_lr)}, "
                    f"matched spatial true q<=0.1={int(row.all_true_lr_matched_spatial_q_0_1)}, "
                    f"matched all-null true q<=0.1={int(row.all_true_lr_matched_all_null_q_0_1)}, "
                    f"strict matched all-null true q<=0.1={int(row.all_true_lr_strict_matched_all_null_q_0_1)}, "
                    f"best true={row.best_true_lr} q={fmt_num(row.best_true_matched_all_null_q)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "post-hoc matched spatial-null diagnostic; not end-to-end whole-atlas FDR",
            }
        )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 matched spatial atlas FDR overall",
                "status": s.overall_matched_spatial_call,
                "key_result": (
                    f"true matched-spatial q<=0.1={int(s.n_true_matched_spatial_q_0_1)}, "
                    f"true matched all-null q<=0.1={int(s.n_true_matched_all_null_q_0_1)}, "
                    f"strict true matched all-null q<=0.1={int(s.n_true_strict_matched_all_null_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "matched spatial-null claim boundary",
            }
        )
    return rows


def guarded_matched_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_dataset.csv"
    story_path = PHASE12_DIR / "guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 guarded matched atlas FDR: {row.dataset_key}",
                "status": row.guarded_dataset_call,
                "key_result": (
                    f"guarded matched-all true={int(row.guarded_matched_all_null_n_true_supported)}, "
                    f"primary-control={int(row.guarded_matched_all_null_n_primary_control_supported)}, "
                    f"receptor-shared={int(row.guarded_matched_all_null_n_receptor_shared_supported)}, "
                    f"background={int(row.guarded_matched_all_null_n_background_supported)}, "
                    f"best true={row.guarded_matched_all_null_best_true_lr} "
                    f"q={fmt_num(row.guarded_matched_all_null_best_true_q)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "full-LR BH using receiver/LR-identity plus matched spatial-null guards; not pooled spatial-distance FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 guarded matched atlas FDR overall: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"supported datasets={int(row.n_supported_datasets)}/{int(row.n_datasets)}, "
                        f"true supported={int(row.total_true_supported)}, "
                        f"primary-control supported={int(row.total_primary_control_supported)}, "
                        f"receptor-shared supported={int(row.total_receptor_shared_supported)}; "
                        f"{row.claim_boundary}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": "guarded matched-null all-LR diagnostic; selected discovery support only",
                }
            )
    return rows


def range_weighted_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_dataset.csv"
    story_path = PHASE12_DIR / "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 range-weighted atlas FDR diagnostic: {row.dataset_key}",
                "status": row.range_weighted_call,
                "key_result": (
                    f"weighted q<=0.1 true={int(row.weighted_true_q_0_1)}, "
                    f"controls={int(row.weighted_control_q_0_1)}, "
                    f"strict true={int(row.strict_weighted_true_q_0_1)}, "
                    f"strict controls={int(row.strict_weighted_control_q_0_1)}, "
                    f"top10 true={int(row.top10_true_lr)}, top10 controls={int(row.top10_primary_control_decoy)}, "
                    f"best={row.best_true_lr} q={fmt_num(row.best_true_weighted_q)}, "
                    f"best strict={row.best_strict_true_lr} q={fmt_num(row.best_strict_true_weighted_q)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "label-independent weighted BH over range-matched all-null p-values; diagnostic, not replacement for pooled FDR",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 range-weighted atlas FDR story: {row.topic}",
                    "status": row.overall_range_weighted_call,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, supportive={int(row.n_supportive_datasets)}, "
                        f"strict-clean supportive={int(row.n_strict_clean_supportive_datasets)}, "
                        f"weighted true q<=0.1={int(row.n_weighted_true_q_0_1)}, "
                        f"weighted controls q<=0.1={int(row.n_weighted_control_q_0_1)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def variant_matched_spatial_fdr_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "variant_matched_spatial_fdr/phase12_variant_matched_spatial_method_summary.csv"
    dataset_path = PHASE12_DIR / "variant_matched_spatial_fdr/phase12_variant_matched_spatial_dataset_summary.csv"
    story_path = PHASE12_DIR / "variant_matched_spatial_fdr/phase12_variant_matched_spatial_story.csv"
    rows: list[dict[str, object]] = []
    if method_path.exists():
        method = pd.read_csv(method_path)
        for row in method.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 variant matched-spatial FDR diagnostic: {row.method}",
                    "status": row.method_call,
                    "key_result": (
                        f"matched true q<=0.1={int(row.matched_all_true_q_0_1)}, "
                        f"strict matched true q<=0.1={int(row.strict_matched_all_true_q_0_1)}, "
                        f"matched controls q<=0.1={int(row.matched_all_control_q_0_1)}, "
                        f"weighted true q<=0.1={int(row.weighted_true_q_0_1)}, "
                        f"weighted controls q<=0.1={int(row.weighted_control_q_0_1)}, "
                        f"best matched q={fmt_num(row.best_true_matched_all_q)}"
                    ),
                    "supporting_file": rel(method_path),
                    "claim_level": "post-hoc variant matched-spatial/range-weighted diagnostic; not broad pooled FDR closure",
                }
            )
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 variant matched-spatial dataset inventory",
                "status": "completed_variant_matched_inventory",
                "key_result": "; ".join(
                    f"{r.method}/{r.dataset_key}: {r.dataset_call}, matchedTrue={int(r.matched_all_true_q_0_1)}, "
                    f"strictTrue={int(r.strict_matched_all_true_q_0_1)}, controls={int(r.matched_all_control_q_0_1)}, "
                    f"weightedTrue={int(r.weighted_true_q_0_1)}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "dataset-level variant matched-spatial/range-weighted diagnostic",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 variant matched-spatial FDR story",
                "status": first["overall_call"],
                "key_result": (
                    f"methods={int(first.n_methods)}, best={first.best_method}, "
                    f"strict true={int(first.best_method_strict_matched_true_q_0_1)}, "
                    f"matched true={int(first.best_method_matched_true_q_0_1)}, "
                    f"weighted true={int(first.best_method_weighted_true_q_0_1)}, "
                    f"weighted controls={int(first.best_method_weighted_control_q_0_1)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def lr_standardized_null_fdr_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_method_summary.csv"
    dataset_path = PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_dataset_summary.csv"
    story_path = PHASE12_DIR / "lr_standardized_null_fdr/phase12_lr_standardized_null_fdr_story.csv"
    rows: list[dict[str, object]] = []
    if method_path.exists():
        method = pd.read_csv(method_path)
        for row in method.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 LR-standardized all-null FDR diagnostic: {row.method}",
                    "status": row.method_call,
                    "key_result": (
                        f"true q<=0.1={int(row.standardized_true_q_0_1)}, "
                        f"controls q<=0.1={int(row.standardized_control_q_0_1)}, "
                        f"strict true q<=0.1={int(row.strict_standardized_true_q_0_1)}, "
                        f"strict controls q<=0.1={int(row.strict_standardized_control_q_0_1)}, "
                        f"best true q={fmt_num(row.best_true_standardized_q)}"
                    ),
                    "supporting_file": rel(method_path),
                    "claim_level": "LR-specific null-baseline diagnostic; negative result, not broad FDR closure",
                }
            )
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 LR-standardized all-null dataset inventory",
                "status": "completed_negative_diagnostic",
                "key_result": "; ".join(
                    f"{r.method}/{r.dataset_key}: {r.dataset_call}, true={int(r.standardized_true_q_0_1)}, "
                    f"strictTrue={int(r.strict_standardized_true_q_0_1)}, controls={int(r.standardized_control_q_0_1)}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "dataset-level LR-standardized all-null FDR diagnostic",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 LR-standardized all-null FDR story",
                "status": first["overall_call"],
                "key_result": (
                    f"methods={int(first.n_methods)}, best={first.best_method}, "
                    f"strict true={int(first.best_method_strict_true_q_0_1)}, "
                    f"strict controls={int(first.best_method_strict_control_q_0_1)}, "
                    f"true={int(first.best_method_true_q_0_1)}, controls={int(first.best_method_control_q_0_1)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def null_margin_guarded_training_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_dataset_comparison.csv"
    )
    story_path = PHASE12_DIR / "null_margin_guarded_training_comparison/phase12_null_margin_guarded_training_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 null-margin guarded training: {row.dataset_key}",
                "status": row.dataset_decision,
                "key_result": (
                    f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
                    f"delta top10 true vs main={fmt_num(row.delta_top10_true_lr_vs_main)}, "
                    f"hard-negative best beats={int(row.hard_negative_best_comparison_beats)}, "
                    f"true q<=0.1={int(row.all_true_lr_conservative_fdr_0_1)}, "
                    f"best true={row.best_true_lr} q={fmt_num(row.best_true_lr_fdr)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "stronger null-margin training comparison; exact-score FDR still required",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 null-margin guarded training story",
                "status": first["overall_call"],
                "key_result": (
                    f"top10 improved={int(first.n_top10_true_improved_vs_main)}/{int(first.n_datasets)}, "
                    f"no added risk={int(first.n_no_added_top10_risk_vs_main)}/{int(first.n_datasets)}, "
                    f"hard-neg clean={int(first.n_hard_negative_best_beat_clean)}/{int(first.n_datasets)}, "
                    f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def fdr_margin_gap_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_method.csv"
    dataset_path = PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_dataset.csv"
    story_path = PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_story.csv"
    rows: list[dict[str, object]] = []
    if method_path.exists():
        method = pd.read_csv(method_path)
        for row in method.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 FDR margin gap: {row.variant}",
                    "status": row.method_gap_call,
                    "key_result": (
                        f"mean top10 true={fmt_num(row.mean_top10_true_lr)}, "
                        f"min q-gap={fmt_num(row.min_q_gap_to_0_1)}x, "
                        f"median q-gap={fmt_num(row.median_q_gap_to_0_1)}x, "
                        f"dominant nulls={row.dominant_null_counts}"
                    ),
                    "supporting_file": rel(method_path),
                    "claim_level": "quantifies distance to broad all-null FDR; not FDR closure",
                }
            )
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 FDR margin gap dataset inventory",
                "status": "completed_gap_inventory",
                "key_result": "; ".join(
                    f"{r.variant}/{r.dataset_key}: {r.dataset_gap_call}, best={r.best_true_by_q}, "
                    f"q-gap={fmt_num(r.best_true_q_gap_to_0_1)}x, null={r.best_true_dominant_p_null}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "dataset-level q-gap and dominant-null inventory",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 FDR margin gap story",
                "status": first["overall_call"],
                "key_result": (
                    f"variants={int(first.n_variants)}, best={first.best_variant_by_min_q_gap}, "
                    f"min q-gap={fmt_num(first.best_variant_min_q_gap_to_0_1)}x, "
                    f"median q-gap={fmt_num(first.best_variant_median_q_gap_to_0_1)}x, "
                    f"dominant={first.best_variant_dominant_null_counts}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def spatial_specificity_comparison_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "spatial_specificity_comparison/phase12_spatial_specificity_dataset_comparison.csv"
    story_path = PHASE12_DIR / "spatial_specificity_comparison/phase12_spatial_specificity_story.csv"
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 spatial-specificity calibration comparison",
                "status": "completed_not_fdr_closure",
                "key_result": "; ".join(
                    f"{r.dataset_key}: {r.dataset_decision}, "
                    f"top10 true {int(r.top10_true_lr_main)}->{int(r.top10_true_lr_spatial_specificity)}, "
                    f"risk {fmt_num(r.top10_negative_or_risk_fraction_main)}->{fmt_num(r.top10_negative_or_risk_fraction_spatial_specificity)}, "
                    f"best true q {fmt_num(r.best_true_lr_fdr_main)}->{fmt_num(r.best_true_lr_fdr_spatial_specificity)}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "post-hoc spatial-specificity diagnostic; shrinks selected q gaps but not broad FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 spatial-specificity calibration story",
                "status": first["overall_call"],
                "key_result": (
                    f"true q<=0.1={int(first.n_true_q_0_1)}, "
                    f"best true q improved vs main={int(first.n_best_true_q_improved_vs_main)}/{int(first.n_datasets)}, "
                    f"top10 true improved={int(first.n_top10_true_improved_vs_main)}/{int(first.n_datasets)}, "
                    f"best q={fmt_num(first.best_true_q_min)} ({fmt_num(first.best_true_q_gap_to_0_1)}x over 0.1)"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def guarded_spatial_specificity_comparison_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_dataset_comparison.csv"
    )
    story_path = PHASE12_DIR / "guarded_spatial_specificity_comparison/phase12_guarded_spatial_specificity_story.csv"
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 guarded spatial-specificity comparison",
                "status": "completed_near_fdr_not_closed",
                "key_result": "; ".join(
                    f"{r.dataset_key}: {r.dataset_decision}, "
                    f"q main/spatial/guarded={fmt_num(r.best_true_lr_fdr_main)}/"
                    f"{fmt_num(r.best_true_lr_fdr_spatial)}/{fmt_num(r.best_true_lr_fdr_guarded)}, "
                    f"risk main/spatial/guarded={fmt_num(r.top10_negative_or_risk_fraction_main)}/"
                    f"{fmt_num(r.top10_negative_or_risk_fraction_spatial)}/"
                    f"{fmt_num(r.top10_negative_or_risk_fraction_guarded)}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "post-hoc guarded spatial-specificity diagnostic; near-threshold in post-Xenium but not broad FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 guarded spatial-specificity story",
                "status": first["overall_call"],
                "key_result": (
                    f"true q<=0.1={int(first.n_true_q_0_1)}, "
                    f"best true q improved vs main={int(first.n_best_true_q_improved_vs_main)}/{int(first.n_datasets)}, "
                    f"risk improved vs spatial={int(first.n_top10_risk_improved_vs_spatial_specificity)}/{int(first.n_datasets)}, "
                    f"best q={fmt_num(first.best_true_q_min)} ({fmt_num(first.best_true_q_gap_to_0_1)}x over 0.1)"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def spatial_null_bottleneck_map_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_dataset.csv"
    story_path = PHASE12_DIR / "spatial_null_bottleneck_map/phase12_spatial_null_bottleneck_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 spatial-null bottleneck map: {row.dataset_key}",
                "status": row.dataset_spatial_null_call,
                "key_result": (
                    f"true pairing/identity supported={int(row.true_pairing_identity_supported)}, "
                    f"pooled spatial true q<=0.1={int(row.true_pooled_spatial_supported)}, "
                    f"spatial-blocked true={int(row.true_spatial_specific_blocker)}, "
                    f"matched spatial true={int(row.true_matched_spatial_supported)}, "
                    f"top20 broad-other={int(row.top20_category_balanced_other)}, "
                    f"next={row.dominant_calibration_target}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "diagnostic map of the remaining spatial-distance null bottleneck; not FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 spatial-null bottleneck story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, true pooled spatial q<=0.1={int(row.n_true_pooled_spatial_supported)}, "
                        f"spatial-blocked true={int(row.n_true_spatial_specific_blocker)}, "
                        f"matched-spatial true={int(row.n_true_matched_spatial_supported)}, "
                        f"top20 broad-other LR={int(row.n_top20_broad_other_lr)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def background_suppression_attempt_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "background_suppression_attempt/phase12_background_suppression_dataset_comparison.csv"
    story_path = PHASE12_DIR / "background_suppression_attempt/phase12_background_suppression_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 background-suppression attempt: {row.dataset_key}",
                "status": row.background_adoption_decision,
                "key_result": (
                    f"top10 true delta={fmt_num(row.background_top10_delta_true_lr)}, "
                    f"top10 risk delta={fmt_num(row.background_top10_delta_risk)}, "
                    f"top20 broad-other delta={fmt_num(row.background_top20_delta_broad_other)}, "
                    f"top100 control-FDR delta={fmt_num(row.background_top100_delta_control_fdr)}, "
                    f"best variant={row.best_variant_for_dataset}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "exploratory background-suppressed ranking audit; not global FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 background-suppression attempt story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, adopt candidates={int(row.n_adopt_candidate)}, "
                        f"top10 promising={int(row.n_top10_promising)}, hold-out={int(row.n_hold_out)}, "
                        f"background-best={int(row.n_background_best_by_score)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def background_suppressed_permutation_comparison_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / (
        "background_suppressed_permutation_comparison/"
        "phase12_background_suppressed_permutation_dataset_comparison.csv"
    )
    story_path = PHASE12_DIR / (
        "background_suppressed_permutation_comparison/"
        "phase12_background_suppressed_permutation_story.csv"
    )
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 background-suppressed exact-score permutation: {row.dataset_key}",
                "status": row.comparison_call,
                "key_result": (
                    f"top10 true delta={fmt_num(row.delta_top10_true_lr)}, "
                    f"top10 risk delta={fmt_num(row.delta_top10_risk_fraction)}, "
                    f"best true q delta={fmt_num(row.delta_best_true_lr_fdr)}, "
                    f"background all-LR q<=0.1={int(row.background_all_lr_q_0_1)}, "
                    f"background true-LR q<=0.1={int(row.background_true_lr_q_0_1)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "exact-score post-hoc permutation audit for background-suppressed ranking; not end-to-end FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 background-suppressed exact-score permutation story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, FDR-closed datasets={int(row.n_fdr_closed_datasets)}, "
                        f"top10-improved datasets={int(row.n_top10_improved_without_risk)}, "
                        f"best-true-q-improved datasets={int(row.n_best_true_q_improved)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def spatial_null_aware_score_permutation_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_dataset.csv"
    story_path = PHASE12_DIR / "spatial_null_aware_score_permutation_audit/phase12_spatial_null_aware_score_permutation_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 spatial-null-aware exact-score prototype: {row.dataset_key}",
                "status": row.dataset_call,
                "key_result": (
                    f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
                    f"all q<=0.1={int(row.all_lr_q_0_1)}, true q<=0.1={int(row.true_lr_q_0_1)}, "
                    f"risk q<=0.1={int(row.negative_or_risk_q_0_1)}, best true={row.best_true_lr} "
                    f"q={fmt_num(row.best_true_q)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "exploratory exact-score null audit for next-model scoring; not a retrained model or FDR closure",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 spatial-null-aware exact-score prototype story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"datasets with true q<=0.1={int(row.n_datasets_with_true_q_0_1)}/{int(row.n_datasets)}, "
                        f"total true q<=0.1={int(row.total_true_lr_q_0_1)}, "
                        f"total risk q<=0.1={int(row.total_negative_or_risk_q_0_1)}; "
                        f"{row.claim_boundary}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": "next-model prototype audit; ranking improvement without whole-atlas FDR closure",
                }
            )
    return rows


def spatial_null_training_smoke_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "spatial_null_training_smoke/phase12_spatial_null_training_smoke_summary.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    rows = []
    for row in table.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 spatial-null contrastive training smoke: {row.dataset_key}",
                "status": row.status,
                "key_result": (
                    f"spatial-null pairs={int(row.n_spatial_null_contrastive_train)}, "
                    f"context pairs={int(row.n_context_contrastive_train)}, "
                    f"hard-negative pairs={int(row.n_hard_negative_train)}, "
                    f"runtime={fmt_num(row.runtime_seconds)}s"
                ),
                "supporting_file": rel(path),
                "claim_level": "next-model training path smoke; proves loss path runs, not FDR improvement",
            }
        )
    return rows


def spatial_null_trained_atlas_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_dataset_summary.csv"
    comparison_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_comparison_to_main.csv"
    status_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_trained/phase12_spatial_null_trained_status.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 spatial-null-trained expanded atlas: {row.dataset_key}",
                "status": row.calibrated_audit_call,
                "key_result": (
                    f"top10 true={int(row.top10_true_lr)}, top10 risk={fmt_num(row.top10_negative_or_risk_fraction)}, "
                    f"top50 true={int(row.top50_true_lr)}, hard-neg beats={int(row.hard_negative_best_comparison_beats)}, "
                    f"runtime={fmt_num(row.runtime_seconds)}s"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "full training variant ranking/control audit; not exact permutation FDR",
            }
        )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 spatial-null-trained vs main: {row.dataset_key}",
                    "status": row.comparison_call,
                    "key_result": (
                        f"delta top10 true={int(row.delta_top10_true_lr)}, "
                        f"delta top10 risk={fmt_num(row.delta_top10_risk_fraction)}, "
                        f"delta top50 true={int(row.delta_top50_true_lr)}, "
                        f"delta hard-neg beats={int(row.delta_hard_negative_beats)}"
                    ),
                    "supporting_file": rel(comparison_path),
                    "claim_level": "model-variant comparison; mixed tradeoff, not replacement baseline",
                }
            )
    if status_path.exists():
        status = pd.read_csv(status_path)
        rows.append(
            {
                "phase12_component": "Goal 1 spatial-null-trained expanded atlas run status",
                "status": "completed" if status["status"].astype(str).eq("completed").all() else "partial_or_failed",
                "key_result": (
                    f"datasets completed={int(status['status'].astype(str).eq('completed').sum())}/{len(status)}, "
                    f"total spatial-null train pairs={int(pd.to_numeric(status['n_spatial_null_contrastive_train'], errors='coerce').fillna(0).sum())}"
                ),
                "supporting_file": rel(status_path),
                "claim_level": "execution status for spatial-null-trained full-atlas variant",
            }
        )
    return rows


def spatial_null_anchor_preserving_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_dataset_summary.csv"
    comparison_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_comparison_to_main.csv"
    status_path = PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_status.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 anchor-preserving spatial-null atlas: {row.dataset_key}",
                "status": row.calibrated_audit_call,
                "key_result": (
                    f"top10 true={int(row.top10_true_lr)}, top10 risk={fmt_num(row.top10_negative_or_risk_fraction)}, "
                    f"top50 true={int(row.top50_true_lr)}, hard-neg beats={int(row.hard_negative_best_comparison_beats)}, "
                    f"runtime={fmt_num(row.runtime_seconds)}s"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "anchor-preserving full training variant ranking/control audit; not exact permutation FDR",
            }
        )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-preserving vs main: {row.dataset_key}",
                    "status": row.comparison_call,
                    "key_result": (
                        f"delta top10 true={int(row.delta_top10_true_lr_vs_main)}, "
                        f"delta top10 risk={fmt_num(row.delta_top10_risk_vs_main)}, "
                        f"delta top50 true={int(row.delta_top50_true_lr_vs_main)}, "
                        f"delta hard-neg beats={int(row.delta_hard_negative_beats_vs_main)}"
                    ),
                    "supporting_file": rel(comparison_path),
                    "claim_level": "anchor-preserving model-variant comparison; candidate direction with CRC caveat",
                }
            )
    if status_path.exists():
        status = pd.read_csv(status_path)
        rows.append(
            {
                "phase12_component": "Goal 1 anchor-preserving spatial-null atlas run status",
                "status": "completed" if status["status"].astype(str).eq("completed").all() else "partial_or_failed",
                "key_result": (
                    f"datasets completed={int(status['status'].astype(str).eq('completed').sum())}/{len(status)}, "
                    f"total spatial-null train pairs={int(pd.to_numeric(status['n_spatial_null_contrastive_train'], errors='coerce').fillna(0).sum())}"
                ),
                "supporting_file": rel(status_path),
                "claim_level": "execution status for anchor-preserving spatial-null variant",
            }
        )
    return rows


def spatial_null_anchor_preserving_permutation_rows() -> list[dict[str, object]]:
    comparison_path = (
        PHASE12_DIR
        / "spatial_null_anchor_preserving_permutation_comparison/"
        / "phase12_spatial_null_anchor_preserving_permutation_dataset_comparison.csv"
    )
    story_path = (
        PHASE12_DIR
        / "spatial_null_anchor_preserving_permutation_comparison/"
        / "phase12_spatial_null_anchor_preserving_permutation_story.csv"
    )
    if not comparison_path.exists():
        return []
    comparison = pd.read_csv(comparison_path)
    rows: list[dict[str, object]] = []
    for row in comparison.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 anchor-preserving exact-score permutation: {row.dataset_key}",
                "status": row.comparison_call,
                "key_result": (
                    f"delta top10 true={int(row.delta_top10_true_lr)}, "
                    f"delta top10 risk={fmt_num(row.delta_top10_risk_fraction)}, "
                    f"delta best true q={fmt_num(row.delta_best_true_lr_fdr)}, "
                    f"anchor true q<=0.1={int(row.anchor_true_lr_q_0_1)}"
                ),
                "supporting_file": rel(comparison_path),
                "claim_level": "exact-score post-hoc permutation comparison; top-k support only, broad whole-atlas FDR not closed",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 anchor-preserving exact-score permutation story",
                "status": first["overall_call"],
                "key_result": (
                    f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}/{int(first.n_datasets)}, "
                    f"top10 improved={int(first.n_top10_improved_without_risk)}, "
                    f"top10 preserved={int(first.n_top10_preserved_without_risk)}, "
                    f"best true q improved={int(first.n_best_true_q_improved)}, "
                    f"CRC weaker={int(first.n_crc_weaker)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def anchor_background_suppressed_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "atlas_model_expanded_anchor_background_suppressed/phase12_anchor_background_suppressed_dataset_summary.csv"
    comparison_path = (
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_comparison/"
        / "phase12_anchor_background_permutation_dataset_comparison.csv"
    )
    story_path = (
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_comparison/"
        / "phase12_anchor_background_permutation_story.csv"
    )
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background calibration: {row.dataset_key}",
                    "status": row.background_suppressed_call,
                    "key_result": (
                        f"top10 true={int(row.top10_true_lr)}, "
                        f"top10 high-expression negatives={int(row.top10_high_expression_negative)}, "
                        f"top10 broad-other={int(row.top10_broad_other_lr)}, "
                        f"top10 control FDR={fmt_num(row.top10_control_panel_empirical_fdr)}, "
                        f"top100 true={int(row.top100_true_lr)}"
                    ),
                    "supporting_file": rel(dataset_path),
                    "claim_level": "post-hoc anchor-preserving plus background suppression calibration; ranking/control diagnostic",
                }
            )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background exact permutation: {row.dataset_key}",
                    "status": row.comparison_call,
                    "key_result": (
                        f"delta top10 true vs base={int(row.delta_anchor_background_vs_base_top10_true_lr)}, "
                        f"delta risk vs base={fmt_num(row.delta_anchor_background_vs_base_top10_risk)}, "
                        f"delta best true q vs base={fmt_num(row.delta_anchor_background_vs_base_best_true_fdr)}, "
                        f"true q<=0.1={int(row.anchor_background_all_true_lr_conservative_fdr_0_1)}"
                    ),
                    "supporting_file": rel(comparison_path),
                    "claim_level": "exact-score post-hoc permutation comparison; top-k improvement but broad FDR not closed",
                }
            )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 anchor-background exact permutation story",
                "status": first["overall_call"],
                "key_result": (
                    f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}/{int(first.n_datasets)}, "
                    f"top10 improved vs base={int(first.n_top10_improved_without_risk_vs_base)}, "
                    f"best true q improved={int(first.n_best_true_q_improved_vs_base)}, "
                    f"CRC recovered={int(first.n_crc_top10_recovered_vs_base)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def anchor_background_guarded_training_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_guarded_training/"
        / "phase12_anchor_background_guarded_training_dataset_summary.csv"
    )
    comparison_path = (
        PHASE12_DIR
        / "anchor_background_guarded_training_comparison/"
        / "phase12_anchor_background_guarded_training_dataset_comparison.csv"
    )
    story_path = (
        PHASE12_DIR
        / "anchor_background_guarded_training_comparison/"
        / "phase12_anchor_background_guarded_training_story.csv"
    )
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background guarded training: {row.dataset_key}",
                    "status": row.anchor_background_guarded_training_call,
                    "key_result": (
                        f"top10 true={int(row.top10_true_lr)}, "
                        f"top10 risk={int(row.top10_negative_or_risk)}, "
                        f"top20 broad-other={int(row.top20_broad_other_lr)}, "
                        f"top100 true={int(row.top100_true_lr)}, "
                        f"hard-neg beats={int(row.hard_negative_best_comparison_beats)}"
                    ),
                    "supporting_file": rel(dataset_path),
                    "claim_level": "training-integrated anchor/background guard; ranking/control diagnostic before FDR audit",
                }
            )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background guarded exact permutation: {row.dataset_key}",
                    "status": row.comparison_call,
                    "key_result": (
                        f"delta top10 true vs base={int(row.delta_train_vs_base_top10_true_lr)}, "
                        f"delta risk vs base={fmt_num(row.delta_train_vs_base_top10_risk)}, "
                        f"delta risk vs posthoc={fmt_num(row.delta_train_vs_anchor_bg_top10_risk)}, "
                        f"true q<=0.1={int(row.train_all_true_lr_conservative_fdr_0_1)}, "
                        f"best true q={fmt_num(row.train_best_true_lr_fdr)}"
                    ),
                    "supporting_file": rel(comparison_path),
                    "claim_level": "exact-score post-hoc permutation comparison for training-integrated variant; broad FDR still not closed",
                }
            )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 anchor-background guarded training story",
                "status": first["overall_call"],
                "key_result": (
                    f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}/{int(first.n_datasets)}, "
                    f"top10 improved vs base={int(first.n_top10_improved_without_risk_vs_base)}, "
                    f"worse risk than posthoc={int(first.n_worse_top10_risk_than_posthoc_anchor_background)}, "
                    f"hard-neg clean={int(first.n_hard_negative_best_beat_clean)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def anchor_background_range_guarded_training_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_dataset_summary.csv"
    )
    comparison_path = (
        PHASE12_DIR
        / "anchor_background_range_guarded_training_comparison/"
        / "phase12_anchor_background_range_guarded_training_dataset_comparison.csv"
    )
    story_path = (
        PHASE12_DIR
        / "anchor_background_range_guarded_training_comparison/"
        / "phase12_anchor_background_range_guarded_training_story.csv"
    )
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background range-guarded training: {row.dataset_key}",
                    "status": row.anchor_background_range_guarded_training_call,
                    "key_result": (
                        f"top10 true={int(row.top10_true_lr)}, "
                        f"top10 risk={int(row.top10_negative_or_risk)}, "
                        f"top20 broad-other={int(row.top20_broad_other_lr)}, "
                        f"top100 true={int(row.top100_true_lr)}, "
                        f"hard-neg beats={int(row.hard_negative_best_comparison_beats)}"
                    ),
                    "supporting_file": rel(dataset_path),
                    "claim_level": "training-integrated low-weight range-violation spatial-null guard; ranking/control diagnostic",
                }
            )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 anchor-background range-guarded exact permutation: {row.dataset_key}",
                    "status": row.comparison_call,
                    "key_result": (
                        f"delta top10 true vs base={int(row.delta_train_vs_base_top10_true_lr)}, "
                        f"delta risk vs base={fmt_num(row.delta_train_vs_base_top10_risk)}, "
                        f"delta best true q vs base={fmt_num(row.delta_train_vs_base_best_true_fdr)}, "
                        f"true q<=0.1={int(row.train_all_true_lr_conservative_fdr_0_1)}, "
                        f"best true q={fmt_num(row.train_best_true_lr_fdr)}"
                    ),
                    "supporting_file": rel(comparison_path),
                    "claim_level": "exact-score post-hoc permutation comparison for range-guarded variant; broad FDR still not closed",
                }
            )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 anchor-background range-guarded training story",
                "status": first["overall_call"],
                "key_result": (
                    f"FDR-closed datasets={int(first.n_fdr_closed_datasets)}/{int(first.n_datasets)}, "
                    f"top10 improved vs base={int(first.n_top10_improved_without_risk_vs_base)}, "
                    f"worse risk than posthoc={int(first.n_worse_top10_risk_than_posthoc_anchor_background)}, "
                    f"hard-neg clean={int(first.n_hard_negative_best_beat_clean)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def integrated_guarded_spatial_training_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "atlas_model_expanded_integrated_guarded_spatial_training/"
        / "phase12_integrated_guarded_spatial_training_dataset_summary.csv"
    )
    perm_path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv"
    )
    story_path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_story.csv"
    )
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 integrated guarded spatial training: {row.dataset_key}",
                    "status": row.integrated_guarded_spatial_training_call,
                    "key_result": (
                        f"top10 true={int(row.top10_true_lr)}, "
                        f"top10 risk={int(row.top10_negative_or_risk)}, "
                        f"top50 true={int(row.top50_true_lr)}, "
                        f"best true={row.best_true_lr}, "
                        f"hard-neg beats={int(getattr(row, 'hard_negative_best_comparison_beats', 0))}"
                    ),
                    "supporting_file": rel(dataset_path),
                    "claim_level": "candidate integrated training/ranking variant; ranking/control diagnostic before null audit",
                }
            )
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        for row in perm.itertuples(index=False):
            call = getattr(
                row,
                "integrated_guarded_spatial_training_permutation_call",
                getattr(row, "spatial_specificity_permutation_call", ""),
            )
            rows.append(
                {
                    "phase12_component": f"Goal 1 integrated guarded spatial exact permutation: {row.dataset_key}",
                    "status": call,
                    "key_result": (
                        f"top10 true={int(row.top10_true_lr)}, "
                        f"top10 risk={fmt_num(row.top10_negative_or_risk_fraction)}, "
                        f"true q<=0.1={int(row.all_true_lr_conservative_fdr_0_1)}, "
                        f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
                        f"best true={row.best_true_lr} q={fmt_num(row.best_true_lr_fdr)}, "
                        f"best spatial q={fmt_num(row.best_true_spatial_fdr)}"
                    ),
                    "supporting_file": rel(perm_path),
                    "claim_level": "exact-score audit for candidate integrated guarded spatial variant; not end-to-end FDR and not reporting baseline",
                }
            )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 integrated guarded spatial training story",
                "status": first["overall_call"],
                "key_result": (
                    f"true q<=0.1={int(first.total_true_q_0_1)}, "
                    f"clean top10 datasets={int(first.n_clean_top10_risk)}, "
                    f"best true q={fmt_num(first.best_true_q_min)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def integrated_guarded_spatial_highres_stability_rows() -> list[dict[str, object]]:
    dataset_path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_training_highres_stability/"
        / "phase12_integrated_guarded_spatial_highres_stability_dataset.csv"
    )
    story_path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_training_highres_stability/"
        / "phase12_integrated_guarded_spatial_highres_stability_story.csv"
    )
    rows: list[dict[str, object]] = []
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 integrated guarded spatial highres stability: {row.dataset_key}",
                    "status": row.q_0_1_stability_call,
                    "key_result": (
                        f"best true q80={fmt_num(row.best_true_lr_fdr_80)} -> "
                        f"q240={fmt_num(row.best_true_lr_fdr_240)}, "
                        f"true q<=0.1 highres={int(row.all_true_lr_conservative_fdr_0_1_240)}, "
                        f"top10 risk highres={fmt_num(row.top10_negative_or_risk_fraction_240)}, "
                        f"use={row.manuscript_use}"
                    ),
                    "supporting_file": rel(dataset_path),
                    "claim_level": "higher-permutation exact-score stability check; not end-to-end FDR",
                }
            )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 integrated guarded spatial highres stability story",
                "status": first["overall_call"],
                "key_result": (
                    f"stable q<=0.1 hits={int(first.n_stable_q_0_1_hits)}, "
                    f"highres true q<=0.1 total={int(first.highres_true_q_0_1_total)}, "
                    f"best={first.best_highres_true} q={fmt_num(first.best_highres_true_q)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def null_component_diagnostic_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "null_component_diagnostic/phase12_null_component_method_summary.csv"
    dataset_path = PHASE12_DIR / "null_component_diagnostic/phase12_null_component_dataset_summary.csv"
    story_path = PHASE12_DIR / "null_component_diagnostic/phase12_null_component_story.csv"
    rows: list[dict[str, object]] = []
    if method_path.exists():
        method = pd.read_csv(method_path)
        for row in method.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 null-component diagnostic: {row.method}",
                    "status": row.method_component_call,
                    "key_result": (
                        f"mean top10 true={fmt_num(row.mean_top10_true_lr)}, "
                        f"broad true q<=0.1={int(row.sum_true_broad_fdr_supported)}, "
                        f"non-spatial true support={int(row.sum_true_non_spatial_supported)}, "
                        f"spatial true support={int(row.sum_true_spatial_supported)}, "
                        f"spatial-blocked true={int(row.sum_true_spatial_global_blocked)}, "
                        f"best spatial q={fmt_num(row.best_true_spatial_q)}"
                    ),
                    "supporting_file": rel(method_path),
                    "claim_level": "component-level null diagnostic; not a replacement broad FDR procedure",
                }
            )
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        rows.append(
            {
                "phase12_component": "Goal 1 null-component dataset inventory",
                "status": "completed_component_inventory",
                "key_result": "; ".join(
                    f"{r.method}/{r.dataset_key}: {r.dataset_component_call}, nonSpatial={int(r.true_non_spatial_supported)}, spatialBlocked={int(r.true_spatial_global_blocked)}"
                    for r in dataset.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "dataset-level decomposition of receiver, LR-identity and spatial-distance null support",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        first = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 null-component diagnostic story",
                "status": first["overall_call"],
                "key_result": (
                    f"methods={int(first.n_methods)}, rows={int(first.n_dataset_method_rows)}, "
                    f"best non-spatial method={first.best_non_spatial_method}, "
                    f"best non-spatial true supported={int(first.best_non_spatial_true_supported)}, "
                    f"best spatial-q method={first.best_spatial_q_method}, "
                    f"best spatial q={fmt_num(first.best_true_spatial_q)}, "
                    f"broad true supported total={int(first.broad_fdr_supported_true_total)}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": first["claim_boundary"],
            }
        )
    return rows


def goal1_model_variant_scoreboard_rows() -> list[dict[str, object]]:
    variant_path = PHASE12_DIR / "goal1_model_variant_scoreboard/phase12_goal1_model_variant_summary.csv"
    story_path = PHASE12_DIR / "goal1_model_variant_scoreboard/phase12_goal1_model_variant_story.csv"
    if not variant_path.exists():
        return []
    variant = pd.read_csv(variant_path)
    rows: list[dict[str, object]] = []
    for row in variant.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 model-variant scoreboard: {row.variant}",
                "status": row.variant_decision,
                "key_result": (
                    f"datasets={int(row.n_datasets)}, mean top10 true={fmt_num(row.mean_top10_true_lr)}, "
                    f"true q<=0.1={int(row.total_true_lr_q_0_1)}, all-LR q<=0.1={int(row.total_all_lr_q_0_1)}, "
                    f"next-model candidates={int(row.n_next_model_candidate)}"
                ),
                "supporting_file": rel(variant_path),
                "claim_level": row.claim_boundary,
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 model-variant scoreboard story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"variants={int(row.n_variants)}, true q<=0.1={int(row.total_true_lr_q_0_1)}, "
                        f"next-model candidate components={int(row.n_next_model_candidate_components)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def range_null_calibration_attempt_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "range_null_calibration_attempt/phase12_range_null_calibration_dataset_comparison.csv"
    story_path = PHASE12_DIR / "range_null_calibration_attempt/phase12_range_null_calibration_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows: list[dict[str, object]] = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 range-null calibration attempt: {row.dataset_key}",
                "status": row.adoption_decision,
                "key_result": (
                    f"top10 true delta={fmt_num(row.delta_true_lr)}, "
                    f"top10 risk delta={fmt_num(row.delta_negative_or_risk)}, "
                    f"top100 control-FDR delta={fmt_num(row.top100_delta_control_panel_empirical_fdr)}; "
                    f"{row.decision_reason}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "exploratory model-variant audit; not adopted as the main Phase 12 ranking",
            }
        )
    if story_path.exists():
        story = pd.read_csv(story_path)
        for row in story.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 range-null calibration attempt story: {row.topic}",
                    "status": row.overall_call,
                    "key_result": (
                        f"datasets={int(row.n_datasets)}, adopt candidates={int(row.n_adopt_candidate)}, "
                        f"hold-out={int(row.n_hold_out)}, top10 true-gain datasets={int(row.n_top10_true_gain)}, "
                        f"top100 control-FDR-worse datasets={int(row.n_top100_control_fdr_worse)}"
                    ),
                    "supporting_file": rel(story_path),
                    "claim_level": row.claim_boundary,
                }
            )
    return rows


def discovery_eligible_atlas_fdr_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv"
    story_path = PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 discovery-eligible atlas FDR audit: {row.dataset_key}",
                "status": row.discovery_eligible_call,
                "key_result": (
                    f"eligible={int(row.n_eligible_lr)}/{int(row.n_total_lr)}, "
                    f"top10 true={int(row.top10_true_lr)}, top10 controls={int(row.top10_primary_control_decoy)}, "
                    f"matched q<=0.1 true={int(row.eligible_matched_true_q_0_1)}, "
                    f"controls={int(row.eligible_matched_control_q_0_1)}, "
                    f"background={int(row.eligible_matched_background_q_0_1)}, "
                    f"best true={row.best_true_lr} q={fmt_num(row.best_true_eligible_matched_q)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "pre-defined functionally annotated/expression-supported LR universe; not broad all-covered-LR FDR",
            }
        )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 discovery-eligible atlas FDR overall",
                "status": s.overall_discovery_eligible_call,
                "key_result": (
                    f"eligible pooled true q<=0.1={int(s.n_eligible_pooled_true_q_0_1)}, "
                    f"eligible matched true q<=0.1={int(s.n_eligible_matched_true_q_0_1)}, "
                    f"strict matched true q<=0.1={int(s.n_eligible_strict_matched_true_q_0_1)}, "
                    f"matched controls q<=0.1={int(s.n_eligible_matched_control_q_0_1)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "discovery-eligible FDR claim boundary",
            }
        )
    return rows


def discovery_eligible_method_comparison_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_summary.csv"
    dataset_path = PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_dataset.csv"
    story_path = PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv"
    comparison_path = PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_comparison.csv"
    if not method_path.exists():
        return []
    method = pd.read_csv(method_path)
    dataset = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    rows = []
    rows.append(
        {
            "phase12_component": "Goal 1 discovery-eligible method comparison",
            "status": "completed_eligible_universe_adapter_comparison",
            "key_result": "; ".join(
                f"{r.method_label}: top10true={fmt_num(r.mean_top10_true_lr)}, controls={fmt_num(r.mean_top10_primary_control_decoy)}, clean={int(r.n_clean_top10_datasets)}/{int(r.n_datasets)}, call={r.method_call}"
                for r in method.itertuples(index=False)
            ),
            "supporting_file": rel(method_path),
            "claim_level": "same discovery-eligible LR universe adapter-light ranking/contamination comparison",
        }
    )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 1 discovery-eligible method comparison overall",
                "status": s.overall_eligible_method_call,
                "key_result": (
                    f"SpatialSelf pairwise wins={int(s.spatialself_pairwise_wins)}, "
                    f"baseline wins={int(s.baseline_pairwise_wins)}, "
                    f"SpatialSelf clean top10 datasets={int(s.spatialself_clean_top10_datasets)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "eligible-universe method comparison boundary",
            }
        )
    if not dataset.empty:
        ss = dataset[dataset["method"].astype(str).eq("spatialself_calibrated")]
        rows.append(
            {
                "phase12_component": "Goal 1 discovery-eligible SpatialSelf dataset details",
                "status": "completed_spatialself_eligible_dataset_audit",
                "key_result": "; ".join(
                    f"{r.dataset_key}: top10 true={int(r.top10_true_lr)}, controls={int(r.top10_primary_control_decoy)}, highExpr={int(r.top10_high_expression_control)}, call={r.dataset_call}"
                    for r in ss.itertuples(index=False)
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "SpatialSelf dataset details within eligible LR universe",
            }
        )
    if not comparison.empty:
        wins = comparison["comparison_call"].astype(str).str.startswith("spatialself").sum()
        rows.append(
            {
                "phase12_component": "Goal 1 discovery-eligible SpatialSelf-vs-baseline pairs",
                "status": "spatialself_pairwise_advantage_in_eligible_universe",
                "key_result": f"SpatialSelf-favoring pairwise comparisons={int(wins)}/{len(comparison)}; calls={comparison['comparison_call'].value_counts().to_dict()}",
                "supporting_file": rel(comparison_path),
                "claim_level": "pairwise adapter-light comparison; not fully tuned official method superiority",
            }
        )
    return rows


def receptor_shared_atlas_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv"
    pair_path = PHASE12_DIR / "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_positive_pair_audit.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    pair = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        sub = pair[pair["dataset_key"].eq(row.dataset_key)] if not pair.empty else pd.DataFrame()
        leaking = sub[pd.to_numeric(sub.get("n_receptor_shared_decoys_beating_positive", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)] if not sub.empty else pd.DataFrame()
        leak_text = (
            "; ".join(
                f"{r.positive_lr_pair} beaten by {r.best_decoy_lr_pair}#{int(r.best_decoy_rank)}"
                for r in leaking.sort_values("positive_rank").head(4).itertuples(index=False)
            )
            if len(leaking)
            else "no positive anchor beaten"
        )
        rows.append(
            {
                "phase12_component": f"Goal 1 receptor-shared atlas audit: {row.dataset_key}",
                "status": row.receptor_shared_atlas_call,
                "key_result": (
                    f"receptor-shared decoys={int(row.n_receptor_shared_decoy)}, "
                    f"top10={int(row.top10_receptor_shared_decoy)}, top50={int(row.top50_receptor_shared_decoy)}, "
                    f"positive anchors beaten={int(row.n_positive_anchors_beaten_by_receptor_shared_decoy)}/{int(row.n_positive_anchor)}, "
                    f"pairwise beats={int(row.total_receptor_shared_decoys_beating_positive)}; {leak_text}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "whole-atlas receptor-shared control contamination audit; not native discovery FDR",
            }
        )
    return rows


def downstream_rows() -> list[dict[str, object]]:
    rows = []
    expanded_summary_path = PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv"
    expanded_status_path = PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_status.csv"
    expanded_dataset_path = PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_dataset_summary.csv"
    expanded_commot_dataset_path = PHASE12_DIR / "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_dataset_summary.csv"
    expanded_commot_comparison_path = PHASE12_DIR / "expanded_atlas_commot_downstream_comparison/phase12_expanded_commot_downstream_method_comparison.csv"
    if expanded_summary_path.exists() and expanded_status_path.exists():
        summary = pd.read_csv(expanded_summary_path)
        status = pd.read_csv(expanded_status_path)
        dataset_summary = pd.read_csv(expanded_dataset_path) if expanded_dataset_path.exists() else pd.DataFrame()
        for stat in status.itertuples(index=False):
            dataset_key = stat.dataset_key
            sub = summary[summary["dataset_key"].eq(dataset_key)].copy()
            evaluable = sub[sub["n_targets_present"].fillna(0).astype(int).gt(0)]
            significant = evaluable[evaluable["target_top300_enrichment_bh_fdr"].fillna(1.0).le(0.1)]
            best = significant.sort_values(["target_top300_enrichment_bh_fdr", "rank"]).head(4)
            best_text = (
                "; ".join(
                    f"{r.lr_pair}#{int(r.rank)} FDR={float(r.target_top300_enrichment_bh_fdr):.3g}"
                    for r in best.itertuples(index=False)
                )
                if len(best)
                else "no LR reached target-enrichment BH FDR <= 0.1"
            )
            ds = dataset_summary[dataset_summary["dataset_key"].eq(dataset_key)].head(1)
            module_rep = int(ds["n_module_representative_lr"].iloc[0]) if len(ds) else 0
            module_hit = int(ds["n_modules_with_target_enriched_rep"].iloc[0]) if len(ds) else 0
            rows.append(
                {
                    "phase12_component": f"Goal 2 expanded-atlas downstream scan: {dataset_key}",
                    "status": stat.status,
                    "key_result": (
                        f"{int(stat.n_lr_scanned)} top/anchor/module LR scanned across {int(stat.n_genes_tested)} genes; "
                        f"module-representative LR={module_rep}; evaluable target sets={len(evaluable)}; "
                        f"significant target-enriched LR={len(significant)}; modules with enriched representative={module_hit}; {best_text}"
                    ),
                    "supporting_file": rel(expanded_summary_path),
                    "claim_level": "expanded-atlas observational downstream support; not causal validation and not pan-LR target proof",
                }
            )
    if expanded_commot_dataset_path.exists() and expanded_commot_comparison_path.exists():
        method_summary = pd.read_csv(expanded_commot_dataset_path)
        comparison = pd.read_csv(expanded_commot_comparison_path)
        for dataset_key, sub in method_summary.groupby("dataset_key", sort=False):
            ss = sub[sub["method"].astype(str).eq("spatialself_calibrated")].head(1)
            cm = sub[sub["method"].astype(str).eq("commot_adapter")].head(1)
            comp = comparison[comparison["dataset_key"].astype(str).eq(str(dataset_key))]
            calls = comp["comparison_call"].value_counts().to_dict() if not comp.empty else {}
            call_text = "; ".join(f"{k}={int(v)}" for k, v in calls.items())
            highlight = comp[
                comp["comparison_call"].astype(str).isin(
                    ["spatialself_target_support_only", "commot_target_support_only"]
                )
            ].sort_values(["comparison_call", "spatialself_rank"]).head(4)
            highlight_text = (
                "; ".join(
                    f"{r.lr_pair}: SS q={fmt_num(r.spatialself_target_fdr)}, COMMOT q={fmt_num(r.commot_target_fdr)}"
                    for r in highlight.itertuples(index=False)
                )
                if len(highlight)
                else "no one-sided FDR-supported LR"
            )
            rows.append(
                {
                    "phase12_component": f"Goal 2 expanded SpatialSelf vs COMMOT downstream comparison: {dataset_key}",
                    "status": "completed_commot_style_adapter_comparison",
                    "key_result": (
                        f"SpatialSelf target-enriched={int(ss['n_target_enriched_q_0_1'].iloc[0]) if len(ss) else 0}; "
                        f"COMMOT adapter target-enriched={int(cm['n_target_enriched_q_0_1'].iloc[0]) if len(cm) else 0}; "
                        f"calls: {call_text}; highlights: {highlight_text}"
                    ),
                    "supporting_file": rel(expanded_commot_comparison_path),
                    "claim_level": "same-universe COMMOT-style adapter/downstream proxy; not official end-to-end COMMOT communication_impact",
                }
            )
    summary_path = PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv"
    status_path = PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_status.csv"
    if not summary_path.exists() or not status_path.exists():
        return rows
    summary = pd.read_csv(summary_path)
    status = pd.read_csv(status_path)
    for stat in status.itertuples(index=False):
        dataset_key = stat.dataset_key
        sub = summary[summary["dataset_key"].eq(dataset_key)].copy()
        evaluable = sub[sub["n_nichenet_targets_present"].fillna(0).astype(int).gt(0)]
        significant = evaluable[evaluable["target_top300_enrichment_bh_fdr"].fillna(1.0).le(0.1)]
        best = significant.sort_values(["target_top300_enrichment_bh_fdr", "target_top300_enrichment"]).head(4)
        if len(best):
            best_text = "; ".join(
                f"{r.lr_pair}#{int(r.lr_rank)} FDR={float(r.target_top300_enrichment_bh_fdr):.3g}"
                for r in best.itertuples(index=False)
            )
        else:
            best_text = "no LR reached target-enrichment BH FDR <= 0.1"
        rows.append(
            {
                "phase12_component": f"Goal 2 whole-transcriptome downstream scan: {dataset_key}",
                "status": stat.status,
                "key_result": f"{int(stat.n_lr_tested)} LR scanned across {int(stat.n_genes_tested)} genes; evaluable target sets={len(evaluable)}; significant target-enriched LR={len(significant)}; {best_text}",
                "supporting_file": rel(summary_path),
                "claim_level": "pilot observational downstream support",
            }
        )
    return rows


def downstream_target_background_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "downstream_target_background_audit/phase12_downstream_target_background_dataset_audit.csv"
    lr_path = PHASE12_DIR / "downstream_target_background_audit/phase12_downstream_target_background_lr_audit.csv"
    module_path = PHASE12_DIR / "downstream_target_background_audit/phase12_downstream_target_background_module_audit.csv"
    gene_path = PHASE12_DIR / "downstream_target_background_audit/phase12_downstream_target_background_gene_program_audit.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    lr = pd.read_csv(lr_path) if lr_path.exists() else pd.DataFrame()
    module = pd.read_csv(module_path) if module_path.exists() else pd.DataFrame()
    gene = pd.read_csv(gene_path) if gene_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 2 target-background audit: {row.dataset_key}",
                "status": row.dataset_target_background_call,
                "key_result": (
                    f"scanned LR={int(row.n_lr_scanned)}, with curated targets={int(row.n_lr_with_curated_targets)}, "
                    f"supported q<=0.1={int(row.n_target_supported_q_0_1)}, "
                    f"true-supported={int(row.n_true_lr_target_supported_q_0_1)}, "
                    f"module-supported={int(row.n_module_rep_target_supported_q_0_1)}, "
                    f"high-expression target leakage={int(row.n_high_expression_negative_target_leakage)}, "
                    f"best={row.best_supported_lr if pd.notna(row.best_supported_lr) and str(row.best_supported_lr) else 'none'}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "selected top/priority/module LR target-background audit; not pan-atlas functional proof",
            }
        )
    if not lr.empty:
        supported = lr[
            lr["target_background_call"].astype(str).isin(
                ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
            )
        ].sort_values(["dataset_key", "target_support_score", "rank"], ascending=[True, False, True])
        if len(supported):
            rows.append(
                {
                    "phase12_component": "Goal 2 top LR with target genes above full-gene background",
                    "status": "completed_target_rank_audit",
                    "key_result": "; ".join(
                        f"{r.dataset_key}/{r.lr_pair}: rank={int(r.rank)}, targetFDR={fmt_num(r.target_top300_enrichment_bh_fdr)}, enrich={fmt_num(r.target_top300_enrichment)}"
                        for r in supported.head(8).itertuples(index=False)
                    ),
                    "supporting_file": rel(lr_path),
                    "claim_level": "receiver target genes rank above full expressed-gene background for selected LR",
                }
            )
    if not module.empty:
        module_supported = module[
            module["module_target_background_call"].astype(str).eq("module_has_target_supported_representative")
        ]
        rows.append(
            {
                "phase12_component": "Goal 2 module-level target-background audit",
                "status": "completed_module_target_audit",
                "key_result": (
                    f"modules tested={len(module)}, modules with supported representative={len(module_supported)}"
                    + (
                        "; "
                        + "; ".join(
                            f"{r.dataset_key}/{r.module}: {int(r.n_target_supported_pairs)}/{int(r.n_top_pairs_with_targets)}"
                            for r in module_supported.head(6).itertuples(index=False)
                        )
                        if len(module_supported)
                        else ""
                    )
                ),
                "supporting_file": rel(module_path),
                "claim_level": "module representative target-rank audit",
            }
        )
    if not gene.empty:
        rows.append(
            {
                "phase12_component": "Goal 2 supported LR receiver gene-program source table",
                "status": "completed_gene_program_table",
                "key_result": f"supported LR gene-program rows={len(gene)}; supported datasets={gene['dataset_key'].nunique() if 'dataset_key' in gene else 0}; scanned datasets={dataset['dataset_key'].nunique() if 'dataset_key' in dataset else 0}",
                "supporting_file": rel(gene_path),
                "claim_level": "source table for top receiver genes and target genes in supported LR",
            }
        )
    return rows


def gene_program_rows() -> list[dict[str, object]]:
    summary_path = PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_enrichment_summary.csv"
    specificity_path = PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_specificity_summary.csv"
    pathway_dataset_path = PHASE12_DIR / "pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv"
    pathway_candidate_path = PHASE12_DIR / "pathway_specific_programs/phase12_pathway_specific_candidate_programs.csv"
    module_path = PHASE12_DIR / "gene_program_enrichment/phase12_module_gene_program_summary.csv"
    commot_path = PHASE12_DIR / "gene_program_enrichment/phase12_commot_impact_comparison_summary.csv"
    status_path = PHASE12_DIR / "gene_program_enrichment/phase12_gene_program_resource_status.csv"
    if not summary_path.exists() and not module_path.exists() and not commot_path.exists():
        return []
    rows = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for dataset_key, sub in summary.groupby("dataset_key"):
            sig_programs = sub[sub["n_significant_terms_fdr_0_05"].fillna(0).astype(int).gt(0)]
            true_sig = sig_programs[sig_programs["true_lr"].astype(bool)]
            neg_sig = sig_programs[sig_programs["false_positive_high_expression"].astype(bool)]
            best = sub.sort_values(["best_term_fdr", "n_significant_terms_fdr_0_05"], ascending=[True, False]).head(1)
            best_text = (
                f"best={best.iloc[0].program_label} {best.iloc[0].library}:{best.iloc[0].best_term} FDR={fmt_num(best.iloc[0].best_term_fdr)}"
                if len(best)
                else "no enrichment rows"
            )
            rows.append(
                {
                    "phase12_component": f"Goal 2 gene-program enrichment: {dataset_key}",
                    "status": "completed_program_enrichment",
                    "key_result": f"program-library rows={len(sub)}, significant true-LR rows={len(true_sig)}, significant negative rows={len(neg_sig)}; {best_text}",
                    "supporting_file": rel(summary_path),
                    "claim_level": "observational gene-program enrichment; specificity must be checked against negative programs",
                }
            )
    if specificity_path.exists():
        specificity = pd.read_csv(specificity_path)
        for dataset_key, sub in specificity.groupby("dataset_key"):
            class_counts = sub["program_specificity_class"].astype(str).value_counts().to_dict()
            pieces = ", ".join(f"{k}={int(v)}" for k, v in class_counts.items())
            true_support = sub[sub["program_specificity_class"].astype(str).eq("specific_true_lr_support")]
            mixed_support = sub[sub["program_specificity_class"].astype(str).eq("mixed_true_lr_support_broad_dominant")]
            leakage = sub[sub["program_specificity_class"].astype(str).eq("negative_specific_leakage")]
            top = pd.concat([true_support, mixed_support], ignore_index=True) if len(true_support) or len(mixed_support) else pd.DataFrame()
            if not top.empty:
                top = top.sort_values(
                    ["n_specific_significant_terms_fdr_0_05", "best_specific_term_fdr"],
                    ascending=[False, True],
                ).head(3)
                top_text = "; ".join(
                    f"{r.program_label} {int(r.n_specific_significant_terms_fdr_0_05)} specific ({r.program_specificity_class})"
                    for r in top.itertuples(index=False)
                )
            else:
                top_text = "no true-LR specific support class"
            rows.append(
                {
                    "phase12_component": f"Goal 2 program specificity audit: {dataset_key}",
                    "status": "completed_specificity_audit",
                    "key_result": f"{pieces}; true/mixed support={len(true_support) + len(mixed_support)}, negative leakage={len(leakage)}; {top_text}",
                    "supporting_file": rel(specificity_path),
                    "claim_level": "specificity audit separating broad programs from interpretable communication programs",
                }
            )
    if module_path.exists():
        modules = pd.read_csv(module_path)
        for dataset_key, sub in modules.groupby("dataset_key"):
            top = sub.sort_values(["n_significant_terms_fdr_0_05", "best_term_fdr"], ascending=[False, True]).head(2)
            text = "; ".join(
                f"{r.module} {r.best_term} FDR={fmt_num(r.best_term_fdr)} sig={int(r.n_significant_terms_fdr_0_05)}"
                for r in top.itertuples(index=False)
            )
            rows.append(
                {
                    "phase12_component": f"Goal 2 module gene-program enrichment: {dataset_key}",
                    "status": "completed_module_program_summary",
                    "key_result": text if text else "no module program rows",
                    "supporting_file": rel(module_path),
                    "claim_level": "module-level receiver program summary",
                }
            )
    if pathway_dataset_path.exists():
        pathway_dataset = pd.read_csv(pathway_dataset_path)
        for row in pathway_dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 2 pathway-specific receiver program audit: {row.dataset_key}",
                    "status": row.dataset_program_call,
                    "key_result": (
                        f"clean={int(row.n_clean_specific_support)}, "
                        f"mixed-broad={int(row.n_mixed_specific_support_broad_dominant)}, "
                        f"negative leakage={int(row.n_negative_specific_leakage)}, "
                        f"best clean={fmt_text(row.best_clean_program)}, best mixed={fmt_text(row.best_mixed_program)}"
                    ),
                    "supporting_file": rel(pathway_dataset_path),
                    "claim_level": "pathway-specific receiver-program audit; observational, not perturbational",
                }
            )
    if pathway_candidate_path.exists():
        candidates = pd.read_csv(pathway_candidate_path)
        clean = candidates[candidates["pathway_specific_call"].astype(str).eq("clean_specific_support")]
        mixed = candidates[candidates["pathway_specific_call"].astype(str).eq("specific_support_broad_dominant")]
        text_parts = []
        for label, sub in [("clean", clean), ("mixed", mixed)]:
            top = sub.sort_values("best_specific_term_fdr").head(3) if not sub.empty else pd.DataFrame()
            if len(top):
                text_parts.append(
                    f"{label}: "
                    + "; ".join(
                        f"{r.dataset_key}/{r.program_label}/{r.best_specific_term} FDR={fmt_num(r.best_specific_term_fdr)}"
                        for r in top.itertuples(index=False)
                    )
                )
        if text_parts:
            rows.append(
                {
                    "phase12_component": "Goal 2 top pathway-specific receiver programs",
                    "status": "completed_candidate_program_audit",
                    "key_result": " | ".join(text_parts),
                    "supporting_file": rel(pathway_candidate_path),
                    "claim_level": "candidate receiver programs for manuscript story selection",
                }
            )
    if commot_path.exists():
        commot = pd.read_csv(commot_path)
        for dataset_key, sub in commot.groupby("dataset_key"):
            pieces = []
            for row in sub.itertuples(index=False):
                if getattr(row, "status", "") != "completed":
                    pieces.append(f"{getattr(row, 'method', 'NA')}: {row.status}")
                    continue
                pieces.append(
                    f"{row.method}: mean positive deltaR2={fmt_num(row.mean_positive_delta_cv_r2)}, max negative deltaR2={fmt_num(row.max_negative_delta_cv_r2)}, best={row.best_positive_lr_by_delta}"
                )
            rows.append(
                {
                    "phase12_component": f"Goal 2 COMMOT impact comparison: {dataset_key}",
                    "status": "completed_existing_impact_summary",
                    "key_result": " | ".join(pieces),
                    "supporting_file": rel(commot_path),
                    "claim_level": "reused existing COMMOT-style communication impact summary",
                }
            )
    if status_path.exists():
        status = pd.read_csv(status_path)
        missing = status[~status["status"].astype(str).eq("loaded")]
        rows.append(
            {
                "phase12_component": "Goal 2 pathway resource status",
                "status": "completed_resource_audit",
                "key_result": f"resource rows={len(status)}, missing/not loaded={len(missing)}",
                "supporting_file": rel(status_path),
                "claim_level": "resource availability audit",
            }
        )
    return rows


def hard_negative_stress_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv"
    positive_path = PHASE12_DIR / "hard_negative_stress/phase12_hard_negative_stress_positive_summary.csv"
    edge_dataset_path = PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_dataset_summary.csv"
    edge_class_path = PHASE12_DIR / "edge_hard_negative_audit/phase12_edge_hard_negative_class_summary.csv"
    receiver_shuffle_path = PHASE12_DIR / "receiver_shuffle_hard_negative/phase12_receiver_shuffle_dataset_summary.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 3 hard-negative stress: {row.dataset_key}",
                "status": "completed_pilot_stress",
                "key_result": f"positive anchors={int(row.n_positive_controls)}, rank-score pass={int(row.n_positive_rank_score_pass)}, downstream pass={int(row.n_positive_downstream_pass)}, negative beat fraction={float(row.negative_comparison_beat_fraction):.2f}, receptor-shared beats={int(row.n_receptor_shared_beats)}",
                "supporting_file": rel(positive_path),
                "claim_level": "pilot hard-negative stress test",
            }
        )
    if edge_dataset_path.exists():
        edge_dataset = pd.read_csv(edge_dataset_path)
        for row in edge_dataset.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 3 edge-level hard-negative audit: {row.dataset_key} {row.score_mode}",
                    "status": "completed_fixed_score_edge_stress",
                    "key_result": (
                        f"negative classes={int(row.n_negative_classes)}, "
                        f"leakage tests={int(row.n_leakage_lr_class_tests)}/{int(row.n_lr_class_tests)}, "
                        f"mean leakage fraction={fmt_num(row.mean_leakage_fraction)}, hardest={row.hardest_negative_class}"
                    ),
                    "supporting_file": rel(edge_dataset_path),
                    "claim_level": "fixed-score edge-level hard-negative stress; exposes leakage, not model-rerun FDR",
                }
            )
        if {"dataset_key", "score_mode", "n_leakage_lr_class_tests", "n_lr_class_tests"}.issubset(edge_dataset.columns):
            pieces = []
            for dataset_key, sub in edge_dataset.groupby("dataset_key", sort=False):
                mode = sub.set_index("score_mode")
                if "raw" in mode.index and "edge_guardrail" in mode.index:
                    raw = mode.loc["raw"]
                    guard = mode.loc["edge_guardrail"]
                    delta = int(guard["n_leakage_lr_class_tests"]) - int(raw["n_leakage_lr_class_tests"])
                    pieces.append(
                        f"{dataset_key}: raw={int(raw['n_leakage_lr_class_tests'])}/{int(raw['n_lr_class_tests'])}, "
                        f"guardrail={int(guard['n_leakage_lr_class_tests'])}/{int(guard['n_lr_class_tests'])}, delta={delta:+d}"
                    )
            if pieces:
                rows.append(
                    {
                        "phase12_component": "Goal 3 edge guardrail calibration delta",
                        "status": "diagnostic_guardrail_mixed",
                        "key_result": "; ".join(pieces),
                        "supporting_file": rel(edge_dataset_path),
                        "claim_level": "diagnostic post-hoc calibration; not final model or FDR",
                    }
                )
    if edge_class_path.exists():
        edge_class = pd.read_csv(edge_class_path)
        hardest = (
            edge_class.sort_values(["negative_leakage_fraction", "median_negative_minus_positive_top_tail"], ascending=False)
            .head(5)
        )
        text = "; ".join(
            f"{r.dataset_key}/{r.score_mode}/{r.negative_class}={int(r.n_negative_leakage)}/{int(r.n_lr_with_negative_edges)}"
            for r in hardest.itertuples(index=False)
        )
        rows.append(
            {
                "phase12_component": "Goal 3 hardest edge-level negative classes",
                "status": "completed_edge_class_leakage_audit",
                "key_result": text,
                "supporting_file": rel(edge_class_path),
                "claim_level": "hardest fixed-score edge-negative classes; use as failure-mode guardrail",
            }
        )
    if receiver_shuffle_path.exists():
        receiver_shuffle = pd.read_csv(receiver_shuffle_path)
        for row in receiver_shuffle.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 3 receiver-shuffle hard-negative audit: {row.dataset_key}",
                    "status": "completed_fixed_score_receiver_shuffle",
                    "key_result": (
                        f"true LR tested={int(row.n_true_lr_tested)}, "
                        f"receiver-shuffle q<=0.1={int(row.n_receiver_shuffle_fdr_0_1)}, "
                        f"leakage={int(row.n_receiver_shuffle_leakage)}/{int(row.n_true_lr_tested)}, "
                        f"median delta={fmt_num(row.median_receiver_shuffle_delta_top_tail)}, "
                        f"best={row.best_supported_lr}"
                    ),
                    "supporting_file": rel(receiver_shuffle_path),
                    "claim_level": "fixed-score receiver-context hard-negative stress; not model-rerun global FDR",
                }
            )
    return rows


def directionality_rows() -> list[dict[str, object]]:
    method_path = PHASE12_DIR / "directionality_benchmark/phase12_directionality_method_summary.csv"
    comparison_path = PHASE12_DIR / "directionality_benchmark/phase12_directionality_method_comparison.csv"
    if not method_path.exists():
        return []
    method = pd.read_csv(method_path)
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    rows = []
    for dataset_key, sub in method.groupby("dataset_key"):
        pieces = []
        for row in sub.itertuples(index=False):
            pieces.append(
                f"{row.method}: mean expected={float(row.mean_expected_flow_fraction):.2f}, top-flow expected={int(row.n_top_flow_expected)}/{int(row.n_lr)}, strong={int(row.n_strong_directionality_support)}"
            )
        calls = comparison[comparison["dataset_key"].eq(dataset_key)]["comparison_call"].value_counts() if not comparison.empty else pd.Series(dtype=int)
        call_text = "; ".join(f"{k}={int(v)}" for k, v in calls.items()) if len(calls) else "no paired method comparison"
        rows.append(
            {
                "phase12_component": f"Goal 5 directionality benchmark: {dataset_key}",
                "status": "completed_directionality_metrics",
                "key_result": " | ".join(pieces) + f"; comparison: {call_text}",
                "supporting_file": rel(method_path),
                "claim_level": "descriptive source-sink consistency",
            }
        )
    return rows


def directionality_statistical_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_dataset_summary.csv"
    comparison_path = PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_method_comparison.csv"
    readiness_path = PHASE12_DIR / "directionality_statistics/phase12_directionality_permutation_readiness.csv"
    permutation_dataset_path = PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_dataset_summary.csv"
    permutation_comparison_path = PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_method_comparison.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 5 directionality statistical audit: {row.dataset_key} {row.method}",
                "status": row.statistical_call,
                "key_result": (
                    f"FDR<=0.1={int(row.n_expected_flow_fdr_0_1)}/{int(row.n_lr)}, "
                    f"FDR<=0.25={int(row.n_expected_flow_fdr_0_25)}/{int(row.n_lr)}, "
                    f"mean expected-flow={fmt_num(row.mean_expected_flow_fraction)}, "
                    f"top-flow expected={int(row.n_top_flow_expected)}/{int(row.n_lr)}, "
                    f"wins/losses={int(row.method_comparison_wins)}/{int(row.method_comparison_losses)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "empirical expected-flow FDR; not coordinate/LR-label permutation p-value",
            }
        )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        text = "; ".join(
            f"{r.dataset_key}: SS higher={int(r.n_spatialself_stronger_or_higher)}, COMMOT higher={int(r.n_commot_stronger_or_higher)}, mixed={int(r.n_similar_or_mixed)}"
            for r in comparison.itertuples(index=False)
        )
        rows.append(
            {
                "phase12_component": "Goal 5 SpatialSelf vs COMMOT directionality statistics",
                "status": "completed_method_comparison_audit",
                "key_result": text,
                "supporting_file": rel(comparison_path),
                "claim_level": "method comparison; no global SpatialSelf directionality superiority",
            }
        )
    if readiness_path.exists():
        readiness = pd.read_csv(readiness_path)
        missing = readiness["missing_metrics"].astype(str).replace("nan", "").unique().tolist()
        has_label_permutation = permutation_dataset_path.exists()
        rows.append(
            {
                "phase12_component": "Goal 5 permutation readiness",
                "status": "aggregate_label_permutation_available_coordinate_missing" if has_label_permutation else "needs_permutation_null",
                "key_result": f"datasets={len(readiness)}, legacy missing metrics={'; '.join(m for m in missing if m) or 'none'}; label permutation audit={'available' if has_label_permutation else 'missing'}",
                "supporting_file": rel(readiness_path),
                "claim_level": "readiness gap audit; aggregated label permutations are available, coordinate-level edge reruns are still missing",
            }
        )
    if permutation_dataset_path.exists():
        perm = pd.read_csv(permutation_dataset_path)
        for row in perm.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 5 directionality label-permutation audit: {row.dataset_key} {row.method}",
                    "status": row.permutation_audit_call,
                    "key_result": (
                        f"both-null FDR<=0.1={int(row.n_both_fdr_0_1)}/{int(row.n_lr)}, "
                        f"LR-label FDR<=0.1={int(row.n_lr_label_fdr_0_1)}, "
                        f"niche-label FDR<=0.1={int(row.n_niche_label_fdr_0_1)}, "
                        f"best={row.best_lr}, q_lr={fmt_num(row.best_lr_lr_label_fdr)}, q_niche={fmt_num(row.best_lr_niche_label_fdr)}"
                    ),
                    "supporting_file": rel(permutation_dataset_path),
                    "claim_level": "aggregated niche-flow label-permutation audit; not coordinate-level edge permutation",
                }
            )
    if permutation_comparison_path.exists():
        comparison = pd.read_csv(permutation_comparison_path)
        text = "; ".join(
            f"{dataset_key}: "
            + ", ".join(f"{call}={int(count)}" for call, count in sub["comparison_call"].value_counts().items())
            for dataset_key, sub in comparison.groupby("dataset_key", sort=False)
        )
        rows.append(
            {
                "phase12_component": "Goal 5 SpatialSelf vs COMMOT label-permutation comparison",
                "status": "completed_label_permutation_comparison",
                "key_result": text,
                "supporting_file": rel(permutation_comparison_path),
                "claim_level": "aggregated label-permutation comparison; no global SpatialSelf directionality superiority",
            }
        )
    return rows


def directionality_edge_permutation_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv"
    comparison_path = PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_method_comparison.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 5 directionality edge-level permutation audit: {row.dataset_key} {row.method}",
                "status": row.edge_permutation_dataset_call,
                "key_result": (
                    f"LR={int(row.n_lr)}, receiver-pair q<=0.1={int(row.n_receiver_pairing_fdr_0_1)}, "
                    f"vector q<=0.1={int(row.n_edge_vector_fdr_0_1)}, both={int(row.n_receiver_and_vector_fdr_0_1)}, "
                    f"best={row.best_lr}, recv_q={fmt_num(row.best_lr_receiver_pairing_fdr)}, vec_q={fmt_num(row.best_lr_edge_vector_fdr)}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "edge-level fixed-score coordinate/niche pairing null; not model retraining or causal transport",
            }
        )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        text = "; ".join(
            f"{dataset_key}: "
            + ", ".join(f"{call}={int(count)}" for call, count in sub["edge_permutation_comparison_call"].value_counts().items())
            for dataset_key, sub in comparison.groupby("dataset_key", sort=False)
        )
        rows.append(
            {
                "phase12_component": "Goal 5 SpatialSelf vs COMMOT edge-level permutation comparison",
                "status": "completed_edge_level_permutation_comparison",
                "key_result": text,
                "supporting_file": rel(comparison_path),
                "claim_level": "edge-level fixed-score method comparison; no global directionality superiority claim",
            }
        )
    return rows


def directionality_claim_guardrail_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "directionality_claim_guardrail/phase12_directionality_claim_guardrail_dataset.csv"
    method_path = PHASE12_DIR / "directionality_claim_guardrail/phase12_directionality_claim_guardrail_method.csv"
    lr_path = PHASE12_DIR / "directionality_claim_guardrail/phase12_directionality_claim_guardrail_lr.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 5 directionality claim guardrail: {row.dataset_key} {row.method}",
                "status": row.dataset_claim_call,
                "key_result": (
                    f"supported LR={int(row.n_supported_directionality_claims)}, vector-supported={int(row.n_vector_supported)}, "
                    f"local={int(row.n_local_retained)}, boundary={int(row.n_boundary_crossing)}, long-range-like={int(row.n_long_range_like)}, "
                    f"negative-supported={int(row.n_high_expression_negative_directionality_supported)}, "
                    f"best={row.best_lr} ({row.best_lr_claim})"
                ),
                "supporting_file": rel(lr_path),
                "claim_level": "directionality claim guardrail; supports selected source-sink claims, not causal transport or global superiority",
            }
        )
    if method_path.exists():
        method = pd.read_csv(method_path)
        for row in method.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 5 directionality method guardrail: {row.method}",
                    "status": "completed_directionality_guardrail",
                    "key_result": (
                        f"LR={int(row.n_lr)}, supported={int(row.n_supported_directionality_claims)}, "
                        f"vector-supported={int(row.n_vector_supported)}, high-expression negative supported={int(row.n_high_expression_negative_supported)}, "
                        f"mean expected flow={fmt_num(row.mean_expected_flow_fraction)}, mean vector coherence={fmt_num(row.mean_vector_coherence)}"
                    ),
                    "supporting_file": rel(method_path),
                    "claim_level": "method-level guardrail summary; adapter/fixed-score caveats still apply",
                }
            )
    return rows


def directionality_consensus_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "directionality_consensus/phase12_directionality_consensus_dataset_summary.csv"
    pair_path = PHASE12_DIR / "directionality_consensus/phase12_directionality_consensus_method_pair.csv"
    method_path = PHASE12_DIR / "directionality_consensus/phase12_directionality_consensus_method_summary.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    pair = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    method = pd.read_csv(method_path) if method_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 5 directionality consensus audit: {row.dataset_key} {row.method}",
                "status": row.dataset_consensus_call,
                "key_result": (
                    f"supported={int(row.n_consensus_supported)}/{int(row.n_lr)}, "
                    f"strict-vector={int(row.n_strict_edge_vector_consensus)}, "
                    f"method-specific={int(row.n_method_specific_supported)}, "
                    f"negative-leakage={int(row.n_negative_directionality_leakage)}, "
                    f"best={row.best_lr} ({row.best_lr_consensus_call})"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "merged directionality p-value and method-comparison guardrail; not causal transport proof",
            }
        )
    if not pair.empty:
        calls = pair["consensus_method_comparison_call"].value_counts()
        highlights = pair[
            pair["consensus_method_comparison_call"].astype(str).isin(
                ["spatialself_stronger_consensus", "commot_stronger_consensus"]
            )
        ].sort_values(["dataset_key", "lr_pair"])
        highlight_text = "; ".join(
            f"{r.dataset_key}/{r.lr_pair}:{r.consensus_method_comparison_call}"
            for r in highlights.head(8).itertuples(index=False)
        )
        rows.append(
            {
                "phase12_component": "Goal 5 SpatialSelf vs COMMOT directionality consensus",
                "status": "completed_consensus_method_comparison",
                "key_result": (
                    "; ".join(f"{idx}={int(val)}" for idx, val in calls.items())
                    + (f"; highlights: {highlight_text}" if highlight_text else "")
                ),
                "supporting_file": rel(pair_path),
                "claim_level": "consensus-level method comparison; supports selected cases, not global SpatialSelf directionality superiority",
            }
        )
    if not method.empty:
        rows.append(
            {
                "phase12_component": "Goal 5 directionality consensus method summary",
                "status": "completed_consensus_method_summary",
                "key_result": "; ".join(
                    f"{r.method}: supported={int(r.n_consensus_supported)}, strict={int(r.n_strict_edge_vector_consensus)}, method-specific={int(r.n_method_specific_supported)}, negative={int(r.n_negative_directionality_leakage)}"
                    for r in method.itertuples(index=False)
                ),
                "supporting_file": rel(method_path),
                "claim_level": "method-level consensus summary; no causal or global-superiority claim",
            }
        )
    return rows


def coordinate_model_rerun_smoke_rows() -> list[dict[str, object]]:
    summary_paths = sorted((PHASE12_DIR / "coordinate_model_rerun_smoke").glob("*/*_coordinate_model_rerun_summary.csv"))
    rows = []
    for path in summary_paths:
        summary = pd.read_csv(path)
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 5 coordinate model-rerun smoke: {row.dataset_key}",
                    "status": row.status,
                    "key_result": (
                        f"completed coordinate reruns={int(row.n_completed_permutations)}, "
                        f"best positive={row.best_positive_lr} p={fmt_num(row.best_positive_p)} BH={fmt_num(row.best_positive_bh_fdr)}, "
                        f"best high-expression negative={row.best_negative_lr} p={fmt_num(row.best_negative_p)} BH={fmt_num(row.best_negative_bh_fdr)}"
                    ),
                    "supporting_file": rel(path),
                    "claim_level": "end-to-end coordinate-permutation model-rerun smoke; low permutation count, not final FDR",
                }
            )
    return rows


def integrated_guarded_spatial_coordinate_rerun_smoke_rows() -> list[dict[str, object]]:
    summary_paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke").glob(
            "*/*_integrated_coordinate_summary.csv"
        )
    )
    rows = []
    summaries = []
    for path in summary_paths:
        summary = pd.read_csv(path)
        summary["supporting_file"] = rel(path)
        summaries.append(summary)
    if not summaries:
        return rows
    table = pd.concat(summaries, ignore_index=True)
    table["n_completed_permutations"] = pd.to_numeric(table["n_completed_permutations"], errors="coerce").fillna(0)
    table = table.sort_values(["dataset_key", "n_completed_permutations"]).groupby("dataset_key", as_index=False).tail(1)
    for _, row in table.iterrows():
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated guarded spatial coordinate-rerun smoke: {row.dataset_key}",
                "status": row.status,
                "key_result": (
                    f"completed coordinate reruns={int(row.n_completed_permutations)}, "
                    f"target={row.target_lr} rank={int(row.target_rank)} "
                    f"score={fmt_num(row.target_score)} null_q95={fmt_num(row.target_null_q95)} "
                    f"p={fmt_num(row.target_p)} BH={fmt_num(row.target_bh_fdr)}, "
                    f"best high-expression negative={row.best_negative_lr} "
                    f"p={fmt_num(row.best_negative_p)} BH={fmt_num(row.best_negative_bh_fdr)}"
                ),
                "supporting_file": row.supporting_file,
                "claim_level": "end-to-end coordinate-rerun smoke for integrated candidate; low permutation count and broad/background leakage, not formal FDR",
            }
        )
    return rows


def integrated_guarded_spatial_coordinate_formal_parameter_smoke_rows() -> list[dict[str, object]]:
    summary_paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke").glob(
            "*formal*/*_integrated_coordinate_summary.csv"
        )
    )
    rows = []
    for path in summary_paths:
        summary = pd.read_csv(path)
        if summary.empty:
            continue
        manifest_path = path.with_name(path.name.replace("_summary.csv", "_manifest.json"))
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "phase12_component": f"Goal 1 integrated formal-parameter coordinate smoke: {row.dataset_key}",
                    "status": row.status,
                    "key_result": (
                        f"completed coordinate reruns={int(row.n_completed_permutations)}, "
                        f"epochs={int(manifest.get('epochs', -1)) if manifest else 'NA'}, "
                        f"max_train_samples={int(manifest.get('max_train_samples', -1)) if manifest else 'NA'}, "
                        f"target={row.target_lr} rank={int(row.target_rank)} "
                        f"score={fmt_num(row.target_score)} null_q95={fmt_num(row.target_null_q95)} "
                        f"p={fmt_num(row.target_p)} BH={fmt_num(row.target_bh_fdr)}, "
                        f"best high-expression negative={row.best_negative_lr} "
                        f"p={fmt_num(row.best_negative_p)} BH={fmt_num(row.best_negative_bh_fdr)}, "
                        f"min-p true={int(row.n_positive_p_min)}, min-p highNeg={int(row.n_negative_p_min)}"
                    ),
                    "supporting_file": rel(path),
                    "claim_level": "formal-parameter end-to-end coordinate-rerun smoke; confirms execution and rank stability but has too few permutations for formal FDR",
                }
            )
    return rows


def integrated_guarded_spatial_coordinate_summary_only_smoke_rows() -> list[dict[str, object]]:
    summary_paths = sorted(
        (PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_summary_only_smoke").glob(
            "*/*_integrated_coordinate_summary.csv"
        )
    )
    frames = []
    for path in summary_paths:
        summary = pd.read_csv(path)
        if summary.empty:
            continue
        summary["supporting_file"] = rel(path)
        frames.append(summary)
    if not frames:
        return []
    table = pd.concat(frames, ignore_index=True)
    table["n_completed_permutations"] = pd.to_numeric(table["n_completed_permutations"], errors="coerce").fillna(0)
    table = table.sort_values(["dataset_key", "n_completed_permutations"]).groupby("dataset_key", as_index=False).tail(1)
    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated coordinate summary-only runner smoke: {row.dataset_key}",
                "status": row.status,
                "key_result": (
                    f"completed coordinate reruns={int(row.n_completed_permutations)}, "
                    f"target={row.target_lr} rank={int(row.target_rank)} "
                    f"score={fmt_num(row.target_score)} null_q95={fmt_num(row.target_null_q95)} "
                    f"p={fmt_num(row.target_p)} BH={fmt_num(row.target_bh_fdr)}; "
                    "null prediction directories are not retained"
                ),
                "supporting_file": row.supporting_file,
                "claim_level": "execution infrastructure smoke for summary-only/no-cache coordinate reruns; not scientific FDR evidence",
            }
        )
    return rows


def coordinate_runner_concordance_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_runner_concordance/"
        / "phase12_coordinate_runner_concordance_summary.csv"
    )
    if not path.exists():
        return []
    table = pd.read_csv(path)
    rows = []
    for row in table.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 coordinate runner concordance: {row.dataset_key}",
                "status": row.runner_concordance_call,
                "key_result": (
                    f"common LR={int(row.n_common_lr)}, target={row.target_lr}, "
                    f"p reference/no-cache={fmt_num(row.reference_target_p)}/{fmt_num(row.nocache_target_p)}, "
                    f"BH reference/no-cache={fmt_num(row.reference_target_bh_fdr)}/{fmt_num(row.nocache_target_bh_fdr)}, "
                    f"null q95 delta={fmt_num(row.target_null_q95_abs_delta)}, "
                    f"best negative same={bool(row.best_negative_same)}, "
                    f"min-p true reference/no-cache={int(row.reference_min_p_true)}/{int(row.nocache_min_p_true)}, "
                    f"min-p highNeg reference/no-cache="
                    f"{int(row.reference_min_p_high_expression_negative)}/"
                    f"{int(row.nocache_min_p_high_expression_negative)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "runner concordance validation only; not new FDR evidence",
            }
        )
    return rows


def integrated_guarded_spatial_coordinate_leakage_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_leakage/"
        / "phase12_integrated_coordinate_leakage_summary.csv"
    )
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated coordinate leakage audit: {row.dataset_key}",
                "status": row.leakage_call,
                "key_result": (
                    f"min-p LR={int(row.n_min_p_lr)} "
                    f"(true={int(row.n_min_p_true_lr)}, "
                    f"highNeg={int(row.n_min_p_high_expression_negative)}, "
                    f"broad/background={int(row.n_min_p_broad_background)}); "
                    f"best true={row.best_true_lr} p={fmt_num(row.best_true_p)} BH={fmt_num(row.best_true_bh_fdr)}, "
                    f"best highNeg={row.best_high_expression_negative_lr} p={fmt_num(row.best_high_expression_negative_p)}, "
                    f"best broad/background={row.best_broad_background_lr} p={fmt_num(row.best_broad_background_p)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "coordinate-rerun leakage decomposition; high-expression negatives suppressed but broad/background leakage blocks whole-atlas FDR",
            }
        )
    return rows


def integrated_guarded_spatial_coordinate_eligible_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_eligible_audit/"
        / "phase12_integrated_coordinate_eligible_dataset.csv"
    )
    if not path.exists():
        return []
    dataset = pd.read_csv(path)
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated coordinate eligible audit: {row.dataset_key}",
                "status": row.eligible_coordinate_call,
                "key_result": (
                    f"eligible={int(row.n_eligible_lr)}/{int(row.n_total_lr)}, "
                    f"q<=0.1 total={int(row.eligible_coordinate_q_0_1)} "
                    f"(true={int(row.eligible_coordinate_true_q_0_1)}, "
                    f"controls={int(row.eligible_coordinate_control_q_0_1)}, "
                    f"background={int(row.eligible_coordinate_background_q_0_1)}); "
                    f"best true={row.best_true_lr} p={fmt_num(row.best_true_coordinate_p)} "
                    f"eligible-q={fmt_num(row.best_true_eligible_coordinate_q)}, "
                    f"best control={row.best_control_lr} eligible-q={fmt_num(row.best_control_eligible_coordinate_q)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "eligibility-restricted coordinate audit; useful diagnostic but still not FDR-supported",
            }
        )
    return rows


def integrated_guarded_spatial_coordinate_magnitude_gate_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_magnitude_gate/"
        / "phase12_integrated_coordinate_magnitude_gate_best.csv"
    )
    if not path.exists():
        return []
    best = pd.read_csv(path)
    rows = []
    for row in best.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated coordinate magnitude-gate diagnostic: {row.dataset_key}",
                "status": row.gate_call,
                "key_result": (
                    f"score threshold={fmt_num(row.score_threshold)}, retained LR={int(row.n_lr)} "
                    f"(true={int(row.n_true_lr)}, highNeg={int(row.n_high_expression_negative)}, "
                    f"broad/background={int(row.n_broad_background)}); "
                    f"q<=0.1 total={int(row.n_q_0_1)} "
                    f"(true={int(row.n_true_q_0_1)}, "
                    f"highNeg={int(row.n_high_expression_negative_q_0_1)}, "
                    f"broad/background={int(row.n_broad_background_q_0_1)}); "
                    f"best true q={fmt_num(row.best_true_q)}, retained={row.top_lr_pairs}"
                ),
                "supporting_file": rel(path),
                "claim_level": "diagnostic score-magnitude gate; clean selected anchors but not a prespecified final FDR rule",
            }
        )
    return rows


def integrated_guarded_spatial_coordinate_background_target_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_background_targets/"
        / "phase12_integrated_coordinate_background_target_summary.csv"
    )
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated coordinate background target audit: {row.dataset_key}",
                "status": row.target_call,
                "key_result": (
                    f"min-p LR={int(row.n_min_p_lr)} "
                    f"(true={int(row.n_min_p_true_lr)}, highNeg={int(row.n_min_p_high_expression_negative)}, "
                    f"broad/background={int(row.n_min_p_broad_background)}); "
                    f"broad/background top10={int(row.n_broad_background_top10)}, top50={int(row.n_broad_background_top50)}; "
                    f"retained true floor={fmt_num(row.retained_true_score_floor)}, "
                    f"max broad/background score={fmt_num(row.max_broad_background_score)}, "
                    f"gap={fmt_num(row.retained_true_to_max_broad_score_gap)}x; "
                    f"broad/background above magnitude gate={int(row.n_broad_background_above_gate)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "diagnostic target list for broad/background leakage; guides next-model guardrails but is not FDR closure",
            }
        )
    return rows


def integrated_background_guard_training_target_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_background_guard_training_targets/"
        / "phase12_integrated_background_guard_training_spec.csv"
    )
    if not path.exists():
        return []
    spec = pd.read_csv(path)
    rows = []
    for row in spec.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated background-guard training targets: {row.dataset_key}",
                "status": "next_model_target_panel_ready",
                "key_result": (
                    f"coordinate reruns={int(row.n_completed_coordinate_reruns)}, "
                    f"background targets={int(row.n_background_targets)} "
                    f"(primary={int(row.n_primary_background_targets)}), "
                    f"min-p LR={int(row.n_min_p_lr)} "
                    f"(true={int(row.n_min_p_true_lr)}, highNeg={int(row.n_min_p_high_expression_negative)}, "
                    f"broad/background={int(row.n_min_p_broad_background)}); "
                    f"gate={fmt_num(row.magnitude_gate_threshold)}, "
                    f"background above gate={int(row.n_background_above_gate)}; "
                    f"best true={row.best_true_lr} BH={fmt_num(row.best_true_coordinate_bh_fdr)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "next-model training target panel derived from formal coordinate leakage; not discovery evidence or FDR closure",
            }
        )
    return rows


def integrated_background_guard_interface_smoke_rows() -> list[dict[str, object]]:
    paths = sorted(
        (
            PHASE12_DIR
            / "integrated_background_guard_training_targets"
            / "interface_smoke"
        ).glob("*/*_metrics.json")
    )
    rows = []
    for path in paths:
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        target_train = int(metrics.get("n_background_guard_target_train", 0))
        guard_train = int(metrics.get("n_background_guard_train", 0))
        status = "target_interface_smoke_passed" if target_train > 0 and guard_train > 0 else "target_interface_smoke_failed"
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated background-guard interface smoke: {metrics.get('replicate', '')}",
                "status": status,
                "key_result": (
                    f"epochs={int(metrics.get('epochs', 0))}, "
                    f"max target samples in guard={target_train}/{guard_train}, "
                    f"target_path={metrics.get('background_guard_target_path', '')}, "
                    f"prediction={metrics.get('prediction_path', '')}"
                ),
                "supporting_file": rel(path),
                "claim_level": "execution smoke for target-panel training interface; not a scientific result or FDR evidence",
            }
        )
    return rows


def integrated_background_target_guarded_training_smoke_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_training_smoke"
        / "phase12_integrated_background_target_guarded_dataset_summary.csv"
    )
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated background-target guarded training smoke: {row.dataset_key}",
                "status": row.target_guard_smoke_call,
                "key_result": (
                    f"epochs={int(row.epochs)}, max_train_samples={int(row.max_train_samples)}, "
                    f"target guard train={int(row.n_background_guard_target_train)}/{int(row.n_background_guard_train)}, "
                    f"top10 true={int(row.top10_true_lr)} vs reference={int(row.reference_top10_true_lr)}, "
                    f"top10 background targets={int(row.top10_background_targets)} vs reference={int(row.reference_top10_background_targets)}, "
                    f"targets above gate={int(row.n_targets_above_magnitude_gate)}, "
                    f"anchor min delta={fmt_num(row.anchor_min_delta_score)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "low-budget target-aware model smoke; useful next-model direction but not reporting baseline or FDR evidence",
            }
        )
    return rows


def integrated_background_target_guarded_permutation_smoke_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_permutation_smoke"
        / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv"
    )
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        call = getattr(row, "target_guarded_permutation_smoke_call", "")
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated background-target guarded exact-score smoke: {row.dataset_key}",
                "status": call,
                "key_result": (
                    f"permutations/null={int(row.n_permutations_per_null)}, "
                    f"top10 true={int(row.top10_true_lr)}, "
                    f"top10 risk={fmt_percent(row.top10_negative_or_risk_fraction)}, "
                    f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
                    f"true LR q<=0.1={int(row.all_true_lr_conservative_fdr_0_1)}, "
                    f"best true={row.best_true_lr} q={fmt_num(row.best_true_lr_fdr)}"
                ),
                "supporting_file": rel(path),
                "claim_level": "low-permutation exact-score smoke for target-aware model path; not formal FDR or baseline replacement",
            }
        )
    return rows


def integrated_background_target_guarded_coordinate_smoke_rows() -> list[dict[str, object]]:
    path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_summary.csv"
    )
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    fdr_path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_lr_fdr.csv"
    )
    fdr = pd.read_csv(fdr_path) if fdr_path.exists() else pd.DataFrame()
    status_path = (
        PHASE12_DIR
        / "integrated_background_target_guarded_coordinate_rerun_smoke"
        / "post_xenium_colon_full"
        / "post_xenium_colon_full_target_guarded_coordinate_status.csv"
    )
    run_status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    skipped_low_disk = (
        int(run_status["status"].astype(str).eq("skipped_low_disk").sum()) if not run_status.empty else 0
    )
    low_disk_detail = ""
    if skipped_low_disk:
        detail = str(run_status[run_status["status"].astype(str).eq("skipped_low_disk")].iloc[-1].get("detail", ""))
        low_disk_detail = f", low-disk skips={skipped_low_disk} ({detail})"
    failed_runs = int(run_status["status"].astype(str).eq("failed").sum()) if not run_status.empty else 0
    failure_detail = ""
    if failed_runs:
        detail = str(run_status[run_status["status"].astype(str).eq("failed")].iloc[-1].get("detail", ""))
        failure_detail = f", failed attempts={failed_runs} ({detail})"
    rows = []
    for row in summary.itertuples(index=False):
        gap_detail = target_coordinate_score_gap_detail(fdr, str(row.target_lr), str(row.best_negative_lr))
        rows.append(
            {
                "phase12_component": f"Goal 1 integrated background-target guarded coordinate-rerun smoke: {row.dataset_key}",
                "status": row.status,
                "key_result": (
                    f"coordinate reruns={int(row.n_completed_permutations)}, "
                    f"target={row.target_lr} rank={int(row.target_rank)}, "
                    f"score={fmt_num(row.target_score)} vs null q95={fmt_num(row.target_null_q95)}, "
                    f"p={fmt_num(row.target_p)}, BH={fmt_num(row.target_bh_fdr)}, "
                    f"best high-expression negative={row.best_negative_lr} p={fmt_num(row.best_negative_p)}, "
                    f"min-p true={int(row.n_positive_p_min)}, min-p high-expression negative={int(row.n_negative_p_min)}"
                    f"{gap_detail}"
                    f"{low_disk_detail}"
                    f"{failure_detail}"
                ),
                "supporting_file": rel(path),
                "claim_level": "low-permutation end-to-end coordinate-rerun smoke for target-aware model path; not formal FDR or baseline replacement",
            }
        )
    return rows


def disk_cleanup_audit_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "disk_cleanup_audit/phase12_disk_cleanup_candidates.csv"
    rows = []
    if path.exists():
        table = pd.read_csv(path)
        if not table.empty:
            total_savings = int(pd.to_numeric(table["estimated_savings_bytes"], errors="coerce").fillna(0).sum())
            primary = table.sort_values("estimated_savings_bytes", ascending=False).iloc[0]
            rows.append(
                {
                    "phase12_component": "Phase 12 disk cleanup audit",
                    "status": "read_only_cleanup_candidates_available",
                    "key_result": (
                        f"candidates={len(table)}, total potential savings={fmt_num(total_savings / (1024 * 1024))} MiB; "
                        f"largest={primary['candidate']} ({fmt_num(float(primary['estimated_savings_mib']))} MiB); "
                        "no files deleted or rewritten"
                    ),
                    "supporting_file": rel(path),
                    "claim_level": "infrastructure guardrail for scaling coordinate-null reruns; not scientific evidence",
                }
            )
    dry_run_path = PHASE12_DIR / "disk_cleanup_audit/phase12_replicate_hardlink_dedup_dry_run.csv"
    if dry_run_path.exists():
        dry = pd.read_csv(dry_run_path)
        candidates = dry[dry["planned_action"].astype(str).eq("candidate_hardlink")].copy()
        savings = float(pd.to_numeric(candidates["size_mib"], errors="coerce").fillna(0).sum()) if len(candidates) else 0.0
        rows.append(
            {
                "phase12_component": "Phase 12 replicate hardlink dedup dry-run",
                "status": "dry_run_candidates_verified",
                "key_result": (
                    f"audited files={len(dry)}, hardlink candidates={len(candidates)}, "
                    f"estimated savings={fmt_num(savings)} MiB; no files modified"
                ),
                "supporting_file": rel(dry_run_path),
                "claim_level": "dry-run infrastructure check only; execute separately if replicate inputs are immutable",
            }
        )
    executed_path = PHASE12_DIR / "disk_cleanup_audit/phase12_replicate_hardlink_dedup_executed.csv"
    if executed_path.exists():
        executed = pd.read_csv(executed_path)
        hardlinked = executed[executed["executed_action"].astype(str).eq("hardlinked_to_base")].copy()
        failures = executed[executed["executed_action"].astype(str).eq("failed")].copy()
        savings = float(pd.to_numeric(hardlinked["size_mib"], errors="coerce").fillna(0).sum()) if len(hardlinked) else 0.0
        rows.append(
            {
                "phase12_component": "Phase 12 replicate hardlink dedup executed",
                "status": "hardlink_dedup_executed",
                "key_result": (
                    f"hardlinked files={len(hardlinked)}, estimated storage reclaimed={fmt_num(savings)} MiB, "
                    f"failures={len(failures)}; non-identical metadata skipped"
                ),
                "supporting_file": rel(executed_path),
                "claim_level": "infrastructure cleanup executed for duplicate immutable replicate inputs; not scientific evidence",
            }
        )
    tiff_path = PHASE12_DIR / "disk_cleanup_audit/phase12_regenerable_tiff_cleanup_executed.csv"
    if tiff_path.exists():
        tiff = pd.read_csv(tiff_path)
        deleted = tiff[tiff["executed_action"].astype(str).eq("deleted_tiff_artifact")].copy()
        failures = tiff[tiff["executed_action"].astype(str).eq("failed")].copy()
        savings = float(pd.to_numeric(deleted["size_mib"], errors="coerce").fillna(0).sum()) if len(deleted) else 0.0
        rows.append(
            {
                "phase12_component": "Phase 12 regenerable TIFF cleanup executed",
                "status": "tiff_artifacts_deleted_with_counterparts",
                "key_result": (
                    f"deleted TIFF files={len(deleted)}, reclaimed={fmt_num(savings)} MiB, "
                    f"failures={len(failures)}; SVG/PDF/PNG counterparts preserved"
                ),
                "supporting_file": rel(tiff_path),
                "claim_level": "storage cleanup for regenerable rendered figure artifacts; source data and vector/raster alternatives preserved",
            }
        )
    return rows


def target_coordinate_score_gap_detail(fdr: pd.DataFrame, target_lr: str, negative_lr: str) -> str:
    if fdr.empty or "lr_pair" not in fdr:
        return ""
    target = fdr[fdr["lr_pair"].astype(str).eq(str(target_lr))]
    negative = fdr[fdr["lr_pair"].astype(str).eq(str(negative_lr))]
    if target.empty or negative.empty:
        return ""
    target_score = float(pd.to_numeric(target.iloc[0].get("calibrated_score"), errors="coerce"))
    negative_score = float(pd.to_numeric(negative.iloc[0].get("calibrated_score"), errors="coerce"))
    negative_rank = pd.to_numeric(negative.iloc[0].get("rank"), errors="coerce")
    if not np.isfinite(target_score) or not np.isfinite(negative_score):
        return ""
    ratio = target_score / max(negative_score, 1e-12)
    rank_text = int(negative_rank) if pd.notna(negative_rank) else "NA"
    return f", negative rank={rank_text}, negative score={fmt_num(negative_score)}, target/negative score ratio={fmt_num(ratio)}x"


def coordinate_control_guardrail_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_dataset_summary.csv"
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 1/5 coordinate control guardrail: {row.dataset_key}",
                "status": "completed_posthoc_control_guardrail",
                "key_result": (
                    f"nominal discoveries={int(row.n_nominal_coordinate_discoveries)} "
                    f"(true={int(row.n_nominal_true_lr)}, highNeg={int(row.n_nominal_high_expression_negative)}); "
                    f"story-guardrail discoveries={int(row.n_story_guardrail_discoveries)} "
                    f"(true={int(row.n_story_guardrail_true_lr)}, highNeg={int(row.n_story_guardrail_high_expression_negative)}); "
                    f"broad-background leakage={int(getattr(row, 'n_control_guardrail_broad_background', 0))} "
                    f"({fmt_percent(getattr(row, 'control_guardrail_broad_background_fraction', float('nan')))} of control-guardrail calls); "
                    f"top={row.top_story_guardrail_lr if pd.notna(row.top_story_guardrail_lr) else ''}"
                ),
                "supporting_file": rel(path),
                "claim_level": "post-hoc high-expression-control guardrail; broad-background leakage quantified; not unbiased whole-atlas FDR",
            }
        )
    return rows


def coordinate_null_closure_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "coordinate_null_closure/phase12_coordinate_null_closure_dataset.csv"
    lr_path = PHASE12_DIR / "coordinate_null_closure/phase12_coordinate_null_closure_lr.csv"
    story_path = PHASE12_DIR / "coordinate_null_closure/phase12_coordinate_null_closure_story.csv"
    if not dataset_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    lr = pd.read_csv(lr_path) if lr_path.exists() else pd.DataFrame()
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 5 coordinate-null closure audit: {row.dataset_key}",
                "status": row.dataset_coordinate_null_call,
                "key_result": (
                    f"reruns={int(row.n_completed_permutations)}, "
                    f"story-supported={int(row.n_story_coordinate_supported)}, "
                    f"true-story={int(row.n_true_story_coordinate_supported)}, "
                    f"high-expression-story={int(row.n_high_expression_story_supported)}, "
                    f"broad-background leakage={int(row.n_broad_background_leakage)}, "
                    f"top={row.top_story_coordinate_supported_lr if pd.notna(row.top_story_coordinate_supported_lr) and str(row.top_story_coordinate_supported_lr) else 'none'}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "end-to-end coordinate-rerun closure for selected story LR; whole-atlas coordinate discovery blocked by broad-background leakage",
            }
        )
    if not lr.empty:
        top = lr[lr["closure_tier"].astype(str).eq("A_story_coordinate_null_supported")]
        if len(top):
            rows.append(
                {
                    "phase12_component": "Goal 5 top coordinate-null supported story LR",
                    "status": "completed_story_coordinate_null_support",
                    "key_result": "; ".join(
                        f"{r.dataset_key}/{r.lr_pair}: BH={fmt_num(r.model_rerun_bh_fdr)}, deltaQ95={fmt_num(r.coordinate_null_delta_vs_q95)}"
                        for r in top.sort_values(["dataset_key", "model_rerun_group_statistic"], ascending=[True, False]).head(10).itertuples(index=False)
                    ),
                    "supporting_file": rel(lr_path),
                    "claim_level": "story-level coordinate-null support after high-expression guardrail",
                }
            )
    if not story.empty:
        s = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 5 coordinate-null closure overall",
                "status": s.overall_coordinate_null_closure_call,
                "key_result": (
                    f"datasets={int(s.n_datasets)}, "
                    f"reruns={int(s.n_completed_coordinate_reruns)}, "
                    f"supported datasets={int(s.n_datasets_with_story_coordinate_support)}, "
                    f"broad-background leakage={int(s.n_broad_background_leakage)}; "
                    f"{s.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "coordinate-null story closure; not whole-atlas coordinate FDR",
            }
        )
    return rows


def strict_story_candidate_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "strict_story_candidates/phase12_strict_story_candidate_dataset_summary.csv"
    evidence_path = PHASE12_DIR / "strict_story_candidates/phase12_strict_story_candidate_evidence.csv"
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 4 strict story-candidate triage: {row.dataset_key}",
                "status": "completed_story_triage",
                "key_result": (
                    f"story-guardrail candidates={int(row.n_story_guardrail_candidates)}, "
                    f"main-ready={int(row.n_main_story_ready)}, "
                    f"biological-consistency={int(row.n_biological_consistency_with_stat_support)}, "
                    f"targeted-missing-functional-closure={int(row.n_targeted_missing_functional_closure)}, "
                    f"top={row.top_candidate} ({row.top_candidate_class})"
                ),
                "supporting_file": rel(evidence_path),
                "claim_level": "manuscript story triage; integrates coordinate guardrail, targeted nulls, downstream support, baseline ranks and directionality; not whole-atlas FDR",
            }
        )
    return rows


def second_story_rows() -> list[dict[str, object]]:
    audit_path = PHASE12_DIR / "second_story/phase12_second_story_candidate_audit.csv"
    evidence_path = PHASE12_DIR / "second_story/phase12_second_story_evidence_matrix.csv"
    if not audit_path.exists() or not evidence_path.exists():
        return []
    audit = pd.read_csv(audit_path)
    evidence = pd.read_csv(evidence_path)
    rows = []
    ss = audit[audit["method"].eq("spatialself")].copy()
    if not ss.empty:
        ss = ss.sort_values(["story_score", "expected_flow_fraction", "target_top500_enrichment"], ascending=False)
        best = ss.iloc[0]
        n_supported = int(ss["story_suitability_class"].astype(str).str.contains("strong_biological_support", na=False).sum())
        n_clean = int(ss["story_suitability_class"].astype(str).str.contains("strong_second_story_candidate", na=False).sum())
        rows.append(
            {
                "phase12_component": "Goal 4 second biological story audit: lymph_node_full",
                "status": "completed_candidate_audit",
                "key_result": f"best candidate={best.lr_pair}; story class={best.story_suitability_class}; SpatialSelf rank={fmt_num(best.mean_score_rank)}, expected-flow={fmt_num(best.expected_flow_fraction)}, target enrichment={fmt_num(best.target_top500_enrichment)}, downstream FDR genes={fmt_num(best.n_impact_fdr_0_1)}; supported-but-negative-competitive candidates={n_supported}, clean candidates={n_clean}",
                "supporting_file": rel(audit_path),
                "claim_level": "second-story audit; biological consistency, not SpatialSelf-exclusive discovery",
            }
        )
    if not evidence.empty:
        rows.append(
            {
                "phase12_component": "Goal 4 second story evidence matrix",
                "status": "completed_candidate_matrix",
                "key_result": "; ".join(
                    f"{r.candidate_story}: {r.spatialself_story_class}"
                    for r in evidence.itertuples(index=False)
                ),
                "supporting_file": rel(evidence_path),
                "claim_level": "candidate prioritization table",
            }
        )
    return rows


def second_story_closure_rows() -> list[dict[str, object]]:
    summary_path = PHASE12_DIR / "second_story_closure/phase12_second_story_closure_summary.csv"
    evidence_path = PHASE12_DIR / "second_story_closure/phase12_second_story_closure_candidate_evidence.csv"
    if not summary_path.exists() or not evidence_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    evidence = pd.read_csv(evidence_path)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 4 second-story closure audit: {row.dataset_key}",
                "status": row.best_closure_class,
                "key_result": (
                    f"best={row.best_candidate}; score={int(row.best_closure_score)}; "
                    f"A candidates={int(row.n_A_candidates)}, B candidates={int(row.n_B_candidates)}; "
                    f"blocker={row.main_blocker}"
                ),
                "supporting_file": rel(evidence_path),
                "claim_level": "candidate-level second-story closure; observational and not perturbational causality",
            }
        )
    top_a = evidence[evidence["closure_class"].astype(str).eq("A_candidate_if_framed_conservatively")]
    if len(top_a):
        best = top_a.sort_values(["closure_score", "dataset_key"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 4 current non-CRC A-class closure candidate",
                "status": "conservative_A_candidate_available",
                "key_result": (
                    f"{best.dataset_key} {best.lr_pair}: rank={fmt_num(best.spatialself_rank)}, "
                    f"target FDR={fmt_num(best.downstream_bh_fdr)}, local/targeted FDR={fmt_num(best.local_receiver_bh_fdr)}, "
                    f"hard-negative/LR-identity FDR={fmt_num(best.lr_identity_bh_fdr)}, "
                    f"direction receiver q={fmt_num(best.direction_receiver_pairing_q)}"
                ),
                "supporting_file": rel(evidence_path),
                "claim_level": "use as conservative second biological story only; not SpatialSelf-exclusive discovery",
            }
        )
    return rows


def cross_platform_closure_rows() -> list[dict[str, object]]:
    dataset_path = PHASE12_DIR / "cross_platform_closure_matrix/phase12_cross_platform_closure_dataset_summary.csv"
    candidate_path = PHASE12_DIR / "cross_platform_closure_matrix/phase12_cross_platform_closure_candidate_matrix.csv"
    story_path = PHASE12_DIR / "cross_platform_closure_matrix/phase12_cross_platform_closure_story_summary.csv"
    if not dataset_path.exists() or not candidate_path.exists():
        return []
    dataset = pd.read_csv(dataset_path)
    candidate = pd.read_csv(candidate_path)
    story = pd.read_csv(story_path) if story_path.exists() else pd.DataFrame()
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 4 cross-platform closure matrix: {row.dataset_key}",
                "status": row.dataset_cross_platform_call,
                "key_result": (
                    f"best={row.best_candidate} ({row.best_closure_tier}), "
                    f"axes={int(row.best_axes_passed)}, "
                    f"A={int(row.n_A_candidates)}, B={int(row.n_B_candidates)}, "
                    f"supportive-or-better={int(row.n_supportive_or_better)}, "
                    f"use={row.recommended_use}"
                ),
                "supporting_file": rel(dataset_path),
                "claim_level": "cross-platform closure audit; separates conservative second story from supportive or failed platforms",
            }
        )
    if not story.empty:
        summary = story.iloc[0]
        rows.append(
            {
                "phase12_component": "Goal 4 cross-platform closure overall",
                "status": summary.overall_cross_platform_closure_call,
                "key_result": (
                    f"datasets={int(summary.n_datasets_audited)}, "
                    f"non-CRC={int(summary.n_non_crc_datasets_audited)}, "
                    f"A candidates={int(summary.n_A_candidates)}, "
                    f"B candidates={int(summary.n_B_candidates)}, "
                    f"supportive datasets={int(summary.n_datasets_supportive_or_better)}; "
                    f"{summary.claim_boundary}"
                ),
                "supporting_file": rel(story_path),
                "claim_level": "conservative cross-platform validation summary, not a claim that every platform is clean",
            }
        )
    top = candidate.sort_values(["closure_tier_rank", "n_validation_axes_passed", "spatialself_rank"], ascending=[True, False, True]).head(6)
    if len(top):
        rows.append(
            {
                "phase12_component": "Goal 4 top cross-platform closure candidates",
                "status": "completed_candidate_matrix",
                "key_result": "; ".join(
                    f"{r.dataset_key}/{r.candidate_lr_pair}:{r.closure_tier}, axes={int(r.n_validation_axes_passed)}, blocker={r.remaining_blocker}"
                    for r in top.itertuples(index=False)
                ),
                "supporting_file": rel(candidate_path),
                "claim_level": "candidate-level closure matrix for choosing manuscript stories",
            }
        )
    return rows


def cross_platform_story_rows() -> list[dict[str, object]]:
    audit_path = PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv"
    if not audit_path.exists():
        return []
    audit = pd.read_csv(audit_path)
    rows = []
    for row in audit.itertuples(index=False):
        rows.append(
            {
                "phase12_component": f"Goal 4 cross-platform story audit: {row.dataset_key}",
                "status": row.story_class,
                "key_result": f"best positive={row.best_positive_control} rank={fmt_num(row.best_positive_rank)}, best negative={row.best_negative_control} rank={fmt_num(row.best_negative_rank)}, rank status={row.rank_validation_status}; local contrast={row.local_contrast_status}; targeted positives={fmt_num(row.n_targeted_positive_significant)}, targeted negative leakage={fmt_num(row.n_targeted_negative_leakage)}; downstream={row.downstream_claim_status}",
                "supporting_file": rel(audit_path),
                "claim_level": "cross-platform story readiness audit",
            }
        )
    return rows


def figure_package_rows() -> list[dict[str, object]]:
    panel_path = PHASE12_DIR / "manuscript_figure_package/phase12_manuscript_figure_panel_index.csv"
    source_path = PHASE12_DIR / "manuscript_figure_package/phase12_manuscript_figure_source_manifest.csv"
    contract_path = PHASE12_DIR / "manuscript_figure_package/phase12_manuscript_figure_contract.md"
    if not panel_path.exists() or not source_path.exists():
        return []
    panels = pd.read_csv(panel_path)
    sources = pd.read_csv(source_path)
    empty = sources[pd.to_numeric(sources["n_rows"], errors="coerce").fillna(0).eq(0)]
    figure_counts = panels["figure_id"].value_counts().sort_index()
    counts_text = "; ".join(f"{k}={int(v)} panels" for k, v in figure_counts.items())
    rows = [
        {
            "phase12_component": "Goal 6 manuscript figure source-data package",
            "status": "completed_source_data_package",
            "key_result": f"figures={panels['figure_id'].nunique()}, panels={len(panels)}, source tables={len(sources)}, empty tables={len(empty)}; {counts_text}",
            "supporting_file": rel(panel_path),
            "claim_level": "figure source-data and panel contract; render QA is tracked separately",
        },
        {
            "phase12_component": "Goal 6 figure contract and review guardrails",
            "status": "completed_contract",
            "key_result": "core conclusion, panel map and review risks written for four planned main figures",
            "supporting_file": rel(contract_path),
            "claim_level": "manuscript figure planning; not visual QA",
        },
    ]
    rendered_path = PHASE12_DIR / "figures/phase12_rendered_figure_status.csv"
    if rendered_path.exists():
        rendered = pd.read_csv(rendered_path)
        n_complete = int(rendered["all_expected_outputs_exist"].sum()) if "all_expected_outputs_exist" in rendered else 0
        n_editable = int(rendered["svg_keeps_editable_text"].sum()) if "svg_keeps_editable_text" in rendered else 0
        rows.append(
            {
                "phase12_component": "Goal 6 rendered manuscript figure drafts",
                "status": "draft_figures_rendered",
                "key_result": f"rendered figure drafts={n_complete}/{len(rendered)}, editable SVG text={n_editable}/{len(rendered)}",
                "supporting_file": rel(rendered_path),
                "claim_level": "manuscript figure drafts rendered and QA-smoke checked; not final journal submission art",
            }
        )
    return rows


def claim_matrix_rows() -> list[dict[str, object]]:
    claim_path = PHASE12_DIR / "claim_matrix/phase12_claim_matrix.csv"
    if not claim_path.exists():
        return []
    claims = pd.read_csv(claim_path)
    unsupported_mask = claims["support_status"].astype(str).str.contains("not_supported|not_yet|do_not", na=False)
    supported = int((~unsupported_mask).sum())
    blocked = int(unsupported_mask.sum())
    main_cases = claims[
        claims["support_status"].astype(str).isin(
            [
                "main_case_supported_with_downstream_caveat",
                "biological_consistency_story",
                "supportive_second_platform_candidate",
            ]
        )
    ]["topic"].astype(str).tolist()
    return [
        {
            "phase12_component": "Goal 6 claim matrix and story consolidation",
            "status": "completed_conservative_claim_matrix",
            "key_result": f"supported or qualified claims={supported}; blocked/not-yet claims={blocked}; main usable story items: {'; '.join(main_cases[:5])}",
            "supporting_file": rel(claim_path),
            "claim_level": "manuscript framing and claim boundary",
        }
    ]


def final_limitations_next_model_rows() -> list[dict[str, object]]:
    path = PHASE12_DIR / "final_limitations_next_model/phase12_final_limitations_next_model.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    blocked = int(table["status"].astype(str).str.contains("not_closed|primary_bottleneck|quantified_not_solved|active_partial", na=False).sum())
    candidate = int(table["status"].astype(str).str.contains("candidate|hold_out|current_reporting", na=False).sum())
    return [
        {
            "phase12_component": "Phase 12 final limitations and next-model requirements",
            "status": "completed_next_model_guardrail",
            "key_result": (
                f"{len(table)} decision rows; unresolved/blocking items={blocked}; "
                f"model-variant decision rows={candidate}; broad whole-atlas discovery FDR remains the main open requirement"
            ),
            "supporting_file": rel(path),
            "claim_level": "final Phase 12 claim boundary and next-model specification; not evidence of complete Goal 1 closure",
        }
    ]


def write_markdown(table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Current Status",
        "",
        "## What Is Done",
        "",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"- **{row.phase12_component}**: `{row.status}`")
        lines.append(f"  - {row.key_result}")
        lines.append(f"  - file: `{row.supporting_file}`")
        lines.append(f"  - claim level: {row.claim_level}")
    lines += [
        "",
        "## Current Interpretation",
        "",
        "- Phase 12 now has whole-LR input coverage, hard-negative templates, a naive atlas smoke baseline, and a real SpatialSelf wavelet atlas pilot.",
        "- The naive expression-distance smoke is heavily contaminated by high-expression negatives, which validates the need for hard-negative stress testing.",
        "- The SpatialSelf model pilot is runnable on full-transcriptome backgrounds. Post-Xenium colon retains the SPP1-CD44 story in the top ranks, but Visium HD CRC still shows strong high-expression negative leakage.",
        "- The atlas quality and hard-negative stress summaries now quantify that leakage instead of treating it as anecdotal.",
        "- The atlas FDR/contamination audit now summarizes top-k control-panel empirical FDR, high-expression/receptor-shared risk, and positive-control hard-negative FDR in one place.",
        "- The atlas FDR closure audit now makes the remaining Goal 1 blocker explicit: target-decoy/control FDR is informative, but strict whole-atlas permutation FDR does not support positive all-LR discoveries.",
        "- The spatial-null bottleneck map now turns that blocker into concrete calibration targets: preserve matched-spatial true signals in lymph node/post-Xenium, while suppressing broad category-balanced background LR especially in CRC.",
        "- A background-suppressed ranking variant now tests that calibration direction: it improves lymph-node top10 and is a post-Xenium candidate, but it is not adopted globally because CRC and broader top-k diagnostics remain caveated.",
        "- The Goal 1 model-variant scoreboard now consolidates main, range-null and background-suppressed variants: main calibrated remains the reporting baseline, background suppression is a next-model component, and no tested variant closes whole-atlas FDR.",
        "- The exploratory range-null calibration variant is documented but not adopted: it improves top10 true recovery in some datasets, while worsening broader control-risk diagnostics.",
        "- The receptor-shared atlas audit now checks same-receptor decoys directly on the full LR ranking: top10 receptor-shared contamination is clean in the expanded atlas audit, but lower-ranked anchors still have limited pairwise same-receptor leakage.",
        "- The expanded whole-transcriptome downstream scan adds receiver-side target-rank evidence across post-Xenium colon, human lymph node and Visium HD CRC. Support is concentrated in selected LR, not every predicted LR, and CRC remains a weak/negative stress case at this layer.",
        "- The target-background audit now explicitly tests whether curated target genes rank above the full expressed-gene background for selected top/priority/module LR: post-Xenium has strong support, lymph node has selected CXCL12-CXCR4 support, and high-expression target leakage is not observed in this layer.",
        "- The gene-program enrichment layer now summarizes LR and module receiver programs, and the new expanded SpatialSelf-vs-COMMOT downstream comparison shows post-Xenium anchors favor SpatialSelf while lymph-node CCL19/CCL21-CCR7 favor COMMOT.",
        "- The program-specificity audit now separates broad housekeeping-like enrichment from immune/ECM/migration/signaling terms and flags negative-specific leakage; the expanded pathway-specific layer should be treated as a guardrail, not clean downstream closure.",
        "- Directionality is now quantified from Phase 11 trajectories. It supports lymph-node chemokine/source-sink consistency, but does not show SpatialSelf beating COMMOT globally.",
        "- The directionality statistical audit now separates empirical expected-flow FDR, aggregated label-permutation p-values, fixed-score edge-level permutation, and a low-permutation coordinate model-rerun smoke.",
        "- The directionality consensus audit now merges those p-value layers into manuscript-safe calls: selected lymph-node source-sink claims are supported, but strict edge-vector consensus and global SpatialSelf-over-COMMOT directionality are not.",
        "- The coordinate-null closure audit now converts end-to-end coordinate model-rerun smoke into story-level evidence: selected story LR pass coordinate rerun plus guardrails, while broad-background leakage blocks whole-atlas coordinate discovery claims.",
        "- Official adapter-light COMMOT, CellChat, SpaTalk, and CellNEST baselines now run on the exact Phase 12 expanded LR universe. COMMOT/CellChat/CellNEST are high-expression contaminated in top10; SpaTalk is cleaner but low recall.",
        "- The second-story closure audit supports lymph-node CXCL12-CXCR4 as the current conservative non-CRC A-candidate; CCL19/CCL21-CCR7 remain supportive chemokine biology but are not SpatialSelf-exclusive because COMMOT is also strong.",
        "- The cross-platform closure matrix now closes Goal 4 conservatively: lymph-node CXCL12-CXCR4 is the A-class non-CRC second story, tonsil A1/D1 provide supportive platform validation, and melanoma remains not clean.",
        "- The cross-platform audit keeps post-Xenium colon as the strongest main case. SpaMosaic tonsil A1 is the best current non-CRC support case, but it still has targeted negative leakage and no clean pathway-specific downstream claim.",
        "- Native melanoma should not be used as a success story under the current control panel.",
        "- The claim matrix now separates claims we can write, claims that need caveats, and claims that should not be made yet.",
        "- Therefore the current result is not mature enough to claim whole-atlas superiority. The next required work is formal empirical FDR, stronger hard-negative calibration, and larger atlas-scale validation.",
        "",
        "## Remaining Phase 12 Work",
        "",
        "- Run final whole-LR or larger atlas scoring with a pre-registered LR set and runtime caps.",
        "- Add empirical FDR for atlas LR ranks, not only pilot hard-negative rank comparisons.",
        "- Improve calibration/model or add stronger end-to-end nulls before making a final positive whole-atlas discovery-FDR or whole-atlas superiority claim.",
        "- Use the spatial-null bottleneck map to separate two next model tasks: range-matched spatial null preservation for real anchors, and broad category-balanced background suppression.",
        "- Treat background suppression as a candidate component for the next model version, then rerun exact-score permutation/control audits before any discovery-FDR claim.",
        "- Use the model-variant scoreboard as the decision guardrail before replacing the current main calibrated ranking.",
        "- Do not replace the current calibrated expanded-atlas ranking with the range-null-calibrated variant unless a future version improves true recovery without raising top-k or top100 control risk.",
        "- Use receptor-shared atlas leakage as a calibration target for lower-ranked anchors, especially where same-receptor decoys beat weak positives.",
        "- Expand curated target resources and receiver-gene association only if we want pan-atlas target support rather than selected LR/module support.",
        "- Improve pathway-specific downstream interpretation by separating immune/chemokine biology from broad translation/ribosome or library-size-like programs.",
        "- Tighten the program-specificity audit further with curated exclusion lists and per-dataset negative-control calibration before using it as a main figure.",
        "- Strengthen the lymph-node CXCL12-CXCR4 second story with cleaner receptor-shared/null calibration if we want it to be more than conservative biological consistency.",
        "- For stronger cross-platform support beyond the conservative Goal 4 closure, prioritize SpaMosaic tonsil A1/D1 with improved downstream target panels or stricter targeted FDR that removes negative leakage.",
        "- Scale the coordinate model-rerun smoke beyond the current story-level closure if we want whole-atlas coordinate-null FDR rather than selected-story coordinate-null support.",
        "- Replace adapter-light official baselines with fully tuned official end-to-end runs if we want a stronger method-superiority claim.",
        "- Polish manuscript-grade figures only after those evidence gaps are stable.",
        "- Use the claim matrix and manuscript figure source-data package as guardrails before treating the Phase 12 draft figures as final manuscript art.",
    ]
    (OUT_DIR / "phase12_current_status_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt_num(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def fmt_percent(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100 * val:.1f}%"


def fmt_text(value: object) -> str:
    if value is None:
        return "none"
    try:
        if pd.isna(value):
            return "none"
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else "none"


if __name__ == "__main__":
    main()
