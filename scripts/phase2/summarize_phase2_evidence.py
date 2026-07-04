#!/usr/bin/env python
"""Build the current Phase 2 evidence matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/phase2_evidence_matrix",
    )
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_evidence_matrix()
    fieldnames = ["goal_id", "goal", "evidence_status", "primary_artifacts", "key_findings", "next_steps"]
    csv_path = output_dir / "phase2_evidence_matrix.csv"
    json_path = output_dir / "phase2_evidence_matrix.json"
    md_path = output_dir / "phase2_evidence_matrix.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(rows), encoding="utf-8")
    print(f"Phase 2 evidence CSV:  {csv_path}")
    print(f"Phase 2 evidence JSON: {json_path}")
    print(f"Phase 2 evidence MD:   {md_path}")
    print(f"Rows: {len(rows)}")


def build_evidence_matrix() -> list[dict[str, str]]:
    return [
        _goal1_row(),
        _goal2_row(),
        _goal3_row(),
        _goal4_row(),
        _goal5_row(),
        _goal6_row(),
    ]


def _goal1_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery"
    smoke = _read_csv(root / "smoke_controlled_proxy_v2/weak_signal_recovery_summary.csv")
    controlled = _read_csv(root / "controlled_five_replicate_initial/weak_signal_recovery_summary.csv")
    full_slide = _read_csv(root / "full_slide_sparse_initial/weak_signal_recovery_summary.csv")
    threshold = _read_csv(root / "full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv")
    lr_balanced = _read_csv(root / "full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv")
    lr_balanced_stability = _read_csv(
        root / "controlled_lr_balanced_threshold_stability/lr_balanced_replicate_aggregate.csv"
    )
    mechanism_threshold = _read_csv(
        root / "full_slide_sparse_mechanism_threshold_tradeoff/mechanism_threshold_tradeoff_summary.csv"
    )
    mechanism_probe = _read_csv(root / "full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_summary.csv")
    mechanism_overview = _first(_read_csv(root / "controlled_five_replicate_mechanism_summary/mechanism_weak_signal_overview.csv"))
    mechanism_q94 = _first(_read_csv(root / "full_slide_sparse_operating_q94_mechanism_summary/mechanism_weak_signal_overview.csv"))
    best = _best_candidate(controlled)
    edge = _first_row(controlled, "ablation", "wavelet_mechanism_niche_edge_prior_ranking")
    default = _first_row(controlled, "ablation", "wavelet_mechanism_niche_lr_prior_lrwise_ranking")
    best_smoke = _best_candidate(smoke)
    full_edge = _first_row(full_slide, "ablation", "wavelet_mechanism_niche_edge_prior_ranking")
    threshold_edge = _first_row(threshold, "label", "edge_prior_ranking")
    lr_balanced_edge = _first_row(lr_balanced, "strategy", "lrwise_quantile_with_global_floor")
    lr_balanced_controlled = _first_row(lr_balanced_stability, "strategy", "lrwise_quantile_with_global_floor")
    long_range_threshold = _first_row(mechanism_threshold, "mechanism", "long_range_chemokine_niche")
    combined_threshold = _first_row(mechanism_threshold, "strategy", "combined_mechanism_specific")
    long_range_probe = _first_row(mechanism_probe, "ablation", "phase2_long_range_soft_weak_edge_prior_wavelet")
    non_contact_probe = _first_row(mechanism_probe, "ablation", "phase2_non_contact_soft_weak_edge_prior_wavelet")
    status = "initial_evidence" if controlled and full_slide and threshold else "planned"
    return {
        "goal_id": "1",
        "goal": "Weak-signal recovery model improvement",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal1_weak_signal_recovery/smoke_controlled_proxy_v2/weak_signal_recovery_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/weak_signal_recovery_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/controlled_lr_balanced_threshold_stability/lr_balanced_replicate_aggregate.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_threshold_tradeoff/mechanism_threshold_tradeoff_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/goal1_mechanism_weak_signal_combined_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_mechanism_summary/mechanism_weak_signal_summary.csv",
                "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_operating_q94_mechanism_summary/mechanism_weak_signal_summary.csv",
                "scripts/benchmark/run_spatialself.py",
                "scripts/phase2/run_goal1_weak_signal_recovery.py",
                "scripts/phase2/run_goal1_threshold_tradeoff.py",
                "scripts/phase2/run_goal1_lr_balanced_threshold.py",
                "scripts/phase2/run_goal1_lr_balanced_replicate_stability.py",
                "scripts/phase2/run_goal1_mechanism_threshold_tradeoff.py",
                "scripts/phase2/run_goal1_mechanism_weak_signal_summary.py",
            ]
        ),
        "key_findings": (
            f"Smoke best={best_smoke.get('ablation', 'NA')} weak recall="
            f"{_fmt(best_smoke, 'weak_signal_recall_at_threshold_mean')}; "
            f"5-replicate default weak recall={_fmt(default, 'weak_signal_recall_at_threshold_mean')}, "
            f"edge-prior={_fmt(edge, 'weak_signal_recall_at_threshold_mean')}, "
            f"best={best.get('ablation', 'NA')} weak recall={_fmt(best, 'weak_signal_recall_at_threshold_mean')}; "
            f"best AUPR={_fmt(best, 'aupr_mean')}, FDR={_fmt(best, 'empirical_fdr_at_threshold_mean')}, "
            f"FP rate={_fmt(best, 'false_positive_high_expression_rate_at_threshold_mean')}. "
            f"Full-slide edge-prior q0.95 weak recall={_fmt(full_edge, 'weak_signal_recall_at_threshold_mean')}; "
            f"threshold tradeoff best q={_fmt(threshold_edge, 'best_quantile')}, "
            f"weak recall={_fmt(threshold_edge, 'best_weak_signal_recall')}, "
            f"FDR={_fmt(threshold_edge, 'best_fdr')}, "
            f"FP rate={_fmt(threshold_edge, 'best_false_positive_high_expression_rate')}. "
            f"LR-balanced operating diagnostic best strategy={lr_balanced_edge.get('strategy', 'NA')}, "
            f"LR q={_fmt(lr_balanced_edge, 'best_lr_quantile')}, global floor="
            f"{_fmt(lr_balanced_edge, 'best_global_floor_quantile')}, weak recall="
            f"{_fmt(lr_balanced_edge, 'best_weak_signal_recall')}, FDR="
            f"{_fmt(lr_balanced_edge, 'best_fdr')}, FDR delta="
            f"{_fmt(lr_balanced_edge, 'best_fdr_delta_vs_reference')}, FP rate="
            f"{_fmt(lr_balanced_edge, 'best_false_positive_high_expression_rate')}. "
            f"Controlled LR-balanced stability did not pass promotion constraints: pass fraction="
            f"{_fmt(lr_balanced_controlled, 'passing_replicate_fraction')}, weak recall mean="
            f"{_fmt(lr_balanced_controlled, 'best_weak_signal_recall_mean')}, FDR delta mean="
            f"{_fmt(lr_balanced_controlled, 'best_fdr_delta_vs_reference_mean')}, FP rate mean="
            f"{_fmt(lr_balanced_controlled, 'best_false_positive_high_expression_rate_mean')}. "
            f"Mechanism diagnostic controlled edge-prior weak recall="
            f"{mechanism_overview.get('edge_prior_mechanism_weak_recall', 'NA')}; "
            f"full-slide q0.94 edge-prior weak recall="
            f"{mechanism_q94.get('edge_prior_mechanism_weak_recall', 'NA')}. "
            f"Mechanism-specific threshold diagnostic: long-range q="
            f"{_fmt(long_range_threshold, 'threshold_quantile')} weak recall="
            f"{_fmt(long_range_threshold, 'weak_signal_recall_at_threshold')}, "
            f"FDR delta={_fmt(long_range_threshold, 'fdr_delta_vs_reference')}; "
            f"combined mechanism-specific weak recall="
            f"{_fmt(combined_threshold, 'weak_signal_recall_at_threshold')} with FDR="
            f"{_fmt(combined_threshold, 'empirical_fdr_at_threshold')}. "
            f"Mechanism-targeted training probes did not improve transfer: long-range soft weak recall="
            f"{_fmt(long_range_probe, 'weak_signal_recall_at_threshold_mean')}, "
            f"non-contact soft weak recall={_fmt(non_contact_probe, 'weak_signal_recall_at_threshold_mean')}. "
            f"Pass flag={best.get('phase2_goal1_candidate_pass', 'NA')}; "
            f"full-slide threshold status={threshold_edge.get('status', 'NA')}."
        ),
        "next_steps": (
            "Promote edge-prior ranking plus calibrated operating point into Goal 3 model selection; avoid "
            "soft weak-positive mining on full-slide unless the training objective is redesigned. Treat "
            "treat the LR-balanced threshold as a full-slide post-hoc diagnostic only because controlled "
            "cross-replicate stability fails the FDR/FP constraints. Keep global q0.94 as the safer current "
            "Goal 3 operating point."
        ),
    }


def _planned_row(goal_id: str, goal: str) -> dict[str, str]:
    return {
        "goal_id": goal_id,
        "goal": goal,
        "evidence_status": "planned",
        "primary_artifacts": "",
        "key_findings": "No Phase 2 result artifact yet.",
        "next_steps": "Implement after Goal 1 candidate narrowing.",
    }


def _goal6_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase2/goal6_paper_figures"
    summary = _first(_read_csv(root / "goal6_figure_assembly_summary.csv"))
    figure_qa = _first(_read_csv(root / "figure_qa/goal6_figure_qa_summary.csv"))
    manifest = _read_json(root / "goal6_figure_assembly_manifest.json")
    source_tables = manifest.get("source_tables", {}) if isinstance(manifest.get("source_tables", {}), dict) else {}
    figures = manifest.get("figures", []) if isinstance(manifest.get("figures", []), list) else []
    manuscript_panels = (
        manifest.get("manuscript_panels", [])
        if isinstance(manifest.get("manuscript_panels", []), list)
        else []
    )
    status = summary.get("evidence_status", "planned") if summary else "planned"
    return {
        "goal_id": "6",
        "goal": "Paper-ready result tables and figures",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal6_paper_figures/goal6_figure_assembly_summary.csv",
                "outputs/phase2/goal6_paper_figures/goal6_figure_assembly_manifest.json",
                "outputs/phase2/goal6_paper_figures/figure_qa/goal6_figure_qa_summary.csv",
                "outputs/phase2/goal6_paper_figures/figure_qa/goal6_figure_file_qa.csv",
                "outputs/phase2/goal6_paper_figures/figure_qa/goal6_figure_source_trace_qa.csv",
                "outputs/phase2/goal6_paper_figures/figure_qa/goal6_figure_qa_manifest.json",
                "outputs/phase2/goal6_paper_figures/figures/phase2_overview.png",
                "outputs/phase2/goal6_paper_figures/figures/phase2_overview.pdf",
                "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/lymph_node_validation_summary.csv",
                "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/lymph_node_validation_manifest.json",
                "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/baseline_matched_validation_summary.csv",
                "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/baseline_matched_validation_manifest.json",
                "outputs/phase2/goal6_paper_figures/manuscript_panels/",
                "outputs/phase2/goal6_paper_figures/source_tables/goal1_controlled_weak_signal.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal1_mechanism_weak_signal.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal1_full_slide_operating_point.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal2_direct_group_fdr.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal3_model_selection.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal4_high_resolution_controls.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal4_high_resolution_scaling.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal4_tonsil_baseline_matched.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal5_interpretability.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal5_interpretability_n1800.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal5_mechanism_category.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal5_lymph_node_interpretability.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal5_interpretability_synthesis.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_controlled_baseline_comparison.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_full_slide_baseline_comparison.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_component_ablation.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_lymph_node_global_validation.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_lymph_node_baseline_matched.csv",
                "outputs/phase2/goal6_paper_figures/source_tables/goal6_claim_traceability.csv",
                "scripts/phase2/run_goal6_lymph_node_validation.py",
                "scripts/phase2/run_goal6_lymph_node_baseline_validation.py",
                "scripts/phase2/run_goal6_figure_assembly.py",
                "scripts/phase2/run_goal6_figure_qa.py",
            ]
        )
        if summary
        else "",
        "key_findings": (
            f"Initial Goal 6 assembly generated {summary.get('n_source_tables', 'NA')} source tables "
            f"and {summary.get('n_figure_files', 'NA')} figure files, including "
            f"{summary.get('n_manuscript_panel_files', len(manuscript_panels))} manuscript-panel files. "
            f"Source table keys={','.join(sorted(source_tables))}; "
            f"overview figures={','.join(figures)}; "
            f"panel file count={len(manuscript_panels)}. "
            f"This is an overview/source-table artifact and explicitly records current limitations: "
            f"Goal 4 is subset-level, Goal 5 synthesis now includes lymph-node injected-control downstream/rank "
            f"support but still constrains broad interpretability to mixed initial support, and baseline rows carry "
            f"explicit evidence-level limitations. "
            f"Component ablation is included as Phase 1 controlled "
            f"component-level evidence, not final Phase 2 default ablation. Lymph node validation is "
            f"now regenerated by a Phase 2 runner as full-slide sparse real-background injected-control evidence, "
            f"not unlabeled real-data ground truth; a baseline-matched validation panel compares retained "
            f"candidate-edge/projection baselines under the same injected-control metrics. "
            f"Figure QA status={figure_qa.get('evidence_status', 'NA') if figure_qa else 'NA'}, "
            f"qa_pass={figure_qa.get('qa_pass', 'NA') if figure_qa else 'NA'}, "
            f"export bundle={figure_qa.get('export_bundle_status', 'NA') if figure_qa else 'NA'}, "
            f"missing figures={figure_qa.get('n_missing_figure_files', 'NA') if figure_qa else 'NA'}, "
            f"missing sources={figure_qa.get('n_missing_source_tables', 'NA') if figure_qa else 'NA'}."
        )
        if summary
        else "No Phase 2 result artifact yet.",
        "next_steps": (
            "Add manual visual review, SVG/TIFF export, final journal sizing, and larger high-resolution validation "
            "panels before treating Goal 6 as paper-ready."
        )
        if summary
        else "Implement after Goal 5 interpretability evidence.",
    }


def _goal4_row() -> dict[str, str]:
    stream_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset"
    fresh_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset"
    scaling_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated"
    baseline_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset"
    stream_profile = _first(_read_csv(stream_root / "high_resolution_dataset_profile.csv"))
    stream_summary = _first(_read_csv(stream_root / "goal4_high_resolution_summary.csv"))
    fresh_profile = _first(_read_csv(fresh_root / "tonsil_fresh_dataset_profile.csv"))
    fresh_summary = _first(_read_csv(fresh_root / "tonsil_fresh_goal4_summary.csv"))
    fresh_controls = _read_csv(fresh_root / "tonsil_fresh_control_summary.csv")
    scaling_summary = _first(_read_csv(scaling_root / "tonsil_scaling_summary.csv"))
    baseline_summary = _first(_read_csv(baseline_root / "tonsil_baseline_matched_summary.csv"))
    baseline_methods = _read_csv(baseline_root / "tonsil_baseline_matched_method_summary.csv")
    module_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset"
    module_overview = _first(_read_csv(module_root / "tonsil_ccc_module_overview.csv"))
    module_enrichment = _read_csv(module_root / "tonsil_ccc_module_control_enrichment.csv")
    module_positive = _first_row(module_enrichment, "control_class", "tonsil_positive_prior")
    module_negative = _first_row(module_enrichment, "control_class", "housekeeping_negative_control")
    module_n1800_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset"
    module_n1800_overview = _first(_read_csv(module_n1800_root / "tonsil_ccc_module_overview.csv"))
    module_n1800_summary = _read_csv(module_n1800_root / "tonsil_ccc_module_summary.csv")
    module_n1800_enrichment = _read_csv(module_n1800_root / "tonsil_ccc_module_control_enrichment.csv")
    module_n1800_positive = _first_row(module_n1800_enrichment, "control_class", "tonsil_positive_prior")
    module_n1800_negative = _first_row(module_n1800_enrichment, "control_class", "housekeeping_negative_control")
    module_n1800_mixed = [
        row for row in module_n1800_summary
        if _to_float(row.get("n_positive_controls")) > 0
        and _to_float(row.get("n_housekeeping_negative_controls")) > 0
    ]
    baseline_spatialself = _first_row(baseline_methods, "method", "spatialself_wavelet_calibrated")
    baseline_commot = _first_row(baseline_methods, "method", "commot")
    baseline_cellchat = _first_row(baseline_methods, "method", "cellchat")
    positives = [
        row for row in fresh_controls if row.get("control_class") == "tonsil_positive_prior"
    ]
    negatives = [
        row for row in fresh_controls if row.get("control_class") != "tonsil_positive_prior"
    ]
    best_positive = min(positives, key=lambda row: _to_float(row.get("rank_by_score_mean")), default={})
    best_negative = min(negatives, key=lambda row: _to_float(row.get("rank_by_score_mean")), default={})
    status = fresh_summary.get("evidence_status", "") or stream_summary.get("evidence_status", "planned")
    return {
        "goal_id": "4",
        "goal": "High-resolution real-data extension",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_dataset_profile.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_goal4_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_lr_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_cell_type_pair_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_control_summary.csv",
                "scripts/phase2/run_goal4_fresh_tonsil_subset.py",
                "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/high_resolution_dataset_profile.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/goal4_high_resolution_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/tonsil_lr_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/tonsil_cell_type_pair_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/tonsil_positive_control_summary.csv",
                "scripts/phase2/run_goal4_high_resolution_tonsil_subset.py",
                "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_run_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_manifest.json",
                "scripts/phase2/run_goal4_high_resolution_scaling.py",
                "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_method_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_status.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_manifest.json",
                "scripts/phase2/run_goal4_tonsil_baseline_matched.py",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/tonsil_ccc_module_overview.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/tonsil_ccc_module_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/tonsil_ccc_module_control_enrichment.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_overview.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_summary.csv",
                "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_control_enrichment.csv",
                "scripts/phase2/run_goal4_tonsil_ccc_modules.py",
            ]
        )
        if (fresh_summary or stream_summary)
        else "",
        "key_findings": (
            f"Fresh Phase 2 default tonsil subset analyzes {fresh_profile.get('n_cells', 'NA')} cells, "
            f"{fresh_profile.get('n_lr_pairs', 'NA')} LR pairs, and "
            f"{fresh_profile.get('n_candidate_edges', 'NA')} candidate edges. "
            f"Wall-clock={_fmt(fresh_profile, 'wall_clock_seconds')}s, peak RSS="
            f"{_fmt(fresh_profile, 'peak_rss_mb')} MB. "
            f"Positive controls top decile={fresh_summary.get('n_positive_controls_top_decile', 'NA')}; "
            f"negative controls top decile={fresh_summary.get('n_negative_controls_top_decile', 'NA')}; "
            f"best positive={best_positive.get('control_name', 'NA')} rank="
            f"{best_positive.get('rank_by_score_mean', 'NA')}; "
            f"best negative={best_negative.get('control_name', 'NA')} rank="
            f"{best_negative.get('rank_by_score_mean', 'NA')}; "
            f"top cell-type pairs={fresh_summary.get('top_cell_type_pairs', 'NA')}. "
            f"Existing stream bridge also processed {stream_profile.get('n_prediction_rows_processed', 'NA')} "
            f"rows across {stream_profile.get('n_cells', 'NA')} cells with node-order check="
            f"{stream_profile.get('all_header_cells_match_spatial', 'NA')}. "
            f"Subset scaling reaches {scaling_summary.get('n_cells_max', 'NA')} cells and "
            f"{scaling_summary.get('n_candidate_edges_max', 'NA')} candidate edges; largest run "
            f"wall-clock={_fmt(scaling_summary, 'largest_run_wall_clock_seconds')}s, peak RSS="
            f"{_fmt(scaling_summary, 'largest_run_peak_rss_mb')} MB, positive/negative top-decile controls="
            f"{scaling_summary.get('largest_run_positive_controls_top_decile', 'NA')}/"
            f"{scaling_summary.get('largest_run_negative_controls_top_decile', 'NA')}. "
            f"Same-subset baseline comparison completed {baseline_summary.get('n_methods_completed', 'NA')} methods; "
            f"SpatialSelf pass={baseline_spatialself.get('rank_validation_pass', 'NA')} with top-decile pos/neg="
            f"{baseline_spatialself.get('n_positive_controls_top_decile', 'NA')}/"
            f"{baseline_spatialself.get('n_negative_controls_top_decile', 'NA')}; "
            f"COMMOT pass={baseline_commot.get('rank_validation_pass', 'NA')}; "
            f"CellChat pass={baseline_cellchat.get('rank_validation_pass', 'NA')}. "
            f"CCC module diagnostic finds {module_overview.get('n_modules', 'NA')} modules; "
            f"positive-control modules={module_overview.get('n_modules_with_positive_controls', 'NA')}; "
            f"housekeeping-negative modules={module_overview.get('n_modules_with_housekeeping_negatives', 'NA')}; "
            f"best positive module={module_overview.get('best_positive_module', 'NA')} with "
            f"{module_overview.get('best_positive_module_positive_controls', 'NA')} positives and dominant pair="
            f"{module_overview.get('best_positive_module_dominant_pair', 'NA')}; "
            f"positive-control module BH-FDR={_fmt(module_positive, 'bh_fdr')}; "
            f"housekeeping-negative module BH-FDR={_fmt(module_negative, 'bh_fdr')}. "
            f"n1800 module stress test finds {module_n1800_overview.get('n_modules', 'NA')} modules; "
            f"positive-control module BH-FDR={_fmt(module_n1800_positive, 'bh_fdr')}; "
            f"housekeeping-negative module BH-FDR={_fmt(module_n1800_negative, 'bh_fdr')}; "
            f"mixed positive+housekeeping modules={len(module_n1800_mixed)}. "
            f"This comparison is control-rank validation, not ground-truth AUROC/AUPR."
        )
        if (fresh_summary or stream_summary)
        else "No Phase 2 result artifact yet.",
        "next_steps": (
            "Treat the fresh tonsil run as initial subset evidence only; expand to a larger tile/full data, "
            "add downstream/pathway validation, and compare against retained baselines before manuscript claims. "
            "The CCC module result supports mechanism/control separation but remains single-subset and weak on "
            "spatial confinement; the n1800 module stress test shows positive/housekeeping module separation can weaken."
        )
        if (fresh_summary or stream_summary)
        else "Implement after Goal 1 candidate narrowing.",
    }


def _goal5_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset"
    n1800_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_n1800_subset"
    embedding_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset"
    mechanism_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset"
    lymph_node_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lymph_node_real_background"
    synthesis_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/synthesis"
    summary = _first(_read_csv(root / "goal5_interpretability_summary.csv"))
    n1800_summary = _first(_read_csv(n1800_root / "goal5_interpretability_summary.csv"))
    n1800_controls = _read_csv(n1800_root / "tonsil_interpretability_control_summary.csv")
    controls = _read_csv(root / "tonsil_interpretability_control_summary.csv")
    downstream = _read_csv(root / "tonsil_downstream_target_correlation.csv")
    embedding_summary = _first(_read_csv(embedding_root / "goal5_lr_embedding_summary.csv"))
    mechanism_summary = _first(_read_csv(mechanism_root / "goal5_mechanism_category_summary.csv"))
    lymph_node_summary = _first(_read_csv(lymph_node_root / "lymph_node_goal5_interpretability_summary.csv"))
    synthesis_summary = _first(_read_csv(synthesis_root / "interpretability_synthesis_summary.csv"))
    synthesis_claims = _read_csv(synthesis_root / "interpretability_synthesis_claims.csv")
    positive = _first_row(controls, "control_class", "tonsil_positive_prior")
    housekeeping = _first_row(controls, "control_class", "housekeeping_negative_control")
    background = _first_row(controls, "control_class", "background_lr")
    n1800_positive = _first_row(n1800_controls, "control_class", "tonsil_positive_prior")
    n1800_housekeeping = _first_row(n1800_controls, "control_class", "housekeeping_negative_control")
    n1800_background = _first_row(n1800_controls, "control_class", "background_lr")
    synthesis_statuses = sorted({row.get("support_status", "NA") for row in synthesis_claims}) if synthesis_claims else []
    status = summary.get("evidence_status", "planned") if summary else "planned"
    return {
        "goal_id": "5",
        "goal": "Biological interpretability",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/goal5_interpretability_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_downstream_target_correlation.csv",
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_interpretability_control_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_pathway_category_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_ccc_module_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_lr_activity_profiles.csv",
                "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/goal5_interpretability_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_downstream_target_correlation.csv",
                "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_interpretability_control_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_pathway_category_summary.csv",
                "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_ccc_module_summary.csv",
                "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/goal5_lr_embedding_summary.csv",
                "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/lr_embedding_coordinates.csv",
                "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/lr_embedding_cluster_summary.csv",
                "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/lr_embedding_cluster_enrichment.csv",
                "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/goal5_mechanism_category_summary.csv",
                "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/mechanism_interpretability_summary.csv",
                "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/category_interpretability_summary.csv",
                "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/mechanism_category_pairwise_tests.csv",
                "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_summary.csv",
                "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_control_interpretability.csv",
                "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_claims.csv",
                "outputs/phase2/goal5_interpretability/synthesis/interpretability_synthesis_summary.csv",
                "outputs/phase2/goal5_interpretability/synthesis/interpretability_synthesis_claims.csv",
                "outputs/phase2/goal5_interpretability/synthesis/interpretability_synthesis_metrics.csv",
                "scripts/phase2/run_goal5_tonsil_interpretability.py",
                "scripts/phase2/run_goal5_lr_embedding_interpretability.py",
                "scripts/phase2/run_goal5_mechanism_category_interpretability.py",
                "scripts/phase2/run_goal5_lymph_node_interpretability.py",
                "scripts/phase2/run_goal5_interpretability_synthesis.py",
            ]
        )
        if summary
        else "",
        "key_findings": (
            f"Tonsil fresh subset downstream validation tested {summary.get('n_tested_lr_pairs', 'NA')}/"
            f"{summary.get('n_lr_pairs', 'NA')} LR pairs with {summary.get('n_permutations', 'NA')} permutations. "
            f"Raw mean Spearman positive/housekeeping/background="
            f"{_fmt(summary, 'positive_control_mean_spearman')}/"
            f"{_fmt(summary, 'housekeeping_negative_mean_spearman')}/"
            f"{_fmt(summary, 'background_lr_mean_spearman')}; "
            f"receiver-cell-type residual mean Spearman positive/housekeeping/background="
            f"{_fmt(summary, 'positive_control_mean_residual_spearman')}/"
            f"{_fmt(summary, 'housekeeping_negative_mean_residual_spearman')}/"
            f"{_fmt(summary, 'background_lr_mean_residual_spearman')}. "
            f"Residual significant controls positive/negative="
            f"{summary.get('significant_positive_controls_residual_fdr_0_10', 'NA')}/"
            f"{summary.get('significant_negative_controls_residual_fdr_0_10', 'NA')}. "
            f"Top raw downstream LR={summary.get('top_lr_by_downstream_spearman', 'NA')}; "
            f"top residual LR={summary.get('top_lr_by_residual_downstream_spearman', 'NA')}. "
            f"Control table means: positive residual={_fmt(positive, 'mean_residual_spearman')}, "
            f"housekeeping residual={_fmt(housekeeping, 'mean_residual_spearman')}, "
            f"background residual={_fmt(background, 'mean_residual_spearman')}. "
            f"Larger 1,800-cell tonsil stress test tested "
            f"{n1800_summary.get('n_tested_lr_pairs', 'NA') if n1800_summary else 'NA'} LR pairs: "
            f"positive/housekeeping/background residual means="
            f"{_fmt(n1800_summary, 'positive_control_mean_residual_spearman')}/"
            f"{_fmt(n1800_summary, 'housekeeping_negative_mean_residual_spearman')}/"
            f"{_fmt(n1800_summary, 'background_lr_mean_residual_spearman')}, "
            f"residual significant controls positive/negative="
            f"{n1800_summary.get('significant_positive_controls_residual_fdr_0_10', 'NA') if n1800_summary else 'NA'}/"
            f"{n1800_summary.get('significant_negative_controls_residual_fdr_0_10', 'NA') if n1800_summary else 'NA'}; "
            f"control table residual means="
            f"{_fmt(n1800_positive, 'mean_residual_spearman')}/"
            f"{_fmt(n1800_housekeeping, 'mean_residual_spearman')}/"
            f"{_fmt(n1800_background, 'mean_residual_spearman')}. "
            f"Current interpretation is mixed: residual positive controls exceed housekeeping negatives, "
            f"but broad background LR still show stronger downstream correlations and the 1,800-cell stress test "
            f"does not preserve the positive-control residual advantage. "
            f"LR embedding structure is weak on this subset: mechanism silhouette="
            f"{_fmt(embedding_summary, 'silhouette_mechanism')}, control-class silhouette="
            f"{_fmt(embedding_summary, 'silhouette_control_class')}, "
            f"BH-significant cluster enrichments="
            f"{embedding_summary.get('n_significant_cluster_enrichments_fdr_0_10', 'NA') if embedding_summary else 'NA'}, "
            f"embedding-norm/spatial-range Spearman="
            f"{_fmt(embedding_summary, 'embedding_norm_spatial_range_spearman')}. "
            f"Mechanism/category summary: positive-housekeeping residual difference="
            f"{_fmt(mechanism_summary, 'positive_minus_housekeeping_residual_spearman')}, "
            f"positive/housekeeping residual significant controls="
            f"{mechanism_summary.get('n_positive_controls_residual_bh_fdr_0_10', 'NA') if mechanism_summary else 'NA'}/"
            f"{mechanism_summary.get('n_housekeeping_negatives_residual_bh_fdr_0_10', 'NA') if mechanism_summary else 'NA'}, "
            f"best mechanism={mechanism_summary.get('best_mechanism_by_residual', 'NA') if mechanism_summary else 'NA'}, "
            f"best category={mechanism_summary.get('best_category_by_residual', 'NA') if mechanism_summary else 'NA'}, "
            f"embedding cluster category purity={_fmt(mechanism_summary, 'embedding_cluster_category_purity')}. "
            f"Mechanism-level result remains mixed because broad background/short-medium groups remain strong. "
            f"Lymph-node real-background injected-control validation: claim level="
            f"{lymph_node_summary.get('claim_level', 'NA') if lymph_node_summary else 'NA'}, "
            f"positive downstream supported="
            f"{lymph_node_summary.get('n_positive_downstream_supported', 'NA') if lymph_node_summary else 'NA'}, "
            f"weak positive downstream supported="
            f"{lymph_node_summary.get('n_weak_positive_downstream_supported', 'NA') if lymph_node_summary else 'NA'}, "
            f"negative top3={lymph_node_summary.get('n_negative_top3', 'NA') if lymph_node_summary else 'NA'}, "
            f"mean positive downstream Spearman="
            f"{_fmt(lymph_node_summary, 'mean_positive_downstream_spearman')}. "
            f"Synthesis claim level={synthesis_summary.get('claim_level', 'NA') if synthesis_summary else 'NA'}, "
            f"support score={synthesis_summary.get('support_score', 'NA') if synthesis_summary else 'NA'}, "
            f"contradiction score={synthesis_summary.get('contradiction_score', 'NA') if synthesis_summary else 'NA'}, "
            f"claim statuses={','.join(synthesis_statuses) if synthesis_statuses else 'NA'}."
        )
        if summary
        else "No Phase 2 result artifact yet.",
        "next_steps": (
            "Goal 5 now has a cross-real-background injected-control support artifact from lymph node, but the "
            "overall synthesis remains cautious rather than manuscript-ready biological validation. The 1,800-cell "
            "tonsil stress test is mixed/negative for downstream positive-control residual enrichment, LR embedding "
            "clustering is weak, and background LR remain strong; the next step is replicate-level real-background "
            "validation and stronger receiver/domain-specific ligand-target priors."
        )
        if summary
        else "Implement after Goal 4 high-resolution subset evidence.",
    }


def _goal3_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase2/goal3_model_selection"
    evidence = _read_csv(root / "model_selection_evidence.csv")
    criteria = _read_csv(root / "model_selection_criteria.csv")
    recommendation = _read_json(root / "model_selection_recommendation.json")
    recommended_id = str(recommendation.get("phase2_default_model", "NA")) if recommendation else "NA"
    recommended = _first_row(evidence, "candidate_id", recommended_id)
    edge_prior = _first_row(evidence, "candidate_id", "wavelet_edge_prior")
    default = _first_row(evidence, "candidate_id", "wavelet_default")
    focal = _first_row(evidence, "candidate_id", "wavelet_focal_weak_mining")
    status = "default_selected" if recommended and recommendation.get("status") == "default_selected" else "planned"
    return {
        "goal_id": "3",
        "goal": "Default model selection",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal3_model_selection/model_selection_evidence.csv",
                "outputs/phase2/goal3_model_selection/model_selection_criteria.csv",
                "outputs/phase2/goal3_model_selection/model_selection_recommendation.json",
                "outputs/phase2/goal3_model_selection/model_selection_summary.md",
                "scripts/phase2/run_goal3_model_selection.py",
            ]
        ),
        "key_findings": (
            f"Recommended Phase 2 default={recommended_id} with score="
            f"{recommended.get('selection_score', 'NA')} and evidence="
            f"{recommended.get('evidence_status', 'NA')}. "
            f"Controlled weak recall={_fmt(recommended, 'controlled_weak_recall')}, "
            f"full-slide weak recall={_fmt(recommended, 'full_slide_weak_recall')}, "
            f"operating weak recall={_fmt(recommended, 'operating_weak_recall')}, "
            f"group FDR positive BH-FDR={_fmt(recommended, 'group_fdr_positive_bh_fdr')}, "
            f"negative significant groups={recommended.get('group_fdr_negative_sig_groups', 'NA')}, "
            f"group FDR seed-stability passing fraction="
            f"{_fmt(recommended, 'group_fdr_seed_stability_passing_fraction')}, "
            f"seed-stability max negative groups="
            f"{recommended.get('group_fdr_seed_stability_max_negative_sig_groups', 'NA')}, "
            f"lymph-node downstream Spearman={_fmt(recommended, 'lymph_node_mean_positive_downstream_spearman')}. "
            f"Edge-prior without calibrated FDR score={edge_prior.get('selection_score', 'NA')}; "
            f"default wavelet full-slide weak recall={_fmt(default, 'full_slide_weak_recall')}; "
            f"focal weak-mining full-slide FDR={_fmt(focal, 'full_slide_fdr')}."
        ),
        "next_steps": (
            "Use wavelet_calibrated_fdr as the Phase 2 default candidate for Goals 4-5; "
            "rerun direct group-level FDR and downstream validation on additional high-resolution/real datasets "
            "before treating it as a final manuscript default. Current FDR support is seed-stable but still "
            "single-replicate."
        ),
    }


def _goal2_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial"
    summary = _read_csv(root / "fdr_calibration_summary.csv")
    group_summary = _read_csv(root / "fdr_calibration_group_summary.csv")
    grid100 = _read_csv(
        PROJECT_ROOT
        / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid/direct_group_grid_summary.csv"
    )
    grid500 = _read_csv(
        PROJECT_ROOT
        / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv"
    )
    seed_stability = _first(
        _read_csv(
            PROJECT_ROOT
            / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/direct_group_seed_stability_summary.csv"
        )
    )
    replicate_stability = _first(
        _read_csv(
            PROJECT_ROOT
            / "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_summary.csv"
        )
    )
    score = _first_row(summary, "null_model", "score_permutation")
    spatial = _first_row(summary, "null_model", "spatial_permutation")
    lr_identity = _first_row(summary, "null_model", "lr_identity_permutation")
    group_best = _best_group_summary(group_summary)
    positive_best = _best_positive_group_summary(group_summary)
    direct_best = _best_goal2_direct_grid_row(grid500 or grid100)
    selected_direct = _selected_goal2_direct_grid_row(grid500 or grid100)
    status = "calibrated_initial" if grid500 and _to_float(direct_best.get("positive_control_min_group_bh_fdr")) <= 0.05 else ("initial_evidence" if summary and group_summary else "planned")
    return {
        "goal_id": "2",
        "goal": "FDR/permutation calibration improvement",
        "evidence_status": status,
        "primary_artifacts": "; ".join(
            [
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial/fdr_calibration_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial/fdr_calibration_group_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial/fdr_calibration_manifest.json",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid/direct_group_grid_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/direct_group_seed_stability_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/direct_group_seed_stability_per_seed.csv",
                "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_summary.csv",
                "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_per_replicate.csv",
                "scripts/phase2/run_goal2_fdr_calibration.py",
                "scripts/phase2/run_goal2_direct_group_grid.py",
                "scripts/phase2/run_goal2_direct_group_seed_stability.py",
                "scripts/phase2/run_goal2_direct_group_replicate_stability.py",
                "src/spitialself/fdr.py",
            ]
        ),
        "key_findings": (
            f"Full-slide edge-prior calibration smoke covers {len(summary)} nulls and "
            f"{len(group_summary)} group-level summaries. "
            f"Score-permutation min empirical p={_fmt(score, 'min_p_value')}, "
            f"min BH-FDR={_fmt(score, 'min_bh_fdr')}, "
            f"Gaussian diagnostic min BH-FDR={_fmt(score, 'min_gaussian_bh_fdr')}, "
            f"local-FDR edges={score.get('n_local_fdr_edges', 'NA') if score else 'NA'}. "
            f"Spatial-permutation min BH-FDR={_fmt(spatial, 'min_bh_fdr')}; "
            f"LR-identity max null z={_fmt(lr_identity, 'max_null_z')}. "
            f"Operating q={_fmt(score, 'operating_quantile')} gives weak recall="
            f"{_fmt(score, 'operating_weak_recall')}, FDR={_fmt(score, 'operating_realized_fdr')}, "
            f"FP rate={_fmt(score, 'operating_fp_high_expression_rate')}. "
            f"Best group-level row={group_best.get('group_level', 'NA')} under "
            f"{group_best.get('null_model', 'NA')} with min group BH-FDR="
            f"{_fmt(group_best, 'min_group_bh_fdr')} using "
            f"{group_best.get('group_test', 'NA')}. "
            f"Best positive-control group BH-FDR={_fmt(positive_best, 'positive_control_min_group_bh_fdr')}; "
            f"test={positive_best.get('group_test', 'NA')}, "
            f"level={positive_best.get('group_level', 'NA')}, "
            f"null={positive_best.get('null_model', 'NA')}; "
            f"positive significant groups={positive_best.get('positive_control_significant_groups', 'NA')}; "
            f"negative significant groups={positive_best.get('negative_control_significant_groups', 'NA')}. "
            f"500-permutation focused direct grid best={direct_best.get('null_model', 'NA')}/"
            f"{direct_best.get('group_level', 'NA')}/top{direct_best.get('top_fraction', 'NA')} with "
            f"positive BH-FDR={_fmt(direct_best, 'positive_control_min_group_bh_fdr')}, "
            f"positive significant groups={direct_best.get('positive_control_significant_groups', 'NA')}, "
            f"negative significant groups={direct_best.get('negative_control_significant_groups', 'NA')}. "
            f"Selected interpretable LR-pair setting={selected_direct.get('null_model', 'NA')}/"
            f"{selected_direct.get('group_level', 'NA')}/top{selected_direct.get('top_fraction', 'NA')} with "
            f"positive BH-FDR={_fmt(selected_direct, 'positive_control_min_group_bh_fdr')}, "
            f"positive significant groups={selected_direct.get('positive_control_significant_groups', 'NA')}, "
            f"negative significant groups={selected_direct.get('negative_control_significant_groups', 'NA')}. "
            f"Selected-setting seed stability ({seed_stability.get('n_permutations', 'NA')} permutations, "
            f"{seed_stability.get('n_completed_seeds', 'NA')}/{seed_stability.get('n_seeds', 'NA')} seeds) "
            f"has passing seed fraction={_fmt(seed_stability, 'passing_seed_fraction')}, "
            f"positive significant groups range="
            f"{seed_stability.get('min_positive_control_significant_groups', 'NA')}-"
            f"{seed_stability.get('max_positive_control_significant_groups', 'NA')}, "
            f"max negative significant groups="
            f"{seed_stability.get('max_negative_control_significant_groups', 'NA')}, "
            f"claim level={seed_stability.get('claim_level', 'NA')}. "
            f"Controlled replicate stability ({replicate_stability.get('n_permutations', 'NA')} permutations, "
            f"{replicate_stability.get('n_completed_replicates', 'NA')}/"
            f"{replicate_stability.get('n_replicates', 'NA')} replicates) has passing fraction="
            f"{_fmt(replicate_stability, 'passing_replicate_fraction')}, "
            f"positive significant groups range="
            f"{replicate_stability.get('min_positive_control_significant_groups', 'NA')}-"
            f"{replicate_stability.get('max_positive_control_significant_groups', 'NA')}, "
            f"max negative significant groups="
            f"{replicate_stability.get('max_negative_control_significant_groups', 'NA')}, "
            f"claim level={replicate_stability.get('claim_level', 'NA')}."
        ),
        "next_steps": (
            "Use direct group-level permutation as the Phase 2 FDR candidate for model selection, "
            "while reporting receiver-permutation as a stricter low-power null. The selected null/statistic "
            "is seed-stable on one full-slide replicate and controlled-replicate stable on synthetic ground-truth "
            "data; validate it on additional realistic-background replicates/datasets and with larger permutation "
            "counts before manuscript claims."
        ),
    }


def _best_candidate(rows: list[dict[str, str]]) -> dict[str, str]:
    passing = [row for row in rows if str(row.get("phase2_goal1_candidate_pass", "")).lower() == "true"]
    candidates = passing if passing else rows
    if not candidates:
        return {}
    return max(candidates, key=lambda row: _to_float(row.get("weak_signal_recall_at_threshold_mean")))


def _best_group_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    return min(rows, key=lambda row: _to_float(row.get("min_group_bh_fdr")))


def _best_positive_group_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if math.isfinite(_to_float(row.get("positive_control_min_group_bh_fdr")))
    ]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: _to_float(row.get("positive_control_min_group_bh_fdr")))


def _best_goal2_direct_grid_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "completed"
        and _to_float(row.get("negative_control_significant_groups")) == 0
        and math.isfinite(_to_float(row.get("positive_control_min_group_bh_fdr")))
    ]
    if not candidates:
        return {}
    passing = [row for row in candidates if str(row.get("phase2_goal2_pass", "")).lower() == "true"]
    candidates = passing if passing else candidates
    return min(candidates, key=lambda row: _to_float(row.get("positive_control_min_group_bh_fdr")))


def _selected_goal2_direct_grid_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if (
            row.get("status") == "completed"
            and row.get("null_model") == "lr_identity_permutation"
            and row.get("group_level") == "lr_pair"
            and abs(_to_float(row.get("top_fraction")) - 0.10) < 1e-9
        ):
            return row
    return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _first_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(row: dict[str, str], column: str) -> str:
    value = _to_float(row.get(column, "nan"))
    return f"{value:.3f}" if math.isfinite(value) else "NA"


def _to_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Phase 2 Evidence Matrix",
        "",
        "| Goal | Status | Primary artifacts | Key findings | Next steps |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["goal"]),
                    _md(row["evidence_status"]),
                    _md(row["primary_artifacts"]),
                    _md(row["key_findings"]),
                    _md(row["next_steps"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
