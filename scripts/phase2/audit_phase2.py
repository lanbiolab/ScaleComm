#!/usr/bin/env python
"""Audit current Phase 2 setup and initial evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.experiments import list_phase2_experiments


@dataclass
class AuditRow:
    goal_id: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2/phase2_audit")
    args = parser.parse_args()
    rows: list[AuditRow] = []
    rows.extend(_audit_registry())
    rows.extend(_audit_goal1())
    rows.extend(_audit_goal2())
    rows.extend(_audit_goal3())
    rows.extend(_audit_goal4())
    rows.extend(_audit_goal5())
    rows.extend(_audit_goal6())
    rows.extend(_audit_evidence_matrix())
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase2_audit_summary.csv"
    json_path = output_dir / "phase2_audit_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal_id", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    print(f"Phase 2 audit CSV:  {csv_path}")
    print(f"Phase 2 audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("\nNon-complete rows:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")


def _audit_registry() -> list[AuditRow]:
    experiments = list_phase2_experiments()
    rows = [
        AuditRow(
            "phase2",
            "registry has six Phase 2 goals",
            "complete" if len(experiments) == 6 else "partial",
            "src/spitialself/experiments/phase2.py",
            f"n={len(experiments)}",
        )
    ]
    for experiment in experiments:
        rows.append(
            AuditRow(
                experiment.exp_id,
                "config file exists",
                "complete" if (PROJECT_ROOT / experiment.config_path).exists() else "missing",
                experiment.config_path,
            )
        )
    return rows


def _audit_goal1() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery"
    smoke_summary = root / "smoke_controlled_proxy_v2/weak_signal_recovery_summary.csv"
    full_summary = root / "controlled_five_replicate_initial/weak_signal_recovery_summary.csv"
    full_metrics = root / "controlled_five_replicate_initial/weak_signal_recovery_metrics.csv"
    full_status = root / "controlled_five_replicate_initial/weak_signal_recovery_status.csv"
    full_slide_summary = root / "full_slide_sparse_initial/weak_signal_recovery_summary.csv"
    threshold_summary = root / "full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv"
    threshold_metrics = root / "full_slide_sparse_threshold_tradeoff/threshold_tradeoff_metrics.csv"
    lr_balanced_summary = root / "full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv"
    lr_balanced_metrics = root / "full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_metrics.csv"
    lr_balanced_replicate = root / "controlled_lr_balanced_threshold_stability/lr_balanced_replicate_aggregate.csv"
    mechanism_threshold_summary = root / "full_slide_sparse_mechanism_threshold_tradeoff/mechanism_threshold_tradeoff_summary.csv"
    mechanism_probe_summary = root / "full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_summary.csv"
    mechanism_probe_metrics = root / "full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_metrics.csv"
    mechanism_combined = root / "goal1_mechanism_weak_signal_combined_summary.csv"
    mechanism_controlled = root / "controlled_five_replicate_mechanism_summary/mechanism_weak_signal_summary.csv"
    mechanism_full_q94 = root / "full_slide_sparse_operating_q94_mechanism_summary/mechanism_weak_signal_summary.csv"
    rows = [
        AuditRow("p2_goal1_weak_signal_recovery", "smoke summary exists", _exists(smoke_summary), str(smoke_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "controlled five-replicate summary exists", _exists(full_summary), str(full_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "controlled five-replicate metrics exists", _exists(full_metrics), str(full_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "controlled five-replicate status exists", _exists(full_status), str(full_status.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide semi-synthetic summary exists", _exists(full_slide_summary), str(full_slide_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide threshold tradeoff summary exists", _exists(threshold_summary), str(threshold_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide threshold tradeoff metrics exists", _exists(threshold_metrics), str(threshold_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide LR-balanced threshold summary exists", _exists(lr_balanced_summary), str(lr_balanced_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide LR-balanced threshold metrics exists", _exists(lr_balanced_metrics), str(lr_balanced_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "controlled LR-balanced replicate-stability aggregate exists", _exists(lr_balanced_replicate), str(lr_balanced_replicate.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "mechanism-specific threshold tradeoff summary exists", _exists(mechanism_threshold_summary), str(mechanism_threshold_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "mechanism-targeted weak-mining probe summary exists", _exists(mechanism_probe_summary), str(mechanism_probe_summary.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "mechanism-targeted weak-mining probe metrics exists", _exists(mechanism_probe_metrics), str(mechanism_probe_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "mechanism weak-signal combined summary exists", _exists(mechanism_combined), str(mechanism_combined.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "controlled mechanism weak-signal summary exists", _exists(mechanism_controlled), str(mechanism_controlled.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal1_weak_signal_recovery", "full-slide q0.94 mechanism weak-signal summary exists", _exists(mechanism_full_q94), str(mechanism_full_q94.relative_to(PROJECT_ROOT))),
    ]
    if full_summary.exists():
        summary = _read_csv(full_summary)
        methods = {row.get("ablation", "") for row in summary}
        required = {
            "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
            "wavelet_mechanism_niche_edge_prior_ranking",
            "phase2_weak_mining_edge_prior_wavelet",
            "phase2_focal_weak_mining_edge_prior_wavelet",
        }
        passing = [row for row in summary if str(row.get("phase2_goal1_candidate_pass", "")).lower() == "true"]
        best = max(summary, key=lambda row: _to_float(row.get("weak_signal_recall_at_threshold_mean"))) if summary else {}
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "controlled panel covers default, edge-prior, and weak-mining candidates",
                "complete" if required.issubset(methods) else "partial",
                str(full_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required - methods)}",
            )
        )
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "at least one candidate exceeds controlled weak-recall reference without FDR/FP-rate inflation",
                "complete" if passing else "partial",
                str(full_summary.relative_to(PROJECT_ROOT)),
                f"passing={[row.get('ablation') for row in passing]}; best={best.get('ablation')} weak={best.get('weak_signal_recall_at_threshold_mean')}",
            )
        )
    if full_metrics.exists():
        metrics = _read_csv(full_metrics)
        reps = {row.get("replicate", "") for row in metrics}
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "controlled metrics cover five replicates",
                "complete" if len(reps) >= 5 else "partial",
                str(full_metrics.relative_to(PROJECT_ROOT)),
                f"replicates={sorted(reps)}",
            )
        )
    if full_slide_summary.exists():
        summary = _read_csv(full_slide_summary)
        edge = _first_row(summary, "ablation", "wavelet_mechanism_niche_edge_prior_ranking")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "full-slide edge-prior ranking reproduces Phase 1 weak-recall boundary",
                "complete"
                if edge and _to_float(edge.get("weak_signal_recall_at_threshold_mean")) >= 0.039
                else "partial",
                str(full_slide_summary.relative_to(PROJECT_ROOT)),
                f"edge_weak={edge.get('weak_signal_recall_at_threshold_mean') if edge else 'NA'}",
            )
        )
    if threshold_summary.exists():
        summary = _read_csv(threshold_summary)
        edge = _first_row(summary, "label", "edge_prior_ranking")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "full-slide threshold tradeoff finds weak-recall operating point without large FDR/FP-rate increase",
                "complete"
                if edge
                and edge.get("status") == "passing_operating_point"
                and _to_float(edge.get("best_weak_signal_recall")) > 0.040
                and _to_float(edge.get("best_fdr_delta_vs_95")) <= 0.02
                and _to_float(edge.get("best_false_positive_high_expression_rate")) <= 0.005
                else "partial",
                str(threshold_summary.relative_to(PROJECT_ROOT)),
                (
                    f"status={edge.get('status') if edge else 'NA'}; "
                    f"q={edge.get('best_quantile') if edge else 'NA'}; "
                    f"weak={edge.get('best_weak_signal_recall') if edge else 'NA'}; "
                    f"fdr_delta={edge.get('best_fdr_delta_vs_95') if edge else 'NA'}; "
                    f"fp={edge.get('best_false_positive_high_expression_rate') if edge else 'NA'}"
                ),
            )
        )
    if lr_balanced_summary.exists():
        summary = _read_csv(lr_balanced_summary)
        lr_balanced = _first_row(summary, "strategy", "lrwise_quantile_with_global_floor")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "LR-balanced threshold diagnostic improves full-slide weak recall under FDR/FP constraints",
                "complete"
                if lr_balanced
                and lr_balanced.get("status") == "passing_operating_point"
                and _to_float(lr_balanced.get("best_weak_signal_recall")) > 0.080
                and _to_float(lr_balanced.get("best_fdr_delta_vs_reference")) <= 0.02
                and _to_float(lr_balanced.get("best_false_positive_high_expression_rate")) <= 0.005
                else "partial",
                str(lr_balanced_summary.relative_to(PROJECT_ROOT)),
                (
                    f"status={lr_balanced.get('status') if lr_balanced else 'NA'}; "
                    f"lr_q={lr_balanced.get('best_lr_quantile') if lr_balanced else 'NA'}; "
                    f"floor={lr_balanced.get('best_global_floor_quantile') if lr_balanced else 'NA'}; "
                    f"weak={lr_balanced.get('best_weak_signal_recall') if lr_balanced else 'NA'}; "
                    f"fdr_delta={lr_balanced.get('best_fdr_delta_vs_reference') if lr_balanced else 'NA'}; "
                    f"fp={lr_balanced.get('best_false_positive_high_expression_rate') if lr_balanced else 'NA'}"
                ),
            )
        )
    if lr_balanced_replicate.exists():
        summary = _read_csv(lr_balanced_replicate)
        lr_balanced = _first_row(summary, "strategy", "lrwise_quantile_with_global_floor")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "controlled LR-balanced stability records failed cross-replicate promotion criterion",
                "complete"
                if lr_balanced
                and _to_float(lr_balanced.get("n_replicates")) >= 5
                and _to_float(lr_balanced.get("passing_replicate_fraction")) == 0.0
                and _to_float(lr_balanced.get("best_fdr_delta_vs_reference_mean")) > 0.02
                and _to_float(lr_balanced.get("best_false_positive_high_expression_rate_mean")) > 0.005
                else "partial",
                str(lr_balanced_replicate.relative_to(PROJECT_ROOT)),
                (
                    f"pass_frac={lr_balanced.get('passing_replicate_fraction') if lr_balanced else 'NA'}; "
                    f"weak_mean={lr_balanced.get('best_weak_signal_recall_mean') if lr_balanced else 'NA'}; "
                    f"fdr_delta_mean={lr_balanced.get('best_fdr_delta_vs_reference_mean') if lr_balanced else 'NA'}; "
                    f"fp_mean={lr_balanced.get('best_false_positive_high_expression_rate_mean') if lr_balanced else 'NA'}"
                ),
            )
        )
    if mechanism_combined.exists():
        mechanism = _read_csv(mechanism_combined)
        mechanisms = {row.get("mechanism", "") for row in mechanism}
        edge_controlled = [
            row for row in mechanism
            if row.get("analysis_label") == "controlled_five_replicate_initial"
            and row.get("ablation") == "wavelet_mechanism_niche_edge_prior_ranking"
        ]
        edge_q94 = [
            row for row in mechanism
            if row.get("analysis_label") == "full_slide_sparse_operating_q94"
            and row.get("ablation") == "wavelet_mechanism_niche_edge_prior_ranking"
            and row.get("mechanism") == "long_range_chemokine_niche"
        ]
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "mechanism summary covers contact, short/medium, and long-range weak-signal diagnostics",
                "complete"
                if {
                    "juxtacrine_contact",
                    "short_medium_paracrine",
                    "long_range_chemokine_niche",
                }.issubset(mechanisms)
                and edge_controlled
                and edge_q94
                and _to_float(edge_q94[0].get("weak_signal_recall_at_threshold_mean")) > 0.04
                else "partial",
                str(mechanism_combined.relative_to(PROJECT_ROOT)),
                f"mechanisms={sorted(mechanisms)}; edge_q94_long={edge_q94[0].get('weak_signal_recall_at_threshold_mean') if edge_q94 else 'NA'}",
            )
        )
    if mechanism_threshold_summary.exists():
        threshold_rows = _read_csv(mechanism_threshold_summary)
        long_range = _first_row(threshold_rows, "mechanism", "long_range_chemokine_niche")
        combined = _first_row(threshold_rows, "strategy", "combined_mechanism_specific")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "mechanism-specific threshold diagnostic records small long-range operating-point gain",
                "complete"
                if long_range
                and str(long_range.get("candidate_pass", "")).lower() == "true"
                and _to_float(long_range.get("weak_signal_recall_at_threshold")) > 0.040
                and _to_float(long_range.get("fdr_delta_vs_reference")) <= 0.01
                and combined
                and str(combined.get("candidate_pass", "")).lower() == "true"
                else "partial",
                str(mechanism_threshold_summary.relative_to(PROJECT_ROOT)),
                (
                    f"long_q={long_range.get('threshold_quantile') if long_range else 'NA'}; "
                    f"long_weak={long_range.get('weak_signal_recall_at_threshold') if long_range else 'NA'}; "
                    f"long_fdr_delta={long_range.get('fdr_delta_vs_reference') if long_range else 'NA'}; "
                    f"combined_weak={combined.get('weak_signal_recall_at_threshold') if combined else 'NA'}"
                ),
            )
        )
    if mechanism_probe_summary.exists():
        probes = _read_csv(mechanism_probe_summary)
        methods = {row.get("ablation", "") for row in probes}
        edge = _first_row(probes, "ablation", "wavelet_mechanism_niche_edge_prior_ranking")
        long_probe = _first_row(probes, "ablation", "phase2_long_range_soft_weak_edge_prior_wavelet")
        rows.append(
            AuditRow(
                "p2_goal1_weak_signal_recovery",
                "mechanism-targeted weak-mining probes are recorded as non-transfer improvements",
                "complete"
                if {
                    "wavelet_mechanism_niche_edge_prior_ranking",
                    "phase2_long_range_soft_weak_edge_prior_wavelet",
                    "phase2_non_contact_soft_weak_edge_prior_wavelet",
                }.issubset(methods)
                and long_probe
                and edge
                and _to_float(long_probe.get("weak_signal_recall_at_threshold_mean"))
                < _to_float(edge.get("weak_signal_recall_at_threshold_mean"))
                else "partial",
                str(mechanism_probe_summary.relative_to(PROJECT_ROOT)),
                (
                    f"methods={sorted(methods)}; "
                    f"edge_weak={edge.get('weak_signal_recall_at_threshold_mean') if edge else 'NA'}; "
                    f"long_probe_weak={long_probe.get('weak_signal_recall_at_threshold_mean') if long_probe else 'NA'}"
                ),
            )
        )
    return rows


def _audit_evidence_matrix() -> list[AuditRow]:
    path = PROJECT_ROOT / "outputs/phase2/phase2_evidence_matrix/phase2_evidence_matrix.csv"
    rows = [AuditRow("phase2", "phase2 evidence matrix exists", _exists(path), str(path.relative_to(PROJECT_ROOT)))]
    if path.exists():
        records = _read_csv(path)
        rows.append(
            AuditRow(
                "phase2",
                "phase2 evidence matrix has six rows and Goal 1 initial evidence",
                "complete"
                if len(records) == 6
                and any(row.get("goal_id") == "1" and row.get("evidence_status") == "initial_evidence" for row in records)
                and any(
                    row.get("goal_id") == "2"
                    and row.get("evidence_status") in {"initial_evidence", "calibrated_initial"}
                    for row in records
                )
                and any(row.get("goal_id") == "3" and row.get("evidence_status") == "default_selected" for row in records)
                else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"n={len(records)}",
            )
        )
    return rows


def _audit_goal3() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal3_model_selection"
    evidence_path = root / "model_selection_evidence.csv"
    criteria_path = root / "model_selection_criteria.csv"
    recommendation_path = root / "model_selection_recommendation.json"
    summary_path = root / "model_selection_summary.md"
    manifest_path = root / "model_selection_manifest.json"
    rows = [
        AuditRow("p2_goal3_model_selection", "model-selection evidence exists", _exists(evidence_path), str(evidence_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal3_model_selection", "model-selection criteria exists", _exists(criteria_path), str(criteria_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal3_model_selection", "model-selection recommendation exists", _exists(recommendation_path), str(recommendation_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal3_model_selection", "model-selection markdown summary exists", _exists(summary_path), str(summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal3_model_selection", "model-selection manifest exists", _exists(manifest_path), str(manifest_path.relative_to(PROJECT_ROOT))),
    ]
    if evidence_path.exists():
        evidence = _read_csv(evidence_path)
        candidates = {row.get("candidate_id", "") for row in evidence}
        required = {
            "wavelet_default",
            "wavelet_edge_prior",
            "wavelet_calibrated_fdr",
            "wavelet_weak_mining",
            "wavelet_focal_weak_mining",
            "gat_reference",
            "no_lr_identity",
            "no_distance",
            "no_expression",
            "hybrid_wavelet_gat",
        }
        recommended = [row for row in evidence if str(row.get("recommended_default", "")).lower() == "true"]
        rows.append(
            AuditRow(
                "p2_goal3_model_selection",
                "model-selection covers planned candidate set",
                "complete" if required.issubset(candidates) else "partial",
                str(evidence_path.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required - candidates)}",
            )
        )
        rows.append(
            AuditRow(
                "p2_goal3_model_selection",
                "exactly one Phase 2 default is recommended",
                "complete" if len(recommended) == 1 else "partial",
                str(evidence_path.relative_to(PROJECT_ROOT)),
                f"recommended={[row.get('candidate_id') for row in recommended]}",
            )
        )
        rec = recommended[0] if recommended else {}
        rows.append(
            AuditRow(
                "p2_goal3_model_selection",
                "recommended default has controlled, full-slide, FDR, and lymph-node evidence",
                "complete"
                if rec.get("candidate_id") == "wavelet_calibrated_fdr"
                and rec.get("evidence_status") == "selection_ready"
                and _to_float(rec.get("controlled_weak_recall")) > 0.155
                and _to_float(rec.get("operating_weak_recall")) > 0.04
                and _to_float(rec.get("group_fdr_positive_bh_fdr")) <= 0.05
                and _to_float(rec.get("group_fdr_negative_sig_groups")) == 0
                and rec.get("group_fdr_seed_stability_status") == "seed_stability_completed"
                and _to_float(rec.get("group_fdr_seed_stability_passing_fraction")) == 1.0
                and _to_float(rec.get("group_fdr_seed_stability_max_negative_sig_groups")) == 0
                and _to_float(rec.get("lymph_node_mean_positive_downstream_spearman")) > 0.7
                else "partial",
                str(evidence_path.relative_to(PROJECT_ROOT)),
                (
                    f"id={rec.get('candidate_id', 'NA')}; "
                    f"score={rec.get('selection_score', 'NA')}; "
                    f"controlled_weak={rec.get('controlled_weak_recall', 'NA')}; "
                    f"operating_weak={rec.get('operating_weak_recall', 'NA')}; "
                    f"group_fdr={rec.get('group_fdr_positive_bh_fdr', 'NA')}; "
                    f"neg_sig={rec.get('group_fdr_negative_sig_groups', 'NA')}; "
                    f"seed_pass={rec.get('group_fdr_seed_stability_passing_fraction', 'NA')}; "
                    f"seed_neg={rec.get('group_fdr_seed_stability_max_negative_sig_groups', 'NA')}; "
                    f"spearman={rec.get('lymph_node_mean_positive_downstream_spearman', 'NA')}"
                ),
            )
        )
    if criteria_path.exists():
        criteria = _read_csv(criteria_path)
        rows.append(
            AuditRow(
                "p2_goal3_model_selection",
                "criteria table includes pass/fail rows",
                "complete" if len(criteria) >= 40 else "partial",
                str(criteria_path.relative_to(PROJECT_ROOT)),
                f"n={len(criteria)}",
            )
        )
    return rows


def _audit_goal4() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset"
    fresh_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset"
    scaling_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated"
    baseline_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset"
    module_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset"
    module_n1800_root = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset"
    profile_path = root / "high_resolution_dataset_profile.csv"
    summary_path = root / "goal4_high_resolution_summary.csv"
    lr_summary_path = root / "tonsil_lr_summary.csv"
    cell_type_path = root / "tonsil_cell_type_pair_summary.csv"
    positive_path = root / "tonsil_positive_control_summary.csv"
    manifest_path = root / "goal4_high_resolution_manifest.json"
    fresh_profile_path = fresh_root / "tonsil_fresh_dataset_profile.csv"
    fresh_summary_path = fresh_root / "tonsil_fresh_goal4_summary.csv"
    fresh_lr_summary_path = fresh_root / "tonsil_fresh_lr_summary.csv"
    fresh_cell_type_path = fresh_root / "tonsil_fresh_cell_type_pair_summary.csv"
    fresh_control_path = fresh_root / "tonsil_fresh_control_summary.csv"
    fresh_manifest_path = fresh_root / "tonsil_fresh_manifest.json"
    scaling_summary_path = scaling_root / "tonsil_scaling_summary.csv"
    scaling_run_path = scaling_root / "tonsil_scaling_run_summary.csv"
    scaling_manifest_path = scaling_root / "tonsil_scaling_manifest.json"
    baseline_summary_path = baseline_root / "tonsil_baseline_matched_summary.csv"
    baseline_method_path = baseline_root / "tonsil_baseline_matched_method_summary.csv"
    baseline_status_path = baseline_root / "tonsil_baseline_matched_status.csv"
    baseline_control_path = baseline_root / "tonsil_baseline_matched_control_lr_summary.csv"
    baseline_manifest_path = baseline_root / "tonsil_baseline_matched_manifest.json"
    module_overview_path = module_root / "tonsil_ccc_module_overview.csv"
    module_summary_path = module_root / "tonsil_ccc_module_summary.csv"
    module_enrichment_path = module_root / "tonsil_ccc_module_control_enrichment.csv"
    module_pair_path = module_root / "tonsil_ccc_module_cell_type_pair_summary.csv"
    module_manifest_path = module_root / "tonsil_ccc_module_manifest.json"
    module_n1800_overview_path = module_n1800_root / "tonsil_ccc_module_overview.csv"
    module_n1800_summary_path = module_n1800_root / "tonsil_ccc_module_summary.csv"
    module_n1800_enrichment_path = module_n1800_root / "tonsil_ccc_module_control_enrichment.csv"
    module_n1800_pair_path = module_n1800_root / "tonsil_ccc_module_cell_type_pair_summary.csv"
    module_n1800_manifest_path = module_n1800_root / "tonsil_ccc_module_manifest.json"
    stream_script_path = PROJECT_ROOT / "scripts/phase2/run_goal4_high_resolution_tonsil_subset.py"
    fresh_script_path = PROJECT_ROOT / "scripts/phase2/run_goal4_fresh_tonsil_subset.py"
    scaling_script_path = PROJECT_ROOT / "scripts/phase2/run_goal4_high_resolution_scaling.py"
    baseline_script_path = PROJECT_ROOT / "scripts/phase2/run_goal4_tonsil_baseline_matched.py"
    module_script_path = PROJECT_ROOT / "scripts/phase2/run_goal4_tonsil_ccc_modules.py"
    rows = [
        AuditRow("p2_goal4_high_resolution", "tonsil existing-stream runner exists", _exists(stream_script_path), str(stream_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh tonsil subset runner exists", _exists(fresh_script_path), str(fresh_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh tonsil scaling runner exists", _exists(scaling_script_path), str(scaling_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh tonsil baseline-matched runner exists", _exists(baseline_script_path), str(baseline_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh tonsil CCC module runner exists", _exists(module_script_path), str(module_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream dataset profile exists", _exists(profile_path), str(profile_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream goal summary exists", _exists(summary_path), str(summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream LR summary exists", _exists(lr_summary_path), str(lr_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream cell-type-pair summary exists", _exists(cell_type_path), str(cell_type_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream positive-control summary exists", _exists(positive_path), str(positive_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "existing-stream manifest exists", _exists(manifest_path), str(manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset dataset profile exists", _exists(fresh_profile_path), str(fresh_profile_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset goal summary exists", _exists(fresh_summary_path), str(fresh_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset LR summary exists", _exists(fresh_lr_summary_path), str(fresh_lr_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset cell-type-pair summary exists", _exists(fresh_cell_type_path), str(fresh_cell_type_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset control summary exists", _exists(fresh_control_path), str(fresh_control_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset manifest exists", _exists(fresh_manifest_path), str(fresh_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset scaling summary exists", _exists(scaling_summary_path), str(scaling_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset scaling run summary exists", _exists(scaling_run_path), str(scaling_run_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset scaling manifest exists", _exists(scaling_manifest_path), str(scaling_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset baseline-matched summary exists", _exists(baseline_summary_path), str(baseline_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset baseline-matched method summary exists", _exists(baseline_method_path), str(baseline_method_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset baseline-matched status exists", _exists(baseline_status_path), str(baseline_status_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset baseline-matched control summary exists", _exists(baseline_control_path), str(baseline_control_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset baseline-matched manifest exists", _exists(baseline_manifest_path), str(baseline_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset CCC module overview exists", _exists(module_overview_path), str(module_overview_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset CCC module summary exists", _exists(module_summary_path), str(module_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset CCC module control enrichment exists", _exists(module_enrichment_path), str(module_enrichment_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset CCC module cell-type-pair summary exists", _exists(module_pair_path), str(module_pair_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "fresh subset CCC module manifest exists", _exists(module_manifest_path), str(module_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "n1800 CCC module stress-test overview exists", _exists(module_n1800_overview_path), str(module_n1800_overview_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "n1800 CCC module stress-test summary exists", _exists(module_n1800_summary_path), str(module_n1800_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "n1800 CCC module stress-test control enrichment exists", _exists(module_n1800_enrichment_path), str(module_n1800_enrichment_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "n1800 CCC module stress-test cell-type-pair summary exists", _exists(module_n1800_pair_path), str(module_n1800_pair_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal4_high_resolution", "n1800 CCC module stress-test manifest exists", _exists(module_n1800_manifest_path), str(module_n1800_manifest_path.relative_to(PROJECT_ROOT))),
    ]
    if profile_path.exists():
        records = _read_csv(profile_path)
        profile = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "tonsil node order is validated against expression header and graph artifact",
                "complete"
                if str(profile.get("all_header_cells_match_spatial", "")).lower() == "true"
                and _to_float(profile.get("graph_nodes")) == _to_float(profile.get("n_cells"))
                else "partial",
                str(profile_path.relative_to(PROJECT_ROOT)),
                (
                    f"n_cells={profile.get('n_cells', 'NA')}; "
                    f"expr_cells={profile.get('n_expression_cells', 'NA')}; "
                    f"graph_nodes={profile.get('graph_nodes', 'NA')}; "
                    f"header_match={profile.get('all_header_cells_match_spatial', 'NA')}"
                ),
            )
        )
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "existing tonsil prediction stream is processed at high-resolution scale",
                "complete"
                if _to_float(profile.get("n_cells")) >= 5000
                and _to_float(profile.get("n_prediction_rows_processed")) >= 50_000_000
                and _to_float(profile.get("n_lr_pairs")) >= 900
                else "partial",
                str(profile_path.relative_to(PROJECT_ROOT)),
                (
                    f"cells={profile.get('n_cells', 'NA')}; "
                    f"rows={profile.get('n_prediction_rows_processed', 'NA')}; "
                    f"lr={profile.get('n_lr_pairs', 'NA')}; "
                    f"seconds={profile.get('wall_clock_seconds', 'NA')}; "
                    f"rss={profile.get('peak_rss_mb', 'NA')}"
                ),
            )
        )
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh Phase 2 default inference remains explicitly marked incomplete",
                "complete" if str(profile.get("is_fresh_phase2_default_inference", "")).lower() == "false" else "partial",
                str(profile_path.relative_to(PROJECT_ROOT)),
                f"is_fresh={profile.get('is_fresh_phase2_default_inference', 'NA')}",
            )
        )
    if summary_path.exists():
        summary_records = _read_csv(summary_path)
        summary = summary_records[0] if summary_records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "goal summary records initial high-resolution evidence status",
                "complete" if summary.get("evidence_status") == "initial_high_resolution_evidence" else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"positive_present={summary.get('n_positive_controls_present', 'NA')}; "
                    f"top_decile={summary.get('n_positive_controls_top_decile', 'NA')}; "
                    f"best={summary.get('best_positive_control', 'NA')}"
                ),
            )
        )
    if positive_path.exists():
        positives = _read_csv(positive_path)
        present = [row for row in positives if str(row.get("present_in_lr_table", "")).lower() == "true"]
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "tonsil positive-control priors are tracked without overclaiming success",
                "complete" if len(present) >= 5 else "partial",
                str(positive_path.relative_to(PROJECT_ROOT)),
                (
                    f"present={len(present)}; "
                    f"top_decile={sum(_to_float(row.get('rank_percentile_by_prob_mean')) <= 0.10 for row in present)}; "
                    f"best_rank={min((_to_float(row.get('rank_by_prob_mean')) for row in present), default=float('nan'))}"
                ),
            )
        )
    if fresh_profile_path.exists():
        records = _read_csv(fresh_profile_path)
        profile = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh Phase 2 default tonsil subset inference ran at nontrivial scale",
                "complete"
                if str(profile.get("is_fresh_phase2_default_inference", "")).lower() == "true"
                and profile.get("phase2_default_candidate") == "wavelet_calibrated_fdr"
                and _to_float(profile.get("n_cells")) >= 700
                and _to_float(profile.get("n_candidate_edges")) >= 300_000
                and _to_float(profile.get("n_lr_pairs")) >= 40
                else "partial",
                str(fresh_profile_path.relative_to(PROJECT_ROOT)),
                (
                    f"fresh={profile.get('is_fresh_phase2_default_inference', 'NA')}; "
                    f"default={profile.get('phase2_default_candidate', 'NA')}; "
                    f"cells={profile.get('n_cells', 'NA')}; "
                    f"edges={profile.get('n_candidate_edges', 'NA')}; "
                    f"lr={profile.get('n_lr_pairs', 'NA')}; "
                    f"seconds={profile.get('wall_clock_seconds', 'NA')}"
                ),
            )
        )
    if fresh_summary_path.exists():
        summary_records = _read_csv(fresh_summary_path)
        summary = summary_records[0] if summary_records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh subset shows positive-control enrichment with high-expression negatives suppressed",
                "complete"
                if summary.get("evidence_status") == "fresh_subset_initial"
                and _to_float(summary.get("n_positive_controls_top_decile")) >= 1
                and _to_float(summary.get("n_negative_controls_top_decile")) == 0
                else "partial",
                str(fresh_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"pos_top_decile={summary.get('n_positive_controls_top_decile', 'NA')}; "
                    f"neg_top_decile={summary.get('n_negative_controls_top_decile', 'NA')}; "
                    f"best_pos={summary.get('best_positive_control', 'NA')}; "
                    f"best_neg={summary.get('best_negative_control', 'NA')}"
                ),
            )
        )
    if fresh_control_path.exists():
        controls = _read_csv(fresh_control_path)
        positives = [row for row in controls if row.get("control_class") == "tonsil_positive_prior"]
        negatives = [row for row in controls if row.get("control_class") != "tonsil_positive_prior"]
        best_positive = min((_to_float(row.get("rank_by_score_mean")) for row in positives), default=float("nan"))
        best_negative = min((_to_float(row.get("rank_by_score_mean")) for row in negatives), default=float("nan"))
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh subset control table includes positive and negative controls",
                "complete" if len(positives) >= 6 and len(negatives) >= 3 and best_positive < best_negative else "partial",
                str(fresh_control_path.relative_to(PROJECT_ROOT)),
                f"positive={len(positives)}; negative={len(negatives)}; best_positive_rank={best_positive}; best_negative_rank={best_negative}",
            )
        )
    if scaling_summary_path.exists():
        summary_records = _read_csv(scaling_summary_path)
        summary = summary_records[0] if summary_records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh tonsil scaling reaches larger subset with negative controls suppressed",
                "complete"
                if summary.get("evidence_status") == "fresh_scaling_initial"
                and _to_float(summary.get("n_runs")) >= 3
                and _to_float(summary.get("n_cells_max")) >= 1800
                and _to_float(summary.get("n_candidate_edges_max")) >= 1_000_000
                and _to_float(summary.get("largest_run_negative_controls_top_decile")) == 0
                else "partial",
                str(scaling_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"runs={summary.get('n_runs', 'NA')}; "
                    f"cells={summary.get('n_cells_max', 'NA')}; "
                    f"edges={summary.get('n_candidate_edges_max', 'NA')}; "
                    f"neg_top10={summary.get('largest_run_negative_controls_top_decile', 'NA')}"
                ),
            )
        )
    if scaling_run_path.exists():
        runs = _read_csv(scaling_run_path)
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh tonsil scaling run table records runtime and memory for each subset",
                "complete"
                if len(runs) >= 3
                and {"n_cells", "n_candidate_edges", "wall_clock_seconds", "peak_rss_mb"}.issubset(runs[0])
                and max(_to_float(row.get("n_cells")) for row in runs) >= 1800
                else "partial",
                str(scaling_run_path.relative_to(PROJECT_ROOT)),
                f"rows={len(runs)}",
            )
        )
    if baseline_summary_path.exists():
        summary_records = _read_csv(baseline_summary_path)
        summary = summary_records[0] if summary_records else {}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh tonsil baseline-matched comparison records retained-method control ranking",
                "complete"
                if summary.get("evidence_status") == "fresh_subset_baseline_matched_initial"
                and _to_float(summary.get("n_methods_completed")) >= 4
                and summary.get("spatialself_rank_validation_pass") == "True"
                and _to_float(summary.get("spatialself_positive_controls_top_decile")) >= 1
                and _to_float(summary.get("spatialself_negative_controls_top_decile")) == 0
                else "partial",
                str(baseline_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"methods={summary.get('n_methods_completed', 'NA')}; "
                    f"best={summary.get('best_rank_validation_method', 'NA')}; "
                    f"spatialself_pass={summary.get('spatialself_rank_validation_pass', 'NA')}; "
                    f"spatialself_pos_neg={summary.get('spatialself_positive_controls_top_decile', 'NA')}/"
                    f"{summary.get('spatialself_negative_controls_top_decile', 'NA')}"
                ),
            )
        )
    if baseline_method_path.exists():
        methods = _read_csv(baseline_method_path)
        by_method = {row.get("method", ""): row for row in methods}
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh tonsil baseline-matched method table separates valid and flat-score baselines",
                "complete"
                if {"spatialself_wavelet_calibrated", "commot", "cellchat", "spatalk"}.issubset(by_method)
                and by_method["spatialself_wavelet_calibrated"].get("rank_validation_pass") == "True"
                and _to_float(by_method["commot"].get("n_unique_scores")) > 1
                and _to_float(by_method["cellchat"].get("n_unique_scores")) > 1
                and _to_float(by_method["spatalk"].get("n_unique_scores")) <= 1
                else "partial",
                str(baseline_method_path.relative_to(PROJECT_ROOT)),
                f"methods={sorted(by_method)}",
            )
        )
    if module_overview_path.exists() and module_summary_path.exists() and module_enrichment_path.exists():
        overview_records = _read_csv(module_overview_path)
        overview = overview_records[0] if overview_records else {}
        modules = _read_csv(module_summary_path)
        enrichment = _read_csv(module_enrichment_path)
        positive_enrich = [
            row for row in enrichment
            if row.get("control_class") == "tonsil_positive_prior"
            and _to_float(row.get("bh_fdr")) <= 0.10
        ]
        negative_enrich = [
            row for row in enrichment
            if row.get("control_class") == "housekeeping_negative_control"
            and _to_float(row.get("bh_fdr")) <= 0.10
        ]
        long_positive = [
            row for row in modules
            if row.get("dominant_mechanism") == "long_range_chemokine_niche"
            and _to_float(row.get("n_positive_controls")) >= 3
            and _to_float(row.get("n_housekeeping_negative_controls")) == 0
        ]
        fp_negative = [
            row for row in modules
            if row.get("dominant_mechanism") == "false_positive_high_expression"
            and _to_float(row.get("n_housekeeping_negative_controls")) >= 3
        ]
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "fresh tonsil CCC modules separate positive mechanisms from housekeeping negative controls",
                "complete"
                if overview.get("evidence_status") == "module_validation_initial"
                and _to_float(overview.get("n_modules")) >= 3
                and long_positive
                and fp_negative
                and positive_enrich
                and negative_enrich
                else "partial",
                str(module_overview_path.relative_to(PROJECT_ROOT)),
                (
                    f"modules={overview.get('n_modules', 'NA')}; "
                    f"positive_modules={overview.get('n_modules_with_positive_controls', 'NA')}; "
                    f"negative_modules={overview.get('n_modules_with_housekeeping_negatives', 'NA')}; "
                    f"positive_enrich={[(row.get('ccc_module'), row.get('bh_fdr')) for row in positive_enrich]}; "
                    f"negative_enrich={[(row.get('ccc_module'), row.get('bh_fdr')) for row in negative_enrich]}"
                ),
            )
        )
    if module_n1800_overview_path.exists() and module_n1800_summary_path.exists() and module_n1800_enrichment_path.exists():
        overview_records = _read_csv(module_n1800_overview_path)
        overview = overview_records[0] if overview_records else {}
        modules = _read_csv(module_n1800_summary_path)
        enrichment = _read_csv(module_n1800_enrichment_path)
        mixed_modules = [
            row for row in modules
            if _to_float(row.get("n_positive_controls")) > 0
            and _to_float(row.get("n_housekeeping_negative_controls")) > 0
        ]
        negative_enrich = [
            row for row in enrichment
            if row.get("control_class") == "housekeeping_negative_control"
            and _to_float(row.get("bh_fdr")) <= 0.10
        ]
        rows.append(
            AuditRow(
                "p2_goal4_high_resolution",
                "n1800 CCC module stress test records weakened positive/housekeeping separation",
                "complete"
                if overview.get("evidence_status") == "module_validation_initial"
                and _to_float(overview.get("n_modules")) >= 3
                and mixed_modules
                and negative_enrich
                else "partial",
                str(module_n1800_overview_path.relative_to(PROJECT_ROOT)),
                (
                    f"modules={overview.get('n_modules', 'NA')}; "
                    f"mixed={[(row.get('ccc_module'), row.get('n_positive_controls'), row.get('n_housekeeping_negative_controls')) for row in mixed_modules]}; "
                    f"negative_enrich={[(row.get('ccc_module'), row.get('bh_fdr')) for row in negative_enrich]}"
                ),
            )
        )
    return rows


def _audit_goal5() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset"
    n1800_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_n1800_subset"
    embedding_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset"
    mechanism_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset"
    lymph_node_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lymph_node_real_background"
    synthesis_root = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/synthesis"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_tonsil_interpretability.py"
    embedding_script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_lr_embedding_interpretability.py"
    mechanism_script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_mechanism_category_interpretability.py"
    lymph_node_script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_lymph_node_interpretability.py"
    synthesis_script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_interpretability_synthesis.py"
    summary_path = root / "goal5_interpretability_summary.csv"
    downstream_path = root / "tonsil_downstream_target_correlation.csv"
    control_path = root / "tonsil_interpretability_control_summary.csv"
    pathway_path = root / "tonsil_pathway_category_summary.csv"
    module_path = root / "tonsil_ccc_module_summary.csv"
    profile_path = root / "tonsil_lr_activity_profiles.csv"
    manifest_path = root / "goal5_interpretability_manifest.json"
    n1800_summary_path = n1800_root / "goal5_interpretability_summary.csv"
    n1800_downstream_path = n1800_root / "tonsil_downstream_target_correlation.csv"
    n1800_control_path = n1800_root / "tonsil_interpretability_control_summary.csv"
    n1800_pathway_path = n1800_root / "tonsil_pathway_category_summary.csv"
    n1800_module_path = n1800_root / "tonsil_ccc_module_summary.csv"
    n1800_manifest_path = n1800_root / "goal5_interpretability_manifest.json"
    embedding_summary_path = embedding_root / "goal5_lr_embedding_summary.csv"
    embedding_coordinates_path = embedding_root / "lr_embedding_coordinates.csv"
    embedding_cluster_path = embedding_root / "lr_embedding_cluster_summary.csv"
    embedding_enrichment_path = embedding_root / "lr_embedding_cluster_enrichment.csv"
    embedding_manifest_path = embedding_root / "goal5_lr_embedding_manifest.json"
    mechanism_summary_path = mechanism_root / "goal5_mechanism_category_summary.csv"
    mechanism_table_path = mechanism_root / "mechanism_interpretability_summary.csv"
    mechanism_category_path = mechanism_root / "category_interpretability_summary.csv"
    mechanism_pairwise_path = mechanism_root / "mechanism_category_pairwise_tests.csv"
    mechanism_manifest_path = mechanism_root / "goal5_mechanism_category_manifest.json"
    lymph_node_summary_path = lymph_node_root / "lymph_node_goal5_interpretability_summary.csv"
    lymph_node_controls_path = lymph_node_root / "lymph_node_goal5_control_interpretability.csv"
    lymph_node_claims_path = lymph_node_root / "lymph_node_goal5_interpretability_claims.csv"
    lymph_node_manifest_path = lymph_node_root / "lymph_node_goal5_interpretability_manifest.json"
    synthesis_summary_path = synthesis_root / "interpretability_synthesis_summary.csv"
    synthesis_claims_path = synthesis_root / "interpretability_synthesis_claims.csv"
    synthesis_metrics_path = synthesis_root / "interpretability_synthesis_metrics.csv"
    synthesis_manifest_path = synthesis_root / "interpretability_synthesis_manifest.json"
    rows = [
        AuditRow("p2_goal5_interpretability", "tonsil interpretability runner exists", _exists(script_path), str(script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding interpretability runner exists", _exists(embedding_script_path), str(embedding_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "mechanism/category interpretability runner exists", _exists(mechanism_script_path), str(mechanism_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "lymph-node interpretability runner exists", _exists(lymph_node_script_path), str(lymph_node_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "interpretability synthesis runner exists", _exists(synthesis_script_path), str(synthesis_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "goal summary exists", _exists(summary_path), str(summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "downstream target correlation table exists", _exists(downstream_path), str(downstream_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "control summary exists", _exists(control_path), str(control_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "pathway category summary exists", _exists(pathway_path), str(pathway_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "CCC module summary exists", _exists(module_path), str(module_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR activity profile table exists", _exists(profile_path), str(profile_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "manifest exists", _exists(manifest_path), str(manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test summary exists", _exists(n1800_summary_path), str(n1800_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test downstream table exists", _exists(n1800_downstream_path), str(n1800_downstream_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test control summary exists", _exists(n1800_control_path), str(n1800_control_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test pathway summary exists", _exists(n1800_pathway_path), str(n1800_pathway_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test module summary exists", _exists(n1800_module_path), str(n1800_module_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "n1800 tonsil stress-test manifest exists", _exists(n1800_manifest_path), str(n1800_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding summary exists", _exists(embedding_summary_path), str(embedding_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding coordinate table exists", _exists(embedding_coordinates_path), str(embedding_coordinates_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding cluster summary exists", _exists(embedding_cluster_path), str(embedding_cluster_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding cluster enrichment exists", _exists(embedding_enrichment_path), str(embedding_enrichment_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR embedding manifest exists", _exists(embedding_manifest_path), str(embedding_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "mechanism/category summary exists", _exists(mechanism_summary_path), str(mechanism_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "mechanism interpretability table exists", _exists(mechanism_table_path), str(mechanism_table_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "LR category interpretability table exists", _exists(mechanism_category_path), str(mechanism_category_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "mechanism/category pairwise tests exists", _exists(mechanism_pairwise_path), str(mechanism_pairwise_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "mechanism/category manifest exists", _exists(mechanism_manifest_path), str(mechanism_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "lymph-node interpretability summary exists", _exists(lymph_node_summary_path), str(lymph_node_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "lymph-node interpretability controls exist", _exists(lymph_node_controls_path), str(lymph_node_controls_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "lymph-node interpretability claims exist", _exists(lymph_node_claims_path), str(lymph_node_claims_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "lymph-node interpretability manifest exists", _exists(lymph_node_manifest_path), str(lymph_node_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "interpretability synthesis summary exists", _exists(synthesis_summary_path), str(synthesis_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "interpretability synthesis claims exists", _exists(synthesis_claims_path), str(synthesis_claims_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "interpretability synthesis metrics exists", _exists(synthesis_metrics_path), str(synthesis_metrics_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal5_interpretability", "interpretability synthesis manifest exists", _exists(synthesis_manifest_path), str(synthesis_manifest_path.relative_to(PROJECT_ROOT))),
    ]
    if downstream_path.exists():
        downstream = _read_csv(downstream_path)
        required_columns = {
            "lr_id",
            "canonical_pair",
            "control_class",
            "target_signatures",
            "spearman_r",
            "empirical_p_value",
            "bh_fdr",
            "residual_spearman_r",
            "residual_empirical_p_value",
            "residual_bh_fdr",
            "within_expected_receiver_spearman_r",
            "rank_by_residual_spearman",
            "status",
        }
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "downstream table includes raw, residualized, and expected-receiver metrics",
                "complete" if downstream and required_columns.issubset(downstream[0]) else "partial",
                str(downstream_path.relative_to(PROJECT_ROOT)),
                f"columns={sorted(downstream[0]) if downstream else []}",
            )
        )
        tested = [row for row in downstream if row.get("status") == "tested"]
        classes = {row.get("control_class", "") for row in downstream}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "downstream table tests all fresh tonsil LR pairs and keeps control strata separate",
                "complete"
                if len(tested) >= 45
                and {"tonsil_positive_prior", "housekeeping_negative_control", "background_lr"}.issubset(classes)
                else "partial",
                str(downstream_path.relative_to(PROJECT_ROOT)),
                f"tested={len(tested)}; classes={sorted(classes)}",
            )
        )
    if summary_path.exists():
        records = _read_csv(summary_path)
        summary = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "goal summary records initial interpretability evidence with cautious mixed-result diagnostics",
                "complete"
                if summary.get("evidence_status") == "initial_interpretability_evidence"
                and _to_float(summary.get("n_tested_lr_pairs")) >= 45
                and math.isfinite(_to_float(summary.get("positive_control_mean_residual_spearman")))
                and math.isfinite(_to_float(summary.get("background_lr_mean_residual_spearman")))
                else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"tested={summary.get('n_tested_lr_pairs', 'NA')}; "
                    f"pos_resid={summary.get('positive_control_mean_residual_spearman', 'NA')}; "
                    f"housekeeping_resid={summary.get('housekeeping_negative_mean_residual_spearman', 'NA')}; "
                    f"background_resid={summary.get('background_lr_mean_residual_spearman', 'NA')}"
                ),
            )
        )
    if control_path.exists():
        controls = _read_csv(control_path)
        positive = _first_row(controls, "control_class", "tonsil_positive_prior")
        housekeeping = _first_row(controls, "control_class", "housekeeping_negative_control")
        background = _first_row(controls, "control_class", "background_lr")
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "control summary separates positive, housekeeping negative, and background LR results",
                "complete" if positive and housekeeping and background else "partial",
                str(control_path.relative_to(PROJECT_ROOT)),
                (
                    f"pos_mean={positive.get('mean_residual_spearman', 'NA') if positive else 'NA'}; "
                    f"housekeeping_mean={housekeeping.get('mean_residual_spearman', 'NA') if housekeeping else 'NA'}; "
                    f"background_mean={background.get('mean_residual_spearman', 'NA') if background else 'NA'}"
                ),
            )
        )
    if n1800_summary_path.exists() and n1800_control_path.exists():
        records = _read_csv(n1800_summary_path)
        summary = records[0] if records else {}
        controls = _read_csv(n1800_control_path)
        positive = _first_row(controls, "control_class", "tonsil_positive_prior")
        housekeeping = _first_row(controls, "control_class", "housekeeping_negative_control")
        background = _first_row(controls, "control_class", "background_lr")
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "n1800 tonsil stress test records larger-subset downstream instability without overclaiming",
                "complete"
                if summary.get("evidence_status") == "initial_interpretability_evidence"
                and _to_float(summary.get("n_tested_lr_pairs")) >= 45
                and positive
                and housekeeping
                and background
                and math.isfinite(_to_float(summary.get("positive_control_mean_residual_spearman")))
                and _to_float(summary.get("significant_negative_controls_residual_fdr_0_10")) == 0
                else "partial",
                str(n1800_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"tested={summary.get('n_tested_lr_pairs', 'NA')}; "
                    f"pos_resid={summary.get('positive_control_mean_residual_spearman', 'NA')}; "
                    f"housekeeping_resid={summary.get('housekeeping_negative_mean_residual_spearman', 'NA')}; "
                    f"background_resid={summary.get('background_lr_mean_residual_spearman', 'NA')}; "
                    f"pos_sig_resid={summary.get('significant_positive_controls_residual_fdr_0_10', 'NA')}; "
                    f"neg_sig_resid={summary.get('significant_negative_controls_residual_fdr_0_10', 'NA')}"
                ),
            )
        )
    if module_path.exists():
        modules = _read_csv(module_path)
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "CCC module summary records module structure without requiring strong enrichment",
                "complete" if modules and {"ccc_module", "n_lr_pairs", "dominant_mechanism"}.issubset(modules[0]) else "partial",
                str(module_path.relative_to(PROJECT_ROOT)),
                f"n_modules={len(modules)}",
            )
        )
    if embedding_summary_path.exists():
        records = _read_csv(embedding_summary_path)
        summary = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "LR embedding summary records weak/mixed structure rather than overclaiming enrichment",
                "complete"
                if summary.get("evidence_status") == "initial_embedding_interpretability_evidence"
                and _to_float(summary.get("n_lr_pairs")) >= 45
                and math.isfinite(_to_float(summary.get("silhouette_mechanism")))
                and _to_float(summary.get("n_significant_cluster_enrichments_fdr_0_10")) == 0
                else "partial",
                str(embedding_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"lr={summary.get('n_lr_pairs', 'NA')}; "
                    f"sil_mech={summary.get('silhouette_mechanism', 'NA')}; "
                    f"sig_enrich={summary.get('n_significant_cluster_enrichments_fdr_0_10', 'NA')}; "
                    f"range_corr={summary.get('embedding_norm_spatial_range_spearman', 'NA')}"
                ),
            )
        )
    if embedding_coordinates_path.exists():
        coordinates = _read_csv(embedding_coordinates_path)
        required_columns = {
            "lr_id",
            "canonical_pair",
            "mechanism",
            "lr_category",
            "pca1",
            "pca2",
            "embedding_cluster",
            "embedding_norm",
            "score_mean",
            "residual_spearman_r",
        }
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "LR embedding coordinate table links embedding, model score, and downstream metrics",
                "complete" if coordinates and required_columns.issubset(coordinates[0]) else "partial",
                str(embedding_coordinates_path.relative_to(PROJECT_ROOT)),
                f"n={len(coordinates)}; columns={sorted(coordinates[0]) if coordinates else []}",
            )
        )
    if mechanism_summary_path.exists():
        records = _read_csv(mechanism_summary_path)
        summary = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "mechanism/category summary records mixed interpretability evidence without overclaiming",
                "complete"
                if summary.get("evidence_status") == "initial_mechanism_category_evidence"
                and _to_float(summary.get("n_lr_pairs")) >= 45
                and _to_float(summary.get("positive_minus_housekeeping_residual_spearman")) > 0
                and _to_float(summary.get("n_housekeeping_negatives_residual_bh_fdr_0_10")) == 0
                and math.isfinite(_to_float(summary.get("embedding_cluster_category_purity")))
                else "partial",
                str(mechanism_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"lr={summary.get('n_lr_pairs', 'NA')}; "
                    f"pos_minus_neg={summary.get('positive_minus_housekeeping_residual_spearman', 'NA')}; "
                    f"neg_sig={summary.get('n_housekeeping_negatives_residual_bh_fdr_0_10', 'NA')}; "
                    f"purity={summary.get('embedding_cluster_category_purity', 'NA')}"
                ),
            )
        )
    if mechanism_table_path.exists():
        mechanism = _read_csv(mechanism_table_path)
        mechanisms = {row.get("mechanism", "") for row in mechanism}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "mechanism table covers contact, paracrine, chemokine, and high-expression negative classes",
                "complete"
                if {
                    "juxtacrine_contact",
                    "short_medium_paracrine",
                    "long_range_chemokine_niche",
                    "false_positive_high_expression",
                }.issubset(mechanisms)
                and {"mean_residual_spearman", "n_residual_bh_fdr_0_10", "top_lr_by_residual"}.issubset(mechanism[0])
                else "partial",
                str(mechanism_table_path.relative_to(PROJECT_ROOT)),
                f"mechanisms={sorted(mechanisms)}",
            )
        )
    if synthesis_summary_path.exists() and synthesis_claims_path.exists():
        records = _read_csv(synthesis_summary_path)
        summary = records[0] if records else {}
        claims = _read_csv(synthesis_claims_path)
        statuses = {row.get("support_status", "") for row in claims}
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "interpretability synthesis records mixed support and explicit negative diagnostics",
                "complete"
                if summary.get("claim_level") == "mixed_initial_support_not_manuscript_ready"
                and _to_float(summary.get("support_score")) >= 4
                and _to_float(summary.get("contradiction_score")) >= 3
                and {"supported_initial", "not_supported", "mixed_support"}.issubset(statuses)
                and any(row.get("claim_id") == "lymph_node_real_background_injected_controls" for row in claims)
                else "partial",
                str(synthesis_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"claim_level={summary.get('claim_level', 'NA')}; "
                    f"support={summary.get('support_score', 'NA')}; "
                    f"contradiction={summary.get('contradiction_score', 'NA')}; "
                    f"statuses={sorted(statuses)}"
                ),
            )
        )
    if lymph_node_summary_path.exists() and lymph_node_claims_path.exists():
        records = _read_csv(lymph_node_summary_path)
        summary = records[0] if records else {}
        claims = _read_csv(lymph_node_claims_path)
        rows.append(
            AuditRow(
                "p2_goal5_interpretability",
                "lymph-node real-background injected-control interpretability records downstream and rank support",
                "complete"
                if summary.get("claim_level") == "real_background_injected_control_support"
                and _to_float(summary.get("n_positive_downstream_supported")) >= 3
                and _to_float(summary.get("n_weak_positive_downstream_supported")) >= 1
                and _to_float(summary.get("n_negative_top3")) == 0
                and _to_float(summary.get("mean_positive_downstream_spearman")) >= 0.5
                and any(row.get("claim_id") == "lymph_node_weak_positive_downstream_response" for row in claims)
                else "partial",
                str(lymph_node_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"claim_level={summary.get('claim_level', 'NA')}; "
                    f"pos={summary.get('n_positive_downstream_supported', 'NA')}; "
                    f"weak={summary.get('n_weak_positive_downstream_supported', 'NA')}; "
                    f"neg_top3={summary.get('n_negative_top3', 'NA')}; "
                    f"mean_downstream={summary.get('mean_positive_downstream_spearman', 'NA')}"
                ),
            )
        )
    return rows


def _audit_goal6() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal6_paper_figures"
    lymph_root = PROJECT_ROOT / "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior"
    lymph_baseline_root = PROJECT_ROOT / "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_figure_assembly.py"
    qa_script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_figure_qa.py"
    lymph_script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_lymph_node_validation.py"
    lymph_baseline_script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_lymph_node_baseline_validation.py"
    summary_path = root / "goal6_figure_assembly_summary.csv"
    manifest_path = root / "goal6_figure_assembly_manifest.json"
    lymph_summary_path = lymph_root / "lymph_node_validation_summary.csv"
    lymph_manifest_path = lymph_root / "lymph_node_validation_manifest.json"
    lymph_baseline_summary_path = lymph_baseline_root / "baseline_matched_validation_summary.csv"
    lymph_baseline_manifest_path = lymph_baseline_root / "baseline_matched_validation_manifest.json"
    figure_png = root / "figures/phase2_overview.png"
    figure_pdf = root / "figures/phase2_overview.pdf"
    panel_dir = root / "manuscript_panels"
    figure_qa_summary_path = root / "figure_qa/goal6_figure_qa_summary.csv"
    figure_qa_file_path = root / "figure_qa/goal6_figure_file_qa.csv"
    figure_qa_source_path = root / "figure_qa/goal6_figure_source_trace_qa.csv"
    figure_qa_manifest_path = root / "figure_qa/goal6_figure_qa_manifest.json"
    source_paths = [
        root / "source_tables/goal1_controlled_weak_signal.csv",
        root / "source_tables/goal1_mechanism_weak_signal.csv",
        root / "source_tables/goal1_full_slide_operating_point.csv",
        root / "source_tables/goal1_lr_balanced_operating_point.csv",
        root / "source_tables/goal1_lr_balanced_controlled_stability.csv",
        root / "source_tables/goal2_direct_group_fdr.csv",
        root / "source_tables/goal2_direct_group_replicate_stability.csv",
        root / "source_tables/goal3_model_selection.csv",
        root / "source_tables/goal4_high_resolution_controls.csv",
        root / "source_tables/goal4_high_resolution_scaling.csv",
        root / "source_tables/goal4_tonsil_baseline_matched.csv",
        root / "source_tables/goal4_tonsil_ccc_modules.csv",
        root / "source_tables/goal4_tonsil_ccc_modules_n1800.csv",
        root / "source_tables/goal5_interpretability.csv",
        root / "source_tables/goal5_interpretability_n1800.csv",
        root / "source_tables/goal5_mechanism_category.csv",
        root / "source_tables/goal5_lymph_node_interpretability.csv",
        root / "source_tables/goal5_interpretability_synthesis.csv",
        root / "source_tables/goal6_controlled_baseline_comparison.csv",
        root / "source_tables/goal6_full_slide_baseline_comparison.csv",
        root / "source_tables/goal6_component_ablation.csv",
        root / "source_tables/goal6_lymph_node_global_validation.csv",
        root / "source_tables/goal6_lymph_node_baseline_matched.csv",
        root / "source_tables/goal6_claim_traceability.csv",
    ]
    rows = [
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure assembly runner exists", _exists(script_path), str(script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure QA runner exists", _exists(qa_script_path), str(qa_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 lymph-node validation runner exists", _exists(lymph_script_path), str(lymph_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 lymph-node baseline validation runner exists", _exists(lymph_baseline_script_path), str(lymph_baseline_script_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 lymph-node validation summary exists", _exists(lymph_summary_path), str(lymph_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 lymph-node validation manifest exists", _exists(lymph_manifest_path), str(lymph_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 lymph-node baseline validation summary exists", _exists(lymph_baseline_summary_path), str(lymph_baseline_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 lymph-node baseline validation manifest exists", _exists(lymph_baseline_manifest_path), str(lymph_baseline_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 summary exists", _exists(summary_path), str(summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 manifest exists", _exists(manifest_path), str(manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 overview PNG exists", _exists(figure_png), str(figure_png.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Phase 2 overview PDF exists", _exists(figure_pdf), str(figure_pdf.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "manuscript panel directory exists", _exists(panel_dir), str(panel_dir.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure QA summary exists", _exists(figure_qa_summary_path), str(figure_qa_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure file QA exists", _exists(figure_qa_file_path), str(figure_qa_file_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure source trace QA exists", _exists(figure_qa_source_path), str(figure_qa_source_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal6_paper_figures", "Goal 6 figure QA manifest exists", _exists(figure_qa_manifest_path), str(figure_qa_manifest_path.relative_to(PROJECT_ROOT))),
    ]
    for path in source_paths:
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                f"source table exists: {path.name}",
                _exists(path),
                str(path.relative_to(PROJECT_ROOT)),
            )
        )
    if summary_path.exists():
        records = _read_csv(summary_path)
        summary = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 6 summary records initial figure-source status",
                "complete"
                if summary.get("evidence_status") == "initial_figure_sources"
                and _to_float(summary.get("n_source_tables")) >= 20
                and _to_float(summary.get("n_figure_files")) >= 32
                and _to_float(summary.get("n_manuscript_panel_files")) >= 30
                else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"tables={summary.get('n_source_tables', 'NA')}; "
                    f"figures={summary.get('n_figure_files', 'NA')}; "
                    f"panels={summary.get('n_manuscript_panel_files', 'NA')}"
                ),
            )
        )
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        source_tables = manifest.get("source_tables", {}) if isinstance(manifest.get("source_tables", {}), dict) else {}
        figures = manifest.get("figures", []) if isinstance(manifest.get("figures", []), list) else []
        manuscript_panels = (
            manifest.get("manuscript_panels", [])
            if isinstance(manifest.get("manuscript_panels", []), list)
            else []
        )
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "manifest links overview and manuscript panels to all current Phase 2 source tables",
                "complete"
                if len(source_tables) >= 20
                and len(figures) >= 2
                and len(manuscript_panels) >= 30
                and all((PROJECT_ROOT / path).exists() for path in source_tables.values())
                and all((PROJECT_ROOT / path).exists() for path in figures)
                and all((PROJECT_ROOT / path).exists() for path in manuscript_panels)
                else "partial",
                str(manifest_path.relative_to(PROJECT_ROOT)),
                f"source_tables={sorted(source_tables)}; figures={len(figures)}; panels={len(manuscript_panels)}",
            )
        )
    for path in source_paths:
        if path.exists():
            records = _read_csv(path)
            rows.append(
                AuditRow(
                    "p2_goal6_paper_figures",
                    f"source table is non-empty: {path.name}",
                    "complete" if len(records) > 0 else "partial",
                    str(path.relative_to(PROJECT_ROOT)),
                    f"rows={len(records)}",
                )
            )
    if figure_png.exists():
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "overview PNG is non-empty",
                "complete" if figure_png.stat().st_size > 100_000 else "partial",
                str(figure_png.relative_to(PROJECT_ROOT)),
                f"bytes={figure_png.stat().st_size}",
            )
        )
    if figure_pdf.exists():
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "overview PDF is non-empty",
                "complete" if figure_pdf.stat().st_size > 10_000 else "partial",
                str(figure_pdf.relative_to(PROJECT_ROOT)),
                f"bytes={figure_pdf.stat().st_size}",
            )
        )
    if panel_dir.exists():
        panel_pngs = sorted(panel_dir.glob("*.png"))
        panel_pdfs = sorted(panel_dir.glob("*.pdf"))
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "manuscript-style panels are exported as PNG and PDF",
                "complete" if len(panel_pngs) >= 15 and len(panel_pdfs) >= 15 else "partial",
                str(panel_dir.relative_to(PROJECT_ROOT)),
                f"png={len(panel_pngs)}; pdf={len(panel_pdfs)}",
            )
        )
        small = [path.name for path in panel_pngs + panel_pdfs if path.stat().st_size <= 10_000]
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "manuscript-style panel files are non-empty",
                "complete" if not small else "partial",
                str(panel_dir.relative_to(PROJECT_ROOT)),
                f"small_files={small}",
            )
        )
    if figure_qa_summary_path.exists():
        records = _read_csv(figure_qa_summary_path)
        summary = records[0] if records else {}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "figure QA records PNG/PDF bundle, source traceability, and remaining export risk",
                "complete"
                if summary.get("evidence_status") == "figure_qa_complete"
                and summary.get("qa_pass") == "True"
                and _to_float(summary.get("n_manuscript_panel_files")) >= 30
                and _to_float(summary.get("n_source_tables")) >= 24
                and _to_float(summary.get("n_missing_figure_files")) == 0
                and _to_float(summary.get("n_missing_source_tables")) == 0
                and summary.get("export_bundle_status") == "png_pdf_ready_svg_tiff_missing"
                and summary.get("manual_review_required") == "True"
                else "partial",
                str(figure_qa_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={summary.get('evidence_status', 'NA')}; "
                    f"pass={summary.get('qa_pass', 'NA')}; "
                    f"panels={summary.get('n_manuscript_panel_files', 'NA')}; "
                    f"sources={summary.get('n_source_tables', 'NA')}; "
                    f"export={summary.get('export_bundle_status', 'NA')}"
                ),
            )
        )
    if figure_qa_file_path.exists() and figure_qa_source_path.exists():
        file_qa = _read_csv(figure_qa_file_path)
        source_qa = _read_csv(figure_qa_source_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "figure QA detail tables have paired exports and non-empty source tables",
                "complete"
                if file_qa
                and source_qa
                and all(row.get("exists") == "True" for row in file_qa)
                and all(row.get("size_pass") == "True" for row in file_qa)
                and all(row.get("paired_format_exists") == "True" for row in file_qa)
                and all(row.get("exists") == "True" for row in source_qa)
                and all(row.get("non_empty") == "True" for row in source_qa)
                else "partial",
                str(figure_qa_file_path.relative_to(PROJECT_ROOT)),
                f"files={len(file_qa)}; sources={len(source_qa)}",
            )
        )
    baseline_controlled_path = root / "source_tables/goal6_controlled_baseline_comparison.csv"
    goal1_mechanism_path = root / "source_tables/goal1_mechanism_weak_signal.csv"
    goal1_lr_balanced_path = root / "source_tables/goal1_lr_balanced_operating_point.csv"
    goal1_lr_balanced_stability_path = root / "source_tables/goal1_lr_balanced_controlled_stability.csv"
    goal2_replicate_path = root / "source_tables/goal2_direct_group_replicate_stability.csv"
    baseline_full_slide_path = root / "source_tables/goal6_full_slide_baseline_comparison.csv"
    component_ablation_path = root / "source_tables/goal6_component_ablation.csv"
    highres_scaling_path = root / "source_tables/goal4_high_resolution_scaling.csv"
    tonsil_baseline_path = root / "source_tables/goal4_tonsil_baseline_matched.csv"
    tonsil_modules_path = root / "source_tables/goal4_tonsil_ccc_modules.csv"
    tonsil_modules_n1800_path = root / "source_tables/goal4_tonsil_ccc_modules_n1800.csv"
    interpretability_n1800_path = root / "source_tables/goal5_interpretability_n1800.csv"
    mechanism_category_path = root / "source_tables/goal5_mechanism_category.csv"
    lymph_node_interpretability_path = root / "source_tables/goal5_lymph_node_interpretability.csv"
    interpretability_synthesis_path = root / "source_tables/goal5_interpretability_synthesis.csv"
    lymph_node_path = root / "source_tables/goal6_lymph_node_global_validation.csv"
    lymph_baseline_path = root / "source_tables/goal6_lymph_node_baseline_matched.csv"
    claim_trace_path = root / "source_tables/goal6_claim_traceability.csv"
    if baseline_controlled_path.exists():
        baseline = _read_csv(baseline_controlled_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "controlled baseline source records evidence levels and claim eligibility",
                "complete"
                if baseline
                and {"baseline_evidence_level", "claim_eligible", "claim_scope"}.issubset(baseline[0])
                and any(row.get("claim_eligible") == "True" for row in baseline)
                and any(row.get("claim_eligible") == "False" for row in baseline)
                else "partial",
                str(baseline_controlled_path.relative_to(PROJECT_ROOT)),
                f"rows={len(baseline)}",
            )
        )
    if goal1_mechanism_path.exists():
        mechanism = _read_csv(goal1_mechanism_path)
        mechanisms = {row.get("mechanism", "") for row in mechanism}
        analyses = {row.get("analysis_label", "") for row in mechanism}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 1 mechanism source records mechanism-specific weak-signal diagnostics",
                "complete"
                if mechanism
                and {"display_ablation", "display_mechanism", "weak_signal_recall_at_threshold_mean", "analysis_scope"}.issubset(mechanism[0])
                and {"controlled_five_replicate_initial", "full_slide_sparse_operating_q94"}.issubset(analyses)
                and "long_range_chemokine_niche" in mechanisms
                else "partial",
                str(goal1_mechanism_path.relative_to(PROJECT_ROOT)),
                f"analyses={sorted(analyses)}; mechanisms={sorted(mechanisms)}",
            )
        )
    if goal1_lr_balanced_path.exists():
        lr_balanced = _read_csv(goal1_lr_balanced_path)
        best = _first_row(lr_balanced, "strategy", "lrwise_quantile_with_global_floor")
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 1 LR-balanced source records stronger post-hoc operating diagnostic",
                "complete"
                if best
                and {"display_label", "analysis_scope", "best_weak_signal_recall", "best_fdr_delta_vs_reference"}.issubset(best)
                and best.get("status") == "passing_operating_point"
                and _to_float(best.get("best_weak_signal_recall")) > 0.080
                and _to_float(best.get("best_fdr_delta_vs_reference")) <= 0.02
                else "partial",
                str(goal1_lr_balanced_path.relative_to(PROJECT_ROOT)),
                (
                    f"weak={best.get('best_weak_signal_recall') if best else 'NA'}; "
                    f"fdr_delta={best.get('best_fdr_delta_vs_reference') if best else 'NA'}"
                ),
            )
        )
    if goal1_lr_balanced_stability_path.exists():
        stability = _read_csv(goal1_lr_balanced_stability_path)
        lr_balanced = _first_row(stability, "strategy", "lrwise_quantile_with_global_floor")
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 1 LR-balanced controlled-stability source records failed promotion criterion",
                "complete"
                if lr_balanced
                and {"display_label", "analysis_scope", "passing_replicate_fraction"}.issubset(lr_balanced)
                and _to_float(lr_balanced.get("passing_replicate_fraction")) == 0.0
                and _to_float(lr_balanced.get("best_fdr_delta_vs_reference_mean")) > 0.02
                else "partial",
                str(goal1_lr_balanced_stability_path.relative_to(PROJECT_ROOT)),
                (
                    f"pass_frac={lr_balanced.get('passing_replicate_fraction') if lr_balanced else 'NA'}; "
                    f"fdr_delta={lr_balanced.get('best_fdr_delta_vs_reference_mean') if lr_balanced else 'NA'}"
                ),
            )
        )
    if goal2_replicate_path.exists():
        replicate = _read_csv(goal2_replicate_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 2 replicate-stability source records controlled cross-replicate FDR support",
                "complete"
                if len(replicate) >= 5
                and {
                    "replicate",
                    "phase2_goal2_pass",
                    "positive_control_significant_groups",
                    "negative_control_significant_groups",
                    "summary_passing_replicate_fraction",
                    "analysis_scope",
                }.issubset(replicate[0])
                and all(str(row.get("phase2_goal2_pass", "")).lower() == "true" for row in replicate)
                and max(_to_float(row.get("negative_control_significant_groups")) for row in replicate) == 0
                and all("controlled synthetic" in row.get("analysis_scope", "") for row in replicate)
                else "partial",
                str(goal2_replicate_path.relative_to(PROJECT_ROOT)),
                f"rows={len(replicate)}",
            )
        )
    if baseline_full_slide_path.exists():
        baseline = _read_csv(baseline_full_slide_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "full-slide baseline source records single-replicate scope",
                "complete"
                if baseline
                and {"baseline_evidence_level", "claim_eligible", "claim_scope"}.issubset(baseline[0])
                and all(row.get("baseline_evidence_level") == "single_full_slide_replicate" for row in baseline)
                else "partial",
                str(baseline_full_slide_path.relative_to(PROJECT_ROOT)),
                f"rows={len(baseline)}",
            )
        )
    if component_ablation_path.exists():
        ablation = _read_csv(component_ablation_path)
        groups = {row.get("component_group", "") for row in ablation}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "component ablation source records Phase 1 component-level evidence",
                "complete"
                if len(ablation) >= 10
                and {"ablation", "display_label", "component_group", "aupr_mean", "claim_scope"}.issubset(ablation[0])
                and {"lr_identity", "architecture_graph", "distance", "expression"}.issubset(groups)
                and all("Phase 1 controlled" in row.get("claim_scope", "") for row in ablation)
                else "partial",
                str(component_ablation_path.relative_to(PROJECT_ROOT)),
                f"rows={len(ablation)}; groups={sorted(groups)}",
            )
        )
    if highres_scaling_path.exists():
        scaling = _read_csv(highres_scaling_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "high-resolution scaling source records subset size, runtime, memory, and controls",
                "complete"
                if scaling
                and {
                    "n_cells",
                    "n_candidate_edges",
                    "wall_clock_seconds",
                    "peak_rss_mb",
                    "n_positive_controls_top_decile",
                    "n_negative_controls_top_decile",
                    "summary_n_cells_max",
                }.issubset(scaling[0])
                and len(scaling) >= 3
                and max(_to_float(row.get("n_cells")) for row in scaling) >= 1800
                else "partial",
                str(highres_scaling_path.relative_to(PROJECT_ROOT)),
                f"rows={len(scaling)}",
            )
        )
    if tonsil_baseline_path.exists():
        tonsil = _read_csv(tonsil_baseline_path)
        by_method = {row.get("method", ""): row for row in tonsil}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "tonsil baseline-matched source records same-subset control ranking and baseline limitations",
                "complete"
                if {"spatialself_wavelet_calibrated", "commot", "cellchat", "spatalk"}.issubset(by_method)
                and {"analysis_scope", "claim_eligible", "rank_validation_pass"}.issubset(tonsil[0])
                and by_method["spatialself_wavelet_calibrated"].get("rank_validation_pass") == "True"
                and by_method["spatalk"].get("claim_eligible") == "False"
                else "partial",
                str(tonsil_baseline_path.relative_to(PROJECT_ROOT)),
                f"methods={sorted(by_method)}",
            )
        )
    if tonsil_modules_path.exists():
        modules = _read_csv(tonsil_modules_path)
        dominant = {row.get("dominant_mechanism", "") for row in modules}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "tonsil CCC module source records module/control separation with cautious scope",
                "complete"
                if modules
                and {
                    "ccc_module",
                    "display_module",
                    "dominant_mechanism",
                    "display_mechanism",
                    "n_positive_controls",
                    "n_housekeeping_negative_controls",
                    "positive_control_bh_fdr",
                    "housekeeping_negative_bh_fdr",
                    "analysis_scope",
                }.issubset(modules[0])
                and len(modules) >= 3
                and "long_range_chemokine_niche" in dominant
                and "false_positive_high_expression" in dominant
                and any(
                    row.get("dominant_mechanism") == "long_range_chemokine_niche"
                    and _to_float(row.get("n_positive_controls")) >= 3
                    and _to_float(row.get("n_housekeeping_negative_controls")) == 0
                    and _to_float(row.get("positive_control_bh_fdr")) <= 0.10
                    for row in modules
                )
                and any(
                    row.get("dominant_mechanism") == "false_positive_high_expression"
                    and _to_float(row.get("n_housekeeping_negative_controls")) >= 3
                    and _to_float(row.get("housekeeping_negative_bh_fdr")) <= 0.10
                    for row in modules
                )
                and all("not strong spatial confinement" in row.get("analysis_scope", "") for row in modules)
                else "partial",
                str(tonsil_modules_path.relative_to(PROJECT_ROOT)),
                f"rows={len(modules)}; mechanisms={sorted(dominant)}",
            )
        )
    if tonsil_modules_n1800_path.exists():
        modules = _read_csv(tonsil_modules_n1800_path)
        mixed = [
            row for row in modules
            if _to_float(row.get("n_positive_controls")) > 0
            and _to_float(row.get("n_housekeeping_negative_controls")) > 0
        ]
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "n1800 CCC module source records larger-subset module-separation stress test",
                "complete"
                if modules
                and {
                    "ccc_module",
                    "n_positive_controls",
                    "n_housekeeping_negative_controls",
                    "positive_control_bh_fdr",
                    "housekeeping_negative_bh_fdr",
                    "analysis_scope",
                }.issubset(modules[0])
                and mixed
                and all("stress test" in row.get("analysis_scope", "") for row in modules)
                else "partial",
                str(tonsil_modules_n1800_path.relative_to(PROJECT_ROOT)),
                f"rows={len(modules)}; mixed={[(row.get('ccc_module'), row.get('n_positive_controls'), row.get('n_housekeeping_negative_controls')) for row in mixed]}",
            )
        )
    if mechanism_category_path.exists():
        mechanism = _read_csv(mechanism_category_path)
        mechanisms = {row.get("mechanism", "") for row in mechanism}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "mechanism/category source records mixed biological interpretability diagnostics",
                "complete"
                if mechanism
                and {"analysis_scope", "display_label", "mean_residual_spearman", "overview_positive_minus_housekeeping_residual_spearman"}.issubset(mechanism[0])
                and {
                    "juxtacrine_contact",
                    "short_medium_paracrine",
                    "long_range_chemokine_niche",
                    "false_positive_high_expression",
                }.issubset(mechanisms)
                else "partial",
                str(mechanism_category_path.relative_to(PROJECT_ROOT)),
                f"mechanisms={sorted(mechanisms)}",
            )
        )
    if interpretability_n1800_path.exists():
        n1800 = _read_csv(interpretability_n1800_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "n1800 interpretability source records larger-subset stress-test scope",
                "complete"
                if n1800
                and {
                    "control_class",
                    "mean_residual_spearman",
                    "summary_positive_control_mean_residual_spearman",
                    "summary_housekeeping_negative_mean_residual_spearman",
                    "analysis_scope",
                }.issubset(n1800[0])
                and all("stress test" in row.get("analysis_scope", "") for row in n1800)
                else "partial",
                str(interpretability_n1800_path.relative_to(PROJECT_ROOT)),
                f"rows={len(n1800)}",
            )
        )
    if interpretability_synthesis_path.exists():
        synthesis = _read_csv(interpretability_synthesis_path)
        statuses = {row.get("support_status", "") for row in synthesis}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Goal 5 synthesis source constrains interpretability claims",
                "complete"
                if synthesis
                and {
                    "claim_id",
                    "support_status",
                    "summary_claim_level",
                    "summary_support_score",
                    "summary_contradiction_score",
                    "analysis_scope",
                }.issubset(synthesis[0])
                and synthesis[0].get("summary_claim_level") == "mixed_initial_support_not_manuscript_ready"
                and {"supported_initial", "not_supported", "mixed_support"}.issubset(statuses)
                and _to_float(synthesis[0].get("summary_support_score")) >= 4
                and _to_float(synthesis[0].get("summary_contradiction_score")) >= 3
                and any(row.get("claim_id") == "lymph_node_real_background_injected_controls" for row in synthesis)
                else "partial",
                str(interpretability_synthesis_path.relative_to(PROJECT_ROOT)),
                f"rows={len(synthesis)}; statuses={sorted(statuses)}",
            )
        )
    if lymph_node_interpretability_path.exists():
        lymph_interp = _read_csv(lymph_node_interpretability_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "lymph-node interpretability source records real-background downstream/rank support",
                "complete"
                if lymph_interp
                and {
                    "claim_id",
                    "support_status",
                    "summary_claim_level",
                    "summary_n_positive_downstream_supported",
                    "summary_n_weak_positive_downstream_supported",
                    "summary_n_negative_top3",
                    "analysis_scope",
                }.issubset(lymph_interp[0])
                and lymph_interp[0].get("summary_claim_level") == "real_background_injected_control_support"
                and _to_float(lymph_interp[0].get("summary_n_positive_downstream_supported")) >= 3
                and _to_float(lymph_interp[0].get("summary_n_weak_positive_downstream_supported")) >= 1
                and _to_float(lymph_interp[0].get("summary_n_negative_top3")) == 0
                else "partial",
                str(lymph_node_interpretability_path.relative_to(PROJECT_ROOT)),
                f"rows={len(lymph_interp)}",
            )
        )
    if lymph_summary_path.exists():
        lymph_summary = _read_csv(lymph_summary_path)
        row = lymph_summary[0] if lymph_summary else {}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Phase 2 lymph-node validation records full-slide injected-control evidence",
                "complete"
                if row.get("evidence_status") == "phase2_initial_lymph_node_validation"
                and _to_float(row.get("n_cells")) >= 4000
                and _to_float(row.get("n_candidate_edges")) >= 300000
                and row.get("positive_controls_rank_ahead_of_negatives") == "True"
                and _to_float(row.get("negative_top3_count")) == 0
                and _to_float(row.get("n_downstream_bh_fdr_0_05")) >= 3
                else "partial",
                str(lymph_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={row.get('evidence_status', 'NA')}; "
                    f"cells={row.get('n_cells', 'NA')}; edges={row.get('n_candidate_edges', 'NA')}; "
                    f"rank_ahead={row.get('positive_controls_rank_ahead_of_negatives', 'NA')}; "
                    f"neg_top3={row.get('negative_top3_count', 'NA')}; "
                    f"downstream_sig={row.get('n_downstream_bh_fdr_0_05', 'NA')}"
                ),
            )
        )
    if lymph_baseline_summary_path.exists():
        lymph_baseline_summary = _read_csv(lymph_baseline_summary_path)
        row = lymph_baseline_summary[0] if lymph_baseline_summary else {}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "Phase 2 lymph-node baseline-matched validation records retained-method comparison",
                "complete"
                if row.get("evidence_status") == "phase2_initial_baseline_matched_lymph_node_validation"
                and _to_float(row.get("n_claim_eligible_methods")) >= 7
                and _to_float(row.get("n_methods_rank_validation_pass")) >= 3
                and row.get("spatialself_edge_prior_rank_validation_pass") == "True"
                else "partial",
                str(lymph_baseline_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"status={row.get('evidence_status', 'NA')}; "
                    f"methods={row.get('n_claim_eligible_methods', 'NA')}; "
                    f"pass={row.get('n_methods_rank_validation_pass', 'NA')}; "
                    f"spatialself_pass={row.get('spatialself_edge_prior_rank_validation_pass', 'NA')}"
                ),
            )
        )
    if lymph_node_path.exists():
        lymph_node = _read_csv(lymph_node_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "lymph node source records real-background injected-control validation scope",
                "complete"
                if lymph_node
                and {
                    "dataset_context",
                    "analysis_scope",
                    "display_label",
                    "control_type",
                    "global_rank",
                    "best_region_enrichment_score",
                    "downstream_spearman_r",
                    "method_negative_top3_count",
                }.issubset(lymph_node[0])
                and len(lymph_node) >= 5
                and any(row.get("control_type") == "positive_weak" for row in lymph_node)
                and any(row.get("false_positive_high_expression") == "True" for row in lymph_node)
                and all("injected-control" in row.get("analysis_scope", "") for row in lymph_node)
                and all("Phase 2" in row.get("analysis_scope", "") for row in lymph_node)
                else "partial",
                str(lymph_node_path.relative_to(PROJECT_ROOT)),
                f"rows={len(lymph_node)}",
            )
        )
    if lymph_baseline_path.exists():
        lymph_baseline = _read_csv(lymph_baseline_path)
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "lymph-node baseline-matched source records method scopes and rank-control outcomes",
                "complete"
                if lymph_baseline
                and {
                    "method",
                    "method_evidence_scope",
                    "claim_eligible",
                    "mean_positive_rank",
                    "mean_negative_rank",
                    "negative_top3_count",
                    "rank_validation_pass",
                    "overview_n_claim_eligible_methods",
                }.issubset(lymph_baseline[0])
                and len(lymph_baseline) >= 7
                and any(row.get("method") == "spatialself_wavelet_edge_prior" and row.get("rank_validation_pass") == "True" for row in lymph_baseline)
                and any(row.get("rank_validation_pass") == "False" for row in lymph_baseline)
                else "partial",
                str(lymph_baseline_path.relative_to(PROJECT_ROOT)),
                f"rows={len(lymph_baseline)}",
            )
        )
    if claim_trace_path.exists():
        claims = _read_csv(claim_trace_path)
        statuses = {row.get("support_status", "") for row in claims}
        rows.append(
            AuditRow(
                "p2_goal6_paper_figures",
                "claim traceability table records supported, mixed, and traceability-only claims",
                "complete"
                if len(claims) >= 14
                and {"supported_initial", "mixed_initial", "traceability_ready"}.issubset(statuses)
                and any(row.get("claim_id") == "goal2_controlled_replicate_fdr" for row in claims)
                and any(row.get("claim_id") == "goal4_tonsil_ccc_modules" for row in claims)
                and any(row.get("claim_id") == "goal4_tonsil_n1800_ccc_module_stress_test" for row in claims)
                and any(row.get("claim_id") == "goal5_n1800_downstream_stress_test" for row in claims)
                and any(row.get("claim_id") == "goal5_interpretability_synthesis" for row in claims)
                and any(row.get("claim_id") == "goal5_lymph_node_real_background_interpretability" for row in claims)
                and any(row.get("claim_id") == "lymph_node_real_background_controls" for row in claims)
                and any(row.get("claim_id") == "lymph_node_baseline_matched_controls" for row in claims)
                and all((PROJECT_ROOT / row.get("evidence_artifact", "")).exists() for row in claims)
                else "partial",
                str(claim_trace_path.relative_to(PROJECT_ROOT)),
                f"rows={len(claims)}; statuses={sorted(statuses)}",
            )
        )
    return rows


def _audit_goal2() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial"
    grid_root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid"
    focused_root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused"
    seed_root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability"
    replicate_root = PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability"
    summary_path = root / "fdr_calibration_summary.csv"
    group_summary_path = root / "fdr_calibration_group_summary.csv"
    manifest_path = root / "fdr_calibration_manifest.json"
    grid_summary_path = grid_root / "direct_group_grid_summary.csv"
    focused_summary_path = focused_root / "direct_group_grid_summary.csv"
    seed_summary_path = seed_root / "direct_group_seed_stability_summary.csv"
    seed_per_seed_path = seed_root / "direct_group_seed_stability_per_seed.csv"
    seed_manifest_path = seed_root / "direct_group_seed_stability_manifest.json"
    replicate_summary_path = replicate_root / "direct_group_replicate_stability_summary.csv"
    replicate_per_replicate_path = replicate_root / "direct_group_replicate_stability_per_replicate.csv"
    replicate_manifest_path = replicate_root / "direct_group_replicate_stability_manifest.json"
    rows = [
        AuditRow("p2_goal2_fdr_calibration", "calibration summary exists", _exists(summary_path), str(summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "group calibration summary exists", _exists(group_summary_path), str(group_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "calibration manifest exists", _exists(manifest_path), str(manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "direct group grid summary exists", _exists(grid_summary_path), str(grid_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "focused 500-permutation direct group grid summary exists", _exists(focused_summary_path), str(focused_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group seed-stability summary exists", _exists(seed_summary_path), str(seed_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group seed-stability per-seed table exists", _exists(seed_per_seed_path), str(seed_per_seed_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group seed-stability manifest exists", _exists(seed_manifest_path), str(seed_manifest_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group replicate-stability summary exists", _exists(replicate_summary_path), str(replicate_summary_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group replicate-stability per-replicate table exists", _exists(replicate_per_replicate_path), str(replicate_per_replicate_path.relative_to(PROJECT_ROOT))),
        AuditRow("p2_goal2_fdr_calibration", "selected direct group replicate-stability manifest exists", _exists(replicate_manifest_path), str(replicate_manifest_path.relative_to(PROJECT_ROOT))),
    ]
    if summary_path.exists():
        summary = _read_csv(summary_path)
        nulls = {row.get("null_model", "") for row in summary}
        required_nulls = {
            "score_permutation",
            "spatial_permutation",
            "receiver_permutation",
            "lr_identity_permutation",
            "within_region_permutation",
            "cell_type_preserving_permutation",
        }
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "calibration covers Phase 2 null models",
                "complete" if required_nulls.issubset(nulls) else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_nulls - nulls)}",
            )
        )
        score = _first_row(summary, "null_model", "score_permutation")
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "operating point carries full-slide weak-recall/FDR diagnostics",
                "complete"
                if score
                and _to_float(score.get("operating_weak_recall")) > 0.040
                and _to_float(score.get("operating_fp_high_expression_rate")) <= 0.005
                else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"q={score.get('operating_quantile') if score else 'NA'}; "
                    f"weak={score.get('operating_weak_recall') if score else 'NA'}; "
                    f"fdr={score.get('operating_realized_fdr') if score else 'NA'}; "
                    f"fp={score.get('operating_fp_high_expression_rate') if score else 'NA'}"
                ),
            )
        )
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "empirical-null diagnostic columns are present",
                "complete"
                if score
                and "min_gaussian_bh_fdr" in score
                and "n_local_fdr_edges" in score
                and "max_null_z" in score
                else "partial",
                str(summary_path.relative_to(PROJECT_ROOT)),
                f"columns={sorted(score) if score else []}",
            )
        )
    if group_summary_path.exists():
        group_summary = _read_csv(group_summary_path)
        levels = {row.get("group_level", "") for row in group_summary}
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "group FDR covers LR-pair, cell-type-pair, and pathway/module levels",
                "complete" if {"lr_pair", "cell_type_pair", "pathway_module"}.issubset(levels) else "partial",
                str(group_summary_path.relative_to(PROJECT_ROOT)),
                f"levels={sorted(levels)}",
            )
        )
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "group FDR tracks positive and high-expression negative controls",
                "complete"
                if group_summary
                and all(
                    {
                        "positive_control_groups",
                        "positive_control_significant_groups",
                        "negative_control_groups",
                        "negative_control_significant_groups",
                    }.issubset(row)
                    for row in group_summary
                )
                else "partial",
                str(group_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"positive_sig={sorted({row.get('positive_control_significant_groups') for row in group_summary})}; "
                    f"negative_sig={sorted({row.get('negative_control_significant_groups') for row in group_summary})}"
                ),
            )
        )
        tests = {row.get("group_test", "") for row in group_summary}
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "group FDR includes Simes edge-p and direct top-tail permutation tests",
                "complete" if {"simes_edge_p", "direct_top_tail_permutation"}.issubset(tests) else "partial",
                str(group_summary_path.relative_to(PROJECT_ROOT)),
                f"tests={sorted(tests)}",
            )
        )
    if focused_summary_path.exists():
        focused = _read_csv(focused_summary_path)
        passing = [
            row
            for row in focused
            if row.get("status") == "completed"
            and str(row.get("phase2_goal2_pass", "")).lower() == "true"
            and _to_float(row.get("positive_control_min_group_bh_fdr")) <= 0.05
            and _to_float(row.get("negative_control_significant_groups")) == 0
        ]
        receiver_rows = [
            row
            for row in focused
            if row.get("null_model") == "receiver_permutation"
            and row.get("status") == "completed"
        ]
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "focused direct group grid finds positive-control discoveries with negative controls suppressed",
                "complete" if passing else "partial",
                str(focused_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"passing={len(passing)}; "
                    f"best={_best_direct_grid_detail(passing)}"
                ),
            )
        )
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "focused direct group grid records strict receiver-permutation low-power null",
                "complete" if receiver_rows else "partial",
                str(focused_summary_path.relative_to(PROJECT_ROOT)),
                f"receiver_rows={len(receiver_rows)}",
            )
        )
    if seed_summary_path.exists():
        seed_records = _read_csv(seed_summary_path)
        seed_summary = seed_records[0] if seed_records else {}
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "selected LR-identity LR-pair direct FDR setting is stable across random seeds",
                "complete"
                if seed_summary
                and seed_summary.get("status") == "seed_stability_completed"
                and _to_float(seed_summary.get("passing_seed_fraction")) == 1.0
                and _to_float(seed_summary.get("n_completed_seeds")) >= 5
                and _to_float(seed_summary.get("max_negative_control_significant_groups")) == 0
                and _to_float(seed_summary.get("min_positive_control_significant_groups")) >= 1
                else "partial",
                str(seed_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"seeds={seed_summary.get('n_completed_seeds', 'NA')}/{seed_summary.get('n_seeds', 'NA')}; "
                    f"pass_frac={seed_summary.get('passing_seed_fraction', 'NA')}; "
                    f"pos_sig_range={seed_summary.get('min_positive_control_significant_groups', 'NA')}-"
                    f"{seed_summary.get('max_positive_control_significant_groups', 'NA')}; "
                    f"max_neg_sig={seed_summary.get('max_negative_control_significant_groups', 'NA')}; "
                    f"claim={seed_summary.get('claim_level', 'NA')}"
                ),
            )
        )
    if replicate_summary_path.exists():
        replicate_records = _read_csv(replicate_summary_path)
        replicate_summary = replicate_records[0] if replicate_records else {}
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "selected LR-identity LR-pair direct FDR setting is stable across controlled replicates",
                "complete"
                if replicate_summary
                and replicate_summary.get("status") == "replicate_stability_completed"
                and _to_float(replicate_summary.get("passing_replicate_fraction")) == 1.0
                and _to_float(replicate_summary.get("n_completed_replicates")) >= 5
                and _to_float(replicate_summary.get("max_negative_control_significant_groups")) == 0
                and _to_float(replicate_summary.get("min_positive_control_significant_groups")) >= 1
                else "partial",
                str(replicate_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"replicates={replicate_summary.get('n_completed_replicates', 'NA')}/"
                    f"{replicate_summary.get('n_replicates', 'NA')}; "
                    f"pass_frac={replicate_summary.get('passing_replicate_fraction', 'NA')}; "
                    f"pos_sig_range={replicate_summary.get('min_positive_control_significant_groups', 'NA')}-"
                    f"{replicate_summary.get('max_positive_control_significant_groups', 'NA')}; "
                    f"max_neg_sig={replicate_summary.get('max_negative_control_significant_groups', 'NA')}; "
                    f"claim={replicate_summary.get('claim_level', 'NA')}"
                ),
            )
        )
    if replicate_per_replicate_path.exists():
        per_replicate = _read_csv(replicate_per_replicate_path)
        rows.append(
            AuditRow(
                "p2_goal2_fdr_calibration",
                "controlled replicate-stability table records every replicate and suppresses high-expression negatives",
                "complete"
                if len(per_replicate) >= 5
                and all(row.get("status") == "completed" for row in per_replicate)
                and all(str(row.get("phase2_goal2_pass", "")).lower() == "true" for row in per_replicate)
                and max(_to_float(row.get("negative_control_significant_groups")) for row in per_replicate) == 0
                else "partial",
                str(replicate_per_replicate_path.relative_to(PROJECT_ROOT)),
                f"rows={len(per_replicate)}",
            )
        )
    return rows


def _best_direct_grid_detail(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "NA"
    best = min(rows, key=lambda row: _to_float(row.get("positive_control_min_group_bh_fdr")))
    return (
        f"{best.get('null_model')}/{best.get('group_level')}/top{best.get('top_fraction')}:"
        f"pos_fdr={best.get('positive_control_min_group_bh_fdr')},"
        f"pos_sig={best.get('positive_control_significant_groups')},"
        f"neg_sig={best.get('negative_control_significant_groups')}"
    )


def _exists(path: Path) -> str:
    return "complete" if path.exists() else "missing"


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


if __name__ == "__main__":
    main()
