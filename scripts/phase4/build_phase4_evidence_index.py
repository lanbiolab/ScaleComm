#!/usr/bin/env python
"""Build the Phase 4 manuscript evidence-lock index.

This script does not rerun expensive experiments. It reads the current Phase 2
and Phase 3 artifacts and classifies each Phase 4 goal against manuscript gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_COLUMNS = [
    "goal_id",
    "goal_title",
    "evidence_area",
    "tier",
    "status",
    "current_value",
    "target_value",
    "evidence",
    "supported_claim",
    "limitation",
    "next_action",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase4_manuscript_evidence_lock.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase4/manuscript_evidence_lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase4_evidence_index.csv"))

    sources = (config.get("source_artifacts") or {})
    gates = config.get("submission_gates") or {}
    rows: list[dict[str, Any]] = []
    rows.extend(_goal1_benchmarks(sources, gates))
    rows.extend(_goal2_baselines(sources, config))
    rows.extend(_goal3_fdr(sources, gates))
    rows.extend(_goal4_weak_signal(sources))
    rows.extend(_goal5_real_validation(sources))
    rows.extend(_goal6_ablation(sources))
    rows.extend(_goal7_figures(sources))

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path, config_path)
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 4 evidence index: {output_path}")
    print(json.dumps(summary, indent=2))


def _goal1_benchmarks(sources: dict[str, Any], gates: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = sources.get("phase2_revised") or {}
    controlled_path = PROJECT_ROOT / p2.get("controlled_simulator_summary", "")
    real_path = PROJECT_ROOT / p2.get("real_background_summary", "")
    required_mechanisms = set(gates.get("required_mechanisms") or [])
    rows: list[dict[str, Any]] = []

    controlled = _first(_read_csv(controlled_path))
    n_controlled = int(float(controlled.get("n_replicates") or 0)) if controlled else 0
    controlled_mechanisms = set(str(controlled.get("mechanisms_observed", "")).split("|")) if controlled else set()
    rows.append(
        _row(
            "goal1_formalize_benchmarks",
            "Formalize key benchmarks",
            "controlled_simulator_v2_replicates",
            _tier_for_count(n_controlled, int(gates.get("benchmark_replicates_min", 5)), representative_min=3),
            "complete" if n_controlled >= int(gates.get("benchmark_replicates_min", 5)) else "needs_recompute",
            str(n_controlled),
            str(gates.get("benchmark_replicates_min", 5)),
            controlled_path,
            "Controlled simulator has mechanism-layered truth tables and replicate summaries.",
            "Current formal simulator replicate count is below the Phase 4 manuscript gate." if n_controlled < int(gates.get("benchmark_replicates_min", 5)) else "",
            "Regenerate controlled simulator v2 with at least five replicates and rerun the matched baseline panel.",
        )
    )
    rows.append(
        _row(
            "goal1_formalize_benchmarks",
            "Formalize key benchmarks",
            "controlled_mechanism_coverage",
            "formal" if required_mechanisms.issubset(controlled_mechanisms) else "partial",
            "complete" if required_mechanisms.issubset(controlled_mechanisms) else "needs_design_fix",
            "|".join(sorted(controlled_mechanisms)),
            "|".join(sorted(required_mechanisms)),
            controlled_path,
            "Controlled benchmark includes contact, paracrine, long-range niche, and high-expression negative-control mechanisms.",
            "",
            "Keep mechanism truth contract frozen when increasing replicates.",
        )
    )

    real_rows = _read_csv(real_path)
    tissues = {row.get("dataset_key", "") for row in real_rows}
    replicate_counts = [int(float(row.get("n_replicates") or 0)) for row in real_rows]
    min_reps = min(replicate_counts) if replicate_counts else 0
    real_mechanisms = set()
    for row in real_rows:
        real_mechanisms.update(str(row.get("mechanisms_observed", "")).split("|"))
    target_tissues = int(gates.get("real_background_tissues_min", 3))
    target_reps = int(gates.get("real_background_replicates_per_tissue_min", 5))
    real_formal = len(tissues) >= target_tissues and min_reps >= target_reps
    rows.append(
        _row(
            "goal1_formalize_benchmarks",
            "Formalize key benchmarks",
            "real_background_replicates",
            "formal" if real_formal else ("representative" if len(tissues) >= target_tissues and min_reps >= 1 else "scaffold"),
            "complete" if real_formal else "needs_recompute",
            f"tissues={len(tissues)};min_replicates_per_tissue={min_reps}",
            f"tissues>={target_tissues};replicates_per_tissue>={target_reps}",
            real_path,
            "Real-background semi-synthetic benchmark covers lymph node, tonsil, and melanoma as current representative substrates.",
            "Current generation summary is one formal replicate per tissue, below the final Phase 4 replicate gate." if not real_formal else "",
            "Generate at least five real-background replicates per tissue or attach the existing 18-replicate tuned panel to a regenerated dataset manifest.",
        )
    )
    rows.append(
        _row(
            "goal1_formalize_benchmarks",
            "Formalize key benchmarks",
            "real_background_mechanism_coverage",
            "formal" if required_mechanisms.issubset(real_mechanisms) else "partial",
            "complete" if required_mechanisms.issubset(real_mechanisms) else "needs_design_fix",
            "|".join(sorted(real_mechanisms)),
            "|".join(sorted(required_mechanisms)),
            real_path,
            "Across real-background tissues, injected controls span the required mechanism classes.",
            "",
            "Keep mechanism coverage explicit in the formal source table.",
        )
    )
    return rows


def _goal2_baselines(sources: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = sources.get("phase2_revised") or {}
    path = PROJECT_ROOT / p2.get("replicated_baseline_metrics", "")
    claims_path = PROJECT_ROOT / p2.get("replicated_baseline_claims", "")
    rows = _read_csv(path)
    methods = {row.get("method", "") for row in rows}
    counts = {row.get("method", ""): int(float(row.get("aupr_count") or 0)) for row in rows}
    retained = set(((config.get("retained_methods") or {}).get("metric_methods") or []))
    excluded = set(((config.get("retained_methods") or {}).get("excluded_methods") or []))
    missing = sorted(retained - methods)
    excluded_present = sorted(excluded & methods)
    min_count = min([counts.get(method, 0) for method in retained if method in methods] or [0])
    formal = not missing and not excluded_present and min_count >= 15
    claim_rows = _read_csv(claims_path)
    mixed_claim = _find_row(claim_rows, "claim_id", "replicated_tuned_panel_tradeoff_not_global_dominance")
    return [
        _row(
            "goal2_strengthen_baselines",
            "Strengthen baseline comparison",
            "replicated_baseline_panel",
            "formal" if formal else "partial",
            "complete" if formal else "needs_recompute",
            f"methods={','.join(sorted(methods))};min_count={min_count};excluded_present={','.join(excluded_present)}",
            f"methods={','.join(sorted(retained))};min_count>=15;excluded_absent={','.join(sorted(excluded))}",
            path,
            "Replicated tuned baseline panel is available with common metrics across retained methods.",
            "CellNEST is a tuned-small run rather than a full seed sweep; method winners are metric- and tissue-specific.",
            "Use this table for formal comparison, then decide whether CellNEST needs a deeper epoch/seed sweep.",
        ),
        _row(
            "goal2_strengthen_baselines",
            "Strengthen baseline comparison",
            "claim_guardrail",
            "formal" if mixed_claim and _truthy(mixed_claim.get("supported")) else "partial",
            "complete" if mixed_claim and _truthy(mixed_claim.get("supported")) else "needs_review",
            mixed_claim.get("summary", "") if mixed_claim else "",
            "no_global_dominance_claim",
            claims_path,
            "Baseline result supports a tradeoff interpretation, not global dominance.",
            "",
            "Keep the no-global-dominance wording in Results and figure captions.",
        ),
    ]


def _goal3_fdr(sources: dict[str, Any], gates: dict[str, Any]) -> list[dict[str, Any]]:
    p3 = sources.get("phase3") or {}
    p2 = sources.get("phase2_revised") or {}
    fdr_paths = [
        PROJECT_ROOT / p3.get("fdr_index", ""),
        PROJECT_ROOT / p2.get("real_background_fdr_100perm", ""),
        PROJECT_ROOT / p2.get("fdr_200perm_smoke", ""),
        PROJECT_ROOT / p2.get("fdr_500perm_key_panel", ""),
    ]
    all_rows: list[dict[str, str]] = []
    for path in fdr_paths:
        for row in _read_csv(path):
            row = dict(row)
            row["_source"] = _rel(path)
            all_rows.append(row)
    nulls = set(gates.get("required_null_models") or [])
    observed_nulls = {row.get("null_model", "") for row in all_rows}
    max_perm = max([int(float(row.get("n_permutations") or 0)) for row in all_rows] or [0])
    formal_min = int(gates.get("formal_fdr_permutations_min", 200))
    final_target = int(gates.get("final_fdr_permutations_target", 500))
    formal_rows = [row for row in all_rows if int(float(row.get("n_permutations") or 0)) >= formal_min]
    final_rows = [row for row in all_rows if int(float(row.get("n_permutations") or 0)) >= final_target]
    group_path = PROJECT_ROOT / p2.get("fdr_500perm_key_group_summary", "")
    group_rows = _read_csv(group_path)
    max_group_perm = max([int(float(row.get("n_permutations") or 0)) for row in group_rows] or [0])
    n_sig_lr_groups = sum(
        int(float(row.get("n_significant_groups") or 0))
        for row in group_rows
        if row.get("group_level") == "lr_pair"
    )
    n_sig_ct_groups = sum(
        int(float(row.get("n_significant_groups") or 0))
        for row in group_rows
        if row.get("group_level") == "cell_type_pair"
    )
    return [
        _row(
            "goal3_upgrade_fdr",
            "Upgrade FDR",
            "fdr_permutation_depth",
            "final" if final_rows else ("formal" if formal_rows else ("representative" if max_perm >= 100 else "scaffold")),
            "complete" if final_rows else "needs_recompute",
            (
                f"max_permutations={max_perm};formal_rows={len(formal_rows)};final_rows={len(final_rows)};"
                f"max_group_permutations={max_group_perm};significant_lr_groups={n_sig_lr_groups};"
                f"significant_cell_type_pair_groups={n_sig_ct_groups}"
            ),
            f"formal>={formal_min};final>={final_target}",
            ";".join(_rel(path) for path in [*fdr_paths, group_path] if path.exists()),
            "FDR module is wired across the three null models and now includes a 500-permutation key lymph-node panel with group-level LR and cell-type-pair outputs.",
            "The 500-permutation final panel is currently one key lymph-node replicate; edge-level discoveries remain absent, so main wording should emphasize group-level FDR.",
            "Extend 500-permutation FDR to additional key tissues or keep this as the designated final key panel with explicit edge-power limitations.",
        ),
        _row(
            "goal3_upgrade_fdr",
            "Upgrade FDR",
            "fdr_null_models",
            "formal" if nulls.issubset(observed_nulls) else "partial",
            "complete" if nulls.issubset(observed_nulls) else "needs_recompute",
            "|".join(sorted(observed_nulls)),
            "|".join(sorted(nulls)),
            ";".join(_rel(path) for path in fdr_paths if path.exists()),
            "All required null models are represented in current FDR outputs.",
            "",
            "Keep all three null models in every formal and final FDR run.",
        ),
    ]


def _goal4_weak_signal(sources: dict[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / ((sources.get("phase2_revised") or {}).get("weak_signal_tradeoff_claims", ""))
    rows = _read_csv(path)
    tradeoff = _find_row(rows, "claim_id", "G2_C1_weak_fp_tradeoff")
    raw = _find_row(rows, "claim_id", "G2_C2_raw_weak_recall")
    fp = _find_row(rows, "claim_id", "G2_C3_false_positive_suppression")
    return [
        _row(
            "goal4_weak_signal_evidence",
            "Weak-signal evidence",
            "weak_signal_tradeoff",
            "representative" if tradeoff and tradeoff.get("status") == "supported" else "partial",
            "complete" if tradeoff and tradeoff.get("status") == "supported" else "needs_recompute",
            tradeoff.get("status", "") if tradeoff else "",
            "supported",
            path,
            tradeoff.get("allowed_wording", "") if tradeoff else "",
            "This supports a tradeoff claim, not highest raw weak-signal recall.",
            "Promote the weak-signal panel to the formal benchmark replicate set and report weak recall beside FDR and FP rate.",
        ),
        _row(
            "goal4_weak_signal_evidence",
            "Weak-signal evidence",
            "raw_weak_recall_guardrail",
            "formal" if raw and raw.get("status") == "unsupported" else "partial",
            "complete" if raw and raw.get("status") == "unsupported" else "needs_review",
            raw.get("status", "") if raw else "",
            "unsupported_global_raw_recall_claim",
            path,
            raw.get("allowed_wording", "") if raw else "",
            raw.get("disallowed_wording", "") if raw else "",
            "Keep Results phrasing as weak-signal/false-positive tradeoff.",
        ),
        _row(
            "goal4_weak_signal_evidence",
            "Weak-signal evidence",
            "false_positive_guardrail",
            "representative" if fp and fp.get("status") in {"supported", "partial"} else "partial",
            "needs_review" if fp and fp.get("status") == "partial" else ("complete" if fp else "missing"),
            fp.get("status", "") if fp else "",
            "supported_or_explicitly_partial",
            path,
            fp.get("allowed_wording", "") if fp else "",
            fp.get("disallowed_wording", "") if fp else "",
            "Report SpaTalk as a conservative low-FP baseline where it is lower than SpatialSelf.",
        ),
    ]


def _goal5_real_validation(sources: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = sources.get("phase2_revised") or {}
    bio_path = PROJECT_ROOT / p2.get("biological_validation_claims", "")
    downstream_path = PROJECT_ROOT / p2.get("downstream_claims", "")
    p4_downstream_summary_path = PROJECT_ROOT / p2.get("phase4_downstream_summary", "")
    p4_downstream_claims_path = PROJECT_ROOT / p2.get("phase4_downstream_claims", "")
    p4_downstream_results_path = PROJECT_ROOT / p2.get("phase4_downstream_results", "")
    bio_rows = _read_csv(bio_path)
    downstream_rows = _read_csv(downstream_path)
    p4_downstream_summary = _first(_read_csv(p4_downstream_summary_path))
    p4_downstream_claims = _read_csv(p4_downstream_claims_path)
    control = _find_row(bio_rows, "claim_id", "G1_C1_control_ranking")
    fp = _find_row(bio_rows, "claim_id", "G1_C2_high_expression_fp_suppression")
    downstream_limited = bool(downstream_rows) and all(_truthy(row.get("coverage_limited")) for row in downstream_rows)
    p4_downstream_supported = (
        bool(p4_downstream_summary)
        and not _truthy(p4_downstream_summary.get("coverage_limited"))
        and _float(p4_downstream_summary.get("positive_negative_spearman_delta")) > 0
    )
    p4_downstream_claim_status = {
        row.get("claim_id"): row.get("status", "")
        for row in p4_downstream_claims
    }
    return [
        _row(
            "goal5_real_validation",
            "Real-sample validation",
            "positive_negative_controls",
            "formal" if control and control.get("status") == "supported" else "partial",
            "complete" if control and control.get("status") == "supported" else "needs_recompute",
            control.get("status", "") if control else "",
            "supported",
            bio_path,
            control.get("allowed_wording", "") if control else "",
            "Injected-control validation is not unlabeled real-data ground truth.",
            "Use global control-ranking statistics before selected biological examples.",
        ),
        _row(
            "goal5_real_validation",
            "Real-sample validation",
            "high_expression_fp_suppression",
            "formal" if fp and fp.get("status") == "supported" else "partial",
            "complete" if fp and fp.get("status") == "supported" else "needs_recompute",
            fp.get("status", "") if fp else "",
            "supported",
            bio_path,
            fp.get("allowed_wording", "") if fp else "",
            fp.get("disallowed_wording", "") if fp else "",
            "Keep SpaTalk conservative-baseline caveat in manuscript wording.",
        ),
        _row(
            "goal5_real_validation",
            "Real-sample validation",
            "downstream_validation",
            "formal" if p4_downstream_supported else ("scaffold" if downstream_limited else "partial"),
            "complete" if p4_downstream_supported else ("needs_recompute" if downstream_limited else "needs_review"),
            (
                f"phase4_key_coverage_limited={p4_downstream_summary.get('coverage_limited', '')};"
                f"positive_tested={p4_downstream_summary.get('n_positive_tested', '')};"
                f"negative_tested={p4_downstream_summary.get('n_negative_tested', '')};"
                f"delta={p4_downstream_summary.get('positive_negative_spearman_delta', '')};"
                f"legacy_coverage_limited={downstream_limited};legacy_rows={len(downstream_rows)}"
            ),
            "key_lymph_node_not_global_downstream_superiority",
            ";".join(_rel(path) for path in [p4_downstream_summary_path, p4_downstream_claims_path, p4_downstream_results_path, downstream_path] if path.exists()),
            "Phase 4 key lymph-node downstream panel tests all injected positive and high-expression negative LR controls and positive controls have higher mean receiver target correlation.",
            (
                "This closes the key lymph-node downstream gate but not global cross-tissue downstream validation; "
                f"cross_tissue_claim={p4_downstream_claim_status.get('P4_D3_cross_tissue_downstream', '')}."
            ),
            "For a stronger manuscript claim, regenerate tonsil and melanoma benchmark expression tables with expanded required target genes.",
        ),
    ]


def _goal6_ablation(sources: dict[str, Any]) -> list[dict[str, Any]]:
    p4 = sources.get("phase4") or {}
    summary_path = PROJECT_ROOT / p4.get("formal_ablation_summary", "")
    metrics_path = PROJECT_ROOT / p4.get("formal_ablation_metrics", "")
    status_path = PROJECT_ROOT / p4.get("formal_ablation_status", "")
    fallback_candidates = [
        PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv",
        PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/spatialself_ablation_summary.csv",
        PROJECT_ROOT / "outputs/phase2/goal6_paper_figures/source_tables/goal6_component_ablation.csv",
    ]
    existing = [path for path in [summary_path, metrics_path, status_path] if path.exists()]
    required = {
        "full_model",
        "no_lr_identity",
        "no_graph_wavelet",
        "no_distance",
        "no_expression",
        "mlp_instead_of_graph",
        "gat_wavelet_comparison",
        "pseudo_label_strategy",
    }
    summary_rows = _read_csv(summary_path)
    status_rows = _read_csv(status_path)
    completed_ablations = {
        row.get("ablation", "")
        for row in status_rows
        if row.get("status") == "completed"
    }
    ablation_names = {row.get("ablation", "") for row in summary_rows} | completed_ablations
    category_map = {
        "full_model": {"wavelet_mechanism_niche_edge_prior"},
        "no_lr_identity": {"wavelet_no_lr_embedding"},
        "no_graph_wavelet": {"gat_hard_expression_distance"},
        "no_distance": {"wavelet_no_distance"},
        "no_expression": {"wavelet_no_expression_features"},
        "mlp_instead_of_graph": {"mlp_hard_expression_distance"},
        "gat_wavelet_comparison": {"wavelet_hard_expression_distance", "gat_hard_expression_distance"},
        "pseudo_label_strategy": {
            "wavelet_hard_expression_distance",
            "wavelet_expression_only",
            "wavelet_distance_only",
            "wavelet_soft_expression_distance",
        },
    }
    present = {
        category
        for category, needed_ablations in category_map.items()
        if needed_ablations.issubset(ablation_names)
    }
    replicate_counts = [
        int(float(row.get("auroc_count") or 0))
        for row in summary_rows
        if row.get("ablation") in ablation_names
    ]
    min_replicates = min(replicate_counts) if replicate_counts else 0
    full = _find_row(summary_rows, "ablation", "wavelet_mechanism_niche_edge_prior")
    no_lr = _find_row(summary_rows, "ablation", "wavelet_no_lr_embedding")
    full_aupr = _float(full.get("aupr_mean")) if full else float("nan")
    no_lr_aupr = _float(no_lr.get("aupr_mean")) if no_lr else float("nan")
    formal = required.issubset(present) and min_replicates >= 5
    if not formal and not existing:
        old_existing = [path for path in fallback_candidates if path.exists()]
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in old_existing)
        present = {name for name in required if _ablation_present(name, text)}
        existing = old_existing
    return [
        _row(
            "goal6_ablation_lock",
            "Lock ablations",
            "required_ablation_coverage",
            "formal" if formal else ("scaffold" if existing else "missing"),
            "complete" if formal else "needs_recompute",
            (
                f"present={','.join(sorted(present))};min_replicates={min_replicates};"
                f"full_aupr_mean={full_aupr:.6g};no_lr_aupr_mean={no_lr_aupr:.6g}"
            ),
            ",".join(sorted(required)),
            ";".join(_rel(path) for path in existing),
            "Phase 4 controlled 5-replicate ablation covers LR identity, spatial graph/wavelet comparison, distance, expression, MLP/GAT alternatives, and pseudo-label strategies.",
            "This is a 10-epoch formal coverage panel; manuscript-final effect sizes may need a longer epoch or seed sweep.",
            "If this panel enters the main text, rerun the same registered ablations with the final training budget and paired statistics.",
        )
    ]


def _goal7_figures(sources: dict[str, Any]) -> list[dict[str, Any]]:
    p3 = sources.get("phase3") or {}
    p2 = sources.get("phase2_revised") or {}
    p4 = sources.get("phase4") or {}
    p4_qa_path = PROJECT_ROOT / p4.get(
        "figure_render_qa",
        "outputs/phase4/manuscript_evidence_lock/phase4_figure_render_qa.csv",
    )
    p4_source_path = PROJECT_ROOT / p4.get(
        "figure_source_index",
        "outputs/phase4/manuscript_evidence_lock/phase4_figure_source_index.csv",
    )
    p4_summary_path = PROJECT_ROOT / p4.get(
        "figure_render_summary",
        "outputs/phase4/manuscript_evidence_lock/phase4_figure_render_summary.json",
    )
    fig_path = PROJECT_ROOT / p3.get("figure_index", "")
    source_path = PROJECT_ROOT / p3.get("manuscript_source_index", "")
    runtime_path = PROJECT_ROOT / p2.get("runtime_claims", "")
    p4_qa_rows = _read_csv(p4_qa_path)
    p4_source_rows = _read_csv(p4_source_path)
    all_exports_exist = bool(p4_qa_rows) and all(
        (PROJECT_ROOT / row.get("pdf", "")).exists()
        and (PROJECT_ROOT / row.get("svg", "")).exists()
        and (PROJECT_ROOT / row.get("tiff", "")).exists()
        and row.get("status") == "rendered"
        for row in p4_qa_rows
    )
    source_claims_complete = bool(p4_source_rows) and all(
        row.get("supported_claim", "").strip() and row.get("limitation", "").strip()
        for row in p4_source_rows
    )
    p4_complete = len(p4_qa_rows) >= 7 and len(p4_source_rows) >= 7 and all_exports_exist and source_claims_complete
    fig_rows = _read_csv(fig_path)
    source_rows = _read_csv(source_path)
    runtime = _first(_read_csv(runtime_path))
    return [
        _row(
            "goal7_manuscript_figures",
            "Manuscript figures",
            "figure_source_scaffold",
            "formal" if p4_complete else ("scaffold" if len(fig_rows) >= 8 and len(source_rows) >= 6 else "partial"),
            "complete" if p4_complete else "needs_polish",
            (
                f"phase4_figures={len(p4_qa_rows)};phase4_source_rows={len(p4_source_rows)};"
                f"all_exports_exist={all_exports_exist};source_claims_complete={source_claims_complete};"
                f"phase3_figures={len(fig_rows)};phase3_source_tables={len(source_rows)}"
            ),
            "figures>=7;source_tables_present;final_exports_qa",
            ";".join(_rel(path) for path in [p4_qa_path, p4_source_path, p4_summary_path, fig_path, source_path] if path.exists()),
            "Phase 4 provides source-linked evidence-lock Fig1-Fig8 exports with PDF, SVG, TIFF, supported claims, and explicit limitations.",
            "These are evidence-lock manuscript figures; final graphical polishing and journal-specific composition can still be refined before submission.",
            "Use these figures as the Phase 4 manuscript evidence backbone, then perform final visual polish in the writing stage.",
        ),
        _row(
            "goal7_manuscript_figures",
            "Manuscript figures",
            "runtime_scalability_claim",
            "representative" if runtime else "missing",
            "needs_review" if runtime else "missing",
            runtime.get("evidence_status", "") if runtime else "",
            "final_50k_or_100k_scaling_or_supplement_only",
            runtime_path,
            runtime.get("interpretation", "") if runtime else "",
            "Current runtime evidence reaches 1,800 cells/1.64M edges, not a final 50k/100k high-resolution benchmark.",
            "Keep Fig8 in supplement unless larger scalability evidence is generated.",
        ),
    ]


def _ablation_present(name: str, text: str) -> bool:
    probes = {
        "full_model": ["full"],
        "no_lr_identity": ["no_lr", "lr identity", "identity"],
        "no_graph_wavelet": ["wavelet"],
        "no_distance": ["distance"],
        "no_expression": ["expression"],
        "mlp_instead_of_graph": ["mlp"],
        "gat_wavelet_comparison": ["gat", "wavelet"],
        "pseudo_label_strategy": ["pseudo"],
    }
    return all(token in text for token in probes[name])


def _row(
    goal_id: str,
    goal_title: str,
    evidence_area: str,
    tier: str,
    status: str,
    current_value: str,
    target_value: str,
    evidence_path: Path | str,
    supported_claim: str,
    limitation: str,
    next_action: str,
) -> dict[str, Any]:
    evidence = _rel(evidence_path) if isinstance(evidence_path, Path) else str(evidence_path)
    return {
        "goal_id": goal_id,
        "goal_title": goal_title,
        "evidence_area": evidence_area,
        "tier": tier,
        "status": status,
        "current_value": current_value,
        "target_value": target_value,
        "evidence": evidence,
        "supported_claim": supported_claim,
        "limitation": limitation,
        "next_action": next_action,
    }


def _tier_for_count(value: int, formal_min: int, representative_min: int = 1) -> str:
    if value >= formal_min:
        return "formal"
    if value >= representative_min:
        return "representative"
    if value > 0:
        return "scaffold"
    return "missing"


def _find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _summary(rows: list[dict[str, Any]], output_path: Path, config_path: Path) -> dict[str, Any]:
    return {
        "config": _rel(config_path),
        "evidence_index_csv": _rel(output_path),
        "n_rows": len(rows),
        "status_counts": _counts(row["status"] for row in rows),
        "tier_counts": _counts(row["tier"] for row in rows),
        "goal_counts": _counts(row["goal_id"] for row in rows),
        "n_complete_rows": sum(1 for row in rows if row["status"] == "complete"),
        "n_rows_requiring_action": sum(1 for row in rows if row["status"] != "complete"),
        "claim_level": "phase4_evidence_lock_not_final_manuscript",
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 4 config loading.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
