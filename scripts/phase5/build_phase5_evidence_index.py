#!/usr/bin/env python
"""Build the Phase 5 submission-manuscript evidence index."""

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
    parser.add_argument("--config", default="configs/experiments/phase5_submission_manuscript_lock.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase5/manuscript_submission_lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase5_evidence_index.csv"))

    sources = config.get("source_artifacts") or {}
    rows: list[dict[str, Any]] = []
    rows.extend(_goal1_claim_boundary(sources))
    rows.extend(_goal2_real_validation(sources, config))
    rows.extend(_goal3_figures(sources, config))
    rows.extend(_goal4_results(sources))
    rows.extend(_goal5_methods(sources, config))
    rows.extend(_goal6_negative_controls(sources, config))

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path, config_path)
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 5 evidence index: {output_path}")
    print(json.dumps(summary, indent=2))


def _goal1_claim_boundary(sources: dict[str, Any]) -> list[dict[str, Any]]:
    p5 = sources.get("phase5") or {}
    p4 = sources.get("phase4") or {}
    claim_path = PROJECT_ROOT / p5.get("claim_boundary", "")
    phase4_index = PROJECT_ROOT / p4.get("evidence_index", "")
    text = _read_text(claim_path)
    required_phrases = [
        "Claims We Can Make",
        "Claims We Cannot Make",
        "SpatialSelf is the best CCC method",
        "edge-level",
        "global cross-tissue",
    ]
    complete = claim_path.exists() and phase4_index.exists() and all(phrase in text for phrase in required_phrases)
    return [
        _row(
            "goal1_claim_boundary",
            "Lock manuscript claim boundary",
            "claim_boundary_document",
            "formal" if complete else "missing",
            "complete" if complete else "needs_writeup",
            f"exists={claim_path.exists()};phase4_index={phase4_index.exists()};phrases={sum(phrase in text for phrase in required_phrases)}/{len(required_phrases)}",
            "CLAIM_BOUNDARY.md with allowed/disallowed claims tied to evidence index",
            f"{_rel(claim_path)};{_rel(phase4_index)}",
            "The manuscript has an explicit claim boundary separating supported wording from disallowed global claims.",
            "This controls wording but does not add new biological evidence.",
            "Use this file as a gate before polishing Results text.",
        )
    ]


def _goal2_real_validation(sources: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = sources.get("phase2_revised") or {}
    p5 = sources.get("phase5") or {}
    p5_tissue_path = PROJECT_ROOT / p5.get("real_validation_tissue_summary", "")
    p5_claims_path = PROJECT_ROOT / p5.get("real_validation_claims", "")
    p5_lr_path = PROJECT_ROOT / p5.get("real_validation_lr_panel", "")
    cross_scope_path = PROJECT_ROOT / p5.get("cross_tissue_validation_scope", "")
    cross_claims_path = PROJECT_ROOT / p5.get("cross_tissue_validation_claims", "")
    claims_path = PROJECT_ROOT / p2.get("full_baseline_matched_claims_all", "")
    method_path = PROJECT_ROOT / p2.get("full_baseline_matched_method_summary_all", "")
    lr_path = PROJECT_ROOT / p2.get("full_baseline_matched_lr_summary_all", "")
    bio_index_path = PROJECT_ROOT / p2.get("biological_validation_index", "")
    p5_tissue_rows = _read_csv(p5_tissue_path)
    p5_claim_rows = _read_csv(p5_claims_path)
    cross_scope_rows = _read_csv(cross_scope_path)
    cross_claim_rows = _read_csv(cross_claims_path)
    claim_rows = _read_csv(claims_path)
    method_rows = _read_csv(method_path)
    bio_rows = _read_csv(bio_index_path)
    required = set(config.get("required_real_validation_datasets") or [])
    dataset_alias = {
        "lymph_node_real_background_smoke": "lymph_node",
        "real_background_lymph_node": "lymph_node",
        "tonsil_real_background_smoke": "tonsil",
        "real_background_tonsil": "tonsil",
        "high_resolution_tonsil_subset": "tonsil",
        "melanoma_real_background_smoke": "melanoma",
        "real_background_melanoma": "melanoma",
    }
    claim_datasets = {dataset_alias.get(row.get("dataset", ""), row.get("dataset", "")) for row in claim_rows}
    method_datasets = {dataset_alias.get(row.get("dataset", ""), row.get("dataset", "")) for row in method_rows}
    bio_datasets = {dataset_alias.get(row.get("dataset_id", ""), row.get("dataset_id", "")) for row in bio_rows}
    covered = required & (claim_datasets | method_datasets | bio_datasets)
    spatialself_rows = [row for row in method_rows if row.get("method") == "spatialself_wavelet_edge_prior"]
    pass_count = sum(1 for row in spatialself_rows if _truthy(row.get("rank_validation_pass")))
    has_baselines = len({row.get("method", "") for row in method_rows}) >= 5
    p5_claim_status = {row.get("claim_id", ""): row.get("status", "") for row in p5_claim_rows}
    p5_supported = (
        p5_claim_status.get("P5_RV1_three_tissue_control_ranking") == "supported"
        and p5_claim_status.get("P5_RV2_high_expression_fp_suppression") == "supported"
        and p5_claim_status.get("P5_RV4_baseline_matched_scope") == "supported"
    )
    cross_claim_status = {row.get("claim_id", ""): row.get("status", "") for row in cross_claim_rows}
    cross_scope_supported = (
        len(cross_scope_rows) >= 3
        and cross_claim_status.get("P5_XT1_three_tissue_validation_scope") == "supported"
        and cross_claim_status.get("P5_XT2_downstream_scope_guardrail") == "supported"
        and cross_claim_status.get("P5_XT3_group_fdr_scope") == "supported"
    )
    complete = required.issubset(covered) and pass_count >= 3 and has_baselines and p5_supported
    tier = "formal" if complete else ("representative" if covered else "missing")
    return [
        _row(
            "goal2_real_biological_validation",
            "Strengthen real biological validation",
            "three_tissue_control_ranking",
            tier,
            "complete" if complete else "needs_validation",
            (
                f"covered={','.join(sorted(covered))};spatialself_rank_pass_count={pass_count};"
                f"baseline_methods={len({row.get('method', '') for row in method_rows})};"
                f"phase5_claims_supported={p5_supported};phase5_tissue_rows={len(p5_tissue_rows)}"
            ),
            "lymph_node,tonsil,melanoma with positive/negative controls and baseline comparison",
            ";".join(_rel(path) for path in [p5_tissue_path, p5_claims_path, p5_lr_path, claims_path, method_path, lr_path, bio_index_path] if path.exists()),
            "SpatialSelf has baseline-matched positive/negative control ranking evidence across lymph node, tonsil, and melanoma real-background settings.",
            "Current evidence includes semi-synthetic real-background and prior-control validation; native real-tissue ground truth is absent.",
            "Promote this into main Fig7/Fig8 only with clear semi-synthetic/prior-control wording and add native-real region/niche enrichment where available.",
        ),
        _row(
            "goal2_real_biological_validation",
            "Strengthen real biological validation",
            "downstream_cross_tissue_scope",
            "formal" if cross_scope_supported else "representative",
            "complete" if cross_scope_supported else "needs_review",
            (
                f"key_lymph_node_complete;tonsil_melanoma_proxy_guarded;"
                f"cross_scope_supported={cross_scope_supported};scope_rows={len(cross_scope_rows)}"
            ),
            "lymph node plus strengthened tonsil/melanoma downstream or niche validation",
            ";".join(
                _rel(path)
                for path in [
                    PROJECT_ROOT / (sources.get("phase4") or {}).get("downstream_key_lymph_node_summary", ""),
                    cross_scope_path,
                    cross_claims_path,
                ]
                if path.exists()
            ),
            "Cross-tissue validation scope is locked: lymph node has key curated downstream validation, while tonsil and melanoma are guarded as proxy/niche-control validation with group-level FDR support.",
            "This is not global cross-tissue curated downstream superiority and not native complete ground truth.",
            "Keep the guarded wording in Results and Methods; add native-real shuffled/random controls only if pursuing a stronger claim.",
        ),
    ]


def _goal3_figures(sources: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    p5 = sources.get("phase5") or {}
    p4 = sources.get("phase4") or {}
    plan_path = PROJECT_ROOT / p5.get("figure_plan", "")
    p5_qa_path = PROJECT_ROOT / p5.get("real_validation_figure_qa", "")
    p5_source_path = PROJECT_ROOT / p5.get("real_validation_figure_source_index", "")
    p5_summary_path = PROJECT_ROOT / p5.get("real_validation_figure_summary", "")
    package_index_path = PROJECT_ROOT / p5.get("main_figure_package_index", "")
    package_source_path = PROJECT_ROOT / p5.get("main_figure_source_data_index", "")
    panel_source_path = PROJECT_ROOT / p5.get("main_figure_panel_source_index", "")
    package_summary_path = PROJECT_ROOT / p5.get("submission_package_summary", "")
    source_path = PROJECT_ROOT / p4.get("figure_source_index", "")
    qa_path = PROJECT_ROOT / p4.get("figure_render_qa", "")
    plan_text = _read_text(plan_path)
    required = config.get("required_main_figures") or []
    mentioned = _count_required_figures(plan_text, required)
    source_rows = _read_csv(source_path)
    qa_rows = _read_csv(qa_path)
    p5_qa_rows = _read_csv(p5_qa_path)
    p5_source_rows = _read_csv(p5_source_path)
    package_rows = _read_csv(package_index_path)
    package_source_rows = _read_csv(package_source_path)
    panel_source_rows = _read_csv(panel_source_path)
    package_summary = _read_json(package_summary_path)
    evidence_lock_complete = len(source_rows) >= 8 and len(qa_rows) >= 8
    phase5_figures_complete = _figure_exports_complete(p5_qa_rows) and _figure_source_complete(p5_source_rows) and {row.get("figure_id") for row in p5_qa_rows} >= {"Fig7", "Fig8"}
    plan_complete = plan_path.exists() and "Main Text Figures" in plan_text and "Supplementary Figures" in plan_text
    package_complete = (
        len(package_rows) == 8
        and len(package_source_rows) >= 20
        and len(panel_source_rows) >= 24
        and all(row.get("status") == "complete" for row in package_rows)
        and all(row.get("status") == "complete" for row in panel_source_rows)
        and package_summary.get("package_complete") is True
    )
    status = "complete" if plan_complete and evidence_lock_complete and phase5_figures_complete and package_complete else "needs_polish"
    tier = "formal" if status == "complete" else ("representative" if phase5_figures_complete else "scaffold")
    return [
        _row(
            "goal3_main_figure_package",
            "Convert evidence-lock figures into main figures",
            "main_figure_plan",
            tier,
            status,
            (
                f"plan={plan_path.exists()};phase4_source_rows={len(source_rows)};phase4_qa_rows={len(qa_rows)};"
                f"phase5_real_validation_figures={len(p5_qa_rows)};phase5_sources={len(p5_source_rows)};"
                f"phase5_figures_complete={phase5_figures_complete};package_complete={package_complete};"
                f"package_figures={len(package_rows)};source_tables={len(package_source_rows)};"
                f"panel_source_rows={len(panel_source_rows)};"
                f"required_mentions={mentioned}/{len(required)}"
            ),
            "8 main figures assigned and packaged with PDF/SVG/TIFF exports, source-data index, supported claims, and limitations",
            ";".join(_rel(path) for path in [plan_path, source_path, qa_path, p5_qa_path, p5_source_path, p5_summary_path, package_index_path, package_source_path, panel_source_path, package_summary_path] if path.exists()),
            "Phase 5 has a source-linked main-figure package covering Fig1-Fig8 and replacing evidence-lock-only figure tracking.",
            "Fig1-Fig6 are packaged from Phase 4 evidence-lock figures and may still need final visual polish, but their source-data and claim mapping are locked.",
            "Use the packaged figure index as the manuscript source-data map.",
        )
    ]


def _figure_exports_complete(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        row.get("status") == "rendered"
        and (PROJECT_ROOT / row.get("pdf", "")).exists()
        and (PROJECT_ROOT / row.get("svg", "")).exists()
        and (PROJECT_ROOT / row.get("tiff", "")).exists()
        for row in rows
    )


def _figure_source_complete(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        row.get("supported_claim", "").strip()
        and row.get("limitation", "").strip()
        and row.get("source_tables", "").strip()
        for row in rows
    )


def _goal4_results(sources: dict[str, Any]) -> list[dict[str, Any]]:
    p5 = sources.get("phase5") or {}
    path = PROJECT_ROOT / p5.get("results_draft", "")
    manuscript_path = PROJECT_ROOT / p5.get("results_manuscript_draft", "")
    qa_path = PROJECT_ROOT / p5.get("results_methods_qa", "")
    text = _read_text(path)
    manuscript_text = _read_text(manuscript_path)
    qa_rows = _read_csv(qa_path)
    sections = [
        "SpatialSelf Models CCC",
        "Realistic Simulations",
        "Weak-Signal / False-Positive",
        "Ablation Studies",
        "Permutation Testing",
        "Lymph Node",
        "High-Resolution Tonsil and Melanoma",
    ]
    count = sum(section in text for section in sections)
    has_claims = text.count("Supported claim:") >= 7 and text.count("Limitation:") >= 7
    manuscript_count = sum(section in manuscript_text for section in sections)
    figure_mentions = sum(1 for idx in range(1, 9) if f"Fig. {idx}" in manuscript_text or f"Figure {idx}" in manuscript_text)
    manuscript_ready = (
        manuscript_path.exists()
        and manuscript_count == len(sections)
        and manuscript_text.count("Supported claim:") >= 7
        and manuscript_text.count("Limitation:") >= 7
        and figure_mentions >= 8
        and "global" in manuscript_text
        and "ground truth" in manuscript_text
    )
    qa_ready = _qa_complete(qa_rows, "results")
    complete = path.exists() and count == len(sections) and has_claims
    tier = "formal" if manuscript_ready and qa_ready else ("representative" if manuscript_ready else ("scaffold" if complete else "missing"))
    status = "complete" if manuscript_ready and qa_ready else ("needs_polish" if manuscript_ready or complete else "needs_writeup")
    return [
        _row(
            "goal4_results_draft",
            "Write Results first draft",
            "results_skeleton",
            tier,
            status,
            (
                f"skeleton_sections={count}/{len(sections)};manuscript_sections={manuscript_count}/{len(sections)};"
                f"figure_mentions={figure_mentions};supported_claims={manuscript_text.count('Supported claim:')};"
                f"limitations={manuscript_text.count('Limitation:')};qa_ready={qa_ready}"
            ),
            "7 Results sections with figure-linked claims, source evidence, statistics, and limitations",
            ";".join(_rel(p) for p in [path, manuscript_path, qa_path] if p.exists()),
            "A conservative Results manuscript draft exists with figure-linked claims, exact Phase 5 validation statistics, source-data callouts, and explicit limitations.",
            "This is a first manuscript draft; final language editing and citation formatting remain outside the evidence lock.",
            "Proceed to final prose editing without expanding claims beyond CLAIM_BOUNDARY.md.",
        )
    ]


def _goal5_methods(sources: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    p5 = sources.get("phase5") or {}
    path = PROJECT_ROOT / p5.get("methods_reproducibility", "")
    manuscript_path = PROJECT_ROOT / p5.get("methods_manuscript_draft", "")
    lr_manifest_path = PROJECT_ROOT / p5.get("lr_database_manifest", "")
    baseline_status_path = PROJECT_ROOT / p5.get("baseline_status_table", "")
    methods_claims_path = PROJECT_ROOT / p5.get("methods_support_claims", "")
    qa_path = PROJECT_ROOT / p5.get("results_methods_qa", "")
    text = _read_text(path)
    manuscript_text = _read_text(manuscript_path)
    lr_rows = _read_csv(lr_manifest_path)
    baseline_rows = _read_csv(baseline_status_path)
    methods_claims = _read_csv(methods_claims_path)
    qa_rows = _read_csv(qa_path)
    topics = [
        "Simulation and Benchmark Construction",
        "LR Database and Complex Handling",
        "Candidate Graph and Range Handling",
        "Self-Supervised Pseudo-Labels",
        "Permutation and FDR",
        "Baseline Fairness",
        "Real-Sample Validation",
        "Reproducibility Entry Points",
    ]
    count = sum(topic in text for topic in topics)
    normalized_text = _normalize_token(text)
    has_excluded = all(_normalize_token(name) in normalized_text for name in config.get("excluded_baselines") or [])
    manuscript_topics = [
        "Controlled CCC Simulator",
        "Real-Background Semi-Synthetic Benchmark",
        "LR Database and Candidate Universe",
        "Spatial Graph, KNN, and Graph-Wavelet Range Handling",
        "Self-Supervised Pseudo-Labels",
        "Permutation-Based FDR",
        "Baseline Comparison",
        "Real-Sample and Real-Background Validation",
        "Reproducibility Commands",
    ]
    manuscript_count = sum(topic in manuscript_text for topic in manuscript_topics)
    normalized_manuscript = _normalize_token(manuscript_text)
    manuscript_ready = (
        manuscript_path.exists()
        and manuscript_count == len(manuscript_topics)
        and all(_normalize_token(name) in normalized_manuscript for name in config.get("excluded_baselines") or [])
        and "DeepTalk" in manuscript_text
        and "NICHES" in manuscript_text
        and "K_NN" in manuscript_text
        and "K_pos" in manuscript_text
        and "K_neg" in manuscript_text
        and "500 permutations" in manuscript_text
        and "pair-level LR controls without multi-subunit complex modeling" in manuscript_text
    )
    claim_status = {row.get("claim_id", ""): row.get("status", "") for row in methods_claims}
    methods_support_ready = (
        lr_manifest_path.exists()
        and baseline_status_path.exists()
        and methods_claims_path.exists()
        and len(lr_rows) >= 3
        and len(baseline_rows) >= 9
        and claim_status.get("P5_M1_lr_universe_locked_for_benchmarks") == "supported"
        and claim_status.get("P5_M2_baseline_status_locked") == "supported"
        and claim_status.get("P5_M3_reproducibility_support") == "supported"
    )
    qa_ready = _qa_complete(qa_rows, "methods")
    complete = path.exists() and count == len(topics) and has_excluded
    tier = "formal" if manuscript_ready and methods_support_ready and qa_ready else ("representative" if manuscript_ready else ("scaffold" if complete else "missing"))
    status = "complete" if manuscript_ready and methods_support_ready and qa_ready else ("needs_polish" if manuscript_ready or complete else "needs_writeup")
    return [
        _row(
            "goal5_methods_reproducibility",
            "Organize Methods and reproducibility",
            "methods_checklist",
            tier,
            status,
            (
                f"checklist_topics={count}/{len(topics)};manuscript_topics={manuscript_count}/{len(manuscript_topics)};"
                f"excluded_baselines_present={has_excluded};manuscript_ready={manuscript_ready};"
                f"methods_support_ready={methods_support_ready};qa_ready={qa_ready};lr_rows={len(lr_rows)};baseline_rows={len(baseline_rows)}"
            ),
            "Methods draft covers simulation, real-background injection, LR database, graph/KNN/wavelet, pseudo-labels, FDR, baselines, validation, and reproducibility commands",
            ";".join(_rel(p) for p in [path, manuscript_path, lr_manifest_path, baseline_status_path, methods_claims_path, qa_path] if p.exists()),
            "A Methods manuscript draft exists with Phase 5 LR-universe, baseline-status, pair-level LR-control scope, and reproducibility support tables.",
            "The benchmark LR universe is pair-level and does not model multi-subunit complexes; this limits claims about external LR database completeness.",
            "Keep the pair-level complex-handling limitation visible in final Methods.",
        )
    ]


def _count_required_figures(text: str, required: list[str]) -> int:
    lowered = text.lower()
    count = 0
    for idx, _item in enumerate(required, start=1):
        probes = [f"fig. {idx}", f"fig {idx}", f"figure {idx}"]
        if any(probe in lowered for probe in probes):
            count += 1
    return count


def _goal6_negative_controls(sources: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    p2 = sources.get("phase2_revised") or {}
    p5 = sources.get("phase5") or {}
    p5_tissue_path = PROJECT_ROOT / p5.get("real_validation_tissue_summary", "")
    p5_claims_path = PROJECT_ROOT / p5.get("real_validation_claims", "")
    p5_lr_path = PROJECT_ROOT / p5.get("real_validation_lr_panel", "")
    claims_path = PROJECT_ROOT / p2.get("full_baseline_matched_claims_all", "")
    method_path = PROJECT_ROOT / p2.get("full_baseline_matched_method_summary_all", "")
    lr_path = PROJECT_ROOT / p2.get("full_baseline_matched_lr_summary_all", "")
    p5_claims = _read_csv(p5_claims_path)
    claims = _read_csv(claims_path)
    methods = _read_csv(method_path)
    lr_rows = _read_csv(lr_path)
    required = set(config.get("required_real_validation_datasets") or [])
    alias = {
        "lymph_node_real_background_smoke": "lymph_node",
        "tonsil_real_background_smoke": "tonsil",
        "melanoma_real_background_smoke": "melanoma",
    }
    datasets = {alias.get(row.get("dataset", ""), row.get("dataset", "")) for row in claims}
    spatialself = [row for row in methods if row.get("method") == "spatialself_wavelet_edge_prior"]
    all_rank_pass = sum(1 for row in spatialself if _truthy(row.get("rank_validation_pass")))
    low_fp_count = sum(1 for row in claims if _truthy(row.get("spatialself_lowest_fp")))
    positive_lr = sum(1 for row in lr_rows if row.get("method") == "spatialself_wavelet_edge_prior" and _truthy(row.get("positive_control")))
    negative_lr = sum(1 for row in lr_rows if row.get("method") == "spatialself_wavelet_edge_prior" and _truthy(row.get("negative_control")))
    claim_status = {row.get("claim_id", ""): row.get("status", "") for row in p5_claims}
    phase5_supported = (
        claim_status.get("P5_RV1_three_tissue_control_ranking") == "supported"
        and claim_status.get("P5_RV2_high_expression_fp_suppression") == "supported"
        and claim_status.get("P5_RV3_receiver_response_proxy") == "supported"
        and claim_status.get("P5_RV4_baseline_matched_scope") == "supported"
    )
    complete = required.issubset(datasets) and all_rank_pass >= 3 and low_fp_count >= 3 and positive_lr >= 9 and negative_lr >= 6 and phase5_supported
    return [
        _row(
            "goal6_real_negative_controls",
            "Add real-data negative-control false-positive suppression",
            "positive_negative_control_ranking",
            "formal" if complete else ("representative" if datasets else "missing"),
            "complete" if complete else "needs_validation",
            (
                f"datasets={','.join(sorted(datasets))};rank_pass={all_rank_pass};lowest_fp={low_fp_count};"
                f"positive_lr={positive_lr};negative_lr={negative_lr};phase5_supported={phase5_supported}"
            ),
            "lymph_node/tonsil/melanoma positive-vs-negative controls, baseline comparison, enrichment/rank/FDR outputs",
            ";".join(_rel(path) for path in [p5_tissue_path, p5_claims_path, p5_lr_path, claims_path, method_path, lr_path] if path.exists()),
            "Current real-background control panels rank injected positives above high-expression negatives across the three target tissues and compare retained baselines.",
            "These are real-background semi-synthetic controls, not native unlabeled real-data ground truth; additional shuffled/native controls would strengthen the claim.",
            "Add random LR, shuffled LR database, spatially shuffled coordinates, and non-permissive cell-type controls for native real datasets.",
        )
    ]


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
    return {
        "goal_id": goal_id,
        "goal_title": goal_title,
        "evidence_area": evidence_area,
        "tier": tier,
        "status": status,
        "current_value": current_value,
        "target_value": target_value,
        "evidence": _rel(evidence_path) if isinstance(evidence_path, Path) else str(evidence_path),
        "supported_claim": supported_claim,
        "limitation": limitation,
        "next_action": next_action,
    }


def _summary(rows: list[dict[str, Any]], output_path: Path, config_path: Path) -> dict[str, Any]:
    n_rows_requiring_action = sum(1 for row in rows if row["status"] != "complete")
    return {
        "config": _rel(config_path),
        "evidence_index_csv": _rel(output_path),
        "n_rows": len(rows),
        "status_counts": _counts(row["status"] for row in rows),
        "tier_counts": _counts(row["tier"] for row in rows),
        "goal_counts": _counts(row["goal_id"] for row in rows),
        "n_complete_rows": sum(1 for row in rows if row["status"] == "complete"),
        "n_rows_requiring_action": n_rows_requiring_action,
        "claim_level": "phase5_submission_manuscript_lock_complete"
        if n_rows_requiring_action == 0
        else "phase5_submission_manuscript_lock_in_progress",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _qa_complete(rows: list[dict[str, str]], artifact: str) -> bool:
    selected = [row for row in rows if row.get("artifact") == artifact]
    return bool(selected) and all(row.get("status") == "complete" for row in selected)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 5 index building.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _normalize_token(value: Any) -> str:
    return str(value).lower().replace("-", "_")


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
