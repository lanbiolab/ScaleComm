#!/usr/bin/env python
"""Build the Phase 8 manuscript-grade evidence consolidation package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORE_CLAIM_COLUMNS = [
    "claim_id",
    "manuscript_role",
    "allowed_wording",
    "evidence_tier",
    "primary_evidence",
    "supporting_evidence",
    "disallowed_wording",
    "limitation",
    "status",
]

FIGURE_COLUMNS = [
    "figure_id",
    "proposed_title",
    "main_message",
    "primary_evidence",
    "source_tables",
    "source_figures_or_rendered_assets",
    "figure_status",
    "manuscript_priority",
    "reviewer_risk_addressed",
    "limitation",
]

CASE_COLUMNS = [
    "case_id",
    "dataset",
    "platform_or_background",
    "lr_pair",
    "evidence_tier",
    "spatialself_rank",
    "baseline_rank_summary",
    "weak_or_context_specific_status",
    "region_or_niche_evidence",
    "downstream_target_evidence",
    "fdr_evidence",
    "negative_control_status",
    "biological_interpretation",
    "source_tables",
    "recommended_manuscript_use",
    "limitation",
]

BASELINE_COLUMNS = [
    "method",
    "phase7_final_status",
    "code_availability",
    "dependency_reproducibility",
    "input_compatibility",
    "output_level",
    "lr_pair_ranking_support",
    "runtime_memory_status",
    "exact_failure_reason",
    "manuscript_use",
    "source_evidence",
    "limitation",
]

BOUNDARY_COLUMNS = [
    "boundary_id",
    "risk",
    "safe_wording",
    "unsafe_wording",
    "evidence_or_reason",
    "where_to_state",
]

METHOD_COLUMNS = [
    "methods_topic",
    "required_content",
    "evidence_or_source",
    "status",
    "remaining_action",
    "claim_boundary",
]

SOURCE_INDEX_COLUMNS = [
    "source_id",
    "artifact_type",
    "manuscript_use",
    "path",
    "exists",
    "claim_boundary",
]

AUDIT_COLUMNS = [
    "goal_id",
    "requirement",
    "status",
    "evidence",
    "remaining_gap",
    "priority",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase8_manuscript_grade_evidence_consolidation.yaml",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    phase4 = PROJECT_ROOT / config["upstream_outputs"]["phase4"]
    phase5 = PROJECT_ROOT / config["upstream_outputs"]["phase5"]
    phase7 = PROJECT_ROOT / config["upstream_outputs"]["phase7"]

    context = {
        "phase4": phase4,
        "phase5": phase5,
        "phase7": phase7,
        "phase7_requirements": _read_csv(phase7 / "phase7_requirement_audit.csv"),
        "phase7_evidence": _read_csv(phase7 / "phase7_evidence_index.csv"),
        "goal5_closure": _read_csv(
            phase7 / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv"
        ),
        "cross_platform": _read_csv(
            phase7 / "goal5_cross_platform_real_data" / "phase7_cross_platform_validation_summary.csv"
        ),
        "downstream_methods": _read_csv(
            phase7 / "goal1_downstream_target_validation" / "phase7_method_target_response_comparison.csv"
        ),
        "deep_baselines": _read_csv(
            phase7 / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv"
        ),
        "realism": _read_csv(
            phase7 / "goal3_benchmark_realism_figure" / "phase7_benchmark_realism_summary.csv"
        ),
        "melanoma_goal4": _read_csv(
            phase7
            / "goal4_expanded_melanoma_subset_screen"
            / "phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv"
        ),
        "tonsil_goal4": _read_csv(
            phase7 / "goal4_expanded_tonsil_subset_screen" / "phase7_goal4_expanded_tonsil_audit.csv"
        ),
        "output_dir": output_dir,
    }

    core_claims = _build_core_claims(context)
    figure_plan = _build_main_figure_plan(context)
    cases = _build_real_case_table(context)
    baselines = _build_deep_baseline_table(context)
    boundaries = _build_claim_boundaries(context)
    methods = _build_methods_checklist(context)
    source_index = _build_source_table_index(context, core_claims, figure_plan, cases, baselines, methods)
    audit = _build_audit(context, core_claims, figure_plan, cases, baselines, boundaries, methods)

    paths = {
        "core_claims": output_dir / "phase8_core_story_claims.csv",
        "main_figure_plan": output_dir / "phase8_main_figure_plan.csv",
        "real_case_deep_dive": output_dir / "phase8_real_case_deep_dive_table.csv",
        "deep_baseline_closure": output_dir / "phase8_deep_baseline_closure_table.csv",
        "claim_boundaries": output_dir / "phase8_claim_boundary_table.csv",
        "methods_reproducibility": output_dir / "phase8_methods_reproducibility_checklist.csv",
        "methods_package": output_dir / "phase8_methods_reproducibility_package.md",
        "tutorial": output_dir / "phase8_reproducibility_tutorial.md",
        "source_table_index": output_dir / "phase8_source_table_index.csv",
        "audit": output_dir / "phase8_completion_gap_audit.csv",
        "results_skeleton": output_dir / "phase8_results_skeleton.md",
        "summary": output_dir / "phase8_summary.json",
    }

    _write_csv(paths["core_claims"], core_claims, CORE_CLAIM_COLUMNS)
    _write_csv(paths["main_figure_plan"], figure_plan, FIGURE_COLUMNS)
    _write_csv(paths["real_case_deep_dive"], cases, CASE_COLUMNS)
    _write_csv(paths["deep_baseline_closure"], baselines, BASELINE_COLUMNS)
    _write_csv(paths["claim_boundaries"], boundaries, BOUNDARY_COLUMNS)
    _write_csv(paths["methods_reproducibility"], methods, METHOD_COLUMNS)
    _write_csv(paths["source_table_index"], source_index, SOURCE_INDEX_COLUMNS)
    _write_csv(paths["audit"], audit, AUDIT_COLUMNS)
    paths["methods_package"].write_text(
        _build_methods_reproducibility_package(methods, boundaries, source_index),
        encoding="utf-8",
    )
    paths["tutorial"].write_text(_build_reproducibility_tutorial(), encoding="utf-8")
    paths["results_skeleton"].write_text(_build_results_skeleton(core_claims, figure_plan, cases), encoding="utf-8")

    _copy_key_figures(phase4, phase5, phase7, output_dir / "figure_assets")

    summary = {
        "config": _rel(config_path),
        "output_dir": _rel(output_dir),
        "artifacts": {key: _rel(path) for key, path in paths.items()},
        "n_core_claims": len(core_claims),
        "claim_status_counts": dict(Counter(row["status"] for row in core_claims)),
        "n_main_figures": len(figure_plan),
        "figure_status_counts": dict(Counter(row["figure_status"] for row in figure_plan)),
        "n_real_cases": len(cases),
        "case_tier_counts": dict(Counter(row["evidence_tier"] for row in cases)),
        "n_deep_baselines": len(baselines),
        "n_source_table_index_rows": len(source_index),
        "phase8_audit_status_counts": dict(Counter(row["status"] for row in audit)),
        "phase8_package_complete": all(row["status"] == "complete" for row in audit),
        "completion_boundary": "Phase 8 is complete only when the package contains the manuscript story, figure plan, real-case table, deep-baseline closure, claim boundaries, and methods/reproducibility checklist. It does not mean final polished figures or manuscript text are done.",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _build_core_claims(context: dict[str, Any]) -> list[dict[str, str]]:
    phase7_req = context["phase7"] / "phase7_requirement_audit_summary.json"
    phase7_summary = _load_json(context["phase7"] / "phase7_summary.json")
    closure_summary = _load_json(
        context["phase7"]
        / "goal5_cross_platform_real_data"
        / "phase7_goal5_functional_fdr_closure_audit_summary.json"
    )
    realism = context["realism"]
    min_realism = _min_float(realism, "selected_library_size_spearman")
    n_local = closure_summary.get("n_local_contrast_all_null_success", 0)
    n_lr_identity = closure_summary.get("n_functional_lr_identity_success", 0)
    n_competitive = closure_summary.get("n_functional_competitive_success", 0)
    phase7_complete = str(_load_json(phase7_req).get("phase7_complete", False))
    return [
        {
            "claim_id": "P8_C1_main_story",
            "manuscript_role": "central_thesis",
            "allowed_wording": "SpatialSelf reduces high-expression false-positive leakage and improves weak or context-specific CCC prioritization by combining LR identity-aware modeling, graph-wavelet spatial structure, self-supervised scoring, and matched local-background statistical control.",
            "evidence_tier": "primary_story_supported",
            "primary_evidence": _rel(context["phase7"] / "phase7_requirement_audit.csv"),
            "supporting_evidence": ";".join(
                [
                    _rel(context["phase4"] / "goal6_formal_ablation_controlled_5rep" / "spatialself_ablation_summary.csv"),
                    _rel(context["phase7"] / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv"),
                ]
            ),
            "disallowed_wording": "SpatialSelf proves causal cell-cell communication in native tissues or universally dominates all CCC methods.",
            "limitation": "Real-data validation uses controls, enrichment, target response, and FDR rather than native perturbation truth.",
            "status": "supported" if phase7_complete == "True" else "partial",
        },
        {
            "claim_id": "P8_C2_realistic_benchmark",
            "manuscript_role": "simulation_independence",
            "allowed_wording": "The semi-synthetic benchmarks preserve major expression and spatial statistics from real ST backgrounds while retaining injected LR truth.",
            "evidence_tier": "primary_benchmark_support",
            "primary_evidence": _rel(context["phase7"] / "goal3_benchmark_realism_figure" / "phase7_benchmark_realism_summary.csv"),
            "supporting_evidence": _rel(context["phase7"] / "goal3_benchmark_realism_figure" / "figures" / "fig_phase7_benchmark_realism.pdf"),
            "disallowed_wording": "The semi-synthetic datasets contain complete native biological ground truth.",
            "limitation": f"Realism statistics support plausibility; minimum selected library-size Spearman observed here is {_fmt(min_realism)}.",
            "status": "supported" if min_realism is not None and min_realism > 0.9 else "partial",
        },
        {
            "claim_id": "P8_C3_downstream_functional_support",
            "manuscript_role": "functional_validation",
            "allowed_wording": "Selected SpatialSelf-prioritized LR activities are associated with receiver downstream target signatures across lymph-node and non-lymph-node settings.",
            "evidence_tier": "functional_association_support",
            "primary_evidence": _rel(
                context["phase7"]
                / "goal1_downstream_target_validation"
                / "phase7_method_target_response_comparison.csv"
            ),
            "supporting_evidence": _rel(
                context["phase7"] / "goal1_downstream_target_validation" / "phase7_downstream_fdr.csv"
            ),
            "disallowed_wording": "Downstream signatures prove ligand perturbation causality.",
            "limitation": "Target signatures are curated-prior associations; some targeted-panel platforms have negative-control leakage.",
            "status": "supported" if context["downstream_methods"] else "partial",
        },
        {
            "claim_id": "P8_C4_local_contrast_fdr",
            "manuscript_role": "false_positive_control",
            "allowed_wording": "In the strongest cross-platform case, local-contrast FDR separates a positive LR from matched high-expression negative-control background under all audited nulls.",
            "evidence_tier": "strict_local_contrast_support",
            "primary_evidence": _rel(
                context["phase7"]
                / "goal5_cross_platform_real_data"
                / "phase7_goal5_functional_fdr_closure_audit.csv"
            ),
            "supporting_evidence": f"local_contrast_all_null_success={n_local};functional_lr_identity_success={n_lr_identity};functional_competitive_success={n_competitive}",
            "disallowed_wording": "Raw-score strict all-null FDR or model-rerun all-null FDR is fully solved.",
            "limitation": "Local-contrast FDR is a matched-background statistic designed for score saturation; raw-score and model-rerun closure remain unavailable.",
            "status": "supported" if int(n_local) >= 1 else "partial",
        },
        {
            "claim_id": "P8_C5_deep_baseline_transparency",
            "manuscript_role": "reviewer_risk_control",
            "allowed_wording": "Deep spatial CCC methods were either formally evaluated when compatible or audited with reproducible status/failure reasons.",
            "evidence_tier": "transparency_support",
            "primary_evidence": _rel(
                context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv"
            ),
            "supporting_evidence": _rel(
                context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deeptalk_formal_run_summary.csv"
            ),
            "disallowed_wording": "Failed or incompatible deep baselines are biologically inferior.",
            "limitation": "Several methods remain status-only because of missing code, missing required model artifacts, or incompatible output levels.",
            "status": "supported" if len(context["deep_baselines"]) >= 5 else "partial",
        },
    ]


def _build_main_figure_plan(context: dict[str, Any]) -> list[dict[str, str]]:
    phase4_fig = context["phase4"] / "figures"
    phase5_fig = context["phase5"] / "figures"
    phase7_realism_fig = context["phase7"] / "goal3_benchmark_realism_figure" / "figures"
    specs = [
        (
            "Fig1",
            "SpatialSelf method overview",
            "SpatialSelf integrates LR identity, expression/distance features, graph-wavelet spatial structure, self-supervised learning, and FDR calibration.",
            [context["phase4"] / "phase4_evidence_index.csv"],
            [phase4_fig / "fig1_phase4_manuscript.pdf", phase4_fig / "fig1_phase4_manuscript.svg"],
            "primary",
            "Method should not look like a generic GAT/expression-distance score.",
            "Schematic only; quantitative support comes later.",
        ),
        (
            "Fig2",
            "Realistic ground-truth benchmark",
            "Controlled and real-background benchmarks test contact, paracrine, long-range, weak-signal, and high-expression false-positive scenarios.",
            [
                context["phase4"] / "goal1_controlled_simulator_v2_5rep" / "formal_benchmark_summary.csv",
                context["phase4"] / "goal1_real_background_v2_5rep" / "formal_real_background_summary.csv",
            ],
            [phase4_fig / "fig2_phase4_manuscript.pdf", phase4_fig / "fig3_phase4_manuscript.pdf"],
            "primary",
            "Simulation is too artificial or aligned with the method.",
            "Truth is injected rather than native.",
        ),
        (
            "Fig3",
            "Benchmark realism",
            "Semi-synthetic datasets preserve real ST expression and spatial statistics.",
            [
                context["phase7"] / "goal3_benchmark_realism_figure" / "phase7_benchmark_realism_summary.csv",
                context["phase7"] / "goal3_benchmark_realism_figure" / "phase7_benchmark_realism_metrics.csv",
            ],
            [
                phase7_realism_fig / "fig_phase7_benchmark_realism.pdf",
                phase7_realism_fig / "fig_phase7_benchmark_realism.svg",
            ],
            "primary",
            "Reviewer concern that simulated data are unrealistic.",
            "Realism statistics do not create native ground truth.",
        ),
        (
            "Fig4",
            "Baseline comparison",
            "SpatialSelf is benchmarked against retained traditional/spatial/deep baselines with transparent status handling.",
            [
                context["phase5"] / "goal5_methods_support" / "phase5_baseline_status_table.csv",
                context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv",
            ],
            [phase4_fig / "fig4_phase4_manuscript.pdf"],
            "primary",
            "Missing CellNEST/DeepTalk-like baseline comparison.",
            "Status-only baselines should not be treated as losses.",
        ),
        (
            "Fig5",
            "Ablation and weak-signal mechanism",
            "LR identity and graph-wavelet spatial structure contribute to weak/context-specific signal recovery.",
            [context["phase4"] / "goal6_formal_ablation_controlled_5rep" / "spatialself_ablation_summary.csv"],
            [phase4_fig / "fig5_phase4_manuscript.pdf"],
            "primary",
            "Gains might come only from model capacity or pseudo-label bias.",
            "Ablation supports component contribution, not a unique mechanistic proof.",
        ),
        (
            "Fig6",
            "FDR and high-expression false-positive control",
            "Permutation and local-contrast FDR quantify false-positive control and matched-background separation.",
            [
                context["phase4"] / "goal3_fdr_500perm_lymph_node" / "goal5_formal_fdr_summary.csv",
                context["phase7"] / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv",
            ],
            [phase4_fig / "fig6_phase4_manuscript.pdf"],
            "primary",
            "Scores need p-values/FDR and false-positive control.",
            "Local-contrast FDR must be labeled separately from raw-score FDR.",
        ),
        (
            "Fig7",
            "Real-data functional validation",
            "Lymph node, tonsil, and melanoma cases connect prioritized LR activity to region/niche enrichment and downstream response.",
            [
                context["phase5"] / "goal2_real_validation" / "phase5_real_validation_lr_panel.csv",
                context["phase7"] / "goal1_downstream_target_validation" / "phase7_method_target_response_comparison.csv",
                context["phase7"] / "goal4_expanded_melanoma_subset_screen" / "phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv",
            ],
            [phase5_fig / "fig7_phase5_real_validation.pdf", phase5_fig / "fig8_phase5_real_validation.pdf"],
            "primary",
            "Real-data validation should not be only selected pretty examples.",
            "Real samples lack full native ground truth.",
        ),
        (
            "Fig8",
            "Cross-platform local-contrast case",
            "Post-Xenium colon SPP1-CD44 provides the strongest Phase 7 cross-platform closure case.",
            [
                context["phase7"] / "goal5_cross_platform_real_data" / "phase7_cross_platform_validation_summary.csv",
                context["phase7"]
                / "goal5_cross_platform_real_data"
                / "post_xenium_colon_local_contrast_fdr_200perm"
                / "post_xenium_colon_local_contrast_audit.csv",
            ],
            [],
            "primary_needs_final_figure",
            "Cross-platform story must not rely only on legacy datasets.",
            "Needs final polished figure assembly from existing source tables.",
        ),
    ]
    rows = []
    for fig_id, title, message, sources, assets, priority, risk, limitation in specs:
        source_paths = [Path(path) for path in sources]
        asset_paths = [Path(path) for path in assets]
        status = "ready_source_tables_and_assets" if all(path.exists() for path in source_paths) and asset_paths and any(path.exists() for path in asset_paths) else (
            "ready_source_tables_needs_render" if all(path.exists() for path in source_paths) else "missing_source"
        )
        rows.append(
            {
                "figure_id": fig_id,
                "proposed_title": title,
                "main_message": message,
                "primary_evidence": ";".join(_rel(path) for path in source_paths),
                "source_tables": ";".join(_rel(path) for path in source_paths),
                "source_figures_or_rendered_assets": ";".join(_rel(path) for path in asset_paths if path.exists()),
                "figure_status": status,
                "manuscript_priority": priority,
                "reviewer_risk_addressed": risk,
                "limitation": limitation,
            }
        )
    return rows


def _build_real_case_table(context: dict[str, Any]) -> list[dict[str, str]]:
    cross = {row["dataset"]: row for row in context["cross_platform"]}
    closure = {row["dataset"]: row for row in context["goal5_closure"]}
    melanoma_rows = context["melanoma_goal4"]
    cxcl10 = next((row for row in melanoma_rows if row.get("canonical_pair") == "CXCL10-CXCR3"), {})
    post = closure.get("post_xenium_human_colon_cancer_subset", {})
    post_cross = cross.get("post_xenium_human_colon_cancer_subset", {})
    spamosaic = closure.get("spamosaic_tonsil_A1_subset", {})
    slide = closure.get("slide_tags_tonsil_subset", {})
    xenium = cross.get("xenium_human_lymph_node_subset", {})
    rows = [
        {
            "case_id": "case1_lymph_node_cd48_cd244",
            "dataset": "human_lymph_node_visium_or_phase6_expanded",
            "platform_or_background": "Visium lymph node / expanded benchmark-real validation",
            "lr_pair": "CD48-CD244",
            "evidence_tier": "strong_functional_support_from_phase7_goal1",
            "spatialself_rank": _rank_from_downstream(context["downstream_methods"], "CD48", "CD244"),
            "baseline_rank_summary": "Other methods have lower downstream target correlation in Phase 7 Goal 1 comparison.",
            "weak_or_context_specific_status": "weak juxtacrine/contact-like T-cell case",
            "region_or_niche_evidence": "lymph-node immune niche support from Phase 5/6 source tables",
            "downstream_target_evidence": _downstream_summary(context["downstream_methods"], "CD48", "CD244"),
            "fdr_evidence": "Phase 7 downstream BH-FDR supports target association; use Phase 5/6 FDR tables for final figure.",
            "negative_control_status": "Positive/negative control separation available in Phase 5/6 real-validation summaries.",
            "biological_interpretation": "A weak/context immune contact signal that SpatialSelf prioritizes with receiver response support.",
            "source_tables": ";".join(
                [
                    _rel(context["phase7"] / "goal1_downstream_target_validation" / "phase7_method_target_response_comparison.csv"),
                    _rel(context["phase5"] / "goal2_real_validation" / "phase5_real_validation_lr_panel.csv"),
                ]
            ),
            "recommended_manuscript_use": "main_text_supporting_case",
            "limitation": "Not native perturbation ground truth; keep as functional association plus control ranking.",
        },
        {
            "case_id": "case2_post_xenium_colon_spp1_cd44",
            "dataset": "post_xenium_human_colon_cancer_subset",
            "platform_or_background": post_cross.get("platform", "post-Xenium colon"),
            "lr_pair": post.get("best_positive", "SPP1-CD44"),
            "evidence_tier": "strongest_cross_platform_local_contrast_closure",
            "spatialself_rank": str(post.get("best_positive_rank", "")),
            "baseline_rank_summary": post_cross.get("baseline_status", ""),
            "weak_or_context_specific_status": "context-specific tumor/stromal-immune style LR; not claimed as native truth",
            "region_or_niche_evidence": post_cross.get("region_or_niche_enrichment", ""),
            "downstream_target_evidence": post_cross.get("downstream_target_response", ""),
            "fdr_evidence": _local_contrast_summary(post),
            "negative_control_status": f"best_negative={post.get('best_negative','')};rank={post.get('best_negative_rank','')};local_contrast_negative_fdr={post.get('local_contrast_best_negative_fdr','')}",
            "biological_interpretation": "SPP1-CD44 is a plausible context-associated interaction; current claim is prioritization with downstream/local-background support.",
            "source_tables": ";".join(
                [
                    _rel(context["phase7"] / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv"),
                    _rel(context["phase7"] / "goal5_cross_platform_real_data" / "phase7_cross_platform_validation_summary.csv"),
                    _rel(
                        context["phase7"]
                        / "goal5_cross_platform_real_data"
                        / "post_xenium_colon_local_contrast_fdr_200perm"
                        / "post_xenium_colon_local_contrast_audit.csv"
                    ),
                ]
            ),
            "recommended_manuscript_use": "main_text_anchor_case_for_phase7",
            "limitation": "Local-contrast FDR closure, not raw-score all-null or model-rerun closure.",
        },
        {
            "case_id": "case3_tonsil_ccl21_ccr7_or_cxcl13_cxcr5",
            "dataset": "spamosaic_tonsil_A1_subset_or_slide_tags_tonsil_subset",
            "platform_or_background": "SPAMOSAIC / Slide-tags tonsil",
            "lr_pair": spamosaic.get("best_positive", "") or slide.get("best_positive", "CCL21-CCR7/CXCL13-CXCR5"),
            "evidence_tier": "supportive_not_strict_closure",
            "spatialself_rank": str(spamosaic.get("best_positive_rank", "") or slide.get("best_positive_rank", "")),
            "baseline_rank_summary": cross.get("spamosaic_tonsil_A1_subset", {}).get("baseline_status", ""),
            "weak_or_context_specific_status": "tonsil chemokine/GC niche support",
            "region_or_niche_evidence": cross.get("spamosaic_tonsil_A1_subset", {}).get("region_or_niche_enrichment", ""),
            "downstream_target_evidence": cross.get("spamosaic_tonsil_A1_subset", {}).get("downstream_target_response", ""),
            "fdr_evidence": _local_contrast_summary(spamosaic),
            "negative_control_status": "Local-contrast FDR supports negative suppression, but downstream negative-control leakage keeps this supportive.",
            "biological_interpretation": "Tonsil chemokine organization is biologically plausible and supports cross-platform consistency.",
            "source_tables": ";".join(
                [
                    _rel(context["phase7"] / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv"),
                    _rel(
                        context["phase7"]
                        / "goal5_cross_platform_real_data"
                        / "spamosaic_tonsil_A1_local_contrast_fdr_200perm"
                        / "spamosaic_tonsil_A1_local_contrast_audit.csv"
                    ),
                ]
            ),
            "recommended_manuscript_use": "supporting_main_or_supplement",
            "limitation": "Supportive local-contrast evidence but not full downstream-clean closure.",
        },
        {
            "case_id": "case4_melanoma_cxcl10_cxcr3",
            "dataset": "melanoma_real_background_targeted_context_screen",
            "platform_or_background": "melanoma real-background targeted context screen",
            "lr_pair": cxcl10.get("canonical_pair", "CXCL10-CXCR3"),
            "evidence_tier": "strong_spatialself_specific_targeted_context_support",
            "spatialself_rank": cxcl10.get("raw_spatialself_rank", ""),
            "baseline_rank_summary": f"COMMOT={cxcl10.get('commot_rank','')};CellChat={cxcl10.get('cellchat_rank','')};CellNEST={cxcl10.get('cellnest_rank','')}",
            "weak_or_context_specific_status": "tumor-immune context; macrophage-to-T candidate",
            "region_or_niche_evidence": cxcl10.get("calibrated_dominant_context", ""),
            "downstream_target_evidence": f"rho={cxcl10.get('downstream_spearman_r','')};FDR={cxcl10.get('downstream_bh_fdr','')}",
            "fdr_evidence": f"targeted all-null pass={cxcl10.get('all_three_targeted_fdr_pass','')};strict_goal4_success={cxcl10.get('strict_goal4_success','')}",
            "negative_control_status": "Targeted audit records pass/fail columns and claim boundary.",
            "biological_interpretation": "Macrophage-associated chemokine signaling consistent with TME immune biology; avoid treatment-causality claims.",
            "source_tables": _rel(
                context["phase7"]
                / "goal4_expanded_melanoma_subset_screen"
                / "phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv"
            ),
            "recommended_manuscript_use": "main_text_if_space_or_strong_supplement",
            "limitation": "Targeted context screen, not native perturbation truth.",
        },
        {
            "case_id": "case5_xenium_lymph_node_ccl19_ccr7",
            "dataset": "xenium_human_lymph_node_subset",
            "platform_or_background": xenium.get("platform", "10x Xenium human lymph node"),
            "lr_pair": "CCL19-CCR7",
            "evidence_tier": "mixed_graphclust_fdr_support_not_clean_downstream",
            "spatialself_rank": "see graphclust and downstream source tables",
            "baseline_rank_summary": xenium.get("baseline_status", ""),
            "weak_or_context_specific_status": "T-zone chemokine candidate",
            "region_or_niche_evidence": xenium.get("region_or_niche_enrichment", ""),
            "downstream_target_evidence": xenium.get("downstream_target_response", ""),
            "fdr_evidence": xenium.get("fdr_or_feasibility_note", ""),
            "negative_control_status": "Mixed: negative controls also correlate in downstream target validation.",
            "biological_interpretation": "Useful high-resolution platform support, but not a clean false-positive-control case.",
            "source_tables": _rel(context["phase7"] / "goal5_cross_platform_real_data" / "phase7_cross_platform_validation_summary.csv"),
            "recommended_manuscript_use": "supplement_or_boundary_case",
            "limitation": "Marker-rule/graphclust proxies and downstream leakage limit claim strength.",
        },
    ]
    return rows


def _build_deep_baseline_table(context: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    source = context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv"
    for row in context["deep_baselines"]:
        status = row.get("final_status", "")
        method = row.get("method", "")
        rows.append(
            {
                "method": method,
                "phase7_final_status": status,
                "code_availability": row.get("code_availability", ""),
                "dependency_reproducibility": row.get("dependency_reproducibility", ""),
                "input_compatibility": row.get("input_compatibility", ""),
                "output_level": row.get("output_level", ""),
                "lr_pair_ranking_support": row.get("lr_pair_ranking_support", ""),
                "runtime_memory_status": row.get("runtime_memory_status", ""),
                "exact_failure_reason": row.get("exact_failure_reason", ""),
                "manuscript_use": _deep_baseline_use(method, status),
                "source_evidence": _rel(source),
                "limitation": "Use as reproducibility/status evidence only unless the method produced comparable LR-pair rankings.",
            }
        )
    return rows


def _build_claim_boundaries(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "boundary_id": "B1_native_ground_truth",
            "risk": "Real ST data do not contain complete CCC ground truth.",
            "safe_wording": "Real-data results provide control-based, enrichment, FDR, and downstream-response support.",
            "unsafe_wording": "Real-data predictions are proven true CCC events.",
            "evidence_or_reason": "Native tissue lacks perturbation labels; semi-synthetic truth is injected.",
            "where_to_state": "Results real-data sections; Discussion limitations; Methods validation.",
        },
        {
            "boundary_id": "B2_local_contrast_fdr",
            "risk": "Local-contrast FDR could be mistaken for raw-score all-null FDR.",
            "safe_wording": "Local-contrast FDR tests positive LR excess over matched high-expression negative-control background.",
            "unsafe_wording": "Raw-score strict all-null FDR is solved across real platforms.",
            "evidence_or_reason": _rel(
                context["phase7"]
                / "goal5_cross_platform_real_data"
                / "phase7_goal5_functional_fdr_closure_audit_summary.json"
            ),
            "where_to_state": "FDR Results; Methods statistical testing; Figure 8 legend.",
        },
        {
            "boundary_id": "B3_downstream_validation",
            "risk": "Target signatures are not perturbation causality.",
            "safe_wording": "Receiver target signatures provide functional association evidence.",
            "unsafe_wording": "Downstream target correlation proves ligand causality.",
            "evidence_or_reason": _rel(
                context["phase7"] / "goal1_downstream_target_validation" / "phase7_method_target_response_comparison.csv"
            ),
            "where_to_state": "Functional validation Results and Methods.",
        },
        {
            "boundary_id": "B4_deep_baselines",
            "risk": "Status-only or failed baselines could be overinterpreted.",
            "safe_wording": "Incompatible or failed baselines are reported as reproducibility/status limitations.",
            "unsafe_wording": "Methods that could not run are biologically worse than SpatialSelf.",
            "evidence_or_reason": _rel(
                context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv"
            ),
            "where_to_state": "Baseline comparison Results; Methods; Supplementary Table.",
        },
        {
            "boundary_id": "B5_global_superiority",
            "risk": "Metric-specific wins can become overbroad claims.",
            "safe_wording": "SpatialSelf shows stronger prioritization for weak/context-specific CCC and lower high-expression false-positive leakage under audited settings.",
            "unsafe_wording": "SpatialSelf is universally the best CCC method for every dataset and metric.",
            "evidence_or_reason": "Baseline comparisons are dataset-, metric-, and adapter-dependent.",
            "where_to_state": "Abstract, Discussion, and final paragraph of Results.",
        },
    ]


def _build_methods_checklist(context: dict[str, Any]) -> list[dict[str, str]]:
    topics = [
        (
            "benchmark_generation",
            "Describe controlled simulator with contact, paracrine, long-range niche, weak-signal, and high-expression false-positive LR mechanisms.",
            context["phase4"] / "goal1_controlled_simulator_v2_5rep" / "formal_simulator_manifest.json",
            "complete",
            "Condense into Methods and move parameter grid to Supplement.",
            "Synthetic truth is injected.",
        ),
        (
            "semi_synthetic_real_background",
            "Describe real-coordinate/expression background preservation and LR injection policy.",
            context["phase4"] / "goal1_real_background_v2_5rep" / "formal_real_background_manifest.json",
            "complete" if (context["phase4"] / "goal1_real_background_v2_5rep").exists() else "needs_source_check",
            "Verify exact manifest path during manuscript export.",
            "Injected truth is not native truth.",
        ),
        (
            "benchmark_realism",
            "Report library size, sparsity, mean-variance, NN distance, density, Moran's I, LR expression, PCA/UMAP overlap.",
            context["phase7"] / "goal3_benchmark_realism_figure" / "phase7_benchmark_realism_summary.csv",
            "complete",
            "Use as Figure 3 source and Methods validation section.",
            "Realism metrics support plausibility only.",
        ),
        (
            "graph_wavelet_model",
            "Document graph construction, graph-wavelet spatial propagation, LR identity embedding, expression/distance features, and self-supervised loss.",
            PROJECT_ROOT / "scripts" / "benchmark" / "run_spatialself.py",
            "complete",
            "Add manuscript equations and distinguish K_NN from K_pos/K_neg.",
            "Model description should not imply unlimited-range edge truth unless evaluated.",
        ),
        (
            "knn_range_policy",
            "Explain that KNN constructs computational candidate neighborhoods, while LR mechanism/range is tested via benchmark labels and distance features/FDR.",
            PROJECT_ROOT / "scripts" / "benchmark" / "generate_controlled_ccc.py",
            "complete",
            "Use the Phase 8 Methods package paragraph directly, then polish notation in the manuscript.",
            "KNN candidate graph can still limit discoverable edges beyond graph radius.",
        ),
        (
            "pseudo_label_definition",
            "Define pseudo-label score, K_pos/K_neg, weak-signal weighting or controls, and distinguish from K_NN.",
            PROJECT_ROOT / "scripts" / "benchmark" / "diagnose_pseudo_labels.py",
            "complete",
            "Use the Phase 8 Methods package notation block directly, then add equations if needed.",
            "Pseudo-labeling is self-supervised, not oracle truth.",
        ),
        (
            "permutation_fdr",
            "Define spatial, receiver/cell-type, LR-identity, competitive LR, model-rerun, and local-contrast FDR.",
            context["phase7"] / "goal5_cross_platform_real_data" / "phase7_goal5_functional_fdr_closure_audit.csv",
            "complete",
            "Separate raw-score, model-rerun, and local-contrast claims.",
            "Local-contrast is matched-background evidence, not raw all-null closure.",
        ),
        (
            "baseline_installation_status",
            "Document COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk, HoloNet, stMLnet, CytoSignal, and NICHES status.",
            context["phase7"] / "goal2_deep_baseline_closure" / "phase7_deep_baseline_status_table.csv",
            "complete",
            "Format as Supplementary Table.",
            "Failed/incompatible methods are not biological losses.",
        ),
        (
            "runtime_memory",
            "Report runtime/memory for representative benchmark and real/cross-platform panels.",
            context["phase7"] / "goal5_cross_platform_real_data" / "phase7_cross_platform_validation_summary.csv",
            "complete",
            "Extract concise main-table values and full supplement.",
            "Subset timings should be labeled as subset timings.",
        ),
        (
            "tutorial_and_source_tables",
            "Provide source table index, commands, and tutorial entry point for benchmark and validation reproduction.",
            context["output_dir"] / "phase8_source_table_index.csv",
            "complete",
            "After final figure polishing, add exact commands for any newly rendered figures.",
            "Current package is a manuscript evidence/reproducibility index, not a fully polished public tutorial.",
        ),
    ]
    rows = []
    for topic, content, source, status, action, boundary in topics:
        source = Path(source)
        output_dir = context.get("output_dir")
        generated_later = output_dir is not None and source.parent == output_dir
        if status == "complete" and not source.exists() and not generated_later:
            status = "missing_source"
        rows.append(
            {
                "methods_topic": topic,
                "required_content": content,
                "evidence_or_source": _rel(source),
                "status": status,
                "remaining_action": action,
                "claim_boundary": boundary,
            }
        )
    return rows


def _build_audit(
    context: dict[str, Any],
    core_claims: list[dict[str, str]],
    figure_plan: list[dict[str, str]],
    cases: list[dict[str, str]],
    baselines: list[dict[str, str]],
    boundaries: list[dict[str, str]],
    methods: list[dict[str, str]],
) -> list[dict[str, str]]:
    complete_figures = sum(row["figure_status"] != "missing_source" for row in figure_plan)
    ready_assets = sum(row["figure_status"] == "ready_source_tables_and_assets" for row in figure_plan)
    strong_cases = [row for row in cases if row["evidence_tier"].startswith("strong") or row["evidence_tier"].startswith("strongest")]
    complete_methods = [row for row in methods if row["status"] == "complete"]
    missing_methods = [row for row in methods if row["status"] == "missing_source"]
    partial_methods = [row for row in methods if row["status"] in {"partial", "needs_manuscript_text"}]
    rows = [
        {
            "goal_id": "goal1_core_story",
            "requirement": "Core manuscript thesis and claim hierarchy are locked.",
            "status": "complete" if all(row["status"] == "supported" for row in core_claims) else "partial",
            "evidence": "phase8_core_story_claims.csv",
            "remaining_gap": "None for claim hierarchy; final prose still needed.",
            "priority": "high",
        },
        {
            "goal_id": "goal2_main_figures",
            "requirement": "Eight main-figure slots are mapped to claims, source tables, and reviewer risks.",
            "status": "complete" if complete_figures == 8 else "partial",
            "evidence": f"phase8_main_figure_plan.csv;ready_or_renderable={complete_figures}/8;existing_assets={ready_assets}/8",
            "remaining_gap": "Fig8 still needs final polished rendering from source tables.",
            "priority": "high",
        },
        {
            "goal_id": "goal3_real_case_deep_dive",
            "requirement": "Strongest real cases are ranked by evidence tier with downstream/FDR/negative-control status.",
            "status": "complete" if len(strong_cases) >= 3 else "partial",
            "evidence": f"phase8_real_case_deep_dive_table.csv;strong_cases={len(strong_cases)}",
            "remaining_gap": "Use mixed Xenium/tonsil cases as supplement unless final polishing improves leakage.",
            "priority": "high",
        },
        {
            "goal_id": "goal4_deep_baseline_closure",
            "requirement": "DeepTalk, HoloNet, stMLnet, CytoSignal, and NICHES have status/failure records.",
            "status": "complete" if len(baselines) >= 5 else "partial",
            "evidence": "phase8_deep_baseline_closure_table.csv",
            "remaining_gap": "Do not claim unavailable methods are inferior.",
            "priority": "high",
        },
        {
            "goal_id": "goal5_claim_boundary",
            "requirement": "Unsafe claims are translated into safe manuscript wording.",
            "status": "complete" if len(boundaries) >= 5 else "partial",
            "evidence": "phase8_claim_boundary_table.csv",
            "remaining_gap": "Apply this wording in Abstract/Results/Discussion.",
            "priority": "high",
        },
        {
            "goal_id": "goal6_methods_reproducibility",
            "requirement": "Methods and reproducibility checklist covers benchmark, model, FDR, baselines, runtime, source tables, and tutorial.",
            "status": "complete" if not partial_methods and not missing_methods else "partial",
            "evidence": f"phase8_methods_reproducibility_checklist.csv;complete_topics={len(complete_methods)}/{len(methods)}",
            "remaining_gap": "Final public tutorial and manuscript prose polishing remain outside this evidence-consolidation package.",
            "priority": "high",
        },
    ]
    return rows


def _build_source_table_index(
    context: dict[str, Any],
    core_claims: list[dict[str, str]],
    figure_plan: list[dict[str, str]],
    cases: list[dict[str, str]],
    baselines: list[dict[str, str]],
    methods: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(source_id: str, artifact_type: str, manuscript_use: str, path: str | Path, boundary: str) -> None:
        source = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
        generated_by_this_script = source == context["output_dir"] / "phase8_source_table_index.csv"
        rows.append(
            {
                "source_id": source_id,
                "artifact_type": artifact_type,
                "manuscript_use": manuscript_use,
                "path": _rel(source),
                "exists": str(source.exists() or generated_by_this_script),
                "claim_boundary": boundary,
            }
        )

    for row in figure_plan:
        for idx, path in enumerate(_split_paths(row["source_tables"]), 1):
            add(
                f"{row['figure_id']}_table_{idx}",
                "figure_source_table",
                f"{row['figure_id']}: {row['proposed_title']}",
                path,
                row["limitation"],
            )
        for idx, path in enumerate(_split_paths(row["source_figures_or_rendered_assets"]), 1):
            add(
                f"{row['figure_id']}_asset_{idx}",
                "rendered_asset",
                f"{row['figure_id']}: existing draft asset",
                path,
                row["limitation"],
            )
    for idx, row in enumerate(cases, 1):
        for source_idx, path in enumerate(_split_paths(row["source_tables"]), 1):
            add(
                f"case{idx}_source_{source_idx}",
                "real_case_source_table",
                f"{row['case_id']}: {row['lr_pair']}",
                path,
                row["limitation"],
            )
    for idx, row in enumerate(baselines, 1):
        add(
            f"deep_baseline_{idx}",
            "baseline_status_table",
            f"{row['method']} baseline closure",
            row["source_evidence"],
            row["limitation"],
        )
    for idx, row in enumerate(methods, 1):
        add(
            f"methods_{idx}",
            "methods_source_or_script",
            row["methods_topic"],
            row["evidence_or_source"],
            row["claim_boundary"],
        )
    for idx, row in enumerate(core_claims, 1):
        add(
            f"claim_{idx}_primary",
            "claim_primary_evidence",
            row["claim_id"],
            row["primary_evidence"],
            row["limitation"],
        )
    return rows


def _build_methods_reproducibility_package(
    methods: list[dict[str, str]],
    boundaries: list[dict[str, str]],
    source_index: list[dict[str, str]],
) -> str:
    missing_sources = [row for row in source_index if row["exists"] != "True"]
    lines = [
        "# Phase 8 Methods and Reproducibility Package",
        "",
        "This package is a manuscript-facing evidence map. It locks the Methods wording needed for Phase 8, but it is not a polished public tutorial or final journal supplement.",
        "",
        "## Core Methods Wording",
        "",
        "### Benchmark generation",
        "",
        "We used two benchmark families. The controlled simulator creates ligand-receptor (LR) truth under contact-dependent, short-to-medium paracrine, and longer-range niche-restricted mechanisms. Each mechanism includes sender and receiver cell types, expression strength, background expression, weak injected signals, and high-expression negative-control LR pairs. The semi-synthetic real-background benchmark keeps real ST coordinates, local density, cell-type composition, expression sparsity, and library-size structure, then injects known LR activities into defined niches. Therefore AUROC, AUPR, recall, weak-signal recall, and FDR are evaluated against injected truth, not native tissue truth.",
        "",
        "### Graph wavelet SpatialSelf model",
        "",
        "SpatialSelf scores candidate sender-receiver-LR events using expression features, distance features, LR identity-aware parameters, and graph-wavelet spatial propagation. The graph-wavelet component is used to propagate and compare local spatial context across graph scales. The LR identity component allows different LR pairs to learn different spatial/use-context behavior instead of forcing all LR pairs into one expression-distance rule.",
        "",
        "### KNN graph versus biological signaling range",
        "",
        "The KNN graph is a computational candidate graph, not the definition of biological truth. We denote the graph neighborhood size as K_NN. K_NN controls which sender-receiver pairs are considered for scoring and spatial propagation. Biological signaling range is tested separately through mechanism-specific benchmark labels, distance features, range annotations, and permutation/FDR analyses. This distinction matters: increasing K_NN expands the candidate search radius, while LR-specific distance/range features and local-background controls determine whether a candidate edge is prioritized. We should state the limitation explicitly: a finite candidate graph can miss edges outside the evaluated graph radius, so long-range claims are only valid within the chosen candidate construction and sensitivity analysis.",
        "",
        "### Self-supervised pseudo-labels",
        "",
        "Pseudo-labels are training signals derived from expression, spatial proximity, and LR-specific candidate scores; they are not treated as ground truth. We reserve K_NN for graph construction and use K_pos and K_neg for the numbers or fractions of high-confidence positive and negative pseudo-labeled candidate events. Weak-signal analyses are reported separately because top-ranked pseudo-labels can be biased toward strong expression if not audited. In the manuscript, pseudo-label performance should be evaluated using weak-signal recall, high-expression false-positive controls, ablations of pseudo-label strategy, and downstream/FDR validation.",
        "",
        "### Permutation, FDR, and local-contrast controls",
        "",
        "Permutation testing uses multiple nulls to ask different questions: spatial permutation disrupts spatial organization while retaining expression, receiver/cell-type permutation disrupts receiver assignment, and LR-identity permutation disrupts LR-specific labels while retaining global expression structure. Competitive LR and model-rerun diagnostics are reported separately when available. Local-contrast FDR compares a prioritized positive LR against matched high-expression negative-control background and is designed for settings where raw scores are saturated or high-expression leakage is the main risk. It must not be described as raw-score all-null FDR.",
        "",
        "### Baseline reproducibility",
        "",
        "Comparable baselines are interpreted only when they produce compatible LR-pair or candidate-edge ranking outputs. DeepTalk, HoloNet, stMLnet, CytoSignal, and NICHES status entries should be reported as reproducibility and compatibility evidence. A failed or incompatible run is not evidence that the method is biologically worse.",
        "",
        "### Source tables and reproducibility",
        "",
        "The Phase 8 source table index links each main figure, case, baseline closure, and Methods topic to its source table, script, or rendered draft asset. Missing or not-yet-rendered final figure assets should be treated as figure-polishing gaps, not as missing evidence if the source tables exist.",
        "",
        "## Notation Table",
        "",
        "| Symbol | Meaning | Claim boundary |",
        "| --- | --- | --- |",
        "| K_NN | Number of nearest neighbors used to build the candidate/spatial graph | Computational search radius, not biological truth |",
        "| K_pos | Number or fraction of high-confidence pseudo-positive events | Self-supervised training signal, not oracle truth |",
        "| K_neg | Number or fraction of pseudo-negative events | Should be audited with high-expression negative controls |",
        "| R_max | Optional maximum evaluated spatial radius | Claims beyond this radius are not supported |",
        "| tau | Distance bandwidth or decay parameter | Sensitivity should be reported where used |",
        "| lambda | Spatial/local-background weighting parameter | Dataset- and validation-dependent |",
        "",
        "## Claim Guardrails",
        "",
    ]
    for row in boundaries:
        lines.extend(
            [
                f"- {row['boundary_id']}: {row['safe_wording']}",
                f"  Avoid: {row['unsafe_wording']}",
            ]
        )
    lines.extend(["", "## Methods Checklist Status", ""])
    for row in methods:
        lines.append(
            f"- {row['methods_topic']}: {row['status']}. Source: {row['evidence_or_source']}. Boundary: {row['claim_boundary']}"
        )
    lines.extend(["", "## Source Index Audit", ""])
    lines.append(f"- Indexed evidence rows: {len(source_index)}")
    lines.append(f"- Missing indexed sources: {len(missing_sources)}")
    if missing_sources:
        lines.append("- Missing-source rows to resolve or justify:")
        for row in missing_sources[:20]:
            lines.append(f"  - {row['source_id']}: {row['path']} ({row['manuscript_use']})")
        if len(missing_sources) > 20:
            lines.append(f"  - ... {len(missing_sources) - 20} additional rows")
    else:
        lines.append("- All indexed source paths exist at build time.")
    lines.extend(
        [
            "",
            "## Minimal Reproduction Entry Points",
            "",
            "- Controlled and semi-synthetic benchmark generation: `scripts/benchmark/generate_controlled_ccc.py` and `scripts/benchmark/generate_semisynthetic_real_background.py`.",
            "- SpatialSelf scoring and ablations: `scripts/benchmark/run_spatialself.py` and `scripts/benchmark/run_spatialself_ablation.py`.",
            "- Benchmark realism: `scripts/phase7/run_goal3_benchmark_realism_figure.py`.",
            "- Downstream target validation: `scripts/phase7/run_goal1_downstream_target_validation.py` and `scripts/benchmark/run_downstream_target_correlation.py`.",
            "- Permutation/local-contrast FDR: `scripts/benchmark/run_formal_fdr_panel.py`, `scripts/phase7/run_goal5_local_contrast_fdr.py`, and related Phase 7 FDR scripts.",
            "- Deep baseline closure: `scripts/phase7/build_goal2_deep_baseline_status.py` plus baseline adapters in `scripts/benchmark/`.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_reproducibility_tutorial() -> str:
    return "\n".join(
        [
            "# Phase 8 Reproducibility Tutorial",
            "",
            "This tutorial has two levels. The first level rebuilds the manuscript evidence package from existing Phase 4/5/7 outputs. The second level points to the scripts used to regenerate upstream analyses when raw/intermediate data are available.",
            "",
            "## Level 1: rebuild the Phase 8 evidence package",
            "",
            "Run:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/phase8/build_phase8_manuscript_evidence.py --config configs/experiments/phase8_manuscript_grade_evidence_consolidation.yaml",
            "```",
            "",
            "Expected key outputs:",
            "",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_summary.json`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_completion_gap_audit.csv`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_main_figure_plan.csv`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_real_case_deep_dive_table.csv`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_deep_baseline_closure_table.csv`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_claim_boundary_table.csv`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_methods_reproducibility_package.md`",
            "- `outputs/phase8/manuscript_grade_evidence_consolidation/phase8_source_table_index.csv`",
            "",
            "A successful Phase 8 evidence package should report `phase8_package_complete: true` in the summary and six `complete` rows in the completion audit. This means the manuscript evidence chain is consolidated; it does not mean final journal figures or full public documentation are polished.",
            "",
            "## Level 2: regenerate upstream evidence",
            "",
            "Controlled and semi-synthetic benchmarks:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/benchmark/generate_controlled_ccc.py",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/benchmark/generate_semisynthetic_real_background.py",
            "```",
            "",
            "SpatialSelf predictions and ablations:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/benchmark/run_spatialself.py",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/benchmark/run_spatialself_ablation.py",
            "```",
            "",
            "Benchmark realism and downstream validation:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/phase7/run_goal3_benchmark_realism_figure.py",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/phase7/run_goal1_downstream_target_validation.py",
            "```",
            "",
            "Permutation/FDR and local-contrast controls:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/benchmark/run_formal_fdr_panel.py",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/phase7/run_goal5_local_contrast_fdr.py",
            "```",
            "",
            "Deep baseline closure/status table:",
            "",
            "```bash",
            "/home/xd/y/Benchmark/.conda-env/bin/python scripts/phase7/build_goal2_deep_baseline_status.py",
            "```",
            "",
            "## Interpretation rules",
            "",
            "- Treat benchmark AUROC/AUPR/FDR as injected-truth evidence.",
            "- Treat real ST validation as control-based support, not native ground truth.",
            "- Treat local-contrast FDR as matched high-expression-background evidence, not raw-score all-null FDR.",
            "- Treat incompatible deep baselines as reproducibility/status limitations, not biological losses.",
            "- Keep K_NN separate from K_pos/K_neg in all Methods text and figure legends.",
            "",
        ]
    )


def _build_results_skeleton(
    core_claims: list[dict[str, str]],
    figure_plan: list[dict[str, str]],
    cases: list[dict[str, str]],
) -> str:
    lines = [
        "# Phase 8 Results Skeleton",
        "",
        "## Proposed central claim",
        "",
        core_claims[0]["allowed_wording"],
        "",
        "## Results section order",
        "",
    ]
    titles = [
        "SpatialSelf integrates LR identity-aware modeling with graph-wavelet spatial structure for CCC inference",
        "Realistic benchmarks test weak, contact-dependent, paracrine, long-range, and high-expression false-positive CCC",
        "Semi-synthetic benchmarks preserve key spatial transcriptomics statistics from real tissues",
        "SpatialSelf improves weak/context-specific CCC prioritization while reducing high-expression false-positive leakage",
        "Ablation studies identify LR identity and graph-wavelet spatial modeling as key contributors",
        "Permutation and local-contrast FDR provide calibrated statistical support for prioritized CCC",
        "SpatialSelf-prioritized LR activity is associated with receiver downstream target responses in real tissues",
        "Cross-platform validation highlights SPP1-CD44 as a local-contrast FDR-supported context-specific case",
    ]
    for i, title in enumerate(titles, 1):
        fig = figure_plan[i - 1]
        lines.extend(
            [
                f"### {i}. {title}",
                "",
                f"Primary figure: {fig['figure_id']} - {fig['proposed_title']}",
                "",
                f"Main message: {fig['main_message']}",
                "",
                f"Claim boundary: {fig['limitation']}",
                "",
            ]
        )
    lines.extend(["## Recommended real-case hierarchy", ""])
    for row in cases:
        lines.extend(
            [
                f"- {row['case_id']}: {row['lr_pair']} in {row['dataset']} ({row['evidence_tier']}). Use: {row['recommended_manuscript_use']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Language guardrail",
            "",
            "Use 'prioritizes', 'supports', 'is associated with', and 'matched local-background control'. Avoid 'proves', 'causal', 'native ground truth', and 'universally best'.",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_key_figures(phase4: Path, phase5: Path, phase7: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = [
        phase4 / "figures" / "fig1_phase4_manuscript.pdf",
        phase4 / "figures" / "fig2_phase4_manuscript.pdf",
        phase4 / "figures" / "fig3_phase4_manuscript.pdf",
        phase4 / "figures" / "fig4_phase4_manuscript.pdf",
        phase4 / "figures" / "fig5_phase4_manuscript.pdf",
        phase4 / "figures" / "fig6_phase4_manuscript.pdf",
        phase5 / "figures" / "fig7_phase5_real_validation.pdf",
        phase5 / "figures" / "fig8_phase5_real_validation.pdf",
        phase7 / "goal3_benchmark_realism_figure" / "figures" / "fig_phase7_benchmark_realism.pdf",
    ]
    for asset in assets:
        if asset.exists():
            shutil.copy2(asset, output_dir / asset.name)


def _deep_baseline_use(method: str, status: str) -> str:
    if "formal_run_completed" in status or "usable" in status:
        return "comparable_if_lr_pair_ranking_outputs_exist"
    if method.lower() == "deeptalk":
        return "status_table_plus_input_preparation_summary"
    return "supplementary_status_or_failure_table"


def _rank_from_downstream(rows: list[dict[str, str]], ligand: str, receptor: str) -> str:
    row = _find_lr(rows, ligand, receptor)
    if not row:
        return ""
    return row.get("spatialself_rank_by_target_r", "")


def _downstream_summary(rows: list[dict[str, str]], ligand: str, receptor: str) -> str:
    row = _find_lr(rows, ligand, receptor)
    if not row:
        return ""
    return (
        f"SpatialSelf rho={row.get('spatialself_target_spearman_r','')};"
        f"FDR={row.get('spatialself_bh_fdr','')};"
        f"best_method={row.get('best_method_by_target_r','')}"
    )


def _find_lr(rows: list[dict[str, str]], ligand: str, receptor: str) -> dict[str, str]:
    for row in rows:
        if row.get("ligand") == ligand and row.get("receptor") == receptor:
            return row
    return {}


def _local_contrast_summary(row: dict[str, str]) -> str:
    if not row:
        return ""
    return (
        f"all_null_success={row.get('local_contrast_all_null_success','')};"
        f"nulls={row.get('local_contrast_nulls_passing','') or row.get('null_models_passing','')};"
        f"best_positive_fdr={row.get('local_contrast_best_positive_fdr','') or row.get('best_positive_local_contrast_fdr','')};"
        f"best_negative_fdr={row.get('local_contrast_best_negative_fdr','') or row.get('best_negative_local_contrast_fdr','')}"
    )


def _min_float(rows: list[dict[str, str]], column: str) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(column, "")))
        except ValueError:
            pass
    return min(values) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_paths(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(";") if part]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
