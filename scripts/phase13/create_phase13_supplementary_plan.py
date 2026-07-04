#!/usr/bin/env python
"""Create Phase 13 supplementary figure plan and source-data index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/phase13"


SUPP_FIGURES = [
    {
        "supplementary_figure": "Supplementary Fig. 1",
        "title": "Benchmark realism audit",
        "review_risk_addressed": "simulation too artificial / benchmark not independent",
        "source_tables": "outputs/phase7/functional_cross_platform_validation/goal3_benchmark_realism_figure/source_data/phase7_realism_cell_library_sparsity.csv;outputs/phase7/functional_cross_platform_validation/goal3_benchmark_realism_figure/source_data/phase7_realism_gene_mean_variance.csv;outputs/phase7/functional_cross_platform_validation/goal3_benchmark_realism_figure/source_data/phase7_realism_moran.csv;outputs/phase7/functional_cross_platform_validation/goal3_benchmark_realism_figure/source_data/phase7_realism_spatial_density.csv",
        "main_claim": "Semi-synthetic benchmarks preserve several real-ST distributional features.",
        "claim_boundary": "Realism statistics support benchmark plausibility, not biological truth.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 2",
        "title": "Full formal baseline metrics",
        "review_risk_addressed": "baseline comparison completeness",
        "source_tables": "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_summary.csv;outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_metrics.csv",
        "main_claim": "AUPR, AUROC, top-k, weak recall, empirical FDR and FP rate are reported across retained baselines.",
        "claim_boundary": "Adapter-light/projected baselines should be labeled as such.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 3",
        "title": "Weak-signal tradeoff and sensitivity",
        "review_risk_addressed": "pseudo-label top-K may miss weak CCC",
        "source_tables": "outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_method_gate.csv;outputs/phase2_revised/goal2_tuned_guardrail_sensitivity/goal2_tuned_guardrail_sensitivity_summary.csv;outputs/phase2_revised/goal2_tuned_guardrail_sensitivity/goal2_tuned_guardrail_sensitivity.csv",
        "main_claim": "SpatialSelf provides a practical weak-signal / false-positive tradeoff under tuning and guardrail stress.",
        "claim_boundary": "Do not claim maximum raw weak recall.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 4",
        "title": "Architecture and LR-identity ablation",
        "review_risk_addressed": "model gain may come from capacity rather than LR identity/wavelet design",
        "source_tables": "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv;outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_metrics.csv",
        "main_claim": "LR identity, expression, distance, spatial graph and graph backbone variants have measurable effects.",
        "claim_boundary": "Use as controlled ablation evidence; align exact full-model variant with final manuscript model.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 5",
        "title": "Deep baseline closure and method status",
        "review_risk_addressed": "CellNEST / DeepTalk / HoloNet / stMLnet / CytoSignal comparability",
        "source_tables": "outputs/phase8/manuscript_grade_evidence_consolidation/phase8_deep_baseline_closure_table.csv;outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_status.csv",
        "main_claim": "Deep and spatial CCC method status, input compatibility and LR-rank comparability are reported transparently.",
        "claim_boundary": "DeepTalk is status-only unless comparable LR-pair scoring output is available.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 6",
        "title": "Runtime and scalability",
        "review_risk_addressed": "computational cost / high-resolution feasibility",
        "source_tables": "outputs/phase2_revised/goal7_runtime_scalability/goal7_method_runtime_profile.csv;outputs/phase2_revised/goal7_runtime_scalability/goal7_highres_scaling_summary.csv;outputs/phase2_revised/goal7_runtime_scalability/goal7_highres_scaling_runs.csv",
        "main_claim": "SpatialSelf runtime is practical on retained high-resolution and scaling benchmarks.",
        "claim_boundary": "Runtime adapters are not equivalent to full biological comparability.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 7",
        "title": "Expanded atlas and top-k contamination",
        "review_risk_addressed": "selected LR-only evaluation",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure1_expanded_calibrated_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure1_expanded_calibrated_topk_summary.csv;outputs/phase12/manuscript_figure_package/figure1_expanded_calibrated_lr_ranking_top25.csv",
        "main_claim": "Expanded LR-atlas scoring and top-k contamination are quantified across three datasets.",
        "claim_boundary": "Calibrated ranking support is not final unbiased whole-atlas discovery FDR.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 8",
        "title": "Targeted, target-decoy and matched-null FDR",
        "review_risk_addressed": "lack of p-value/FDR",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure1_expanded_target_decoy_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure1_expanded_target_decoy_topk_summary.csv;outputs/phase12/manuscript_figure_package/figure1_targeted_matched_null_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure1_targeted_matched_null_supported_lr.csv",
        "main_claim": "Target-decoy and targeted matched-null audits quantify control contamination and story-level support.",
        "claim_boundary": "Do not call these strict whole-atlas permutation FDR.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 9",
        "title": "Post-Xenium colon source evidence",
        "review_risk_addressed": "biological case could be cherry-picked",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure2_crc_method_lr_rank_table.csv;outputs/phase12/manuscript_figure_package/figure2_crc_local_contrast_fdr.csv;outputs/phase12/manuscript_figure_package/figure2_crc_whole_transcriptome_downstream.csv;outputs/phase12/manuscript_figure_package/figure2_crc_commot_style_impact.csv",
        "main_claim": "The SPP1-CD44 case has ranking, local contrast and receiver-response source evidence.",
        "claim_boundary": "Observational validation, not perturbational causality.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 10",
        "title": "Lymph node and tonsil supportive validation",
        "review_risk_addressed": "single biological case / lack of cross-platform validation",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure3_lymph_node_directionality_edge_permutation.csv;outputs/phase12/manuscript_figure_package/figure3_lymph_node_program_specificity.csv;outputs/phase12/manuscript_figure_package/figure3_tonsil_A1_method_lr_rank.csv;outputs/phase12/manuscript_figure_package/figure3_tonsil_A1_local_contrast_fdr.csv;outputs/phase12/manuscript_figure_package/figure3_tonsil_D1_method_lr_rank.csv",
        "main_claim": "Lymph node and tonsil provide supportive, caveated cross-platform validation.",
        "claim_boundary": "Not a clean second main story; disclose negative leakage and mixed downstream closure.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 11",
        "title": "Hard-negative and coordinate-control guardrails",
        "review_risk_addressed": "false positives and coordinate overfitting",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure4_hard_negative_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure4_edge_hard_negative_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure4_coordinate_control_guardrail_dataset_summary.csv;outputs/phase12/manuscript_figure_package/figure4_coordinate_model_rerun_smoke_summary.csv",
        "main_claim": "False-positive modes and coordinate-control behavior are explicitly quantified.",
        "claim_boundary": "Coordinate rerun is smoke/guardrail evidence, not final whole-atlas coordinate-null FDR.",
        "status": "planned",
    },
    {
        "supplementary_figure": "Supplementary Fig. 12",
        "title": "Claim matrix and forbidden extensions",
        "review_risk_addressed": "overclaiming risk",
        "source_tables": "outputs/phase12/manuscript_figure_package/figure4_claim_matrix_guardrails.csv;outputs/phase12/manuscript_figure_package/phase12_manuscript_figure_panel_index.csv",
        "main_claim": "Every major manuscript claim is linked to a source table and caveat.",
        "claim_boundary": "This is a writing and review guardrail, not new evidence.",
        "status": "planned",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SUPP_FIGURES)
    df["all_source_tables_exist"] = df["source_tables"].map(all_sources_exist)
    out_csv = OUT / "phase13_supplementary_figure_plan.csv"
    out_md = OUT / "phase13_supplementary_figure_plan.md"
    df.to_csv(out_csv, index=False)
    out_md.write_text(render_markdown(df), encoding="utf-8")
    print(df[["supplementary_figure", "title", "status", "all_source_tables_exist"]].to_string(index=False))


def all_sources_exist(source_tables: str) -> bool:
    return all((ROOT / p.strip()).exists() for p in str(source_tables).split(";") if p.strip())


def render_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Phase 13 Supplementary Figure Plan",
        "",
        "This plan turns the large evidence base into traceable supplementary figures. It is deliberately conservative: each figure has a claim boundary so that supplementary data do not inflate the manuscript claims.",
        "",
    ]
    for row in df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.supplementary_figure}: {row.title}",
                "",
                f"- Review risk addressed: {row.review_risk_addressed}",
                f"- Main claim: {row.main_claim}",
                f"- Claim boundary: {row.claim_boundary}",
                f"- Source tables exist: `{row.all_source_tables_exist}`",
                f"- Status: `{row.status}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
