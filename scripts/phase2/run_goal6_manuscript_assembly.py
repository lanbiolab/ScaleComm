#!/usr/bin/env python
"""Assemble revised Phase 2 manuscript source tables and claim matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal6_manuscript_assembly.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal6_manuscript_assembly"))
    source_dir = output_dir / "source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_source_tables(source_dir)

    copied = _copy_source_tables(config["source_tables"], source_dir)
    claim_matrix = _build_claim_matrix(copied)
    limitations = _build_limitations_table()
    figure_plan = _build_figure_plan()
    panel_index = _build_panel_index(copied)

    claim_matrix.to_csv(output_dir / "claim_evidence_matrix.csv", index=False)
    limitations.to_csv(output_dir / "manuscript_limitations.csv", index=False)
    figure_plan.to_csv(output_dir / "figure_panel_plan.csv", index=False)
    panel_index.to_csv(output_dir / "source_table_index.csv", index=False)

    manifest = {
        "config": args.config,
        "output_dir": str(output_dir),
        "source_tables_dir": str(source_dir),
        "n_source_tables": int(len(copied)),
        "canonical_goal": config.get("canonical_goal", "goal6_manuscript_evidence_figure_convergence"),
        "historical_artifact_mapping": config.get("historical_artifact_mapping", {}),
        "claim_matrix": str(output_dir / "claim_evidence_matrix.csv"),
        "limitations": str(output_dir / "manuscript_limitations.csv"),
        "figure_plan": str(output_dir / "figure_panel_plan.csv"),
        "source_table_index": str(output_dir / "source_table_index.csv"),
        "claim_level": config.get("claim_level", "source_table_assembly_not_final_figure_bundle"),
        "next_required_step": "Generate and QA manuscript figures from these source tables.",
    }
    (output_dir / "goal6_manuscript_assembly_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(claim_matrix.to_string(index=False))
    print(f"Wrote revised Goal 6 manuscript assembly outputs to {output_dir}")


def _copy_source_tables(source_tables: dict[str, str], source_dir: Path) -> list[dict]:
    copied: list[dict] = []
    for table_id, relative_path in source_tables.items():
        src = PROJECT_ROOT / relative_path
        dest = source_dir / f"{table_id}.csv"
        status = "copied" if src.exists() else "missing"
        n_rows = 0
        n_columns = 0
        if src.exists():
            shutil.copy2(src, dest)
            try:
                table = pd.read_csv(dest)
                n_rows = int(len(table))
                n_columns = int(len(table.columns))
            except Exception:  # noqa: BLE001 - index should still record malformed tables.
                status = "copied_unreadable"
        copied.append(
            {
                "table_id": table_id,
                "source_path": str(src),
                "assembled_path": str(dest),
                "status": status,
                "n_rows": n_rows,
                "n_columns": n_columns,
            }
        )
    return copied


def _clear_stale_source_tables(source_dir: Path) -> None:
    for path in source_dir.glob("*.csv"):
        path.unlink()


def _build_claim_matrix(copied: list[dict]) -> pd.DataFrame:
    table_status = {row["table_id"]: row for row in copied}
    rows = [
        {
            "claim_id": "C1_simulator_v2_mechanism_layered_truth",
            "main_text_role": "benchmark design",
            "supported_statement": "Controlled benchmark v2 formal outputs contain contact, paracrine, chemokine-like, weak, and high-expression negative-control LR truth tables across replicated generated datasets; the Goal3/4 benchmark quality report and shared truth-contract validator confirm the required truth schema, mechanism coverage, weak positives, and zero high-expression false-positive truth edges.",
            "primary_source_tables": "goal3_formal_benchmark_summary;goal3_formal_claims;goal3_mechanism_summary;goal3_lr_summary;goal34_benchmark_quality_summary;goal34_mechanism_coverage_matrix;goal34_truth_contract_summary;goal34_benchmark_quality_claims;goal34_truth_contract_replicate_summary;goal34_truth_contract_dataset_summary;goal34_truth_contract_claims",
            "evidence_status": _table_status(table_status, ["goal3_formal_benchmark_summary", "goal3_formal_claims", "goal3_mechanism_summary", "goal3_lr_summary", "goal34_benchmark_quality_summary", "goal34_mechanism_coverage_matrix", "goal34_truth_contract_summary", "goal34_benchmark_quality_claims", "goal34_truth_contract_replicate_summary", "goal34_truth_contract_dataset_summary", "goal34_truth_contract_claims"]),
            "claim_scope": "formal mechanism-layered generated datasets plus benchmark quality report and shared truth-contract validator",
            "allowed_wording": "mechanism-layered benchmark with auditable truth tables and quality report",
            "disallowed_wording": "fully realistic simulator of all CCC biology",
        },
        {
            "claim_id": "C2_real_background_v2_preserves_st_properties",
            "main_text_role": "real-background benchmark",
            "supported_statement": "Real-background semi-synthetic v2 formal outputs preserve coordinates, sparsity, library size, cell types, local density, and tissue-domain labels for lymph node, tonsil, and melanoma backgrounds with injected truth; the quality report records measured ST-property ranges, truth-contract checks, and non-saturating heuristic performance. The replicated tuned-baseline dataset layer contains 18 generated real-background replicates, six per tissue background, and the shared truth-contract validator confirms all 18 satisfy the common truth schema with weak positives and high-expression false-positive controls kept ground-truth negative.",
            "primary_source_tables": "goal4_formal_real_background_summary;goal4_formal_claims;goal4_lymph_validation;goal4_tonsil_validation;goal4_melanoma_validation;goal4_replicated_tuned_dataset_status;goal4_replicated_tuned_validation_summary;goal4_replicated_tuned_real_background_summary;goal4_replicated_tuned_claims;goal4_replicated_tuned_manifest;goal34_benchmark_quality_summary;goal34_real_background_property_summary;goal34_truth_contract_summary;goal34_benchmark_quality_claims;goal34_truth_contract_replicate_summary;goal34_truth_contract_dataset_summary;goal34_truth_contract_claims",
            "evidence_status": _table_status(table_status, ["goal4_formal_real_background_summary", "goal4_formal_claims", "goal4_lymph_validation", "goal4_tonsil_validation", "goal4_melanoma_validation", "goal4_replicated_tuned_dataset_status", "goal4_replicated_tuned_validation_summary", "goal4_replicated_tuned_real_background_summary", "goal4_replicated_tuned_claims", "goal4_replicated_tuned_manifest", "goal34_benchmark_quality_summary", "goal34_real_background_property_summary", "goal34_truth_contract_summary", "goal34_benchmark_quality_claims", "goal34_truth_contract_replicate_summary", "goal34_truth_contract_dataset_summary", "goal34_truth_contract_claims"]),
            "claim_scope": "formal real-background semi-synthetic benchmarks with injected truth, retained smoke summaries, benchmark quality report, and generated 18-replicate dataset layer validated by shared truth contract",
            "allowed_wording": "real ST background with injected controls, measured preservation summary, and generated 18-replicate tuned-baseline dataset layer",
            "disallowed_wording": "true unlabeled real-data ground truth",
        },
        {
            "claim_id": "C3_spatialself_fp_suppression_tradeoff",
            "main_text_role": "baseline comparison and weak signal tradeoff",
            "supported_statement": "Across retained smoke panels, the formal Goal3/4 adapter-aware baseline panel, and the formal weak-signal stress summary, SpatialSelf has a FP/FDR-constrained weak-signal tradeoff signal. The Goal2 readiness gate covers SpatialSelf wavelet edge prior, SpatialSelf wavelet, COMMOT, CellChat, SpaTalk, and CellNEST, with DeepTalk and NICHES retained as status-recorded baselines. It supports the tradeoff claim in the mechanism-stratified stress gate and aggregate frozen endpoint, while explicitly marking highest raw weak recall, tissue-level dominance, and paired significance against every baseline as unsupported. The replicated tuned baseline panel contains 18 matched real-background replicates and 108 metric rows; it favors SpatialSelf on the aggregate frozen endpoint and paired mean deltas, but method-mean lowest FP and best replicated-panel tradeoff can favor SpaTalk, and highest raw weak recall favors CellNEST.",
            "primary_source_tables": "goal2_metrics;goal2_summary;goal2_claims;goal2_status;goal2_formal_metrics;goal2_formal_summary;goal2_formal_claims;goal2_formal_status;goal2_formal_tuned_small_metrics;goal2_formal_tuned_small_summary;goal2_formal_tuned_small_claims;goal2_formal_tuned_small_status;goal2_formal_tuned_small_parameters;goal2_tuned_guardrail_sensitivity;goal2_tuned_guardrail_sensitivity_summary;goal2_tuned_guardrail_sensitivity_claims;goal2_tuned_guardrail_sensitivity_status;goal2_preregistered_endpoint_scores;goal2_preregistered_endpoint_summary;goal2_preregistered_endpoint_claims;goal2_preregistered_endpoint_policy_table;goal2_preregistered_endpoint_status;goal2_preregistered_fp_suppression_by_tissue;goal2_preregistered_paired_deltas;goal2_preregistered_paired_summary;goal2_preregistered_paired_claims;goal2_preregistered_paired_status;goal2_replicate_power_plan_by_baseline;goal2_replicate_power_plan_overall;goal2_replicate_power_plan_design;goal2_replicate_power_plan_claims;goal2_replicate_power_plan_status;goal2_replicated_tuned_execution_plan;goal2_replicated_tuned_replicate_manifest;goal2_replicated_tuned_method_jobs;goal2_replicated_tuned_job_status_template;goal2_replicated_tuned_job_claims;goal2_replicated_tuned_metric_rows;goal2_replicated_tuned_metric_summary;goal2_replicated_tuned_tissue_metric_summary;goal2_replicated_tuned_replicate_summary;goal2_replicated_tuned_run_status;goal2_replicated_tuned_run_status_summary;goal2_replicated_tuned_panel_claims;goal2_replicated_tuned_endpoint_scores;goal2_replicated_tuned_endpoint_summary;goal2_replicated_tuned_endpoint_claims;goal2_replicated_tuned_endpoint_policy_table;goal2_replicated_tuned_paired_deltas;goal2_replicated_tuned_paired_summary;goal2_replicated_tuned_paired_claims;goal2_replicated_tuned_paired_status;goal2_weak_stress_metrics;goal2_weak_stress_summary;goal2_weak_stress_claims;goal2_weak_stress_status;goal2_weak_tradeoff_method_gate;goal2_weak_tradeoff_tissue_gate;goal2_weak_tradeoff_stress_gate;goal2_weak_tradeoff_paired_gate;goal2_weak_tradeoff_claims",
            "evidence_status": _table_status(table_status, ["goal2_metrics", "goal2_summary", "goal2_claims", "goal2_status", "goal2_formal_metrics", "goal2_formal_summary", "goal2_formal_claims", "goal2_formal_status", "goal2_formal_tuned_small_metrics", "goal2_formal_tuned_small_summary", "goal2_formal_tuned_small_claims", "goal2_formal_tuned_small_status", "goal2_formal_tuned_small_parameters", "goal2_tuned_guardrail_sensitivity", "goal2_tuned_guardrail_sensitivity_summary", "goal2_tuned_guardrail_sensitivity_claims", "goal2_tuned_guardrail_sensitivity_status", "goal2_preregistered_endpoint_scores", "goal2_preregistered_endpoint_summary", "goal2_preregistered_endpoint_claims", "goal2_preregistered_endpoint_policy_table", "goal2_preregistered_endpoint_status", "goal2_preregistered_fp_suppression_by_tissue", "goal2_preregistered_paired_deltas", "goal2_preregistered_paired_summary", "goal2_preregistered_paired_claims", "goal2_preregistered_paired_status", "goal2_replicate_power_plan_by_baseline", "goal2_replicate_power_plan_overall", "goal2_replicate_power_plan_design", "goal2_replicate_power_plan_claims", "goal2_replicate_power_plan_status", "goal2_replicated_tuned_execution_plan", "goal2_replicated_tuned_replicate_manifest", "goal2_replicated_tuned_method_jobs", "goal2_replicated_tuned_job_status_template", "goal2_replicated_tuned_job_claims", "goal2_replicated_tuned_metric_rows", "goal2_replicated_tuned_metric_summary", "goal2_replicated_tuned_tissue_metric_summary", "goal2_replicated_tuned_replicate_summary", "goal2_replicated_tuned_run_status", "goal2_replicated_tuned_run_status_summary", "goal2_replicated_tuned_panel_claims", "goal2_replicated_tuned_endpoint_scores", "goal2_replicated_tuned_endpoint_summary", "goal2_replicated_tuned_endpoint_claims", "goal2_replicated_tuned_endpoint_policy_table", "goal2_replicated_tuned_paired_deltas", "goal2_replicated_tuned_paired_summary", "goal2_replicated_tuned_paired_claims", "goal2_replicated_tuned_paired_status", "goal2_weak_stress_metrics", "goal2_weak_stress_summary", "goal2_weak_stress_claims", "goal2_weak_stress_status", "goal2_weak_tradeoff_method_gate", "goal2_weak_tradeoff_tissue_gate", "goal2_weak_tradeoff_stress_gate", "goal2_weak_tradeoff_paired_gate", "goal2_weak_tradeoff_claims"]),
            "claim_scope": "three real-background smoke datasets, formal controlled/real-background adapter-aware panels, weak-stress mechanism summary, tuned guardrail sensitivity, frozen endpoint policy, completed 18-replicate tuned baseline panel, paired endpoint diagnostics, and Goal2 readiness gate; DeepTalk/NICHES remain status-recorded",
            "allowed_wording": "SpatialSelf supports an FP/FDR-guardrailed weak-signal tradeoff in the mechanism-stratified stress gate and aggregate frozen endpoint, while replicated-panel method means and tissue winners remain mixed",
            "disallowed_wording": "highest weak recall, lowest false-positive rate against every baseline, global dominance across all baselines or tissues, SpatialSelf winning every tissue, paired significance against every baseline, or endpoint rules chosen without disclosure",
        },
        {
            "claim_id": "C4_baseline_matched_control_ranking",
            "main_text_role": "biological validation",
            "supported_statement": "In formal lymph-node, tonsil, and melanoma real-background injected-control validation, SpatialSelf ranks injected positive controls ahead of high-expression negative controls and keeps high-expression false-positive call rates low relative to COMMOT, CellChat, and CellNEST; SpaTalk can be equally or more conservative on false positives. The Goal1 readiness gate classifies all three datasets as main-text-ready for control ranking, false-positive suppression, and representative group-level FDR support, while curated pathway/Reactome/TRRUST downstream evidence remains coverage-limited and does not support SpatialSelf downstream-superiority yet.",
            "primary_source_tables": "goal1_formal_method_summary;goal1_formal_lr_control_summary;goal1_formal_claims;goal1_formal_status;goal1_downstream_proxy_lr_summary;goal1_downstream_proxy_method_summary;goal1_downstream_proxy_claims;goal1_downstream_proxy_status;goal1_curated_downstream_results;goal1_curated_downstream_method_summary;goal1_curated_downstream_coverage;goal1_curated_downstream_claims;goal1_curated_downstream_status;goal1_biological_validation_dataset_gate;goal1_biological_validation_method_gate;goal1_biological_validation_lr_gate;goal1_biological_validation_claims;goal1_method_summary;goal1_lr_control_summary;goal1_claims",
            "evidence_status": _table_status(table_status, ["goal1_formal_method_summary", "goal1_formal_lr_control_summary", "goal1_formal_claims", "goal1_formal_status", "goal1_downstream_proxy_lr_summary", "goal1_downstream_proxy_method_summary", "goal1_downstream_proxy_claims", "goal1_downstream_proxy_status", "goal1_curated_downstream_results", "goal1_curated_downstream_method_summary", "goal1_curated_downstream_coverage", "goal1_curated_downstream_claims", "goal1_curated_downstream_status", "goal1_biological_validation_dataset_gate", "goal1_biological_validation_method_gate", "goal1_biological_validation_lr_gate", "goal1_biological_validation_claims", "goal1_method_summary", "goal1_lr_control_summary", "goal1_claims"]),
            "claim_scope": "formal three-dataset real-background injected-control validation with readiness gate, representative group-level FDR support, mixed injected and curated downstream panels, and retained smoke/full tables",
            "allowed_wording": "formal baseline-matched injected-control validation supports control ranking and high-expression false-positive suppression relative to COMMOT, CellChat, and CellNEST, with SpaTalk caveat and downstream limitation stated",
            "disallowed_wording": "validated biological ground truth in unlabeled real tissue, SpatialSelf has the lowest false-positive rate against every baseline, or SpatialSelf has demonstrated global downstream superiority",
        },
        {
            "claim_id": "C5_formal_fdr_schema",
            "main_text_role": "statistical calibration",
            "supported_statement": "Formal FDR module outputs empirical p-values, BH FDR, candidate-edge, LR-pair, cell-type-pair, and significant-result tables under spatial, receiver, and LR-identity nulls. A dedicated fixed-null CLI exposes the manuscript-facing three-null contract and writes the standard discovery tables. A 200-permutation tonsil smoke improves p-value resolution, and the representative 100-permutation real-background panel covers lymph node, tonsil, and melanoma with LR-pair direct discoveries while candidate-edge discoveries remain absent.",
            "primary_source_tables": "goal5_fdr_summary;goal5_fdr_group_summary;goal5_fdr_200perm_summary;goal5_fdr_200perm_group_summary;goal5_fixed_null_cli_smoke_summary;goal5_fixed_null_cli_smoke_group_summary;goal5_real_background_fdr_summary;goal5_real_background_fdr_group_summary;goal5_real_background_fdr_claims;goal5_real_background_fdr_status;goal5_real_background_fdr_100perm_summary;goal5_real_background_fdr_100perm_group_summary;goal5_real_background_fdr_100perm_claims;goal5_real_background_fdr_100perm_status",
            "evidence_status": _table_status(table_status, ["goal5_fdr_summary", "goal5_fdr_group_summary", "goal5_fdr_200perm_summary", "goal5_fdr_200perm_group_summary", "goal5_fixed_null_cli_smoke_summary", "goal5_fixed_null_cli_smoke_group_summary", "goal5_real_background_fdr_summary", "goal5_real_background_fdr_group_summary", "goal5_real_background_fdr_claims", "goal5_real_background_fdr_status", "goal5_real_background_fdr_100perm_summary", "goal5_real_background_fdr_100perm_group_summary", "goal5_real_background_fdr_100perm_claims", "goal5_real_background_fdr_100perm_status"]),
            "claim_scope": "FDR schema, fixed-null CLI smoke, one stronger 200-permutation smoke panel, three-dataset 50-permutation schema panel, and representative three-dataset 100-permutation real-background panel",
            "allowed_wording": "FDR output schema, fixed three-null CLI, 200-permutation single-dataset smoke, and representative three-dataset 100-permutation real-background panel are implemented",
            "disallowed_wording": "final calibrated alpha control, edge-level discovery power, or manuscript-ready 500-permutation FDR panel",
        },
        {
            "claim_id": "C6_auditability_reproducibility",
            "main_text_role": "reproducibility",
            "supported_statement": "Revised Phase 2 outputs are checked by an external audit runner; the current assembly records source-table traceability and preserves non-final evidence scope.",
            "primary_source_tables": "source_table_index;claim_evidence_matrix;figure_panel_plan;manuscript_limitations",
            "evidence_status": "source_ready",
            "claim_scope": "current revised Phase 2 smoke and assembly artifacts",
            "allowed_wording": "auditable evidence chain",
            "disallowed_wording": "final complete manuscript package",
        },
        {
            "claim_id": "C6_runtime_scalability_support",
            "main_text_role": "runtime and scalability",
            "supported_statement": "Supporting runtime evidence profiles SpatialSelf and baseline adapters on a 4,035-cell/332k-edge real-background benchmark and shows SpatialSelf tonsil subset scaling to 1,800 cells and 1.64M candidate edges with high-expression negative controls suppressed.",
            "primary_source_tables": "goal6_runtime_method_profile;goal6_runtime_highres_scaling_runs;goal6_runtime_claims",
            "evidence_status": _table_status(table_status, ["goal6_runtime_method_profile", "goal6_runtime_highres_scaling_runs", "goal6_runtime_claims"]),
            "claim_scope": "supporting runtime/scalability evidence, not final 50k/100k high-resolution benchmark",
            "allowed_wording": "supporting runtime and subset-scaling evidence",
            "disallowed_wording": "full Xenium/Visium HD scalability proven to 100k cells",
        },
    ]
    return pd.DataFrame(rows)


def _build_limitations_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "limitation_id": "L1_smoke_scale",
                "scope": "all revised Goal 1-5 outputs",
                "limitation": "Current revised evidence is smoke-scale or small-panel; full replicates and larger tiles are still needed before final manuscript claims.",
                "next_action": "Run expanded replicate panels and larger tonsil/melanoma tiles.",
            },
            {
                "limitation_id": "L2_baseline_panel_incomplete",
                "scope": "Goal 1 and Goal 2",
                "limitation": "The 18-replicate tuned baseline panel is now complete for six metric methods, with DeepTalk/NICHES status-recorded. Results are mixed: SpatialSelf wins the aggregate frozen endpoint and has positive paired mean endpoint deltas, but tissue-level winners differ and paired CIs still cross zero for SpaTalk and CellNEST.",
                "next_action": "Promote the completed replicated panel into manuscript figures/source tables, report tissue-specific winners and paired endpoint diagnostics, and keep DeepTalk/NICHES limitations explicit unless executable LR-specific outputs are obtained.",
            },
            {
                "limitation_id": "L3_downstream_proxy",
                "scope": "Goal 1",
                "limitation": "Receiver-response evidence is mixed, and the current curated pathway/Reactome/TRRUST downstream panel is target-coverage limited for several chemokine-like positive controls; it does not support a SpatialSelf downstream-superiority claim.",
                "next_action": "Add stronger NicheNet/OmniPath/Reactome target maps for revised v2 LR pairs, ensure target coverage for chemokine controls, and rerun baseline-matched downstream validation.",
            },
            {
                "limitation_id": "L4_fdr_power",
                "scope": "Goal 5",
                "limitation": "FDR evidence includes a fixed-null schema smoke, one 200-permutation tonsil smoke, and a three-dataset 100-permutation representative real-background panel; candidate-edge discoveries remain absent, and final manuscript calibration still needs broader 200-500 permutation panels.",
                "next_action": "Run 200-500 permutations across lymph node, tonsil, melanoma, and selected baselines; report group-level calibration, candidate-edge power limits, and null p-value diagnostics.",
            },
            {
                "limitation_id": "L5_no_final_figures",
                "scope": "Goal 6",
                "limitation": "This assembly creates source-ready figure exports with automated QA, but not final journal-ready manually reviewed figures.",
                "next_action": "Perform manual visual review, journal sizing, panel-label polish, and final SVG/PDF/TIFF export checks.",
            },
            {
                "limitation_id": "L6_runtime_scope",
                "scope": "Goal 6 runtime/scalability evidence",
                "limitation": "Runtime evidence combines inherited method profiling and tonsil subset scaling; it is not a final 50k/100k high-resolution full-dataset benchmark.",
                "next_action": "Run revised profiling on selected real-background datasets and add 10k/50k/100k high-resolution subsets where data are available.",
            },
        ]
    )


def _build_figure_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": "Fig1",
                "panel": "A",
                "title": "SpatialSelf revised evidence chain",
                "source_tables": "claim_evidence_matrix.csv",
                "plot_type": "schematic/table-derived workflow",
                "status": "planned",
            },
            {
                "figure": "Fig1",
                "panel": "B",
                "title": "Simulator v2 mechanism coverage",
                "source_tables": "source_tables/goal3_mechanism_summary.csv",
                "plot_type": "stacked bar or dot plot",
                "status": "source_ready",
            },
            {
                "figure": "Fig1",
                "panel": "C",
                "title": "Real-background preserved properties",
                "source_tables": "source_tables/goal4_*_validation.csv",
                "plot_type": "small multiple summary bars",
                "status": "source_ready",
            },
            {
                "figure": "Fig2",
                "panel": "A",
                "title": "FP/FDR-constrained weak-signal tradeoff across baselines",
                "source_tables": "source_tables/goal2_metrics.csv",
                "plot_type": "scatter or paired bars with adapter-light status",
                "status": "source_ready",
            },
            {
                "figure": "Fig2",
                "panel": "B",
                "title": "Baseline-matched control ranking",
                "source_tables": "source_tables/goal1_method_summary.csv;source_tables/goal1_lr_control_summary.csv",
                "plot_type": "rank plot",
                "status": "source_ready",
            },
            {
                "figure": "Fig2",
                "panel": "C",
                "title": "Formal FDR null outputs",
                "source_tables": "source_tables/goal5_fixed_null_cli_smoke_summary.csv;source_tables/goal5_fdr_summary.csv;source_tables/goal5_fdr_200perm_summary.csv;source_tables/goal5_fdr_200perm_group_summary.csv;source_tables/goal5_real_background_fdr_100perm_summary.csv;source_tables/goal5_real_background_fdr_100perm_group_summary.csv",
                "plot_type": "null/FDR summary bars",
                "status": "source_ready_fixed_null_cli_200perm_smoke_and_100perm_representative_panel",
            },
            {
                "figure": "Fig2",
                "panel": "D",
                "title": "Runtime and high-resolution subset scaling",
                "source_tables": "source_tables/goal6_runtime_method_profile.csv;source_tables/goal6_runtime_highres_scaling_runs.csv;source_tables/goal6_runtime_claims.csv",
                "plot_type": "runtime bars and scaling line",
                "status": "source_ready_supporting_runtime",
            },
        ]
    )


def _build_panel_index(copied: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(copied)
    table["ready_for_plotting"] = (table["status"] == "copied") & (table["n_rows"] > 0) & (table["n_columns"] > 0)
    return table


def _table_status(table_status: dict[str, dict], ids: list[str]) -> str:
    missing = [table_id for table_id in ids if table_status.get(table_id, {}).get("status") != "copied"]
    empty = [
        table_id
        for table_id in ids
        if table_status.get(table_id, {}).get("status") == "copied"
        and int(table_status.get(table_id, {}).get("n_rows", 0)) == 0
    ]
    if missing:
        return f"missing_source:{','.join(missing)}"
    if empty:
        return f"empty_source:{','.join(empty)}"
    return "source_ready"


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
