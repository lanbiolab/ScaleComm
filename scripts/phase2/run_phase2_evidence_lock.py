#!/usr/bin/env python
"""Lock revised Phase 2 goals to current evidence, caveats, and next actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GOAL_ROWS = [
    {
        "goal_id": "Goal 1",
        "goal_title": "Baseline-matched biological validation",
        "priority_order": 3,
        "target_conclusion": "SpatialSelf ranks biologically plausible controls ahead of high-expression negative controls in realistic tissue backgrounds.",
        "current_status": "source_ready_with_downstream_limitation",
        "source_tables": "goal1_biological_validation_claims;goal1_biological_validation_dataset_gate;goal1_biological_validation_method_gate;goal1_biological_validation_lr_gate;goal1_formal_method_summary;goal1_formal_lr_control_summary;goal1_curated_downstream_coverage",
        "evidence_files": "outputs/phase2_revised/goal1_biological_validation_readiness/goal1_biological_validation_claims.csv;outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv",
        "supported_claims": "Control ranking and high-expression false-positive suppression are supported across lymph node, tonsil, and melanoma formal real-background validations; representative group-level FDR support is available.",
        "partial_or_unsupported_claims": "Curated downstream target validation remains coverage-limited and does not support downstream-superiority claims.",
        "allowed_wording": "SpatialSelf supports baseline-matched injected-control ranking and high-expression false-positive suppression relative to COMMOT, CellChat, and CellNEST, with SpaTalk noted as conservative.",
        "disallowed_wording": "SpatialSelf proves unlabeled real-data ground truth, has the lowest false-positive rate against every baseline, or has demonstrated global downstream target superiority.",
        "remaining_next_action": "Upgrade NicheNet/OmniPath/Reactome target coverage and rerun baseline-matched downstream Spearman for lymph node, tonsil, and melanoma.",
        "manuscript_status": "main_text_ready_for_control_ranking_not_downstream_superiority",
    },
    {
        "goal_id": "Goal 2",
        "goal_title": "Weak-signal claim",
        "priority_order": 5,
        "target_conclusion": "SpatialSelf has a better weak-signal versus false-positive tradeoff, not necessarily the highest raw weak recall.",
        "current_status": "source_ready_tradeoff_supported_raw_recall_not_supported",
        "source_tables": "goal2_weak_tradeoff_claims;goal2_weak_tradeoff_method_gate;goal2_weak_tradeoff_tissue_gate;goal2_weak_tradeoff_stress_gate;goal2_weak_tradeoff_paired_gate;goal2_replicated_tuned_metric_summary;goal2_replicated_tuned_endpoint_summary",
        "evidence_files": "outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_claims.csv;outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv",
        "supported_claims": "The mechanism-stratified stress gate and aggregate frozen endpoint support an FP/FDR-guardrailed weak-signal tradeoff.",
        "partial_or_unsupported_claims": "Highest raw weak recall, lowest false-positive rate against every baseline, tissue-level dominance, and paired significance against every baseline are not supported.",
        "allowed_wording": "SpatialSelf improves the FP/FDR-guardrailed weak-signal tradeoff in the mechanism-stratified stress gate and aggregate endpoint, while tissue winners remain mixed.",
        "disallowed_wording": "SpatialSelf is globally best on every weak-signal metric, has the highest weak recall, wins every tissue, or has paired significance against every baseline.",
        "remaining_next_action": "Run the weak-signal-specific benchmark panel after freezing simulator v2 and real-background v2 schemas, reporting weak recall beside empirical FDR, top-k precision, and high-expression false-positive rate.",
        "manuscript_status": "main_text_ready_as_tradeoff_claim_not_raw_recall_claim",
    },
    {
        "goal_id": "Goal 3",
        "goal_title": "Benchmark simulator v2",
        "priority_order": 1,
        "target_conclusion": "The benchmark is mechanism-layered and no longer reducible to high expression plus short distance.",
        "current_status": "source_ready_truth_contract_validated",
        "source_tables": "goal34_truth_contract_claims;goal34_truth_contract_dataset_summary;goal34_truth_contract_replicate_summary;goal3_formal_benchmark_summary;goal34_mechanism_coverage_matrix",
        "evidence_files": "outputs/phase2_revised/goal34_truth_contract/goal34_truth_contract_claims.csv;outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv",
        "supported_claims": "Controlled simulator v2 satisfies the shared truth-table contract and covers contact, short/medium paracrine, longer-range niche, weak signals, and high-expression negative controls.",
        "partial_or_unsupported_claims": "The simulator should not be described as fully realistic biology for all CCC modes.",
        "allowed_wording": "Mechanism-layered controlled benchmark with auditable candidate, LR-pair, cell-type-pair, niche, weak/strong, and mechanism truth.",
        "disallowed_wording": "Fully realistic simulator of all CCC biology.",
        "remaining_next_action": "Keep the truth schema frozen before launching new large baseline panels and preserve the contract validator as a required gate.",
        "manuscript_status": "main_text_ready_as_benchmark_design",
    },
    {
        "goal_id": "Goal 4",
        "goal_title": "Real-background semi-synthetic benchmark",
        "priority_order": 2,
        "target_conclusion": "SpatialSelf's advantage is tested under real spatial structure and expression background, not only artificial simulation.",
        "current_status": "source_ready_truth_contract_validated",
        "source_tables": "goal34_truth_contract_claims;goal34_truth_contract_dataset_summary;goal4_replicated_tuned_real_background_summary;goal34_real_background_property_summary;goal4_formal_real_background_summary",
        "evidence_files": "outputs/phase2_revised/goal34_truth_contract/goal34_truth_contract_claims.csv;outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv",
        "supported_claims": "Validated real-background v2 replicate sets exist for lymph node, tonsil, and melanoma with preserved measured ST properties and high-expression false-positive controls kept ground-truth negative.",
        "partial_or_unsupported_claims": "These are injected-truth semi-synthetic benchmarks, not true ground truth in unlabeled real tissue.",
        "allowed_wording": "Real ST backgrounds with injected controls, measured property preservation, and validated 18-replicate tuned-baseline dataset layer.",
        "disallowed_wording": "True unlabeled real-data ground truth.",
        "remaining_next_action": "Use the validated real-background replicates as the common substrate for expanded baseline-matched validation and weak-signal panels.",
        "manuscript_status": "main_text_ready_as_real_background_benchmark",
    },
    {
        "goal_id": "Goal 5",
        "goal_title": "Formal FDR and calibration module",
        "priority_order": 4,
        "target_conclusion": "SpatialSelf reports statistically calibrated CCC discoveries rather than scores only.",
        "current_status": "source_ready_schema_and_100perm_representative_panel_not_final_500perm",
        "source_tables": "goal5_real_background_fdr_100perm_claims;goal5_real_background_fdr_100perm_summary;goal5_real_background_fdr_100perm_group_summary;goal5_fixed_null_cli_smoke_summary;goal5_fdr_200perm_summary",
        "evidence_files": "outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_claims.csv;outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv",
        "supported_claims": "The fixed three-null FDR module outputs empirical p-values, BH FDR, significant LR pairs, cell-type pairs, and spatial-edge tables; representative 100-permutation panels support LR-pair-level discoveries.",
        "partial_or_unsupported_claims": "Final 500-permutation calibration and candidate-edge discovery power are not established.",
        "allowed_wording": "FDR output schema, fixed three-null CLI, 200-permutation single-dataset smoke, and representative three-dataset 100-permutation real-background panel are implemented.",
        "disallowed_wording": "Final calibrated alpha control, edge-level discovery power, or manuscript-ready 500-permutation FDR panel.",
        "remaining_next_action": "Run selected 200-500 permutation panels and report null p-value diagnostics, group-level calibration, and candidate-edge power limits.",
        "manuscript_status": "main_text_ready_as_framework_schema_and_representative_panel_not_final_calibration",
    },
    {
        "goal_id": "Goal 6",
        "goal_title": "Manuscript evidence and figure convergence",
        "priority_order": 6,
        "target_conclusion": "SpatialSelf is presented as a calibrated, LR-aware, weak-signal-sensitive CCC inference framework with auditable source tables.",
        "current_status": "source_ready_not_manuscript_final",
        "source_tables": "claim_evidence_matrix;source_table_index;manuscript_results_outline;manuscript_gap_priority;manuscript_wording_guardrails;manuscript_readiness_summary",
        "evidence_files": "outputs/phase2_revised/goal6_manuscript_assembly/claim_evidence_matrix.csv;outputs/phase2_revised/goal6_manuscript_assembly/source_table_index.csv;outputs/phase2_revised/goal6_manuscript_assembly/manuscript_readiness_summary.csv",
        "supported_claims": "The evidence chain is auditable, source-table based, and guarded against unsupported wording.",
        "partial_or_unsupported_claims": "The manuscript is not final; journal-ready figure exports, stronger downstream validation, final FDR calibration, and larger high-resolution runtime evidence remain open.",
        "allowed_wording": "Auditable source-table and figure-scaffold evidence chain.",
        "disallowed_wording": "Final complete manuscript package or full Xenium/Visium HD scalability proven to 100k cells.",
        "remaining_next_action": "Generate final figure source panels from locked tables, QA them, and keep source tables synchronized with claim guardrails.",
        "manuscript_status": "source_table_scaffold_ready_not_final_figures",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default="outputs/phase2_revised/goal6_manuscript_assembly")
    args = parser.parse_args()

    assembly_dir = PROJECT_ROOT / args.assembly_dir
    assembly_dir.mkdir(parents=True, exist_ok=True)

    claim_matrix = _read_csv(assembly_dir / "claim_evidence_matrix.csv")
    source_index = _read_csv(assembly_dir / "source_table_index.csv")
    gap_table = _read_csv(assembly_dir / "manuscript_gap_priority.csv")
    readiness = _read_csv(assembly_dir / "manuscript_readiness_summary.csv")

    goal_lock = _goal_lock(source_index)
    guardrail_lock = _guardrail_lock(claim_matrix)
    remaining_actions = _remaining_actions(goal_lock, gap_table)
    completion_matrix = _completion_matrix(goal_lock, readiness)

    goal_lock.to_csv(assembly_dir / "phase2_goal_evidence_lock.csv", index=False)
    guardrail_lock.to_csv(assembly_dir / "phase2_claim_guardrail_lock.csv", index=False)
    remaining_actions.to_csv(assembly_dir / "phase2_remaining_actions.csv", index=False)
    completion_matrix.to_csv(assembly_dir / "phase2_goal_completion_matrix.csv", index=False)

    manifest = {
        "analysis_label": "phase2_six_goal_evidence_lock",
        "assembly_dir": str(assembly_dir.relative_to(PROJECT_ROOT)),
        "canonical_goal_count": int(len(goal_lock)),
        "canonical_goals": goal_lock["goal_id"].tolist(),
        "claim_level": "phase2_goal_lock_not_phase2_completion",
        "outputs": {
            "goal_evidence_lock": str((assembly_dir / "phase2_goal_evidence_lock.csv").relative_to(PROJECT_ROOT)),
            "claim_guardrail_lock": str((assembly_dir / "phase2_claim_guardrail_lock.csv").relative_to(PROJECT_ROOT)),
            "remaining_actions": str((assembly_dir / "phase2_remaining_actions.csv").relative_to(PROJECT_ROOT)),
            "goal_completion_matrix": str((assembly_dir / "phase2_goal_completion_matrix.csv").relative_to(PROJECT_ROOT)),
        },
        "readiness_context": _readiness_context(readiness),
    }
    (assembly_dir / "phase2_evidence_lock_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(goal_lock[["goal_id", "goal_title", "current_status", "manuscript_status"]].to_string(index=False))
    print(f"Wrote Phase 2 evidence lock outputs to {assembly_dir}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _goal_lock(source_index: pd.DataFrame) -> pd.DataFrame:
    source_lookup = {}
    if not source_index.empty and "table_id" in source_index:
        source_lookup = {
            str(row.table_id): {
                "status": str(row.status),
                "n_rows": int(row.n_rows),
                "ready_for_plotting": bool(row.ready_for_plotting),
            }
            for row in source_index.itertuples(index=False)
        }

    rows = []
    for row in GOAL_ROWS:
        table_ids = [item for item in row["source_tables"].split(";") if item]
        present = [table_id for table_id in table_ids if table_id in source_lookup]
        missing = [table_id for table_id in table_ids if table_id not in source_lookup and not table_id.startswith("manuscript_") and table_id not in {"claim_evidence_matrix", "source_table_index"}]
        rows.append(
            {
                **row,
                "n_declared_source_tables": len(table_ids),
                "n_assembled_source_tables_present": len(present),
                "missing_assembled_source_tables": ";".join(missing),
                "source_table_gate": "pass" if not missing else "partial",
            }
        )
    return pd.DataFrame(rows)


def _guardrail_lock(claim_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not claim_matrix.empty:
        for row in claim_matrix.itertuples(index=False):
            rows.append(
                {
                    "source": "claim_evidence_matrix",
                    "scope": getattr(row, "claim_id", ""),
                    "status": getattr(row, "evidence_status", ""),
                    "allowed_wording": getattr(row, "allowed_wording", ""),
                    "disallowed_wording": getattr(row, "disallowed_wording", ""),
                    "reason": getattr(row, "claim_scope", ""),
                }
            )

    for row in GOAL_ROWS:
        rows.append(
            {
                "source": "phase2_goal_lock",
                "scope": row["goal_id"],
                "status": row["current_status"],
                "allowed_wording": row["allowed_wording"],
                "disallowed_wording": row["disallowed_wording"],
                "reason": row["partial_or_unsupported_claims"],
            }
        )
    return pd.DataFrame(rows)


def _remaining_actions(goal_lock: pd.DataFrame, gap_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in goal_lock.itertuples(index=False):
        rows.append(
            {
                "scope": row.goal_id,
                "priority_order": row.priority_order,
                "severity": "major" if row.goal_id in {"Goal 1", "Goal 2", "Goal 5"} else "moderate",
                "remaining_next_action": row.remaining_next_action,
                "blocks_final_manuscript": row.goal_id in {"Goal 1", "Goal 2", "Goal 5", "Goal 6"},
            }
        )
    if not gap_table.empty:
        for row in gap_table.itertuples(index=False):
            rows.append(
                {
                    "scope": getattr(row, "limitation_id", ""),
                    "priority_order": int(getattr(row, "priority_rank", 99)),
                    "severity": getattr(row, "severity", "moderate"),
                    "remaining_next_action": getattr(row, "next_action", ""),
                    "blocks_final_manuscript": bool(getattr(row, "blocks_final_manuscript_claim", True)),
                }
            )
    return pd.DataFrame(rows).sort_values(["priority_order", "scope"]).reset_index(drop=True)


def _completion_matrix(goal_lock: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    manuscript_final_ready = False
    if not readiness.empty and "manuscript_final_ready" in readiness:
        manuscript_final_ready = bool(readiness.iloc[0]["manuscript_final_ready"])

    rows = []
    for row in goal_lock.itertuples(index=False):
        phase2_goal_locked = row.source_table_gate == "pass"
        evidence_supports_target = "unsupported" not in str(row.current_status).split("_")
        rows.append(
            {
                "goal_id": row.goal_id,
                "goal_title": row.goal_title,
                "phase2_goal_locked": phase2_goal_locked,
                "evidence_supports_limited_target": evidence_supports_target,
                "manuscript_final_ready": manuscript_final_ready,
                "completion_interpretation": "goal_definition_and_current_evidence_locked_not_experimentally_complete",
            }
        )
    return pd.DataFrame(rows)


def _readiness_context(readiness: pd.DataFrame) -> dict:
    if readiness.empty:
        return {}
    return readiness.iloc[0].to_dict()


if __name__ == "__main__":
    main()
