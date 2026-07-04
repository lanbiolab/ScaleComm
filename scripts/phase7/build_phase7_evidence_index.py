#!/usr/bin/env python
"""Build the Phase 7 evidence index scaffold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_COLUMNS = [
    "goal_id",
    "goal_name",
    "status",
    "required_outputs",
    "current_evidence",
    "completion_gate",
    "supported_claim_if_complete",
    "limitation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(config)
    index_path = output_dir / "phase7_evidence_index.csv"
    _write_csv(index_path, rows)

    summary = {
        "config": _rel(config_path),
        "output_dir": _rel(output_dir),
        "n_goals": len(rows),
        "status_counts": _counts(row["status"] for row in rows),
        "evidence_index": _rel(index_path),
    }
    summary_path = output_dir / "phase7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _build_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    goals = config["goals"]
    return [
        {
            "goal_id": "goal1_downstream_target_validation",
            "goal_name": "Downstream target validation",
            "status": _goal1_status(goals["goal1_downstream_target_validation"]["output_dir"]),
            "required_outputs": ";".join(
                [
                    "phase7_downstream_target_signature_table.csv",
                    "phase7_lr_target_response_correlation.csv",
                    "phase7_method_target_response_comparison.csv",
                    "phase7_downstream_fdr.csv",
                ]
            ),
            "current_evidence": _rel(PROJECT_ROOT / goals["goal1_downstream_target_validation"]["output_dir"]),
            "completion_gate": "At least one lymph-node and one non-lymph-node LR have target-signature validation, method comparison, and FDR.",
            "supported_claim_if_complete": "SpatialSelf-ranked LR activity is associated with receiver downstream target activity in selected tissue backgrounds.",
            "limitation": "Target signatures are curated-prior validation, not perturbation evidence.",
        },
        {
            "goal_id": "goal2_deep_baseline_closure",
            "goal_name": "DeepTalk and deep baseline closure",
            "status": _status_for_required_outputs(
                goals["goal2_deep_baseline_closure"]["output_dir"],
                ["phase7_deep_baseline_status_table.csv"],
            ),
            "required_outputs": "phase7_deep_baseline_status_table.csv;optional_phase7_deeptalk_formal_run_summary.csv",
            "current_evidence": _rel(PROJECT_ROOT / goals["goal2_deep_baseline_closure"]["output_dir"]),
            "completion_gate": "DeepTalk formal run or reproducible failure/status table; HoloNet/stMLnet/CytoSignal/NICHES status entries.",
            "supported_claim_if_complete": "Deep spatial CCC baselines were formally evaluated or audited with reproducible failure/status reasons.",
            "limitation": "Failed or incompatible baselines must not be described as inferior.",
        },
        {
            "goal_id": "goal3_benchmark_realism_figure",
            "goal_name": "Benchmark realism figure",
            "status": _status_for_required_outputs(
                goals["goal3_benchmark_realism_figure"]["output_dir"],
                [
                    "phase7_benchmark_realism_metrics.csv",
                    "phase7_benchmark_realism_summary.csv",
                ],
            ),
            "required_outputs": "phase7_benchmark_realism_metrics.csv;phase7_benchmark_realism_summary.csv;fig_phase7_benchmark_realism.pdf/svg/tiff",
            "current_evidence": _rel(PROJECT_ROOT / goals["goal3_benchmark_realism_figure"]["output_dir"]),
            "completion_gate": "Real vs semi-synthetic statistics for lymph node, tonsil, and melanoma where available, with panel source tables.",
            "supported_claim_if_complete": "Semi-synthetic benchmarks preserve major spatial and expression properties of real ST backgrounds.",
            "limitation": "Injected truth is not native real-tissue ground truth.",
        },
        {
            "goal_id": "goal4_cross_background_spatialself_specific_discovery",
            "goal_name": "Cross-background SpatialSelf-specific discovery",
            "status": _goal4_status(
                goals["goal4_cross_background_spatialself_specific_discovery"]["output_dir"],
                [
                    "phase7_cross_background_spatialself_only_candidates.csv",
                    "phase7_cross_background_candidate_validation.csv",
                    "phase7_cross_background_candidate_fdr.csv",
                ],
            ),
            "required_outputs": "phase7_cross_background_spatialself_only_candidates.csv;phase7_cross_background_candidate_validation.csv;phase7_cross_background_candidate_fdr.csv",
            "current_evidence": _rel(PROJECT_ROOT / goals["goal4_cross_background_spatialself_specific_discovery"]["output_dir"]),
            "completion_gate": "At least one tonsil or melanoma strict candidate passes ranking, weak/context, validation, FDR, and controls, or negative evidence is recorded.",
            "supported_claim_if_complete": "SpatialSelf-specific weak/context LR discovery is not limited to the lymph-node expanded benchmark.",
            "limitation": "A second benchmark-supported case is not a universal native-tissue discovery claim.",
        },
        {
            "goal_id": "goal5_cross_platform_real_data",
            "goal_name": "Cross-platform real data",
            "status": _goal5_status(
                goals["goal5_cross_platform_real_data"]["output_dir"],
                [
                    "phase7_cross_platform_dataset_manifest.csv",
                    "phase7_cross_platform_validation_summary.csv",
                ],
            ),
            "required_outputs": "phase7_cross_platform_dataset_manifest.csv;phase7_cross_platform_control_panel.csv;phase7_cross_platform_validation_summary.csv;phase7_cross_platform_runtime_summary.csv",
            "current_evidence": _rel(PROJECT_ROOT / goals["goal5_cross_platform_real_data"]["output_dir"]),
            "completion_gate": "At least one new platform has manifest, predictions, validation summary, runtime, and source tables.",
            "supported_claim_if_complete": "SpatialSelf can be evaluated on additional spatial platforms with biologically plausible CCC profiles.",
            "limitation": "Real-platform validation relies on controls/enrichment/target response/FDR rather than complete native ground truth.",
        },
    ]


def _status_for_required_outputs(output_dir: str, required_files: list[str]) -> str:
    path = PROJECT_ROOT / output_dir
    if not path.exists():
        return "not_started"
    existing = [filename for filename in required_files if (path / filename).exists()]
    if len(existing) == len(required_files):
        return "complete_pending_audit"
    if existing:
        return "in_progress"
    return "not_started"


def _goal1_status(output_dir: str) -> str:
    path = PROJECT_ROOT / output_dir
    required = [
        "phase7_downstream_target_signature_table.csv",
        "phase7_lr_target_response_correlation.csv",
        "phase7_method_target_response_comparison.csv",
        "phase7_downstream_fdr.csv",
    ]
    base_status = _status_for_required_outputs(output_dir, required)
    if base_status == "not_started":
        return base_status
    summary_path = path / "phase7_goal1_summary.json"
    if not summary_path.exists():
        return "in_progress"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "in_progress"
    if summary.get("has_lymph_node_target_validation") and summary.get("has_non_lymph_node_target_validation"):
        return "complete_pending_audit"
    return "in_progress"


def _goal4_status(output_dir: str, required_files: list[str]) -> str:
    base_status = _status_for_required_outputs(output_dir, required_files)
    targeted_audit = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv"
    )
    if targeted_audit.exists() and _targeted_audit_has_strict_success(targeted_audit):
        return "complete_pending_audit"
    expanded_manifests = [
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_tonsil_subset_screen/phase7_goal4_expanded_tonsil_manifest.json",
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_expanded_melanoma_manifest.json",
    ]
    partial_candidate_found = False
    for expanded_manifest in expanded_manifests:
        if not expanded_manifest.exists():
            continue
        try:
            expanded = json.loads(expanded_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if expanded.get("strict_success"):
            return "complete_pending_audit"
        if expanded.get("available_baseline_candidate_found"):
            partial_candidate_found = True
    if partial_candidate_found:
        return "partial_candidate_pending_functional_or_final_fdr"
    if base_status != "complete_pending_audit":
        return base_status
    manifest_path = PROJECT_ROOT / output_dir / "phase7_goal4_manifest.json"
    if not manifest_path.exists():
        return base_status
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "in_progress"
    if manifest.get("strict_success"):
        return "complete_pending_audit"
    if manifest.get("negative_evidence_recorded"):
        return "complete_negative_evidence"
    return base_status


def _targeted_audit_has_strict_success(path: Path) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    frame = pd.read_csv(path)
    if frame.empty or "strict_goal4_success" not in frame:
        return False
    return bool(frame["strict_goal4_success"].fillna(False).astype(bool).any())


def _goal5_status(output_dir: str, required_files: list[str]) -> str:
    requirement_status = _goal5_status_from_requirement_audit()
    if requirement_status:
        return requirement_status
    base_status = _status_for_required_outputs(output_dir, required_files)
    if base_status != "complete_pending_audit":
        return base_status
    manifest_path = PROJECT_ROOT / output_dir / "phase7_goal5_manifest.json"
    if not manifest_path.exists():
        return base_status
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "in_progress"
    if int(manifest.get("priority_platforms_available", 0)) > 0:
        if _goal5_has_completed_platform_validation(output_dir):
            return "complete_pending_audit"
        if _goal5_has_mixed_platform_validation(output_dir):
            return "in_progress_mixed_platform_evidence"
        return "in_progress_standardized_platform_ready"
    if int(manifest.get("non_priority_high_resolution_runs", 0)) > 0:
        return "partial_with_feasibility_gap"
    return "not_started"


def _goal5_status_from_requirement_audit() -> str:
    audit_path = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/phase7_requirement_audit.csv"
    if not audit_path.exists():
        return ""
    try:
        import pandas as pd
    except ImportError:
        return ""
    audit = pd.read_csv(audit_path)
    if audit.empty or "goal_id" not in audit or "status" not in audit:
        return ""
    goal5 = audit[audit["goal_id"].astype(str).eq("goal5_cross_platform_real_data")]
    if goal5.empty:
        return ""
    if goal5["status"].astype(str).eq("complete").all():
        return "complete_pending_audit"
    if goal5["status"].astype(str).eq("mixed").any():
        return "in_progress_mixed_platform_evidence"
    return ""


def _goal5_has_completed_platform_validation(output_dir: str) -> bool:
    validation_path = PROJECT_ROOT / output_dir / "phase7_cross_platform_validation_summary.csv"
    if not validation_path.exists():
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    frame = pd.read_csv(validation_path)
    if frame.empty:
        return False
    statuses = set(frame.get("validation_status", pd.Series(dtype=str)).astype(str))
    return bool(statuses & {"complete_pending_audit", "final_supported", "completed_with_fdr"})


def _goal5_has_mixed_platform_validation(output_dir: str) -> bool:
    validation_path = PROJECT_ROOT / output_dir / "phase7_cross_platform_validation_summary.csv"
    if not validation_path.exists():
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    frame = pd.read_csv(validation_path)
    if frame.empty:
        return False
    statuses = set(frame.get("validation_status", pd.Series(dtype=str)).astype(str))
    return bool(
        statuses
        & {
            "five_method_panel_with_mixed_downstream_and_conservative_prediction_fdr",
            "five_method_panel_with_mixed_downstream_control_risk",
            "model_run_with_mixed_downstream_control_risk",
            "calibrated_spatialself_candidate_pending_downstream_fdr",
            "functional_support_negative_suppressed_but_fdr_no_recovery",
            "high_resolution_functional_support_negative_suppressed_but_fdr_no_recovery",
        }
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
