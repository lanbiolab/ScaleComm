#!/usr/bin/env python
"""Build Phase 5 Methods support tables for LR universe and baseline status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETAINED_BASELINES = ["commot", "cellchat", "spatalk", "cellnest"]
STATUS_METHODS = ["deeptalk", "niches"]
EXCLUDED_METHODS = ["lr_coexp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase5/manuscript_submission_lock/goal5_methods_support")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_rows = _build_lr_manifest()
    baseline_rows = _build_baseline_status()
    claim_rows = _build_claims(lr_rows, baseline_rows)

    _write_csv(output_dir / "phase5_lr_database_manifest.csv", lr_rows, lr_rows[0].keys())
    _write_csv(output_dir / "phase5_baseline_status_table.csv", baseline_rows, baseline_rows[0].keys())
    _write_csv(output_dir / "phase5_methods_support_claims.csv", claim_rows, claim_rows[0].keys())
    manifest = {
        "lr_database_manifest": _rel(output_dir / "phase5_lr_database_manifest.csv"),
        "baseline_status_table": _rel(output_dir / "phase5_baseline_status_table.csv"),
        "methods_support_claims": _rel(output_dir / "phase5_methods_support_claims.csv"),
        "claim_level": "phase5_methods_support_tables_for_manuscript",
    }
    (output_dir / "phase5_methods_support_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _build_lr_manifest() -> list[dict[str, Any]]:
    sources = [
        (
            "controlled_simulator_v2_5rep",
            PROJECT_ROOT / "outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep",
            "benchmark_internal_injected_lr_universe",
        ),
        (
            "real_background_v2_5rep",
            PROJECT_ROOT / "outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep",
            "benchmark_internal_injected_lr_universe",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source_id, root, source_type in sources:
        files = sorted(root.glob("**/lr_pairs.csv"))
        all_pairs: set[tuple[str, str, str]] = set()
        mechanisms: set[str] = set()
        weak_count = 0
        high_fp_count = 0
        for path in files:
            for row in _read_csv(path):
                ligand = row.get("ligand") or row.get("source") or row.get("Ligand") or ""
                receptor = row.get("receptor") or row.get("target") or row.get("Receptor") or ""
                lr_id = row.get("lr_id") or row.get("pair_id") or f"{ligand}_{receptor}"
                if ligand or receptor or lr_id:
                    all_pairs.add((lr_id, ligand, receptor))
                mechanism = row.get("mechanism", "")
                if mechanism:
                    mechanisms.add(mechanism)
                if _truthy(row.get("weak_signal")):
                    weak_count += 1
                if _truthy(row.get("false_positive_high_expression")):
                    high_fp_count += 1
        rows.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "root": _rel(root),
                "n_lr_pair_files": len(files),
                "n_unique_lr_pairs": len(all_pairs),
                "mechanisms": "|".join(sorted(mechanisms)),
                "weak_signal_entries": weak_count,
                "high_expression_false_positive_entries": high_fp_count,
                "complex_handling": "no_multisubunit_complex_resolution_in_current_benchmark_manifest",
                "absent_gene_handling": "filtered_by_expression_matrix_in_method_adapters_when_applicable",
                "candidate_universe_policy": "common_benchmark_lr_universe_for_methods_when_adapter_supports_projection",
                "limitation": "This manifest locks benchmark LR universes; it is not a final external LR database version manifest.",
            }
        )

    cellnest_files = sorted((PROJECT_ROOT / "outputs").glob("**/cellnest_lr_database.csv"))
    unique_cellnest: set[tuple[str, str]] = set()
    for path in cellnest_files:
        for row in _read_csv(path):
            ligand = row.get("ligand") or row.get("Ligand") or row.get("from") or ""
            receptor = row.get("receptor") or row.get("Receptor") or row.get("to") or ""
            if ligand or receptor:
                unique_cellnest.add((ligand, receptor))
    rows.append(
        {
            "source_id": "cellnest_adapter_lr_database_exports",
            "source_type": "baseline_adapter_export",
            "root": "outputs/**/cellnest_lr_database.csv",
            "n_lr_pair_files": len(cellnest_files),
            "n_unique_lr_pairs": len(unique_cellnest),
            "mechanisms": "",
            "weak_signal_entries": "",
            "high_expression_false_positive_entries": "",
            "complex_handling": "exported_pair_level_adapter_database",
            "absent_gene_handling": "adapter_export_after_dataset_filtering",
            "candidate_universe_policy": "used_for_cellnest_adapter_inputs",
            "limitation": "Adapter export confirms CellNEST input LR rows but does not replace the shared benchmark LR manifest.",
        }
    )
    return rows


def _build_baseline_status() -> list[dict[str, Any]]:
    policy = PROJECT_ROOT / "configs/baselines/phase1_baselines.yaml"
    status_files = {
        "replicated_tuned_panel": PROJECT_ROOT / "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_metric_summary.csv",
        "full_baseline_all": PROJECT_ROOT / "outputs/phase2_revised/goal2_full_baseline_panel_all/goal2_full_baseline_status.csv",
        "formal_tuned_small": PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_tuned_small_panel/method_runs",
        "tonsil_smoke": PROJECT_ROOT / "outputs/phase2_revised/goal2_full_baseline_panel/tonsil_real_background_smoke/baseline_status.csv",
    }
    metric_rows = _read_csv(status_files["replicated_tuned_panel"])
    methods_in_metrics = {row.get("method", "") for row in metric_rows}
    counts = {
        row.get("method", ""): max(
            _int(row.get("aupr_count")),
            _int(row.get("auroc_count")),
            _int(row.get("weak_signal_recall_at_threshold_count")),
        )
        for row in metric_rows
    }
    rows: list[dict[str, Any]] = []
    for method in ["spatialself_wavelet_edge_prior", "spatialself_wavelet", *RETAINED_BASELINES]:
        rows.append(
            {
                "method": method,
                "role": "representative_method" if method == "spatialself_wavelet_edge_prior" else ("spatialself_variant" if method == "spatialself_wavelet" else "retained_baseline"),
                "formal_metric_panel_status": "included" if method in methods_in_metrics else "not_included",
                "formal_metric_count": counts.get(method, 0),
                "output_level": _output_level(method),
                "adapter_status": _adapter_status(method),
                "fairness_policy": "common_metrics_and_common_candidate_universe_when_method_adapter_supports_projection",
                "source_evidence": _rel(status_files["replicated_tuned_panel"]),
                "limitation": _method_limitation(method),
            }
        )
    for method in STATUS_METHODS:
        rows.append(
            {
                "method": method,
                "role": "status_method",
                "formal_metric_panel_status": "status_only",
                "formal_metric_count": 0,
                "output_level": _output_level(method),
                "adapter_status": _adapter_status(method),
                "fairness_policy": "record_status_or_failure_reason_instead_of_forcing_incompatible_metric",
                "source_evidence": _rel(policy),
                "limitation": _method_limitation(method),
            }
        )
    for method in EXCLUDED_METHODS:
        rows.append(
            {
                "method": method,
                "role": "excluded_method",
                "formal_metric_panel_status": "excluded",
                "formal_metric_count": 0,
                "output_level": "not_applicable",
                "adapter_status": "excluded_by_phase5_policy",
                "fairness_policy": "excluded_from_phase5_retained_baselines",
                "source_evidence": _rel(PROJECT_ROOT / "configs/experiments/phase5_submission_manuscript_lock.yaml"),
                "limitation": "Removed by user decision and Phase 5 retained-baseline policy.",
            }
        )
    return rows


def _build_claims(lr_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal_methods = [row for row in baseline_rows if row["formal_metric_panel_status"] == "included"]
    status_methods = [row for row in baseline_rows if row["role"] == "status_method"]
    excluded = [row for row in baseline_rows if row["role"] == "excluded_method"]
    benchmark_lr_rows = [row for row in lr_rows if row["source_type"] == "benchmark_internal_injected_lr_universe"]
    return [
        {
            "claim_id": "P5_M1_lr_universe_locked_for_benchmarks",
            "status": "supported" if benchmark_lr_rows and all(_int(row["n_unique_lr_pairs"]) > 0 for row in benchmark_lr_rows) else "partial",
            "allowed_wording": "Benchmark LR universes are locked as benchmark-internal injected LR sets with explicit mechanism labels and false-positive controls.",
            "disallowed_wording": "The current manifest is a final external LR database version with complete complex handling.",
            "evidence_value": "|".join(f"{row['source_id']}:{row['n_unique_lr_pairs']}" for row in benchmark_lr_rows),
            "limitation": "External LR database version and complex handling still need final manuscript locking.",
        },
        {
            "claim_id": "P5_M2_baseline_status_locked",
            "status": "supported" if len(formal_methods) >= 6 and status_methods and excluded else "partial",
            "allowed_wording": "COMMOT, CellChat, SpaTalk, and CellNEST are retained baselines; DeepTalk and NICHES are status methods; LR-Coexp is excluded.",
            "disallowed_wording": "Every possible baseline is fully optimized and directly comparable at edge level.",
            "evidence_value": f"formal_methods={len(formal_methods)};status_methods={len(status_methods)};excluded={len(excluded)}",
            "limitation": "Some adapters project cell-type-level outputs to candidate edges; CellNEST and status methods retain implementation caveats.",
        },
        {
            "claim_id": "P5_M3_reproducibility_support",
            "status": "supported",
            "allowed_wording": "Phase 5 provides source tables and commands for evidence index, validation summaries, and real-validation figures.",
            "disallowed_wording": "The repository already contains a complete one-command submission reproduction workflow for all manuscript figures.",
            "evidence_value": "phase5_methods_support_tables_generated",
            "limitation": "A final one-command tutorial remains to be assembled after manuscript figure polishing.",
        },
    ]


def _output_level(method: str) -> str:
    return {
        "spatialself_wavelet_edge_prior": "candidate_edge",
        "spatialself_wavelet": "candidate_edge",
        "commot": "candidate_edge",
        "cellchat": "cell_type_pair_projected_to_candidate_edge",
        "spatalk": "cell_type_pair_projected_to_candidate_edge",
        "cellnest": "candidate_edge",
        "deeptalk": "candidate_edge_when_predict_ccc_exists",
        "niches": "blocked_or_not_formal_candidate_edge",
    }.get(method, "")


def _adapter_status(method: str) -> str:
    return {
        "spatialself_wavelet_edge_prior": "formal_replicated_metric_panel",
        "spatialself_wavelet": "formal_replicated_metric_panel",
        "commot": "formal_replicated_metric_panel",
        "cellchat": "formal_replicated_metric_panel_with_projection_adapter",
        "spatalk": "formal_replicated_metric_panel_with_projection_adapter",
        "cellnest": "formal_replicated_metric_panel_tuned_small_caveat",
        "deeptalk": "projection_smoke_tested_model_required_status_only",
        "niches": "optional_blocked_dependency_status_only",
    }.get(method, "")


def _method_limitation(method: str) -> str:
    return {
        "spatialself_wavelet_edge_prior": "Representative SpatialSelf method; claims remain metric-specific.",
        "spatialself_wavelet": "SpatialSelf variant, not the representative edge-prior method.",
        "commot": "Retained baseline with formal metric rows.",
        "cellchat": "Cell-type/spatial output is projected back to candidate edges.",
        "spatalk": "Adapter uses known benchmark cell types and projects cell-type-pair LR scores.",
        "cellnest": "Deep-learning baseline included, but current evidence keeps tuned-small/status caveats.",
        "deeptalk": "Status-only because LR-specific trained/pretrained model requirements prevent formal replicated metric inclusion.",
        "niches": "Status-only because local R dependency chain was not reproducible.",
    }.get(method, "")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.is_dir():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Any) -> None:
    columns = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
