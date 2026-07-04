#!/usr/bin/env python
"""Audit Phase 3 end-to-end manuscript pipeline scaffolding."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AuditRow:
    goal_id: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase3_end_to_end.yaml")
    parser.add_argument("--output-dir", default="outputs/phase3/end_to_end/audit")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path) if config_path.exists() else {}
    rows: list[AuditRow] = []
    rows.extend(_audit_docs_and_config(config_path, config))
    rows.extend(_audit_manifest(config))
    rows.extend(_audit_pipeline_contract(config))

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase3_audit_summary.csv"
    json_path = out_dir / "phase3_audit_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal_id", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    print(f"Phase 3 audit CSV:  {csv_path}")
    print(f"Phase 3 audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("\nNon-complete rows:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 3 config audit.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _audit_docs_and_config(config_path: Path, config: dict[str, Any]) -> list[AuditRow]:
    docs_path = PROJECT_ROOT / "docs/PHASE3_END_TO_END_PIPELINE.md"
    manifest_script = PROJECT_ROOT / "scripts/phase3/build_phase3_dataset_manifest.py"
    prediction_script = PROJECT_ROOT / "scripts/phase3/build_phase3_prediction_index.py"
    metric_script = PROJECT_ROOT / "scripts/phase3/build_phase3_metric_index.py"
    fdr_script = PROJECT_ROOT / "scripts/phase3/build_phase3_fdr_index.py"
    bio_script = PROJECT_ROOT / "scripts/phase3/build_phase3_biological_validation_index.py"
    manuscript_script = PROJECT_ROOT / "scripts/phase3/build_phase3_manuscript_index.py"
    required_goals = {
        "goal1_dataset_manifest",
        "goal2_main_model_and_baselines",
        "goal3_unified_evaluation",
        "goal4_fdr_permutation",
        "goal5_biological_validation",
        "goal6_manuscript_tables_figures",
    }
    goals = set(config.get("phase3_goals") or [])
    text = docs_path.read_text(encoding="utf-8") if docs_path.exists() else ""
    return [
        AuditRow("phase3", "Phase 3 end-to-end plan document exists", _exists(docs_path), _rel(docs_path)),
        AuditRow("phase3", "Phase 3 config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase3_goal1_dataset_manifest", "dataset manifest builder exists", _exists(manifest_script), _rel(manifest_script)),
        AuditRow("phase3_goal2_main_model_and_baselines", "prediction index builder exists", _exists(prediction_script), _rel(prediction_script)),
        AuditRow("phase3_goal3_unified_evaluation", "metric index builder exists", _exists(metric_script), _rel(metric_script)),
        AuditRow("phase3_goal4_fdr_permutation", "FDR index builder exists", _exists(fdr_script), _rel(fdr_script)),
        AuditRow("phase3_goal5_biological_validation", "biological validation index builder exists", _exists(bio_script), _rel(bio_script)),
        AuditRow("phase3_goal6_manuscript_tables_figures", "manuscript index builder exists", _exists(manuscript_script), _rel(manuscript_script)),
        AuditRow(
            "phase3",
            "config declares exactly the six Phase 3 goals",
            "complete" if goals == required_goals else "partial",
            _rel(config_path),
            f"goals={sorted(goals)}",
        ),
        AuditRow(
            "phase3",
            "document states end-to-end pipeline rather than model optimization",
            "complete" if "not a model-optimization phase" in text and "dataset manifest -> predictions" in text else "partial",
            _rel(docs_path),
        ),
    ]


def _audit_manifest(config: dict[str, Any]) -> list[AuditRow]:
    manifest_cfg = config.get("manifest") or {}
    csv_path = PROJECT_ROOT / manifest_cfg.get("output_csv", "outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    json_path = PROJECT_ROOT / manifest_cfg.get("output_json", "outputs/phase3/end_to_end/phase3_dataset_manifest.json")
    summary_path = PROJECT_ROOT / config.get("output_dir", "outputs/phase3/end_to_end") / "phase3_dataset_manifest_summary.json"
    rows = [
        AuditRow("phase3_goal1_dataset_manifest", "dataset manifest CSV exists", _exists(csv_path), _rel(csv_path)),
        AuditRow("phase3_goal1_dataset_manifest", "dataset manifest JSON exists", _exists(json_path), _rel(json_path)),
        AuditRow("phase3_goal1_dataset_manifest", "dataset manifest summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not csv_path.exists():
        return rows

    table = _read_csv(csv_path)
    required_columns = set(manifest_cfg.get("required_columns") or [])
    columns = set(table[0].keys()) if table else set()
    dataset_ids = {row["dataset_id"] for row in table}
    required_dataset_ids = {
        "controlled_simulator_v2",
        "real_background_lymph_node",
        "real_background_tonsil",
        "real_background_melanoma",
        "high_resolution_tonsil_subset",
    }
    optional_ids = {"optional_xenium_subset", "optional_visium_hd_subset"}
    required_rows = [row for row in table if _truthy(row.get("required"))]
    truth_required_rows = [
        row
        for row in required_rows
        if row.get("dataset_type") in {"controlled_simulator_v2", "real_background_semisynthetic"}
    ]
    optional_rows = [row for row in table if not _truthy(row.get("required"))]
    rows.extend(
        [
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "manifest includes required columns",
                "complete" if required_columns.issubset(columns) else "partial",
                _rel(csv_path),
                f"missing={sorted(required_columns - columns)}",
            ),
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "manifest includes required and optional dataset entries",
                "complete" if required_dataset_ids.issubset(dataset_ids) and optional_ids.issubset(dataset_ids) else "partial",
                _rel(csv_path),
                f"dataset_ids={sorted(dataset_ids)}",
            ),
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "all required dataset entries are ready",
                "complete" if required_rows and all(row.get("dataset_status") == "ready" for row in required_rows) else "partial",
                _rel(csv_path),
                _status_detail(required_rows),
            ),
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "optional Xenium and Visium HD placeholders are non-blocking",
                "complete" if optional_ids == {row["dataset_id"] for row in optional_rows} and all(row.get("dataset_status") == "missing_optional" for row in optional_rows) else "partial",
                _rel(csv_path),
                _status_detail(optional_rows),
            ),
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "controlled and real-background benchmark entries expose ground truth",
                "complete"
                if all(
                    _truthy(row.get("has_ground_truth"))
                    and int(float(row.get("n_candidate_edges") or 0)) > 0
                    for row in truth_required_rows
                )
                else "partial",
                _rel(csv_path),
                _status_detail(truth_required_rows),
            ),
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "all required entries expose injected controls or positive/negative controls",
                "complete"
                if all(
                    _truthy(row.get("has_injected_controls"))
                    and int(float(row.get("n_candidate_edges") or 0)) > 0
                    for row in required_rows
                )
                else "partial",
                _rel(csv_path),
                _status_detail(required_rows),
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal1_dataset_manifest",
                "manifest summary marks all required datasets ready",
                "complete" if summary.get("all_required_ready") is True and int(summary.get("n_ready_required_datasets", 0)) == 5 else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _audit_pipeline_contract(config: dict[str, Any]) -> list[AuditRow]:
    methods = config.get("retained_methods") or {}
    metric_methods = set(methods.get("metric_methods") or [])
    status_methods = set(methods.get("status_methods") or [])
    nulls = set((config.get("fdr") or {}).get("null_models") or [])
    figures = {row.get("figure_id") for row in (config.get("figures") or [])}
    required_figures = {f"Fig{i}" for i in range(1, 9)}
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase3/end_to_end")
    rows = [
        AuditRow(
            "phase3_goal2_main_model_and_baselines",
            "representative SpatialSelf method is frozen",
            "complete" if config.get("representative_method") == "spatialself_wavelet_edge_prior" else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
            f"representative_method={config.get('representative_method')}",
        ),
        AuditRow(
            "phase3_goal2_main_model_and_baselines",
            "retained metric and status baselines are declared",
            "complete"
            if {"spatialself_wavelet_edge_prior", "commot", "cellchat", "spatalk", "cellnest"}.issubset(metric_methods)
            and {"deeptalk", "niches"}.issubset(status_methods)
            else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
            f"metric_methods={sorted(metric_methods)} status_methods={sorted(status_methods)}",
        ),
        AuditRow(
            "phase3_goal3_unified_evaluation",
            "Phase 3 output directory exists for unified evaluation artifacts",
            _exists(output_dir),
            _rel(output_dir),
            "later gates will require unified prediction and metric indexes",
        ),
        AuditRow(
            "phase3_goal4_fdr_permutation",
            "FDR contract declares smoke and representative tiers plus three fixed nulls",
            "complete"
            if int((config.get("fdr") or {}).get("smoke_permutations", 0)) == 20
            and int((config.get("fdr") or {}).get("representative_permutations", 0)) == 100
            and {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}.issubset(nulls)
            else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
            f"nulls={sorted(nulls)}",
        ),
        AuditRow(
            "phase3_goal5_biological_validation",
            "manifest dataset sources cover lymph node, tonsil, melanoma, and high-resolution tonsil contexts",
            "complete"
            if {"real_background_lymph_node", "real_background_tonsil", "real_background_melanoma", "high_resolution_tonsil_subset"}.issubset(set((config.get("dataset_sources") or {}).keys()))
            else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
        ),
        AuditRow(
            "phase3_goal6_manuscript_tables_figures",
            "figure scaffold declares Fig1-Fig8",
            "complete" if required_figures == figures else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
            f"figures={sorted(figures)}",
        ),
        AuditRow(
            "phase3",
            "claim level marks scaffold as non-final manuscript",
            "complete" if config.get("claim_level") == "end_to_end_pipeline_scaffold_not_final_manuscript" else "partial",
            "configs/experiments/phase3_end_to_end.yaml",
            f"claim_level={config.get('claim_level')}",
        ),
    ]
    rows.extend(_audit_prediction_index(config, metric_methods, status_methods, output_dir))
    rows.extend(_audit_metric_index(config, metric_methods, status_methods, output_dir))
    rows.extend(_audit_fdr_index(config, output_dir))
    rows.extend(_audit_biological_validation_index(config, output_dir))
    rows.extend(_audit_manuscript_index(config, output_dir))
    return rows


def _audit_prediction_index(
    config: dict[str, Any],
    metric_methods: set[str],
    status_methods: set[str],
    output_dir: Path,
) -> list[AuditRow]:
    path = output_dir / "phase3_prediction_index.csv"
    summary_path = output_dir / "phase3_prediction_index.summary.json"
    rows = [
        AuditRow("phase3_goal2_main_model_and_baselines", "prediction index CSV exists", _exists(path), _rel(path)),
        AuditRow("phase3_goal2_main_model_and_baselines", "prediction index summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not path.exists():
        return rows
    table = _read_csv(path)
    columns = set(table[0].keys()) if table else set()
    required_columns = {
        "dataset_id",
        "dataset_key",
        "method",
        "method_role",
        "status",
        "prediction_path",
        "prediction_exists",
        "source_phase2_artifact",
        "adapter_notes",
    }
    required_datasets = _ready_required_dataset_ids(config)
    expected_methods = metric_methods | status_methods
    present_pairs = {(row.get("dataset_id"), row.get("method")) for row in table}
    missing_pairs = sorted(
        (dataset_id, method)
        for dataset_id in required_datasets
        for method in expected_methods
        if (dataset_id, method) not in present_pairs
    )
    scored_metric_rows = [
        row
        for row in table
        if row.get("dataset_id") in required_datasets
        and row.get("method") in metric_methods
        and row.get("status") == "completed"
        and _truthy(row.get("prediction_exists"))
    ]
    expected_metric_rows = len(required_datasets) * len(metric_methods)
    high_res_cellnest = [
        row
        for row in table
        if row.get("dataset_id") == "high_resolution_tonsil_subset"
        and row.get("method") == "cellnest"
    ]
    rows.extend(
        [
            AuditRow(
                "phase3_goal2_main_model_and_baselines",
                "prediction index has required columns",
                "complete" if required_columns.issubset(columns) else "partial",
                _rel(path),
                f"missing={sorted(required_columns - columns)}",
            ),
            AuditRow(
                "phase3_goal2_main_model_and_baselines",
                "prediction index covers every ready required dataset and declared method",
                "complete" if not missing_pairs else "partial",
                _rel(path),
                f"missing_pairs={missing_pairs[:20]}",
            ),
            AuditRow(
                "phase3_goal2_main_model_and_baselines",
                "completed metric-method predictions are indexed where Phase 2 artifacts exist",
                "complete" if len(scored_metric_rows) >= expected_metric_rows - 1 else "partial",
                _rel(path),
                f"completed={len(scored_metric_rows)} expected={expected_metric_rows}; high_res_cellnest_status={high_res_cellnest[0].get('status') if high_res_cellnest else 'missing'}",
            ),
            AuditRow(
                "phase3_goal2_main_model_and_baselines",
                "DeepTalk and NICHES are retained as explicit status rows",
                "complete"
                if all(
                    (dataset_id, method) in present_pairs
                    for dataset_id in required_datasets
                    for method in {"deeptalk", "niches"}
                )
                else "partial",
                _rel(path),
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal2_main_model_and_baselines",
                "prediction summary reports high required metric coverage",
                "complete" if float(summary.get("required_metric_coverage_fraction", 0.0)) >= 0.95 else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _audit_metric_index(
    config: dict[str, Any],
    metric_methods: set[str],
    status_methods: set[str],
    output_dir: Path,
) -> list[AuditRow]:
    path = output_dir / "phase3_metric_index.csv"
    summary_path = output_dir / "phase3_metric_index.summary.json"
    rows = [
        AuditRow("phase3_goal3_unified_evaluation", "metric index CSV exists", _exists(path), _rel(path)),
        AuditRow("phase3_goal3_unified_evaluation", "metric index summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not path.exists():
        return rows
    table = _read_csv(path)
    columns = set(table[0].keys()) if table else set()
    required_columns = {
        "dataset_id",
        "method",
        "prediction_status",
        "metric_status",
        "evaluation_context",
        "auroc",
        "aupr",
        "weak_signal_recall",
        "empirical_fdr",
        "high_expression_false_positive_rate",
        "positive_control_rank",
        "negative_control_rank",
        "downstream_spearman",
        "runtime_seconds",
        "limitations",
    }
    required_datasets = _ready_required_dataset_ids(config)
    expected_methods = metric_methods | status_methods
    present_pairs = {(row.get("dataset_id"), row.get("method")) for row in table}
    truth_rows = [
        row
        for row in table
        if row.get("dataset_id") in {
            "controlled_simulator_v2",
            "real_background_lymph_node",
            "real_background_tonsil",
            "real_background_melanoma",
        }
        and row.get("method") in metric_methods
    ]
    control_rows = [
        row
        for row in table
        if row.get("dataset_id") in {
            "real_background_lymph_node",
            "real_background_tonsil",
            "real_background_melanoma",
            "high_resolution_tonsil_subset",
        }
        and row.get("method") in metric_methods
    ]
    rows.extend(
        [
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "metric index has required unified columns",
                "complete" if required_columns.issubset(columns) else "partial",
                _rel(path),
                f"missing={sorted(required_columns - columns)}",
            ),
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "metric index covers every ready required dataset and declared method",
                "complete"
                if all(
                    (dataset_id, method) in present_pairs
                    for dataset_id in required_datasets
                    for method in expected_methods
                )
                else "partial",
                _rel(path),
            ),
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "truth-enabled rows expose AUROC/AUPR/FDR/weak-signal columns when scored",
                "complete"
                if truth_rows
                and all(
                    row.get("metric_status") == "complete"
                    and row.get("auroc") != ""
                    and row.get("aupr") != ""
                    and row.get("weak_signal_recall") != ""
                    and row.get("empirical_fdr") != ""
                    for row in truth_rows
                    if row.get("prediction_status") == "completed"
                )
                else "partial",
                _rel(path),
                f"truth_rows={len(truth_rows)}",
            ),
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "control-enabled real/high-resolution rows expose positive/negative ranking or limitation",
                "complete"
                if control_rows
                and all(
                    row.get("positive_control_rank") != ""
                    or row.get("limitations") != ""
                    or row.get("method") == "cellnest"
                    for row in control_rows
                )
                else "partial",
                _rel(path),
                f"control_rows={len(control_rows)}",
            ),
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "status-only methods are not treated as scored benchmark failures",
                "complete"
                if all(
                    row.get("metric_status") == "status_only"
                    for row in table
                    if row.get("method") in status_methods
                )
                else "partial",
                _rel(path),
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal3_unified_evaluation",
                "metric summary reports high required metric coverage",
                "complete" if float(summary.get("required_metric_coverage_fraction", 0.0)) >= 0.95 else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _ready_required_dataset_ids(config: dict[str, Any]) -> set[str]:
    manifest_cfg = config.get("manifest") or {}
    path = PROJECT_ROOT / manifest_cfg.get("output_csv", "outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    if not path.exists():
        return {
            "controlled_simulator_v2",
            "real_background_lymph_node",
            "real_background_tonsil",
            "real_background_melanoma",
            "high_resolution_tonsil_subset",
        }
    return {
        row["dataset_id"]
        for row in _read_csv(path)
        if _truthy(row.get("required")) and row.get("dataset_status") == "ready"
    }


def _audit_fdr_index(config: dict[str, Any], output_dir: Path) -> list[AuditRow]:
    path = output_dir / "phase3_fdr_index.csv"
    summary_path = output_dir / "phase3_fdr_index.summary.json"
    rows = [
        AuditRow("phase3_goal4_fdr_permutation", "FDR index CSV exists", _exists(path), _rel(path)),
        AuditRow("phase3_goal4_fdr_permutation", "FDR index summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not path.exists():
        return rows
    table = _read_csv(path)
    columns = set(table[0].keys()) if table else set()
    required_columns = {
        "dataset_id",
        "method",
        "null_model",
        "tier",
        "n_permutations",
        "status",
        "candidate_edge_fdr_path",
        "lr_pair_fdr_path",
        "cell_type_pair_fdr_path",
        "significant_edges_path",
        "n_significant_edges",
        "n_significant_lr_pairs",
        "n_significant_cell_type_pairs",
        "limitations",
    }
    nulls = set((config.get("fdr") or {}).get("null_models") or [])
    required_datasets = _ready_required_dataset_ids(config)
    present_pairs = {(row.get("dataset_id"), row.get("null_model")) for row in table}
    real_background_ids = {"real_background_lymph_node", "real_background_tonsil", "real_background_melanoma"}
    real_background_complete = [
        row
        for row in table
        if row.get("dataset_id") in real_background_ids
        and row.get("status") == "complete"
        and row.get("tier") == "representative_100perm"
    ]
    missing_expected = sorted(
        (dataset_id, null_model)
        for dataset_id in required_datasets
        for null_model in nulls
        if (dataset_id, null_model) not in present_pairs
    )
    rows.extend(
        [
            AuditRow(
                "phase3_goal4_fdr_permutation",
                "FDR index has required columns",
                "complete" if required_columns.issubset(columns) else "partial",
                _rel(path),
                f"missing={sorted(required_columns - columns)}",
            ),
            AuditRow(
                "phase3_goal4_fdr_permutation",
                "FDR index covers every ready required dataset and fixed null model",
                "complete" if not missing_expected else "partial",
                _rel(path),
                f"missing_pairs={missing_expected[:20]}",
            ),
            AuditRow(
                "phase3_goal4_fdr_permutation",
                "representative 100-permutation real-background FDR rows are attached",
                "complete" if len(real_background_complete) == len(real_background_ids) * len(nulls) else "partial",
                _rel(path),
                f"complete_real_background_rows={len(real_background_complete)}",
            ),
            AuditRow(
                "phase3_goal4_fdr_permutation",
                "not-yet-run FDR contexts are explicit rather than silent",
                "complete" if all(row.get("status") == "complete" for row in table) else "partial",
                _rel(path),
                "all required Phase 3 FDR rows should now be complete",
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal4_fdr_permutation",
                "FDR summary reports nine complete real-background 100-perm rows",
                "complete" if int(summary.get("complete_real_background_100perm_rows", 0)) == 9 else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _audit_biological_validation_index(config: dict[str, Any], output_dir: Path) -> list[AuditRow]:
    path = output_dir / "phase3_biological_validation_index.csv"
    summary_path = output_dir / "phase3_biological_validation_index.summary.json"
    rows = [
        AuditRow("phase3_goal5_biological_validation", "biological validation index CSV exists", _exists(path), _rel(path)),
        AuditRow("phase3_goal5_biological_validation", "biological validation index summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not path.exists():
        return rows
    table = _read_csv(path)
    columns = set(table[0].keys()) if table else set()
    required_columns = {
        "dataset_id",
        "method",
        "validation_context",
        "status",
        "n_positive_controls",
        "n_negative_controls",
        "positive_top3_count",
        "negative_top3_count",
        "best_positive_rank",
        "best_negative_rank",
        "mean_positive_downstream_spearman",
        "top_lr_or_control",
        "representative_spatial_map_source",
        "limitations",
    }
    expected_contexts = {
        "real_background_lymph_node",
        "real_background_tonsil",
        "real_background_melanoma",
        "high_resolution_tonsil_subset",
    }
    present_contexts = {row.get("dataset_id") for row in table}
    spatialself_rows = [
        row
        for row in table
        if row.get("method") == "spatialself_wavelet_edge_prior"
        and row.get("dataset_id") in expected_contexts
    ]
    completed_rows = [row for row in table if row.get("status") == "complete"]
    rows.extend(
        [
            AuditRow(
                "phase3_goal5_biological_validation",
                "biological validation index has required columns",
                "complete" if required_columns.issubset(columns) else "partial",
                _rel(path),
                f"missing={sorted(required_columns - columns)}",
            ),
            AuditRow(
                "phase3_goal5_biological_validation",
                "biological validation covers lymph node, tonsil, melanoma, and high-resolution tonsil",
                "complete" if expected_contexts.issubset(present_contexts) else "partial",
                _rel(path),
                f"contexts={sorted(present_contexts)}",
            ),
            AuditRow(
                "phase3_goal5_biological_validation",
                "SpatialSelf has complete biological validation rows in all required real/high-resolution contexts",
                "complete" if len(spatialself_rows) == 4 and all(row.get("status") == "complete" for row in spatialself_rows) else "partial",
                _rel(path),
                f"spatialself_rows={len(spatialself_rows)}",
            ),
            AuditRow(
                "phase3_goal5_biological_validation",
                "completed biological validation rows include control ranking evidence",
                "complete"
                if completed_rows
                and all(row.get("best_positive_rank") != "" or row.get("top_lr_or_control") != "" for row in completed_rows)
                else "partial",
                _rel(path),
                f"completed_rows={len(completed_rows)}",
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal5_biological_validation",
                "biological validation summary reports SpatialSelf coverage across four contexts",
                "complete" if int(summary.get("spatialself_complete_contexts", 0)) == 4 else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _audit_manuscript_index(config: dict[str, Any], output_dir: Path) -> list[AuditRow]:
    source_path = output_dir / "phase3_manuscript_source_table_index.csv"
    figure_path = output_dir / "phase3_figure_scaffold_index.csv"
    summary_path = output_dir / "phase3_manuscript_index_summary.json"
    render_qa_path = output_dir / "phase3_figure_render_qa.csv"
    render_summary_path = output_dir / "phase3_figure_render_summary.json"
    rows = [
        AuditRow("phase3_goal6_manuscript_tables_figures", "manuscript source table index exists", _exists(source_path), _rel(source_path)),
        AuditRow("phase3_goal6_manuscript_tables_figures", "figure scaffold index exists", _exists(figure_path), _rel(figure_path)),
        AuditRow("phase3_goal6_manuscript_tables_figures", "manuscript index summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase3_goal6_manuscript_tables_figures", "figure render QA exists", _exists(render_qa_path), _rel(render_qa_path)),
        AuditRow("phase3_goal6_manuscript_tables_figures", "figure render summary exists", _exists(render_summary_path), _rel(render_summary_path)),
    ]
    if not source_path.exists() or not figure_path.exists():
        return rows
    sources = _read_csv(source_path)
    figures = _read_csv(figure_path)
    source_columns = set(sources[0].keys()) if sources else set()
    figure_columns = set(figures[0].keys()) if figures else set()
    required_source_columns = {
        "source_table_id",
        "phase3_goal",
        "path",
        "exists",
        "n_rows",
        "intended_figures",
        "supported_claim",
        "limitation",
    }
    required_figure_columns = {
        "figure_id",
        "title",
        "source_table_ids",
        "source_paths",
        "plotting_script",
        "figure_output",
        "qa_status",
        "supported_claim",
        "limitation",
    }
    expected_figures = {f"Fig{i}" for i in range(1, 9)}
    present_figures = {row.get("figure_id") for row in figures}
    rows.extend(
        [
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "source table index has required columns",
                "complete" if required_source_columns.issubset(source_columns) else "partial",
                _rel(source_path),
                f"missing={sorted(required_source_columns - source_columns)}",
            ),
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "figure scaffold index has required columns",
                "complete" if required_figure_columns.issubset(figure_columns) else "partial",
                _rel(figure_path),
                f"missing={sorted(required_figure_columns - figure_columns)}",
            ),
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "figure scaffold covers Fig1-Fig8",
                "complete" if expected_figures == present_figures else "partial",
                _rel(figure_path),
                f"figures={sorted(present_figures)}",
            ),
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "all figure scaffold rows have source tables, supported claim, and limitation",
                "complete"
                if figures
                and all(
                    row.get("source_table_ids")
                    and row.get("supported_claim")
                    and row.get("limitation")
                    and row.get("qa_status") == "source_tables_available_plot_pending"
                    for row in figures
                )
                else "partial",
                _rel(figure_path),
            ),
        ]
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "manuscript summary reports six existing source tables and eight figure scaffolds",
                "complete"
                if int(summary.get("n_existing_source_tables", 0)) == 6 and int(summary.get("n_figures", 0)) == 8
                else "partial",
                _rel(summary_path),
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
        )
    if render_qa_path.exists():
        qa_rows = _read_csv(render_qa_path)
        expected_figures = {f"Fig{i}" for i in range(1, 9)}
        rendered_figures = {row.get("figure_id") for row in qa_rows if row.get("status") == "rendered"}
        missing_files: list[str] = []
        for row in qa_rows:
            for column in ("pdf", "svg", "tiff"):
                file_path = PROJECT_ROOT / row.get(column, "")
                if not file_path.exists() or file_path.stat().st_size == 0:
                    missing_files.append(row.get(column, ""))
        rows.append(
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "Fig1-Fig8 scaffold figure files are rendered as PDF/SVG/TIFF",
                "complete" if rendered_figures == expected_figures and not missing_files else "partial",
                _rel(render_qa_path),
                f"rendered={sorted(rendered_figures)} missing_files={missing_files[:10]}",
            )
        )
    if render_summary_path.exists():
        render_summary = json.loads(render_summary_path.read_text(encoding="utf-8"))
        rows.append(
            AuditRow(
                "phase3_goal6_manuscript_tables_figures",
                "figure render summary reports eight rendered figures",
                "complete" if int(render_summary.get("n_rendered_figures", 0)) == 8 else "partial",
                _rel(render_summary_path),
                json.dumps(render_summary, ensure_ascii=True, sort_keys=True),
            )
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _status_detail(rows: list[dict[str, str]]) -> str:
    return json.dumps(
        [
            {
                "dataset_id": row.get("dataset_id"),
                "dataset_status": row.get("dataset_status"),
                "exists": row.get("exists"),
                "has_ground_truth": row.get("has_ground_truth"),
                "has_injected_controls": row.get("has_injected_controls"),
                "n_candidate_edges": row.get("n_candidate_edges"),
            }
            for row in rows
        ],
        ensure_ascii=True,
    )


def _exists(path: Path) -> str:
    return "complete" if path.exists() else "missing"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
