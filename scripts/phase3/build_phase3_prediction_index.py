#!/usr/bin/env python
"""Build a unified Phase 3 prediction/status index from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_COLUMNS = [
    "dataset_id",
    "dataset_key",
    "dataset_type",
    "method",
    "method_role",
    "method_scope",
    "status",
    "prediction_path",
    "prediction_exists",
    "n_prediction_rows",
    "n_unique_scores",
    "n_nonzero_scores",
    "runtime_seconds",
    "status_path",
    "runtime_path",
    "source_phase2_artifact",
    "baseline_evidence_level",
    "adapter_notes",
]

DATASET_KEYS = {
    "controlled_simulator_v2": "replicate_00",
    "real_background_lymph_node": "human_lymph_node_visium_replicate_00",
    "real_background_tonsil": "human_tonsil_slide_tags_replicate_00",
    "real_background_melanoma": "human_melanoma_multiome_replicate_00",
    "high_resolution_tonsil_subset": "constructed_replicate",
}

REAL_BACKGROUND_DATASET_NAMES = {
    "real_background_lymph_node": "real_background_v2_formal_lymph_node",
    "real_background_tonsil": "real_background_v2_formal_tonsil",
    "real_background_melanoma": "real_background_v2_formal_melanoma",
}

METHOD_ALIASES = {
    "spatialself_wavelet": "spatialself_wavelet_edge_prior",
    "spatialself_wavelet_calibrated": "spatialself_wavelet_edge_prior",
}

METHOD_SCOPES = {
    "spatialself_wavelet_edge_prior": "candidate_edge_metric",
    "commot": "candidate_edge_metric",
    "cellchat": "cell_type_pair_projected_adapter_light",
    "spatalk": "cell_type_pair_projected_adapter_light",
    "cellnest": "candidate_edge_adapter_light",
    "deeptalk": "status_only_model_required_unless_predict_ccc_exists",
    "niches": "status_only_dependency_blocked",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase3_end_to_end.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase3/end_to_end")
    manifest_path = PROJECT_ROOT / (
        args.manifest
        or (config.get("manifest") or {}).get("output_csv", "outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    )
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase3_prediction_index.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(manifest_path)
    metric_methods = list(((config.get("retained_methods") or {}).get("metric_methods") or []))
    status_methods = list(((config.get("retained_methods") or {}).get("status_methods") or []))
    all_methods = metric_methods + [method for method in status_methods if method not in metric_methods]

    artifact_rows = _load_artifact_rows()
    rows: list[dict[str, Any]] = []
    for dataset in manifest_rows:
        if dataset.get("dataset_status") not in {"ready", "missing_optional"}:
            continue
        for method in all_methods:
            rows.append(_index_row(dataset, method, artifact_rows))

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 prediction index: {output_path}")
    print(json.dumps(summary, indent=2))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 3 config loading.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_artifact_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    candidates = [
        PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_metrics.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_status.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal1_formal_baseline_matched_validation/goal1_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_status.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch/tonsil_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch/tonsil_baseline_matched_status.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for row in _read_csv(path):
            row = dict(row)
            row["_source_artifact"] = _rel(path)
            rows.append(row)
    return rows


def _index_row(dataset: dict[str, str], method: str, artifacts: list[dict[str, str]]) -> dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    dataset_key = DATASET_KEYS.get(dataset_id, dataset_id)
    dataset_type = dataset.get("dataset_type", "")
    method_role = "metric_method" if method not in {"deeptalk", "niches"} else "status_method"
    if dataset.get("dataset_status") == "missing_optional":
        return _base_row(dataset, method, dataset_key, "optional_dataset_missing", "optional dataset placeholder is non-blocking")

    match = _find_artifact(dataset_id, dataset_key, method, artifacts)
    if not match:
        status = "status_only" if method_role == "status_method" else "missing_prediction"
        return _base_row(dataset, method, dataset_key, status, "no matching Phase 2 artifact found")

    prediction_path = _normalize_path(match.get("prediction_path", ""))
    prediction_exists = bool(prediction_path) and (PROJECT_ROOT / prediction_path).exists()
    raw_status = match.get("status") or ("completed" if prediction_exists else "")
    if prediction_exists and raw_status in {"", "completed_or_model_required"}:
        status = "completed"
    elif raw_status:
        status = raw_status
    else:
        status = "missing_prediction"
    if method_role == "status_method" and not prediction_exists:
        status = raw_status or "status_only"

    return {
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "dataset_type": dataset_type,
        "method": method,
        "method_role": method_role,
        "method_scope": match.get("method_scope") or METHOD_SCOPES.get(method, ""),
        "status": status,
        "prediction_path": prediction_path,
        "prediction_exists": prediction_exists,
        "n_prediction_rows": match.get("n_prediction_rows") or match.get("n_candidates") or "",
        "n_unique_scores": match.get("n_unique_scores", ""),
        "n_nonzero_scores": match.get("n_nonzero_scores", ""),
        "runtime_seconds": match.get("wall_clock_seconds", ""),
        "status_path": _normalize_path(match.get("status_path", "")),
        "runtime_path": "",
        "source_phase2_artifact": match.get("_source_artifact", ""),
        "baseline_evidence_level": match.get("baseline_evidence_level") or match.get("method_scope") or METHOD_SCOPES.get(method, ""),
        "adapter_notes": match.get("interpretation_note") or match.get("detail") or "",
    }


def _find_artifact(
    dataset_id: str,
    dataset_key: str,
    method: str,
    artifacts: list[dict[str, str]],
) -> dict[str, str] | None:
    aliases = {method}
    aliases.update(alias for alias, canonical in METHOD_ALIASES.items() if canonical == method)
    scored: list[tuple[int, dict[str, str]]] = []
    for row in artifacts:
        row_method = row.get("method", "")
        if row_method not in aliases:
            continue
        score = _dataset_match_score(dataset_id, dataset_key, row)
        if score > 0:
            scored.append((score, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1].get("_source_artifact", "")), reverse=True)
    return scored[0][1]


def _dataset_match_score(dataset_id: str, dataset_key: str, row: dict[str, str]) -> int:
    row_dataset = row.get("dataset", "")
    prediction_path = row.get("prediction_path", "")
    source = row.get("_source_artifact", "")
    text = " ".join([row_dataset, prediction_path, source])
    if dataset_id == "controlled_simulator_v2":
        if dataset_key in text and "controlled_v2_formal" in text:
            return 100
        return 0
    if dataset_id in REAL_BACKGROUND_DATASET_NAMES:
        expected_dataset = REAL_BACKGROUND_DATASET_NAMES[dataset_id]
        if row_dataset == expected_dataset:
            return 120
        if dataset_key in text and "real_background_v2_formal" in text:
            return 100
        return 0
    if dataset_id == "high_resolution_tonsil_subset":
        if "tonsil_baseline_matched_fresh_subset" in text or dataset_key in text:
            return 100
        return 0
    return 0


def _base_row(dataset: dict[str, str], method: str, dataset_key: str, status: str, note: str) -> dict[str, Any]:
    method_role = "metric_method" if method not in {"deeptalk", "niches"} else "status_method"
    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_key": dataset_key,
        "dataset_type": dataset.get("dataset_type", ""),
        "method": method,
        "method_role": method_role,
        "method_scope": METHOD_SCOPES.get(method, ""),
        "status": status,
        "prediction_path": "",
        "prediction_exists": False,
        "n_prediction_rows": "",
        "n_unique_scores": "",
        "n_nonzero_scores": "",
        "runtime_seconds": "",
        "status_path": "",
        "runtime_path": "",
        "source_phase2_artifact": "",
        "baseline_evidence_level": METHOD_SCOPES.get(method, ""),
        "adapter_notes": note,
    }


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _summary(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    required_rows = [row for row in rows if row["dataset_id"] not in {"optional_xenium_subset", "optional_visium_hd_subset"}]
    metric_rows = [row for row in required_rows if row["method_role"] == "metric_method"]
    completed_metric_rows = [row for row in metric_rows if row["status"] == "completed" and str(row["prediction_exists"]) == "True"]
    return {
        "prediction_index_csv": _rel(path),
        "n_rows": len(rows),
        "n_required_rows": len(required_rows),
        "n_required_metric_rows": len(metric_rows),
        "n_completed_required_metric_rows": len(completed_metric_rows),
        "required_metric_coverage_fraction": len(completed_metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "status_counts": _counts(row["status"] for row in rows),
        "method_counts": _counts(row["method"] for row in rows),
    }


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
