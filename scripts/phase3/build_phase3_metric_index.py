#!/usr/bin/env python
"""Build a unified Phase 3 metric index for manuscript source tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_COLUMNS = [
    "dataset_id",
    "dataset_type",
    "method",
    "prediction_status",
    "metric_status",
    "evaluation_context",
    "has_ground_truth",
    "auroc",
    "aupr",
    "top_k_precision",
    "weak_signal_recall",
    "empirical_fdr",
    "high_expression_false_positive_rate",
    "calibration_error",
    "positive_control_rank",
    "negative_control_rank",
    "positive_top3_count",
    "negative_top3_count",
    "high_expression_negative_topk",
    "downstream_spearman",
    "downstream_validation_level",
    "region_enrichment_available",
    "fdr_available",
    "runtime_seconds",
    "source_metric_artifact",
    "source_control_artifact",
    "source_downstream_artifact",
    "limitations",
]

REAL_BACKGROUND_DATASET_NAMES = {
    "real_background_lymph_node": "real_background_v2_formal_lymph_node",
    "real_background_tonsil": "real_background_v2_formal_tonsil",
    "real_background_melanoma": "real_background_v2_formal_melanoma",
}

METHOD_ALIASES = {
    "spatialself_wavelet": "spatialself_wavelet_edge_prior",
    "spatialself_wavelet_calibrated": "spatialself_wavelet_edge_prior",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase3_end_to_end.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--prediction-index", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase3/end_to_end")
    manifest_path = PROJECT_ROOT / (
        args.manifest
        or (config.get("manifest") or {}).get("output_csv", "outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    )
    prediction_path = PROJECT_ROOT / (args.prediction_index or str(output_dir / "phase3_prediction_index.csv"))
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase3_metric_index.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {row["dataset_id"]: row for row in _read_csv(manifest_path)}
    prediction_rows = _read_csv(prediction_path)
    metric_sources = _load_metric_sources()
    control_sources = _load_control_sources()
    downstream_sources = _load_downstream_sources()

    rows = [
        _metric_row(row, manifest.get(row["dataset_id"], {}), metric_sources, control_sources, downstream_sources)
        for row in prediction_rows
        if row.get("dataset_id") not in {"optional_xenium_subset", "optional_visium_hd_subset"}
    ]

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 metric index: {output_path}")
    print(json.dumps(summary, indent=2))


def _metric_row(
    prediction: dict[str, str],
    dataset: dict[str, str],
    metric_sources: list[dict[str, str]],
    control_sources: list[dict[str, str]],
    downstream_sources: list[dict[str, str]],
) -> dict[str, Any]:
    dataset_id = prediction["dataset_id"]
    method = prediction["method"]
    has_ground_truth = _truthy(dataset.get("has_ground_truth", ""))
    metric = _find_metric(dataset_id, method, metric_sources)
    control = _find_control(dataset_id, method, control_sources)
    downstream = _find_downstream(dataset_id, method, downstream_sources)

    row = {
        "dataset_id": dataset_id,
        "dataset_type": dataset.get("dataset_type", ""),
        "method": method,
        "prediction_status": prediction.get("status", ""),
        "metric_status": "complete" if metric or control or downstream else "missing",
        "evaluation_context": _evaluation_context(dataset_id, has_ground_truth),
        "has_ground_truth": has_ground_truth,
        "auroc": _get(metric, "auroc"),
        "aupr": _get(metric, "aupr"),
        "top_k_precision": _get(metric, "top_k_precision"),
        "weak_signal_recall": _get(metric, "weak_signal_recall_at_threshold"),
        "empirical_fdr": _first_nonempty(_get(metric, "empirical_fdr_at_threshold"), _get(control, "empirical_fdr_at_q95")),
        "high_expression_false_positive_rate": _first_nonempty(
            _get(metric, "false_positive_high_expression_rate_at_threshold"),
            _get(control, "high_expression_false_positive_rate_at_q95"),
        ),
        "calibration_error": _get(metric, "calibration_error_10bin"),
        "positive_control_rank": _first_nonempty(_get(control, "best_positive_rank"), _get(control, "best_positive_control_rank")),
        "negative_control_rank": _first_nonempty(_get(control, "best_negative_rank"), _get(control, "best_negative_control_rank")),
        "positive_top3_count": _get(control, "positive_top3_count"),
        "negative_top3_count": _get(control, "negative_top3_count"),
        "high_expression_negative_topk": _get(control, "negative_top3_count"),
        "downstream_spearman": _first_nonempty(
            _get(downstream, "mean_positive_spearman"),
            _get(control, "mean_positive_receiver_response_proxy_spearman"),
        ),
        "downstream_validation_level": _first_nonempty(
            _get(downstream, "target_validation_level"),
            _get(control, "receiver_response_proxy_level"),
        ),
        "region_enrichment_available": "yes" if dataset_id == "high_resolution_tonsil_subset" and control else "",
        "fdr_available": _fdr_available(dataset_id, method),
        "runtime_seconds": _first_nonempty(prediction.get("runtime_seconds", ""), _get(control, "wall_clock_seconds")),
        "source_metric_artifact": _get(metric, "_source_artifact"),
        "source_control_artifact": _get(control, "_source_artifact"),
        "source_downstream_artifact": _get(downstream, "_source_artifact"),
        "limitations": _limitations(dataset_id, method, prediction, metric, control, downstream),
    }
    if prediction.get("method_role") == "status_method":
        row["metric_status"] = "status_only"
    return row


def _load_metric_sources() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_metrics.csv",
    ]
    return _load_rows(paths)


def _load_control_sources() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal1_formal_baseline_matched_validation/goal1_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch/tonsil_baseline_matched_method_summary.csv",
    ]
    return _load_rows(paths)


def _load_downstream_sources() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal1_curated_downstream_panel/goal1_curated_downstream_method_summary.csv",
    ]
    return _load_rows(paths)


def _load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            row = dict(row)
            row["_source_artifact"] = _rel(path)
            row["method"] = _canonical_method(row.get("method", ""))
            rows.append(row)
    return rows


def _find_metric(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_metric_dataset(dataset_id, row)]
    return _best_by_source(matches)


def _find_control(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_control_dataset(dataset_id, row)]
    return _best_by_source(matches)


def _find_downstream(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_control_dataset(dataset_id, row)]
    return _best_by_source(matches)


def _matches_metric_dataset(dataset_id: str, row: dict[str, str]) -> bool:
    dataset_name = row.get("dataset", "")
    replicate = row.get("replicate", "")
    prediction_path = row.get("prediction_path", "")
    text = " ".join([dataset_name, replicate, prediction_path])
    if dataset_id == "controlled_simulator_v2":
        return dataset_name == "controlled_v2_formal" and replicate == "replicate_00"
    if dataset_id in REAL_BACKGROUND_DATASET_NAMES:
        return REAL_BACKGROUND_DATASET_NAMES[dataset_id].replace("real_background_v2_formal_", "") in dataset_name or _expected_key(dataset_id) in text
    return False


def _matches_control_dataset(dataset_id: str, row: dict[str, str]) -> bool:
    dataset_name = row.get("dataset", "")
    prediction_path = row.get("prediction_path", "")
    text = " ".join([dataset_name, prediction_path, row.get("_source_artifact", "")])
    if dataset_id in REAL_BACKGROUND_DATASET_NAMES:
        return dataset_name == REAL_BACKGROUND_DATASET_NAMES[dataset_id] or _expected_key(dataset_id) in text
    if dataset_id == "high_resolution_tonsil_subset":
        return "tonsil_baseline_matched_fresh_subset" in text or "constructed_replicate" in text
    return False


def _expected_key(dataset_id: str) -> str:
    return {
        "real_background_lymph_node": "human_lymph_node_visium_replicate_00",
        "real_background_tonsil": "human_tonsil_slide_tags_replicate_00",
        "real_background_melanoma": "human_melanoma_multiome_replicate_00",
    }.get(dataset_id, dataset_id)


def _best_by_source(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: row.get("_source_artifact", ""))
    return rows[0]


def _canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def _evaluation_context(dataset_id: str, has_ground_truth: bool) -> str:
    if dataset_id == "high_resolution_tonsil_subset":
        return "real_subset_controls_no_full_truth"
    return "ground_truth_candidate_edges" if has_ground_truth else "controls_only"


def _fdr_available(dataset_id: str, method: str) -> str:
    if method != "spatialself_wavelet_edge_prior":
        return ""
    fdr_root = PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_100perm"
    if dataset_id == "real_background_tonsil" and fdr_root.exists():
        return "representative_100perm_available"
    formal_root = PROJECT_ROOT / "outputs/phase2_revised/goal5_formal_fdr"
    if dataset_id == "real_background_tonsil" and formal_root.exists():
        return "formal_fdr_available"
    return ""


def _limitations(
    dataset_id: str,
    method: str,
    prediction: dict[str, str],
    metric: dict[str, str] | None,
    control: dict[str, str] | None,
    downstream: dict[str, str] | None,
) -> str:
    notes: list[str] = []
    if prediction.get("status") != "completed":
        notes.append(f"prediction_status={prediction.get('status')}")
    if method in {"cellchat", "spatalk"}:
        notes.append("cell_type_pair_scores_projected_to_candidate_edges")
    if method == "cellnest":
        notes.append("adapter_light_short_training")
    if method in {"deeptalk", "niches"}:
        notes.append("status_record_only_not_scored")
    if dataset_id == "high_resolution_tonsil_subset":
        notes.append("controls_only_no_full_ground_truth")
    if not downstream and dataset_id.startswith("real_background"):
        notes.append("curated_downstream_missing")
    if not metric and dataset_id != "high_resolution_tonsil_subset" and method not in {"deeptalk", "niches"}:
        notes.append("candidate_truth_metrics_missing")
    if not control and dataset_id != "controlled_simulator_v2" and method not in {"deeptalk", "niches"}:
        notes.append("control_metrics_missing")
    return ";".join(notes)


def _get(row: dict[str, str] | None, key: str) -> str:
    if not row:
        return ""
    return row.get(key, "")


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value) != "":
            return str(value)
    return ""


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
    required_metric_rows = [row for row in rows if row["method"] not in {"deeptalk", "niches"}]
    complete_rows = [row for row in required_metric_rows if row["metric_status"] == "complete"]
    return {
        "metric_index_csv": _rel(path),
        "n_rows": len(rows),
        "n_required_metric_rows": len(required_metric_rows),
        "n_complete_required_metric_rows": len(complete_rows),
        "required_metric_coverage_fraction": len(complete_rows) / len(required_metric_rows) if required_metric_rows else 0.0,
        "metric_status_counts": _counts(row["metric_status"] for row in rows),
        "evaluation_context_counts": _counts(row["evaluation_context"] for row in rows),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 3 config loading.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
