#!/usr/bin/env python
"""Build a Phase 3 biological validation index."""

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
    "validation_context",
    "status",
    "n_positive_controls",
    "n_negative_controls",
    "positive_top3_count",
    "negative_top3_count",
    "best_positive_rank",
    "best_negative_rank",
    "rank_validation_pass",
    "high_expression_fp_rate",
    "mean_positive_downstream_spearman",
    "mean_negative_downstream_spearman",
    "downstream_delta",
    "downstream_validation_level",
    "top_lr_or_control",
    "representative_spatial_map_source",
    "source_control_artifact",
    "source_downstream_artifact",
    "limitations",
]

METHOD_ALIASES = {
    "spatialself_wavelet": "spatialself_wavelet_edge_prior",
    "spatialself_wavelet_calibrated": "spatialself_wavelet_edge_prior",
}

REAL_BACKGROUND_DATASET_NAMES = {
    "real_background_lymph_node": "real_background_v2_formal_lymph_node",
    "real_background_tonsil": "real_background_v2_formal_tonsil",
    "real_background_melanoma": "real_background_v2_formal_melanoma",
}

METHODS = ["spatialself_wavelet_edge_prior", "commot", "cellchat", "spatalk", "cellnest"]


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
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase3_biological_validation_index.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = [row for row in _read_csv(manifest_path) if _truthy(row.get("required")) and row.get("dataset_status") == "ready"]
    control_rows = _load_control_rows()
    downstream_rows = _load_downstream_rows()
    top_lr_rows = _load_top_lr_rows()

    rows: list[dict[str, Any]] = []
    for dataset in manifest:
        if dataset["dataset_id"] == "controlled_simulator_v2":
            continue
        for method in METHODS:
            rows.append(_bio_row(dataset, method, control_rows, downstream_rows, top_lr_rows))

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path)
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 biological validation index: {output_path}")
    print(json.dumps(summary, indent=2))


def _bio_row(
    dataset: dict[str, str],
    method: str,
    control_rows: list[dict[str, str]],
    downstream_rows: list[dict[str, str]],
    top_lr_rows: list[dict[str, str]],
) -> dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    control = _find_control(dataset_id, method, control_rows)
    downstream = _find_downstream(dataset_id, method, downstream_rows)
    top_lr = _find_top_lr(dataset_id, method, top_lr_rows)
    status = "complete" if control or downstream or top_lr else "missing"
    return {
        "dataset_id": dataset_id,
        "dataset_type": dataset.get("dataset_type", ""),
        "method": method,
        "validation_context": _validation_context(dataset_id),
        "status": status,
        "n_positive_controls": _first_nonempty(_get(control, "n_positive_controls"), _count_class(dataset_id, method, top_lr_rows, positive=True)),
        "n_negative_controls": _first_nonempty(_get(control, "n_negative_controls"), _count_class(dataset_id, method, top_lr_rows, positive=False)),
        "positive_top3_count": _first_nonempty(_get(control, "positive_top3_count"), _get(control, "n_positive_controls_top_decile")),
        "negative_top3_count": _first_nonempty(_get(control, "negative_top3_count"), _get(control, "n_negative_controls_top_decile")),
        "best_positive_rank": _first_nonempty(_get(control, "best_positive_rank"), _get(control, "best_positive_control_rank")),
        "best_negative_rank": _first_nonempty(_get(control, "best_negative_rank"), _get(control, "best_negative_control_rank")),
        "rank_validation_pass": _get(control, "rank_validation_pass"),
        "high_expression_fp_rate": _get(control, "high_expression_false_positive_rate_at_q95"),
        "mean_positive_downstream_spearman": _get(downstream, "mean_positive_spearman"),
        "mean_negative_downstream_spearman": _get(downstream, "mean_negative_spearman"),
        "downstream_delta": _get(downstream, "positive_negative_spearman_delta"),
        "downstream_validation_level": _first_nonempty(_get(downstream, "target_validation_level"), _get(control, "receiver_response_proxy_level")),
        "top_lr_or_control": _top_name(top_lr, control),
        "representative_spatial_map_source": _representative_source(dataset_id, method, top_lr, control),
        "source_control_artifact": _get(control, "_source_artifact"),
        "source_downstream_artifact": _get(downstream, "_source_artifact"),
        "limitations": _limitations(dataset_id, method, control, downstream),
    }


def _load_control_rows() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal1_formal_baseline_matched_validation/goal1_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_method_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch/tonsil_baseline_matched_method_summary.csv",
    ]
    return _load_rows(paths)


def _load_downstream_rows() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal1_curated_downstream_panel/goal1_curated_downstream_method_summary.csv",
    ]
    return _load_rows(paths)


def _load_top_lr_rows() -> list[dict[str, str]]:
    paths = [
        PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_control_lr_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch/tonsil_baseline_matched_control_lr_summary.csv",
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
            row["method"] = METHOD_ALIASES.get(row.get("method", ""), row.get("method", ""))
            rows.append(row)
    return rows


def _find_control(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_dataset(dataset_id, row)]
    return matches[0] if matches else None


def _find_downstream(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_dataset(dataset_id, row)]
    return matches[0] if matches else None


def _find_top_lr(dataset_id: str, method: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("method") == method and _matches_dataset(dataset_id, row)]
    if not matches:
        return None
    matches.sort(key=lambda row: int(float(row.get("rank_by_score_mean") or 10**9)))
    return matches[0]


def _matches_dataset(dataset_id: str, row: dict[str, str]) -> bool:
    dataset = row.get("dataset", "")
    source = row.get("_source_artifact", "")
    prediction_path = row.get("prediction_path", "")
    text = " ".join([dataset, source, prediction_path])
    if dataset_id in REAL_BACKGROUND_DATASET_NAMES:
        return dataset == REAL_BACKGROUND_DATASET_NAMES[dataset_id] or REAL_BACKGROUND_DATASET_NAMES[dataset_id] in text
    if dataset_id == "high_resolution_tonsil_subset":
        return "tonsil_baseline_matched_fresh_subset" in text or "constructed_replicate" in text
    return False


def _validation_context(dataset_id: str) -> str:
    if dataset_id == "high_resolution_tonsil_subset":
        return "high_resolution_tonsil_prior_controls"
    return "real_background_injected_controls_and_curated_downstream"


def _count_class(dataset_id: str, method: str, rows: list[dict[str, str]], positive: bool) -> str:
    if dataset_id != "high_resolution_tonsil_subset":
        return ""
    wanted = "tonsil_positive_prior" if positive else "housekeeping_negative_control"
    return str(sum(1 for row in rows if row.get("method") == method and row.get("control_class") == wanted))


def _top_name(top_lr: dict[str, str] | None, control: dict[str, str] | None) -> str:
    if top_lr:
        return _first_nonempty(top_lr.get("canonical_pair", ""), top_lr.get("lr_id", ""))
    return _first_nonempty(_get(control, "best_positive_control"), _get(control, "top_cell_type_pairs"))


def _representative_source(
    dataset_id: str,
    method: str,
    top_lr: dict[str, str] | None,
    control: dict[str, str] | None,
) -> str:
    if dataset_id == "high_resolution_tonsil_subset":
        source = _get(top_lr, "_source_artifact") or _get(control, "_source_artifact")
        return source
    return _get(control, "prediction_path")


def _limitations(dataset_id: str, method: str, control: dict[str, str] | None, downstream: dict[str, str] | None) -> str:
    notes: list[str] = []
    if not control:
        notes.append("control_validation_missing")
    if not downstream and dataset_id != "high_resolution_tonsil_subset":
        notes.append("curated_downstream_missing")
    if dataset_id == "high_resolution_tonsil_subset":
        notes.append("prior_controls_only_no_full_truth")
    if method in {"cellchat", "spatalk"}:
        notes.append("cell_type_pair_projection_adapter")
    if method == "cellnest":
        notes.append("adapter_light_or_missing_for_high_res")
    return ";".join(notes)


def _get(row: dict[str, str] | None, key: str) -> str:
    return row.get(key, "") if row else ""


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
    complete = [row for row in rows if row["status"] == "complete"]
    return {
        "biological_validation_index_csv": _rel(path),
        "n_rows": len(rows),
        "n_complete_rows": len(complete),
        "status_counts": _counts(row["status"] for row in rows),
        "validation_context_counts": _counts(row["validation_context"] for row in rows),
        "spatialself_complete_contexts": sum(1 for row in complete if row["method"] == "spatialself_wavelet_edge_prior"),
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
