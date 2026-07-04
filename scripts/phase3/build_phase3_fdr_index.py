#!/usr/bin/env python
"""Build a Phase 3 FDR/permutation artifact index."""

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
    "method",
    "null_model",
    "tier",
    "n_permutations",
    "status",
    "candidate_edge_fdr_path",
    "lr_pair_fdr_path",
    "cell_type_pair_fdr_path",
    "significant_edges_path",
    "significant_lr_pairs_path",
    "significant_cell_type_pairs_path",
    "n_significant_edges",
    "n_significant_lr_pairs",
    "n_significant_cell_type_pairs",
    "min_candidate_bh_fdr",
    "min_lr_pair_bh_fdr",
    "min_cell_type_pair_bh_fdr",
    "p_value_ks_uniform_stat",
    "source_summary_artifact",
    "source_group_artifact",
    "limitations",
]

DATASET_KEYS = {
    "controlled_simulator_v2": "replicate_00",
    "real_background_lymph_node": "human_lymph_node_visium_replicate_00",
    "real_background_tonsil": "human_tonsil_slide_tags_replicate_00",
    "real_background_melanoma": "human_melanoma_multiome_replicate_00",
    "high_resolution_tonsil_subset": "constructed_replicate",
}

REAL_BACKGROUND_NAMES = {
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
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase3/end_to_end")
    manifest_path = PROJECT_ROOT / (
        args.manifest
        or (config.get("manifest") or {}).get("output_csv", "outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    )
    output_path = PROJECT_ROOT / (args.output or str(output_dir / "phase3_fdr_index.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = [row for row in _read_csv(manifest_path) if _truthy(row.get("required")) and row.get("dataset_status") == "ready"]
    null_models = list((config.get("fdr") or {}).get("null_models") or [])
    summary_rows, group_rows = _load_fdr_sources()
    rows: list[dict[str, Any]] = []
    for dataset in manifest:
        for null_model in null_models:
            rows.append(_fdr_row(dataset, null_model, summary_rows, group_rows))

    _write_csv(output_path, rows, OUT_COLUMNS)
    summary = _summary(rows, output_path)
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 FDR index: {output_path}")
    print(json.dumps(summary, indent=2))


def _load_fdr_sources() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/fdr_smoke_missing/phase3_missing_fdr_summary.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal5_formal_fdr/goal5_formal_fdr_summary.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal5_fdr_200perm/goal5_formal_fdr_summary.csv",
    ]
    group_paths = [
        PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_group_summary.csv",
        PROJECT_ROOT / "outputs/phase3/end_to_end/fdr_smoke_missing/phase3_missing_fdr_group_summary.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal5_formal_fdr/goal5_formal_fdr_group_summary.csv",
        PROJECT_ROOT / "outputs/phase2_revised/goal5_fdr_200perm/goal5_formal_fdr_group_summary.csv",
    ]
    return _load_rows(summary_paths), _load_rows(group_paths)


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


def _fdr_row(
    dataset: dict[str, str],
    null_model: str,
    summary_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
) -> dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    dataset_key = DATASET_KEYS.get(dataset_id, dataset_id)
    match = _find_summary(dataset_id, null_model, summary_rows)
    lr_group = _find_group(dataset_id, null_model, "lr_pair", group_rows)
    ct_group = _find_group(dataset_id, null_model, "cell_type_pair", group_rows)
    if not match:
        return _missing_row(dataset_id, dataset_key, null_model)
    return {
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "method": "spatialself_wavelet_edge_prior",
        "null_model": null_model,
        "tier": _tier(match),
        "n_permutations": match.get("n_permutations", ""),
        "status": "complete",
        "candidate_edge_fdr_path": _normalize_path(match.get("candidate_edge_fdr_path", "")),
        "lr_pair_fdr_path": _normalize_path(match.get("lr_pair_direct_fdr_path", "")),
        "cell_type_pair_fdr_path": _normalize_path(match.get("cell_type_pair_direct_fdr_path", "")),
        "significant_edges_path": _normalize_path(match.get("significant_spatial_edges_path", "")),
        "significant_lr_pairs_path": _significant_path(match.get("lr_pair_direct_fdr_path", ""), "lr_pairs"),
        "significant_cell_type_pairs_path": _significant_path(match.get("cell_type_pair_direct_fdr_path", ""), "cell_type_pairs"),
        "n_significant_edges": match.get("n_significant_edges", ""),
        "n_significant_lr_pairs": _first_nonempty(match.get("n_significant_lr_pairs_direct", ""), match.get("n_significant_lr_pairs", "")),
        "n_significant_cell_type_pairs": match.get("n_significant_cell_type_pairs_direct", ""),
        "min_candidate_bh_fdr": match.get("min_bh_fdr", ""),
        "min_lr_pair_bh_fdr": lr_group.get("min_group_bh_fdr", "") if lr_group else "",
        "min_cell_type_pair_bh_fdr": ct_group.get("min_group_bh_fdr", "") if ct_group else "",
        "p_value_ks_uniform_stat": match.get("p_value_ks_uniform_stat", ""),
        "source_summary_artifact": match.get("_source_artifact", ""),
        "source_group_artifact": _first_nonempty(_get(lr_group, "_source_artifact"), _get(ct_group, "_source_artifact")),
        "limitations": _limitations(dataset_id, match),
    }


def _find_summary(dataset_id: str, null_model: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    expected_dataset = REAL_BACKGROUND_NAMES.get(dataset_id, dataset_id)
    matches = [
        row
        for row in rows
        if row.get("dataset") == expected_dataset
        and row.get("null_model") == null_model
        and row.get("method") == "spatialself_wavelet_edge_prior"
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: int(float(row.get("n_permutations") or 0)), reverse=True)
    return matches[0]


def _find_group(dataset_id: str, null_model: str, group_level: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    expected_dataset = REAL_BACKGROUND_NAMES.get(dataset_id, dataset_id)
    matches = [
        row
        for row in rows
        if row.get("dataset") == expected_dataset
        and row.get("null_model") == null_model
        and row.get("group_level") == group_level
        and row.get("method") == "spatialself_wavelet_edge_prior"
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: int(float(row.get("n_permutations") or 0)) if row.get("n_permutations") else 0, reverse=True)
    return matches[0]


def _missing_row(dataset_id: str, dataset_key: str, null_model: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "method": "spatialself_wavelet_edge_prior",
        "null_model": null_model,
        "tier": "",
        "n_permutations": "",
        "status": "missing_or_not_run",
        "candidate_edge_fdr_path": "",
        "lr_pair_fdr_path": "",
        "cell_type_pair_fdr_path": "",
        "significant_edges_path": "",
        "significant_lr_pairs_path": "",
        "significant_cell_type_pairs_path": "",
        "n_significant_edges": "",
        "n_significant_lr_pairs": "",
        "n_significant_cell_type_pairs": "",
        "min_candidate_bh_fdr": "",
        "min_lr_pair_bh_fdr": "",
        "min_cell_type_pair_bh_fdr": "",
        "p_value_ks_uniform_stat": "",
        "source_summary_artifact": "",
        "source_group_artifact": "",
        "limitations": "FDR not yet run for this Phase 3 dataset",
    }


def _tier(row: dict[str, str]) -> str:
    n_perm = int(float(row.get("n_permutations") or 0))
    if n_perm >= 100:
        return "representative_100perm"
    if n_perm >= 20:
        return "smoke_20perm_or_higher"
    return "smoke"


def _limitations(dataset_id: str, row: dict[str, str]) -> str:
    n_perm = int(float(row.get("n_permutations") or 0))
    notes = ["representative_panel_not_final_500perm"] if n_perm >= 100 else ["smoke_20perm_not_final_calibration"]
    if row.get("n_significant_edges") in {"0", "0.0"}:
        notes.append("candidate_edge_discoveries_absent_at_alpha_0p05")
    if dataset_id in REAL_BACKGROUND_NAMES:
        notes.append("real_background_spatialself_only")
    if dataset_id == "high_resolution_tonsil_subset":
        notes.append("high_resolution_controls_no_full_truth")
    return ";".join(notes)


def _significant_path(path_value: str, level: str) -> str:
    if not path_value:
        return ""
    text = _normalize_path(path_value)
    if level == "lr_pairs":
        return text.replace("lr_pair_direct_fdr.csv", "significant_lr_pairs.csv")
    return text.replace("cell_type_pair_direct_fdr.csv", "significant_cell_type_pairs.csv")


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
        "fdr_index_csv": _rel(path),
        "n_rows": len(rows),
        "n_complete_rows": len(complete),
        "status_counts": _counts(row["status"] for row in rows),
        "dataset_counts": _counts(row["dataset_id"] for row in rows),
        "null_model_counts": _counts(row["null_model"] for row in rows),
        "complete_real_background_100perm_rows": sum(1 for row in complete if row["tier"] == "representative_100perm"),
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
