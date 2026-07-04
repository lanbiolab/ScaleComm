#!/usr/bin/env python
"""Build the unified Phase 3 dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_DATASET_FILES = (
    "cells.csv",
    "lr_pairs.csv",
    "candidate_interactions.csv",
)
TRUTH_FILES = (
    "ground_truth_table.csv",
    "truth_lr_pairs.csv",
    "truth_cell_type_pairs.csv",
    "truth_spatial_niches.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase3_end_to_end.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase3/end_to_end"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_cfg = config.get("manifest", {})
    csv_path = PROJECT_ROOT / manifest_cfg.get(
        "output_csv",
        str(output_dir / "phase3_dataset_manifest.csv"),
    )
    json_path = PROJECT_ROOT / manifest_cfg.get(
        "output_json",
        str(output_dir / "phase3_dataset_manifest.json"),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        _dataset_row(dataset_id, dataset_cfg)
        for dataset_id, dataset_cfg in (config.get("dataset_sources") or {}).items()
    ]
    rows = sorted(rows, key=lambda row: (not row["required"], row["dataset_id"]))
    fieldnames = _fieldnames(config)
    _write_csv(csv_path, rows, fieldnames)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary = _summary(config, rows, csv_path, json_path)
    summary_path = output_dir / "phase3_dataset_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote Phase 3 dataset manifest: {csv_path}")
    print(f"Wrote Phase 3 dataset manifest JSON: {json_path}")
    print(json.dumps(summary, indent=2))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 3 config loading.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataset_row(dataset_id: str, dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / str(dataset_cfg.get("path", ""))
    required = bool(dataset_cfg.get("required", False))
    exists = path.exists()
    file_status = _file_status(path) if exists else "dataset_path_missing"
    has_ground_truth = exists and all((path / item).exists() for item in TRUTH_FILES)
    candidate_columns = _csv_header(path / "candidate_interactions.csv") if exists else []
    lr_columns = _csv_header(path / "lr_pairs.csv") if exists else []
    cells_columns = _csv_header(path / "cells.csv") if exists else []
    available_labels = _available_labels(has_ground_truth, candidate_columns, lr_columns, cells_columns)
    has_injected_controls = _has_injected_controls(path, candidate_columns, lr_columns) if exists else False
    n_cells = _count_csv_rows(path / "cells.csv") if exists else 0
    n_lr_pairs = _count_csv_rows(path / "lr_pairs.csv") if exists else 0
    n_candidate_edges = _count_csv_rows(path / "candidate_interactions.csv") if exists else 0
    n_truth_edges = _sum_int_column(path / "ground_truth_table.csv", "true_edge") if has_ground_truth else 0
    n_weak_truth_edges = _sum_truth_where(path / "ground_truth_table.csv", "true_edge", "weak_signal") if has_ground_truth else 0
    n_high_expression_negative_controls = _count_truth_where(
        path / "ground_truth_table.csv",
        "false_positive_high_expression",
    ) if has_ground_truth else _count_lr_controls(path / "lr_pairs.csv")
    metadata = _load_json(path / "metadata.json") if exists and (path / "metadata.json").exists() else {}
    dataset_context = metadata.get("benchmark", "")
    if not dataset_context and str(dataset_cfg.get("dataset_type", "")).startswith("optional"):
        dataset_context = "optional_placeholder_not_available"
    return {
        "dataset_id": dataset_id,
        "dataset_type": str(dataset_cfg.get("dataset_type", "")),
        "tissue": str(dataset_cfg.get("tissue", "")),
        "path": str(path.relative_to(PROJECT_ROOT)) if _inside_project(path) else str(path),
        "required": required,
        "exists": exists,
        "dataset_status": "ready" if exists and file_status == "complete" else ("missing_optional" if not required else file_status),
        "file_status": file_status,
        "n_cells": n_cells,
        "n_lr_pairs": n_lr_pairs,
        "n_candidate_edges": n_candidate_edges,
        "has_ground_truth": has_ground_truth,
        "has_injected_controls": has_injected_controls,
        "available_labels": ";".join(available_labels),
        "intended_figure_panel": str(dataset_cfg.get("intended_figure_panel", "")),
        "n_truth_edges": n_truth_edges,
        "n_weak_truth_edges": n_weak_truth_edges,
        "n_high_expression_negative_control_edges": n_high_expression_negative_controls,
        "dataset_context": str(dataset_context),
        "benchmark_version": str(metadata.get("benchmark_version", "")),
        "notes": "optional placeholder" if not required and not exists else "",
    }


def _file_status(path: Path) -> str:
    missing = [item for item in REQUIRED_DATASET_FILES if not (path / item).exists()]
    if missing:
        return "missing_required_files:" + ",".join(missing)
    return "complete"


def _available_labels(
    has_ground_truth: bool,
    candidate_columns: list[str],
    lr_columns: list[str],
    cells_columns: list[str],
) -> list[str]:
    labels: list[str] = []
    if has_ground_truth:
        labels.extend(["candidate_truth", "lr_pair_truth", "cell_type_pair_truth", "spatial_niche_truth"])
    for column in (
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
        "positive_control",
        "negative_control",
        "region",
        "niche",
        "cell_type",
    ):
        if column in candidate_columns or column in lr_columns or column in cells_columns:
            labels.append(column)
    return sorted(set(labels))


def _has_injected_controls(path: Path, candidate_columns: list[str], lr_columns: list[str]) -> bool:
    for csv_path, columns in ((path / "candidate_interactions.csv", candidate_columns), (path / "lr_pairs.csv", lr_columns)):
        if "false_positive_high_expression" in columns and _any_truthy(csv_path, "false_positive_high_expression"):
            return True
        if "weak_signal" in columns and _any_truthy(csv_path, "weak_signal"):
            return True
    return False


def _csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _sum_int_column(path: Path, column: str) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            return 0
        for row in reader:
            total += int(float(row.get(column) or 0))
    return total


def _sum_truth_where(path: Path, value_column: str, mask_column: str) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if value_column not in (reader.fieldnames or []) or mask_column not in (reader.fieldnames or []):
            return 0
        for row in reader:
            if _truthy(row.get(mask_column, "")):
                total += int(float(row.get(value_column) or 0))
    return total


def _count_truth_where(path: Path, mask_column: str) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if mask_column not in (reader.fieldnames or []):
            return 0
        for row in reader:
            total += int(_truthy(row.get(mask_column, "")))
    return total


def _count_lr_controls(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "false_positive_high_expression" not in (reader.fieldnames or []):
            return 0
        return sum(1 for row in reader if _truthy(row.get("false_positive_high_expression", "")))


def _any_truthy(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            return False
        return any(_truthy(row.get(column, "")) for row in reader)


def _truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _inside_project(path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _fieldnames(config: dict[str, Any]) -> list[str]:
    required = list((config.get("manifest") or {}).get("required_columns") or [])
    extras = [
        "tissue",
        "required",
        "exists",
        "dataset_status",
        "file_status",
        "n_truth_edges",
        "n_weak_truth_edges",
        "n_high_expression_negative_control_edges",
        "dataset_context",
        "benchmark_version",
        "notes",
    ]
    return list(dict.fromkeys(required + extras))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summary(config: dict[str, Any], rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> dict[str, Any]:
    required_rows = [row for row in rows if row["required"]]
    ready_required = [row for row in required_rows if row["dataset_status"] == "ready"]
    return {
        "experiment_id": config.get("experiment_id", ""),
        "claim_level": config.get("claim_level", ""),
        "manifest_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "manifest_json": str(json_path.relative_to(PROJECT_ROOT)),
        "n_datasets": len(rows),
        "n_required_datasets": len(required_rows),
        "n_ready_required_datasets": len(ready_required),
        "all_required_ready": len(required_rows) == len(ready_required),
        "n_optional_placeholders": sum(1 for row in rows if not row["required"]),
        "representative_method": config.get("representative_method", ""),
        "metric_methods": (config.get("retained_methods") or {}).get("metric_methods", []),
        "status_methods": (config.get("retained_methods") or {}).get("status_methods", []),
        "fdr_null_models": (config.get("fdr") or {}).get("null_models", []),
    }


if __name__ == "__main__":
    main()
