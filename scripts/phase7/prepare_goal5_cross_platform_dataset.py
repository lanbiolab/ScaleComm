#!/usr/bin/env python
"""Prepare or audit Phase 7 cross-platform real datasets.

This script is intentionally conservative: by default it performs readiness
checks and writes a schema contract. Full conversion is implemented only for
simple CSV tables that already expose cell ids, coordinates, expression, and
annotations. Platform-specific raw bundles are reported with exact missing
pieces instead of guessed conversions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

STANDARD_FILES = {
    "cells": "cells.csv",
    "expression": "expression.csv",
    "lr_pairs": "lr_pairs.csv",
    "candidate_interactions": "candidate_interactions.csv",
}

REQUIRED_COLUMNS = {
    "cells": ["cell_id", "x", "y", "cell_type", "region", "niche"],
    "expression": ["cell_id"],
    "lr_pairs": ["lr_id", "ligand", "receptor", "mechanism", "true_lr"],
    "candidate_interactions": [
        "candidate_id",
        "sender_id",
        "receiver_id",
        "lr_id",
        "ligand",
        "receptor",
        "mechanism",
        "distance",
        "sender_cell_type",
        "receiver_cell_type",
        "sender_niche",
        "receiver_niche",
        "weak_signal",
        "false_positive_high_expression",
        "ligand_expr",
        "receptor_expr",
    ],
}

PLATFORM_FILE_HINTS = {
    "xenium_human_lymph_node": {
        "raw_candidates": [
            "cell_feature_matrix.h5",
            "cells.csv.gz",
            "cells.parquet",
            "transcripts.parquet",
            "analysis.zarr.zip",
        ],
        "annotation_candidates": [
            "cell_types.csv",
            "cell_annotations.csv",
            "analysis/clustering/gene_expression_graphclust/clusters.csv",
        ],
    },
    "visium_hd_crc": {
        "raw_candidates": [
            "binned_outputs",
            "filtered_feature_bc_matrix.h5",
            "filtered_feature_bc_matrix",
            "spatial/tissue_positions.parquet",
            "spatial/tissue_positions.csv",
        ],
        "annotation_candidates": [
            "cell_types.csv",
            "spot_annotations.csv",
            "region_annotations.csv",
        ],
    },
    "cosmx_melanoma": {
        "raw_candidates": [
            "exprMat_file.csv",
            "exprMat_file.csv.gz",
            "metadata.csv",
            "fov_positions_file.csv",
            "cell_metadata.csv",
        ],
        "annotation_candidates": [
            "cell_metadata.csv",
            "metadata.csv",
            "cell_types.csv",
        ],
    },
    "merfish_or_seqfish_or_slideseqv2": {
        "raw_candidates": [
            "adata.h5ad",
            "counts.h5ad",
            "cell_by_gene.csv",
            "cell_metadata.csv",
            "coordinates.csv",
        ],
        "annotation_candidates": [
            "cell_metadata.csv",
            "annotations.csv",
            "clusters.csv",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORM_FILE_HINTS))
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs",
    )
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = PROJECT_ROOT / args.output_root
    dataset_id = args.dataset_id or args.platform
    dataset_out = output_root / dataset_id
    dataset_out.mkdir(parents=True, exist_ok=True)

    readiness = _readiness(args.platform, input_root, dataset_out)
    contract = _schema_contract(args.platform, input_root, dataset_out, args.dry_run)

    GOAL5_DIR.mkdir(parents=True, exist_ok=True)
    readiness_path = GOAL5_DIR / f"{dataset_id}__standardization_readiness.csv"
    contract_path = GOAL5_DIR / f"{dataset_id}__standardization_contract.json"
    pd.DataFrame([readiness]).to_csv(readiness_path, index=False)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    if not args.dry_run and readiness["standard_schema_present"]:
        _copy_standard_schema(input_root, dataset_out)
        readiness["conversion_status"] = "copied_existing_standard_schema"
        pd.DataFrame([readiness]).to_csv(readiness_path, index=False)

    print(pd.DataFrame([readiness]).to_string(index=False))
    print(json.dumps({"readiness": _rel(readiness_path), "contract": _rel(contract_path)}, indent=2))


def _readiness(platform: str, input_root: Path, output_root: Path) -> dict[str, Any]:
    standard_status = _standard_status(input_root)
    hints = PLATFORM_FILE_HINTS[platform]
    raw_matches = _match_hints(input_root, hints["raw_candidates"])
    annotation_matches = _match_hints(input_root, hints["annotation_candidates"])
    invalid_raw_files = _invalid_raw_files(input_root)
    missing_standard = [
        file_name for key, file_name in STANDARD_FILES.items() if not standard_status[f"{key}_exists"]
    ]
    if all(standard_status[f"{key}_valid"] for key in STANDARD_FILES):
        status = "ready_standard_schema"
        action = "run SpatialSelf/baseline/FDR workflow directly"
    elif invalid_raw_files and not raw_matches:
        status = "input_root_exists_with_invalid_download_placeholders"
        action = "redownload raw expression, metadata, and coordinate files with valid access token or provide standardized CSV exports"
    elif raw_matches:
        status = "raw_or_partial_platform_bundle_found"
        action = "write platform-specific parser or provide standardized CSV exports"
    elif input_root.exists():
        status = "input_root_exists_but_no_recognized_bundle"
        action = "inspect directory and add file hints or standardized CSV files"
    else:
        status = "input_root_missing"
        action = "download dataset and rerun readiness check"
    return {
        "platform": platform,
        "input_root": _rel(input_root),
        "output_root": _rel(output_root),
        "input_root_exists": input_root.exists(),
        "standard_schema_present": all(standard_status[f"{key}_exists"] for key in STANDARD_FILES),
        "standard_schema_valid": all(standard_status[f"{key}_valid"] for key in STANDARD_FILES),
        "readiness_status": status,
        "recommended_next_action": action,
        "missing_standard_files": ";".join(missing_standard),
        "raw_platform_file_matches": ";".join(raw_matches[:50]),
        "annotation_file_matches": ";".join(annotation_matches[:50]),
        "invalid_raw_files": ";".join(invalid_raw_files[:50]),
        **standard_status,
    }


def _standard_status(input_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, filename in STANDARD_FILES.items():
        path = input_root / filename
        out[f"{key}_exists"] = path.exists()
        missing_cols: list[str] = []
        n_rows: int | str = ""
        if path.exists():
            try:
                frame = pd.read_csv(path, nrows=25)
                missing_cols = [col for col in REQUIRED_COLUMNS[key] if col not in frame.columns]
                n_rows = _count_rows(path)
            except Exception as exc:  # pragma: no cover - defensive audit output
                missing_cols = [f"read_error:{type(exc).__name__}"]
        else:
            missing_cols = REQUIRED_COLUMNS[key]
        out[f"{key}_valid"] = path.exists() and not missing_cols
        out[f"{key}_missing_columns"] = ";".join(missing_cols)
        out[f"{key}_n_rows"] = n_rows
    return out


def _match_hints(root: Path, hints: list[str]) -> list[str]:
    if not root.exists():
        return []
    matches: list[str] = []
    lowered_hints = [hint.lower() for hint in hints]
    for path in root.rglob("*"):
        rel = _rel(path)
        lower = rel.lower()
        if any(part in lower for part in ("__pycache__", ".git", "site-packages")):
            continue
        if any(hint in lower for hint in lowered_hints):
            matches.append(rel)
        if len(matches) >= 100:
            break
    return sorted(set(matches))


def _invalid_raw_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    invalid: list[str] = []
    markers = [
        "unauthorized",
        "must have current bearer token",
        "access denied",
        "forbidden",
        "not found",
    ]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 4096:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip().lower()
        except OSError:
            continue
        if any(marker in text for marker in markers):
            invalid.append(_rel(path))
    return sorted(invalid)


def _schema_contract(platform: str, input_root: Path, output_root: Path, dry_run: bool) -> dict[str, Any]:
    return {
        "platform": platform,
        "input_root": _rel(input_root),
        "output_root": _rel(output_root),
        "dry_run": bool(dry_run),
        "standard_schema": {
            "cells.csv": {
                "required_columns": REQUIRED_COLUMNS["cells"],
                "notes": "cell_type/region/niche may be derived from annotations; use 'unknown' only for dry-run smoke, not final validation.",
            },
            "expression.csv": {
                "required_columns": REQUIRED_COLUMNS["expression"],
                "notes": "wide cell-by-gene table with one row per cell/spot/bin; columns after cell_id are gene symbols.",
            },
            "lr_pairs.csv": {
                "required_columns": REQUIRED_COLUMNS["lr_pairs"],
                "notes": "for real data, true_lr can be blank/unknown unless using injected semi-synthetic truth.",
            },
            "candidate_interactions.csv": {
                "required_columns": REQUIRED_COLUMNS["candidate_interactions"],
                "notes": "candidate edges should be generated after choosing LR panel and spatial range rules.",
            },
        },
        "platform_raw_file_hints": PLATFORM_FILE_HINTS[platform],
        "claim_boundary": "readiness/standardization contract only; not model performance evidence until predictions, controls, downstream, and FDR are run.",
    }


def _copy_standard_schema(input_root: Path, output_root: Path) -> None:
    for filename in STANDARD_FILES.values():
        source = input_root / filename
        target = output_root / filename
        target.write_bytes(source.read_bytes())


def _count_rows(path: Path) -> int | str:
    try:
        with path.open("rb") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return ""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
