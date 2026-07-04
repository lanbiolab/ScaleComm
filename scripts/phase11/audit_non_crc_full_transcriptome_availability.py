#!/usr/bin/env python
"""Audit local availability of native non-CRC full-transcriptome ST inputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story"

RAW_MATRIX_GLOBS = [
    "data/**/*.h5",
    "data/**/*.h5ad",
    "data/**/*.zarr.zip",
]

STANDARDIZED_EXPRESSION_GLOBS = [
    "outputs/phase6/expanded_lr_universe/datasets/*/expression.csv",
    "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/*/expression.csv",
]

TISSUE_HINTS = {
    "crc": ["crc", "colon_cancer", "p2crc"],
    "colon": ["colon", "post_xenium"],
    "lymph_node": ["lymph", "hlymphnode"],
    "tonsil": ["tonsil", "spamosaic", "slide_tags"],
    "melanoma": ["melanoma"],
    "lung": ["lung"],
}

PLATFORM_HINTS = {
    "visium_hd": ["visium_hd"],
    "visium": ["visium", "cytassist"],
    "xenium": ["xenium"],
    "slide_tags": ["slide_tags"],
    "spamosaic": ["spamosaic"],
    "cosmx": ["cosmx"],
    "multiome": ["multiome"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pattern in RAW_MATRIX_GLOBS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            rows.append(audit_matrix_path(path, source_class="raw_or_downloaded_matrix"))

    for pattern in STANDARDIZED_EXPRESSION_GLOBS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if is_null_input(path):
                continue
            rows.append(audit_expression_csv(path, source_class="standardized_or_constructed_expression"))

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=output_columns())
    else:
        table = table[output_columns()]
        table = table.sort_values(
            ["native_non_crc_full_transcriptome_candidate", "source_class", "dataset_label", "path"],
            ascending=[False, True, True, True],
        )

    audit_path = output_dir / "non_crc_full_transcriptome_availability_audit.csv"
    table.to_csv(audit_path, index=False)

    summary = build_summary(table, audit_path)
    summary_path = output_dir / "non_crc_full_transcriptome_availability_summary.md"
    summary_path.write_text(render_markdown(summary, table), encoding="utf-8")
    json_path = output_dir / "non_crc_full_transcriptome_availability_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(table.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


def audit_matrix_path(path: Path, source_class: str) -> dict[str, Any]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    suffix = "".join(path.suffixes).lower()
    dataset_label = infer_dataset_label(path)
    n_obs = None
    n_genes = None
    read_status = "unread"
    notes = []

    try:
        if suffix.endswith(".h5"):
            n_obs, n_genes, notes = read_10x_h5_shape(path)
            read_status = "ok"
        elif suffix.endswith(".h5ad"):
            n_obs, n_genes = read_h5ad_shape(path)
            read_status = "ok"
        elif suffix.endswith(".zarr.zip"):
            n_obs, n_genes, notes = inspect_zarr_zip(path)
            read_status = "metadata_only" if n_genes is None else "ok"
        else:
            read_status = "unsupported"
    except Exception as exc:  # pragma: no cover - audit should continue on corrupt optional files.
        read_status = f"failed:{type(exc).__name__}"
        notes.append(str(exc)[:240])

    return classify_row(
        path=rel,
        source_class=source_class,
        dataset_label=dataset_label,
        platform=infer_platform(rel),
        tissue=infer_tissue(rel),
        n_obs=n_obs,
        n_genes=n_genes,
        read_status=read_status,
        notes="; ".join(notes),
    )


def audit_expression_csv(path: Path, source_class: str) -> dict[str, Any]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    dataset_dir = path.parent
    dataset_label = dataset_dir.name
    n_obs = count_csv_rows(path)
    n_genes = count_expression_genes(path)
    profile_notes = []
    manifest_scope = ""

    for candidate in sorted(dataset_dir.glob("*manifest.json")):
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifest_scope = str(manifest.get("scope") or manifest.get("dataset") or manifest.get("platform") or "")
        if manifest_scope:
            profile_notes.append(f"manifest_scope={manifest_scope}")
            break

    return classify_row(
        path=rel,
        source_class=source_class,
        dataset_label=dataset_label,
        platform=infer_platform(rel),
        tissue=infer_tissue(rel),
        n_obs=n_obs,
        n_genes=n_genes,
        read_status="ok",
        notes="; ".join(profile_notes),
    )


def read_10x_h5_shape(path: Path) -> tuple[int | None, int | None, list[str]]:
    notes: list[str] = []
    with h5py.File(path, "r") as handle:
        if "matrix" in handle:
            matrix = handle["matrix"]
            shape = matrix["shape"][()]
            n_genes = int(shape[0])
            n_obs = int(shape[1])
            feature_type_counts = {}
            if "feature_type" in matrix.get("features", {}):
                values = decode_array(matrix["features"]["feature_type"][()])
                for value in values:
                    feature_type_counts[value] = feature_type_counts.get(value, 0) + 1
                if feature_type_counts:
                    notes.append(f"feature_types={feature_type_counts}")
            return n_obs, n_genes, notes
        if "X" in handle:
            n_obs, n_genes = read_h5ad_shape(path)
            return n_obs, n_genes, notes
        keys = ",".join(handle.keys())
        notes.append(f"unknown_h5_keys={keys}")
    return None, None, notes


def read_h5ad_shape(path: Path) -> tuple[int | None, int | None]:
    with h5py.File(path, "r") as handle:
        if "X" not in handle:
            return None, None
        x = handle["X"]
        if hasattr(x, "shape") and len(x.shape) == 2:
            return int(x.shape[0]), int(x.shape[1])
        if isinstance(x, h5py.Group) and "shape" in x:
            shape = x["shape"][()]
            return int(shape[0]), int(shape[1])
    return None, None


def inspect_zarr_zip(path: Path) -> tuple[int | None, int | None, list[str]]:
    notes = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        notes.append(f"zip_entries={len(names)}")
        panel_matches = [name for name in names if "gene_panel" in name.lower()]
        if panel_matches:
            notes.append(f"contains_gene_panel={panel_matches[0]}")
    return None, None, notes


def decode_array(values: Any) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def count_expression_genes(path: Path) -> int | None:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    gene_columns = [column for column in header if column not in {"cell_id", "spot_id", "barcode"}]
    return len(gene_columns)


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def classify_row(
    *,
    path: str,
    source_class: str,
    dataset_label: str,
    platform: str,
    tissue: str,
    n_obs: int | None,
    n_genes: int | None,
    read_status: str,
    notes: str,
) -> dict[str, Any]:
    is_crc_or_colon = tissue in {"crc", "colon"}
    full_tx = bool(n_genes is not None and n_genes >= 10000)
    broad_tx = bool(n_genes is not None and 1000 <= n_genes < 10000)
    targeted = bool(n_genes is not None and n_genes < 1000)
    raw_native = source_class == "raw_or_downloaded_matrix"
    native_non_crc_full = bool(raw_native and full_tx and not is_crc_or_colon)
    if native_non_crc_full:
        classification = "native_non_crc_full_transcriptome_candidate"
    elif raw_native and full_tx and is_crc_or_colon:
        classification = "native_crc_or_colon_full_transcriptome"
    elif broad_tx:
        classification = "broader_transcriptome_not_native_full"
    elif targeted:
        classification = "targeted_panel_or_small_expression"
    elif raw_native and read_status.startswith("metadata_only"):
        classification = "raw_metadata_without_expression_shape"
    else:
        classification = "unclassified_or_unreadable"

    return {
        "dataset_label": dataset_label,
        "path": path,
        "source_class": source_class,
        "platform": platform,
        "tissue": tissue,
        "n_obs": n_obs,
        "n_genes": n_genes,
        "read_status": read_status,
        "transcriptome_scope_class": classification,
        "native_non_crc_full_transcriptome_candidate": native_non_crc_full,
        "notes": notes,
    }


def build_summary(table: pd.DataFrame, audit_path: Path) -> dict[str, Any]:
    candidates = table[table["native_non_crc_full_transcriptome_candidate"].astype(bool)] if not table.empty else table
    raw_full_crc = table[table["transcriptome_scope_class"].eq("native_crc_or_colon_full_transcriptome")] if not table.empty else table
    broad_non_crc = table[
        table["transcriptome_scope_class"].eq("broader_transcriptome_not_native_full")
        & ~table["tissue"].isin(["crc", "colon"])
    ] if not table.empty else table
    targeted_non_crc = table[
        table["transcriptome_scope_class"].eq("targeted_panel_or_small_expression")
        & ~table["tissue"].isin(["crc", "colon"])
    ] if not table.empty else table
    return {
        "audit_path": audit_path.relative_to(PROJECT_ROOT).as_posix(),
        "n_entries": int(len(table)),
        "n_native_non_crc_full_transcriptome_candidates": int(len(candidates)),
        "native_non_crc_full_transcriptome_available": bool(len(candidates) > 0),
        "native_crc_or_colon_full_transcriptome_entries": raw_full_crc["path"].tolist(),
        "non_crc_broader_transcriptome_entries": broad_non_crc["path"].tolist(),
        "non_crc_targeted_or_small_entries": targeted_non_crc["path"].tolist(),
        "claim_boundary": (
            "Native non-CRC full-transcriptome ST validation is available locally only when the candidate count is "
            "greater than zero. Broader-transcriptome and targeted lymph node/tonsil/melanoma outputs remain useful "
            "supporting evidence, but should not be described as native full-transcriptome validation."
        ),
    }


def render_markdown(summary: dict[str, Any], table: pd.DataFrame) -> str:
    lines = [
        "# Non-CRC native full-transcriptome availability audit",
        "",
        f"- Entries audited: {summary['n_entries']}",
        f"- Native non-CRC full-transcriptome candidates: {summary['n_native_non_crc_full_transcriptome_candidates']}",
        f"- Native non-CRC full-transcriptome available locally: {summary['native_non_crc_full_transcriptome_available']}",
        "",
        "## Interpretation",
        "",
        summary["claim_boundary"],
        "",
        "## Local full-transcriptome entries",
        "",
    ]
    full = table[table["transcriptome_scope_class"].str.contains("full_transcriptome", na=False)]
    if full.empty:
        lines.append("No local full-transcriptome matrix was identified.")
    else:
        for _, row in full.iterrows():
            lines.append(
                f"- `{row['path']}`: {row['transcriptome_scope_class']}, "
                f"tissue={row['tissue']}, platform={row['platform']}, "
                f"n_obs={row['n_obs']}, n_genes={row['n_genes']}"
            )
    lines.extend(["", "## Non-CRC non-full-transcriptome entries", ""])
    non_crc_non_full = table[
        ~table["tissue"].isin(["crc", "colon"])
        & ~table["transcriptome_scope_class"].str.contains("full_transcriptome", na=False)
    ]
    if non_crc_non_full.empty:
        lines.append("No non-CRC non-full-transcriptome entries were identified.")
    else:
        for _, row in non_crc_non_full.iterrows():
            lines.append(
                f"- `{row['path']}`: {row['transcriptome_scope_class']}, "
                f"tissue={row['tissue']}, platform={row['platform']}, "
                f"n_obs={row['n_obs']}, n_genes={row['n_genes']}"
            )
    lines.append("")
    return "\n".join(lines)


def output_columns() -> list[str]:
    return [
        "dataset_label",
        "path",
        "source_class",
        "platform",
        "tissue",
        "n_obs",
        "n_genes",
        "read_status",
        "transcriptome_scope_class",
        "native_non_crc_full_transcriptome_candidate",
        "notes",
    ]


def infer_dataset_label(path: Path) -> str:
    lower_parts = [part.lower() for part in path.parts]
    for part in reversed(lower_parts):
        if part not in {"raw", "data", "optional", "outputs", "extracted"} and not part.endswith((".h5", ".h5ad", ".zip")):
            return part
    return path.stem


def infer_tissue(text: str) -> str:
    lower = text.lower()
    for tissue, hints in TISSUE_HINTS.items():
        if any(hint in lower for hint in hints):
            return tissue
    return "unknown"


def infer_platform(text: str) -> str:
    lower = text.lower()
    for platform, hints in PLATFORM_HINTS.items():
        if any(hint in lower for hint in hints):
            return platform
    return "unknown"


def is_null_input(path: Path) -> bool:
    return "null_inputs" in path.parts


if __name__ == "__main__":
    main()
