#!/usr/bin/env python
"""Audit Phase 12 disk usage and cleanup candidates.

The script is intentionally read-only. It identifies large output directories,
large files, duplicate replicate inputs that may be hard-link deduplicated, and
generated figure artifacts that can be regenerated from source tables/scripts.
It does not delete, compress, or rewrite any project output.
"""

from __future__ import annotations

import filecmp
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "disk_cleanup_audit"


DUPLICATE_REPLICATE_PAIRS = [
    (
        PHASE12_DIR / "atlas_model_expanded/replicates",
        PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training/replicates",
        "expanded_vs_integrated_replicates",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = directory_usage_rows(PHASE12_DIR, max_depth=2)
    files = large_file_rows(PHASE12_DIR, min_bytes=50 * 1024 * 1024)
    duplicates = duplicate_replicate_rows()
    candidates = cleanup_candidate_rows(dirs, files, duplicates)

    pd.DataFrame(dirs).to_csv(OUT_DIR / "phase12_disk_directory_usage.csv", index=False)
    pd.DataFrame(files).to_csv(OUT_DIR / "phase12_disk_large_files.csv", index=False)
    pd.DataFrame(duplicates).to_csv(OUT_DIR / "phase12_disk_duplicate_replicate_candidates.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT_DIR / "phase12_disk_cleanup_candidates.csv", index=False)
    write_status(dirs, files, duplicates, candidates)
    manifest = {
        "task": "phase12_disk_cleanup_audit",
        "read_only": True,
        "outputs": [
            rel(OUT_DIR / "phase12_disk_directory_usage.csv"),
            rel(OUT_DIR / "phase12_disk_large_files.csv"),
            rel(OUT_DIR / "phase12_disk_duplicate_replicate_candidates.csv"),
            rel(OUT_DIR / "phase12_disk_cleanup_candidates.csv"),
            rel(OUT_DIR / "phase12_disk_cleanup_audit_status.md"),
        ],
        "claim_boundary": "Read-only disk audit. It identifies cleanup candidates but does not delete or rewrite outputs.",
    }
    (OUT_DIR / "phase12_disk_cleanup_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(candidates).to_string(index=False))


def directory_usage_rows(root: Path, max_depth: int) -> list[dict[str, object]]:
    rows = []
    for path in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(rel_parts) > max_depth:
            continue
        size = path_size(path)
        rows.append(
            {
                "path": rel(path),
                "depth": len(rel_parts),
                "size_bytes": int(size),
                "size_mib": round(size / (1024 * 1024), 3),
            }
        )
    return sorted(rows, key=lambda x: int(x["size_bytes"]), reverse=True)


def large_file_rows(root: Path, min_bytes: int) -> list[dict[str, object]]:
    rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_bytes:
            continue
        rows.append(
            {
                "path": rel(path),
                "size_bytes": int(size),
                "size_mib": round(size / (1024 * 1024), 3),
                "suffix": path.suffix,
            }
        )
    return sorted(rows, key=lambda x: int(x["size_bytes"]), reverse=True)


def duplicate_replicate_rows() -> list[dict[str, object]]:
    rows = []
    for left_root, right_root, group in DUPLICATE_REPLICATE_PAIRS:
        if not left_root.exists() or not right_root.exists():
            continue
        for left in left_root.rglob("*"):
            if not left.is_file():
                continue
            rel_path = left.relative_to(left_root)
            right = right_root / rel_path
            if not right.exists() or not right.is_file():
                continue
            try:
                left_stat = left.stat()
                right_stat = right.stat()
            except OSError:
                continue
            same_inode = (left_stat.st_ino == right_stat.st_ino) and (left_stat.st_dev == right_stat.st_dev)
            same_size = left_stat.st_size == right_stat.st_size
            identical = same_inode or (same_size and filecmp.cmp(left, right, shallow=False))
            if identical:
                rows.append(
                    {
                        "candidate_group": group,
                        "primary_path": rel(left),
                        "duplicate_path": rel(right),
                        "size_bytes": int(right_stat.st_size),
                        "size_mib": round(right_stat.st_size / (1024 * 1024), 3),
                        "already_hardlinked": bool(same_inode),
                        "recommended_action": "hardlink_duplicate_to_primary" if not same_inode else "already_deduplicated",
                        "risk": "low_for_static_inputs_but_requires_no_inplace_editing",
                    }
                )
    return sorted(rows, key=lambda x: int(x["size_bytes"]), reverse=True)


def cleanup_candidate_rows(
    dirs: list[dict[str, object]],
    files: list[dict[str, object]],
    duplicates: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    duplicate_savings = sum(
        int(row["size_bytes"]) for row in duplicates if not bool(row.get("already_hardlinked", False))
    )
    if duplicate_savings:
        rows.append(
            {
                "candidate": "hardlink_duplicate_expanded_replicate_inputs",
                "estimated_savings_bytes": int(duplicate_savings),
                "estimated_savings_mib": round(duplicate_savings / (1024 * 1024), 3),
                "action_type": "deduplicate_by_hardlink",
                "requires_user_approval": True,
                "rationale": "atlas_model_expanded and integrated_guarded_spatial_training replicate inputs contain identical large CSV files.",
                "risk": "Do only if replicate inputs are treated as immutable; hardlinked files share storage.",
            }
        )
    figure_dirs = [row for row in dirs if str(row["path"]).startswith("outputs/phase12/figures")]
    if figure_dirs:
        size = max(int(row["size_bytes"]) for row in figure_dirs)
        rows.append(
            {
                "candidate": "archive_or_regenerate_rendered_figure_artifacts",
                "estimated_savings_bytes": int(size),
                "estimated_savings_mib": round(size / (1024 * 1024), 3),
                "action_type": "manual_archive_or_delete_regenerable_outputs",
                "requires_user_approval": True,
                "rationale": "Rendered PNG/TIFF/PDF/SVG drafts are large compared with source tables.",
                "risk": "Do not delete unless figure rendering scripts and source tables are confirmed sufficient to regenerate.",
            }
        )
    large_csv = [row for row in files if str(row["suffix"]).lower() == ".csv"]
    if large_csv:
        size = sum(int(row["size_bytes"]) for row in large_csv)
        rows.append(
            {
                "candidate": "compress_or_deduplicate_large_csv_inputs",
                "estimated_savings_bytes": int(size),
                "estimated_savings_mib": round(size / (1024 * 1024), 3),
                "action_type": "manual_compress_or_repoint_inputs",
                "requires_user_approval": True,
                "rationale": "Large candidate_interactions.csv files dominate Phase 12 storage.",
                "risk": "Compression can break scripts expecting plain CSV; prefer hardlinking identical static inputs first.",
            }
        )
    return rows


def write_status(
    dirs: list[dict[str, object]],
    files: list[dict[str, object]],
    duplicates: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> None:
    total = path_size(PHASE12_DIR)
    duplicate_savings = sum(
        int(row["size_bytes"]) for row in duplicates if not bool(row.get("already_hardlinked", False))
    )
    top_dirs = dirs[:10]
    lines = [
        "# Phase 12 Disk Cleanup Audit",
        "",
        "## Scope",
        "",
        "Read-only audit. No files were deleted, compressed, hardlinked, or rewritten.",
        "",
        "## Summary",
        "",
        f"- Phase 12 total size: {total / (1024 * 1024):.1f} MiB",
        f"- Large files >=50 MiB: {len(files)}",
        f"- Duplicate replicate files detected: {len(duplicates)}",
        f"- Potential duplicate-input hardlink savings: {duplicate_savings / (1024 * 1024):.1f} MiB",
        "",
        "## Largest Directories",
        "",
    ]
    for row in top_dirs:
        lines.append(f"- `{row['path']}`: {float(row['size_mib']):.1f} MiB")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This audit only identifies candidates. Any deletion, hardlinking, or compression should be treated as a separate action.",
        ]
    )
    (OUT_DIR / "phase12_disk_cleanup_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def path_size(path: Path) -> int:
    total = 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    for current_root, _, files in os.walk(path):
        root_path = Path(current_root)
        for name in files:
            try:
                total += int((root_path / name).stat().st_size)
            except OSError:
                continue
    return total


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
