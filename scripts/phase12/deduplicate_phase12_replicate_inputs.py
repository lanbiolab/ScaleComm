#!/usr/bin/env python
"""Hardlink-deduplicate duplicate Phase 12 replicate input files.

Default mode is dry-run. With --execute, each duplicate file in the integrated
guarded spatial replicate directory is atomically replaced by a hardlink to the
identical file in the base expanded-atlas replicate directory.

This preserves paths and file contents, but hardlinked files share storage.
Only use --execute if these replicate inputs are treated as immutable.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_REPLICATES = PHASE12_DIR / "atlas_model_expanded/replicates"
DUPLICATE_REPLICATES = PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training/replicates"
OUT_DIR = PHASE12_DIR / "disk_cleanup_audit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Replace duplicate files with hardlinks.")
    parser.add_argument(
        "--min-size-mib",
        type=float,
        default=0.0,
        help="Only consider files at least this size. Default considers all duplicate files.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_duplicate_rows(min_size_bytes=int(args.min_size_mib * 1024 * 1024))
    if args.execute:
        rows = execute_hardlink(rows)
    table = pd.DataFrame(rows)
    suffix = "executed" if args.execute else "dry_run"
    out_path = OUT_DIR / f"phase12_replicate_hardlink_dedup_{suffix}.csv"
    table.to_csv(out_path, index=False)
    status_path = OUT_DIR / f"phase12_replicate_hardlink_dedup_{suffix}.md"
    write_status(table, status_path, executed=bool(args.execute), min_size_mib=float(args.min_size_mib))
    manifest = {
        "task": "phase12_replicate_hardlink_dedup",
        "executed": bool(args.execute),
        "min_size_mib": float(args.min_size_mib),
        "base_replicates": rel(BASE_REPLICATES),
        "duplicate_replicates": rel(DUPLICATE_REPLICATES),
        "outputs": [rel(out_path), rel(status_path)],
        "claim_boundary": (
            "Dry-run unless --execute is passed. Hardlink dedup is an infrastructure action, "
            "not scientific evidence."
        ),
    }
    (OUT_DIR / f"phase12_replicate_hardlink_dedup_{suffix}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


def collect_duplicate_rows(min_size_bytes: int) -> list[dict[str, object]]:
    if not BASE_REPLICATES.exists():
        raise FileNotFoundError(BASE_REPLICATES)
    if not DUPLICATE_REPLICATES.exists():
        raise FileNotFoundError(DUPLICATE_REPLICATES)
    rows = []
    for base_path in BASE_REPLICATES.rglob("*"):
        if not base_path.is_file():
            continue
        duplicate_path = DUPLICATE_REPLICATES / base_path.relative_to(BASE_REPLICATES)
        if not duplicate_path.exists() or not duplicate_path.is_file():
            continue
        base_stat = base_path.stat()
        dup_stat = duplicate_path.stat()
        if dup_stat.st_size < min_size_bytes:
            continue
        same_inode = (base_stat.st_dev == dup_stat.st_dev) and (base_stat.st_ino == dup_stat.st_ino)
        same_size = base_stat.st_size == dup_stat.st_size
        identical = same_inode or (same_size and filecmp.cmp(base_path, duplicate_path, shallow=False))
        action = "already_hardlinked" if same_inode else "candidate_hardlink" if identical else "not_identical"
        rows.append(
            {
                "relative_path": str(base_path.relative_to(BASE_REPLICATES)),
                "base_path": rel(base_path),
                "duplicate_path": rel(duplicate_path),
                "size_bytes": int(dup_stat.st_size),
                "size_mib": round(dup_stat.st_size / (1024 * 1024), 3),
                "same_size": bool(same_size),
                "identical": bool(identical),
                "already_hardlinked": bool(same_inode),
                "planned_action": action,
                "executed_action": "",
                "error": "",
            }
        )
    return sorted(rows, key=lambda x: int(x["size_bytes"]), reverse=True)


def execute_hardlink(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    updated = []
    for row in rows:
        item = dict(row)
        if item["planned_action"] != "candidate_hardlink":
            updated.append(item)
            continue
        base_path = PROJECT_ROOT / str(item["base_path"])
        duplicate_path = PROJECT_ROOT / str(item["duplicate_path"])
        try:
            replace_with_hardlink(base_path, duplicate_path)
            item["executed_action"] = "hardlinked_to_base"
            after_base = base_path.stat()
            after_dup = duplicate_path.stat()
            item["already_hardlinked"] = bool(
                after_base.st_dev == after_dup.st_dev and after_base.st_ino == after_dup.st_ino
            )
        except Exception as exc:  # noqa: BLE001
            item["executed_action"] = "failed"
            item["error"] = repr(exc)
        updated.append(item)
    return updated


def replace_with_hardlink(base_path: Path, duplicate_path: Path) -> None:
    if not filecmp.cmp(base_path, duplicate_path, shallow=False):
        raise ValueError(f"Refusing to hardlink non-identical file: {duplicate_path}")
    parent = duplicate_path.parent
    with tempfile.NamedTemporaryFile(prefix=f".{duplicate_path.name}.", dir=parent, delete=False) as handle:
        temp_path = Path(handle.name)
    temp_path.unlink()
    os.link(base_path, temp_path)
    os.replace(temp_path, duplicate_path)


def write_status(table: pd.DataFrame, path: Path, *, executed: bool, min_size_mib: float) -> None:
    if table.empty:
        total = candidates = already = failures = 0
        savings = 0.0
    else:
        candidates = int(table["planned_action"].eq("candidate_hardlink").sum())
        already = int(table["planned_action"].eq("already_hardlinked").sum())
        failures = int(table["executed_action"].eq("failed").sum()) if "executed_action" in table else 0
        savings = float(table.loc[table["planned_action"].eq("candidate_hardlink"), "size_mib"].sum())
        total = len(table)
    lines = [
        "# Phase 12 Replicate Hardlink Dedup",
        "",
        f"- Mode: `{'execute' if executed else 'dry-run'}`",
        f"- Minimum file size: {min_size_mib:.3g} MiB",
        f"- Files audited: {total}",
        f"- Candidate hardlinks: {candidates}",
        f"- Already hardlinked: {already}",
        f"- Estimated savings from candidates: {savings:.1f} MiB",
        f"- Failures: {failures}",
        "",
        "## Boundary",
        "",
        "Hardlink dedup preserves paths and content but makes duplicate files share storage. Treat replicate inputs as immutable before execution.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
