#!/usr/bin/env python
"""Remove regenerable Phase 12 TIFF figure artifacts.

The script only targets TIFF files in outputs/phase12/figures that have same-
stem SVG, PDF and PNG counterparts. Default mode is dry-run. With --execute, it
writes a manifest and deletes the TIFF artifacts to recover disk space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "outputs/phase12/figures"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/disk_cleanup_audit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Delete eligible TIFF artifacts.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if args.execute:
        rows = execute_cleanup(rows)
    table = pd.DataFrame(rows)
    suffix = "executed" if args.execute else "dry_run"
    out_path = OUT_DIR / f"phase12_regenerable_tiff_cleanup_{suffix}.csv"
    status_path = OUT_DIR / f"phase12_regenerable_tiff_cleanup_{suffix}.md"
    table.to_csv(out_path, index=False)
    write_status(table, status_path, executed=bool(args.execute))
    manifest = {
        "task": "phase12_regenerable_tiff_cleanup",
        "executed": bool(args.execute),
        "figure_dir": rel(FIGURE_DIR),
        "outputs": [rel(out_path), rel(status_path)],
        "claim_boundary": (
            "Only TIFF artifacts with same-stem SVG/PDF/PNG counterparts are eligible. "
            "This is storage cleanup, not scientific evidence."
        ),
    }
    (OUT_DIR / f"phase12_regenerable_tiff_cleanup_{suffix}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for tiff in sorted(FIGURE_DIR.glob("*.tiff")):
        stem = tiff.with_suffix("")
        svg = stem.with_suffix(".svg")
        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        has_all = svg.exists() and pdf.exists() and png.exists()
        size = tiff.stat().st_size
        rows.append(
            {
                "tiff_path": rel(tiff),
                "size_bytes": int(size),
                "size_mib": round(size / (1024 * 1024), 3),
                "has_svg": bool(svg.exists()),
                "has_pdf": bool(pdf.exists()),
                "has_png": bool(png.exists()),
                "eligible_for_cleanup": bool(has_all),
                "executed_action": "",
                "error": "",
            }
        )
    return rows


def execute_cleanup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    updated = []
    for row in rows:
        item = dict(row)
        path = PROJECT_ROOT / str(item["tiff_path"])
        if not item["eligible_for_cleanup"]:
            item["executed_action"] = "skipped_missing_counterpart"
            updated.append(item)
            continue
        try:
            path.unlink()
            item["executed_action"] = "deleted_tiff_artifact"
        except Exception as exc:  # noqa: BLE001
            item["executed_action"] = "failed"
            item["error"] = repr(exc)
        updated.append(item)
    return updated


def write_status(table: pd.DataFrame, path: Path, *, executed: bool) -> None:
    eligible = table[table["eligible_for_cleanup"].fillna(False).astype(bool)] if not table.empty else table
    deleted = table[table["executed_action"].astype(str).eq("deleted_tiff_artifact")] if not table.empty else table
    failures = table[table["executed_action"].astype(str).eq("failed")] if not table.empty else table
    eligible_size = float(pd.to_numeric(eligible["size_mib"], errors="coerce").fillna(0).sum()) if len(eligible) else 0.0
    deleted_size = float(pd.to_numeric(deleted["size_mib"], errors="coerce").fillna(0).sum()) if len(deleted) else 0.0
    lines = [
        "# Phase 12 Regenerable TIFF Cleanup",
        "",
        f"- Mode: `{'execute' if executed else 'dry-run'}`",
        f"- TIFF files audited: {len(table)}",
        f"- Eligible TIFF files: {len(eligible)}",
        f"- Eligible size: {eligible_size:.1f} MiB",
        f"- Deleted TIFF files: {len(deleted)}",
        f"- Deleted size: {deleted_size:.1f} MiB",
        f"- Failures: {len(failures)}",
        "",
        "## Boundary",
        "",
        "Only TIFF artifacts with SVG/PDF/PNG counterparts are eligible. Source tables and vector/PNG/PDF figure outputs are preserved.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
