#!/usr/bin/env python
"""Run publication-oriented QA checks for Phase 2 figure assembly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MIN_PNG_BYTES = 50_000
MIN_PDF_BYTES = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", default="outputs/phase2/goal6_paper_figures")
    parser.add_argument("--output-dir", default="outputs/phase2/goal6_paper_figures/figure_qa")
    args = parser.parse_args()

    figure_dir = PROJECT_ROOT / args.figure_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = figure_dir / "goal6_figure_assembly_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_qa = _file_qa(manifest)
    source_trace = _source_trace(manifest)
    summary = _summary_row(file_qa, source_trace, manifest, figure_dir, output_dir)

    file_qa.to_csv(output_dir / "goal6_figure_file_qa.csv", index=False)
    source_trace.to_csv(output_dir / "goal6_figure_source_trace_qa.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "goal6_figure_qa_summary.csv", index=False)
    (output_dir / "goal6_figure_qa_manifest.json").write_text(
        json.dumps(
            {
                "goal_id": "6",
                "analysis_label": "goal6_figure_qa",
                "evidence_status": summary["evidence_status"],
                "figure_dir": str(figure_dir.relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "inputs": {
                    "figure_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                    "source_tables": manifest.get("source_tables", {}),
                    "figures": manifest.get("figures", []),
                    "manuscript_panels": manifest.get("manuscript_panels", []),
                },
                "outputs": {
                    "summary": str((output_dir / "goal6_figure_qa_summary.csv").relative_to(PROJECT_ROOT)),
                    "file_qa": str((output_dir / "goal6_figure_file_qa.csv").relative_to(PROJECT_ROOT)),
                    "source_trace": str((output_dir / "goal6_figure_source_trace_qa.csv").relative_to(PROJECT_ROOT)),
                },
                "limitations": [
                    "Automated QA checks file presence, source traceability, export pairing, and obvious size risks.",
                    "It does not replace manual visual inspection for label overlap, final journal sizing, or exact typography.",
                    "Current assembly exports PNG/PDF but not SVG/TIFF; editable-vector risk is recorded explicitly.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Wrote Goal 6 figure QA outputs to {output_dir}")


def _file_qa(manifest: dict) -> pd.DataFrame:
    paths = []
    for group, values in [
        ("overview", manifest.get("figures", [])),
        ("manuscript_panel", manifest.get("manuscript_panels", [])),
    ]:
        if isinstance(values, list):
            for value in values:
                paths.append((group, PROJECT_ROOT / value))
    rows = []
    for group, path in paths:
        exists = path.exists()
        suffix = path.suffix.lower().lstrip(".")
        size_bytes = path.stat().st_size if exists else 0
        min_bytes = MIN_PNG_BYTES if suffix == "png" else MIN_PDF_BYTES if suffix == "pdf" else 1
        stem = path.with_suffix("")
        paired_path = None
        if suffix == "png":
            paired_path = stem.with_suffix(".pdf")
        elif suffix == "pdf":
            paired_path = stem.with_suffix(".png")
        pair_exists = bool(paired_path and paired_path.exists())
        rows.append(
            {
                "group": group,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "stem": path.stem,
                "format": suffix,
                "exists": exists,
                "size_bytes": int(size_bytes),
                "minimum_size_bytes": int(min_bytes),
                "size_pass": bool(size_bytes > min_bytes),
                "paired_format_path": str(paired_path.relative_to(PROJECT_ROOT)) if paired_path else "",
                "paired_format_exists": pair_exists,
                "export_risk": _export_risk(suffix),
            }
        )
    return pd.DataFrame(rows)


def _source_trace(manifest: dict) -> pd.DataFrame:
    source_tables = manifest.get("source_tables", {})
    if not isinstance(source_tables, dict):
        source_tables = {}
    claim_trace_path = PROJECT_ROOT / source_tables.get("goal6_claim_traceability", "")
    claim_trace = pd.read_csv(claim_trace_path) if claim_trace_path.exists() else pd.DataFrame()
    rows = []
    for key, value in sorted(source_tables.items()):
        path = PROJECT_ROOT / value
        exists = path.exists()
        n_rows = int(pd.read_csv(path).shape[0]) if exists else 0
        referenced_by_claim = False
        if not claim_trace.empty and "evidence_artifact" in claim_trace:
            referenced_by_claim = value in set(claim_trace["evidence_artifact"].astype(str))
        rows.append(
            {
                "source_key": key,
                "source_path": value,
                "exists": exists,
                "n_rows": n_rows,
                "non_empty": n_rows > 0,
                "referenced_by_claim_trace": referenced_by_claim,
                "traceability_scope": (
                    "claim-traced" if referenced_by_claim else "supporting-source-table"
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(
    file_qa: pd.DataFrame,
    source_trace: pd.DataFrame,
    manifest: dict,
    figure_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    n_overview = int((file_qa["group"] == "overview").sum()) if not file_qa.empty else 0
    n_panel_files = int((file_qa["group"] == "manuscript_panel").sum()) if not file_qa.empty else 0
    n_png = int((file_qa["format"] == "png").sum()) if not file_qa.empty else 0
    n_pdf = int((file_qa["format"] == "pdf").sum()) if not file_qa.empty else 0
    n_svg = int((file_qa["format"] == "svg").sum()) if not file_qa.empty else 0
    n_tiff = int(file_qa["format"].isin(["tif", "tiff"]).sum()) if not file_qa.empty else 0
    n_missing = int((~file_qa["exists"]).sum()) if not file_qa.empty else 0
    n_small = int((~file_qa["size_pass"]).sum()) if not file_qa.empty else 0
    n_unpaired = int((~file_qa["paired_format_exists"]).sum()) if not file_qa.empty else 0
    n_sources = len(source_trace)
    n_source_missing = int((~source_trace["exists"]).sum()) if not source_trace.empty else 0
    n_source_empty = int((~source_trace["non_empty"]).sum()) if not source_trace.empty else 0
    n_claim_traced = int(source_trace["referenced_by_claim_trace"].sum()) if not source_trace.empty else 0
    hard_pass = (
        n_overview >= 2
        and n_panel_files >= 30
        and n_png >= 16
        and n_pdf >= 16
        and n_missing == 0
        and n_small == 0
        and n_unpaired == 0
        and n_source_missing == 0
        and n_source_empty == 0
        and n_claim_traced >= 14
    )
    export_bundle_status = "png_pdf_ready_svg_tiff_missing" if n_svg == 0 and n_tiff == 0 else "mixed_exports"
    return {
        "goal_id": "6",
        "analysis_label": "goal6_figure_qa",
        "evidence_status": "figure_qa_complete" if hard_pass else "figure_qa_has_warnings",
        "qa_pass": bool(hard_pass),
        "figure_dir": str(figure_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "n_overview_files": n_overview,
        "n_manuscript_panel_files": n_panel_files,
        "n_png_files": n_png,
        "n_pdf_files": n_pdf,
        "n_svg_files": n_svg,
        "n_tiff_files": n_tiff,
        "n_missing_figure_files": n_missing,
        "n_small_figure_files": n_small,
        "n_unpaired_panel_files": n_unpaired,
        "n_source_tables": n_sources,
        "n_missing_source_tables": n_source_missing,
        "n_empty_source_tables": n_source_empty,
        "n_claim_traced_source_tables": n_claim_traced,
        "export_bundle_status": export_bundle_status,
        "manual_review_required": True,
        "interpretation": (
            "Automated Goal 6 figure QA passes file/source traceability checks and records a remaining export risk: "
            "the current figure bundle is PNG/PDF-only, so final manuscript packaging still needs SVG or TIFF export "
            "and manual visual review for label overlap and journal sizing."
        ),
    }


def _export_risk(suffix: str) -> str:
    if suffix == "pdf":
        return "vector_pdf_export"
    if suffix == "png":
        return "raster_preview_export"
    if suffix in {"svg", "tif", "tiff"}:
        return "submission_export_format"
    return "unknown_export_format"


if __name__ == "__main__":
    main()
