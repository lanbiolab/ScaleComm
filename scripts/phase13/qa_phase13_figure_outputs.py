#!/usr/bin/env python
"""QA Phase 13 manuscript figure outputs.

The checks are intentionally mechanical: presence, file size, nonblank PNG
content, source-data availability and editable SVG text hints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase13"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"


EXPECTED_MAIN = [
    "phase13_figure1_method_benchmark",
    "phase13_figure2_baseline_hardnegative",
    "phase13_figure3_post_xenium_colon",
    "phase13_figure4_cross_platform_validation",
    "phase13_figure5_fdr_guardrails",
]

EXPECTED_SUPP = [
    "phase13_supplementary_figure01_benchmark_realism",
    "phase13_supplementary_figure02_full_baseline_metrics",
    "phase13_supplementary_figure03_weak_signal_sensitivity",
    "phase13_supplementary_figure04_architecture_ablation",
    "phase13_supplementary_figure05_deep_baseline_status",
    "phase13_supplementary_figure06_runtime_scalability",
    "phase13_supplementary_figure07_expanded_atlas_topk",
    "phase13_supplementary_figure08_target_decoy_fdr",
    "phase13_supplementary_figure09_post_xenium_colon_sources",
    "phase13_supplementary_figure10_lymph_tonsil_sources",
    "phase13_supplementary_figure11_hard_negative_coordinate_guardrails",
    "phase13_supplementary_figure12_claim_matrix",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [qa_stem(stem, "main") for stem in EXPECTED_MAIN]
    rows.extend(qa_stem(stem, "supplementary") for stem in EXPECTED_SUPP)
    qa = pd.DataFrame(rows)

    manifest_checks = manifest_status()
    source_checks = source_status()
    overall_pass = bool(qa["pass"].all() and manifest_checks["pass"].all() and source_checks["pass"].all())

    qa_path = OUT / "phase13_figure_output_qa.csv"
    manifest_path = OUT / "phase13_manifest_qa.csv"
    source_path = OUT / "phase13_source_data_qa.csv"
    report_path = OUT / "phase13_figure_output_qa.md"
    qa.to_csv(qa_path, index=False)
    manifest_checks.to_csv(manifest_path, index=False)
    source_checks.to_csv(source_path, index=False)
    report_path.write_text(render_report(qa, manifest_checks, source_checks, overall_pass), encoding="utf-8")
    print(render_report(qa, manifest_checks, source_checks, overall_pass))


def qa_stem(stem: str, group: str) -> dict[str, object]:
    svg = FIG / f"{stem}.svg"
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    png_stats = png_content_stats(png)
    svg_text = svg.read_text(encoding="utf-8", errors="ignore") if svg.exists() else ""
    svg_has_text = "<text" in svg_text or "font-family" in svg_text
    manifest = FIG / f"{stem}_manifest.csv"
    if group == "supplementary":
        manifest = FIG / "phase13_supplementary_figures_manifest.csv"
    checks = {
        "svg_exists": svg.exists(),
        "pdf_exists": pdf.exists(),
        "png_exists": png.exists(),
        "svg_min_size": svg.exists() and svg.stat().st_size > 5_000,
        "pdf_min_size": pdf.exists() and pdf.stat().st_size > 5_000,
        "png_min_size": png.exists() and png.stat().st_size > 50_000,
        "png_nonblank": png_stats["nonwhite_fraction"] > 0.02,
        "svg_editable_text_hint": svg_has_text,
        "manifest_exists": manifest.exists(),
    }
    return {
        "figure_stem": stem,
        "group": group,
        **checks,
        "png_width": png_stats["width"],
        "png_height": png_stats["height"],
        "png_nonwhite_fraction": png_stats["nonwhite_fraction"],
        "png_dark_fraction": png_stats["dark_fraction"],
        "pass": all(checks.values()),
    }


def png_content_stats(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"width": 0, "height": 0, "nonwhite_fraction": 0.0, "dark_fraction": 0.0}
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    return {
        "width": image.size[0],
        "height": image.size[1],
        "nonwhite_fraction": float(np.mean(np.any(arr < 245, axis=2))),
        "dark_fraction": float(np.mean(np.any(arr < 180, axis=2))),
    }


def manifest_status() -> pd.DataFrame:
    paths = [FIG / f"{stem}_manifest.csv" for stem in EXPECTED_MAIN]
    paths.append(FIG / "phase13_supplementary_figures_manifest.csv")
    rows = []
    for path in paths:
        exists = path.exists()
        n_rows = len(pd.read_csv(path)) if exists else 0
        rows.append(
            {
                "manifest": rel(path),
                "exists": exists,
                "n_rows": n_rows,
                "pass": exists and n_rows > 0,
            }
        )
    return pd.DataFrame(rows)


def source_status() -> pd.DataFrame:
    source_files = sorted(SOURCE.glob("*.csv"))
    rows = []
    for path in source_files:
        try:
            n_rows = len(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            n_rows = 0
        rows.append({"source_data": rel(path), "n_rows": n_rows, "pass": n_rows > 0})
    if not rows:
        rows.append({"source_data": "", "n_rows": 0, "pass": False})
    return pd.DataFrame(rows)


def render_report(qa: pd.DataFrame, manifests: pd.DataFrame, sources: pd.DataFrame, overall_pass: bool) -> str:
    n_main = int((qa["group"] == "main").sum())
    n_supp = int((qa["group"] == "supplementary").sum())
    n_pass = int(qa["pass"].sum())
    lines = [
        "# Phase 13 Figure Output QA",
        "",
        f"- Overall pass: `{overall_pass}`",
        f"- Figure files checked: {len(qa)} ({n_main} main, {n_supp} supplementary)",
        f"- Figure stems passing mechanical QA: {n_pass}/{len(qa)}",
        f"- Manifest files passing: {int(manifests['pass'].sum())}/{len(manifests)}",
        f"- Source-data files passing: {int(sources['pass'].sum())}/{len(sources)}",
        "",
        "## Mechanical Criteria",
        "",
        "Each figure requires SVG/PDF/PNG, nontrivial file size, nonblank PNG content, editable-text SVG hints, and a manifest.",
        "",
        "## Remaining Human QA",
        "",
        "Check final caption wording, exact statistical descriptions, and panel-level label crowding after manuscript text is drafted.",
    ]
    return "\n".join(lines) + "\n"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
