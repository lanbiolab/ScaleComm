#!/usr/bin/env python
"""Summarize rendered Phase 12 manuscript figure drafts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "outputs" / "phase12" / "figures"


FIGURES = [
    ("Figure 1", "phase12_figure1_expanded_atlas_audit", "Expanded whole-LR atlas audit"),
    ("Figure 2", "phase12_figure2_crc_spp1_cd44_story", "SPP1-CD44 CRC/post-Xenium story"),
    ("Figure 3", "phase12_figure3_second_tissue_validation", "Second-tissue validation"),
    ("Figure 4", "phase12_figure4_hard_negative_guardrails", "Hard-negative and claim guardrails"),
]


def inspect_figure(figure_id: str, stem: str, description: str) -> dict[str, object]:
    paths = {suffix: FIGDIR / f"{stem}.{suffix}" for suffix in ["svg", "pdf", "png", "tiff"]}
    manifest = FIGDIR / f"{stem}_manifest.csv"
    svg_text_count = None
    svg_keeps_text = False
    if paths["svg"].exists():
        text = paths["svg"].read_text(errors="ignore")
        svg_text_count = text.count("<text")
        svg_keeps_text = svg_text_count > 0
    return {
        "figure_id": figure_id,
        "description": description,
        "stem": stem,
        "svg_exists": paths["svg"].exists(),
        "pdf_exists": paths["pdf"].exists(),
        "png_exists": paths["png"].exists(),
        "tiff_exists": paths["tiff"].exists(),
        "manifest_exists": manifest.exists(),
        "svg_text_count": svg_text_count,
        "svg_keeps_editable_text": svg_keeps_text,
        "all_expected_outputs_exist": all(p.exists() for p in paths.values()) and manifest.exists(),
        "primary_svg_path": str(paths["svg"]),
        "primary_png_path": str(paths["png"]),
    }


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = [inspect_figure(*item) for item in FIGURES]
    df = pd.DataFrame(rows)
    df.to_csv(FIGDIR / "phase12_rendered_figure_status.csv", index=False)

    complete = bool(df["all_expected_outputs_exist"].all() and df["svg_keeps_editable_text"].all())
    lines = [
        "# Phase 12 Rendered Figure Status",
        "",
        f"- Rendered figure drafts: {int(df['all_expected_outputs_exist'].sum())}/{len(df)}",
        f"- Editable SVG text check: {int(df['svg_keeps_editable_text'].sum())}/{len(df)}",
        f"- Goal 6 render status: {'draft_figures_rendered' if complete else 'incomplete_render_outputs'}",
        "",
        "## Figure Outputs",
        "",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"- {row.figure_id}: {row.description}; outputs={row.all_expected_outputs_exist}; "
            f"editable_svg={row.svg_keeps_editable_text}; stem=`{row.stem}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- These are manuscript draft figures, not final evidence that Phase 12 is complete.",
            "- Remaining Phase 12 gaps include formal whole-atlas permutation FDR support, fully tuned official end-to-end baselines, and end-to-end model reruns under coordinate permutations.",
        ]
    )
    (FIGDIR / "phase12_rendered_figure_status.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
