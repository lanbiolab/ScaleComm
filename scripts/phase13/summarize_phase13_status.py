#!/usr/bin/env python
"""Summarize Phase 13 manuscript figure-system progress."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase13"
FIG = OUT / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        figure_row(
            figure="Figure 1",
            status="draft_rendered",
            title="Method and realistic benchmark",
            stem="phase13_figure1_method_benchmark",
            claim="SpatialSelf models LR-specific, multi-scale spatial communication and improves realistic benchmark recovery.",
            next_step="Visual polish after manuscript text is drafted; replace early component audit if a final ablation table supersedes it.",
        ),
        figure_row(
            figure="Figure 2",
            status="draft_rendered",
            title="Baseline comparison and false-positive suppression",
            stem="phase13_figure2_baseline_hardnegative",
            claim="SpatialSelf prioritizes true and weak CCC while suppressing high-expression false positives better in the tradeoff sense than retained CCC baselines.",
            next_step="Visual polish after manuscript text is drafted; keep DeepTalk as status-only unless comparable LR-pair ranking output is produced.",
        ),
        figure_row(
            figure="Figure 3",
            status="draft_rendered",
            title="Post-Xenium colon biological case",
            stem="phase13_figure3_post_xenium_colon",
            claim="SpatialSelf identifies context-specific colon communication supported by local spatial enrichment and receiver transcriptional response.",
            next_step="Visual polish and possible replacement of the schematic-like spatial map if final tissue/domain annotations become available.",
        ),
        figure_row(
            figure="Figure 4",
            status="draft_rendered",
            title="Cross-platform immune communication",
            stem="phase13_figure4_cross_platform_validation",
            claim="SpatialSelf-supported immune communication is reproducible across lymph-node and tonsil contexts when presented as caveated biological consistency evidence.",
            next_step="Visual polish after manuscript text is drafted; keep lymph-node and tonsil as supportive validation rather than SpatialSelf-exclusive causal discovery.",
        ),
        figure_row(
            figure="Figure 5",
            status="draft_rendered",
            title="FDR and claim guardrails",
            stem="phase13_figure5_fdr_guardrails",
            claim="SpatialSelf reduces major false-positive modes, while strict whole-atlas FDR remains conservatively bounded and transparently audited.",
            next_step="Use this figure to police manuscript language; do not upgrade targeted or control-null evidence into unbiased whole-atlas discovery FDR.",
        ),
        supplementary_row(),
    ]
    status = pd.DataFrame(rows)
    status.to_csv(OUT / "phase13_figure_system_status.csv", index=False)
    (OUT / "phase13_figure_system_status.md").write_text(render_markdown(status), encoding="utf-8")
    print(status.to_string(index=False))


def figure_row(*, figure: str, status: str, title: str, stem: str, claim: str, next_step: str) -> dict[str, object]:
    svg = FIG / f"{stem}.svg"
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    manifest = FIG / f"{stem}_manifest.csv"
    outputs_present = all(path.exists() for path in [svg, pdf, png, manifest])
    return {
        "figure": figure,
        "status": status if outputs_present else "missing_outputs",
        "title": title,
        "claim": claim,
        "outputs_present": bool(outputs_present),
        "svg": rel(svg) if svg.exists() else "",
        "pdf": rel(pdf) if pdf.exists() else "",
        "png": rel(png) if png.exists() else "",
        "manifest": rel(manifest) if manifest.exists() else "",
        "next_step": next_step,
    }


def supplementary_row() -> dict[str, object]:
    stems = [FIG / f"phase13_supplementary_figure{i:02d}_{suffix}" for i, suffix in [
        (1, "benchmark_realism"),
        (2, "full_baseline_metrics"),
        (3, "weak_signal_sensitivity"),
        (4, "architecture_ablation"),
        (5, "deep_baseline_status"),
        (6, "runtime_scalability"),
        (7, "expanded_atlas_topk"),
        (8, "target_decoy_fdr"),
        (9, "post_xenium_colon_sources"),
        (10, "lymph_tonsil_sources"),
        (11, "hard_negative_coordinate_guardrails"),
        (12, "claim_matrix"),
    ]]
    manifest = FIG / "phase13_supplementary_figures_manifest.csv"
    outputs_present = all((Path(f"{stem}.svg").exists() and Path(f"{stem}.pdf").exists() and Path(f"{stem}.png").exists()) for stem in stems)
    outputs_present = outputs_present and manifest.exists()
    return {
        "figure": "Supplementary figure system",
        "status": "draft_rendered" if outputs_present else "missing_outputs",
        "title": "Extended benchmark, ablation, baseline and FDR panels",
        "claim": "Complex evidence is traceable in supplement without overloading the main figures.",
        "outputs_present": bool(outputs_present),
        "svg": rel(Path(f"{stems[0]}.svg")) if Path(f"{stems[0]}.svg").exists() else "",
        "pdf": rel(Path(f"{stems[0]}.pdf")) if Path(f"{stems[0]}.pdf").exists() else "",
        "png": rel(Path(f"{stems[0]}.png")) if Path(f"{stems[0]}.png").exists() else "",
        "manifest": rel(manifest) if manifest.exists() else "",
        "next_step": "Manuscript-level caption polish and possible Illustrator/Inkscape micro-adjustments after text is drafted.",
    }


def render_markdown(status: pd.DataFrame) -> str:
    lines = [
        "# Phase 13 Figure-System Status",
        "",
        "## Scope",
        "",
        "Phase 13 upgrades existing result-heavy drafts into a Nature-family manuscript figure system. It does not create new biological evidence by itself.",
        "",
        "## Current Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.extend(
            [
                f"- **{row.figure}: {row.title}**",
                f"  - Status: `{row.status}`",
                f"  - Claim: {row.claim}",
                f"  - Outputs present: {row.outputs_present}",
                f"  - Next: {row.next_step}",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Figures 1-5 and the 12-figure supplementary system have been rendered as Phase 13 drafts. The next step is manuscript-level caption polish and final visual micro-adjustment.",
        ]
    )
    return "\n".join(lines) + "\n"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
