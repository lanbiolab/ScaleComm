#!/usr/bin/env python
"""Summarize Phase 14 redesign status."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"


ROWS = [
    {
        "item": "Visual audit",
        "status": "complete",
        "claim": "Phase 13 main figures were audited for message, hierarchy, readability and journal feel.",
        "primary_output": "outputs/phase14/phase14_figure_visual_audit.csv",
        "next_step": "Use the audit to guide remaining figure redesigns.",
    },
    {
        "item": "Figure 1 redesign",
        "status": "draft_rendered",
        "claim": "Figure 1 is now a schematic-led method front-door figure with compact benchmark and ablation evidence.",
        "primary_output": "outputs/phase14/figures/phase14_figure1_method_benchmark_redesign.svg",
        "next_step": "Human visual review for spacing and final caption language.",
    },
    {
        "item": "Figure 3 redesign",
        "status": "draft_rendered",
        "claim": "Figure 3 is now a biological anchor with spatial hero, rank strip, local FDR, target response and hard-negative guardrail.",
        "primary_output": "outputs/phase14/figures/phase14_figure3_post_xenium_colon_redesign.svg",
        "next_step": "Human visual review of spatial panel density and final biological wording.",
    },
    {
        "item": "Figure 2 redesign",
        "status": "draft_rendered",
        "claim": "Baseline figure is rebuilt around weak/FDR tradeoff, hard-negative suppression, runtime and deep baseline closure.",
        "primary_output": "outputs/phase14/figures/phase14_figure2_baseline_tradeoff_redesign.svg",
        "next_step": "Human visual review and possible label micro-adjustment after manuscript text is drafted.",
    },
    {
        "item": "Figure 5 redesign",
        "status": "draft_rendered",
        "claim": "FDR figure is rebuilt as an evidence hierarchy with targeted, matched-null, target-decoy, hard-negative and claim-boundary panels.",
        "primary_output": "outputs/phase14/figures/phase14_figure5_fdr_guardrails_redesign.svg",
        "next_step": "Human visual review and final wording alignment to keep whole-atlas FDR conservative.",
    },
    {
        "item": "Figure 4 redesign",
        "status": "draft_rendered",
        "claim": "Cross-platform immune support is rebuilt as a caveated supportive validation plate.",
        "primary_output": "outputs/phase14/figures/phase14_figure4_cross_platform_redesign.svg",
        "next_step": "Human visual review and final wording to avoid overclaiming mixed lymph-node support.",
    },
    {
        "item": "Supplement redesign",
        "status": "draft_rendered",
        "claim": "Supplement is rebuilt as a 12-figure supporting evidence system with cleaner statistical and guardrail plates.",
        "primary_output": "outputs/phase14/figures/phase14_supplementary_figures_manifest.csv",
        "next_step": "Human visual review and possible final micro-adjustment after main-text captions are locked.",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame(ROWS)
    status["output_exists"] = status["primary_output"].map(lambda p: bool(p) and (ROOT / p).exists())
    status.to_csv(OUT / "phase14_status.csv", index=False)
    (OUT / "phase14_status.md").write_text(render_markdown(status), encoding="utf-8")
    print(status.to_string(index=False))


def render_markdown(status: pd.DataFrame) -> str:
    lines = [
        "# Phase 14 Status",
        "",
        "Phase 14 is a visual redesign and quality-control phase. It is not complete until all main figures and supplement are redesigned and visually reviewed.",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.extend(
            [
                f"## {row.item}",
                "",
                f"- Status: `{row.status}`",
                f"- Claim: {row.claim}",
                f"- Output: {row.primary_output or 'not yet generated'}",
                f"- Output exists: {row.output_exists}",
                f"- Next: {row.next_step}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
