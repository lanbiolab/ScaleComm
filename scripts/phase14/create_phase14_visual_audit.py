#!/usr/bin/env python
"""Create Phase 14 visual audit for Phase 13 main figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase14"


AUDIT_ROWS = [
    {
        "figure": "Figure 1",
        "current_role": "method and benchmark entry figure",
        "scientific_message_score_1_5": 3,
        "visual_hierarchy_score_1_5": 2,
        "readability_score_1_5": 2,
        "journal_feel_score_1_5": 2,
        "main_problem": "Architecture panel still reads like an engineering flow chart; benchmark panel is table-like; ablation is dense.",
        "hero_panel": "a: refined architecture schematic with LR identity, graph-wavelet scale, hard negatives and CCC score",
        "keep": "core claim, benchmark mechanisms, weak-signal/FDR metrics, ablation source data",
        "redesign": "all panels",
        "move_to_supplement": "detailed ablation variants and component audit caveats",
        "remove": "small explanatory footnotes inside panels",
        "merge": "benchmark mechanisms and stressors into a schematic/cartoon rather than separate table-like summary",
        "needs_new_layout": True,
        "phase14_action": "redraw as schematic-led composite with one large architecture hero and three compact evidence panels",
    },
    {
        "figure": "Figure 2",
        "current_role": "baseline comparison and hard-negative suppression",
        "scientific_message_score_1_5": 4,
        "visual_hierarchy_score_1_5": 2,
        "readability_score_1_5": 2,
        "journal_feel_score_1_5": 2,
        "main_problem": "Too many baseline audit concepts compete; deep-status table makes the main figure feel like a report page.",
        "hero_panel": "c: weak-signal versus FDR tradeoff",
        "keep": "formal AUPR, weak/FDR tradeoff, high-expression FP suppression, runtime, deep baseline status",
        "redesign": "method landscape, tradeoff, FP suppression",
        "move_to_supplement": "full deep baseline status table and adapter caveats",
        "remove": "repeated method-status labels from the main visual",
        "merge": "AUPR and hard-negative suppression into a compact two-metric comparison",
        "needs_new_layout": True,
        "phase14_action": "redraw after Fig.1/Fig.3 with tradeoff as hero and deep baseline as compact status strip",
    },
    {
        "figure": "Figure 3",
        "current_role": "post-Xenium colon biological anchor",
        "scientific_message_score_1_5": 4,
        "visual_hierarchy_score_1_5": 2,
        "readability_score_1_5": 2,
        "journal_feel_score_1_5": 2,
        "main_problem": "Spatial map is useful but cramped; rank/FDR/downstream panels do not form a clean biological validation flow.",
        "hero_panel": "a-b: tissue context plus SPP1-CD44 communication hotspot",
        "keep": "SPP1-CD44 spatial map, baseline rank, local contrast FDR, target enrichment, high-expression controls",
        "redesign": "all panels",
        "move_to_supplement": "null-model audit card and verbose local contrast text",
        "remove": "dense multiline annotations inside axes",
        "merge": "negative control rank and q-value into one concise guardrail strip",
        "needs_new_layout": True,
        "phase14_action": "redraw as biological anchor with large spatial hero and four clean validation panels",
    },
    {
        "figure": "Figure 4",
        "current_role": "cross-platform immune support",
        "scientific_message_score_1_5": 3,
        "visual_hierarchy_score_1_5": 2,
        "readability_score_1_5": 2,
        "journal_feel_score_1_5": 2,
        "main_problem": "Supportive datasets are mixed with caveats; figure does not yet look like a cross-platform validation plate.",
        "hero_panel": "compact tissue/platform strip showing lymph node and tonsil as supportive validation",
        "keep": "tonsil ranking/FDR, lymph-node receiver-pairing support, cross-platform caveat",
        "redesign": "story triage and cross-platform summary",
        "move_to_supplement": "program specificity table and strict evidence details",
        "remove": "A-class/B-class wording from visual if not self-explanatory",
        "merge": "rank, FDR and support status into per-dataset mini-panels",
        "needs_new_layout": True,
        "phase14_action": "redraw after primary figures; keep as supportive validation rather than major discovery figure",
    },
    {
        "figure": "Figure 5",
        "current_role": "FDR guardrails and limitations",
        "scientific_message_score_1_5": 4,
        "visual_hierarchy_score_1_5": 2,
        "readability_score_1_5": 2,
        "journal_feel_score_1_5": 2,
        "main_problem": "Important honesty figure, but currently looks like multiple audit widgets rather than an evidence hierarchy.",
        "hero_panel": "a: FDR evidence hierarchy from whole-atlas to targeted/local guardrails",
        "keep": "whole-atlas limitation, targeted/matched null, target-decoy, coordinate control, hard-negative leakage, claim matrix",
        "redesign": "all panels",
        "move_to_supplement": "pathway leakage and detailed claim matrix rows",
        "remove": "small dense tables from main figure",
        "merge": "guardrail summaries into one hierarchy plus two quantitative panels",
        "needs_new_layout": True,
        "phase14_action": "redraw as evidence hierarchy after Fig.1/Fig.3; keep conservative language explicit",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(AUDIT_ROWS)
    audit.to_csv(OUT / "phase14_figure_visual_audit.csv", index=False)
    (OUT / "phase14_figure_visual_audit.md").write_text(render_markdown(audit), encoding="utf-8")
    print(audit.to_string(index=False))


def render_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# Phase 14 Figure Visual Audit",
        "",
        "This audit treats Phase 13 figures as mechanically complete but visually immature. The purpose is to decide what to keep, redraw, merge or move to supplement.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.extend(
            [
                f"## {row.figure}",
                "",
                f"- Current role: {row.current_role}",
                f"- Scores: message {row.scientific_message_score_1_5}/5, hierarchy {row.visual_hierarchy_score_1_5}/5, readability {row.readability_score_1_5}/5, journal feel {row.journal_feel_score_1_5}/5",
                f"- Main problem: {row.main_problem}",
                f"- Hero panel: {row.hero_panel}",
                f"- Keep: {row.keep}",
                f"- Redesign: {row.redesign}",
                f"- Move to supplement: {row.move_to_supplement}",
                f"- Merge: {row.merge}",
                f"- Phase 14 action: {row.phase14_action}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
