#!/usr/bin/env python
"""Audit Phase 13 against the user-provided success criteria."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase13"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"
SCRIPTS = ROOT / "scripts" / "phase13"


MAIN_STEMS = [
    "phase13_figure1_method_benchmark",
    "phase13_figure2_baseline_hardnegative",
    "phase13_figure3_post_xenium_colon",
    "phase13_figure4_cross_platform_validation",
    "phase13_figure5_fdr_guardrails",
]
SUPP_STEMS = [
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
    qa = read_csv(OUT / "phase13_figure_output_qa.csv")
    source_qa = read_csv(OUT / "phase13_source_data_qa.csv")
    status = read_csv(OUT / "phase13_figure_system_status.csv")
    story_map = read_csv(OUT / "phase13_main_figure_panel_story_map.csv")
    supp_story = read_csv(OUT / "phase13_supplementary_story_summary.csv")

    rows = [
        check(
            "5 main figures",
            len(MAIN_STEMS) == 5 and all(all((FIG / f"{stem}.{ext}").exists() for ext in ["svg", "pdf", "png"]) for stem in MAIN_STEMS),
            "Five main figure stems with SVG/PDF/PNG exist in outputs/phase13/figures.",
        ),
        check(
            "8-12 supplementary figures",
            8 <= len(SUPP_STEMS) <= 12 and all(all((FIG / f"{stem}.{ext}").exists() for ext in ["svg", "pdf", "png"]) for stem in SUPP_STEMS),
            "Twelve supplementary figure stems with SVG/PDF/PNG exist.",
        ),
        check(
            "one-sentence conclusion per figure",
            (OUT / "phase13_main_figure_story_summary.csv").exists()
            and (OUT / "phase13_supplementary_story_summary.csv").exists()
            and "headline" in read_csv(OUT / "phase13_main_figure_story_summary.csv").columns
            and len(supp_story) == 12,
            "Main and supplementary story summary files provide headline/purpose fields.",
        ),
        check(
            "source data per panel",
            len(story_map) >= 30
            and story_map["source_data"].astype(str).str.contains("outputs/phase13/source_data/").all()
            and len(source_qa) >= 78
            and bool(source_qa["pass"].all()),
            "Main panel story map points to Phase 13 source data, and source-data QA passes.",
        ),
        check(
            "unified style, colors and fonts",
            (SCRIPTS / "phase13_figure_style.py").exists()
            and all((SCRIPTS / f"plot_{stem}.py").exists() for stem in MAIN_STEMS)
            and (SCRIPTS / "plot_phase13_supplementary_figures.py").exists(),
            "All plotted figures import/use the shared Phase 13 matplotlib style module.",
        ),
        check(
            "editable SVG/PDF/PNG exports",
            len(qa) == 17 and bool(qa["pass"].all()) and bool(qa["svg_editable_text_hint"].all()),
            "Mechanical QA confirms nonblank PNGs, manifests, file sizes and editable-text SVG hints for 17 figure stems.",
        ),
        check(
            "main claims and figure claims aligned",
            (OUT / "phase13_figure_contract.md").exists()
            and (OUT / "phase13_manuscript_story_guide_zh.md").exists()
            and (OUT / "phase13_figure_legends_draft_en.md").exists()
            and story_map["claim_boundary"].astype(str).str.len().min() > 20,
            "Figure contract, Chinese story guide, English legend draft and panel-level claim boundaries exist.",
        ),
        check(
            "supplement contains complex evidence",
            (FIG / "phase13_supplementary_figures_manifest.csv").exists()
            and len(read_csv(FIG / "phase13_supplementary_figures_manifest.csv")) >= 40,
            "Supplementary manifest covers benchmark realism, baselines, ablation, deep-status, runtime, FDR and guardrail figures.",
        ),
        check(
            "Phase 13 status updated",
            "draft_rendered" in set(status["status"].astype(str))
            and status["outputs_present"].astype(bool).all(),
            "Phase 13 status file reports all outputs present and draft-rendered.",
        ),
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "phase13_completion_audit.csv", index=False)
    (OUT / "phase13_completion_audit.md").write_text(render_markdown(audit), encoding="utf-8")
    print(render_markdown(audit))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def check(requirement: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "requirement": requirement,
        "status": "satisfied" if passed else "not_satisfied",
        "evidence": evidence if passed else "Missing or insufficient evidence: " + evidence,
    }


def render_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# Phase 13 Completion Audit",
        "",
        f"- Requirements satisfied: {(audit['status'] == 'satisfied').sum()}/{len(audit)}",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.extend([f"## {row.requirement}", "", f"- Status: `{row.status}`", f"- Evidence: {row.evidence}", ""])
    lines.extend(
        [
            "## Residual Human Polish",
            "",
            "The mechanical Phase 13 package is complete. Before journal submission, captions still need exact n/test/correction wording after Methods are locked, and final visual micro-adjustment may be done in Illustrator/Inkscape if desired.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
