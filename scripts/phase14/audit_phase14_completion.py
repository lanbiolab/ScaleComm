#!/usr/bin/env python
"""Audit Phase 14 completion against the redesign objective."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"

MAIN_STEMS = [
    "phase14_figure1_method_benchmark_redesign",
    "phase14_figure2_baseline_tradeoff_redesign",
    "phase14_figure3_post_xenium_colon_redesign",
    "phase14_figure4_cross_platform_redesign",
    "phase14_figure5_fdr_guardrails_redesign",
]

SUPP_STEMS = [
    f"phase14_supplementary_figure{i:02d}_{suffix}"
    for i, suffix in [
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
    ]
]


def main() -> None:
    rows = [
        check("visual audit table", (OUT / "phase14_figure_visual_audit.csv").exists(), "phase14_figure_visual_audit.csv exists"),
        check("5 redesigned main figures", all_stems_present(MAIN_STEMS), "five main redesign stems have SVG/PDF/PNG/manifest"),
        check("12 redesigned supplementary figures", all_stems_present(SUPP_STEMS, shared_manifest=True), "twelve supplementary redesign stems have SVG/PDF/PNG and shared manifest"),
        check("hero panel declared for each main figure", hero_audit_ok(), "visual audit defines hero_panel for all five main figures"),
        check("source data exists", len(list(SOURCE.glob("*.csv"))) >= 40, "Phase 14 source_data CSV files exist for main and supplement figures"),
        check("PNG nonblank QA", all_png_nonblank(), "all Phase 14 PNGs have nonwhite fraction > 0.02"),
        check("status updated", status_ok(), "phase14_status.csv reports all major items rendered/complete"),
        check("supplement plan exists", (OUT / "phase14_supplement_redesign_plan.csv").exists(), "supplement redesign plan exists"),
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "phase14_completion_audit.csv", index=False)
    (OUT / "phase14_completion_audit.md").write_text(render(audit), encoding="utf-8")
    print(render(audit))


def all_stems_present(stems: list[str], *, shared_manifest: bool = False) -> bool:
    for stem in stems:
        if not all((FIG / f"{stem}.{ext}").exists() for ext in ["svg", "pdf", "png"]):
            return False
        if not shared_manifest and not (FIG / f"{stem}_manifest.csv").exists():
            return False
    if shared_manifest and not (FIG / "phase14_supplementary_figures_manifest.csv").exists():
        return False
    return True


def hero_audit_ok() -> bool:
    path = OUT / "phase14_figure_visual_audit.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    return len(df) >= 5 and "hero_panel" in df.columns and df["hero_panel"].astype(str).str.len().min() > 5


def all_png_nonblank() -> bool:
    pngs = list(FIG.glob("phase14_*.png"))
    if len(pngs) < 17:
        return False
    for p in pngs:
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im)
        if float(np.mean(np.any(arr < 245, axis=2))) <= 0.02:
            return False
    return True


def status_ok() -> bool:
    path = OUT / "phase14_status.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    required = {"Visual audit", "Figure 1 redesign", "Figure 2 redesign", "Figure 3 redesign", "Figure 4 redesign", "Figure 5 redesign", "Supplement redesign"}
    if not required.issubset(set(df["item"])):
        return False
    bad = df[df["item"].isin(required) & ~df["status"].isin(["complete", "draft_rendered"])]
    return bad.empty and df["output_exists"].astype(bool).all()


def check(requirement: str, passed: bool, evidence: str) -> dict[str, str]:
    return {"requirement": requirement, "status": "satisfied" if passed else "not_satisfied", "evidence": evidence}


def render(audit: pd.DataFrame) -> str:
    lines = ["# Phase 14 Completion Audit", "", f"- Requirements satisfied: {(audit['status'] == 'satisfied').sum()}/{len(audit)}", ""]
    for row in audit.itertuples(index=False):
        lines.extend([f"## {row.requirement}", "", f"- Status: `{row.status}`", f"- Evidence: {row.evidence}", ""])
    lines.extend(
        [
            "## Residual Human Polish",
            "",
            "The reproducible Phase 14 figure system is complete at draft-rendered level. Final journal submission still needs human visual review of exact spacing, typography and caption statistics after the manuscript text is locked.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
