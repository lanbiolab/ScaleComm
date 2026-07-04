#!/usr/bin/env python
"""Summarize spatial-specificity calibration against existing Goal 1 audits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
SPATIAL_DIR = PHASE12_DIR / "spatial_specificity_permutation_audit"
MAIN_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
OUT_DIR = PHASE12_DIR / "spatial_specificity_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial = pd.read_csv(SPATIAL_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv")
    main_perm = pd.read_csv(MAIN_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv")
    dataset = build_dataset(spatial, main_perm)
    story = build_story(dataset)
    dataset.to_csv(OUT_DIR / "phase12_spatial_specificity_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_specificity_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_dataset(spatial: pd.DataFrame, main_perm: pd.DataFrame) -> pd.DataFrame:
    main_cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk_fraction",
        "all_true_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
    ]
    spatial_cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk_fraction",
        "all_true_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
        "best_true_spatial_lr",
        "best_true_spatial_fdr",
        "spatial_specificity_permutation_call",
    ]
    out = spatial[spatial_cols].merge(
        main_perm[main_cols],
        on="dataset_key",
        how="left",
        suffixes=("_spatial_specificity", "_main"),
    )
    out["delta_top10_true_vs_main"] = (
        pd.to_numeric(out["top10_true_lr_spatial_specificity"], errors="coerce")
        - pd.to_numeric(out["top10_true_lr_main"], errors="coerce")
    )
    out["delta_top10_risk_vs_main"] = (
        pd.to_numeric(out["top10_negative_or_risk_fraction_spatial_specificity"], errors="coerce")
        - pd.to_numeric(out["top10_negative_or_risk_fraction_main"], errors="coerce")
    )
    out["delta_best_true_q_vs_main"] = (
        pd.to_numeric(out["best_true_lr_fdr_spatial_specificity"], errors="coerce")
        - pd.to_numeric(out["best_true_lr_fdr_main"], errors="coerce")
    )
    out["best_true_q_gap_to_0_1"] = pd.to_numeric(
        out["best_true_lr_fdr_spatial_specificity"], errors="coerce"
    ) / 0.1
    out["dataset_decision"] = [dataset_decision(row) for row in out.itertuples(index=False)]
    return out.sort_values("dataset_key")


def build_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n = len(dataset)
    true_q = int(
        pd.to_numeric(dataset["all_true_lr_conservative_fdr_0_1_spatial_specificity"], errors="coerce")
        .fillna(0)
        .sum()
    )
    q_improved = int(pd.to_numeric(dataset["delta_best_true_q_vs_main"], errors="coerce").fillna(0).lt(0).sum())
    top10_improved = int(pd.to_numeric(dataset["delta_top10_true_vs_main"], errors="coerce").fillna(0).gt(0).sum())
    clean_or_equal_risk = int(pd.to_numeric(dataset["delta_top10_risk_vs_main"], errors="coerce").fillna(1).le(0).sum())
    best_q = float(pd.to_numeric(dataset["best_true_lr_fdr_spatial_specificity"], errors="coerce").min())
    if true_q > 0:
        call = "spatial_specificity_fdr_candidate"
    elif q_improved > 0:
        call = "spatial_specificity_shrinks_q_gap_not_closed"
    else:
        call = "spatial_specificity_not_useful"
    return pd.DataFrame(
        [
            {
                "topic": "spatial_specificity_comparison",
                "overall_call": call,
                "n_datasets": n,
                "n_true_q_0_1": true_q,
                "n_best_true_q_improved_vs_main": q_improved,
                "n_top10_true_improved_vs_main": top10_improved,
                "n_no_added_top10_risk_vs_main": clean_or_equal_risk,
                "best_true_q_min": best_q,
                "best_true_q_gap_to_0_1": best_q / 0.1,
                "claim_boundary": (
                    "Spatial-specificity post-hoc calibration can shrink selected spatial-null q gaps, "
                    "but it does not close broad whole-atlas FDR."
                ),
            }
        ]
    )


def dataset_decision(row: object) -> str:
    true_q = int(getattr(row, "all_true_lr_conservative_fdr_0_1_spatial_specificity", 0))
    delta_q = float(getattr(row, "delta_best_true_q_vs_main", 0.0))
    delta_true = float(getattr(row, "delta_top10_true_vs_main", 0.0))
    delta_risk = float(getattr(row, "delta_top10_risk_vs_main", 0.0))
    if true_q > 0 and delta_risk <= 0:
        return "candidate_fdr_support_clean_or_equal_risk"
    if delta_q < 0 and delta_true >= 0 and delta_risk <= 0:
        return "q_gap_improved_without_top10_tradeoff"
    if delta_q < 0:
        return "q_gap_improved_with_ranking_or_risk_tradeoff"
    return "no_q_gap_improvement"


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    first = story.iloc[0]
    lines = [
        "# Phase 12 Spatial-Specificity Comparison",
        "",
        f"- Overall: `{first.overall_call}`.",
        f"- True q<=0.1 discoveries: {int(first.n_true_q_0_1)}.",
        f"- Best true q: {fmt(first.best_true_q_min)} ({fmt(first.best_true_q_gap_to_0_1)}x over q<=0.1).",
        "",
        "## Dataset Decisions",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_decision}`: "
            f"top10 true {int(row.top10_true_lr_main)} -> {int(row.top10_true_lr_spatial_specificity)}, "
            f"top10 risk {fmt(row.top10_negative_or_risk_fraction_main)} -> "
            f"{fmt(row.top10_negative_or_risk_fraction_spatial_specificity)}, "
            f"best true q {fmt(row.best_true_lr_fdr_main)} -> {fmt(row.best_true_lr_fdr_spatial_specificity)} "
            f"({row.best_true_lr_spatial_specificity})."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is useful as a next-model diagnostic, not as a reporting baseline replacement.",
        "- Broad all-LR exact-score FDR remains unclosed.",
    ]
    (OUT_DIR / "phase12_spatial_specificity_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_specificity_comparison",
        "inputs": [
            rel(SPATIAL_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv"),
            rel(MAIN_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_specificity_dataset_comparison.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_story.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_comparison_status.md"),
        ],
        "claim_boundary": "Diagnostic comparison only; no broad FDR closure.",
    }
    (OUT_DIR / "phase12_spatial_specificity_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
