#!/usr/bin/env python
"""Compare anchor-background exact-score permutation audit with main and anchor audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
ANCHOR_DIR = PHASE12_DIR / "spatial_null_anchor_preserving_permutation_audit"
AB_DIR = PHASE12_DIR / "anchor_background_suppressed_permutation_audit"
OUT_DIR = PHASE12_DIR / "anchor_background_suppressed_permutation_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_summary(BASE_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv", "base")
    anchor = load_summary(
        ANCHOR_DIR / "phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv",
        "anchor",
    )
    anchor_bg = load_summary(
        AB_DIR / "phase12_anchor_background_suppressed_permutation_dataset_summary.csv",
        "anchor_background",
    )
    comparison = base.merge(anchor, on="dataset_key", how="outer").merge(anchor_bg, on="dataset_key", how="outer")
    add_deltas(comparison)
    story = summarize_story(comparison)
    comparison.to_csv(OUT_DIR / "phase12_anchor_background_permutation_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_anchor_background_permutation_story.csv", index=False)
    write_status(comparison, story)
    write_manifest()
    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def load_summary(path: Path, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    keep = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk_fraction",
        "all_lr_conservative_fdr_0_1",
        "all_true_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
    ]
    return table[keep].rename(columns={col: f"{prefix}_{col}" for col in keep if col != "dataset_key"})


def add_deltas(table: pd.DataFrame) -> None:
    table["delta_anchor_background_vs_base_top10_true_lr"] = (
        table["anchor_background_top10_true_lr"].fillna(0) - table["base_top10_true_lr"].fillna(0)
    )
    table["delta_anchor_background_vs_base_top10_risk"] = (
        table["anchor_background_top10_negative_or_risk_fraction"]
        - table["base_top10_negative_or_risk_fraction"]
    )
    table["delta_anchor_background_vs_base_best_true_fdr"] = (
        table["anchor_background_best_true_lr_fdr"] - table["base_best_true_lr_fdr"]
    )
    table["delta_anchor_background_vs_anchor_top10_true_lr"] = (
        table["anchor_background_top10_true_lr"].fillna(0) - table["anchor_top10_true_lr"].fillna(0)
    )
    table["delta_anchor_background_vs_anchor_top10_risk"] = (
        table["anchor_background_top10_negative_or_risk_fraction"]
        - table["anchor_top10_negative_or_risk_fraction"]
    )
    table["delta_anchor_background_vs_anchor_best_true_fdr"] = (
        table["anchor_background_best_true_lr_fdr"] - table["anchor_best_true_lr_fdr"]
    )
    table["anchor_background_fdr_closed"] = (
        pd.to_numeric(table["anchor_background_all_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(table["anchor_background_all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).gt(0)
    )
    table["comparison_call"] = np.select(
        [
            table["anchor_background_fdr_closed"],
            (table["delta_anchor_background_vs_base_top10_true_lr"] > 0)
            & (table["delta_anchor_background_vs_base_top10_risk"] <= 0)
            & (table["delta_anchor_background_vs_base_best_true_fdr"] <= 0),
            (table["delta_anchor_background_vs_base_top10_true_lr"] > 0)
            & (table["delta_anchor_background_vs_base_top10_risk"] <= 0),
        ],
        [
            "anchor_background_exact_score_fdr_supported",
            "top10_and_best_q_improved_but_fdr_not_closed",
            "top10_improved_but_best_q_not_improved",
        ],
        default="tradeoff_or_not_closed",
    )


def summarize_story(comparison: pd.DataFrame) -> pd.DataFrame:
    n_datasets = len(comparison)
    n_fdr_closed = int(comparison["anchor_background_fdr_closed"].sum())
    n_top10_improved = int(
        (
            (comparison["delta_anchor_background_vs_base_top10_true_lr"] > 0)
            & (comparison["delta_anchor_background_vs_base_top10_risk"] <= 0)
        ).sum()
    )
    n_best_q_improved = int((comparison["delta_anchor_background_vs_base_best_true_fdr"] < 0).sum())
    n_crc_top10_recovered = int(
        (
            comparison["dataset_key"].astype(str).eq("visium_hd_crc_full")
            & (comparison["delta_anchor_background_vs_base_top10_true_lr"] > 0)
        ).sum()
    )
    overall = (
        "anchor_background_exact_score_permutation_partial_support"
        if n_fdr_closed > 0
        else "anchor_background_exact_score_permutation_not_closed"
    )
    return pd.DataFrame(
        [
            {
                "topic": "anchor_background_suppressed_exact_score_permutation",
                "overall_call": overall,
                "n_datasets": n_datasets,
                "n_fdr_closed_datasets": n_fdr_closed,
                "n_top10_improved_without_risk_vs_base": n_top10_improved,
                "n_best_true_q_improved_vs_base": n_best_q_improved,
                "n_crc_top10_recovered_vs_base": n_crc_top10_recovered,
                "claim_boundary": (
                    "Anchor plus background suppression improves selected top-k diagnostics, especially lymph-node top10 "
                    "and CRC true-anchor recovery, but exact-score broad whole-atlas q<=0.1 discoveries remain absent."
                ),
            }
        ]
    )


def write_status(comparison: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor Background-Suppressed Permutation Comparison",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        "- This is useful calibration evidence, but not FDR closure.",
        "",
        "## Dataset Comparison",
        "",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.comparison_call}`: "
            f"top10 true delta vs base={fmt(row.delta_anchor_background_vs_base_top10_true_lr)}, "
            f"top10 risk delta vs base={fmt(row.delta_anchor_background_vs_base_top10_risk)}, "
            f"best true q delta vs base={fmt(row.delta_anchor_background_vs_base_best_true_fdr)}, "
            f"anchor-background true q<=0.1={int(row.anchor_background_all_true_lr_conservative_fdr_0_1)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: the combined calibration clarifies the next-model direction but still fails conservative exact-score FDR.",
        "- Unsafe wording: background-suppressed anchor training controls whole-atlas FDR.",
    ]
    (OUT_DIR / "phase12_anchor_background_permutation_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_anchor_background_suppressed_permutation_comparison",
        "inputs": [
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "spatial_null_anchor_preserving_permutation_audit/phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv",
            "anchor_background_suppressed_permutation_audit/phase12_anchor_background_suppressed_permutation_dataset_summary.csv",
        ],
        "outputs": [
            "phase12_anchor_background_permutation_dataset_comparison.csv",
            "phase12_anchor_background_permutation_story.csv",
            "phase12_anchor_background_permutation_comparison_status.md",
        ],
        "claim_boundary": "Post-hoc exact-score permutation comparison; not end-to-end FDR closure.",
    }
    (OUT_DIR / "phase12_anchor_background_permutation_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
