#!/usr/bin/env python
"""Compare anchor-preserving exact-score permutation audit with the main audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
ANCHOR_DIR = PHASE12_DIR / "spatial_null_anchor_preserving_permutation_audit"
OUT_DIR = PHASE12_DIR / "spatial_null_anchor_preserving_permutation_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv")
    anchor = pd.read_csv(ANCHOR_DIR / "phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv")
    comparison = compare_dataset(base, anchor)
    story = summarize_story(comparison)

    comparison.to_csv(
        OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_dataset_comparison.csv",
        index=False,
    )
    story.to_csv(OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_story.csv", index=False)
    write_status(comparison, story)
    write_manifest()

    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def compare_dataset(base: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    base_keep = base[
        [
            "dataset_key",
            "top10_true_lr",
            "top10_negative_or_risk_fraction",
            "top10_conservative_fdr_0_1",
            "all_lr_conservative_fdr_0_1",
            "all_true_lr_conservative_fdr_0_1",
            "best_true_lr",
            "best_true_lr_fdr",
            "permutation_audit_call",
        ]
    ].rename(
        columns={
            "top10_true_lr": "base_top10_true_lr",
            "top10_negative_or_risk_fraction": "base_top10_risk_fraction",
            "top10_conservative_fdr_0_1": "base_top10_q_0_1",
            "all_lr_conservative_fdr_0_1": "base_all_lr_q_0_1",
            "all_true_lr_conservative_fdr_0_1": "base_true_lr_q_0_1",
            "best_true_lr": "base_best_true_lr",
            "best_true_lr_fdr": "base_best_true_lr_fdr",
            "permutation_audit_call": "base_permutation_call",
        }
    )
    anchor_keep = anchor[
        [
            "dataset_key",
            "top10_true_lr",
            "top10_negative_or_risk_fraction",
            "top10_conservative_fdr_0_1",
            "all_lr_conservative_fdr_0_1",
            "all_true_lr_conservative_fdr_0_1",
            "best_true_lr",
            "best_true_lr_fdr",
            "anchor_preserving_permutation_call",
        ]
    ].rename(
        columns={
            "top10_true_lr": "anchor_top10_true_lr",
            "top10_negative_or_risk_fraction": "anchor_top10_risk_fraction",
            "top10_conservative_fdr_0_1": "anchor_top10_q_0_1",
            "all_lr_conservative_fdr_0_1": "anchor_all_lr_q_0_1",
            "all_true_lr_conservative_fdr_0_1": "anchor_true_lr_q_0_1",
            "best_true_lr": "anchor_best_true_lr",
            "best_true_lr_fdr": "anchor_best_true_lr_fdr",
            "anchor_preserving_permutation_call": "anchor_permutation_call",
        }
    )
    out = base_keep.merge(anchor_keep, on="dataset_key", how="outer")
    out["delta_top10_true_lr"] = out["anchor_top10_true_lr"].fillna(0) - out["base_top10_true_lr"].fillna(0)
    out["delta_top10_risk_fraction"] = (
        out["anchor_top10_risk_fraction"].fillna(np.nan) - out["base_top10_risk_fraction"].fillna(np.nan)
    )
    out["delta_best_true_lr_fdr"] = (
        out["anchor_best_true_lr_fdr"].fillna(np.nan) - out["base_best_true_lr_fdr"].fillna(np.nan)
    )
    out["discovery_fdr_closed"] = (
        pd.to_numeric(out["anchor_all_lr_q_0_1"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["anchor_true_lr_q_0_1"], errors="coerce").fillna(0).gt(0)
    )
    out["comparison_call"] = np.select(
        [
            out["discovery_fdr_closed"],
            (out["delta_top10_true_lr"] > 0) & (out["delta_top10_risk_fraction"] <= 0),
            (out["delta_top10_true_lr"] == 0) & (out["delta_top10_risk_fraction"] <= 0),
            out["delta_best_true_lr_fdr"] < 0,
        ],
        [
            "anchor_exact_score_fdr_supported",
            "top10_improved_without_risk_but_fdr_not_closed",
            "top10_preserved_without_risk_but_fdr_not_closed",
            "best_true_fdr_improved_but_not_closed",
        ],
        default="tradeoff_or_not_closed",
    )
    return out.sort_values("dataset_key").reset_index(drop=True)


def summarize_story(comparison: pd.DataFrame) -> pd.DataFrame:
    n_datasets = len(comparison)
    n_fdr_closed = int(comparison["discovery_fdr_closed"].sum())
    n_top10_improved = int(
        ((comparison["delta_top10_true_lr"] > 0) & (comparison["delta_top10_risk_fraction"] <= 0)).sum()
    )
    n_top10_preserved = int(
        ((comparison["delta_top10_true_lr"] == 0) & (comparison["delta_top10_risk_fraction"] <= 0)).sum()
    )
    n_best_true_q_improved = int((comparison["delta_best_true_lr_fdr"] < 0).sum())
    n_crc_weaker = int(
        (
            comparison["dataset_key"].astype(str).eq("visium_hd_crc_full")
            & (comparison["delta_top10_true_lr"] < 0)
        ).sum()
    )
    overall = (
        "anchor_exact_score_permutation_partial_support"
        if n_fdr_closed > 0
        else "anchor_exact_score_permutation_not_closed"
    )
    return pd.DataFrame(
        [
            {
                "topic": "anchor_preserving_spatial_null_exact_score_permutation",
                "overall_call": overall,
                "n_datasets": n_datasets,
                "n_fdr_closed_datasets": n_fdr_closed,
                "n_top10_improved_without_risk": n_top10_improved,
                "n_top10_preserved_without_risk": n_top10_preserved,
                "n_best_true_q_improved": n_best_true_q_improved,
                "n_crc_weaker": n_crc_weaker,
                "claim_boundary": (
                    "Anchor-preserving spatial-null training improves/preserves selected top-k control diagnostics, "
                    "but exact-score permutation still produces zero whole-atlas q<=0.1 discoveries; use as next-model "
                    "evidence, not FDR closure."
                ),
            }
        ]
    )


def write_status(comparison: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor-Preserving Exact-Score Permutation Comparison",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        "- Anchor-preserving spatial-null training remains a next-model candidate, not the reporting baseline.",
        "- Exact-score whole-atlas FDR is still not closed.",
        "",
        "## Dataset Comparison",
        "",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.comparison_call}`: "
            f"top10 true delta={fmt(row.delta_top10_true_lr)}, "
            f"top10 risk delta={fmt(row.delta_top10_risk_fraction)}, "
            f"best true q delta={fmt(row.delta_best_true_lr_fdr)}, "
            f"anchor all-LR q<=0.1={int(row.anchor_all_lr_q_0_1)}, "
            f"anchor true-LR q<=0.1={int(row.anchor_true_lr_q_0_1)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: anchor-preserving spatial-null training makes top-k rankings cleaner in selected datasets but does not solve broad whole-atlas FDR.",
        "- Unsafe wording: spatial-null training controls atlas-scale FDR.",
    ]
    (OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_null_anchor_preserving_permutation_comparison",
        "inputs": [
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "spatial_null_anchor_preserving_permutation_audit/phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv",
        ],
        "outputs": [
            "phase12_spatial_null_anchor_preserving_permutation_dataset_comparison.csv",
            "phase12_spatial_null_anchor_preserving_permutation_story.csv",
            "phase12_spatial_null_anchor_preserving_permutation_comparison_status.md",
        ],
        "claim_boundary": "Compares exact-score post-hoc permutation audits; not end-to-end FDR closure.",
    }
    (OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_comparison_manifest.json").write_text(
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
