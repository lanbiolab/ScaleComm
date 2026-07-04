#!/usr/bin/env python
"""Compare background-suppressed exact-score permutation audit with the main audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
BG_DIR = PHASE12_DIR / "background_suppressed_permutation_audit"
OUT_DIR = PHASE12_DIR / "background_suppressed_permutation_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv")
    bg = pd.read_csv(BG_DIR / "phase12_background_suppressed_permutation_dataset_summary.csv")
    comparison = compare_dataset(base, bg)
    story = summarize_story(comparison)

    comparison.to_csv(OUT_DIR / "phase12_background_suppressed_permutation_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_background_suppressed_permutation_story.csv", index=False)
    write_status(comparison, story)
    write_manifest()

    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def compare_dataset(base: pd.DataFrame, bg: pd.DataFrame) -> pd.DataFrame:
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
    bg_keep = bg[
        [
            "dataset_key",
            "top10_true_lr",
            "top10_negative_or_risk_fraction",
            "top10_conservative_fdr_0_1",
            "all_lr_conservative_fdr_0_1",
            "all_true_lr_conservative_fdr_0_1",
            "best_true_lr",
            "best_true_lr_fdr",
            "background_suppressed_permutation_call",
        ]
    ].rename(
        columns={
            "top10_true_lr": "background_top10_true_lr",
            "top10_negative_or_risk_fraction": "background_top10_risk_fraction",
            "top10_conservative_fdr_0_1": "background_top10_q_0_1",
            "all_lr_conservative_fdr_0_1": "background_all_lr_q_0_1",
            "all_true_lr_conservative_fdr_0_1": "background_true_lr_q_0_1",
            "best_true_lr": "background_best_true_lr",
            "best_true_lr_fdr": "background_best_true_lr_fdr",
            "background_suppressed_permutation_call": "background_permutation_call",
        }
    )
    out = base_keep.merge(bg_keep, on="dataset_key", how="outer")
    out["delta_top10_true_lr"] = out["background_top10_true_lr"].fillna(0) - out["base_top10_true_lr"].fillna(0)
    out["delta_top10_risk_fraction"] = (
        out["background_top10_risk_fraction"].fillna(np.nan) - out["base_top10_risk_fraction"].fillna(np.nan)
    )
    out["delta_best_true_lr_fdr"] = (
        out["background_best_true_lr_fdr"].fillna(np.nan) - out["base_best_true_lr_fdr"].fillna(np.nan)
    )
    out["discovery_fdr_closed"] = (
        pd.to_numeric(out["background_all_lr_q_0_1"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["background_true_lr_q_0_1"], errors="coerce").fillna(0).gt(0)
    )
    out["comparison_call"] = np.select(
        [
            out["discovery_fdr_closed"],
            (out["delta_top10_true_lr"] > 0) & (out["delta_top10_risk_fraction"] <= 0),
            out["delta_best_true_lr_fdr"] < 0,
        ],
        [
            "background_exact_score_fdr_supported",
            "top10_improved_without_risk_but_fdr_not_closed",
            "best_true_fdr_improved_but_not_closed",
        ],
        default="not_improved_or_not_closed",
    )
    return out.sort_values("dataset_key").reset_index(drop=True)


def summarize_story(comparison: pd.DataFrame) -> pd.DataFrame:
    n_datasets = len(comparison)
    n_fdr_closed = int(comparison["discovery_fdr_closed"].sum())
    n_top10_improved = int(
        ((comparison["delta_top10_true_lr"] > 0) & (comparison["delta_top10_risk_fraction"] <= 0)).sum()
    )
    n_best_true_q_improved = int((comparison["delta_best_true_lr_fdr"] < 0).sum())
    overall = (
        "background_exact_score_permutation_not_closed"
        if n_fdr_closed == 0
        else "background_exact_score_permutation_partial_support"
    )
    return pd.DataFrame(
        [
            {
                "topic": "background_suppressed_exact_score_permutation",
                "overall_call": overall,
                "n_datasets": n_datasets,
                "n_fdr_closed_datasets": n_fdr_closed,
                "n_top10_improved_without_risk": n_top10_improved,
                "n_best_true_q_improved": n_best_true_q_improved,
                "claim_boundary": (
                    "Exact-score background-suppressed permutation audit improves selected top-k ranking but does not "
                    "produce whole-atlas q<=0.1 discoveries; use as calibration evidence, not FDR closure."
                ),
            }
        ]
    )


def write_status(comparison: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Background-Suppressed Exact-Score Permutation Comparison",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        "- Background suppression should not replace the main ranking for FDR claims.",
        "- It remains useful as a calibration direction because some top-k diagnostics improve.",
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
            f"background all-LR q<=0.1={int(row.background_all_lr_q_0_1)}, "
            f"background true-LR q<=0.1={int(row.background_true_lr_q_0_1)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: exact-score null audit confirms the variant is not enough to close broad whole-atlas FDR.",
        "- Unsafe wording: background suppression controls whole-atlas FDR.",
    ]
    (OUT_DIR / "phase12_background_suppressed_permutation_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_background_suppressed_permutation_comparison",
        "inputs": [
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "background_suppressed_permutation_audit/phase12_background_suppressed_permutation_dataset_summary.csv",
        ],
        "outputs": [
            "phase12_background_suppressed_permutation_dataset_comparison.csv",
            "phase12_background_suppressed_permutation_story.csv",
            "phase12_background_suppressed_permutation_comparison_status.md",
        ],
        "claim_boundary": "Compares exact-score post-hoc permutation audits; not end-to-end FDR closure.",
    }
    (OUT_DIR / "phase12_background_suppressed_permutation_comparison_manifest.json").write_text(
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
