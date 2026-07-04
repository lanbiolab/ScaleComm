#!/usr/bin/env python
"""Summarize the Phase 12 range-null calibration attempt.

The range-null-aware ranking variant was introduced as an exploratory model
calibration layer. This audit compares it against the current calibrated atlas
ranking and records whether it is strong enough to replace the main result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_DIR = PHASE12_DIR / "atlas_model_expanded"
RANGE_DIR = PHASE12_DIR / "atlas_model_expanded_range_calibrated"
OUT_DIR = PHASE12_DIR / "range_null_calibration_attempt"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_DIR / "phase12_expanded_atlas_calibrated_topk_summary.csv")
    base_dataset = pd.read_csv(BASE_DIR / "phase12_expanded_atlas_calibrated_dataset_summary.csv")
    range_topk = pd.read_csv(RANGE_DIR / "phase12_range_null_calibrated_topk_summary.csv")
    range_dataset = pd.read_csv(RANGE_DIR / "phase12_range_null_calibrated_dataset_summary.csv")

    comparison = compare_topk(base, range_topk)
    dataset = compare_dataset(base_dataset, range_dataset, comparison)
    story = summarize_story(dataset)

    comparison.to_csv(OUT_DIR / "phase12_range_null_calibration_topk_comparison.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_range_null_calibration_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_range_null_calibration_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def compare_topk(base: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_key",
        "top_k",
        "n_true_lr",
        "n_high_expression_negative",
        "n_any_negative_or_risk",
        "control_panel_empirical_fdr",
        "top_lr_pairs",
    ]
    left = base[keep].rename(
        columns={
            "n_true_lr": "base_true_lr",
            "n_high_expression_negative": "base_high_expression_negative",
            "n_any_negative_or_risk": "base_negative_or_risk",
            "control_panel_empirical_fdr": "base_control_panel_empirical_fdr",
            "top_lr_pairs": "base_top_lr_pairs",
        }
    )
    right = variant[keep].rename(
        columns={
            "n_true_lr": "range_null_true_lr",
            "n_high_expression_negative": "range_null_high_expression_negative",
            "n_any_negative_or_risk": "range_null_negative_or_risk",
            "control_panel_empirical_fdr": "range_null_control_panel_empirical_fdr",
            "top_lr_pairs": "range_null_top_lr_pairs",
        }
    )
    out = left.merge(right, on=["dataset_key", "top_k"], how="outer")
    out["delta_true_lr"] = out["range_null_true_lr"].fillna(0) - out["base_true_lr"].fillna(0)
    out["delta_negative_or_risk"] = out["range_null_negative_or_risk"].fillna(0) - out["base_negative_or_risk"].fillna(0)
    out["delta_control_panel_empirical_fdr"] = (
        out["range_null_control_panel_empirical_fdr"].fillna(np.nan)
        - out["base_control_panel_empirical_fdr"].fillna(np.nan)
    )
    out["topk_call"] = np.select(
        [
            (out["delta_true_lr"] > 0) & (out["delta_negative_or_risk"] <= 0) & (out["delta_control_panel_empirical_fdr"] <= 0),
            (out["delta_true_lr"] > 0) & (out["delta_negative_or_risk"] > 0),
            (out["delta_true_lr"] <= 0) & (out["delta_negative_or_risk"] > 0),
        ],
        [
            "improves_without_extra_control_risk",
            "recovers_more_true_but_adds_control_risk",
            "no_true_gain_and_adds_control_risk",
        ],
        default="mixed_or_neutral",
    )
    return out.sort_values(["dataset_key", "top_k"]).reset_index(drop=True)


def compare_dataset(base_dataset: pd.DataFrame, range_dataset: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    base = base_dataset[
        [
            "dataset_key",
            "top10_true_lr",
            "top10_high_expression_negative",
            "top10_control_panel_empirical_fdr",
            "top50_true_lr",
            "top50_high_expression_negative",
            "calibrated_audit_call",
        ]
    ].rename(
        columns={
            "top10_true_lr": "base_top10_true_lr",
            "top10_high_expression_negative": "base_top10_high_expression_negative",
            "top10_control_panel_empirical_fdr": "base_top10_control_panel_empirical_fdr",
            "top50_true_lr": "base_top50_true_lr",
            "top50_high_expression_negative": "base_top50_high_expression_negative",
            "calibrated_audit_call": "base_call",
        }
    )
    variant = range_dataset[
        [
            "dataset_key",
            "top10_true_lr",
            "top10_high_expression_negative",
            "top10_negative_or_risk",
            "top10_control_panel_empirical_fdr",
            "top100_true_lr",
            "top100_control_panel_empirical_fdr",
            "range_null_calibrated_call",
        ]
    ].rename(
        columns={
            "top10_true_lr": "range_null_top10_true_lr",
            "top10_high_expression_negative": "range_null_top10_high_expression_negative",
            "top10_negative_or_risk": "range_null_top10_negative_or_risk",
            "top10_control_panel_empirical_fdr": "range_null_top10_control_panel_empirical_fdr",
            "top100_true_lr": "range_null_top100_true_lr",
            "top100_control_panel_empirical_fdr": "range_null_top100_control_panel_empirical_fdr",
            "range_null_calibrated_call": "range_null_call",
        }
    )
    out = base.merge(variant, on="dataset_key", how="outer")
    t10 = comparison[comparison["top_k"].eq(10)][
        ["dataset_key", "delta_true_lr", "delta_negative_or_risk", "delta_control_panel_empirical_fdr", "topk_call"]
    ].rename(columns={"topk_call": "top10_variant_call"})
    t100 = comparison[comparison["top_k"].eq(100)][
        ["dataset_key", "delta_true_lr", "delta_negative_or_risk", "delta_control_panel_empirical_fdr", "topk_call"]
    ].rename(
        columns={
            "delta_true_lr": "top100_delta_true_lr",
            "delta_negative_or_risk": "top100_delta_negative_or_risk",
            "delta_control_panel_empirical_fdr": "top100_delta_control_panel_empirical_fdr",
            "topk_call": "top100_variant_call",
        }
    )
    out = out.merge(t10, on="dataset_key", how="left").merge(t100, on="dataset_key", how="left")
    out["adoption_decision"] = np.where(
        (out["delta_true_lr"].fillna(0) > 0)
        & (out["delta_negative_or_risk"].fillna(0) <= 0)
        & (out["top100_delta_control_panel_empirical_fdr"].fillna(1) <= 0),
        "adopt_candidate",
        "hold_out_exploratory_variant",
    )
    out["decision_reason"] = np.where(
        out["adoption_decision"].eq("adopt_candidate"),
        "range-null variant improves top-k recovery without increasing control-risk diagnostics",
        "range-null variant either adds top-k control risk or worsens broader control-panel FDR",
    )
    return out.sort_values("dataset_key").reset_index(drop=True)


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n_dataset = len(dataset)
    n_adopt = int(dataset["adoption_decision"].eq("adopt_candidate").sum())
    n_hold = int(dataset["adoption_decision"].eq("hold_out_exploratory_variant").sum())
    n_top10_true_gain = int(pd.to_numeric(dataset["delta_true_lr"], errors="coerce").fillna(0).gt(0).sum())
    n_top10_risk_gain = int(pd.to_numeric(dataset["delta_negative_or_risk"], errors="coerce").fillna(0).gt(0).sum())
    n_top100_fdr_worse = int(
        pd.to_numeric(dataset["top100_delta_control_panel_empirical_fdr"], errors="coerce").fillna(0).gt(0).sum()
    )
    overall = "not_adopted_exploratory_variant" if n_adopt < n_dataset else "adoptable_candidate"
    return pd.DataFrame(
        [
            {
                "topic": "range_null_calibration_attempt",
                "overall_call": overall,
                "n_datasets": n_dataset,
                "n_adopt_candidate": n_adopt,
                "n_hold_out": n_hold,
                "n_top10_true_gain": n_top10_true_gain,
                "n_top10_extra_control_risk": n_top10_risk_gain,
                "n_top100_control_fdr_worse": n_top100_fdr_worse,
                "claim_boundary": (
                    "Exploratory model-variant audit only. The current range-null-calibrated score is not "
                    "adopted as the main Phase 12 result because it does not improve control-risk diagnostics "
                    "consistently across datasets."
                ),
            }
        ]
    )


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Range-Null Calibration Attempt",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        "- Keep the existing calibrated expanded-atlas ranking as the main Phase 12 ranking.",
        "- Treat the range-null-calibrated ranking as an exploratory attempt until it improves control-risk diagnostics consistently.",
        "",
        "## Dataset Comparison",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: decision `{row.adoption_decision}`; "
            f"top10 true delta={fmt(row.delta_true_lr)}, top10 risk delta={fmt(row.delta_negative_or_risk)}, "
            f"top100 control-FDR delta={fmt(row.top100_delta_control_panel_empirical_fdr)}. "
            f"Reason: {row.decision_reason}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Do not use this variant for the main model claim yet.",
        "- If mentioned, describe it as a calibration attempt that helped identify the remaining broad-background/risk leakage problem.",
    ]
    (OUT_DIR / "phase12_range_null_calibration_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_range_null_calibration_attempt_audit",
        "inputs": [
            "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv",
            "atlas_model_expanded_range_calibrated/phase12_range_null_calibrated_topk_summary.csv",
        ],
        "outputs": [
            "phase12_range_null_calibration_topk_comparison.csv",
            "phase12_range_null_calibration_dataset_comparison.csv",
            "phase12_range_null_calibration_story.csv",
            "phase12_range_null_calibration_status.md",
        ],
        "claim_boundary": "Exploratory range-null calibration attempt; not adopted as main result.",
    }
    (OUT_DIR / "phase12_range_null_calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
