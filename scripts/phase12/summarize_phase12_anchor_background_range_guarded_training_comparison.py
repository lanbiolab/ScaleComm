#!/usr/bin/env python
"""Compare training-integrated anchor/background range-guarded variant with prior variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_PERM_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
ANCHOR_BG_PERM_DIR = PHASE12_DIR / "anchor_background_suppressed_permutation_audit"
TRAIN_PERM_DIR = PHASE12_DIR / "anchor_background_range_guarded_training_permutation_audit"
OUT_DIR = PHASE12_DIR / "anchor_background_range_guarded_training_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset_comparison()
    story = summarize_story(dataset)
    dataset.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_dataset_comparison() -> pd.DataFrame:
    train_rank = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_dataset_summary.csv"
    )
    train_perm = load_perm(
        TRAIN_PERM_DIR / "phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv",
        "train",
    )
    base_perm = load_perm(BASE_PERM_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv", "base")
    anchor_bg_perm = load_perm(
        ANCHOR_BG_PERM_DIR / "phase12_anchor_background_suppressed_permutation_dataset_summary.csv",
        "anchor_bg",
    )
    out = normalize_rank(train_rank).merge(train_perm, on="dataset_key", how="left")
    out = out.merge(base_perm, on="dataset_key", how="left").merge(anchor_bg_perm, on="dataset_key", how="left")
    add_deltas(out)
    out["comparison_call"] = [comparison_call(row) for row in out.itertuples(index=False)]
    return out


def normalize_rank(table: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_key",
        "n_lr",
        "n_true_lr",
        "top10_true_lr",
        "top10_negative_or_risk",
        "top10_control_panel_empirical_fdr",
        "top20_broad_other_lr",
        "top100_true_lr",
        "top100_control_panel_empirical_fdr",
        "hard_negative_best_comparison_beats",
        "hard_negative_comparisons",
        "anchor_background_range_guarded_training_call",
    ]
    existing = [col for col in keep if col in table.columns]
    return table[existing].copy()


def load_perm(path: Path, prefix: str) -> pd.DataFrame:
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
    table["train_fdr_closed"] = (
        pd.to_numeric(table["train_all_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(table["train_all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).gt(0)
    )
    table["delta_train_vs_base_top10_true_lr"] = table["train_top10_true_lr"] - table["base_top10_true_lr"]
    table["delta_train_vs_base_top10_risk"] = (
        table["train_top10_negative_or_risk_fraction"] - table["base_top10_negative_or_risk_fraction"]
    )
    table["delta_train_vs_base_best_true_fdr"] = (
        table["train_best_true_lr_fdr"] - table["base_best_true_lr_fdr"]
    )
    table["delta_train_vs_anchor_bg_top10_true_lr"] = (
        table["train_top10_true_lr"] - table["anchor_bg_top10_true_lr"]
    )
    table["delta_train_vs_anchor_bg_top10_risk"] = (
        table["train_top10_negative_or_risk_fraction"] - table["anchor_bg_top10_negative_or_risk_fraction"]
    )
    table["delta_train_vs_anchor_bg_best_true_fdr"] = (
        table["train_best_true_lr_fdr"] - table["anchor_bg_best_true_lr_fdr"]
    )


def comparison_call(row: object) -> str:
    if bool(getattr(row, "train_fdr_closed", False)):
        return "training_integrated_fdr_supported"
    delta_base_true = float(getattr(row, "delta_train_vs_base_top10_true_lr", 0.0))
    delta_base_risk = float(getattr(row, "delta_train_vs_base_top10_risk", 0.0))
    delta_anchor_risk = float(getattr(row, "delta_train_vs_anchor_bg_top10_risk", 0.0))
    if delta_base_true > 0 and delta_base_risk <= 0 and delta_anchor_risk <= 0:
        return "training_integrated_topk_improved_not_fdr_closed"
    if delta_base_true > 0 and delta_base_risk <= 0:
        return "training_integrated_improves_base_but_worse_than_anchor_bg"
    if delta_base_true >= 0 and delta_base_risk < 0:
        return "training_integrated_reduces_risk_not_fdr_closed"
    return "training_integrated_tradeoff_not_fdr_closed"


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n_datasets = int(len(dataset))
    n_fdr_closed = int(dataset["train_fdr_closed"].sum())
    n_improved_vs_base = int(
        (
            (dataset["delta_train_vs_base_top10_true_lr"] > 0)
            & (dataset["delta_train_vs_base_top10_risk"] <= 0)
        ).sum()
    )
    n_worse_risk_than_anchor_bg = int((dataset["delta_train_vs_anchor_bg_top10_risk"] > 0).sum())
    n_hard_clean = int(pd.to_numeric(dataset["hard_negative_best_comparison_beats"], errors="coerce").fillna(0).eq(0).sum())
    return pd.DataFrame(
        [
            {
                "topic": "anchor_background_range_guarded_training_exact_score_permutation",
                "overall_call": (
                    "training_integrated_anchor_background_fdr_partial_support"
                    if n_fdr_closed > 0
                    else "training_integrated_anchor_background_fdr_not_closed"
                ),
                "n_datasets": n_datasets,
                "n_fdr_closed_datasets": n_fdr_closed,
                "n_top10_improved_without_risk_vs_base": n_improved_vs_base,
                "n_worse_top10_risk_than_posthoc_anchor_background": n_worse_risk_than_anchor_bg,
                "n_hard_negative_best_beat_clean": n_hard_clean,
                "claim_boundary": (
                    "Training integration with range-violation spatial null improves top-k recovery over the main calibrated ranking and removes best hard-negative beats, "
                    "but it still has zero broad whole-atlas q<=0.1 discoveries and can be riskier than the post-hoc anchor/background score."
                ),
            }
        ]
    )


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor-Background Range-Guarded Training Comparison",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        f"- FDR-closed datasets: {int(story.iloc[0].n_fdr_closed_datasets)}/{int(story.iloc[0].n_datasets)}.",
        f"- Top10 improved without added risk vs base: {int(story.iloc[0].n_top10_improved_without_risk_vs_base)}/{int(story.iloc[0].n_datasets)}.",
        f"- Hard-negative best-beat clean datasets: {int(story.iloc[0].n_hard_negative_best_beat_clean)}/{int(story.iloc[0].n_datasets)}.",
        "",
        "## Dataset Comparison",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.comparison_call}`: "
            f"top10 true={int(row.train_top10_true_lr)}, "
            f"top10 risk={fmt(row.train_top10_negative_or_risk_fraction)}, "
            f"best true q={fmt(row.train_best_true_lr_fdr)}, "
            f"delta risk vs posthoc anchor-bg={fmt(row.delta_train_vs_anchor_bg_top10_risk)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: training integration with range-violation spatial null is a useful next-model direction and hard-negative guard, not a solved whole-atlas FDR procedure.",
        "- Unsafe wording: the new training variant closes broad all-LR FDR.",
    ]
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_anchor_background_range_guarded_training_comparison",
        "inputs": [
            "atlas_model_expanded_anchor_background_range_guarded_training/phase12_anchor_background_range_guarded_training_dataset_summary.csv",
            "anchor_background_range_guarded_training_permutation_audit/phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv",
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv",
            "anchor_background_suppressed_permutation_audit/phase12_anchor_background_suppressed_permutation_dataset_summary.csv",
        ],
        "outputs": [
            "phase12_anchor_background_range_guarded_training_dataset_comparison.csv",
            "phase12_anchor_background_range_guarded_training_story.csv",
            "phase12_anchor_background_range_guarded_training_comparison_status.md",
        ],
        "claim_boundary": "Comparison for training-integrated anchor/background range-guarded variant; not a new FDR procedure.",
    }
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_comparison_manifest.json").write_text(
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
