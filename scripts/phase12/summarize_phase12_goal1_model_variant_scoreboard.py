#!/usr/bin/env python
"""Build a Goal 1 model-variant scoreboard for Phase 12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "goal1_model_variant_scoreboard"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset_scoreboard()
    variant = summarize_variant(dataset)
    story = summarize_story(variant)

    dataset.to_csv(OUT_DIR / "phase12_goal1_model_variant_dataset_scoreboard.csv", index=False)
    variant.to_csv(OUT_DIR / "phase12_goal1_model_variant_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_goal1_model_variant_story.csv", index=False)
    write_status(variant, story)
    write_manifest()

    print(variant.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_dataset_scoreboard() -> pd.DataFrame:
    base = pd.read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv")
    base_topk = pd.read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv")
    range_null = pd.read_csv(
        PHASE12_DIR / "atlas_model_expanded_range_calibrated/phase12_range_null_calibrated_dataset_summary.csv"
    )
    bg = pd.read_csv(
        PHASE12_DIR / "atlas_model_expanded_background_suppressed/phase12_background_suppressed_dataset_summary.csv"
    )
    base_perm = pd.read_csv(
        PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv"
    )
    bg_perm = pd.read_csv(
        PHASE12_DIR
        / "background_suppressed_permutation_audit/phase12_background_suppressed_permutation_dataset_summary.csv"
    )
    range_attempt = pd.read_csv(
        PHASE12_DIR / "range_null_calibration_attempt/phase12_range_null_calibration_dataset_comparison.csv"
    )
    bg_attempt = pd.read_csv(
        PHASE12_DIR / "background_suppression_attempt/phase12_background_suppression_dataset_comparison.csv"
    )
    bg_perm_cmp = pd.read_csv(
        PHASE12_DIR
        / "background_suppressed_permutation_comparison/phase12_background_suppressed_permutation_dataset_comparison.csv"
    )
    guarded_training = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_background_guarded_training/phase12_background_guarded_training_dataset_summary.csv"
    )
    guarded_training_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_background_guarded_training/phase12_background_guarded_training_topk_summary.csv"
    )
    guarded_training_perm = pd.read_csv(
        PHASE12_DIR
        / "background_guarded_training_permutation_audit/"
        / "phase12_background_guarded_training_permutation_dataset_summary.csv"
    )
    guarded_training_cmp = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_background_guarded_training/phase12_background_guarded_training_comparison.csv"
    )
    range_violation = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_range_violation_guarded_training/"
        / "phase12_range_violation_guarded_training_dataset_summary.csv"
    )
    range_violation_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_range_violation_guarded_training/"
        / "phase12_range_violation_guarded_training_topk_summary.csv"
    )
    range_violation_perm = pd.read_csv(
        PHASE12_DIR
        / "range_violation_guarded_training_permutation_audit/"
        / "phase12_range_violation_guarded_training_permutation_dataset_summary.csv"
    )
    range_violation_cmp = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_range_violation_guarded_training/"
        / "phase12_range_violation_guarded_training_comparison.csv"
    )
    anchor_background = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_suppressed/"
        / "phase12_anchor_background_suppressed_dataset_summary.csv"
    )
    anchor_background_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_suppressed/"
        / "phase12_anchor_background_suppressed_topk_summary.csv"
    )
    anchor_background_perm = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_audit/"
        / "phase12_anchor_background_suppressed_permutation_dataset_summary.csv"
    )
    anchor_background_cmp = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_suppressed_permutation_comparison/"
        / "phase12_anchor_background_permutation_dataset_comparison.csv"
    )
    anchor_background_guarded = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_guarded_training/"
        / "phase12_anchor_background_guarded_training_dataset_summary.csv"
    )
    anchor_background_guarded_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_guarded_training/"
        / "phase12_anchor_background_guarded_training_topk_summary.csv"
    )
    anchor_background_guarded_perm = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_guarded_training_permutation_audit/"
        / "phase12_anchor_background_guarded_training_permutation_dataset_summary.csv"
    )
    anchor_background_guarded_cmp = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_guarded_training_comparison/"
        / "phase12_anchor_background_guarded_training_dataset_comparison.csv"
    )
    anchor_background_range_guarded = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_dataset_summary.csv"
    )
    anchor_background_range_guarded_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_topk_summary.csv"
    )
    anchor_background_range_guarded_perm = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_range_guarded_training_permutation_audit/"
        / "phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv"
    )
    anchor_background_range_guarded_cmp = pd.read_csv(
        PHASE12_DIR
        / "anchor_background_range_guarded_training_comparison/"
        / "phase12_anchor_background_range_guarded_training_dataset_comparison.csv"
    )
    integrated_guarded_spatial = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_integrated_guarded_spatial_training/"
        / "phase12_integrated_guarded_spatial_training_dataset_summary.csv"
    )
    integrated_guarded_spatial_topk = pd.read_csv(
        PHASE12_DIR
        / "atlas_model_expanded_integrated_guarded_spatial_training/"
        / "phase12_integrated_guarded_spatial_training_topk_summary.csv"
    )
    integrated_guarded_spatial_perm = pd.read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv"
    )

    datasets = sorted(set(base["dataset_key"].astype(str)))
    rows = []
    for dataset_key in datasets:
        rows.append(base_row(dataset_key, base, base_topk, base_perm))
        rows.append(range_null_row(dataset_key, range_null, range_attempt))
        rows.append(background_row(dataset_key, bg, bg_attempt, bg_perm, bg_perm_cmp))
        rows.append(
            background_guarded_training_row(
                dataset_key,
                guarded_training,
                guarded_training_topk,
                guarded_training_perm,
                guarded_training_cmp,
            )
        )
        rows.append(
            range_violation_guarded_training_row(
                dataset_key,
                range_violation,
                range_violation_topk,
                range_violation_perm,
                range_violation_cmp,
            )
        )
        rows.append(
            anchor_background_suppressed_row(
                dataset_key,
                anchor_background,
                anchor_background_topk,
                anchor_background_perm,
                anchor_background_cmp,
            )
        )
        rows.append(
            anchor_background_guarded_training_row(
                dataset_key,
                anchor_background_guarded,
                anchor_background_guarded_topk,
                anchor_background_guarded_perm,
                anchor_background_guarded_cmp,
            )
        )
        rows.append(
            anchor_background_range_guarded_training_row(
                dataset_key,
                anchor_background_range_guarded,
                anchor_background_range_guarded_topk,
                anchor_background_range_guarded_perm,
                anchor_background_range_guarded_cmp,
            )
        )
        rows.append(
            integrated_guarded_spatial_training_row(
                dataset_key,
                integrated_guarded_spatial,
                integrated_guarded_spatial_topk,
                integrated_guarded_spatial_perm,
            )
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "variant"]).reset_index(drop=True)


def base_row(dataset_key: str, base: pd.DataFrame, topk: pd.DataFrame, perm: pd.DataFrame) -> dict[str, object]:
    row = one(base, dataset_key)
    p = one(perm, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    return {
        "dataset_key": dataset_key,
        "variant": "main_calibrated",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(round(float(row.get("top10_negative_or_risk_fraction", 0)) * 10))
        if "top10_negative_or_risk_fraction" in row
        else np.nan,
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": np.nan,
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": row.get("calibrated_audit_call", ""),
        "adoption_call": "current_main_ranking",
        "use_in_next_model": "baseline_to_improve",
        "claim_boundary": "Main calibrated ranking; ranking/control support only, not whole-atlas FDR closure.",
    }


def range_null_row(dataset_key: str, dataset: pd.DataFrame, attempt: pd.DataFrame) -> dict[str, object]:
    row = one(dataset, dataset_key)
    a = one(attempt, dataset_key)
    return {
        "dataset_key": dataset_key,
        "variant": "range_null_calibrated",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(row.get("top10_negative_or_risk", 0)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": np.nan,
        "top100_control_fdr": float(row.get("top100_control_panel_empirical_fdr", np.nan)),
        "all_lr_q_0_1": np.nan,
        "true_lr_q_0_1": np.nan,
        "best_true_lr": "",
        "best_true_lr_q": np.nan,
        "dataset_call": row.get("range_null_calibrated_call", ""),
        "adoption_call": a.get("adoption_decision", "hold_out_exploratory_variant"),
        "use_in_next_model": "do_not_adopt_global_variant",
        "claim_boundary": "Exploratory score; improves some top10 true recovery but worsens broader control risk.",
    }


def background_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    attempt: pd.DataFrame,
    perm: pd.DataFrame,
    perm_cmp: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    a = one(attempt, dataset_key)
    p = one(perm, dataset_key)
    pc = one(perm_cmp, dataset_key)
    next_use = "candidate_component"
    if str(pc.get("comparison_call", "")).endswith("fdr_not_closed"):
        next_use = "candidate_component_requires_new_null_training"
    if dataset_key == "visium_hd_crc_full":
        next_use = "stress_case_not_adopt"
    return {
        "dataset_key": dataset_key,
        "variant": "background_suppressed",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(row.get("top10_negative_or_risk", 0)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(row.get("top20_broad_other_lr", 0)),
        "top100_control_fdr": float(row.get("top100_control_panel_empirical_fdr", np.nan)),
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": pc.get("comparison_call", row.get("background_suppressed_call", "")),
        "adoption_call": a.get("background_adoption_decision", "hold_out_exploratory_variant"),
        "use_in_next_model": next_use,
        "claim_boundary": "Exact-score null audit done; top-k can improve but whole-atlas FDR remains unclosed.",
    }


def background_guarded_training_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    c = one(comparison, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    next_use = "candidate_component_requires_stronger_crc_guard"
    if dataset_key == "visium_hd_crc_full":
        next_use = "stress_case_not_adopt"
    if str(c.get("comparison_call", "")).endswith("improves_main_top10"):
        next_use = "candidate_component_for_anchor_preservation"
    return {
        "dataset_key": dataset_key,
        "variant": "background_guarded_training",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(round(float(row.get("top10_negative_or_risk_fraction", 0.0)) * 10)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": np.nan,
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": p.get("background_guarded_permutation_call", row.get("calibrated_audit_call", "")),
        "adoption_call": "hold_out_exploratory_variant",
        "use_in_next_model": next_use,
        "claim_boundary": "Training-integrated guard improves selected top-k diagnostics but exact-score whole-atlas FDR remains unclosed.",
    }


def range_violation_guarded_training_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    c = one(comparison, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    next_use = "hold_out_range_violation_tradeoff"
    if str(c.get("comparison_call", "")).endswith("reduces_main_risk"):
        next_use = "diagnostic_component_for_crc_risk_reduction"
    return {
        "dataset_key": dataset_key,
        "variant": "range_violation_guarded_training",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(round(float(row.get("top10_negative_or_risk_fraction", 0.0)) * 10)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": np.nan,
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": p.get("range_violation_permutation_call", row.get("calibrated_audit_call", "")),
        "adoption_call": "hold_out_exploratory_variant",
        "use_in_next_model": next_use,
        "claim_boundary": (
            "Range-violation spatial-null negatives reduce selected control risk but sacrifice true anchors "
            "and do not close exact-score whole-atlas FDR."
        ),
    }


def anchor_background_suppressed_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    c = one(comparison, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    next_use = "candidate_component_requires_training_integration"
    if dataset_key == "visium_hd_crc_full":
        next_use = "candidate_component_with_crc_high_expression_guard"
    return {
        "dataset_key": dataset_key,
        "variant": "anchor_background_suppressed",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(row.get("top10_negative_or_risk", 0)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": float(row.get("top100_control_panel_empirical_fdr", np.nan)),
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": c.get(
            "comparison_call",
            p.get("anchor_background_permutation_call", row.get("background_suppressed_call", "")),
        ),
        "adoption_call": "hold_out_exploratory_variant",
        "use_in_next_model": next_use,
        "claim_boundary": (
            "Anchor-preserving scores plus background suppression improve top-k recovery in all three datasets, "
            "but exact-score whole-atlas FDR remains unclosed."
        ),
    }


def anchor_background_guarded_training_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    c = one(comparison, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    next_use = "candidate_component_requires_stronger_spatial_null_fdr"
    if float(c.get("delta_train_vs_anchor_bg_top10_risk", 0.0)) > 0:
        next_use = "candidate_component_with_risk_regression_vs_posthoc"
    return {
        "dataset_key": dataset_key,
        "variant": "anchor_background_guarded_training",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(row.get("top10_negative_or_risk", 0)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": float(row.get("top100_control_panel_empirical_fdr", np.nan)),
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": c.get(
            "comparison_call",
            p.get("anchor_background_guarded_permutation_call", row.get("anchor_background_guarded_training_call", "")),
        ),
        "adoption_call": "hold_out_exploratory_variant",
        "use_in_next_model": next_use,
        "claim_boundary": (
            "Training-integrated anchor/background guards improve top-k recovery versus main and remove best hard-negative beats, "
            "but exact-score whole-atlas FDR remains unclosed and post-Xenium risk regresses versus the post-hoc score."
        ),
    }


def anchor_background_range_guarded_training_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    c = one(comparison, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    next_use = "candidate_component_requires_stronger_spatial_null_fdr"
    if float(c.get("delta_train_vs_anchor_bg_top10_risk", 0.0)) > 0:
        next_use = "candidate_component_with_residual_risk_vs_posthoc"
    return {
        "dataset_key": dataset_key,
        "variant": "anchor_background_range_guarded_training",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": int(row.get("top10_negative_or_risk", 0)),
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": float(row.get("top100_control_panel_empirical_fdr", np.nan)),
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": int(p.get("all_true_lr_conservative_fdr_0_1", 0)),
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": c.get(
            "comparison_call",
            p.get(
                "anchor_background_range_guarded_permutation_call",
                row.get("anchor_background_range_guarded_training_call", ""),
            ),
        ),
        "adoption_call": "hold_out_exploratory_variant",
        "use_in_next_model": next_use,
        "claim_boundary": (
            "Low-weight range-violation spatial-null training preserves top-k anchors and improves selected best-q values, "
            "but exact-score whole-atlas FDR remains unclosed."
        ),
    }


def integrated_guarded_spatial_training_row(
    dataset_key: str,
    dataset: pd.DataFrame,
    topk: pd.DataFrame,
    perm: pd.DataFrame,
) -> dict[str, object]:
    row = one(dataset, dataset_key)
    p = one(perm, dataset_key)
    t20 = one_topk(topk, dataset_key, 20)
    true_q = int(p.get("all_true_lr_conservative_fdr_0_1", 0))
    top10_risk = int(round(float(row.get("top10_negative_or_risk_fraction", 0.0)) * 10))
    next_use = "candidate_for_fdr_followup_requires_reproducibility"
    if true_q > 0 and top10_risk == 0:
        next_use = "leading_candidate_for_fdr_followup"
    if dataset_key == "visium_hd_crc_full" and top10_risk > 0:
        next_use = "stress_case_requires_crc_background_guard"
    return {
        "dataset_key": dataset_key,
        "variant": "integrated_guarded_spatial_training",
        "top10_true_lr": int(row.get("top10_true_lr", 0)),
        "top10_risk": top10_risk,
        "top10_control_fdr": float(row.get("top10_control_panel_empirical_fdr", np.nan)),
        "top20_broad_other_lr": int(t20.get("n_broad_other_lr", np.nan)) if "n_broad_other_lr" in t20 else np.nan,
        "top100_control_fdr": np.nan,
        "all_lr_q_0_1": int(p.get("all_lr_conservative_fdr_0_1", 0)),
        "true_lr_q_0_1": true_q,
        "best_true_lr": p.get("best_true_lr", ""),
        "best_true_lr_q": float(p.get("best_true_lr_fdr", np.nan)),
        "dataset_call": p.get(
            "integrated_guarded_spatial_training_permutation_call",
            row.get("integrated_guarded_spatial_training_call", ""),
        ),
        "adoption_call": "hold_out_candidate_until_end_to_end_audit",
        "use_in_next_model": next_use,
        "claim_boundary": (
            "First candidate with a conservative broad true q<=0.1 hit, but not current baseline; "
            "requires more permutations/end-to-end nulls and CRC risk repair."
        ),
    }


def summarize_variant(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in dataset.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "n_datasets": int(len(sub)),
                "mean_top10_true_lr": mean_float(sub["top10_true_lr"]),
                "total_true_lr_q_0_1": int(pd.to_numeric(sub["true_lr_q_0_1"], errors="coerce").fillna(0).sum()),
                "total_all_lr_q_0_1": int(pd.to_numeric(sub["all_lr_q_0_1"], errors="coerce").fillna(0).sum()),
                "n_adopt_candidate": int(sub["adoption_call"].astype(str).eq("adopt_candidate").sum()),
                "n_hold_out": int(sub["adoption_call"].astype(str).str.contains("hold_out", na=False).sum()),
                "n_next_model_candidate": int(
                    sub["use_in_next_model"]
                    .astype(str)
                    .str.contains("candidate_component|candidate_for_fdr_followup|leading_candidate", na=False)
                    .sum()
                ),
                "variant_decision": classify_variant(variant, sub),
                "claim_boundary": summarize_boundary(variant, sub),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def summarize_story(variant: pd.DataFrame) -> pd.DataFrame:
    fdr_closed = int(pd.to_numeric(variant["total_true_lr_q_0_1"], errors="coerce").fillna(0).sum())
    candidate_components = int(pd.to_numeric(variant["n_next_model_candidate"], errors="coerce").fillna(0).sum())
    return pd.DataFrame(
        [
            {
                "topic": "goal1_model_variant_scoreboard",
                "overall_call": "integrated_candidate_has_first_true_fdr_hit_goal1_still_partial"
                if fdr_closed > 0
                else "variants_inform_next_model_but_do_not_close_goal1_fdr",
                "n_variants": int(len(variant)),
                "total_true_lr_q_0_1": fdr_closed,
                "n_next_model_candidate_components": candidate_components,
                "claim_boundary": (
                    "Main ranking remains the reporting baseline. The integrated guarded spatial candidate has the "
                    "first broad true q<=0.1 hit, but it needs reproducibility/end-to-end null audits and CRC risk "
                    "repair before baseline replacement."
                    if fdr_closed > 0
                    else "Main ranking remains the reporting baseline. Background/background-guard variants can inform "
                    "the next model, while range-violation training exposes an anchor-recovery tradeoff. No tested "
                    "variant closes broad whole-atlas FDR."
                ),
            }
        ]
    )


def classify_variant(variant: str, sub: pd.DataFrame) -> str:
    true_q = int(pd.to_numeric(sub["true_lr_q_0_1"], errors="coerce").fillna(0).sum())
    if true_q > 0:
        return "candidate_for_fdr_followup"
    if variant == "main_calibrated":
        return "keep_as_current_reporting_baseline"
    if variant in {
        "background_suppressed",
        "background_guarded_training",
        "anchor_background_suppressed",
        "anchor_background_guarded_training",
        "anchor_background_range_guarded_training",
        "integrated_guarded_spatial_training",
    } and sub[
        "use_in_next_model"
    ].astype(str).str.contains("candidate_component|candidate_for_fdr_followup|leading_candidate", na=False).any():
        return "use_as_next_model_component_not_reporting_baseline"
    return "hold_out_not_adopted"


def summarize_boundary(variant: str, sub: pd.DataFrame) -> str:
    if variant == "main_calibrated":
        return "Use for current ranking/control-panel reporting; do not claim broad whole-atlas FDR."
    if variant == "background_suppressed":
        return "Use as calibration direction; exact-score permutation still has zero q<=0.1 discoveries."
    if variant == "background_guarded_training":
        return "Use as training direction; lymph-node top-k improves but exact-score permutation still has zero q<=0.1 discoveries."
    if variant == "anchor_background_suppressed":
        return "Use as strongest calibration direction; top-k recovery improves across datasets but exact-score permutation still has zero q<=0.1 discoveries."
    if variant == "anchor_background_guarded_training":
        return "Use as training-integration evidence; top-k and hard-negative guards improve but broad FDR remains unclosed and post-hoc risk can be cleaner."
    if variant == "anchor_background_range_guarded_training":
        return "Use as range-null training evidence; it preserves top-k anchors and improves selected q-values but broad FDR remains unclosed."
    if variant == "integrated_guarded_spatial_training":
        return "Use as leading FDR follow-up candidate; it has one broad true q<=0.1 hit but is not a reporting baseline until reproduced and CRC risk is repaired."
    if variant == "range_violation_guarded_training":
        return "Use as a diagnostic negative result; it reduces selected CRC risk but sacrifices lymph-node anchors and still has zero q<=0.1 discoveries."
    return "Exploratory only; not adopted."


def one(table: pd.DataFrame, dataset_key: str) -> dict[str, object]:
    sub = table[table["dataset_key"].astype(str).eq(dataset_key)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def one_topk(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    sub = table[table["dataset_key"].astype(str).eq(dataset_key) & table["top_k"].eq(top_k)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def mean_float(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def write_status(variant: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Goal 1 Model-Variant Scoreboard",
        "",
        "## Overall",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        f"- Tested variants: {int(story.iloc[0].n_variants)}.",
        f"- Total true LR q<=0.1 across tested variants: {int(story.iloc[0].total_true_lr_q_0_1)}.",
        "",
        "## Variant Decisions",
        "",
    ]
    for row in variant.itertuples(index=False):
        lines.append(
            f"- `{row.variant}` `{row.variant_decision}`: mean top10 true={row.mean_top10_true_lr:.2f}, "
            f"true q<=0.1={int(row.total_true_lr_q_0_1)}, all-LR q<=0.1={int(row.total_all_lr_q_0_1)}, "
            f"next-model candidates={int(row.n_next_model_candidate)}. {row.claim_boundary}"
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: model variants identify useful calibration directions but do not solve broad whole-atlas FDR.",
        "- Unsafe wording: any tested variant proves whole-atlas FDR control.",
    ]
    (OUT_DIR / "phase12_goal1_model_variant_scoreboard_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_goal1_model_variant_scoreboard",
        "inputs": [
            "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv",
            "atlas_model_expanded_range_calibrated/phase12_range_null_calibrated_dataset_summary.csv",
            "atlas_model_expanded_background_suppressed/phase12_background_suppressed_dataset_summary.csv",
            "atlas_model_expanded_background_guarded_training/phase12_background_guarded_training_dataset_summary.csv",
            "atlas_model_expanded_range_violation_guarded_training/phase12_range_violation_guarded_training_dataset_summary.csv",
            "atlas_model_expanded_anchor_background_suppressed/phase12_anchor_background_suppressed_dataset_summary.csv",
            "atlas_model_expanded_anchor_background_guarded_training/phase12_anchor_background_guarded_training_dataset_summary.csv",
            "atlas_model_expanded_anchor_background_range_guarded_training/phase12_anchor_background_range_guarded_training_dataset_summary.csv",
            "background_suppressed_permutation_comparison/phase12_background_suppressed_permutation_dataset_comparison.csv",
            "background_guarded_training_permutation_audit/phase12_background_guarded_training_permutation_dataset_summary.csv",
            "range_violation_guarded_training_permutation_audit/phase12_range_violation_guarded_training_permutation_dataset_summary.csv",
            "anchor_background_suppressed_permutation_audit/phase12_anchor_background_suppressed_permutation_dataset_summary.csv",
            "anchor_background_guarded_training_permutation_audit/phase12_anchor_background_guarded_training_permutation_dataset_summary.csv",
            "anchor_background_range_guarded_training_permutation_audit/phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv",
        ],
        "outputs": [
            "phase12_goal1_model_variant_dataset_scoreboard.csv",
            "phase12_goal1_model_variant_summary.csv",
            "phase12_goal1_model_variant_story.csv",
            "phase12_goal1_model_variant_scoreboard_status.md",
        ],
        "claim_boundary": "Variant scoreboard for Goal 1 model decisions; not a new FDR procedure.",
    }
    (OUT_DIR / "phase12_goal1_model_variant_scoreboard_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
