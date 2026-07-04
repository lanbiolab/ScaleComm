#!/usr/bin/env python
"""Summarize the remaining Phase 12 Goal 1 FDR bottleneck.

This is a writing and planning guardrail. It does not introduce a new FDR
procedure. It collates the existing broad, null-specific, matched, eligible and
control-panel audits so the manuscript cannot accidentally present scoped
diagnostics as solved whole-atlas discovery FDR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "goal1_fdr_bottleneck"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    broad = read_csv("atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv")
    strict = read_csv("strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv")
    null_specific = read_csv("null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv")
    matched = read_csv("matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv")
    range_weighted = read_csv("range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_dataset.csv")
    eligible = read_csv("discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv")
    target_decoy = read_csv("atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv")

    rows = []
    for dataset_key in sorted(set(broad.get("dataset_key", pd.Series(dtype=str)).astype(str))):
        b = one_row(broad, dataset_key)
        s = one_row(strict, dataset_key)
        n = one_row(null_specific, dataset_key)
        m = one_row(matched, dataset_key)
        w = one_row(range_weighted, dataset_key)
        e = one_row(eligible, dataset_key)
        t = one_row(target_decoy, dataset_key)
        rows.append(
            {
                "dataset_key": dataset_key,
                "broad_all_lr_q_0_1": to_int(b.get("all_lr_conservative_fdr_0_1")),
                "broad_true_lr_q_0_1": to_int(b.get("all_true_lr_conservative_fdr_0_1")),
                "strict_all_lr_q_0_1": to_int(s.get("all_lr_strict_permutation_q_0_1")),
                "strict_true_lr_q_0_1": to_int(s.get("all_true_lr_strict_permutation_q_0_1")),
                "dominant_bottleneck_null": text(n.get("dominant_bottleneck_null")),
                "pairing_identity_true_q_0_1": to_int(n.get("two_null_pairing_identity_true_q_0_1")),
                "spatial_distance_true_q_0_1": to_int(n.get("spatial_distance_true_q_0_1")),
                "matched_spatial_true_q_0_1": to_int(m.get("all_true_lr_matched_spatial_q_0_1")),
                "matched_all_null_true_q_0_1": to_int(m.get("all_true_lr_matched_all_null_q_0_1")),
                "range_weighted_true_q_0_1": to_int(w.get("weighted_true_q_0_1")),
                "range_weighted_control_q_0_1": to_int(w.get("weighted_control_q_0_1")),
                "strict_range_weighted_true_q_0_1": to_int(w.get("strict_weighted_true_q_0_1")),
                "strict_range_weighted_control_q_0_1": to_int(w.get("strict_weighted_control_q_0_1")),
                "eligible_matched_true_q_0_1": to_int(e.get("eligible_matched_true_q_0_1")),
                "eligible_matched_control_q_0_1": to_int(e.get("eligible_matched_control_q_0_1")),
                "target_decoy_top10_fdr": to_float(t.get("top10_primary_control_prefix_fdr")),
                "target_decoy_top100_fdr": to_float(t.get("top100_primary_control_prefix_fdr")),
                "goal1_fdr_state": classify_dataset(b, s, n, m, e, t),
                "next_model_or_null_action": next_action(n, e, t),
            }
        )

    dataset = pd.DataFrame(rows)
    story = summarize_story(dataset)
    dataset.to_csv(OUT_DIR / "phase12_goal1_fdr_bottleneck_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_goal1_fdr_bottleneck_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def read_csv(rel_path: str) -> pd.DataFrame:
    path = PHASE12_DIR / rel_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def one_row(table: pd.DataFrame, dataset_key: str) -> pd.Series:
    if table.empty or "dataset_key" not in table:
        return pd.Series(dtype=object)
    sub = table[table["dataset_key"].astype(str).eq(dataset_key)]
    return sub.iloc[0] if len(sub) else pd.Series(dtype=object)


def to_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def classify_dataset(
    broad: pd.Series,
    strict: pd.Series,
    null_specific: pd.Series,
    matched: pd.Series,
    eligible: pd.Series,
    target_decoy: pd.Series,
) -> str:
    broad_hits = to_int(broad.get("all_true_lr_conservative_fdr_0_1"))
    strict_hits = to_int(strict.get("all_true_lr_strict_permutation_q_0_1"))
    if broad_hits > 0 and strict_hits > 0:
        return "broad_whole_atlas_fdr_supported"

    eligible_true = to_int(eligible.get("eligible_matched_true_q_0_1"))
    eligible_ctrl = to_int(eligible.get("eligible_matched_control_q_0_1"))
    matched_true = to_int(matched.get("all_true_lr_matched_spatial_q_0_1"))
    top100_fdr = to_float(target_decoy.get("top100_control_fdr"))
    if eligible_true > 0 and eligible_ctrl == 0:
        return "eligible_universe_supported_broad_fdr_blocked"
    if matched_true > 0 and (pd.isna(top100_fdr) or top100_fdr <= 0.1):
        return "matched_spatial_or_control_support_broad_fdr_blocked"
    if eligible_true > 0:
        return "eligible_signal_with_control_caveat_broad_fdr_blocked"
    return "ranking_or_control_only_broad_fdr_blocked"


def next_action(null_specific: pd.Series, eligible: pd.Series, target_decoy: pd.Series) -> str:
    bottleneck = text(null_specific.get("dominant_bottleneck_null"))
    eligible_ctrl = to_int(eligible.get("eligible_matched_control_q_0_1"))
    top10_fdr = to_float(target_decoy.get("top10_control_fdr"))
    if "spatial_distance" in bottleneck:
        return "train_or_calibrate_against_range_matched_spatial_distance_null"
    if eligible_ctrl > 0:
        return "tighten_discovery_eligible_control_penalty_and_receptor_ligand_shared_decoys"
    if not pd.isna(top10_fdr) and top10_fdr > 0.1:
        return "reduce_top10_control_contamination_before_claiming_fdr"
    return "increase_end_to_end_null_replicates_or_keep_as_scoped_diagnostic"


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    broad_supported = int(dataset["goal1_fdr_state"].eq("broad_whole_atlas_fdr_supported").sum()) if len(dataset) else 0
    spatial_bottleneck = int(dataset["dominant_bottleneck_null"].astype(str).str.contains("spatial_distance").sum()) if len(dataset) else 0
    eligible_support = int(dataset["eligible_matched_true_q_0_1"].gt(0).sum()) if len(dataset) else 0
    range_weighted_support = int(dataset["strict_range_weighted_true_q_0_1"].gt(0).sum()) if len(dataset) else 0
    top100_clean = int(dataset["target_decoy_top100_fdr"].le(0.1).sum()) if len(dataset) else 0
    if broad_supported == len(dataset) and len(dataset):
        overall = "goal1_broad_fdr_supported"
    else:
        overall = "goal1_broad_fdr_not_closed_spatial_null_bottleneck"
    return pd.DataFrame(
        [
            {
                "topic": "goal1_fdr_bottleneck",
                "overall_state": overall,
                "n_datasets": int(len(dataset)),
                "n_broad_supported": broad_supported,
                "n_spatial_distance_bottleneck": spatial_bottleneck,
                "n_eligible_universe_with_true_hits": eligible_support,
                "n_range_weighted_strict_true_hit_datasets": range_weighted_support,
                "n_target_decoy_top100_clean": top100_clean,
                "manuscript_boundary": (
                    "Use this as the Goal 1 blocker table. SpatialSelf has calibrated ranking, "
                    "hard-control suppression and scoped FDR diagnostics, but broad all-covered-LR "
                    "whole-atlas discovery FDR is not closed."
                ),
            }
        ]
    )


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Goal 1 FDR Bottleneck",
        "",
        "## Dataset-Level State",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: `{row.goal1_fdr_state}`; broad true q<=0.1={row.broad_true_lr_q_0_1}, "
            f"spatial true q<=0.1={row.spatial_distance_true_q_0_1}, "
            f"eligible matched true q<=0.1={row.eligible_matched_true_q_0_1}, "
            f"strict range-weighted true q<=0.1={row.strict_range_weighted_true_q_0_1}, "
            f"top100 target-decoy FDR={fmt(row.target_decoy_top100_fdr)}; next `{row.next_model_or_null_action}`."
        )
    if not story.empty:
        row = story.iloc[0]
        lines += [
            "",
            "## Manuscript Boundary",
            "",
            f"- Overall: `{row.overall_state}`.",
            f"- {row.manuscript_boundary}",
        ]
    (OUT_DIR / "phase12_goal1_fdr_bottleneck_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_goal1_fdr_bottleneck_summary",
        "claim_boundary": "Collates existing audits; does not create a new FDR procedure.",
        "outputs": [
            "phase12_goal1_fdr_bottleneck_dataset.csv",
            "phase12_goal1_fdr_bottleneck_story.csv",
            "phase12_goal1_fdr_bottleneck_status.md",
        ],
    }
    (OUT_DIR / "phase12_goal1_fdr_bottleneck_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    val = to_float(value)
    if pd.isna(val):
        return "NA"
    return f"{val:.3g}"


if __name__ == "__main__":
    main()
