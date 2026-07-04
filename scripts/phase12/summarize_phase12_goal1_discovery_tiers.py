#!/usr/bin/env python
"""Summarize Goal 1 whole-atlas discovery evidence by claim tier.

This table is intentionally conservative. Phase 12 now has several FDR-like
audits with different scopes: broad all-covered LR permutation, target-decoy
control contamination, matched/spatial diagnostics, discovery-eligible subsets,
and pre-specified anchor panels. Mixing those scopes would overstate the result,
so this script makes the allowed claim boundary explicit in one source table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "goal1_discovery_tiers"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows = build_dataset_rows()
    story_rows = build_story_rows(dataset_rows)
    dataset_table = pd.DataFrame(dataset_rows)
    story_table = pd.DataFrame(story_rows)
    dataset_table.to_csv(OUT_DIR / "phase12_goal1_discovery_tier_dataset.csv", index=False)
    story_table.to_csv(OUT_DIR / "phase12_goal1_discovery_tier_story.csv", index=False)
    write_status(dataset_table, story_table)
    manifest = {
        "task": "phase12_goal1_discovery_tiers",
        "outputs": [
            rel(OUT_DIR / "phase12_goal1_discovery_tier_dataset.csv"),
            rel(OUT_DIR / "phase12_goal1_discovery_tier_story.csv"),
            rel(OUT_DIR / "phase12_goal1_discovery_tier_status.md"),
        ],
        "claim_boundary": (
            "Broad all-covered-LR whole-atlas FDR remains incomplete. "
            "Eligible, matched, targeted, and target-decoy rows are narrower diagnostics."
        ),
    }
    (OUT_DIR / "phase12_goal1_discovery_tier_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(dataset_table.to_string(index=False))
    print()
    print(story_table.to_string(index=False))


def build_dataset_rows() -> list[dict[str, object]]:
    calibrated = read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv")
    perm = read_csv(PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv")
    target_decoy = read_csv(PHASE12_DIR / "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv")
    strict = read_csv(PHASE12_DIR / "strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv")
    matched = read_csv(PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv")
    eligible = read_csv(PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv")
    targeted = read_csv(PHASE12_DIR / "targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv")

    datasets = sorted(
        set(calibrated.get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
        | set(perm.get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
        | set(target_decoy.get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, object]] = []
    for dataset_key in datasets:
        cal = first(calibrated, dataset_key)
        per = first(perm, dataset_key)
        decoy = first(target_decoy, dataset_key)
        strict_row = first(strict, dataset_key)
        matched_row = first(matched, dataset_key)
        eligible_row = first(eligible, dataset_key)
        targeted_row = first(targeted, dataset_key)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": val(cal, "n_lr", per, strict_row),
                "top10_true_lr": val(cal, "top10_true_lr", per, strict_row),
                "top10_high_expression_negative": val(cal, "top10_high_expression_negative", strict_row),
                "top10_negative_or_risk_fraction": val(cal, "top10_negative_or_risk_fraction", per),
                "top10_primary_control_fdr": val(decoy, "top10_primary_control_prefix_fdr", strict_row),
                "top100_primary_control_fdr": val(decoy, "top100_primary_control_prefix_fdr", strict_row),
                "broad_all_lr_q_0_1": val(per, "all_lr_conservative_fdr_0_1", strict_row),
                "broad_true_lr_q_0_1": val(per, "all_true_lr_conservative_fdr_0_1", strict_row),
                "strict_all_lr_q_0_1": val(strict_row, "all_lr_strict_permutation_q_0_1"),
                "matched_spatial_true_q_0_1": val(matched_row, "all_true_lr_matched_spatial_q_0_1"),
                "matched_all_null_true_q_0_1": val(matched_row, "all_true_lr_matched_all_null_q_0_1"),
                "strict_matched_all_null_true_q_0_1": val(matched_row, "all_true_lr_strict_matched_all_null_q_0_1"),
                "eligible_lr": val(eligible_row, "n_eligible_lr"),
                "eligible_matched_true_q_0_1": val(eligible_row, "eligible_matched_true_q_0_1"),
                "eligible_strict_matched_true_q_0_1": val(eligible_row, "eligible_strict_matched_true_q_0_1"),
                "eligible_matched_control_q_0_1": val(eligible_row, "eligible_matched_control_q_0_1"),
                "targeted_anchor_top10": val(targeted_row, "n_rank_top10"),
                "targeted_panel_q_0_1": val(targeted_row, "n_targeted_panel_q_0_1"),
                "targeted_matched_q_0_1": val(targeted_row, "n_matched_control_q_0_1"),
                "best_supported_lr": val(targeted_row, "best_supported_lr", cal),
                "dataset_support_tier": dataset_support_tier(cal, per, decoy, matched_row, eligible_row, targeted_row),
                "dataset_claim_boundary": dataset_claim_boundary(dataset_key, cal, per, decoy, matched_row, eligible_row, targeted_row),
            }
        )
    return rows


def build_story_rows(dataset_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dataset_table = pd.DataFrame(dataset_rows)
    eligible_story = read_csv(PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv")
    eligible_method_story = read_csv(PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv")
    matched_story = read_csv(PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv")
    closure_story = read_csv(PHASE12_DIR / "atlas_fdr_closure/phase12_atlas_fdr_closure_story.csv")
    official_method = read_csv(
        PHASE12_DIR / "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_method_summary.csv"
    )

    n_datasets = int(dataset_table["dataset_key"].nunique()) if not dataset_table.empty else 0
    broad_hits = int(num(dataset_table.get("broad_all_lr_q_0_1", pd.Series(dtype=float))).sum()) if not dataset_table.empty else 0
    strict_hits = int(num(dataset_table.get("strict_all_lr_q_0_1", pd.Series(dtype=float))).sum()) if not dataset_table.empty else 0
    top10_clean = int((num(dataset_table.get("top10_primary_control_fdr", pd.Series(dtype=float))) <= 0.1).sum()) if not dataset_table.empty else 0
    top100_clean = int((num(dataset_table.get("top100_primary_control_fdr", pd.Series(dtype=float))) <= 0.1).sum()) if not dataset_table.empty else 0
    eligible_true = int(num(dataset_table.get("eligible_matched_true_q_0_1", pd.Series(dtype=float))).sum()) if not dataset_table.empty else 0
    eligible_controls = int(num(dataset_table.get("eligible_matched_control_q_0_1", pd.Series(dtype=float))).sum()) if not dataset_table.empty else 0
    targeted_matched = int(num(dataset_table.get("targeted_matched_q_0_1", pd.Series(dtype=float))).sum()) if not dataset_table.empty else 0

    baseline_contaminated = 0
    if not official_method.empty and "summary_call" in official_method:
        baseline_contaminated = int(official_method["summary_call"].astype(str).str.contains("contaminated|risk", na=False).sum())

    rows = [
        story(
            "broad_all_covered_lr_permutation_fdr",
            "not_supported",
            f"{n_datasets} expanded datasets; broad all-LR q<=0.1 discoveries={broad_hits}; strict all-LR q<=0.1 discoveries={strict_hits}.",
            "Do not claim broad whole-atlas discovery FDR or unbiased whole-atlas superiority.",
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv; strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv",
        ),
        story(
            "target_decoy_control_fdr",
            "supportive_contamination_audit",
            f"Top10 control-FDR clean datasets={top10_clean}/{n_datasets}; top100 control-FDR clean datasets={top100_clean}/{n_datasets}.",
            "Can claim hard-control contamination is quantified and often lower after calibration; cannot call this native real-ST discovery FDR.",
            "atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv",
        ),
        story(
            "matched_spatial_fdr_diagnostic",
            safe_story_status(matched_story, "overall_matched_spatial_call", "matched_spatial_not_available"),
            story_value(
                matched_story,
                "claim_boundary",
                "Matched spatial diagnostics are available and should be interpreted as post-hoc matched null evidence.",
            ),
            "Use only as a matched-null diagnostic; it does not replace end-to-end whole-atlas FDR.",
            "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_story.csv",
        ),
        story(
            "discovery_eligible_universe_fdr",
            safe_story_status(eligible_story, "overall_discovery_eligible_call", "eligible_not_available"),
            f"Eligible matched true q<=0.1={eligible_true}; eligible matched controls q<=0.1={eligible_controls}. "
            + story_value(eligible_story, "claim_boundary", ""),
            "This supports a pre-defined functional/expression-supported LR universe, not the broader all-covered-LR universe.",
            "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv",
        ),
        story(
            "targeted_anchor_panel_fdr",
            "targeted_anchor_supportive_with_caveats" if targeted_matched > 0 else "targeted_anchor_not_supported",
            f"Targeted matched anchor q<=0.1 counts across datasets={targeted_matched}. Strongest current anchors: "
            + "; ".join(dataset_table["dataset_key"].astype(str) + "=" + dataset_table["best_supported_lr"].astype(str))
            if not dataset_table.empty
            else "No targeted anchor table available.",
            "Can support pre-specified biological anchors; cannot be sold as discovery from the full LR atlas.",
            "targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv",
        ),
        story(
            "adapter_light_baseline_comparison",
            safe_story_status(eligible_method_story, "overall_eligible_method_call", "adapter_light_comparison_available"),
            f"Adapter-light baseline methods with contaminated/risk-prone method summaries={baseline_contaminated}. "
            + story_value(eligible_method_story, "claim_boundary", ""),
            "This supports same-universe ranking/contamination differences, not fully tuned official end-to-end method superiority.",
            "atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_method_summary.csv; discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv",
        ),
        story(
            "goal1_overall_claim",
            "goal1_partial_not_broad_fdr_complete",
            story_value(
                closure_story,
                "claim_boundary",
                "Goal 1 has strong ranking/control/eligible/targeted evidence, but broad whole-atlas FDR remains incomplete.",
            ),
            "Write Goal 1 as calibrated whole-LR ranking plus hard-control suppression and scoped FDR diagnostics. Do not write it as solved global discovery FDR.",
            "atlas_fdr_closure/phase12_atlas_fdr_closure_story.csv; completion_audit/phase12_requirement_completion_audit.csv",
        ),
    ]
    return rows


def dataset_support_tier(
    cal: pd.Series,
    per: pd.Series,
    decoy: pd.Series,
    matched: pd.Series,
    eligible: pd.Series,
    targeted: pd.Series,
) -> str:
    broad_hits = int_or_zero(val(per, "all_lr_conservative_fdr_0_1"))
    top10_fdr = float_or_na(val(decoy, "top10_primary_control_prefix_fdr"))
    matched_all = int_or_zero(val(matched, "all_true_lr_matched_all_null_q_0_1"))
    eligible_true = int_or_zero(val(eligible, "eligible_matched_true_q_0_1"))
    targeted_matched = int_or_zero(val(targeted, "n_matched_control_q_0_1"))
    top10_true = int_or_zero(val(cal, "top10_true_lr"))
    if broad_hits > 0:
        return "A_broad_whole_atlas_fdr_supported"
    if top10_fdr <= 0.1 and matched_all > 0 and eligible_true > 0:
        return "B_strong_scoped_fdr_and_clean_controls"
    if targeted_matched > 0 or eligible_true > 0:
        return "C_scoped_or_targeted_support_with_caveat"
    if top10_true > 0:
        return "D_ranking_support_only"
    return "E_failure_or_stress_case"


def dataset_claim_boundary(
    dataset_key: str,
    cal: pd.Series,
    per: pd.Series,
    decoy: pd.Series,
    matched: pd.Series,
    eligible: pd.Series,
    targeted: pd.Series,
) -> str:
    top10_true = int_or_zero(val(cal, "top10_true_lr"))
    top10_fdr = val(decoy, "top10_primary_control_prefix_fdr")
    top100_fdr = val(decoy, "top100_primary_control_prefix_fdr")
    matched_all = int_or_zero(val(matched, "all_true_lr_matched_all_null_q_0_1"))
    eligible_true = int_or_zero(val(eligible, "eligible_matched_true_q_0_1"))
    targeted_matched = int_or_zero(val(targeted, "n_matched_control_q_0_1"))
    broad_hits = int_or_zero(val(per, "all_lr_conservative_fdr_0_1"))
    if broad_hits > 0:
        return "Broad whole-atlas discovery FDR is supported for this dataset."
    parts = [
        f"top10 true LR={top10_true}",
        f"top10 control FDR={fmt(top10_fdr)}",
        f"top100 control FDR={fmt(top100_fdr)}",
        f"matched all-null true hits={matched_all}",
        f"eligible matched true hits={eligible_true}",
        f"targeted matched anchor hits={targeted_matched}",
    ]
    if dataset_key == "visium_hd_crc_full":
        parts.append("Use CRC as a high-resolution stress case unless additional calibration removes control contamination.")
    elif dataset_key == "post_xenium_colon_full":
        parts.append("Use SPP1-CD44 as the main scoped biological story, not as broad all-LR discovery FDR.")
    elif dataset_key == "human_lymph_node_visium_raw_full":
        parts.append("Use as the cleanest whole-LR ranking/control case and conservative second-story support.")
    return "; ".join(parts)


def story(topic: str, support_status: str, evidence_summary: str, claim_boundary: str, evidence_file: str) -> dict[str, object]:
    return {
        "topic": topic,
        "support_status": support_status,
        "evidence_summary": evidence_summary,
        "claim_boundary": claim_boundary,
        "evidence_file": "outputs/phase12/" + evidence_file,
    }


def write_status(dataset_table: pd.DataFrame, story_table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Goal 1 Discovery Tiers",
        "",
        "## Dataset Tiers",
        "",
    ]
    if dataset_table.empty:
        lines.append("- No dataset-tier evidence available.")
    else:
        for row in dataset_table.itertuples(index=False):
            lines.append(f"- **{row.dataset_key}**: `{row.dataset_support_tier}`")
            lines.append(f"  Evidence: {row.dataset_claim_boundary}")
    lines += ["", "## Claim Tiers", ""]
    for row in story_table.itertuples(index=False):
        lines.append(f"- **{row.topic}** (`{row.support_status}`): {row.evidence_summary}")
        lines.append(f"  Boundary: {row.claim_boundary}")
    lines += [
        "",
        "## Manuscript-Safe Bottom Line",
        "",
        "SpatialSelf currently supports calibrated whole-LR ranking, hard-control suppression, and scoped FDR diagnostics across expanded real-background atlases. Broad all-covered-LR whole-atlas discovery FDR remains incomplete and should not be claimed.",
    ]
    (OUT_DIR / "phase12_goal1_discovery_tier_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first(table: pd.DataFrame, dataset_key: str) -> pd.Series:
    if table.empty or "dataset_key" not in table:
        return pd.Series(dtype=object)
    sub = table[table["dataset_key"].astype(str).eq(dataset_key)]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def val(row: pd.Series, column: str, *fallback_rows: pd.Series) -> object:
    rows = (row, *fallback_rows)
    for item in rows:
        if isinstance(item, pd.Series) and column in item and pd.notna(item[column]):
            return item[column]
    return pd.NA


def safe_story_status(table: pd.DataFrame, column: str, default: str) -> str:
    if table.empty or column not in table or pd.isna(table.iloc[0][column]):
        return default
    return str(table.iloc[0][column])


def story_value(table: pd.DataFrame, column: str, default: str) -> str:
    if table.empty or column not in table or pd.isna(table.iloc[0][column]):
        return default
    return str(table.iloc[0][column])


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def int_or_zero(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_or_na(value: object) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
