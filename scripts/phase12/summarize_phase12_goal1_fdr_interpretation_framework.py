#!/usr/bin/env python
"""Build a manuscript-safe Goal 1 FDR interpretation framework.

This script does not create a new statistical test. It collates existing Phase
12 broad, null-specific, matched, target-decoy, discovery-eligible and targeted
anchor audits into one layer-by-layer table. The purpose is to make the
manuscript claim boundary explicit:

- broad all-covered-LR FDR is not closed;
- target-decoy/control FDR is a contamination audit;
- matched/guarded/eligible/targeted layers support narrower claims;
- spatial-distance null calibration is the main blocker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "goal1_fdr_interpretation_framework"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    layers = build_layer_table(tables)
    dataset = summarize_dataset_layers(layers)
    story = summarize_story(layers, dataset)

    layers.to_csv(OUT_DIR / "phase12_goal1_fdr_interpretation_layers.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_goal1_fdr_interpretation_dataset_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_goal1_fdr_interpretation_story.csv", index=False)
    write_status(layers, dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "bottleneck": read_csv("goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv"),
        "discovery_tier": read_csv("goal1_discovery_tiers/phase12_goal1_discovery_tier_dataset.csv"),
        "target_decoy": read_csv("atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv"),
        "guarded": read_csv("guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_dataset.csv"),
        "eligible": read_csv("discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv"),
        "null_specific": read_csv("null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv"),
        "matched": read_csv("matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv"),
        "strict": read_csv("strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv"),
        "targeted": read_csv("targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv"),
        "supplementary_panel": read_csv(
            "supplementary_cross_platform_expanded_panel_atlas/phase12_supplementary_expanded_panel_dataset_summary.csv"
        ),
    }


def build_layer_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dataset_keys = sorted(
        set(tables["bottleneck"].get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
        | set(tables["discovery_tier"].get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
        | set(tables["guarded"].get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, object]] = []
    for dataset_key in dataset_keys:
        b = one_row(tables["bottleneck"], dataset_key)
        d = one_row(tables["discovery_tier"], dataset_key)
        t = one_row(tables["target_decoy"], dataset_key)
        g = one_row(tables["guarded"], dataset_key)
        e = one_row(tables["eligible"], dataset_key)
        n = one_row(tables["null_specific"], dataset_key)
        m = one_row(tables["matched"], dataset_key)
        s = one_row(tables["strict"], dataset_key)
        targeted = one_row(tables["targeted"], dataset_key)

        rows.extend(
            [
                layer(
                    dataset_key,
                    "broad_all_covered_lr_fdr",
                    "broad_all_covered_lr",
                    count=b.get("broad_true_lr_q_0_1"),
                    denominator=d.get("n_lr"),
                    control_count=b.get("broad_all_lr_q_0_1"),
                    support_scope="not_supported" if to_int(b.get("broad_true_lr_q_0_1")) == 0 else "broad_supported",
                    allowed_claim="Do not claim broad whole-atlas discovery FDR unless this layer has true LR q<=0.1 discoveries.",
                    forbidden_claim="SpatialSelf controls unbiased whole-atlas FDR over every covered LR pair.",
                    evidence_file="goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "strict_discovery_score_fdr",
                    "broad_all_covered_lr_strict_score",
                    count=s.get("all_true_lr_strict_permutation_q_0_1"),
                    denominator=s.get("n_lr"),
                    control_count=s.get("top10_primary_control_decoy"),
                    support_scope=scope_from_count(s.get("all_true_lr_strict_permutation_q_0_1"), "broad_strict_supported"),
                    allowed_claim="Strict score improves ranking/control diagnostics when supported, but remains a broad FDR layer.",
                    forbidden_claim="Strict top-k enrichment is equivalent to permutation-calibrated whole-atlas FDR.",
                    evidence_file="strict_atlas_discovery_audit/phase12_strict_atlas_discovery_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "spatial_distance_null",
                    "pooled_spatial_distance_null",
                    count=n.get("spatial_distance_true_q_0_1"),
                    denominator=n.get("n_true_lr"),
                    control_count=n.get("spatial_distance_q_0_1"),
                    support_scope="primary_blocker" if to_int(n.get("spatial_distance_true_q_0_1")) == 0 else "spatial_distance_supported",
                    allowed_claim="Use this layer to identify whether spatial-distance calibration blocks broad FDR.",
                    forbidden_claim="Receiver/LR-identity support alone proves spatially specific CCC.",
                    evidence_file="null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "receiver_pairing_lr_identity_null",
                    "pairing_identity_diagnostic",
                    count=n.get("two_null_pairing_identity_true_q_0_1"),
                    denominator=n.get("n_true_lr"),
                    control_count=n.get("two_null_pairing_identity_q_0_1"),
                    support_scope=scope_from_count(n.get("two_null_pairing_identity_true_q_0_1"), "pairing_identity_supported"),
                    allowed_claim="Can support receiver pairing and LR-identity specificity, not spatial-distance FDR.",
                    forbidden_claim="Pairing/identity support closes all-null whole-atlas discovery.",
                    evidence_file="null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "matched_spatial_null",
                    "matched_spatial_diagnostic",
                    count=m.get("all_true_lr_matched_spatial_q_0_1"),
                    denominator=m.get("n_true_lr"),
                    control_count=m.get("all_lr_matched_spatial_q_0_1"),
                    support_scope=scope_from_count(m.get("all_true_lr_matched_spatial_q_0_1"), "matched_spatial_supported"),
                    allowed_claim="Can support matched-null spatial specificity for selected or matched LR universes.",
                    forbidden_claim="Matched spatial diagnostics replace the conservative pooled spatial-distance FDR.",
                    evidence_file="matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "guarded_matched_full_lr",
                    "guarded_full_lr_matched_null",
                    count=g.get("guarded_matched_all_null_n_true_supported"),
                    denominator=g.get("n_true_lr"),
                    control_count=sum_values(
                        g.get("guarded_matched_all_null_n_primary_control_supported"),
                        g.get("guarded_matched_all_null_n_receptor_shared_supported"),
                        g.get("guarded_matched_all_null_n_ligand_shared_supported"),
                        g.get("guarded_matched_all_null_n_background_supported"),
                    ),
                    support_scope=guarded_scope(g),
                    allowed_claim="Can support selected full-LR matched-null discoveries when true support is present and controls are zero.",
                    forbidden_claim="Selected guarded matched support solves broad pooled whole-atlas FDR.",
                    evidence_file="guarded_matched_atlas_fdr/phase12_guarded_matched_atlas_fdr_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "target_decoy_control_fdr_top100",
                    "hard_control_contamination_audit",
                    count=int(to_float(t.get("top100_primary_control_prefix_fdr")) <= 0.1)
                    if not np.isnan(to_float(t.get("top100_primary_control_prefix_fdr")))
                    else 0,
                    denominator=1,
                    control_count=t.get("n_primary_control_decoy"),
                    support_scope="control_contamination_clean_top100"
                    if to_float(t.get("top100_primary_control_prefix_fdr")) <= 0.1
                    else "control_contamination_caveat",
                    allowed_claim="Can claim hard-control contamination is quantified and top100 control FDR is clean when supported.",
                    forbidden_claim="Target-decoy control FDR is native real-ST ground-truth discovery FDR.",
                    evidence_file="atlas_model_expanded/target_decoy_fdr/phase12_expanded_target_decoy_dataset_summary.csv",
                    metric_value=t.get("top100_primary_control_prefix_fdr"),
                ),
                layer(
                    dataset_key,
                    "discovery_eligible_universe",
                    "predefined_functional_expression_supported_universe",
                    count=e.get("eligible_matched_true_q_0_1"),
                    denominator=e.get("n_eligible_true_lr"),
                    control_count=e.get("eligible_matched_control_q_0_1"),
                    support_scope=eligible_scope(e),
                    allowed_claim="Can claim support inside a predefined eligible LR universe with explicit control caveats.",
                    forbidden_claim="Eligible-universe FDR is the same as all-covered-LR whole-atlas FDR.",
                    evidence_file="discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv",
                ),
                layer(
                    dataset_key,
                    "targeted_anchor_panel",
                    "prespecified_biological_anchor_panel",
                    count=targeted.get("n_matched_control_q_0_1"),
                    denominator=targeted.get("n_anchor_lr"),
                    control_count=targeted.get("n_hard_negative_q_0_1"),
                    support_scope=scope_from_count(targeted.get("n_matched_control_q_0_1"), "targeted_anchor_supported"),
                    allowed_claim="Can validate pre-specified biological anchors and story LR.",
                    forbidden_claim="Targeted anchors are unbiased discoveries from the full LR atlas.",
                    evidence_file="targeted_matched_null/phase12_targeted_matched_null_dataset_summary.csv",
                ),
                layer(
                    dataset_key,
                    "range_weighted_diagnostic",
                    "weighted_range_matched_sensitivity",
                    count=b.get("strict_range_weighted_true_q_0_1"),
                    denominator=d.get("n_lr"),
                    control_count=b.get("strict_range_weighted_control_q_0_1"),
                    support_scope=range_weighted_scope(b),
                    allowed_claim="Can be reported as a sensitivity analysis showing range/expression prior calibration helps true anchors.",
                    forbidden_claim="Weighted diagnostics replace the original unweighted pooled FDR.",
                    evidence_file="goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_dataset.csv",
                ),
            ]
        )

    panel = tables["supplementary_panel"]
    if not panel.empty:
        for row in panel.itertuples(index=False):
            rows.append(
                layer(
                    str(row.dataset_key),
                    "supplementary_expanded_panel",
                    "cross_platform_2011_gene_panel",
                    count=row.top10_true_lr,
                    denominator=row.n_lr,
                    control_count=row.hard_negative_best_comparison_beats,
                    support_scope=str(row.supplementary_call),
                    allowed_claim="Can be used as cross-platform expanded-panel support.",
                    forbidden_claim="This is strict full-transcriptome whole-atlas FDR closure.",
                    evidence_file=(
                        "supplementary_cross_platform_expanded_panel_atlas/"
                        "phase12_supplementary_expanded_panel_dataset_summary.csv"
                    ),
                    metric_value=row.top10_negative_or_risk_fraction,
                )
            )
    return pd.DataFrame(rows)


def layer(
    dataset_key: str,
    layer_name: str,
    null_or_scope: str,
    *,
    count: object,
    denominator: object,
    control_count: object,
    support_scope: str,
    allowed_claim: str,
    forbidden_claim: str,
    evidence_file: str,
    metric_value: object = np.nan,
) -> dict[str, object]:
    supported_count = to_int(count)
    control_supported_count = to_int(control_count)
    denom = to_int(denominator)
    return {
        "dataset_key": dataset_key,
        "layer_name": layer_name,
        "null_or_scope": null_or_scope,
        "supported_count": supported_count,
        "denominator": denom,
        "control_or_risk_count": control_supported_count,
        "metric_value": to_float(metric_value),
        "support_scope": support_scope,
        "claim_tier": classify_layer(layer_name, supported_count, control_supported_count, support_scope),
        "allowed_claim": allowed_claim,
        "forbidden_claim": forbidden_claim,
        "evidence_file": f"outputs/phase12/{evidence_file}",
    }


def classify_layer(layer_name: str, count: int, control_count: int, support_scope: str) -> str:
    if layer_name in {"broad_all_covered_lr_fdr", "strict_discovery_score_fdr"}:
        return "broad_fdr_supported" if count > 0 else "broad_fdr_not_supported"
    if layer_name == "spatial_distance_null":
        return "spatial_null_supported" if count > 0 else "spatial_null_blocker"
    if layer_name == "guarded_matched_full_lr":
        return "selected_guarded_support" if count > 0 and control_count == 0 else "not_guarded_supported"
    if layer_name == "target_decoy_control_fdr_top100":
        return "control_contamination_supported" if "clean" in support_scope else "control_contamination_caveat"
    if layer_name == "discovery_eligible_universe":
        if count > 0 and control_count == 0:
            return "eligible_clean_support"
        if count > 0:
            return "eligible_support_with_control_caveat"
        return "eligible_not_supported"
    if layer_name == "supplementary_expanded_panel":
        return "supplementary_cross_platform_support"
    return "scoped_support" if count > 0 else "not_supported"


def summarize_dataset_layers(layers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in layers.groupby("dataset_key"):
        broad = sub[sub["claim_tier"].eq("broad_fdr_supported")]
        blockers = sub[sub["claim_tier"].isin(["broad_fdr_not_supported", "spatial_null_blocker"])]
        clean_controls = sub[sub["claim_tier"].eq("control_contamination_supported")]
        scoped = sub[
            sub["claim_tier"].isin(
                [
                    "selected_guarded_support",
                    "eligible_clean_support",
                    "eligible_support_with_control_caveat",
                    "scoped_support",
                    "supplementary_cross_platform_support",
                ]
            )
        ]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_layers": int(len(sub)),
                "n_broad_fdr_supported_layers": int(len(broad)),
                "n_blocking_layers": int(len(blockers)),
                "n_control_contamination_supported_layers": int(len(clean_controls)),
                "n_scoped_or_supplementary_support_layers": int(len(scoped)),
                "has_spatial_null_blocker": bool(sub["claim_tier"].eq("spatial_null_blocker").any()),
                "has_selected_guarded_support": bool(sub["claim_tier"].eq("selected_guarded_support").any()),
                "has_eligible_clean_support": bool(sub["claim_tier"].eq("eligible_clean_support").any()),
                "dataset_fdr_interpretation": dataset_interpretation(sub),
                "allowed_dataset_claim": allowed_dataset_claim(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(layers: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    n_strict = int(dataset[~dataset["dataset_key"].astype(str).str.contains("expanded_panel")]["dataset_key"].nunique())
    n_panel = int(dataset[dataset["dataset_key"].astype(str).str.contains("expanded_panel")]["dataset_key"].nunique())
    broad_supported = int(dataset["n_broad_fdr_supported_layers"].gt(0).sum()) if len(dataset) else 0
    spatial_blocked = int(dataset["has_spatial_null_blocker"].sum()) if len(dataset) else 0
    selected_guarded = int(dataset["has_selected_guarded_support"].sum()) if len(dataset) else 0
    eligible_clean = int(dataset["has_eligible_clean_support"].sum()) if len(dataset) else 0
    control_supported = int(dataset["n_control_contamination_supported_layers"].gt(0).sum()) if len(dataset) else 0
    overall = "broad_fdr_closed" if broad_supported >= max(n_strict, 1) else "broad_fdr_not_closed_scoped_support_available"
    return pd.DataFrame(
        [
            {
                "topic": "goal1_fdr_interpretation_framework",
                "overall_call": overall,
                "n_strict_full_transcriptome_datasets": n_strict,
                "n_supplementary_panel_datasets": n_panel,
                "n_datasets_with_broad_fdr_support": broad_supported,
                "n_datasets_with_spatial_null_blocker": spatial_blocked,
                "n_datasets_with_selected_guarded_support": selected_guarded,
                "n_datasets_with_eligible_clean_support": eligible_clean,
                "n_datasets_with_control_contamination_support": control_supported,
                "manuscript_claim": (
                    "SpatialSelf supports calibrated whole-LR ranking, hard-control contamination auditing, "
                    "selected guarded matched-null discoveries, eligible-universe support, and supplementary "
                    "cross-platform expanded-panel validation, but broad all-covered-LR discovery FDR remains unclosed."
                ),
                "next_requirement": (
                    "Use anchor-preserving spatial-null training plus background/receptor-shared hard-negative "
                    "suppression, then rerun exact-score and end-to-end null audits."
                ),
            }
        ]
    )


def dataset_interpretation(sub: pd.DataFrame) -> str:
    if sub["claim_tier"].eq("broad_fdr_supported").any():
        return "broad_fdr_supported"
    if sub["claim_tier"].eq("selected_guarded_support").any():
        return "selected_guarded_support_broad_fdr_blocked"
    if sub["claim_tier"].eq("eligible_clean_support").any():
        return "eligible_clean_support_broad_fdr_blocked"
    if sub["claim_tier"].eq("eligible_support_with_control_caveat").any():
        return "eligible_support_with_control_caveat_broad_fdr_blocked"
    if sub["claim_tier"].eq("supplementary_cross_platform_support").any():
        return "supplementary_cross_platform_support_only"
    return "ranking_or_control_support_broad_fdr_blocked"


def allowed_dataset_claim(sub: pd.DataFrame) -> str:
    dataset_key = str(sub["dataset_key"].iloc[0])
    interpretation = dataset_interpretation(sub)
    if interpretation == "selected_guarded_support_broad_fdr_blocked":
        return f"{dataset_key}: selected guarded matched-null support; keep broad FDR caveat."
    if interpretation == "eligible_clean_support_broad_fdr_blocked":
        return f"{dataset_key}: eligible-universe support with clean controls; not all-LR FDR."
    if interpretation == "eligible_support_with_control_caveat_broad_fdr_blocked":
        return f"{dataset_key}: eligible-universe signal with control caveat; use as scoped evidence only."
    if interpretation == "supplementary_cross_platform_support_only":
        return f"{dataset_key}: supplementary expanded-panel cross-platform support only."
    return f"{dataset_key}: ranking/control diagnostics only; do not claim discovery FDR."


def guarded_scope(row: pd.Series) -> str:
    true_count = to_int(row.get("guarded_matched_all_null_n_true_supported"))
    controls = sum_values(
        row.get("guarded_matched_all_null_n_primary_control_supported"),
        row.get("guarded_matched_all_null_n_receptor_shared_supported"),
        row.get("guarded_matched_all_null_n_ligand_shared_supported"),
        row.get("guarded_matched_all_null_n_background_supported"),
    )
    if true_count > 0 and controls == 0:
        return "clean_selected_guarded_support"
    if true_count > 0:
        return "guarded_support_with_control_caveat"
    return "not_guarded_supported"


def eligible_scope(row: pd.Series) -> str:
    true_count = to_int(row.get("eligible_matched_true_q_0_1"))
    controls = to_int(row.get("eligible_matched_control_q_0_1"))
    if true_count > 0 and controls == 0:
        return "eligible_clean_support"
    if true_count > 0:
        return "eligible_support_with_control_caveat"
    return "eligible_not_supported"


def range_weighted_scope(row: pd.Series) -> str:
    true_count = to_int(row.get("strict_range_weighted_true_q_0_1"))
    controls = to_int(row.get("strict_range_weighted_control_q_0_1"))
    if true_count > 0 and controls == 0:
        return "range_weighted_true_support_clean_controls"
    if true_count > 0:
        return "range_weighted_true_support_control_caveat"
    return "range_weighted_not_supported"


def scope_from_count(value: object, supported_label: str) -> str:
    return supported_label if to_int(value) > 0 else "not_supported"


def read_csv(rel_path: str) -> pd.DataFrame:
    path = PHASE12_DIR / rel_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def one_row(table: pd.DataFrame, dataset_key: str) -> pd.Series:
    if table.empty or "dataset_key" not in table:
        return pd.Series(dtype=object)
    sub = table[table["dataset_key"].astype(str).eq(str(dataset_key))]
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


def sum_values(*values: object) -> int:
    return int(sum(to_int(value) for value in values))


def write_status(layers: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Goal 1 FDR Interpretation Framework",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: `{row.dataset_fdr_interpretation}`; "
            f"broad layers={int(row.n_broad_fdr_supported_layers)}, blockers={int(row.n_blocking_layers)}, "
            f"scoped/supplementary support layers={int(row.n_scoped_or_supplementary_support_layers)}. "
            f"{row.allowed_dataset_claim}"
        )
    if not story.empty:
        row = story.iloc[0]
        lines += [
            "",
            "## Overall",
            "",
            f"- `{row.overall_call}`.",
            f"- {row.manuscript_claim}",
            f"- Next: {row.next_requirement}",
        ]
    lines += [
        "",
        "## Layer Boundary",
        "",
        "- Broad all-covered-LR FDR is the only layer that would support an unbiased whole-atlas discovery claim.",
        "- Target-decoy FDR is a hard-control contamination audit.",
        "- Guarded, matched, eligible and targeted layers are scoped validation layers.",
        "- Supplementary expanded-panel rows are cross-platform support, not strict full-transcriptome FDR.",
    ]
    (OUT_DIR / "phase12_goal1_fdr_interpretation_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_goal1_fdr_interpretation_framework",
        "outputs": [
            "phase12_goal1_fdr_interpretation_layers.csv",
            "phase12_goal1_fdr_interpretation_dataset_summary.csv",
            "phase12_goal1_fdr_interpretation_story.csv",
            "phase12_goal1_fdr_interpretation_status.md",
        ],
        "claim_boundary": "Interpretation framework only; collates existing audits and does not create a new FDR procedure.",
    }
    (OUT_DIR / "phase12_goal1_fdr_interpretation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
