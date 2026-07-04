#!/usr/bin/env python
"""Map the spatial-distance null bottleneck in Phase 12 Goal 1.

This audit turns the broad statement "spatial-distance permutation blocks
whole-atlas FDR" into LR-, category- and dataset-level calibration targets. It
does not change the model ranking and does not claim that spatial-null FDR is
solved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
PERM_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
OUT_DIR = PHASE12_DIR / "spatial_null_bottleneck_map"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = build_lr_table()
    dataset = summarize_dataset(lr)
    category = summarize_category(lr)
    story = summarize_story(dataset, category)

    lr.to_csv(OUT_DIR / "phase12_spatial_null_bottleneck_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_null_bottleneck_dataset.csv", index=False)
    category.to_csv(OUT_DIR / "phase12_spatial_null_bottleneck_category.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_null_bottleneck_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_lr_table() -> pd.DataFrame:
    perm = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    null_specific = pd.read_csv(PHASE12_DIR / "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_lr.csv")
    matched = pd.read_csv(PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv")
    weighted = pd.read_csv(PHASE12_DIR / "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_lr.csv")

    keys = ["dataset_key", "lr_id", "lr_pair"]
    keep_perm = [
        "dataset_key",
        "rank",
        "strict_rank",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "calibrated_score",
        "strict_discovery_score",
        "conservative_global_bh_fdr",
        "strict_conservative_global_bh_fdr",
        "receiver_permutation_global_bh_fdr",
        "spatial_distance_permutation_global_bh_fdr",
        "lr_identity_permutation_global_bh_fdr",
        "strict_receiver_permutation_global_bh_fdr",
        "strict_spatial_distance_permutation_global_bh_fdr",
        "strict_lr_identity_permutation_global_bh_fdr",
    ]
    out = perm[keep_perm].copy()
    out = out.merge(
        null_specific[
            keys
            + [
                "two_null_pairing_identity_q",
                "strict_two_null_pairing_identity_q",
                "bottleneck_null",
                "strict_bottleneck_null",
                "null_specific_call",
                "strict_null_specific_call",
            ]
        ],
        on=keys,
        how="left",
    )
    out = out.merge(
        matched[
            keys
            + [
                "matched_null_scope",
                "matched_spatial_bh_fdr",
                "matched_all_null_bh_fdr",
                "strict_matched_spatial_bh_fdr",
                "strict_matched_all_null_bh_fdr",
                "matched_spatial_call",
            ]
        ],
        on=keys,
        how="left",
    )
    out = out.merge(
        weighted[
            keys
            + [
                "n_nonzero_expr",
                "paired_expr_fraction",
                "expr_gate",
                "source_support",
                "category_weight",
                "receptor_shared_decoy",
                "ligand_shared_decoy",
                "primary_control_decoy",
                "decoy_class",
                "weighted_matched_spatial_bh_fdr",
                "weighted_matched_all_null_bh_fdr",
                "weighted_strict_matched_all_null_bh_fdr",
                "range_weighted_fdr_call",
                "range_weighted_rank",
            ]
        ],
        on=keys,
        how="left",
    )

    out["pairing_identity_supported"] = q_le(out["two_null_pairing_identity_q"], 0.1)
    out["strict_pairing_identity_supported"] = q_le(out["strict_two_null_pairing_identity_q"], 0.1)
    out["pooled_spatial_supported"] = q_le(out["spatial_distance_permutation_global_bh_fdr"], 0.1)
    out["strict_pooled_spatial_supported"] = q_le(out["strict_spatial_distance_permutation_global_bh_fdr"], 0.1)
    out["matched_spatial_supported"] = q_le(out["matched_spatial_bh_fdr"], 0.1)
    out["strict_matched_spatial_supported"] = q_le(out["strict_matched_spatial_bh_fdr"], 0.1)
    out["range_weighted_spatial_supported"] = q_le(out["weighted_matched_spatial_bh_fdr"], 0.1)
    out["range_weighted_all_null_supported"] = q_le(out["weighted_matched_all_null_bh_fdr"], 0.1)
    out["range_weighted_strict_all_null_supported"] = q_le(out["weighted_strict_matched_all_null_bh_fdr"], 0.1)
    out["spatial_specific_blocker"] = (
        out["pairing_identity_supported"]
        & ~out["pooled_spatial_supported"]
        & out["bottleneck_null"].astype(str).eq("spatial_distance_permutation")
    )
    out["strict_spatial_specific_blocker"] = (
        out["strict_pairing_identity_supported"]
        & ~out["strict_pooled_spatial_supported"]
        & out["strict_bottleneck_null"].astype(str).eq("spatial_distance_permutation")
    )
    out["spatial_q_minus_pairing_q"] = (
        pd.to_numeric(out["spatial_distance_permutation_global_bh_fdr"], errors="coerce")
        - pd.to_numeric(out["two_null_pairing_identity_q"], errors="coerce")
    )
    out["spatial_null_status"] = np.select(
        [
            out["pooled_spatial_supported"],
            out["spatial_specific_blocker"] & out["matched_spatial_supported"],
            out["spatial_specific_blocker"] & out["range_weighted_spatial_supported"],
            out["spatial_specific_blocker"],
            out["true_lr"].astype(bool) & ~out["pairing_identity_supported"],
            out["primary_control_decoy"].fillna(False).astype(bool),
        ],
        [
            "pooled_spatial_supported",
            "blocked_in_pooled_but_matched_spatial_supported",
            "blocked_in_pooled_but_range_weighted_supported",
            "spatial_distance_bottleneck",
            "not_supported_before_spatial_null",
            "control_or_decoy",
        ],
        default="background_or_unresolved",
    )
    out["calibration_target"] = np.select(
        [
            out["spatial_null_status"].eq("blocked_in_pooled_but_matched_spatial_supported"),
            out["spatial_null_status"].eq("blocked_in_pooled_but_range_weighted_supported"),
            out["spatial_null_status"].eq("spatial_distance_bottleneck"),
            out["primary_control_decoy"].fillna(False).astype(bool),
            out["selected_reason"].astype(str).eq("category_balanced") & out["biological_category"].astype(str).eq("other"),
            out["selected_reason"].astype(str).eq("category_balanced"),
        ],
        [
            "preserve_signal_with_range_matched_spatial_null",
            "preserve_signal_with_weighted_range_spatial_null",
            "train_against_spatial_distance_null",
            "suppress_primary_control_decoy",
            "suppress_broad_other_category_balanced_lr",
            "tighten_category_balanced_lr",
        ],
        default="monitor",
    )
    columns = [
        "dataset_key",
        "rank",
        "strict_rank",
        "range_weighted_rank",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "primary_control_decoy",
        "decoy_class",
        "calibrated_score",
        "strict_discovery_score",
        "conservative_global_bh_fdr",
        "spatial_distance_permutation_global_bh_fdr",
        "receiver_permutation_global_bh_fdr",
        "lr_identity_permutation_global_bh_fdr",
        "two_null_pairing_identity_q",
        "matched_spatial_bh_fdr",
        "matched_all_null_bh_fdr",
        "weighted_matched_spatial_bh_fdr",
        "weighted_matched_all_null_bh_fdr",
        "strict_spatial_distance_permutation_global_bh_fdr",
        "strict_two_null_pairing_identity_q",
        "strict_matched_spatial_bh_fdr",
        "weighted_strict_matched_all_null_bh_fdr",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "expr_gate",
        "source_support",
        "bottleneck_null",
        "strict_bottleneck_null",
        "pairing_identity_supported",
        "strict_pairing_identity_supported",
        "pooled_spatial_supported",
        "strict_pooled_spatial_supported",
        "matched_spatial_supported",
        "strict_matched_spatial_supported",
        "range_weighted_spatial_supported",
        "range_weighted_all_null_supported",
        "range_weighted_strict_all_null_supported",
        "spatial_specific_blocker",
        "strict_spatial_specific_blocker",
        "spatial_q_minus_pairing_q",
        "spatial_null_status",
        "calibration_target",
    ]
    return out[columns].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def summarize_dataset(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        true = sub[sub["true_lr"].astype(bool)]
        top20 = sub.sort_values("rank").head(20)
        blocked_true = true[true["spatial_specific_blocker"]]
        matched_true = true[true["matched_spatial_supported"]]
        weighted_true = true[true["range_weighted_all_null_supported"] | true["range_weighted_strict_all_null_supported"]]
        primary_controls = sub[sub["primary_control_decoy"].fillna(False).astype(bool)]
        broad_top20 = top20[
            top20["selected_reason"].astype(str).eq("category_balanced")
            & top20["biological_category"].astype(str).eq("other")
        ]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(len(true)),
                "top10_true_lr": int(top20.head(10)["true_lr"].astype(bool).sum()),
                "top20_category_balanced_other": int(len(broad_top20)),
                "true_pairing_identity_supported": int(true["pairing_identity_supported"].sum()),
                "true_pooled_spatial_supported": int(true["pooled_spatial_supported"].sum()),
                "true_spatial_specific_blocker": int(len(blocked_true)),
                "true_matched_spatial_supported": int(len(matched_true)),
                "true_range_weighted_all_null_supported": int(len(weighted_true)),
                "primary_control_pooled_spatial_supported": int(primary_controls["pooled_spatial_supported"].sum()),
                "best_true_spatial_q": min_float(true["spatial_distance_permutation_global_bh_fdr"]),
                "best_true_pairing_identity_q": min_float(true["two_null_pairing_identity_q"]),
                "best_true_matched_spatial_q": min_float(true["matched_spatial_bh_fdr"]),
                "best_true_weighted_all_null_q": min_float(true["weighted_matched_all_null_bh_fdr"]),
                "top_spatial_blocked_true_lr": join_lr(blocked_true.sort_values(["rank", "spatial_distance_permutation_global_bh_fdr"]), 8),
                "top_broad_other_top20_lr": join_lr(broad_top20.sort_values("rank"), 8),
                "dominant_calibration_target": dominant_target(sub),
                "dataset_spatial_null_call": classify_dataset(true, top20, primary_controls),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key").reset_index(drop=True)


def summarize_category(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, category), sub in lr.groupby(["dataset_key", "biological_category"], sort=False):
        top20 = sub[sub["rank"].le(20)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "biological_category": category,
                "n_lr": int(len(sub)),
                "n_top20": int(len(top20)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "n_primary_control": int(sub["primary_control_decoy"].fillna(False).astype(bool).sum()),
                "n_spatial_blocked_true": int((sub["true_lr"].astype(bool) & sub["spatial_specific_blocker"]).sum()),
                "median_spatial_q": median_float(sub["spatial_distance_permutation_global_bh_fdr"]),
                "median_pairing_identity_q": median_float(sub["two_null_pairing_identity_q"]),
                "top_lr_pairs": join_lr(sub.sort_values("rank"), 10),
                "category_call": classify_category(sub, top20),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "n_top20", "n_lr"], ascending=[True, False, False])


def summarize_story(dataset: pd.DataFrame, category: pd.DataFrame) -> pd.DataFrame:
    n_datasets = len(dataset)
    n_spatial_blocked = int(pd.to_numeric(dataset["true_spatial_specific_blocker"], errors="coerce").fillna(0).sum())
    n_pooled_spatial_true = int(pd.to_numeric(dataset["true_pooled_spatial_supported"], errors="coerce").fillna(0).sum())
    n_matched_true = int(pd.to_numeric(dataset["true_matched_spatial_supported"], errors="coerce").fillna(0).sum())
    n_broad_top20 = int(pd.to_numeric(dataset["top20_category_balanced_other"], errors="coerce").fillna(0).sum())
    top_leakage_cats = category[
        category["category_call"].astype(str).isin(["top20_broad_background_leakage", "top20_category_balanced_leakage"])
    ]
    return pd.DataFrame(
        [
            {
                "topic": "spatial_null_bottleneck_map",
                "overall_call": "spatial_distance_null_is_primary_goal1_calibration_target",
                "n_datasets": n_datasets,
                "n_true_pooled_spatial_supported": n_pooled_spatial_true,
                "n_true_spatial_specific_blocker": n_spatial_blocked,
                "n_true_matched_spatial_supported": n_matched_true,
                "n_top20_broad_other_lr": n_broad_top20,
                "top_leakage_categories": ";".join(
                    f"{r.dataset_key}:{r.biological_category}" for r in top_leakage_cats.head(8).itertuples(index=False)
                ),
                "claim_boundary": (
                    "Diagnostic only. This maps why broad whole-atlas FDR is blocked and identifies calibration targets; "
                    "it does not solve pooled spatial-distance FDR."
                ),
            }
        ]
    )


def classify_dataset(true: pd.DataFrame, top20: pd.DataFrame, primary_controls: pd.DataFrame) -> str:
    true_pooled = int(true["pooled_spatial_supported"].sum())
    true_blocked = int(true["spatial_specific_blocker"].sum())
    true_matched = int(true["matched_spatial_supported"].sum())
    top20_broad = int(
        (
            top20["selected_reason"].astype(str).eq("category_balanced")
            & top20["biological_category"].astype(str).eq("other")
        ).sum()
    )
    control_pooled = int(primary_controls["pooled_spatial_supported"].sum()) if len(primary_controls) else 0
    if true_pooled > 0 and control_pooled == 0:
        return "pooled_spatial_supported"
    if true_matched > 0 and true_blocked > 0 and top20_broad <= 2:
        return "range_matched_signal_spatial_pooled_bottleneck"
    if true_matched > 0 and true_blocked > 0:
        return "range_matched_signal_with_background_leakage"
    if top20_broad > 5:
        return "background_leakage_dominates_top_ranks"
    return "spatial_null_bottleneck_unresolved"


def classify_category(sub: pd.DataFrame, top20: pd.DataFrame) -> str:
    if int((sub["true_lr"].astype(bool) & sub["spatial_specific_blocker"]).sum()) > 0:
        return "true_signal_spatially_blocked"
    if len(top20) and sub["biological_category"].astype(str).iloc[0] == "other":
        return "top20_broad_background_leakage"
    if len(top20) and top20["selected_reason"].astype(str).eq("category_balanced").any():
        return "top20_category_balanced_leakage"
    if int(sub["primary_control_decoy"].fillna(False).astype(bool).sum()) > 0:
        return "control_category_monitor"
    return "background_monitor"


def dominant_target(sub: pd.DataFrame) -> str:
    targets = sub["calibration_target"].astype(str)
    priority = [
        "preserve_signal_with_range_matched_spatial_null",
        "preserve_signal_with_weighted_range_spatial_null",
        "train_against_spatial_distance_null",
        "suppress_broad_other_category_balanced_lr",
        "suppress_primary_control_decoy",
        "tighten_category_balanced_lr",
    ]
    counts = targets.value_counts()
    for target in priority:
        if counts.get(target, 0) > 0:
            return target
    return str(counts.index[0]) if len(counts) else "none"


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Spatial-Null Bottleneck Map",
        "",
        "## Overall",
        "",
        f"- Overall call: `{story.iloc[0].overall_call}`.",
        f"- True LR with pooled spatial q<=0.1: {int(story.iloc[0].n_true_pooled_spatial_supported)}.",
        f"- True LR blocked specifically by spatial-distance null: {int(story.iloc[0].n_true_spatial_specific_blocker)}.",
        f"- True LR supported by matched spatial q<=0.1: {int(story.iloc[0].n_true_matched_spatial_supported)}.",
        "",
        "## Dataset Calls",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_spatial_null_call}`: "
            f"true pairing/identity supported={int(row.true_pairing_identity_supported)}, "
            f"pooled spatial supported={int(row.true_pooled_spatial_supported)}, "
            f"spatial-blocked true={int(row.true_spatial_specific_blocker)}, "
            f"matched spatial true={int(row.true_matched_spatial_supported)}, "
            f"top20 broad-other={int(row.top20_category_balanced_other)}; "
            f"next={row.dominant_calibration_target}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- This is a calibration-target map, not a positive FDR result.",
        "- Safe wording: spatial-distance null is the remaining broad whole-atlas FDR bottleneck, while matched/range diagnostics identify signals worth preserving.",
        "- Unsafe wording: whole-atlas spatial FDR is solved.",
    ]
    (OUT_DIR / "phase12_spatial_null_bottleneck_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_null_bottleneck_map",
        "inputs": [
            "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "null_specific_atlas_fdr/phase12_null_specific_atlas_fdr_lr.csv",
            "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv",
            "range_weighted_atlas_fdr/phase12_range_weighted_atlas_fdr_lr.csv",
        ],
        "outputs": [
            "phase12_spatial_null_bottleneck_lr.csv",
            "phase12_spatial_null_bottleneck_dataset.csv",
            "phase12_spatial_null_bottleneck_category.csv",
            "phase12_spatial_null_bottleneck_story.csv",
            "phase12_spatial_null_bottleneck_status.md",
        ],
        "claim_boundary": "Diagnostic map of the remaining spatial-distance null bottleneck; not FDR closure.",
    }
    (OUT_DIR / "phase12_spatial_null_bottleneck_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def q_le(series: pd.Series, threshold: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").le(threshold).fillna(False)


def min_float(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.min()) if len(vals) else float("nan")


def median_float(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else float("nan")


def join_lr(table: pd.DataFrame, n: int) -> str:
    if table.empty:
        return ""
    return ";".join(table["lr_pair"].astype(str).head(n))


if __name__ == "__main__":
    main()
