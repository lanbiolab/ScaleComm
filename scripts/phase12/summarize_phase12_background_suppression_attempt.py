#!/usr/bin/env python
"""Audit the Phase 12 background-suppression ranking attempt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
BASE_DIR = PHASE12_DIR / "atlas_model_expanded"
BACKGROUND_DIR = PHASE12_DIR / "atlas_model_expanded_background_suppressed"
RANGE_DIR = PHASE12_DIR / "atlas_model_expanded_range_calibrated"
OUT_DIR = PHASE12_DIR / "background_suppression_attempt"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_DIR / "phase12_expanded_atlas_calibrated_topk_summary.csv")
    background = pd.read_csv(BACKGROUND_DIR / "phase12_background_suppressed_topk_summary.csv")
    range_null = pd.read_csv(RANGE_DIR / "phase12_range_null_calibrated_topk_summary.csv")

    comparison = compare_topk(base, background, range_null)
    dataset = summarize_dataset(comparison)
    story = summarize_story(dataset)

    comparison.to_csv(OUT_DIR / "phase12_background_suppression_topk_comparison.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_background_suppression_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_background_suppression_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def compare_topk(base: pd.DataFrame, background: pd.DataFrame, range_null: pd.DataFrame) -> pd.DataFrame:
    base_norm = normalize_topk(base, "base")
    bg_norm = normalize_topk(background, "background")
    range_norm = normalize_topk(range_null, "range_null")
    out = base_norm.merge(bg_norm, on=["dataset_key", "top_k"], how="outer").merge(
        range_norm, on=["dataset_key", "top_k"], how="outer"
    )
    for variant in ["background", "range_null"]:
        out[f"{variant}_delta_true_lr"] = out[f"{variant}_true_lr"].fillna(0) - out["base_true_lr"].fillna(0)
        out[f"{variant}_delta_risk"] = out[f"{variant}_negative_or_risk"].fillna(0) - out[
            "base_negative_or_risk"
        ].fillna(0)
        out[f"{variant}_delta_broad_other"] = out[f"{variant}_broad_other_lr"].fillna(0) - out[
            "base_broad_other_lr"
        ].fillna(0)
        out[f"{variant}_delta_control_fdr"] = out[f"{variant}_control_fdr"].fillna(np.nan) - out[
            "base_control_fdr"
        ].fillna(np.nan)
    return out.sort_values(["dataset_key", "top_k"]).reset_index(drop=True)


def normalize_topk(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = table.copy()
    if "n_broad_other_lr" not in out.columns:
        broad = []
        for value in out.get("top_lr_pairs", pd.Series([""] * len(out))).astype(str):
            broad.append(np.nan)
        out["n_broad_other_lr"] = broad
    keep = [
        "dataset_key",
        "top_k",
        "n_true_lr",
        "n_high_expression_negative",
        "n_any_negative_or_risk",
        "n_broad_other_lr",
        "control_panel_empirical_fdr",
        "top_lr_pairs",
    ]
    out = out[keep].rename(
        columns={
            "n_true_lr": f"{prefix}_true_lr",
            "n_high_expression_negative": f"{prefix}_high_expression_negative",
            "n_any_negative_or_risk": f"{prefix}_negative_or_risk",
            "n_broad_other_lr": f"{prefix}_broad_other_lr",
            "control_panel_empirical_fdr": f"{prefix}_control_fdr",
            "top_lr_pairs": f"{prefix}_top_lr_pairs",
        }
    )
    return out


def summarize_dataset(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in comparison.groupby("dataset_key", sort=False):
        t10 = one_row(sub, 10)
        t20 = one_row(sub, 20)
        t100 = one_row(sub, 100)
        rows.append(
            {
                "dataset_key": dataset_key,
                "base_top10_true_lr": int(t10.get("base_true_lr", 0)),
                "background_top10_true_lr": int(t10.get("background_true_lr", 0)),
                "range_null_top10_true_lr": int(t10.get("range_null_true_lr", 0)),
                "background_top10_delta_true_lr": float(t10.get("background_delta_true_lr", np.nan)),
                "background_top10_delta_risk": float(t10.get("background_delta_risk", np.nan)),
                "background_top20_delta_broad_other": float(t20.get("background_delta_broad_other", np.nan)),
                "background_top100_delta_control_fdr": float(t100.get("background_delta_control_fdr", np.nan)),
                "range_null_top10_delta_true_lr": float(t10.get("range_null_delta_true_lr", np.nan)),
                "range_null_top10_delta_risk": float(t10.get("range_null_delta_risk", np.nan)),
                "range_null_top100_delta_control_fdr": float(t100.get("range_null_delta_control_fdr", np.nan)),
                "background_adoption_decision": classify_background(t10, t20, t100),
                "best_variant_for_dataset": choose_best_variant(t10, t20, t100),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key").reset_index(drop=True)


def classify_background(top10: dict[str, object], top20: dict[str, object], top100: dict[str, object]) -> str:
    true_gain = float(top10.get("background_delta_true_lr", 0))
    risk_delta = float(top10.get("background_delta_risk", 0))
    broad_delta = float(top20.get("background_delta_broad_other", 0))
    fdr_delta = float(top100.get("background_delta_control_fdr", np.nan))
    if true_gain > 0 and risk_delta <= 0 and broad_delta <= 0 and fdr_delta <= 0:
        return "adopt_candidate"
    if true_gain > 0 and risk_delta <= 0:
        return "top10_promising_but_broad_or_top100_caveat"
    if broad_delta < 0 and risk_delta <= 0:
        return "leakage_reduced_low_recall_or_fdr_caveat"
    return "hold_out_exploratory_variant"


def choose_best_variant(top10: dict[str, object], top20: dict[str, object], top100: dict[str, object]) -> str:
    bg_score = variant_score(top10, top20, top100, "background")
    rn_score = variant_score(top10, top20, top100, "range_null")
    base_score = 0.0
    if bg_score > max(rn_score, base_score):
        return "background_suppressed_topk_candidate"
    if rn_score > max(bg_score, base_score):
        return "range_null_candidate"
    return "keep_current_main_ranking"


def variant_score(top10: dict[str, object], top20: dict[str, object], top100: dict[str, object], prefix: str) -> float:
    return (
        1.0 * float(top10.get(f"{prefix}_delta_true_lr", 0) or 0)
        - 1.5 * max(float(top10.get(f"{prefix}_delta_risk", 0) or 0), 0)
        - 0.4 * max(float(top20.get(f"{prefix}_delta_broad_other", 0) or 0), 0)
        - 4.0 * max(float(top100.get(f"{prefix}_delta_control_fdr", 0) or 0), 0)
    )


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n_datasets = len(dataset)
    n_adopt = int(dataset["background_adoption_decision"].eq("adopt_candidate").sum())
    n_promising = int(dataset["background_adoption_decision"].str.contains("promising", na=False).sum())
    n_hold = int(dataset["background_adoption_decision"].eq("hold_out_exploratory_variant").sum())
    n_background_best = int(dataset["best_variant_for_dataset"].eq("background_suppressed_topk_candidate").sum())
    overall = (
        "background_suppression_not_adopted_global_variant"
        if n_adopt < n_datasets
        else "background_suppression_adoptable_candidate"
    )
    return pd.DataFrame(
        [
            {
                "topic": "background_suppression_attempt",
                "overall_call": overall,
                "n_datasets": n_datasets,
                "n_adopt_candidate": n_adopt,
                "n_top10_promising": n_promising,
                "n_hold_out": n_hold,
                "n_background_best_by_score": n_background_best,
                "claim_boundary": (
                    "Exploratory ranking-variant audit. Background suppression improves lymph-node top10 recovery, "
                    "but is not a global replacement because post-Xenium/CRC retain risk or broad-background leakage."
                ),
            }
        ]
    )


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Background-Suppression Attempt",
        "",
        "## Decision",
        "",
        f"- Overall: `{story.iloc[0].overall_call}`.",
        "- Do not replace the current main calibrated ranking globally.",
        "- Treat background suppression as a promising lymph-node/top10 calibration direction and a failed global replacement.",
        "",
        "## Dataset Comparison",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.background_adoption_decision}`: "
            f"top10 true delta={fmt(row.background_top10_delta_true_lr)}, "
            f"top10 risk delta={fmt(row.background_top10_delta_risk)}, "
            f"top20 broad-other delta={fmt(row.background_top20_delta_broad_other)}, "
            f"top100 control-FDR delta={fmt(row.background_top100_delta_control_fdr)}; "
            f"best={row.best_variant_for_dataset}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: background suppression is a useful calibration direction for lymph-node ranking.",
        "- Unsafe wording: the background-suppressed variant solves whole-atlas FDR or replaces the main model.",
    ]
    (OUT_DIR / "phase12_background_suppression_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_background_suppression_attempt_audit",
        "inputs": [
            "atlas_model_expanded/phase12_expanded_atlas_calibrated_topk_summary.csv",
            "atlas_model_expanded_background_suppressed/phase12_background_suppressed_topk_summary.csv",
            "atlas_model_expanded_range_calibrated/phase12_range_null_calibrated_topk_summary.csv",
        ],
        "outputs": [
            "phase12_background_suppression_topk_comparison.csv",
            "phase12_background_suppression_dataset_comparison.csv",
            "phase12_background_suppression_story.csv",
            "phase12_background_suppression_status.md",
        ],
        "claim_boundary": "Exploratory background suppression audit; not global FDR closure.",
    }
    (OUT_DIR / "phase12_background_suppression_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def one_row(table: pd.DataFrame, top_k: int) -> dict[str, object]:
    sub = table[table["top_k"].eq(top_k)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
