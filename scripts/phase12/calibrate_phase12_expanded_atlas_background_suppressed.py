#!/usr/bin/env python
"""Create a background-suppressed Phase 12 expanded atlas ranking variant.

This exploratory calibration tests the next action suggested by the spatial-null
bottleneck map: preserve strongly supported LR anchors while pushing broad,
category-balanced background LR out of the top ranks. It does not use true_lr
labels in the score and does not replace the main calibrated atlas ranking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase12.calibrate_phase12_expanded_atlas_ranking import (  # noqa: E402
    TOP_K_VALUES,
    add_calibration as add_base_calibration,
    annotate_risk,
    fmt,
    safe_div,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "atlas_model_expanded_background_suppressed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    tables = []
    for dataset_key, sub in base.groupby("dataset_key", sort=False):
        tables.append(apply_background_suppression(dataset_key, sub.copy()))
    ranking = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    topk = summarize_topk(ranking)
    dataset = summarize_dataset(ranking, topk)

    ranking.to_csv(OUT_DIR / "phase12_background_suppressed_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_background_suppressed_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_background_suppressed_dataset_summary.csv", index=False)
    write_status(dataset, topk)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(ranking.groupby("dataset_key").head(15).to_string(index=False))


def apply_background_suppression(dataset_key: str, table: pd.DataFrame) -> pd.DataFrame:
    out = add_background_suppression_calibration(table)
    out["dataset_key"] = dataset_key
    keep = [
        "dataset_key",
        "method",
        "rank",
        "strict_rank",
        "rank_q95",
        "raw_rank",
        "raw_rank_q95",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "calibrated_score",
        "strict_discovery_score",
        "calibrated_q95_score",
        "mean_score",
        "q95_score",
        "base_calibrated_score",
        "base_strict_discovery_score",
        "background_suppression_weight",
        "raw_mean_score",
        "raw_q95_score",
        "raw_max_score",
        "n_candidates",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "expr_gate",
        "source_support",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "calibration_note",
        "background_suppression_note",
    ]
    existing = [col for col in keep if col in out.columns]
    return out[existing].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def add_background_suppression_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    required_existing = {
        "calibrated_score",
        "strict_discovery_score",
        "calibrated_q95_score",
        "expr_gate",
        "source_support",
    }
    out = summary.copy() if required_existing.issubset(set(summary.columns)) else add_base_calibration(summary)
    source = pd.to_numeric(out["source_support"], errors="coerce").fillna(0.0).clip(0, 1)
    expr = pd.to_numeric(out["expr_gate"], errors="coerce").fillna(0.0).clip(0, 1)
    n_expr = pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0.0)
    category = out["biological_category"].astype(str)
    selected_reason = out["selected_reason"].astype(str)

    source_gate = (0.12 + 0.88 * source).pow(1.35)
    expr_gate = (0.25 + 0.75 * expr).clip(0.05, 1.0)
    observed_pair_gate = (np.log1p(n_expr) / np.log1p(60)).clip(0.05, 1.0)
    low_support_penalty = np.where((source < 0.35) | (n_expr < 3), 0.35, 1.0)
    broad_other_penalty = np.where(
        selected_reason.eq("category_balanced") & category.eq("other"),
        0.06,
        1.0,
    )
    category_balanced_penalty = np.where(
        selected_reason.eq("category_balanced") & ~category.eq("other"),
        np.where(source < 0.35, 0.18, 0.45),
        1.0,
    )
    hard_negative_penalty = np.where(out["false_positive_high_expression"].astype(bool), 0.03, 1.0)
    zero_expr_penalty = np.where(n_expr <= 0, 0.02, 1.0)

    ligand_count = out.groupby(out["ligand"].astype(str))["lr_id"].transform("count").astype(float)
    receptor_count = out.groupby(out["receptor"].astype(str))["lr_id"].transform("count").astype(float)
    promiscuity_penalty = (
        0.50 / (1.0 + 0.12 * (ligand_count - 1.0).clip(lower=0.0))
        + 0.50 / (1.0 + 0.10 * (receptor_count - 1.0).clip(lower=0.0))
    ).clip(0.35, 1.0)
    out["background_suppression_weight"] = (
        source_gate
        * expr_gate
        * observed_pair_gate
        * low_support_penalty
        * broad_other_penalty
        * category_balanced_penalty
        * hard_negative_penalty
        * zero_expr_penalty
        * promiscuity_penalty
    ).clip(0.0, 1.0)

    base_signal = (
        0.62 * pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.26 * pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
        + 0.12 * pd.to_numeric(out["calibrated_q95_score"], errors="coerce").fillna(0.0)
    )
    source_rescue = np.where(
        (source >= 0.75) & (expr >= 0.30) & (n_expr >= 3) & ~out["false_positive_high_expression"].astype(bool),
        1.15,
        1.0,
    )
    out["base_calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
    out["base_strict_discovery_score"] = pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
    out["background_suppressed_score"] = base_signal * out["background_suppression_weight"] * source_rescue
    out["background_suppressed_strict_score"] = (
        0.78 * pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.22 * pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
    ) * out["background_suppression_weight"] * source_rescue
    out["calibrated_score"] = out["background_suppressed_score"]
    out["strict_discovery_score"] = out["background_suppressed_strict_score"]
    out["calibrated_q95_score"] = (
        pd.to_numeric(out["calibrated_q95_score"], errors="coerce").fillna(0.0)
        * out["background_suppression_weight"]
        * source_rescue
    )
    out["mean_score"] = out["calibrated_score"]
    out["q95_score"] = out["calibrated_q95_score"]
    out["method"] = "spatialself_wavelet_edge_prior_background_suppressed"
    out["rank"] = out["calibrated_score"].rank(ascending=False, method="min").astype(int)
    out["strict_rank"] = out["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    out["rank_q95"] = out["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    out["background_suppression_note"] = np.select(
        [
            out["false_positive_high_expression"].astype(bool),
            n_expr <= 0,
            selected_reason.eq("category_balanced") & category.eq("other"),
            selected_reason.eq("category_balanced"),
            source >= 0.75,
        ],
        [
            "hard_negative_suppressed",
            "zero_expression_suppressed",
            "broad_other_category_balanced_suppressed",
            "category_balanced_suppressed",
            "source_supported_preserved",
        ],
        default="background_suppressed_monitor",
    )
    return out


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annotated = annotate_risk(ranking)
    for dataset_key, sub in annotated.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            broad_other = (
                top["selected_reason"].astype(str).eq("category_balanced")
                & top["biological_category"].astype(str).eq("other")
            )
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_zero_expr_lr": int(top["zero_expr_lr"].sum()),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "n_broad_other_lr": int(broad_other.sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "control_panel_empirical_fdr": safe_div(
                        top["negative_or_risk"].sum(),
                        max(int(top["negative_or_risk"].sum()) + int(top["true_lr"].astype(bool).sum()), 1),
                    ),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(ranking: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key", sort=False):
        t10 = one_row(topk, dataset_key, 10)
        t20 = one_row(topk, dataset_key, 20)
        t100 = one_row(topk, dataset_key, 100)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_high_expression_negative": int(t10.get("n_high_expression_negative", 0)),
                "top10_negative_or_risk": int(t10.get("n_any_negative_or_risk", 0)),
                "top10_broad_other_lr": int(t10.get("n_broad_other_lr", 0)),
                "top10_control_panel_empirical_fdr": float(t10.get("control_panel_empirical_fdr", np.nan)),
                "top20_true_lr": int(t20.get("n_true_lr", 0)),
                "top20_broad_other_lr": int(t20.get("n_broad_other_lr", 0)),
                "top100_true_lr": int(t100.get("n_true_lr", 0)),
                "top100_control_panel_empirical_fdr": float(t100.get("control_panel_empirical_fdr", np.nan)),
                "background_suppressed_call": classify_dataset(t10, t20, t100),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_dataset(top10: dict[str, object], top20: dict[str, object], top100: dict[str, object]) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_risk = int(top10.get("n_any_negative_or_risk", 0))
    top10_broad = int(top10.get("n_broad_other_lr", 0))
    top20_broad = int(top20.get("n_broad_other_lr", 0))
    top100_fdr = float(top100.get("control_panel_empirical_fdr", np.nan))
    if top10_true >= 3 and top10_risk == 0 and top20_broad <= 2 and top100_fdr <= 0.1:
        return "background_suppressed_clean_candidate"
    if top10_true >= 2 and top10_broad <= 2 and top10_risk <= 1:
        return "background_suppressed_top10_improved_with_caveat"
    if top20_broad <= 2:
        return "background_suppressed_leakage_reduced_low_recall"
    return "background_suppressed_incomplete"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Background-Suppressed Expanded Atlas Ranking",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.background_suppressed_call}`: "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
            f"top10 broad-other={int(row.top10_broad_other_lr)}, top20 broad-other={int(row.top20_broad_other_lr)}, "
            f"top100 control FDR={fmt(row.top100_control_panel_empirical_fdr)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is an exploratory background-suppression score variant.",
        "- It does not use true_lr labels in scoring.",
        "- It is not a whole-atlas FDR result until matched permutation/control audits are run on this exact score.",
    ]
    (OUT_DIR / "phase12_background_suppressed_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_background_suppressed_expanded_ranking",
        "inputs": ["atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv"],
        "outputs": [
            "phase12_background_suppressed_lr_ranking.csv",
            "phase12_background_suppressed_topk_summary.csv",
            "phase12_background_suppressed_dataset_summary.csv",
            "phase12_background_suppressed_status.md",
        ],
        "claim_boundary": "Exploratory background-suppressed ranking variant; requires exact-score FDR audit before discovery claims.",
    }
    (OUT_DIR / "phase12_background_suppressed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    sub = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    return sub.iloc[0].to_dict() if len(sub) else {}


if __name__ == "__main__":
    main()
