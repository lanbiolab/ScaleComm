#!/usr/bin/env python
"""Guarded spatial-specificity calibration for Phase 12 Goal 1.

This variant combines the spatial-window concentration signal with stricter
background and ligand/receptor-shared guards. It is a post-hoc diagnostic, not a
replacement baseline.
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
    annotate_risk,
    fmt,
    safe_div,
)
from scripts.phase12.calibrate_phase12_spatial_specificity import (  # noqa: E402
    add_spatial_specificity_calibration,
    summarize_spatial_specificity,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "atlas_model_expanded_guarded_spatial_specificity"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    all_rankings = []
    for dataset_key, ranking in base_ranking.groupby("dataset_key", sort=False):
        replicate_dir = EXPANDED_DIR / "replicates" / str(dataset_key)
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        features = summarize_spatial_specificity(candidates, lr_pairs)
        table = ranking.merge(features, on="lr_id", how="left", suffixes=("", "_spatial"))
        table = add_guarded_spatial_specificity_calibration(table)
        table["dataset_key"] = str(dataset_key)
        table["method"] = "spatialself_wavelet_edge_prior_guarded_spatial_specificity"
        table["mean_score"] = table["calibrated_score"]
        table["q95_score"] = table["calibrated_q95_score"]
        table["rank"] = table["calibrated_score"].rank(ascending=False, method="min").astype(int)
        table["strict_rank"] = table["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
        table["rank_q95"] = table["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
        all_rankings.append(select_columns(table))

    ranking = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    topk = summarize_topk(ranking)
    dataset = summarize_dataset(ranking, topk)
    story = summarize_story(dataset)
    ranking.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_dataset_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))
    print()
    print(ranking.groupby("dataset_key").head(15).to_string(index=False))


def add_guarded_spatial_specificity_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    out = add_spatial_specificity_calibration(summary)
    selected_reason = out["selected_reason"].astype(str)
    category = out["biological_category"].astype(str)
    source = pd.to_numeric(out["source_support"], errors="coerce").fillna(0.0).clip(0, 1)
    n_expr = pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0.0)
    specificity = pd.to_numeric(out["spatial_specificity_score"], errors="coerce").fillna(0.0).clip(0, 1)
    paired = pd.to_numeric(out["paired_expr_fraction"], errors="coerce").fillna(0.0).clip(0, 1)

    broad_other = selected_reason.eq("category_balanced") & category.eq("other")
    category_balanced = selected_reason.eq("category_balanced")
    high_expression_negative = out["false_positive_high_expression"].astype(bool)
    priority_anchor = selected_reason.eq("phase12_priority_anchor")

    ligand_count = out.groupby(out["ligand"].astype(str))["lr_id"].transform("count").astype(float)
    receptor_count = out.groupby(out["receptor"].astype(str))["lr_id"].transform("count").astype(float)
    ligand_promiscuity = 1.0 / (1.0 + 0.18 * (ligand_count - 1.0).clip(lower=0.0))
    receptor_promiscuity = 1.0 / (1.0 + 0.14 * (receptor_count - 1.0).clip(lower=0.0))
    out["guarded_promiscuity_weight"] = (0.55 * ligand_promiscuity + 0.45 * receptor_promiscuity).clip(0.25, 1.0)
    out["guarded_background_weight"] = np.select(
        [
            high_expression_negative,
            broad_other,
            category_balanced & source.lt(0.35),
            category_balanced & source.lt(0.60),
            category_balanced,
        ],
        [0.015, 0.012, 0.04, 0.10, 0.24],
        default=1.0,
    )
    out["guarded_anchor_rescue"] = np.where(priority_anchor & n_expr.ge(3), 1.18, 1.0)
    out["guarded_specificity_floor"] = np.where(
        priority_anchor,
        (0.18 + 0.82 * specificity).clip(0.10, 1.0),
        (0.05 + 0.95 * specificity).pow(1.35).clip(0.02, 1.0),
    )
    out["guarded_support_weight"] = (
        0.12
        + 0.48 * (np.log1p(n_expr.clip(lower=0.0)) / np.log1p(160)).clip(0, 1)
        + 0.40 * paired
    ).clip(0.06, 1.0)
    out["guarded_zero_expression_weight"] = np.where(
        n_expr.le(0),
        0.015,
        np.where(n_expr.lt(3), 0.25, 1.0),
    )
    shared_decoy_like = category_balanced & (
        out["ligand"].astype(str).isin(set(out.loc[priority_anchor, "ligand"].astype(str)))
        | out["receptor"].astype(str).isin(set(out.loc[priority_anchor, "receptor"].astype(str)))
    )
    out["guarded_shared_anchor_weight"] = np.where(shared_decoy_like, 0.18, 1.0)
    out["guarded_spatial_weight"] = (
        out["guarded_background_weight"]
        * out["guarded_promiscuity_weight"]
        * out["guarded_anchor_rescue"]
        * out["guarded_specificity_floor"]
        * out["guarded_support_weight"]
        * out["guarded_zero_expression_weight"]
        * out["guarded_shared_anchor_weight"]
    ).clip(0, 1.25)

    base_signal = (
        0.62 * pd.to_numeric(out["base_strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.24 * pd.to_numeric(out["base_calibrated_score"], errors="coerce").fillna(0.0)
        + 0.14 * pd.to_numeric(out["spatial_specificity_raw"], errors="coerce").fillna(0.0)
    )
    strict_signal = (
        0.70 * pd.to_numeric(out["base_strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.16 * pd.to_numeric(out["base_calibrated_score"], errors="coerce").fillna(0.0)
        + 0.14 * pd.to_numeric(out["spatial_specificity_raw"], errors="coerce").fillna(0.0)
    )
    q95_signal = (
        0.54 * pd.to_numeric(out["calibrated_q95_score"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(out["base_strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.16 * specificity
    )
    out["calibrated_score"] = base_signal * out["guarded_spatial_weight"]
    out["strict_discovery_score"] = strict_signal * out["guarded_spatial_weight"]
    out["calibrated_q95_score"] = q95_signal * out["guarded_spatial_weight"]
    out["guarded_spatial_specificity_note"] = np.select(
        [
            high_expression_negative,
            shared_decoy_like,
            broad_other,
            category_balanced,
            priority_anchor & specificity.lt(0.15),
            priority_anchor,
        ],
        [
            "high_expression_negative_guarded",
            "anchor_shared_decoy_guarded",
            "broad_other_guarded",
            "category_balanced_guarded",
            "priority_anchor_low_specificity_retained_with_penalty",
            "priority_anchor_guarded_retained",
        ],
        default="background_monitor",
    )
    return out


def select_columns(table: pd.DataFrame) -> pd.DataFrame:
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
        "raw_mean_score",
        "raw_q95_score",
        "raw_max_score",
        "n_candidates",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_pair_expr",
        "q95_pair_expr",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "niche_satisfied_fraction",
        "spatial_expr_contrast",
        "spatial_coexpr_contrast",
        "spatial_lift",
        "spatial_specificity_score",
        "guarded_spatial_weight",
        "guarded_background_weight",
        "guarded_promiscuity_weight",
        "guarded_shared_anchor_weight",
        "guarded_support_weight",
        "guarded_zero_expression_weight",
        "guarded_specificity_floor",
        "guarded_anchor_rescue",
        "expr_gate",
        "source_support",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "guarded_spatial_specificity_note",
    ]
    existing = [col for col in keep if col in table.columns]
    return table[existing].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annotated = annotate_risk(ranking)
    for dataset_key, sub in annotated.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
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
        t50 = one_row(topk, dataset_key, 50)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_negative_or_risk": int(t10.get("n_any_negative_or_risk", 0)),
                "top10_negative_or_risk_fraction": float(t10.get("negative_or_risk_fraction", np.nan)),
                "top10_control_panel_empirical_fdr": float(t10.get("control_panel_empirical_fdr", np.nan)),
                "top50_true_lr": int(t50.get("n_true_lr", 0)),
                "top50_negative_or_risk_fraction": float(t50.get("negative_or_risk_fraction", np.nan)),
                "best_true_lr": best_true_lr(sub),
                "guarded_spatial_specificity_call": classify_dataset(t10, t50),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n = len(dataset)
    mean_true = float(pd.to_numeric(dataset["top10_true_lr"], errors="coerce").mean()) if n else np.nan
    clean = int(pd.to_numeric(dataset["top10_negative_or_risk"], errors="coerce").fillna(999).eq(0).sum())
    if clean == n and mean_true >= 4:
        call = "guarded_spatial_specificity_clean_ranking"
    elif mean_true >= 3:
        call = "guarded_spatial_specificity_mixed_ranking"
    else:
        call = "guarded_spatial_specificity_not_supported"
    return pd.DataFrame(
        [
            {
                "topic": "guarded_spatial_specificity_calibration",
                "overall_call": call,
                "n_datasets": n,
                "mean_top10_true_lr": mean_true,
                "n_clean_top10_risk": clean,
                "claim_boundary": "Ranking-only diagnostic; exact-score permutation audit is required for FDR claims.",
            }
        ]
    )


def classify_dataset(top10: dict[str, object], top50: dict[str, object]) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_risk = int(top10.get("n_any_negative_or_risk", 999))
    top50_true = int(top50.get("n_true_lr", 0))
    if top10_true >= 5 and top10_risk == 0:
        return "strong_guarded_ranking_clean"
    if top10_true >= 3 and top10_risk <= 1 and top50_true >= top10_true:
        return "guarded_ranking_support"
    return "guarded_ranking_mixed"


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return str(true["lr_pair"].iloc[0]) if not true.empty else "NA"


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    row = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    return row.iloc[0].to_dict() if len(row) else {}


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Guarded Spatial-Specificity Calibration",
        "",
        f"- Overall: `{story.iloc[0].overall_call if not story.empty else 'NA'}`.",
        "- Scope: ranking-only calibration; exact-score permutation audit is required before any FDR claim.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.guarded_spatial_specificity_call}`: "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
            f"best true={row.best_true_lr}."
        )
    (OUT_DIR / "phase12_guarded_spatial_specificity_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_guarded_spatial_specificity_calibration",
        "inputs": [
            rel(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv"),
            rel(EXPANDED_DIR / "replicates/*/candidate_interactions.csv"),
            rel(EXPANDED_DIR / "replicates/*/lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_topk_summary.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_story.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_status.md"),
        ],
        "claim_boundary": "Post-hoc ranking diagnostic only; exact-score audit required.",
    }
    (OUT_DIR / "phase12_guarded_spatial_specificity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
