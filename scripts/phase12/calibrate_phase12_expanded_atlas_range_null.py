#!/usr/bin/env python
"""Create a range-null-calibrated Phase 12 expanded atlas ranking.

This is a method-variant calibration layer. It uses no true_lr labels. The
score combines the existing calibrated/strict scores with expression support,
source support, LR category support and hard-negative penalties, and keeps the
same function available for permutation-null recomputation.
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
    add_calibration,
    annotate_risk,
    fmt,
    safe_div,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "atlas_model_expanded_range_calibrated"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    ranking = []
    for dataset_key, sub in base.groupby("dataset_key", sort=False):
        ranking.append(apply_dataset_calibration(dataset_key, sub.copy()))
    table = pd.concat(ranking, ignore_index=True) if ranking else pd.DataFrame()
    topk = summarize_topk(table)
    dataset = summarize_dataset(table, topk)

    table.to_csv(OUT_DIR / "phase12_range_null_calibrated_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_range_null_calibrated_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_range_null_calibrated_dataset_summary.csv", index=False)
    write_status(dataset, topk)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(table.groupby("dataset_key").head(15).to_string(index=False))


def apply_dataset_calibration(dataset_key: str, table: pd.DataFrame) -> pd.DataFrame:
    out = add_range_null_calibration(table)
    out["base_calibrated_score"] = pd.to_numeric(table["calibrated_score"], errors="coerce").fillna(0.0)
    out["base_strict_discovery_score"] = pd.to_numeric(table["strict_discovery_score"], errors="coerce").fillna(0.0)
    out["calibrated_score"] = out["range_null_calibrated_score"]
    out["strict_discovery_score"] = out["range_null_strict_score"]
    out["calibrated_q95_score"] = out["range_null_q95_score"]
    out["mean_score"] = out["calibrated_score"]
    out["q95_score"] = out["calibrated_q95_score"]
    out["method"] = "spatialself_wavelet_edge_prior_range_null_calibrated"
    out["rank"] = out["calibrated_score"].rank(ascending=False, method="min").astype(int)
    out["strict_rank"] = out["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    out["rank_q95"] = out["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
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
        "strict_source_weight",
        "strict_expr_weight",
        "strict_low_annotation_weight",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "range_null_prior",
        "range_null_prior_pct",
        "range_null_weight",
        "range_null_promiscuity_weight",
        "range_null_spatial_rule",
        "calibration_note",
        "range_null_calibration_note",
    ]
    existing = [col for col in keep if col in out.columns]
    return out[existing].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def add_range_null_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    required_existing = {"calibrated_score", "strict_discovery_score", "expr_gate", "source_support"}
    if required_existing.issubset(set(summary.columns)) and "source_count" not in summary.columns:
        out = summary.copy()
    else:
        out = add_calibration(summary)
    expr = pd.to_numeric(out["expr_gate"], errors="coerce").fillna(0.0).clip(0, 1)
    source = pd.to_numeric(out["source_support"], errors="coerce").fillna(0.0).clip(0, 1)
    annotated = out["biological_category"].astype(str).ne("other").astype(float)
    ligand_count = out.groupby(out["ligand"].astype(str))["lr_id"].transform("count").astype(float)
    receptor_count = out.groupby(out["receptor"].astype(str))["lr_id"].transform("count").astype(float)
    ligand_promiscuity = 1.0 / (1.0 + 0.08 * (ligand_count - 1.0).clip(lower=0.0))
    receptor_promiscuity = 1.0 / (1.0 + 0.06 * (receptor_count - 1.0).clip(lower=0.0))
    out["range_null_promiscuity_weight"] = (0.55 * ligand_promiscuity + 0.45 * receptor_promiscuity).clip(0.35, 1.0)
    penalties = (
        pd.to_numeric(out["hard_negative_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(out["zero_expression_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(out["low_support_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(out["category_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * out["range_null_promiscuity_weight"]
    )
    out["range_null_prior"] = (0.42 * source + 0.38 * expr + 0.20 * annotated) * penalties
    out["range_null_prior_pct"] = pct(out["range_null_prior"])
    out["range_null_weight"] = (0.20 + 1.80 * out["range_null_prior_pct"]).clip(0.20, 2.0)
    out["range_null_spatial_rule"] = (
        0.55 * pd.to_numeric(out["rule_satisfied_fraction"], errors="coerce").fillna(0.0)
        + 0.45 * pd.to_numeric(out["range_satisfied_fraction"], errors="coerce").fillna(0.0)
    ).clip(0, 1)
    base_score = (
        0.56 * pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.28 * pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
        + 0.16 * out["range_null_spatial_rule"]
    )
    strict_score = (
        0.72 * pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.18 * pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
        + 0.10 * out["range_null_spatial_rule"]
    )
    q95_score = (
        0.56 * pd.to_numeric(out["calibrated_q95_score"], errors="coerce").fillna(0.0)
        + 0.24 * pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
        + 0.20 * out["range_null_spatial_rule"]
    )
    out["range_null_calibrated_score"] = base_score * out["range_null_weight"]
    out["range_null_strict_score"] = strict_score * out["range_null_weight"]
    out["range_null_q95_score"] = q95_score * out["range_null_weight"]
    out["range_null_calibration_note"] = np.select(
        [
            out["false_positive_high_expression"].astype(bool),
            out["n_nonzero_expr"].le(0),
            out["biological_category"].astype(str).eq("other"),
        ],
        [
            "range_null_hard_negative_penalized",
            "range_null_zero_expression_penalized",
            "range_null_broad_other_downweighted",
        ],
        default="range_null_expression_source_supported",
    )
    return out


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
                    "n_zero_expr_lr": int(top["zero_expr_lr"].sum()),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
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
        t100 = one_row(topk, dataset_key, 100)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_high_expression_negative": int(t10.get("n_high_expression_negative", 0)),
                "top10_negative_or_risk": int(t10.get("n_any_negative_or_risk", 0)),
                "top10_control_panel_empirical_fdr": float(t10.get("control_panel_empirical_fdr", np.nan)),
                "top100_true_lr": int(t100.get("n_true_lr", 0)),
                "top100_control_panel_empirical_fdr": float(t100.get("control_panel_empirical_fdr", np.nan)),
                "range_null_calibrated_call": classify_dataset(t10, t100),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_dataset(top10: dict[str, object], top100: dict[str, object]) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_high = int(top10.get("n_high_expression_negative", 0))
    top10_risk = int(top10.get("n_any_negative_or_risk", 0))
    top100_fdr = float(top100.get("control_panel_empirical_fdr", np.nan))
    if top10_true >= 3 and top10_high == 0 and top10_risk <= 2 and top100_fdr <= 0.1:
        return "range_null_rank_clean_top100_control_supported"
    if top10_true >= 2 and top10_high == 0:
        return "range_null_rank_support_with_risk_caveat"
    return "range_null_rank_mixed_or_incomplete"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Range-Null-Calibrated Expanded Atlas Ranking",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.range_null_calibrated_call}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 high-neg={int(row.top10_high_expression_negative)}, top10 risk={int(row.top10_negative_or_risk)}, "
            f"top100 control FDR={fmt(row.top100_control_panel_empirical_fdr)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is a range-null-aware score variant using label-independent support weights.",
        "- It is not complete whole-atlas FDR until the matching permutation audit is run.",
    ]
    (OUT_DIR / "phase12_range_null_calibrated_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_range_null_calibrated_expanded_ranking",
        "inputs": ["atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv"],
        "outputs": [
            "phase12_range_null_calibrated_lr_ranking.csv",
            "phase12_range_null_calibrated_topk_summary.csv",
            "phase12_range_null_calibrated_dataset_summary.csv",
            "phase12_range_null_calibrated_status.md",
        ],
        "claim_boundary": "Range-null-aware ranking variant; requires matching permutation audit before FDR claims.",
    }
    (OUT_DIR / "phase12_range_null_calibrated_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pct(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).rank(pct=True)


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    sub = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    return sub.iloc[0].to_dict() if len(sub) else {}


if __name__ == "__main__":
    main()
