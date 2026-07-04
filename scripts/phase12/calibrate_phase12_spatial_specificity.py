#!/usr/bin/env python
"""Spatial-specificity calibration for Phase 12 expanded atlas ranking.

This is a post-hoc calibration variant for Goal 1. It asks whether LR pairs with
expression concentrated inside their expected spatial/type/niche window can move
farther away from the spatial-distance null. It does not use true_lr labels in
the score and does not replace the main SpatialSelf baseline.
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
    pct,
    safe_div,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "atlas_model_expanded_spatial_specificity"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    all_rankings = []
    for dataset_key, ranking in base_ranking.groupby("dataset_key", sort=False):
        replicate_dir = EXPANDED_DIR / "replicates" / str(dataset_key)
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        features = summarize_spatial_specificity(candidates, lr_pairs)
        calibrated = apply_spatial_specificity_calibration(str(dataset_key), ranking.copy(), features)
        all_rankings.append(calibrated)

    table = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    topk = summarize_topk(table)
    dataset = summarize_dataset(table, topk)
    story = summarize_story(dataset)

    table.to_csv(OUT_DIR / "phase12_spatial_specificity_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_spatial_specificity_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_specificity_dataset_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_specificity_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))
    print()
    print(table.groupby("dataset_key").head(15).to_string(index=False))


def summarize_spatial_specificity(candidates: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    lr_pairs = lr_pairs[["lr_id", "spatial_range", "source_count"]].copy()
    table = candidates.merge(lr_pairs, on="lr_id", how="left")
    table["spatial_range"] = pd.to_numeric(table["spatial_range"], errors="coerce").fillna(
        table["mechanism"].map(range_from_mechanism)
    )
    return summarize_candidate_table(table)


def summarize_candidate_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    ligand = pd.to_numeric(out["ligand_expr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    receptor = pd.to_numeric(out["receptor_expr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    distance = pd.to_numeric(out["distance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    spatial_range = pd.to_numeric(out["spatial_range"], errors="coerce").fillna(420.0).to_numpy(dtype=float)
    pair_expr = np.log1p(ligand) * np.log1p(receptor)
    distance_scale = np.maximum(spatial_range, 1.0)
    distance_weight = np.exp(-distance / distance_scale)
    coexpr_distance = pair_expr * distance_weight
    range_rule = distance <= spatial_range
    type_rule = out["type_rule_satisfied"].astype(bool).to_numpy()
    niche_rule = (
        out["niche_rule_satisfied"].astype(bool).to_numpy()
        if "niche_rule_satisfied" in out.columns
        else np.ones(len(out), dtype=bool)
    )
    in_window = range_rule & type_rule & niche_rule
    off_window = ~in_window
    out["_pair_expr"] = pair_expr
    out["_coexpr_distance"] = coexpr_distance
    out["_in_window"] = in_window.astype(float)
    out["_off_window"] = off_window.astype(float)
    out["_in_window_expr"] = pair_expr * in_window
    out["_off_window_expr"] = pair_expr * off_window
    out["_in_window_coexpr"] = coexpr_distance * in_window
    out["_off_window_coexpr"] = coexpr_distance * off_window
    out["_distance_weight"] = distance_weight

    grouped = out.groupby("lr_id", as_index=False).agg(
        n_candidates=("candidate_id", "size"),
        n_nonzero_expr=("_pair_expr", lambda x: int((x > 0).sum())),
        paired_expr_fraction=("_pair_expr", lambda x: float((x > 0).mean())),
        ligand_positive_fraction=("ligand_expr", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
        receptor_positive_fraction=("receptor_expr", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
        mean_pair_expr=("_pair_expr", "mean"),
        q95_pair_expr=("_pair_expr", lambda x: float(np.quantile(x, 0.95))),
        mean_coexpression_distance=("_coexpr_distance", "mean"),
        q95_coexpression_distance=("_coexpr_distance", lambda x: float(np.quantile(x, 0.95))),
        rule_satisfied_fraction=("type_rule_satisfied", "mean"),
        range_satisfied_fraction=("_in_window", "mean"),
        niche_satisfied_fraction=("niche_rule_satisfied", "mean") if "niche_rule_satisfied" in out.columns else ("_in_window", "mean"),
        in_window_expr_sum=("_in_window_expr", "sum"),
        off_window_expr_sum=("_off_window_expr", "sum"),
        in_window_coexpr_sum=("_in_window_coexpr", "sum"),
        off_window_coexpr_sum=("_off_window_coexpr", "sum"),
        in_window_count=("_in_window", "sum"),
        off_window_count=("_off_window", "sum"),
        mean_distance_weight=("_distance_weight", "mean"),
    )
    return add_spatial_specificity_features(grouped)


def add_spatial_specificity_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    in_count = pd.to_numeric(out["in_window_count"], errors="coerce").fillna(0.0)
    off_count = pd.to_numeric(out["off_window_count"], errors="coerce").fillna(0.0)
    in_expr = pd.to_numeric(out["in_window_expr_sum"], errors="coerce").fillna(0.0) / np.maximum(in_count, 1.0)
    off_expr = pd.to_numeric(out["off_window_expr_sum"], errors="coerce").fillna(0.0) / np.maximum(off_count, 1.0)
    in_coexpr = pd.to_numeric(out["in_window_coexpr_sum"], errors="coerce").fillna(0.0) / np.maximum(in_count, 1.0)
    off_coexpr = pd.to_numeric(out["off_window_coexpr_sum"], errors="coerce").fillna(0.0) / np.maximum(off_count, 1.0)
    out["in_window_expr_mean"] = in_expr
    out["off_window_expr_mean"] = off_expr
    out["in_window_coexpr_mean"] = in_coexpr
    out["off_window_coexpr_mean"] = off_coexpr
    out["spatial_expr_contrast"] = (in_expr + 1e-6) / (in_expr + off_expr + 2e-6)
    out["spatial_coexpr_contrast"] = (in_coexpr + 1e-6) / (in_coexpr + off_coexpr + 2e-6)
    out["spatial_lift"] = np.log1p(in_expr) - np.log1p(off_expr)
    out["spatial_specificity_score"] = (
        0.35 * out["spatial_expr_contrast"].clip(0, 1)
        + 0.35 * out["spatial_coexpr_contrast"].clip(0, 1)
        + 0.20 * pct(out["spatial_lift"])
        + 0.10 * pd.to_numeric(out["range_satisfied_fraction"], errors="coerce").fillna(0.0).clip(0, 1)
    ).clip(0, 1)
    return out


def apply_spatial_specificity_calibration(
    dataset_key: str,
    ranking: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    merged = ranking.merge(features, on="lr_id", how="left", suffixes=("", "_spatial"))
    calibrated = add_spatial_specificity_calibration(merged)
    calibrated["dataset_key"] = dataset_key
    calibrated["method"] = "spatialself_wavelet_edge_prior_spatial_specificity"
    calibrated["mean_score"] = calibrated["calibrated_score"]
    calibrated["q95_score"] = calibrated["calibrated_q95_score"]
    calibrated["rank"] = calibrated["calibrated_score"].rank(ascending=False, method="min").astype(int)
    calibrated["strict_rank"] = calibrated["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    calibrated["rank_q95"] = calibrated["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
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
        "in_window_expr_mean",
        "off_window_expr_mean",
        "in_window_coexpr_mean",
        "off_window_coexpr_mean",
        "spatial_expr_contrast",
        "spatial_coexpr_contrast",
        "spatial_lift",
        "spatial_specificity_score",
        "spatial_specificity_gate",
        "spatial_specificity_raw",
        "spatial_specificity_weight",
        "background_candidate_weight",
        "support_abundance_weight",
        "source_anchor_rescue",
        "expr_gate",
        "source_support",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "spatial_specificity_note",
    ]
    existing = [col for col in keep if col in calibrated.columns]
    return calibrated[existing].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def add_spatial_specificity_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["base_calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
    out["base_strict_discovery_score"] = pd.to_numeric(out["strict_discovery_score"], errors="coerce").fillna(0.0)
    n_support = (np.log1p(pd.to_numeric(out.get("n_nonzero_expr", 0), errors="coerce").fillna(0.0)) / np.log1p(40)).clip(0, 1)
    side_support = np.sqrt(
        pd.to_numeric(out.get("ligand_positive_fraction", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        * pd.to_numeric(out.get("receptor_positive_fraction", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    )
    expr_gate = (0.72 * n_support + 0.28 * side_support).clip(0, 1)
    out["expr_gate"] = expr_gate
    if "source_count" in out.columns:
        source = (
            np.log1p(pd.to_numeric(out["source_count"], errors="coerce").fillna(1).clip(lower=0.0)) / np.log1p(12)
        ).clip(0, 1)
    else:
        source = pd.to_numeric(out.get("source_support", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        if float(source.max()) > 1.0:
            source = (np.log1p(source.clip(lower=0.0)) / np.log1p(12)).clip(0, 1)
    out["source_support"] = source
    out["category_weight"] = np.select(
        [
            out["biological_category"].astype(str).eq("other"),
            out["biological_category"].astype(str).eq("high_expression_negative_risk"),
        ],
        [0.72, 0.20],
        default=1.0,
    )
    out["hard_negative_weight"] = np.where(out["false_positive_high_expression"].astype(bool), 0.20, 1.0)
    out["zero_expression_weight"] = np.where(
        pd.to_numeric(out.get("n_nonzero_expr", 0), errors="coerce").fillna(0).le(0),
        0.03,
        1.0,
    )
    out["low_support_weight"] = np.where(
        (pd.to_numeric(out.get("n_nonzero_expr", 0), errors="coerce").fillna(0).lt(3))
        | pd.to_numeric(out.get("ligand_positive_fraction", 0), errors="coerce").fillna(0).le(0)
        | pd.to_numeric(out.get("receptor_positive_fraction", 0), errors="coerce").fillna(0).le(0),
        0.35,
        1.0,
    )
    specificity = pd.to_numeric(out.get("spatial_specificity_score", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    n_expr = pd.to_numeric(out.get("n_nonzero_expr", 0), errors="coerce").fillna(0.0)
    paired_fraction = pd.to_numeric(out.get("paired_expr_fraction", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    spatial_lift_pct = pct(pd.to_numeric(out.get("spatial_lift", 0.0), errors="coerce").fillna(0.0))
    coexpr_pct = pct(pd.to_numeric(out.get("mean_coexpression_distance", 0.0), errors="coerce").fillna(0.0))
    q95_pct = pct(pd.to_numeric(out.get("q95_coexpression_distance", 0.0), errors="coerce").fillna(0.0))
    rule = (
        0.45 * pd.to_numeric(out.get("rule_satisfied_fraction", 0.0), errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(out.get("range_satisfied_fraction", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("niche_satisfied_fraction", 0.0), errors="coerce").fillna(0.0)
    ).clip(0, 1)

    selected_reason = out["selected_reason"].astype(str)
    category = out["biological_category"].astype(str)
    broad_other = selected_reason.eq("category_balanced") & category.eq("other")
    category_balanced = selected_reason.eq("category_balanced")
    out["background_candidate_weight"] = np.select(
        [
            out["false_positive_high_expression"].astype(bool),
            broad_other,
            category_balanced & source.lt(0.35),
            category_balanced & source.lt(0.60),
            category_balanced,
        ],
        [0.03, 0.035, 0.10, 0.24, 0.55],
        default=1.0,
    )
    out["support_abundance_weight"] = (
        0.25
        + 0.40 * (np.log1p(n_expr.clip(lower=0.0)) / np.log1p(120)).clip(0, 1)
        + 0.35 * paired_fraction.clip(0, 1)
    ).clip(0.08, 1.0)
    out["source_anchor_rescue"] = np.where(
        source.ge(0.75) & n_expr.ge(3) & ~out["false_positive_high_expression"].astype(bool),
        1.12,
        1.0,
    )
    out["spatial_specificity_gate"] = (0.22 + 0.78 * specificity).pow(1.25)
    out["spatial_specificity_weight"] = (
        (0.10 + 0.90 * expr_gate).pow(1.10)
        * (0.12 + 0.88 * source).pow(1.35)
        * out["spatial_specificity_gate"]
        * out["support_abundance_weight"]
        * out["background_candidate_weight"]
        * out["source_anchor_rescue"]
        * pd.to_numeric(out.get("category_weight", 1.0), errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(out.get("zero_expression_weight", 1.0), errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(out.get("low_support_weight", 1.0), errors="coerce").fillna(1.0).clip(0, 1)
    ).clip(0, 1.2)

    out["spatial_specificity_raw"] = (
        0.42 * pct(out["base_strict_discovery_score"])
        + 0.22 * pct(out["base_calibrated_score"])
        + 0.12 * coexpr_pct
        + 0.06 * q95_pct
        + 0.10 * specificity
        + 0.04 * spatial_lift_pct
        + 0.04 * rule
    )
    strict_raw = (
        0.48 * pct(out["base_strict_discovery_score"])
        + 0.17 * pct(out["base_calibrated_score"])
        + 0.10 * coexpr_pct
        + 0.05 * q95_pct
        + 0.12 * specificity
        + 0.04 * spatial_lift_pct
        + 0.04 * rule
    )
    out["calibrated_score"] = out["spatial_specificity_raw"] * out["spatial_specificity_weight"]
    out["strict_discovery_score"] = strict_raw * out["spatial_specificity_weight"]
    out["calibrated_q95_score"] = (
        0.42 * q95_pct + 0.24 * coexpr_pct + 0.22 * specificity + 0.12 * rule
    ) * out["spatial_specificity_weight"]
    out["spatial_specificity_note"] = np.select(
        [
            out["false_positive_high_expression"].astype(bool),
            pd.to_numeric(out.get("n_nonzero_expr", 0), errors="coerce").fillna(0).le(0),
            specificity.lt(0.35),
            specificity.ge(0.70),
        ],
        [
            "hard_negative_penalized",
            "zero_expression_penalized",
            "low_spatial_specificity_penalized",
            "spatial_window_concentrated",
        ],
        default="spatial_specificity_moderate",
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
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
                    "median_spatial_specificity": float(top["spatial_specificity_score"].median()),
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
                "top10_median_spatial_specificity": float(t10.get("median_spatial_specificity", np.nan)),
                "top50_true_lr": int(t50.get("n_true_lr", 0)),
                "top50_negative_or_risk_fraction": float(t50.get("negative_or_risk_fraction", np.nan)),
                "best_true_lr": best_true_lr(sub),
                "spatial_specificity_call": classify_dataset(t10, t50),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n = int(len(dataset))
    clean_top10 = int(pd.to_numeric(dataset["top10_negative_or_risk"], errors="coerce").fillna(999).eq(0).sum())
    mean_top10_true = float(pd.to_numeric(dataset["top10_true_lr"], errors="coerce").mean()) if n else np.nan
    if clean_top10 == n and mean_top10_true >= 4:
        call = "spatial_specificity_ranking_clean"
    elif mean_top10_true >= 3:
        call = "spatial_specificity_ranking_support_with_risk"
    else:
        call = "spatial_specificity_ranking_mixed"
    return pd.DataFrame(
        [
            {
                "topic": "spatial_specificity_calibration",
                "overall_call": call,
                "n_datasets": n,
                "mean_top10_true_lr": mean_top10_true,
                "n_clean_top10_risk": clean_top10,
                "claim_boundary": "Ranking calibration only; exact-score permutation audit is required for FDR claims.",
            }
        ]
    )


def classify_dataset(top10: dict[str, object], top50: dict[str, object]) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_risk = int(top10.get("n_any_negative_or_risk", 999))
    top50_true = int(top50.get("n_true_lr", 0))
    if top10_true >= 5 and top10_risk == 0:
        return "strong_spatial_specific_ranking_clean"
    if top10_true >= 3 and top10_risk <= 1 and top50_true >= top10_true:
        return "spatial_specific_ranking_support"
    return "spatial_specific_ranking_mixed"


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return str(true["lr_pair"].iloc[0]) if not true.empty else "NA"


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    if table.empty:
        return {}
    row = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    return row.iloc[0].to_dict() if len(row) else {}


def range_from_mechanism(mechanism: str) -> float:
    mechanism = str(mechanism)
    if mechanism == "long_range_chemokine_niche":
        return 700.0
    if mechanism == "juxtacrine_contact":
        return 220.0
    if mechanism == "short_medium_paracrine":
        return 420.0
    return 420.0


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Spatial-Specificity Calibration",
        "",
        f"- Overall: `{story.iloc[0].overall_call if not story.empty else 'NA'}`.",
        "- Scope: ranking-only calibration; no FDR claim without exact-score permutation audit.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.spatial_specificity_call}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 risk={int(row.top10_negative_or_risk)}, "
            f"median spatial specificity={fmt(row.top10_median_spatial_specificity)}, "
            f"best true={row.best_true_lr}."
        )
    (OUT_DIR / "phase12_spatial_specificity_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_specificity_calibration",
        "inputs": [
            rel(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv"),
            rel(EXPANDED_DIR / "replicates/*/candidate_interactions.csv"),
            rel(EXPANDED_DIR / "replicates/*/lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_specificity_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_topk_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_story.csv"),
            rel(OUT_DIR / "phase12_spatial_specificity_status.md"),
        ],
        "claim_boundary": "Post-hoc ranking calibration only; requires exact-score permutation audit for FDR claims.",
    }
    (OUT_DIR / "phase12_spatial_specificity_manifest.json").write_text(
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
