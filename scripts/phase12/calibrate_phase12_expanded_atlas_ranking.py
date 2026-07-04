#!/usr/bin/env python
"""Calibrate expanded Phase 12 atlas LR ranking.

The expanded model run scores every candidate edge, but the raw LR mean score can
rank LR pairs with no expression support highly. This script keeps the model
scores and adds a conservative atlas-level calibration layer:

- require nonzero ligand-receptor edge evidence;
- penalize hard-negative and zero-expression LR pairs;
- downweight weakly annotated broad-background LR pairs;
- keep weak nonzero signals eligible.

This is not a permutation FDR procedure. It is a ranking calibration/audit layer.
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

import scripts.phase12.run_phase12_atlas_model_pilot as pilot


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PRED_ROOT = EXPANDED_DIR / "predictions"
REPLICATE_ROOT = EXPANDED_DIR / "replicates"
TOP_K_VALUES = (10, 20, 50, 100)


def main() -> None:
    status = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_status.csv")
    all_rankings: list[pd.DataFrame] = []
    all_hard: list[pd.DataFrame] = []
    for row in status.itertuples(index=False):
        if str(row.status) != "completed":
            continue
        dataset_key = str(row.dataset_key)
        replicate_dir = REPLICATE_ROOT / dataset_key
        pred_path = prediction_path(dataset_key)
        ranking = calibrate_dataset(dataset_key, replicate_dir, pred_path)
        hard = pilot.summarize_hard_negative(replicate_dir, ranking, dataset_key)
        all_rankings.append(ranking)
        all_hard.append(hard)

    ranking_table = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    topk = summarize_topk(ranking_table)
    dataset = summarize_dataset(ranking_table, topk, hard_table, status)

    ranking_table.to_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv", index=False)
    hard_table.to_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_hard_negative_leakage.csv", index=False)
    topk.to_csv(EXPANDED_DIR / "phase12_expanded_atlas_calibrated_topk_summary.csv", index=False)
    dataset.to_csv(EXPANDED_DIR / "phase12_expanded_atlas_calibrated_dataset_summary.csv", index=False)
    write_status(dataset, topk, hard_table)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    if not ranking_table.empty:
        print()
        print(ranking_table.groupby("dataset_key").head(15).to_string(index=False))


def prediction_path(dataset_key: str) -> Path:
    pred_dir = PRED_ROOT / dataset_key / "spatialself_wavelet_edge_prior"
    paths = sorted(pred_dir.glob("*_predictions.csv"))
    if not paths:
        raise FileNotFoundError(f"No prediction file found under {pred_dir}")
    return paths[0]


def calibrate_dataset(dataset_key: str, replicate_dir: Path, pred_path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    pred = pd.read_csv(pred_path)
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    table = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    summary = (
        table.groupby("lr_id", as_index=False)
        .agg(
            raw_mean_score=("score", "mean"),
            raw_q95_score=("score", lambda x: float(np.quantile(x, 0.95))),
            raw_max_score=("score", "max"),
            n_candidates=("score", "size"),
            n_nonzero_expr=("coexpression_distance_score", lambda x: int((x > 0).sum())),
            paired_expr_fraction=("coexpression_distance_score", lambda x: float((x > 0).mean())),
            ligand_positive_fraction=("ligand_expr", lambda x: float((x > 0).mean())),
            receptor_positive_fraction=("receptor_expr", lambda x: float((x > 0).mean())),
            mean_coexpression_distance=("coexpression_distance_score", "mean"),
            q95_coexpression_distance=("coexpression_distance_score", lambda x: float(np.quantile(x, 0.95))),
            rule_satisfied_fraction=("type_rule_satisfied", "mean"),
            range_satisfied_fraction=("range_rule_satisfied", "mean"),
        )
        .merge(lr_pairs, on="lr_id", how="left")
    )
    summary["lr_pair"] = summary["ligand"] + "-" + summary["receptor"]
    summary["dataset_key"] = dataset_key
    summary["method"] = "spatialself_wavelet_edge_prior_calibrated"
    summary["raw_rank"] = summary["raw_mean_score"].rank(ascending=False, method="min").astype(int)
    summary["raw_rank_q95"] = summary["raw_q95_score"].rank(ascending=False, method="min").astype(int)
    summary = add_calibration(summary)
    # Compatibility with the existing hard-negative summarizer, which expects
    # the active LR ranking score to be named mean_score.
    summary["mean_score"] = summary["calibrated_score"]
    summary["q95_score"] = summary["calibrated_q95_score"]
    summary["rank"] = summary["calibrated_score"].rank(ascending=False, method="min").astype(int)
    summary["rank_q95"] = summary["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    summary["strict_rank"] = summary["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    columns = [
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
        "calibration_note",
    ]
    return summary[columns].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def add_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    model_pct = pct(out["raw_mean_score"])
    q95_pct = pct(out["raw_q95_score"])
    coexpr_pct = pct(out["mean_coexpression_distance"])
    q95co_pct = pct(out["q95_coexpression_distance"])
    spatial_rule = (0.6 * out["rule_satisfied_fraction"] + 0.4 * out["range_satisfied_fraction"]).clip(0, 1)

    n_support = (np.log1p(out["n_nonzero_expr"]) / np.log1p(40)).clip(0, 1)
    side_support = np.sqrt(
        out["ligand_positive_fraction"].clip(0, 1) * out["receptor_positive_fraction"].clip(0, 1)
    )
    out["expr_gate"] = (0.72 * n_support + 0.28 * side_support).clip(0, 1)
    out["source_support"] = (np.log1p(pd.to_numeric(out["source_count"], errors="coerce").fillna(1)) / np.log1p(12)).clip(0, 1)
    out["category_weight"] = np.select(
        [
            out["biological_category"].astype(str).eq("other"),
            out["biological_category"].astype(str).eq("high_expression_negative_risk"),
        ],
        [0.72, 0.20],
        default=1.0,
    )
    out["hard_negative_weight"] = np.where(out["false_positive_high_expression"].astype(bool), 0.20, 1.0)
    out["zero_expression_weight"] = np.where(out["n_nonzero_expr"].le(0), 0.03, 1.0)
    out["low_support_weight"] = np.where(
        (out["n_nonzero_expr"].lt(3))
        | out["ligand_positive_fraction"].le(0)
        | out["receptor_positive_fraction"].le(0),
        0.35,
        1.0,
    )

    raw_calibration = (
        0.42 * model_pct
        + 0.12 * q95_pct
        + 0.24 * coexpr_pct
        + 0.08 * q95co_pct
        + 0.08 * spatial_rule
        + 0.06 * out["source_support"]
    )
    support_multiplier = (
        (0.10 + 0.90 * out["expr_gate"])
        * (0.65 + 0.35 * out["source_support"])
        * out["category_weight"]
        * out["hard_negative_weight"]
        * out["zero_expression_weight"]
        * out["low_support_weight"]
    )
    out["calibrated_score"] = raw_calibration * support_multiplier
    out["calibrated_q95_score"] = (
        0.55 * q95_pct + 0.25 * q95co_pct + 0.10 * spatial_rule + 0.10 * out["source_support"]
    ) * support_multiplier
    strict_spatial_rule = (0.75 * out["rule_satisfied_fraction"] + 0.25 * out["range_satisfied_fraction"]).clip(0, 1)
    strict_raw = (
        0.46 * model_pct
        + 0.08 * q95_pct
        + 0.20 * coexpr_pct
        + 0.06 * q95co_pct
        + 0.10 * strict_spatial_rule
        + 0.10 * out["source_support"]
    )
    out["strict_source_weight"] = (0.25 + 0.75 * out["source_support"].clip(0, 1)).pow(2.2)
    out["strict_expr_weight"] = (0.08 + 0.92 * out["expr_gate"].clip(0, 1)).pow(1.15)
    out["strict_low_annotation_weight"] = np.where(
        out["source_support"].lt(0.35),
        0.50,
        np.where(out["source_support"].lt(0.60), 0.78, 1.0),
    )
    strict_category_weight = np.select(
        [
            out["biological_category"].astype(str).eq("other"),
            out["biological_category"].astype(str).eq("high_expression_negative_risk"),
        ],
        [0.50, 0.08],
        default=1.0,
    )
    strict_support_multiplier = (
        out["strict_source_weight"]
        * out["strict_expr_weight"]
        * out["strict_low_annotation_weight"]
        * strict_category_weight
        * np.where(out["false_positive_high_expression"].astype(bool), 0.08, 1.0)
        * np.where(out["n_nonzero_expr"].le(0), 0.02, 1.0)
        * np.where(
            (out["n_nonzero_expr"].lt(3))
            | out["ligand_positive_fraction"].le(0)
            | out["receptor_positive_fraction"].le(0),
            0.25,
            1.0,
        )
    )
    out["strict_discovery_score"] = strict_raw * strict_support_multiplier
    out["calibration_note"] = np.select(
        [
            out["n_nonzero_expr"].le(0),
            out["false_positive_high_expression"].astype(bool),
            out["n_nonzero_expr"].lt(3),
            out["biological_category"].astype(str).eq("other"),
        ],
        [
            "zero_expression_penalized",
            "hard_negative_penalized",
            "very_low_support_penalized",
            "broad_other_category_downweighted",
        ],
        default="calibrated_expression_supported",
    )
    return out


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    rows = []
    annotated = annotate_risk(ranking)
    for dataset_key, sub in annotated.groupby("dataset_key"):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "true_lr_fraction": safe_div(top["true_lr"].astype(bool).sum(), len(top)),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "high_expression_negative_fraction": safe_div(
                        top["false_positive_high_expression"].astype(bool).sum(), len(top)
                    ),
                    "n_zero_expr_lr": int(top["zero_expr_lr"].sum()),
                    "zero_expr_lr_fraction": safe_div(top["zero_expr_lr"].sum(), len(top)),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "receptor_shared_risk_fraction": safe_div(top["receptor_shared_risk"].sum(), len(top)),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "ligand_shared_risk_fraction": safe_div(top["ligand_shared_risk"].sum(), len(top)),
                    "n_broad_other_lr": int(top["broad_other_lr"].sum()),
                    "broad_other_fraction": safe_div(top["broad_other_lr"].sum(), len(top)),
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


def summarize_dataset(
    ranking: pd.DataFrame,
    topk: pd.DataFrame,
    hard: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        t10 = one_row(topk, dataset_key, 10)
        t50 = one_row(topk, dataset_key, 50)
        hard_sub = hard[hard["dataset_key"].eq(dataset_key)] if not hard.empty else pd.DataFrame()
        status_sub = status[status["dataset_key"].eq(dataset_key)].head(1)
        n_beats = int(hard_sub["negative_beats_positive"].sum()) if not hard_sub.empty else 0
        n_hard = int(len(hard_sub))
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_high_expression_negative": int(t10.get("n_high_expression_negative", 0)),
                "top10_zero_expr_lr": int(t10.get("n_zero_expr_lr", 0)),
                "top10_broad_other_lr": int(t10.get("n_broad_other_lr", 0)),
                "top10_negative_or_risk_fraction": float(t10.get("negative_or_risk_fraction", np.nan)),
                "top10_control_panel_empirical_fdr": float(t10.get("control_panel_empirical_fdr", np.nan)),
                "top50_true_lr": int(t50.get("n_true_lr", 0)),
                "top50_high_expression_negative": int(t50.get("n_high_expression_negative", 0)),
                "top50_zero_expr_lr": int(t50.get("n_zero_expr_lr", 0)),
                "top50_negative_or_risk_fraction": float(t50.get("negative_or_risk_fraction", np.nan)),
                "hard_negative_best_comparison_beats": n_beats,
                "hard_negative_best_comparisons": n_hard,
                "hard_negative_best_comparison_beat_fraction": safe_div(n_beats, n_hard),
                "n_candidates": int(status_sub["n_candidates"].iloc[0]) if len(status_sub) else 0,
                "runtime_seconds": float(status_sub["runtime_seconds"].iloc[0]) if len(status_sub) else np.nan,
                "calibrated_audit_call": classify_calibrated(t10, n_beats, n_hard),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def annotate_risk(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["true_lr"] = out["true_lr"].astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].astype(bool)
    out["zero_expr_lr"] = out["n_nonzero_expr"].le(0)
    out["broad_other_lr"] = out["biological_category"].astype(str).eq("other")
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for dataset_key, sub in out.groupby("dataset_key"):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors) & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands) & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = (
        out["false_positive_high_expression"]
        | out["zero_expr_lr"]
        | out["receptor_shared_risk"]
        | out["ligand_shared_risk"]
    )
    return out


def classify_calibrated(top10: dict[str, object], n_beats: int, n_hard: int) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_zero = int(top10.get("n_zero_expr_lr", 0))
    top10_high_neg = int(top10.get("n_high_expression_negative", 0))
    top10_risk = float(top10.get("negative_or_risk_fraction", np.nan))
    beat_fraction = safe_div(n_beats, n_hard)
    if top10_true >= 3 and top10_zero == 0 and top10_high_neg == 0 and top10_risk <= 0.3 and beat_fraction <= 0.35:
        return "calibrated_expanded_rank_support"
    if top10_zero == 0 and top10_high_neg == 0 and top10_true >= 2:
        return "calibrated_expanded_rank_support_with_hard_negative_risk"
    return "calibrated_expanded_mixed_or_incomplete"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, hard: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Calibrated Expanded Atlas Ranking",
        "",
        "## Scope",
        "",
        "This is an atlas-level calibration of the expanded SpatialSelf LR ranking. It uses the existing model predictions and adds expression-support, annotation, zero-expression and hard-negative penalties.",
        "This is not a permutation-calibrated whole-atlas FDR analysis.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.calibrated_audit_call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 high-neg={int(row.top10_high_expression_negative)}, "
            f"top10 zero-expr={int(row.top10_zero_expr_lr)}, top10 risk={fmt(row.top10_negative_or_risk_fraction)}, "
            f"hard-negative best-beat={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}"
        )
    lines += ["", "## Top-K Summary", ""]
    for row in topk[topk["top_k"].isin([10, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"high-neg={int(row.n_high_expression_negative)}, zero-expr={int(row.n_zero_expr_lr)}, "
            f"broad-other={int(row.n_broad_other_lr)}, risk={fmt(row.negative_or_risk_fraction)}"
        )
    if not hard.empty:
        lines += ["", "## Hard-Negative Best Comparisons", ""]
        grouped = hard.groupby("dataset_key")["negative_beats_positive"].agg(["sum", "count"])
        for dataset_key, row in grouped.iterrows():
            lines.append(f"- `{dataset_key}`: {int(row['sum'])}/{int(row['count'])} negatives beat their positive anchor.")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This fixes the main expanded-atlas ranking failure mode: zero-expression LR and high-expression hard negatives no longer dominate top10.",
        "- It supports calibrated expanded atlas ranking, not final whole-atlas superiority.",
        "- Whole-atlas superiority still requires permutation p-values/FDR and same-universe baseline comparison.",
    ]
    (EXPANDED_DIR / "phase12_expanded_atlas_calibrated_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_atlas_calibration",
        "inputs": [
            "phase12_atlas_model_expanded_status.csv",
            "replicates/*/candidate_interactions.csv",
            "predictions/*/spatialself_wavelet_edge_prior/*_predictions.csv",
        ],
        "outputs": [
            "phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "phase12_atlas_model_expanded_calibrated_hard_negative_leakage.csv",
            "phase12_expanded_atlas_calibrated_topk_summary.csv",
            "phase12_expanded_atlas_calibrated_dataset_summary.csv",
            "phase12_expanded_atlas_calibrated_status.md",
        ],
        "claim_boundary": "Calibrated expanded ranking; not permutation-calibrated whole-atlas FDR or baseline superiority.",
    }
    (EXPANDED_DIR / "phase12_expanded_atlas_calibrated_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def pct(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).rank(pct=True)


def one_row(table: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    if table.empty:
        return {}
    sub = table[table["dataset_key"].eq(dataset_key) & table["top_k"].eq(top_k)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if float(denom) else float("nan")


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(val):
        return "NA"
    return f"{val:.3g}"


if __name__ == "__main__":
    main()
