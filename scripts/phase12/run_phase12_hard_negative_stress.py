#!/usr/bin/env python
"""Formal hard-negative stress test for Phase 12 atlas model pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "hard_negative_stress"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = pd.read_csv(PHASE12_DIR / "atlas_readiness/phase12_hard_negative_templates.csv")
    ranking = pd.read_csv(PHASE12_DIR / "atlas_model_pilot/phase12_atlas_model_pilot_lr_ranking.csv")
    downstream = read_optional(PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv")
    gene_table = read_optional(PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_gene_table.csv")

    comparisons = []
    positive_rows = []
    dataset_keys = sorted(set(ranking["dataset_key"].dropna().astype(str)))
    for dataset_key in dataset_keys:
        result = run_dataset(dataset_key, templates, ranking, downstream, gene_table)
        comparisons.append(result["comparisons"])
        positive_rows.append(result["positive_summary"])

    comparison_table = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    positive_summary = pd.concat(positive_rows, ignore_index=True) if positive_rows else pd.DataFrame()
    if not positive_summary.empty:
        positive_summary["empirical_bh_fdr_score"] = bh_adjust_by_group(
            positive_summary,
            group_col="dataset_key",
            p_col="empirical_p_score",
        )
    dataset_summary = summarize_datasets(positive_summary, comparison_table)

    comparison_table.to_csv(OUT_DIR / "phase12_hard_negative_stress_comparisons.csv", index=False)
    positive_summary.to_csv(OUT_DIR / "phase12_hard_negative_stress_positive_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_hard_negative_stress_dataset_summary.csv", index=False)
    write_status(dataset_summary, positive_summary)
    write_manifest()
    print(dataset_summary.to_string(index=False))
    print()
    print(positive_summary.sort_values(["dataset_key", "positive_rank"]).to_string(index=False))


def run_dataset(
    dataset_key: str,
    templates: pd.DataFrame,
    ranking: pd.DataFrame,
    downstream: pd.DataFrame,
    gene_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    rank_sub = ranking[ranking["dataset_key"].eq(dataset_key)].copy()
    rank_lookup = rank_sub.set_index("lr_pair").to_dict("index")
    template_sub = templates[templates["dataset_key"].eq(dataset_key)].copy()
    if template_sub.empty:
        return {"comparisons": pd.DataFrame(), "positive_summary": pd.DataFrame()}

    comparison_rows = []
    for row in template_sub.itertuples(index=False):
        pos = rank_lookup.get(row.positive_lr_pair)
        neg = rank_lookup.get(row.negative_pair)
        comparison_rows.append(
            {
                "dataset_key": dataset_key,
                "positive_lr_pair": row.positive_lr_pair,
                "positive_category": row.positive_category,
                "negative_pair": row.negative_pair,
                "negative_type": row.negative_type,
                "is_receptor_shared": bool(row.is_receptor_shared),
                "is_ligand_shared": bool(row.is_ligand_shared),
                "positive_scored": pos is not None,
                "negative_scored": neg is not None,
                "positive_rank": int(pos["rank"]) if pos else -1,
                "negative_rank": int(neg["rank"]) if neg else -1,
                "positive_score": float(pos["mean_score"]) if pos else np.nan,
                "negative_score": float(neg["mean_score"]) if neg else np.nan,
                "negative_minus_positive_score": float(neg["mean_score"] - pos["mean_score"]) if pos and neg else np.nan,
                "negative_beats_positive": bool(neg["rank"] < pos["rank"]) if pos and neg else False,
                "negative_false_positive_high_expression": bool(neg["false_positive_high_expression"]) if neg else False,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    positive_summary = summarize_positive_controls(dataset_key, comparisons, rank_sub, downstream, gene_table)
    return {"comparisons": comparisons, "positive_summary": positive_summary}


def summarize_positive_controls(
    dataset_key: str,
    comparisons: pd.DataFrame,
    ranking: pd.DataFrame,
    downstream: pd.DataFrame,
    gene_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    positive_pairs = sorted(comparisons["positive_lr_pair"].dropna().astype(str).unique())
    rank_lookup = ranking.set_index("lr_pair").to_dict("index")
    downstream_lookup = (
        downstream[downstream["dataset_key"].eq(dataset_key)].set_index("lr_pair").to_dict("index")
        if not downstream.empty and "dataset_key" in downstream.columns
        else {}
    )
    for positive in positive_pairs:
        pos = rank_lookup.get(positive)
        sub = comparisons[comparisons["positive_lr_pair"].eq(positive)].copy()
        scored = sub[sub["positive_scored"] & sub["negative_scored"]].copy()
        if pos is None:
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "positive_lr_pair": positive,
                    "status": "positive_not_scored",
                    "n_templates": int(len(sub)),
                    "n_scored_negatives": int(len(scored)),
                }
            )
            continue
        if len(scored):
            best_neg = scored.sort_values(["negative_rank", "negative_score"], ascending=[True, False]).iloc[0]
            empirical_p = float((1 + np.sum(scored["negative_score"].to_numpy(dtype=float) >= float(pos["mean_score"]))) / (1 + len(scored)))
            n_receptor_shared_beats = int((scored["is_receptor_shared"] & scored["negative_beats_positive"]).sum())
            n_ligand_shared_beats = int((scored["is_ligand_shared"] & scored["negative_beats_positive"]).sum())
            n_beats = int(scored["negative_beats_positive"].sum())
            best_negative_pair = str(best_neg["negative_pair"])
            best_negative_rank = int(best_neg["negative_rank"])
            best_negative_score = float(best_neg["negative_score"])
            best_negative_type = str(best_neg["negative_type"])
            top_gene_overlap = top_gene_jaccard(gene_table, dataset_key, positive, best_negative_pair)
        else:
            empirical_p = np.nan
            n_receptor_shared_beats = n_ligand_shared_beats = n_beats = 0
            best_negative_pair = ""
            best_negative_rank = -1
            best_negative_score = np.nan
            best_negative_type = ""
            top_gene_overlap = np.nan
        ds_pos = downstream_lookup.get(positive, {})
        ds_neg = downstream_lookup.get(best_negative_pair, {}) if best_negative_pair else {}
        rows.append(
            {
                "dataset_key": dataset_key,
                "positive_lr_pair": positive,
                "status": "completed",
                "positive_rank": int(pos["rank"]),
                "positive_score": float(pos["mean_score"]),
                "positive_biological_category": pos.get("biological_category", ""),
                "n_templates": int(len(sub)),
                "n_scored_negatives": int(len(scored)),
                "template_coverage_fraction": safe_divide(len(scored), len(sub)),
                "n_negatives_beating_positive": n_beats,
                "negative_beat_fraction": safe_divide(n_beats, len(scored)),
                "n_receptor_shared_negatives_beating_positive": n_receptor_shared_beats,
                "n_ligand_shared_negatives_beating_positive": n_ligand_shared_beats,
                "best_negative_pair": best_negative_pair,
                "best_negative_type": best_negative_type,
                "best_negative_rank": best_negative_rank,
                "best_negative_score": best_negative_score,
                "best_negative_minus_positive_score": best_negative_score - float(pos["mean_score"]) if np.isfinite(best_negative_score) else np.nan,
                "empirical_p_score": empirical_p,
                "positive_nichenet_targets_present": to_int(ds_pos.get("n_nichenet_targets_present", 0)),
                "positive_targets_top300": to_int(ds_pos.get("n_targets_top300", 0)),
                "positive_target_top300_enrichment": to_float(ds_pos.get("target_top300_enrichment")),
                "positive_target_top300_bh_fdr": to_float(ds_pos.get("target_top300_enrichment_bh_fdr")),
                "best_negative_nichenet_targets_present": to_int(ds_neg.get("n_nichenet_targets_present", 0)),
                "best_negative_targets_top300": to_int(ds_neg.get("n_targets_top300", 0)),
                "best_negative_target_top300_enrichment": to_float(ds_neg.get("target_top300_enrichment")),
                "best_negative_target_top300_bh_fdr": to_float(ds_neg.get("target_top300_enrichment_bh_fdr")),
                "positive_vs_best_negative_top300_gene_jaccard": top_gene_overlap,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["stress_pass_rank_score"] = (
            table["empirical_p_score"].fillna(1.0).le(0.1)
            & table["n_receptor_shared_negatives_beating_positive"].fillna(0).eq(0)
            & table["n_negatives_beating_positive"].fillna(0).eq(0)
        )
        table["stress_pass_downstream"] = (
            table["positive_target_top300_bh_fdr"].fillna(1.0).le(0.1)
            & (
                table["best_negative_target_top300_bh_fdr"].isna()
                | (table["positive_target_top300_enrichment"].fillna(-np.inf) >= table["best_negative_target_top300_enrichment"].fillna(-np.inf))
            )
        )
    return table


def summarize_datasets(positive_summary: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in positive_summary.groupby("dataset_key"):
        comp = comparisons[comparisons["dataset_key"].eq(dataset_key)]
        scored_comp = comp[comp["positive_scored"] & comp["negative_scored"]]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_positive_controls": int(len(sub)),
                "n_positive_rank_score_pass": int(sub["stress_pass_rank_score"].fillna(False).sum()),
                "n_positive_downstream_pass": int(sub["stress_pass_downstream"].fillna(False).sum()),
                "median_positive_rank": float(sub["positive_rank"].replace(-1, np.nan).median()),
                "mean_negative_beat_fraction": float(sub["negative_beat_fraction"].mean()),
                "n_scored_comparisons": int(len(scored_comp)),
                "n_negative_beats_positive_comparisons": int(scored_comp["negative_beats_positive"].sum()) if len(scored_comp) else 0,
                "negative_comparison_beat_fraction": safe_divide(
                    int(scored_comp["negative_beats_positive"].sum()) if len(scored_comp) else 0,
                    len(scored_comp),
                ),
                "n_receptor_shared_beats": int((scored_comp["is_receptor_shared"] & scored_comp["negative_beats_positive"]).sum()) if len(scored_comp) else 0,
                "n_significant_empirical_fdr_0_1": int(sub["empirical_bh_fdr_score"].fillna(1.0).le(0.1).sum()) if "empirical_bh_fdr_score" in sub else 0,
                "interpretation": interpret_dataset(sub, scored_comp),
            }
        )
    return pd.DataFrame(rows)


def interpret_dataset(sub: pd.DataFrame, comp: pd.DataFrame) -> str:
    if sub["stress_pass_rank_score"].fillna(False).sum() >= max(1, len(sub) // 3):
        return "some anchors pass hard-negative rank-score stress"
    if len(comp) and comp["negative_beats_positive"].mean() > 0.5:
        return "hard negatives frequently outrank positive anchors; calibration needed"
    return "mixed hard-negative stress behavior"


def top_gene_jaccard(gene_table: pd.DataFrame, dataset_key: str, positive: str, negative: str, *, top_k: int = 300) -> float:
    if gene_table.empty:
        return np.nan
    sub = gene_table[gene_table["dataset_key"].eq(dataset_key)]
    if sub.empty:
        return np.nan
    pos_genes = set(
        sub[sub["lr_pair"].eq(positive)]
        .sort_values("whole_transcriptome_rank")
        .head(top_k)["gene"]
        .astype(str)
    )
    neg_genes = set(
        sub[sub["lr_pair"].eq(negative)]
        .sort_values("whole_transcriptome_rank")
        .head(top_k)["gene"]
        .astype(str)
    )
    if not pos_genes or not neg_genes:
        return np.nan
    return float(len(pos_genes & neg_genes) / len(pos_genes | neg_genes))


def bh_adjust_by_group(table: pd.DataFrame, *, group_col: str, p_col: str) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=float)
    for _, idx in table.groupby(group_col).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        out[idx] = bh_adjust(table.iloc[idx][p_col].to_numpy(dtype=float))
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def write_status(dataset_summary: pd.DataFrame, positive_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Hard-Negative Stress Status",
        "",
        "## Scope",
        "",
        "This stress test compares each scored positive anchor against explicit hard-negative templates using LR rank, mean score, empirical score FDR, receptor/ligand sharing, and available downstream target-rank support.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {int(row.n_positive_controls)} positives; rank-score pass {int(row.n_positive_rank_score_pass)}; downstream pass {int(row.n_positive_downstream_pass)}; negative beat fraction {float(row.negative_comparison_beat_fraction):.2f}; {row.interpretation}"
        )
    if not positive_summary.empty:
        lines += ["", "## Anchor-Level Stress", ""]
        for dataset_key, sub in positive_summary.groupby("dataset_key"):
            lines.append(f"- `{dataset_key}`:")
            for row in sub.sort_values("positive_rank").itertuples(index=False):
                lines.append(
                    f"  - {row.positive_lr_pair}: rank {int(row.positive_rank)}, best negative {row.best_negative_pair} rank {int(row.best_negative_rank)}, beat fraction {float(row.negative_beat_fraction) if pd.notna(row.negative_beat_fraction) else float('nan'):.2f}, empirical FDR {float(row.empirical_bh_fdr_score) if pd.notna(row.empirical_bh_fdr_score) else float('nan'):.3g}, downstream FDR {float(row.positive_target_top300_bh_fdr) if pd.notna(row.positive_target_top300_bh_fdr) else float('nan'):.3g}"
                )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a pilot-scale stress test on the Phase 12 atlas model pilot LR subset. It exposes and quantifies hard false-positive modes; it is not yet final global FDR control.",
    ]
    (OUT_DIR / "phase12_hard_negative_stress_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_hard_negative_stress",
        "outputs": [
            "phase12_hard_negative_stress_comparisons.csv",
            "phase12_hard_negative_stress_positive_summary.csv",
            "phase12_hard_negative_stress_dataset_summary.csv",
            "phase12_hard_negative_stress_status.md",
        ],
        "claim_boundary": "Pilot-scale hard-negative stress quantification, not final global FDR.",
    }
    (OUT_DIR / "phase12_hard_negative_stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_divide(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
