#!/usr/bin/env python
"""Summarize Goal 5 interpretability by LR mechanism and functional category."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import bh_adjust  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal4-dir",
        default="outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset",
    )
    parser.add_argument(
        "--goal5-dir",
        default="outputs/phase2/goal5_interpretability/tonsil_fresh_subset",
    )
    parser.add_argument(
        "--embedding-dir",
        default="outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset",
    )
    args = parser.parse_args()

    goal4_dir = PROJECT_ROOT / args.goal4_dir
    goal5_dir = PROJECT_ROOT / args.goal5_dir
    embedding_dir = PROJECT_ROOT / args.embedding_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_pairs = pd.read_csv(goal4_dir / "constructed_replicate/lr_pairs.csv")
    lr_summary = pd.read_csv(goal4_dir / "tonsil_fresh_lr_summary.csv")
    downstream = pd.read_csv(goal5_dir / "tonsil_downstream_target_correlation.csv")
    embedding = pd.read_csv(embedding_dir / "lr_embedding_coordinates.csv")
    table = _merged_lr_table(lr_pairs, lr_summary, downstream, embedding)
    mechanism_summary = _group_summary(table, group_column="mechanism")
    category_summary = _group_summary(table, group_column="lr_category")
    control_summary = _group_summary(table, group_column="control_class")
    pairwise = _pairwise_category_tests(table)
    overview = _overview_summary(table, mechanism_summary, category_summary, pairwise)

    table.to_csv(output_dir / "mechanism_category_lr_table.csv", index=False)
    mechanism_summary.to_csv(output_dir / "mechanism_interpretability_summary.csv", index=False)
    category_summary.to_csv(output_dir / "category_interpretability_summary.csv", index=False)
    control_summary.to_csv(output_dir / "control_class_interpretability_summary.csv", index=False)
    pairwise.to_csv(output_dir / "mechanism_category_pairwise_tests.csv", index=False)
    pd.DataFrame([overview]).to_csv(output_dir / "goal5_mechanism_category_summary.csv", index=False)
    (output_dir / "goal5_mechanism_category_manifest.json").write_text(
        json.dumps(
            {
                "goal_id": "5",
                "analysis_label": "mechanism_category_tonsil_fresh_subset",
                "evidence_status": "initial_mechanism_category_evidence",
                "goal4_dir": args.goal4_dir,
                "goal5_dir": args.goal5_dir,
                "embedding_dir": args.embedding_dir,
                "output_dir": args.output_dir,
                "tables": {
                    "lr_table": str((output_dir / "mechanism_category_lr_table.csv").relative_to(PROJECT_ROOT)),
                    "mechanism_summary": str(
                        (output_dir / "mechanism_interpretability_summary.csv").relative_to(PROJECT_ROOT)
                    ),
                    "category_summary": str(
                        (output_dir / "category_interpretability_summary.csv").relative_to(PROJECT_ROOT)
                    ),
                    "control_summary": str(
                        (output_dir / "control_class_interpretability_summary.csv").relative_to(PROJECT_ROOT)
                    ),
                    "pairwise_tests": str(
                        (output_dir / "mechanism_category_pairwise_tests.csv").relative_to(PROJECT_ROOT)
                    ),
                    "overview": str(
                        (output_dir / "goal5_mechanism_category_summary.csv").relative_to(PROJECT_ROOT)
                    ),
                },
                "limitations": [
                    "Single fresh tonsil subset; mechanism/category groups are small for positive controls.",
                    "Mechanism labels combine curated controls and heuristic LR metadata.",
                    "Embedding and downstream summaries are association evidence, not proof of biological mechanism.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Goal 5 mechanism summary: {output_dir / 'goal5_mechanism_category_summary.csv'}")
    print(f"Goal 5 mechanism table:   {output_dir / 'mechanism_interpretability_summary.csv'}")


def _merged_lr_table(
    lr_pairs: pd.DataFrame,
    lr_summary: pd.DataFrame,
    downstream: pd.DataFrame,
    embedding: pd.DataFrame,
) -> pd.DataFrame:
    table = lr_pairs.copy()
    table["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in table.itertuples(index=False)]
    keep_lr = [
        "lr_id",
        "score_mean",
        "score_max",
        "top_edge_count",
        "rank_by_score_mean",
        "receiver_spatial_concentration",
        "cell_type_specificity",
    ]
    table = table.merge(lr_summary[[col for col in keep_lr if col in lr_summary.columns]], on="lr_id", how="left")
    keep_down = [
        "lr_id",
        "target_signatures",
        "spearman_r",
        "bh_fdr",
        "residual_spearman_r",
        "residual_bh_fdr",
        "within_expected_receiver_spearman_r",
        "within_expected_receiver_n_cells",
        "rank_by_residual_spearman",
        "status",
    ]
    table = table.merge(downstream[[col for col in keep_down if col in downstream.columns]], on="lr_id", how="left")
    keep_embedding = [
        "lr_id",
        "lr_category",
        "embedding_cluster",
        "embedding_norm",
        "pca1",
        "pca2",
        "umap1",
        "umap2",
    ]
    table = table.merge(embedding[[col for col in keep_embedding if col in embedding.columns]], on="lr_id", how="left")
    if "lr_category" not in table:
        table["lr_category"] = [
            _lr_category(row.mechanism, row.ligand, row.receptor, row.control_class)
            for row in table.itertuples(index=False)
        ]
    table["is_positive_control"] = table["control_class"] == "tonsil_positive_prior"
    table["is_housekeeping_negative"] = table["control_class"] == "housekeeping_negative_control"
    table["is_background_lr"] = table["control_class"] == "background_lr"
    table["residual_significant_0_10"] = (
        (table["residual_bh_fdr"].astype(float) <= 0.10)
        & (table["residual_spearman_r"].astype(float) > 0)
    )
    return table


def _group_summary(table: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    rows = []
    for group_name, subset in table.groupby(group_column, sort=True, dropna=False):
        tested = subset[subset["status"] == "tested"] if "status" in subset else subset
        rows.append(
            {
                group_column: group_name,
                "n_lr_pairs": int(len(subset)),
                "n_tested": int(len(tested)),
                "n_positive_controls": int(subset["is_positive_control"].sum()),
                "n_housekeeping_negative_controls": int(subset["is_housekeeping_negative"].sum()),
                "n_background_lr": int(subset["is_background_lr"].sum()),
                "mean_score_rank": _mean(tested, "rank_by_score_mean"),
                "median_score_rank": _median(tested, "rank_by_score_mean"),
                "mean_score": _mean(tested, "score_mean"),
                "mean_residual_spearman": _mean(tested, "residual_spearman_r"),
                "median_residual_spearman": _median(tested, "residual_spearman_r"),
                "mean_raw_spearman": _mean(tested, "spearman_r"),
                "n_residual_bh_fdr_0_10": int(tested["residual_significant_0_10"].sum())
                if "residual_significant_0_10" in tested
                else 0,
                "mean_expected_receiver_spearman": _mean(tested, "within_expected_receiver_spearman_r"),
                "mean_receiver_spatial_concentration": _mean(tested, "receiver_spatial_concentration"),
                "mean_cell_type_specificity": _mean(tested, "cell_type_specificity"),
                "mean_embedding_norm": _mean(tested, "embedding_norm"),
                "n_embedding_clusters": int(tested["embedding_cluster"].nunique()) if "embedding_cluster" in tested else 0,
                "top_lr_by_residual": _top_lr(tested),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["mean_residual_spearman", "n_lr_pairs"], ascending=[False, False]).reset_index(drop=True)


def _pairwise_category_tests(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tested = table[table["status"] == "tested"].copy()
    for group_column in ["mechanism", "lr_category", "control_class"]:
        labels = sorted(tested[group_column].dropna().astype(str).unique())
        for label in labels:
            in_group = tested[group_column].astype(str) == label
            is_sig = tested["residual_significant_0_10"].astype(bool)
            a = int((in_group & is_sig).sum())
            b = int((in_group & ~is_sig).sum())
            c = int((~in_group & is_sig).sum())
            d = int((~in_group & ~is_sig).sum())
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "group_column": group_column,
                    "label": label,
                    "in_group_significant": a,
                    "in_group_not_significant": b,
                    "out_group_significant": c,
                    "out_group_not_significant": d,
                    "fisher_p_value": float(p_value),
                    "mean_residual_spearman": float(tested.loc[in_group, "residual_spearman_r"].mean())
                    if in_group.any()
                    else math.nan,
                    "mean_spatial_concentration": float(
                        tested.loc[in_group, "receiver_spatial_concentration"].mean()
                    )
                    if in_group.any()
                    else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_fdr"] = bh_adjust(result["fisher_p_value"].fillna(1.0).to_numpy(dtype=float))
    return result.sort_values(["bh_fdr", "fisher_p_value", "group_column", "label"]).reset_index(drop=True)


def _overview_summary(
    table: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> dict[str, object]:
    positives = table[(table["control_class"] == "tonsil_positive_prior") & (table["status"] == "tested")]
    negatives = table[(table["control_class"] == "housekeeping_negative_control") & (table["status"] == "tested")]
    best_mechanism = mechanism_summary.iloc[0].to_dict() if not mechanism_summary.empty else {}
    best_category = category_summary.iloc[0].to_dict() if not category_summary.empty else {}
    significant_groups = pairwise[pairwise["bh_fdr"] <= 0.10] if "bh_fdr" in pairwise else pd.DataFrame()
    embedding_cluster_purity = _embedding_cluster_purity(table)
    corr_range = _spearman(table, "spatial_range", "residual_spearman_r")
    corr_spatial = _spearman(table, "receiver_spatial_concentration", "residual_spearman_r")
    return {
        "goal_id": "5",
        "analysis_label": "mechanism_category_tonsil_fresh_subset",
        "evidence_status": "initial_mechanism_category_evidence",
        "n_lr_pairs": int(len(table)),
        "n_mechanisms": int(table["mechanism"].nunique()),
        "n_lr_categories": int(table["lr_category"].nunique()),
        "positive_control_mean_residual_spearman": float(positives["residual_spearman_r"].mean())
        if len(positives)
        else math.nan,
        "housekeeping_negative_mean_residual_spearman": float(negatives["residual_spearman_r"].mean())
        if len(negatives)
        else math.nan,
        "positive_minus_housekeeping_residual_spearman": float(
            positives["residual_spearman_r"].mean() - negatives["residual_spearman_r"].mean()
        )
        if len(positives) and len(negatives)
        else math.nan,
        "n_positive_controls_residual_bh_fdr_0_10": int(positives["residual_significant_0_10"].sum())
        if len(positives)
        else 0,
        "n_housekeeping_negatives_residual_bh_fdr_0_10": int(negatives["residual_significant_0_10"].sum())
        if len(negatives)
        else 0,
        "best_mechanism_by_residual": best_mechanism.get("mechanism", ""),
        "best_mechanism_mean_residual_spearman": best_mechanism.get("mean_residual_spearman", math.nan),
        "best_category_by_residual": best_category.get("lr_category", ""),
        "best_category_mean_residual_spearman": best_category.get("mean_residual_spearman", math.nan),
        "n_significant_group_enrichments_fdr_0_10": int(len(significant_groups)),
        "embedding_cluster_category_purity": embedding_cluster_purity,
        "spatial_range_residual_spearman": corr_range,
        "spatial_concentration_residual_spearman": corr_spatial,
        "interpretation": (
            "Mechanism/category-level association evidence for the fresh tonsil subset. "
            "Use cautiously because positive-control categories are small and broad background LR remain competitive."
        ),
    }


def _embedding_cluster_purity(table: pd.DataFrame) -> float:
    if "embedding_cluster" not in table or table["embedding_cluster"].isna().all():
        return math.nan
    weighted = 0.0
    total = 0
    for _, subset in table.groupby("embedding_cluster"):
        if subset.empty:
            continue
        counts = subset["lr_category"].value_counts()
        weighted += float(counts.iloc[0])
        total += int(len(subset))
    return float(weighted / total) if total else math.nan


def _spearman(table: pd.DataFrame, x: str, y: str) -> float:
    if x not in table or y not in table:
        return math.nan
    subset = table[[x, y]].dropna()
    if len(subset) < 4:
        return math.nan
    if subset[x].std() <= 1e-12 or subset[y].std() <= 1e-12:
        return math.nan
    corr, _ = spearmanr(subset[x].to_numpy(dtype=float), subset[y].to_numpy(dtype=float))
    return float(corr)


def _top_lr(table: pd.DataFrame) -> str:
    if table.empty or "residual_spearman_r" not in table:
        return ""
    rows = table.sort_values("residual_spearman_r", ascending=False).head(3)
    return ";".join(
        f"{row.canonical_pair}:{row.residual_spearman_r:.3f}" for row in rows.itertuples(index=False)
    )


def _mean(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return math.nan
    return float(table[column].mean())


def _median(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return math.nan
    return float(table[column].median())


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _lr_category(mechanism: str, ligand: str, receptor: str, control_class: str) -> str:
    ligand = str(ligand)
    receptor = str(receptor)
    pair = f"{ligand}-{receptor}"
    if control_class == "housekeeping_negative_control":
        return "housekeeping_negative"
    if ligand.startswith(("CCL", "CXCL", "XCL", "CX3CL")) or receptor.startswith(("CCR", "CXCR", "XCR", "CX3CR")):
        return "chemokine"
    if any(token in pair for token in ("CD40", "TNF", "TNFRSF", "BTLA")):
        return "tnf_superfamily_checkpoint"
    if any(token in pair for token in ("HLA", "CD74", "LILR")):
        return "antigen_presentation"
    if any(token in pair for token in ("ICAM", "ITG", "CD2", "CD58", "CD28", "CD86", "ALCAM", "SEMA")):
        return "adhesion_contact"
    if any(token in pair for token in ("LGALS", "CD44", "SIRP", "CD47", "PDCD1")):
        return "immune_checkpoint_like"
    if mechanism == "juxtacrine_contact":
        return "contact_other"
    if mechanism == "long_range_chemokine_niche":
        return "secreted_long_range"
    return "secreted_or_background"


if __name__ == "__main__":
    main()
