#!/usr/bin/env python
"""Run a lightweight Phase 12 whole-LR atlas smoke ranking on full-transcriptome subsets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
READINESS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_readiness"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_smoke"

DATASETS = [
    {
        "dataset_key": "visium_hd_crc_full",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
    },
    {
        "dataset_key": "post_xenium_colon_full",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/post_xenium_human_colon_cancer_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_universe = pd.read_csv(READINESS_DIR / "phase12_lr_universe.csv")
    hard_negatives = pd.read_csv(READINESS_DIR / "phase12_hard_negative_templates.csv")
    all_rankings = []
    all_modules = []
    all_hard = []
    for spec in DATASETS:
        result = run_dataset(spec, lr_universe, hard_negatives)
        all_rankings.append(result["ranking"])
        all_modules.append(result["modules"])
        all_hard.append(result["hard_negative"])

    ranking = pd.concat(all_rankings, ignore_index=True)
    modules = pd.concat(all_modules, ignore_index=True)
    hard = pd.concat(all_hard, ignore_index=True)
    ranking.to_csv(OUT_DIR / "phase12_atlas_smoke_lr_ranking.csv", index=False)
    modules.to_csv(OUT_DIR / "phase12_atlas_smoke_modules.csv", index=False)
    hard.to_csv(OUT_DIR / "phase12_atlas_smoke_hard_negative_leakage.csv", index=False)
    write_status(ranking, modules, hard)
    manifest = {
        "task": "phase12_whole_lr_atlas_smoke",
        "datasets": [x["dataset_key"] for x in DATASETS],
        "scoring": "mean over receiver cells of max neighbor log1p(ligand sender expression) * log1p(receptor receiver expression) / (1 + distance / median_neighbor_distance)",
        "n_lr_ranked": int(len(ranking)),
        "claim_boundary": "Smoke atlas ranking only; this is an expression-distance baseline to validate atlas-scale inputs, not final SpatialSelf model scoring.",
        "outputs": [
            "phase12_atlas_smoke_lr_ranking.csv",
            "phase12_atlas_smoke_modules.csv",
            "phase12_atlas_smoke_hard_negative_leakage.csv",
            "phase12_atlas_smoke_status.md",
        ],
    }
    (OUT_DIR / "phase12_atlas_smoke_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(ranking.groupby("dataset_key").head(10).to_string(index=False))
    print()
    print(hard.groupby("dataset_key").head(10).to_string(index=False))


def run_dataset(spec: dict[str, str], lr_universe: pd.DataFrame, hard_negatives: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = pd.read_csv(PROJECT_ROOT / spec["cells"])
    genes = pd.read_csv(PROJECT_ROOT / spec["genes"])["gene"].astype(str).str.upper().tolist()
    barcodes = pd.read_csv(PROJECT_ROOT / spec["barcodes"])["cell_id"].astype(str).tolist()
    counts = sparse.load_npz(PROJECT_ROOT / spec["counts"]).tocsr()
    if counts.shape[0] != len(barcodes):
        raise ValueError(f"Count matrix/barcode mismatch for {spec['dataset_key']}")

    cells = cells[cells["cell_id"].astype(str).isin(barcodes)].copy()
    cells = select_spots(cells, max_spots=1500)
    row_index = {cell_id: i for i, cell_id in enumerate(barcodes)}
    rows = [row_index[x] for x in cells["cell_id"].astype(str)]
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=min(16, len(cells)))
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]
    median_dist = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    decay = 1.0 / (1.0 + distances / max(median_dist, 1e-8))

    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    covered = lr_universe[lr_universe["ligand"].isin(gene_to_idx) & lr_universe["receptor"].isin(gene_to_idx)].copy()
    covered = prioritize_lr(covered, max_lr=300)
    required_genes = sorted(set(covered["ligand"]) | set(covered["receptor"]))
    expr = load_expression_block(counts, rows, gene_to_idx, required_genes)

    ranking_rows = []
    activity_profiles = []
    activity_names = []
    for lr in covered.itertuples(index=False):
        ligand = str(lr.ligand)
        receptor = str(lr.receptor)
        ligand_values = expr[ligand]
        receptor_values = expr[receptor]
        neigh_ligand = ligand_values[neighbors]
        incoming = (np.log1p(neigh_ligand) * decay).max(axis=1) * np.log1p(receptor_values)
        ranking_rows.append(
            {
                "dataset_key": spec["dataset_key"],
                "lr_pair": lr.lr_pair,
                "ligand": ligand,
                "receptor": receptor,
                "atlas_smoke_score_mean": float(np.mean(incoming)),
                "atlas_smoke_score_q95": float(np.quantile(incoming, 0.95)),
                "n_receivers_nonzero": int(np.sum(incoming > 0)),
                "receiver_nonzero_fraction": float(np.mean(incoming > 0)),
                "biological_category": lr.biological_category,
                "phase12_priority_anchor": bool(lr.phase12_priority_anchor),
                "prior_high_expression_negative": bool(lr.any_prior_high_expression_negative),
                "receptor_shared_risk": receptor in {"CCR7", "CXCR4", "CXCR3", "CD44", "CD274"},
            }
        )
        activity_profiles.append(incoming)
        activity_names.append(lr.lr_pair)
    ranking = pd.DataFrame(ranking_rows)
    ranking["rank"] = ranking["atlas_smoke_score_mean"].rank(ascending=False, method="min").astype(int)
    ranking = ranking.sort_values(["rank", "lr_pair"])
    modules = module_summary(spec["dataset_key"], ranking, activity_names, activity_profiles)
    hard = hard_negative_leakage(spec["dataset_key"], ranking, hard_negatives)
    return {"ranking": ranking, "modules": modules, "hard_negative": hard}


def select_spots(cells: pd.DataFrame, max_spots: int) -> pd.DataFrame:
    if len(cells) <= max_spots:
        return cells.reset_index(drop=True)
    if "niche" in cells.columns:
        return (
            cells.groupby("niche", group_keys=False, sort=False)
            .apply(lambda x: x.sample(min(len(x), max(1, int(max_spots * len(x) / len(cells)))), random_state=1207))
            .head(max_spots)
            .reset_index(drop=True)
        )
    return cells.sample(max_spots, random_state=1207).reset_index(drop=True)


def prioritize_lr(covered: pd.DataFrame, max_lr: int) -> pd.DataFrame:
    priority = covered[covered["phase12_priority_anchor"].astype(bool)]
    categories = []
    for _, group in covered.groupby("biological_category", sort=False):
        categories.append(group.head(max(8, max_lr // 24)))
    combined = pd.concat([priority, *categories, covered.head(max_lr)], ignore_index=True)
    combined = combined.drop_duplicates("lr_pair").head(max_lr)
    return combined.reset_index(drop=True)


def load_expression_block(counts: sparse.csr_matrix, rows: list[int], gene_to_idx: dict[str, int], genes: list[str]) -> dict[str, np.ndarray]:
    cols = [gene_to_idx[g] for g in genes]
    block = counts[rows, :][:, cols].astype(np.float32).toarray()
    return {gene: block[:, idx] for idx, gene in enumerate(genes)}


def module_summary(dataset_key: str, ranking: pd.DataFrame, activity_names: list[str], activity_profiles: list[np.ndarray]) -> pd.DataFrame:
    if len(activity_profiles) < 5:
        return pd.DataFrame()
    profile_rows = []
    profile_names = []
    for name, profile in zip(activity_names, activity_profiles):
        if np.nanstd(profile) > 1e-10 and np.isfinite(profile).all():
            profile_rows.append(profile)
            profile_names.append(name)
    if len(profile_rows) < 5:
        return pd.DataFrame(
            [
                {
                    "dataset_key": dataset_key,
                    "module": "unclustered",
                    "n_lr": int(len(activity_profiles)),
                    "mean_score": float(ranking["atlas_smoke_score_mean"].mean()),
                    "top_lr_pairs": ";".join(ranking.sort_values("atlas_smoke_score_mean", ascending=False)["lr_pair"].head(8)),
                    "top_biological_category": str(ranking["biological_category"].value_counts().idxmax()),
                    "n_priority_anchor": int(ranking["phase12_priority_anchor"].sum()),
                    "n_prior_high_expression_negative": int(ranking["prior_high_expression_negative"].sum()),
                    "n_receptor_shared_risk": int(ranking["receptor_shared_risk"].sum()),
                }
            ]
        )
    matrix = np.vstack(profile_rows)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(denom, 1e-8)
    n_modules = min(6, max(2, len(activity_profiles) // 35))
    clusters = fcluster(linkage(pdist(matrix, metric="cosine"), method="average"), t=n_modules, criterion="maxclust")
    cluster_table = pd.DataFrame({"lr_pair": profile_names, "module": clusters}).merge(ranking, on="lr_pair", how="left")
    rows = []
    for module, sub in cluster_table.groupby("module"):
        rows.append(
            {
                "dataset_key": dataset_key,
                "module": f"M{int(module)}",
                "n_lr": int(len(sub)),
                "mean_score": float(sub["atlas_smoke_score_mean"].mean()),
                "top_lr_pairs": ";".join(sub.sort_values("atlas_smoke_score_mean", ascending=False)["lr_pair"].head(8)),
                "top_biological_category": str(sub["biological_category"].value_counts().idxmax()),
                "n_priority_anchor": int(sub["phase12_priority_anchor"].sum()),
                "n_prior_high_expression_negative": int(sub["prior_high_expression_negative"].sum()),
                "n_receptor_shared_risk": int(sub["receptor_shared_risk"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "mean_score"], ascending=[True, False])


def hard_negative_leakage(dataset_key: str, ranking: pd.DataFrame, hard_negatives: pd.DataFrame) -> pd.DataFrame:
    rank_lookup = ranking.set_index("lr_pair").to_dict("index")
    rows = []
    subset = hard_negatives[hard_negatives["dataset_key"].eq(dataset_key)]
    for row in subset.itertuples(index=False):
        pos = rank_lookup.get(row.positive_lr_pair)
        neg = rank_lookup.get(row.negative_pair)
        if pos is None or neg is None:
            continue
        rows.append(
            {
                "dataset_key": dataset_key,
                "positive_lr_pair": row.positive_lr_pair,
                "negative_pair": row.negative_pair,
                "negative_type": row.negative_type,
                "positive_rank": int(pos["rank"]),
                "negative_rank": int(neg["rank"]),
                "positive_score": float(pos["atlas_smoke_score_mean"]),
                "negative_score": float(neg["atlas_smoke_score_mean"]),
                "negative_minus_positive_score": float(neg["atlas_smoke_score_mean"] - pos["atlas_smoke_score_mean"]),
                "negative_beats_positive": bool(neg["rank"] < pos["rank"]),
                "is_receptor_shared": bool(row.is_receptor_shared),
                "is_ligand_shared": bool(row.is_ligand_shared),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["negative_beats_positive", "negative_minus_positive_score"], ascending=[False, False])


def write_status(ranking: pd.DataFrame, modules: pd.DataFrame, hard: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas Smoke Status",
        "",
        "## Scope",
        "",
        "This is a lightweight expression-distance atlas smoke test. It validates whole-LR atlas input coverage, LR-level ranking, module clustering and hard-negative leakage calculations before running full SpatialSelf/baseline atlas models.",
        "",
        "## Key results",
        "",
    ]
    for dataset, sub in ranking.groupby("dataset_key"):
        hard_sub = hard[hard["dataset_key"].eq(dataset)] if not hard.empty else pd.DataFrame()
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- LR pairs ranked: {len(sub)}.",
                f"- Priority anchors ranked: {int(sub['phase12_priority_anchor'].sum())}.",
                f"- Prior high-expression negatives ranked: {int(sub['prior_high_expression_negative'].sum())}.",
                f"- Hard-negative comparisons scored: {len(hard_sub)}.",
                f"- Hard negatives beating positives: {int(hard_sub['negative_beats_positive'].sum()) if len(hard_sub) else 0}.",
                "",
                "Top 10 LR pairs:",
                "",
            ]
        )
        for row in sub.sort_values("rank").head(10).itertuples(index=False):
            lines.append(f"- #{row.rank} {row.lr_pair}: score={row.atlas_smoke_score_mean:.4g}, category={row.biological_category}")
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            "These scores are not final SpatialSelf atlas predictions. They are an expression-distance smoke baseline to check Phase 12 atlas-scale plumbing and identify hard-negative risk before full model/baseline runs.",
            "",
            "## Outputs",
            "",
            "- `phase12_atlas_smoke_lr_ranking.csv`",
            "- `phase12_atlas_smoke_modules.csv`",
            "- `phase12_atlas_smoke_hard_negative_leakage.csv`",
        ]
    )
    (OUT_DIR / "phase12_atlas_smoke_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
