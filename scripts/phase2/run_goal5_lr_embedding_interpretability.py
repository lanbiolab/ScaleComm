#!/usr/bin/env python
"""Analyze learned LR identity embeddings for Phase 2 Goal 5."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

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
    parser.add_argument("--output-dir", default="outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset")
    parser.add_argument("--cluster-distance-threshold", type=float, default=0.75)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    goal4_dir = PROJECT_ROOT / args.goal4_dir
    goal5_dir = PROJECT_ROOT / args.goal5_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_pairs = pd.read_csv(goal4_dir / "constructed_replicate/lr_pairs.csv")
    lr_summary = pd.read_csv(goal4_dir / "tonsil_fresh_lr_summary.csv")
    downstream = _read_optional(goal5_dir / "tonsil_downstream_target_correlation.csv")
    embeddings = _load_lr_embeddings(goal4_dir / "tonsil_fresh_spatialself_wavelet_model.pt")
    table = _embedding_table(lr_pairs, lr_summary, downstream, embeddings, random_seed=args.random_seed)
    table = _assign_embedding_clusters(table, embeddings, threshold=args.cluster_distance_threshold)
    cluster_summary = _cluster_summary(table)
    enrichment = _cluster_enrichment(table)
    metrics = _embedding_metrics(table, embeddings)
    summary = _goal_summary(table, cluster_summary, enrichment, metrics)

    table.to_csv(output_dir / "lr_embedding_coordinates.csv", index=False)
    cluster_summary.to_csv(output_dir / "lr_embedding_cluster_summary.csv", index=False)
    enrichment.to_csv(output_dir / "lr_embedding_cluster_enrichment.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "lr_embedding_metric_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "goal5_lr_embedding_summary.csv", index=False)
    (output_dir / "goal5_lr_embedding_manifest.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "goal4_dir": args.goal4_dir,
                "goal5_dir": args.goal5_dir,
                "output_dir": args.output_dir,
                "cluster_distance_threshold": args.cluster_distance_threshold,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"LR embedding summary: {output_dir / 'goal5_lr_embedding_summary.csv'}")
    print(f"LR embedding table:   {output_dir / 'lr_embedding_coordinates.csv'}")


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_lr_embeddings(model_path: Path) -> np.ndarray:
    state = torch.load(model_path, map_location="cpu")
    if "lr_encoder.emb.weight" not in state:
        raise ValueError(f"Model state has no learned LR embedding: {model_path}")
    return state["lr_encoder.emb.weight"].detach().cpu().numpy().astype(float)


def _embedding_table(
    lr_pairs: pd.DataFrame,
    lr_summary: pd.DataFrame,
    downstream: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    random_seed: int,
) -> pd.DataFrame:
    if len(lr_pairs) != embeddings.shape[0]:
        raise ValueError(f"LR table rows ({len(lr_pairs)}) do not match embeddings ({embeddings.shape[0]}).")
    pca = PCA(n_components=2, random_state=random_seed)
    pca_xy = pca.fit_transform(embeddings)
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_seed, n_neighbors=min(12, len(lr_pairs) - 1), min_dist=0.1)
        umap_xy = reducer.fit_transform(embeddings)
    except Exception:
        umap_xy = pca_xy.copy()
    table = lr_pairs.copy().reset_index(drop=True)
    table["canonical_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
    table["embedding_index"] = np.arange(len(table), dtype=int)
    table["lr_category"] = [
        _lr_category(row.mechanism, row.ligand, row.receptor, row.control_class)
        for row in table.itertuples(index=False)
    ]
    table["pca1"] = pca_xy[:, 0]
    table["pca2"] = pca_xy[:, 1]
    table["umap1"] = umap_xy[:, 0]
    table["umap2"] = umap_xy[:, 1]
    table["embedding_norm"] = np.linalg.norm(embeddings, axis=1)
    for i in range(embeddings.shape[1]):
        table[f"emb_{i:02d}"] = embeddings[:, i]
    merge_columns = [
        "lr_id",
        "score_mean",
        "score_max",
        "top_edge_count",
        "rank_by_score_mean",
        "receiver_spatial_concentration",
    ]
    if set(merge_columns).issubset(lr_summary.columns):
        table = table.merge(lr_summary[merge_columns], on="lr_id", how="left")
    if not downstream.empty:
        down_cols = [
            "lr_id",
            "spearman_r",
            "residual_spearman_r",
            "within_expected_receiver_spearman_r",
            "rank_by_residual_spearman",
        ]
        available = [col for col in down_cols if col in downstream.columns]
        table = table.merge(downstream[available], on="lr_id", how="left")
    return table


def _assign_embedding_clusters(table: pd.DataFrame, embeddings: np.ndarray, *, threshold: float) -> pd.DataFrame:
    if len(table) <= 1:
        table["embedding_cluster"] = 1
        return table
    cosine = cosine_similarity(embeddings)
    distance = np.clip(1.0 - cosine, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    clusters = fcluster(linkage(squareform(distance, checks=False), method="average"), t=float(threshold), criterion="distance")
    table = table.copy()
    table["embedding_cluster"] = clusters.astype(int)
    return table


def _cluster_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, subset in table.groupby("embedding_cluster", sort=True):
        mechanisms = subset["mechanism"].value_counts()
        categories = subset["lr_category"].value_counts()
        controls = subset["control_class"].value_counts()
        rows.append(
            {
                "embedding_cluster": int(cluster),
                "n_lr_pairs": int(len(subset)),
                "dominant_mechanism": mechanisms.index[0] if len(mechanisms) else "",
                "dominant_mechanism_fraction": float(mechanisms.iloc[0] / len(subset)) if len(mechanisms) else 0.0,
                "dominant_lr_category": categories.index[0] if len(categories) else "",
                "dominant_lr_category_fraction": float(categories.iloc[0] / len(subset)) if len(categories) else 0.0,
                "n_positive_controls": int((subset["control_class"] == "tonsil_positive_prior").sum()),
                "n_housekeeping_negative_controls": int((subset["control_class"] == "housekeeping_negative_control").sum()),
                "n_background_lr": int((subset["control_class"] == "background_lr").sum()),
                "mean_spatial_range": float(subset["spatial_range"].mean()),
                "mean_score_rank": float(subset["rank_by_score_mean"].mean()) if "rank_by_score_mean" in subset else math.nan,
                "mean_residual_spearman": float(subset["residual_spearman_r"].mean()) if "residual_spearman_r" in subset else math.nan,
                "lr_pairs": ";".join(subset["ligand"].astype(str) + "-" + subset["receptor"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_enrichment(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cluster_values = sorted(table["embedding_cluster"].unique())
    labels = {
        "mechanism": sorted(table["mechanism"].dropna().unique()),
        "lr_category": sorted(table["lr_category"].dropna().unique()),
        "control_class": sorted(table["control_class"].dropna().unique()),
    }
    for label_column, label_values in labels.items():
        for cluster in cluster_values:
            in_cluster = table["embedding_cluster"] == cluster
            for label in label_values:
                has_label = table[label_column] == label
                a = int((in_cluster & has_label).sum())
                b = int((in_cluster & ~has_label).sum())
                c = int((~in_cluster & has_label).sum())
                d = int((~in_cluster & ~has_label).sum())
                _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
                rows.append(
                    {
                        "label_column": label_column,
                        "embedding_cluster": int(cluster),
                        "label": label,
                        "in_cluster_with_label": a,
                        "in_cluster_without_label": b,
                        "out_cluster_with_label": c,
                        "out_cluster_without_label": d,
                        "enrichment_p_value": float(p_value),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["enrichment_bh_fdr"] = bh_adjust(result["enrichment_p_value"].to_numpy(dtype=float))
    return result.sort_values(["enrichment_bh_fdr", "enrichment_p_value", "label_column", "embedding_cluster"])


def _embedding_metrics(table: pd.DataFrame, embeddings: np.ndarray) -> dict[str, object]:
    metrics: dict[str, object] = {
        "n_lr_pairs": int(len(table)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_embedding_clusters": int(table["embedding_cluster"].nunique()),
    }
    for label in ["mechanism", "lr_category", "control_class"]:
        codes, uniques = pd.factorize(table[label].astype(str))
        if len(uniques) >= 2 and min(np.bincount(codes)) >= 2:
            metrics[f"silhouette_{label}"] = float(silhouette_score(embeddings, codes, metric="cosine"))
        else:
            metrics[f"silhouette_{label}"] = math.nan
    for column in ["spatial_range", "score_mean", "top_edge_count", "residual_spearman_r"]:
        if column in table.columns and table[column].notna().sum() >= 4:
            corr, p_value = spearmanr(table["embedding_norm"], table[column], nan_policy="omit")
            metrics[f"embedding_norm_spearman__{column}"] = float(corr)
            metrics[f"embedding_norm_spearman_p__{column}"] = float(p_value)
    cosine = cosine_similarity(embeddings)
    metrics["mean_pairwise_cosine"] = float(cosine[np.triu_indices_from(cosine, k=1)].mean())
    return metrics


def _goal_summary(
    table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    enrichment: pd.DataFrame,
    metrics: dict[str, object],
) -> dict[str, object]:
    top_enrich = enrichment.head(5)
    significant = enrichment[enrichment["enrichment_bh_fdr"] <= 0.10] if "enrichment_bh_fdr" in enrichment else pd.DataFrame()
    largest_cluster = cluster_summary.sort_values("n_lr_pairs", ascending=False).iloc[0].to_dict() if not cluster_summary.empty else {}
    return {
        "goal_id": "5",
        "analysis_label": "lr_embedding_tonsil_fresh_subset",
        "evidence_status": "initial_embedding_interpretability_evidence",
        "n_lr_pairs": int(len(table)),
        "embedding_dim": int(metrics.get("embedding_dim", 0)),
        "n_embedding_clusters": int(metrics.get("n_embedding_clusters", 0)),
        "silhouette_mechanism": metrics.get("silhouette_mechanism", math.nan),
        "silhouette_lr_category": metrics.get("silhouette_lr_category", math.nan),
        "silhouette_control_class": metrics.get("silhouette_control_class", math.nan),
        "n_significant_cluster_enrichments_fdr_0_10": int(len(significant)),
        "largest_cluster_size": int(largest_cluster.get("n_lr_pairs", 0)),
        "largest_cluster_dominant_category": largest_cluster.get("dominant_lr_category", ""),
        "top_cluster_enrichments": ";".join(
            f"c{row.embedding_cluster}:{row.label_column}={row.label}:p={row.enrichment_p_value:.3g}"
            for row in top_enrich.itertuples(index=False)
        ),
        "embedding_norm_spatial_range_spearman": metrics.get("embedding_norm_spearman__spatial_range", math.nan),
        "embedding_norm_residual_spearman_spearman": metrics.get("embedding_norm_spearman__residual_spearman_r", math.nan),
        "interpretation": (
            "Initial LR embedding structure analysis. Treat positive enrichment or clustering as latent structure "
            "associated with LR categories, not as proof that embeddings recover true biological mechanism."
        ),
    }


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
