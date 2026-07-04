#!/usr/bin/env python
"""Whole-transcriptome target-rank scan for native human lymph node Visium."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
RESULT_DIR = OUT_DIR / "results"
RAW_H5 = PROJECT_ROOT / "data/optional/human_lymph_node_visium_full/raw/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5"

FOCUS_METHODS = ["spatialself", "commot", "cellchat", "cellnest"]
FOCUS_LR_IDS = [
    "ln_full_lr_000_CCL19_CCR7",
    "ln_full_lr_001_CCL21_CCR7",
    "ln_full_lr_002_CXCL13_CXCR5",
    "ln_full_lr_004_CD48_CD244",
    "ln_full_lr_008_EEF1A1_CCR7_negative",
]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv").set_index("lr_id")
    target_map = load_targets(DATASET_DIR / "target_pathway_pairs.csv")
    incoming = pd.read_csv(RESULT_DIR / "lymph_node_full_receiver_level_incoming_scores.csv")
    counts, genes, barcodes = read_10x_matrix(RAW_H5)

    selected = cells["cell_id"].astype(str).to_list()
    barcode_index = {barcode: i for i, barcode in enumerate(barcodes)}
    row_idx = [barcode_index[cell] for cell in selected if cell in barcode_index]
    if len(row_idx) != len(selected):
        missing = sorted(set(selected) - set(barcode_index))
        raise ValueError(f"Selected cells missing from raw matrix: {missing[:5]}")

    expression = counts[row_idx, :].tocsr().astype(np.float32)
    expressed = np.asarray((expression > 0).sum(axis=0)).ravel() >= 15
    expression = expression[:, expressed].tocsr()
    genes = genes[expressed]
    expression.data = np.log1p(expression.data)
    dense = np.asarray(expression.toarray(), dtype=np.float32)
    covariates = build_covariates(cells)
    residual_expr = residualize_matrix(dense, covariates)
    gene_sd = residual_expr.std(axis=0)
    valid = gene_sd > 1e-8
    residual_expr = residual_expr[:, valid]
    genes = genes[valid]
    gene_ranked = rank_columns(residual_expr)

    incoming = incoming[incoming["method"].isin(FOCUS_METHODS)].copy()
    incoming_by_method = {method: table.set_index("cell_id").reindex(selected) for method, table in incoming.groupby("method", sort=False)}
    gene_rows = []
    summary_rows = []
    for method, table in incoming_by_method.items():
        for lr_id in FOCUS_LR_IDS:
            if lr_id not in table.columns:
                continue
            signal = pd.to_numeric(table[lr_id], errors="coerce").fillna(0).to_numpy(dtype=float)
            if np.nanstd(signal) <= 1e-10:
                continue
            signal_resid = residualize_vector(signal, covariates)
            if np.nanstd(signal_resid) <= 1e-10:
                continue
            rho = spearman_against_ranked(signal_resid, gene_ranked)
            rank = rank_descending(rho)
            ann = lr_pairs.loc[lr_id].to_dict() if lr_id in lr_pairs.index else {}
            target_genes = [gene for gene in target_map.get(lr_id, []) if gene in set(genes)]
            target_set = set(target_genes)
            target_ranks = []
            for gene, value, gene_rank in zip(genes, rho, rank):
                is_target = gene in target_set
                if is_target:
                    target_ranks.append(int(gene_rank))
                if is_target or gene_rank <= 200:
                    gene_rows.append(
                        {
                            "method": method,
                            "lr_id": lr_id,
                            "lr_pair": f"{ann.get('ligand', '')}-{ann.get('receptor', '')}",
                            "gene": gene,
                            "spearman_r": float(value),
                            "whole_transcriptome_rank": int(gene_rank),
                            "is_curated_target": bool(is_target),
                            "true_lr": bool(ann.get("true_lr", False)),
                            "false_positive_high_expression": bool(ann.get("false_positive_high_expression", False)),
                            "control_class": ann.get("control_class", ""),
                        }
                    )
            n_targets = len(target_ranks)
            n_genes = len(genes)
            if n_targets:
                target_ranks_arr = np.asarray(target_ranks, dtype=float)
                top200 = int(np.sum(target_ranks_arr <= 200))
                top500 = int(np.sum(target_ranks_arr <= 500))
                top1000 = int(np.sum(target_ranks_arr <= 1000))
                mean_rank = float(np.mean(target_ranks_arr))
                median_rank = float(np.median(target_ranks_arr))
                best_rank = int(np.min(target_ranks_arr))
                enrichment_top500 = float((top500 / n_targets) / (500 / n_genes))
            else:
                top200 = top500 = top1000 = 0
                mean_rank = median_rank = enrichment_top500 = np.nan
                best_rank = -1
            summary_rows.append(
                {
                    "method": method,
                    "lr_id": lr_id,
                    "lr_pair": f"{ann.get('ligand', '')}-{ann.get('receptor', '')}",
                    "true_lr": bool(ann.get("true_lr", False)),
                    "false_positive_high_expression": bool(ann.get("false_positive_high_expression", False)),
                    "control_class": ann.get("control_class", ""),
                    "n_genes_tested": int(n_genes),
                    "n_curated_targets_present": int(n_targets),
                    "n_targets_top200": top200,
                    "n_targets_top500": top500,
                    "n_targets_top1000": top1000,
                    "target_top500_enrichment": enrichment_top500,
                    "mean_target_rank": mean_rank,
                    "median_target_rank": median_rank,
                    "best_target_rank": best_rank,
                }
            )

    gene_table = pd.DataFrame(gene_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = add_negative_specificity(summary)
    gene_table.to_csv(RESULT_DIR / "lymph_node_full_transcriptome_target_rank_gene_table.csv", index=False)
    summary.to_csv(RESULT_DIR / "lymph_node_full_transcriptome_target_rank_summary.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_full_subset",
        "raw_matrix": rel(RAW_H5),
        "n_spots": int(len(cells)),
        "n_raw_genes": int(read_raw_gene_count(RAW_H5)),
        "n_expressed_genes_tested": int(summary["n_genes_tested"].max()) if not summary.empty else 0,
        "methods": FOCUS_METHODS,
        "focus_lr_ids": FOCUS_LR_IDS,
        "model": "Residualize log1p gene expression and incoming score against library size + x + y; rank all expressed genes by Spearman association with residual incoming score.",
        "claim_boundary": "Whole-transcriptome target-rank scan uses all expressed genes from the native 36,601-gene Visium matrix as ranking background. Curated target sets are literature/proxy target lists, not causal ground truth.",
        "outputs": [
            "lymph_node_full_transcriptome_target_rank_gene_table.csv",
            "lymph_node_full_transcriptome_target_rank_summary.csv",
        ],
    }
    (RESULT_DIR / "lymph_node_full_transcriptome_target_rank_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def read_10x_matrix(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(group["shape"][()])
        matrix = sparse.csc_matrix(
            (group["data"][()], group["indices"][()], group["indptr"][()]),
            shape=(int(shape[0]), int(shape[1])),
        ).T.tocsr()
        genes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in group["features"]["name"][()]])
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in group["barcodes"][()]]
    return matrix, genes, barcodes


def read_raw_gene_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["matrix"]["shape"][()][0])


def load_targets(path: Path) -> dict[str, list[str]]:
    table = pd.read_csv(path)
    targets = {}
    for row in table.itertuples(index=False):
        targets[str(row.lr_id)] = [gene.strip() for gene in str(row.target_genes).replace(",", ";").split(";") if gene.strip()]
    return targets


def build_covariates(cells: pd.DataFrame) -> np.ndarray:
    library = pd.to_numeric(cells["library_size"], errors="coerce").fillna(0).to_numpy(dtype=float)
    x = pd.to_numeric(cells["x"], errors="coerce").fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(cells["y"], errors="coerce").fillna(0).to_numpy(dtype=float)
    cov = np.column_stack([np.ones(len(cells)), np.log1p(library), x, y])
    cov[:, 1:] = (cov[:, 1:] - cov[:, 1:].mean(axis=0)) / cov[:, 1:].std(axis=0)
    return cov


def residualize_matrix(matrix: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(covariates) @ matrix
    return matrix - covariates @ beta


def residualize_vector(vector: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(covariates) @ vector
    return vector - covariates @ beta


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(matrix, dtype=np.float32)
    for idx in range(matrix.shape[1]):
        ranked[:, idx] = rankdata(matrix[:, idx]).astype(np.float32)
    ranked -= ranked.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(ranked**2, axis=0, keepdims=True))
    ranked /= np.maximum(norms, 1e-8)
    return ranked


def spearman_against_ranked(signal: np.ndarray, ranked_expression: np.ndarray) -> np.ndarray:
    ranked_signal = rankdata(signal).astype(np.float32)
    ranked_signal -= ranked_signal.mean()
    ranked_signal /= max(float(np.sqrt(np.sum(ranked_signal**2))), 1e-8)
    return np.asarray(ranked_signal @ ranked_expression, dtype=float)


def rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def add_negative_specificity(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, table in summary.groupby("method", sort=False):
        negatives = table[table["false_positive_high_expression"].astype(bool)]
        best_negative_enrichment = float(negatives["target_top500_enrichment"].max()) if len(negatives) else np.nan
        best_negative_top500 = float(negatives["n_targets_top500"].max()) if len(negatives) else np.nan
        for row in table.to_dict("records"):
            row["top500_enrichment_minus_best_negative"] = (
                float(row["target_top500_enrichment"]) - best_negative_enrichment
                if np.isfinite(row["target_top500_enrichment"]) and np.isfinite(best_negative_enrichment)
                else np.nan
            )
            row["n_targets_top500_minus_best_negative"] = (
                float(row["n_targets_top500"]) - best_negative_top500
                if np.isfinite(best_negative_top500)
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
