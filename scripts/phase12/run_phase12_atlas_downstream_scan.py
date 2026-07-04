#!/usr/bin/env python
"""Whole-transcriptome downstream scan for Phase 12 atlas model pilot LR scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_pilot"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_downstream_scan"

DATASETS = {
    "visium_hd_crc_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome",
        "nichenet": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/nichenet_targets/lr_pairs__nichenet_targets.csv",
    },
    "post_xenium_colon_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome",
        "nichenet": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/nichenet_targets/lr_pairs__nichenet_targets.csv",
    },
    "human_lymph_node_visium_raw_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome",
        "nichenet": PROJECT_ROOT
        / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset/target_pathway_pairs.csv",
    },
}

FOCUS_LR = {
    "visium_hd_crc_full": ["SPP1-CD44", "CXCL12-CXCR4", "TGFB1-TGFBR2", "FTL-ITGB1", "B2M-CD44"],
    "post_xenium_colon_full": ["SPP1-CD44", "CXCL12-CXCR4", "TGFB1-TGFBR2", "CXCL13-CXCR5", "CD19-CD3E"],
    "human_lymph_node_visium_raw_full": ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CD48-CD244", "EEF1A1-CCR7"],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_gene_rows: list[pd.DataFrame] = []
    all_summary_rows: list[pd.DataFrame] = []
    all_status: list[dict[str, Any]] = []
    ranking = pd.read_csv(PILOT_DIR / "phase12_atlas_model_pilot_lr_ranking.csv")

    for dataset_key, spec in DATASETS.items():
        try:
            result = run_dataset(dataset_key, spec, ranking)
            all_gene_rows.append(result["gene_table"])
            all_summary_rows.append(result["summary"])
            all_status.append(result["status"])
        except Exception as exc:  # noqa: BLE001
            all_status.append({"dataset_key": dataset_key, "status": "failed", "error": repr(exc)})

    gene_table = pd.concat(all_gene_rows, ignore_index=True) if all_gene_rows else pd.DataFrame()
    summary = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()
    status = pd.DataFrame(all_status)
    gene_table.to_csv(OUT_DIR / "phase12_atlas_downstream_gene_table.csv", index=False)
    summary.to_csv(OUT_DIR / "phase12_atlas_downstream_lr_summary.csv", index=False)
    status.to_csv(OUT_DIR / "phase12_atlas_downstream_status.csv", index=False)
    write_status(summary, status)
    write_manifest(status)
    print(status.to_string(index=False))
    if not summary.empty:
        print()
        print(summary.sort_values(["dataset_key", "lr_rank"]).to_string(index=False))


def run_dataset(dataset_key: str, spec: dict[str, Path], ranking: pd.DataFrame) -> dict[str, Any]:
    replicate_dir = PILOT_DIR / "replicates" / dataset_key
    pred_path = (
        PILOT_DIR
        / "predictions"
        / dataset_key
        / "spatialself_wavelet_edge_prior"
        / f"{dataset_key}__spatialself_wavelet_predictions.csv"
    )
    cells = pd.read_csv(replicate_dir / "cells.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    pred = pd.read_csv(pred_path)
    incoming = receiver_incoming(candidates, pred, cells, lr_pairs)
    target_map = load_nichenet_targets(spec["nichenet"])
    counts, genes, barcodes = load_full_tx(spec["full_tx_dir"])
    expression, genes = align_expression_to_cells(counts, genes, barcodes, cells)

    covariates = build_covariates(cells)
    expression.data = np.log1p(expression.data)
    expressed = np.asarray((expression > 0).sum(axis=0)).ravel() >= max(8, int(round(0.03 * expression.shape[0])))
    expression = expression[:, expressed].tocsr()
    genes = genes[expressed]
    dense = np.asarray(expression.toarray(), dtype=np.float32)
    residual_expr = residualize_matrix(dense, covariates)
    valid = residual_expr.std(axis=0) > 1e-8
    residual_expr = residual_expr[:, valid]
    genes = genes[valid]
    gene_set = set(genes.tolist())
    ranked_expr = rank_columns(residual_expr)

    lr_rank = ranking[ranking["dataset_key"].eq(dataset_key)].copy()
    focus_pairs = set(FOCUS_LR.get(dataset_key, []))
    top_pairs = set(lr_rank.sort_values("rank")["lr_pair"].head(8).astype(str))
    selected_pairs = sorted(focus_pairs | top_pairs)
    selected_lr = lr_pairs[(lr_pairs["ligand"] + "-" + lr_pairs["receptor"]).isin(selected_pairs)].copy()

    gene_rows = []
    summary_rows = []
    for lr in selected_lr.itertuples(index=False):
        lr_pair = f"{lr.ligand}-{lr.receptor}"
        if lr.lr_id not in incoming.columns:
            continue
        signal = pd.to_numeric(incoming[lr.lr_id], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.nanstd(signal) <= 1e-10:
            continue
        signal_resid = residualize_vector(signal, covariates)
        if np.nanstd(signal_resid) <= 1e-10:
            continue
        rho = spearman_against_ranked(signal_resid, ranked_expr)
        ranks = rank_descending(rho)
        targets = [gene for gene in target_map.get(lr_pair, []) if gene in gene_set]
        target_set = set(targets)
        top_gene_cutoff = min(300, len(genes))
        target_rank_values = []
        for gene, corr, rank in zip(genes, rho, ranks):
            is_target = gene in target_set
            if is_target:
                target_rank_values.append(int(rank))
            if is_target or rank <= top_gene_cutoff:
                gene_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "method": "spatialself_wavelet_edge_prior",
                        "lr_id": lr.lr_id,
                        "lr_pair": lr_pair,
                        "gene": gene,
                        "spearman_r": float(corr),
                        "whole_transcriptome_rank": int(rank),
                        "is_nichenet_target": bool(is_target),
                        "true_lr": bool(lr.true_lr),
                        "false_positive_high_expression": bool(lr.false_positive_high_expression),
                        "biological_category": lr.biological_category,
                    }
                )
        target_summary = summarize_target_ranks(target_rank_values, len(genes))
        rank_row = lr_rank[lr_rank["lr_pair"].eq(lr_pair)].head(1)
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "method": "spatialself_wavelet_edge_prior",
                "lr_id": lr.lr_id,
                "lr_pair": lr_pair,
                "lr_rank": int(rank_row["rank"].iloc[0]) if len(rank_row) else -1,
                "mean_lr_score": float(rank_row["mean_score"].iloc[0]) if len(rank_row) else np.nan,
                "true_lr": bool(lr.true_lr),
                "false_positive_high_expression": bool(lr.false_positive_high_expression),
                "biological_category": lr.biological_category,
                "n_genes_tested": int(len(genes)),
                "n_nichenet_targets_present": int(len(target_rank_values)),
                **target_summary,
            }
        )

    gene_table = pd.DataFrame(gene_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["target_top300_enrichment_p"] = [
            target_enrichment_pvalue(
                int(row.n_genes_tested),
                int(row.n_nichenet_targets_present),
                min(300, int(row.n_genes_tested)),
                int(row.n_targets_top300),
            )
            for row in summary.itertuples(index=False)
        ]
        summary["target_top300_enrichment_bh_fdr"] = bh_adjust(
            summary["target_top300_enrichment_p"].to_numpy(dtype=float)
        )
        summary = add_specificity_metrics(summary)
    return {
        "gene_table": gene_table,
        "summary": summary,
        "status": {
            "dataset_key": dataset_key,
            "status": "completed",
            "n_cells": int(len(cells)),
            "n_genes_tested": int(summary["n_genes_tested"].max()) if not summary.empty else 0,
            "n_lr_tested": int(len(summary)),
            "n_gene_rows": int(len(gene_table)),
            "error": "",
        },
    }


def receiver_incoming(candidates: pd.DataFrame, pred: pd.DataFrame, cells: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    incoming = pd.DataFrame({"cell_id": cells["cell_id"].astype(str)})
    for lr_id in lr_pairs["lr_id"].astype(str):
        sub = table[table["lr_id"].astype(str).eq(lr_id)]
        agg = sub.groupby("receiver_id")["score"].sum()
        incoming[lr_id] = incoming["cell_id"].map(agg).fillna(0.0).to_numpy(dtype=float)
    return incoming


def load_nichenet_targets(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out = {}
    for row in table.itertuples(index=False):
        pair = f"{str(row.ligand).upper()}-{str(row.receptor).upper()}"
        genes = [gene.strip().upper() for gene in str(row.target_genes).replace(",", ";").split(";") if gene.strip()]
        out[pair] = genes
    return out


def load_full_tx(full_tx_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray, list[str]]:
    counts = sparse.load_npz(full_tx_dir / "expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(full_tx_dir / "genes.csv")["gene"].astype(str).str.upper().to_numpy()
    barcodes = pd.read_csv(full_tx_dir / "barcodes.csv")["cell_id"].astype(str).tolist()
    return counts, genes, barcodes


def align_expression_to_cells(
    counts: sparse.csr_matrix,
    genes: np.ndarray,
    barcodes: list[str],
    cells: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    lookup = {cell_id: idx for idx, cell_id in enumerate(barcodes)}
    selected = cells["cell_id"].astype(str).tolist()
    missing = [cell_id for cell_id in selected if cell_id not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} selected cells missing from full-transcriptome matrix; examples={missing[:5]}")
    rows = [lookup[cell_id] for cell_id in selected]
    return counts[rows, :].tocsr().astype(np.float32), genes


def build_covariates(cells: pd.DataFrame) -> np.ndarray:
    library_col = "library_size" if "library_size" in cells.columns else "total_counts"
    library = pd.to_numeric(cells.get(library_col, pd.Series([0] * len(cells))), errors="coerce").fillna(0).to_numpy(dtype=float)
    x = pd.to_numeric(cells["x"], errors="coerce").fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(cells["y"], errors="coerce").fillna(0).to_numpy(dtype=float)
    cov = np.column_stack([np.ones(len(cells)), np.log1p(library), x, y])
    scale = cov[:, 1:].std(axis=0)
    scale[scale <= 1e-8] = 1.0
    cov[:, 1:] = (cov[:, 1:] - cov[:, 1:].mean(axis=0)) / scale
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


def summarize_target_ranks(target_ranks: list[int], n_genes: int) -> dict[str, float | int]:
    if not target_ranks:
        return {
            "n_targets_top100": 0,
            "n_targets_top300": 0,
            "n_targets_top500": 0,
            "best_target_rank": -1,
            "median_target_rank": np.nan,
            "target_top300_enrichment": np.nan,
            "target_mean_percentile": np.nan,
        }
    arr = np.asarray(target_ranks, dtype=float)
    top300_bg = min(300, n_genes) / max(n_genes, 1)
    target_top300 = int(np.sum(arr <= 300))
    return {
        "n_targets_top100": int(np.sum(arr <= 100)),
        "n_targets_top300": target_top300,
        "n_targets_top500": int(np.sum(arr <= 500)),
        "best_target_rank": int(np.min(arr)),
        "median_target_rank": float(np.median(arr)),
        "target_top300_enrichment": float((target_top300 / len(arr)) / top300_bg) if top300_bg > 0 else np.nan,
        "target_mean_percentile": float(np.mean(1.0 - (arr - 1.0) / max(n_genes, 1))),
    }


def target_enrichment_pvalue(n_genes: int, n_targets: int, top_k: int, observed: int) -> float:
    if n_genes <= 0 or n_targets <= 0 or top_k <= 0 or observed <= 0:
        return np.nan
    return float(hypergeom.sf(int(observed) - 1, int(n_genes), int(n_targets), int(top_k)))


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


def add_specificity_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for dataset_key, group in summary.groupby("dataset_key"):
        negatives = group[group["false_positive_high_expression"].astype(bool)]
        best_negative = float(negatives["target_top300_enrichment"].max()) if len(negatives) else np.nan
        mask = summary["dataset_key"].eq(dataset_key)
        summary.loc[mask, "best_negative_target_top300_enrichment"] = best_negative
        summary.loc[mask, "target_enrichment_minus_best_negative"] = (
            summary.loc[mask, "target_top300_enrichment"].astype(float) - best_negative
        )
    return summary


def write_status(summary: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas Downstream Scan Status",
        "",
        "## Scope",
        "",
        "This is a whole-transcriptome receiver-side association scan for the Phase 12 SpatialSelf atlas model pilot. It ranks all expressed genes by association with LR-specific incoming scores after residualizing library size and x/y coordinates.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; cells={getattr(row, 'n_cells', 'NA')}; genes={getattr(row, 'n_genes_tested', 'NA')}; LR tested={getattr(row, 'n_lr_tested', 'NA')}"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not summary.empty:
        lines += ["", "## Target-Rank Highlights", ""]
        for dataset_key, sub in summary.groupby("dataset_key"):
            top = sub.sort_values(["true_lr", "target_enrichment_minus_best_negative", "target_top300_enrichment"], ascending=[False, False, False]).head(6)
            lines.append(f"- `{dataset_key}`:")
            for row in top.itertuples(index=False):
                lines.append(
                    f"  - {row.lr_pair}: LR rank {int(row.lr_rank)}, targets top300 {int(row.n_targets_top300)}/{int(row.n_nichenet_targets_present)}, enrichment {float(row.target_top300_enrichment) if pd.notna(row.target_top300_enrichment) else float('nan'):.2f}, delta vs best negative {float(row.target_enrichment_minus_best_negative) if pd.notna(row.target_enrichment_minus_best_negative) else float('nan'):.2f}"
                )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is observational downstream support. It does not prove causal ligand-to-target regulation, and it is still pilot-scale because it uses the Phase 12 atlas model pilot LR/candidate subset.",
    ]
    (OUT_DIR / "phase12_atlas_downstream_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_atlas_downstream_scan",
        "datasets": status["dataset_key"].tolist() if "dataset_key" in status.columns else [],
        "model": "receiver incoming SpatialSelf score vs all expressed genes; residualized by library size and x/y coordinates",
        "outputs": [
            "phase12_atlas_downstream_gene_table.csv",
            "phase12_atlas_downstream_lr_summary.csv",
            "phase12_atlas_downstream_status.csv",
            "phase12_atlas_downstream_status.md",
        ],
        "claim_boundary": "Pilot-scale observational downstream association, not causal validation.",
    }
    (OUT_DIR / "phase12_atlas_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
