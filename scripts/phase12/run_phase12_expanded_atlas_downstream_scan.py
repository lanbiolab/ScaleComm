#!/usr/bin/env python
"""Expanded-atlas downstream scan for Phase 12 LR and module candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATLAS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/expanded_atlas_downstream_scan"

DATASETS = {
    "visium_hd_crc_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome",
        "target_file": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/nichenet_targets/lr_pairs__nichenet_targets.csv",
    },
    "post_xenium_colon_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome",
        "target_file": PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/nichenet_targets/lr_pairs__nichenet_targets.csv",
    },
    "human_lymph_node_visium_raw_full": {
        "full_tx_dir": PROJECT_ROOT
        / "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome",
        "target_file": PROJECT_ROOT
        / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset/target_pathway_pairs.csv",
    },
}

TOP_RANK_N = 50
MODULE_TOP_N = 8
PRIORITY_RANK_MAX = 250
TOP_GENE_CUTOFF = 300


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(ATLAS_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    modules = pd.read_csv(ATLAS_DIR / "phase12_expanded_atlas_modules.csv")

    gene_tables: list[pd.DataFrame] = []
    lr_summaries: list[pd.DataFrame] = []
    module_summaries: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for dataset_key, spec in DATASETS.items():
        try:
            result = run_dataset(dataset_key, spec, ranking, modules)
            gene_tables.append(result["gene_table"])
            lr_summaries.append(result["lr_summary"])
            module_summaries.append(result["module_summary"])
            dataset_rows.append(result["dataset_summary"])
            status_rows.append(result["status"])
        except Exception as exc:  # noqa: BLE001
            status_rows.append({"dataset_key": dataset_key, "status": "failed", "error": repr(exc)})

    gene_table = pd.concat(gene_tables, ignore_index=True) if gene_tables else pd.DataFrame()
    lr_summary = pd.concat(lr_summaries, ignore_index=True) if lr_summaries else pd.DataFrame()
    module_summary = pd.concat(module_summaries, ignore_index=True) if module_summaries else pd.DataFrame()
    dataset_summary = pd.DataFrame(dataset_rows)
    status = pd.DataFrame(status_rows)

    gene_table.to_csv(OUT_DIR / "phase12_expanded_atlas_downstream_gene_table.csv", index=False)
    lr_summary.to_csv(OUT_DIR / "phase12_expanded_atlas_downstream_lr_summary.csv", index=False)
    module_summary.to_csv(OUT_DIR / "phase12_expanded_atlas_downstream_module_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_expanded_atlas_downstream_dataset_summary.csv", index=False)
    status.to_csv(OUT_DIR / "phase12_expanded_atlas_downstream_status.csv", index=False)
    write_status(lr_summary, module_summary, dataset_summary, status)
    write_manifest(status)

    print(status.to_string(index=False))
    if not dataset_summary.empty:
        print()
        print(dataset_summary.to_string(index=False))
    if not lr_summary.empty:
        cols = [
            "dataset_key",
            "lr_pair",
            "rank",
            "selection_sources",
            "n_targets_top300",
            "n_targets_present",
            "target_top300_enrichment",
            "target_top300_enrichment_bh_fdr",
        ]
        print()
        print(lr_summary.sort_values(["dataset_key", "rank"])[cols].head(40).to_string(index=False))


def run_dataset(
    dataset_key: str,
    spec: dict[str, Path],
    ranking: pd.DataFrame,
    modules: pd.DataFrame,
) -> dict[str, Any]:
    replicate_dir = ATLAS_DIR / "replicates" / dataset_key
    pred_path = (
        ATLAS_DIR
        / "predictions"
        / dataset_key
        / "spatialself_wavelet_edge_prior"
        / f"{dataset_key}__spatialself_wavelet_predictions.csv"
    )
    cells = pd.read_csv(replicate_dir / "cells.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(
        replicate_dir / "candidate_interactions.csv",
        usecols=["candidate_id", "receiver_id", "lr_id"],
    )
    pred = pd.read_csv(pred_path, usecols=["candidate_id", "score"])
    target_map = load_targets(spec["target_file"])

    dataset_ranking = ranking[ranking["dataset_key"].astype(str).eq(dataset_key)].copy()
    selected = select_lr_pairs(dataset_key, dataset_ranking, modules)
    selected_lr = lr_pairs[(lr_pairs["ligand"] + "-" + lr_pairs["receptor"]).isin(selected["lr_pair"])].copy()
    selected_lr["lr_pair"] = selected_lr["ligand"] + "-" + selected_lr["receptor"]
    selected_lr = selected_lr.merge(selected, on="lr_pair", how="left")
    selected_lr = selected_lr.merge(
        dataset_ranking[
            [
                "lr_id",
                "rank",
                "rank_q95",
                "calibrated_score",
                "calibrated_q95_score",
                "true_lr",
                "false_positive_high_expression",
                "biological_category",
                "mechanism",
                "selected_reason",
            ]
        ],
        on="lr_id",
        how="left",
        suffixes=("", "_rank"),
    )
    selected_lr = selected_lr.drop_duplicates("lr_id").sort_values(["rank", "lr_pair"])

    incoming = receiver_incoming(candidates, pred, cells, selected_lr["lr_id"].astype(str).tolist())
    counts, genes, barcodes = load_full_tx(spec["full_tx_dir"])
    expression, genes = align_expression_to_cells(counts, genes, barcodes, cells)
    covariates = build_covariates(cells)

    expression.data = np.log1p(expression.data)
    min_cells = max(8, int(round(0.03 * expression.shape[0])))
    expressed = np.asarray((expression > 0).sum(axis=0)).ravel() >= min_cells
    expression = expression[:, expressed].tocsr()
    genes = genes[expressed]
    dense = np.asarray(expression.toarray(), dtype=np.float32)
    residual_expr = residualize_matrix(dense, covariates)
    valid = residual_expr.std(axis=0) > 1e-8
    residual_expr = residual_expr[:, valid]
    genes = genes[valid]
    gene_set = set(genes.tolist())
    ranked_expr = rank_columns(residual_expr)

    gene_rows = []
    summary_rows = []
    for lr in selected_lr.itertuples(index=False):
        lr_id = str(lr.lr_id)
        lr_pair = str(lr.lr_pair)
        if lr_id not in incoming.columns:
            continue
        signal = incoming[lr_id].to_numpy(dtype=float)
        if np.nanstd(signal) <= 1e-10:
            continue
        signal_resid = residualize_vector(signal, covariates)
        if np.nanstd(signal_resid) <= 1e-10:
            continue
        rho = spearman_against_ranked(signal_resid, ranked_expr)
        ranks = rank_descending(rho)
        targets = [gene for gene in target_map.get(lr_pair.upper(), []) if gene in gene_set]
        target_set = set(targets)
        target_rank_values = []
        cutoff = min(TOP_GENE_CUTOFF, len(genes))
        for gene, corr, gene_rank in zip(genes, rho, ranks):
            is_target = gene in target_set
            if is_target:
                target_rank_values.append(int(gene_rank))
            if is_target or gene_rank <= cutoff:
                gene_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "method": "spatialself_wavelet_edge_prior_calibrated",
                        "lr_id": lr_id,
                        "lr_pair": lr_pair,
                        "gene": gene,
                        "spearman_r": float(corr),
                        "whole_transcriptome_rank": int(gene_rank),
                        "is_target": bool(is_target),
                        "rank": int(lr.rank) if pd.notna(lr.rank) else -1,
                        "selection_sources": str(lr.selection_sources),
                        "biological_category": str(lr.biological_category),
                        "mechanism": str(lr.mechanism),
                    }
                )
        target_summary = summarize_target_ranks(target_rank_values, len(genes))
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "method": "spatialself_wavelet_edge_prior_calibrated",
                "lr_id": lr_id,
                "lr_pair": lr_pair,
                "rank": int(lr.rank) if pd.notna(lr.rank) else -1,
                "rank_q95": int(lr.rank_q95) if pd.notna(lr.rank_q95) else -1,
                "calibrated_score": float(lr.calibrated_score) if pd.notna(lr.calibrated_score) else np.nan,
                "calibrated_q95_score": float(lr.calibrated_q95_score) if pd.notna(lr.calibrated_q95_score) else np.nan,
                "selection_sources": str(lr.selection_sources),
                "true_lr": bool(lr.true_lr),
                "false_positive_high_expression": bool(lr.false_positive_high_expression),
                "biological_category": str(lr.biological_category),
                "mechanism": str(lr.mechanism),
                "n_genes_tested": int(len(genes)),
                "n_targets_present": int(len(target_rank_values)),
                **target_summary,
            }
        )

    gene_table = pd.DataFrame(gene_rows)
    lr_summary = pd.DataFrame(summary_rows)
    if not lr_summary.empty:
        lr_summary["target_top300_enrichment_p"] = [
            target_enrichment_pvalue(
                int(row.n_genes_tested),
                int(row.n_targets_present),
                min(TOP_GENE_CUTOFF, int(row.n_genes_tested)),
                int(row.n_targets_top300),
            )
            for row in lr_summary.itertuples(index=False)
        ]
        lr_summary["target_top300_enrichment_bh_fdr"] = bh_adjust(
            lr_summary["target_top300_enrichment_p"].to_numpy(dtype=float)
        )
        lr_summary = add_specificity_metrics(lr_summary)

    module_summary = summarize_modules(dataset_key, lr_summary, modules)
    dataset_summary = summarize_dataset(dataset_key, cells, genes, selected_lr, lr_summary, module_summary)
    return {
        "gene_table": gene_table,
        "lr_summary": lr_summary,
        "module_summary": module_summary,
        "dataset_summary": dataset_summary,
        "status": {
            "dataset_key": dataset_key,
            "status": "completed",
            "n_cells": int(len(cells)),
            "n_genes_tested": int(len(genes)),
            "n_lr_selected": int(len(selected_lr)),
            "n_lr_scanned": int(len(lr_summary)),
            "n_gene_rows": int(len(gene_table)),
            "error": "",
        },
    }


def select_lr_pairs(dataset_key: str, ranking: pd.DataFrame, modules: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, set[str]] = {}

    def add(pair: str, source: str) -> None:
        pair = str(pair)
        if not pair or pair == "nan":
            return
        records.setdefault(pair, set()).add(source)

    for pair in ranking.sort_values("rank")["lr_pair"].head(TOP_RANK_N).astype(str):
        add(pair, f"top{TOP_RANK_N}")
    priority = ranking[
        ranking["selected_reason"].astype(str).eq("phase12_priority_anchor")
        & (pd.to_numeric(ranking["rank"], errors="coerce") <= PRIORITY_RANK_MAX)
    ]
    for pair in priority["lr_pair"].astype(str):
        add(pair, f"priority_rank_le_{PRIORITY_RANK_MAX}")
    module_rows = modules[modules["dataset_key"].astype(str).eq(dataset_key)]
    for row in module_rows.itertuples(index=False):
        pairs = [p for p in str(row.top_lr_pairs).split(";") if p]
        for pair in pairs[:MODULE_TOP_N]:
            add(pair, f"module_{row.module}_top{MODULE_TOP_N}")

    return pd.DataFrame(
        [{"lr_pair": pair, "selection_sources": ";".join(sorted(sources))} for pair, sources in records.items()]
    )


def receiver_incoming(
    candidates: pd.DataFrame,
    pred: pd.DataFrame,
    cells: pd.DataFrame,
    selected_lr_ids: list[str],
) -> pd.DataFrame:
    selected = set(selected_lr_ids)
    sub_candidates = candidates[candidates["lr_id"].astype(str).isin(selected)].copy()
    table = sub_candidates.merge(pred, on="candidate_id", how="inner")
    grouped = table.groupby(["receiver_id", "lr_id"], sort=False)["score"].sum().reset_index()
    incoming = grouped.pivot(index="receiver_id", columns="lr_id", values="score")
    incoming = incoming.reindex(cells["cell_id"].astype(str)).fillna(0.0)
    incoming.index.name = "cell_id"
    return incoming.reset_index()


def load_targets(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out: dict[str, list[str]] = {}
    for row in table.itertuples(index=False):
        ligand = str(getattr(row, "ligand", "")).upper()
        receptor = str(getattr(row, "receptor", "")).upper()
        if not ligand or not receptor:
            continue
        genes = [
            gene.strip().upper()
            for gene in str(getattr(row, "target_genes", "")).replace(",", ";").split(";")
            if gene.strip()
        ]
        out[f"{ligand}-{receptor}"] = genes
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
    cutoff_bg = min(TOP_GENE_CUTOFF, n_genes) / max(n_genes, 1)
    target_top300 = int(np.sum(arr <= TOP_GENE_CUTOFF))
    return {
        "n_targets_top100": int(np.sum(arr <= 100)),
        "n_targets_top300": target_top300,
        "n_targets_top500": int(np.sum(arr <= 500)),
        "best_target_rank": int(np.min(arr)),
        "median_target_rank": float(np.median(arr)),
        "target_top300_enrichment": float((target_top300 / len(arr)) / cutoff_bg) if cutoff_bg > 0 else np.nan,
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


def summarize_modules(dataset_key: str, lr_summary: pd.DataFrame, modules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lr_summary.empty:
        return pd.DataFrame()
    module_rows = modules[modules["dataset_key"].astype(str).eq(dataset_key)]
    by_pair = lr_summary.set_index("lr_pair")
    for row in module_rows.itertuples(index=False):
        pairs = [pair for pair in str(row.top_lr_pairs).split(";") if pair]
        present = by_pair.reindex([pair for pair in pairs if pair in by_pair.index])
        sig = int(pd.to_numeric(present.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).sum()) if len(present) else 0
        rows.append(
            {
                "dataset_key": dataset_key,
                "module": row.module,
                "n_lr_in_module": int(row.n_lr),
                "n_top_pairs_tested": int(len(present)),
                "n_target_enriched_q_0_1": sig,
                "best_target_enrichment_fdr": float(pd.to_numeric(present.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce").min()) if len(present) else np.nan,
                "mean_target_top300_enrichment": float(pd.to_numeric(present.get("target_top300_enrichment", pd.Series(dtype=float)), errors="coerce").mean()) if len(present) else np.nan,
                "top_pairs_tested": ";".join(present.index.astype(str).tolist()) if len(present) else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(
    dataset_key: str,
    cells: pd.DataFrame,
    genes: np.ndarray,
    selected_lr: pd.DataFrame,
    lr_summary: pd.DataFrame,
    module_summary: pd.DataFrame,
) -> dict[str, Any]:
    q = pd.to_numeric(lr_summary.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce")
    return {
        "dataset_key": dataset_key,
        "n_cells": int(len(cells)),
        "n_genes_tested": int(len(genes)),
        "n_lr_selected": int(len(selected_lr)),
        "n_lr_scanned": int(len(lr_summary)),
        "n_top_rank_lr": int(lr_summary["selection_sources"].astype(str).str.contains(f"top{TOP_RANK_N}", regex=False).sum()) if not lr_summary.empty else 0,
        "n_module_representative_lr": int(lr_summary["selection_sources"].astype(str).str.contains("module_", regex=False).sum()) if not lr_summary.empty else 0,
        "n_target_enriched_q_0_1": int(q.le(0.1).sum()) if len(q) else 0,
        "n_target_enriched_q_0_2": int(q.le(0.2).sum()) if len(q) else 0,
        "n_modules_with_target_enriched_rep": int(pd.to_numeric(module_summary.get("n_target_enriched_q_0_1", pd.Series(dtype=float)), errors="coerce").gt(0).sum()) if not module_summary.empty else 0,
    }


def write_status(
    lr_summary: pd.DataFrame,
    module_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 12 Expanded Atlas Downstream Scan Status",
        "",
        "## Scope",
        "",
        "This scan aligns receiver-side whole-transcriptome association with the expanded calibrated atlas. It covers calibrated top LR, priority anchors and module representative LR, rather than only the earlier pilot anchors.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; cells={getattr(row, 'n_cells', 'NA')}; genes={getattr(row, 'n_genes_tested', 'NA')}; LR scanned={getattr(row, 'n_lr_scanned', 'NA')}"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not dataset_summary.empty:
        lines += ["", "## Coverage", ""]
        for row in dataset_summary.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}`: selected LR={int(row.n_lr_selected)}, scanned LR={int(row.n_lr_scanned)}, module-rep LR={int(row.n_module_representative_lr)}, target-enriched q<=0.1={int(row.n_target_enriched_q_0_1)}, modules with enriched representative={int(row.n_modules_with_target_enriched_rep)}"
            )
    if not lr_summary.empty:
        lines += ["", "## Strongest Target-Rank Examples", ""]
        valid = lr_summary[pd.to_numeric(lr_summary["target_top300_enrichment_bh_fdr"], errors="coerce").notna()].copy()
        for dataset_key, sub in valid.groupby("dataset_key"):
            lines.append(f"- `{dataset_key}`:")
            top = sub.sort_values(["target_top300_enrichment_bh_fdr", "rank"]).head(8)
            for row in top.itertuples(index=False):
                lines.append(
                    f"  - {row.lr_pair}: rank={int(row.rank)}, q={float(row.target_top300_enrichment_bh_fdr):.3g}, targets top300={int(row.n_targets_top300)}/{int(row.n_targets_present)}, sources={row.selection_sources}"
                )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is observational functional support, not perturbational proof. It strengthens expanded-atlas and module coverage, but should not be described as causal ligand-to-target validation.",
    ]
    (OUT_DIR / "phase12_expanded_atlas_downstream_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_expanded_atlas_downstream_scan",
        "datasets": status["dataset_key"].tolist() if "dataset_key" in status.columns else [],
        "selection": {
            "top_rank_n": TOP_RANK_N,
            "module_top_n": MODULE_TOP_N,
            "priority_rank_max": PRIORITY_RANK_MAX,
        },
        "model": "receiver incoming SpatialSelf calibrated expanded-atlas score vs all expressed genes; residualized by library size and x/y coordinates",
        "outputs": [
            "phase12_expanded_atlas_downstream_gene_table.csv",
            "phase12_expanded_atlas_downstream_lr_summary.csv",
            "phase12_expanded_atlas_downstream_module_summary.csv",
            "phase12_expanded_atlas_downstream_dataset_summary.csv",
            "phase12_expanded_atlas_downstream_status.csv",
            "phase12_expanded_atlas_downstream_status.md",
        ],
        "claim_boundary": "Expanded-atlas observational downstream association, not causal validation.",
    }
    (OUT_DIR / "phase12_expanded_atlas_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
