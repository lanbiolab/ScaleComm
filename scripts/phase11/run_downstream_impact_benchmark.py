#!/usr/bin/env python
"""Target gene-level downstream impact benchmark with permutation FDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    / "standardized_inputs/visium_hd_crc_p2crc_subset"
)

CURATED_TARGETS = {
    "SPP1-CD44": [
        "LYZ",
        "CD68",
        "CD14",
        "C1QA",
        "C1QB",
        "C1QC",
        "LST1",
        "FCGR3A",
        "MSR1",
        "MRC1",
        "COL1A1",
        "COL1A2",
        "FN1",
        "LGALS3",
        "CTSS",
        "APOE",
        "TREM2",
    ],
    "TGFB1-TGFBR2": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "DCN", "LUM", "ACTA2", "TAGLN", "FN1", "SERPINE1", "THBS1", "CTGF", "MMP2", "TIMP1", "SPARC"],
    "VEGFA-KDR": ["KDR", "FLT1", "PECAM1", "VWF", "ENG", "ESAM", "RAMP2", "EMCN", "CDH5", "ANGPT2", "SOX18", "PLVAP", "MCAM"],
    "CXCL12-CXCR4": ["CXCR4", "CXCL12", "JUN", "FOS", "FOSB", "EGR1", "EGR2", "NR4A1", "DUSP1", "DUSP2", "SOCS3", "NFKBIA", "ICAM1", "MMP9", "PTGS2"],
    "CD274-PDCD1": ["PDCD1", "CD274", "CD8A", "CD8B", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "IFNG", "GZMB", "PRF1", "CXCL9", "CXCL10", "STAT1", "IRF1"],
    "ACTB-KRT8": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
    "COL1A1-EPCAM": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=11001)
    args = parser.parse_args()

    out = PHASE11_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    counts, genes = load_expression()
    log_counts = counts.astype(np.float64)
    log_counts.data = np.log1p(log_counts.data)
    log_dense = np.asarray(log_counts.toarray(), dtype=np.float32)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    covariates = pd.DataFrame(
        {
            "library_log_sum": np.asarray(log_dense.sum(axis=1)).ravel(),
            "x": pd.to_numeric(cells["x"], errors="coerce").to_numpy(),
            "y": pd.to_numeric(cells["y"], errors="coerce").to_numpy(),
        }
    )
    covariate_matrix = with_intercept(covariates.to_numpy(dtype=float))
    covariate_hat = np.linalg.pinv(covariate_matrix)
    rng = np.random.default_rng(args.random_seed)

    rows = []
    for method, table in incoming.groupby("method", sort=False):
        table = table.set_index("cell_id").reindex(cells["cell_id"].astype(str)).reset_index()
        for lr_pair, targets in CURATED_TARGETS.items():
            if lr_pair not in table.columns:
                continue
            signal = table[lr_pair].fillna(0.0).to_numpy(dtype=float)
            for gene in targets:
                if gene not in gene_index:
                    continue
                y = log_dense[:, gene_index[gene]]
                observed = residualized_delta_r2(covariate_matrix, covariate_hat, signal, y)
                perm = np.array(
                    [
                        residualized_delta_r2(covariate_matrix, covariate_hat, rng.permutation(signal), y)[
                            "delta_cv_r2"
                        ]
                        for _ in range(args.n_permutations)
                    ],
                    dtype=float,
                )
                p_value = float((np.sum(perm >= observed["delta_cv_r2"]) + 1) / (len(perm) + 1))
                rows.append(
                    {
                        "method": method,
                        "lr_pair": lr_pair,
                        "gene": gene,
                        "n_permutations": args.n_permutations,
                        "permutation_p": p_value,
                        "perm_delta_cv_r2_mean": float(np.mean(perm)),
                        "perm_delta_cv_r2_q95": float(np.quantile(perm, 0.95)),
                        **observed,
                    }
                )

    gene_impact = pd.DataFrame(rows)
    gene_impact["impact_bh_fdr"] = bh_adjust(gene_impact["permutation_p"].to_numpy(dtype=float))
    summary = summarize(gene_impact)
    gene_impact.to_csv(out / "downstream_gene_level_impact.csv", index=False)
    summary.to_csv(out / "downstream_lr_level_impact_summary.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "n_permutations": args.n_permutations,
        "model": "target gene expression is residualized against library size + x + y; incoming score residual explains remaining target variation",
        "outputs": ["downstream_gene_level_impact.csv", "downstream_lr_level_impact_summary.csv"],
        "claim_boundary": "Gene-level impact is observational and permutation-calibrated against shuffled incoming scores.",
    }
    (out / "downstream_impact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def load_expression() -> tuple[sparse.csr_matrix, np.ndarray]:
    counts = sparse.load_npz(PHASE10_DIR / "full_transcriptome/expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(PHASE10_DIR / "full_transcriptome/genes.csv")["gene"].astype(str).to_numpy()
    return counts, genes


def with_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x])


def residualize(y: np.ndarray, x: np.ndarray, x_pinv: np.ndarray) -> np.ndarray:
    return y - x @ (x_pinv @ y)


def residualized_delta_r2(covariates: np.ndarray, covariate_pinv: np.ndarray, signal: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if np.nanstd(y) <= 1e-12 or np.nanstd(signal) <= 1e-12:
        return {"cv_r2_without_comm": np.nan, "cv_r2_with_comm": np.nan, "delta_cv_r2": np.nan}
    y_resid = residualize(np.asarray(y, dtype=float), covariates, covariate_pinv)
    signal_resid = residualize(np.asarray(signal, dtype=float), covariates, covariate_pinv)
    denom = float(np.sum(y_resid**2))
    if denom <= 1e-12 or np.nanstd(signal_resid) <= 1e-12:
        return {"cv_r2_without_comm": 0.0, "cv_r2_with_comm": np.nan, "delta_cv_r2": np.nan}
    beta = float(np.dot(signal_resid, y_resid) / np.dot(signal_resid, signal_resid))
    fitted = beta * signal_resid
    delta = 1.0 - float(np.sum((y_resid - fitted) ** 2) / denom)
    return {"cv_r2_without_comm": 0.0, "cv_r2_with_comm": delta, "delta_cv_r2": delta}


def summarize(gene_impact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, lr_pair), table in gene_impact.groupby(["method", "lr_pair"], sort=False):
        sig = table[table["impact_bh_fdr"].le(0.1)]
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                "n_target_genes_tested": int(len(table)),
                "n_positive_delta_genes": int(table["delta_cv_r2"].gt(0).sum()),
                "n_impact_fdr_0_1": int(len(sig)),
                "mean_delta_cv_r2": float(table["delta_cv_r2"].mean()),
                "median_delta_cv_r2": float(table["delta_cv_r2"].median()),
                "max_delta_cv_r2": float(table["delta_cv_r2"].max()),
                "best_gene": table.sort_values("delta_cv_r2", ascending=False).iloc[0]["gene"],
                "best_gene_delta_cv_r2": float(table["delta_cv_r2"].max()),
                "best_gene_fdr": float(table.sort_values("delta_cv_r2", ascending=False).iloc[0]["impact_bh_fdr"]),
            }
        )
    summary = pd.DataFrame(rows)
    negatives = {"ACTB-KRT8", "COL1A1-EPCAM"}
    spec_rows = []
    for method, table in summary.groupby("method", sort=False):
        best_neg = table[table["lr_pair"].isin(negatives)]["mean_delta_cv_r2"].max()
        for _, row in table.iterrows():
            spec_rows.append({**row.to_dict(), "mean_delta_specificity_vs_best_negative": row["mean_delta_cv_r2"] - best_neg})
    return pd.DataFrame(spec_rows)


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


if __name__ == "__main__":
    main()
