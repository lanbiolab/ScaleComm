#!/usr/bin/env python
"""Broader-transcriptome downstream impact benchmark for lymph node Visium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/datasets/human_lymph_node_visium_phase6_expanded_replicate_00"
BASELINE_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/lymph_node_phase6_expanded"

PREDICTION_PATHS = {
    "spatialself": BASELINE_DIR
    / "spatialself_wavelet_edge_prior/human_lymph_node_visium_phase6_expanded_replicate_00__spatialself_wavelet_predictions.csv",
    "commot": BASELINE_DIR / "commot/human_lymph_node_visium_phase6_expanded_replicate_00__commot_predictions.csv",
    "cellchat": BASELINE_DIR / "cellchat/human_lymph_node_visium_phase6_expanded_replicate_00__cellchat_predictions.csv",
    "cellnest": BASELINE_DIR / "cellnest/human_lymph_node_visium_phase6_expanded_replicate_00__cellnest_predictions.csv",
    "spatalk": BASELINE_DIR / "spatalk/human_lymph_node_visium_phase6_expanded_replicate_00__spatalk_predictions.csv",
}

TARGETS = {
    "CCL19_CCR7_Tzone_semisynthetic": ["CCR7", "CXCR4", "IL7R", "LTB", "JUNB", "NFKBIA", "STAT1", "TCF7", "SELL", "CD3D", "CD3E"],
    "CCL21_CCR7_Tzone_phase6_weak": ["CCR7", "CXCR4", "IL7R", "LTB", "JUNB", "NFKBIA", "STAT1", "TCF7", "SELL", "CD3D", "CD3E"],
    "CXCL13_CXCR5_GC_semisynthetic_weak": ["CXCR5", "MS4A1", "PRDM1", "JUNB", "NFKBIA", "LTB", "CD79A", "CD79B", "FCER2"],
    "CXCL12_CXCR4_follicle_phase6_context": ["CXCR4", "CXCL12", "JUNB", "NFKBIA", "STAT1", "DUSP1", "NR4A1", "IL7R", "LTB"],
    "CD48_CD244_Tcell_phase6_contact_weak": ["CD244", "CD48", "CD247", "LCK", "FYN", "ZAP70", "RUNX3", "CD3D", "CD3E", "NCR3", "KLRD1", "BTLA", "CD96", "STAT1", "JUNB", "NFKBIA"],
    "SPP1_CD44_boundary_semisynthetic": ["CD44", "SPP1", "JUNB", "NFKBIA", "STAT1", "CXCR4", "LTB", "VIM", "ITGB1"],
    "ACTB_EGFR_false_positive_phase6": ["EGFR", "JUNB", "NFKBIA", "STAT1"],
    "GAPDH_ITGB1_false_positive_phase6": ["ITGB1", "JUNB", "NFKBIA", "VIM", "ACTG1"],
    "RPLP0_CD44_false_positive_phase6": ["CD44", "JUNB", "NFKBIA", "STAT1"],
    "EEF1A1_CXCR4_false_positive_phase6": ["CXCR4", "JUNB", "NFKBIA", "STAT1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=11807)
    parser.add_argument("--min-targets", type=int, default=3)
    args = parser.parse_args()

    results_dir = OUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    expression = pd.read_csv(DATASET_DIR / "expression.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    expression = expression.set_index("cell_id").reindex(cells["cell_id"].astype(str)).fillna(0.0)
    genes = list(expression.columns)
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    log_expr = np.log1p(expression.to_numpy(dtype=np.float32))

    covariates = np.column_stack(
        [
            np.ones(len(cells)),
            np.log1p(pd.to_numeric(cells["library_size"], errors="coerce").fillna(0).to_numpy(dtype=float)),
            pd.to_numeric(cells["x"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(cells["y"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    covariate_pinv = np.linalg.pinv(covariates)
    receiver_order = cells["cell_id"].astype(str).to_numpy()
    rng = np.random.default_rng(args.random_seed)

    gene_rows = []
    incoming_frames = []
    for method, prediction_path in PREDICTION_PATHS.items():
        predictions = pd.read_csv(prediction_path)
        incoming = build_receiver_incoming(candidates, predictions, receiver_order)
        incoming["method"] = method
        incoming_frames.append(incoming.reset_index(names="cell_id"))
        for lr_id, target_genes in TARGETS.items():
            present = [gene for gene in target_genes if gene in gene_index]
            if len(present) < args.min_targets or lr_id not in incoming.columns:
                continue
            signal = incoming[lr_id].to_numpy(dtype=float)
            for gene in present:
                y = log_expr[:, gene_index[gene]]
                observed = residualized_delta_r2(covariates, covariate_pinv, signal, y)
                perm = np.array(
                    [
                        residualized_delta_r2(covariates, covariate_pinv, rng.permutation(signal), y)["delta_r2"]
                        for _ in range(args.n_permutations)
                    ],
                    dtype=float,
                )
                valid_perm = perm[np.isfinite(perm)]
                if np.isfinite(observed["delta_r2"]) and len(valid_perm):
                    p_value = float((np.sum(valid_perm >= observed["delta_r2"]) + 1) / (len(valid_perm) + 1))
                    perm_mean = float(np.mean(valid_perm))
                    perm_q95 = float(np.quantile(valid_perm, 0.95))
                else:
                    p_value = np.nan
                    perm_mean = np.nan
                    perm_q95 = np.nan
                gene_rows.append(
                    {
                        "method": method,
                        "lr_id": lr_id,
                        "gene": gene,
                        "n_permutations": args.n_permutations,
                        "permutation_p": p_value,
                        "perm_delta_r2_mean": perm_mean,
                        "perm_delta_r2_q95": perm_q95,
                        **observed,
                    }
                )

    incoming_all = pd.concat(incoming_frames, ignore_index=True)
    gene_impact = pd.DataFrame(gene_rows)
    gene_impact["impact_bh_fdr"] = bh_adjust(gene_impact["permutation_p"].to_numpy(dtype=float))
    summary = summarize(gene_impact, lr_pairs)
    method_summary = summarize_methods(summary)

    incoming_all.to_csv(results_dir / "lymph_node_receiver_level_incoming_scores.csv", index=False)
    gene_impact.to_csv(results_dir / "lymph_node_broader_gene_level_impact.csv", index=False)
    summary.to_csv(results_dir / "lymph_node_broader_lr_level_impact_summary.csv", index=False)
    method_summary.to_csv(results_dir / "lymph_node_broader_method_impact_summary.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_phase6_expanded_replicate_00",
        "scope": "broader_transcriptome_2017_genes_not_full_18k_transcriptome",
        "n_spots": int(len(cells)),
        "n_genes": int(len(genes)),
        "n_candidate_edges": int(len(candidates)),
        "n_permutations": int(args.n_permutations),
        "methods": sorted(PREDICTION_PATHS),
        "model": "target gene expression residualized against library size + x + y; receiver incoming signal residual explains remaining target variation",
        "outputs": [
            "lymph_node_receiver_level_incoming_scores.csv",
            "lymph_node_broader_gene_level_impact.csv",
            "lymph_node_broader_lr_level_impact_summary.csv",
            "lymph_node_broader_method_impact_summary.csv",
        ],
        "claim_boundary": "This is a non-CRC broader-transcriptome replication with 2017 genes, not a native full-transcriptome validation.",
    }
    (results_dir / "lymph_node_broader_downstream_impact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(method_summary.to_string(index=False))
    print()
    print(summary[summary["priority_lr"]].sort_values(["method", "mean_delta_specificity_vs_best_negative"], ascending=[True, False]).to_string(index=False))


def build_receiver_incoming(candidates: pd.DataFrame, predictions: pd.DataFrame, receiver_order: np.ndarray) -> pd.DataFrame:
    table = candidates[["candidate_id", "receiver_id", "lr_id"]].merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="left",
    )
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    incoming = table.pivot_table(index="receiver_id", columns="lr_id", values="score", aggfunc="sum", fill_value=0.0)
    return incoming.reindex(receiver_order).fillna(0.0)


def residualize(y: np.ndarray, covariates: np.ndarray, covariate_pinv: np.ndarray) -> np.ndarray:
    return y - covariates @ (covariate_pinv @ y)


def residualized_delta_r2(covariates: np.ndarray, covariate_pinv: np.ndarray, signal: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if np.nanstd(y) <= 1e-12 or np.nanstd(signal) <= 1e-12:
        return {"r2_without_comm": np.nan, "r2_with_comm": np.nan, "delta_r2": np.nan}
    y_resid = residualize(np.asarray(y, dtype=float), covariates, covariate_pinv)
    signal_resid = residualize(np.asarray(signal, dtype=float), covariates, covariate_pinv)
    denom = float(np.sum(y_resid**2))
    sig_denom = float(np.sum(signal_resid**2))
    if denom <= 1e-12 or sig_denom <= 1e-12:
        return {"r2_without_comm": 0.0, "r2_with_comm": np.nan, "delta_r2": np.nan}
    beta = float(np.dot(signal_resid, y_resid) / sig_denom)
    fitted = beta * signal_resid
    delta = 1.0 - float(np.sum((y_resid - fitted) ** 2) / denom)
    return {"r2_without_comm": 0.0, "r2_with_comm": delta, "delta_r2": delta}


def summarize(gene_impact: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    annotations = lr_pairs.set_index("lr_id")[["ligand", "receptor", "mechanism", "weak_signal", "false_positive_high_expression", "true_lr"]]
    rows = []
    for (method, lr_id), table in gene_impact.groupby(["method", "lr_id"], sort=False):
        sig = table[table["impact_bh_fdr"].le(0.1)]
        best = table.sort_values("delta_r2", ascending=False).iloc[0]
        ann = annotations.loc[lr_id].to_dict() if lr_id in annotations.index else {}
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                **ann,
                "priority_lr": lr_id
                in {
                    "CCL19_CCR7_Tzone_semisynthetic",
                    "CCL21_CCR7_Tzone_phase6_weak",
                    "CXCL13_CXCR5_GC_semisynthetic_weak",
                    "CXCL12_CXCR4_follicle_phase6_context",
                    "CD48_CD244_Tcell_phase6_contact_weak",
                    "SPP1_CD44_boundary_semisynthetic",
                },
                "n_target_genes_tested": int(len(table)),
                "n_positive_delta_genes": int(table["delta_r2"].gt(0).sum()),
                "n_impact_fdr_0_1": int(len(sig)),
                "mean_delta_r2": float(table["delta_r2"].mean()),
                "median_delta_r2": float(table["delta_r2"].median()),
                "max_delta_r2": float(table["delta_r2"].max()),
                "best_gene": best["gene"],
                "best_gene_delta_r2": float(best["delta_r2"]),
                "best_gene_fdr": float(best["impact_bh_fdr"]),
            }
        )
    summary = pd.DataFrame(rows)
    out = []
    for method, table in summary.groupby("method", sort=False):
        negatives = table[table["false_positive_high_expression"].astype(bool)]
        best_negative = float(negatives["mean_delta_r2"].max()) if len(negatives) else np.nan
        for _, row in table.iterrows():
            out.append({**row.to_dict(), "best_negative_mean_delta_r2": best_negative, "mean_delta_specificity_vs_best_negative": row["mean_delta_r2"] - best_negative})
    return pd.DataFrame(out)


def summarize_methods(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, table in summary.groupby("method", sort=False):
        priority = table[table["priority_lr"]]
        weak = priority[priority["weak_signal"].astype(bool)]
        negatives = table[table["false_positive_high_expression"].astype(bool)]
        rows.append(
            {
                "method": method,
                "n_lr_tested": int(table["lr_id"].nunique()),
                "mean_priority_delta_r2": float(priority["mean_delta_r2"].mean()),
                "mean_weak_priority_delta_r2": float(weak["mean_delta_r2"].mean()),
                "mean_negative_delta_r2": float(negatives["mean_delta_r2"].mean()),
                "max_negative_delta_r2": float(negatives["mean_delta_r2"].max()),
                "priority_minus_max_negative": float(priority["mean_delta_r2"].mean() - negatives["mean_delta_r2"].max()),
                "n_priority_fdr_0_1": int(priority["n_impact_fdr_0_1"].gt(0).sum()),
                "n_negative_fdr_0_1": int(negatives["n_impact_fdr_0_1"].gt(0).sum()),
                "best_priority_lr": priority.sort_values("mean_delta_r2", ascending=False).iloc[0]["lr_id"] if len(priority) else "",
                "best_priority_delta_r2": float(priority["mean_delta_r2"].max()) if len(priority) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("priority_minus_max_negative", ascending=False)


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out
    finite_indices = np.flatnonzero(finite)
    finite_p = pvals[finite]
    n = len(finite_p)
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out[finite_indices[order]] = adjusted
    return out


if __name__ == "__main__":
    main()
