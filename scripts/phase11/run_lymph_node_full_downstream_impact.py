#!/usr/bin/env python
"""Downstream delta-R2 impact benchmark for native full-transcriptome lymph node Visium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
BASELINE_DIR = OUT_DIR / "baselines"

PREDICTION_PATHS = {
    "spatialself": BASELINE_DIR / "spatialself_wavelet_edge_prior/dataset__spatialself_wavelet_predictions.csv",
    "commot": BASELINE_DIR / "commot/dataset__commot_predictions.csv",
    "cellchat": BASELINE_DIR / "cellchat/dataset__cellchat_predictions.csv",
    "spatalk": BASELINE_DIR / "spatalk/dataset__spatalk_predictions.csv",
    "cellnest": BASELINE_DIR / "cellnest/dataset__cellnest_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=110719)
    parser.add_argument("--min-targets", type=int, default=3)
    args = parser.parse_args()

    result_dir = OUT_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    expression = pd.read_csv(DATASET_DIR / "expression.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    target_map = load_targets(DATASET_DIR / "target_pathway_pairs.csv")

    expression = expression.set_index("cell_id").reindex(cells["cell_id"].astype(str)).fillna(0.0)
    genes = list(expression.columns)
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    log_expr = np.log1p(expression.to_numpy(dtype=np.float32))
    covariates = build_covariates(cells)
    covariate_pinv = np.linalg.pinv(covariates)
    receiver_order = cells["cell_id"].astype(str).to_numpy()
    rng = np.random.default_rng(args.random_seed)

    gene_rows = []
    incoming_frames = []
    status_rows = []
    for method, prediction_path in PREDICTION_PATHS.items():
        if not prediction_path.exists():
            status_rows.append({"method": method, "status": "prediction_missing", "prediction_path": rel(prediction_path)})
            continue
        predictions = pd.read_csv(prediction_path)
        incoming = build_receiver_incoming(candidates, predictions, receiver_order)
        incoming["method"] = method
        incoming_frames.append(incoming.reset_index(names="cell_id"))
        status_rows.append({"method": method, "status": "completed", "prediction_path": rel(prediction_path)})
        for lr_id, target_genes in target_map.items():
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

    incoming_all = pd.concat(incoming_frames, ignore_index=True) if incoming_frames else pd.DataFrame()
    gene_impact = pd.DataFrame(gene_rows)
    if not gene_impact.empty:
        gene_impact["impact_bh_fdr"] = bh_adjust(gene_impact["permutation_p"].to_numpy(dtype=float))
    summary = summarize_lr(gene_impact, lr_pairs)
    method_summary = summarize_methods(summary)

    incoming_all.to_csv(result_dir / "lymph_node_full_receiver_level_incoming_scores.csv", index=False)
    gene_impact.to_csv(result_dir / "lymph_node_full_gene_level_impact.csv", index=False)
    summary.to_csv(result_dir / "lymph_node_full_lr_level_impact_summary.csv", index=False)
    method_summary.to_csv(result_dir / "lymph_node_full_method_impact_summary.csv", index=False)
    pd.DataFrame(status_rows).to_csv(result_dir / "lymph_node_full_downstream_status.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_full_subset",
        "scope": "native_non_crc_full_transcriptome_matrix_36601_raw_genes_subset_expression_54_target_lr_marker_genes",
        "dataset_dir": rel(DATASET_DIR),
        "baseline_dir": rel(BASELINE_DIR),
        "n_spots": int(len(cells)),
        "n_raw_matrix_genes": int(json.loads((DATASET_DIR / "metadata.json").read_text()).get("n_raw_matrix_genes", 0)),
        "n_expression_genes_for_test": int(len(genes)),
        "n_candidate_edges": int(len(candidates)),
        "n_permutations": int(args.n_permutations),
        "methods": sorted(PREDICTION_PATHS),
        "model": "target gene expression residualized against library size + x + y; receiver incoming signal residual explains remaining target variation",
        "claim_boundary": (
            "This is native full-transcriptome input availability plus curated LR/target subset testing. "
            "The expression table used for this first-pass impact model contains LR, marker and target genes only; "
            "claims should be limited to curated lymph-node chemokine/contact axes and not whole-transcriptome target discovery."
        ),
    }
    (result_dir / "lymph_node_full_downstream_impact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(method_summary.to_string(index=False))
    print()
    print(summary.to_string(index=False))


def load_targets(path: Path) -> dict[str, list[str]]:
    table = pd.read_csv(path)
    targets = {}
    for row in table.itertuples(index=False):
        targets[str(row.lr_id)] = split_genes(getattr(row, "target_genes", ""))
    return targets


def split_genes(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [gene.strip() for gene in str(value).replace(",", ";").split(";") if gene.strip()]


def build_receiver_incoming(candidates: pd.DataFrame, predictions: pd.DataFrame, receiver_order: np.ndarray) -> pd.DataFrame:
    table = candidates[["candidate_id", "receiver_id", "lr_id"]].merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="left",
    )
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    incoming = table.pivot_table(index="receiver_id", columns="lr_id", values="score", aggfunc="sum", fill_value=0.0)
    return incoming.reindex(receiver_order).fillna(0.0)


def build_covariates(cells: pd.DataFrame) -> np.ndarray:
    if "library_size" in cells.columns:
        library = pd.to_numeric(cells["library_size"], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        library = np.zeros(len(cells), dtype=float)
    return np.column_stack(
        [
            np.ones(len(cells)),
            np.log1p(library),
            pd.to_numeric(cells["x"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(cells["y"], errors="coerce").to_numpy(dtype=float),
        ]
    )


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


def summarize_lr(gene_impact: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    if gene_impact.empty:
        return pd.DataFrame()
    annotations = lr_pairs.set_index("lr_id")[
        ["ligand", "receptor", "mechanism", "false_positive_high_expression", "true_lr", "control_class"]
    ]
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
                "priority_lr": not bool(ann.get("false_positive_high_expression", False)),
                "weak_signal": "weak" in str(ann.get("control_class", "")) or "context_specific" in str(ann.get("control_class", "")),
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
            out.append(
                {
                    **row.to_dict(),
                    "best_negative_mean_delta_r2": best_negative,
                    "mean_delta_specificity_vs_best_negative": float(row["mean_delta_r2"] - best_negative),
                }
            )
    return pd.DataFrame(out)


def summarize_methods(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for method, table in summary.groupby("method", sort=False):
        priority = table[table["priority_lr"].astype(bool)]
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
