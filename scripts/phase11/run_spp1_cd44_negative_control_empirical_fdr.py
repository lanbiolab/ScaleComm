#!/usr/bin/env python
"""Add receiver-shuffle empirical FDR to the SPP1-CD44 negative-control panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase10.run_phase10_control_specificity import (  # noqa: E402
    CURATED_TARGETS,
    DATASETS,
    communication_impact,
    target_enrichment,
)


PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    / "standardized_inputs/visium_hd_crc_p2crc_subset"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=20261107)
    args = parser.parse_args()

    result_dir = PHASE11_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    controls = pd.read_csv(result_dir / "spp1_cd44_case_control_summary.csv")
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    counts = sparse.load_npz(PHASE10_DIR / "full_transcriptome/expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(PHASE10_DIR / "full_transcriptome/genes.csv")["gene"].astype(str).to_numpy()
    barcodes = pd.read_csv(PHASE10_DIR / "full_transcriptome/barcodes.csv")["cell_id"].astype(str).tolist()
    if cells["cell_id"].astype(str).tolist() != barcodes:
        raise ValueError("Cell order mismatch between standardized cells and full-transcriptome matrix.")

    expressed = np.asarray((counts > 0).sum(axis=0)).ravel() >= args.min_cells
    gene_idx = np.flatnonzero(expressed)
    genes = genes[gene_idx]
    log_counts = counts[:, gene_idx].astype(np.float64)
    log_counts.data = np.log1p(log_counts.data)
    log_dense = np.asarray(log_counts.toarray(), dtype=np.float32)
    ranked_expr, valid_mask = rank_expression_matrix(log_dense)
    genes = genes[valid_mask]
    log_dense = log_dense[:, valid_mask]

    cfg = DATASETS["visium_hd_crc"]
    observed_signal = (
        incoming[incoming["method"].eq(cfg["focus_method"])][["cell_id", cfg["focus_pair"]]]
        .set_index("cell_id")
        .reindex(cells["cell_id"].astype(str))[cfg["focus_pair"]]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    targets = CURATED_TARGETS[cfg["focus_pair"]]
    rng = np.random.default_rng(args.random_seed)

    null_rows = []
    for idx in range(args.n_permutations):
        signal = rng.permutation(observed_signal)
        scan = scan_signal_fast(signal, ranked_expr, genes)
        enrichment = target_enrichment(scan, genes, targets, args.top_n)
        impact = communication_impact(signal, log_dense, genes, cells, targets, args.random_seed + idx + 1)
        null_rows.append(
            {
                "permutation": idx,
                "target_overlap_top_n": enrichment["target_overlap_top_n"],
                "mean_target_rank": enrichment["mean_target_rank"],
                "delta_cv_r2": impact["delta_cv_r2"],
            }
        )

    null = pd.DataFrame(null_rows)
    enhanced = add_empirical_fdr(controls, null)
    enhanced.to_csv(result_dir / "spp1_cd44_negative_control_empirical_fdr_summary.csv", index=False)
    null.to_csv(result_dir / "spp1_cd44_negative_control_empirical_fdr_null.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "focus_lr": "SPP1-CD44",
        "null_model": "receiver-shuffled SpatialSelf SPP1-CD44 incoming scores",
        "n_permutations": int(args.n_permutations),
        "top_n": int(args.top_n),
        "empirical_p_definitions": {
            "overlap": "Pr(null target_overlap_top_n >= observed/control target_overlap_top_n)",
            "delta_r2": "Pr(null delta_cv_r2 >= observed/control delta_cv_r2)",
            "rank": "Pr(null mean_target_rank <= observed/control mean_target_rank)",
        },
        "fdr": "BH adjustment across control rows for each empirical p-value family",
        "outputs": [
            "spp1_cd44_negative_control_empirical_fdr_summary.csv",
            "spp1_cd44_negative_control_empirical_fdr_null.csv",
        ],
        "claim_boundary": (
            "Empirical FDR is calibrated against receiver-shuffled incoming scores. "
            "It is a statistical specificity check for this panel, not perturbational causal validation."
        ),
    }
    (result_dir / "spp1_cd44_negative_control_empirical_fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(enhanced.to_string(index=False))


def scan_signal_fast(signal: np.ndarray, ranked_expression: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    if np.nanstd(signal) <= 1e-12:
        return pd.DataFrame({"gene": genes, "spearman_r": np.nan, "bh_fdr": np.nan, "status": "constant_signal"})
    signal_rank = rankdata(signal).astype(np.float64)
    signal_centered = signal_rank - signal_rank.mean()
    signal_norm = np.sqrt(np.sum(signal_centered**2))
    expr_centered = ranked_expression - ranked_expression.mean(axis=0, keepdims=True)
    expr_norm = np.sqrt(np.sum(expr_centered**2, axis=0))
    rho = (signal_centered @ expr_centered) / (signal_norm * expr_norm)
    rho = np.clip(rho, -0.999999, 0.999999)
    return pd.DataFrame({"gene": genes, "spearman_r": rho, "bh_fdr": np.nan, "status": "tested"})


def add_empirical_fdr(controls: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    out = controls.copy()
    n = len(null)
    out["empirical_p_overlap"] = [
        empirical_ge(null["target_overlap_top_n"], value) for value in out["target_overlap_top_n"]
    ]
    out["empirical_p_delta_r2"] = [empirical_ge(null["delta_cv_r2"], value) for value in out["delta_cv_r2"]]
    out["empirical_p_mean_rank"] = [empirical_le(null["mean_target_rank"], value) for value in out["mean_target_rank"]]
    out["empirical_fdr_overlap"] = bh_adjust(out["empirical_p_overlap"].to_numpy(dtype=float))
    out["empirical_fdr_delta_r2"] = bh_adjust(out["empirical_p_delta_r2"].to_numpy(dtype=float))
    out["empirical_fdr_mean_rank"] = bh_adjust(out["empirical_p_mean_rank"].to_numpy(dtype=float))
    out["empirical_fdr_combined"] = out[
        ["empirical_fdr_overlap", "empirical_fdr_delta_r2", "empirical_fdr_mean_rank"]
    ].max(axis=1)
    out["null_target_overlap_q95"] = float(null["target_overlap_top_n"].quantile(0.95))
    out["null_delta_r2_q95"] = float(null["delta_cv_r2"].quantile(0.95))
    out["null_mean_target_rank_q05"] = float(null["mean_target_rank"].quantile(0.05))
    out["n_empirical_permutations"] = n
    return out


def empirical_ge(null_values: pd.Series, observed: object) -> float:
    values = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    obs = float(observed) if pd.notna(observed) else np.nan
    if not np.isfinite(obs) or not len(values):
        return np.nan
    return float((np.sum(values >= obs) + 1) / (len(values) + 1))


def empirical_le(null_values: pd.Series, observed: object) -> float:
    values = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    obs = float(observed) if pd.notna(observed) else np.nan
    if not np.isfinite(obs) or not len(values):
        return np.nan
    return float((np.sum(values <= obs) + 1) / (len(values) + 1))


def rank_expression_matrix(log_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = log_dense.std(axis=0)
    valid = std > 1e-12
    ranked = np.apply_along_axis(rankdata, 0, log_dense[:, valid]).astype(np.float32)
    return ranked, valid


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    finite_p = pvals[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out[idx[order]] = adjusted
    return out


if __name__ == "__main__":
    main()
