#!/usr/bin/env python
"""Run COMMOT official communication_impact on SpatialSelf and COMMOT incoming scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import commot as ct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    / "standardized_inputs/visium_hd_crc_p2crc_subset"
)

CURATED_TARGETS = {
    "SPP1-CD44": ["LYZ", "CD68", "CD14", "C1QA", "C1QB", "C1QC", "LST1", "FCGR3A", "MSR1", "MRC1", "COL1A1", "COL1A2", "FN1", "LGALS3", "CTSS", "APOE", "TREM2"],
    "TGFB1-TGFBR2": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "DCN", "LUM", "ACTA2", "TAGLN", "FN1", "SERPINE1", "THBS1", "CTGF", "MMP2", "TIMP1", "SPARC"],
    "VEGFA-KDR": ["KDR", "FLT1", "PECAM1", "VWF", "ENG", "ESAM", "RAMP2", "EMCN", "CDH5", "ANGPT2", "SOX18", "PLVAP", "MCAM"],
    "CXCL12-CXCR4": ["CXCR4", "CXCL12", "JUN", "FOS", "FOSB", "EGR1", "EGR2", "NR4A1", "DUSP1", "DUSP2", "SOCS3", "NFKBIA", "ICAM1", "MMP9", "PTGS2"],
    "CD274-PDCD1": ["PDCD1", "CD274", "CD8A", "CD8B", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "IFNG", "GZMB", "PRF1", "CXCL9", "CXCL10", "STAT1", "IRF1"],
    "ACTB-KRT8": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
    "COL1A1-EPCAM": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["semipartial_corr", "partial_corr", "treebased_score"], default="semipartial_corr")
    parser.add_argument("--bg-genes", type=int, default=200)
    parser.add_argument("--tree-ntrees", type=int, default=25)
    parser.add_argument("--tree-repeat", type=int, default=5)
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-combined", action="store_true")
    parser.add_argument(
        "--bg-mode",
        choices=["precomputed_list", "commot_hvg"],
        default="precomputed_list",
        help="Use a precomputed finite background gene list or let COMMOT call scanpy HVG internally.",
    )
    args = parser.parse_args()

    out = PHASE11_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    adata = build_adata()
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    genes = set(adata.var_names.astype(str))

    rows = []
    raw_outputs = []
    for method_name in ["spatialself", "commot"]:
        method_incoming = incoming[incoming["method"].eq(method_name)].set_index("cell_id").reindex(adata.obs_names)
        db_name = method_name
        setup_commot_slots(adata, db_name, lr_pairs, method_incoming)
        ds_genes = sorted({g for targets in CURATED_TARGETS.values() for g in targets if g in genes})
        try:
            bg_genes = (
                select_background_genes(adata, ds_genes, n_genes=args.bg_genes)
                if args.bg_mode == "precomputed_list"
                else args.bg_genes
            )
            impact = ct.tl.communication_impact(
                adata,
                database_name=db_name,
                pathway_sum_only=False,
                normalize=False,
                method=args.method,
                corr_method="spearman",
                tree_ntrees=args.tree_ntrees,
                tree_repeat=args.tree_repeat,
                tree_max_depth=args.tree_max_depth,
                tree_combined=args.tree_combined,
                ds_genes=ds_genes,
                bg_genes=bg_genes,
            )
            impact.to_csv(out / f"commot_official_impact_{method_name}.csv")
            raw_outputs.append(impact)
            rows.extend(summarize_impact(method_name, impact))
        except Exception as exc:
            rows.append(
                {
                    "method": method_name,
                    "lr_pair": "all",
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "commot_official_impact_summary.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "commot_function": "commot.tl.communication_impact",
        "method": args.method,
        "bg_genes": args.bg_genes,
        "bg_mode": args.bg_mode,
        "tree_ntrees": args.tree_ntrees,
        "tree_repeat": args.tree_repeat,
        "tree_max_depth": args.tree_max_depth,
        "tree_combined": args.tree_combined,
        "outputs": ["commot_official_impact_summary.csv", "commot_official_impact_spatialself.csv", "commot_official_impact_commot.csv"],
        "claim_boundary": (
            "This uses COMMOT's official communication_impact function after injecting SpatialSelf or COMMOT receiver scores "
            "into COMMOT-compatible AnnData slots. It is a package-native impact comparison, not official COMMOT DEG/tradeSeq."
        ),
    }
    (out / "commot_official_impact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def build_adata() -> ad.AnnData:
    counts = sparse.load_npz(PHASE10_DIR / "full_transcriptome/expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(PHASE10_DIR / "full_transcriptome/genes.csv")["gene"].astype(str).to_numpy()
    barcodes = pd.read_csv(PHASE10_DIR / "full_transcriptome/barcodes.csv")["cell_id"].astype(str).to_numpy()
    cells = pd.read_csv(DATASET_DIR / "cells.csv").set_index("cell_id").reindex(barcodes)
    adata = ad.AnnData(X=counts, obs=cells, var=pd.DataFrame(index=genes))
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.layers["counts"] = counts.copy()
    adata.raw = adata.copy()
    adata.obsm["spatial"] = cells[["x", "y"]].to_numpy(dtype=float)
    return adata


def select_background_genes(adata: ad.AnnData, ds_genes: list[str], n_genes: int) -> list[str]:
    """Pick a finite, high-variance background set before calling COMMOT.

    COMMOT's official function accepts either an integer, in which case it calls
    scanpy highly_variable_genes internally, or an explicit list. The explicit
    list keeps this comparison package-native while avoiding HVG binning failures
    caused by duplicate or extreme feature statistics in this matrix.
    """
    x = adata.X
    means = np.asarray(x.mean(axis=0)).ravel()
    sq_means = np.asarray(x.multiply(x).mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray((x * x).mean(axis=0)).ravel()
    vars_ = sq_means - means * means
    score = vars_ / (means + 1e-6)
    genes = np.asarray(adata.var_names.astype(str))
    excluded = set(ds_genes)
    ok = np.isfinite(score) & np.isfinite(means) & (means > 0)
    order = np.argsort(score[ok])[::-1]
    candidates = genes[ok][order]
    selected = [g for g in candidates if g not in excluded]
    return selected[: max(1, int(n_genes))]


def setup_commot_slots(adata: ad.AnnData, database_name: str, lr_pairs: pd.DataFrame, incoming: pd.DataFrame) -> None:
    ligrec = lr_pairs[["ligand", "receptor", "control_name"]].copy()
    ligrec.columns = [0, 1, 2]
    adata.uns[f"commot-{database_name}-info"] = {"df_ligrec": ligrec}
    receiver = pd.DataFrame(index=adata.obs_names)
    sender = pd.DataFrame(index=adata.obs_names)
    for _, row in lr_pairs.iterrows():
        pair = str(row["control_name"])
        key = f"{row['ligand']}-{row['receptor']}"
        col = pair if pair in incoming.columns else key
        vals = incoming[col].fillna(0.0).to_numpy(dtype=float) if col in incoming.columns else np.zeros(adata.n_obs)
        receiver[f"r-{key}"] = vals
        sender[f"s-{key}"] = vals
    receiver["r-total-total"] = receiver.sum(axis=1)
    sender["s-total-total"] = sender.sum(axis=1)
    adata.obsm[f"commot-{database_name}-sum-receiver"] = receiver
    adata.obsm[f"commot-{database_name}-sum-sender"] = sender


def summarize_impact(method_name: str, impact: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for lr_pair, targets in CURATED_TARGETS.items():
        row_name = f"r-{lr_pair}"
        if row_name not in impact.index:
            continue
        present = [g for g in targets if g in impact.columns]
        vals = pd.Series(dtype=float)
        if present:
            selected = impact.loc[[row_name], present].apply(pd.to_numeric, errors="coerce")
            vals = selected.mean(axis=0)
        rows.append(
            {
                "method": method_name,
                "lr_pair": lr_pair,
                "status": "tested",
                "n_targets_present": len(present),
                "mean_official_impact": float(vals.mean()) if len(vals) else np.nan,
                "median_official_impact": float(vals.median()) if len(vals) else np.nan,
                "max_official_impact": float(vals.max()) if len(vals) else np.nan,
                "best_target_gene": str(vals.sort_values(ascending=False).index[0]) if len(vals) else "",
                "best_target_impact": float(vals.max()) if len(vals) else np.nan,
            }
        )
    return rows


if __name__ == "__main__":
    main()
