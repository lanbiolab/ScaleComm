#!/usr/bin/env python
"""Standardize SFEData HeNSCLC CosMx data for the Fig. 5 scale/context story.

The R side exports a compact CosMx table from SFEData::HeNSCLCData(); Python
then assigns coarse cell states, selects a tumor/immune/stromal-enriched subset,
and builds LR candidate edges in the existing SpatialSelf benchmark schema.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/standardized_inputs/cosmx_nsclc_he_subset"
DEFAULT_R = Path("/home/xd/.conda/envs/cosmx_r/bin/Rscript")

GENES = [
    "ACTA2",
    "C1QA",
    "C1QB",
    "CCL5",
    "CCR5",
    "CD14",
    "CD274",
    "CD3D",
    "CD3E",
    "CD4",
    "CD44",
    "CD68",
    "CD8A",
    "COL1A1",
    "COL1A2",
    "CSF1",
    "CSF1R",
    "CXCL9",
    "CXCL10",
    "CXCL12",
    "CXCR3",
    "CXCR4",
    "EPCAM",
    "FN1",
    "IFNG",
    "KDR",
    "KRT8",
    "KRT19",
    "LYZ",
    "MMP2",
    "MMP9",
    "PECAM1",
    "PDCD1",
    "PTPRC",
    "SPP1",
    "TGFB1",
    "TGFBR2",
    "VEGFA",
    "VWF",
]

LR_DESIGN = [
    {
        "ligand": "CD274",
        "receptor": "PDCD1",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 42.0,
        "active_niche": "tumor_immune_interface",
        "control_class": "checkpoint_contact",
        "biological_category": "immune_checkpoint",
        "selected_reason": "local tumor-T checkpoint axis",
        "sender_types": {"tumor_proxy", "myeloid_proxy"},
        "receiver_types": {"t_cell_proxy"},
    },
    {
        "ligand": "CXCL9",
        "receptor": "CXCR3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 230.0,
        "active_niche": "immune_hotspot",
        "control_class": "chemokine_long_range",
        "biological_category": "chemokine",
        "selected_reason": "IFN-associated immune recruitment axis",
        "sender_types": {"myeloid_proxy", "immune_proxy", "tumor_proxy"},
        "receiver_types": {"t_cell_proxy", "immune_proxy"},
    },
    {
        "ligand": "CXCL10",
        "receptor": "CXCR3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 230.0,
        "active_niche": "immune_hotspot",
        "control_class": "chemokine_long_range",
        "biological_category": "chemokine",
        "selected_reason": "IFN-associated immune recruitment axis",
        "sender_types": {"myeloid_proxy", "immune_proxy", "tumor_proxy"},
        "receiver_types": {"t_cell_proxy", "immune_proxy"},
    },
    {
        "ligand": "CCL5",
        "receptor": "CCR5",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 135.0,
        "active_niche": "immune_hotspot",
        "control_class": "immune_paracrine",
        "biological_category": "chemokine",
        "selected_reason": "immune-local paracrine axis",
        "sender_types": {"t_cell_proxy", "immune_proxy"},
        "receiver_types": {"immune_proxy", "myeloid_proxy", "t_cell_proxy"},
    },
    {
        "ligand": "SPP1",
        "receptor": "CD44",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "tumor_stromal_interface",
        "control_class": "macrophage_matrix",
        "biological_category": "matrix_remodeling",
        "selected_reason": "SPP1 macrophage/stromal remodeling axis",
        "sender_types": {"myeloid_proxy", "tumor_proxy"},
        "receiver_types": {"stromal_proxy", "tumor_proxy", "myeloid_proxy"},
    },
    {
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 150.0,
        "active_niche": "tumor_stromal_interface",
        "control_class": "stromal_remodeling",
        "biological_category": "tgfb_ecm",
        "selected_reason": "TGF-beta stromal remodeling axis",
        "sender_types": {"myeloid_proxy", "stromal_proxy", "tumor_proxy"},
        "receiver_types": {"stromal_proxy", "tumor_proxy"},
    },
    {
        "ligand": "CSF1",
        "receptor": "CSF1R",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 150.0,
        "active_niche": "tumor_myeloid_interface",
        "control_class": "tumor_myeloid",
        "biological_category": "growth_factor",
        "selected_reason": "tumor-myeloid support axis",
        "sender_types": {"tumor_proxy", "stromal_proxy"},
        "receiver_types": {"myeloid_proxy"},
    },
    {
        "ligand": "VEGFA",
        "receptor": "KDR",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 210.0,
        "active_niche": "vascular_stromal_interface",
        "control_class": "vascular",
        "biological_category": "angiogenesis",
        "selected_reason": "vascular niche signaling axis",
        "sender_types": {"tumor_proxy", "stromal_proxy", "myeloid_proxy"},
        "receiver_types": {"endothelial_proxy"},
    },
    {
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 240.0,
        "active_niche": "stromal_immune_interface",
        "control_class": "stromal_chemokine",
        "biological_category": "chemokine",
        "selected_reason": "stromal-to-immune chemokine axis",
        "sender_types": {"stromal_proxy"},
        "receiver_types": {"immune_proxy", "t_cell_proxy", "tumor_proxy"},
    },
    {
        "ligand": "KRT8",
        "receptor": "ACTA2",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 80.0,
        "active_niche": "any",
        "control_class": "high_expression_decoy",
        "biological_category": "high_expression_negative_risk",
        "selected_reason": "expressed non-LR decoy",
        "sender_types": {"tumor_proxy"},
        "receiver_types": {"stromal_proxy"},
    },
    {
        "ligand": "PTPRC",
        "receptor": "EPCAM",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 80.0,
        "active_niche": "any",
        "control_class": "high_expression_decoy",
        "biological_category": "high_expression_negative_risk",
        "selected_reason": "expressed non-LR decoy",
        "sender_types": {"immune_proxy", "t_cell_proxy", "myeloid_proxy"},
        "receiver_types": {"tumor_proxy"},
    },
]

TARGET_SIGNATURES = {
    ("CD274", "PDCD1"): ["PDCD1", "CD3D", "CD3E", "CD8A", "IFNG", "CXCR3"],
    ("CXCL9", "CXCR3"): ["CXCR3", "CD3D", "CD3E", "CD8A", "IFNG", "PTPRC"],
    ("CXCL10", "CXCR3"): ["CXCR3", "CD3D", "CD3E", "CD8A", "IFNG", "PTPRC"],
    ("CCL5", "CCR5"): ["CCR5", "PTPRC", "CD3D", "CD68", "LYZ", "IFNG"],
    ("SPP1", "CD44"): ["CD44", "MMP9", "MMP2", "FN1", "COL1A1", "TGFB1"],
    ("TGFB1", "TGFBR2"): ["TGFBR2", "COL1A1", "COL1A2", "ACTA2", "FN1", "MMP2"],
    ("CSF1", "CSF1R"): ["CSF1R", "CD68", "CD14", "LYZ", "C1QA", "C1QB"],
    ("VEGFA", "KDR"): ["KDR", "PECAM1", "VWF", "MMP9", "CXCL12"],
    ("CXCL12", "CXCR4"): ["CXCR4", "CXCL12", "PTPRC", "CD3D", "CD44", "TGFB1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--rscript", default=str(DEFAULT_R))
    parser.add_argument("--subset-size", type=int, default=9000)
    parser.add_argument("--max-neighbors", type=int, default=96)
    parser.add_argument("--max-candidates-per-lr", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--keep-r-export", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_r_export"
    if tmp_dir.exists() and not args.keep_r_export:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    export_cosmx_tables(Path(args.rscript), tmp_dir)

    cells_raw = pd.read_csv(tmp_dir / "cosmx_cells_raw.csv")
    expression_raw = pd.read_csv(tmp_dir / "cosmx_expression_raw.csv")
    genes_present = [g for g in GENES if g in expression_raw.columns]
    expression_raw = expression_raw[["cell_id", *genes_present]].copy()

    cells = annotate_cells(cells_raw, expression_raw)
    selected_ids = select_subset(cells, expression_raw, int(args.subset_size), int(args.seed))
    cells = cells[cells["cell_id"].isin(selected_ids)].copy().reset_index(drop=True)
    expression = expression_raw[expression_raw["cell_id"].isin(selected_ids)].copy()
    expression = expression.set_index("cell_id").loc[cells["cell_id"]].reset_index()

    lr_pairs = build_lr_pairs(expression.columns)
    candidates = build_candidates(
        cells=cells,
        expression=expression,
        lr_pairs=lr_pairs,
        max_neighbors=int(args.max_neighbors),
        max_candidates_per_lr=int(args.max_candidates_per_lr),
        seed=int(args.seed),
    )
    target_signatures = build_target_signatures(expression.columns)

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)
    target_signatures.to_csv(output_dir / "target_signatures.csv", index=False)

    manifest = {
        "dataset": "SFEData::HeNSCLCData / NanoString FFPE CosMx human NSCLC",
        "platform": "CosMx SMI",
        "source_object": "ExperimentHub EH7743 via SFEData::HeNSCLCData",
        "n_cells_raw": int(len(cells_raw)),
        "n_cells_subset": int(len(cells)),
        "n_genes_exported": int(len(expression.columns) - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidates": int(len(candidates)),
        "subset_strategy": "tumor/immune/stromal/interface-enriched coarse cell-state subset",
        "claim_boundary": "Coarse cell-state annotations are proxy labels from CosMx protein markers and marker genes.",
        "files": {
            "cells": rel(output_dir / "cells.csv"),
            "expression": rel(output_dir / "expression.csv"),
            "lr_pairs": rel(output_dir / "lr_pairs.csv"),
            "candidate_interactions": rel(output_dir / "candidate_interactions.csv"),
            "target_signatures": rel(output_dir / "target_signatures.csv"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def export_cosmx_tables(rscript: Path, out_dir: Path) -> None:
    cells_path = out_dir / "cosmx_cells_raw.csv"
    expr_path = out_dir / "cosmx_expression_raw.csv"
    if cells_path.exists() and expr_path.exists():
        return
    if not rscript.exists():
        raise FileNotFoundError(f"Rscript not found: {rscript}")
    gene_vec = ",".join(f'"{g}"' for g in GENES)
    r_code = f"""
suppressPackageStartupMessages({{
  library(SFEData)
  library(SummarizedExperiment)
  library(SpatialExperiment)
}})
sfe <- HeNSCLCData(verbose=FALSE)
genes <- c({gene_vec})
genes <- intersect(genes, rownames(sfe))
counts <- as.matrix(assay(sfe, "counts")[genes, , drop=FALSE])
coords <- as.data.frame(spatialCoords(sfe))
cd <- as.data.frame(colData(sfe))
cells <- data.frame(
  cell_id = colnames(sfe),
  x = coords[, 1],
  y = coords[, 2],
  Area = cd$Area,
  AspectRatio = cd$AspectRatio,
  Width = cd$Width,
  Height = cd$Height,
  Mean_PanCK = cd$Mean.PanCK,
  Mean_CD45 = cd$Mean.CD45,
  Mean_CD3 = cd$Mean.CD3,
  nCounts = cd$nCounts,
  nGenes = cd$nGenes,
  sample_id = cd$sample_id,
  check.names = FALSE
)
expr <- as.data.frame(t(counts), check.names=FALSE)
expr <- data.frame(cell_id = colnames(sfe), expr, check.names=FALSE)
write.csv(cells, "{cells_path}", row.names=FALSE, quote=TRUE)
write.csv(expr, "{expr_path}", row.names=FALSE, quote=TRUE)
"""
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    env["PATH"] = f"{rscript.parent}:{env.get('PATH', '')}"
    env["PKG_CONFIG_PATH"] = f"{rscript.parent.parent / 'lib/pkgconfig'}:{env.get('PKG_CONFIG_PATH', '')}"
    subprocess.run([str(rscript), "-e", r_code], cwd=PROJECT_ROOT, env=env, check=True)


def annotate_cells(cells: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    expr = expression.set_index("cell_id")
    idx = cells["cell_id"]
    panck = rank01(np.log1p(cells["Mean_PanCK"].to_numpy(float)))
    cd45 = rank01(np.log1p(cells["Mean_CD45"].to_numpy(float)))
    cd3p = rank01(np.log1p(cells["Mean_CD3"].to_numpy(float)))
    tumor = rank01(gene_score(expr, idx, ["EPCAM", "KRT8", "KRT19"])) + 0.85 * panck
    immune = rank01(gene_score(expr, idx, ["PTPRC"])) + 0.55 * cd45
    tcell = rank01(gene_score(expr, idx, ["CD3D", "CD3E", "CD4", "CD8A", "PDCD1", "CXCR3"])) + 0.90 * cd3p
    myeloid = rank01(gene_score(expr, idx, ["CD68", "CD14", "LYZ", "C1QA", "C1QB", "CSF1R"]))
    stromal = rank01(gene_score(expr, idx, ["COL1A1", "COL1A2", "ACTA2", "FN1", "CXCL12"]))
    endothelial = rank01(gene_score(expr, idx, ["PECAM1", "VWF", "KDR"]))

    scores = np.vstack([tumor, tcell, myeloid, stromal, endothelial, immune]).T
    labels = np.array(
        [
            "tumor_proxy",
            "t_cell_proxy",
            "myeloid_proxy",
            "stromal_proxy",
            "endothelial_proxy",
            "immune_proxy",
        ]
    )
    best = labels[np.argmax(scores, axis=1)]
    best_score = scores.max(axis=1)
    best[best_score < 0.20] = "other_proxy"
    best[(best == "immune_proxy") & (tcell > 0.55)] = "t_cell_proxy"
    best[(best == "immune_proxy") & (myeloid > 0.55)] = "myeloid_proxy"
    cells["cell_type"] = best
    cells["region"] = coarse_region(cells)
    cells["niche"] = assign_niche(cells, tumor, tcell, myeloid, stromal, endothelial, immune)
    cells["deconvolution_label1"] = cells["cell_type"]
    cells["deconvolution_label2"] = cells["niche"]
    cells["unsupervised_l1"] = cells["region"]
    cells["total_counts"] = cells["nCounts"]
    for name, values in {
        "tumor_score": tumor,
        "immune_score": immune,
        "t_cell_score": tcell,
        "myeloid_score": myeloid,
        "stromal_score": stromal,
        "endothelial_score": endothelial,
    }.items():
        cells[name] = values
    keep_cols = [
        "cell_id",
        "x",
        "y",
        "cell_type",
        "region",
        "niche",
        "deconvolution_label1",
        "deconvolution_label2",
        "unsupervised_l1",
        "total_counts",
        "Area",
        "Mean_PanCK",
        "Mean_CD45",
        "Mean_CD3",
        "nGenes",
        "tumor_score",
        "immune_score",
        "t_cell_score",
        "myeloid_score",
        "stromal_score",
        "endothelial_score",
    ]
    return cells[keep_cols].copy()


def coarse_region(cells: pd.DataFrame) -> pd.Series:
    region = np.full(len(cells), "mixed_region", dtype=object)
    region[cells["cell_type"].eq("tumor_proxy").to_numpy()] = "tumor_enriched"
    region[cells["cell_type"].isin(["t_cell_proxy", "immune_proxy", "myeloid_proxy"]).to_numpy()] = "immune_enriched"
    region[cells["cell_type"].isin(["stromal_proxy", "endothelial_proxy"]).to_numpy()] = "stromal_vascular"
    return pd.Series(region, index=cells.index)


def assign_niche(
    cells: pd.DataFrame,
    tumor: np.ndarray,
    tcell: np.ndarray,
    myeloid: np.ndarray,
    stromal: np.ndarray,
    endothelial: np.ndarray,
    immune: np.ndarray,
) -> pd.Series:
    coords = cells[["x", "y"]].to_numpy(float)
    tree = cKDTree(coords)
    k = min(32, len(cells))
    _, neigh = tree.query(coords, k=k)
    if neigh.ndim == 1:
        neigh = neigh[:, None]
    local_tumor = tumor[neigh].mean(axis=1)
    local_tcell = tcell[neigh].mean(axis=1)
    local_myeloid = myeloid[neigh].mean(axis=1)
    local_stromal = stromal[neigh].mean(axis=1)
    local_endothelial = endothelial[neigh].mean(axis=1)
    local_immune = immune[neigh].mean(axis=1)
    niche = np.full(len(cells), "mixed_niche", dtype=object)
    niche[(local_tumor > 0.50) & (local_immune < 0.35) & (local_stromal < 0.35)] = "tumor_core"
    niche[(local_immune > 0.46) & (local_tumor < 0.42)] = "immune_hotspot"
    niche[(local_tumor > 0.38) & (local_immune > 0.36)] = "tumor_immune_interface"
    niche[(local_tumor > 0.34) & (local_stromal > 0.36)] = "tumor_stromal_interface"
    niche[(local_tumor > 0.34) & (local_myeloid > 0.35)] = "tumor_myeloid_interface"
    niche[(local_stromal > 0.42) & (local_immune > 0.34)] = "stromal_immune_interface"
    niche[(local_endothelial > 0.42) | ((local_stromal > 0.35) & (local_endothelial > 0.30))] = (
        "vascular_stromal_interface"
    )
    return pd.Series(niche, index=cells.index)


def select_subset(cells: pd.DataFrame, expression: pd.DataFrame, subset_size: int, seed: int) -> set[str]:
    if len(cells) <= subset_size:
        return set(cells["cell_id"])
    rng = np.random.default_rng(seed)
    expr = expression.set_index("cell_id")
    priority_genes = ["CD274", "PDCD1", "CXCL9", "CXCL10", "CXCR3", "SPP1", "CD44", "TGFB1", "TGFBR2", "CXCL12", "CXCR4"]
    expr_score = rank01(gene_score(expr, cells["cell_id"], priority_genes))
    niche_priority = cells["niche"].isin(
        [
            "tumor_immune_interface",
            "immune_hotspot",
            "tumor_stromal_interface",
            "tumor_myeloid_interface",
            "stromal_immune_interface",
            "vascular_stromal_interface",
        ]
    ).to_numpy(float)
    type_priority = cells["cell_type"].isin(
        ["tumor_proxy", "t_cell_proxy", "myeloid_proxy", "stromal_proxy", "endothelial_proxy", "immune_proxy"]
    ).to_numpy(float)
    priority = 0.50 * expr_score + 0.30 * niche_priority + 0.20 * type_priority
    keep = []
    # Keep broad coverage across cell types/niches before filling by priority.
    for _, group in cells.groupby("cell_type", sort=False):
        n = min(max(subset_size // 24, 120), len(group))
        local_priority = priority[group.index.to_numpy()]
        order = np.argsort(-local_priority)[:n]
        keep.extend(group.iloc[order]["cell_id"].tolist())
    for _, group in cells.groupby("niche", sort=False):
        n = min(max(subset_size // 30, 80), len(group))
        local_priority = priority[group.index.to_numpy()]
        order = np.argsort(-local_priority)[:n]
        keep.extend(group.iloc[order]["cell_id"].tolist())
    keep = list(dict.fromkeys(keep))
    remaining = subset_size - len(keep)
    if remaining > 0:
        used = set(keep)
        pool = cells[~cells["cell_id"].isin(used)].copy()
        pool_priority = priority[pool.index.to_numpy()]
        # Mostly priority-driven, with a small random component for spatial coverage.
        score = 0.90 * rank01(pool_priority) + 0.10 * rng.random(len(pool))
        order = np.argsort(-score)[:remaining]
        keep.extend(pool.iloc[order]["cell_id"].tolist())
    if len(keep) > subset_size:
        keep_df = cells[cells["cell_id"].isin(keep)].copy()
        keep_priority = priority[keep_df.index.to_numpy()]
        order = np.argsort(-keep_priority)[:subset_size]
        keep = keep_df.iloc[order]["cell_id"].tolist()
    return set(keep)


def build_lr_pairs(expression_columns: pd.Index) -> pd.DataFrame:
    genes = set(expression_columns) - {"cell_id"}
    rows = []
    for i, spec in enumerate(LR_DESIGN):
        if spec["ligand"] not in genes or spec["receptor"] not in genes:
            continue
        rows.append(
            {
                "lr_id": f"cosmx_nsclc_lr_{i:03d}_{spec['ligand']}_{spec['receptor']}",
                "ligand": spec["ligand"],
                "receptor": spec["receptor"],
                "mechanism": spec["mechanism"],
                "spatial_range": float(spec["spatial_range"]),
                "active_niche": spec["active_niche"],
                "control_name": f"{spec['ligand']}-{spec['receptor']}",
                "control_class": spec["control_class"],
                "biological_category": spec["biological_category"],
                "selected_reason": spec["selected_reason"],
                "false_positive_high_expression": spec["control_class"] == "high_expression_decoy",
                "true_lr": spec["control_class"] != "high_expression_decoy",
            }
        )
    return pd.DataFrame(rows)


def build_candidates(
    *,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expr = expression.set_index("cell_id")
    coords = cells[["x", "y"]].to_numpy(float)
    tree = cKDTree(coords)
    max_range = max(float(lr_pairs["spatial_range"].max()), 80.0)
    query_radius = max(max_range * 1.05, 120.0)
    neighbor_lists = tree.query_ball_point(coords, r=query_radius)
    cell_index = pd.Series(np.arange(len(cells)), index=cells["cell_id"])
    rows = []
    cid = 0
    for lr in lr_pairs.itertuples(index=False):
        spec = next(
            item for item in LR_DESIGN if item["ligand"] == lr.ligand and item["receptor"] == lr.receptor
        )
        lr_rows = []
        ligand_values = expr[lr.ligand].to_numpy(float)
        receptor_values = expr[lr.receptor].to_numpy(float)
        candidate_sender = cells["cell_type"].isin(spec["sender_types"]).to_numpy()
        if not candidate_sender.any():
            candidate_sender = ligand_values > 0
        sender_idx = np.flatnonzero(candidate_sender | (ligand_values > 0))
        for sidx in sender_idx:
            neigh = [j for j in neighbor_lists[sidx] if j != sidx]
            if not neigh:
                continue
            neigh = np.asarray(neigh, dtype=int)
            dist = np.sqrt(((coords[neigh] - coords[sidx]) ** 2).sum(axis=1))
            # Keep closest neighbors and receptor-positive/context-relevant receivers.
            priority = receptor_values[neigh] > 0
            priority |= cells.iloc[neigh]["cell_type"].isin(spec["receiver_types"]).to_numpy()
            order = np.lexsort((dist, ~priority))
            neigh = neigh[order[:max_neighbors]]
            dist = dist[order[:max_neighbors]]
            for ridx, d in zip(neigh, dist, strict=False):
                sender = cells.iloc[sidx]
                receiver = cells.iloc[ridx]
                ligand_expr = float(ligand_values[sidx])
                receptor_expr = float(receptor_values[ridx])
                type_ok = bool(
                    sender["cell_type"] in spec["sender_types"] and receiver["cell_type"] in spec["receiver_types"]
                )
                range_ok = range_rule(str(lr.mechanism), float(d), float(lr.spatial_range))
                niche_ok = niche_rule(str(lr.active_niche), str(sender["niche"]), str(receiver["niche"]))
                lr_rows.append(
                    {
                        "candidate_id": f"cosmx_nsclc_cand_{cid:08d}",
                        "sender_id": sender["cell_id"],
                        "receiver_id": receiver["cell_id"],
                        "lr_id": lr.lr_id,
                        "ligand": lr.ligand,
                        "receptor": lr.receptor,
                        "mechanism": lr.mechanism,
                        "distance": float(d),
                        "sender_cell_type": sender["cell_type"],
                        "receiver_cell_type": receiver["cell_type"],
                        "sender_niche": sender["niche"],
                        "receiver_niche": receiver["niche"],
                        "weak_signal": False,
                        "false_positive_high_expression": bool(lr.false_positive_high_expression),
                        "type_rule_satisfied": type_ok,
                        "range_rule_satisfied": range_ok,
                        "niche_rule_satisfied": niche_ok,
                        "ligand_expr": ligand_expr,
                        "receptor_expr": receptor_expr,
                        "coexpression_distance_score": coexpression_score(ligand_expr, receptor_expr, float(d), float(lr.spatial_range)),
                    }
                )
                cid += 1
        if len(lr_rows) > max_candidates_per_lr:
            dat = pd.DataFrame(lr_rows)
            expr_prod = np.log1p(dat["ligand_expr"].to_numpy(float)) * np.log1p(dat["receptor_expr"].to_numpy(float))
            rule = (
                dat["type_rule_satisfied"].astype(bool).to_numpy()
                & dat["range_rule_satisfied"].astype(bool).to_numpy()
                & dat["niche_rule_satisfied"].astype(bool).to_numpy()
            )
            score = 0.55 * rank01(expr_prod) + 0.25 * rule.astype(float) + 0.20 * rank01(-dat["distance"].to_numpy(float))
            keep = np.argsort(-score)[: max_candidates_per_lr]
            random_keep = rng.choice(np.arange(len(dat)), size=min(max_candidates_per_lr // 10, len(dat)), replace=False)
            keep = np.unique(np.concatenate([keep, random_keep]))[:max_candidates_per_lr]
            lr_rows = dat.iloc[keep].to_dict("records")
        rows.extend(lr_rows)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No candidate interactions were generated.")
    out["candidate_id"] = [f"cosmx_nsclc_cand_{i:08d}" for i in range(len(out))]
    return out


def build_target_signatures(expression_columns: pd.Index) -> pd.DataFrame:
    genes = set(expression_columns) - {"cell_id"}
    rows = []
    for (ligand, receptor), targets in TARGET_SIGNATURES.items():
        present = [gene for gene in targets if gene in genes]
        if ligand not in genes or receptor not in genes or len(present) < 3:
            continue
        rows.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "target_genes": ",".join(present),
                "targets": ",".join(present),
                "target_source": "cosmx_nsclc_curated_marker_response_proxy",
                "evidence_note": "Curated NSCLC/TME response proxy using genes present in the CosMx panel.",
            }
        )
    return pd.DataFrame(rows)


def gene_score(expr: pd.DataFrame, cell_ids: pd.Series, genes: list[str]) -> np.ndarray:
    present = [gene for gene in genes if gene in expr.columns]
    if not present:
        return np.zeros(len(cell_ids), dtype=float)
    values = np.log1p(expr.loc[cell_ids, present].to_numpy(float))
    return values.mean(axis=1)


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def range_rule(mechanism: str, distance: float, spatial_range: float) -> bool:
    if mechanism == "juxtacrine_contact":
        return distance <= spatial_range
    if mechanism == "negative_control_high_expression":
        return False
    if mechanism == "long_range_chemokine_niche":
        return 0.20 * spatial_range <= distance <= spatial_range
    return distance <= spatial_range


def niche_rule(active_niche: str, sender_niche: str, receiver_niche: str) -> bool:
    if active_niche in {"any", "", "nan"}:
        return True
    if active_niche == "immune_hotspot":
        return sender_niche == active_niche or receiver_niche == active_niche
    if active_niche == "tumor_immune_interface":
        return sender_niche == active_niche or receiver_niche == active_niche or {
            sender_niche,
            receiver_niche,
        } == {"tumor_core", "immune_hotspot"}
    if active_niche == "tumor_stromal_interface":
        return sender_niche == active_niche or receiver_niche == active_niche
    if active_niche == "tumor_myeloid_interface":
        return sender_niche == active_niche or receiver_niche == active_niche
    if active_niche == "stromal_immune_interface":
        return sender_niche == active_niche or receiver_niche == active_niche
    if active_niche == "vascular_stromal_interface":
        return sender_niche == active_niche or receiver_niche == active_niche
    return sender_niche == active_niche and receiver_niche == active_niche


def coexpression_score(ligand_expr: float, receptor_expr: float, distance: float, spatial_range: float) -> float:
    expr = np.log1p(max(ligand_expr, 0.0)) * np.log1p(max(receptor_expr, 0.0))
    bandwidth = max(spatial_range, 1.0)
    return float(expr * np.exp(-distance / bandwidth))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
