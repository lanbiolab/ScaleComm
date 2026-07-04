#!/usr/bin/env python
"""Standardize a Visium HD CRC P2CRC ROI for Phase 7 Goal 5.

The source is GEO GSM8594568, a 10x Visium HD human colon cancer section with
8um bins and published deconvolution/annotation metadata. The output follows
the Phase 7 real-platform schema. It is a real annotated high-resolution ROI,
not a synthetic benchmark and not native CCC ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PREFIX = "GSM8594568_P2CRC"


LR_PANEL = [
    {
        "lr_id": "visium_hd_crc_lr_000_SPP1_CD44",
        "ligand": "SPP1",
        "receptor": "CD44",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 96.0,
        "active_niche": "tumor_myeloid_boundary",
        "control_name": "SPP1-CD44",
        "control_class": "crc_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "visium_hd_crc_lr_001_CXCL12_CXCR4",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 160.0,
        "active_niche": "tumor_stromal_boundary",
        "control_name": "CXCL12-CXCR4",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "visium_hd_crc_lr_002_VEGFA_KDR",
        "ligand": "VEGFA",
        "receptor": "KDR",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 96.0,
        "active_niche": "vascular_stromal_boundary",
        "control_name": "VEGFA-KDR",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "visium_hd_crc_lr_003_TGFB1_TGFBR2",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 128.0,
        "active_niche": "tumor_stromal_boundary",
        "control_name": "TGFB1-TGFBR2",
        "control_class": "crc_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "visium_hd_crc_lr_004_CD274_PDCD1",
        "ligand": "CD274",
        "receptor": "PDCD1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 80.0,
        "active_niche": "tumor_immune_boundary",
        "control_name": "CD274-PDCD1",
        "control_class": "immune_checkpoint_context_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "visium_hd_crc_lr_005_ACTB_KRT8_negative",
        "ligand": "ACTB",
        "receptor": "KRT8",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 160.0,
        "active_niche": "any",
        "control_name": "ACTB-KRT8",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "visium_hd_crc_lr_006_COL1A1_EPCAM_negative",
        "ligand": "COL1A1",
        "receptor": "EPCAM",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 160.0,
        "active_niche": "any",
        "control_name": "COL1A1-EPCAM",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
]


TARGET_GENES = {
    "SPP1-CD44": ["MMP9", "ITGAV", "ITGB1", "FN1", "LGALS3", "CXCL8", "CD44"],
    "CXCL12-CXCR4": ["CXCR4", "CXCL12", "JUN", "FOS", "PTPRC", "IL7R", "CCR7"],
    "VEGFA-KDR": ["KDR", "FLT1", "PECAM1", "VWF", "ENG", "ESAM", "CDH5"],
    "TGFB1-TGFBR2": ["TGFBR2", "SERPINE1", "COL1A1", "COL1A2", "ACTA2", "TAGLN", "FN1"],
    "CD274-PDCD1": ["PDCD1", "CD3D", "CD3E", "CD8A", "LAG3", "HAVCR2", "TOX"],
    "ACTB-KRT8": ["KRT8", "KRT18", "EPCAM", "ACTB"],
    "COL1A1-EPCAM": ["EPCAM", "KRT8", "KRT18", "COL1A1"],
}


RICH_TARGET_GENES = {
    "SPP1-CD44": [
        "MMP9",
        "MMP2",
        "ITGAV",
        "ITGB1",
        "FN1",
        "LGALS3",
        "CXCL8",
        "IL1B",
        "NFKBIA",
        "TNFAIP3",
        "JUN",
        "FOS",
        "CD44",
    ],
    "CXCL12-CXCR4": [
        "CXCR4",
        "CXCL12",
        "JUN",
        "FOS",
        "PTPRC",
        "IL7R",
        "CCR7",
        "LTB",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "STAT1",
    ],
    "VEGFA-KDR": [
        "KDR",
        "FLT1",
        "PECAM1",
        "VWF",
        "ENG",
        "ESAM",
        "CDH5",
        "RAMP2",
        "EMCN",
        "PLVAP",
        "CLDN5",
        "SELE",
        "ICAM1",
    ],
    "TGFB1-TGFBR2": [
        "TGFBR2",
        "SERPINE1",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL6A1",
        "ACTA2",
        "TAGLN",
        "FN1",
        "CTGF",
        "MMP2",
        "MMP9",
        "JUN",
        "FOS",
        "SMAD7",
    ],
    "CD274-PDCD1": [
        "PDCD1",
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",
        "LAG3",
        "HAVCR2",
        "TOX",
        "GZMB",
        "IFNG",
        "CXCL9",
        "CXCL10",
        "STAT1",
        "IRF1",
        "NKG7",
    ],
    "ACTB-KRT8": ["KRT8", "KRT18", "KRT19", "EPCAM", "MUC1", "ACTB"],
    "COL1A1-EPCAM": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "COL1A1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/optional/visium_hd_crc/raw/GSM8594568_P2CRC",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
            "standardized_inputs/visium_hd_crc_p2crc_subset"
        ),
    )
    parser.add_argument("--n-spots", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--max-neighbors", type=int, default=28)
    parser.add_argument("--max-candidates-per-lr", type=int, default=120000)
    parser.add_argument("--dataset-id", default="visium_hd_crc_p2crc_subset")
    parser.add_argument("--target-profile", choices=("compact", "rich"), default="compact")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = raw_dir / f"{RAW_PREFIX}_filtered_feature_bc_matrix.h5"
    metadata_path = raw_dir / f"extracted/{RAW_PREFIX}_Metadata.parquet"
    positions_path = raw_dir / f"extracted/{RAW_PREFIX}_tissue_positions.parquet"
    _require_files([matrix_path, metadata_path, positions_path])

    bundle = _read_10x_h5(matrix_path)
    lr_pairs = pd.DataFrame(LR_PANEL)
    target_genes = RICH_TARGET_GENES if args.target_profile == "rich" else TARGET_GENES
    genes_needed = sorted(
        set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | _marker_genes() | _target_gene_set(target_genes)
    )
    genes_present = [gene for gene in genes_needed if gene in bundle["gene_to_idx"]]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR panel contains genes absent from Visium HD CRC matrix: {missing_lr_genes}")

    metadata = pd.read_parquet(metadata_path)
    positions = pd.read_parquet(positions_path)
    annotated = _annotated_positions(metadata, positions, bundle)
    selected = _select_roi(annotated, n_spots=int(args.n_spots), seed=int(args.seed))
    expression = _expression_table(bundle, selected["barcode"].to_list(), genes_present)
    cells = _cells_table(selected, expression)
    candidates = _candidate_interactions(
        cells,
        expression,
        lr_pairs,
        max_neighbors=int(args.max_neighbors),
        max_candidates_per_lr=int(args.max_candidates_per_lr),
        seed=int(args.seed),
    )
    target_signatures = _target_signature_table(lr_pairs, genes_present, target_genes)

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)
    target_signatures.to_csv(output_dir / "target_signatures.csv", index=False)

    dataset_metadata = {
        "dataset_id": str(args.dataset_id),
        "platform": "10x Visium HD human colon cancer P2CRC 8um bins",
        "source": "GEO GSM8594568 P2CRC supplemental Visium HD files",
        "raw_dir": _rel(raw_dir),
        "output_dir": _rel(output_dir),
        "n_spots": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "n_spots_requested": int(args.n_spots),
            "seed": int(args.seed),
            "rule": (
                "balanced ROI from real deconvolution labels: tumor, CAF/fibroblast, macrophage/SPP1+ macrophage, "
                "T cells, endothelial, epithelial/goblet/enterocyte, plus boundary-enriched bins"
            ),
        },
        "label_source": (
            "GEO metadata columns DeconvolutionLabel1/2, UnsupervisedL1, Periphery, MacrophageSubtype, "
            "and GobletSubcluster; these are external annotations/deconvolution labels, not SpatialSelf-generated labels."
        ),
        "claim_boundary": (
            "Real 10x Visium HD coordinates, expression, and published deconvolution metadata are preserved. Positive "
            "controls are literature/context priors and negative controls are deliberately high-expression invalid pairs. "
            "This is a real annotated cross-platform validation dataset, not native perturbation CCC ground truth."
        ),
        "source_files": {
            "matrix_h5": _rel(matrix_path),
            "metadata_parquet": _rel(metadata_path),
            "positions_parquet": _rel(positions_path),
        },
        "missing_lr_genes": missing_lr_genes,
        "target_profile": str(args.target_profile),
    }
    (output_dir / "metadata.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": dataset_metadata["dataset_id"],
                "platform": dataset_metadata["platform"],
                "n_cells": dataset_metadata["n_spots"],
                "n_expression_genes": dataset_metadata["n_expression_genes"],
                "n_lr_pairs": dataset_metadata["n_lr_pairs"],
                "n_candidate_interactions": dataset_metadata["n_candidate_interactions"],
                "output_dir": dataset_metadata["output_dir"],
                "claim_boundary": dataset_metadata["claim_boundary"],
            }
        ]
    ).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(dataset_metadata, indent=2))


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required Visium HD CRC files: " + ";".join(missing))


def _read_10x_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        genes = [item.decode() for item in handle["matrix/features/name"][:]]
        barcodes = [item.decode() for item in handle["matrix/barcodes"][:]]
        data = handle["matrix/data"][:]
        indices = handle["matrix/indices"][:]
        indptr = handle["matrix/indptr"][:]
        shape = tuple(int(x) for x in handle["matrix/shape"][:])
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return {
        "genes": genes,
        "gene_to_idx": {gene: idx for idx, gene in enumerate(genes)},
        "barcodes": barcodes,
        "barcode_to_idx": {barcode: idx for idx, barcode in enumerate(barcodes)},
        "matrix": matrix,
    }


def _annotated_positions(metadata: pd.DataFrame, positions: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "barcode",
        "tissue",
        "X",
        "Y",
        "DeconvolutionClass",
        "DeconvolutionLabel1",
        "DeconvolutionLabel2",
        "Periphery",
        "UnsupervisedL1",
        "UnsupervisedL2",
        "MacrophageSubtype",
        "GobletSubcluster",
    ]
    meta = metadata[[column for column in columns if column in metadata.columns]].copy()
    pos = positions[["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]].copy()
    table = meta.merge(pos, on="barcode", how="inner", validate="one_to_one")
    table = table[(table["in_tissue"] == 1) & (table["barcode"].isin(bundle["barcode_to_idx"]))].copy()
    table["cell_type"] = table.apply(_cell_type_from_metadata, axis=1)
    table["niche"] = table.apply(_niche_from_metadata, axis=1)
    table["region"] = table["niche"]
    table["is_boundary"] = table["Periphery"].astype(str).isin(["50 micron", "Tumor"])
    return table.reset_index(drop=True)


def _select_roi(table: pd.DataFrame, *, n_spots: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_spots = min(int(n_spots), len(table))
    selected: list[int] = []
    targets = {
        "tumor_proxy": 1200,
        "caf_proxy": 900,
        "myeloid_proxy": 800,
        "spp1_macrophage_proxy": 500,
        "T_cell_proxy": 650,
        "endothelial_proxy": 450,
        "epithelial_proxy": 450,
    }
    for label, target in targets.items():
        idx = np.flatnonzero(table["cell_type"].astype(str).to_numpy() == label)
        if len(idx) == 0:
            continue
        boundary = idx[table.iloc[idx]["is_boundary"].to_numpy(dtype=bool)]
        pool = boundary if len(boundary) >= min(target, 25) else idx
        take = min(int(target), len(pool), n_spots - len(selected))
        if take <= 0:
            break
        selected.extend(int(i) for i in rng.choice(pool, size=take, replace=False))
    remaining = n_spots - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        boundary_pool = np.array(
            [idx for idx in np.flatnonzero(table["is_boundary"].to_numpy(dtype=bool)) if int(idx) not in selected_set],
            dtype=int,
        )
        all_pool = np.array([idx for idx in range(len(table)) if int(idx) not in selected_set], dtype=int)
        pool = boundary_pool if len(boundary_pool) >= remaining else all_pool
        selected.extend(int(i) for i in rng.choice(pool, size=min(remaining, len(pool)), replace=False))
    selected = np.sort(np.array(selected[:n_spots], dtype=int))
    return table.iloc[selected].reset_index(drop=True)


def _expression_table(bundle: dict[str, Any], barcodes: list[str], genes: list[str]) -> pd.DataFrame:
    spot_idx = [bundle["barcode_to_idx"][barcode] for barcode in barcodes]
    gene_idx = [bundle["gene_to_idx"][gene] for gene in genes]
    sub = bundle["matrix"][gene_idx, :][:, spot_idx].T.toarray()
    frame = pd.DataFrame(sub, columns=genes)
    frame.insert(0, "cell_id", barcodes)
    return frame


def _cells_table(selected: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    expr = expression.set_index("cell_id")
    library = expr.sum(axis=1)
    return pd.DataFrame(
        {
            "cell_id": selected["barcode"].to_numpy(),
            "x": selected["pxl_col_in_fullres"].to_numpy(dtype=float),
            "y": selected["pxl_row_in_fullres"].to_numpy(dtype=float),
            "cell_type": selected["cell_type"].to_numpy(),
            "region": selected["region"].to_numpy(),
            "niche": selected["niche"].to_numpy(),
            "deconvolution_label1": selected.get("DeconvolutionLabel1", pd.Series([""] * len(selected))).fillna("").to_numpy(),
            "deconvolution_label2": selected.get("DeconvolutionLabel2", pd.Series([""] * len(selected))).fillna("").to_numpy(),
            "unsupervised_l1": selected.get("UnsupervisedL1", pd.Series([""] * len(selected))).fillna("").to_numpy(),
            "periphery": selected.get("Periphery", pd.Series([""] * len(selected))).fillna("").to_numpy(),
            "macrophage_subtype": selected.get("MacrophageSubtype", pd.Series([""] * len(selected))).fillna("").to_numpy(),
            "array_row": selected["array_row"].to_numpy(dtype=int),
            "array_col": selected["array_col"].to_numpy(dtype=int),
            "total_counts": library.reindex(selected["barcode"]).to_numpy(dtype=float),
        }
    )


def _candidate_interactions(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    expr = expression.set_index("cell_id")
    rows = []
    for lr in lr_pairs.itertuples(index=False):
        pairs = []
        for sender_idx, neighbors in enumerate(tree.query_ball_point(coords, r=float(lr.spatial_range))):
            neighbors = [idx for idx in neighbors if idx != sender_idx]
            if len(neighbors) > max_neighbors:
                distances = np.linalg.norm(coords[neighbors] - coords[sender_idx], axis=1)
                order = np.argsort(distances)[:max_neighbors]
                neighbors = [neighbors[int(i)] for i in order]
            for receiver_idx in neighbors:
                pairs.append((sender_idx, receiver_idx))
        if len(pairs) > max_candidates_per_lr:
            keep = rng.choice(np.arange(len(pairs)), size=int(max_candidates_per_lr), replace=False)
            pairs = [pairs[int(i)] for i in np.sort(keep)]
        for sender_idx, receiver_idx in pairs:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            distance = float(np.linalg.norm(coords[int(sender_idx)] - coords[int(receiver_idx)]))
            ligand_expr = float(expr.loc[sender["cell_id"], lr.ligand])
            receptor_expr = float(expr.loc[receiver["cell_id"], lr.receptor])
            type_ok = _type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))
            niche_ok = _niche_rule(str(lr.active_niche), str(sender["niche"]), str(receiver["niche"]))
            rows.append(
                {
                    "sender_id": sender["cell_id"],
                    "receiver_id": receiver["cell_id"],
                    "lr_id": lr.lr_id,
                    "ligand": lr.ligand,
                    "receptor": lr.receptor,
                    "mechanism": lr.mechanism,
                    "distance": distance,
                    "sender_cell_type": sender["cell_type"],
                    "receiver_cell_type": receiver["cell_type"],
                    "sender_niche": sender["niche"],
                    "receiver_niche": receiver["niche"],
                    "weak_signal": str(lr.control_class) in {"context_specific_prior", "immune_checkpoint_context_prior"},
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "type_rule_satisfied": bool(type_ok),
                    "range_rule_satisfied": bool(distance <= float(lr.spatial_range)),
                    "niche_rule_satisfied": bool(niche_ok),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates.insert(0, "candidate_id", [f"visium_hd_crc_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _target_signature_table(
    lr_pairs: pd.DataFrame,
    genes_present: list[str],
    target_genes: dict[str, list[str]],
) -> pd.DataFrame:
    present = set(genes_present)
    rows = []
    for row in lr_pairs.itertuples(index=False):
        targets = [gene for gene in target_genes.get(row.control_name, []) if gene in present]
        rows.append(
            {
                "ligand": row.ligand,
                "receptor": row.receptor,
                "target_genes": ",".join(targets),
                "targets": ",".join(targets),
                "target_source": "phase7_visium_hd_crc_curated_context_targets",
                "evidence_note": (
                    "Curated CRC/TME target-response proxy using genes present in the Visium HD whole-transcriptome panel."
                ),
            }
        )
    return pd.DataFrame(rows)


def _cell_type_from_metadata(row: pd.Series) -> str:
    label_text = " ".join(
        str(row.get(column, ""))
        for column in ["DeconvolutionLabel1", "DeconvolutionLabel2", "UnsupervisedL1", "MacrophageSubtype"]
    ).lower()
    if "spp1" in label_text and "macrophage" in label_text:
        return "spp1_macrophage_proxy"
    if "macrophage" in label_text or "myeloid" in label_text or "dendritic" in label_text or "neutrophil" in label_text:
        return "myeloid_proxy"
    if "cd8 t" in label_text or "cd4 t" in label_text or "t cells" in label_text or "t cell" in label_text:
        return "T_cell_proxy"
    if "caf" in label_text or "fibroblast" in label_text or "myofibroblast" in label_text or "vsm" in label_text:
        return "caf_proxy"
    if "endothelial" in label_text or "pericytes" in label_text:
        return "endothelial_proxy"
    if "tumor" in label_text:
        return "tumor_proxy"
    if "goblet" in label_text or "enterocyte" in label_text or "epithelial" in label_text:
        return "epithelial_proxy"
    if "b cells" in label_text or "plasma" in label_text:
        return "B_cell_proxy"
    return "unknown"


def _niche_from_metadata(row: pd.Series) -> str:
    cell_type = _cell_type_from_metadata(row)
    periphery = str(row.get("Periphery", ""))
    if cell_type == "spp1_macrophage_proxy":
        return "tumor_myeloid_boundary"
    if cell_type == "caf_proxy":
        return "tumor_stromal_boundary"
    if cell_type == "endothelial_proxy":
        return "vascular_stromal_boundary"
    if periphery == "50 micron":
        if cell_type in {"T_cell_proxy", "myeloid_proxy", "B_cell_proxy"}:
            return "tumor_immune_boundary"
    if periphery == "Tumor" or cell_type == "tumor_proxy":
        return "tumor_core"
    if cell_type in {"T_cell_proxy", "myeloid_proxy", "B_cell_proxy"}:
        return "immune_region"
    if cell_type == "epithelial_proxy":
        return "epithelial_region"
    return "unknown"


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name == "SPP1-CD44":
        return sender_type in {"spp1_macrophage_proxy", "myeloid_proxy", "tumor_proxy"} and receiver_type in {
            "tumor_proxy",
            "caf_proxy",
            "T_cell_proxy",
            "epithelial_proxy",
        }
    if control_name == "CXCL12-CXCR4":
        return sender_type in {"caf_proxy", "endothelial_proxy"} and receiver_type in {
            "T_cell_proxy",
            "myeloid_proxy",
            "tumor_proxy",
        }
    if control_name == "VEGFA-KDR":
        return sender_type in {"tumor_proxy", "myeloid_proxy", "caf_proxy"} and receiver_type == "endothelial_proxy"
    if control_name == "TGFB1-TGFBR2":
        return sender_type in {"tumor_proxy", "myeloid_proxy", "caf_proxy"} and receiver_type in {
            "caf_proxy",
            "tumor_proxy",
            "T_cell_proxy",
        }
    if control_name == "CD274-PDCD1":
        return sender_type in {"tumor_proxy", "myeloid_proxy"} and receiver_type == "T_cell_proxy"
    return False


def _niche_rule(active_niche: str, sender_niche: str, receiver_niche: str) -> bool:
    if active_niche in {"any", "", "nan"}:
        return True
    return active_niche in {sender_niche, receiver_niche}


def _marker_genes() -> set[str]:
    return {
        "EPCAM",
        "KRT8",
        "KRT18",
        "COL1A1",
        "COL1A2",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
        "PTPRC",
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",
        "IL7R",
        "LYZ",
        "CD68",
        "CD14",
        "C1QA",
        "C1QB",
        "PECAM1",
        "VWF",
        "KDR",
        "PDCD1",
        "CD274",
        "SPP1",
        "CD44",
        "CXCL12",
        "CXCR4",
        "VEGFA",
        "TGFB1",
        "TGFBR2",
    }


def _target_gene_set(target_genes: dict[str, list[str]]) -> set[str]:
    genes: set[str] = set()
    for targets in target_genes.values():
        genes.update(targets)
    return genes


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
