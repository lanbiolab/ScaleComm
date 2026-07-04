#!/usr/bin/env python
"""Standardize 10x post-Xenium human colon cancer Visium data for Phase 7.

The output follows the shared Phase 7 real-platform schema. It preserves real
10x Visium CytAssist coordinates and expression. Marker-rule cell type and
niche labels are workflow proxies only, not native pathology labels or CCC
ground truth.
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
RAW_PREFIX = "CytAssist_FFPE_Human_Colon_Post_Xenium_Rep1"

DEFAULT_LR_PANEL = [
    {
        "lr_id": "post_xenium_crc_lr_000_TGFB1_TGFBR2",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "stromal_immune_proxy",
        "control_name": "TGFB1-TGFBR2",
        "control_class": "crc_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "post_xenium_crc_lr_001_SPP1_CD44",
        "ligand": "SPP1",
        "receptor": "CD44",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "myeloid_stromal_proxy",
        "control_name": "SPP1-CD44",
        "control_class": "crc_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "post_xenium_crc_lr_002_CXCL12_CXCR4",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 3600.0,
        "active_niche": "stromal_immune_proxy",
        "control_name": "CXCL12-CXCR4",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "post_xenium_crc_lr_003_VEGFA_KDR",
        "ligand": "VEGFA",
        "receptor": "KDR",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "vascular_stromal_proxy",
        "control_name": "VEGFA-KDR",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "post_xenium_crc_lr_004_EGF_EGFR",
        "ligand": "EGF",
        "receptor": "EGFR",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "epithelial_proxy",
        "control_name": "EGF-EGFR",
        "control_class": "crc_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "post_xenium_crc_lr_005_ACTB_KRT8_negative",
        "ligand": "ACTB",
        "receptor": "KRT8",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 3600.0,
        "active_niche": "any",
        "control_name": "ACTB-KRT8",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "post_xenium_crc_lr_006_COL1A1_EPCAM_negative",
        "ligand": "COL1A1",
        "receptor": "EPCAM",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 3600.0,
        "active_niche": "any",
        "control_name": "COL1A1-EPCAM",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/optional/post_xenium_human_colon_cancer/raw",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/"
            "goal5_cross_platform_real_data/standardized_inputs/"
            "post_xenium_human_colon_cancer_subset"
        ),
    )
    parser.add_argument("--n-spots", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-candidates-per-lr", type=int, default=100000)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = raw_dir / f"{RAW_PREFIX}_filtered_feature_bc_matrix.h5"
    spatial_path = raw_dir / "extracted/spatial/tissue_positions.csv"
    matrix_bundle = _read_10x_h5(h5_path)
    spatial = pd.read_csv(spatial_path)
    spatial = spatial[(spatial["in_tissue"] == 1) & spatial["barcode"].isin(matrix_bundle["barcodes"])].copy()

    lr_pairs = pd.DataFrame(DEFAULT_LR_PANEL)
    genes_needed = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | _marker_genes())
    genes_present = [gene for gene in genes_needed if gene in matrix_bundle["gene_to_idx"]]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR panel contains genes absent from post-Xenium colon matrix: {missing_lr_genes}")

    selected = _select_spots(spatial, matrix_bundle, genes_present, args.n_spots, args.seed)
    expression = _expression_table(matrix_bundle, selected["barcode"].to_list(), genes_present)
    cells = _cells_table(selected, expression)
    candidates = _candidate_interactions(
        cells,
        expression,
        lr_pairs,
        max_neighbors=args.max_neighbors,
        max_candidates_per_lr=args.max_candidates_per_lr,
        seed=args.seed,
    )

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)

    metadata = {
        "dataset_id": "post_xenium_human_colon_cancer_subset",
        "platform": "10x Visium CytAssist post-Xenium human colon cancer FFPE",
        "source": "10x Genomics CytAssist_FFPE_Human_Colon_Post_Xenium_Rep1",
        "raw_dir": _rel(raw_dir),
        "output_dir": _rel(output_dir),
        "n_spots": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "n_spots_requested": int(args.n_spots),
            "seed": int(args.seed),
            "rule": "sample from high marker-diversity in-tissue spots to retain epithelial, stromal, immune, myeloid, and vascular proxies",
        },
        "claim_boundary": (
            "Real 10x Visium CytAssist coordinates and expression are preserved. cell_type, region, and niche are "
            "marker-rule proxies for cross-platform validation wiring, not curated pathology labels or native CCC "
            "ground truth. Positive controls are literature/context priors; negative controls are deliberately "
            "high-expression biologically invalid pairs for false-positive stress testing."
        ),
        "source_files": {
            "matrix_h5": _rel(h5_path),
            "tissue_positions_csv": _rel(spatial_path),
            "spatial_bundle": _rel(raw_dir / f"{RAW_PREFIX}_spatial.tar.gz"),
        },
        "missing_lr_genes": missing_lr_genes,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": metadata["dataset_id"],
                "platform": metadata["platform"],
                "n_cells": metadata["n_spots"],
                "n_expression_genes": metadata["n_expression_genes"],
                "n_lr_pairs": metadata["n_lr_pairs"],
                "n_candidate_interactions": metadata["n_candidate_interactions"],
                "output_dir": metadata["output_dir"],
                "claim_boundary": metadata["claim_boundary"],
            }
        ]
    ).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(metadata, indent=2))


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


def _select_spots(
    spatial: pd.DataFrame,
    bundle: dict[str, Any],
    genes_present: list[str],
    n_spots: int,
    seed: int,
) -> pd.DataFrame:
    expression = _expression_table(bundle, spatial["barcode"].to_list(), genes_present)
    provisional_cells = _cells_table(spatial.reset_index(drop=True), expression)
    expr = expression.drop(columns=["cell_id"])
    marker_score = np.log1p(expr).sum(axis=1).to_numpy(dtype=float)
    library = np.asarray(bundle["matrix"][:, [bundle["barcode_to_idx"][b] for b in spatial["barcode"]]].sum(axis=0)).ravel()
    score = marker_score + 0.0005 * np.log1p(library)
    n_spots = min(int(n_spots), len(spatial))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    target_by_type = {
        "epithelial_proxy": 350,
        "myeloid_proxy": 350,
        "T_cell_proxy": 150,
        "endothelial_proxy": 100,
        "unknown": 250,
    }
    labels = provisional_cells["cell_type"].to_numpy()
    for label, target in target_by_type.items():
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        ranked = idx[np.argsort(score[idx])[::-1]]
        take = min(int(target), len(ranked), n_spots - len(selected))
        selected.extend(int(i) for i in ranked[:take])
    remaining = n_spots - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        pool = np.array([idx for idx in np.argsort(score)[::-1] if int(idx) not in selected_set], dtype=int)
        pool_size = min(max(remaining * 3, remaining), len(pool))
        pool = pool[:pool_size]
        if len(pool) > remaining:
            pool = np.sort(rng.choice(pool, size=remaining, replace=False))
        selected.extend(int(i) for i in pool)
    return spatial.iloc[np.sort(np.array(selected, dtype=int))].reset_index(drop=True)


def _expression_table(bundle: dict[str, Any], barcodes: list[str], genes: list[str]) -> pd.DataFrame:
    spot_idx = [bundle["barcode_to_idx"][barcode] for barcode in barcodes]
    gene_idx = [bundle["gene_to_idx"][gene] for gene in genes]
    sub = bundle["matrix"][gene_idx, :][:, spot_idx].T.toarray()
    frame = pd.DataFrame(sub, columns=genes)
    frame.insert(0, "cell_id", barcodes)
    return frame


def _cells_table(spatial: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    expr = expression.set_index("cell_id")
    labels = []
    niches = []
    for barcode in spatial["barcode"]:
        row = expr.loc[barcode]
        scores = {
            "epithelial_proxy": _sum_genes(row, ["EPCAM", "KRT8", "KRT19", "MUC1", "EGFR"]),
            "fibroblast_proxy": _sum_genes(row, ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "CXCL12"]),
            "T_cell_proxy": _sum_genes(row, ["CD3D", "CD3E", "CD4", "CD8A", "CXCR4"]),
            "myeloid_proxy": _sum_genes(row, ["LYZ", "CD68", "CD14", "C1QA", "C1QB", "SPP1"]),
            "endothelial_proxy": _sum_genes(row, ["PECAM1", "VWF", "KDR"]),
        }
        label = max(scores, key=scores.get)
        if scores[label] <= 0:
            label = "unknown"
        labels.append(label)
        niches.append(_niche(label, row))
    return pd.DataFrame(
        {
            "cell_id": spatial["barcode"].to_numpy(),
            "x": spatial["pxl_col_in_fullres"].to_numpy(dtype=float),
            "y": spatial["pxl_row_in_fullres"].to_numpy(dtype=float),
            "cell_type": labels,
            "region": niches,
            "niche": niches,
            "array_row": spatial["array_row"].to_numpy(dtype=int),
            "array_col": spatial["array_col"].to_numpy(dtype=int),
        }
    )


def _niche(label: str, row: pd.Series) -> str:
    if row.get("SPP1", 0) > 0 and row.get("CD44", 0) > 0:
        return "myeloid_stromal_proxy"
    if row.get("KDR", 0) > 0 or row.get("VWF", 0) > 0:
        return "vascular_stromal_proxy"
    if label == "epithelial_proxy":
        return "epithelial_proxy"
    if label == "fibroblast_proxy":
        return "stromal_immune_proxy"
    if label == "endothelial_proxy":
        return "vascular_stromal_proxy"
    if label == "myeloid_proxy":
        return "myeloid_stromal_proxy"
    if label == "T_cell_proxy":
        return "immune_proxy"
    return "unknown"


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
        radius = float(lr.spatial_range)
        pairs = []
        for sender_idx, neighbors in enumerate(tree.query_ball_point(coords, r=radius)):
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
            active_niche = str(lr.active_niche)
            type_ok = _type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))
            range_ok = distance <= float(lr.spatial_range)
            niche_ok = active_niche in {"any", "", "nan"} or (
                str(sender["niche"]) == active_niche or str(receiver["niche"]) == active_niche
            )
            ligand_expr = float(expr.loc[sender["cell_id"], lr.ligand])
            receptor_expr = float(expr.loc[receiver["cell_id"], lr.receptor])
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
                    "weak_signal": str(lr.control_class) == "context_specific_prior",
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "type_rule_satisfied": bool(type_ok),
                    "range_rule_satisfied": bool(range_ok),
                    "niche_rule_satisfied": bool(niche_ok),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates.insert(0, "candidate_id", [f"post_xenium_crc_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name == "TGFB1-TGFBR2":
        return sender_type in {"fibroblast_proxy", "myeloid_proxy", "epithelial_proxy"} and receiver_type in {
            "fibroblast_proxy",
            "epithelial_proxy",
            "T_cell_proxy",
        }
    if control_name == "SPP1-CD44":
        return sender_type in {"myeloid_proxy", "epithelial_proxy"} and receiver_type in {
            "epithelial_proxy",
            "fibroblast_proxy",
            "T_cell_proxy",
        }
    if control_name == "CXCL12-CXCR4":
        return sender_type in {"fibroblast_proxy", "endothelial_proxy"} and receiver_type in {
            "T_cell_proxy",
            "myeloid_proxy",
            "epithelial_proxy",
        }
    if control_name == "VEGFA-KDR":
        return sender_type in {"myeloid_proxy", "epithelial_proxy", "fibroblast_proxy"} and receiver_type == "endothelial_proxy"
    if control_name == "EGF-EGFR":
        return sender_type in {"epithelial_proxy", "myeloid_proxy"} and receiver_type == "epithelial_proxy"
    return False


def _sum_genes(row: pd.Series, genes: list[str]) -> float:
    return float(sum(row.get(gene, 0.0) for gene in genes))


def _marker_genes() -> set[str]:
    return {
        "EPCAM",
        "KRT8",
        "KRT19",
        "MUC1",
        "COL1A1",
        "COL1A2",
        "DCN",
        "LUM",
        "ACTA2",
        "PTPRC",
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",
        "LYZ",
        "CD68",
        "CD14",
        "C1QA",
        "C1QB",
        "PECAM1",
        "VWF",
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
