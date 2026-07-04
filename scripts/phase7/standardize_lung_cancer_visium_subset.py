#!/usr/bin/env python
"""Standardize a lung-cancer Visium subset for Phase 7 supplemental validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LR_PANEL = [
    {
        "lr_id": "lung_lr_000_CXCL12_CXCR4",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 3600.0,
        "active_niche": "stromal_immune_proxy",
        "control_name": "CXCL12-CXCR4",
        "control_class": "lung_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "lung_lr_001_SPP1_CD44",
        "ligand": "SPP1",
        "receptor": "CD44",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "myeloid_epithelial_proxy",
        "control_name": "SPP1-CD44",
        "control_class": "lung_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "lung_lr_002_TGFB1_TGFBR2",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "stromal_immune_proxy",
        "control_name": "TGFB1-TGFBR2",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "lung_lr_003_EGF_EGFR",
        "ligand": "EGF",
        "receptor": "EGFR",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 2400.0,
        "active_niche": "epithelial_proxy",
        "control_name": "EGF-EGFR",
        "control_class": "lung_tme_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "lung_lr_004_ALB_EPCAM_negative",
        "ligand": "ALB",
        "receptor": "EPCAM",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 3600.0,
        "active_niche": "any",
        "control_name": "ALB-EPCAM",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "lung_lr_005_ACTB_KRT8_negative",
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
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", default="lung_cancer/lung_mapping_results.h5ad")
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/"
            "goal5_cross_platform_real_data/standardized_inputs/"
            "lung_cancer_visium_subset"
        ),
    )
    parser.add_argument("--n-spots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-candidates-per-lr", type=int, default=100000)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input_h5ad
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(input_path)
    lr_pairs = pd.DataFrame(DEFAULT_LR_PANEL)
    genes_needed = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | _marker_genes())
    genes_present = [gene for gene in genes_needed if gene in data.var_names]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR panel contains genes absent from lung h5ad: {missing_lr_genes}")

    selected = _select_spots(data, args.n_spots, args.seed)
    expression = _expression_table(data, selected, genes_present)
    cells = _cells_table(data, selected, expression)
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
        "dataset_id": "lung_cancer_visium_subset",
        "platform": "10x Visium / lung cancer h5ad",
        "source": _rel(input_path),
        "output_dir": _rel(output_dir),
        "n_spots": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "n_spots_requested": int(args.n_spots),
            "seed": int(args.seed),
            "rule": "sample from high-abundance epithelial, myeloid, fibroblast, and T/NK neighborhoods",
        },
        "claim_boundary": (
            "Real Visium coordinates and expression are preserved. cell_type, region, and niche are derived from "
            "cell2location abundance proxies and are not manual pathology labels or native CCC ground truth. "
            "This is supplemental cross-tissue real-background evidence, not a replacement for priority platforms "
            "such as Visium HD, Xenium, CosMx, MERFISH, or Slide-seqV2."
        ),
        "source_obsm": "means_cell_abundance_w_sf;spatial",
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


def _select_spots(data: ad.AnnData, n_spots: int, seed: int) -> list[int]:
    abundance = data.obsm["means_cell_abundance_w_sf"]
    target_columns = [
        "meanscell_abundance_w_sf_Epithelial cells",
        "meanscell_abundance_w_sf_Fibroblasts",
        "meanscell_abundance_w_sf_Myeloid cells",
        "meanscell_abundance_w_sf_T_NK cells",
    ]
    score = abundance[target_columns].sum(axis=1).to_numpy(dtype=float)
    n_spots = min(int(n_spots), data.n_obs)
    pool_size = min(max(n_spots * 4, n_spots), data.n_obs)
    pool = np.argsort(score)[::-1][:pool_size]
    rng = np.random.default_rng(seed)
    if len(pool) > n_spots:
        pool = np.sort(rng.choice(pool, size=n_spots, replace=False))
    return [int(idx) for idx in pool]


def _expression_table(data: ad.AnnData, selected: list[int], genes: list[str]) -> pd.DataFrame:
    gene_idx = [int(np.where(data.var_names == gene)[0][0]) for gene in genes]
    sub = data.X[selected, :][:, gene_idx]
    if sparse.issparse(sub):
        values = sub.toarray()
    else:
        values = np.asarray(sub)
    frame = pd.DataFrame(values, columns=genes)
    frame.insert(0, "cell_id", [str(data.obs_names[idx]) for idx in selected])
    return frame


def _cells_table(data: ad.AnnData, selected: list[int], expression: pd.DataFrame) -> pd.DataFrame:
    abundance = data.obsm["means_cell_abundance_w_sf"].iloc[selected].copy()
    coords = np.asarray(data.obsm["spatial"])[selected]
    dominant = abundance.idxmax(axis=1).str.replace("meanscell_abundance_w_sf_", "", regex=False)
    labels = dominant.map(
        {
            "Epithelial cells": "epithelial_proxy",
            "Fibroblasts": "fibroblast_proxy",
            "Myeloid cells": "myeloid_proxy",
            "T_NK cells": "T_NK_proxy",
            "Endothelial cells": "endothelial_proxy",
            "MAST cells": "mast_proxy",
            "B lymphocytes": "B_cell_proxy",
            "Unknown": "unknown",
        }
    ).fillna("unknown")
    niches = [_niche(label) for label in labels]
    out = pd.DataFrame(
        {
            "cell_id": expression["cell_id"].to_numpy(),
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
            "cell_type": labels.to_numpy(),
            "region": niches,
            "niche": niches,
        }
    )
    for column in abundance.columns:
        out[column] = abundance[column].to_numpy(dtype=float)
    return out


def _niche(label: str) -> str:
    if label == "epithelial_proxy":
        return "epithelial_proxy"
    if label in {"fibroblast_proxy", "endothelial_proxy"}:
        return "stromal_immune_proxy"
    if label in {"myeloid_proxy", "T_NK_proxy", "B_cell_proxy"}:
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
    candidates.insert(0, "candidate_id", [f"lung_visium_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name == "CXCL12-CXCR4":
        return sender_type in {"fibroblast_proxy", "endothelial_proxy"} and receiver_type in {"T_NK_proxy", "myeloid_proxy"}
    if control_name == "SPP1-CD44":
        return sender_type in {"myeloid_proxy", "epithelial_proxy"} and receiver_type in {"epithelial_proxy", "fibroblast_proxy", "T_NK_proxy"}
    if control_name == "TGFB1-TGFBR2":
        return sender_type in {"myeloid_proxy", "fibroblast_proxy", "epithelial_proxy"} and receiver_type in {"fibroblast_proxy", "epithelial_proxy", "T_NK_proxy"}
    if control_name == "EGF-EGFR":
        return sender_type in {"epithelial_proxy", "myeloid_proxy"} and receiver_type == "epithelial_proxy"
    return False


def _marker_genes() -> set[str]:
    return {
        "EPCAM",
        "KRT18",
        "KRT19",
        "COL1A1",
        "COL1A2",
        "VIM",
        "PECAM1",
        "CD3D",
        "CD3E",
        "NKG7",
        "CD14",
        "LYZ",
        "MS4A1",
        "PTPRC",
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
