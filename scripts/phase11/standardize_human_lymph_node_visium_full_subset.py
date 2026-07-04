#!/usr/bin/env python
"""Standardize 10x V1 Human Lymph Node Visium full-transcriptome subset."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PREFIX = "V1_Human_Lymph_Node"

LR_PANEL = [
    {
        "lr_id": "ln_full_lr_000_CCL19_CCR7",
        "ligand": "CCL19",
        "receptor": "CCR7",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 700.0,
        "active_niche": "T_cell_zone_proxy",
        "control_name": "CCL19-CCR7",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_001_CCL21_CCR7",
        "ligand": "CCL21",
        "receptor": "CCR7",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 700.0,
        "active_niche": "T_cell_zone_proxy",
        "control_name": "CCL21-CCR7",
        "control_class": "lymph_node_positive_prior_weak",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_002_CXCL13_CXCR5",
        "ligand": "CXCL13",
        "receptor": "CXCR5",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 700.0,
        "active_niche": "B_follicle_proxy",
        "control_name": "CXCL13-CXCR5",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_003_CXCL12_CXCR4",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 700.0,
        "active_niche": "stromal_follicle_proxy",
        "control_name": "CXCL12-CXCR4",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_004_CD48_CD244",
        "ligand": "CD48",
        "receptor": "CD244",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 220.0,
        "active_niche": "T_cell_zone_proxy",
        "control_name": "CD48-CD244",
        "control_class": "context_specific_prior_weak",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_005_CD40LG_CD40",
        "ligand": "CD40LG",
        "receptor": "CD40",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 220.0,
        "active_niche": "B_follicle_proxy",
        "control_name": "CD40LG-CD40",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "ln_full_lr_006_ACTB_CD44_negative",
        "ligand": "ACTB",
        "receptor": "CD44",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 900.0,
        "active_niche": "any",
        "control_name": "ACTB-CD44",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "ln_full_lr_007_RPLP0_CXCR4_negative",
        "ligand": "RPLP0",
        "receptor": "CXCR4",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 900.0,
        "active_niche": "any",
        "control_name": "RPLP0-CXCR4",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "ln_full_lr_008_EEF1A1_CCR7_negative",
        "ligand": "EEF1A1",
        "receptor": "CCR7",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 900.0,
        "active_niche": "any",
        "control_name": "EEF1A1-CCR7",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
]

TARGET_SIGNATURES = {
    "ln_full_lr_000_CCL19_CCR7": ["CCR7", "IL7R", "TCF7", "SELL", "LTB", "JUNB", "NFKBIA", "DUSP1", "CD3D", "CD3E"],
    "ln_full_lr_001_CCL21_CCR7": ["CCR7", "IL7R", "TCF7", "SELL", "LTB", "JUNB", "NFKBIA", "DUSP1", "CD3D", "CD3E"],
    "ln_full_lr_002_CXCL13_CXCR5": ["CXCR5", "MS4A1", "CD79A", "CD79B", "LTB", "JUNB", "NFKBIA", "BCL6", "FCER2"],
    "ln_full_lr_003_CXCL12_CXCR4": ["CXCR4", "IL7R", "LTB", "JUNB", "NFKBIA", "DUSP1", "NR4A1", "STAT1"],
    "ln_full_lr_004_CD48_CD244": ["CD244", "CD247", "LCK", "FYN", "ZAP70", "RUNX3", "KLRD1", "NCR3", "BTLA", "CD96"],
    "ln_full_lr_005_CD40LG_CD40": ["CD40", "MS4A1", "CD79A", "CD79B", "NFKBIA", "JUNB", "BCL6", "AICDA", "IRF4"],
    "ln_full_lr_006_ACTB_CD44_negative": ["CD44", "JUNB", "NFKBIA", "STAT1"],
    "ln_full_lr_007_RPLP0_CXCR4_negative": ["CXCR4", "JUNB", "NFKBIA", "STAT1"],
    "ln_full_lr_008_EEF1A1_CCR7_negative": ["CCR7", "JUNB", "NFKBIA", "STAT1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/optional/human_lymph_node_visium_full/raw")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset",
    )
    parser.add_argument("--n-spots", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--max-candidates-per-lr", type=int, default=120000)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = raw_dir / f"{RAW_PREFIX}_filtered_feature_bc_matrix.h5"
    spatial_dir = _ensure_spatial(raw_dir)
    spatial_path = spatial_dir / "tissue_positions_list.csv"
    bundle = read_10x_h5(h5_path)
    spatial = read_spatial_positions(spatial_path, bundle["barcodes"])

    lr_pairs = pd.DataFrame(LR_PANEL)
    genes_needed = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | marker_genes() | target_genes())
    genes_present = [gene for gene in genes_needed if gene in bundle["gene_to_idx"]]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR genes absent from V1 human lymph node matrix: {missing_lr_genes}")

    selected = select_spots(spatial, bundle, genes_present, args.n_spots, args.seed)
    expression = expression_table(bundle, selected["barcode"].to_list(), genes_present)
    cells = cells_table(selected, expression, bundle)
    candidates = candidate_interactions(
        cells,
        expression,
        lr_pairs,
        max_neighbors=args.max_neighbors,
        max_candidates_per_lr=args.max_candidates_per_lr,
        seed=args.seed,
    )
    target_signatures = target_signature_table(genes_present)
    target_pathways = target_pathway_table(lr_pairs, genes_present)

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)
    target_signatures.to_csv(output_dir / "target_signatures.csv", index=False)
    target_pathways.to_csv(output_dir / "target_pathway_pairs.csv", index=False)

    metadata = {
        "dataset_id": "human_lymph_node_visium_full_subset",
        "platform": "10x Visium V1 Human Lymph Node",
        "source": "10x Genomics V1_Human_Lymph_Node public dataset",
        "raw_dir": rel(raw_dir),
        "output_dir": rel(output_dir),
        "n_spots": int(len(cells)),
        "n_raw_matrix_spots": int(len(bundle["barcodes"])),
        "n_raw_matrix_genes": int(len(bundle["genes"])),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "n_target_signature_rows": int(len(target_signatures)),
        "n_target_pathway_rows": int(len(target_pathways)),
        "selection": {
            "n_spots_requested": int(args.n_spots),
            "seed": int(args.seed),
            "rule": "marker-diversity subset retaining T-zone, B-follicle/GC and stromal proxy spots",
        },
        "claim_boundary": (
            "Raw full-transcriptome 10x Visium expression and spatial coordinates are preserved. "
            "cell_type, region and niche are marker-rule proxies, not manual pathology labels or native CCC ground truth. "
            "Positive LR pairs are literature/context priors; high-expression pairs are false-positive stress controls."
        ),
        "source_files": {
            "matrix_h5": rel(h5_path),
            "spatial_tar_gz": rel(raw_dir / f"{RAW_PREFIX}_spatial.tar.gz"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": metadata["dataset_id"],
                "platform": metadata["platform"],
                "n_cells": metadata["n_spots"],
                "n_raw_matrix_genes": metadata["n_raw_matrix_genes"],
                "n_expression_genes": metadata["n_expression_genes"],
                "n_lr_pairs": metadata["n_lr_pairs"],
                "n_candidate_interactions": metadata["n_candidate_interactions"],
                "output_dir": metadata["output_dir"],
                "claim_boundary": metadata["claim_boundary"],
            }
        ]
    ).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(metadata, indent=2))


def _ensure_spatial(raw_dir: Path) -> Path:
    spatial_dir = raw_dir / "spatial"
    if (spatial_dir / "tissue_positions_list.csv").exists():
        return spatial_dir
    archive_path = raw_dir / f"{RAW_PREFIX}_spatial.tar.gz"
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(raw_dir)
    if not (spatial_dir / "tissue_positions_list.csv").exists():
        raise FileNotFoundError(spatial_dir / "tissue_positions_list.csv")
    return spatial_dir


def read_10x_h5(path: Path) -> dict[str, Any]:
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


def read_spatial_positions(path: Path, barcodes: list[str]) -> pd.DataFrame:
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    spatial = pd.read_csv(path, header=None, names=columns)
    return spatial[(spatial["in_tissue"] == 1) & spatial["barcode"].isin(barcodes)].copy()


def select_spots(
    spatial: pd.DataFrame,
    bundle: dict[str, Any],
    genes_present: list[str],
    n_spots: int,
    seed: int,
) -> pd.DataFrame:
    expression = expression_table(bundle, spatial["barcode"].to_list(), genes_present)
    provisional = cells_table(spatial.reset_index(drop=True), expression, bundle)
    expr = expression.drop(columns=["cell_id"])
    marker_score = np.log1p(expr).sum(axis=1).to_numpy(dtype=float)
    library = np.asarray(bundle["matrix"][:, [bundle["barcode_to_idx"][b] for b in spatial["barcode"]]].sum(axis=0)).ravel()
    score = marker_score + 0.0002 * np.log1p(library)
    n_spots = min(int(n_spots), len(spatial))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    target_by_type = {
        "T_cell_proxy": 650,
        "B_cell_proxy": 650,
        "GC_B_cell_proxy": 350,
        "stromal_proxy": 350,
        "myeloid_proxy": 250,
        "unknown": 250,
    }
    labels = provisional["cell_type"].to_numpy()
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


def expression_table(bundle: dict[str, Any], barcodes: list[str], genes: list[str]) -> pd.DataFrame:
    spot_idx = [bundle["barcode_to_idx"][barcode] for barcode in barcodes]
    gene_idx = [bundle["gene_to_idx"][gene] for gene in genes]
    sub = bundle["matrix"][gene_idx, :][:, spot_idx].T.toarray()
    frame = pd.DataFrame(sub, columns=genes)
    frame.insert(0, "cell_id", barcodes)
    return frame


def cells_table(spatial: pd.DataFrame, expression: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    expr = expression.set_index("cell_id")
    labels = []
    niches = []
    for barcode in spatial["barcode"]:
        row = expr.loc[barcode]
        scores = {
            "T_cell_proxy": sum_genes(row, ["CD3D", "CD3E", "CD4", "CD8A", "CCR7", "IL7R", "TCF7"]),
            "B_cell_proxy": sum_genes(row, ["MS4A1", "CD79A", "CD79B", "CXCR5", "CD40"]),
            "GC_B_cell_proxy": sum_genes(row, ["BCL6", "AICDA", "MKI67", "CXCL13", "CXCR5"]),
            "stromal_proxy": sum_genes(row, ["CXCL12", "CCL19", "CCL21", "DCN", "LUM", "COL1A1"]),
            "myeloid_proxy": sum_genes(row, ["LYZ", "CD68", "CD14", "C1QA", "C1QB", "SPP1"]),
        }
        label = max(scores, key=scores.get)
        if scores[label] <= 0:
            label = "unknown"
        labels.append(label)
        niches.append(niche(label, row))
    barcodes = spatial["barcode"].to_list()
    library = np.asarray(bundle["matrix"][:, [bundle["barcode_to_idx"][b] for b in barcodes]].sum(axis=0)).ravel()
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
            "library_size": library.astype(float),
        }
    )


def niche(label: str, row: pd.Series) -> str:
    if label == "T_cell_proxy" or row.get("CCR7", 0) > 0 and row.get("IL7R", 0) > 0:
        return "T_cell_zone_proxy"
    if label in {"B_cell_proxy", "GC_B_cell_proxy"}:
        if row.get("BCL6", 0) > 0 or row.get("CXCL13", 0) > 0:
            return "germinal_center_proxy"
        return "B_follicle_proxy"
    if label == "stromal_proxy":
        return "stromal_follicle_proxy"
    if label == "myeloid_proxy":
        return "myeloid_proxy"
    return "unknown"


def candidate_interactions(
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
            type_ok = type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))
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
                    "weak_signal": str(lr.control_class).endswith("_weak") or str(lr.control_class).startswith("context_specific"),
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
    candidates.insert(0, "candidate_id", [f"ln_full_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name in {"CCL19-CCR7", "CCL21-CCR7"}:
        return sender_type in {"stromal_proxy", "T_cell_proxy"} and receiver_type == "T_cell_proxy"
    if control_name == "CXCL13-CXCR5":
        return sender_type in {"stromal_proxy", "GC_B_cell_proxy", "B_cell_proxy"} and receiver_type in {"B_cell_proxy", "GC_B_cell_proxy"}
    if control_name == "CXCL12-CXCR4":
        return sender_type == "stromal_proxy" and receiver_type in {"T_cell_proxy", "B_cell_proxy", "myeloid_proxy"}
    if control_name == "CD48-CD244":
        return sender_type in {"T_cell_proxy", "B_cell_proxy"} and receiver_type in {"T_cell_proxy", "myeloid_proxy"}
    if control_name == "CD40LG-CD40":
        return sender_type == "T_cell_proxy" and receiver_type in {"B_cell_proxy", "GC_B_cell_proxy"}
    return False


def target_signature_table(genes_present: list[str]) -> pd.DataFrame:
    present = set(genes_present)
    rows = []
    for lr_id, genes in TARGET_SIGNATURES.items():
        for gene in genes:
            rows.append({"lr_id": lr_id, "target_gene": gene, "present_in_expression": gene in present})
    return pd.DataFrame(rows)


def target_pathway_table(lr_pairs: pd.DataFrame, genes_present: list[str]) -> pd.DataFrame:
    present = set(genes_present)
    rows = []
    for row in lr_pairs.itertuples(index=False):
        targets = [gene for gene in TARGET_SIGNATURES.get(str(row.lr_id), []) if gene in present]
        rows.append(
            {
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "pathway": row.control_name,
                "target_genes": ";".join(targets),
                "n_targets_present": int(len(targets)),
            }
        )
    return pd.DataFrame(rows)


def sum_genes(row: pd.Series, genes: list[str]) -> float:
    return float(sum(row.get(gene, 0.0) for gene in genes))


def marker_genes() -> set[str]:
    return {
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CCR7",
        "IL7R",
        "TCF7",
        "MS4A1",
        "CD79A",
        "CD79B",
        "CXCR5",
        "CD40",
        "BCL6",
        "AICDA",
        "MKI67",
        "CXCL13",
        "CXCL12",
        "CCL19",
        "CCL21",
        "DCN",
        "LUM",
        "COL1A1",
        "LYZ",
        "CD68",
        "CD14",
        "C1QA",
        "C1QB",
        "SPP1",
    }


def target_genes() -> set[str]:
    return {gene for genes in TARGET_SIGNATURES.values() for gene in genes}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
