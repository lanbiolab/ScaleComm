#!/usr/bin/env python
"""Standardize a small Xenium human lymph-node subset for Phase 7 Goal 5.

This creates a runnable real-platform dataset from public 10x Xenium output
files. Cell type and niche labels are marker-rule proxies for stratification
only; they are not manual annotations or ground truth.
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
RAW_PREFIX = "Xenium_V1_hLymphNode_nondiseased_section"

DEFAULT_LR_PANEL = [
    {
        "lr_id": "xenium_ln_lr_000_CCL19_CCR7",
        "ligand": "CCL19",
        "receptor": "CCR7",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 180.0,
        "active_niche": "T_zone_proxy",
        "control_name": "CCL19-CCR7",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "xenium_ln_lr_001_CD86_CD28",
        "ligand": "CD86",
        "receptor": "CD28",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 55.0,
        "active_niche": "immune_activation_proxy",
        "control_name": "CD86-CD28",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "xenium_ln_lr_002_CD70_CD27",
        "ligand": "CD70",
        "receptor": "CD27",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 55.0,
        "active_niche": "immune_activation_proxy",
        "control_name": "CD70-CD27",
        "control_class": "lymph_node_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "xenium_ln_lr_003_CD274_PDCD1",
        "ligand": "CD274",
        "receptor": "PDCD1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 100.0,
        "active_niche": "immune_activation_proxy",
        "control_name": "CD274-PDCD1",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "xenium_ln_lr_004_MS4A1_CD3D_negative",
        "ligand": "MS4A1",
        "receptor": "CD3D",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 180.0,
        "active_niche": "any",
        "control_name": "MS4A1-CD3D",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "xenium_ln_lr_005_CD19_CD3E_negative",
        "ligand": "CD19",
        "receptor": "CD3E",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 180.0,
        "active_niche": "any",
        "control_name": "CD19-CD3E",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
]


COMPACT_TARGET_GENES = {
    "CCL19-CCR7": ["CCR7", "CD3D", "CD3E", "CD4", "CD8A", "CD28", "CD27"],
    "CD274-PDCD1": ["PDCD1", "CD3D", "CD3E", "CD4", "CD8A", "CD28", "CD27"],
    "CD86-CD28": ["CD28", "CD3D", "CD3E", "CD4", "CD8A", "CD27", "PDCD1"],
    "CD70-CD27": ["CD27", "CD3D", "CD3E", "CD4", "CD8A", "CD28", "PDCD1"],
    "MS4A1-CD3D": ["CD3D", "CD3E", "CD4", "CD8A", "CD28", "CD27"],
    "CD19-CD3E": ["CD3E", "CD3D", "CD4", "CD8A", "CD28", "CD27"],
}


RICH_TARGET_GENES = {
    "CCL19-CCR7": [
        "CCR7",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD28",
        "CD27",
        "IL7R",
        "SELL",
        "TCF7",
        "LEF1",
        "MAL",
        "LTB",
    ],
    "CD274-PDCD1": [
        "PDCD1",
        "LAG3",
        "HAVCR2",
        "GZMB",
        "CTLA4",
        "TIGIT",
        "ENTPD1",
        "CXCR6",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD27",
        "CD28",
    ],
    "CD86-CD28": [
        "CD28",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD27",
        "PDCD1",
        "ICOS",
        "CTLA4",
        "GZMB",
        "LAG3",
    ],
    "CD70-CD27": [
        "CD27",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD28",
        "PDCD1",
        "GZMB",
        "LAG3",
        "HAVCR2",
    ],
    "MS4A1-CD3D": ["CD3D", "CD3E", "CD4", "CD8A", "CD28", "CD27", "IL7R", "TCF7", "LEF1"],
    "CD19-CD3E": ["CD3E", "CD3D", "CD4", "CD8A", "CD28", "CD27", "IL7R", "TCF7", "LEF1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/optional/xenium_human_lymph_node/raw",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/"
            "goal5_cross_platform_real_data/standardized_inputs/"
            "xenium_human_lymph_node_subset"
        ),
    )
    parser.add_argument("--n-cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-candidates-per-lr", type=int, default=120000)
    parser.add_argument("--target-profile", choices=("compact", "rich"), default="compact")
    parser.add_argument(
        "--reference-dataset-dir",
        default=None,
        help="Optional standardized dataset whose cells/candidates should be reused while rebuilding expression targets.",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_raw = pd.read_parquet(raw_dir / f"{RAW_PREFIX}_cells.parquet")
    h5_path = raw_dir / f"{RAW_PREFIX}_cell_feature_matrix.h5"
    matrix_bundle = _read_xenium_h5(h5_path)
    lr_pairs = pd.DataFrame(DEFAULT_LR_PANEL)
    target_genes = RICH_TARGET_GENES if args.target_profile == "rich" else COMPACT_TARGET_GENES
    genes_needed = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | _marker_genes() | _target_gene_set(target_genes))
    genes_present = [gene for gene in genes_needed if gene in matrix_bundle["gene_to_idx"]]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR panel contains genes absent from Xenium matrix: {missing_lr_genes}")

    reference_dir = PROJECT_ROOT / args.reference_dataset_dir if args.reference_dataset_dir else None
    if reference_dir is not None:
        cells = pd.read_csv(reference_dir / "cells.csv")
        candidates = pd.read_csv(reference_dir / "candidate_interactions.csv")
        expression = _expression_table(matrix_bundle, cells["cell_id"].to_list(), genes_present)
    else:
        selected_cells = _select_cells(cells_raw, matrix_bundle["barcodes"], args.n_cells, args.seed)
        expression = _expression_table(matrix_bundle, selected_cells["cell_id"].to_list(), genes_present)
        cells = _annotate_cells(selected_cells, expression)
        candidates = _candidate_interactions(
            cells,
            expression,
            lr_pairs,
            max_neighbors=args.max_neighbors,
            max_candidates_per_lr=args.max_candidates_per_lr,
            seed=args.seed,
        )
    target_signatures = _target_signature_table(lr_pairs, genes_present, target_genes)

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)
    target_signatures.to_csv(output_dir / "target_signatures.csv", index=False)

    metadata = {
        "dataset_id": "xenium_human_lymph_node_subset",
        "platform": "Xenium In Situ Gene Expression",
        "source": "10x Genomics Xenium_V1_hLymphNode_nondiseased_section",
        "raw_dir": _rel(raw_dir),
        "output_dir": _rel(output_dir),
        "n_cells": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "n_cells_requested": int(args.n_cells),
            "seed": int(args.seed),
            "rule": "top transcript-count cells sampled after ranking to keep a compact but information-rich subset",
        },
        "claim_boundary": (
            "Real Xenium coordinates and expression are preserved. cell_type, region, and niche are marker-rule "
            "proxies for workflow stratification only, not curated annotations or ground truth. candidate_interactions "
            "contains candidate edges and controls for cross-platform evaluation wiring, not native true CCC labels."
        ),
        "missing_lr_genes": missing_lr_genes,
        "target_profile": str(args.target_profile),
        "reference_dataset_dir": _rel(reference_dir) if reference_dir is not None else None,
        "source_files": {
            "cells_parquet": _rel(raw_dir / f"{RAW_PREFIX}_cells.parquet"),
            "cell_feature_matrix_h5": _rel(h5_path),
            "gene_panel_json": _rel(raw_dir / f"{RAW_PREFIX}_gene_panel.json"),
            "analysis_zarr_zip": _rel(raw_dir / f"{RAW_PREFIX}_analysis.zarr.zip"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "dataset": metadata["dataset_id"],
                "platform": metadata["platform"],
                "n_cells": metadata["n_cells"],
                "n_expression_genes": metadata["n_expression_genes"],
                "n_lr_pairs": metadata["n_lr_pairs"],
                "n_candidate_interactions": metadata["n_candidate_interactions"],
                "output_dir": metadata["output_dir"],
                "claim_boundary": metadata["claim_boundary"],
            }
        ]
    ).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(metadata, indent=2))


def _read_xenium_h5(path: Path) -> dict[str, Any]:
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


def _select_cells(cells: pd.DataFrame, barcodes: list[str], n_cells: int, seed: int) -> pd.DataFrame:
    available = cells[cells["cell_id"].isin(barcodes)].copy()
    n_cells = min(int(n_cells), len(available))
    ranked = available.sort_values(["transcript_counts", "cell_id"], ascending=[False, True]).head(max(n_cells * 6, n_cells))
    rng = np.random.default_rng(seed)
    if len(ranked) > n_cells:
        keep = np.sort(rng.choice(np.arange(len(ranked)), size=n_cells, replace=False))
        ranked = ranked.iloc[keep]
    out = ranked.rename(columns={"x_centroid": "x", "y_centroid": "y"}).copy()
    return out[["cell_id", "x", "y", "transcript_counts", "total_counts", "cell_area", "nucleus_area"]].reset_index(drop=True)


def _expression_table(bundle: dict[str, Any], cell_ids: list[str], genes: list[str]) -> pd.DataFrame:
    cell_idx = [bundle["barcode_to_idx"][cell_id] for cell_id in cell_ids]
    gene_idx = [bundle["gene_to_idx"][gene] for gene in genes]
    sub = bundle["matrix"][gene_idx, :][:, cell_idx].T.toarray()
    frame = pd.DataFrame(sub, columns=genes)
    frame.insert(0, "cell_id", cell_ids)
    return frame


def _annotate_cells(cells: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    expr = expression.set_index("cell_id")
    labels = []
    niches = []
    for cell_id in cells["cell_id"]:
        row = expr.loc[cell_id]
        t_score = _sum_genes(row, ["CD3D", "CD3E", "CD4", "CD8A", "CCR7", "CD28", "PDCD1"])
        b_score = _sum_genes(row, ["MS4A1", "CD19", "BANK1"])
        myeloid_score = _sum_genes(row, ["CD14", "CD68", "CD163", "AIF1", "ADGRE1"])
        stromal_score = _sum_genes(row, ["ACTA2", "CAV1", "CAVIN1", "ACKR1"])
        scores = {
            "T_cell_proxy": t_score,
            "B_cell_proxy": b_score,
            "myeloid_proxy": myeloid_score,
            "stromal_proxy": stromal_score,
        }
        label = max(scores, key=scores.get)
        if scores[label] <= 0:
            label = "unknown"
        labels.append(label)
        if label == "T_cell_proxy" or row.get("CCR7", 0) > 0 or row.get("CCL19", 0) > 0:
            niches.append("T_zone_proxy")
        elif label == "B_cell_proxy":
            niches.append("B_zone_proxy")
        elif label in {"myeloid_proxy", "stromal_proxy"}:
            niches.append("immune_activation_proxy")
        else:
            niches.append("unknown")
    out = cells.copy()
    out["cell_type"] = labels
    out["region"] = niches
    out["niche"] = niches
    return out[["cell_id", "x", "y", "cell_type", "region", "niche", "transcript_counts", "total_counts", "cell_area", "nucleus_area"]]


def _sum_genes(row: pd.Series, genes: list[str]) -> float:
    return float(sum(row.get(gene, 0.0) for gene in genes))


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
                str(sender["niche"]) == active_niche and str(receiver["niche"]) == active_niche
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
    candidates.insert(0, "candidate_id", [f"xenium_ln_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name == "CCL19-CCR7":
        return sender_type in {"T_cell_proxy", "stromal_proxy", "unknown"} and receiver_type == "T_cell_proxy"
    if control_name in {"CD86-CD28", "CD70-CD27", "CD274-PDCD1"}:
        return sender_type in {"myeloid_proxy", "B_cell_proxy", "stromal_proxy", "unknown"} and receiver_type == "T_cell_proxy"
    return False


def _marker_genes() -> set[str]:
    return {
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CCR7",
        "CD28",
        "PDCD1",
        "MS4A1",
        "CD19",
        "BANK1",
        "CD14",
        "CD68",
        "CD163",
        "AIF1",
        "ADGRE1",
        "ACTA2",
        "CAV1",
        "CAVIN1",
        "ACKR1",
        "CCL19",
    }


def _target_gene_set(target_genes: dict[str, list[str]]) -> set[str]:
    genes: set[str] = set()
    for targets in target_genes.values():
        genes.update(targets)
    return genes


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
                "target_source": f"phase7_xenium_{'rich' if target_genes is RICH_TARGET_GENES else 'compact'}_within_panel_targets",
                "evidence_note": "Curated Xenium within-panel receiver target-response proxy using genes present in the 541-gene panel.",
            }
        )
    return pd.DataFrame(rows)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
