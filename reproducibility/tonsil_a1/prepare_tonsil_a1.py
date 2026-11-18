#!/usr/bin/env python
"""Prepare the public SpaMosaic tonsil A1 input for the ScaleComm tutorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LR_PANEL = [
    {
        "lr_id": "spamosaic_tonsil_lr_000_CXCL13_CXCR5",
        "ligand": "CXCL13",
        "receptor": "CXCR5",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 260.0,
        "active_niche": "germinal_center_proxy",
        "control_name": "CXCL13-CXCR5",
        "control_class": "tonsil_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_001_CCL21_CCR7",
        "ligand": "CCL21",
        "receptor": "CCR7",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 260.0,
        "active_niche": "T_zone_proxy",
        "control_name": "CCL21-CCR7",
        "control_class": "tonsil_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_002_CD40LG_CD40",
        "ligand": "CD40LG",
        "receptor": "CD40",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 70.0,
        "active_niche": "germinal_center_proxy",
        "control_name": "CD40LG-CD40",
        "control_class": "tonsil_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_003_IL21_IL21R",
        "ligand": "IL21",
        "receptor": "IL21R",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 150.0,
        "active_niche": "germinal_center_proxy",
        "control_name": "IL21-IL21R",
        "control_class": "context_specific_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_004_TNFSF13B_TNFRSF13C",
        "ligand": "TNFSF13B",
        "receptor": "TNFRSF13C",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 150.0,
        "active_niche": "germinal_center_proxy",
        "control_name": "TNFSF13B-TNFRSF13C",
        "control_class": "tonsil_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_005_ACTB_EGFR_negative",
        "ligand": "ACTB",
        "receptor": "EGFR",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 260.0,
        "active_niche": "any",
        "control_name": "ACTB-EGFR",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
    {
        "lr_id": "spamosaic_tonsil_lr_006_MS4A1_CD3E_negative",
        "ligand": "MS4A1",
        "receptor": "CD3E",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 260.0,
        "active_niche": "any",
        "control_name": "MS4A1-CD3E",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
    },
]


COMPACT_TARGET_GENES = {
    "CXCL13-CXCR5": ["CXCR5", "MS4A1", "CD19", "CR2", "CD40", "TNFRSF13C"],
    "CCL21-CCR7": ["CCR7", "CD3D", "CD3E", "CD4", "CD8A"],
    "CD40LG-CD40": ["CD40", "MS4A1", "CD19", "TNFRSF13C", "CR2"],
    "IL21-IL21R": ["IL21R", "MS4A1", "CD19", "CD40", "TNFRSF13C", "CR2"],
    "TNFSF13B-TNFRSF13C": ["TNFRSF13C", "MS4A1", "CD19", "CD40", "CR2"],
    "ACTB-EGFR": ["EGFR", "EPCAM", "VIM", "ACTA2"],
    "MS4A1-CD3E": ["CD3E", "CD3D", "CD4", "CD8A"],
}


RICH_TARGET_GENES = {
    "CXCL13-CXCR5": [
        "CXCR5",
        "MS4A1",
        "CD19",
        "CR2",
        "CD40",
        "TNFRSF13C",
        "AICDA",
        "BCL6",
        "PAX5",
        "CD79A",
        "CD79B",
        "BANK1",
        "TNFRSF13B",
    ],
    "CCL21-CCR7": [
        "CCR7",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "IL7R",
        "LTB",
        "SELL",
        "TCF7",
        "LEF1",
        "MAL",
        "CCR6",
    ],
    "CD40LG-CD40": [
        "CD40",
        "MS4A1",
        "CD19",
        "TNFRSF13C",
        "CR2",
        "NFKBIA",
        "TNFAIP3",
        "IRF4",
        "AICDA",
        "BCL6",
        "CD83",
        "CD86",
    ],
    "IL21-IL21R": [
        "IL21R",
        "MS4A1",
        "CD19",
        "CD40",
        "TNFRSF13C",
        "CR2",
        "PRDM1",
        "IRF4",
        "BCL6",
        "AICDA",
        "JCHAIN",
        "XBP1",
        "MZB1",
    ],
    "TNFSF13B-TNFRSF13C": [
        "TNFRSF13C",
        "TNFRSF13B",
        "MS4A1",
        "CD19",
        "CD40",
        "CR2",
        "BANK1",
        "PAX5",
        "CD79A",
        "CD79B",
        "NFKBIA",
        "TNFAIP3",
    ],
    "ACTB-EGFR": ["EGFR", "EPCAM", "KRT8", "KRT18", "KRT19", "VIM", "ACTA2", "ACTB"],
    "MS4A1-CD3E": ["CD3E", "CD3D", "CD4", "CD8A", "IL7R", "LTB", "TCF7", "LEF1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        required=True,
        help="Public SpaMosaic tonsil A1 RNA+ADT h5ad input.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "reproducibility/tonsil_a1/work/prepared/"
            "spamosaic_tonsil_A1_rich_targets_subset"
        ),
    )
    parser.add_argument("--dataset-id", default="spamosaic_tonsil_A1_rich_targets_subset")
    parser.add_argument("--n-cells", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-neighbors", type=int, default=30)
    parser.add_argument("--max-candidates-per-lr", type=int, default=90000)
    parser.add_argument("--target-profile", choices=("compact", "rich"), default="rich")
    parser.add_argument(
        "--reference-dataset-dir",
        default=None,
        help=(
            "Optional standardized dataset directory whose cells/candidates should be reused. "
            "This is useful for rebuilding a richer expression/target panel while preserving "
            "candidate_id alignment with existing prediction files."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_h5ad)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(input_path)
    lr_pairs = pd.DataFrame(DEFAULT_LR_PANEL)
    target_genes = RICH_TARGET_GENES if args.target_profile == "rich" else COMPACT_TARGET_GENES
    genes_needed = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | _marker_genes() | _target_gene_set(target_genes))
    genes_present = [gene for gene in genes_needed if gene in data.var_names]
    missing_lr_genes = sorted((set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])) - set(genes_present))
    if missing_lr_genes:
        raise ValueError(f"LR panel contains genes absent from SPAMOSAIC tonsil h5ad: {missing_lr_genes}")

    reference_dir = PROJECT_ROOT / args.reference_dataset_dir if args.reference_dataset_dir else None
    if reference_dir is not None:
        selected = _select_cells_from_reference(data, reference_dir)
        expression = _expression_table(data, selected, genes_present)
        cells = pd.read_csv(reference_dir / "cells.csv")
        candidates = pd.read_csv(reference_dir / "candidate_interactions.csv")
    else:
        selected = _select_cells(data, args.n_cells, args.seed)
        expression = _expression_table(data, selected, genes_present)
        protein = _protein_table(data, selected)
        cells = _cells_table(data, selected, expression, protein)
        candidates = _candidate_interactions(
            cells,
            expression,
            lr_pairs,
            max_neighbors=args.max_neighbors,
            max_candidates_per_lr=args.max_candidates_per_lr,
            seed=args.seed,
            candidate_prefix=args.dataset_id.replace("_subset", ""),
        )
    target_signatures = _target_signature_table(lr_pairs, genes_present, target_genes)

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)
    target_signatures.to_csv(output_dir / "target_signatures.csv", index=False)

    metadata = {
        "dataset_id": args.dataset_id,
        "platform": "SPAMOSAIC tonsil RNA+ADT h5ad",
        "source": _rel(input_path),
        "output_dir": _rel(output_dir),
        "n_cells": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "n_cells_requested": int(args.n_cells),
            "seed": int(args.seed),
            "rule": "balanced sample from ADT marker-derived B/T/Tfh/FDC-myeloid/stromal proxy niches",
        },
        "claim_boundary": (
            "Real SPAMOSAIC tonsil coordinates and RNA expression are preserved. cell_type, region, and niche are "
            "ADT marker-rule proxies, not curated histology labels or native CCC ground truth."
        ),
        "source_obsm": "protein_counts;spatial",
        "missing_lr_genes": missing_lr_genes,
        "target_profile": str(args.target_profile),
        "reference_dataset_dir": _rel(reference_dir) if reference_dir is not None else None,
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


def _select_cells(data: ad.AnnData, n_cells: int, seed: int) -> list[int]:
    protein = _protein_table(data, list(range(data.n_obs)))
    labels = _assign_labels(protein)
    priority = labels["niche"].isin(["germinal_center_proxy", "T_zone_proxy", "myeloid_stromal_proxy"])
    pool = np.where(priority.to_numpy())[0]
    n_cells = min(int(n_cells), data.n_obs)
    rng = np.random.default_rng(seed)
    if len(pool) < n_cells:
        pool = np.arange(data.n_obs)
    selected = rng.choice(pool, size=n_cells, replace=False) if len(pool) > n_cells else pool
    return [int(idx) for idx in np.sort(selected)]


def _select_cells_from_reference(data: ad.AnnData, reference_dir: Path) -> list[int]:
    reference_cells = pd.read_csv(reference_dir / "cells.csv", usecols=["cell_id"])
    index_by_cell = {str(cell_id): int(i) for i, cell_id in enumerate(data.obs_names)}
    missing = [str(cell_id) for cell_id in reference_cells["cell_id"] if str(cell_id) not in index_by_cell]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Reference cells absent from source h5ad: {preview}")
    return [index_by_cell[str(cell_id)] for cell_id in reference_cells["cell_id"]]


def _expression_table(data: ad.AnnData, selected: list[int], genes: list[str]) -> pd.DataFrame:
    gene_idx = [int(np.where(data.var_names == gene)[0][0]) for gene in genes]
    sub = data.X[selected, :][:, gene_idx]
    values = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    frame = pd.DataFrame(values, columns=genes)
    frame.insert(0, "cell_id", [str(data.obs_names[idx]) for idx in selected])
    return frame


def _protein_table(data: ad.AnnData, selected: list[int]) -> pd.DataFrame:
    protein = data.obsm["protein_counts"]
    values = protein[selected, :] if not isinstance(protein, pd.DataFrame) else protein.iloc[selected]
    if sparse.issparse(values):
        values = values.toarray()
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        features = data.uns["protein_features"]
        names = list(features["protein_name"]) if "protein_name" in features else list(features.index)
        frame = pd.DataFrame(np.asarray(values), columns=[str(name) for name in names])
    frame.index = [str(data.obs_names[idx]) for idx in selected]
    return frame


def _cells_table(data: ad.AnnData, selected: list[int], expression: pd.DataFrame, protein: pd.DataFrame) -> pd.DataFrame:
    coords = np.asarray(data.obsm["spatial"])[selected]
    labels = _assign_labels(protein)
    out = pd.DataFrame(
        {
            "cell_id": expression["cell_id"].to_numpy(),
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
            "cell_type": labels["cell_type"].to_numpy(),
            "region": labels["niche"].to_numpy(),
            "niche": labels["niche"].to_numpy(),
        }
    )
    for column in ["CD19", "MS4A1", "CD3E", "CD4", "CD8A", "CXCR5", "CCR7", "CD68", "CD14", "CR2", "VIM", "ACTA2"]:
        if column in protein:
            out[f"adt_{column}"] = protein[column].to_numpy(dtype=float)
    return out


def _assign_labels(protein: pd.DataFrame) -> pd.DataFrame:
    z = protein.copy()
    for column in z.columns:
        values = pd.to_numeric(z[column], errors="coerce").fillna(0.0)
        z[column] = (values - values.median()) / (values.std(ddof=0) + 1e-6)

    def score(*markers: str) -> pd.Series:
        present = [marker for marker in markers if marker in z]
        if not present:
            return pd.Series(0.0, index=z.index)
        return z[present].mean(axis=1)

    scores = pd.DataFrame(
        {
            "B_cell_proxy": score("CD19", "MS4A1", "PAX5"),
            "T_cell_proxy": score("CD3E", "CD4", "CD8A"),
            "Tfh_proxy": score("CD3E", "CD4", "CXCR5", "PDCD1"),
            "FDC_proxy": score("CR2", "CXCR5"),
            "myeloid_proxy": score("CD68", "CD14", "CD163", "ITGAM"),
            "stromal_proxy": score("VIM", "ACTA2"),
            "epithelial_proxy": score("EPCAM"),
        }
    )
    cell_type = scores.idxmax(axis=1)
    niche = cell_type.map(
        {
            "B_cell_proxy": "germinal_center_proxy",
            "Tfh_proxy": "germinal_center_proxy",
            "FDC_proxy": "germinal_center_proxy",
            "T_cell_proxy": "T_zone_proxy",
            "myeloid_proxy": "myeloid_stromal_proxy",
            "stromal_proxy": "myeloid_stromal_proxy",
            "epithelial_proxy": "epithelial_proxy",
        }
    ).fillna("unknown")
    return pd.DataFrame({"cell_type": cell_type.to_numpy(), "niche": niche.to_numpy()}, index=protein.index)


def _candidate_interactions(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
    candidate_prefix: str,
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
                neighbors = [neighbors[int(i)] for i in np.argsort(distances)[:max_neighbors]]
            pairs.extend((sender_idx, receiver_idx) for receiver_idx in neighbors)
        if len(pairs) > max_candidates_per_lr:
            keep = rng.choice(np.arange(len(pairs)), size=int(max_candidates_per_lr), replace=False)
            pairs = [pairs[int(i)] for i in np.sort(keep)]
        for sender_idx, receiver_idx in pairs:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            distance = float(np.linalg.norm(coords[int(sender_idx)] - coords[int(receiver_idx)]))
            active_niche = str(lr.active_niche)
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
                    "type_rule_satisfied": bool(_type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))),
                    "range_rule_satisfied": bool(distance <= float(lr.spatial_range)),
                    "niche_rule_satisfied": bool(
                        active_niche in {"any", "", "nan"}
                        or str(sender["niche"]) == active_niche
                        or str(receiver["niche"]) == active_niche
                    ),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates.insert(0, "candidate_id", [f"{candidate_prefix}_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name == "CXCL13-CXCR5":
        return sender_type in {"FDC_proxy", "B_cell_proxy"} and receiver_type in {"B_cell_proxy", "Tfh_proxy"}
    if control_name == "CCL21-CCR7":
        return sender_type in {"FDC_proxy", "stromal_proxy", "myeloid_proxy"} and receiver_type in {"T_cell_proxy", "Tfh_proxy"}
    if control_name == "CD40LG-CD40":
        return sender_type in {"T_cell_proxy", "Tfh_proxy"} and receiver_type in {"B_cell_proxy", "FDC_proxy", "myeloid_proxy"}
    if control_name == "IL21-IL21R":
        return sender_type == "Tfh_proxy" and receiver_type in {"B_cell_proxy", "T_cell_proxy"}
    if control_name == "TNFSF13B-TNFRSF13C":
        return sender_type in {"myeloid_proxy", "FDC_proxy", "stromal_proxy"} and receiver_type == "B_cell_proxy"
    return False


def _marker_genes() -> set[str]:
    return {
        "MS4A1",
        "CD19",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD68",
        "CD14",
        "CR2",
        "VIM",
        "ACTA2",
        "EPCAM",
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
                "target_source": "phase7_spamosaic_tonsil_curated_context_targets",
                "evidence_note": "Curated tonsil target-response proxy using genes present in the SPAMOSAIC RNA panel.",
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
