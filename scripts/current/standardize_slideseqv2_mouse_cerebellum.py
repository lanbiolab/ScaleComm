#!/usr/bin/env python
"""Standardize OpenProblems Slide-seqV2 mouse cerebellum for platform extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/slideseqv2_openproblems/mouse_cerebellum.h5ad"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "outputs/current_slideseqv2_extension/standardized_inputs/slideseqv2_mouse_cerebellum_subset"
)


LR_DESIGN = [
    {
        "ligand": "Efnb3",
        "receptor": "Ephb2",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 36.0,
        "active_niche": "neuronal_interface",
        "control_class": "axon_guidance_contact",
        "biological_category": "ephrin_contact",
        "selected_reason": "contact-dependent neuronal guidance prior",
        "sender_types": {"neuronal_cluster", "interneuron_cluster", "granule_cluster", "purkinje_cluster"},
        "receiver_types": {"neuronal_cluster", "interneuron_cluster", "granule_cluster", "purkinje_cluster"},
    },
    {
        "ligand": "Fgf1",
        "receptor": "Fgfr1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 90.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "growth_factor_paracrine",
        "biological_category": "growth_factor",
        "selected_reason": "FGF signaling axis present in cerebellum HVG panel",
        "sender_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster", "granule_cluster"},
        "receiver_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster", "granule_cluster"},
    },
    {
        "ligand": "Fgf9",
        "receptor": "Fgfr1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 90.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "growth_factor_paracrine",
        "biological_category": "growth_factor",
        "selected_reason": "FGF signaling axis present in cerebellum HVG panel",
        "sender_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster"},
        "receiver_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster", "granule_cluster"},
    },
    {
        "ligand": "Nrg2",
        "receptor": "Erbb4",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "neuronal_interface",
        "control_class": "neuronal_paracrine",
        "biological_category": "neuregulin",
        "selected_reason": "neuregulin-Erbb signaling axis present in cerebellum HVG panel",
        "sender_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster"},
        "receiver_types": {"interneuron_cluster", "neuronal_cluster", "purkinje_cluster"},
    },
    {
        "ligand": "Nrg3",
        "receptor": "Erbb4",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "neuronal_interface",
        "control_class": "neuronal_paracrine",
        "biological_category": "neuregulin",
        "selected_reason": "neuregulin-Erbb signaling axis present in cerebellum HVG panel",
        "sender_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster"},
        "receiver_types": {"interneuron_cluster", "neuronal_cluster", "purkinje_cluster"},
    },
    {
        "ligand": "Ptn",
        "receptor": "Ptprz1",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 185.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "matrix_growth_context",
        "biological_category": "pleiotrophin",
        "selected_reason": "context-dependent PTN-PTPRZ1 signaling axis present in panel",
        "sender_types": {"neuronal_cluster", "granule_cluster", "glial_cluster"},
        "receiver_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster"},
    },
    {
        "ligand": "Ptn",
        "receptor": "Sdc3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 185.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "matrix_growth_context",
        "biological_category": "pleiotrophin",
        "selected_reason": "context-dependent PTN-SDC3 signaling axis present in panel",
        "sender_types": {"neuronal_cluster", "granule_cluster", "glial_cluster"},
        "receiver_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster"},
    },
    {
        "ligand": "Mdk",
        "receptor": "Ptprz1",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 185.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "matrix_growth_context",
        "biological_category": "midkine",
        "selected_reason": "context-dependent MDK-PTPRZ1 signaling axis present in panel",
        "sender_types": {"glial_cluster", "neuronal_cluster", "granule_cluster"},
        "receiver_types": {"glial_cluster", "oligodendrocyte_cluster", "neuronal_cluster"},
    },
    {
        "ligand": "Reln",
        "receptor": "Vldlr",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 170.0,
        "active_niche": "neuronal_interface",
        "control_class": "neuronal_positioning_context",
        "biological_category": "reelin",
        "selected_reason": "Reelin receptor axis present in cerebellum HVG panel",
        "sender_types": {"interneuron_cluster", "neuronal_cluster", "purkinje_cluster"},
        "receiver_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster"},
    },
    {
        "ligand": "Reln",
        "receptor": "Lrp8",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 170.0,
        "active_niche": "neuronal_interface",
        "control_class": "neuronal_positioning_context",
        "biological_category": "reelin",
        "selected_reason": "Reelin receptor axis present in cerebellum HVG panel",
        "sender_types": {"interneuron_cluster", "neuronal_cluster", "purkinje_cluster"},
        "receiver_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster"},
    },
    {
        "ligand": "Apoe",
        "receptor": "Lrp1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "lipid_glial_context",
        "biological_category": "apolipoprotein",
        "selected_reason": "glial-neuronal lipid signaling proxy present in panel",
        "sender_types": {"glial_cluster", "oligodendrocyte_cluster"},
        "receiver_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster", "glial_cluster"},
    },
    {
        "ligand": "App",
        "receptor": "Lrp1",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "neuronal_interface",
        "control_class": "neuronal_paracrine",
        "biological_category": "app_lrp",
        "selected_reason": "APP-LRP1 signaling/protein-interaction proxy present in panel",
        "sender_types": {"neuronal_cluster", "granule_cluster", "purkinje_cluster"},
        "receiver_types": {"neuronal_cluster", "glial_cluster", "oligodendrocyte_cluster"},
    },
    {
        "ligand": "Gdf10",
        "receptor": "Bmpr1a",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 120.0,
        "active_niche": "glial_neuronal_interface",
        "control_class": "bmp_context",
        "biological_category": "bmp",
        "selected_reason": "BMP-family axis present in panel",
        "sender_types": {"glial_cluster", "neuronal_cluster"},
        "receiver_types": {"glial_cluster", "neuronal_cluster", "granule_cluster"},
    },
    {
        "ligand": "Mbp",
        "receptor": "Erbb4",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 120.0,
        "active_niche": "any",
        "control_class": "high_expression_decoy",
        "biological_category": "high_expression_negative_risk",
        "selected_reason": "high-expression non-LR decoy",
        "sender_types": {"oligodendrocyte_cluster"},
        "receiver_types": {"interneuron_cluster", "neuronal_cluster"},
    },
    {
        "ligand": "Pcp2",
        "receptor": "Fgfr1",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 120.0,
        "active_niche": "any",
        "control_class": "high_expression_decoy",
        "biological_category": "high_expression_negative_risk",
        "selected_reason": "high-expression non-LR decoy",
        "sender_types": {"purkinje_cluster"},
        "receiver_types": {"neuronal_cluster", "glial_cluster"},
    },
]

MARKER_GENES = [
    "Aldoc",
    "Apoe",
    "Aqp4",
    "Calb1",
    "Car8",
    "Gabra6",
    "Gad1",
    "Gad2",
    "Grid2",
    "Mbp",
    "Mog",
    "Neurod1",
    "Olig1",
    "Pcp2",
    "Plp1",
    "Reln",
    "Slc1a3",
    "Slc17a7",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--max-neighbors", type=int, default=72)
    parser.add_argument("--max-candidates-per-lr", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    input_path = resolve(args.input_h5ad)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    lr_pairs = build_lr_pairs(adata)
    selected = select_spatial_subset(adata, int(args.subset_size), int(args.seed))
    genes = expression_genes(adata, lr_pairs)
    expression = expression_table(adata, selected, genes)
    cells = cells_table(adata, selected, expression)
    candidates = candidate_interactions(
        cells,
        expression,
        lr_pairs,
        max_neighbors=int(args.max_neighbors),
        max_candidates_per_lr=int(args.max_candidates_per_lr),
        seed=int(args.seed),
    )

    cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)

    metadata = {
        "dataset_id": "slideseqv2_mouse_cerebellum_subset",
        "platform": "Slide-seqV2 mouse cerebellum",
        "source": rel(input_path),
        "openproblems_dataset_id": adata.uns.get("dataset_id", "zenodo_spatial/slideseqv2/mouse_cerebellum"),
        "source_url": adata.uns.get("dataset_url", ""),
        "output_dir": rel(output_dir),
        "n_beads": int(len(cells)),
        "n_source_beads": int(adata.n_obs),
        "n_source_genes": int(adata.n_vars),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "subset_size_requested": int(args.subset_size),
            "seed": int(args.seed),
            "rule": "spatially contiguous region around a high-quality bead selected from real Slide-seqV2 coordinates",
        },
        "label_source": (
            "cell_type and niche are coarse proxies derived from marker expression and leiden clusters in the "
            "OpenProblems processed AnnData. They are used for context annotation, not native cell-type ground truth."
        ),
        "claim_boundary": (
            "Real Slide-seqV2 coordinates and expression are preserved. LR axes are literature/plausibility priors "
            "restricted to genes present in this high-variable-gene AnnData. This run supports cross-platform "
            "execution and interpretable spatial CCC maps, not perturbation ground truth."
        ),
        "missing_requested_lr_genes": sorted(set(gene for spec in LR_DESIGN for gene in (spec["ligand"], spec["receptor"])) - set(adata.var_names)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": metadata["dataset_id"],
                "platform": metadata["platform"],
                "n_cells": metadata["n_beads"],
                "n_expression_genes": metadata["n_expression_genes"],
                "n_lr_pairs": metadata["n_lr_pairs"],
                "n_candidate_interactions": metadata["n_candidate_interactions"],
                "output_dir": metadata["output_dir"],
                "claim_boundary": metadata["claim_boundary"],
            }
        ]
    ).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(metadata, indent=2))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_lr_pairs(adata) -> pd.DataFrame:
    genes = set(map(str, adata.var_names))
    rows = []
    for i, spec in enumerate(LR_DESIGN):
        if spec["ligand"] not in genes or spec["receptor"] not in genes:
            continue
        row = {
            "lr_id": f"slideseqv2_cb_lr_{i:03d}_{spec['ligand']}_{spec['receptor']}",
            "ligand": spec["ligand"],
            "receptor": spec["receptor"],
            "mechanism": spec["mechanism"],
            "spatial_range": float(spec["spatial_range"]),
            "active_niche": spec["active_niche"],
            "control_name": f"{spec['ligand']}-{spec['receptor']}",
            "control_class": spec["control_class"],
            "false_positive_high_expression": spec["mechanism"] == "negative_control_high_expression",
            "true_lr": spec["mechanism"] != "negative_control_high_expression",
            "biological_category": spec["biological_category"],
            "selected_reason": spec["selected_reason"],
            "sender_types": "|".join(sorted(spec["sender_types"])),
            "receiver_types": "|".join(sorted(spec["receiver_types"])),
        }
        rows.append(row)
    if not rows:
        raise ValueError("No LR pairs survived gene-presence filtering.")
    return pd.DataFrame(rows)


def select_spatial_subset(adata, subset_size: int, seed: int) -> np.ndarray:
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    n = min(int(subset_size), coords.shape[0])
    quality = np.asarray(adata.obs["n_genes_by_counts"], dtype=float)
    candidates = np.flatnonzero(quality >= np.quantile(quality, 0.65))
    rng = np.random.default_rng(seed)
    center_idx = int(rng.choice(candidates))
    distance = np.linalg.norm(coords - coords[center_idx], axis=1)
    selected = np.argsort(distance)[:n]
    return np.sort(selected).astype(int)


def expression_genes(adata, lr_pairs: pd.DataFrame) -> list[str]:
    genes = set(map(str, adata.var_names))
    requested = set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]) | set(MARKER_GENES)
    present = [gene for gene in sorted(requested) if gene in genes]
    return present


def expression_table(adata, selected: np.ndarray, genes: list[str]) -> pd.DataFrame:
    gene_idx = [int(np.where(adata.var_names == gene)[0][0]) for gene in genes]
    matrix = adata.layers["normalized"] if "normalized" in adata.layers else adata.X
    sub = matrix[selected, :][:, gene_idx]
    values = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame(values, columns=genes)
    out.insert(0, "cell_id", [str(adata.obs_names[i]) for i in selected])
    return out


def cells_table(adata, selected: np.ndarray, expression: pd.DataFrame) -> pd.DataFrame:
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[selected]
    expr = expression.set_index("cell_id")
    cell_types = [assign_cell_type(expr.loc[cell_id]) for cell_id in expression["cell_id"]]
    niches = [assign_niche(cell_type) for cell_type in cell_types]
    cells = pd.DataFrame(
        {
            "cell_id": expression["cell_id"].to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cell_type": cell_types,
            "region": niches,
            "niche": niches,
            "leiden": adata.obs.iloc[selected]["leiden"].astype(str).to_numpy(),
            "total_counts": adata.obs.iloc[selected]["total_counts"].to_numpy(dtype=float),
            "n_genes_by_counts": adata.obs.iloc[selected]["n_genes_by_counts"].to_numpy(dtype=float),
        }
    )
    return cells


def assign_cell_type(row: pd.Series) -> str:
    markers = {
        "purkinje_cluster": ["Pcp2", "Calb1", "Car8", "Grid2"],
        "granule_cluster": ["Gabra6", "Neurod1"],
        "interneuron_cluster": ["Gad1", "Gad2", "Reln"],
        "oligodendrocyte_cluster": ["Mbp", "Mog", "Plp1", "Olig1"],
        "glial_cluster": ["Aldoc", "Aqp4", "Slc1a3", "Apoe"],
        "neuronal_cluster": ["Slc17a7", "Reln", "Grid2"],
    }
    scores = {}
    for label, genes in markers.items():
        present = [gene for gene in genes if gene in row.index]
        scores[label] = float(row[present].mean()) if present else 0.0
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "unassigned_cluster"
    return best


def assign_niche(cell_type: str) -> str:
    if cell_type in {"oligodendrocyte_cluster", "glial_cluster"}:
        return "glial_neuronal_interface"
    if cell_type in {"purkinje_cluster", "granule_cluster", "interneuron_cluster", "neuronal_cluster"}:
        return "neuronal_interface"
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
        sender_allowed = set(str(lr.sender_types).split("|")) if hasattr(lr, "sender_types") else set()
        receiver_allowed = set(str(lr.receiver_types).split("|")) if hasattr(lr, "receiver_types") else set()
        pairs = []
        for sender_idx, neighbors in enumerate(tree.query_ball_point(coords, r=float(lr.spatial_range))):
            neighbors = [idx for idx in neighbors if idx != sender_idx]
            if len(neighbors) > max_neighbors:
                distances = np.linalg.norm(coords[neighbors] - coords[sender_idx], axis=1)
                order = np.argsort(distances)[:max_neighbors]
                neighbors = [neighbors[int(i)] for i in order]
            pairs.extend((sender_idx, receiver_idx) for receiver_idx in neighbors)
        if len(pairs) > max_candidates_per_lr:
            keep = rng.choice(np.arange(len(pairs)), size=int(max_candidates_per_lr), replace=False)
            pairs = [pairs[int(i)] for i in np.sort(keep)]
        for sender_idx, receiver_idx in pairs:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            distance = float(np.linalg.norm(coords[int(sender_idx)] - coords[int(receiver_idx)]))
            active_niche = str(lr.active_niche)
            sender_type = str(sender["cell_type"])
            receiver_type = str(receiver["cell_type"])
            sender_niche = str(sender["niche"])
            receiver_niche = str(receiver["niche"])
            type_ok = (sender_type in sender_allowed) and (receiver_type in receiver_allowed)
            niche_ok = active_niche in {"any", "", "nan"} or sender_niche == active_niche or receiver_niche == active_niche
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
                    "spatial_range": float(lr.spatial_range),
                    "active_niche": active_niche,
                    "distance": distance,
                    "sender_cell_type": sender_type,
                    "receiver_cell_type": receiver_type,
                    "sender_niche": sender_niche,
                    "receiver_niche": receiver_niche,
                    "weak_signal": str(lr.control_class) in {"matrix_growth_context", "neuronal_positioning_context"},
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "type_rule_satisfied": bool(type_ok),
                    "range_rule_satisfied": distance <= float(lr.spatial_range),
                    "niche_rule_satisfied": bool(niche_ok),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)),
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("Candidate construction produced no edges.")
    candidates.insert(0, "candidate_id", [f"slideseqv2_cb_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
