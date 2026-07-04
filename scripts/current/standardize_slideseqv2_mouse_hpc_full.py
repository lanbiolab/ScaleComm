#!/usr/bin/env python
"""Standardize full-gene STexampleData Slide-seqV2 mouse hippocampus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RDS = PROJECT_ROOT / "data/slideseqv2_mouse_hpc_stexample/SlideSeqV2_mouseHPC.rds"
DEFAULT_OUT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
DEFAULT_RSCRIPT = Path("/home/xd/.conda/envs/Benchmark/bin/Rscript")

LR_DESIGN = [
    ("Efnb3", "Ephb2", "juxtacrine_contact", 42.0, "neuronal_interface", "ephrin_contact", "axon guidance/contact"),
    ("Dll1", "Notch1", "juxtacrine_contact", 42.0, "neurogenic_interface", "notch_contact", "Notch contact-dependent axis"),
    ("Jag1", "Notch2", "juxtacrine_contact", 42.0, "neurogenic_interface", "notch_contact", "Notch contact-dependent axis"),
    ("Nrg1", "Erbb4", "short_medium_paracrine", 120.0, "neuronal_interface", "neuregulin", "neuregulin-Erbb interneuron axis"),
    ("Nrg2", "Erbb4", "short_medium_paracrine", 120.0, "neuronal_interface", "neuregulin", "neuregulin-Erbb interneuron axis"),
    ("Nrg3", "Erbb4", "short_medium_paracrine", 120.0, "neuronal_interface", "neuregulin", "neuregulin-Erbb interneuron axis"),
    ("Fgf1", "Fgfr1", "short_medium_paracrine", 115.0, "glial_neuronal_interface", "growth_factor", "FGF growth-factor signaling"),
    ("Fgf2", "Fgfr1", "short_medium_paracrine", 115.0, "glial_neuronal_interface", "growth_factor", "FGF growth-factor signaling"),
    ("Fgf9", "Fgfr1", "short_medium_paracrine", 115.0, "glial_neuronal_interface", "growth_factor", "FGF growth-factor signaling"),
    ("Reln", "Vldlr", "long_range_chemokine_niche", 220.0, "neuronal_interface", "reelin", "Reelin spatial positioning axis"),
    ("Reln", "Lrp8", "long_range_chemokine_niche", 220.0, "neuronal_interface", "reelin", "Reelin spatial positioning axis"),
    ("Ptn", "Ptprz1", "long_range_chemokine_niche", 240.0, "glial_neuronal_interface", "pleiotrophin", "PTN glial-neuronal context axis"),
    ("Mdk", "Ptprz1", "long_range_chemokine_niche", 240.0, "glial_neuronal_interface", "midkine", "MDK glial-neuronal context axis"),
    ("Cxcl12", "Cxcr4", "long_range_chemokine_niche", 260.0, "vascular_neurogenic_interface", "chemokine", "CXCL12-CXCR4 niche axis"),
    ("Vegfa", "Kdr", "long_range_chemokine_niche", 240.0, "vascular_neurogenic_interface", "vascular", "VEGFA vascular niche axis"),
    ("Vegfa", "Flt1", "long_range_chemokine_niche", 240.0, "vascular_neurogenic_interface", "vascular", "VEGFA vascular niche axis"),
    ("Sema3a", "Nrp1", "long_range_chemokine_niche", 220.0, "neuronal_interface", "semaphorin", "semaphorin guidance axis"),
    ("Sema3a", "Nrp2", "long_range_chemokine_niche", 220.0, "neuronal_interface", "semaphorin", "semaphorin guidance axis"),
    ("Apoe", "Lrp1", "short_medium_paracrine", 130.0, "glial_neuronal_interface", "apolipoprotein", "glial-neuronal lipid signaling"),
    ("App", "Lrp1", "short_medium_paracrine", 130.0, "neuronal_interface", "app_lrp", "APP-LRP1 signaling proxy"),
    ("Mbp", "Erbb4", "negative_control_high_expression", 120.0, "any", "high_expression_decoy", "high-expression invalid LR decoy"),
    ("Pcp2", "Fgfr1", "negative_control_high_expression", 120.0, "any", "high_expression_decoy", "high-expression invalid LR decoy"),
    ("Gfap", "Cxcr4", "negative_control_high_expression", 160.0, "any", "high_expression_decoy", "high-expression invalid LR decoy"),
]

GENES = sorted(
    {
        gene
        for ligand, receptor, *_ in LR_DESIGN
        for gene in (ligand, receptor)
    }
    | {
        "Aqp4",
        "Apoe",
        "Calb1",
        "Cldn5",
        "Cx3cr1",
        "Gad1",
        "Gad2",
        "Gfap",
        "Mbp",
        "Mog",
        "Olig1",
        "Olig2",
        "Pcp2",
        "Pdgfra",
        "Pecam1",
        "Plp1",
        "Prox1",
        "Reln",
        "Slc1a2",
        "Slc17a7",
        "Sox10",
        "Tbr1",
    }
)

CELL_TYPE_MAP = {
    "Astrocyte": "astrocyte",
    "CA1": "CA1_neuron",
    "CA3": "CA3_neuron",
    "Cajal_Retzius": "cajal_retzius",
    "Choroid": "choroid",
    "Denate": "dentate_neuron",
    "Endothelial_Stalk": "endothelial",
    "Endothelial_Tip": "endothelial",
    "Entorihinal": "entorhinal_neuron",
    "Ependymal": "ependymal",
    "Interneuron": "interneuron",
    "Microglia_Macrophages": "microglia_macrophage",
    "Mural": "mural",
    "Neurogenesis": "neurogenic",
    "Neuron.Slc17a6": "excitatory_neuron",
    "Oligodendrocyte": "oligodendrocyte",
    "Polydendrocyte": "polydendrocyte",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rds", default=str(DEFAULT_RDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--rscript", default=str(DEFAULT_RSCRIPT))
    parser.add_argument("--subset-size", type=int, default=6000)
    parser.add_argument("--use-all", action="store_true", help="Retain all Slide-seqV2 beads instead of selecting a subset.")
    parser.add_argument("--max-neighbors", type=int, default=72)
    parser.add_argument("--max-candidates-per-lr", type=int, default=22000)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_dir = output_dir / "_r_export"
    exported_dir.mkdir(exist_ok=True)

    export_r_tables(
        rscript=Path(args.rscript),
        input_rds=resolve(args.input_rds),
        output_dir=exported_dir,
        genes=GENES,
    )
    cells_raw = pd.read_csv(exported_dir / "cells_raw.csv")
    expression_raw = pd.read_csv(exported_dir / "expression_raw.csv")
    source_summary = json.loads((exported_dir / "source_summary.json").read_text(encoding="utf-8"))

    lr_pairs = build_lr_pairs(expression_raw.columns)
    if args.use_all:
        selected_ids = cells_raw["cell_id"].astype(str).tolist()
        selection_rule = "all Slide-seqV2 beads retained; no spatial or annotation subset selection"
    else:
        selected_ids = select_annotated_spatial_subset(cells_raw, expression_raw, int(args.subset_size), int(args.seed))
        selection_rule = "spatially contiguous subset centered on annotated, high-library-size Slide-seqV2 beads"
    cells = cells_raw[cells_raw["cell_id"].isin(selected_ids)].copy()
    cells = cells.set_index("cell_id").loc[selected_ids].reset_index()
    expression = expression_raw[expression_raw["cell_id"].isin(selected_ids)].copy()
    expression = expression.set_index("cell_id").loc[selected_ids].reset_index()

    cells = annotate_cells(cells, expression)
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
        "dataset_id": "slideseqv2_mouse_hpc_full" if args.use_all else "slideseqv2_mouse_hpc_annotated_subset",
        "platform": "Slide-seqV2 mouse hippocampus",
        "source": rel(resolve(args.input_rds)),
        "source_reference": "STexampleData::SlideSeqV2_mouseHPC; Stickels et al. 2020; Cable et al. 2021 cell type annotations",
        "output_dir": rel(output_dir),
        "n_source_genes": int(source_summary["n_genes"]),
        "n_source_beads": int(source_summary["n_beads"]),
        "n_beads": int(len(cells)),
        "n_annotated_beads": int(cells["native_celltype"].notna().sum()),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "selection": {
            "subset_size_requested": None if args.use_all else int(args.subset_size),
            "use_all": bool(args.use_all),
            "seed": int(args.seed),
            "rule": selection_rule,
        },
        "label_source": "Native celltype column from STexampleData/Cable et al. plus coarse niche mapping used for SpatialSelf context features.",
        "claim_boundary": (
            "Full-gene Slide-seqV2 counts, spatial coordinates, and published cell type annotations are preserved. "
            "LR axes are literature/context priors for real-data platform extension, not perturbation ground truth."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([metadata]).to_csv(output_dir / "standardization_summary.csv", index=False)
    print(json.dumps(metadata, indent=2))


def export_r_tables(*, rscript: Path, input_rds: Path, output_dir: Path, genes: list[str]) -> None:
    gene_literal = "c(" + ",".join(f'"{g}"' for g in genes) + ")"
    script = f"""
    suppressPackageStartupMessages({{
      library(SpatialExperiment)
      library(SingleCellExperiment)
      library(Matrix)
      library(jsonlite)
    }})
    spe <- readRDS({json.dumps(str(input_rds))})
    genes_requested <- {gene_literal}
    gene_names <- as.character(rowData(spe)$gene_name)
    genes_present <- intersect(genes_requested, gene_names)
    idx <- match(genes_present, gene_names)
    counts <- assay(spe, "counts")[idx, , drop=FALSE]
    lib <- Matrix::colSums(assay(spe, "counts"))
    norm <- t(t(counts) / pmax(lib, 1)) * 10000
    norm <- log1p(norm)
    expr <- as.data.frame(t(as.matrix(norm)), check.names=FALSE)
    expr <- cbind(cell_id=colnames(spe), expr)
    coords <- spatialCoords(spe)
    cells <- as.data.frame(colData(spe))
    cells$cell_id <- colnames(spe)
    cells$x <- coords[,1]
    cells$y <- coords[,2]
    cells$total_counts <- as.numeric(lib)
    cells$n_genes_by_counts <- Matrix::colSums(assay(spe, "counts") > 0)
    if (!"celltype" %in% colnames(cells)) cells$celltype <- NA
    cells <- cells[, c("cell_id", "x", "y", "celltype", "total_counts", "n_genes_by_counts")]
    write.csv(cells, file=file.path({json.dumps(str(output_dir))}, "cells_raw.csv"), row.names=FALSE, quote=TRUE)
    write.csv(expr, file=file.path({json.dumps(str(output_dir))}, "expression_raw.csv"), row.names=FALSE, quote=TRUE)
    writeLines(jsonlite::toJSON(list(
      n_genes=nrow(spe),
      n_beads=ncol(spe),
      genes_requested=genes_requested,
      genes_present=genes_present,
      genes_missing=setdiff(genes_requested, genes_present)
    ), auto_unbox=TRUE, pretty=TRUE), file.path({json.dumps(str(output_dir))}, "source_summary.json"))
    """
    script_path = output_dir / "export_slideseqv2_hpc_tables.R"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([str(rscript), str(script_path)], check=True, cwd=PROJECT_ROOT)


def build_lr_pairs(columns: pd.Index) -> pd.DataFrame:
    genes_present = set(columns) - {"cell_id"}
    rows = []
    for i, (ligand, receptor, mechanism, spatial_range, active_niche, category, reason) in enumerate(LR_DESIGN):
        if ligand not in genes_present or receptor not in genes_present:
            continue
        rows.append(
            {
                "lr_id": f"slideseqv2_hpc_lr_{i:03d}_{ligand}_{receptor}",
                "ligand": ligand,
                "receptor": receptor,
                "mechanism": mechanism,
                "spatial_range": float(spatial_range),
                "active_niche": active_niche,
                "control_name": f"{ligand}-{receptor}",
                "control_class": category,
                "false_positive_high_expression": mechanism == "negative_control_high_expression",
                "true_lr": mechanism != "negative_control_high_expression",
                "biological_category": category,
                "selected_reason": reason,
            }
        )
    if not rows:
        raise ValueError("No LR pairs remain after gene filtering.")
    return pd.DataFrame(rows)


def select_annotated_spatial_subset(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    subset_size: int,
    seed: int,
) -> list[str]:
    merged = cells.merge(expression[["cell_id", "Cxcl12", "Cxcr4", "Nrg1", "Erbb4", "Reln", "Ptn"]], on="cell_id", how="left")
    annotated = merged["celltype"].notna() & merged["celltype"].astype(str).ne("")
    signal = (
        np.log1p(merged[["Cxcl12", "Nrg1", "Erbb4", "Reln", "Ptn"]].fillna(0).to_numpy(dtype=float)).sum(axis=1)
        + np.log1p(merged["total_counts"].to_numpy(dtype=float)) * 0.15
        + annotated.to_numpy(dtype=float) * 2.0
    )
    pool = np.flatnonzero(annotated.to_numpy())
    if len(pool) == 0:
        pool = np.arange(len(merged))
    rng = np.random.default_rng(seed)
    top_pool = pool[np.argsort(signal[pool])[::-1][: min(max(300, subset_size), len(pool))]]
    center_idx = int(rng.choice(top_pool[: min(80, len(top_pool))]))
    coords = merged[["x", "y"]].to_numpy(dtype=float)
    distance = np.linalg.norm(coords - coords[center_idx], axis=1)
    selected = np.argsort(distance)[: min(int(subset_size), len(merged))]
    selected = sorted(selected, key=lambda idx: distance[idx])
    return merged.iloc[selected]["cell_id"].astype(str).tolist()


def annotate_cells(cells: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    cells = cells.rename(columns={"celltype": "native_celltype"})
    cells["cell_type"] = cells["native_celltype"].map(CELL_TYPE_MAP).fillna("unannotated")
    cells["region"] = cells["cell_type"].map(region_for_cell_type).fillna("unannotated")
    cells["niche"] = cells["cell_type"].map(niche_for_cell_type).fillna("unknown")
    expr = expression.set_index("cell_id")
    for gene in ["Cxcl12", "Cxcr4", "Nrg1", "Erbb4", "Reln", "Ptn", "Gfap", "Mbp", "Slc17a7", "Gad1"]:
        if gene in expr.columns:
            cells[f"{gene}_expr"] = expr.loc[cells["cell_id"], gene].to_numpy(dtype=float)
    return cells


def region_for_cell_type(cell_type: str) -> str:
    if cell_type in {"CA1_neuron", "CA3_neuron", "dentate_neuron", "entorhinal_neuron"}:
        return cell_type.replace("_neuron", "")
    if cell_type in {"interneuron", "excitatory_neuron", "cajal_retzius"}:
        return "neuronal_region"
    if cell_type in {"astrocyte", "oligodendrocyte", "polydendrocyte", "microglia_macrophage"}:
        return "glial_region"
    if cell_type in {"endothelial", "mural", "neurogenic", "ependymal", "choroid"}:
        return "vascular_neurogenic_region"
    return "unannotated"


def niche_for_cell_type(cell_type: str) -> str:
    if cell_type in {"CA1_neuron", "CA3_neuron", "dentate_neuron", "entorhinal_neuron", "interneuron", "excitatory_neuron", "cajal_retzius"}:
        return "neuronal_interface"
    if cell_type in {"astrocyte", "oligodendrocyte", "polydendrocyte", "microglia_macrophage"}:
        return "glial_neuronal_interface"
    if cell_type in {"endothelial", "mural", "neurogenic", "ependymal", "choroid"}:
        return "vascular_neurogenic_interface"
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
    expr = expression.set_index("cell_id").loc[cells["cell_id"]]
    expr_values = expr.to_numpy(dtype=float)
    gene_to_col = {gene: idx for idx, gene in enumerate(expr.columns)}
    cell_ids = cells["cell_id"].astype(str).to_numpy()
    cell_type = cells["cell_type"].astype(str).to_numpy()
    niche = cells["niche"].astype(str).to_numpy()
    k = min(max(int(max_neighbors) + 1, 2), len(cells))
    nn_distance, nn_index = tree.query(coords, k=k)
    if nn_index.ndim == 1:
        nn_index = nn_index[:, None]
        nn_distance = nn_distance[:, None]
    rows = []
    for lr in lr_pairs.itertuples(index=False):
        pair_src = []
        pair_dst = []
        pair_dist = []
        radius = float(lr.spatial_range)
        for sender_idx in range(len(cells)):
            neighbors = nn_index[sender_idx]
            distances = nn_distance[sender_idx]
            keep = (neighbors != sender_idx) & np.isfinite(distances) & (distances <= radius)
            if not bool(np.any(keep)):
                continue
            kept_neighbors = neighbors[keep][:max_neighbors]
            kept_distances = distances[keep][:max_neighbors]
            pair_src.extend([sender_idx] * len(kept_neighbors))
            pair_dst.extend([int(idx) for idx in kept_neighbors])
            pair_dist.extend([float(d) for d in kept_distances])
        pairs = list(zip(pair_src, pair_dst, pair_dist))
        if len(pairs) > max_candidates_per_lr:
            keep = rng.choice(np.arange(len(pairs)), size=max_candidates_per_lr, replace=False)
            pairs = [pairs[int(i)] for i in np.sort(keep)]
        ligand_col = gene_to_col[str(lr.ligand)]
        receptor_col = gene_to_col[str(lr.receptor)]
        for sender_idx, receiver_idx, distance in pairs:
            active_niche = str(lr.active_niche)
            sender_niche = niche[int(sender_idx)]
            receiver_niche = niche[int(receiver_idx)]
            niche_ok = active_niche in {"any", "", "nan"} or sender_niche == active_niche or receiver_niche == active_niche
            sender_type = cell_type[int(sender_idx)]
            receiver_type = cell_type[int(receiver_idx)]
            type_ok = type_rule(str(lr.biological_category), sender_type, receiver_type)
            ligand_expr = float(expr_values[int(sender_idx), ligand_col])
            receptor_expr = float(expr_values[int(receiver_idx), receptor_col])
            rows.append(
                {
                    "sender_id": cell_ids[int(sender_idx)],
                    "receiver_id": cell_ids[int(receiver_idx)],
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
                    "weak_signal": str(lr.biological_category) in {"chemokine", "pleiotrophin", "midkine", "semaphorin"},
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
    candidates.insert(0, "candidate_id", [f"slideseqv2_hpc_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def type_rule(category: str, sender_type: str, receiver_type: str) -> bool:
    neuronal = {"CA1_neuron", "CA3_neuron", "dentate_neuron", "entorhinal_neuron", "interneuron", "excitatory_neuron", "cajal_retzius"}
    glial = {"astrocyte", "oligodendrocyte", "polydendrocyte", "microglia_macrophage"}
    vascular = {"endothelial", "mural", "neurogenic", "ependymal", "choroid"}
    if category in {"ephrin_contact", "notch_contact", "neuregulin", "reelin", "semaphorin", "app_lrp"}:
        return sender_type in neuronal and receiver_type in neuronal
    if category in {"growth_factor", "pleiotrophin", "midkine", "apolipoprotein"}:
        return sender_type in (neuronal | glial) and receiver_type in (neuronal | glial)
    if category in {"chemokine", "vascular"}:
        return sender_type in (glial | vascular | neuronal) and receiver_type in (vascular | neuronal | glial)
    if category == "high_expression_decoy":
        return False
    return True


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
