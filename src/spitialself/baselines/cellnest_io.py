"""CellNEST input preparation for generated CCC benchmark replicates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def prepare_cellnest_inputs(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    output_dir: Path,
    *,
    dataset_name: str,
) -> dict:
    """Write h5ad, LR database, and annotation files consumable by CellNEST."""

    _patch_numpy2_legacy_aliases()

    _check_columns(cells, {"cell_id", "x", "y"}, "cells")
    _check_columns(expression, {"cell_id"}, "expression")
    _check_columns(lr_pairs, {"ligand", "receptor"}, "lr_pairs")

    ad = __import__("anndata")

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_order = cells["cell_id"].astype(str).to_list()
    expr = expression.set_index("cell_id").reindex(cell_order)
    if expr.isnull().any().any():
        missing = expr.index[expr.isnull().any(axis=1)].to_list()[:5]
        raise ValueError(f"Expression table missing cells present in cells.csv: {missing}")

    coords = cells.set_index("cell_id").loc[cell_order, ["x", "y"]].to_numpy(dtype=float)
    adata = ad.AnnData(
        X=sparse.csr_matrix(expr.to_numpy(dtype=float)),
        obs=pd.DataFrame(index=cell_order),
        var=pd.DataFrame(index=[str(gene).upper() for gene in expr.columns]),
    )
    adata.obsm["spatial"] = coords
    h5ad_path = output_dir / f"{dataset_name}.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")

    database = _cellnest_database(lr_pairs)
    database_path = output_dir / "cellnest_lr_database.csv"
    database.to_csv(database_path, index=False)

    annotation_path = output_dir / "cellnest_celltype_annotation.csv"
    cell_type_col = "cell_type" if "cell_type" in cells.columns else None
    if cell_type_col:
        annotation = cells[["cell_id", cell_type_col]].rename(
            columns={"cell_id": "Barcode", cell_type_col: "Type"}
        )
        annotation.to_csv(annotation_path, index=False)
    else:
        annotation_path = None

    command_path = output_dir / "cellnest_preprocess_command.sh"
    command = (
        "python /home/xd/y/BenchmarkBaselines/CellNEST/data_preprocess_CellNEST.py "
        f"--data_name='{dataset_name}' "
        f"--data_from='{h5ad_path}' "
        "--data_type=anndata "
        f"--database_path='{database_path}' "
        f"--data_to='{output_dir / 'input_graph' / dataset_name}/' "
        f"--metadata_to='{output_dir / 'metadata' / dataset_name}/' "
        "--skip_normalize=1 "
        "--filter_min_cell=1 "
        "--threshold_gene_exp=90 "
        "--distance_measure=knn "
        "--k=12"
    )
    if annotation_path is not None:
        command += f" --use_celltype=1 --celltype_path='{annotation_path}'"
    command_path.write_text(command + "\n", encoding="utf-8")

    return {
        "dataset_name": dataset_name,
        "h5ad_path": str(h5ad_path),
        "database_path": str(database_path),
        "annotation_path": str(annotation_path) if annotation_path is not None else "",
        "preprocess_command_path": str(command_path),
        "n_cells": int(len(cells)),
        "n_genes": int(expr.shape[1]),
        "n_lr_pairs": int(len(database)),
        "status": "input_ready_preprocess_required",
    }


def _cellnest_database(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in lr_pairs.drop_duplicates(subset=["ligand", "receptor"]).itertuples(index=False):
        ligand = str(getattr(row, "ligand")).upper()
        receptor = str(getattr(row, "receptor")).upper()
        mechanism = str(getattr(row, "mechanism", ""))
        annotation = "Cell-Cell Contact" if "contact" in mechanism or "juxtacrine" in mechanism else "Secreted Signaling"
        rows.append(
            {
                "Ligand": ligand,
                "Receptor": receptor,
                "Annotation": annotation,
                "Reference": "SpatialSelf_Phase1_Benchmark",
            }
        )
    return pd.DataFrame(rows)


def _check_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} table missing columns: {sorted(missing)}")


def _patch_numpy2_legacy_aliases() -> None:
    """Provide old aliases required by older anndata/h5ad dependency stacks."""

    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
