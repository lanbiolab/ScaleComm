"""COMMOT adapter for generated CCC benchmark replicates."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def score_commot(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    dis_thr: float | None = None,
    database_name: str = "phase1",
    cot_eps_p: float = 0.1,
    cot_nitermax: int = 2000,
) -> pd.DataFrame:
    """Run COMMOT and project its edge scores onto benchmark candidates."""

    _patch_numpy2_legacy_aliases()

    required_cells = {"cell_id", "x", "y"}
    required_expression = {"cell_id"}
    required_lr = {"ligand", "receptor"}
    required_candidates = {"candidate_id", "sender_id", "receiver_id", "ligand", "receptor"}
    _check_columns(cells, required_cells, "cells")
    _check_columns(expression, required_expression, "expression")
    _check_columns(lr_pairs, required_lr, "lr_pairs")
    _check_columns(candidates, required_candidates, "candidates")

    ad = importlib.import_module("anndata")
    ct = importlib.import_module("commot")

    cell_order = cells["cell_id"].astype(str).to_list()
    expr = expression.set_index("cell_id").reindex(cell_order)
    if expr.isnull().any().any():
        missing = expr.index[expr.isnull().any(axis=1)].to_list()[:5]
        raise ValueError(f"Expression table missing cells present in cells.csv: {missing}")

    coords = cells.set_index("cell_id").loc[cell_order, ["x", "y"]].to_numpy(dtype=float)
    adata = ad.AnnData(
        X=expr.to_numpy(dtype=float),
        obs=pd.DataFrame(index=cell_order),
        var=pd.DataFrame(index=expr.columns.astype(str)),
    )
    adata.obsm["spatial"] = coords

    df_ligrec = _commot_lr_table(lr_pairs, expr.columns)
    if df_ligrec.empty:
        return _zero_predictions(candidates, method="commot")

    if dis_thr is None:
        dis_thr = float(candidates["distance"].max()) if "distance" in candidates.columns else None

    ct.tl.spatial_communication(
        adata,
        database_name=database_name,
        df_ligrec=df_ligrec,
        pathway_sum=False,
        heteromeric=False,
        dis_thr=dis_thr,
        cot_eps_p=cot_eps_p,
        cot_nitermax=cot_nitermax,
    )

    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_order)}
    score_cache: dict[tuple[str, str], object | None] = {}
    scores = np.zeros(len(candidates), dtype=float)
    for row_idx, row in candidates.reset_index(drop=True).iterrows():
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        matrix = score_cache.get((ligand, receptor))
        if matrix is None and (ligand, receptor) not in score_cache:
            matrix = adata.obsp.get(f"commot-{database_name}-{ligand}-{receptor}")
            score_cache[(ligand, receptor)] = matrix
        if matrix is None:
            continue

        sender_idx = cell_to_idx.get(str(row["sender_id"]))
        receiver_idx = cell_to_idx.get(str(row["receiver_id"]))
        if sender_idx is None or receiver_idx is None:
            continue
        scores[row_idx] = float(matrix[sender_idx, receiver_idx])

    return pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "method": "commot",
            "score": scores,
        }
    )


def _commot_lr_table(lr_pairs: pd.DataFrame, genes: pd.Index) -> pd.DataFrame:
    genes = set(genes.astype(str))
    rows = []
    for row in lr_pairs.drop_duplicates(subset=["ligand", "receptor"]).itertuples(index=False):
        ligand = str(getattr(row, "ligand"))
        receptor = str(getattr(row, "receptor"))
        if ligand in genes and receptor in genes:
            pathway = str(getattr(row, "mechanism", f"{ligand}_{receptor}"))
            rows.append((ligand, receptor, pathway))
    return pd.DataFrame(rows, columns=["ligand", "receptor", "pathway"])


def _zero_predictions(candidates: pd.DataFrame, *, method: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "method": method,
            "score": np.zeros(len(candidates), dtype=float),
        }
    )


def _check_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} table missing columns: {sorted(missing)}")


def _patch_numpy2_legacy_aliases() -> None:
    """Provide old aliases required by older anndata/commot dependency stacks."""

    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
