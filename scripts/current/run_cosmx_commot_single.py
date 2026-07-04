#!/usr/bin/env python
"""Run COMMOT on one standardized CosMx dataset without importing SpatialSelf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dis-thr", type=float, default=None)
    parser.add_argument("--cot-eps-p", type=float, default=0.1)
    parser.add_argument("--cot-nitermax", type=int, default=2000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")

    predictions = score_commot(
        cells,
        expression,
        lr_pairs,
        candidates,
        dis_thr=args.dis_thr,
        cot_eps_p=float(args.cot_eps_p),
        cot_nitermax=int(args.cot_nitermax),
    )
    prediction_path = output_dir / f"{dataset_dir.name}__commot_predictions.csv"
    metrics_path = output_dir / f"{dataset_dir.name}__commot_metrics.json"
    predictions.to_csv(prediction_path, index=False)

    metrics = summarize_real_data_predictions(candidates, predictions)
    metrics.update(
        {
            "replicate": dataset_dir.name,
            "method": "commot",
            "prediction_path": str(prediction_path),
            "dis_thr": float(args.dis_thr) if args.dis_thr is not None else float(candidates["distance"].max()),
            "cot_eps_p": float(args.cot_eps_p),
            "cot_nitermax": int(args.cot_nitermax),
            "output_level": "candidate_edge",
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=True))


def score_commot(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    dis_thr: float | None,
    cot_eps_p: float,
    cot_nitermax: int,
) -> pd.DataFrame:
    import anndata as ad
    import commot as ct

    cell_order = cells["cell_id"].astype(str).to_list()
    expr = expression.set_index("cell_id").reindex(cell_order)
    if expr.isnull().any().any():
        raise ValueError("Expression table is missing cells from cells.csv")
    coords = cells.set_index("cell_id").loc[cell_order, ["x", "y"]].to_numpy(dtype=float)
    adata = ad.AnnData(
        X=expr.to_numpy(dtype=float),
        obs=pd.DataFrame(index=cell_order),
        var=pd.DataFrame(index=expr.columns.astype(str)),
    )
    adata.obsm["spatial"] = coords

    df_ligrec = commot_lr_table(lr_pairs, expr.columns)
    if df_ligrec.empty:
        return zero_predictions(candidates)
    if dis_thr is None:
        dis_thr = float(candidates["distance"].max()) if "distance" in candidates.columns else None

    ct.tl.spatial_communication(
        adata,
        database_name="cosmx_fig5",
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
        key = (ligand, receptor)
        matrix = score_cache.get(key)
        if matrix is None and key not in score_cache:
            matrix = adata.obsp.get(f"commot-cosmx_fig5-{ligand}-{receptor}")
            score_cache[key] = matrix
        if matrix is None:
            continue
        sender_idx = cell_to_idx.get(str(row["sender_id"]))
        receiver_idx = cell_to_idx.get(str(row["receiver_id"]))
        if sender_idx is None or receiver_idx is None:
            continue
        scores[row_idx] = float(matrix[sender_idx, receiver_idx])
    return pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "commot", "score": scores})


def commot_lr_table(lr_pairs: pd.DataFrame, genes: pd.Index) -> pd.DataFrame:
    genes = set(genes.astype(str))
    rows = []
    for row in lr_pairs.drop_duplicates(subset=["ligand", "receptor"]).itertuples(index=False):
        ligand = str(getattr(row, "ligand"))
        receptor = str(getattr(row, "receptor"))
        if ligand in genes and receptor in genes:
            pathway = str(getattr(row, "mechanism", f"{ligand}_{receptor}"))
            rows.append((ligand, receptor, pathway))
    return pd.DataFrame(rows, columns=["ligand", "receptor", "pathway"])


def zero_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "commot", "score": 0.0})


def summarize_real_data_predictions(candidates: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    scored = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    if scored.empty:
        return {
            "n_candidates": int(len(candidates)),
            "score_min": float("nan"),
            "score_max": float("nan"),
            "evaluation_mode": "real_data_no_ground_truth",
            "truth_metrics_available": False,
        }
    lr_mean = (
        scored.groupby("lr_id", as_index=False)
        .agg(mean_score=("score", "mean"), max_score=("score", "max"))
        .sort_values(["mean_score", "max_score"], ascending=False)
        .reset_index(drop=True)
    )
    top_counts = scored.nlargest(min(100, len(scored)), "score")["lr_id"].value_counts()
    return {
        "n_candidates": int(len(candidates)),
        "score_min": float(scored["score"].min()),
        "score_max": float(scored["score"].max()),
        "evaluation_mode": "real_data_no_ground_truth",
        "truth_metrics_available": False,
        "n_lr_pairs_scored": int(lr_mean.shape[0]),
        "top_lr_by_mean_score": str(lr_mean["lr_id"].iloc[0]) if not lr_mean.empty else "",
        "top_lr_mean_score": float(lr_mean["mean_score"].iloc[0]) if not lr_mean.empty else float("nan"),
        "top100_lr_counts": ";".join(f"{idx}:{int(val)}" for idx, val in top_counts.items()),
    }


if __name__ == "__main__":
    main()
