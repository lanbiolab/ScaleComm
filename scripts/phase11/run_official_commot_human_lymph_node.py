#!/usr/bin/env python
"""Run COMMOT's tutorial workflow on full 10x Human Lymph Node Visium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import commot as ct


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_tzone_labels(dataset_dir: Path, obs_names: pd.Index) -> pd.DataFrame:
    cells = pd.read_csv(dataset_dir / "cells.csv")
    cells = cells.set_index("cell_id").reindex(obs_names)
    return cells[["niche", "region"]].fillna("")


def summarize_commot(adata, database_name: str, labels: pd.DataFrame, output_dir: Path) -> dict:
    sender = adata.obsm[f"commot-{database_name}-sum-sender"]
    receiver = adata.obsm[f"commot-{database_name}-sum-receiver"]
    lr_rows = []
    for col in receiver.columns:
        if not col.startswith("r-") or col == "r-total-total":
            continue
        lr_pair = col.removeprefix("r-")
        r_score = pd.to_numeric(receiver[col], errors="coerce").fillna(0.0)
        s_col = "s-" + lr_pair
        s_score = pd.to_numeric(sender[s_col], errors="coerce").fillna(0.0) if s_col in sender else pd.Series(0.0, index=receiver.index)
        tzone = labels["niche"].astype(str).eq("T_cell_zone_proxy")
        threshold = float(np.quantile(r_score.to_numpy(dtype=float), 0.9)) if len(r_score) else np.nan
        top = r_score.ge(threshold) if np.isfinite(threshold) else pd.Series(False, index=r_score.index)
        lr_rows.append(
            {
                "lr_pair": lr_pair,
                "receiver_score_sum": float(r_score.sum()),
                "sender_score_sum": float(s_score.sum()),
                "total_score": float(r_score.sum() + s_score.sum()),
                "receiver_score_q95": float(np.quantile(r_score.to_numpy(dtype=float), 0.95)) if len(r_score) else np.nan,
                "tzone_receiver_score_sum": float(r_score[tzone].sum()),
                "tzone_receiver_score_fraction": float(r_score[tzone].sum() / r_score.sum()) if r_score.sum() > 0 else np.nan,
                "tzone_receiver_top10_fraction": float(tzone[top].mean()) if top.any() else np.nan,
            }
        )
    lr_summary = pd.DataFrame(lr_rows)
    if not lr_summary.empty:
        lr_summary["total_score_rank"] = lr_summary["total_score"].rank(ascending=False, method="min").astype(int)
        lr_summary["receiver_score_rank"] = lr_summary["receiver_score_sum"].rank(ascending=False, method="min").astype(int)
        lr_summary = lr_summary.sort_values(["total_score_rank", "lr_pair"])

    focus = lr_summary[lr_summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7"])] if not lr_summary.empty else lr_summary
    lr_summary.to_csv(output_dir / "commot_official_lr_summary.csv", index=False)
    focus.to_csv(output_dir / "commot_official_ccr7_focus.csv", index=False)

    result = {
        "method": "COMMOT official tutorial workflow",
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_lr_pairs_scored": int(len(lr_summary)),
        "database_name": database_name,
        "distance_threshold": 200,
        "normalization": "scanpy.pp.normalize_total + scanpy.pp.log1p",
        "lr_database": "commot.pp.ligand_receptor_database(database='CellChat', species='human')",
    }
    for pair in ["CCL19-CCR7", "CCL21-CCR7"]:
        row = focus[focus["lr_pair"].eq(pair)]
        prefix = pair.lower().replace("-", "_")
        if len(row):
            item = row.iloc[0].to_dict()
            result[f"{prefix}_total_score_rank"] = int(item["total_score_rank"])
            result[f"{prefix}_receiver_score_rank"] = int(item["receiver_score_rank"])
            result[f"{prefix}_tzone_receiver_score_fraction"] = float(item["tzone_receiver_score_fraction"])
            result[f"{prefix}_tzone_receiver_top10_fraction"] = float(item["tzone_receiver_top10_fraction"])
        else:
            result[f"{prefix}_total_score_rank"] = None
    (output_dir / "commot_official_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/optional/human_lymph_node_visium_full/raw")
    parser.add_argument("--dataset-dir", default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset")
    parser.add_argument("--output-dir", default="outputs/phase11/official_method_reproduction/commot_human_lymph_node")
    parser.add_argument("--distance-threshold", type=float, default=200.0)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_visium(path=str(raw_dir), count_file="V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    labels = load_tzone_labels(dataset_dir, adata.obs_names)
    adata.obs["spatialself_marker_niche"] = labels["niche"].values
    adata.obs["spatialself_marker_region"] = labels["region"].values

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    df_ligrec = ct.pp.ligand_receptor_database(database="CellChat", species="human")
    df_ligrec.to_csv(output_dir / "commot_cellchat_human_lr_database.csv", index=False)

    database_name = "CellChat"
    ct.tl.spatial_communication(
        adata,
        database_name=database_name,
        df_ligrec=df_ligrec,
        dis_thr=args.distance_threshold,
        heteromeric=True,
    )
    result = summarize_commot(adata, database_name, labels, output_dir)
    adata.write_h5ad(output_dir / "commot_official_human_lymph_node.h5ad", compression="gzip")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
