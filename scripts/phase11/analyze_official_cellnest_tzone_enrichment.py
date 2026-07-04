#!/usr/bin/env python
"""Analyze T-zone localization of official CellNEST CCR7-axis predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOCUS_PAIRS = ["CCL19-CCR7", "CCL21-CCR7"]
T_ZONE = "T_cell_zone_proxy"


def load_cellnest_edges(path: Path) -> pd.DataFrame:
    expected = [
        "from_cell",
        "to_cell",
        "ligand",
        "receptor",
        "edge_rank",
        "component",
        "from_id",
        "to_id",
        "attention_score",
    ]
    df = pd.read_csv(path)
    if list(df.columns[: len(expected)]) != expected:
        df = pd.read_csv(path, header=None, names=expected)
        df = df[df["from_cell"].ne("from_cell")]
    df["edge_rank"] = pd.to_numeric(df["edge_rank"], errors="coerce")
    df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce").fillna(0.0)
    df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
    return df.dropna(subset=["edge_rank", "from_cell", "to_cell", "ligand", "receptor"]).copy()


def add_cell_annotations(edges: pd.DataFrame, cells_path: Path) -> pd.DataFrame:
    cells = pd.read_csv(cells_path)
    niche = cells.set_index("cell_id")["niche"].astype(str).to_dict()
    region = cells.set_index("cell_id")["region"].astype(str).to_dict()
    cell_type = cells.set_index("cell_id")["cell_type"].astype(str).to_dict()
    x = cells.set_index("cell_id")["x"].to_dict()
    y = cells.set_index("cell_id")["y"].to_dict()

    out = edges.copy()
    out["sender_niche"] = out["from_cell"].map(niche).fillna("missing")
    out["receiver_niche"] = out["to_cell"].map(niche).fillna("missing")
    out["sender_region"] = out["from_cell"].map(region).fillna("missing")
    out["receiver_region"] = out["to_cell"].map(region).fillna("missing")
    out["sender_cell_type"] = out["from_cell"].map(cell_type).fillna("missing")
    out["receiver_cell_type"] = out["to_cell"].map(cell_type).fillna("missing")
    out["sender_x"] = out["from_cell"].map(x)
    out["sender_y"] = out["from_cell"].map(y)
    out["receiver_x"] = out["to_cell"].map(x)
    out["receiver_y"] = out["to_cell"].map(y)
    out["distance"] = np.sqrt((out["sender_x"] - out["receiver_x"]) ** 2 + (out["sender_y"] - out["receiver_y"]) ** 2)
    return out


def top_fraction_by_rank(table: pd.DataFrame, flag_col: str, fraction: float = 0.1) -> float:
    if table.empty:
        return float("nan")
    n_top = max(1, int(np.ceil(len(table) * fraction)))
    top = table.sort_values("edge_rank", ascending=True).head(n_top)
    return float(top[flag_col].mean())


def summarize_lr(edges: pd.DataFrame) -> pd.DataFrame:
    edges = edges.copy()
    edges["receiver_is_tzone"] = edges["receiver_niche"].eq(T_ZONE)
    edges["sender_is_tzone"] = edges["sender_niche"].eq(T_ZONE)
    edges["tzone_to_tzone"] = edges["sender_is_tzone"] & edges["receiver_is_tzone"]

    rows = []
    for lr_pair, sub in edges.groupby("lr_pair", sort=False):
        total_score = float(sub["attention_score"].sum())
        tzone_score = float(sub.loc[sub["receiver_is_tzone"], "attention_score"].sum())
        tzone_to_tzone_score = float(sub.loc[sub["tzone_to_tzone"], "attention_score"].sum())
        rows.append(
            {
                "lr_pair": lr_pair,
                "n_edges": int(len(sub)),
                "min_edge_rank": float(sub["edge_rank"].min()),
                "median_edge_rank": float(sub["edge_rank"].median()),
                "mean_attention": float(sub["attention_score"].mean()),
                "score_sum": total_score,
                "receiver_tzone_n_edges": int(sub["receiver_is_tzone"].sum()),
                "receiver_tzone_edge_fraction": float(sub["receiver_is_tzone"].mean()),
                "receiver_tzone_score_fraction": tzone_score / total_score if total_score else float("nan"),
                "receiver_tzone_top10_by_rank_fraction": top_fraction_by_rank(sub, "receiver_is_tzone", 0.1),
                "sender_tzone_edge_fraction": float(sub["sender_is_tzone"].mean()),
                "tzone_to_tzone_edge_fraction": float(sub["tzone_to_tzone"].mean()),
                "tzone_to_tzone_score_fraction": tzone_to_tzone_score / total_score if total_score else float("nan"),
                "median_distance": float(sub["distance"].median()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["min_edge_rank_lr_rank"] = out["min_edge_rank"].rank(ascending=True, method="min").astype(int)
    out["score_sum_lr_rank"] = out["score_sum"].rank(ascending=False, method="min").astype(int)
    out["receiver_tzone_score_fraction_rank"] = out["receiver_tzone_score_fraction"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["min_edge_rank_lr_rank", "lr_pair"])


def summarize_tzone_subset(edges: pd.DataFrame) -> pd.DataFrame:
    tzone = edges[edges["receiver_niche"].eq(T_ZONE)].copy()
    out = summarize_lr(tzone)
    if not out.empty:
        out = out.rename(
            columns={
                "min_edge_rank_lr_rank": "tzone_receiver_min_edge_rank_lr_rank",
                "score_sum_lr_rank": "tzone_receiver_score_sum_lr_rank",
            }
        )
    return out


def summarize_flow(edges: pd.DataFrame) -> pd.DataFrame:
    flow = (
        edges.groupby(["lr_pair", "sender_niche", "receiver_niche"], as_index=False)
        .agg(
            n_edges=("lr_pair", "size"),
            score_sum=("attention_score", "sum"),
            mean_attention=("attention_score", "mean"),
            min_edge_rank=("edge_rank", "min"),
            median_distance=("distance", "median"),
        )
    )
    flow["score_fraction_within_lr"] = flow["score_sum"] / flow.groupby("lr_pair")["score_sum"].transform("sum").replace(0, np.nan)
    flow["rank_within_lr"] = flow.groupby("lr_pair")["score_sum"].rank(ascending=False, method="dense").astype(int)
    return flow.sort_values(["lr_pair", "rank_within_lr", "score_sum"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cellnest-top20",
        default="/home/xd/y/BenchmarkBaselines/CellNEST/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_top20percent.csv",
    )
    parser.add_argument(
        "--cells",
        default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset/cells.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase11/official_method_reproduction/cellnest_human_lymph_node",
    )
    args = parser.parse_args()

    top20_path = Path(args.cellnest_top20)
    cells_path = PROJECT_ROOT / args.cells
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = add_cell_annotations(load_cellnest_edges(top20_path), cells_path)
    lr_summary = summarize_lr(edges)
    tzone_summary = summarize_tzone_subset(edges)
    flow = summarize_flow(edges)

    focus_lr = lr_summary[lr_summary["lr_pair"].isin(FOCUS_PAIRS)].copy()
    focus_tzone = tzone_summary[tzone_summary["lr_pair"].isin(FOCUS_PAIRS)].copy() if not tzone_summary.empty else tzone_summary
    focus_flow = flow[flow["lr_pair"].isin(FOCUS_PAIRS)].copy()
    focus_flow = focus_flow.sort_values(["lr_pair", "rank_within_lr", "score_sum"], ascending=[True, True, False])

    lr_summary.to_csv(output_dir / "cellnest_official_tzone_lr_summary.csv", index=False)
    tzone_summary.to_csv(output_dir / "cellnest_official_tzone_receiver_subset_lr_summary.csv", index=False)
    flow.to_csv(output_dir / "cellnest_official_niche_flow.csv", index=False)
    focus_lr.to_csv(output_dir / "cellnest_official_ccr7_tzone_focus.csv", index=False)
    focus_tzone.to_csv(output_dir / "cellnest_official_ccr7_tzone_receiver_subset_focus.csv", index=False)
    focus_flow.to_csv(output_dir / "cellnest_official_ccr7_niche_flow.csv", index=False)

    background_receiver_tzone_fraction = float(edges["receiver_niche"].eq(T_ZONE).mean())
    background_receiver_tzone_score_fraction = float(
        edges.loc[edges["receiver_niche"].eq(T_ZONE), "attention_score"].sum() / edges["attention_score"].sum()
    )
    result = {
        "cellnest_top20": str(top20_path),
        "cells": str(cells_path),
        "n_edges": int(len(edges)),
        "n_lr_pairs": int(lr_summary["lr_pair"].nunique()),
        "background_receiver_tzone_edge_fraction": background_receiver_tzone_fraction,
        "background_receiver_tzone_score_fraction": background_receiver_tzone_score_fraction,
        "tzone_label": T_ZONE,
        "rank_definitions": {
            "global_lr_rank": "rank among LR pairs in official CellNEST top20% output by minimum official edge_rank",
            "tzone_receiver_lr_rank": "same ranking after filtering CellNEST edges to receiver spots annotated as T_cell_zone_proxy",
        },
        "focus": {},
    }
    for pair in FOCUS_PAIRS:
        row = focus_lr[focus_lr["lr_pair"].eq(pair)]
        trow = focus_tzone[focus_tzone["lr_pair"].eq(pair)] if not focus_tzone.empty else pd.DataFrame()
        top_flow = focus_flow[focus_flow["lr_pair"].eq(pair)].head(1)
        item = {}
        if len(row):
            item.update(row.iloc[0].to_dict())
        if len(trow):
            item["tzone_receiver_min_edge_rank_lr_rank"] = int(trow.iloc[0]["tzone_receiver_min_edge_rank_lr_rank"])
            item["tzone_receiver_score_sum_lr_rank"] = int(trow.iloc[0]["tzone_receiver_score_sum_lr_rank"])
            item["tzone_receiver_n_edges_after_filter"] = int(trow.iloc[0]["n_edges"])
        if len(top_flow):
            item["top_niche_flow"] = f"{top_flow.iloc[0]['sender_niche']} -> {top_flow.iloc[0]['receiver_niche']}"
            item["top_niche_flow_score_fraction"] = float(top_flow.iloc[0]["score_fraction_within_lr"])
            item["top_niche_flow_n_edges"] = int(top_flow.iloc[0]["n_edges"])
        result["focus"][pair] = item

    (output_dir / "cellnest_official_ccr7_tzone_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
