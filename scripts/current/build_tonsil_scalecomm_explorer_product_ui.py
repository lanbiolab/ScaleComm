#!/usr/bin/env python
"""Build a standalone ScaleComm Explorer UI for the tonsil CXCL13-CXCR5 story."""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current import build_spatialself_explorer_product_ui as product_ui


OUT_DIR = ROOT / "outputs/current_web_ui"
DATA_DIR = (
    ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset"
)
PRED_PATH = (
    ROOT
    / "outputs/current_spamosaic_tonsil_scalecomm_rerun/a1_rich_spatialself_wavelet_scale_e200/"
    "spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_predictions.csv"
)
STORY_SOURCE = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story/source_data"

DEFAULT_LR = "CXCL13-CXCR5"

TONSIL_CELL_COLORS = {
    "B_cell_proxy": "#FF1E18",
    "FDC_proxy": "#F5A7A8",
    "T_cell_proxy": "#66A9E8",
    "Tfh_proxy": "#BFD8EC",
    "myeloid_proxy": "#9BC8C1",
    "stromal_proxy": "#B9B0A6",
    "epithelial_proxy": "#D6D1C9",
    "unannotated": "#DCE2E5",
}


def _round_list(values: pd.Series | np.ndarray, digits: int = 5) -> list[float]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _scale(values: pd.Series, vmin: float, vmax: float) -> np.ndarray:
    span = max(float(vmax - vmin), 1e-9)
    return (pd.to_numeric(values, errors="coerce").to_numpy(float) - vmin) / span


def _pair_summary(lr_edges: pd.DataFrame) -> pd.DataFrame:
    annotated = lr_edges[
        ~lr_edges["sender_cell_type"].astype(str).eq("unannotated")
        & ~lr_edges["receiver_cell_type"].astype(str).eq("unannotated")
    ].copy()
    if annotated.empty:
        return pd.DataFrame(columns=["pair", "n_edges", "mean_score", "median_distance"])
    annotated["pair"] = annotated["sender_cell_type"].astype(str) + " -> " + annotated["receiver_cell_type"].astype(str)
    return (
        annotated.groupby("pair", as_index=False)
        .agg(n_edges=("score", "size"), mean_score=("score", "mean"), median_distance=("distance", "median"))
        .sort_values(["n_edges", "mean_score"], ascending=False)
        .reset_index(drop=True)
    )


def _load_cells() -> pd.DataFrame:
    cells = pd.read_csv(DATA_DIR / "cells.csv")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["cell_type"] = cells["cell_type"].fillna("unannotated").astype(str)
    cells["niche"] = cells["niche"].fillna(cells["cell_type"]).astype(str)
    cells["x_plot"] = pd.to_numeric(cells["x"], errors="coerce")
    cells["y_plot"] = -pd.to_numeric(cells["y"], errors="coerce")
    return cells.dropna(subset=["x_plot", "y_plot"]).copy()


def _load_edges(cells: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "candidate_id",
        "sender_id",
        "receiver_id",
        "ligand",
        "receptor",
        "mechanism",
        "distance",
        "sender_cell_type",
        "receiver_cell_type",
        "sender_niche",
        "receiver_niche",
        "ligand_expr",
        "receptor_expr",
    ]
    candidates = pd.read_csv(DATA_DIR / "candidate_interactions.csv", usecols=usecols)
    pred = pd.read_csv(PRED_PATH, usecols=["candidate_id", "score"])
    edges = candidates.merge(pred, on="candidate_id", how="inner")
    edges["sender_id"] = edges["sender_id"].astype(str)
    edges["receiver_id"] = edges["receiver_id"].astype(str)
    edges["ligand"] = edges["ligand"].astype(str)
    edges["receptor"] = edges["receptor"].astype(str)
    edges["LR"] = edges["ligand"] + "-" + edges["receptor"]
    for col in ["score", "distance", "ligand_expr", "receptor_expr"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")
    for col in ["sender_cell_type", "receiver_cell_type", "sender_niche", "receiver_niche", "mechanism"]:
        edges[col] = edges[col].fillna("unknown").astype(str)

    pos = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].drop_duplicates("cell_id")
    sender = pos.rename(
        columns={
            "cell_id": "sender_id",
            "x_plot": "sender_x_plot",
            "y_plot": "sender_y_plot",
            "cell_type": "sender_cell_type_meta",
            "niche": "sender_niche_meta",
        }
    )
    receiver = pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x_plot": "receiver_x_plot",
            "y_plot": "receiver_y_plot",
            "cell_type": "receiver_cell_type_meta",
            "niche": "receiver_niche_meta",
        }
    )
    edges = edges.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")
    edges["sender_cell_type"] = edges["sender_cell_type"].where(edges["sender_cell_type"].ne("unknown"), edges["sender_cell_type_meta"])
    edges["receiver_cell_type"] = edges["receiver_cell_type"].where(
        edges["receiver_cell_type"].ne("unknown"), edges["receiver_cell_type_meta"]
    )
    edges["sender_niche"] = edges["sender_niche"].where(edges["sender_niche"].ne("unknown"), edges["sender_niche_meta"])
    edges["receiver_niche"] = edges["receiver_niche"].where(edges["receiver_niche"].ne("unknown"), edges["receiver_niche_meta"])
    edges = edges.dropna(subset=["sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot", "score"]).copy()
    return edges


def _load_rank_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lr_pairs = pd.read_csv(DATA_DIR / "lr_pairs.csv")
    lr_pairs["LR"] = lr_pairs["ligand"].astype(str) + "-" + lr_pairs["receptor"].astype(str)

    gc_rank = pd.read_csv(STORY_SOURCE / "panel_g_gc_top20_lr_hotspot_counts.csv")
    if "lr_pair" in gc_rank:
        gc_rank = gc_rank.rename(columns={"lr_pair": "LR"})
    if "section" in gc_rank.columns:
        gc_rank = gc_rank[gc_rank["section"].astype(str).eq("A1")].copy()
    rank_col = "top20_gc_n_rank" if "top20_gc_n_rank" in gc_rank.columns else "top20_gc_fraction_rank"
    gc_rank = gc_rank[["LR", rank_col, "top20_gc_n", "top20_gc_fraction"]].copy()
    gc_rank = gc_rank.rename(columns={rank_col: "gc_hotspot_rank"})

    scale = pd.read_csv(STORY_SOURCE / "source_lr_scale_distribution.csv")
    scale["LR"] = scale["ligand"].astype(str) + "-" + scale["receptor"].astype(str)
    if "section" in scale.columns:
        scale = scale[scale["section"].astype(str).eq("A1")].copy()
    return lr_pairs, gc_rank, scale


def build_payload() -> dict:
    cells = _load_cells()
    edges = _load_edges(cells)
    lr_pairs, gc_rank, scale = _load_rank_tables()

    available_lrs = set(edges["LR"].astype(str))
    rank_order = (
        gc_rank[gc_rank["LR"].isin(available_lrs)]
        .sort_values(["gc_hotspot_rank", "top20_gc_n"], ascending=[True, False])["LR"]
        .astype(str)
        .tolist()
    )
    ranked_lrs = [DEFAULT_LR] + [lr for lr in rank_order if lr != DEFAULT_LR]
    ranked_lrs += [lr for lr in lr_pairs["LR"].astype(str).tolist() if lr in available_lrs and lr not in ranked_lrs]

    xmin = float(min(cells["x_plot"].min(), edges["sender_x_plot"].min(), edges["receiver_x_plot"].min()))
    xmax = float(max(cells["x_plot"].max(), edges["sender_x_plot"].max(), edges["receiver_x_plot"].max()))
    ymin = float(min(cells["y_plot"].min(), edges["sender_y_plot"].min(), edges["receiver_y_plot"].min()))
    ymax = float(max(cells["y_plot"].max(), edges["sender_y_plot"].max(), edges["receiver_y_plot"].max()))

    background = {
        "x": _round_list(_scale(cells["x_plot"], xmin, xmax), 5),
        "y": _round_list(_scale(cells["y_plot"], ymin, ymax), 5),
        "cell_type": cells["cell_type"].astype(str).tolist(),
        "niche": cells["niche"].astype(str).tolist(),
        "cell_id": cells["cell_id"].astype(str).tolist(),
    }

    gc_rank_map = gc_rank.set_index("LR").to_dict("index") if not gc_rank.empty else {}
    lr_pair_meta = lr_pairs.set_index("LR").to_dict("index")
    scale_map = scale.set_index("LR")
    lr_payload: dict[str, dict] = {}

    for lr in ranked_lrs:
        lr_edges = edges[edges["LR"].eq(lr)].sort_values("score", ascending=False).copy()
        if lr_edges.empty:
            continue
        pairs = _pair_summary(lr_edges)
        pair_names = ["All interactions", "Annotated interactions"] + pairs.head(24)["pair"].astype(str).tolist()
        pair_names = list(dict.fromkeys(pair_names))
        scale_vals = []
        if lr in scale_map.index:
            row = scale_map.loc[lr]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for i in range(6):
                scale_vals.append(float(pd.to_numeric(pd.Series([row.get(f"scale_{i}", 0.0)]), errors="coerce").iloc[0]))
        else:
            scale_vals = [0.0] * 6
        meta = lr_pair_meta.get(lr, {})
        rank_meta = gc_rank_map.get(lr, {})
        rank = int(rank_meta.get("gc_hotspot_rank")) if pd.notna(rank_meta.get("gc_hotspot_rank", np.nan)) else None
        lr_payload[lr] = {
            "mechanism": str(meta.get("mechanism", "ScaleComm-ranked LR")).replace("_", " "),
            "rank_q95": rank,
            "median_distance_all": round(float(lr_edges["distance"].median()), 1),
            "all_lr_edges": int(edges["LR"].eq(lr).sum()),
            "embedded_edges": int(len(lr_edges)),
            "annotated_edges": int(len(lr_edges)),
            "pair_names": pair_names,
            "pair_summary": [
                {
                    "pair": str(row.pair),
                    "n_edges": int(row.n_edges),
                    "median_distance": round(float(row.median_distance), 1) if pd.notna(row.median_distance) else None,
                }
                for row in pairs.head(12).itertuples(index=False)
            ],
            "scale": [round(float(v), 4) for v in scale_vals],
            "edges": {
                "sender_id": lr_edges["sender_id"].astype(str).tolist(),
                "receiver_id": lr_edges["receiver_id"].astype(str).tolist(),
                "sender_type": lr_edges["sender_cell_type"].astype(str).tolist(),
                "receiver_type": lr_edges["receiver_cell_type"].astype(str).tolist(),
                "sender_niche": lr_edges["sender_niche"].astype(str).tolist(),
                "receiver_niche": lr_edges["receiver_niche"].astype(str).tolist(),
                "sx": _round_list(_scale(lr_edges["sender_x_plot"], xmin, xmax), 6),
                "sy": _round_list(_scale(lr_edges["sender_y_plot"], ymin, ymax), 6),
                "rx": _round_list(_scale(lr_edges["receiver_x_plot"], xmin, xmax), 6),
                "ry": _round_list(_scale(lr_edges["receiver_y_plot"], ymin, ymax), 6),
                "score": _round_list(lr_edges["score"], 6),
                "distance": _round_list(lr_edges["distance"], 2),
            },
            "annotated_pool": int(len(lr_edges)),
        }

    return {
        "dataset": "SPAMOSAIC tonsil A1",
        "default_lr": DEFAULT_LR,
        "total_cells": int(len(cells)),
        "annotated_cells": int(len(cells)),
        "background": background,
        "cell_colors": TONSIL_CELL_COLORS,
        "lrs": lr_payload,
    }


def build_html() -> str:
    product_ui.build_payload = build_payload
    html = product_ui.build_html()
    html = html.replace("Slide-seqV2 hippocampus", "SPAMOSAIC tonsil A1")
    html = html.replace("Cxcl12-Cxcr4", DEFAULT_LR)
    html = html.replace("<title>ScaleComm Explorer</title>", "<title>ScaleComm Explorer | SPAMOSAIC tonsil</title>")
    return html


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    named = OUT_DIR / "scalecomm_explorer_tonsil_cxcl13_cxcr5.html"
    product = OUT_DIR / "spatialself_explorer_product_ui.html"
    paper = OUT_DIR / "spatialself_explorer_paper_screenshot.html"
    manifest = OUT_DIR / "scalecomm_explorer_tonsil_cxcl13_cxcr5_manifest.json"
    named.write_text(html, encoding="utf-8")
    product.write_text(html, encoding="utf-8")
    paper_html = (
        html.replace("<title>ScaleComm Explorer | SPAMOSAIC tonsil</title>", "<title>ScaleComm Explorer | SPAMOSAIC tonsil paper view</title>")
        .replace('<div class="page">', '<div class="page paper-view presentation-view">', 1)
        .replace(
            '<button id="tool-view" class="active">Tool</button><button id="paper-view">Figure</button>',
            '<button id="tool-view">Tool</button><button id="paper-view" class="active">Figure</button>',
        )
    )
    paper.write_text(paper_html, encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "dataset": "SPAMOSAIC tonsil A1",
                "default_lr": DEFAULT_LR,
                "source_input": str(DATA_DIR.relative_to(ROOT)),
                "source_prediction": str(PRED_PATH.relative_to(ROOT)),
                "named_html": str(named.relative_to(ROOT)),
                "product_html": str(product.relative_to(ROOT)),
                "paper_html": str(paper.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(named)
    print(product)
    print(paper)
    print(manifest)


if __name__ == "__main__":
    main()
