#!/usr/bin/env python
"""ScaleComm Explorer.

Screenshot-first interactive viewer for ScaleComm communication outputs.
The design goal is a compact CellNEST-style tool panel: one large spatial
communication canvas, minimal controls, and click-to-inspect edge details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[2]
SLIDESEQ_ROOT = ROOT / "outputs/current_slideseqv2_hpc_full"
SLIDESEQ_INPUT = SLIDESEQ_ROOT / "standardized_inputs/slideseqv2_mouse_hpc_full"
SLIDESEQ_RUN = SLIDESEQ_ROOT / "spatialself_scale_full_e80"

INK = "#20272C"
MUTED = "#66727A"
LINE = "#E3E8EC"
SOFT = "#F7F8F9"
SENDER = "#78B66B"
RECEIVER = "#536FA8"
EDGE = "#263137"

CELL_COLORS = {
    "astrocyte": "#93CC82",
    "oligodendrocyte": "#B9C1D1",
    "CA1_neuron": "#B9E5FA",
    "CA3_neuron": "#8CB9D9",
    "dentate_neuron": "#F3D78A",
    "interneuron": "#FBC9C4",
    "microglia_macrophage": "#C6B2D6",
    "endothelial": "#8BC2B3",
    "polydendrocyte": "#C7B5A7",
    "entorhinal_neuron": "#A9CDEB",
    "ependymal": "#D7C4E9",
    "cajal_retzius": "#E8C4A7",
    "neurogenic": "#B9DDB4",
    "excitatory_neuron": "#F4CF90",
    "mural": "#B9B0A6",
    "choroid": "#D6C6A8",
}


@dataclass(frozen=True)
class ExplorerStats:
    lr: str
    pair: str
    total_cells: int
    annotated_cells: int
    lr_edges: int
    pair_edges: int
    displayed_edges: int
    top_fraction: int
    median_distance: float
    top_sender: str
    top_receiver: str
    mechanism: str


def page_style() -> None:
    st.set_page_config(
        page_title="ScaleComm Explorer",
        page_icon="SC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
          --ink: #20272C;
          --muted: #66727A;
          --line: #E3E8EC;
          --soft: #F7F8F9;
        }
        .stApp { background: #FFFFFF; color: var(--ink); }
        #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="collapsedControl"] {
          visibility: hidden;
          display: none;
        }
        .block-container {
          padding: 0.70rem 1.05rem 0.75rem 1.05rem;
          max-width: 1840px;
        }
        div[data-testid="stVerticalBlock"] { gap: 0.42rem; }
        div[data-testid="column"] { padding-top: 0 !important; }
        .ss-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          border: 1px solid var(--line);
          background: #FFFFFF;
          padding: 0.58rem 0.74rem;
        }
        .ss-title {
          font: 700 1.04rem/1.05 Arial, Helvetica, sans-serif;
          color: var(--ink);
          letter-spacing: 0;
        }
        .ss-subtitle {
          margin-top: 0.17rem;
          font: 400 0.74rem/1.25 Arial, Helvetica, sans-serif;
          color: var(--muted);
        }
        .ss-badges {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 0.35rem;
          flex-wrap: wrap;
        }
        .ss-badge {
          display: inline-block;
          border: 1px solid var(--line);
          background: #FBFCFD;
          color: var(--ink);
          padding: 0.18rem 0.42rem;
          font: 400 0.68rem/1 Arial, Helvetica, sans-serif;
          white-space: nowrap;
        }
        .ss-control-card {
          border: 1px solid var(--line);
          background: #FFFFFF;
          padding: 0.50rem 0.62rem 0.58rem 0.62rem;
        }
        .ss-control-title {
          font: 700 0.76rem/1 Arial, Helvetica, sans-serif;
          color: var(--ink);
          margin-bottom: 0.28rem;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stCheckbox"] label {
          color: var(--muted) !important;
          font: 400 0.70rem/1.1 Arial, Helvetica, sans-serif !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
          min-height: 2.02rem;
          border-radius: 0 !important;
          border-color: var(--line) !important;
          box-shadow: none !important;
          background: #FFFFFF !important;
        }
        div[data-testid="stSlider"] { padding-top: 0.05rem; }
        .stSlider [data-baseweb="slider"] div { border-radius: 0 !important; }
        .ss-caption {
          margin: -0.10rem 0 0.12rem 0;
          color: var(--muted);
          font: 400 0.70rem/1.3 Arial, Helvetica, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_slideseq_data() -> dict[str, pd.DataFrame]:
    cells_path = SLIDESEQ_INPUT / "cells.csv"
    edges_path = SLIDESEQ_RUN / "top10000_edges_merged.csv"
    ranking_path = SLIDESEQ_RUN / "lr_ranking_summary.csv"
    scale_path = SLIDESEQ_RUN / "slideseqv2_mouse_hpc_full__spatialself_wavelet_scale_lr_scale_distribution.csv"
    missing = [str(p) for p in [cells_path, edges_path, ranking_path, scale_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SpatialSelf viewer source table(s): " + ", ".join(missing))

    cells = pd.read_csv(cells_path)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["cell_type"] = cells.get("cell_type", "unannotated").fillna("unannotated").astype(str)
    cells["niche"] = cells.get("niche", cells["cell_type"]).fillna("unknown").astype(str)
    cells["x_plot"] = pd.to_numeric(cells["x"], errors="coerce")
    cells["y_plot"] = -pd.to_numeric(cells["y"], errors="coerce")
    cells = cells.dropna(subset=["x_plot", "y_plot"]).copy()

    edges = pd.read_csv(edges_path)
    edges["sender_id"] = edges["sender_id"].astype(str)
    edges["receiver_id"] = edges["receiver_id"].astype(str)
    edges["ligand"] = edges["ligand"].astype(str)
    edges["receptor"] = edges["receptor"].astype(str)
    edges["LR"] = edges["ligand"] + "-" + edges["receptor"]
    for col in ["score", "distance", "ligand_expr", "receptor_expr"]:
        if col in edges:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
    for col in ["sender_cell_type", "receiver_cell_type", "sender_niche", "receiver_niche", "mechanism"]:
        if col in edges:
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
    edges["receiver_cell_type"] = edges["receiver_cell_type"].where(edges["receiver_cell_type"].ne("unknown"), edges["receiver_cell_type_meta"])
    edges["sender_niche"] = edges["sender_niche"].where(edges["sender_niche"].ne("unknown"), edges["sender_niche_meta"])
    edges["receiver_niche"] = edges["receiver_niche"].where(edges["receiver_niche"].ne("unknown"), edges["receiver_niche_meta"])
    edges = edges.dropna(subset=["sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot", "score"]).copy()

    ranking = pd.read_csv(ranking_path)
    ranking["LR"] = ranking["ligand"].astype(str) + "-" + ranking["receptor"].astype(str)
    scale = pd.read_csv(scale_path)
    scale["LR"] = scale["ligand"].astype(str) + "-" + scale["receptor"].astype(str)
    return {"cells": cells, "edges": edges, "ranking": ranking, "scale": scale}


def lr_options(ranking: pd.DataFrame, edges: pd.DataFrame) -> list[str]:
    available = set(edges["LR"].astype(str))
    ranked = ranking[ranking["LR"].isin(available)].sort_values("rank_q95", ascending=True)
    opts = ranked["LR"].tolist()
    return opts or sorted(available)


def pair_options(edges: pd.DataFrame, lr: str) -> tuple[list[str], pd.DataFrame]:
    dat = edges[edges["LR"].eq(lr)].copy()
    dat = dat[
        ~dat["sender_cell_type"].astype(str).eq("unannotated")
        & ~dat["receiver_cell_type"].astype(str).eq("unannotated")
    ].copy()
    if dat.empty:
        return ["All detected interactions"], pd.DataFrame()
    dat["pair"] = dat["sender_cell_type"].astype(str) + " -> " + dat["receiver_cell_type"].astype(str)
    summary = (
        dat.groupby(["pair", "sender_cell_type", "receiver_cell_type"], as_index=False)
        .agg(n_edges=("score", "size"), mean_score=("score", "mean"), median_distance=("distance", "median"))
        .sort_values(["n_edges", "mean_score"], ascending=False)
        .reset_index(drop=True)
    )
    opts = ["All annotated interactions"] + summary.head(24)["pair"].tolist()
    return opts, summary


def normalize_xy(cells: pd.DataFrame, edges: pd.DataFrame) -> tuple[float, float, float, float]:
    xs = pd.concat([cells["x_plot"], edges["sender_x_plot"], edges["receiver_x_plot"]], ignore_index=True)
    ys = pd.concat([cells["y_plot"], edges["sender_y_plot"], edges["receiver_y_plot"]], ignore_index=True)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    return xmin, xmax, ymin, ymax


def scale_array(values: pd.Series, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-9)
    return (pd.to_numeric(values, errors="coerce").to_numpy(float) - vmin) / span


def selected_edges(edges: pd.DataFrame, lr: str, pair: str, top_fraction: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat = edges[edges["LR"].eq(lr)].copy()
    if pair == "All annotated interactions":
        dat = dat[
            ~dat["sender_cell_type"].astype(str).eq("unannotated")
            & ~dat["receiver_cell_type"].astype(str).eq("unannotated")
        ].copy()
    elif pair != "All detected interactions" and " -> " in pair:
        sender_type, receiver_type = pair.split(" -> ", 1)
        dat = dat[
            dat["sender_cell_type"].astype(str).eq(sender_type)
            & dat["receiver_cell_type"].astype(str).eq(receiver_type)
        ].copy()
    dat = dat.sort_values("score", ascending=False).reset_index(drop=True)
    if dat.empty:
        return dat, dat
    n_show = min(len(dat), max(20, int(np.ceil(len(dat) * top_fraction / 100.0))))
    return dat, dat.head(n_show).copy()


def make_explorer_figure(cells: pd.DataFrame, all_edges: pd.DataFrame, show_edges: pd.DataFrame, lr: str, pair: str) -> go.Figure:
    annotated = cells[~cells["cell_type"].astype(str).eq("unannotated")].copy()
    context_types = sorted(
        set(show_edges["sender_cell_type"].astype(str)).union(show_edges["receiver_cell_type"].astype(str))
        - {"unannotated", "unknown", "nan"}
    )
    if len(context_types) > 6:
        counts = pd.concat([show_edges["sender_cell_type"], show_edges["receiver_cell_type"]]).value_counts()
        context_types = [ct for ct in counts.index.astype(str).tolist() if ct not in {"unannotated", "unknown", "nan"}][:6]
    context_cells = annotated[annotated["cell_type"].isin(context_types)].copy()
    xmin, xmax, ymin, ymax = normalize_xy(annotated, show_edges if not show_edges.empty else all_edges)

    fig = go.Figure()

    for z, fill, outline, label in [
        (1.0, "#F4F8F2", "rgba(120,182,107,0.22)", "sending layer"),
        (0.0, "#F4F6FB", "rgba(83,111,168,0.22)", "receiving layer"),
    ]:
        fig.add_trace(
            go.Surface(
                x=[[0.0, 1.0], [0.0, 1.0]],
                y=[[0.0, 0.0], [1.0, 1.0]],
                z=[[z, z], [z, z]],
                surfacecolor=[[0, 0], [0, 0]],
                colorscale=[[0, fill], [1, fill]],
                opacity=0.18,
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
                name=label,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 1, 1, 0, 0],
                y=[0, 0, 1, 1, 0],
                z=[z, z, z, z, z],
                mode="lines",
                line={"color": outline, "width": 1.4},
                hoverinfo="skip",
                showlegend=False,
                name=f"{label} border",
            )
        )

    bg_x = scale_array(annotated["x_plot"], xmin, xmax)
    bg_y = scale_array(annotated["y_plot"], ymin, ymax)
    bg_custom = np.column_stack(
        [
            np.full(len(annotated), "cell"),
            annotated["cell_id"].astype(str),
            annotated["cell_type"].astype(str),
            annotated["niche"].astype(str),
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=bg_x,
            y=bg_y,
            z=np.full(len(annotated), -0.025),
            mode="markers",
            name="annotated tissue",
            marker={"size": 1.45, "color": "#CAD1D6", "opacity": 0.18, "line": {"width": 0}},
            customdata=bg_custom,
            hovertemplate="cell=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )

    for cell_type, group in context_cells.groupby("cell_type", sort=True):
        z = np.full(len(group), -0.015)
        fig.add_trace(
            go.Scatter3d(
                x=scale_array(group["x_plot"], xmin, xmax),
                y=scale_array(group["y_plot"], ymin, ymax),
                z=z,
                mode="markers",
                name=cell_type,
                marker={
                    "size": 2.35,
                    "color": CELL_COLORS.get(cell_type, "#B9C1D1"),
                    "opacity": 0.54,
                    "line": {"width": 0.10, "color": "white"},
                },
                customdata=np.column_stack(
                    [
                        np.full(len(group), "cell"),
                        group["cell_id"].astype(str),
                        group["cell_type"].astype(str),
                        group["niche"].astype(str),
                    ]
                ),
                hovertemplate="cell=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<extra></extra>",
                showlegend=True,
            )
        )

    if show_edges.empty:
        return fig

    sx = scale_array(show_edges["sender_x_plot"], xmin, xmax)
    sy = scale_array(show_edges["sender_y_plot"], ymin, ymax)
    rx = scale_array(show_edges["receiver_x_plot"], xmin, xmax)
    ry = scale_array(show_edges["receiver_y_plot"], ymin, ymax)
    scores = pd.to_numeric(show_edges["score"], errors="coerce").fillna(0).to_numpy(float)
    score_norm = (scores - np.nanmin(scores)) / max(float(np.nanmax(scores) - np.nanmin(scores)), 1e-9)

    q = pd.Series(scores).rank(method="average", pct=True).to_numpy(float)
    edge_rank = np.arange(len(show_edges), dtype=float)

    def curved_edge_points(x0: float, y0: float, x1: float, y1: float, idx: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
        dx = x1 - x0
        dy = y1 - y0
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        px = -dy / length
        py = dx / length
        sign = -1.0 if int(idx) % 2 else 1.0
        bend = sign * (0.012 + 0.018 * ((idx % 11) / 10.0))
        if length < 0.030:
            bend *= 1.65
        ts = [0.0, 0.22, 0.50, 0.78, 1.0]
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for t in ts:
            arc = np.sin(np.pi * t)
            xs.append(float((1 - t) * x0 + t * x1 + bend * px * arc))
            ys.append(float((1 - t) * y0 + t * y1 + bend * py * arc))
            zs.append(float(1.0 - t))
        xs.append(None)
        ys.append(None)
        zs.append(None)
        return xs, ys, zs

    masks = [
        (q < 0.70, 0.62, 0.115, "#9EADB2"),
        ((q >= 0.70) & (q < 0.90), 1.02, 0.29, "#64777C"),
        (q >= 0.90, 1.62, 0.60, "#2D3A40"),
    ]
    for idx, (mask, width, opacity, edge_color) in enumerate(masks):
        if not mask.any():
            continue
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        line_z: list[float | None] = []
        for x0, y0, x1, y1, edge_idx in zip(sx[mask], sy[mask], rx[mask], ry[mask], edge_rank[mask], strict=False):
            cx, cy, cz = curved_edge_points(float(x0), float(y0), float(x1), float(y1), float(edge_idx))
            line_x.extend(cx)
            line_y.extend(cy)
            line_z.extend(cz)
        fig.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                name="predicted communication" if idx == 2 else None,
                line={"color": edge_color, "width": width},
                opacity=opacity,
                hoverinfo="skip",
                showlegend=idx == 2,
            )
        )

    edge_custom = np.column_stack(
        [
            np.full(len(show_edges), "edge"),
            show_edges["sender_id"].astype(str),
            show_edges["sender_cell_type"].astype(str),
            show_edges["receiver_id"].astype(str),
            show_edges["receiver_cell_type"].astype(str),
            pd.to_numeric(show_edges["distance"], errors="coerce").round(1).astype(str),
            show_edges["sender_niche"].astype(str),
            show_edges["receiver_niche"].astype(str),
            pd.Series(sx).round(6).astype(str),
            pd.Series(sy).round(6).astype(str),
            pd.Series(rx).round(6).astype(str),
            pd.Series(ry).round(6).astype(str),
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=(sx + rx) / 2,
            y=(sy + ry) / 2,
            z=np.full(len(show_edges), 0.50),
            mode="markers",
            name="edge selector",
            marker={"size": 8.0, "color": "rgba(0,0,0,0.01)", "opacity": 0.03, "line": {"width": 0}},
            customdata=edge_custom,
            hovertemplate=(
                "sender=%{customdata[1]} (%{customdata[2]})<br>"
                "receiver=%{customdata[3]} (%{customdata[4]})<br>"
                "distance=%{customdata[5]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    sender_custom = np.column_stack(
        [
            np.full(len(show_edges), "sender"),
            show_edges["sender_id"].astype(str),
            show_edges["sender_cell_type"].astype(str),
            show_edges["sender_niche"].astype(str),
            show_edges["receiver_id"].astype(str),
            show_edges["receiver_cell_type"].astype(str),
            pd.to_numeric(show_edges["distance"], errors="coerce").round(1).astype(str),
            pd.Series(sx).round(6).astype(str),
            pd.Series(sy).round(6).astype(str),
            pd.Series(rx).round(6).astype(str),
            pd.Series(ry).round(6).astype(str),
        ]
    )
    receiver_custom = np.column_stack(
        [
            np.full(len(show_edges), "receiver"),
            show_edges["receiver_id"].astype(str),
            show_edges["receiver_cell_type"].astype(str),
            show_edges["receiver_niche"].astype(str),
            show_edges["sender_id"].astype(str),
            show_edges["sender_cell_type"].astype(str),
            pd.to_numeric(show_edges["distance"], errors="coerce").round(1).astype(str),
            pd.Series(rx).round(6).astype(str),
            pd.Series(ry).round(6).astype(str),
            pd.Series(sx).round(6).astype(str),
            pd.Series(sy).round(6).astype(str),
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=np.ones(len(show_edges)),
            mode="markers",
            name="sending cells",
            marker={
                "size": 5.0 + 2.8 * score_norm,
                "color": SENDER,
                "opacity": 0.94,
                "line": {"width": 0.58, "color": "white"},
            },
            customdata=sender_custom,
            hovertemplate=(
                "sender=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>"
                "receiver=%{customdata[4]} (%{customdata[5]})<br>"
                "distance=%{customdata[6]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=rx,
            y=ry,
            z=np.zeros(len(show_edges)),
            mode="markers",
            name="receiving cells",
            marker={
                "size": 5.2 + 3.0 * score_norm,
                "color": RECEIVER,
                "opacity": 0.95,
                "line": {"width": 0.58, "color": "white"},
            },
            customdata=receiver_custom,
            hovertemplate=(
                "receiver=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>"
                "sender=%{customdata[4]} (%{customdata[5]})<br>"
                "distance=%{customdata[6]}<extra></extra>"
            ),
        )
    )

    fig.add_annotation(x=0.025, y=0.955, xref="paper", yref="paper", text="sending layer", showarrow=False, xanchor="left", font={"size": 12, "color": SENDER})
    fig.add_annotation(x=0.025, y=0.090, xref="paper", yref="paper", text="receiving layer", showarrow=False, xanchor="left", font={"size": 12, "color": RECEIVER})
    fig.add_annotation(x=0.985, y=0.040, xref="paper", yref="paper", text="click a cell or edge", showarrow=False, xanchor="right", font={"size": 10, "color": MUTED})

    fig.update_layout(
        height=760,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False, "range": [-0.08, 1.08]},
            "aspectmode": "manual",
            "aspectratio": {"x": 2.55, "y": 0.92, "z": 0.56},
            "camera": {
                "eye": {"x": 0.72, "y": -0.98, "z": 0.54},
                "center": {"x": 0.02, "y": 0.0, "z": -0.10},
                "projection": {"type": "perspective"},
            },
        },
        hovermode="closest",
        uirevision=f"{lr}-{pair}-{len(show_edges)}",
    )
    return fig


def make_stats(
    cells: pd.DataFrame,
    ranking: pd.DataFrame,
    lr_edges: pd.DataFrame,
    pair_edges: pd.DataFrame,
    show_edges: pd.DataFrame,
    lr: str,
    pair: str,
    top_fraction: int,
) -> ExplorerStats:
    rank_row = ranking[ranking["LR"].eq(lr)]
    mechanism = str(rank_row["mechanism"].iloc[0]).replace("_", " ") if not rank_row.empty and "mechanism" in rank_row else "ScaleComm-ranked LR"
    if pair_edges.empty:
        median_distance = float("nan")
        top_sender = "NA"
        top_receiver = "NA"
    else:
        median_distance = float(pd.to_numeric(show_edges["distance"], errors="coerce").median())
        top_sender = str(pair_edges["sender_cell_type"].value_counts().index[0])
        top_receiver = str(pair_edges["receiver_cell_type"].value_counts().index[0])
    return ExplorerStats(
        lr=lr,
        pair=pair,
        total_cells=int(len(cells)),
        annotated_cells=int((~cells["cell_type"].astype(str).eq("unannotated")).sum()),
        lr_edges=int(len(lr_edges)),
        pair_edges=int(len(pair_edges)),
        displayed_edges=int(len(show_edges)),
        top_fraction=int(top_fraction),
        median_distance=median_distance,
        top_sender=top_sender,
        top_receiver=top_receiver,
        mechanism=mechanism,
    )


def scale_profile(scale: pd.DataFrame, lr: str) -> list[tuple[str, float]]:
    row = scale[scale["LR"].eq(lr)]
    if row.empty:
        return []
    out = []
    for i in range(6):
        col = f"scale_{i}"
        if col in row:
            value = pd.to_numeric(row[col], errors="coerce").iloc[0]
            if pd.notna(value):
                out.append((f"S{i}", float(value)))
    return out


def top_pair_rows(summary: pd.DataFrame, current_pair: str) -> str:
    if summary.empty:
        return "<div class='empty'>No annotated cell-type pair table.</div>"
    rows = []
    for _, row in summary.head(4).iterrows():
        active = " active" if str(row["pair"]) == current_pair else ""
        rows.append(
            "<div class='pair-row%s'>"
            "<span>%s</span><b>%d</b><em>%.1f</em>"
            "</div>"
            % (
                active,
                str(row["pair"]).replace("_", " "),
                int(row["n_edges"]),
                float(row["median_distance"]) if pd.notna(row["median_distance"]) else 0.0,
            )
        )
    return "".join(rows)


def render_explorer(fig: go.Figure, stats: ExplorerStats, scale_vals: list[tuple[str, float]], pair_summary: pd.DataFrame) -> None:
    fig_json = fig.to_json()
    scale_html = ""
    if scale_vals:
        max_val = max(v for _, v in scale_vals) or 1.0
        bars = []
        for name, val in scale_vals:
            bars.append(
                "<div class='scale-row'><span>%s</span><div><i style='width:%.2f%%'></i></div><b>%.2f</b></div>"
                % (name, 100.0 * val / max_val, val)
            )
        scale_html = "".join(bars)
    else:
        scale_html = "<div class='empty'>Scale profile not available.</div>"

    metrics = [
        ("displayed edges", f"{stats.displayed_edges:,}"),
        ("selected pool", f"{stats.pair_edges:,}"),
        ("all LR edges", f"{stats.lr_edges:,}"),
        ("annotated beads", f"{stats.annotated_cells:,}"),
        ("median distance", "NA" if np.isnan(stats.median_distance) else f"{stats.median_distance:.1f}"),
    ]
    metric_html = "".join(f"<div class='metric'><span>{k}</span><b>{v}</b></div>" for k, v in metrics)
    pair_html = top_pair_rows(pair_summary, stats.pair)
    payload = {
        "lr": stats.lr,
        "pair": stats.pair,
        "top_fraction": stats.top_fraction,
        "mechanism": stats.mechanism,
        "top_sender": stats.top_sender,
        "top_receiver": stats.top_receiver,
    }

    html = f"""
    <div id="ss-shell">
      <div id="ss-canvas-card">
        <div id="ss-plot"></div>
        <div class="floating-toolbar">
          <button class="mode active" data-mode="overview">Overview</button>
          <button class="mode" data-mode="focus">Focus</button>
          <button class="mode" data-mode="network">Network</button>
          <button id="reset-camera" class="mode">Reset</button>
        </div>
        <div class="layer-key">
          <div><span class="dot sender-dot"></span>sender cells</div>
          <div><span class="dot receiver-dot"></span>receiver cells</div>
          <div><span class="line-key"></span>predicted communication</div>
        </div>
        <div class="canvas-note">Drag to rotate · scroll to zoom · click an edge or endpoint</div>
      </div>
      <aside id="ss-inspector">
        <div class="eyebrow">ScaleComm Explorer</div>
        <div class="lr-title">{stats.lr}</div>
        <div class="pair-title">{stats.pair.replace("_", " ")}</div>
        <div class="pill-row">
          <span>Slide-seqV2</span><span>top {stats.top_fraction}%</span><span>{stats.mechanism}</span>
        </div>
        <div class="metric-grid">{metric_html}</div>
        <div class="section">
          <div class="section-head">Selected event</div>
          <div id="detail-box" class="detail-box">
            <div class="detail-title">No event selected</div>
            <div class="detail-text">Click a sender, receiver, or line midpoint to inspect one ScaleComm communication event.</div>
          </div>
        </div>
        <div class="section">
          <div class="section-head">Learned scale profile</div>
          <div class="scale-box">{scale_html}</div>
        </div>
        <div class="section">
          <div class="section-head">Top cell-type pairs</div>
          <div class="pair-list">
            <div class="pair-header"><span>pair</span><b>edges</b><em>dist.</em></div>
            {pair_html}
          </div>
        </div>
      </aside>
    </div>
    <style>
      #ss-shell {{
        width: 100%;
        height: 704px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 324px;
        gap: 12px;
        background: #FFFFFF;
        font-family: Arial, Helvetica, sans-serif;
        color: {INK};
      }}
      #ss-canvas-card {{
        position: relative;
        border: 1px solid {LINE};
        background: #FFFFFF;
        overflow: hidden;
      }}
      #ss-plot {{ width: 100%; height: 100%; }}
      #ss-inspector {{
        border: 1px solid {LINE};
        background: #FBFCFD;
        padding: 13px 13px 11px 13px;
        overflow: hidden;
      }}
      .eyebrow {{
        color: {MUTED};
        font-size: 10px;
        letter-spacing: .055em;
        text-transform: uppercase;
        line-height: 1;
      }}
      .lr-title {{
        margin-top: 6px;
        font-size: 22px;
        font-weight: 700;
        line-height: 1.04;
        color: {INK};
      }}
      .pair-title {{
        margin-top: 5px;
        color: #3E4A52;
        font-size: 12px;
        line-height: 1.25;
      }}
      .pill-row {{
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
        margin: 10px 0 11px 0;
      }}
      .pill-row span {{
        border: 1px solid {LINE};
        background: #FFFFFF;
        color: {MUTED};
        font-size: 10px;
        padding: 4px 6px;
        line-height: 1;
      }}
      .metric-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 5px;
        margin-bottom: 9px;
      }}
      .metric {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        padding: 6px 7px;
        min-height: 38px;
      }}
      .metric span {{
        display: block;
        color: {MUTED};
        font-size: 10px;
        line-height: 1.05;
      }}
      .metric b {{
        display: block;
        color: {INK};
        font-size: 15px;
        line-height: 1.12;
        margin-top: 4px;
      }}
      .section {{
        border-top: 1px solid {LINE};
        padding-top: 9px;
        margin-top: 9px;
      }}
      .section-head {{
        font-size: 12px;
        font-weight: 700;
        color: {INK};
        margin-bottom: 7px;
      }}
      .detail-box {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        padding: 9px 9px 8px 9px;
        min-height: 82px;
      }}
      .detail-title {{
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
        color: {INK};
        margin-bottom: 6px;
      }}
      .detail-text, .detail-row {{
        color: #4C5860;
        font-size: 11px;
        line-height: 1.42;
      }}
      .detail-row {{
        display: grid;
        grid-template-columns: 70px minmax(0, 1fr);
        gap: 7px;
        border-top: 1px solid #EDF1F4;
        padding: 5px 0;
      }}
      .detail-row:first-of-type {{ border-top: 0; }}
      .detail-row span {{ color: {MUTED}; }}
      .detail-row b {{ color: {INK}; font-weight: 600; word-break: break-word; }}
      .scale-box {{
        display: grid;
        gap: 5px;
      }}
      .scale-row {{
        display: grid;
        grid-template-columns: 24px minmax(0, 1fr) 34px;
        gap: 7px;
        align-items: center;
        color: {MUTED};
        font-size: 10px;
      }}
      .scale-row div {{
        height: 7px;
        background: #E9EEF1;
        overflow: hidden;
      }}
      .scale-row i {{
        display: block;
        height: 100%;
        background: linear-gradient(90deg, {SENDER}, {RECEIVER});
      }}
      .scale-row b {{
        color: {INK};
        font-weight: 600;
        text-align: right;
      }}
      .pair-list {{
        background: #FFFFFF;
        border: 1px solid {LINE};
      }}
      .pair-header, .pair-row {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 46px 42px;
        gap: 6px;
        align-items: center;
        padding: 6px 8px;
        border-bottom: 1px solid #EDF1F4;
        font-size: 10.5px;
      }}
      .pair-header {{
        color: {MUTED};
        background: #F7F9FA;
        text-transform: uppercase;
        letter-spacing: .02em;
      }}
      .pair-row:last-child {{ border-bottom: 0; }}
      .pair-row.active {{
        background: #F2F6F3;
        box-shadow: inset 3px 0 0 {SENDER};
      }}
      .pair-row span {{
        color: {INK};
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}
      .pair-row b, .pair-row em {{
        color: {INK};
        text-align: right;
        font-style: normal;
      }}
      .pair-row em {{ color: {MUTED}; }}
      .floating-toolbar {{
        position: absolute;
        z-index: 10;
        top: 12px;
        left: 12px;
        display: flex;
        gap: 6px;
      }}
      .mode {{
        border: 1px solid rgba(211, 218, 223, .95);
        background: rgba(255, 255, 255, .86);
        color: #4D5961;
        font: 400 11px/1 Arial, Helvetica, sans-serif;
        padding: 6px 8px;
        cursor: pointer;
      }}
      .mode:hover {{ background: #FFFFFF; color: {INK}; }}
      .mode.active {{
        background: {INK};
        border-color: {INK};
        color: #FFFFFF;
      }}
      .canvas-note {{
        position: absolute;
        left: 13px;
        bottom: 12px;
        z-index: 10;
        border: 1px solid rgba(227,232,236,.82);
        background: rgba(255,255,255,.82);
        color: {MUTED};
        font-size: 10.5px;
        padding: 6px 8px;
      }}
      .layer-key {{
        position: absolute;
        right: 13px;
        bottom: 12px;
        z-index: 10;
        display: grid;
        gap: 5px;
        border: 1px solid rgba(227,232,236,.78);
        background: rgba(255,255,255,.72);
        padding: 7px 8px;
        color: #526069;
        font-size: 10.5px;
        line-height: 1.1;
      }}
      .layer-key div {{
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
      }}
      .dot {{
        width: 8px;
        height: 8px;
        display: inline-block;
        border-radius: 999px;
        box-shadow: 0 0 0 1px rgba(255,255,255,.9);
      }}
      .sender-dot {{ background: {SENDER}; }}
      .receiver-dot {{ background: {RECEIVER}; }}
      .line-key {{
        display: inline-block;
        width: 18px;
        height: 2px;
        background: #344247;
      }}
      .empty {{ color: {MUTED}; font-size: 11px; }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const payload = {json.dumps(payload)};
      const baseFig = {fig_json};
      const plot = document.getElementById("ss-plot");
      const detail = document.getElementById("detail-box");
      const modeButtons = Array.from(document.querySelectorAll(".mode[data-mode]"));
      const resetButton = document.getElementById("reset-camera");
      const baseData = baseFig.data.map(t => JSON.parse(JSON.stringify(t)));
      const layout = JSON.parse(JSON.stringify(baseFig.layout));
      layout.margin = {{l:0, r:0, t:0, b:0}};
      const defaultCamera = layout.scene && layout.scene.camera ? JSON.parse(JSON.stringify(layout.scene.camera)) : null;
      const config = {{responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:true}};
      let currentMode = "overview";
      let selected = null;

      function esc(v) {{
        if (v === undefined || v === null || String(v) === "nan") return "";
        const div = document.createElement("div");
        div.textContent = String(v);
        return div.innerHTML;
      }}
      function row(k, v) {{
        if (v === undefined || v === null || v === "") return "";
        return `<div class="detail-row"><span>${{esc(k)}}</span><b>${{esc(v)}}</b></div>`;
      }}
      function num(v) {{
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
      }}
      function selectionTraces(d) {{
        if (!d) return [];
        let sx, sy, rx, ry;
        if (d[0] === "edge") {{
          sx = num(d[8]); sy = num(d[9]); rx = num(d[10]); ry = num(d[11]);
        }} else if (d[0] === "sender") {{
          sx = num(d[7]); sy = num(d[8]); rx = num(d[9]); ry = num(d[10]);
        }} else if (d[0] === "receiver") {{
          rx = num(d[7]); ry = num(d[8]); sx = num(d[9]); sy = num(d[10]);
        }} else {{
          return [];
        }}
        if ([sx, sy, rx, ry].some(v => v === null)) return [];
        return [
          {{type:"scatter3d", mode:"lines", x:[sx,rx], y:[sy,ry], z:[1,0], line:{{color:"#11181D", width:5}}, opacity:.78, hoverinfo:"skip", showlegend:false, name:"selected edge"}},
          {{type:"scatter3d", mode:"markers", x:[sx], y:[sy], z:[1], marker:{{size:9, color:"{SENDER}", line:{{width:1.3, color:"#FFFFFF"}}}}, hoverinfo:"skip", showlegend:false, name:"selected sender"}},
          {{type:"scatter3d", mode:"markers", x:[rx], y:[ry], z:[0], marker:{{size:9, color:"{RECEIVER}", line:{{width:1.3, color:"#FFFFFF"}}}}, hoverinfo:"skip", showlegend:false, name:"selected receiver"}}
        ];
      }}
      function dataForMode() {{
        const data = baseData.map(t => JSON.parse(JSON.stringify(t)));
        data.forEach(t => {{
          if (currentMode === "overview") {{
            if (t.name === "predicted communication") t.opacity = .28;
            if (t.name === "sending cells" || t.name === "receiving cells") t.opacity = .86;
          }} else if (currentMode === "focus") {{
            if (t.mode === "lines") t.opacity = .055;
            if (t.name === "sending cells" || t.name === "receiving cells") t.opacity = selected ? .22 : .78;
          }} else if (currentMode === "network") {{
            if (t.mode === "lines") t.opacity = Math.min((t.opacity || .2) * 1.75, .55);
            if (t.name === "sending cells" || t.name === "receiving cells") t.opacity = .92;
          }}
        }});
        return data.concat(selectionTraces(selected));
      }}
      function redraw() {{
        Plotly.react(plot, dataForMode(), layout, config);
      }}
      function setMode(mode) {{
        currentMode = mode;
        modeButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.mode === mode));
        redraw();
      }}
      function updateDetail(d) {{
        selected = d;
        const kind = d && d[0] ? String(d[0]) : "cell";
        if (kind === "edge") {{
          detail.innerHTML =
            `<div class="detail-title">${{esc(d[1])}} -> ${{esc(d[3])}}</div>` +
            row("sender", `${{d[1]}} (${{d[2]}})`) +
            row("receiver", `${{d[3]}} (${{d[4]}})`) +
            row("distance", d[5]) +
            row("sender niche", d[6]) +
            row("receiver niche", d[7]);
        }} else if (kind === "sender") {{
          detail.innerHTML =
            `<div class="detail-title">Sender: ${{esc(d[1])}}</div>` +
            row("cell type", d[2]) +
            row("niche", d[3]) +
            row("receiver", `${{d[4]}} (${{d[5]}})`) +
            row("distance", d[6]);
        }} else if (kind === "receiver") {{
          detail.innerHTML =
            `<div class="detail-title">Receiver: ${{esc(d[1])}}</div>` +
            row("cell type", d[2]) +
            row("niche", d[3]) +
            row("sender", `${{d[4]}} (${{d[5]}})`) +
            row("distance", d[6]);
        }} else {{
          detail.innerHTML =
            `<div class="detail-title">${{esc(d[1] || "cell")}}</div>` +
            row("cell type", d[2]) +
            row("niche", d[3]);
        }}
        if (currentMode === "overview") setMode("focus"); else redraw();
      }}
      Plotly.newPlot(plot, dataForMode(), layout, config).then(function() {{
        plot.on("plotly_click", function(ev) {{
          if (!ev || !ev.points || !ev.points.length) return;
          const d = ev.points[0].customdata;
          if (d) updateDetail(d);
        }});
      }});
      modeButtons.forEach(btn => btn.onclick = function() {{ setMode(this.dataset.mode); }});
      resetButton.onclick = function() {{
        if (defaultCamera) Plotly.relayout(plot, {{"scene.camera": defaultCamera}});
      }};
    </script>
    """
    components.html(html, height=786, scrolling=False)


def main() -> None:
    page_style()
    tables = load_slideseq_data()
    cells = tables["cells"]
    edges = tables["edges"]
    ranking = tables["ranking"]
    scale = tables["scale"]

    options = lr_options(ranking, edges)
    default_lr = "Cxcl12-Cxcr4" if "Cxcl12-Cxcr4" in options else options[0]

    st.markdown(
        """
        <div class="ss-topbar">
          <div>
            <div class="ss-title">ScaleComm Explorer</div>
            <div class="ss-subtitle">Interactive single-cell communication browser for ScaleComm outputs.</div>
          </div>
          <div class="ss-badges">
            <span class="ss-badge">ScaleComm only</span>
            <span class="ss-badge">LR filtering</span>
            <span class="ss-badge">cell-type pairs</span>
            <span class="ss-badge">clickable edges</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl = st.columns([1.05, 1.15, 1.50, 0.72], gap="small")
    with ctrl[0]:
        st.markdown('<div class="ss-control-card"><div class="ss-control-title">Dataset</div>', unsafe_allow_html=True)
        st.selectbox("Dataset", ["Slide-seqV2 hippocampus"], label_visibility="collapsed", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with ctrl[1]:
        st.markdown('<div class="ss-control-card"><div class="ss-control-title">Ligand-receptor</div>', unsafe_allow_html=True)
        lr = st.selectbox("Ligand-receptor", options, index=options.index(default_lr), label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    pair_opts, pair_summary = pair_options(edges, lr)
    with ctrl[2]:
        st.markdown('<div class="ss-control-card"><div class="ss-control-title">Cell-type interaction</div>', unsafe_allow_html=True)
        pair = st.selectbox("Cell-type interaction", pair_opts, index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with ctrl[3]:
        st.markdown('<div class="ss-control-card"><div class="ss-control-title">Edge threshold</div>', unsafe_allow_html=True)
        top_fraction = st.slider("top percent", min_value=5, max_value=50, value=20, step=5, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    lr_edges = edges[edges["LR"].eq(lr)].copy()
    pair_edges, show_edges = selected_edges(edges, lr, pair, top_fraction)
    stats = make_stats(cells, ranking, lr_edges, pair_edges, show_edges, lr, pair, top_fraction)
    fig = make_explorer_figure(cells, lr_edges, show_edges, lr, pair)
    st.markdown(
        '<div class="ss-caption">Default view hides unannotated beads from the selectable cell-type pairs; the canvas remains zoomable and clickable for event-level inspection.</div>',
        unsafe_allow_html=True,
    )
    render_explorer(fig, stats, scale_profile(scale, lr), pair_summary)


if __name__ == "__main__":
    main()
