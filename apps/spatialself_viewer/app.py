#!/usr/bin/env python
"""Interactive ScaleComm result viewer.

The app intentionally reads manuscript source tables instead of recomputing
model outputs, so it can be used as a lightweight companion to the figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(os.environ.get("SCALECOMM_VIEWER_ROOT", Path(__file__).resolve().parents[2])).resolve()


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    source_dir: Path
    description: str
    kind: str = "cosmx"


DATASETS = {
    "slideseqv2_hpc": DatasetConfig(
        key="slideseqv2_hpc",
        label="Slide-seqV2 hippocampus Cxcl12-Cxcr4",
        source_dir=ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v19_cytosignal_full_interactions/source_data",
        description="CytoSignal-style complete sending/receiving interaction view for ScaleComm Cxcl12-Cxcr4 results.",
        kind="slideseq",
    ),
    "cosmx_nsclc": DatasetConfig(
        key="cosmx_nsclc",
        label="CosMx NSCLC scale/context validation",
        source_dir=ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables",
        description="ScaleComm predictions, LR scale weights, receiver hotspots, and downstream target support.",
        kind="cosmx",
    ),
}

CELL_COLORS = {
    "tumor_proxy": "#FBC9C4",
    "t_cell_proxy": "#B9E5FA",
    "immune_proxy": "#D6EFF8",
    "myeloid_proxy": "#B9C1D1",
    "stromal_proxy": "#F3D78A",
    "endothelial_proxy": "#93CC82",
    "other_proxy": "#DDE2E9",
    "astrocyte": "#8CCB84",
    "oligodendrocyte": "#B8C1D2",
    "CA1_neuron": "#B8DFF2",
    "CA3_neuron": "#8EBBD8",
    "dentate_neuron": "#F2D578",
    "interneuron": "#F2B5AE",
    "microglia_macrophage": "#C9B3D7",
    "endothelial": "#9CC8BA",
    "polydendrocyte": "#C8B6A6",
    "entorhinal_neuron": "#A8C8E6",
    "ependymal": "#D8C3E6",
    "cajal_retzius": "#E7C4A9",
    "neurogenic": "#B7D9B1",
    "excitatory_neuron": "#F4CF90",
    "mural": "#B9B0A6",
    "choroid": "#D6C6A8",
}

LR_COLORS = {
    "SPP1-CD44": "#B98F2E",
    "CXCL10-CXCR3": "#C87370",
    "CXCL12-CXCR4": "#4F9B59",
    "CXCL9-CXCR3": "#5BAFD0",
    "CCL5-CCR5": "#80C3DE",
    "VEGFA-KDR": "#79B984",
    "CSF1-CSF1R": "#93A0B4",
    "TGFB1-TGFBR2": "#C87370",
}


def page_style() -> None:
    st.set_page_config(
        page_title="ScaleComm Viewer",
        page_icon="SS",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
          --ink: #20272c;
          --muted: #68747d;
          --line: #e2e6ea;
          --soft: #f7f8f9;
          --accent: #2f78a8;
        }
        .stApp { background: #ffffff; }
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stHeader"], [data-testid="collapsedControl"] { display: none; }
        .block-container {
          padding-top: 0.55rem;
          padding-bottom: 0.70rem;
          padding-left: 1.05rem;
          padding-right: 1.05rem;
          max-width: 1840px;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        h1 {
          font-size: 1.12rem !important;
          line-height: 1.1 !important;
          margin: 0 !important;
        }
        h2, h3 {
          font-size: 0.82rem !important;
          margin: 0.16rem 0 0.12rem 0 !important;
        }
        div[data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid var(--line);
          padding: 0.30rem 0.42rem;
          border-radius: 0;
        }
        div[data-testid="stMetricValue"] { font-size: 0.88rem; }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        .viewer-caption {
          color: var(--muted);
          font-size: 0.74rem;
          margin-top: 0;
          margin-bottom: 0;
        }
        .small-note {
          color: var(--muted);
          font-size: 0.72rem;
        }
        .control-card {
          border: 1px solid var(--line);
          background: #ffffff;
          padding: 0.62rem 0.72rem;
          margin-bottom: 0.55rem;
        }
        .lr-name {
          font-size: 1.26rem;
          font-weight: 700;
          line-height: 1.05;
          margin: 0.08rem 0 0.28rem 0;
        }
        .control-label {
          color: var(--muted);
          font-size: 0.70rem;
          text-transform: uppercase;
          letter-spacing: 0.02em;
          margin-bottom: 0.10rem;
        }
        .status-pill {
          display: inline-block;
          border: 1px solid var(--line);
          padding: 0.12rem 0.34rem;
          margin: 0.07rem 0.10rem 0.07rem 0;
          color: var(--ink);
          background: #fbfbfb;
          font-size: 0.68rem;
          white-space: nowrap;
        }
        .topbar {
          border: 1px solid var(--line);
          background: #ffffff;
          padding: 0.48rem 0.68rem;
          margin-bottom: 0.62rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
        }
        .topbar-title {
          font-size: 1.06rem;
          font-weight: 700;
          color: var(--ink);
          line-height: 1;
        }
        .topbar-sub {
          color: var(--muted);
          font-size: 0.74rem;
          margin-top: 0.16rem;
        }
        .topbar-badges {
          display: flex;
          gap: 0.35rem;
          align-items: center;
          flex-wrap: wrap;
          justify-content: flex-end;
        }
        .topbar-badge {
          border: 1px solid var(--line);
          background: #fbfbfb;
          color: var(--ink);
          padding: 0.13rem 0.36rem;
          font-size: 0.68rem;
          white-space: nowrap;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.42rem;
          margin: 0.32rem 0 0.24rem 0;
        }
        .metric-tile {
          border: 1px solid var(--line);
          background: #ffffff;
          padding: 0.40rem 0.46rem;
          min-height: 2.72rem;
        }
        .metric-label {
          color: var(--muted);
          font-size: 0.65rem;
          white-space: nowrap;
        }
        .metric-value {
          color: var(--ink);
          font-size: 0.94rem;
          font-weight: 700;
          line-height: 1.15;
          margin-top: 0.12rem;
        }
        .section-head {
          color: var(--ink);
          font-weight: 700;
          font-size: 0.80rem;
          line-height: 1.05;
          margin: 0.22rem 0 0.12rem 0;
        }
        .edge-list {
          border: 1px solid var(--line);
          background: #ffffff;
        }
        .edge-row {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 3.2rem 3.2rem;
          gap: 0.35rem;
          align-items: center;
          padding: 0.34rem 0.42rem;
          border-bottom: 1px solid #edf0f2;
          font-size: 0.70rem;
          color: var(--ink);
        }
        .edge-row:last-child { border-bottom: 0; }
        .edge-row span {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .edge-row b {
          font-size: 0.70rem;
          text-align: right;
          color: var(--accent);
        }
        .edge-row em {
          font-size: 0.68rem;
          text-align: right;
          color: var(--muted);
          font-style: normal;
        }
        .stPlotlyChart { margin-bottom: 0 !important; }
        div[data-testid="stVerticalBlock"] { gap: 0.30rem; }
        div[data-testid="column"] { padding-top: 0 !important; }
        section[data-testid="stSidebar"] { width: 21rem !important; }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stTextInput"] label {
          color: var(--muted) !important;
          font-size: 0.72rem !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
        .stTabs [data-baseweb="tab"] {
          height: 2.0rem;
          padding: 0 0.55rem;
        }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div { min-height: 2.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_tables(dataset_key: str) -> dict[str, pd.DataFrame]:
    cfg = DATASETS[dataset_key]
    if cfg.kind == "slideseq":
        return load_slideseq_tables(cfg)
    return load_cosmx_tables(cfg)


def load_cosmx_tables(cfg: DatasetConfig) -> dict[str, pd.DataFrame]:
    source = cfg.source_dir
    required = {
        "summary": "cosmx_lr_scale_context_summary.csv",
        "support": "cosmx_receiver_target_support.csv",
        "cells": "cosmx_cells_for_context_map.csv",
        "hotspots": "cosmx_hotspot_receiver_cells.csv",
        "scored": "cosmx_spatialself_scored_candidates.csv",
    }
    missing = [name for name in required.values() if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source table(s): {', '.join(missing)}")
    tables = {key: pd.read_csv(source / filename) for key, filename in required.items()}
    for key in ["cells", "hotspots"]:
        tables[key] = with_plot_coordinates(tables[key])
    tables["ranking"] = build_ranking(tables["summary"], tables["support"])
    tables["interactions"] = pd.DataFrame()
    return tables


def load_slideseq_tables(cfg: DatasetConfig) -> dict[str, pd.DataFrame]:
    source = cfg.source_dir
    run_dir = ROOT / "outputs/current_slideseqv2_hpc_full"
    full_input_dir = run_dir / "standardized_inputs/slideseqv2_mouse_hpc_full"
    full_pred_dir = run_dir / "spatialself_scale_full_e80"
    required = {
        "cells": "panel_a_cells.csv",
        "expr_score": "panel_b_expression_score.csv",
        "response": "panel_e_response.csv",
        "validation": "panel_f_validation.csv",
        "interactions": "panel_e_full_edges.csv",
        "summary": "summary.csv",
    }
    missing = [name for name in required.values() if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source table(s): {', '.join(missing)}")
    raw = {key: pd.read_csv(source / filename) for key, filename in required.items()}
    full_cells_path = full_input_dir / "cells.csv"
    if full_cells_path.exists():
        cells = pd.read_csv(full_cells_path)
        cells["x_plot"] = pd.to_numeric(cells["x"], errors="coerce")
        cells["y_plot"] = -pd.to_numeric(cells["y"], errors="coerce")
        cells["cell_type"] = cells.get("cell_type", "unannotated").fillna("unannotated").astype(str)
        cells["niche"] = cells.get("niche", cells["cell_type"]).fillna("unknown").astype(str)
        cells["Cxcl12"] = pd.to_numeric(cells.get("Cxcl12_expr", 0.0), errors="coerce").fillna(0.0)
        cells["Cxcr4"] = pd.to_numeric(cells.get("Cxcr4_expr", 0.0), errors="coerce").fillna(0.0)
    else:
        cells = raw["cells"].copy()
        cells = cells.rename(columns={"cell_type": "niche"}) if "niche" not in cells.columns else cells
        cells["cell_type"] = raw["cells"].get("cell_type", "annotated").astype(str)
        cells["niche"] = raw["cells"].get("cell_type", "annotated").astype(str)
        for col in ["x_plot", "y_plot"]:
            cells[col] = pd.to_numeric(cells[col], errors="coerce")
    expr = raw["expr_score"].copy()
    response = raw["response"].copy()
    lr = "Cxcl12-Cxcr4"
    hotspots = response.copy()
    hotspots["LR"] = lr
    hotspots["x"] = hotspots["x_plot"]
    hotspots["y"] = -hotspots["y_plot"]
    hotspots["receiver_activity"] = pd.to_numeric(hotspots.get("spatialself_score", 0.0), errors="coerce").fillna(0.0)
    hotspots["target_score"] = pd.to_numeric(hotspots.get("response_score", 0.0), errors="coerce").fillna(0.0)
    hotspots["is_top_receiver"] = hotspots.get("hotspot", False).astype(bool)
    hotspots["niche"] = hotspots.get("cell_type", "annotated").astype(str)
    hotspots["cell_type"] = hotspots.get("cell_type", "annotated").astype(str)

    top_edges_path = full_pred_dir / "top10000_edges_merged.csv"
    interactions = pd.read_csv(top_edges_path) if top_edges_path.exists() else raw["interactions"].copy()
    interactions = interactions[
        interactions.get("ligand", pd.Series(dtype=str)).astype(str).eq("Cxcl12")
        & interactions.get("receptor", pd.Series(dtype=str)).astype(str).eq("Cxcr4")
    ].copy()
    interactions["LR"] = lr
    interactions["score"] = pd.to_numeric(interactions["score"], errors="coerce")
    interactions["distance"] = pd.to_numeric(interactions["distance"], errors="coerce")
    if "sender_cell_type" not in interactions:
        interactions["sender_cell_type"] = interactions.get("sender_cell_type_y", interactions.get("sender_cell_type_x", pd.Series(["sender"] * len(interactions), index=interactions.index)))
    if "receiver_cell_type" not in interactions:
        interactions["receiver_cell_type"] = interactions.get("receiver_cell_type_y", interactions.get("receiver_cell_type_x", pd.Series(["receiver"] * len(interactions), index=interactions.index)))
    interactions["sender_cell_type"] = interactions["sender_cell_type"].fillna("unannotated").astype(str)
    interactions["receiver_cell_type"] = interactions["receiver_cell_type"].fillna("unannotated").astype(str)
    interactions["sender_niche"] = interactions.get("sender_niche", interactions["sender_cell_type"]).astype(str)
    interactions["receiver_niche"] = interactions.get("receiver_niche", interactions["receiver_cell_type"]).astype(str)
    if "sender_x_plot" not in interactions or "receiver_x_plot" not in interactions:
        pos = cells[["cell_id", "x_plot", "y_plot"]].drop_duplicates("cell_id")
        sender_pos = pos.rename(columns={"cell_id": "sender_id", "x_plot": "sender_x_plot", "y_plot": "sender_y_plot"})
        receiver_pos = pos.rename(columns={"cell_id": "receiver_id", "x_plot": "receiver_x_plot", "y_plot": "receiver_y_plot"})
        interactions = interactions.merge(sender_pos, on="sender_id", how="left").merge(receiver_pos, on="receiver_id", how="left")
    interactions = interactions.dropna(subset=["sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot"]).copy()

    ranking = pd.DataFrame([
        {
            "LR": lr,
            "mechanism": "long-range chemokine niche",
            "biological_category": "chemokine",
            "scale_class": "context-specific",
            "active_niche": "hippocampal Cxcl12-Cxcr4 niche",
            "mean_score": float(interactions["score"].mean()),
            "q95_score": float(interactions["score"].quantile(0.95)),
            "max_score": float(interactions["score"].max()),
            "target_spearman_r": np.nan,
            "top_receiver_target_lift": float(raw["validation"].loc[raw["validation"].group.eq("hotspot receivers"), "response_score"].mean() - raw["validation"].loc[raw["validation"].group.eq("other beads"), "response_score"].mean()),
            "top_median_distance": float(interactions["distance"].median()),
            "top_niche_match_rate": np.nan,
            "broad_weight": np.nan,
            "scale_0": 0.02,
            "scale_1": 0.08,
            "scale_2": 0.17,
            "scale_3": 0.26,
            "scale_4": 0.29,
            "scale_5": 0.18,
        }
    ])
    return {
        "cells": cells,
        "hotspots": hotspots,
        "scored": interactions,
        "interactions": interactions,
        "expr_score": expr,
        "response": response,
        "validation": raw["validation"],
        "ranking": ranking,
        "summary": raw["summary"],
        "support": pd.DataFrame(),
    }


def with_plot_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_plot"] = pd.to_numeric(out["x"], errors="coerce")
    out["y_plot"] = -pd.to_numeric(out["y"], errors="coerce")
    return out


def build_ranking(summary: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    support_cols = [
        "LR",
        "target_spearman_r",
        "target_spearman_p",
        "top_receiver_target_mean",
        "background_target_mean",
        "top_receiver_target_lift",
        "top_receiver_target_fold",
        "top_receiver_niche_mode",
        "top_receiver_cell_type_mode",
    ]
    cols = [c for c in support_cols if c in support.columns]
    merged = summary.merge(support[cols], on="LR", how="left", suffixes=("", "_support"))
    for col in [
        "mean_score",
        "q95_score",
        "max_score",
        "target_spearman_r",
        "top_receiver_target_lift",
        "broad_weight",
        "local_weight",
    ]:
        if col in merged:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def sort_ranking(df: pd.DataFrame, sort_by: str, query: str) -> pd.DataFrame:
    options = {
        "Mean score": "mean_score",
        "Target correlation": "target_spearman_r",
        "Target lift": "top_receiver_target_lift",
        "Broad scale weight": "broad_weight",
        "Max score": "max_score",
    }
    out = df.copy()
    if query:
        q = query.lower().replace(" ", "")
        out = out[out["LR"].astype(str).str.lower().str.replace(" ", "", regex=False).str.contains(q, regex=False)]
    col = options.get(sort_by, "mean_score")
    if col in out:
        out = out.sort_values(col, ascending=False, na_position="last")
    return out.reset_index(drop=True)


def format_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def lr_row(ranking: pd.DataFrame, lr: str) -> pd.Series:
    match = ranking[ranking["LR"].eq(lr)]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def selected_hotspots(hotspots: pd.DataFrame, lr: str) -> pd.DataFrame:
    out = hotspots[hotspots["LR"].eq(lr)].copy()
    if out.empty:
        return out
    for col in ["receiver_activity", "target_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "is_top_receiver" in out:
        out["is_top_receiver"] = out["is_top_receiver"].astype(bool)
    else:
        out["is_top_receiver"] = False
    return out


def selected_scored(scored: pd.DataFrame, lr: str) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    if "LR" in scored:
        out = scored[scored["LR"].astype(str).eq(lr)].copy()
    else:
        ligand, receptor = lr.split("-", 1)
        out = scored[(scored["ligand"].astype(str).eq(ligand)) & (scored["receptor"].astype(str).eq(receptor))].copy()
    for col in ["score", "distance", "ligand_expr", "receptor_expr", "coexpression_distance_score"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def make_spatial_map(cells: pd.DataFrame, hotspots: pd.DataFrame, lr: str, color_by: str) -> go.Figure:
    fig = go.Figure()
    for cell_type, group in cells.groupby("cell_type", sort=False):
        fig.add_trace(
            go.Scattergl(
                x=group["x_plot"],
                y=group["y_plot"],
                mode="markers",
                name=str(cell_type).replace("_proxy", ""),
                marker={
                    "size": 4.0,
                    "color": CELL_COLORS.get(cell_type, "#DDE2E9"),
                    "opacity": 0.38,
                    "line": {"width": 0},
                },
                hovertemplate=(
                    "cell=%{customdata[0]}<br>"
                    "type=%{customdata[1]}<br>"
                    "niche=%{customdata[2]}<extra></extra>"
                ),
                customdata=group[["cell_id", "cell_type", "niche"]].astype(str).to_numpy(),
                showlegend=False,
            )
        )

    hot = hotspots.copy()
    value_col = "target_score" if color_by == "Target signature" else "receiver_activity"
    values = hot[value_col].to_numpy(float)
    if len(values):
        cutoff = np.nanquantile(values, 0.55)
        display = hot[hot[value_col].ge(cutoff)].copy()
    else:
        display = hot
    fig.add_trace(
        go.Scattergl(
            x=display["x_plot"],
            y=display["y_plot"],
            mode="markers",
            name=color_by,
            marker={
                "size": 5.5,
                "color": display[value_col],
                "colorscale": [[0, "#F7F4EC"], [0.55, "#E6C06A"], [1, LR_COLORS.get(lr, "#4F9B59")]],
                "showscale": True,
                "colorbar": {"title": color_by, "thickness": 10, "len": 0.72},
                "opacity": 0.82,
                "line": {"width": 0},
            },
            customdata=display[["cell_id", "cell_type", "niche", "receiver_activity", "target_score", "is_top_receiver"]].to_numpy(),
            hovertemplate=(
                "cell=%{customdata[0]}<br>"
                "type=%{customdata[1]}<br>"
                "niche=%{customdata[2]}<br>"
                "receiver score=%{customdata[3]:.4f}<br>"
                "target signature=%{customdata[4]:.4f}<br>"
                "top hotspot=%{customdata[5]}<extra></extra>"
            ),
        )
    )
    top = hot[hot["is_top_receiver"]].copy()
    fig.add_trace(
        go.Scattergl(
            x=top["x_plot"],
            y=top["y_plot"],
            mode="markers",
            name="Top receiver",
            marker={
                "size": 9.5,
                "color": "rgba(255,255,255,0)",
                "line": {"color": LR_COLORS.get(lr, "#263137"), "width": 1.3},
            },
            customdata=top[["cell_id", "cell_type", "niche", "receiver_activity", "target_score"]].to_numpy(),
            hovertemplate=(
                "top receiver<br>"
                "cell=%{customdata[0]}<br>"
                "type=%{customdata[1]}<br>"
                "niche=%{customdata[2]}<br>"
                "receiver score=%{customdata[3]:.4f}<br>"
                "target signature=%{customdata[4]:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        title={"text": lr, "font": {"size": 16}},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": 0.01, "x": 0.01, "bgcolor": "rgba(255,255,255,0.75)"},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
    )
    return fig


def make_communication_view(
    cells: pd.DataFrame,
    hotspots: pd.DataFrame,
    scored: pd.DataFrame,
    lr: str,
    *,
    top_n: int,
    color_by: str,
    show_cell_types: bool,
    show_hotspots: bool,
    show_edges: bool,
) -> go.Figure:
    dat = scored.sort_values("score", ascending=False).head(top_n).copy()
    pos = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].copy()
    sender = pos.rename(
        columns={
            "cell_id": "sender_id",
            "x_plot": "sender_x",
            "y_plot": "sender_y",
            "cell_type": "sender_type",
            "niche": "sender_niche_meta",
        }
    )
    receiver = pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x_plot": "receiver_x",
            "y_plot": "receiver_y",
            "cell_type": "receiver_type",
            "niche": "receiver_niche_meta",
        }
    )
    if not dat.empty:
        dat = dat.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")

    fig = go.Figure()
    if show_cell_types:
        for cell_type, group in cells.groupby("cell_type", sort=False):
            fig.add_trace(
                go.Scattergl(
                    x=group["x_plot"],
                    y=group["y_plot"],
                    mode="markers",
                    name=str(cell_type).replace("_proxy", ""),
                    marker={
                        "size": 4.9,
                        "color": CELL_COLORS.get(cell_type, "#DDE2E9"),
                        "opacity": 0.78,
                        "line": {"width": 0.15, "color": "rgba(255,255,255,0.72)"},
                    },
                    customdata=group[["cell_id", "cell_type", "niche"]].astype(str).to_numpy(),
                    hovertemplate="cell=%{customdata[0]}<br>type=%{customdata[1]}<br>niche=%{customdata[2]}<extra></extra>",
                    showlegend=True,
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=cells["x_plot"],
                y=cells["y_plot"],
                mode="markers",
                name="cells",
                marker={"size": 4.4, "color": "#CDD4DA", "opacity": 0.58, "line": {"width": 0.15, "color": "rgba(255,255,255,0.65)"}},
                customdata=cells[["cell_id", "cell_type", "niche"]].astype(str).to_numpy(),
                hovertemplate="cell=%{customdata[0]}<br>type=%{customdata[1]}<br>niche=%{customdata[2]}<extra></extra>",
                showlegend=False,
            )
        )

    color = LR_COLORS.get(lr, "#4F9B59")
    edge_trace_count = 0
    if show_edges and not dat.empty:
        score = dat["score"].to_numpy(float)
        if np.nanmax(score) > np.nanmin(score):
            edge_width = 0.55 + 2.15 * (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score))
        else:
            edge_width = np.full(len(dat), 1.15)
        # Plot in small batches so stronger edges are visually clearer without
        # creating one trace per edge for large selections.
        bucket_count = min(4, max(1, len(dat)))
        ranks = pd.Series(score).rank(method="first", pct=True).to_numpy(float)
        buckets = np.minimum((ranks * bucket_count).astype(int), bucket_count - 1)
        for bucket in sorted(np.unique(buckets)):
            sub = dat[buckets == bucket]
            x_lines: list[float | None] = []
            y_lines: list[float | None] = []
            for row in sub.itertuples():
                x_lines.extend([row.sender_x, row.receiver_x, None])
                y_lines.extend([row.sender_y, row.receiver_y, None])
            fig.add_trace(
                go.Scattergl(
                    x=x_lines,
                    y=y_lines,
                    mode="lines",
                    line={"color": color, "width": float(np.nanmedian(edge_width[buckets == bucket]))},
                    opacity=0.10 + 0.11 * float(bucket + 1),
                    hoverinfo="skip",
                    name="communication lines" if bucket == int(np.max(buckets)) else None,
                    showlegend=bool(bucket == int(np.max(buckets))),
                )
            )
            edge_trace_count += 1

        mid = dat.copy()
        mid["mid_x"] = (mid["sender_x"] + mid["receiver_x"]) / 2
        mid["mid_y"] = (mid["sender_y"] + mid["receiver_y"]) / 2
        fig.add_trace(
            go.Scattergl(
                x=mid["mid_x"],
                y=mid["mid_y"],
                mode="markers",
                marker={"size": 8, "color": "rgba(255,255,255,0)", "line": {"width": 0}},
                customdata=mid[["sender_id", "receiver_id", "sender_cell_type", "receiver_cell_type", "score", "distance", "sender_niche", "receiver_niche"]].astype(object).to_numpy(),
                hovertemplate=(
                    "sender=%{customdata[0]}<br>"
                    "receiver=%{customdata[1]}<br>"
                    "%{customdata[2]} -> %{customdata[3]}<br>"
                    "score=%{customdata[4]:.4f}<br>"
                    "distance=%{customdata[5]:.1f}<br>"
                    "sender niche=%{customdata[6]}<br>"
                    "receiver niche=%{customdata[7]}<extra></extra>"
                ),
                showlegend=False,
                name="edge hover",
            )
        )

    if show_hotspots and not hotspots.empty:
        value_col = "target_score" if color_by == "Target signature" else "receiver_activity"
        hot = hotspots.copy()
        hot[value_col] = pd.to_numeric(hot[value_col], errors="coerce").fillna(0.0)
        cutoff = np.nanquantile(hot[value_col].to_numpy(float), 0.80)
        display = hot[hot[value_col].ge(cutoff)].copy()
        fig.add_trace(
            go.Scattergl(
                x=display["x_plot"],
                y=display["y_plot"],
                mode="markers",
                name="receiver hotspot",
                marker={
                    "size": 7.8,
                    "color": display[value_col],
                    "colorscale": [[0, "#F7F4EC"], [0.52, "#E6C06A"], [1, color]],
                    "showscale": False,
                    "opacity": 0.84,
                    "line": {"width": 0.35, "color": "rgba(255,255,255,0.78)"},
                },
                customdata=display[["cell_id", "cell_type", "niche", "receiver_activity", "target_score", "is_top_receiver"]].to_numpy(),
                hovertemplate=(
                    "cell=%{customdata[0]}<br>"
                    "type=%{customdata[1]}<br>"
                    "niche=%{customdata[2]}<br>"
                    "receiver score=%{customdata[3]:.4f}<br>"
                    "target signature=%{customdata[4]:.4f}<br>"
                    "top hotspot=%{customdata[5]}<extra></extra>"
                ),
            )
        )

    if not dat.empty:
        endpoints_sender = dat[["sender_id", "sender_x", "sender_y", "sender_type", "sender_niche_meta", "score", "distance"]].rename(
            columns={"sender_id": "cell_id", "sender_x": "x", "sender_y": "y", "sender_type": "cell_type", "sender_niche_meta": "niche"}
        )
        endpoints_receiver = dat[["receiver_id", "receiver_x", "receiver_y", "receiver_type", "receiver_niche_meta", "score", "distance"]].rename(
            columns={"receiver_id": "cell_id", "receiver_x": "x", "receiver_y": "y", "receiver_type": "cell_type", "receiver_niche_meta": "niche"}
        )
        fig.add_trace(
            go.Scattergl(
                x=endpoints_sender["x"],
                y=endpoints_sender["y"],
                mode="markers",
                name="sender",
                marker={"size": 7.2, "symbol": "circle", "color": color, "opacity": 0.88, "line": {"color": "white", "width": 0.85}},
                customdata=endpoints_sender[["cell_id", "cell_type", "niche", "score", "distance"]].to_numpy(),
                hovertemplate="sender=%{customdata[0]}<br>type=%{customdata[1]}<br>niche=%{customdata[2]}<br>edge score=%{customdata[3]:.4f}<br>distance=%{customdata[4]:.1f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=endpoints_receiver["x"],
                y=endpoints_receiver["y"],
                mode="markers",
                name="receiver",
                marker={"size": 8.8, "symbol": "circle-open", "color": color, "opacity": 0.98, "line": {"width": 1.55}},
                customdata=endpoints_receiver[["cell_id", "cell_type", "niche", "score", "distance"]].to_numpy(),
                hovertemplate="receiver=%{customdata[0]}<br>type=%{customdata[1]}<br>niche=%{customdata[2]}<br>edge score=%{customdata[3]:.4f}<br>distance=%{customdata[4]:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=705,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "y": 0.005,
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(226,230,234,0.75)",
            "borderwidth": 1,
            "font": {"size": 9, "color": "#263137"},
            "itemsizing": "constant",
        },
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        dragmode="pan",
        hovermode="closest",
        uirevision=f"{lr}-{top_n}-{show_cell_types}-{show_hotspots}-{show_edges}-{edge_trace_count}",
    )
    return fig



def make_3d_layered_interaction_view(
    cells: pd.DataFrame,
    interactions: pd.DataFrame,
    lr: str,
    *,
    selected_cell_types: list[str] | None = None,
    show_background: bool = True,
    filter_edges_by_cell_type: bool = False,
) -> go.Figure:
    dat = interactions[interactions.get("LR", lr).astype(str).eq(lr)].copy() if "LR" in interactions else interactions.copy()
    dat = dat.sort_values("score", ascending=False).copy()
    if dat.empty:
        return empty_figure("No complete sending-receiving interactions for this LR")

    required = ["sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot"]
    dat = dat.dropna(subset=[c for c in required if c in dat.columns]).copy()
    selected = [str(x) for x in (selected_cell_types or [])]
    if filter_edges_by_cell_type and selected:
        dat = dat[
            dat["sender_cell_type"].astype(str).isin(selected)
            | dat["receiver_cell_type"].astype(str).isin(selected)
        ].copy()
    if dat.empty:
        return empty_figure("No interactions remain after the current cell-type filter")

    xmin = float(np.nanmin([dat["sender_x_plot"].min(), dat["receiver_x_plot"].min(), cells["x_plot"].min()]))
    xmax = float(np.nanmax([dat["sender_x_plot"].max(), dat["receiver_x_plot"].max(), cells["x_plot"].max()]))
    ymin = float(np.nanmin([dat["sender_y_plot"].min(), dat["receiver_y_plot"].min(), cells["y_plot"].min()]))
    ymax = float(np.nanmax([dat["sender_y_plot"].max(), dat["receiver_y_plot"].max(), cells["y_plot"].max()]))
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    def nx(v: pd.Series) -> np.ndarray:
        return (pd.to_numeric(v, errors="coerce").to_numpy(float) - xmin) / xspan

    def ny(v: pd.Series) -> np.ndarray:
        return (pd.to_numeric(v, errors="coerce").to_numpy(float) - ymin) / yspan

    fig = go.Figure()

    def add_layer_plane(z: float, color: str, name: str) -> None:
        fig.add_trace(
            go.Surface(
                x=[[0.0, 1.0], [0.0, 1.0]],
                y=[[0.0, 0.0], [1.0, 1.0]],
                z=[[z, z], [z, z]],
                surfacecolor=[[0, 0], [0, 0]],
                colorscale=[[0, color], [1, color]],
                opacity=0.070,
                showscale=False,
                hoverinfo="skip",
                name=name,
                showlegend=False,
            )
        )
        border_x = [0.0, 1.0, 1.0, 0.0, 0.0, None]
        border_y = [0.0, 0.0, 1.0, 1.0, 0.0, None]
        fig.add_trace(
            go.Scatter3d(
                x=border_x,
                y=border_y,
                z=[z, z, z, z, z, None],
                mode="lines",
                line={"color": "rgba(70,82,90,0.28)", "width": 1.2},
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} outline",
            )
        )

    add_layer_plane(1.0, "#E9F3E5", "sending layer")
    add_layer_plane(0.0, "#E8EEF8", "receiving layer")

    if show_background:
        bg_cells = cells.dropna(subset=["x_plot", "y_plot"]).copy()
        bg_custom = np.column_stack(
            [
                np.full(len(bg_cells), "cell"),
                bg_cells["cell_id"].astype(str).to_numpy(),
                bg_cells["cell_type"].astype(str).to_numpy(),
                np.full(len(bg_cells), "background"),
                np.full(len(bg_cells), ""),
                np.full(len(bg_cells), ""),
                np.full(len(bg_cells), ""),
                np.full(len(bg_cells), ""),
            ]
        )
        bg_x = nx(bg_cells["x_plot"])
        bg_y = ny(bg_cells["y_plot"])
        for z, opacity, name in [(1.0, 0.13, "tissue context: sender layer"), (0.0, 0.09, "tissue context: receiver layer")]:
            fig.add_trace(
                go.Scatter3d(
                    x=bg_x,
                    y=bg_y,
                    z=np.full(len(bg_cells), z),
                    mode="markers",
                    name=name,
                    marker={"size": 1.75, "color": "#BFC7CF", "opacity": opacity, "line": {"width": 0}},
                    customdata=bg_custom,
                    hovertemplate="cell=%{customdata[1]}<br>type=%{customdata[2]}<extra></extra>",
                    showlegend=False,
                )
            )

        highlighted = bg_cells[bg_cells["cell_type"].astype(str).isin(selected)].copy() if selected else bg_cells.iloc[0:0].copy()
        for cell_type, group in highlighted.groupby(highlighted["cell_type"].astype(str), sort=True):
            color = CELL_COLORS.get(cell_type, "#D6DAE2")
            cx = nx(group["x_plot"])
            cy = ny(group["y_plot"])
            custom = np.column_stack(
                [
                    np.full(len(group), "cell"),
                    group["cell_id"].astype(str).to_numpy(),
                    group["cell_type"].astype(str).to_numpy(),
                    np.full(len(group), "highlighted cell type"),
                    np.full(len(group), ""),
                    np.full(len(group), ""),
                    np.full(len(group), ""),
                    np.full(len(group), ""),
                ]
            )
            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=np.full(len(group), 1.0),
                    mode="markers",
                    name=cell_type,
                    legendgroup=f"cell-{cell_type}",
                    marker={"size": 3.2, "color": color, "opacity": 0.82, "line": {"width": 0.18, "color": "rgba(255,255,255,0.65)"}},
                    customdata=custom,
                    hovertemplate="cell=%{customdata[1]}<br>type=%{customdata[2]}<br>layer=sending<extra></extra>",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=np.zeros(len(group)),
                    mode="markers",
                    name=cell_type,
                    legendgroup=f"cell-{cell_type}",
                    marker={"size": 3.2, "color": color, "opacity": 0.44, "line": {"width": 0.18, "color": "rgba(255,255,255,0.65)"}},
                    customdata=custom,
                    hovertemplate="cell=%{customdata[1]}<br>type=%{customdata[2]}<br>layer=receiving<extra></extra>",
                    showlegend=False,
                )
            )

    sx = nx(dat["sender_x_plot"])
    sy = ny(dat["sender_y_plot"])
    rx = nx(dat["receiver_x_plot"])
    ry = ny(dat["receiver_y_plot"])
    score = pd.to_numeric(dat["score"], errors="coerce").fillna(0).to_numpy(float)
    q = pd.Series(score).rank(method="average", pct=True).to_numpy(float)
    color = LR_COLORS.get(lr, "#243B73")

    score_norm = (score - np.nanmin(score)) / max(float(np.nanmax(score) - np.nanmin(score)), 1e-9)
    low_mask = q < 0.82
    high_mask = ~low_mask

    def add_edge_bundle(mask: np.ndarray, *, width: float, opacity: float, name: str, showlegend: bool) -> None:
        if not mask.any():
            return
        x_lines: list[float | None] = []
        y_lines: list[float | None] = []
        z_lines: list[float | None] = []
        for x0, y0, x1, y1 in zip(sx[mask], sy[mask], rx[mask], ry[mask], strict=False):
            x_lines.extend([float(x0), float(x1), None])
            y_lines.extend([float(y0), float(y1), None])
            z_lines.extend([1.0, 0.0, None])
        fig.add_trace(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line={"color": color, "width": width},
                opacity=opacity,
                hoverinfo="skip",
                showlegend=showlegend,
                name=name,
            )
        )

    add_edge_bundle(low_mask, width=0.65, opacity=0.075, name="all interactions", showlegend=False)
    add_edge_bundle(high_mask, width=1.9, opacity=0.36, name="top-scoring interactions", showlegend=True)

    fig.add_trace(
        go.Scatter3d(
            x=(sx + rx) / 2,
            y=(sy + ry) / 2,
            z=np.full(len(dat), 0.5),
            mode="markers",
            name="edge hover",
            marker={"size": 3.6, "color": "rgba(0,0,0,0.01)", "opacity": 0.02, "line": {"width": 0}},
            customdata=np.column_stack(
                [
                    np.full(len(dat), "edge"),
                    dat["sender_id"].astype(str).to_numpy(),
                    dat["sender_cell_type"].astype(str).to_numpy(),
                    dat["receiver_id"].astype(str).to_numpy(),
                    dat["receiver_cell_type"].astype(str).to_numpy(),
                    pd.to_numeric(dat["score"], errors="coerce").round(4).astype(str).to_numpy(),
                    pd.to_numeric(dat["distance"], errors="coerce").round(1).astype(str).to_numpy(),
                    dat.get("sender_niche", pd.Series([""] * len(dat))).astype(str).to_numpy(),
                    dat.get("receiver_niche", pd.Series([""] * len(dat))).astype(str).to_numpy(),
                    pd.Series(sx).round(5).astype(str).to_numpy(),
                    pd.Series(sy).round(5).astype(str).to_numpy(),
                    pd.Series(rx).round(5).astype(str).to_numpy(),
                    pd.Series(ry).round(5).astype(str).to_numpy(),
                ]
            ),
            hovertemplate="sender=%{customdata[1]} (%{customdata[2]})<br>receiver=%{customdata[3]} (%{customdata[4]})<br>score=%{customdata[5]}<br>distance=%{customdata[6]}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=np.ones(len(dat)),
            mode="markers",
            name="sending cells",
            marker={
                "size": 4.2 + 2.0 * score_norm,
                "color": score,
                "colorscale": [[0, "#DCEED8"], [0.55, "#93CC82"], [1, "#3F8E50"]],
                "showscale": False,
                "opacity": 0.84,
                "line": {"width": 0.35, "color": "white"},
            },
            customdata=np.column_stack(
                [
                    np.full(len(dat), "sender"),
                    dat["sender_id"].astype(str).to_numpy(),
                    dat["sender_cell_type"].astype(str).to_numpy(),
                    dat.get("sender_niche", pd.Series([""] * len(dat))).astype(str).to_numpy(),
                    dat["receiver_id"].astype(str).to_numpy(),
                    dat["receiver_cell_type"].astype(str).to_numpy(),
                    pd.to_numeric(dat["score"], errors="coerce").round(4).astype(str).to_numpy(),
                    pd.to_numeric(dat["distance"], errors="coerce").round(1).astype(str).to_numpy(),
                    pd.Series(sx).round(5).astype(str).to_numpy(),
                    pd.Series(sy).round(5).astype(str).to_numpy(),
                    pd.Series(rx).round(5).astype(str).to_numpy(),
                    pd.Series(ry).round(5).astype(str).to_numpy(),
                ]
            ),
            hovertemplate="sender=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>receiver=%{customdata[4]} (%{customdata[5]})<br>score=%{customdata[6]}<br>distance=%{customdata[7]}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=rx,
            y=ry,
            z=np.zeros(len(dat)),
            mode="markers",
            name="receiving cells",
            marker={
                "size": 4.2 + 2.0 * score_norm,
                "color": score,
                "colorscale": [[0, "#DCE5F4"], [0.55, "#7F9BCB"], [1, "#243B73"]],
                "showscale": False,
                "opacity": 0.86,
                "line": {"width": 0.35, "color": "white"},
            },
            customdata=np.column_stack(
                [
                    np.full(len(dat), "receiver"),
                    dat["receiver_id"].astype(str).to_numpy(),
                    dat["receiver_cell_type"].astype(str).to_numpy(),
                    dat.get("receiver_niche", pd.Series([""] * len(dat))).astype(str).to_numpy(),
                    dat["sender_id"].astype(str).to_numpy(),
                    dat["sender_cell_type"].astype(str).to_numpy(),
                    pd.to_numeric(dat["score"], errors="coerce").round(4).astype(str).to_numpy(),
                    pd.to_numeric(dat["distance"], errors="coerce").round(1).astype(str).to_numpy(),
                    pd.Series(rx).round(5).astype(str).to_numpy(),
                    pd.Series(ry).round(5).astype(str).to_numpy(),
                    pd.Series(sx).round(5).astype(str).to_numpy(),
                    pd.Series(sy).round(5).astype(str).to_numpy(),
                ]
            ),
            hovertemplate="receiver=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>sender=%{customdata[4]} (%{customdata[5]})<br>score=%{customdata[6]}<br>distance=%{customdata[7]}<extra></extra>",
        )
    )

    fig.add_annotation(x=0.025, y=0.965, xref="paper", yref="paper", text="Cxcl12 sending layer", showarrow=False, font={"size": 12, "color": "#3F8E50"}, xanchor="left")
    fig.add_annotation(x=0.025, y=0.075, xref="paper", yref="paper", text="Cxcr4 receiving layer", showarrow=False, font={"size": 12, "color": "#243B73"}, xanchor="left")
    fig.add_annotation(x=0.985, y=0.045, xref="paper", yref="paper", text=f"{len(dat):,} complete interactions", showarrow=False, font={"size": 10, "color": "#68747d"}, xanchor="right")

    fig.update_layout(
        height=730,
        margin={"l": 0, "r": 0, "t": 6, "b": 0},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "y": 0.010,
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.74)",
            "bordercolor": "rgba(226,230,234,0.75)",
            "borderwidth": 1,
            "font": {"size": 8.5, "color": "#263137"},
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        scene={
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False, "showbackground": False, "range": [-0.06, 1.08]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.55, "y": 1.0, "z": 0.48},
            "camera": {"eye": {"x": 1.35, "y": -1.62, "z": 0.72}, "center": {"x": 0.0, "y": 0.0, "z": -0.06}},
        },
        hovermode="closest",
        uirevision=f"3d-layered-{lr}-{len(dat)}-{show_background}-{filter_edges_by_cell_type}-{'|'.join(selected)}",
    )
    return fig


def render_clickable_3d_figure(fig: go.Figure, *, height: int = 730) -> None:
    fig_json = fig.to_json()
    html = f"""
    <div id="spatialself-viewer-shell" style="width:100%;height:{height}px;display:grid;grid-template-columns:minmax(0,1fr) 318px;gap:12px;background:#fff;font-family:Arial,Helvetica,sans-serif;">
      <div id="spatialself-plot-card" style="position:relative;background:#ffffff;border:1px solid #E2E6EA;overflow:hidden;">
        <div id="spatialself-3d-plot" style="width:100%;height:100%;"></div>
        <div id="spatialself-modebar" style="position:absolute;left:14px;top:12px;display:flex;gap:6px;z-index:12;">
          <button class="ss-mode active" data-mode="overview">Overview</button>
          <button class="ss-mode" data-mode="focus">Focus selected</button>
          <button class="ss-mode" data-mode="full">Full network</button>
          <button id="ss-reset" class="ss-mode">Reset view</button>
        </div>
        <div style="position:absolute;left:16px;bottom:14px;background:rgba(255,255,255,0.82);border:1px solid rgba(226,230,234,0.72);padding:6px 8px;font-size:11px;color:#6F7A83;z-index:12;">
          Drag to rotate · scroll to zoom · click a cell or edge
        </div>
      </div>
      <aside id="spatialself-detail" style="border:1px solid #E2E6EA;background:#FBFCFD;padding:14px 14px 12px 14px;color:#20272c;overflow:hidden;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
          <div>
            <div style="font-size:10px;color:#6D777F;text-transform:uppercase;letter-spacing:.05em;">ScaleComm selection</div>
            <div id="ss-detail-title" style="font-size:15px;font-weight:700;color:#20272c;margin-top:3px;line-height:1.25;">Cxcl12-Cxcr4 overview</div>
          </div>
          <div id="ss-detail-badge" style="border:1px solid #DDE3E8;background:#FFFFFF;color:#68747D;font-size:10px;padding:4px 6px;white-space:nowrap;">overview</div>
        </div>
        <div id="ss-detail-body" style="font-size:12px;line-height:1.5;color:#35414A;">
          <div style="border-top:1px solid #E8EDF1;padding:9px 0;">The 3D map separates Cxcl12-sending cells and Cxcr4-receiving cells into two spatial layers.</div>
          <div style="border-top:1px solid #E8EDF1;padding:9px 0;">Click a sender, receiver, or interaction to inspect one communication event.</div>
        </div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid #E8EDF1;">
          <div style="font-size:10px;color:#6D777F;text-transform:uppercase;letter-spacing:.05em;margin-bottom:7px;">View logic</div>
          <div style="display:grid;gap:7px;font-size:11px;color:#5F6B74;">
            <div><span style="display:inline-block;width:9px;height:9px;background:#93CC82;margin-right:6px;"></span>Sender layer: Cxcl12-associated cells</div>
            <div><span style="display:inline-block;width:9px;height:9px;background:#7F9BCB;margin-right:6px;"></span>Receiver layer: Cxcr4-associated cells</div>
            <div><span style="display:inline-block;width:18px;height:2px;background:#4F9B59;margin-right:6px;vertical-align:middle;"></span>Line: predicted communication</div>
          </div>
        </div>
      </aside>
    </div>
    <style>
      .ss-mode {{
        border:1px solid rgba(210,216,222,0.95);background:rgba(255,255,255,0.82);color:#55616A;
        font-family:Arial,Helvetica,sans-serif;font-size:11px;padding:5px 8px;cursor:pointer;line-height:1;
      }}
      .ss-mode:hover {{ background:#FFFFFF; color:#20272c; }}
      .ss-mode.active {{ background:#263137; border-color:#263137; color:#FFFFFF; }}
      @media (max-width: 950px) {{
        #spatialself-viewer-shell {{ grid-template-columns:1fr; height:auto; }}
        #spatialself-plot-card {{ height:560px; }}
        #spatialself-detail {{ min-height:260px; }}
      }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
    const baseFig = {fig_json};
    const plot = document.getElementById("spatialself-3d-plot");
    const title = document.getElementById("ss-detail-title");
    const body = document.getElementById("ss-detail-body");
    const badge = document.getElementById("ss-detail-badge");
    const modeButtons = Array.from(document.querySelectorAll(".ss-mode[data-mode]"));
    const resetButton = document.getElementById("ss-reset");
    const baseData = baseFig.data.map(trace => JSON.parse(JSON.stringify(trace)));
    const layout = JSON.parse(JSON.stringify(baseFig.layout));
    layout.margin = {{l:0, r:0, t:0, b:0}};
    layout.showlegend = false;
    const defaultCamera = layout.scene && layout.scene.camera ? JSON.parse(JSON.stringify(layout.scene.camera)) : null;
    let selected = null;
    let currentMode = "overview";
    const config = {{responsive:true, displaylogo:false, scrollZoom:true, modeBarButtonsToRemove:["lasso2d","select2d"]}};

    function esc(v) {{
      if (v === undefined || v === null || v === "nan") return "";
      const div = document.createElement("div");
      div.textContent = String(v);
      return div.innerHTML;
    }}
    function row(label, value) {{
      if (value === undefined || value === null || value === "") return "";
      return `<div style="display:grid;grid-template-columns:86px minmax(0,1fr);gap:8px;border-top:1px solid #E8EDF1;padding:7px 0;">
        <div style="color:#78838B;">${{esc(label)}}</div><div style="font-weight:600;color:#263137;word-break:break-word;">${{esc(value)}}</div></div>`;
    }}
    function asNum(v) {{ const n = Number(v); return Number.isFinite(n) ? n : null; }}
    function selectedTrace(d) {{
      if (!d) return [];
      let sx, sy, rx, ry;
      if (d[0] === "edge") {{ sx=asNum(d[9]); sy=asNum(d[10]); rx=asNum(d[11]); ry=asNum(d[12]); }}
      else if (d[0] === "sender") {{ sx=asNum(d[8]); sy=asNum(d[9]); rx=asNum(d[10]); ry=asNum(d[11]); }}
      else if (d[0] === "receiver") {{ rx=asNum(d[8]); ry=asNum(d[9]); sx=asNum(d[10]); sy=asNum(d[11]); }}
      else return [];
      if ([sx, sy, rx, ry].some(v => v === null)) return [];
      return [
        {{type:"scatter3d", mode:"lines", x:[sx,rx], y:[sy,ry], z:[1,0], hoverinfo:"skip", showlegend:false, line:{{color:"#20272C", width:5}}, opacity:0.82, name:"selected interaction"}},
        {{type:"scatter3d", mode:"markers", x:[sx], y:[sy], z:[1], hoverinfo:"skip", showlegend:false, marker:{{size:9, color:"#3F8E50", line:{{width:1.3,color:"white"}}}}, name:"selected sender"}},
        {{type:"scatter3d", mode:"markers", x:[rx], y:[ry], z:[0], hoverinfo:"skip", showlegend:false, marker:{{size:9, color:"#243B73", line:{{width:1.3,color:"white"}}}}, name:"selected receiver"}}
      ];
    }}
    function dataForMode() {{
      const d = baseData.map(t => JSON.parse(JSON.stringify(t)));
      if (currentMode === "overview") {{
        d.forEach(t => {{
          if (t.name === "all interactions") t.opacity = 0.018;
          if (t.name === "top-scoring interactions") t.opacity = 0.22;
          if (t.name === "edge hover") {{ t.marker = t.marker || {{}}; t.marker.size = 7; t.marker.opacity = 0.018; }}
        }});
      }} else if (currentMode === "focus" && selected) {{
        d.forEach(t => {{
          if (["all interactions", "top-scoring interactions"].includes(t.name)) {{ t.opacity = 0.025; }}
          if (["sending cells", "receiving cells"].includes(t.name)) {{ t.opacity = 0.22; }}
        }});
      }} else if (currentMode === "full") {{
        d.forEach(t => {{ if (t.name === "all interactions") t.opacity = 0.16; if (t.name === "top-scoring interactions") t.opacity = 0.50; }});
      }}
      return d.concat(selectedTrace(selected));
    }}
    function draw() {{ Plotly.react(plot, dataForMode(), layout, config); }}
    function setMode(mode) {{
      currentMode = mode;
      modeButtons.forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
      draw();
    }}
    function updatePanel(d) {{
      selected = d;
      const kind = d && d[0] ? String(d[0]) : "point";
      if (kind === "edge") {{
        badge.textContent = "communication";
        title.textContent = `${{d[1]}} -> ${{d[3]}}`;
        body.innerHTML = row("sender", `${{d[1]}} (${{d[2]}})`) + row("receiver", `${{d[3]}} (${{d[4]}})`) + row("score", d[5]) + row("distance", d[6]) + row("sender niche", d[7]) + row("receiver niche", d[8]);
      }} else if (kind === "sender") {{
        badge.textContent = "sender";
        title.textContent = d[1];
        body.innerHTML = row("cell type", d[2]) + row("niche", d[3]) + row("receiver", `${{d[4]}} (${{d[5]}})`) + row("score", d[6]) + row("distance", d[7]);
      }} else if (kind === "receiver") {{
        badge.textContent = "receiver";
        title.textContent = d[1];
        body.innerHTML = row("cell type", d[2]) + row("niche", d[3]) + row("sender", `${{d[4]}} (${{d[5]}})`) + row("score", d[6]) + row("distance", d[7]);
      }} else {{
        badge.textContent = "cell";
        title.textContent = d[1] || "cell";
        body.innerHTML = row("cell type", d[2]) + row("class", d[3]);
      }}
      if (currentMode === "overview") setMode("focus"); else draw();
    }}
    Plotly.newPlot(plot, dataForMode(), layout, config).then(function() {{
      plot.on("plotly_click", function(ev) {{
        if (!ev || !ev.points || !ev.points.length) return;
        const d = ev.points[0].customdata;
        if (d) updatePanel(d);
      }});
    }});
    modeButtons.forEach(b => b.onclick = function() {{ setMode(this.dataset.mode); }});
    resetButton.onclick = function() {{ if (defaultCamera) Plotly.relayout(plot, {{"scene.camera": defaultCamera}}); }};
    </script>
    """
    components.html(html, height=height + 8, scrolling=False)


def slideseq_pair_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame(columns=["pair", "sender_cell_type", "receiver_cell_type", "n_edges", "mean_score", "median_distance", "max_score"])
    dat = interactions.copy()
    dat["sender_cell_type"] = dat["sender_cell_type"].astype(str)
    dat["receiver_cell_type"] = dat["receiver_cell_type"].astype(str)
    dat["pair"] = dat["sender_cell_type"] + " -> " + dat["receiver_cell_type"]
    out = (
        dat.groupby(["pair", "sender_cell_type", "receiver_cell_type"], as_index=False)
        .agg(
            n_edges=("score", "size"),
            mean_score=("score", "mean"),
            median_distance=("distance", "median"),
            max_score=("score", "max"),
        )
        .sort_values(["n_edges", "mean_score"], ascending=False)
        .reset_index(drop=True)
    )
    return out


def make_slideseq_v2_spatial_map(
    cells: pd.DataFrame,
    expr_score: pd.DataFrame,
    response: pd.DataFrame,
    interactions: pd.DataFrame,
    selected_pair: str,
    *,
    overlay: str = "Communication",
    height: int = 690,
    show_pair_edges: bool = True,
) -> go.Figure:
    fig = go.Figure()
    cells = cells.dropna(subset=["x_plot", "y_plot"]).copy()
    expr_score = expr_score.dropna(subset=["x_plot", "y_plot"]).copy()
    response = response.dropna(subset=["x_plot", "y_plot"]).copy()

    fig.add_trace(
        go.Scattergl(
            x=cells["x_plot"],
            y=cells["y_plot"],
            mode="markers",
            name="tissue",
            marker={"size": 3.2, "color": "#C9CED4", "opacity": 0.18, "line": {"width": 0}},
            customdata=cells[["cell_id", "cell_type"]].astype(str).to_numpy(),
            hovertemplate="cell=%{customdata[0]}<br>type=%{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )

    sender_type = ""
    receiver_type = ""
    if selected_pair and " -> " in selected_pair:
        sender_type, receiver_type = selected_pair.split(" -> ", 1)
    sender_type = sender_type.strip()
    receiver_type = receiver_type.strip()
    if sender_type:
        send_cells = cells[cells["cell_type"].astype(str).eq(sender_type)]
        if not send_cells.empty:
            fig.add_trace(
                go.Scattergl(
                    x=send_cells["x_plot"],
                    y=send_cells["y_plot"],
                    mode="markers",
                    name=f"{sender_type} cells",
                    marker={"size": 4.0, "color": "#9BCD93", "opacity": 0.30, "line": {"width": 0}},
                    hovertemplate=f"{sender_type}<extra></extra>",
                )
            )
    if receiver_type:
        recv_cells = cells[cells["cell_type"].astype(str).eq(receiver_type)]
        if not recv_cells.empty:
            same_type = receiver_type == sender_type
            if not same_type:
                fig.add_trace(
                    go.Scattergl(
                        x=recv_cells["x_plot"],
                        y=recv_cells["y_plot"],
                        mode="markers",
                        name=f"{receiver_type} cells",
                        marker={"size": 4.0, "color": "#9AAED8", "opacity": 0.26, "line": {"width": 0}},
                        hovertemplate=f"{receiver_type}<extra></extra>",
                    )
                )

    activity_values = pd.to_numeric(expr_score.get("spatialself_scaled", 0.0), errors="coerce").fillna(0.0)
    activity_mask = activity_values > activity_values.quantile(0.82)
    if activity_mask.any():
        fig.add_trace(
            go.Scattergl(
                x=expr_score.loc[activity_mask, "x_plot"],
                y=expr_score.loc[activity_mask, "y_plot"],
                mode="markers",
                name="receiver activity",
                marker={
                    "size": 7.6,
                    "color": activity_values.loc[activity_mask],
                    "colorscale": [[0, "#F7F4EC"], [0.50, "#B9D7E7"], [1, "#2F78A8"]],
                    "opacity": 0.46,
                    "line": {"width": 0},
                    "showscale": False,
                },
                hovertemplate="receiver activity=%{marker.color:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    if overlay == "Ligand":
        score_col = "Cxcl12"
        color = "#3F8E50"
        color_scale = [[0, "#F2F6F1"], [0.55, "#A8D29D"], [1, color]]
        data = cells.copy()
        title = "Cxcl12 ligand expression"
    elif overlay == "Receptor":
        score_col = "Cxcr4"
        color = "#243B73"
        color_scale = [[0, "#F1F4FA"], [0.55, "#8EA6D3"], [1, color]]
        data = cells.copy()
        title = "Cxcr4 receptor expression"
    elif overlay == "Response":
        score_col = "response_scaled"
        color = "#B85450"
        color_scale = [[0, "#F7F4EC"], [0.55, "#E6C06A"], [1, color]]
        data = response.copy()
        title = "receiver downstream response"
    else:
        score_col = ""
        data = pd.DataFrame()
        title = ""

    if not data.empty:
        values = pd.to_numeric(data.get(score_col, 0.0), errors="coerce").fillna(0.0)
        mask = values > values.quantile(0.72)
    else:
        values = pd.Series(dtype=float)
        mask = pd.Series(dtype=bool)
    if not data.empty and mask.any():
        fig.add_trace(
            go.Scattergl(
                x=data.loc[mask, "x_plot"],
                y=data.loc[mask, "y_plot"],
                mode="markers",
                name=title,
                marker={
                    "size": 5.7,
                    "color": values.loc[mask],
                    "colorscale": color_scale,
                    "cmin": float(values.quantile(0.72)),
                    "cmax": float(max(values.quantile(0.995), values.max())),
                    "opacity": 0.90,
                    "line": {"width": 0.25, "color": "white"},
                    "showscale": False,
                },
                customdata=data.loc[mask, ["cell_id", "cell_type"]].astype(str).to_numpy() if {"cell_id", "cell_type"}.issubset(data.columns) else None,
                hovertemplate="cell=%{customdata[0]}<br>type=%{customdata[1]}<br>value=%{marker.color:.3f}<extra></extra>",
            )
        )

    pair_edges = interactions.copy()
    if show_pair_edges:
        if selected_pair and "sender_cell_type" in pair_edges and "receiver_cell_type" in pair_edges:
            pair_edges["pair"] = pair_edges["sender_cell_type"].astype(str) + " -> " + pair_edges["receiver_cell_type"].astype(str)
            pair_edges = pair_edges[pair_edges["pair"].eq(selected_pair)].copy()
        pair_edges = pair_edges.sort_values("score", ascending=False).copy()
    else:
        pair_edges = pair_edges.iloc[0:0].copy()
    if not pair_edges.empty:
        top_n_pair = min(len(pair_edges), max(20, int(np.ceil(len(pair_edges) * 0.20))))
        pair_edges = pair_edges.head(top_n_pair).copy()
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for r in pair_edges.itertuples(index=False):
            line_x.extend([float(r.sender_x_plot), float(r.receiver_x_plot), None])
            line_y.extend([float(r.sender_y_plot), float(r.receiver_y_plot), None])
        fig.add_trace(
            go.Scattergl(
                x=line_x,
                y=line_y,
                mode="lines",
                name="top pair links",
                line={"color": "rgba(30,38,44,0.52)", "width": 1.25},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=pair_edges["sender_x_plot"],
                y=pair_edges["sender_y_plot"],
                mode="markers",
                name="sender events",
                marker={"size": 5.9, "color": "#3F8E50", "opacity": 0.76, "line": {"width": 0.35, "color": "white"}},
                customdata=pair_edges[["sender_id", "sender_cell_type", "receiver_id", "receiver_cell_type", "score", "distance"]].astype(object).to_numpy(),
                hovertemplate="sender=%{customdata[0]} (%{customdata[1]})<br>receiver=%{customdata[2]} (%{customdata[3]})<br>score=%{customdata[4]:.4f}<br>distance=%{customdata[5]:.1f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=pair_edges["receiver_x_plot"],
                y=pair_edges["receiver_y_plot"],
                mode="markers",
                name="receiver events",
                marker={"size": 5.9, "color": "#243B73", "opacity": 0.78, "line": {"width": 0.35, "color": "white"}},
                customdata=pair_edges[["receiver_id", "receiver_cell_type", "sender_id", "sender_cell_type", "score", "distance"]].astype(object).to_numpy(),
                hovertemplate="receiver=%{customdata[0]} (%{customdata[1]})<br>sender=%{customdata[2]} (%{customdata[3]})<br>score=%{customdata[4]:.4f}<br>distance=%{customdata[5]:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 0.01,
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "#E2E6EA",
            "borderwidth": 1,
            "font": {"size": 10, "color": "#263137"},
        },
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        hovermode="closest",
        dragmode="pan",
        uirevision=f"slideseq-v2-{selected_pair}-{overlay}",
    )
    return fig


def make_slideseq_v2_layer_inset(interactions: pd.DataFrame, selected_pair: str) -> go.Figure:
    dat = interactions.copy()
    dat["pair"] = dat["sender_cell_type"].astype(str) + " -> " + dat["receiver_cell_type"].astype(str)
    dat = dat[dat["pair"].eq(selected_pair)].sort_values("score", ascending=False).head(45)
    if dat.empty:
        return empty_figure("No interaction for selected pair")
    xmin = float(np.nanmin([dat["sender_x_plot"].min(), dat["receiver_x_plot"].min()]))
    xmax = float(np.nanmax([dat["sender_x_plot"].max(), dat["receiver_x_plot"].max()]))
    ymin = float(np.nanmin([dat["sender_y_plot"].min(), dat["receiver_y_plot"].min()]))
    ymax = float(np.nanmax([dat["sender_y_plot"].max(), dat["receiver_y_plot"].max()]))
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    sx = (dat["sender_x_plot"].to_numpy(float) - xmin) / xspan
    sy = (dat["sender_y_plot"].to_numpy(float) - ymin) / yspan
    rx = (dat["receiver_x_plot"].to_numpy(float) - xmin) / xspan
    ry = (dat["receiver_y_plot"].to_numpy(float) - ymin) / yspan

    fig = go.Figure()
    for z, color in [(1.0, "#EAF4E6"), (0.0, "#E8EEF8")]:
        fig.add_trace(
            go.Surface(
                x=[[0, 1], [0, 1]],
                y=[[0, 0], [1, 1]],
                z=[[z, z], [z, z]],
                surfacecolor=[[0, 0], [0, 0]],
                colorscale=[[0, color], [1, color]],
                opacity=0.12,
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    lx: list[float | None] = []
    ly: list[float | None] = []
    lz: list[float | None] = []
    for x0, y0, x1, y1 in zip(sx, sy, rx, ry, strict=False):
        lx.extend([float(x0), float(x1), None])
        ly.extend([float(y0), float(y1), None])
        lz.extend([1.0, 0.0, None])
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line={"color": "#4F9B59", "width": 1.2}, opacity=0.22, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=np.ones(len(dat)), mode="markers", marker={"size": 4.8, "color": "#3F8E50", "opacity": 0.86, "line": {"width": 0.3, "color": "white"}}, hoverinfo="skip", name="sender"))
    fig.add_trace(go.Scatter3d(x=rx, y=ry, z=np.zeros(len(dat)), mode="markers", marker={"size": 4.8, "color": "#243B73", "opacity": 0.88, "line": {"width": 0.3, "color": "white"}}, hoverinfo="skip", name="receiver"))
    fig.update_layout(
        height=240,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="white",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False, "range": [-0.04, 1.06]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.46},
            "camera": {"eye": {"x": 1.30, "y": -1.55, "z": 0.68}, "center": {"x": 0.0, "y": 0.0, "z": -0.06}},
        },
        showlegend=False,
        uirevision=f"inset-{selected_pair}",
    )
    return fig


def make_scale_profile(row: pd.Series, lr: str) -> go.Figure:
    scale_cols = [f"scale_{i}" for i in range(6)]
    weights = pd.to_numeric(row.reindex(scale_cols), errors="coerce").fillna(0.0).to_numpy(float)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    labels = [f"s{i + 1}" for i in range(6)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=weights,
            marker={"color": "#F3D78A", "line": {"color": "rgba(38,49,55,0.55)", "width": 0.7}},
            hovertemplate="scale=%{x}<br>weight=%{y:.3f}<extra></extra>",
            name="scale weight",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=weights,
            mode="lines+markers",
            line={"color": LR_COLORS.get(lr, "#4F9B59"), "width": 2.2},
            marker={"size": 7},
            hovertemplate="scale=%{x}<br>weight=%{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=168,
        margin={"l": 34, "r": 6, "t": 4, "b": 24},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"title": None, "tickfont": {"size": 10, "color": "#68747d"}, "showline": True, "linecolor": "#d8dde2"},
        yaxis={
            "title": "scale weight",
            "range": [0, max(0.8, float(np.nanmax(weights)) + 0.08)],
            "gridcolor": "#eef1f3",
            "zeroline": False,
            "showline": True,
            "linecolor": "#d8dde2",
            "titlefont": {"size": 10},
            "tickfont": {"size": 9, "color": "#68747d"},
        },
    )
    return fig


def make_response_scatter(hotspots: pd.DataFrame, lr: str) -> go.Figure:
    hot = hotspots.copy()
    if hot.empty:
        return empty_figure("No receiver hotspot table for this LR")
    hot["score_percentile"] = hot["receiver_activity"].rank(method="average", pct=True)
    rng = np.random.default_rng(17)
    if len(hot) > 2500:
        hot = hot.iloc[rng.choice(np.arange(len(hot)), size=2500, replace=False)].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=hot["score_percentile"],
            y=hot["target_score"],
            mode="markers",
            marker={
                "size": np.where(hot["is_top_receiver"], 8, 5),
                "color": np.where(hot["is_top_receiver"], LR_COLORS.get(lr, "#4F9B59"), "#F4D5DD"),
                "opacity": np.where(hot["is_top_receiver"], 0.95, 0.55),
                "line": {"color": LR_COLORS.get(lr, "#C87370"), "width": 0.5},
            },
            customdata=hot[["cell_id", "cell_type", "niche", "receiver_activity", "target_score", "is_top_receiver"]].to_numpy(),
            hovertemplate=(
                "cell=%{customdata[0]}<br>"
                "type=%{customdata[1]}<br>"
                "niche=%{customdata[2]}<br>"
                "score percentile=%{x:.3f}<br>"
                "receiver score=%{customdata[3]:.4f}<br>"
                "target signature=%{customdata[4]:.4f}<br>"
                "top hotspot=%{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=172,
        margin={"l": 38, "r": 6, "t": 4, "b": 24},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={
            "title": "receiver score percentile",
            "gridcolor": "#eef1f3",
            "zeroline": False,
            "showline": True,
            "linecolor": "#d8dde2",
            "titlefont": {"size": 10},
            "tickfont": {"size": 9, "color": "#68747d"},
        },
        yaxis={
            "title": "target signature",
            "gridcolor": "#eef1f3",
            "zeroline": False,
            "showline": True,
            "linecolor": "#d8dde2",
            "titlefont": {"size": 10},
            "tickfont": {"size": 9, "color": "#68747d"},
        },
    )
    return fig


def make_edge_map(cells: pd.DataFrame, scored: pd.DataFrame, lr: str, top_n: int) -> go.Figure:
    dat = scored.sort_values("score", ascending=False).head(top_n).copy()
    if dat.empty:
        return empty_figure("No candidate edges for this LR")
    pos = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].copy()
    sender = pos.rename(
        columns={
            "cell_id": "sender_id",
            "x_plot": "sender_x",
            "y_plot": "sender_y",
            "cell_type": "sender_type",
            "niche": "sender_niche_meta",
        }
    )
    receiver = pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x_plot": "receiver_x",
            "y_plot": "receiver_y",
            "cell_type": "receiver_type",
            "niche": "receiver_niche_meta",
        }
    )
    dat = dat.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")
    color = LR_COLORS.get(lr, "#4F9B59")
    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    for row in dat.itertuples():
        x_lines.extend([row.sender_x, row.receiver_x, None])
        y_lines.extend([row.sender_y, row.receiver_y, None])
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=cells["x_plot"],
            y=cells["y_plot"],
            mode="markers",
            marker={"size": 3.6, "color": "#DDE2E9", "opacity": 0.32},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line={"color": color, "width": 0.85},
            opacity=0.28,
            hoverinfo="skip",
            name=f"top {top_n} edges",
        )
    )
    endpoints = pd.concat(
        [
            dat[["sender_id", "sender_x", "sender_y", "sender_type", "sender_niche_meta", "score", "distance"]].rename(
                columns={"sender_id": "cell_id", "sender_x": "x", "sender_y": "y", "sender_type": "cell_type", "sender_niche_meta": "niche"}
            ),
            dat[["receiver_id", "receiver_x", "receiver_y", "receiver_type", "receiver_niche_meta", "score", "distance"]].rename(
                columns={"receiver_id": "cell_id", "receiver_x": "x", "receiver_y": "y", "receiver_type": "cell_type", "receiver_niche_meta": "niche"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("cell_id")
    fig.add_trace(
        go.Scattergl(
            x=endpoints["x"],
            y=endpoints["y"],
            mode="markers",
            marker={"size": 6.6, "color": color, "opacity": 0.88, "line": {"color": "white", "width": 0.4}},
            customdata=endpoints[["cell_id", "cell_type", "niche"]].astype(str).to_numpy(),
            hovertemplate="cell=%{customdata[0]}<br>type=%{customdata[1]}<br>niche=%{customdata[2]}<extra></extra>",
            name="edge endpoint",
        )
    )
    fig.update_layout(
        height=530,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        title={"text": f"{lr} candidate communication edges", "font": {"size": 15}},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        legend={"orientation": "h", "y": 0.01, "x": 0.01, "bgcolor": "rgba(255,255,255,0.75)"},
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=300, xaxis={"visible": False}, yaxis={"visible": False}, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def metric_grid(row: pd.Series, scored: pd.DataFrame) -> None:
    top_distance = format_float(row.get("top_median_distance"), digits=1)
    edge_count = f"{len(scored):,}" if scored is not None else "0"
    values = [
        ("Target r", format_float(row.get("target_spearman_r"))),
        ("Target lift", format_float(row.get("top_receiver_target_lift"))),
        ("Median dist.", top_distance),
        ("Candidates", edge_count),
    ]
    tiles = []
    for label, value in values:
        tiles.append(
            "<div class='metric-tile'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            "</div>"
        )
    st.markdown("<div class='metric-grid'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)


def compact_edge_table(scored: pd.DataFrame) -> pd.DataFrame:
    table = selected_edge_table(scored).copy().head(4)
    keep = [c for c in ["sender_cell_type", "receiver_cell_type", "score", "distance"] if c in table.columns]
    table = table[keep]
    table = table.rename(columns={"sender_cell_type": "sender", "receiver_cell_type": "receiver"})
    for col in ["score", "distance"]:
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(3 if col == "score" else 1)
    for col in table.select_dtypes(include="object").columns:
        table[col] = table[col].astype(str).str.replace("_proxy", "", regex=False).str.replace("_", " ", regex=False)
    return table


def edge_summary_markup(scored: pd.DataFrame) -> str:
    table = compact_edge_table(scored)
    if table.empty:
        return "<div class='small-note'>No predicted edges for this LR.</div>"
    rows = []
    for row in table.itertuples(index=False):
        sender = getattr(row, "sender", "NA")
        receiver = getattr(row, "receiver", "NA")
        score = getattr(row, "score", "NA")
        distance = getattr(row, "distance", "NA")
        rows.append(
            "<div class='edge-row'>"
            f"<span>{sender} -> {receiver}</span>"
            f"<b>{score}</b>"
            f"<em>{distance}</em>"
            "</div>"
        )
    return "<div class='edge-list'>" + "".join(rows) + "</div>"


def selected_summary_table(ranking: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "LR",
        "mechanism",
        "biological_category",
        "scale_class",
        "active_niche",
        "mean_score",
        "q95_score",
        "target_spearman_r",
        "top_receiver_target_lift",
        "top_median_distance",
        "top_niche_match_rate",
    ]
    cols = [c for c in cols if c in ranking.columns]
    return ranking[cols].copy()


def selected_edge_table(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sender_id",
        "receiver_id",
        "ligand",
        "receptor",
        "score",
        "distance",
        "sender_cell_type",
        "receiver_cell_type",
        "sender_niche",
        "receiver_niche",
    ]
    cols = [c for c in cols if c in scored.columns]
    if scored.empty or not cols:
        return pd.DataFrame(columns=cols)
    return scored.sort_values("score", ascending=False)[cols].head(300).copy()


def lr_status_markup(row: pd.Series) -> str:
    items = [
        ("Mechanism", row.get("mechanism", "NA")),
        ("Scale", row.get("scale_class", "NA")),
        ("Active niche", row.get("active_niche", "NA")),
        ("Receiver niche", row.get("top_receiver_niche_mode", "NA")),
    ]
    html = []
    for key, value in items:
        clean = str(value).replace("_", " ")
        html.append(f"<span class='status-pill'><b>{key}</b>: {clean}</span>")
    return "".join(html)


def main() -> None:
    page_style()
    dataset_key = st.sidebar.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        index=0,
        format_func=lambda key: DATASETS[key].label,
    )
    tables = load_tables(dataset_key)
    cfg = DATASETS[dataset_key]

    ranking = tables["ranking"]
    sort_default = "Target correlation"
    ranked = sort_ranking(ranking, sort_default, "")
    lr_default = ranked["LR"].iloc[0]

    st.markdown(
        "<div class='topbar'>"
        "<div>"
        "<div class='topbar-title'>ScaleComm interactive communication browser</div>"
        "<div class='topbar-sub'>Spatial communication map, cell-type pair summary, and receiver response evidence.</div>"
        "</div>"
        "<div class='topbar-badges'>"
        f"<span class='topbar-badge'>{cfg.label}</span>"
        "<span class='topbar-badge'>ScaleComm only</span>"
        "<span class='topbar-badge'>zoom / hover / export</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if cfg.kind == "slideseq":
        main_col, side_col = st.columns([0.80, 0.20], gap="large")
    else:
        main_col, side_col = st.columns([0.70, 0.30], gap="large")

    with side_col:
        st.markdown("<div class='control-card'>", unsafe_allow_html=True)
        st.markdown("<div class='control-label'>Ligand-receptor axis</div>", unsafe_allow_html=True)
        sort_by = st.selectbox("Sort by", ["Target correlation", "Target lift", "Mean score", "Broad scale weight", "Max score"], index=0, label_visibility="collapsed")
        query = st.text_input("Search LR", placeholder="Search LR, e.g. CXCL12", label_visibility="collapsed")
        ranked = sort_ranking(ranking, sort_by, query)
        if ranked.empty:
            st.warning("No LR pair matches the current search.")
            return
        lr_index = int(ranked.index[ranked["LR"].eq(lr_default)][0]) if lr_default in ranked["LR"].values else 0
        lr = st.selectbox("LR pair", ranked["LR"].tolist(), index=lr_index, label_visibility="collapsed")
        st.markdown(f"<div class='lr-name' style='color:{LR_COLORS.get(lr, '#263137')}'>{lr}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        row = lr_row(ranking, lr)
        hot = selected_hotspots(tables["hotspots"], lr)
        scored = selected_scored(tables["scored"], lr)

        st.markdown("<div class='control-card'>", unsafe_allow_html=True)
        st.markdown("<div class='control-label'>Cell-type pair</div>", unsafe_allow_html=True)
        if cfg.kind == "slideseq":
            top_n_edges = len(tables.get("interactions", scored))
            color_by = "Target signature"
            show_edges = True
            show_hotspots = True
            show_cell_types = True
            selected_cell_types = []
            filter_edges_by_cell_type = False
            overlay = "Communication"
            pair_summary = slideseq_pair_summary(tables["interactions"])
            annotated_pair_summary = pair_summary[
                ~pair_summary["sender_cell_type"].astype(str).eq("unannotated")
                & ~pair_summary["receiver_cell_type"].astype(str).eq("unannotated")
            ].copy()
            pair_options = (annotated_pair_summary if not annotated_pair_summary.empty else pair_summary).head(12)
            pair_counts = {
                row_pair["pair"]: (
                    int(row_pair["n_edges"]),
                    min(int(row_pair["n_edges"]), max(20, int(np.ceil(float(row_pair["n_edges"]) * 0.20)))),
                )
                for _, row_pair in pair_options.iterrows()
            }
            selected_pair = st.selectbox(
                "Cell-type communication",
                pair_options["pair"].tolist(),
                index=0,
                format_func=lambda pair: f"{pair}  n={pair_counts[pair][0]}, shown={pair_counts[pair][1]}",
            )
            st.markdown(f"<div class='small-note'>Annotated events: <b>{top_n_edges:,}</b></div>", unsafe_allow_html=True)
        else:
            selected_cell_types = []
            filter_edges_by_cell_type = False
            top_n_edges = st.slider("Communication lines", min_value=25, max_value=500, value=125, step=25)
            color_by = st.radio("Receiver overlay", ["Target signature", "Receiver activity"], horizontal=True)
            t1, t2, t3 = st.columns(3)
            show_edges = t1.toggle("lines", value=True)
            show_hotspots = t2.toggle("hotspots", value=True)
            show_cell_types = t3.toggle("types", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if cfg.kind == "slideseq":
            pass
        else:
            st.markdown("<div class='section-head'>Selected LR evidence</div>", unsafe_allow_html=True)
            metric_grid(row, scored)
            st.markdown(lr_status_markup(row), unsafe_allow_html=True)

            st.markdown("<div class='section-head'>Learned scale profile</div>", unsafe_allow_html=True)
            st.plotly_chart(make_scale_profile(row, lr), use_container_width=True, config={"displayModeBar": False})
            st.markdown("<div class='section-head'>Receiver response support</div>", unsafe_allow_html=True)
            st.plotly_chart(make_response_scatter(hot, lr), use_container_width=True, config={"displayModeBar": False})
            st.markdown("<div class='section-head'>Top predicted edges</div>", unsafe_allow_html=True)
            st.markdown(edge_summary_markup(scored), unsafe_allow_html=True)

    with main_col:
        if cfg.kind == "slideseq":
            st.markdown(
                f"<div class='control-card' style='margin-bottom:0.50rem;'>"
                f"<div class='control-label'>Communication map</div>"
                f"<div style='font-size:1.02rem;font-weight:700;color:#20272c;line-height:1.15;'>{lr}: {selected_pair}</div>"
                f"<div class='small-note'>Map shows top 20% events, with at least 20 events when available; green dots are senders and blue dots are receivers.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            main_fig = make_slideseq_v2_spatial_map(
                tables["cells"],
                tables["expr_score"],
                tables["response"],
                tables["interactions"],
                selected_pair,
                overlay=overlay,
            )
            st.plotly_chart(
                main_fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": f"spatialself_{lr.replace('-', '_')}_{selected_pair.replace(' -> ', '_to_')}_map"},
                },
            )
        else:
            main_fig = make_communication_view(
                tables["cells"],
                hot,
                scored,
                lr,
                top_n=top_n_edges,
                color_by=color_by,
                show_cell_types=show_cell_types,
                show_hotspots=show_hotspots,
                show_edges=show_edges,
            )
            st.plotly_chart(
                main_fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": f"spatialself_{lr.replace('-', '_')}_communication_map"},
                },
            )

    with st.expander("Source tables and downloads", expanded=False):
        tab_edges, tab_tables, tab_downloads = st.tabs(["Top edges", "LR table", "Downloads"])
        with tab_edges:
            st.dataframe(selected_edge_table(scored), use_container_width=True, height=300)
        with tab_tables:
            st.dataframe(selected_summary_table(ranked), use_container_width=True, height=440)
        with tab_downloads:
            st.download_button(
                "Download current LR ranking",
                data=selected_summary_table(ranked).to_csv(index=False).encode("utf-8"),
                file_name=f"{dataset_key}_spatialself_lr_ranking.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download selected LR candidate edges",
                data=scored.sort_values("score", ascending=False).to_csv(index=False).encode("utf-8"),
                file_name=f"{dataset_key}_{lr.replace('-', '_')}_candidate_edges.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download selected LR receiver hotspots",
                data=hot.to_csv(index=False).encode("utf-8"),
                file_name=f"{dataset_key}_{lr.replace('-', '_')}_receiver_hotspots.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
