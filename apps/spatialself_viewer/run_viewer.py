#!/usr/bin/env python
"""Interactive viewer for the standardized outputs of one completed ScaleComm run.

Launch through ``launch_run_viewer.py`` or directly with Streamlit:

    streamlit run apps/spatialself_viewer/run_viewer.py -- \
        --prepared-dir path/to/prepared --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PALETTE = ("#3976A8", "#BD8F3D", "#5D9885", "#A66B82", "#7D6AAE", "#71808B", "#C7675A", "#719F5D")
INK = "#20272C"
MUTED = "#66727A"
LINE = "#E3E8EC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--scale-profile", type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _one_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(path.name for path in matches)
        raise FileNotFoundError(f"Expected one {label} in {directory}; found {detail}.")
    return matches[0]


def _require_columns(table: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {', '.join(missing)}")


def _color_for(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).digest()[0]
    return PALETTE[digest % len(PALETTE)]


def _label_series(table: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in table:
        return table[preferred].fillna("unannotated").astype(str)
    return pd.Series(fallback, index=table.index, dtype="object")


@st.cache_data(show_spinner=False)
def load_run(
    prepared_dir_value: str,
    results_dir_value: str,
    prediction_value: str | None,
    scale_value: str | None,
) -> dict[str, pd.DataFrame]:
    prepared_dir = Path(prepared_dir_value)
    results_dir = Path(results_dir_value)
    cells_path = prepared_dir / "cells.csv"
    candidates_path = prepared_dir / "candidate_interactions.csv"
    lr_pairs_path = prepared_dir / "lr_pairs.csv"
    if not cells_path.exists() or not candidates_path.exists():
        raise FileNotFoundError("Prepared directory must contain cells.csv and candidate_interactions.csv.")

    prediction_path = Path(prediction_value) if prediction_value else _one_file(results_dir, "*_predictions.csv", "prediction CSV")
    scale_path = Path(scale_value) if scale_value else _one_file(results_dir, "*_lr_scale_distribution.csv", "scale-profile CSV")
    if not prediction_path.exists() or not scale_path.exists():
        raise FileNotFoundError("The selected prediction or scale-profile CSV does not exist.")

    cells = pd.read_csv(cells_path)
    candidates = pd.read_csv(candidates_path)
    prediction = pd.read_csv(prediction_path)
    scale = pd.read_csv(scale_path)
    _require_columns(cells, {"cell_id", "x", "y"}, cells_path)
    _require_columns(candidates, {"candidate_id", "sender_id", "receiver_id", "ligand", "receptor"}, candidates_path)
    _require_columns(prediction, {"candidate_id", "score"}, prediction_path)
    _require_columns(scale, {"ligand", "receptor"}, scale_path)

    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["x_plot"] = pd.to_numeric(cells["x"], errors="coerce")
    cells["y_plot"] = -pd.to_numeric(cells["y"], errors="coerce")
    cells["cell_type_display"] = _label_series(cells, "cell_type", "unannotated")
    cells["niche_display"] = _label_series(cells, "niche", "unknown")
    cells = cells.dropna(subset=["x_plot", "y_plot"]).drop_duplicates("cell_id")

    candidates = candidates.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidates["sender_id"] = candidates["sender_id"].astype(str)
    candidates["receiver_id"] = candidates["receiver_id"].astype(str)
    candidates["LR"] = candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)
    candidates = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="left", validate="one_to_one")
    candidates["score"] = pd.to_numeric(candidates["score"], errors="coerce")
    scored = candidates.dropna(subset=["score"]).copy()

    positions = cells[["cell_id", "x_plot", "y_plot", "cell_type_display", "niche_display"]]
    sender = positions.rename(columns={
        "cell_id": "sender_id", "x_plot": "sender_x", "y_plot": "sender_y",
        "cell_type_display": "sender_cell_type_plot", "niche_display": "sender_niche_plot",
    })
    receiver = positions.rename(columns={
        "cell_id": "receiver_id", "x_plot": "receiver_x", "y_plot": "receiver_y",
        "cell_type_display": "receiver_cell_type_plot", "niche_display": "receiver_niche_plot",
    })
    scored = scored.merge(sender, on="sender_id", how="inner").merge(receiver, on="receiver_id", how="inner")
    for name in ("sender_cell_type", "receiver_cell_type", "sender_niche", "receiver_niche", "mechanism", "distance"):
        if name not in scored:
            scored[name] = np.nan

    scale = scale.copy()
    scale["LR"] = scale["ligand"].astype(str) + "-" + scale["receptor"].astype(str)
    component_columns = [column for column in scale.columns if column.startswith("scale_") and column[6:].isdigit()]
    component_columns.sort(key=lambda column: int(column.split("_")[1]))
    if not component_columns:
        raise ValueError(f"{scale_path.name} does not contain scale_0 ... scale_N columns.")
    scale[component_columns] = scale[component_columns].apply(pd.to_numeric, errors="coerce")

    lr_metadata = pd.read_csv(lr_pairs_path) if lr_pairs_path.exists() else pd.DataFrame()
    return {"cells": cells, "scored": scored, "scale": scale, "lr_metadata": lr_metadata}


def page_style() -> None:
    st.set_page_config(page_title="ScaleComm Run Viewer", page_icon="SC", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
        :root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; }}
        .stApp {{ background:#ffffff; }}
        #MainMenu, footer, header {{ visibility:hidden; }}
        .block-container {{ max-width:1800px; padding-top:0.8rem; padding-bottom:0.7rem; }}
        h1 {{ font-size:1.2rem !important; margin-bottom:0.15rem !important; }}
        .run-subtitle {{ color:var(--muted); font-size:0.76rem; margin-bottom:0.55rem; }}
        div[data-testid="stMetric"] {{ border:1px solid var(--line); border-radius:0; padding:0.35rem 0.5rem; }}
        div[data-testid="stMetricLabel"] {{ color:var(--muted); }}
        div[data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
        .stPlotlyChart {{ border:1px solid var(--line); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def draw_map(cells: pd.DataFrame, edges: pd.DataFrame, top_n: int) -> go.Figure:
    top_edges = edges.nlargest(top_n, "score")
    figure = go.Figure()
    for cell_type, group in cells.groupby("cell_type_display", sort=True):
        figure.add_trace(go.Scattergl(
            x=group["x_plot"], y=group["y_plot"], mode="markers", name=cell_type,
            marker={"size": 4, "color": _color_for(cell_type), "opacity": 0.58},
            hovertemplate=f"{cell_type}<extra></extra>",
        ))
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for row in top_edges.itertuples(index=False):
        line_x.extend([row.sender_x, row.receiver_x, None])
        line_y.extend([row.sender_y, row.receiver_y, None])
    figure.add_trace(go.Scattergl(
        x=line_x, y=line_y, mode="lines", name="Top candidate events",
        line={"color": "rgba(32,39,44,0.28)", "width": 1}, hoverinfo="skip",
    ))
    receiver_scores = top_edges.groupby("receiver_id", as_index=False)["score"].mean().merge(
        cells[["cell_id", "x_plot", "y_plot"]], left_on="receiver_id", right_on="cell_id", how="inner"
    )
    figure.add_trace(go.Scattergl(
        x=receiver_scores["x_plot"], y=receiver_scores["y_plot"], mode="markers", name="Top receivers",
        marker={"size": 7, "color": receiver_scores["score"], "colorscale": "Blues", "showscale": True,
                "colorbar": {"title": "Mean ranking score", "thickness": 12}},
        customdata=np.stack([receiver_scores["receiver_id"], receiver_scores["score"]], axis=-1),
        hovertemplate="Receiver %{customdata[0]}<br>Mean score %{customdata[1]:.3g}<extra></extra>",
    ))
    figure.update_layout(
        height=650, margin={"l": 14, "r": 18, "t": 38, "b": 14},
        title={"text": f"Top {len(top_edges):,} candidate events", "font": {"size": 15, "color": INK}},
        legend={"orientation": "h", "y": -0.10, "font": {"size": 10}},
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis={"visible": False, "scaleanchor": "y"}, yaxis={"visible": False},
    )
    return figure


def draw_profile(scale_row: pd.Series, components: list[str]) -> go.Figure:
    values = [float(scale_row[column]) for column in components]
    figure = go.Figure(go.Bar(
        x=[f"s{index}" for index in range(len(components))], y=values,
        marker_color="#2E6EA5", hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    figure.update_layout(
        height=280, margin={"l": 44, "r": 14, "t": 36, "b": 36},
        title={"text": "LR-specific scale profile", "font": {"size": 15, "color": INK}},
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", showlegend=False,
        yaxis={"title": "Weight", "range": [0, max(0.05, max(values) * 1.15)], "gridcolor": "#E7EBEE"},
        xaxis={"title": "Diffusion-derived component"},
    )
    return figure


def main() -> None:
    page_style()
    args = parse_args()
    try:
        tables = load_run(str(args.prepared_dir.resolve()), str(args.results_dir.resolve()), str(args.prediction) if args.prediction else None, str(args.scale_profile) if args.scale_profile else None)
    except Exception as error:
        st.error(str(error))
        st.stop()

    cells, scored, scale = tables["cells"], tables["scored"], tables["scale"]
    all_lrs = sorted(set(scored["LR"]).intersection(scale["LR"]))
    if not all_lrs:
        st.error("No LR pairs are shared between scored candidates and scale profiles.")
        st.stop()
    st.title("ScaleComm Run Viewer")
    st.markdown(
        f'<div class="run-subtitle">Prepared input: <code>{args.prepared_dir.name}</code> &nbsp; | &nbsp; Results: <code>{args.results_dir.name}</code></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Display")
        selected_lr = st.selectbox("Ligand-receptor pair", all_lrs)
        fraction = st.slider("Displayed top fraction (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        sender_types = sorted(scored.loc[scored["LR"].eq(selected_lr), "sender_cell_type"].dropna().astype(str).unique())
        receiver_types = sorted(scored.loc[scored["LR"].eq(selected_lr), "receiver_cell_type"].dropna().astype(str).unique())
        sender_filter = st.selectbox("Sender cell type", ["All"] + sender_types)
        receiver_filter = st.selectbox("Receiver cell type", ["All"] + receiver_types)
        st.caption("Scores are within-dataset ranking quantities, not calibrated probabilities of communication.")

    selected = scored.loc[scored["LR"].eq(selected_lr)].copy()
    if sender_filter != "All":
        selected = selected.loc[selected["sender_cell_type"].astype(str).eq(sender_filter)]
    if receiver_filter != "All":
        selected = selected.loc[selected["receiver_cell_type"].astype(str).eq(receiver_filter)]
    if selected.empty:
        st.warning("No candidate events remain after the selected cell-type filters.")
        st.stop()
    top_n = max(1, int(np.ceil(len(selected) * fraction / 100.0)))

    scale_row = scale.loc[scale["LR"].eq(selected_lr)].iloc[0]
    component_columns = sorted([column for column in scale.columns if column.startswith("scale_") and column[6:].isdigit()], key=lambda column: int(column.split("_")[1]))
    top = selected.nlargest(top_n, "score")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Cells", f"{len(cells):,}")
    metric_columns[1].metric("Candidate events", f"{len(selected):,}")
    metric_columns[2].metric("Displayed events", f"{len(top):,}")
    metric_columns[3].metric("Top-event score", f"{top['score'].iloc[0]:.3g}")
    distance = pd.to_numeric(top["distance"], errors="coerce").dropna()
    metric_columns[4].metric("Top-event median distance", "NA" if distance.empty else f"{distance.median():.2f}")

    left, right = st.columns([2.25, 1.0], gap="large")
    with left:
        st.plotly_chart(draw_map(cells, selected, top_n), use_container_width=True, config={"toImageButtonOptions": {"format": "png", "filename": f"scalecomm_{selected_lr}"}})
    with right:
        st.plotly_chart(draw_profile(scale_row, component_columns), use_container_width=True, config={"displayModeBar": False})
        st.subheader("Selected LR summary")
        summary = pd.DataFrame({
            "Field": ["Ligand", "Receptor", "Mechanism", "Expected scale", "Local weight", "Broad weight"],
            "Value": [
                str(scale_row.get("ligand", "")), str(scale_row.get("receptor", "")), str(scale_row.get("mechanism", "not supplied")),
                f"{float(scale_row.get('expected_scale', np.nan)):.3f}" if pd.notna(scale_row.get("expected_scale", np.nan)) else "not supplied",
                f"{float(scale_row.get('local_weight', np.nan)):.3f}" if pd.notna(scale_row.get("local_weight", np.nan)) else "not supplied",
                f"{float(scale_row.get('broad_weight', np.nan)):.3f}" if pd.notna(scale_row.get("broad_weight", np.nan)) else "not supplied",
            ],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    display_columns = [column for column in ["candidate_id", "sender_id", "receiver_id", "ligand", "receptor", "sender_cell_type", "receiver_cell_type", "distance", "score"] if column in top]
    st.subheader("Displayed candidate events")
    st.dataframe(top[display_columns].sort_values("score", ascending=False), hide_index=True, use_container_width=True, height=330)
    st.download_button(
        "Download displayed events as CSV",
        data=top[display_columns].sort_values("score", ascending=False).to_csv(index=False).encode("utf-8"),
        file_name=f"scalecomm_{selected_lr.replace('-', '_')}_displayed_events.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
