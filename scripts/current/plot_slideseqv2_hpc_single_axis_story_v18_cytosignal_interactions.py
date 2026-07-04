#!/usr/bin/env python
"""Slide-seqV2 hippocampus single-axis story for Cxcl12-Cxcr4.

The figure is intentionally focused on one biological axis:
tissue context -> ligand/receptor availability -> SpatialSelf localization ->
sender/receiver interpretation -> receiver-side marker response support.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v18_cytosignal_interactions"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
LR_PAIR = f"{LIGAND}-{RECEPTOR}"
RESPONSE_GENES = ["Gfap", "Aqp4", "Slc1a2", "Pdgfra"]

INK = "#202124"
TEXT = "#333840"
MUTED = "#5E6470"
GRID = "#E5E7EB"
TISSUE = "#DDE3EF"
SPATIAL = "#243B73"
RESPONSE = "#A43B73"
SENDER = "#63B35D"
BG = "#FFFFFF"
BLACK = "#111111"

CELL_COLORS = {
    "astrocyte": "#8CCB84",
    "oligodendrocyte": "#B8C1D2",
    "CA1_neuron": "#B8DFF2",
    "CA3_neuron": "#8EBBD8",
    "dentate_neuron": "#F2D578",
    "interneuron": "#F2B5AE",
    "endothelial": "#9CC8BA",
    "microglia_macrophage": "#C9B3D7",
    "polydendrocyte": "#B8C1D2",
    "entorhinal_neuron": "#C9CDD6",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()

    configure()
    cells = pd.read_csv(DATA_ROOT / "cells.csv")
    expr = pd.read_csv(DATA_ROOT / "expression.csv")
    merged = pd.read_csv(RESULT_ROOT / "merged_predictions.csv")

    source = build_source(cells, expr, merged)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_figure(source)
    out = OUT_DIR / "slideseqv2_hpc_cxcl12_cxcr4_single_axis_story"
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 1000
        fig.savefig(f"{out}.{ext}", **kwargs)
    plt.close(fig)
    print(f"{out}.png")
    print(f"{out}.pdf")


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.58,
            "axes.edgecolor": "#706860",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def build_source(cells: pd.DataFrame, expr: pd.DataFrame, merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    annotated = cells[cells["cell_type"].ne("unannotated")].copy()
    annotated = annotated[["cell_id", "x_plot", "y_plot", "cell_type"]]

    expr = expr.copy()
    expr["cell_id"] = expr["cell_id"].astype(str)
    gene_cols = [g for g in [LIGAND, RECEPTOR, *RESPONSE_GENES] if g in expr.columns]
    base = annotated.merge(expr[["cell_id", *gene_cols]], on="cell_id", how="left")
    for gene in gene_cols:
        base[gene] = base[gene].fillna(0.0).astype(float)

    sub = merged[(merged["ligand"].eq(LIGAND)) & (merged["receptor"].eq(RECEPTOR))].copy()
    cutoff_edge = float(sub["score"].quantile(0.985))
    top_edges = sub[sub["score"].ge(cutoff_edge)].copy()
    top_edges = top_edges.merge(
        cells[["cell_id", "x_plot", "y_plot", "cell_type"]].add_prefix("sender_"),
        left_on="sender_id",
        right_on="sender_cell_id",
        how="left",
    ).merge(
        cells[["cell_id", "x_plot", "y_plot", "cell_type"]].add_prefix("receiver_"),
        left_on="receiver_id",
        right_on="receiver_cell_id",
        how="left",
    )
    top_edges_annot = top_edges[
        top_edges["sender_cell_type_y"].ne("unannotated")
        & top_edges["receiver_cell_type_y"].ne("unannotated")
    ].copy()

    receiver_score = sub.groupby("receiver_id", as_index=False).agg(
        spatialself_score=("score", "mean"),
        n_edges=("score", "size"),
    )
    quad = base.merge(receiver_score, left_on="cell_id", right_on="receiver_id", how="left")
    quad["spatialself_score"] = quad["spatialself_score"].fillna(0.0)
    quad["n_edges"] = quad["n_edges"].fillna(0).astype(int)
    pos = quad.loc[quad["spatialself_score"].gt(0), "spatialself_score"]
    cutoff_receiver = float(pos.quantile(0.985)) if len(pos) else 1.0
    quad["hotspot"] = quad["spatialself_score"].ge(cutoff_receiver)
    quad["spatialself_scaled"] = scaled_positive(quad["spatialself_score"], lo_pct=50, hi_pct=99.5)

    sender_layer = top_edges_annot.groupby("sender_id", as_index=False).agg(
        sender_score=("score", "mean"),
        n_edges=("score", "size"),
    )
    receiver_layer = top_edges_annot.groupby("receiver_id", as_index=False).agg(
        receiver_score=("score", "mean"),
        n_edges=("score", "size"),
    )
    sender_layer = base.merge(sender_layer, left_on="cell_id", right_on="sender_id", how="left")
    receiver_layer = base.merge(receiver_layer, left_on="cell_id", right_on="receiver_id", how="left")
    sender_layer["sender_score"] = sender_layer["sender_score"].fillna(0.0)
    receiver_layer["receiver_score"] = receiver_layer["receiver_score"].fillna(0.0)
    sender_layer["sender_scaled"] = scaled_positive(sender_layer["sender_score"])
    receiver_layer["receiver_scaled"] = scaled_positive(receiver_layer["receiver_score"])

    local_edges, local_cells = select_local_edges(base, top_edges_annot)

    response = base.copy()
    genes = [g for g in RESPONSE_GENES if g in response.columns]
    values = response[genes].fillna(0).astype(float)
    z = (values - values.mean(axis=0)) / (values.std(axis=0, ddof=0) + 1e-9)
    response["response_score"] = z.mean(axis=1)
    lo, hi = np.nanpercentile(response["response_score"], [1, 99])
    response["response_scaled"] = np.clip((response["response_score"] - lo) / (hi - lo + 1e-9), 0, 1)
    response = response.merge(
        quad[["cell_id", "hotspot", "spatialself_score"]],
        on="cell_id",
        how="left",
    )
    response["hotspot"] = response["hotspot"].fillna(False).astype(bool)
    response["spatialself_score"] = response["spatialself_score"].fillna(0.0)

    validation = response[["cell_id", "hotspot", "response_score"]].copy()
    validation["group"] = np.where(validation["hotspot"], "hotspot receivers", "other beads")

    summary = pd.DataFrame(
        [
            {"metric": "annotated_beads", "value": len(base)},
            {"metric": "top_edge_cutoff", "value": cutoff_edge},
            {"metric": "top_edges", "value": len(top_edges)},
            {"metric": "annotated_top_edges", "value": len(top_edges_annot)},
            {"metric": "hotspot_receivers", "value": int(quad["hotspot"].sum())},
            {"metric": "sender_beads", "value": int(sender_layer["sender_score"].gt(0).sum())},
            {"metric": "receiver_beads", "value": int(receiver_layer["receiver_score"].gt(0).sum())},
        ]
    )

    return {
        "panel_a_cells": base,
        "panel_b_expression_score": quad,
        "panel_d_sender_layer": sender_layer,
        "panel_d_receiver_layer": receiver_layer,
        "panel_d_local_edges": local_edges,
        "panel_d_local_cells": local_cells,
        "panel_e_response": response,
        "panel_f_validation": validation,
        "summary": summary,
    }


def scaled_positive(values: pd.Series, lo_pct: float = 5, hi_pct: float = 99.5) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(pos, [lo_pct, hi_pct])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def select_local_edges(cells: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return edges.copy(), cells.copy()
    top_receivers = edges.sort_values("score", ascending=False).head(120)
    cx = float(top_receivers["receiver_x_plot"].median())
    cy = float(top_receivers["receiver_y_plot"].median())
    span = max(
        float(cells["x_plot"].max() - cells["x_plot"].min()),
        float(cells["y_plot"].max() - cells["y_plot"].min()),
    )
    half = span * 0.14
    local_edges = edges[
        edges["receiver_x_plot"].between(cx - half, cx + half)
        & edges["receiver_y_plot"].between(cy - half * 0.8, cy + half * 0.8)
    ].sort_values("score", ascending=False).head(90)
    if local_edges.empty:
        local_edges = edges.sort_values("score", ascending=False).head(90)
    xmin = float(local_edges[["sender_x_plot", "receiver_x_plot"]].min().min())
    xmax = float(local_edges[["sender_x_plot", "receiver_x_plot"]].max().max())
    ymin = float(local_edges[["sender_y_plot", "receiver_y_plot"]].min().min())
    ymax = float(local_edges[["sender_y_plot", "receiver_y_plot"]].max().max())
    xpad = (xmax - xmin + 1e-9) * 0.30
    ypad = (ymax - ymin + 1e-9) * 0.30
    local_cells = cells[
        cells["x_plot"].between(xmin - xpad, xmax + xpad)
        & cells["y_plot"].between(ymin - ypad, ymax + ypad)
    ].copy()
    return local_edges, local_cells


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(7.70, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        left=0.035,
        right=0.992,
        bottom=0.078,
        top=0.955,
        wspace=0.105,
        hspace=0.30,
        width_ratios=[1.0, 1.0, 1.0, 1.08],
        height_ratios=[1.0, 1.0],
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[0, 3]),
        "e": fig.add_subplot(gs[1, 0:2]),
        "f": fig.add_subplot(gs[1, 2]),
        "g": fig.add_subplot(gs[1, 3]),
    }
    data = source["panel_b_expression_score"]
    plot_cell_map(axes["a"], source["panel_a_cells"])
    expr_cmap = LinearSegmentedColormap.from_list("expr", ["#F7F8FA", "#F7DF6E", "#EC8C3D", "#6A3D9A"])
    plot_expression(axes["b"], data, LIGAND, expr_cmap, "Cxcl12")
    plot_expression(axes["c"], data, RECEPTOR, expr_cmap, "Cxcr4")
    plot_hotspot_locations(axes["d"], data, large=True)
    plot_interaction_layers(axes["e"], source["panel_d_local_cells"], source["panel_d_local_edges"])
    plot_response_map(axes["f"], source["panel_e_response"])
    plot_validation(axes["g"], source["panel_f_validation"])
    for label, ax in axes.items():
        panel_label(fig, ax, label)
    return fig


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    background = cells[~cells["cell_type"].isin(["astrocyte", "dentate_neuron", "CA1_neuron", "CA3_neuron", "interneuron"])]
    ax.scatter(
        background["x_plot"],
        background["y_plot"],
        s=1.65,
        c=TISSUE,
        alpha=0.38,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    order = ["astrocyte", "dentate_neuron", "CA1_neuron", "CA3_neuron", "interneuron"]
    for cell_type in order:
        sub = cells[cells["cell_type"].eq(cell_type)]
        ax.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=1.95,
            c=CELL_COLORS.get(cell_type, TISSUE),
            alpha=0.82,
            linewidths=0,
            edgecolors="none",
            antialiaseds=False,
            rasterized=False,
            label=pretty_cell(cell_type),
        )
    set_title(ax, "Slide-seqV2 hippocampus")
    add_spatial_frame(ax, cells)
    add_celltype_key(ax)


def add_celltype_key(ax: plt.Axes) -> None:
    labels = [
        ("astrocyte", CELL_COLORS["astrocyte"]),
        ("dentate", CELL_COLORS["dentate_neuron"]),
        ("CA1/CA3", CELL_COLORS["CA1_neuron"]),
        ("interneuron", CELL_COLORS["interneuron"]),
        ("other", TISSUE),
    ]
    x0, y0 = 0.060, 0.075
    for i, (label, color) in enumerate(labels):
        x = x0 + (i % 3) * 0.235
        y = y0 + (1 - i // 3) * 0.052
        ax.scatter([x], [y], s=7.2, c=color, transform=ax.transAxes, linewidths=0, clip_on=False, zorder=20)
        ax.text(x + 0.018, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.4, color=MUTED, zorder=20)


def add_spatial_frame(ax: plt.Axes, data: pd.DataFrame, pad_frac: float = 0.018) -> None:
    ax.set_aspect("equal")
    if len(data):
        xspan = float(data["x_plot"].max() - data["x_plot"].min())
        yspan = float(data["y_plot"].max() - data["y_plot"].min())
        pad = max(xspan, yspan) * pad_frac
        ax.set_xlim(float(data["x_plot"].min()) - pad, float(data["x_plot"].max()) + pad)
        ax.set_ylim(float(data["y_plot"].min()) - pad, float(data["y_plot"].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_expression_pair(fig: plt.Figure, host: plt.Axes, data: pd.DataFrame) -> None:
    host.set_axis_off()
    sub = host.get_subplotspec().subgridspec(1, 2, wspace=0.13)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    expr_cmap = LinearSegmentedColormap.from_list("expr", ["#F7F8FA", "#F7DF6E", "#EC8C3D", "#6A3D9A"])
    plot_expression(axes[0], data, LIGAND, expr_cmap, "Cxcl12")
    plot_expression(axes[1], data, RECEPTOR, expr_cmap, "Cxcr4")


def plot_sender_receiver_layers(fig: plt.Figure, host: plt.Axes, sender: pd.DataFrame, receiver: pd.DataFrame) -> None:
    host.set_axis_off()
    sub = host.get_subplotspec().subgridspec(1, 2, wspace=0.13)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    plot_layer(axes[0], sender, "sender", SENDER, "Cxcl12-sending")
    plot_layer(axes[1], receiver, "receiver", SPATIAL, "Cxcr4-receiving")


def plot_expression(ax: plt.Axes, data: pd.DataFrame, gene: str, cmap: LinearSegmentedColormap, title: str) -> None:
    frame = data.sort_values([gene, "cell_id"], ascending=[True, True]).copy()
    values = frame[gene].to_numpy(dtype=float)
    vmax = float(np.nanquantile(values[values > 0], 0.99)) if np.any(values > 0) else 1.0
    vmax = max(vmax, 1e-9)
    ax.scatter(
        frame["x_plot"],
        frame["y_plot"],
        s=2.0,
        c=TISSUE,
        alpha=0.34,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    pos = frame[frame[gene].gt(0)].copy()
    ax.scatter(
        pos["x_plot"],
        pos["y_plot"],
        s=3.4,
        c=np.clip(pos[gene].to_numpy(dtype=float), 0, vmax),
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=vmax),
        alpha=0.98,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    set_title(ax, title)
    add_spatial_frame(ax, data)
    add_colorbar(ax, cmap, Normalize(0, vmax), width=0.42)


def plot_spatialself_score(ax: plt.Axes, data: pd.DataFrame, cmap: LinearSegmentedColormap) -> None:
    frame = data.sort_values(["spatialself_scaled", "cell_id"], ascending=[True, True]).copy()
    ax.scatter(
        frame["x_plot"],
        frame["y_plot"],
        s=2.0,
        c=TISSUE,
        alpha=0.34,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    active = frame[frame["spatialself_scaled"].gt(0)].copy()
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=6.0,
        c=active["spatialself_scaled"],
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.98,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    set_title(ax, "SpatialSelf score")
    add_spatial_frame(ax, data)
    add_colorbar(ax, cmap, Normalize(0, 1), width=0.42)


def plot_hotspot_locations(ax: plt.Axes, data: pd.DataFrame, large: bool = False) -> None:
    ax.scatter(
        data["x_plot"],
        data["y_plot"],
        s=2.1 if large else 2.0,
        c=TISSUE,
        alpha=0.34 if large else 0.30,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    hot = data[data["hotspot"]].copy()
    ax.scatter(
        hot["x_plot"],
        hot["y_plot"],
        s=5.8 if large else 4.8,
        c=SPATIAL,
        alpha=0.98,
        linewidths=0.0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    set_title(ax, f"{LR_PAIR} hotspots" if large else "hotspots")
    add_spatial_frame(ax, data)
    ax.text(
        0.035,
        0.052,
        f"n={len(hot):,}",
        transform=ax.transAxes,
        fontsize=4.8,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def plot_layer(ax: plt.Axes, data: pd.DataFrame, kind: str, color: str, title: str) -> None:
    score_col = f"{kind}_score"
    scaled_col = f"{kind}_scaled"
    ax.scatter(
        data["x_plot"],
        data["y_plot"],
        s=1.9,
        c=TISSUE,
        alpha=0.32,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    active = data[data[score_col].gt(0)].sort_values(scaled_col)
    cmap = LinearSegmentedColormap.from_list(kind, ["#F7F8FA", "#DCE8F2", color])
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=4.0,
        c=active[scaled_col],
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.94,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    set_title(ax, title)
    add_spatial_frame(ax, data)
    ax.text(0.035, 0.052, f"n={len(active):,}", transform=ax.transAxes, fontsize=4.7, color=MUTED, ha="left", va="bottom")



def plot_interaction_layers(ax: plt.Axes, local_cells: pd.DataFrame, edges: pd.DataFrame) -> None:
    ax.set_axis_off()
    if edges.empty or local_cells.empty:
        set_title(ax, "sending-receiving interactions")
        return
    display_edges = select_display_interactions(edges, max_edges=18)
    xmin = float(local_cells["x_plot"].min())
    xmax = float(local_cells["x_plot"].max())
    ymin = float(local_cells["y_plot"].min())
    ymax = float(local_cells["y_plot"].max())
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    def norm_x(values: pd.Series) -> np.ndarray:
        return (values.to_numpy(dtype=float) - xmin) / xspan

    def norm_y(values: pd.Series, offset: float) -> np.ndarray:
        return offset + 0.30 * ((values.to_numpy(dtype=float) - ymin) / yspan - 0.5)

    # Two muted tissue layers, mirroring the same local neighborhood.
    ax.scatter(norm_x(local_cells["x_plot"]), norm_y(local_cells["y_plot"], 0.72), s=4.0, c=TISSUE, alpha=0.28, linewidths=0, zorder=1)
    ax.scatter(norm_x(local_cells["x_plot"]), norm_y(local_cells["y_plot"], 0.28), s=4.0, c=TISSUE, alpha=0.28, linewidths=0, zorder=1)

    sx = (display_edges["sender_x_plot"].to_numpy(dtype=float) - xmin) / xspan
    sy = 0.72 + 0.30 * ((display_edges["sender_y_plot"].to_numpy(dtype=float) - ymin) / yspan - 0.5)
    rx = (display_edges["receiver_x_plot"].to_numpy(dtype=float) - xmin) / xspan
    ry = 0.28 + 0.30 * ((display_edges["receiver_y_plot"].to_numpy(dtype=float) - ymin) / yspan - 0.5)
    score = display_edges["score"].to_numpy(dtype=float)
    score_scaled = (score - score.min()) / (score.max() - score.min() + 1e-9)

    for x0, y0, x1, y1, ss in zip(sx, sy, rx, ry, score_scaled, strict=False):
        ax.plot([x0, x1], [y0, y1], color=mpl.colors.to_rgba(SPATIAL, 0.22 + 0.34 * float(ss)), lw=0.38 + 0.34 * float(ss), zorder=2)
    ax.scatter(sx, sy, s=13.5, c=SENDER, edgecolors=BG, linewidths=0.32, alpha=0.95, zorder=4)
    ax.scatter(rx, ry, s=13.5, c=SPATIAL, edgecolors=BG, linewidths=0.32, alpha=0.95, zorder=4)

    ax.text(0.02, 0.93, "Cxcl12-sending layer", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=SENDER)
    ax.text(0.02, 0.49, "Cxcr4-receiving layer", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=SPATIAL)
    ax.text(0.98, 0.045, f"representative edges: {len(display_edges)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7, color=MUTED)
    set_title(ax, "inferred sending-receiving interactions")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.04, 0.96)


def select_display_interactions(edges: pd.DataFrame, max_edges: int = 18) -> pd.DataFrame:
    table = edges.copy()
    table = table.sort_values(["score", "distance"], ascending=[False, False])
    selected: list[int] = []
    mids: list[tuple[float, float]] = []
    min_sep = 36.0
    for idx, row in table.iterrows():
        mx = (float(row.sender_x_plot) + float(row.receiver_x_plot)) / 2.0
        my = (float(row.sender_y_plot) + float(row.receiver_y_plot)) / 2.0
        if all(((mx - ox) ** 2 + (my - oy) ** 2) ** 0.5 >= min_sep for ox, oy in mids):
            selected.append(idx)
            mids.append((mx, my))
        if len(selected) >= max_edges:
            break
    if len(selected) < min(max_edges, len(table)):
        for idx in table.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min(max_edges, len(table)):
                break
    return edges.loc[selected].copy()


def plot_response_map(ax: plt.Axes, response: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("resp", ["#F9F7EC", "#F2D777", "#E98C3C", "#C8452D"])
    frame = response.sort_values(["response_scaled", "cell_id"], ascending=[True, True]).copy()
    ax.scatter(
        frame["x_plot"],
        frame["y_plot"],
        s=2.1,
        c=frame["response_scaled"].astype(float),
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.88,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    hot = response[response["hotspot"]]
    ax.scatter(
        hot["x_plot"],
        hot["y_plot"],
        s=5.8,
        facecolors="none",
        edgecolors=SPATIAL,
        alpha=0.38,
        linewidths=0.16,
        rasterized=False,
    )
    set_title(ax, "receiver response")
    add_spatial_frame(ax, response)
    add_colorbar(ax, cmap, Normalize(0, 1), width=0.36)


def plot_validation(ax: plt.Axes, validation: pd.DataFrame) -> None:
    rng = np.random.default_rng(19)
    groups = ["other beads", "hotspot receivers"]
    colors = ["#D7D1C7", SPATIAL]
    data = [
        validation.loc[validation["group"].eq(group), "response_score"].dropna().to_numpy(dtype=float)
        for group in groups
    ]
    bp = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": BLACK, "linewidth": 0.70},
        whiskerprops={"color": BLACK, "linewidth": 0.54},
        capprops={"color": BLACK, "linewidth": 0.54},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set(facecolor=mpl.colors.to_rgba(color, 0.26), edgecolor=BLACK, linewidth=0.42)
    for idx, vals in enumerate(data):
        sample = vals if len(vals) <= 180 else rng.choice(vals, 180, replace=False)
        x = idx + rng.normal(0, 0.040, len(sample))
        alpha = 0.42 if idx == 0 else 0.72
        ax.scatter(x, sample, s=4.8, c=colors[idx], alpha=alpha, edgecolors="none", linewidths=0, zorder=3)
        ax.scatter([idx], [np.nanmean(vals)], s=13, marker="D", c=colors[idx], edgecolors=BLACK, linewidths=0.22, zorder=5)
    bg, hot = data
    delta = float(np.nanmean(hot) - np.nanmean(bg))
    ax.text(0.5, 0.96, f"Delta z={delta:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=4.9, color=SPATIAL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nbeads", "hotspot\nreceivers"], fontsize=5.4)
    ax.set_ylabel("receiver response\n(z-score)", fontsize=5.6, color=MUTED)
    set_title(ax, "response support")
    style_quant_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.32)


def add_colorbar(ax: plt.Axes, cmap: LinearSegmentedColormap, norm: Normalize, width: float = 0.52) -> None:
    x0 = (1.0 - width) / 2.0
    cax = ax.inset_axes([x0, -0.075, width, 0.030])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.34)
    cb.set_ticks([norm.vmin, norm.vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.1, length=1.2, width=0.30, pad=0.6)


def add_spatial_style(ax: plt.Axes, data: pd.DataFrame) -> None:
    add_spatial_frame(ax, data)


def style_quant_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.58)
    ax.spines["bottom"].set_linewidth(0.58)
    ax.grid(axis="x", color=GRID, lw=0.38)
    ax.tick_params(axis="both", labelsize=5.3, length=2.0, pad=1.1)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=5.8, pad=0.8, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.016,
        box.y1 + 0.010,
        label,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )


def pretty_cell(label: str) -> str:
    return label.replace("_neuron", "").replace("_", " ")


if __name__ == "__main__":
    main()
