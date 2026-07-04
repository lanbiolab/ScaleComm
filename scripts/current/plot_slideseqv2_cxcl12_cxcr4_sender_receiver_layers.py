#!/usr/bin/env python
"""Sender/receiver layered view for Cxcl12-Cxcr4 in Slide-seqV2 hippocampus."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_cxcl12_cxcr4_sender_receiver_layers_v2"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
LR_PAIR = f"{LIGAND}-{RECEPTOR}"

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
TISSUE = "#D8D2C8"
SENDER = "#93CC82"
RECEIVER = "#2F78A8"
EDGE = "#2F78A8"
BG = "#FFFFFF"
BLACK = "#111111"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()

    configure()
    cells = pd.read_csv(DATA_ROOT / "cells.csv")
    expr = pd.read_csv(DATA_ROOT / "expression.csv", usecols=["cell_id", LIGAND, RECEPTOR])
    pred = pd.read_csv(
        RESULT_ROOT / "merged_predictions.csv",
        usecols=["ligand", "receptor", "sender_id", "receiver_id", "score", "sender_cell_type", "receiver_cell_type"],
    )
    source = build_source(cells, expr, pred)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_layers(source)
    out = OUT_DIR / "slideseqv2_cxcl12_cxcr4_sender_receiver_layers"
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
            "axes.linewidth": 0.62,
            "axes.edgecolor": "#6A625B",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def build_source(cells: pd.DataFrame, expr: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    annotated = cells[cells["cell_type"].ne("unannotated")][["cell_id", "x_plot", "y_plot", "cell_type"]].copy()
    expr = expr.fillna(0.0)
    annotated = annotated.merge(expr, on="cell_id", how="left")

    sub = pred[(pred["ligand"].eq(LIGAND)) & (pred["receptor"].eq(RECEPTOR))].copy()
    cutoff = float(sub["score"].quantile(0.985))
    top = sub[sub["score"].ge(cutoff)].copy()
    top = top.merge(
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

    top_annot = top[
        top["sender_cell_type_y"].ne("unannotated") & top["receiver_cell_type_y"].ne("unannotated")
    ].copy()
    sender = top_annot.groupby("sender_id", as_index=False).agg(sender_score=("score", "mean"), n_edges=("score", "size"))
    receiver = top_annot.groupby("receiver_id", as_index=False).agg(receiver_score=("score", "mean"), n_edges=("score", "size"))
    sender = annotated.merge(sender, left_on="cell_id", right_on="sender_id", how="left")
    receiver = annotated.merge(receiver, left_on="cell_id", right_on="receiver_id", how="left")
    sender["sender_score"] = sender["sender_score"].fillna(0.0)
    sender["sender_scaled"] = scaled_positive(sender["sender_score"])
    receiver["receiver_score"] = receiver["receiver_score"].fillna(0.0)
    receiver["receiver_scaled"] = scaled_positive(receiver["receiver_score"])

    if top_annot.empty:
        local_edges = top_annot
    else:
        top_receivers = top_annot.sort_values("score", ascending=False).head(120)
        cx = float(top_receivers["receiver_x_plot"].median())
        cy = float(top_receivers["receiver_y_plot"].median())
        span = max(float(annotated["x_plot"].max() - annotated["x_plot"].min()), float(annotated["y_plot"].max() - annotated["y_plot"].min()))
        half = span * 0.14
        local_edges = top_annot[
            top_annot["receiver_x_plot"].between(cx - half, cx + half)
            & top_annot["receiver_y_plot"].between(cy - half * 0.8, cy + half * 0.8)
        ].sort_values("score", ascending=False).head(38)
    local_cells = annotated.copy()
    if not local_edges.empty:
        xmin, xmax = local_edges[["sender_x_plot", "receiver_x_plot"]].min().min(), local_edges[["sender_x_plot", "receiver_x_plot"]].max().max()
        ymin, ymax = local_edges[["sender_y_plot", "receiver_y_plot"]].min().min(), local_edges[["sender_y_plot", "receiver_y_plot"]].max().max()
        xpad = (xmax - xmin + 1e-9) * 0.30
        ypad = (ymax - ymin + 1e-9) * 0.30
        local_cells = annotated[annotated["x_plot"].between(xmin - xpad, xmax + xpad) & annotated["y_plot"].between(ymin - ypad, ymax + ypad)].copy()

    summary = pd.DataFrame(
        [
            {"metric": "top_edges", "value": len(top)},
            {"metric": "annotated_top_edges", "value": len(top_annot)},
            {"metric": "annotated_senders", "value": int(sender["sender_score"].gt(0).sum())},
            {"metric": "annotated_receivers", "value": int(receiver["receiver_score"].gt(0).sum())},
        ]
    )
    return {
        "annotated_cells": annotated,
        "sender_layer": sender,
        "receiver_layer": receiver,
        "local_edges": local_edges,
        "local_cells": local_cells,
        "summary": summary,
    }


def scaled_positive(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(pos, [5, 99.5])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def plot_layers(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(5.55, 2.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.988, bottom=0.12, top=0.86, wspace=0.19)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    plot_layer(axes[0], source["sender_layer"], "sender", SENDER, "Cxcl12-sending beads")
    plot_layer(axes[1], source["receiver_layer"], "receiver", RECEIVER, "Cxcr4-receiving beads")
    plot_local_edges(axes[2], source["local_cells"], source["local_edges"])
    for label, ax in zip(["a", "b", "c"], axes, strict=True):
        panel_label(fig, ax, label)
    return fig


def plot_layer(ax: plt.Axes, data: pd.DataFrame, kind: str, color: str, title: str) -> None:
    score_col = f"{kind}_score"
    scaled_col = f"{kind}_scaled"
    ax.scatter(data["x_plot"], data["y_plot"], s=3.2, c=TISSUE, alpha=0.24, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    active = data[data[score_col].gt(0)].sort_values(scaled_col)
    cmap = LinearSegmentedColormap.from_list(kind, ["#F3EFE7", "#D9E6E0", color])
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=7.4,
        c=active[scaled_col],
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.98,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    set_title(ax, title)
    add_spatial_style(ax, data)
    ax.text(0.04, 0.055, f"n={len(active):,}", transform=ax.transAxes, fontsize=5.0, color=MUTED, ha="left", va="bottom")


def plot_local_edges(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame) -> None:
    ax.scatter(cells["x_plot"], cells["y_plot"], s=3.2, c=TISSUE, alpha=0.16, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    if not edges.empty:
        vmax = float(edges["score"].max())
        send = edges[["sender_x_plot", "sender_y_plot"]].drop_duplicates()
        rec = edges[["receiver_x_plot", "receiver_y_plot"]].drop_duplicates()
        ax.scatter(send["sender_x_plot"], send["sender_y_plot"], s=7.8, c=SENDER, edgecolors=BLACK, linewidths=0.10, alpha=0.80, zorder=4)
        ax.scatter(rec["receiver_x_plot"], rec["receiver_y_plot"], s=7.8, c=RECEIVER, edgecolors=BLACK, linewidths=0.10, alpha=0.82, zorder=5)
        for row in edges.itertuples(index=False):
            val = float(row.score) / vmax
            patch = FancyArrowPatch(
                (row.sender_x_plot, row.sender_y_plot),
                (row.receiver_x_plot, row.receiver_y_plot),
                arrowstyle="-|>",
                mutation_scale=5.2,
                linewidth=0.22 + 0.16 * val,
                color="#1F5E7F",
                alpha=0.44 + 0.22 * val,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=6,
            )
            ax.add_patch(patch)
        ax.scatter(send["sender_x_plot"], send["sender_y_plot"], s=8.8, facecolors="none", edgecolors=SENDER, linewidths=0.24, alpha=0.90, zorder=7)
        ax.scatter(rec["receiver_x_plot"], rec["receiver_y_plot"], s=8.8, facecolors="none", edgecolors=RECEIVER, linewidths=0.24, alpha=0.92, zorder=8)
    set_title(ax, "local sender-to-receiver edges")
    add_spatial_style(ax, cells)
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=SENDER, markeredgecolor=BLACK, markeredgewidth=0.18, markersize=3.5, label="sender"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=RECEIVER, markeredgecolor=BLACK, markeredgewidth=0.18, markersize=3.5, label="receiver"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=4.8, handletextpad=0.22, borderaxespad=0.08)


def add_spatial_style(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    xspan = float(data["x_plot"].max() - data["x_plot"].min())
    yspan = float(data["y_plot"].max() - data["y_plot"].min())
    pad = max(xspan, yspan) * 0.018
    ax.set_xlim(float(data["x_plot"].min()) - pad, float(data["x_plot"].max()) + pad)
    ax.set_ylim(float(data["y_plot"].min()) - pad, float(data["y_plot"].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#7A716A")
        ax.spines[side].set_linewidth(0.54)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=6.8, pad=2.0, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.022, box.y1 + 0.014, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
