#!/usr/bin/env python
"""CytoSignal-style four-panel view for Cxcl12-Cxcr4 in Slide-seqV2 hippocampus."""

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
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_cxcl12_cxcr4_quad_panel_v2"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
LR_PAIR = f"{LIGAND}-{RECEPTOR}"

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
TISSUE = "#D8D2C8"
SPATIAL = "#2F78A8"
RESPONSE = "#C86B53"
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
    pred = pd.read_csv(RESULT_ROOT / "merged_predictions.csv", usecols=["ligand", "receptor", "receiver_id", "score"])

    source = build_source(cells, expr, pred)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_quad(source)
    out = OUT_DIR / "slideseqv2_cxcl12_cxcr4_ligand_receptor_score_hotspot"
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
            "font.size": 6.5,
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
    base = annotated.merge(expr, on="cell_id", how="left")
    base[[LIGAND, RECEPTOR]] = base[[LIGAND, RECEPTOR]].fillna(0.0)

    sub = pred[(pred["ligand"].eq(LIGAND)) & (pred["receptor"].eq(RECEPTOR))].copy()
    receiver = sub.groupby("receiver_id", as_index=False).agg(spatialself_score=("score", "mean"), n_edges=("score", "size"))
    receiver = base.merge(receiver, left_on="cell_id", right_on="receiver_id", how="left")
    receiver["spatialself_score"] = receiver["spatialself_score"].fillna(0.0)
    receiver["n_edges"] = receiver["n_edges"].fillna(0).astype(int)
    cutoff = float(receiver.loc[receiver["spatialself_score"].gt(0), "spatialself_score"].quantile(0.985))
    receiver["hotspot"] = receiver["spatialself_score"].ge(cutoff)
    positive = receiver.loc[receiver["spatialself_score"].gt(0), "spatialself_score"]
    if len(positive):
        lo, hi = np.nanpercentile(positive, [50, 99.5])
        receiver["spatialself_scaled"] = np.clip((receiver["spatialself_score"] - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        receiver["spatialself_scaled"] = 0.0
    return {"quad_data": receiver}


def plot_quad(source: dict[str, pd.DataFrame]) -> plt.Figure:
    data = source["quad_data"]
    fig = plt.figure(figsize=(7.2, 2.05), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        4,
        left=0.035,
        right=0.992,
        bottom=0.16,
        top=0.88,
        wspace=0.18,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    expr_cmap = LinearSegmentedColormap.from_list("expr", ["#F3EFE7", "#E6C06A", RESPONSE])
    score_cmap = LinearSegmentedColormap.from_list("score", ["#F3EFE7", "#B9D5DF", SPATIAL])

    plot_expression(axes[0], data, LIGAND, expr_cmap, f"{LIGAND} expression")
    plot_expression(axes[1], data, RECEPTOR, expr_cmap, f"{RECEPTOR} expression")
    plot_score(axes[2], data, score_cmap)
    plot_hotspot(axes[3], data)

    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        panel_label(fig, ax, label)
    return fig


def plot_expression(ax: plt.Axes, data: pd.DataFrame, gene: str, cmap: LinearSegmentedColormap, title: str) -> None:
    frame = data.sort_values([gene, "cell_id"], ascending=[True, True]).copy()
    values = frame[gene].to_numpy(dtype=float)
    vmax = float(np.nanquantile(values[values > 0], 0.99)) if np.any(values > 0) else 1.0
    vmax = max(vmax, 1e-9)
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.2, c=TISSUE, alpha=0.22, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    pos = frame[frame[gene].gt(0)].copy()
    ax.scatter(
        pos["x_plot"],
        pos["y_plot"],
        s=5.6,
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
    add_spatial_style(ax, data)
    add_colorbar(ax, cmap, Normalize(0, vmax))


def plot_score(ax: plt.Axes, data: pd.DataFrame, cmap: LinearSegmentedColormap) -> None:
    frame = data.sort_values(["spatialself_scaled", "cell_id"], ascending=[True, True]).copy()
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.2, c=TISSUE, alpha=0.22, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    active = frame[frame["spatialself_scaled"].gt(0)].copy()
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=5.8,
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
    add_spatial_style(ax, data)
    add_colorbar(ax, cmap, Normalize(0, 1))


def plot_hotspot(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.scatter(data["x_plot"], data["y_plot"], s=3.2, c=TISSUE, alpha=0.24, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    hot = data[data["hotspot"]].copy()
    ax.scatter(
        hot["x_plot"],
        hot["y_plot"],
        s=7.6,
        c=SPATIAL,
        alpha=0.98,
        linewidths=0.10,
        edgecolors=BLACK,
        rasterized=False,
    )
    set_title(ax, "Hotspot locations")
    add_spatial_style(ax, data)
    ax.text(0.04, 0.06, f"top {len(hot):,} receivers", transform=ax.transAxes, fontsize=5.1, color=MUTED, ha="left", va="bottom")


def add_colorbar(ax: plt.Axes, cmap: LinearSegmentedColormap, norm: Normalize) -> None:
    cax = ax.inset_axes([0.18, -0.125, 0.55, 0.042])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.38)
    cb.set_ticks([norm.vmin, norm.vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.6, length=1.8, width=0.4, pad=1)


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
    fig.text(box.x0 - 0.020, box.y1 + 0.012, label, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
