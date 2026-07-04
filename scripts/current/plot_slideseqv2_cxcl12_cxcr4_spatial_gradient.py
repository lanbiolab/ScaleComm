#!/usr/bin/env python
"""Spatial gradient plot for Cxcl12-Cxcr4 SpatialSelf signaling in Slide-seqV2 hippocampus."""

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
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_cxcl12_cxcr4_spatial_gradient_v1"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
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
    pred = pd.read_csv(RESULT_ROOT / "merged_predictions.csv", usecols=["ligand", "receptor", "receiver_id", "score"])
    source = build_source(cells, pred)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_gradient(source)
    out = OUT_DIR / "slideseqv2_cxcl12_cxcr4_spatial_gradient"
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
            "axes.linewidth": 0.58,
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


def build_source(cells: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    annotated = cells[cells["cell_type"].ne("unannotated")][["cell_id", "x_plot", "y_plot", "cell_type"]].copy()

    sub = pred[(pred["ligand"].eq(LIGAND)) & (pred["receptor"].eq(RECEPTOR))].copy()
    receiver = sub.groupby("receiver_id", as_index=False).agg(score=("score", "mean"), n_edges=("score", "size"))
    table = annotated.merge(receiver, left_on="cell_id", right_on="receiver_id", how="left")
    table["score"] = table["score"].fillna(0.0)
    pos = table.loc[table["score"].gt(0), "score"]
    lo, hi = np.nanpercentile(pos, [50, 99.5]) if len(pos) else (0.0, 1.0)
    table["score_scaled"] = np.clip((table["score"] - lo) / (hi - lo + 1e-9), 0, 1)
    cutoff = float(pos.quantile(0.985)) if len(pos) else 1.0
    table["hotspot"] = table["score"].ge(cutoff)

    core = table[table["hotspot"]].sort_values("score", ascending=False).head(120)
    cx, cy = float(core["x_plot"].median()), float(core["y_plot"].median())
    table["distance_to_core"] = np.sqrt((table["x_plot"] - cx) ** 2 + (table["y_plot"] - cy) ** 2)
    max_dist = float(np.nanpercentile(table["distance_to_core"], 99))
    table["distance_norm"] = np.clip(table["distance_to_core"] / (max_dist + 1e-9), 0, 1)

    bins = np.linspace(0, 1, 16)
    table["distance_bin"] = pd.cut(table["distance_norm"], bins=bins, include_lowest=True)
    trend = (
        table.groupby("distance_bin", observed=True)
        .agg(
            distance=("distance_norm", "mean"),
            mean_score=("score_scaled", "mean"),
            q25=("score_scaled", lambda x: float(np.nanquantile(x, 0.25))),
            q75=("score_scaled", lambda x: float(np.nanquantile(x, 0.75))),
            n=("score_scaled", "size"),
        )
        .reset_index(drop=True)
    )
    return {"spatial_gradient": table, "gradient_trend": trend, "hotspot_core": core}


def plot_gradient(source: dict[str, pd.DataFrame]) -> plt.Figure:
    table = source["spatial_gradient"]
    trend = source["gradient_trend"]
    core = source["hotspot_core"]
    fig = plt.figure(figsize=(5.45, 2.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, bottom=0.16, top=0.86, wspace=0.30, width_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_spatial_core(ax_a, table, core)
    plot_distance_trend(ax_b, table, trend)
    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


def plot_spatial_core(ax: plt.Axes, table: pd.DataFrame, core: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("score", ["#F3EFE7", "#B9D5DF", SPATIAL])
    frame = table.sort_values("score_scaled")
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.2, c=TISSUE, alpha=0.22, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    active = frame[frame["score_scaled"].gt(0)]
    ax.scatter(active["x_plot"], active["y_plot"], s=6.6, c=active["score_scaled"], cmap=cmap, vmin=0, vmax=1, alpha=0.98, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    ax.scatter(core["x_plot"], core["y_plot"], s=10.0, facecolors="none", edgecolors=RESPONSE, linewidths=0.32, alpha=0.78, rasterized=False)
    cx, cy = float(core["x_plot"].median()), float(core["y_plot"].median())
    ax.scatter([cx], [cy], s=22, marker="x", c=RESPONSE, linewidths=0.8, zorder=5)
    set_title(ax, "hotspot core and score field")
    add_spatial_style(ax, table)


def plot_distance_trend(ax: plt.Axes, table: pd.DataFrame, trend: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    sample = table.sample(min(3600, len(table)), random_state=7).copy()
    jitter = rng.normal(0, 0.003, size=len(sample))
    ax.scatter(sample["distance_norm"] + jitter, sample["score_scaled"], s=4.0, c="#CFC7BC", alpha=0.20, linewidths=0, rasterized=True)
    x = trend["distance"].to_numpy(dtype=float)
    y = trend["mean_score"].to_numpy(dtype=float)
    ax.fill_between(x, trend["q25"].to_numpy(dtype=float), trend["q75"].to_numpy(dtype=float), color="#B9D5DF", alpha=0.42, linewidth=0)
    ax.plot(x, y, color=SPATIAL, linewidth=1.25)
    ax.scatter(x, y, s=13, color=SPATIAL, edgecolors=BG, linewidths=0.35, zorder=4)
    corr = float(np.corrcoef(table["distance_norm"], table["score_scaled"])[0, 1])
    ax.text(0.97, 0.92, f"r = {corr:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED)
    ax.set_xlabel("distance from hotspot core", fontsize=6.1)
    ax.set_ylabel("SpatialSelf score", fontsize=6.1)
    set_title(ax, "spatial signaling gradient")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color=GRID, lw=0.42)
    ax.tick_params(axis="both", labelsize=5.3, length=2.0, pad=1.0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)


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
    fig.text(box.x0 - 0.026, box.y1 + 0.014, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
