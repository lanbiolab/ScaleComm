#!/usr/bin/env python
"""Article-style CosMx NSCLC scale/context evidence additions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables"
OUT = ROOT / "outputs/article_style_figure_additions/cosmx_nsclc_article_style"
SOURCE_OUT = OUT / "source_data"

INK = "#171717"
TEXT = "#2D2A27"
MUTED = "#6E665B"
GRID = "#E7E1D8"
BG = "#FFFFFF"
RED = "#C94F48"
TEAL = "#237A78"
BLUE = "#2F78A8"
AMBER = "#D7B56D"
TAUPE = "#BDB8AE"
PURPLE = "#7E6C9A"

CELL_COLORS = {
    "tumor_proxy": "#D8D2C8",
    "t_cell_proxy": "#C94F48",
    "myeloid_proxy": "#D7B56D",
    "stromal_proxy": "#237A78",
    "endothelial_proxy": "#6D88A8",
    "other_proxy": "#BDB8AE",
}
LR_COLORS = {
    "CXCL12-CXCR4": TEAL,
    "CXCL10-CXCR3": RED,
    "CXCL9-CXCR3": "#B85450",
    "VEGFA-KDR": BLUE,
    "TGFB1-TGFBR2": AMBER,
    "SPP1-CD44": PURPLE,
    "CSF1-CSF1R": "#8A8177",
}
ACTIVITY_CMAP = LinearSegmentedColormap.from_list("cosmx_activity", ["#F7F4EC", "#E0B46C", "#C94F48", "#641C1C"])
TARGET_CMAP = LinearSegmentedColormap.from_list("cosmx_target", ["#F7F4EC", "#BFD0C7", "#2F78A8"])
SCALE_CMAP = LinearSegmentedColormap.from_list("cosmx_scale", ["#F7F4EC", "#D9B66F", "#B85450"])

SELECTED_LRS = ["CXCL12-CXCR4", "CXCL10-CXCR3", "TGFB1-TGFBR2", "VEGFA-KDR", "SPP1-CD44"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.1,
            "axes.linewidth": 0.55,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.025, dy: float = 0.014) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.45, length=2.0, pad=1.4, colors=TEXT)
    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.35, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.35, zorder=0)
    ax.set_axisbelow(True)


def robust_norm(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo, hi = arr.quantile(0.02), arr.quantile(0.985)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = arr.min(), arr.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def spatial_axis(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    px = (frame["x_plot"].max() - frame["x_plot"].min()) * 0.012
    py = (frame["y_plot"].max() - frame["y_plot"].min()) * 0.012
    ax.set_xlim(frame["x_plot"].min() - px, frame["x_plot"].max() + px)
    ax.set_ylim(frame["y_plot"].min() - py, frame["y_plot"].max() + py)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_source() -> dict[str, pd.DataFrame]:
    cells = pd.read_csv(SOURCE / "cosmx_cells_for_context_map.csv")
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    return {
        "cells": cells,
        "lr_summary": pd.read_csv(SOURCE / "cosmx_lr_scale_context_summary.csv"),
        "support": pd.read_csv(SOURCE / "cosmx_receiver_target_support.csv"),
        "scale_vs_noscale": pd.read_csv(SOURCE / "cosmx_scale_vs_noscale_target_support.csv"),
        "hotspots": pd.read_csv(SOURCE / "cosmx_hotspot_receiver_cells.csv"),
        "ranking": pd.read_csv(SOURCE / "cosmx_fig5_story_candidate_ranking.csv"),
    }


def plot_context_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    order = ["other_proxy", "stromal_proxy", "endothelial_proxy", "tumor_proxy", "myeloid_proxy", "t_cell_proxy"]
    for ct in order:
        sub = cells[cells["cell_type"].eq(ct)]
        if sub.empty:
            continue
        ax.scatter(sub["x_plot"], sub["y_plot"], s=1.85, color=CELL_COLORS.get(ct, TAUPE), alpha=0.78, linewidths=0, rasterized=True)
    spatial_axis(ax, cells)
    ax.set_title("CosMx NSCLC context", pad=4)
    handles = [
        Line2D([], [], marker="o", ls="", color=CELL_COLORS[k], markersize=4.0, label=v)
        for k, v in [
            ("tumor_proxy", "tumor"),
            ("t_cell_proxy", "T cell"),
            ("myeloid_proxy", "myeloid"),
            ("stromal_proxy", "stroma"),
            ("endothelial_proxy", "endothelial"),
        ]
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.00, -0.03), ncol=2, frameon=False, handletextpad=0.20, columnspacing=0.55)


def plot_scale_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    scale_cols = [f"scale_{i}" for i in range(6)]
    dat = summary[summary["LR"].isin(SELECTED_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("order")
    mat = dat[scale_cols].astype(float).to_numpy()
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SCALE_CMAP(mat[i, j]), edgecolor="#8F8B85", linewidth=0.20))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_xticks([0, 2.5, 5])
    ax.set_xticklabels(["local", "mid", "broad"])
    ax.set_yticks(range(len(dat)))
    ax.set_yticklabels(dat["LR"].tolist(), fontsize=5.1)
    for tick, lr in zip(ax.get_yticklabels(), dat["LR"], strict=False):
        tick.set_color(LR_COLORS.get(lr, TEXT))
    ax.tick_params(length=0, pad=1.4)
    ax.set_title("Learned LR scale profile", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_delta(ax: plt.Axes, table: pd.DataFrame) -> None:
    dat = table[table["method"].isin(["scale-aware", "no-scale"])].copy()
    wide = dat.pivot_table(index="LR", columns="method", values="target_spearman_r", aggfunc="mean")
    wide = wide.dropna(subset=["scale-aware", "no-scale"]).copy()
    wide["delta"] = wide["scale-aware"] - wide["no-scale"]
    wide = wide.reindex([lr for lr in SELECTED_LRS if lr in wide.index]).dropna(subset=["delta"])
    vals = wide["delta"].to_numpy(float)
    y = np.arange(len(wide))
    colors = [RED if v >= 0 else BLUE for v in vals]
    ax.axvline(0, color=INK, lw=0.55)
    ax.barh(y, vals, height=0.56, color=colors, alpha=0.86, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index.tolist(), fontsize=5.1)
    for tick, lr in zip(ax.get_yticklabels(), wide.index, strict=False):
        tick.set_color(LR_COLORS.get(lr, TEXT))
    ax.invert_yaxis()
    ax.set_xlabel("delta target correlation")
    ax.set_title("Scale/context gate gain", pad=4)
    clean_axis(ax, "x")


def plot_target_lift(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support[support["LR"].isin(SELECTED_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("top_receiver_target_lift", ascending=True)
    y = np.arange(len(dat))
    colors = [LR_COLORS.get(lr, TAUPE) for lr in dat["LR"]]
    ax.barh(y, dat["top_receiver_target_lift"], color=colors, alpha=0.84, height=0.56, edgecolor=INK, linewidth=0.28)
    ax.axvline(0, color=INK, lw=0.50)
    ax.set_yticks(y)
    ax.set_yticklabels(dat["LR"].tolist(), fontsize=5.1)
    ax.set_xlabel("target-response lift")
    ax.set_title("Receiver target support", pad=4)
    clean_axis(ax, "x")


def plot_spatial_pair(fig: plt.Figure, outer_ax: plt.Axes, cells: pd.DataFrame, hotspots: pd.DataFrame) -> None:
    outer_ax.set_axis_off()
    outer_ax.set_title("Context-specific receiver activity", pad=4, fontsize=7.2)
    plot_lrs = ["CXCL12-CXCR4", "CXCL10-CXCR3", "TGFB1-TGFBR2"]
    for i, lr in enumerate(plot_lrs):
        ax = outer_ax.inset_axes([0.02 + i * 0.325, 0.08, 0.29, 0.82])
        sub = hotspots[hotspots["LR"].eq(lr)].copy()
        frame = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].merge(
            sub[["cell_id", "receiver_activity", "target_score", "is_top_receiver"]],
            on="cell_id",
            how="left",
        )
        frame["receiver_activity"] = frame["receiver_activity"].fillna(0.0)
        frame["activity_norm"] = robust_norm(frame["receiver_activity"])
        ax.scatter(frame["x_plot"], frame["y_plot"], s=1.45, color="#D8D3CC", alpha=0.28, linewidths=0, rasterized=True)
        active = frame[frame["activity_norm"].ge(0.04)].sort_values("activity_norm")
        ax.scatter(active["x_plot"], active["y_plot"], s=2.5, c=active["activity_norm"], cmap=ACTIVITY_CMAP, norm=Normalize(0, 1), alpha=0.90, linewidths=0, rasterized=True)
        top = frame[frame["is_top_receiver"].fillna(False)]
        ax.scatter(top["x_plot"], top["y_plot"], s=8.5, facecolors="none", edgecolors=LR_COLORS.get(lr, RED), linewidths=0.30, alpha=0.52)
        ax.set_title(lr, fontsize=6.2, pad=1.0, color=LR_COLORS.get(lr, TEXT))
        spatial_axis(ax, frame)
    cax = outer_ax.inset_axes([0.75, -0.03, 0.20, 0.035])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=ACTIVITY_CMAP, aspect="auto")
    cax.set_xticks([0, 127]); cax.set_xticklabels(["low", "high"]); cax.set_yticks([])
    cax.tick_params(length=0, labelsize=5.0, pad=0.8)
    for s in cax.spines.values():
        s.set_visible(False)


def plot_story_score(ax: plt.Axes, ranking: pd.DataFrame) -> None:
    dat = ranking[ranking["LR"].isin(SELECTED_LRS)].sort_values("story_score", ascending=True)
    y = np.arange(len(dat))
    colors = [LR_COLORS.get(lr, TAUPE) for lr in dat["LR"]]
    ax.barh(y, dat["story_score"], color=colors, alpha=0.86, height=0.58, edgecolor=INK, linewidth=0.28)
    ax.set_yticks(y)
    ax.set_yticklabels(dat["LR"].tolist(), fontsize=5.1)
    ax.set_xlabel("integrated support score")
    ax.set_title("Prioritized context LR axes", pad=4)
    clean_axis(ax, "x")


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(8.75, 5.10), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        12,
        left=0.065,
        right=0.985,
        bottom=0.110,
        top=0.920,
        wspace=0.94,
        hspace=0.58,
        height_ratios=[0.98, 1.08],
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[0, 6:9])
    ax_d = fig.add_subplot(gs[0, 9:12])
    ax_e = fig.add_subplot(gs[1, 0:8])
    ax_f = fig.add_subplot(gs[1, 8:12])

    plot_context_map(ax_a, source["cells"])
    plot_scale_heatmap(ax_b, source["lr_summary"])
    plot_delta(ax_c, source["scale_vs_noscale"])
    plot_target_lift(ax_d, source["support"])
    plot_spatial_pair(fig, ax_e, source["cells"], source["hotspots"])
    plot_story_score(ax_f, source["ranking"])

    for label, ax in zip(list("abcdef"), [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        panel_label(fig, ax, label)
    return fig


def export(fig: plt.Figure, stem: Path, dpi: int = 850) -> None:
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    for stale in SOURCE_OUT.glob("*.csv"):
        stale.unlink()
    setup_style()
    source = load_source()
    for name, table in source.items():
        table.to_csv(SOURCE_OUT / f"{name}.csv", index=False)
    fig = plot_figure(source)
    export(fig, OUT / "cosmx_nsclc_article_style_additions")
    (OUT / "README.md").write_text(
        "CosMx NSCLC article-style additions for ScaleComm scale/context evidence. "
        "Panels summarize learned LR scale profiles, scale-aware target-correlation gain, receiver target support, and spatial receiver activity.\n",
        encoding="utf-8",
    )
    print(OUT / "cosmx_nsclc_article_style_additions.pdf")


if __name__ == "__main__":
    main()
