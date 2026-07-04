#!/usr/bin/env python
"""CellNEST-style single-axis Slide-seqV2 hippocampus story for Cxcl12-Cxcr4."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v5_crisp_points"

LR_PAIR = "Cxcl12-Cxcr4"
LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
RESPONSE_GENES = ["Cxcr4", "Gfap", "Aqp4", "Slc1a2", "Pdgfra"]

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
TISSUE = "#D6D0C6"
SPATIAL = "#2F78A8"
RESPONSE = "#C86B53"
ASTRO = "#93CC82"
BG = "#FFFFFF"
BLACK = "#111111"

CELL_COLORS = {
    "astrocyte": "#93CC82",
    "oligodendrocyte": "#B9C1D1",
    "CA1_neuron": "#B9E5FA",
    "CA3_neuron": "#8ABEDC",
    "dentate_neuron": "#F3D78A",
    "interneuron": "#FBC9C4",
    "endothelial": "#9FC7BD",
    "microglia_macrophage": "#CAB2D6",
    "polydendrocyte": "#B9C1D1",
    "entorhinal_neuron": "#D8D2C8",
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
    lr_summary = pd.read_csv(RESULT_ROOT / "lr_ranking_summary.csv")
    merged = pd.read_csv(RESULT_ROOT / "merged_predictions.csv")

    source = build_source(cells, expr, lr_summary, merged)
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


def build_source(
    cells: pd.DataFrame,
    expr: pd.DataFrame,
    lr_summary: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    annotated = cells[cells["cell_type"].ne("unannotated")].copy()

    lr_rank = lr_summary.copy()
    lr_rank["lr_pair"] = lr_rank["ligand"] + "-" + lr_rank["receptor"]
    lr_rank = lr_rank[~lr_rank["mechanism"].astype(str).str.startswith("negative")].copy()
    lr_rank = lr_rank.sort_values("mean_score", ascending=False).head(8).copy()
    lr_rank["highlight"] = lr_rank["lr_pair"].eq(LR_PAIR)

    sub = merged[(merged["ligand"].eq(LIGAND)) & (merged["receptor"].eq(RECEPTOR))].copy()
    cutoff = float(sub["score"].quantile(0.985))
    top_edges = sub[sub["score"] >= cutoff].copy()
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

    receiver = (
        top_edges.groupby("receiver_id", as_index=False)
        .agg(receiver_score=("score", "mean"), n_edges=("score", "size"))
        .sort_values("receiver_score", ascending=False)
    )
    receiver = annotated[["cell_id", "x_plot", "y_plot", "cell_type"]].merge(
        receiver,
        left_on="cell_id",
        right_on="receiver_id",
        how="left",
    )
    receiver["receiver_score"] = receiver["receiver_score"].fillna(0.0)
    receiver["n_edges"] = receiver["n_edges"].fillna(0).astype(int)
    receiver["is_hotspot"] = receiver["receiver_score"].gt(0)
    if receiver["receiver_score"].max() > 0:
        receiver["receiver_scaled"] = receiver["receiver_score"] / receiver["receiver_score"].max()
    else:
        receiver["receiver_scaled"] = 0.0

    genes = [gene for gene in RESPONSE_GENES if gene in expr.columns]
    response = annotated[["cell_id", "x_plot", "y_plot", "cell_type"]].merge(expr[["cell_id"] + genes], on="cell_id", how="left")
    values = response[genes].fillna(0).astype(float)
    z = (values - values.mean(axis=0)) / (values.std(axis=0, ddof=0) + 1e-9)
    response["response_score"] = z.mean(axis=1)
    lo, hi = np.nanpercentile(response["response_score"], [1, 99])
    response["response_scaled"] = np.clip((response["response_score"] - lo) / (hi - lo + 1e-9), 0, 1)
    response = response.merge(receiver[["cell_id", "is_hotspot", "receiver_score"]], on="cell_id", how="left")
    response["is_hotspot"] = response["is_hotspot"].fillna(False).astype(bool)
    response["receiver_score"] = response["receiver_score"].fillna(0.0)

    # Use the densest hotspot region for a local communication zoom.
    hot = receiver[receiver["is_hotspot"]].copy()
    if hot.empty:
        xmin, xmax = annotated["x_plot"].quantile([0.35, 0.65])
        ymin, ymax = annotated["y_plot"].quantile([0.35, 0.65])
    else:
        center = hot.sort_values("receiver_score", ascending=False).head(80)
        cx, cy = float(center["x_plot"].median()), float(center["y_plot"].median())
        span = max(
            float(annotated["x_plot"].max() - annotated["x_plot"].min()),
            float(annotated["y_plot"].max() - annotated["y_plot"].min()),
        )
        half = span * 0.16
        xmin, xmax = cx - half, cx + half
        ymin, ymax = cy - half * 0.78, cy + half * 0.78

    zoom_box = pd.DataFrame([{"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}])
    local_edges = top_edges[
        top_edges["receiver_x_plot"].between(xmin, xmax)
        & top_edges["receiver_y_plot"].between(ymin, ymax)
        & top_edges["sender_x_plot"].between(xmin - (xmax - xmin) * 0.4, xmax + (xmax - xmin) * 0.4)
        & top_edges["sender_y_plot"].between(ymin - (ymax - ymin) * 0.4, ymax + (ymax - ymin) * 0.4)
    ].copy()
    local_edges = local_edges.sort_values(["score", "candidate_id"], ascending=[False, True]).head(70)

    validation = response[["cell_id", "is_hotspot", "response_score"]].copy()
    validation["group"] = np.where(validation["is_hotspot"], "predicted receiver", "background")

    return {
        "panel_a_cells": annotated,
        "panel_b_lr_rank": lr_rank,
        "panel_c_receivers": receiver,
        "panel_d_local_edges": local_edges,
        "panel_d_zoom_box": zoom_box,
        "panel_e_response": response,
        "panel_f_validation": validation,
    }


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 5.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        bottom=0.078,
        top=0.955,
        wspace=0.25,
        hspace=0.42,
        height_ratios=[1.0, 0.95],
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[1, 0]),
        "e": fig.add_subplot(gs[1, 1]),
        "f": fig.add_subplot(gs[1, 2]),
    }
    plot_cell_map(axes["a"], source["panel_a_cells"])
    plot_lr_rank(axes["b"], source["panel_b_lr_rank"])
    plot_receiver_hotspot(axes["c"], source["panel_c_receivers"], source["panel_d_zoom_box"])
    plot_local_edges(axes["d"], source["panel_a_cells"], source["panel_d_local_edges"], source["panel_d_zoom_box"])
    plot_response_map(axes["e"], source["panel_e_response"])
    plot_validation(axes["f"], source["panel_f_validation"])
    for label, ax in axes.items():
        panel_label(fig, ax, label)
    return fig


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    order = cells["cell_type"].value_counts().index.tolist()[:10]
    for cell_type in order:
        sub = cells[cells["cell_type"].eq(cell_type)]
        ax.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=3.8,
            c=CELL_COLORS.get(cell_type, TISSUE),
            alpha=1.0,
            linewidths=0,
            edgecolors="none",
            antialiaseds=False,
            rasterized=False,
            label=pretty_cell(cell_type),
        )
    set_title(ax, "Slide-seqV2 hippocampus")
    add_spatial_style(ax, cells)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.085),
        ncol=3,
        fontsize=4.5,
        handletextpad=0.22,
        columnspacing=0.5,
        labelspacing=0.25,
        markerscale=1.45,
    )


def plot_lr_rank(ax: plt.Axes, rank: pd.DataFrame) -> None:
    table = rank.sort_values("mean_score", ascending=True).copy()
    y = np.arange(len(table))
    colors = [SPATIAL if flag else "#D8CDBD" for flag in table["highlight"]]
    edges = [BLACK if flag else "#B6AEA5" for flag in table["highlight"]]
    ax.barh(y, table["mean_score"], height=0.62, color=colors, edgecolor=edges, linewidth=0.35)
    labels = [label.replace("-", "\n") for label in table["lr_pair"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.9)
    for tick, flag in zip(ax.get_yticklabels(), table["highlight"]):
        if flag:
            tick.set_color(SPATIAL)
            tick.set_fontweight("bold")
    xmax = float(table["mean_score"].max()) * 1.13
    ax.set_xlim(0, xmax)
    for yi, row in enumerate(table.itertuples(index=False)):
        if bool(row.highlight):
            ax.scatter(
                float(row.mean_score) + xmax * 0.035,
                yi,
                marker="v",
                s=20,
                facecolors=BG,
                edgecolors=SPATIAL,
                linewidths=0.75,
                clip_on=False,
                zorder=5,
            )
            ax.text(
                float(row.mean_score) + xmax * 0.060,
                yi,
                "selected",
                ha="left",
                va="center",
                fontsize=5.0,
                color=SPATIAL,
                clip_on=False,
            )
    ax.set_xlabel("mean SpatialSelf activity", fontsize=5.6, color=MUTED, labelpad=1.0)
    set_title(ax, "Prioritized LR axis")
    style_quant_axis(ax)
    ax.tick_params(axis="y", length=0, pad=1)


def plot_receiver_hotspot(ax: plt.Axes, receiver: pd.DataFrame, zoom_box: pd.DataFrame) -> None:
    ax.scatter(receiver["x_plot"], receiver["y_plot"], s=2.4, c=TISSUE, alpha=0.34, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    active = receiver[receiver["is_hotspot"]].sort_values("receiver_scaled")
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=7.2,
        c=active["receiver_scaled"],
        cmap=LinearSegmentedColormap.from_list("hot", ["#EDE7D8", "#D6B06A", SPATIAL]),
        vmin=0,
        vmax=1,
        alpha=0.98,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
    )
    box = zoom_box.iloc[0]
    ax.add_patch(
        Rectangle(
            (box.xmin, box.ymin),
            box.xmax - box.xmin,
            box.ymax - box.ymin,
            fill=False,
            ec=BLACK,
            lw=0.55,
            alpha=0.72,
        )
    )
    set_title(ax, f"{LR_PAIR} hotspots")
    add_spatial_style(ax, receiver)


def plot_local_edges(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame, zoom_box: pd.DataFrame) -> None:
    box = zoom_box.iloc[0]
    local_cells = cells[cells["x_plot"].between(box.xmin, box.xmax) & cells["y_plot"].between(box.ymin, box.ymax)].copy()
    ax.scatter(local_cells["x_plot"], local_cells["y_plot"], s=3.6, c=TISSUE, alpha=0.30, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    if not edges.empty:
        vmax = max(float(edges["score"].max()), 1e-9)
        for row in edges.head(24).itertuples(index=False):
            val = float(row.score) / vmax
            patch = FancyArrowPatch(
                (float(row.sender_x_plot), float(row.sender_y_plot)),
                (float(row.receiver_x_plot), float(row.receiver_y_plot)),
                arrowstyle="-|>",
                mutation_scale=6.8,
                linewidth=0.36 + 0.26 * val,
                color=SPATIAL,
                alpha=0.30 + 0.22 * val,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
            )
            ax.add_patch(patch)
        send = edges[["sender_x_plot", "sender_y_plot"]].drop_duplicates()
        rec = edges[["receiver_x_plot", "receiver_y_plot"]].drop_duplicates()
        ax.scatter(send["sender_x_plot"], send["sender_y_plot"], s=8.4, c=ASTRO, edgecolors=BLACK, linewidths=0.12, alpha=0.92, zorder=6)
        ax.scatter(rec["receiver_x_plot"], rec["receiver_y_plot"], s=8.8, c=SPATIAL, edgecolors=BLACK, linewidths=0.12, alpha=0.94, zorder=7)
        handles = [
            mpl.lines.Line2D([], [], marker="o", ls="", color=ASTRO, markeredgecolor=BLACK, markeredgewidth=0.2, markersize=3.6, label="sender"),
            mpl.lines.Line2D([], [], marker="o", ls="", color=SPATIAL, markeredgecolor=BLACK, markeredgewidth=0.2, markersize=3.6, label="receiver"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=4.8, handletextpad=0.2, borderaxespad=0.1)
    ax.set_xlim(box.xmin, box.xmax)
    ax.set_ylim(box.ymin, box.ymax)
    set_title(ax, "Local communication edges")
    add_spatial_style(ax, local_cells, preserve_limits=True)


def plot_response_map(ax: plt.Axes, response: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("resp", ["#F2EFE8", "#E6C06A", RESPONSE])
    frame = response.sort_values(["response_scaled", "cell_id"], ascending=[True, True]).copy()
    values = frame["response_scaled"].astype(float)
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.9, c=values, cmap=cmap, vmin=0, vmax=1, alpha=0.96, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    hot = response[response["is_hotspot"]]
    ax.scatter(hot["x_plot"], hot["y_plot"], s=8.4, facecolors="none", edgecolors=SPATIAL, alpha=0.48, linewidths=0.22, rasterized=False)
    set_title(ax, "Receiver response")
    add_spatial_style(ax, response)
    cax = ax.inset_axes([0.13, -0.115, 0.52, 0.042])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation="horizontal")
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.42)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.8, length=1.8, width=0.42, pad=1)


def plot_validation(ax: plt.Axes, validation: pd.DataFrame) -> None:
    rng = np.random.default_rng(19)
    groups = ["background", "predicted receiver"]
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
        medianprops={"color": BLACK, "linewidth": 0.72},
        whiskerprops={"color": BLACK, "linewidth": 0.56},
        capprops={"color": BLACK, "linewidth": 0.56},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=mpl.colors.to_rgba(color, 0.28), edgecolor=BLACK, linewidth=0.58)
    for idx, vals in enumerate(data):
        sample = vals if len(vals) <= 180 else rng.choice(vals, 180, replace=False)
        x = idx + rng.normal(0, 0.045, len(sample))
        ax.scatter(x, sample, s=7.5, c=colors[idx], alpha=0.48 if idx == 0 else 0.72, edgecolors=BLACK, linewidths=0.10, zorder=3)
        ax.scatter([idx], [np.nanmean(vals)], s=18, marker="D", c=colors[idx], edgecolors=BLACK, linewidths=0.28, zorder=5)
    bg, hot = data
    delta = float(np.nanmean(hot) - np.nanmean(bg))
    ax.text(0.5, 0.96, f"Delta z = {delta:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=SPATIAL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nbeads", "hotspot\nreceivers"], fontsize=5.5)
    ax.set_ylabel("response signature (z-score)", fontsize=5.9, color=MUTED)
    set_title(ax, "Response support")
    style_quant_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.42)


def add_spatial_style(ax: plt.Axes, data: pd.DataFrame, preserve_limits: bool = False) -> None:
    ax.set_aspect("equal")
    if not preserve_limits and len(data):
        xspan = float(data["x_plot"].max() - data["x_plot"].min())
        yspan = float(data["y_plot"].max() - data["y_plot"].min())
        pad = max(xspan, yspan) * 0.025
        ax.set_xlim(float(data["x_plot"].min()) - pad, float(data["x_plot"].max()) + pad)
        ax.set_ylim(float(data["y_plot"].min()) - pad, float(data["y_plot"].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#7A716A")
        ax.spines[side].set_linewidth(0.58)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def style_quant_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.62)
    ax.spines["bottom"].set_linewidth(0.62)
    ax.grid(axis="x", color=GRID, lw=0.42)
    ax.tick_params(axis="both", labelsize=5.4, length=2.1, pad=1.2)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=7.0, pad=2.0, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.035, box.y1 + 0.018, label, ha="left", va="bottom", fontsize=9.0, fontweight="bold", color=INK)


def pretty_cell(label: str) -> str:
    return label.replace("_neuron", "").replace("_", " ")


if __name__ == "__main__":
    main()
