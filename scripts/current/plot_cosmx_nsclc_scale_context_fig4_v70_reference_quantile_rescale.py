#!/usr/bin/env python
"""Plot a focused CosMx NSCLC figure for LR-specific scale/context validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables"
OUT = ROOT / "outputs/manuscript_experiment_figures_unified_v14/fig4_cosmx_nsclc"
ARTICLE_EXTRA_SOURCE = ROOT / "outputs/article_style_figure_additions/high_priority_remaining/source_data"

SPECTRUM_LRS = ["CXCL10-CXCR3", "SPP1-CD44", "TGFB1-TGFBR2", "VEGFA-KDR", "CXCL12-CXCR4"]
SELECTED_LRS = ["SPP1-CD44", "CXCL10-CXCR3", "CXCL12-CXCR4"]
FUNCTIONAL_LRS = ["SPP1-CD44", "CXCL10-CXCR3", "CXCL12-CXCR4", "CXCL9-CXCR3", "CCL5-CCR5", "VEGFA-KDR", "CSF1-CSF1R"]
FOCUS_LR = "CXCL10-CXCR3"
COL = {
    "ink": "#263137",
    "text": "#384247",
    "muted": "#707A82",
    "grid": "#E8ECEF",
    "bg": "#FFFFFF",
    "cell_bg": "#DDE2E9",
    "tumor": "#FBC9C4",
    "immune": "#B9E5FA",
    "stroma": "#F3D78A",
    "vascular": "#93CC82",
    "other": "#B9C1D1",
    "cream": "#FFEEBC",
    "green_dark": "#4F9B59",
    "blue_dark": "#5BAFD0",
    "pink_dark": "#C87370",
    "gold_dark": "#B98F2E",
    "slate_dark": "#76849A",
    "ours": "#4F9B59",
    "commot": "#5BAFD0",
    "cellnest": "#76849A",
    "cellchat": "#B98F2E",
    "spatalk": "#C87370",
}

LR_COLORS = {
    "CCL5-CCR5": COL["blue_dark"],
    "CXCL9-CXCR3": COL["blue_dark"],
    "CXCL10-CXCR3": COL["pink_dark"],
    "CD274-PDCD1": COL["slate_dark"],
    "SPP1-CD44": COL["gold_dark"],
    "CSF1-CSF1R": COL["slate_dark"],
    "TGFB1-TGFBR2": COL["pink_dark"],
    "VEGFA-KDR": COL["green_dark"],
    "CXCL12-CXCR4": COL["green_dark"],
    "KRT8-ACTA2": "#9EA7B6",
    "PTPRC-EPCAM": "#9EA7B6",
}

LR_FILLS = {
    "CCL5-CCR5": COL["immune"],
    "CXCL9-CXCR3": COL["immune"],
    "CXCL10-CXCR3": COL["tumor"],
    "CD274-PDCD1": COL["other"],
    "SPP1-CD44": COL["stroma"],
    "CSF1-CSF1R": COL["other"],
    "TGFB1-TGFBR2": COL["tumor"],
    "VEGFA-KDR": COL["vascular"],
    "CXCL12-CXCR4": COL["vascular"],
}

CELL_COLORS = {
    "tumor_proxy": "#C87873",
    "t_cell_proxy": "#4E8BA8",
    "immune_proxy": "#8FB3C2",
    "myeloid_proxy": "#7D8799",
    "stromal_proxy": "#B79A51",
    "endothelial_proxy": "#5F9B6B",
    "other_proxy": "#D4D9DF",
}

ACTIVITY_CMAP = LinearSegmentedColormap.from_list(
    "activity_cmap",
    [COL["cream"], COL["stroma"], COL["green_dark"]],
)

TARGET_RESPONSE_CMAP = LinearSegmentedColormap.from_list(
    "target_response_heatmap",
    ["#F8F5EC", "#E6C06A", "#D78656", "#9A3A35"],
)

REGION_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "region_heat_reference",
    ["#FAF8F1", "#E8C56F", "#D4845C", "#9A3A35"],
)

CELLNEST_SPOT_CMAP = LinearSegmentedColormap.from_list(
    "cellnest_spot_heat",
    ["#B5B2BD", "#8472A6", "#B95D9A", "#E76F83", "#F6A37B", "#FFF2B3"],
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    source = load_source()
    fig = plot_figure(source)
    stem = OUT / "fig4_cosmx_nsclc_scale_context_v14_reference_quantile_rescale"
    export(fig, stem)
    g_fig = plot_g_panel_standalone(source)
    g_stem = OUT / "fig4_cosmx_nsclc_scale_context_v14_panel_g_standalone"
    export(g_fig, g_stem)
    print(f"Wrote {stem}.png/.pdf/.svg/.tiff")


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 6.8,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "axes.labelcolor": COL["text"],
            "text.color": COL["text"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_source() -> dict[str, pd.DataFrame]:
    return {
        "cells": pd.read_csv(SOURCE / "cosmx_cells_for_context_map.csv"),
        "lr_summary": pd.read_csv(SOURCE / "cosmx_lr_scale_context_summary.csv"),
        "support": pd.read_csv(SOURCE / "cosmx_receiver_target_support.csv"),
        "distance": pd.read_csv(SOURCE / "cosmx_edge_distance_summary.csv"),
        "hotspots": pd.read_csv(SOURCE / "cosmx_hotspot_receiver_cells.csv"),
        "scored": pd.read_csv(SOURCE / "cosmx_spatialself_scored_candidates.csv"),
        "scale_vs_noscale": pd.read_csv(SOURCE / "cosmx_scale_vs_noscale_target_support.csv"),
        "receiver_burden": pd.read_csv(ARTICLE_EXTRA_SOURCE / "realdata_cosmx_receiver_celltype_burden.csv")
        if (ARTICLE_EXTRA_SOURCE / "realdata_cosmx_receiver_celltype_burden.csv").exists()
        else pd.DataFrame(),
    }


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(8.25, 9.15))
    gs = fig.add_gridspec(
        4,
        4,
        width_ratios=[1.22, 1.02, 1.02, 1.08],
        height_ratios=[0.86, 0.78, 1.78, 0.72],
        hspace=0.24,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[1, 3])

    plot_context_map(ax_a, source["cells"])
    plot_scale_distribution(ax_b, source["lr_summary"])
    plot_selected_scale_fingerprints(ax_c, source["lr_summary"])
    plot_cosmx_target_alignment_delta(ax_d, source["scale_vs_noscale"])
    plot_receiver_response_screen(ax_e, source["support"])
    plot_selected_target_lift(ax_f, source["support"], source["hotspots"])

    spatial_gs = gs[2, :].subgridspec(2, len(SELECTED_LRS), hspace=0.035, wspace=0.050)
    g_axes: list[plt.Axes] = []
    for col_idx, lr in enumerate(SELECTED_LRS):
        roi = compute_top_fraction_roi(source["scored"], source["cells"], source["hotspots"], lr, fraction=0.20)
        ax_comm = fig.add_subplot(spatial_gs[0, col_idx])
        plot_top_fraction_communication_map(
            ax_comm,
            source["scored"],
            source["cells"],
            source["hotspots"],
            lr,
            fraction=0.20,
            roi=roi,
            show_row_label=col_idx == 0,
        )
        ax_resp = fig.add_subplot(spatial_gs[1, col_idx])
        plot_target_response_heatmap(
            ax_resp,
            source["hotspots"],
            lr,
            roi=roi,
            show_row_label=col_idx == 0,
        )
        g_axes.extend([ax_comm, ax_resp])
    add_spot_heatmap_colorbar(fig, g_axes, label="relative activity")

    ax_h = fig.add_subplot(gs[3, 0:2])
    ax_i = fig.add_subplot(gs[3, 2:4])
    plot_receiver_celltype_burden_heatmap(ax_h, source["receiver_burden"])
    plot_scale_vs_noscale_pairs(ax_i, source["scale_vs_noscale"])

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d, "d", x=-0.075, y=1.075)
    add_panel_label(ax_e, "e")
    add_panel_label(ax_f, "f")
    add_panel_label(g_axes[0], "g", x=-0.10, y=1.025)
    add_panel_label(ax_h, "h", x=-0.055, y=1.055)
    add_panel_label(ax_i, "i", x=-0.055, y=1.055)

    return fig


def plot_g_panel_standalone(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(8.25, 4.25))
    gs = fig.add_gridspec(2, len(SELECTED_LRS), hspace=0.035, wspace=0.050)
    axes: list[plt.Axes] = []
    for col_idx, lr in enumerate(SELECTED_LRS):
        roi = compute_top_fraction_roi(source["scored"], source["cells"], source["hotspots"], lr, fraction=0.20)
        ax_comm = fig.add_subplot(gs[0, col_idx])
        plot_top_fraction_communication_map(
            ax_comm,
            source["scored"],
            source["cells"],
            source["hotspots"],
            lr,
            fraction=0.20,
            roi=roi,
            show_row_label=col_idx == 0,
        )
        ax_resp = fig.add_subplot(gs[1, col_idx])
        plot_target_response_heatmap(
            ax_resp,
            source["hotspots"],
            lr,
            roi=roi,
            show_row_label=col_idx == 0,
        )
        axes.extend([ax_comm, ax_resp])
    add_spot_heatmap_colorbar(fig, axes, label="relative activity")
    add_panel_label(axes[0], "g", x=-0.090, y=1.105)
    return fig


def plot_context_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    dat = cells.copy()
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    order = ["other_proxy", "stromal_proxy", "endothelial_proxy", "myeloid_proxy", "t_cell_proxy", "tumor_proxy"]
    label_map = {
        "tumor_proxy": "tumor",
        "t_cell_proxy": "T cell",
        "myeloid_proxy": "myeloid",
        "stromal_proxy": "stroma",
        "endothelial_proxy": "endo.",
        "other_proxy": "other",
    }
    for cell_type in order:
        group = dat[dat["cell_type"].eq(cell_type)]
        if group.empty:
            continue
        color = CELL_COLORS.get(cell_type, COL["other"])
        size = 1.55 if cell_type == "other_proxy" else 2.08
        alpha = 0.25 if cell_type == "other_proxy" else 0.76
        if cell_type in {"tumor_proxy", "t_cell_proxy"}:
            alpha = 0.88
            size = 2.30
        ax.scatter(group["x_plot"], group["y_plot"], s=size, c=color, alpha=alpha, linewidths=0, rasterized=False)
    ax.set_aspect("equal")
    ax.margins(0.0)
    strip_spatial_axis(ax)
    ax.set_title("Cell context", fontsize=7.4, pad=1.8, color=COL["ink"])
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CELL_COLORS[k], markeredgecolor="none", markersize=3.6, label=label)
        for k, label in [
            ("tumor_proxy", "tumor"),
            ("t_cell_proxy", "T cell"),
            ("myeloid_proxy", "myeloid"),
            ("stromal_proxy", "stroma"),
            ("endothelial_proxy", "endo."),
        ]
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.012, 0.50),
        ncol=1,
        fontsize=5.05,
        handletextpad=0.30,
        labelspacing=0.32,
        borderaxespad=0.0,
        frameon=False,
        labelcolor=COL["text"],
    )


def plot_scale_distribution(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    scale_cols = [f"scale_{i}" for i in range(6)]
    dat = lr_summary.dropna(subset=["LR"]).copy()
    dat["is_control"] = dat["mechanism"].fillna("").str.contains("negative_control", case=False)
    dat = dat[~dat["is_control"]].copy()
    raw = dat[scale_cols].to_numpy(float)
    raw = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-8, None)
    dat["scale_center"] = np.dot(raw, np.arange(len(scale_cols)))
    dat = dat.sort_values(["is_control", "scale_center", "LR"], ascending=[True, True, True]).reset_index(drop=True)

    values = dat[scale_cols].to_numpy(float)
    values = values / np.clip(values.sum(axis=1, keepdims=True), 1e-8, None)
    cmap = LinearSegmentedColormap.from_list(
        "scale_weight_reference_v7",
        ["#FAF8F1", "#E8C56F", "#D4845C", "#9A3A35"],
    )
    vmax = max(0.75, float(np.nanmax(values)))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_box_aspect(0.58)

    ax.set_xticks(np.arange(len(scale_cols)))
    ax.set_xticklabels([f"s{i + 1}" for i in range(len(scale_cols))], fontsize=5.6)
    ax.set_yticks(np.arange(len(dat)))
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0, pad=1.4)
    ax.set_xticks(np.arange(-0.5, len(scale_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dat), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.72)
    ax.tick_params(which="minor", bottom=False, left=False)

    short_labels = {
        "CD274-PDCD1": "CD274-PDCD1",
        "CCL5-CCR5": "CCL5-CCR5",
        "CXCL9-CXCR3": "CXCL9-CXCR3",
        "CXCL10-CXCR3": "CXCL10-CXCR3",
        "SPP1-CD44": "SPP1-CD44",
        "CSF1-CSF1R": "CSF1-CSF1R",
        "TGFB1-TGFBR2": "TGFB1-TGFBR2",
        "VEGFA-KDR": "VEGFA-KDR",
        "CXCL12-CXCR4": "CXCL12-CXCR4",
        "KRT8-ACTA2": "KRT8-ACTA2",
        "PTPRC-EPCAM": "PTPRC-EPCAM",
    }
    for yi, row in dat.iterrows():
        lr = row["LR"]
        is_selected = lr in SELECTED_LRS
        color = LR_COLORS.get(lr, COL["ink"]) if is_selected else COL["text"]
        ax.text(
            -0.70,
            yi,
            short_labels.get(lr, lr),
            ha="right",
            va="center",
            fontsize=5.5,
            color=color,
            fontweight="bold" if is_selected else "normal",
            clip_on=False,
        )
        if lr in SELECTED_LRS:
            xi = int(np.nanargmax(values[yi]))
            val = float(values[yi, xi])
            text_color = "white" if val > 0.52 * vmax else "#1F2529"
            ax.text(
                xi,
                yi,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=4.2,
                color=text_color,
                fontweight="bold",
                alpha=0.96,
            )

    ax.text(0.0, 1.075, "Scale atlas", transform=ax.transAxes, fontsize=7.5, color=COL["ink"], ha="left", va="bottom")
    ax.text(0.00, 1.006, "local", transform=ax.transAxes, fontsize=5.4, color=COL["muted"], ha="left", va="bottom")
    ax.text(1.00, 1.006, "broad", transform=ax.transAxes, fontsize=5.4, color=COL["muted"], ha="right", va="bottom")
    ax.set_xlabel("scale bin", fontsize=6.2, labelpad=1.5)

    cax = ax.inset_axes([1.022, 0.17, 0.022, 0.62])
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.42)
    cb.outline.set_edgecolor("#6B625B")
    cb.set_ticks([0, 0.5 * vmax, vmax])
    cb.set_ticklabels(["0", f"{0.5 * vmax:.2f}", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=5.0, width=0.0, length=0.0, pad=1.2, colors=COL["muted"])
    cb.set_label("weight", fontsize=5.5, labelpad=2.0, color=COL["muted"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#6B625B")
        spine.set_linewidth(0.55)
    for spine in cax.spines.values():
        spine.set_visible(False)


def plot_expected_scale_summary(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[lr_summary["LR"].isin(SPECTRUM_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SPECTRUM_LRS)})
    dat = dat.sort_values("expected_scale")
    if dat.empty:
        ax.set_axis_off()
        return
    y = np.arange(len(dat))
    short = {
        "CCL5-CCR5": "CCL5",
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    colors = [LR_COLORS.get(lr, COL["muted"]) for lr in dat["LR"]]
    ax.barh(y, dat["expected_scale"], height=0.54, color=colors, edgecolor="none", alpha=0.92)
    ax.set_xlim(0, 5)
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(lr, lr) for lr in dat["LR"]], fontsize=5.6)
    ax.set_xticks([0, 2.5, 5.0])
    ax.set_xlabel("expected scale", fontsize=6.6, labelpad=1.4)
    ax.set_title("Scale order", fontsize=7.4, pad=2.5, color=COL["ink"])
    ax.axvspan(0, 1.5, color="#C8C3B8", alpha=0.13, lw=0, zorder=0)
    ax.axvspan(1.5, 3.5, color="#D7835C", alpha=0.10, lw=0, zorder=0)
    ax.axvspan(3.5, 5, color="#237A78", alpha=0.10, lw=0, zorder=0)
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.35, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.2)


def plot_top_edge_distance_distribution(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    dat = candidates[candidates["LR"].isin(SELECTED_LRS)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    rows: list[pd.DataFrame] = []
    for lr in SELECTED_LRS:
        group = dat[dat["LR"].eq(lr)].copy()
        if group.empty:
            continue
        cutoff = group["score"].quantile(0.985)
        top = group[group["score"].ge(cutoff)].copy()
        top["short"] = {"CCL5-CCR5": "CCL5", "TGFB1-TGFBR2": "TGFB1", "CXCL12-CXCR4": "CXCL12"}.get(lr, lr)
        rows.append(top)
    if not rows:
        ax.set_axis_off()
        return
    top_dat = pd.concat(rows, ignore_index=True)
    order = ["CCL5", "TGFB1", "CXCL12"]
    positions = np.arange(len(order))

    for i, short in enumerate(order):
        lr = {"CCL5": "CCL5-CCR5", "TGFB1": "TGFB1-TGFBR2", "CXCL12": "CXCL12-CXCR4"}[short]
        vals = top_dat[top_dat["short"].eq(short)]["distance"].to_numpy(float)
        if len(vals) == 0:
            continue
        parts = ax.violinplot([vals], positions=[i], widths=0.62, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(LR_COLORS.get(lr, COL["muted"]))
            body.set_edgecolor("none")
            body.set_alpha(0.33)
        q10, q25, med, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
        ax.plot([i, i], [q10, q90], color=LR_COLORS.get(lr, COL["muted"]), lw=0.85, alpha=0.82, solid_capstyle="round")
        ax.scatter([i], [med], s=18, color=LR_COLORS.get(lr, COL["muted"]), edgecolor="white", linewidth=0.50, zorder=3)
        ax.plot([i - 0.15, i + 0.15], [q25, q25], color=LR_COLORS.get(lr, COL["muted"]), lw=0.8, alpha=0.70)
        ax.plot([i - 0.15, i + 0.15], [q75, q75], color=LR_COLORS.get(lr, COL["muted"]), lw=0.8, alpha=0.70)

    ax.set_xticks(positions)
    ax.set_xticklabels(["CCL5", "TGFB1", "CXCL12"], fontsize=5.8, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel("top-edge distance", fontsize=6.6, labelpad=1.4)
    ax.set_title("Predicted range", fontsize=7.4, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.38, zorder=0)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=1.0)


def plot_selected_distance_footprint(ax: plt.Axes, distance: pd.DataFrame) -> None:
    dat = distance[distance["LR"].isin(SELECTED_LRS)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    xmin = float(dat["bin_mid"].min())
    xmax = float(dat["bin_mid"].max())
    ax.axvspan(xmin, xmin + (xmax - xmin) / 3, color="#C8C3B8", alpha=0.10, lw=0, zorder=0)
    ax.axvspan(xmin + (xmax - xmin) / 3, xmin + 2 * (xmax - xmin) / 3, color="#D7835C", alpha=0.08, lw=0, zorder=0)
    ax.axvspan(xmin + 2 * (xmax - xmin) / 3, xmax, color="#237A78", alpha=0.10, lw=0, zorder=0)
    for lr in SELECTED_LRS:
        group = dat[dat["LR"].eq(lr)].sort_values("bin_mid")
        if group.empty:
            continue
        ax.plot(
            group["bin_mid"],
            group["mean_score_norm_by_lr"],
            color=LR_COLORS.get(lr, COL["muted"]),
            lw=1.35,
            alpha=0.96,
            zorder=3,
        )
        ax.scatter(
            group["bin_mid"],
            group["mean_score_norm_by_lr"],
            s=9,
            color=LR_COLORS.get(lr, COL["muted"]),
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )
    labels = {
        "CCL5-CCR5": (0.04, 0.76, "CCL5"),
        "TGFB1-TGFBR2": (0.18, 0.28, "TGFB1"),
        "CXCL12-CXCR4": (0.60, 0.83, "CXCL12"),
    }
    for lr, (x_ax, y_ax, label) in labels.items():
        ax.text(x_ax, y_ax, label, transform=ax.transAxes, fontsize=5.4, color=LR_COLORS.get(lr, COL["muted"]), ha="left", va="center")
    ax.set_xlim(xmin - 3, xmax + 3)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks([xmin, (xmin + xmax) / 2, xmax])
    ax.set_xticklabels([f"{xmin:.0f}", f"{(xmin + xmax) / 2:.0f}", f"{xmax:.0f}"], fontsize=5.8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("edge distance", fontsize=6.6, labelpad=1.2)
    ax.set_ylabel("relative activity", fontsize=6.6, labelpad=1.2)
    ax.set_title("Distance footprint", fontsize=7.4, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.38, zorder=0)




def plot_selected_scale_fingerprints(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[lr_summary["LR"].isin(SELECTED_LRS)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("order")
    scale_cols = [f"scale_{i}" for i in range(6)]
    x = np.arange(len(scale_cols))
    short = {"SPP1-CD44": "SPP1-CD44", "CXCL10-CXCR3": "CXCL10-CXCR3", "CXCL12-CXCR4": "CXCL12-CXCR4"}
    handles: list[Line2D] = []
    for _, row in dat.iterrows():
        lr = str(row["LR"])
        vals = row[scale_cols].to_numpy(dtype=float)
        vals = vals / max(vals.sum(), 1e-8)
        color = LR_COLORS.get(lr, COL["muted"])
        xs = np.linspace(x.min(), x.max(), 160)
        ys = np.interp(xs, x, vals)
        kernel_x = np.linspace(-2.4, 2.4, 21)
        kernel = np.exp(-0.5 * kernel_x**2)
        kernel = kernel / kernel.sum()
        pad = len(kernel) // 2
        ys = np.convolve(np.pad(ys, pad, mode="edge"), kernel, mode="valid")
        ax.plot(xs, ys, color=color, lw=1.55, alpha=0.98, zorder=3, solid_capstyle="round")
        ax.scatter(x, vals, s=22, color=color, edgecolor=color, linewidth=0.25, zorder=4)
        handles.append(Line2D([0], [0], color=color, lw=1.55, marker="o", markersize=4.0, label=short.get(lr, lr)))

    ax.set_xlim(-0.35, 5.35)
    ax.set_ylim(0, 0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{i + 1}" for i in x], fontsize=5.8)
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlabel("scale bin", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("scale weight", fontsize=6.4, labelpad=1.2)
    ax.set_title("Scale profiles", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color="#DDE2E9", linewidth=0.32, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#15191C")
        spine.set_linewidth(0.72)


def plot_scale_selectivity_summary(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[lr_summary["LR"].isin(SELECTED_LRS)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    scale_cols = [f"scale_{i}" for i in range(6)]
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("order").reset_index(drop=True)

    weights = dat[scale_cols].to_numpy(float)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    scale_index = np.arange(len(scale_cols), dtype=float)
    centers = weights @ scale_index
    entropy = -(weights * np.log(np.clip(weights, 1e-8, None))).sum(axis=1) / np.log(len(scale_cols))
    selectivity = 1.0 - entropy
    peak = weights.max(axis=1)

    y = np.arange(len(dat))
    ax.axvspan(-0.35, 1.5, color=COL["other"], alpha=0.15, lw=0, zorder=0)
    ax.axvspan(1.5, 3.5, color=COL["stroma"], alpha=0.14, lw=0, zorder=0)
    ax.axvspan(3.5, 5.35, color=COL["vascular"], alpha=0.13, lw=0, zorder=0)

    short = {
        "SPP1-CD44": "SPP1\nCD44",
        "CXCL10-CXCR3": "CXCL10\nCXCR3",
        "CXCL12-CXCR4": "CXCL12\nCXCR4",
    }
    for yi, (_, row) in enumerate(dat.iterrows()):
        lr = str(row["LR"])
        color = LR_COLORS.get(lr, COL["muted"])
        vals = weights[yi]
        q_center = centers[yi]
        spread = float(np.sqrt(((scale_index - q_center) ** 2 * vals).sum()))
        ax.plot(
            [max(-0.1, q_center - spread), min(5.1, q_center + spread)],
            [yi, yi],
            color=color,
            lw=2.1,
            alpha=0.28,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [q_center],
            [yi],
            s=48 + 95 * selectivity[yi],
            color=color,
            edgecolor="white",
            linewidth=0.62,
            alpha=0.96,
            zorder=3,
        )
        ax.text(
            min(5.15, q_center + spread + 0.20),
            yi,
            f"peak {peak[yi]:.2f}",
            ha="left",
            va="center",
            fontsize=5.15,
            color=color,
        )

    ax.set_xlim(-0.25, 5.25)
    ax.set_ylim(-0.55, len(dat) - 0.45)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(lr, lr) for lr in dat["LR"]], fontsize=5.7)
    ax.set_xticks([0, 2.5, 5])
    ax.set_xticklabels(["local", "mid", "broad"], fontsize=5.8)
    ax.set_xlabel("dominant context scale", fontsize=6.3, labelpad=1.0)
    ax.set_title("Scale selectivity", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.34, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.3)


def plot_scale_utility_ablation(ax: plt.Axes, ablation: pd.DataFrame) -> None:
    """Show that the scale/context gate improves a ground-truth mechanism benchmark."""
    keep = ["Final gate", "Spatial-null", "Wavelet scale", "Fixed prior", "Contact kernel"]
    dat = ablation[ablation["variant_label"].isin(keep)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    order = {label: i for i, label in enumerate(keep)}
    dat["order"] = dat["variant_label"].map(order)
    dat = dat.sort_values("order").reset_index(drop=True)

    y = np.arange(len(dat))
    base_color = "#CBD2DA"
    colors = [COL["ours"] if label == "Final gate" else base_color for label in dat["variant_label"]]
    edges = [COL["green_dark"] if label == "Final gate" else "white" for label in dat["variant_label"]]
    label_color = [COL["ours"] if label == "Final gate" else COL["muted"] for label in dat["variant_label"]]

    scores = dat["selection_score"].to_numpy(float)
    ax.barh(
        y,
        scores,
        height=0.52,
        color=colors,
        edgecolor=edges,
        linewidth=0.55,
        alpha=0.96,
        zorder=3,
    )
    for yi, (_, row) in enumerate(dat.iterrows()):
        label = str(row["variant_label"])
        val = float(row["selection_score"])
        ax.text(
            max(0.018, val - 0.018),
            yi,
            f"{val:.2f}",
            ha="right",
            va="center",
            fontsize=5.15,
            color="white" if label == "Final gate" else COL["muted"],
            fontweight="bold" if label == "Final gate" else "normal",
            zorder=4,
        )

    display = {
        "Final gate": "full gate",
        "Spatial-null": "spatial-null",
        "Wavelet scale": "scale only",
        "Fixed prior": "fixed prior",
        "Contact kernel": "contact only",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([display.get(label, label) for label in dat["variant_label"]], fontsize=5.45)
    for tick, color in zip(ax.get_yticklabels(), label_color, strict=False):
        tick.set_color(color)
        if tick.get_text() == "full gate":
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0.50, 0.84)
    ax.set_xticks([0.55, 0.70, 0.85])
    ax.set_xticklabels(["0.55", "0.70", "0.85"], fontsize=5.6)
    ax.set_xlabel("mechanism benchmark score", fontsize=6.2, labelpad=1.0)
    ax.set_title("Scale-gate utility", fontsize=7.2, pad=2.0, color=COL["ink"])
    ax.text(
        0.985,
        0.085,
        "200-LR benchmark",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.15,
        color=COL["muted"],
    )
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.34, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.1)


def plot_cosmx_target_alignment_delta(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support[support["method"].isin(["scale-aware", "no-scale"])].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    wide = dat.pivot_table(index="LR", columns="method", values="target_spearman_r", aggfunc="mean")
    if {"scale-aware", "no-scale"} - set(wide.columns):
        ax.set_axis_off()
        return
    wide = wide.dropna(subset=["scale-aware", "no-scale"]).copy()
    wide["delta"] = wide["scale-aware"] - wide["no-scale"]
    short = {
        "CXCL12-CXCR4": "CXCL12",
        "CXCL9-CXCR3": "CXCL9",
        "CXCL10-CXCR3": "CXCL10",
        "VEGFA-KDR": "VEGFA",
        "SPP1-CD44": "SPP1",
        "TGFB1-TGFBR2": "TGFB1",
        "CCL5-CCR5": "CCL5",
        "CSF1-CSF1R": "CSF1",
        "CD274-PDCD1": "PD-L1",
    }
    order = [
        "CXCL12-CXCR4",
        "CXCL9-CXCR3",
        "CXCL10-CXCR3",
        "VEGFA-KDR",
        "SPP1-CD44",
    ]
    wide = wide.reindex([lr for lr in order if lr in wide.index]).dropna(subset=["delta"])
    if wide.empty:
        ax.set_axis_off()
        return

    y = np.arange(len(wide))
    vals = wide["delta"].to_numpy(float)
    pos_color = "#D30D14"
    neg_color = "#156D96"
    colors = [pos_color if val >= 0 else neg_color for val in vals]

    ax.axvline(0, color="#15191C", lw=0.74, zorder=2)
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="none", alpha=0.98, zorder=3)
    for yi, (lr, row) in enumerate(wide.iterrows()):
        val = float(row["delta"])
        ha = "left" if val >= 0 else "right"
        x_text = val + (0.006 if val >= 0 else -0.006)
        ax.text(
            x_text,
            yi,
            f"{val:+.02f}",
            ha=ha,
            va="center",
            fontsize=4.9,
            color=pos_color if val >= 0 else neg_color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([short.get(lr, lr) for lr in wide.index], fontsize=5.25)
    for tick, lr in zip(ax.get_yticklabels(), wide.index, strict=False):
        tick.set_color(COL["ink"])
        if lr in SELECTED_LRS:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(-0.125, 0.135)
    ax.set_xticks([-0.10, 0.00, 0.10])
    ax.set_xticklabels(["-0.10", "", "+0.10"], fontsize=5.55)
    ax.set_xlabel("Delta target Spearman r", fontsize=6.2, labelpad=1.0)
    ax.set_title("Response gain", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="x", color="#E1E6EA", linewidth=0.32, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.spines["bottom"].set_color("#15191C")
    ax.spines["bottom"].set_linewidth(0.72)


def plot_scale_range_coupling(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[~lr_summary["mechanism"].astype(str).str.contains("negative_control", regex=False)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    scale_cols = [f"scale_{i}" for i in range(6)]
    weights = dat[scale_cols].to_numpy(float)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    dat["scale_center"] = np.dot(weights, np.arange(len(scale_cols)))
    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)

    ax.axvspan(-0.2, 1.7, color="#C8C3B8", alpha=0.12, lw=0, zorder=0)
    ax.axvspan(1.7, 3.7, color="#D7835C", alpha=0.08, lw=0, zorder=0)
    ax.axvspan(3.7, 5.2, color="#237A78", alpha=0.10, lw=0, zorder=0)

    for _, row in dat[~dat["is_selected"]].iterrows():
        ax.scatter(
            row["scale_center"],
            row["top_median_distance"],
            s=38,
            color="#BDB8AE",
            edgecolor="white",
            linewidth=0.45,
            alpha=0.72,
            zorder=2,
        )
    for _, row in dat[dat["is_selected"]].iterrows():
        lr = str(row["LR"])
        ax.scatter(
            row["scale_center"],
            row["top_median_distance"],
            s=56,
            color=LR_COLORS.get(lr, COL["ours"]),
            edgecolor="white",
            linewidth=0.55,
            alpha=0.96,
            zorder=3,
        )

    if len(dat) >= 3:
        x = dat["scale_center"].to_numpy(float)
        y = dat["top_median_distance"].to_numpy(float)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(max(-0.05, x.min() - 0.10), min(5.05, x.max() + 0.10), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=COL["ink"], lw=0.75, alpha=0.55, zorder=1)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(0.04, 0.93, f"Spearman r={rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=COL["muted"])

    short = {
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "CXCL12-CXCR4": "CXCL12",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
    }
    offsets = {
        "SPP1-CD44": (0.10, -6, "left"),
        "CSF1-CSF1R": (0.08, 8, "left"),
        "CXCL12-CXCR4": (-0.12, 8, "right"),
        "VEGFA-KDR": (0.08, 8, "left"),
        "TGFB1-TGFBR2": (0.08, -8, "left"),
    }
    for lr in ["SPP1-CD44", "CSF1-CSF1R", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2"]:
        row = dat[dat["LR"].eq(lr)]
        if row.empty:
            continue
        row = row.iloc[0]
        dx, dy, ha = offsets.get(lr, (0.08, 6, "left"))
        ax.text(
            row["scale_center"] + dx,
            row["top_median_distance"] + dy,
            short.get(lr, lr),
            fontsize=5.2,
            color=LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"],
            ha=ha,
            va="center",
        )

    ax.set_xlim(-0.10, 5.10)
    ax.set_ylim(45, 190)
    ax.set_xticks([0, 2.5, 5.0])
    ax.set_xticklabels(["local", "mid", "broad"], fontsize=5.7)
    ax.set_yticks([50, 100, 150])
    ax.set_xlabel("learned scale center", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("top-edge distance", fontsize=6.4, labelpad=1.2)
    ax.set_title("Scale-to-range", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.36, zorder=0)


def plot_selected_top_edge_distance_distribution(ax: plt.Axes, scored: pd.DataFrame) -> None:
    scored = scored.copy()
    scored["LR"] = scored["ligand"].astype(str).str.cat(scored["receptor"].astype(str), sep="-")
    rows: list[pd.DataFrame] = []
    for lr in SELECTED_LRS:
        group = scored[scored["LR"].eq(lr)].copy()
        if group.empty:
            continue
        top = group.nlargest(min(300, len(group)), "score").copy()
        top["LR"] = lr
        rows.append(top)
    if not rows:
        ax.set_axis_off()
        return
    dat = pd.concat(rows, ignore_index=True)
    labels = {"SPP1-CD44": "SPP1\nCD44", "CCL5-CCR5": "CCL5\nCCR5", "CXCL12-CXCR4": "CXCL12\nCXCR4"}
    positions = np.arange(len(SELECTED_LRS))
    rng = np.random.default_rng(7)

    for i, lr in enumerate(SELECTED_LRS):
        vals = dat[dat["LR"].eq(lr)]["distance"].to_numpy(float)
        if len(vals) == 0:
            continue
        color = LR_COLORS.get(lr, COL["muted"])
        fill = LR_FILLS.get(lr, color)
        parts = ax.violinplot([vals], positions=[i], widths=0.68, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(fill)
            body.set_edgecolor("none")
            body.set_alpha(0.52)
        jitter_idx = rng.choice(np.arange(len(vals)), size=min(70, len(vals)), replace=False)
        jitter = rng.normal(0, 0.045, size=len(jitter_idx))
        ax.scatter(
            np.full(len(jitter_idx), i) + jitter,
            vals[jitter_idx],
            s=5.2,
            color=color,
            alpha=0.24,
            linewidths=0,
            zorder=2,
        )
        q25, med, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color=color, lw=1.2, zorder=4, solid_capstyle="round")
        ax.plot([i, i], [q25, q75], color=color, lw=2.1, alpha=0.82, solid_capstyle="round", zorder=3)
        ax.text(i, med + 8, f"{med:.0f}", ha="center", va="bottom", fontsize=5.0, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([labels.get(lr, lr) for lr in SELECTED_LRS], fontsize=5.7)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_ylabel("top-edge distance", fontsize=6.3, labelpad=1.2)
    ax.set_title("Range distribution", fontsize=7.2, pad=2.0, color=COL["ink"])
    ax.set_ylim(35, 220)
    ax.set_yticks([50, 100, 150, 200])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.36, zorder=0)
    ax.spines["bottom"].set_visible(False)


def plot_receiver_response_screen(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support.copy()
    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)
    dat = dat[np.isfinite(dat["target_spearman_r"]) & np.isfinite(dat["top_receiver_target_lift"])].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    x = dat["target_spearman_r"].to_numpy(float)
    y = dat["top_receiver_target_lift"].to_numpy(float)
    color = "#C85B78"

    for _, row in dat[~dat["is_selected"]].iterrows():
        ax.scatter(
            row["target_spearman_r"],
            row["top_receiver_target_lift"],
            s=38,
            facecolors="#F4D5DD",
            edgecolors=color,
            linewidths=0.58,
            alpha=0.62,
            zorder=3,
        )
    for _, row in dat[dat["is_selected"]].iterrows():
        lr = str(row["LR"])
        lr_color = LR_COLORS.get(lr, color)
        ax.scatter(
            row["target_spearman_r"],
            row["top_receiver_target_lift"],
            s=62,
            facecolors="white",
            edgecolors=lr_color,
            linewidths=1.15,
            alpha=0.98,
            zorder=4,
        )

    short = {
        "SPP1-CD44": "SPP1",
        "CXCL10-CXCR3": "CXCL10",
        "CXCL12-CXCR4": "CXCL12",
        "CSF1-CSF1R": "CSF1",
        "VEGFA-KDR": "VEGFA",
        "TGFB1-TGFBR2": "TGFB1",
        "CCL5-CCR5": "CCL5",
    }
    offsets = {
        "SPP1-CD44": (0.011, -0.032, "left"),
        "CXCL10-CXCR3": (0.013, 0.016, "left"),
        "CXCL12-CXCR4": (-0.018, 0.032, "right"),
        "VEGFA-KDR": (0.011, 0.034, "left"),
        "CSF1-CSF1R": (0.010, 0.034, "left"),
        "TGFB1-TGFBR2": (-0.012, -0.034, "right"),
    }
    for lr in ["SPP1-CD44", "CXCL10-CXCR3", "CXCL12-CXCR4", "VEGFA-KDR", "CSF1-CSF1R", "TGFB1-TGFBR2"]:
        row = dat[dat["LR"].eq(lr)]
        if row.empty:
            continue
        row = row.iloc[0]
        dx, dy, ha = offsets.get(lr, (0.008, 0.018, "left"))
        label_color = LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"]
        ax.text(
            row["target_spearman_r"] + dx,
            row["top_receiver_target_lift"] + dy,
            short.get(lr, lr),
            fontsize=5.05,
            color=label_color,
            ha=ha,
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.15),
            zorder=6,
        )

    ax.axvspan(0.30, 0.49, color="#F4D5DD", alpha=0.10, lw=0, zorder=0)
    ax.axhspan(0.20, 0.68, color="#F4D5DD", alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(0.15, 0.49)
    ax.set_ylim(0.02, 0.68)
    ax.set_xticks([0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_xlabel("score-target Spearman r", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("target lift", fontsize=6.4, labelpad=1.2)
    ax.set_title("Receiver response", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax, grid=False)
    ax.spines["left"].set_color("#15191C")
    ax.spines["bottom"].set_color("#15191C")
    ax.spines["left"].set_linewidth(0.72)
    ax.spines["bottom"].set_linewidth(0.72)

def plot_scale_calibration(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[~lr_summary["mechanism"].astype(str).str.contains("negative_control", regex=False)].copy()
    if dat.empty or "target_expected_scale" not in dat:
        ax.set_axis_off()
        return
    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)
    class_colors = {"local": "#8A8177", "intermediate": "#C94F48", "broad": "#237A78"}
    ax.axvspan(0, 1.5, color="#C8C3B8", alpha=0.12, lw=0, zorder=0)
    ax.axvspan(1.5, 3.5, color="#D7835C", alpha=0.08, lw=0, zorder=0)
    ax.axvspan(3.5, 5.0, color="#237A78", alpha=0.10, lw=0, zorder=0)
    ax.plot([0, 5], [0, 5], color=COL["grid"], lw=0.8, zorder=1)
    for _, row in dat.iterrows():
        lr = str(row["LR"])
        selected = bool(row["is_selected"])
        color = LR_COLORS.get(lr, class_colors.get(row.get("scale_class"), COL["muted"])) if selected else class_colors.get(row.get("scale_class"), "#AFA79B")
        ax.scatter(
            row["target_expected_scale"],
            row["expected_scale"],
            s=52 if selected else 34,
            color=color,
            alpha=0.95 if selected else 0.58,
            edgecolor="white",
            linewidth=0.55,
            zorder=3 if selected else 2,
        )
    short = {"CCL5-CCR5": "CCL5", "TGFB1-TGFBR2": "TGFB1", "CXCL12-CXCR4": "CXCL12", "VEGFA-KDR": "VEGFA"}
    offsets = {
        "CCL5-CCR5": (0.18, 0.12, "left"),
        "TGFB1-TGFBR2": (0.13, 0.12, "left"),
        "CXCL12-CXCR4": (-0.22, 0.15, "right"),
        "VEGFA-KDR": (-0.18, -0.18, "right"),
    }
    for lr, label in short.items():
        row = dat[dat["LR"].eq(lr)]
        if row.empty:
            continue
        row = row.iloc[0]
        dx, dy, ha = offsets.get(lr, (0.10, 0.10, "left"))
        ax.text(
            row["target_expected_scale"] + dx,
            row["expected_scale"] + dy,
            label,
            fontsize=5.2,
            color=LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"],
            ha=ha,
            va="center",
        )
    ax.text(0.05, 0.94, "auxiliary scale target", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=COL["muted"])
    ax.set_xlim(-0.05, 5.05)
    ax.set_ylim(-0.05, 5.05)
    ax.set_xticks([0, 2.5, 5])
    ax.set_yticks([0, 2.5, 5])
    ax.set_xticklabels(["local", "mid", "broad"], fontsize=5.7)
    ax.set_yticklabels(["local", "mid", "broad"], fontsize=5.7)
    ax.set_xlabel("expected scale", fontsize=6.5, labelpad=1.1)
    ax.set_ylabel("learned scale", fontsize=6.5, labelpad=1.1)
    ax.set_title("Scale gate diagnostic", fontsize=7.4, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="both", color=COL["grid"], linewidth=0.34, zorder=0)


def plot_empirical_range_summary(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[lr_summary["LR"].isin(SELECTED_LRS)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("top_median_distance")
    y = np.arange(len(dat))
    short = {
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "CXCL12-CXCR4": "CXCL12",
        "CCL5-CCR5": "CCL5",
    }
    for yi, (_, row) in enumerate(dat.iterrows()):
        lr = str(row["LR"])
        color = LR_COLORS.get(lr, COL["muted"])
        q25 = float(row["top_q25_distance"])
        q75 = float(row["top_q75_distance"])
        med = float(row["top_median_distance"])
        ax.plot([q25, q75], [yi, yi], color=color, lw=2.2, alpha=0.42, solid_capstyle="round", zorder=2)
        ax.scatter(med, yi, s=44, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(
            q75 + 6,
            yi,
            short.get(lr, lr),
            ha="left",
            va="center",
            fontsize=5.6,
            color=color,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(["short", "mid", "long"], fontsize=5.8)
    ax.set_ylim(-0.55, len(dat) - 0.45)
    ax.invert_yaxis()
    xmin = max(0, float(dat["top_q25_distance"].min()) - 18)
    xmax = float(dat["top_q75_distance"].max()) + 32
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([75, 125, 175])
    ax.set_xlabel("top-edge distance", fontsize=6.5, labelpad=1.1)
    ax.set_title("Empirical range", fontsize=7.4, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.36, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.2)


def plot_context_support(ax: plt.Axes, lr_summary: pd.DataFrame) -> None:
    dat = lr_summary[lr_summary["LR"].isin(SPECTRUM_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SPECTRUM_LRS)})
    dat = dat.sort_values("order")
    if dat.empty:
        ax.set_axis_off()
        return
    y = np.arange(len(dat))
    type_vals = dat["top_type_match_rate"].to_numpy(float)
    niche_vals = dat["top_niche_match_rate"].to_numpy(float)
    for yi, xv0, xv1 in zip(y, type_vals, niche_vals):
        ax.plot([xv0, xv1], [yi, yi], color=COL["grid"], lw=0.8, zorder=1)
    ax.scatter(type_vals, y, s=20, color="#BDB8AE", edgecolor="white", linewidth=0.45, label="cell type", zorder=2)
    ax.scatter(niche_vals, y, s=22, color=COL["immune"], edgecolor="white", linewidth=0.45, label="niche", zorder=3)
    short = {
        "CCL5-CCR5": "CCL5",
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(dat) - 0.45)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(lr, lr) for lr in dat["LR"]], fontsize=5.8)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("top-edge support", fontsize=6.8, labelpad=1.5)
    ax.set_title("Context support", fontsize=7.4, pad=7.0, color=COL["ink"])
    ax.text(0.28, 1.015, "cell type", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.2, color=COL["muted"])
    ax.text(0.70, 1.015, "niche", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.2, color=COL["immune"])
    ax.scatter([0.245], [1.032], transform=ax.transAxes, s=15, color="#BDB8AE", edgecolor="white", linewidth=0.35, clip_on=False)
    ax.scatter([0.665], [1.032], transform=ax.transAxes, s=15, color=COL["immune"], edgecolor="white", linewidth=0.35, clip_on=False)
    clean_axis(ax, grid=False)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines["left"].set_visible(False)


def plot_distance_programs(ax: plt.Axes, distance: pd.DataFrame) -> None:
    dat = distance[distance["LR"].isin(SELECTED_LRS)].copy()
    for lr in SELECTED_LRS:
        group = dat[dat["LR"].eq(lr)].sort_values("bin_mid")
        if group.empty:
            continue
        ax.plot(group["bin_mid"], group["mean_score_norm_by_lr"], color=LR_COLORS[lr], lw=1.25, alpha=0.96)
    ax.set_xlabel("edge distance", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("relative activity", fontsize=6.8, labelpad=1.5)
    ax.set_title("Distance footprint", fontsize=7.4, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlim(dat["bin_mid"].min() - 6, dat["bin_mid"].max() + 8)
    labels = {
        "CCL5-CCR5": (0.48, 0.26),
        "TGFB1-TGFBR2": (0.39, 0.70),
        "CXCL12-CXCR4": (0.57, 0.91),
    }
    for lr, (x_ax, y_ax) in labels.items():
        ax.text(x_ax, y_ax, lr, transform=ax.transAxes, fontsize=5.2, color=LR_COLORS[lr], ha="left", va="center")


def plot_hotspot_map(ax: plt.Axes, hotspots: pd.DataFrame, lr: str) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    bg_colors = {
        "tumor_proxy": "#DFA6A2",
        "t_cell_proxy": "#A9CFE2",
        "myeloid_proxy": "#C5BCE8",
        "stromal_proxy": "#E8C97B",
        "endothelial_proxy": "#9FCBC8",
        "other_proxy": "#D8D3CB",
    }
    for cell_type, group in dat.groupby("cell_type"):
        color = bg_colors.get(cell_type, COL["cell_bg"])
        ax.scatter(
            group["x_plot"],
            group["y_plot"],
            s=0.46,
            c=color,
            alpha=0.095,
            linewidths=0,
        )
    values = dat["receiver_activity"].to_numpy(float)
    vmax = np.quantile(values[values > 0], 0.992) if np.any(values > 0) else 1.0
    show = dat[dat["receiver_activity"] > np.quantile(dat["receiver_activity"], 0.955)].copy()
    ax.scatter(
        show["x_plot"],
        show["y_plot"],
        s=5.8,
        c=np.clip(show["receiver_activity"], 0, vmax),
        cmap=ACTIVITY_CMAP,
        vmin=0,
        vmax=vmax,
        alpha=0.92,
        linewidths=0.12,
        edgecolors="white",
        zorder=3,
    )
    top = dat[dat["is_top_receiver"].astype(bool)]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=10.0,
        facecolors="none",
        edgecolors=LR_COLORS.get(lr, COL["ours"]),
        linewidths=0.42,
        alpha=0.82,
        zorder=4,
    )
    ax.set_aspect("equal")
    strip_spatial_axis(ax)
    title = {
        "CCL5-CCR5": "local",
        "SPP1-CD44": "short",
        "CSF1-CSF1R": "mid",
        "TGFB1-TGFBR2": "intermediate",
        "CXCL12-CXCR4": "broad",
    }.get(lr, lr)
    ax.set_title(title, fontsize=7.0, pad=2.5, color=LR_COLORS.get(lr, COL["ink"]))


def plot_edge_footprint_map(ax: plt.Axes, scored: pd.DataFrame, cells: pd.DataFrame, lr: str, scale_label: str) -> None:
    dat = scored[scored["ligand"].astype(str).str.cat(scored["receptor"].astype(str), sep="-").eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return

    cell_pos = cells[["cell_id", "x", "y", "cell_type"]].copy()
    sender = cell_pos.rename(columns={"cell_id": "sender_id", "x": "sender_x", "y": "sender_y", "cell_type": "sender_type"})
    receiver = cell_pos.rename(columns={"cell_id": "receiver_id", "x": "receiver_x", "y": "receiver_y", "cell_type": "receiver_type"})
    dat = dat.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")
    dat = dat.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y", "score"])
    if dat.empty:
        ax.set_axis_off()
        return

    bg = cell_pos.copy()
    bg["x_plot"] = bg["x"].astype(float)
    bg["y_plot"] = -bg["y"].astype(float)
    for _, group in bg.groupby("cell_type"):
        ax.scatter(
            group["x_plot"],
            group["y_plot"],
            s=0.30,
            c="#D8D3CB",
            alpha=0.105,
            linewidths=0,
            zorder=0,
        )

    top_n = min(58, len(dat))
    top = dat.nlargest(top_n, "score").copy()
    top["sender_y_plot"] = -top["sender_y"].astype(float)
    top["receiver_y_plot"] = -top["receiver_y"].astype(float)
    line_segments = [
        [(row["sender_x"], row["sender_y_plot"]), (row["receiver_x"], row["receiver_y_plot"])]
        for _, row in top.iterrows()
    ]
    color = LR_COLORS.get(lr, COL["ours"])
    lc = LineCollection(line_segments, colors=color, linewidths=0.78, alpha=0.46, zorder=2)
    ax.add_collection(lc)
    ax.scatter(
        top["receiver_x"],
        top["receiver_y_plot"],
        s=13.0,
        facecolors="none",
        edgecolors=color,
        linewidths=0.70,
        alpha=0.86,
        zorder=3,
    )
    ax.scatter(
        top["sender_x"],
        top["sender_y_plot"],
        s=11.0,
        c=color,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.32,
        zorder=4,
    )

    med = float(top["distance"].median())
    q25 = float(top["distance"].quantile(0.25))
    q75 = float(top["distance"].quantile(0.75))
    lr_short = {"SPP1-CD44": "SPP1-CD44", "CSF1-CSF1R": "CSF1-CSF1R", "CXCL12-CXCR4": "CXCL12-CXCR4"}.get(lr, lr)
    ax.text(
        0.02,
        1.015,
        scale_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=color,
        clip_on=False,
    )
    ax.text(
        0.02,
        0.94,
        lr_short,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=COL["text"],
    )
    ax.text(
        0.02,
        0.865,
        f"{med:.0f} ({q25:.0f}-{q75:.0f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=COL["muted"],
    )
    ax.set_aspect("equal")
    strip_spatial_axis(ax)


def plot_target_response_map(ax: plt.Axes, hotspots: pd.DataFrame, lr: str) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    ax.scatter(dat["x_plot"], dat["y_plot"], s=0.42, c="#D8D3CB", alpha=0.12, linewidths=0)
    values = dat["target_score"].to_numpy(float)
    positive = values[values > 0]
    vmax = np.quantile(positive, 0.985) if len(positive) else 1.0
    show = dat[dat["target_score"] > np.quantile(dat["target_score"], 0.94)].copy()
    cmap = LinearSegmentedColormap.from_list("target_response", ["#F7F4EC", "#E6C06A", LR_COLORS.get(lr, COL["ours"])])
    ax.scatter(
        show["x_plot"],
        show["y_plot"],
        s=5.0,
        c=np.clip(show["target_score"], 0, vmax),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.88,
        linewidths=0.10,
        edgecolors="white",
        zorder=3,
    )
    top = dat[dat["is_top_receiver"].astype(bool)]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=10.0,
        facecolors="none",
        edgecolors=LR_COLORS.get(lr, COL["ours"]),
        linewidths=0.40,
        alpha=0.78,
        zorder=4,
    )
    ax.set_aspect("equal")
    strip_spatial_axis(ax)
    ax.set_title("TGFB1 response", fontsize=7.0, pad=2.5, color=LR_COLORS.get(lr, COL["ink"]))


def plot_receiver_response_map(ax: plt.Axes, hotspots: pd.DataFrame, lr: str) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    ax.scatter(dat["x_plot"], dat["y_plot"], s=0.42, c="#D8D3CB", alpha=0.16, linewidths=0)

    values = dat["target_score"].to_numpy(float)
    vmax = np.quantile(values[values > 0], 0.985) if np.any(values > 0) else 1.0
    show = dat[dat["target_score"] > np.quantile(dat["target_score"], 0.945)].copy()
    response_cmap = LinearSegmentedColormap.from_list("csf1_response", ["#F7F4EC", "#C9B9D8", LR_COLORS.get(lr, "#7E6C9A")])
    ax.scatter(
        show["x_plot"],
        show["y_plot"],
        s=5.8,
        c=np.clip(show["target_score"], 0, vmax),
        cmap=response_cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.88,
        linewidths=0.08,
        edgecolors="white",
        zorder=3,
    )
    top = dat[dat["is_top_receiver"].astype(bool)]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=12.0,
        facecolors="none",
        edgecolors=LR_COLORS.get(lr, COL["ours"]),
        linewidths=0.52,
        alpha=0.90,
        zorder=4,
    )
    ax.set_aspect("equal")
    strip_spatial_axis(ax)
    ax.set_title("CSF1 response", fontsize=7.2, pad=2.5, color=LR_COLORS.get(lr, COL["ink"]))


def plot_spatial_colocalization_map(
    ax: plt.Axes,
    hotspots: pd.DataFrame,
    lr: str,
    *,
    mode: str,
    show_lr: bool,
) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    color = LR_COLORS.get(lr, COL["green_dark"])
    fill = LR_FILLS.get(lr, color)
    short = {
        "SPP1-CD44": "SPP1-CD44",
        "CSF1-CSF1R": "CSF1-CSF1R",
        "CXCL12-CXCR4": "CXCL12-CXCR4",
    }.get(lr, lr)

    ax.scatter(
        dat["x_plot"],
        dat["y_plot"],
        s=0.36,
        c="#DDE2E9",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    top = dat[dat["is_top_receiver"].astype(bool)].copy()
    if mode == "communication":
        values = dat["receiver_activity"].to_numpy(float)
        cutoff = np.quantile(values, 0.955)
        show = dat[dat["receiver_activity"].ge(cutoff)].copy()
        vmax = np.quantile(values[values > 0], 0.992) if np.any(values > 0) else 1.0
        cmap = LinearSegmentedColormap.from_list(f"{lr}_pred_cmap", [COL["cream"], fill, color])
        color_values = np.clip(show["receiver_activity"], 0, vmax)
        row_label = "communication"
        point_size = 4.8
    else:
        values = dat["target_score"].to_numpy(float)
        cutoff = np.quantile(values, 0.940)
        show = dat[dat["target_score"].ge(cutoff)].copy()
        vmax = np.quantile(values[values > 0], 0.985) if np.any(values > 0) else 1.0
        cmap = LinearSegmentedColormap.from_list(f"{lr}_resp_cmap", ["#F7FAFC", COL["cream"], color])
        color_values = np.clip(show["target_score"], 0, vmax)
        row_label = "target response"
        point_size = 4.5

    ax.scatter(
        show["x_plot"],
        show["y_plot"],
        s=point_size,
        c=color_values,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.94,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=9.5 if mode == "communication" else 8.4,
        facecolors="none",
        edgecolors=color,
        linewidths=0.46 if mode == "communication" else 0.38,
        alpha=0.86 if mode == "communication" else 0.58,
        rasterized=True,
        zorder=3,
    )

    if show_lr:
        ax.text(
            0.02,
            1.025,
            short,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color=color,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_aspect("equal")
    strip_spatial_axis(ax)


def _lr_scored_edges(scored: pd.DataFrame, lr: str) -> pd.DataFrame:
    lig, rec = lr.split("-", 1)
    dat = scored[scored["ligand"].astype(str).eq(lig) & scored["receptor"].astype(str).eq(rec)].copy()
    return dat.dropna(subset=["score"])


def compute_top_fraction_roi(
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    hotspots: pd.DataFrame,
    lr: str,
    *,
    fraction: float = 0.20,
) -> tuple[float, float, float, float]:
    edge = _lr_scored_edges(scored, lr)
    if not edge.empty:
        cutoff = edge["score"].quantile(1.0 - fraction)
        top_edge = edge[edge["score"].ge(cutoff)].copy()
        top_edge = top_edge.nlargest(min(900, len(top_edge)), "score")
    else:
        top_edge = edge

    hot = hotspots[hotspots["LR"].eq(lr)].copy()
    if not hot.empty:
        top_target = hot.nlargest(min(700, len(hot)), "target_score")
        top_receiver = hot[hot["is_top_receiver"].astype(bool)] if "is_top_receiver" in hot else hot.iloc[0:0]
    else:
        top_target = hot
        top_receiver = hot

    ids = set(top_edge.get("sender_id", pd.Series(dtype=str)).astype(str))
    ids |= set(top_edge.get("receiver_id", pd.Series(dtype=str)).astype(str))
    ids |= set(top_target.get("cell_id", pd.Series(dtype=str)).astype(str))
    ids |= set(top_receiver.get("cell_id", pd.Series(dtype=str)).astype(str))
    pts = cells[cells["cell_id"].astype(str).isin(ids)].copy()
    if pts.empty:
        pts = cells.copy()

    pts["x_plot"] = pts["x"].astype(float)
    pts["y_plot"] = -pts["y"].astype(float)
    x0, x1 = pts["x_plot"].quantile([0.16, 0.84]).to_numpy(float)
    y0, y1 = pts["y_plot"].quantile([0.16, 0.84]).to_numpy(float)
    if x0 == x1 or y0 == y1:
        x0, x1 = float(cells["x"].min()), float(cells["x"].max())
        y0, y1 = float((-cells["y"]).min()), float((-cells["y"]).max())
    pad_x = 0.12 * (x1 - x0)
    pad_y = 0.12 * (y1 - y0)
    x0 = max(float(cells["x"].min()), x0 - pad_x)
    x1 = min(float(cells["x"].max()), x1 + pad_x)
    all_y = -cells["y"].astype(float)
    y0 = max(float(all_y.min()), y0 - pad_y)
    y1 = min(float(all_y.max()), y1 + pad_y)
    return x0, x1, y0, y1


def plot_top_fraction_communication_map(
    ax: plt.Axes,
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    hotspots: pd.DataFrame,
    lr: str,
    *,
    fraction: float = 0.20,
    roi: tuple[float, float, float, float] | None = None,
    show_row_label: bool = False,
) -> None:
    bg = hotspots[hotspots["LR"].eq(lr)].copy()
    if bg.empty:
        ax.set_axis_off()
        return
    bg["x_plot"] = bg["x"].astype(float)
    bg["y_plot"] = -bg["y"].astype(float)
    bg["score"] = bg["receiver_activity"].astype(float)
    if roi is not None:
        x0, x1, y0, y1 = roi
        bg = bg[bg["x_plot"].between(x0, x1) & bg["y_plot"].between(y0, y1)]
    plot_cellnest_like_spot_heatmap(ax, bg, "score", vmax_quantile=0.985)

    if show_row_label:
        ax.text(
            -0.030,
            0.50,
            "communication",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=5.35,
            color="#8A9199",
            clip_on=False,
        )
    ax.text(
        0.02,
        1.010,
        lr,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.95,
        color="#3A4148",
        fontweight="semibold",
        clip_on=False,
    )
    ax.text(
        0.98,
        1.010,
        "receiver activity",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.05,
        color="#8A9199",
        clip_on=False,
    )
    ax.set_aspect("equal")
    if roi is not None:
        apply_roi(ax, roi)
    strip_spatial_axis(ax)


def plot_overlay_validation_map(
    ax: plt.Axes,
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    hotspots: pd.DataFrame,
    support: pd.DataFrame,
    lr: str,
    *,
    fraction: float = 0.20,
    roi: tuple[float, float, float, float] | None = None,
    show_row_label: bool = False,
) -> None:
    hot = hotspots[hotspots["LR"].eq(lr)].copy()
    if hot.empty:
        ax.set_axis_off()
        return

    hot["x_plot"] = hot["x"].astype(float)
    hot["y_plot"] = -hot["y"].astype(float)
    if roi is not None:
        x0, x1, y0, y1 = roi
        hot = hot[hot["x_plot"].between(x0, x1) & hot["y_plot"].between(y0, y1)].copy()
    if hot.empty:
        ax.set_axis_off()
        return

    color = LR_COLORS.get(lr, COL["slate_dark"])
    fill = LR_FILLS.get(lr, color)
    values = hot["target_score"].to_numpy(float)
    positive = values[values > 0]
    vmax = np.quantile(positive, 0.985) if len(positive) else 1.0

    ax.scatter(hot["x_plot"], hot["y_plot"], s=0.26, c="#E7EAF0", alpha=0.17, linewidths=0, zorder=0)
    draw_spatial_tilemap(
        ax,
        hot["x_plot"].to_numpy(float),
        hot["y_plot"].to_numpy(float),
        np.clip(values, 0, vmax),
        roi=roi,
        cmap=TARGET_RESPONSE_CMAP,
        norm=PowerNorm(gamma=0.64, vmin=0, vmax=max(float(vmax), 1e-8)),
        bins=80,
    )

    receivers = get_top_fraction_receivers(scored, cells, lr, fraction=fraction, roi=roi)
    if not receivers.empty:
        draw_hotspot_contours(ax, receivers, roi=roi, color=color, fill=fill, bins=84)

    support_row = support[support["LR"].eq(lr)]
    if support_row.empty:
        lift_txt = "lift n/a"
        corr_txt = "r n/a"
    else:
        row = support_row.iloc[0]
        lift_txt = f"lift {float(row['top_receiver_target_lift']):.2f}"
        corr_txt = f"r {float(row['target_spearman_r']):.2f}"

    ax.text(
        0.02,
        1.035,
        lr,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=color,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.035,
        0.045,
        f"{lift_txt}\n{corr_txt}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.35,
        color=COL["ink"],
        linespacing=1.18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, boxstyle="round,pad=0.18"),
        zorder=5,
    )
    if show_row_label:
        ax.text(
            -0.036,
            0.50,
            "response + hotspot",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=5.35,
            color="#8A9199",
            linespacing=1.12,
            clip_on=False,
        )
    ax.set_aspect("equal")
    if roi is not None:
        apply_roi(ax, roi)
    strip_spatial_axis(ax)


def get_top_fraction_receivers(
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    lr: str,
    *,
    fraction: float,
    roi: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    dat = _lr_scored_edges(scored, lr)
    if dat.empty:
        return pd.DataFrame()
    cutoff = dat["score"].quantile(1.0 - fraction)
    top = dat[dat["score"].ge(cutoff)].copy()
    cell_pos = cells[["cell_id", "x", "y"]].copy()
    receiver = cell_pos.rename(columns={"cell_id": "receiver_id", "x": "receiver_x", "y": "receiver_y"})
    top = top.merge(receiver, on="receiver_id", how="left")
    top = top.dropna(subset=["receiver_x", "receiver_y", "score"])
    top["receiver_y_plot"] = -top["receiver_y"].astype(float)
    if roi is not None:
        x0, x1, y0, y1 = roi
        top = top[top["receiver_x"].between(x0, x1) & top["receiver_y_plot"].between(y0, y1)]
    if top.empty:
        return pd.DataFrame()
    receivers = (
        top.groupby("receiver_id", as_index=False)
        .agg(receiver_x=("receiver_x", "first"), receiver_y_plot=("receiver_y_plot", "first"), score=("score", "mean"))
        .sort_values("score", ascending=False)
    )
    return receivers


def draw_hotspot_contours(
    ax: plt.Axes,
    receivers: pd.DataFrame,
    *,
    roi: tuple[float, float, float, float] | None,
    color: str,
    fill: str,
    bins: int = 84,
) -> None:
    if receivers.empty:
        return
    x = receivers["receiver_x"].to_numpy(float)
    y = receivers["receiver_y_plot"].to_numpy(float)
    weights = receivers["score"].to_numpy(float)
    if roi is None:
        x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
        y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    else:
        x0, x1, y0, y1 = roi
    x_edges = np.linspace(x0, x1, bins + 1)
    y_edges = np.linspace(y0, y1, bins + 1)
    score_sum, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=weights)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    density = score_sum * np.sqrt(np.clip(counts, 0, None))
    if np.nanmax(density) <= 0:
        return
    density = smooth_grid(density, passes=2)
    density = density / max(float(np.nanmax(density)), 1e-8)
    xx = (x_edges[:-1] + x_edges[1:]) / 2
    yy = (y_edges[:-1] + y_edges[1:]) / 2
    high = density[density > 0]
    if len(high) == 0:
        return
    levels = [float(np.quantile(high, 0.84)), float(np.quantile(high, 0.93))]
    levels = sorted(set([lv for lv in levels if 0 < lv < 1]))
    if not levels:
        return
    ax.contourf(xx, yy, density, levels=[levels[0], 1.0], colors=[fill], alpha=0.12, antialiased=True, zorder=2)
    ax.contour(xx, yy, density, levels=levels, colors=[color] * len(levels), linewidths=[0.82, 1.18][: len(levels)], alpha=0.96, zorder=4)
    top_pts = receivers.nlargest(min(42, len(receivers)), "score")
    ax.scatter(
        top_pts["receiver_x"],
        top_pts["receiver_y_plot"],
        s=8.5,
        c=color,
        alpha=0.42,
        linewidths=0,
        zorder=5,
    )


def smooth_grid(grid: np.ndarray, *, passes: int = 2) -> np.ndarray:
    out = np.nan_to_num(grid.astype(float), nan=0.0)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2]
            + 2 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2 * padded[1:-1, :-2]
            + 4 * padded[1:-1, 1:-1]
            + 2 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return out


def plot_target_response_heatmap(
    ax: plt.Axes,
    hotspots: pd.DataFrame,
    lr: str,
    *,
    roi: tuple[float, float, float, float] | None = None,
    show_row_label: bool = False,
) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return

    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    if roi is not None:
        x0, x1, y0, y1 = roi
        dat = dat[dat["x_plot"].between(x0, x1) & dat["y_plot"].between(y0, y1)].copy()
    if dat.empty:
        ax.set_axis_off()
        return

    plot_cellnest_like_spot_heatmap(ax, dat, "target_score", vmax_quantile=0.985)

    if show_row_label:
        ax.text(
            -0.030,
            0.50,
            "target response",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=5.8,
            color=COL["muted"],
            clip_on=False,
        )
    ax.set_aspect("equal")
    if roi is not None:
        apply_roi(ax, roi)
    strip_spatial_axis(ax)


def draw_spatial_tilemap(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None,
    cmap: LinearSegmentedColormap,
    norm: PowerNorm,
    bins: int = 76,
    smooth_passes: int = 0,
    alpha: float = 0.88,
) -> None:
    if len(x) == 0:
        return
    if roi is None:
        x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
        y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    else:
        x0, x1, y0, y1 = roi

    x_edges = np.linspace(x0, x1, bins + 1)
    y_edges = np.linspace(y0, y1, bins + 1)
    sums, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=values)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    mean[counts < 2] = np.nan
    if smooth_passes > 0:
        valid = np.isfinite(mean).astype(float)
        filled = np.nan_to_num(mean, nan=0.0)
        smooth_vals = smooth_grid(filled, passes=smooth_passes)
        smooth_valid = smooth_grid(valid, passes=smooth_passes)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = smooth_vals / smooth_valid
        mean[smooth_valid < 0.10] = np.nan
    masked = np.ma.masked_invalid(mean)
    ax.imshow(
        masked,
        extent=(x0, x1, y0, y1),
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear" if smooth_passes > 0 else "nearest",
        alpha=alpha,
        zorder=1,
    )


def plot_cellnest_like_spot_heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_col: str,
    *,
    vmax_quantile: float,
) -> None:
    if frame.empty:
        return
    plot = frame.dropna(subset=["x_plot", "y_plot", value_col]).copy()
    if plot.empty:
        return
    values = plot[value_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive = values[values > 0]
    if len(positive):
        vmin = float(np.nanquantile(positive, 0.05))
        vmax = float(np.nanquantile(positive, vmax_quantile))
    else:
        vmin, vmax = 0.0, max(float(values.max()), 1.0)
    if vmax <= vmin:
        vmax = max(vmin + 1e-6, 1.0)
    scaled = (values - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0, 1)
    scaled = np.where(values > 0, 0.12 + 0.88 * scaled, 0.0)
    plot["_plot_value"] = scaled
    plot = plot.sort_values(["_plot_value", "cell_id"], ascending=[True, True])
    ax.scatter(
        plot["x_plot"],
        plot["y_plot"],
        s=3.85,
        c=plot["_plot_value"],
        cmap=CELLNEST_SPOT_CMAP,
        norm=PowerNorm(gamma=0.42, vmin=0, vmax=1),
        marker="o",
        alpha=0.82,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )


def add_spot_heatmap_colorbar(fig: plt.Figure, axes: list[plt.Axes], *, label: str) -> None:
    if not axes:
        return
    boxes = [ax.get_position() for ax in axes]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    bottom = min(box.y0 for box in boxes)
    top = max(box.y1 for box in boxes)
    height = top - bottom
    cax = fig.add_axes([right + 0.006, bottom + 0.18 * height, 0.010, 0.56 * height])
    sm = mpl.cm.ScalarMappable(cmap=CELLNEST_SPOT_CMAP, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])
    cb.ax.tick_params(labelsize=5.1, length=0, pad=1.0, colors=COL["muted"])
    cb.set_label(label, fontsize=5.2, labelpad=2.0, color=COL["muted"])


def draw_region_heat_surface(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    roi: tuple[float, float, float, float] | None,
    bins: int = 92,
    smooth_passes: int = 4,
    mask_quantile: float = 0.18,
    contour_quantiles: tuple[float, ...] = (0.65, 0.84, 0.94),
    contour_color: str = "#7D342F",
    alpha: float = 0.94,
    use_density: bool = False,
) -> None:
    if len(x) == 0:
        return
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[finite]
    y = y[finite]
    values = values[finite]
    if len(x) == 0:
        return
    if roi is None:
        x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
        y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    else:
        x0, x1, y0, y1 = roi
    x_edges = np.linspace(x0, x1, bins + 1)
    y_edges = np.linspace(y0, y1, bins + 1)
    sums, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=values)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = sums / counts
    grid[counts < 1] = np.nan
    if use_density:
        grid = np.nan_to_num(grid, nan=0.0) * np.sqrt(np.clip(counts, 0, None))
        grid[counts < 1] = np.nan
    valid = np.isfinite(grid).astype(float)
    filled = np.nan_to_num(grid, nan=0.0)
    if smooth_passes > 0:
        smooth_vals = smooth_grid(filled, passes=smooth_passes)
        smooth_valid = smooth_grid(valid, passes=smooth_passes)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = smooth_vals / smooth_valid
        grid[smooth_valid < 0.015] = np.nan
    positive = grid[np.isfinite(grid) & (grid > 0)]
    if len(positive) == 0:
        return
    vmax = float(np.nanquantile(positive, 0.985))
    if vmax <= 0:
        return
    grid = np.clip(grid / vmax, 0, 1)
    positive = grid[np.isfinite(grid) & (grid > 0)]
    threshold = float(np.nanquantile(positive, mask_quantile)) if len(positive) else 0.0
    masked = np.ma.masked_where(~np.isfinite(grid) | (grid <= threshold), grid)
    ax.imshow(
        masked,
        extent=(x0, x1, y0, y1),
        origin="lower",
        cmap=REGION_HEAT_CMAP,
        norm=PowerNorm(gamma=0.72, vmin=0, vmax=1),
        interpolation="bilinear",
        alpha=alpha,
        zorder=1,
    )
    xx = (x_edges[:-1] + x_edges[1:]) / 2
    yy = (y_edges[:-1] + y_edges[1:]) / 2
    levels = []
    for q in contour_quantiles:
        level = float(np.nanquantile(positive, q))
        if threshold < level < 1:
            levels.append(level)
    levels = sorted(set(np.round(levels, 4)))
    if levels:
        ax.contour(
            xx,
            yy,
            np.ma.masked_invalid(grid),
            levels=levels,
            colors=[contour_color] * len(levels),
            linewidths=np.linspace(0.40, 0.88, len(levels)),
            alpha=0.84,
            zorder=3,
        )


def compute_focus_roi(scored: pd.DataFrame, cells: pd.DataFrame, hotspots: pd.DataFrame, lr: str) -> tuple[float, float, float, float]:
    lig, rec = lr.split("-", 1)
    edge = scored[scored["ligand"].astype(str).eq(lig) & scored["receptor"].astype(str).eq(rec)].copy()
    top_edge = edge.nlargest(min(350, len(edge)), "score") if not edge.empty else edge
    hot = hotspots[hotspots["LR"].eq(lr)].copy()
    top_target = hot.nlargest(min(350, len(hot)), "target_score") if not hot.empty else hot
    ids = set(top_edge.get("sender_id", pd.Series(dtype=str)).astype(str))
    ids |= set(top_edge.get("receiver_id", pd.Series(dtype=str)).astype(str))
    ids |= set(top_target.get("cell_id", pd.Series(dtype=str)).astype(str))
    pts = cells[cells["cell_id"].astype(str).isin(ids)].copy()
    if pts.empty:
        pts = cells.copy()
    pts["x_plot"] = pts["x"].astype(float)
    pts["y_plot"] = -pts["y"].astype(float)
    x0, x1 = pts["x_plot"].quantile([0.10, 0.90]).to_numpy(float)
    y0, y1 = pts["y_plot"].quantile([0.10, 0.90]).to_numpy(float)
    if x0 == x1 or y0 == y1:
        x0, x1 = float(cells["x"].min()), float(cells["x"].max())
        y0, y1 = float((-cells["y"]).min()), float((-cells["y"]).max())
    pad_x = 0.18 * (x1 - x0)
    pad_y = 0.18 * (y1 - y0)
    x0 = max(float(cells["x"].min()), x0 - pad_x)
    x1 = min(float(cells["x"].max()), x1 + pad_x)
    all_y = -cells["y"].astype(float)
    y0 = max(float(all_y.min()), y0 - pad_y)
    y1 = min(float(all_y.max()), y1 + pad_y)
    return x0, x1, y0, y1


def apply_roi(ax: plt.Axes, roi: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = roi
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def plot_focus_edge_map(
    ax: plt.Axes,
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    lr: str,
    *,
    top_n: int,
    label: str,
    roi: tuple[float, float, float, float] | None = None,
) -> None:
    dat = scored.copy()
    dat["LR"] = dat["ligand"].astype(str).str.cat(dat["receptor"].astype(str), sep="-")
    dat = dat[dat["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    color = LR_COLORS.get(lr, COL["slate_dark"])
    fill = LR_FILLS.get(lr, color)
    cell_pos = cells[["cell_id", "x", "y"]].copy()
    sender = cell_pos.rename(columns={"cell_id": "sender_id", "x": "sender_x", "y": "sender_y"})
    receiver = cell_pos.rename(columns={"cell_id": "receiver_id", "x": "receiver_x", "y": "receiver_y"})
    dat = dat.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")
    dat = dat.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y", "score"])
    top = dat.nlargest(min(top_n, len(dat)), "score").copy()

    bg = cells.copy()
    bg["x_plot"] = bg["x"].astype(float)
    bg["y_plot"] = -bg["y"].astype(float)
    if roi is not None:
        x0, x1, y0, y1 = roi
        bg = bg[bg["x_plot"].between(x0, x1) & bg["y_plot"].between(y0, y1)]
    ax.scatter(bg["x_plot"], bg["y_plot"], s=1.05, c="#E4E8EF", alpha=0.30, linewidths=0, rasterized=True, zorder=0)

    if len(top) > 0:
        draw = top.nlargest(min(220, len(top)), "score").copy()
        draw["sender_y_plot"] = -draw["sender_y"].astype(float)
        draw["receiver_y_plot"] = -draw["receiver_y"].astype(float)
        if roi is not None:
            x0, x1, y0, y1 = roi
            draw = draw[
                (draw["sender_x"].between(x0, x1) & draw["sender_y_plot"].between(y0, y1))
                | (draw["receiver_x"].between(x0, x1) & draw["receiver_y_plot"].between(y0, y1))
            ]
        segments = [
            [(row["sender_x"], row["sender_y_plot"]), (row["receiver_x"], row["receiver_y_plot"])]
            for _, row in draw.iterrows()
        ]
        if segments:
            lc = LineCollection(segments, colors=color, linewidths=0.42, alpha=0.20, zorder=1)
            ax.add_collection(lc)

        top["sender_y_plot"] = -top["sender_y"].astype(float)
        top["receiver_y_plot"] = -top["receiver_y"].astype(float)
        if roi is not None:
            x0, x1, y0, y1 = roi
            top = top[
                (top["sender_x"].between(x0, x1) & top["sender_y_plot"].between(y0, y1))
                | (top["receiver_x"].between(x0, x1) & top["receiver_y_plot"].between(y0, y1))
            ]
        ax.scatter(
            top["sender_x"],
            top["sender_y_plot"],
            s=8.0,
            c=fill,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax.scatter(
            top["receiver_x"],
            top["receiver_y_plot"],
            s=14.0,
            facecolors="none",
            edgecolors=color,
            linewidths=0.58,
            alpha=0.80,
            rasterized=True,
            zorder=3,
        )

    ax.text(0.02, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=COL["muted"], clip_on=False)
    if top_n == 3000:
        ax.text(
            0.02,
            1.115,
            lr,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
            color=color,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_aspect("equal")
    if roi is not None:
        apply_roi(ax, roi)
    strip_spatial_axis(ax)


def plot_focus_target_response_map(
    ax: plt.Axes,
    hotspots: pd.DataFrame,
    lr: str,
    *,
    roi: tuple[float, float, float, float] | None = None,
) -> None:
    dat = hotspots[hotspots["LR"].eq(lr)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["x_plot"] = dat["x"].astype(float)
    dat["y_plot"] = -dat["y"].astype(float)
    color = LR_COLORS.get(lr, COL["slate_dark"])
    fill = LR_FILLS.get(lr, color)
    bg = dat.copy()
    if roi is not None:
        x0, x1, y0, y1 = roi
        bg = bg[bg["x_plot"].between(x0, x1) & bg["y_plot"].between(y0, y1)]
    ax.scatter(bg["x_plot"], bg["y_plot"], s=1.05, c="#E4E8EF", alpha=0.30, linewidths=0, rasterized=True, zorder=0)
    values = dat["target_score"].to_numpy(float)
    cutoff = np.quantile(values, 0.950)
    show = dat[dat["target_score"].ge(cutoff)].copy()
    if roi is not None:
        x0, x1, y0, y1 = roi
        show = show[show["x_plot"].between(x0, x1) & show["y_plot"].between(y0, y1)]
    vmax = np.quantile(values[values > 0], 0.985) if np.any(values > 0) else 1.0
    cmap = LinearSegmentedColormap.from_list(f"{lr}_focus_response", ["#F7FAFC", fill, color])
    ax.scatter(
        show["x_plot"],
        show["y_plot"],
        s=14.0,
        c=np.clip(show["target_score"], 0, vmax),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.90,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    top = dat[dat["is_top_receiver"].astype(bool)].copy()
    if roi is not None:
        x0, x1, y0, y1 = roi
        top = top[top["x_plot"].between(x0, x1) & top["y_plot"].between(y0, y1)]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=13.5,
        facecolors="none",
        edgecolors=color,
        linewidths=0.50,
        alpha=0.68,
        rasterized=True,
        zorder=3,
    )
    ax.text(0.02, 1.03, "target response", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=COL["muted"], clip_on=False)
    ax.set_aspect("equal")
    if roi is not None:
        apply_roi(ax, roi)
    strip_spatial_axis(ax)


def plot_target_correlation_ranking(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support.copy()
    dat = dat.sort_values("target_spearman_r", ascending=False).head(6).copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat = dat.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(dat))
    short = {
        "CXCL12-CXCR4": "CXCL12",
        "CXCL9-CXCR3": "CXCL9",
        "CXCL10-CXCR3": "CXCL10",
        "CCL5-CCR5": "CCL5",
        "TGFB1-TGFBR2": "TGFB1",
        "CD274-PDCD1": "PD-L1",
        "VEGFA-KDR": "VEGFA",
        "CSF1-CSF1R": "CSF1",
        "SPP1-CD44": "SPP1",
    }
    colors = [LR_COLORS.get(lr, "#BDB8AE") if lr in SELECTED_LRS else "#C7C1B8" for lr in dat["LR"]]
    ax.barh(y, dat["target_spearman_r"], height=0.54, color=colors, edgecolor="none", zorder=3)
    for yi, (_, row) in enumerate(dat.iterrows()):
        lr = str(row["LR"])
        label_color = LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"]
        ax.text(
            0.012,
            yi,
            short.get(lr, lr),
            ha="left",
            va="center",
            fontsize=5.5,
            color=label_color,
            fontweight="bold" if lr in SELECTED_LRS else "normal",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.035"),
            zorder=4,
        )
        ax.text(
            float(row["target_spearman_r"]) + 0.012,
            yi,
            f"{row['target_spearman_r']:.2f}",
            ha="left",
            va="center",
            fontsize=5.2,
            color=label_color,
            zorder=4,
        )
    ax.set_xlim(0, 0.52)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.50])
    ax.set_xlabel("score-target Spearman r", fontsize=6.4, labelpad=1.0)
    ax.set_title("Target correlation", fontsize=7.2, pad=2.5, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.36, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_selected_functional_summary(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support[support["LR"].isin(SELECTED_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(SELECTED_LRS)})
    dat = dat.sort_values("order")
    if dat.empty:
        ax.set_axis_off()
        return
    metrics = [
        ("target_spearman_r", "corr."),
        ("top_receiver_target_lift", "lift"),
    ]
    mat = dat[[m[0] for m in metrics]].to_numpy(float)
    col_max = np.nanmax(mat, axis=0)
    col_max[col_max == 0] = 1.0
    norm = mat / col_max
    cmap = LinearSegmentedColormap.from_list("functional_summary", ["#F7F4EC", "#E6C06A", "#7E6C9A"])
    im = ax.imshow(norm, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    short = {
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "CXCL12-CXCR4": "CXCL12",
    }
    ax.set_yticks(np.arange(len(dat)))
    ax.set_yticklabels([short.get(lr, lr) for lr in dat["LR"]], fontsize=5.8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=5.8)
    for yi, (_, row) in enumerate(dat.iterrows()):
        lr = str(row["LR"])
        ax.get_yticklabels()[yi].set_color(LR_COLORS.get(lr, COL["text"]))
        for xi, (col, _) in enumerate(metrics):
            val = float(row[col])
            ax.text(
                xi,
                yi,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color=COL["ink"] if norm[yi, xi] < 0.66 else "white",
                zorder=3,
            )
    ax.set_title("Functional support", fontsize=7.2, pad=2.5, color=COL["ink"])
    ax.tick_params(axis="both", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dat), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_target_support(ax: plt.Axes, support: pd.DataFrame) -> None:
    dat = support[support["LR"].isin(SPECTRUM_LRS)].copy()
    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)
    ax.axhline(0, color=COL["grid"], lw=0.5, zorder=0)
    ax.axvline(0, color=COL["grid"], lw=0.5, zorder=0)
    for _, row in dat[~dat["is_selected"]].iterrows():
        ax.scatter(
            row["target_spearman_r"],
            row["top_receiver_target_lift"],
            s=35,
            color="#BDB8AE",
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
    for _, row in dat[dat["is_selected"]].iterrows():
        ax.scatter(
            row["target_spearman_r"],
            row["top_receiver_target_lift"],
            s=50,
            color=LR_COLORS.get(row["LR"], COL["ours"]),
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    label_offsets = {
        "CCL5-CCR5": (0.008, 0.022),
        "SPP1-CD44": (0.008, -0.026),
        "CSF1-CSF1R": (0.007, 0.020),
        "TGFB1-TGFBR2": (0.007, 0.018),
        "VEGFA-KDR": (0.006, 0.018),
        "CXCL12-CXCR4": (-0.095, 0.018),
    }
    short = {
        "CCL5-CCR5": "CCL5",
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    for _, row in dat.iterrows():
        dx, dy = label_offsets.get(row["LR"], (0.006, 0.012))
        color = LR_COLORS.get(row["LR"], COL["muted"]) if row["is_selected"] else COL["muted"]
        ax.text(
            row["target_spearman_r"] + dx,
            row["top_receiver_target_lift"] + dy,
            short.get(row["LR"], row["LR"]),
            fontsize=5.5 if row["is_selected"] else 5.1,
            color=color,
            ha="left",
            va="center",
        )
    ax.set_xlabel("score-target Spearman r", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("hotspot target lift", fontsize=6.8, labelpad=1.5)
    ax.set_title("Receiver-response support", fontsize=7.4, pad=2.5, color=COL["ink"])
    ax.set_xlim(0.16, 0.50)
    ax.set_ylim(0.04, 0.66)
    clean_axis(ax)
    ax.grid(axis="both", color=COL["grid"], linewidth=0.38, zorder=0)


def plot_scale_target_coupling(ax: plt.Axes, lr_summary: pd.DataFrame, support: pd.DataFrame) -> None:
    dat = lr_summary.merge(
        support[["LR", "target_spearman_r", "top_receiver_target_lift", "top_vs_rest_mannwhitney_p"]],
        on="LR",
        how="inner",
    )
    dat = dat[~dat["mechanism"].astype(str).str.contains("negative_control", regex=False)].copy()
    if dat.empty:
        ax.set_axis_off()
        return

    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)
    dat = dat.sort_values("expected_scale")
    ax.axvspan(0, 1.5, color="#C8C3B8", alpha=0.13, lw=0, zorder=0)
    ax.axvspan(1.5, 3.5, color="#D7835C", alpha=0.10, lw=0, zorder=0)
    ax.axvspan(3.5, 5.0, color="#237A78", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color=COL["grid"], lw=0.55, zorder=0)

    def size_from_r(r: float) -> float:
        return 28 + 150 * np.clip((float(r) - 0.15) / 0.35, 0, 1)

    for _, row in dat.iterrows():
        lr = str(row["LR"])
        selected = bool(row["is_selected"])
        color = LR_COLORS.get(lr, COL["muted"]) if selected else "#AFA79B"
        ax.scatter(
            row["expected_scale"],
            row["top_receiver_target_lift"],
            s=size_from_r(row["target_spearman_r"]),
            color=color,
            alpha=0.94 if selected else 0.64,
            edgecolor="white",
            linewidth=0.55,
            zorder=3 if selected else 2,
        )

    short = {
        "CCL5-CCR5": "CCL5",
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    label_lrs = ["CCL5-CCR5", "TGFB1-TGFBR2", "CXCL12-CXCR4", "CSF1-CSF1R", "VEGFA-KDR"]
    offsets = {
        "CCL5-CCR5": (0.10, 0.030, "left"),
        "TGFB1-TGFBR2": (0.10, 0.020, "left"),
        "CXCL12-CXCR4": (-0.18, 0.034, "right"),
        "CSF1-CSF1R": (0.08, 0.022, "left"),
        "VEGFA-KDR": (0.08, 0.020, "left"),
    }
    for lr in label_lrs:
        row = dat[dat["LR"].eq(lr)]
        if row.empty:
            continue
        row = row.iloc[0]
        dx, dy, ha = offsets.get(lr, (0.08, 0.02, "left"))
        ax.text(
            row["expected_scale"] + dx,
            row["top_receiver_target_lift"] + dy,
            short.get(lr, lr),
            fontsize=5.2,
            color=LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"],
            ha=ha,
            va="center",
        )

    ax.set_xlim(-0.05, 5.05)
    ymax = max(0.68, float(dat["top_receiver_target_lift"].max()) + 0.12)
    ax.set_ylim(0.00, ymax)
    ax.set_xticks([0, 2.5, 5.0])
    ax.set_xticklabels(["local", "mid", "broad"], fontsize=5.8)
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_xlabel("learned scale", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("target lift", fontsize=6.4, labelpad=1.2)
    ax.set_title("Scale-response", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.38, zorder=0)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def plot_empirical_range_target_coupling(ax: plt.Axes, lr_summary: pd.DataFrame, support: pd.DataFrame) -> None:
    dat = lr_summary.merge(
        support[["LR", "target_spearman_r", "top_receiver_target_lift", "top_vs_rest_mannwhitney_p"]],
        on="LR",
        how="inner",
    )
    dat = dat[~dat["mechanism"].astype(str).str.contains("negative_control", regex=False)].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    dat["is_selected"] = dat["LR"].isin(SELECTED_LRS)
    xmin = float(dat["top_median_distance"].min()) - 10
    xmax = float(dat["top_median_distance"].max()) + 10
    ax.axvspan(xmin, 95, color="#C8C3B8", alpha=0.10, lw=0, zorder=0)
    ax.axvspan(95, 145, color="#D7835C", alpha=0.08, lw=0, zorder=0)
    ax.axvspan(145, xmax, color="#237A78", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color=COL["grid"], lw=0.55, zorder=0)

    def size_from_r(r: float) -> float:
        return 28 + 140 * np.clip((float(r) - 0.15) / 0.35, 0, 1)

    for _, row in dat.iterrows():
        lr = str(row["LR"])
        selected = bool(row["is_selected"])
        color = LR_COLORS.get(lr, COL["muted"]) if selected else "#AFA79B"
        ax.scatter(
            row["top_median_distance"],
            row["top_receiver_target_lift"],
            s=size_from_r(row["target_spearman_r"]),
            color=color,
            alpha=0.95 if selected else 0.62,
            edgecolor="white",
            linewidth=0.55,
            zorder=3 if selected else 2,
        )

    short = {
        "CCL5-CCR5": "CCL5",
        "SPP1-CD44": "SPP1",
        "CSF1-CSF1R": "CSF1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    label_lrs = ["SPP1-CD44", "CSF1-CSF1R", "CXCL12-CXCR4", "TGFB1-TGFBR2", "VEGFA-KDR"]
    offsets = {
        "SPP1-CD44": (5, 0.020, "left"),
        "CSF1-CSF1R": (5, 0.020, "left"),
        "CXCL12-CXCR4": (-6, 0.038, "right"),
        "TGFB1-TGFBR2": (5, 0.018, "left"),
        "VEGFA-KDR": (5, 0.018, "left"),
    }
    for lr in label_lrs:
        row = dat[dat["LR"].eq(lr)]
        if row.empty:
            continue
        row = row.iloc[0]
        dx, dy, ha = offsets.get(lr, (5, 0.02, "left"))
        ax.text(
            row["top_median_distance"] + dx,
            row["top_receiver_target_lift"] + dy,
            short.get(lr, lr),
            fontsize=5.2,
            color=LR_COLORS.get(lr, COL["muted"]) if lr in SELECTED_LRS else COL["muted"],
            ha=ha,
            va="center",
        )
    ax.set_xlim(xmin, xmax)
    ymax = max(0.68, float(dat["top_receiver_target_lift"].max()) + 0.12)
    ax.set_ylim(0.00, ymax)
    ax.set_xticks([75, 125, 175])
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_xlabel("empirical range", fontsize=6.4, labelpad=1.0)
    ax.set_ylabel("target lift", fontsize=6.4, labelpad=1.2)
    ax.set_title("Range-response", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.38, zorder=0)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def plot_selected_target_lift(ax: plt.Axes, support: pd.DataFrame, hotspots: pd.DataFrame | None = None) -> None:
    dat = support[support["LR"].isin(FUNCTIONAL_LRS)].copy()
    dat["order"] = dat["LR"].map({lr: i for i, lr in enumerate(FUNCTIONAL_LRS)})
    dat = dat.sort_values("order")
    if dat.empty:
        ax.set_axis_off()
        return
    x_positions = np.array([0.0, 1.0])
    short = {
        "CXCL10-CXCR3": "CXCL10-CXCR3",
        "SPP1-CD44": "SPP1-CD44",
        "CXCL12-CXCR4": "CXCL12-CXCR4",
        "CXCL9-CXCR3": "CXCL9-CXCR3",
        "CCL5-CCR5": "CCL5-CCR5",
        "VEGFA-KDR": "VEGFA-KDR",
        "CSF1-CSF1R": "CSF1-CSF1R",
    }
    offsets = {
        "SPP1-CD44": -0.21,
        "CXCL10-CXCR3": -0.14,
        "CXCL12-CXCR4": -0.07,
        "CXCL9-CXCR3": 0.00,
        "CCL5-CCR5": 0.07,
        "VEGFA-KDR": 0.14,
        "CSF1-CSF1R": 0.21,
    }
    handles: list[Line2D] = []
    for row in dat.to_dict("records"):
        lr = str(row["LR"])
        color = LR_COLORS.get(lr, COL["ours"])
        primary = lr in SELECTED_LRS
        bg = float(row["background_target_mean"])
        hot = float(row["top_receiver_target_mean"])
        if hotspots is not None:
            lr_hot = hotspots[hotspots["LR"].eq(lr)].copy()
            if not lr_hot.empty:
                top = lr_hot[lr_hot["is_top_receiver"].astype(bool)]["target_score"].to_numpy(float)
                rest = lr_hot[~lr_hot["is_top_receiver"].astype(bool)]["target_score"].to_numpy(float)
                bg_sem = float(np.nanstd(rest) / np.sqrt(max(1, np.isfinite(rest).sum()))) if len(rest) else 0.0
                hot_sem = float(np.nanstd(top) / np.sqrt(max(1, np.isfinite(top).sum()))) if len(top) else 0.0
            else:
                bg_sem = hot_sem = 0.0
        else:
            bg_sem = hot_sem = 0.0
        xpos = x_positions + offsets.get(lr, 0.0)
        means = np.array([bg, hot], dtype=float)
        sems = np.array([bg_sem, hot_sem], dtype=float)
        ax.errorbar(
            xpos,
            means,
            yerr=sems,
            color=color,
            lw=0.96 if primary else 0.72,
            linestyle=(0, (2.0, 2.0)),
            marker="o",
            markersize=5.4 if primary else 4.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.25 if primary else 0.95,
            capsize=2.2,
            elinewidth=0.98 if primary else 0.78,
            alpha=0.98 if primary else 0.68,
            zorder=4 if primary else 3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=0.96 if primary else 0.72,
                linestyle=(0, (2.0, 2.0)),
                marker="o",
                markersize=5.2 if primary else 4.4,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.15 if primary else 0.9,
                alpha=0.98 if primary else 0.68,
                label=short.get(lr, lr),
            )
        )

    ax.axhline(0, color="#8B8F94", lw=0.58, linestyle=(0, (3.0, 3.0)), zorder=0)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0, max(0.75, max(float(dat["top_receiver_target_mean"].max()), float(dat["background_target_mean"].max())) + 0.20))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["background", "hotspot"], fontsize=5.9)
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_ylabel("target signature", fontsize=6.4, labelpad=1.2)
    ax.set_title("Hotspot response", fontsize=7.2, pad=2.0, color=COL["ink"])
    clean_axis(ax, grid=False)
    ax.grid(axis="y", color="#E1E6EA", linewidth=0.34, zorder=0)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        fontsize=4.75,
        handlelength=1.35,
        handletextpad=0.42,
        borderaxespad=0.0,
        labelspacing=0.22,
    )
    ax.spines["left"].set_color("#15191C")
    ax.spines["bottom"].set_color("#15191C")
    ax.spines["left"].set_linewidth(0.72)
    ax.spines["bottom"].set_linewidth(0.72)


def plot_distance_activity_atlas(ax: plt.Axes, distance: pd.DataFrame, lr_summary: pd.DataFrame) -> None:
    summary = lr_summary[~lr_summary["mechanism"].astype(str).str.contains("negative_control", regex=False)].copy()
    summary = summary.sort_values("expected_scale")
    lr_order = summary["LR"].tolist()
    dat = distance[distance["LR"].isin(lr_order)].copy()
    if dat.empty:
        ax.set_axis_off()
        return

    pivot = (
        dat.pivot_table(index="LR", columns="bin_mid", values="mean_score_norm_by_lr", aggfunc="mean")
        .reindex(lr_order)
        .sort_index(axis=1)
    )
    values = pivot.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("distance_atlas", ["#F7F4EC", "#E6C06A", "#C94F48"])
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    short = {
        "CCL5-CCR5": "CCL5",
        "CXCL9-CXCR3": "CXCL9",
        "CXCL10-CXCR3": "CXCL10",
        "CSF1-CSF1R": "CSF1",
        "SPP1-CD44": "SPP1",
        "CD274-PDCD1": "PD-L1",
        "TGFB1-TGFBR2": "TGFB1",
        "VEGFA-KDR": "VEGFA",
        "CXCL12-CXCR4": "CXCL12",
    }
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([short.get(lr, lr) for lr in pivot.index], fontsize=5.7)
    class_colors = {"local": "#8A8177", "intermediate": "#C94F48", "broad": "#237A78"}
    class_by_lr = summary.set_index("LR")["scale_class"].to_dict()
    for tick, lr in zip(ax.get_yticklabels(), pivot.index):
        tick.set_color(class_colors.get(class_by_lr.get(lr), COL["text"]))

    mids = pivot.columns.to_numpy(float)
    if len(mids) >= 3:
        xticks = [0, len(mids) // 2, len(mids) - 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{mids[i]:.0f}" for i in xticks], fontsize=6.0)
    ax.set_xlabel("edge distance", fontsize=6.8, labelpad=1.5)
    ax.set_title("Distance activity atlas", fontsize=7.4, pad=5.0, color=COL["ink"])
    ax.tick_params(axis="both", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = ax.inset_axes([0.62, 1.005, 0.34, 0.038])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=5.2, length=0, pad=1.0, colors=COL["muted"])
    cax.text(-0.06, 0.5, "relative activity", transform=cax.transAxes, ha="right", va="center", fontsize=5.2, color=COL["muted"])


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.tick_params(labelsize=6.1, length=2.0, pad=1.5, colors=COL["muted"])
    if grid:
        ax.grid(axis="y", color=COL["grid"], linewidth=0.40, zorder=0)


def strip_spatial_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_receiver_celltype_burden_heatmap(ax: plt.Axes, burden: pd.DataFrame) -> None:
    if burden.empty:
        ax.set_axis_off()
        return
    keep = [lr for lr in SELECTED_LRS + ["TGFB1-TGFBR2", "VEGFA-KDR", "CSF1-CSF1R"] if lr in set(burden["LR"])]
    dat = burden.set_index("LR").loc[keep].copy()
    cols = [col for col in ["T cell", "tumor", "endothelial", "myeloid", "immune", "stromal"] if col in dat.columns]
    dat = dat[cols]
    values = dat.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("cosmx_receiver_burden_red", ["#FFFFFF", "#F7D8D5", "#C94F48", "#8F3431"])
    vmax = max(0.05, float(np.nanmax(values)))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(dat.index)))
    ax.set_yticklabels(dat.index, fontsize=5.2)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", rotation_mode="anchor", fontsize=5.2)
    ax.tick_params(length=0, pad=1.0)
    ax.set_title("Receiver cell-type burden", fontsize=7.2, color=COL["ink"], pad=2.0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dat.index), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.58)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick, lr in zip(ax.get_yticklabels(), dat.index, strict=False):
        if lr in SELECTED_LRS:
            tick.set_fontweight("bold")
            tick.set_color("#B85450")
    cax = ax.inset_axes([0.78, 0.84, 0.18, 0.050])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.95, length=0, pad=1.0, colors=COL["muted"])
    cax.text(-0.08, 0.50, "fraction", transform=cax.transAxes, ha="right", va="center", fontsize=5.0, color=COL["muted"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#15191C")
        spine.set_linewidth(0.58)


def plot_scale_vs_noscale_pairs(ax: plt.Axes, scale_vs_noscale: pd.DataFrame) -> None:
    dat = scale_vs_noscale[scale_vs_noscale["LR"].isin(SELECTED_LRS + ["VEGFA-KDR", "TGFB1-TGFBR2"])].copy()
    if dat.empty:
        ax.set_axis_off()
        return
    wide = dat.pivot_table(index="LR", columns="method", values="target_spearman_r", aggfunc="mean")
    if {"scale-aware", "no-scale"} - set(wide.columns):
        ax.set_axis_off()
        return
    wide = wide.dropna(subset=["scale-aware", "no-scale"]).copy()
    wide["delta"] = wide["scale-aware"] - wide["no-scale"]
    wide = wide.sort_values("delta", ascending=True)
    y = np.arange(len(wide))
    for yi, (lr, row) in enumerate(wide.iterrows()):
        color = LR_COLORS.get(lr, COL["blue_dark"])
        ax.plot(
            [row["no-scale"], row["scale-aware"]],
            [yi, yi],
            color=mpl.colors.to_rgba(color, 0.42),
            lw=1.15,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(row["no-scale"], yi, s=23, facecolor="white", edgecolor=color, linewidth=0.72, zorder=3)
        ax.scatter(row["scale-aware"], yi, s=28, facecolor=color, edgecolor="white", linewidth=0.48, zorder=4)
    short = {
        "CXCL12-CXCR4": "CXCL12",
        "CXCL10-CXCR3": "CXCL10",
        "VEGFA-KDR": "VEGFA",
        "SPP1-CD44": "SPP1",
        "TGFB1-TGFBR2": "TGFB1",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(lr, lr) for lr in wide.index], fontsize=5.2)
    for tick, lr in zip(ax.get_yticklabels(), wide.index, strict=False):
        if lr in SELECTED_LRS:
            tick.set_fontweight("bold")
            tick.set_color(LR_COLORS.get(lr, COL["ink"]))
    xmin = float(np.nanmin(wide[["scale-aware", "no-scale"]].to_numpy())) - 0.035
    xmax = float(np.nanmax(wide[["scale-aware", "no-scale"]].to_numpy())) + 0.035
    ax.set_xlim(max(0, xmin), min(0.55, xmax))
    ax.set_xlabel("target Spearman r", fontsize=6.2, labelpad=1.0)
    ax.set_title("Scale-aware target support", fontsize=7.2, color=COL["ink"], pad=2.0)
    ax.tick_params(axis="both", labelsize=5.4, length=2.0, width=0.48, pad=1.2)
    clean_axis(ax)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.34, zorder=0)
    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor=COL["muted"], markersize=3.8, label="no-scale"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=COL["green_dark"], markeredgecolor="white", markersize=4.0, label="scale-aware"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.2, handletextpad=0.35, borderaxespad=0.2)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.075, y: float = 1.045) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=COL["ink"])


def export(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=900, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
