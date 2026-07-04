#!/usr/bin/env python
"""Panel-first DSS MERFISH figure for SpatialSelf biological validation and comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (
    BASE,
    INFLAMMATORY_AXES,
    METHOD_COLORS,
    METHODS,
    THREE_METHOD,
    aggregate_receiver_scores,
    broad_compartment,
    load_expression_slice,
    load_slice,
    normalize_extent,
    set_spatial_limits,
)


OUT = BASE / "figures/merfish_dss_main_figure_v7_vector_spots"
PANEL_OUT = OUT / "panels"

SPATIAL_STAGES = ["Healthy_D0", "DSS9", "DSS21"]
SPATIAL_LABELS = ["D0", "D9", "D21"]
ALL_STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
ALL_LABELS = ["D0", "D3", "D9", "D21"]
LIGANDS = ["Cxcl10", "Ccl5", "Ccl2"]
RECEPTORS = ["Cxcr3", "Ccr5", "Ccr2"]

INK = "#202020"
MUTED = "#6E6A62"
TISSUE = "#D8D2C7"
GRID = "#ECE7DE"

COMPARTMENT_COLORS = {
    "Epithelial": "#D49335",
    "Immune": "#B84E68",
    "Mesenchymal": "#519C8D",
    "Vascular": "#5D7FA6",
    "Neural": "#8B7BA4",
    "Other": "#BBB5AA",
}

WARM = mpl.colors.LinearSegmentedColormap.from_list("warm_field", ["#fffaf0", "#e8b66f", "#bf563e", "#5f1518"])
TEAL = mpl.colors.LinearSegmentedColormap.from_list("teal_field", ["#f8f5eb", "#b8d0c6", "#4f9489", "#14545b"])
PURPLE = mpl.colors.LinearSegmentedColormap.from_list("purple_coloc", ["#fbf6ec", "#d6bd79", "#af665e", "#50304c"])
PROGRAM = mpl.colors.LinearSegmentedColormap.from_list("program", ["#f5eee3", "#e2bd81", "#bd6844", "#7a2b28"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.8,
            "axes.titlesize": 7.7,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export(fig: plt.Figure, stem: Path, dpi: int = 1200) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.026, dy: float = 0.010) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=INK)


def panel_title(fig: plt.Figure, ax: plt.Axes, title: str, dx: float = 0.0, dy: float = 0.030) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, title, ha="left", va="bottom", fontsize=7.7, color=INK)


def field_grid(cells: pd.DataFrame, score: np.ndarray, bins: int = 320, sigma: float = 2.8, mean_field: bool = False):
    x0, x1, y0, y1 = normalize_extent(cells)
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    density, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]])
    weighted, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]], weights=score)
    density_s = gaussian_filter(density.T, sigma=sigma)
    weighted_s = gaussian_filter(weighted.T, sigma=sigma)
    tissue = density_s / max(density_s.max(), 1e-9)
    if mean_field:
        field = weighted_s / (density_s + 1e-8)
    else:
        field = weighted_s
    field[tissue < 0.015] = np.nan
    return field, tissue, (x0, x1, y0, y1)


def sample_grid_at_cells(cells: pd.DataFrame, field: np.ndarray, extent) -> np.ndarray:
    x0, x1, y0, y1 = extent
    h, w = field.shape
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    xi = np.clip(np.rint((x - x0) / max(x1 - x0, 1e-9) * (w - 1)).astype(int), 0, w - 1)
    yi = np.clip(np.rint((y - y0) / max(y1 - y0, 1e-9) * (h - 1)).astype(int), 0, h - 1)
    values = field[yi, xi]
    values = np.where(np.isfinite(values), values, 0.0)
    return values


def draw_tissue_context(ax: plt.Axes, cells: pd.DataFrame, tissue: np.ndarray, extent) -> None:
    ax.scatter(cells["x"], cells["y"], s=0.52, color="#D8D2C8", alpha=0.27, linewidths=0)
    try:
        outline = gaussian_filter(np.nan_to_num(tissue), sigma=5.2)
        ax.contour(
            outline,
            levels=[0.026],
            origin="lower",
            extent=extent,
            colors=["#C8C0B4"],
            linewidths=0.30,
            alpha=0.58,
        )
    except ValueError:
        pass


def draw_spot_score_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    score: np.ndarray,
    tissue: np.ndarray,
    extent,
    cmap,
    vmax: float,
    low_clip: float = 0.055,
    size_base: float = 3.2,
    size_scale: float = 12.0,
) -> None:
    draw_tissue_context(ax, cells, tissue, extent)
    values = np.asarray(score, dtype=float)
    norm = np.clip(values / max(vmax, 1e-9), 0, 1)
    mask = np.isfinite(norm) & (norm >= low_clip)
    if np.any(mask):
        sizes = size_base + size_scale * np.power(norm[mask], 0.62)
        ax.scatter(
            cells.loc[mask, "x"],
            cells.loc[mask, "y"],
            c=norm[mask],
            s=sizes,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=0.94,
            marker="o",
            edgecolors="white",
            linewidths=0.10,
            zorder=3,
        )
    set_spatial_limits(ax, cells)


def draw_field(
    ax: plt.Axes,
    cells: pd.DataFrame,
    field: np.ndarray,
    tissue: np.ndarray,
    extent,
    cmap,
    vmax: float,
    alpha_power: float = 0.42,
    low_clip: float = 0.020,
) -> None:
    ax.scatter(cells["x"], cells["y"], s=0.26, color="#D9D2C8", alpha=0.13, linewidths=0)
    ax.imshow(np.ones_like(tissue), origin="lower", extent=extent, cmap=mpl.colors.ListedColormap([TISSUE]), alpha=tissue * 0.055, interpolation="nearest")
    try:
        outline = gaussian_filter(np.nan_to_num(tissue), sigma=5.0)
        ax.contour(
            outline,
            levels=[0.026],
            origin="lower",
            extent=extent,
            colors=["#CFC7BB"],
            linewidths=0.24,
            alpha=0.48,
        )
    except ValueError:
        pass
    norm = np.clip(field / max(vmax, 1e-9), 0, 1)
    alpha = np.where(np.isfinite(norm), np.clip((norm**alpha_power) * 0.98, 0, 0.98), 0)
    alpha[norm < low_clip] = 0
    ax.imshow(norm, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
    set_spatial_limits(ax, cells)


def add_low_high_bar(fig: plt.Figure, ax: plt.Axes, cmap, label: str = "") -> None:
    cax = ax.inset_axes([0.58, -0.105, 0.32, 0.052])
    n = 96
    for i in range(n):
        cax.add_patch(Rectangle((i / n, 0), 1 / n, 1, facecolor=cmap(i / (n - 1)), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.tick_params(length=0, pad=1.0, labelsize=5.6)
    if label:
        cax.set_xlabel(label, fontsize=5.8, labelpad=0.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)


def draw_matrix_rects(ax: plt.Axes, matrix: np.ndarray, cmap, norm=None, vmin: float = 0.0, vmax: float = 1.0) -> None:
    values = np.asarray(matrix, dtype=float)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(values[i, j])),
                    edgecolor="white",
                    linewidth=0.65,
                )
            )
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)


def expression_panels():
    panels = []
    for stage in SPATIAL_STAGES:
        cells = load_expression_slice(stage)
        cells = cells.copy()
        ligand_genes = [g for g in LIGANDS if g in cells.columns]
        receptor_genes = [g for g in RECEPTORS if g in cells.columns]
        cells["ligand_program"] = np.log1p(cells[ligand_genes].sum(axis=1)) if ligand_genes else 0.0
        cells["receptor_program"] = np.log1p(cells[receptor_genes].sum(axis=1)) if receptor_genes else 0.0
        ligand, tissue, extent = field_grid(cells, cells["ligand_program"].to_numpy(), bins=340, sigma=2.6, mean_field=True)
        receptor, _, _ = field_grid(cells, cells["receptor_program"].to_numpy(), bins=340, sigma=2.6, mean_field=True)
        panels.append({"cells": cells, "ligand": ligand, "receptor": receptor, "tissue": tissue, "extent": extent})
    ligand_vals = np.concatenate([p["ligand"][np.isfinite(p["ligand"])] for p in panels])
    receptor_vals = np.concatenate([p["receptor"][np.isfinite(p["receptor"])] for p in panels])
    ligand_vmax = max(float(np.nanquantile(ligand_vals, 0.990)), 1e-6)
    receptor_vmax = max(float(np.nanquantile(receptor_vals, 0.990)), 1e-6)
    for p in panels:
        ln = np.clip(p["ligand"] / ligand_vmax, 0, 1)
        rn = np.clip(p["receptor"] / receptor_vmax, 0, 1)
        p["coloc"] = np.sqrt(ln * rn)
        p["ligand_cell"] = sample_grid_at_cells(p["cells"], p["ligand"], p["extent"])
        p["receptor_cell"] = sample_grid_at_cells(p["cells"], p["receptor"], p["extent"])
        p["coloc_cell"] = sample_grid_at_cells(p["cells"], p["coloc"], p["extent"])
    coloc_vals = np.concatenate([p["coloc"][np.isfinite(p["coloc"])] for p in panels])
    coloc_vmax = max(float(np.nanquantile(coloc_vals, 0.975)), 1e-6)
    return panels, ligand_vmax, receptor_vmax, coloc_vmax


def plot_panel_a(fig: plt.Figure, gs, label: str | None = "a"):
    axes = []
    for i, (stage, stage_label) in enumerate(zip(SPATIAL_STAGES, SPATIAL_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        _, cells, _ = load_slice(stage)
        cells = cells.copy()
        cells["compartment"] = cells["cell_type"].map(broad_compartment)
        for comp, color in COMPARTMENT_COLORS.items():
            sub = cells.loc[cells["compartment"] == comp]
            if not sub.empty:
                ax.scatter(sub["x"], sub["y"], s=5.2, color=color, alpha=0.92, linewidths=0)
        set_spatial_limits(ax, cells)
        ax.set_title(stage_label, fontsize=7.6, pad=2.0)
        axes.append(ax)
    if label:
        panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "DSS MERFISH tissue context", dy=0.034)
    handles = [Line2D([0], [0], marker="o", lw=0, markersize=3.8, color=COMPARTMENT_COLORS[k], label=k) for k in ["Epithelial", "Immune", "Mesenchymal", "Vascular", "Neural"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(axes[1].get_position().x0 + 0.03, axes[0].get_position().y1 + 0.060), ncol=5, frameon=False, handletextpad=0.25, columnspacing=0.85, fontsize=6.1)
    return axes


def plot_panel_b(fig: plt.Figure, gs, panels=None, ligand_vmax=None, receptor_vmax=None, label: str | None = "b"):
    if panels is None:
        panels, ligand_vmax, receptor_vmax, _ = expression_panels()
    axes = []
    for row, (name, key, cmap, vmax) in enumerate(
        [
            ("ligands\nCxcl10/Ccl5/Ccl2", "ligand", WARM, ligand_vmax),
            ("receptors\nCxcr3/Ccr5/Ccr2", "receptor", TEAL, receptor_vmax),
        ]
    ):
        for col, (stage_label, p) in enumerate(zip(SPATIAL_LABELS, panels)):
            ax = fig.add_subplot(gs[row, col])
            draw_spot_score_map(
                ax,
                p["cells"],
                p[f"{key}_cell"],
                p["tissue"],
                p["extent"],
                cmap,
                vmax,
                low_clip=0.055,
                size_base=2.8,
                size_scale=10.5,
            )
            if row == 0:
                ax.set_title(stage_label, fontsize=7.4, pad=1.8)
            if col == 0:
                ax.text(-0.070, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=6.8, color=INK)
            axes.append(ax)
    if label:
        panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "Inflammatory chemokine ligand and receptor fields", dy=0.034)
    return axes


def plot_panel_c(fig: plt.Figure, gs, panels=None, coloc_vmax=None, label: str | None = "c"):
    if panels is None:
        panels, _, _, coloc_vmax = expression_panels()
    axes = []
    for i, (stage_label, p) in enumerate(zip(SPATIAL_LABELS, panels)):
        ax = fig.add_subplot(gs[0, i])
        draw_spot_score_map(
            ax,
            p["cells"],
            p["coloc_cell"],
            p["tissue"],
            p["extent"],
            PURPLE,
            coloc_vmax,
            low_clip=0.070,
            size_base=3.5,
            size_scale=13.5,
        )
        ax.set_title(stage_label, fontsize=7.8, pad=2.0)
        axes.append(ax)
    if label:
        panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "Spatial ligand-receptor co-localization niche", dy=0.034)
    add_low_high_bar(fig, axes[-1], PURPLE)
    return axes


def communication_panels():
    cells_by_stage = [aggregate_receiver_scores(stage, INFLAMMATORY_AXES, "SpatialSelf") for stage in SPATIAL_STAGES]
    maps = []
    for cells in cells_by_stage:
        field, tissue, extent = field_grid(cells, cells["incoming_score"].to_numpy(), bins=340, sigma=2.7, mean_field=False)
        maps.append({"cells": cells, "score": cells["incoming_score"].to_numpy(float), "field": field, "tissue": tissue, "extent": extent})
    vals = np.concatenate([m["score"][np.isfinite(m["score"]) & (m["score"] > 0)] for m in maps])
    vmax = max(float(np.nanquantile(vals, 0.980)), 1e-6)
    return maps, vmax


def plot_panel_d(fig: plt.Figure, gs, comm_maps=None, comm_vmax=None, label: str | None = "d"):
    if comm_maps is None:
        comm_maps, comm_vmax = communication_panels()
    axes = []
    for i, (stage_label, p) in enumerate(zip(SPATIAL_LABELS, comm_maps)):
        ax = fig.add_subplot(gs[0, i])
        draw_spot_score_map(
            ax,
            p["cells"],
            p["score"],
            p["tissue"],
            p["extent"],
            WARM,
            comm_vmax,
            low_clip=0.060,
            size_base=3.3,
            size_scale=14.0,
        )
        ax.set_title(stage_label, fontsize=7.8, pad=2.0)
        axes.append(ax)
    if label:
        panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "SpatialSelf inflammatory chemokine communication activity", dy=0.034)
    add_low_high_bar(fig, axes[-1], WARM)
    return axes


def plot_panel_e(fig: plt.Figure, gs, label: str | None = "e"):
    df = pd.read_csv(THREE_METHOD)
    axes = []
    for i, lr_pair in enumerate(INFLAMMATORY_AXES):
        ax = fig.add_subplot(gs[0, i])
        endpoint_y: dict[str, float] = {}
        for method in METHODS:
            rec = df.loc[(df["method"] == method) & (df["lr_pair"] == lr_pair)]
            if rec.empty:
                continue
            vals = rec[ALL_STAGES].iloc[0].to_numpy(float)
            ax.plot(np.arange(len(ALL_LABELS)), vals, color=METHOD_COLORS[method], lw=1.25, marker="o", ms=2.8, label=method)
            endpoint_y[method] = float(vals[-1])
        ax.set_title(lr_pair, fontsize=7.0, pad=2.0)
        ax.set_xticks(np.arange(len(ALL_LABELS)))
        ax.set_xticklabels(ALL_LABELS)
        ax.set_xlim(-0.15, 3.58 if i == len(INFLAMMATORY_AXES) - 1 else 3.15)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_ylabel("stage score")
        else:
            ax.set_yticklabels([])
        if i == len(INFLAMMATORY_AXES) - 1:
            label_offsets = {"SpatialSelf": -0.035, "COMMOT": 0.012, "CellNEST": 0.000}
            for method in METHODS:
                if method not in endpoint_y:
                    continue
                ax.text(
                    3.18,
                    endpoint_y[method] + label_offsets.get(method, 0.0),
                    method,
                    color=METHOD_COLORS[method],
                    fontsize=5.8,
                    ha="left",
                    va="center",
                    clip_on=False,
                )
        axes.append(ax)
    if label:
        panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "Method trajectories", dy=0.034)
    return axes


def specificity_table() -> pd.DataFrame:
    df = pd.read_csv(THREE_METHOD)
    rows = []
    for method in METHODS:
        for lr in INFLAMMATORY_AXES:
            rec = df.loc[(df["method"] == method) & (df["lr_pair"] == lr)]
            if rec.empty:
                continue
            values = rec[ALL_STAGES].iloc[0]
            rows.append({"method": method, "lr_pair": lr, "specificity": float(values["DSS9"] - values.drop("DSS9").max())})
    return pd.DataFrame(rows)


def plot_panel_f(fig: plt.Figure, ax: plt.Axes, label: str | None = "f"):
    spec = specificity_table()
    matrix = (
        spec.pivot(index="method", columns="lr_pair", values="specificity")
        .reindex(index=METHODS, columns=INFLAMMATORY_AXES)
        .to_numpy(float)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "d9_specificity",
        ["#305F72", "#E9E0D2", "#A33A32"],
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.52, vcenter=0.0, vmax=0.055)
    draw_matrix_rects(ax, matrix, cmap, norm=norm)
    ax.set_xticks(range(len(INFLAMMATORY_AXES)))
    ax.set_xticklabels([s.replace("-", "\n") for s in INFLAMMATORY_AXES])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.tick_params(length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, len(INFLAMMATORY_AXES), 1):
        ax.axvline(x, color="white", lw=0.75)
    for y in np.arange(-0.5, len(METHODS), 1):
        ax.axhline(y, color="white", lw=0.75)
    ax.set_title("D9 specificity", loc="left", pad=4.0)
    cax = ax.inset_axes([1.035, 0.10, 0.050, 0.76])
    n = 96
    vals = np.linspace(-0.52, 0.055, n)
    for i, val in enumerate(vals):
        cax.add_patch(Rectangle((0, i / n), 1, 1 / n, facecolor=cmap(norm(val)), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([0, 0.5, 1])
    cax.set_yticklabels(["other", "0", "D9"])
    cax.tick_params(length=0, pad=1.0, labelsize=5.4)
    cax.yaxis.tick_right()
    cax.set_ylabel("D9 - best other", fontsize=5.5, labelpad=1.0, color=MUTED)
    cax.yaxis.set_label_position("right")
    for spine in cax.spines.values():
        spine.set_visible(False)
    if label:
        panel_label(fig, ax, label, dx=-0.030)
    return ax


def render_single_panels() -> None:
    panels, ligand_vmax, receptor_vmax, coloc_vmax = expression_panels()
    comm_maps, comm_vmax = communication_panels()

    fig = plt.figure(figsize=(6.20, 2.18))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.07, top=0.84, wspace=0.040)
    plot_panel_a(fig, gs)
    export(fig, PANEL_OUT / "panel_a_tissue_context")

    fig = plt.figure(figsize=(6.20, 3.36))
    gs = fig.add_gridspec(2, 3, left=0.155, right=0.985, bottom=0.055, top=0.88, wspace=0.040, hspace=0.070)
    plot_panel_b(fig, gs, panels, ligand_vmax, receptor_vmax)
    export(fig, PANEL_OUT / "panel_b_ligand_receptor_fields")

    fig = plt.figure(figsize=(6.20, 2.35))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.095, top=0.83, wspace=0.050)
    plot_panel_c(fig, gs, panels, coloc_vmax)
    export(fig, PANEL_OUT / "panel_c_colocalization_niche")

    fig = plt.figure(figsize=(6.20, 2.35))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.095, top=0.83, wspace=0.050)
    plot_panel_d(fig, gs, comm_maps, comm_vmax)
    export(fig, PANEL_OUT / "panel_d_spatialself_activity")

    fig = plt.figure(figsize=(5.20, 2.15))
    gs = fig.add_gridspec(1, 3, left=0.095, right=0.985, bottom=0.18, top=0.70, wspace=0.36)
    plot_panel_e(fig, gs)
    export(fig, PANEL_OUT / "panel_e_method_trajectories")

    fig, ax = plt.subplots(figsize=(3.00, 1.92))
    fig.subplots_adjust(left=0.23, right=0.90, bottom=0.20, top=0.82)
    plot_panel_f(fig, ax)
    export(fig, PANEL_OUT / "panel_f_d9_specificity")


def render_composite() -> None:
    panels, ligand_vmax, receptor_vmax, coloc_vmax = expression_panels()
    comm_maps, comm_vmax = communication_panels()
    fig = plt.figure(figsize=(7.20, 9.65))
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.28, 2.25, 1.68, 1.68, 1.58],
        left=0.055,
        right=0.970,
        bottom=0.060,
        top=0.940,
        hspace=0.52,
    )
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[4.1, 2.1], wspace=0.34)
    plot_panel_a(fig, top[0, 0].subgridspec(1, 3, wspace=0.035))
    ax_b0 = fig.add_subplot(top[0, 1])
    # Compact program heatmap inside panel b in the top right.
    traj = pd.read_csv(BASE / "progression_story/scipilot_story_v2_program_activity.csv")
    matrix = traj.pivot(index="program", columns="stage_label", values="relative_activity").loc[
        ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"],
        ALL_LABELS,
    ]
    draw_matrix_rects(ax_b0, matrix.to_numpy(), PROGRAM, vmin=0, vmax=1)
    ax_b0.set_xticks(range(4))
    ax_b0.set_xticklabels(ALL_LABELS)
    ax_b0.set_yticks(range(4))
    ax_b0.set_yticklabels(["homeostasis", "early", "inflam.", "repair"])
    ax_b0.tick_params(length=0, pad=1.8)
    for spine in ax_b0.spines.values():
        spine.set_visible(False)
    ax_b0.set_title("Communication programs", loc="left", pad=4.5)
    panel_label(fig, ax_b0, "b")

    gs_bio = outer[1, 0].subgridspec(2, 3, wspace=0.040, hspace=0.060)
    plot_panel_b(fig, gs_bio, panels, ligand_vmax, receptor_vmax, label="c")

    gs_coloc = outer[2, 0].subgridspec(1, 3, wspace=0.048)
    plot_panel_c(fig, gs_coloc, panels, coloc_vmax, label="d")

    gs_comm = outer[3, 0].subgridspec(1, 3, wspace=0.048)
    plot_panel_d(fig, gs_comm, comm_maps, comm_vmax, label="e")

    bottom = outer[4, 0].subgridspec(1, 2, width_ratios=[4.55, 1.85], wspace=0.34)
    plot_panel_e(fig, bottom[0, 0].subgridspec(1, 3, wspace=0.34), label="f")
    ax_g = fig.add_subplot(bottom[0, 1])
    plot_panel_f(fig, ax_g, label="g")

    export(fig, OUT / "full_figure_biovalidation_method_comparison")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    render_single_panels()
    render_composite()
    print(OUT)


if __name__ == "__main__":
    main()
