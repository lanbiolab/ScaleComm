#!/usr/bin/env python
"""Redraw manuscript experimental Fig2-Fig5 with one consistent visual system.

This script intentionally leaves outputs/manuscript_experiment_figures_fused_v1
untouched. It reads existing source tables only and writes a fresh v2 package.
"""

from __future__ import annotations

import json
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/manuscript_experiment_figures_fused_v2"
FIG_DIR = OUT / "figures"
QA_DIR = OUT / "qa"
SPEC_PATH = OUT / "figure_specs.json"

CRC_SRC = ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/source_data"
CRC_ART = ROOT / "outputs/article_style_figure_additions/crc_visium_hd_article_style/source_data"
DSS_SRC = ROOT / "outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v21_scale_baseline/source_tables"
DSS_ART = ROOT / "outputs/article_style_figure_additions/dss_merfish_article_style/source_data"
DSS_BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
COSMX_SRC = ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables"
COSMX_ART = ROOT / "outputs/article_style_figure_additions/cosmx_nsclc_article_style/source_data"
TONSIL_SRC = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story/source_data"
TONSIL_ART = ROOT / "outputs/article_style_figure_additions/tonsil_article_style/source_data"
HP_SRC = ROOT / "outputs/article_style_figure_additions/high_priority_remaining/source_data"
TONSIL_UI = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story/figures/ScaleComm (2).png"


COL = {
    "ink": "#171717",
    "text": "#2D2A27",
    "muted": "#6E665B",
    "grid": "#E7E1D8",
    "axis": "#333333",
    "bg": "#FFFFFF",
    "paper": "#FFFFFF",
    "tissue": "#D9D5CC",
    "tissue_dark": "#BDB8AE",
    "signal": "#B85450",
    "signal_dark": "#7A2D2A",
    "signal_soft": "#FBC9C4",
    "blue": "#2F78A8",
    "blue_soft": "#B9E5FA",
    "teal": "#237A78",
    "green": "#93CC82",
    "amber": "#D7B56D",
    "sand": "#F3D78A",
    "cream": "#FFEEBC",
    "slate": "#B9C1D1",
    "purple": "#7E6C9A",
    "grey": "#8A8177",
    "pale": "#F7F4EC",
}

CELL_COLORS = {
    "tumor": COL["slate"],
    "tumor_proxy": COL["slate"],
    "epithelial": "#D8D0C1",
    "epithelial_proxy": "#D8D0C1",
    "immune": COL["blue_soft"],
    "immune_enriched": COL["blue_soft"],
    "t_cell_proxy": "#86C8EE",
    "T cell": "#86C8EE",
    "T_cell_proxy": "#86C8EE",
    "B_cell_proxy": COL["signal_soft"],
    "B cell": COL["signal_soft"],
    "FDC_proxy": "#E4A7A1",
    "Tfh_proxy": "#A9D7EF",
    "myeloid": "#F0DFA8",
    "myeloid_proxy": "#F0DFA8",
    "myeloid_stromal_proxy": "#E5D7AC",
    "stromal": COL["green"],
    "stromal_proxy": COL["green"],
    "caf_proxy": "#A8C980",
    "CAF": "#A8C980",
    "mesenchymal": COL["green"],
    "vascular": "#99A8C6",
    "endothelial_proxy": "#99A8C6",
    "endothelial": "#99A8C6",
    "germinal_center_proxy": COL["signal"],
    "T_zone_proxy": "#86C8EE",
    "Other": "#C8C1B8",
}

CONTEXT_CMAP = LinearSegmentedColormap.from_list("context_signal", [COL["pale"], COL["signal_soft"], COL["signal"]])
TARGET_CMAP = LinearSegmentedColormap.from_list("target_blue", [COL["pale"], COL["blue_soft"], COL["blue"]])
HEAT_CMAP = LinearSegmentedColormap.from_list("paper_heat", ["#E6C06A", "#F7F4EC", "#2F78A8"])
RED_CMAP = LinearSegmentedColormap.from_list("paper_red", [COL["pale"], "#E7C6C2", COL["signal"]])
GOLD_CMAP = LinearSegmentedColormap.from_list("paper_gold", [COL["pale"], "#F3D78A", "#B85450"])
PURPLE_CMAP = LinearSegmentedColormap.from_list("paper_purple", [COL["pale"], "#D7CCE2", COL["purple"]])


FIG_SPECS = {
    "fig2": {
        "claim": "In CRC Visium HD, ScaleComm localizes TGFB1-TGFBR2 communication to the tumor-stromal context and aligns with downstream TGF-beta/ECM response better than comparison maps.",
        "panels": [
            "spatial context",
            "local LR abundance",
            "ScaleComm hotspot",
            "receiver response map",
            "response alignment",
            "target gene heatmap",
            "COMMOT map",
            "CellNEST map",
            "sender/receiver layers",
            "cell-type pair network",
        ],
    },
    "fig3": {
        "claim": "In DSS MERFISH time course, ScaleComm recovers stage-specific communication programs spanning homeostasis, inflammatory peak, and repair/remodeling.",
        "panels": [
            "cell-type maps",
            "communication regions",
            "stage top LR axes",
            "program heatmap",
            "temporal LR dynamics",
            "compartment flow rewiring",
            "DSS9 receiver activity",
            "DSS9 local LR ranking",
            "DSS21 local LR ranking",
            "replicate consistency",
        ],
    },
    "fig4": {
        "claim": "In CosMx NSCLC, the learned scale/context gate yields LR-specific spatial profiles and improves functional receiver-response support.",
        "panels": [
            "context map",
            "LR scale atlas",
            "representative scale profiles",
            "scale-aware response gain",
            "context-specific receiver maps",
            "target-response support",
            "receiver cell-type burden",
            "prioritized context LR axes",
        ],
    },
    "fig5": {
        "claim": "In SPAMOSAIC tonsil, ScaleComm localizes CXCL13-CXCR5 receiver activity to GC/B-cell regions with target response, local rank, replicate, and browser support.",
        "panels": [
            "tonsil context",
            "CXCL13/CXCR5 availability",
            "ScaleComm receiver hotspot",
            "GC response",
            "sender-receiver cell pairs",
            "replicate support",
            "GC-localized rank",
            "LR scale profile",
            "interactive browser",
        ],
    },
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.1,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.3,
            "axes.linewidth": 0.55,
            "axes.edgecolor": COL["axis"],
            "axes.labelcolor": COL["text"],
            "xtick.color": COL["text"],
            "ytick.color": COL["text"],
            "text.color": COL["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": COL["bg"],
            "axes.facecolor": COL["bg"],
            "savefig.facecolor": COL["bg"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def robust_norm(values: pd.Series | np.ndarray, qlo: float = 0.02, qhi: float = 0.985) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).to_numpy(float)
    lo = np.quantile(arr, qlo) if len(arr) else 0.0
    hi = np.quantile(arr, qhi) if len(arr) else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mpl.colors.to_rgba(color, alpha)


def nice_label(value: str) -> str:
    text = str(value)
    for old, new in [
        ("_proxy", ""),
        ("germinal_center", "GC"),
        ("myeloid_stromal", "myeloid/stromal"),
        ("T_zone", "T zone"),
        ("tumor_stromal_boundary", "tumor-stromal"),
        ("tumor_immune_boundary", "tumor-immune"),
        ("vascular_stromal_interface", "vascular-stromal"),
        ("stromal_immune_interface", "stromal-immune"),
        ("_", " "),
    ]:
        text = text.replace(old, new)
    return text


def short_lr(label: str) -> str:
    return str(label).replace("-", "\n")


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.028, dy: float = 0.012) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=COL["ink"])


def clean_quant_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.45, length=2.0, pad=1.5)
    ax.set_axisbelow(True)
    if grid == "x":
        ax.grid(axis="x", color=COL["grid"], lw=0.35, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=COL["grid"], lw=0.35, zorder=0)
    else:
        ax.grid(False)


def spatial_axis(ax: plt.Axes, frame: pd.DataFrame, x: str = "x_plot", y: str = "y_plot", pad: float = 0.035) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    xx = pd.to_numeric(frame[x], errors="coerce")
    yy = pd.to_numeric(frame[y], errors="coerce")
    px = (xx.max() - xx.min()) * pad if xx.max() > xx.min() else 1.0
    py = (yy.max() - yy.min()) * pad if yy.max() > yy.min() else 1.0
    ax.set_xlim(xx.min() - px, xx.max() + px)
    ax.set_ylim(yy.min() - py, yy.max() + py)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_category_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    category: str,
    *,
    x: str = "x_plot",
    y: str = "y_plot",
    max_categories: int = 7,
    point_size: float = 3.0,
    legend: bool = True,
    highlight: set[str] | None = None,
) -> None:
    dat = frame.dropna(subset=[x, y]).copy()
    counts = dat[category].astype(str).value_counts()
    cats = counts.head(max_categories).index.tolist()
    if highlight:
        cats = list(dict.fromkeys(list(highlight) + cats))
    other = dat.loc[~dat[category].astype(str).isin(cats)]
    if not other.empty:
        ax.scatter(other[x], other[y], s=point_size * 0.72, color=COL["tissue"], alpha=0.34, linewidths=0, zorder=1)
    handles = []
    for cat in cats:
        sub = dat.loc[dat[category].astype(str).eq(cat)]
        if sub.empty:
            continue
        color = CELL_COLORS.get(cat, CELL_COLORS.get(cat.lower(), COL["grey"]))
        alpha = 0.88 if not highlight or cat in highlight else 0.64
        size = point_size * (1.25 if highlight and cat in highlight else 1.0)
        ax.scatter(sub[x], sub[y], s=size, color=color, alpha=alpha, linewidths=0, zorder=2)
        handles.append(Line2D([], [], marker="o", ls="", color=color, markersize=3.8, label=nice_label(cat)))
    spatial_axis(ax, dat, x, y)
    if legend and handles:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.06), ncol=2, handletextpad=0.2, columnspacing=0.6)


def draw_signal_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_col: str,
    *,
    x: str = "x_plot",
    y: str = "y_plot",
    cmap=RED_CMAP,
    point_size: float = 3.0,
    active_threshold: float = 0.04,
    top_col: str | None = None,
    top_color: str = COL["signal"],
    colorbar: bool = True,
) -> None:
    dat = frame.dropna(subset=[x, y]).copy()
    values = pd.to_numeric(dat[value_col], errors="coerce").fillna(0)
    if values.max() > 1.2 or values.min() < 0:
        dat["_value_norm"] = robust_norm(values)
    else:
        dat["_value_norm"] = np.clip(values.to_numpy(float), 0, 1)
    dat = dat.sort_values("_value_norm")
    ax.scatter(dat[x], dat[y], s=point_size * 0.72, color=COL["tissue"], alpha=0.30, linewidths=0, zorder=1)
    active = dat.loc[dat["_value_norm"].ge(active_threshold)]
    if not active.empty:
        ax.scatter(
            active[x],
            active[y],
            s=point_size * 1.30,
            c=active["_value_norm"],
            cmap=cmap,
            norm=Normalize(0, 1),
            alpha=0.90,
            linewidths=0,
            zorder=2,
        )
    if top_col and top_col in dat.columns:
        top = dat.loc[dat[top_col].fillna(False).astype(bool)]
        if not top.empty:
            ax.scatter(top[x], top[y], s=point_size * 2.8, facecolors="none", edgecolors=top_color, linewidths=0.34, alpha=0.78, zorder=3)
    spatial_axis(ax, dat, x, y)
    if colorbar:
        cax = ax.inset_axes([0.62, -0.025, 0.30, 0.035])
        grad = np.linspace(0, 1, 128).reshape(1, -1)
        cax.imshow(grad, cmap=cmap, aspect="auto")
        cax.set_xticks([0, 127])
        cax.set_xticklabels(["low", "high"], fontsize=4.8)
        cax.set_yticks([])
        cax.tick_params(length=0, pad=0.4)
        for spine in cax.spines.values():
            spine.set_visible(False)


def draw_edge_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    color: str = COL["signal"],
    max_edges: int = 220,
    linewidth: float = 0.28,
    alpha: float = 0.42,
    cell_x: str = "x_plot",
    cell_y: str = "y_plot",
) -> None:
    ax.scatter(cells[cell_x], cells[cell_y], s=2.1, color=COL["tissue"], alpha=0.26, linewidths=0, zorder=1)
    if not edges.empty:
        plot_edges = edges.sort_values("score", ascending=False).head(max_edges).copy()
        max_score = max(float(plot_edges["score"].max()), 1e-9) if "score" in plot_edges else 1.0
        sx = "sender_x_plot" if "sender_x_plot" in plot_edges.columns else "sender_x"
        sy = "sender_y_plot" if "sender_y_plot" in plot_edges.columns else "sender_y"
        rx = "receiver_x_plot" if "receiver_x_plot" in plot_edges.columns else "receiver_x"
        ry = "receiver_y_plot" if "receiver_y_plot" in plot_edges.columns else "receiver_y"
        if sy == "sender_y" and "sender_y_plot" not in plot_edges.columns:
            plot_edges["_sy"] = -pd.to_numeric(plot_edges[sy], errors="coerce")
            plot_edges["_ry"] = -pd.to_numeric(plot_edges[ry], errors="coerce")
            sy, ry = "_sy", "_ry"
        for _, row in plot_edges.iterrows():
            score = float(row.get("score", max_score)) / max_score
            ax.plot(
                [row[sx], row[rx]],
                [row[sy], row[ry]],
                color=color,
                lw=linewidth + 0.35 * score,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2,
            )
        receiver_col = "receiver_id"
        if receiver_col in plot_edges.columns and "cell_id" in cells.columns:
            receivers = cells[cells["cell_id"].isin(plot_edges[receiver_col].astype(str))]
            ax.scatter(receivers[cell_x], receivers[cell_y], s=6.0, color=color, edgecolors=COL["bg"], linewidths=0.12, alpha=0.92, zorder=3)
    spatial_axis(ax, cells, cell_x, cell_y)


def barh_rank(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    *,
    highlight_labels: set[str] | None = None,
    xlabel: str = "",
    color: str = COL["signal"],
    neutral: str = "#D8D2C8",
    max_items: int | None = None,
) -> None:
    data = pd.DataFrame({"label": labels, "value": values}).dropna()
    if max_items:
        data = data.nlargest(max_items, "value")
    data = data.sort_values("value", ascending=True)
    y = np.arange(len(data))
    colors = [color if highlight_labels and lab in highlight_labels else neutral for lab in data["label"]]
    ax.barh(y, data["value"], color=colors, edgecolor=COL["axis"], linewidth=0.28, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"].tolist())
    ax.set_xlabel(xlabel)
    clean_quant_axis(ax, "x")


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap=HEAT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar: bool = True,
    xtick_rotation: int = 35,
) -> None:
    mat = matrix.astype(float)
    norm = TwoSlopeNorm(vmin=float(np.nanmin(mat.values)) if vmin is None else vmin, vcenter=center, vmax=float(np.nanmax(mat.values)) if vmax is None else vmax) if center is not None else Normalize(vmin=float(np.nanmin(mat.values)) if vmin is None else vmin, vmax=float(np.nanmax(mat.values)) if vmax is None else vmax)
    x = np.arange(mat.shape[1] + 1)
    y = np.arange(mat.shape[0] + 1)
    ax.pcolormesh(x, y, mat.values, cmap=cmap, norm=norm, edgecolors="#3E3A35", linewidth=0.18)
    ax.set_xlim(0, mat.shape[1])
    ax.set_ylim(mat.shape[0], 0)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5)
    ax.set_xticklabels(mat.columns.tolist(), rotation=xtick_rotation, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5)
    ax.set_yticklabels(mat.index.tolist())
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cbar:
        cax = ax.inset_axes([1.02, 0.12, 0.035, 0.76])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
        cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.35, pad=1)
        cb.outline.set_linewidth(0.35)


def draw_matrix_network(ax: plt.Axes, table: pd.DataFrame, *, cmap=GOLD_CMAP) -> None:
    mat = table.copy()
    if "index" in mat.columns:
        mat = mat.set_index("index")
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep_rows = mat.sum(axis=1).sort_values(ascending=False).head(min(7, len(mat))).index
    keep_cols = mat.sum(axis=0).sort_values(ascending=False).head(min(7, mat.shape[1])).index
    mat = mat.loc[keep_rows, keep_cols]
    mat.index = [nice_label(v) for v in mat.index]
    mat.columns = [nice_label(v) for v in mat.columns]
    draw_heatmap(ax, mat, cmap=cmap, cbar=True, xtick_rotation=35)
    ax.set_xlabel("receiver")
    ax.set_ylabel("sender")


def draw_box_strip(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, *, order: list[str] | None = None, palette: dict[str, str] | None = None, ylabel: str = "") -> None:
    dat = data[[x, y]].dropna().copy()
    if order is None:
        order = dat[x].drop_duplicates().tolist()
    vals = [pd.to_numeric(dat.loc[dat[x].eq(cat), y], errors="coerce").dropna().to_numpy(float) for cat in order]
    bp = ax.boxplot(vals, positions=np.arange(len(order)), widths=0.46, patch_artist=True, showfliers=False, medianprops={"color": COL["axis"], "lw": 0.72}, whiskerprops={"color": COL["axis"], "lw": 0.55}, capprops={"color": COL["axis"], "lw": 0.55})
    for patch, cat in zip(bp["boxes"], order):
        patch.set_facecolor(with_alpha((palette or {}).get(cat, COL["signal_soft"]), 0.58))
        patch.set_edgecolor(COL["axis"])
        patch.set_linewidth(0.55)
    rng = np.random.default_rng(5)
    for i, (cat, arr) in enumerate(zip(order, vals)):
        if len(arr) == 0:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=len(arr))
        ax.scatter(np.full(len(arr), i) + jitter, arr, s=10, color=(palette or {}).get(cat, COL["signal"]), alpha=0.72, edgecolors=COL["bg"], linewidths=0.18, zorder=3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    clean_quant_axis(ax, "y")


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def broad_compartment(cell_type: str) -> str:
    s = str(cell_type).lower()
    if any(k in s for k in ["tumor"]):
        return "tumor"
    if any(k in s for k in ["epithelial", "colonocyte", "goblet", "stem", "tuft", "ta", "eec", "clu"]):
        return "epithelial"
    if any(k in s for k in ["b_cell", "b cell", "t_cell", "t cell", "tfh", "immune", "macrophage", "myeloid", "dc", "nk", "plasma", "mast", "neutrophil", "monocyte"]):
        if "myeloid" in s or "macrophage" in s or "monocyte" in s:
            return "myeloid"
        if "b_cell" in s or "b cell" in s:
            return "B_cell_proxy"
        if "tfh" in s:
            return "Tfh_proxy"
        if "t_cell" in s or "t cell" in s:
            return "t_cell_proxy"
        return "immune"
    if any(k in s for k in ["stromal", "fibro", "frc", "smc", "pericyte", "caf", "mesenchymal", "fdc"]):
        if "fdc" in s:
            return "FDC_proxy"
        return "stromal"
    if any(k in s for k in ["vascular", "endothelial", "ec"]):
        return "vascular"
    return "Other"


def write_specs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(json.dumps(FIG_SPECS, indent=2), encoding="utf-8")


def fig2_crc() -> None:
    cells = read_csv(CRC_SRC / "panel_a_niche_map.csv")
    abundance = read_csv(CRC_SRC / "panel_b_region_lr_abundance.csv")
    hotspot = read_csv(CRC_SRC / "panel_e_spatialself_top_edge_maps.csv")
    target = read_csv(CRC_SRC / "panel_f_tgfb_response_map.csv")
    align = read_csv(CRC_ART / "alignment_auc.csv")
    genes = read_csv(CRC_ART / "target_gene_heatmap.csv")
    layers = read_csv(CRC_ART / "sender_receiver_layers.csv")
    commot_edges = read_csv(CRC_SRC / "panel_h_commot_arrow_edges.csv")
    cellnest = read_csv(CRC_SRC / "panel_i_cellnest_heat_map.csv")
    pairs = read_csv(HP_SRC / "realdata_crc_tgfb1_tgfbr2_celltype_pairs.csv")

    fig = plt.figure(figsize=(7.2, 7.35))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.16, 1.00, 1.02, 1.02],
        height_ratios=[1.12, 1.02, 1.06],
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.965,
        wspace=0.48,
        hspace=0.58,
    )
    axa = fig.add_subplot(gs[0, 0])
    context = cells.copy()
    context["plot_group"] = np.where(context["niche"].astype(str).eq("tumor_stromal_boundary"), "tumor_stromal_boundary", context["cell_type"].map(broad_compartment))
    draw_category_map(axa, context, "plot_group", max_categories=6, point_size=3.0, highlight={"tumor_stromal_boundary"})
    axa.set_title("CRC tissue context", pad=3)
    add_panel_label(fig, axa, "a")

    axb = fig.add_subplot(gs[0, 1])
    barh_rank(
        axb,
        abundance["control_name"].tolist(),
        abundance["top_edge_count"].tolist(),
        highlight_labels={"TGFB1-TGFBR2"},
        xlabel="top-edge count",
        color=COL["signal"],
        max_items=7,
    )
    axb.set_title("Local LR abundance", pad=3)
    add_panel_label(fig, axb, "b")

    axc = fig.add_subplot(gs[0, 2])
    h300 = hotspot[hotspot["threshold_label"].eq("top 300")].copy()
    draw_signal_map(axc, h300, "incoming_score", cmap=RED_CMAP, top_col="top_edge_receiver", point_size=3.0)
    axc.set_title("ScaleComm hotspot", pad=3)
    add_panel_label(fig, axc, "c")

    axd = fig.add_subplot(gs[0, 3])
    draw_signal_map(axd, target, "target_activity", cmap=TARGET_CMAP, top_col="target_hotspot", top_color=COL["blue"], point_size=3.0)
    axd.set_title("TGF-beta response", pad=3)
    add_panel_label(fig, axd, "d")

    axe = fig.add_subplot(gs[1, 0])
    align = align.copy()
    align["method_label"] = align["method_label"].replace({"SpatialSelf": "ScaleComm"})
    order = ["ScaleComm", "COMMOT", "CellNEST"]
    align = align.set_index("method_label").reindex(order).reset_index()
    colors = [COL["signal"], COL["blue"], COL["purple"]]
    axe.bar(np.arange(len(align)), align["auc"], color=colors, edgecolor=COL["axis"], linewidth=0.35, width=0.58)
    axe.axhline(0.5, color=COL["muted"], lw=0.55, ls=(0, (2, 2)))
    axe.set_xticks(np.arange(len(align)))
    axe.set_xticklabels(order, rotation=35, ha="right")
    axe.set_ylabel("target-hotspot AUC")
    axe.set_ylim(0.46, max(0.65, float(align["auc"].max()) + 0.03))
    clean_quant_axis(axe, "y")
    axe.set_title("Response alignment", pad=3)
    add_panel_label(fig, axe, "e")

    axf = fig.add_subplot(gs[1, 1])
    pivot = genes.pivot_table(index="gene", columns="group", values="z", aggfunc="mean").fillna(0)
    pivot = pivot[[c for c in ["other", "ScaleComm top", "response hotspot"] if c in pivot.columns]]
    pivot.columns = [c.replace("ScaleComm top", "ScaleComm").replace("response hotspot", "response") for c in pivot.columns]
    draw_heatmap(axf, pivot, cmap=HEAT_CMAP, center=0, cbar=True, xtick_rotation=35)
    axf.set_title("Receiver target genes", pad=3)
    add_panel_label(fig, axf, "f")

    axg = fig.add_subplot(gs[1, 2])
    draw_edge_map(axg, cells, commot_edges, color=COL["blue"], max_edges=210, linewidth=0.20, alpha=0.34)
    axg.set_title("COMMOT", pad=3)
    add_panel_label(fig, axg, "g")

    axh = fig.add_subplot(gs[1, 3])
    draw_signal_map(axh, cellnest, "regional_heat", cmap=PURPLE_CMAP, top_col=None, point_size=3.0, active_threshold=0.02)
    axh.set_title("CellNEST", pad=3)
    add_panel_label(fig, axh, "h")

    axi = fig.add_subplot(gs[2, 0:2])
    ldat = layers.copy()
    ldat["score_norm"] = robust_norm(ldat["score"])
    for role, color in [("sender", COL["teal"]), ("receiver", COL["signal"])]:
        sub = ldat[ldat["role"].eq(role)].sort_values("score_norm")
        axi.scatter(sub["x_plot"], sub["y_plot"], s=5.0 + 14.0 * sub["score_norm"], color=color, alpha=0.52, linewidths=0, label=role)
    axi.scatter(cells["x_plot"], cells["y_plot"], s=1.2, color=COL["tissue"], alpha=0.14, linewidths=0, zorder=0)
    spatial_axis(axi, cells)
    axi.legend(loc="lower left", bbox_to_anchor=(0, -0.035), ncol=2, handletextpad=0.25, columnspacing=0.7)
    axi.set_title("Inferred sending and receiving cells", pad=3)
    add_panel_label(fig, axi, "i")

    axj = fig.add_subplot(gs[2, 2:4])
    draw_matrix_network(axj, pairs, cmap=GOLD_CMAP)
    axj.set_title("TGFB1-TGFBR2 cell-type pairs", pad=3)
    add_panel_label(fig, axj, "j")

    save_figure(fig, FIG_DIR / "fig2_crc_visium_hd_tgfb1_tgfbr2_fused_v2")


def dss_stage_replicate_dir(stage: str, replicate: int = 1) -> Path:
    manifest = read_csv(DSS_BASE / "multislice_replicates/multislice_replicate_manifest.csv")
    row = manifest.loc[(manifest["stage"].eq(stage)) & (manifest["stage_replicate"].eq(replicate))].iloc[0]
    return ROOT / str(row["replicate_dir"])


def dss_load_stage(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    d = dss_stage_replicate_dir(stage, replicate)
    return d.name, read_csv(d / "cells.csv"), read_csv(d / "candidate_interactions.csv")


def dss_prediction_path(stage: str, replicate_name: str) -> Path:
    return DSS_BASE / "multislice_spatialself_scale_v21/predictions" / stage / "spatialself_wavelet_scale" / f"{replicate_name}__spatialself_wavelet_scale_predictions.csv"


def dss_top_edges(stage: str, replicate: int = 1, fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep, cells, cand = dss_load_stage(stage, replicate)
    pred = read_csv(dss_prediction_path(stage, rep))
    edges = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    xy = cells.set_index("cell_id")[["x", "y", "cell_type"]]
    edges = edges.join(xy.rename(columns={"x": "sender_x", "y": "sender_y", "cell_type": "sender_type"}), on="sender_id")
    edges = edges.join(xy.rename(columns={"x": "receiver_x", "y": "receiver_y", "cell_type": "receiver_type"}), on="receiver_id")
    edges["sender_y"] = pd.to_numeric(edges["sender_y"], errors="coerce")
    edges["receiver_y"] = pd.to_numeric(edges["receiver_y"], errors="coerce")
    n = max(80, int(len(edges) * fraction))
    return cells, edges.nlargest(n, "score").copy()


def dss_cells_for_plot(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    out["x_plot"] = out["x"]
    out["y_plot"] = out["y"]
    out["compartment"] = out["cell_type"].map(broad_compartment)
    return out


def fig3_dss() -> None:
    stages = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
    labels = ["D0", "D3", "D9", "D21"]
    stage_colors = {"D0": COL["grey"], "D3": "#D9992B", "D9": COL["signal"], "D21": COL["teal"]}
    top_lr = read_csv(DSS_SRC / "panel_c_top_lr_histograms.csv")
    program = read_csv(DSS_SRC / "panel_d_program_activity.csv")
    local_d9 = read_csv(DSS_SRC / "panel_g_local_lr_DSS9.csv")
    local_d21 = read_csv(DSS_SRC / "panel_i_local_lr_DSS21.csv")
    replicate = read_csv(DSS_SRC / "panel_j_replicate_consistency.csv")
    trajectories = read_csv(DSS_ART / "trajectories.csv")
    flow = read_csv(DSS_ART / "flow.csv")
    d9_activity = read_csv(DSS_ART / "panel_d_replicate_spatial_activity.csv")

    fig = plt.figure(figsize=(7.2, 9.15))
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.05, 1.05, 1.42, 1.18, 1.18],
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=0.965,
        hspace=0.52,
    )

    row_a = outer[0, 0].subgridspec(1, 4, wspace=0.06)
    row_b = outer[1, 0].subgridspec(1, 4, wspace=0.06)
    first_a = None
    first_b = None
    for i, (stage, lab) in enumerate(zip(stages, labels)):
        cells, edges = dss_top_edges(stage)
        cp = dss_cells_for_plot(cells)
        ax = fig.add_subplot(row_a[0, i])
        draw_category_map(ax, cp, "compartment", max_categories=6, point_size=2.3, legend=(i == 0))
        ax.set_title(lab, pad=2.5)
        if i == 0:
            first_a = ax
        ax2 = fig.add_subplot(row_b[0, i])
        cc = cp.copy()
        cc["x_plot"] = cc["x"]
        cc["y_plot"] = cc["y"]
        draw_edge_map(ax2, cc, edges, color=stage_colors[lab], max_edges=170, linewidth=0.16, alpha=0.32)
        ax2.set_title(lab, pad=2.5)
        if i == 0:
            first_b = ax2
    add_panel_label(fig, first_a, "a")
    add_panel_label(fig, first_b, "b")

    row_c = outer[2, 0].subgridspec(1, 4, wspace=0.68)
    for i, (stage, lab) in enumerate(zip(stages, labels)):
        ax = fig.add_subplot(row_c[0, i])
        dat = top_lr[top_lr["stage"].eq(stage)].nlargest(6, "edge_count").sort_values("edge_count")
        high = {"Cxcl10-Cxcr3"} if stage == "DSS9" else ({"Tgfb3-Acvrl1"} if stage == "DSS21" else set())
        barh_rank(ax, dat["lr_pair"].tolist(), dat["edge_count"].tolist(), highlight_labels=high, xlabel="n edges", color=stage_colors[lab])
        ax.set_title(lab, pad=2.5)
        if i == 0:
            add_panel_label(fig, ax, "c")

    row_d = outer[3, 0].subgridspec(1, 3, width_ratios=[1.05, 1.45, 1.15], wspace=0.42)
    axd = fig.add_subplot(row_d[0, 0])
    pmat = program.pivot_table(index="program", columns="stage_label", values="relative_activity", aggfunc="mean").reindex(["homeostasis", "early response", "inflammatory peak", "repair/remodeling"])
    pmat = pmat[labels]
    draw_heatmap(axd, pmat, cmap=GOLD_CMAP, vmin=0, vmax=1, cbar=True, xtick_rotation=0)
    axd.set_title("Communication programs", pad=3)
    add_panel_label(fig, axd, "d")

    axe = fig.add_subplot(row_d[0, 1])
    lr_lines = ["Rspo3-Lgr5", "Il1b-Il1r2", "Cxcl10-Cxcr3", "Tgfb3-Acvrl1"]
    line_colors = {"Rspo3-Lgr5": COL["grey"], "Il1b-Il1r2": "#D9992B", "Cxcl10-Cxcr3": COL["signal"], "Tgfb3-Acvrl1": COL["teal"]}
    for lr in lr_lines:
        row = trajectories[trajectories["lr_pair"].eq(lr)]
        if row.empty:
            continue
        vals = row.iloc[0][stages].astype(float).to_numpy()
        vals = vals / max(vals.max(), 1e-9)
        axe.plot(np.arange(4), vals, marker="o", ms=3.0, lw=1.15 if lr not in {"Cxcl10-Cxcr3", "Tgfb3-Acvrl1"} else 1.45, color=line_colors[lr], label=lr)
    axe.set_xticks(np.arange(4))
    axe.set_xticklabels(labels)
    axe.set_ylabel("max-normalized activity")
    axe.set_ylim(-0.03, 1.08)
    clean_quant_axis(axe, "y")
    axe.legend(loc="upper left", bbox_to_anchor=(0.02, 1.03), ncol=2, handlelength=1.5, columnspacing=0.7)
    axe.set_title("Stage-specific LR dynamics", pad=3)
    add_panel_label(fig, axe, "e")

    axf = fig.add_subplot(row_d[0, 2])
    fdat = flow.copy()
    fdat["delta"] = fdat["stage_fraction"] * np.where(fdat["stage_label"].eq("D21"), 1, np.where(fdat["stage_label"].eq("D0"), -1, 0))
    pivot = fdat.pivot_table(index="flow", columns="stage_label", values="stage_fraction", aggfunc="sum").fillna(0)
    top = pivot.sum(axis=1).sort_values(ascending=False).head(8).index
    pivot = pivot.loc[top, labels]
    pivot.index = [nice_label(v).replace(" -> ", "\n-> ") for v in pivot.index]
    draw_heatmap(axf, pivot, cmap=GOLD_CMAP, vmin=0, vmax=float(pivot.values.max()), cbar=True, xtick_rotation=0)
    axf.set_title("Compartment rewiring", pad=3)
    add_panel_label(fig, axf, "f")

    row_e = outer[4, 0].subgridspec(1, 4, width_ratios=[1.10, 1.0, 1.0, 0.72], wspace=0.50)
    axg = fig.add_subplot(row_e[0, 0])
    for rep, x0 in [(1, 0.08), (2, 0.54)]:
        sub = d9_activity[d9_activity["replicate"].eq(rep)].copy()
        inset = axg.inset_axes([x0, 0.10, 0.38, 0.78])
        sub["x_plot"] = sub["x"]
        sub["y_plot"] = -sub["y"]
        draw_signal_map(inset, sub, "activity_norm", cmap=RED_CMAP, point_size=2.6, colorbar=False)
        inset.set_title(f"rep {rep}", fontsize=5.8, pad=0.5)
    axg.set_axis_off()
    axg.set_title("DSS9 receiver activity", pad=3)
    add_panel_label(fig, axg, "g")

    axh = fig.add_subplot(row_e[0, 1])
    d9 = local_d9.nlargest(7, "edge_count")
    barh_rank(axh, d9["lr_pair"].tolist(), d9["edge_count"].tolist(), highlight_labels={"Cxcl10-Cxcr3"}, xlabel="local n", color=COL["signal"])
    axh.set_title("D9 local LR rank", pad=3)
    add_panel_label(fig, axh, "h")

    axi = fig.add_subplot(row_e[0, 2])
    d21 = local_d21.nlargest(7, "edge_count")
    barh_rank(axi, d21["lr_pair"].tolist(), d21["edge_count"].tolist(), highlight_labels={"Tgfb3-Acvrl1"}, xlabel="local n", color=COL["teal"])
    axi.set_title("D21 local LR rank", pad=3)
    add_panel_label(fig, axi, "i")

    axj = fig.add_subplot(row_e[0, 3])
    axj.plot(replicate["stage_label"], replicate["top_lr_jaccard"], color=COL["axis"], lw=1.0, marker="o", ms=3.4)
    for lab, val in zip(replicate["stage_label"], replicate["top_lr_jaccard"]):
        axj.scatter([lab], [val], s=25, color=stage_colors.get(lab, COL["grey"]), edgecolors=COL["axis"], linewidths=0.25, zorder=3)
    axj.set_ylim(0, max(0.75, replicate["top_lr_jaccard"].max() + 0.08))
    axj.set_ylabel("top LR overlap\nJaccard")
    clean_quant_axis(axj, "y")
    axj.set_title("Replicate support", pad=3)
    add_panel_label(fig, axj, "j")

    save_figure(fig, FIG_DIR / "fig3_mouse_colon_dss_merfish_timecourse_fused_v2")


def fig4_cosmx() -> None:
    cells = read_csv(COSMX_SRC / "cosmx_cells_for_context_map.csv")
    scale = read_csv(COSMX_SRC / "cosmx_lr_scale_distribution.csv")
    hotspots = read_csv(COSMX_SRC / "cosmx_hotspot_receiver_cells.csv")
    support = read_csv(COSMX_SRC / "cosmx_receiver_target_support.csv")
    scale_gain = read_csv(COSMX_SRC / "cosmx_scale_vs_noscale_target_support.csv")
    burden = read_csv(HP_SRC / "realdata_cosmx_receiver_celltype_burden.csv")
    ranking = read_csv(COSMX_SRC / "cosmx_fig5_story_candidate_ranking.csv")

    fig = plt.figure(figsize=(7.2, 7.05))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.16, 1.08, 1.05, 1.05],
        height_ratios=[1.05, 1.05, 1.05],
        left=0.055,
        right=0.985,
        bottom=0.06,
        top=0.965,
        wspace=0.50,
        hspace=0.62,
    )
    cells = cells.copy()
    cells["x_plot"] = cells["x"]
    cells["y_plot"] = -cells["y"]
    cells["plot_group"] = cells["cell_type"].map(broad_compartment)
    axa = fig.add_subplot(gs[0, 0])
    draw_category_map(axa, cells, "plot_group", max_categories=7, point_size=2.25, legend=True, highlight={"tumor", "t_cell_proxy", "stromal", "myeloid"})
    axa.set_title("CosMx NSCLC context", pad=3)
    add_panel_label(fig, axa, "a")

    axb = fig.add_subplot(gs[0, 1:3])
    scale_cols = [f"scale_{i}" for i in range(6) if f"scale_{i}" in scale.columns]
    select_lrs = ["CXCL12-CXCR4", "CXCL10-CXCR3", "CXCL9-CXCR3", "VEGFA-KDR", "TGFB1-TGFBR2", "SPP1-CD44", "CD274-PDCD1"]
    smat = scale[scale["lr_pair"].isin(select_lrs)].drop_duplicates("lr_pair").set_index("lr_pair")[scale_cols]
    smat.index = [s.replace("-", "\n") for s in smat.index]
    smat.columns = [f"s{i}" for i in range(len(smat.columns))]
    draw_heatmap(axb, smat, cmap=GOLD_CMAP, vmin=0, vmax=max(1e-9, float(smat.values.max())), cbar=True, xtick_rotation=0)
    axb.set_title("Learned LR scale/context atlas", pad=3)
    add_panel_label(fig, axb, "b")

    axc = fig.add_subplot(gs[0, 3])
    for lr, color in [("CXCL12-CXCR4", COL["teal"]), ("CXCL10-CXCR3", COL["signal"]), ("TGFB1-TGFBR2", COL["amber"]), ("SPP1-CD44", COL["grey"])]:
        row = scale[scale["lr_pair"].eq(lr)]
        if row.empty:
            continue
        vals = row.iloc[0][scale_cols].astype(float).to_numpy()
        vals = vals / max(vals.max(), 1e-9)
        axc.plot(np.arange(len(vals)), vals, marker="o", ms=3.0, lw=1.22, color=color, label=lr)
    axc.set_xticks(np.arange(len(scale_cols)))
    axc.set_xticklabels([f"s{i}" for i in range(len(scale_cols))])
    axc.set_ylabel("relative gate weight")
    axc.set_ylim(-0.04, 1.08)
    clean_quant_axis(axc, "y")
    axc.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.02), handlelength=1.3, fontsize=4.9)
    axc.set_title("Representative profiles", pad=3)
    add_panel_label(fig, axc, "c")

    axd = fig.add_subplot(gs[1, 0])
    gain = scale_gain.copy()
    gain = gain[gain["method"].isin(["scale-aware", "no-scale"])]
    selected = (
        gain.pivot_table(index="LR", columns="method", values="target_spearman_r", aggfunc="mean")
        .dropna()
        .assign(delta=lambda x: x["scale-aware"] - x["no-scale"])
        .sort_values("delta", ascending=False)
        .head(5)
        .index.tolist()
    )
    plot_gain = gain[gain["LR"].isin(selected)].copy()
    wide = plot_gain.pivot_table(index="LR", columns="method", values="target_spearman_r", aggfunc="mean").loc[selected]
    for i, lr in enumerate(wide.index):
        axd.plot([0, 1], [wide.loc[lr, "no-scale"], wide.loc[lr, "scale-aware"]], color=COL["muted"], lw=0.65, alpha=0.68, zorder=1)
        axd.scatter([0], [wide.loc[lr, "no-scale"]], s=22, color=COL["tissue_dark"], edgecolors=COL["axis"], linewidths=0.25, zorder=2)
        axd.scatter([1], [wide.loc[lr, "scale-aware"]], s=22, color=COL["signal"], edgecolors=COL["axis"], linewidths=0.25, zorder=2)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["no scale", "scale-aware"], rotation=25, ha="right")
    axd.set_ylabel("target Spearman r")
    clean_quant_axis(axd, "y")
    axd.set_title("Scale/context response gain", pad=3)
    add_panel_label(fig, axd, "d")

    axe = fig.add_subplot(gs[1, 1:4])
    axe.set_axis_off()
    for i, (lr, cmap, color) in enumerate([("CXCL12-CXCR4", TARGET_CMAP, COL["teal"]), ("CXCL10-CXCR3", RED_CMAP, COL["signal"]), ("VEGFA-KDR", GOLD_CMAP, COL["amber"])]):
        ax = axe.inset_axes([0.01 + i * 0.33, 0.06, 0.30, 0.86])
        sub = hotspots[hotspots["LR"].eq(lr)].copy()
        sub["x_plot"] = sub["x"]
        sub["y_plot"] = -sub["y"]
        draw_signal_map(ax, sub, "receiver_activity", cmap=cmap, top_col="is_top_receiver", top_color=color, point_size=2.3, colorbar=False)
        ax.set_title(lr, fontsize=5.9, color=color, pad=1.2)
    axe.set_title("Context-specific receiver activity", pad=3)
    add_panel_label(fig, axe, "e")

    axf = fig.add_subplot(gs[2, 0])
    sup = support.sort_values("top_receiver_target_lift", ascending=False).head(7).sort_values("top_receiver_target_lift")
    barh_rank(axf, sup["LR"].tolist(), sup["top_receiver_target_lift"].tolist(), highlight_labels={"CXCL12-CXCR4", "CXCL10-CXCR3"}, xlabel="target-response lift", color=COL["signal"], neutral="#D8D2C8")
    axf.set_title("Receiver target support", pad=3)
    add_panel_label(fig, axf, "f")

    axg = fig.add_subplot(gs[2, 1:3])
    mat = burden.copy().set_index("LR")
    keep = [lr for lr in ["CXCL12-CXCR4", "CXCL10-CXCR3", "TGFB1-TGFBR2", "SPP1-CD44", "VEGFA-KDR"] if lr in mat.index]
    mat = mat.loc[keep]
    mat.index = [s.replace("-", "\n") for s in mat.index]
    draw_heatmap(axg, mat, cmap=GOLD_CMAP, vmin=0, vmax=max(1e-9, float(mat.values.max())), cbar=False, xtick_rotation=35)
    axg.set_title("Receiver cell-type burden", pad=3)
    add_panel_label(fig, axg, "g")

    axh = fig.add_subplot(gs[2, 3])
    rank_cols = ["LR", "story_score", "top_niche_match_rate", "top_receiver_target_lift"]
    rank = ranking[rank_cols].dropna().nlargest(6, "story_score").sort_values("story_score")
    barh_rank(axh, rank["LR"].tolist(), rank["story_score"].tolist(), highlight_labels={"CXCL12-CXCR4", "CXCL10-CXCR3"}, xlabel="integrated support", color=COL["teal"], neutral="#D8D2C8")
    axh.set_title("Prioritized context axes", pad=3)
    add_panel_label(fig, axh, "h")

    save_figure(fig, FIG_DIR / "fig4_cosmx_nsclc_scale_context_fused_v2")


def fig5_tonsil() -> None:
    context = read_csv(TONSIL_SRC / "panel_a_tonsil_context.csv")
    expr = read_csv(TONSIL_SRC / "panel_b_ligand_receptor_expression.csv")
    hotspot = read_csv(TONSIL_SRC / "panel_c_scalecomm_receiver_hotspot.csv")
    target = read_csv(TONSIL_SRC / "panel_e_target_response_map.csv")
    sender_receiver = read_csv(TONSIL_SRC / "source_sender_receiver_summary.csv")
    replicate = read_csv(TONSIL_SRC / "panel_f_replicate_support.csv")
    gc_rank = read_csv(TONSIL_SRC / "panel_g_gc_top20_lr_hotspot_counts.csv")
    scale = read_csv(TONSIL_SRC / "source_lr_scale_distribution.csv")

    fig = plt.figure(figsize=(7.2, 8.75))
    gs = fig.add_gridspec(
        4,
        4,
        width_ratios=[1.12, 1.00, 1.00, 1.18],
        height_ratios=[1.08, 1.02, 1.32, 1.42],
        left=0.055,
        right=0.985,
        bottom=0.052,
        top=0.965,
        wspace=0.56,
        hspace=0.60,
    )

    axa = fig.add_subplot(gs[0, 0])
    draw_category_map(axa, context, "niche", max_categories=5, point_size=4.0, legend=True, highlight={"germinal_center_proxy"})
    axa.set_title("SPAMOSAIC tonsil context", pad=3)
    add_panel_label(fig, axa, "a")

    axb = fig.add_subplot(gs[0, 1:4])
    axb.set_axis_off()
    strip = [
        ("CXCL13", expr[expr["gene"].eq("CXCL13")], "expression_norm", RED_CMAP, COL["signal"]),
        ("CXCR5", expr[expr["gene"].eq("CXCR5")], "expression_norm", RED_CMAP, COL["signal"]),
        ("ScaleComm", hotspot, "incoming_norm", RED_CMAP, COL["signal"]),
        ("GC response", target, "target_score_norm", TARGET_CMAP, COL["blue"]),
    ]
    for i, (title, frame, value, cmap, color) in enumerate(strip):
        ax = axb.inset_axes([0.01 + i * 0.246, 0.10, 0.222, 0.78])
        draw_signal_map(ax, frame, value, cmap=cmap, top_col="top_receiver" if title == "ScaleComm" else None, top_color=color, point_size=4.2, active_threshold=0.05, colorbar=False)
        ax.set_title(title, fontsize=6.2, color=color if title in {"ScaleComm", "GC response"} else COL["text"], pad=1.2)
    axb.set_title("CXCL13-CXCR5 spatial program", pad=3)
    add_panel_label(fig, axb, "b")

    axc = fig.add_subplot(gs[1, 0])
    sr = sender_receiver[sender_receiver["lr_pair"].eq("CXCL13-CXCR5")].copy()
    pivot = sr.pivot_table(index="sender_cell_type", columns="receiver_cell_type", values="n_edges", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(6).index, pivot.sum(axis=0).sort_values(ascending=False).head(6).index]
    pivot.index = [nice_label(v) for v in pivot.index]
    pivot.columns = [nice_label(v) for v in pivot.columns]
    draw_heatmap(axc, pivot, cmap=GOLD_CMAP, vmin=0, vmax=float(pivot.values.max()), cbar=True, xtick_rotation=35)
    axc.set_title("Sender-receiver pairs", pad=3)
    axc.set_xlabel("receiver")
    axc.set_ylabel("sender")
    add_panel_label(fig, axc, "c")

    axd = fig.add_subplot(gs[1, 1])
    comp = read_csv(TONSIL_SRC / "panel_d_top_receiver_composition.csv")
    cd = comp[comp["feature"].eq("niche")].sort_values("fold_enrichment", ascending=True)
    d_labels = [str(v).replace("myeloid/stromal", "myeloid\nstromal") for v in cd["display_label"].tolist()]
    barh_rank(axd, d_labels, cd["fold_enrichment"].tolist(), highlight_labels={"GC proxy", "GC"}, xlabel="fold enrichment", color=COL["signal"], neutral="#D8D2C8")
    axd.axvline(1, color=COL["muted"], lw=0.55, ls=(0, (2, 2)))
    axd.set_title("GC enrichment", pad=3)
    add_panel_label(fig, axd, "d")

    axe = fig.add_subplot(gs[1, 2])
    gc = gc_rank[gc_rank["section"].eq("A1")].nlargest(7, "top20_gc_n").sort_values("top20_gc_n")
    gc_labels = [short_lr(v) for v in gc["lr_pair"].tolist()]
    barh_rank(axe, gc_labels, gc["top20_gc_n"].tolist(), highlight_labels={"CXCL13\nCXCR5"}, xlabel="GC cells in top 20%", color=COL["signal"], neutral="#D8D2C8")
    axe.set_title("GC-localized LR rank", pad=3)
    add_panel_label(fig, axe, "e")

    axf = fig.add_subplot(gs[1, 3])
    rep_long = replicate.melt(
        id_vars=["section"],
        value_vars=["target_lift_top_vs_background", "spearman_incoming_target", "gc_fold_enrichment"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "target_lift_top_vs_background": "target lift",
        "spearman_incoming_target": "Spearman r",
        "gc_fold_enrichment": "GC fold",
    }
    rep_long["metric"] = rep_long["metric"].map(labels)
    draw_box_strip(axf, rep_long, "metric", "value", order=["target lift", "Spearman r", "GC fold"], palette={"target lift": COL["signal"], "Spearman r": COL["blue"], "GC fold": COL["teal"]}, ylabel="support value")
    axf.set_xticklabels(["target\nlift", "Spearman\nr", "GC\nfold"])
    axf.set_title("A1/D1 replicate support", pad=3)
    add_panel_label(fig, axf, "f")

    axg = fig.add_subplot(gs[2, 0:2])
    scale_cols = [f"scale_{i}" for i in range(6) if f"scale_{i}" in scale.columns]
    keep = ["CXCL13-CXCR5", "CCL21-CCR7", "CXCL12-CXCR4", "IL21-IL21R", "CD40LG-CD40", "TNFSF13B-TNFRSF13C"]
    smat = scale[scale["lr_pair"].isin(keep)].drop_duplicates("lr_pair").set_index("lr_pair")[scale_cols]
    smat.index = [s.replace("-", "\n") for s in smat.index]
    smat.columns = [f"s{i}" for i in range(len(scale_cols))]
    draw_heatmap(axg, smat, cmap=GOLD_CMAP, vmin=0, vmax=float(smat.values.max()), cbar=True, xtick_rotation=0)
    axg.set_title("Tonsil LR scale profiles", pad=3)
    add_panel_label(fig, axg, "g")

    axi = fig.add_subplot(gs[2, 2:4])
    draw_signal_map(axi, hotspot, "incoming_norm", cmap=RED_CMAP, top_col="top_receiver", top_color=COL["signal"], point_size=4.0, active_threshold=0.03)
    axi.set_title("CXCL13-CXCR5 receiver hotspot", pad=3)
    add_panel_label(fig, axi, "h")

    axh = fig.add_subplot(gs[3, 0:4])
    axh.set_axis_off()
    if TONSIL_UI.exists():
        im = Image.open(TONSIL_UI).convert("RGB")
        axh.imshow(im)
    axh.set_title("Interactive ScaleComm browser", pad=3)
    add_panel_label(fig, axh, "i")

    save_figure(fig, FIG_DIR / "fig5_spamosaic_tonsil_cxcl13_cxcr5_fused_v2")


def make_contact_sheet() -> None:
    paths = [
        FIG_DIR / "fig2_crc_visium_hd_tgfb1_tgfbr2_fused_v2.png",
        FIG_DIR / "fig3_mouse_colon_dss_merfish_timecourse_fused_v2.png",
        FIG_DIR / "fig4_cosmx_nsclc_scale_context_fused_v2.png",
        FIG_DIR / "fig5_spamosaic_tonsil_cxcl13_cxcr5_fused_v2.png",
    ]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((820, 980), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (850, 1040), "white")
        canvas.paste(im, ((850 - im.width) // 2, 12))
        imgs.append(canvas)
    out = Image.new("RGB", (1700, 2080), "white")
    for i, im in enumerate(imgs):
        out.paste(im, ((i % 2) * 850, (i // 2) * 1040))
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out.save(QA_DIR / "fig2_fig5_fused_v2_contact_sheet.png", dpi=(220, 220))


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    write_specs()
    fig2_crc()
    fig3_dss()
    fig4_cosmx()
    fig5_tonsil()
    make_contact_sheet()
    print(f"Wrote fused Fig2-Fig5 to {FIG_DIR}")
    print(f"Wrote QA contact sheet to {QA_DIR / 'fig2_fig5_fused_v2_contact_sheet.png'}")


if __name__ == "__main__":
    main()
