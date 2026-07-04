#!/usr/bin/env python
"""Shared ScaleComm manuscript figure style.

The module keeps visual language consistent across manuscript case-study
figures while leaving each figure's data logic inside its original script.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
BORDER = "#BEB8AE"
FIG_BG = "#FFFFFF"
PANEL_FACE = "#FFFFFF"

TISSUE_BG = "#D6D2C8"
TISSUE_BG_DARK = "#BDB8AE"
NEUTRAL = "#C8C1B7"
NEUTRAL_DARK = "#827B72"

SIGNAL_RED = "#B85450"
SIGNAL_RED_SOFT = "#E7C6C2"
SIGNAL_RED_DARK = "#7A2D2A"
SOURCE_GOLD = "#D6A84F"
SOURCE_GOLD_SOFT = "#E6C06A"
RESPONSE_BLUE = "#3E7FA3"
RESPONSE_BLUE_SOFT = "#BFD8EC"
SCALE_TEAL = "#237A78"
SCALE_TEAL_SOFT = "#9BC8C1"
PURPLE_MUTED = "#7E6C9A"

METHOD_COLORS = {
    "spatialself": SIGNAL_RED,
    "scalecomm": SIGNAL_RED,
    "commot": RESPONSE_BLUE,
    "cellnest": PURPLE_MUTED,
    "cellchat": SOURCE_GOLD,
    "spatalk": "#8A8177",
}

COMPARTMENT_COLORS = {
    "Epithelial": "#D9992B",
    "Immune": SIGNAL_RED,
    "Mesenchymal": SCALE_TEAL,
    "Vascular": "#6D88A8",
    "Neural": PURPLE_MUTED,
    "Other": TISSUE_BG_DARK,
}

PROXY_CELL_COLORS = {
    "tumor_proxy": SIGNAL_RED_SOFT,
    "caf_proxy": "#CDA77A",
    "stromal_proxy": "#B9B0A6",
    "endothelial_proxy": "#93CC82",
    "immune_proxy": RESPONSE_BLUE_SOFT,
    "t_cell_proxy": RESPONSE_BLUE_SOFT,
    "T_cell_proxy": "#66A9E8",
    "myeloid_proxy": "#B9C1D1",
    "epithelial_proxy": "#D6D1C9",
    "B_cell_proxy": SIGNAL_RED,
    "FDC_proxy": "#D9A6A2",
    "Tfh_proxy": RESPONSE_BLUE_SOFT,
    "other_proxy": "#D8DEE8",
}

TONSIL_NICHE_COLORS = {
    "germinal_center_proxy": SIGNAL_RED,
    "T_zone_proxy": "#66A9E8",
    "myeloid_stromal_proxy": SCALE_TEAL_SOFT,
    "epithelial_proxy": "#D6D1C9",
}

CRC_NICHE_COLORS = {
    "tumor_core": "#D8D2C8",
    "tumor_stromal_boundary": "#B88455",
    "tumor_immune_boundary": SCALE_TEAL,
    "tumor_myeloid_boundary": SIGNAL_RED,
    "vascular_stromal_boundary": "#6D88A8",
    "epithelial_region": PURPLE_MUTED,
}


def make_cmaps() -> dict[str, LinearSegmentedColormap]:
    return {
        "activity": LinearSegmentedColormap.from_list(
            "scalecomm_activity", ["#F8F5EE", SIGNAL_RED_SOFT, SIGNAL_RED]
        ),
        "response": LinearSegmentedColormap.from_list(
            "scalecomm_response", ["#F7F4EC", RESPONSE_BLUE_SOFT, RESPONSE_BLUE]
        ),
        "diverging": LinearSegmentedColormap.from_list(
            "scalecomm_diverging", [SOURCE_GOLD_SOFT, "#F7F4EC", RESPONSE_BLUE]
        ),
        "flow": LinearSegmentedColormap.from_list(
            "scalecomm_flow", ["#F7F4EC", SOURCE_GOLD_SOFT, SIGNAL_RED]
        ),
        "scale": LinearSegmentedColormap.from_list(
            "scalecomm_scale", [SOURCE_GOLD_SOFT, "#F7F4EC", SCALE_TEAL]
        ),
        "burden": LinearSegmentedColormap.from_list(
            "scalecomm_burden", ["#F7F4EC", "#E8BF78", SIGNAL_RED]
        ),
    }


CMAPS = make_cmaps()


def configure_matplotlib(font_size: float = 6.7) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.titlesize": font_size + 0.4,
            "axes.labelsize": font_size - 0.2,
            "xtick.labelsize": font_size - 0.8,
            "ytick.labelsize": font_size - 0.8,
            "legend.fontsize": font_size - 1.0,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "axes.facecolor": PANEL_FACE,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
        }
    )


def clean_quant_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_linewidth(0.52)
    ax.spines["bottom"].set_linewidth(0.52)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, lw=0.34)
    ax.tick_params(axis="both", colors=TEXT, width=0.48, length=2.3, pad=1.2)


def spatial_background(ax: plt.Axes, x, y, *, size: float = 1.2, alpha: float = 0.24) -> None:
    ax.set_facecolor(FIG_BG)
    ax.scatter(x, y, s=size, c=TISSUE_BG, alpha=alpha, linewidths=0, zorder=1)


def style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, *, dx: float = -0.046, dy: float = 0.024) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )


def short_panel_title(ax: plt.Axes, title: str, *, pad: float = 2.0) -> None:
    ax.set_title(title, fontsize=7.1, color=TEXT, pad=pad, fontweight="normal")


def export_figure(fig: plt.Figure, stem, *, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
