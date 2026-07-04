"""Shared Phase 14 manuscript figure style.

Phase 14 prioritizes visual hierarchy over table density: one hero panel,
restrained colors, larger readable labels and explicit claim boundaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "ink": "#1F2328",
    "muted": "#66717E",
    "line": "#C7CDD4",
    "grid": "#E4E8EC",
    "paper": "#FFFFFF",
    "soft_bg": "#F6F8FA",
    "spatialself": "#2468A2",
    "spatialself_dark": "#173F61",
    "signal": "#2F8C62",
    "signal_soft": "#DDEFE6",
    "wavelet": "#3D9A9B",
    "lr": "#7B5CA8",
    "weak": "#C9952F",
    "weak_soft": "#F4E1B8",
    "risk": "#B85C5A",
    "risk_soft": "#F1D8D6",
    "neutral": "#AEB6BF",
    "neutral_dark": "#5E6873",
    "neutral_soft": "#E9EDF1",
}


METHOD_COLORS = {
    "SpatialSelf": PALETTE["spatialself"],
    "COMMOT": "#6C737C",
    "CellChat": "#9EA5AD",
    "SpaTalk": "#B58A3C",
    "CellNEST": "#8B78A8",
    "DeepTalk": "#B9C0C8",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 500) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def clean_axes(ax: plt.Axes, *, grid: str | None = "y") -> None:
    if grid:
        ax.grid(axis=grid, color=PALETTE["grid"], linewidth=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])


def add_figure_title(fig: plt.Figure, title: str, subtitle: str, *, x: float = 0.055) -> None:
    fig.suptitle(title, x=x, y=0.985, ha="left", fontsize=12.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(x, 0.948, subtitle, ha="left", va="top", fontsize=6.6, color=PALETTE["muted"])

