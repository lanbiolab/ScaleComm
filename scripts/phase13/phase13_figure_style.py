"""Shared Phase 13 manuscript figure style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "ink": "#222222",
    "muted": "#6A6A6A",
    "grid": "#D9DDE2",
    "paper": "#FFFFFF",
    "soft_bg": "#F5F7F8",
    "line": "#BFC7D0",
    "spatialself": "#2C6FBB",
    "spatialself_dark": "#174A7C",
    "wavelet": "#4CA3A3",
    "lr": "#7E57A0",
    "hard_negative": "#B95550",
    "hard_negative_soft": "#E8B6B1",
    "weak": "#D39B2A",
    "weak_soft": "#F1D6A0",
    "neutral": "#B8BDC4",
    "neutral_soft": "#E6E9ED",
    "success": "#4C8A63",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def save_figure(fig: plt.Figure, out_stem: Path, *, dpi: int = 450) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=dpi, bbox_inches="tight")


def soften_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
