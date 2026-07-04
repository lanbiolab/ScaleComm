#!/usr/bin/env python
"""Plot SPP1-CD44 observed and negative-control spatial maps for Visium HD CRC."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

TEXT = "#202020"

PANELS = [
    ("spatialself_spp1_cd44_incoming", "SPP1-CD44 incoming", "YlGn", "observed"),
    ("SPP1", "SPP1 ligand", "YlOrBr", "biology"),
    ("CD44", "CD44 receptor", "PuBuGn", "biology"),
    ("target_signature", "Target signature", "YlGnBu", "biology"),
    ("APOE", "APOE target", "YlGnBu", "target"),
    ("CD68", "CD68 target", "YlGnBu", "target"),
    ("COL1A1", "COL1A1 target", "YlGnBu", "target"),
    ("FN1", "FN1 target", "YlGnBu", "target"),
    ("spatialself_high_expression_negative_incoming", "High-expression negative", "Greys", "control"),
    ("spatialself_shuffled_receiver_spp1_cd44", "Shuffled receiver", "Greys", "control"),
    ("spatialself_wrong_niche_spp1_cd44", "Wrong niche", "Greys", "control"),
    ("spatialself_wrong_type_spp1_cd44", "Wrong type", "Reds", "competitive"),
]


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    spatial = pd.read_csv(RESULT_DIR / "spp1_cd44_case_spatial_features.csv")
    source_columns = ["cell_id", "x", "y", "cell_type", "niche"] + [name for name, *_ in PANELS if name in spatial.columns]
    spatial[source_columns].to_csv(SOURCE_DIR / "spp1_cd44_spatial_control_maps.csv", index=False)

    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        left=0.055,
        right=0.985,
        top=0.885,
        bottom=0.095,
        wspace=0.13,
        hspace=0.26,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]
    for ax, panel in zip(axes, PANELS):
        column, title, cmap, group = panel
        plot_spatial(ax, spatial, column, title, cmap, group)

    for label, ax in zip(list("abcdefghijkl"), axes):
        add_panel_label(ax, label)

    fig.suptitle(
        "SPP1-CD44 spatial signal is aligned with receiver targets and stress-tested by spatial controls",
        x=0.055,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.055,
        0.925,
        "Observed incoming, sender/receptor expression and target responses share a tumor-myeloid spatial context; shuffled and wrong-niche controls lose this pattern, while wrong-type remains a flagged limitation.",
        fontsize=6.2,
        color="#555555",
    )
    fig.text(
        0.055,
        0.035,
        "Control maps use the same receiver-level masking definitions as the Phase 10 target-impact analysis. Values are independently color-scaled within each panel to show spatial pattern, not absolute cross-panel magnitude.",
        fontsize=5.7,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_spp1_cd44_spatial_control_maps")
    plt.close(fig)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def plot_spatial(ax: plt.Axes, table: pd.DataFrame, column: str, title: str, cmap: str, group: str) -> None:
    if column not in table.columns:
        ax.axis("off")
        ax.set_title(f"{title}\nmissing", loc="left", pad=3, color="#9A9A9A")
        return
    values = pd.to_numeric(table[column], errors="coerce").fillna(0).to_numpy(dtype=float)
    if np.nanstd(values) <= 1e-12:
        vmax = max(float(np.nanmax(values)), 1e-8)
        vmin = 0.0
    else:
        vmax = float(np.nanquantile(values, 0.985))
        vmin = float(np.nanquantile(values, 0.02)) if np.nanmin(values) < 0 else 0.0
    sc = ax.scatter(
        table["x"],
        -table["y"],
        c=values,
        s=2.7,
        cmap=cmap,
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-8),
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, loc="left", pad=3, color=title_color(group), fontsize=6.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.ax.tick_params(labelsize=4.4, length=1.8, pad=1)


def title_color(group: str) -> str:
    if group == "observed":
        return "#2F6B4F"
    if group == "competitive":
        return "#A34242"
    if group == "control":
        return "#555555"
    return TEXT


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.03, 1.06, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
