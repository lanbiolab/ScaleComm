#!/usr/bin/env python
"""CytoSignal-style stage activity maps for DSS MERFISH SpatialSelf results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (  # noqa: E402
    ACTIVITY_CMAP,
    BASE,
    INFLAMMATORY_AXES,
    STAGES,
    STAGE_LABELS,
    aggregate_receiver_scores,
    draw_spatial_heatmap,
    setup_publication_style,
    spatial_activity_grid,
)


OUT = BASE / "figures/merfish_dss_cytosignal_style_stage_activity_v2"
INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
BG = "#FFFFFF"
STAGE_ACCENTS = {
    "D0": "#8A8177",
    "D3": "#D9992B",
    "D9": "#C94F48",
    "D21": "#237A78",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()
    setup_style()
    stage_cells = [aggregate_receiver_scores(stage, INFLAMMATORY_AXES, "SpatialSelf") for stage in STAGES]
    for label, cells in zip(STAGE_LABELS, stage_cells, strict=True):
        cells.to_csv(source_dir / f"{label}_inflammatory_activity_cells.csv", index=False)
    fig = plot_stage_maps(stage_cells)
    out = OUT / "mouse_colon_merfish_dss_cytosignal_style_stage_activity"
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 900
        fig.savefig(f"{out}.{ext}", **kwargs)
    plt.close(fig)
    print(f"{out}.png")
    print(f"{out}.pdf")


def setup_style() -> None:
    setup_publication_style()
    mpl.rcParams.update(
        {
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.55,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_stage_maps(stage_cells: list[pd.DataFrame]) -> plt.Figure:
    grids = [spatial_activity_grid(c, bins=220, sigma=2.55)[0] for c in stage_cells]
    vmax = max(float(np.nanquantile(g[g > 0], 0.955)) if np.any(g > 0) else 0.0 for g in grids)
    vmax = max(vmax, 1e-6)
    fig = plt.figure(figsize=(7.45, 2.22), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, left=0.038, right=0.992, bottom=0.14, top=0.76, wspace=0.055)
    axes = []
    for i, (cells, label) in enumerate(zip(stage_cells, STAGE_LABELS, strict=True)):
        ax = fig.add_subplot(gs[0, i])
        draw_spatial_heatmap(ax, cells, vmax=vmax, cmap=ACTIVITY_CMAP, bins=220, sigma=2.55)
        ax.set_title(label, fontsize=7.4, pad=1.0, color=STAGE_ACCENTS[label])
        for spine in ax.spines.values():
            spine.set_linewidth(0.48)
            spine.set_color("#8A8177")
        axes.append(ax)
    panel_label(fig, axes[0], "a")
    fig.text(0.045, 0.935, "SpatialSelf inflammatory chemokine activity across DSS stages", ha="left", va="top", fontsize=7.8, color=INK)
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=ACTIVITY_CMAP)
    cax = axes[-1].inset_axes([0.60, -0.14, 0.36, 0.052])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["0", "high"])
    cb.ax.tick_params(length=0, pad=1.0, labelsize=5.4)
    cb.outline.set_visible(False)
    return fig


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.022, box.y1 + 0.018, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
