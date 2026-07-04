#!/usr/bin/env python
"""DSS9 versus D0 differential signaling plot, CytoSignal-style."""

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
    aggregate_receiver_scores,
    draw_spatial_heatmap,
    setup_publication_style,
    spatial_activity_grid,
)


OUT = BASE / "figures/merfish_dss_differential_signaling_v1"
TRAJ = BASE / "stagewise_spatialself/summary/stage_lr_trajectories.csv"
LR_PAIR = "Cxcl10-Cxcr3"
STAGES = ["Healthy_D0", "DSS9"]
LABELS = ["D0", "D9"]
INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
BG = "#FFFFFF"
D9 = "#C94F48"
D0 = "#8A8177"
TEAL = "#237A78"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()
    setup_style()
    traj = pd.read_csv(TRAJ)
    volcano = build_volcano(traj)
    stage_cells = [aggregate_receiver_scores(stage, [LR_PAIR], "SpatialSelf") for stage in STAGES]
    volcano.to_csv(source_dir / "dss9_vs_d0_lr_volcano.csv", index=False)
    for label, cells in zip(LABELS, stage_cells, strict=True):
        cells.to_csv(source_dir / f"{label}_{LR_PAIR}_receiver_activity.csv", index=False)
    fig = plot_differential(volcano, stage_cells)
    out = OUT / "mouse_colon_merfish_dss9_vs_d0_differential_signaling"
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 950
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
            "legend.fontsize": 5.4,
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


def build_volcano(traj: pd.DataFrame) -> pd.DataFrame:
    df = traj.copy()
    df = df[np.isfinite(df["dss9_delta"].to_numpy(dtype=float, copy=True))].copy()
    df["effect"] = df["dss9_delta"].astype(float)
    df["abs_effect"] = df["effect"].abs()
    # This is an effect-size volcano-like plot; no formal p-values are available in the stage summary.
    df["evidence_score"] = -np.log10(np.clip(1.0 - df["abs_effect"], 1e-4, 1.0))
    df["highlight"] = df["lr_pair"].eq(LR_PAIR)
    df["direction"] = np.where(df["effect"] > 0, "D9-up", "D9-down")
    return df


def plot_differential(volcano: pd.DataFrame, stage_cells: list[pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(6.65, 2.72), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.985, bottom=0.17, top=0.84, wspace=0.23, width_ratios=[1.15, 0.82, 0.82])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    plot_volcano(ax_a, volcano)
    grids = [spatial_activity_grid(c, bins=210, sigma=2.45)[0] for c in stage_cells]
    vmax = max(float(np.nanquantile(g[g > 0], 0.96)) if np.any(g > 0) else 0.0 for g in grids)
    vmax = max(vmax, 1e-6)
    for ax, cells, label, color in zip([ax_b, ax_c], stage_cells, LABELS, [D0, D9], strict=True):
        draw_spatial_heatmap(ax, cells, vmax=vmax, cmap=ACTIVITY_CMAP, bins=210, sigma=2.45)
        ax.set_title(f"{label} {LR_PAIR}", fontsize=6.8, pad=2.0, color=color)
    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    panel_label(fig, ax_c, "c")
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=ACTIVITY_CMAP)
    cax = ax_c.inset_axes([0.58, -0.17, 0.37, 0.055])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["0", "high"])
    cb.ax.tick_params(length=0, pad=1.0, labelsize=5.3)
    cb.outline.set_visible(False)
    return fig


def plot_volcano(ax: plt.Axes, volcano: pd.DataFrame) -> None:
    colors = np.where(volcano["effect"].gt(0), D9, D0)
    ax.scatter(volcano["effect"], volcano["evidence_score"], s=9.0, c=colors, alpha=0.50, linewidths=0, rasterized=True)
    hit = volcano[volcano["highlight"]]
    ax.scatter(hit["effect"], hit["evidence_score"], s=38, facecolors=BG, edgecolors=D9, linewidths=0.9, zorder=5)
    for row in hit.itertuples(index=False):
        ax.text(row.effect + 0.018, row.evidence_score + 0.02, row.lr_pair, fontsize=5.4, color=D9, ha="left", va="bottom")
    ax.axvline(0, color="#8A8177", lw=0.55)
    ax.set_xlabel("DSS9 - D0 SpatialSelf activity", fontsize=5.9)
    ax.set_ylabel("effect-size evidence", fontsize=5.9)
    ax.set_title("DSS9 differential signaling", fontsize=6.8, pad=2.0)
    ax.grid(axis="y", color=GRID, lw=0.42)
    ax.tick_params(axis="both", labelsize=5.2, length=2.0, pad=1)
    ax.set_xlim(-0.72, 0.96)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.030, box.y1 + 0.016, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
