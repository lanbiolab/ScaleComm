#!/usr/bin/env python
"""Plot a clean MERFISH DSS colitis communication story figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
TRAJ = BASE / "multislice_spatialself/summary/stage_lr_trajectories.csv"
LONG = BASE / "multislice_spatialself/summary/stage_lr_summary_long.csv"
THREE = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "figures/merfish_dss_story"

STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
KEY_AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
DISPLAY_LR = {
    "Cxcl10-Cxcr3": "Cxcl10-Cxcr3",
    "Ccl5-Ccr5": "Ccl5-Ccr5",
    "Ccl2-Ccr2": "Ccl2-Ccr2",
    "Cxcl9-Cxcr3": "Cxcl9-Cxcr3",
    "Cxcl12-Cxcr4": "Cxcl12-Cxcr4",
    "Tnf-Tnfrsf1a": "Tnf-Tnfrsf1a",
    "Tnf-Tnfrsf1b": "Tnf-Tnfrsf1b",
}

INK = "#232323"
MUTED = "#7D7D7D"
LIGHT = "#DADADA"
GRID = "#E9E6E0"
BACKGROUND = "#FFFFFF"
OUR = "#B64F3F"
COMMOT = "#4E7B9D"
CELLNEST = "#9A8A58"
METHOD_COLORS = {"SpatialSelf": OUR, "COMMOT": COMMOT, "CellNEST": CELLNEST}
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "warm_clay", ["#F7F3EC", "#E9C6A1", "#C87752", "#7E2D25"]
)
DIVERGE = LinearSegmentedColormap.from_list(
    "soft_diverge", ["#496B8D", "#F7F3EC", "#A8473C"]
)


def setup_style():
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
        }
    )


def panel_label(ax, label):
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )


def clean_axes(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def zscore_rows(mat):
    arr = mat.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def plot_timeline(ax):
    ax.set_axis_off()
    panel_label(ax, "a")

    xs = np.array([0.08, 0.36, 0.64, 0.92])
    y = 0.53
    ax.plot([xs[0], xs[-1]], [y, y], color=INK, lw=0.9, solid_capstyle="round")

    for i, (x, lab) in enumerate(zip(xs, STAGE_LABELS)):
        color = OUR if lab == "D9" else ("#C8B58C" if lab == "D21" else "#E3D6C5")
        edge = INK if lab in {"D0", "D9"} else MUTED
        circ = Circle((x, y), 0.052, facecolor=color, edgecolor=edge, lw=0.7)
        ax.add_patch(circ)
        ax.text(x, y + 0.13, lab, ha="center", va="center", fontsize=8, color=INK)
        if i > 0:
            ax.text(x, y - 0.14, "DSS", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(0.08, y - 0.14, "control", ha="center", va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def select_heatmap_rows(traj):
    curated = [
        "Cxcl10-Cxcr3",
        "Ccl5-Ccr5",
        "Ccl2-Ccr2",
        "Cxcl9-Cxcr3",
        "Cxcl12-Cxcr4",
        "Cxcl5-Cxcr2",
        "Tnf-Tnfrsf1a",
        "Tnf-Tnfrsf1b",
    ]
    available = [x for x in curated if x in set(traj["lr_pair"])]
    return available


def plot_spatialself_heatmap(ax, traj):
    panel_label(ax, "b")
    rows = select_heatmap_rows(traj)
    sub = traj.set_index("lr_pair").loc[rows, STAGES]
    z = np.clip(zscore_rows(sub), -1.8, 1.8)

    im = ax.imshow(z, cmap=HEAT_CMAP, vmin=-1.8, vmax=1.8, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY_LR.get(x, x) for x in rows], fontsize=6.7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("SpatialSelf LR activity", loc="left", fontsize=7.5, color=INK, pad=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
    cbar.outline.set_visible(False)


def plot_key_trajectories(ax, long_df, traj):
    panel_label(ax, "c")
    offsets = {"Cxcl10-Cxcr3": -0.035, "Ccl5-Ccr5": 0.0, "Ccl2-Ccr2": 0.035}
    x = np.arange(len(STAGES))

    for lr in KEY_AXES:
        rep = long_df.loc[long_df["lr_pair"] == lr].copy()
        rep["stage"] = pd.Categorical(rep["stage"], categories=STAGES, ordered=True)
        means = traj.loc[traj["lr_pair"] == lr, STAGES].iloc[0].astype(float).to_numpy()
        color = {"Cxcl10-Cxcr3": OUR, "Ccl5-Ccr5": "#C97842", "Ccl2-Ccr2": "#8F5B37"}[lr]
        jitter = offsets[lr]
        for i, st in enumerate(STAGES):
            vals = rep.loc[rep["stage"] == st, "score_q95"].astype(float).to_numpy()
            if len(vals):
                xpos = i + jitter + np.linspace(-0.01, 0.01, len(vals))
                ax.scatter(
                    xpos,
                    vals,
                    s=13,
                    facecolor=BACKGROUND,
                    edgecolor=color,
                    lw=0.75,
                    zorder=3,
                )
        ax.plot(x + jitter, means, color=color, lw=1.35, marker="o", ms=2.8, label=DISPLAY_LR[lr])

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=7)
    ax.set_ylabel("score", fontsize=7)
    ax.set_ylim(0.72, 1.015)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.7, borderaxespad=0.1)
    clean_axes(ax)


def plot_method_delta(ax, three):
    panel_label(ax, "d")
    rows = KEY_AXES
    methods = ["SpatialSelf", "COMMOT", "CellNEST"]
    mat = np.full((len(rows), len(methods)), np.nan)
    for i, lr in enumerate(rows):
        for j, method in enumerate(methods):
            r = three.loc[(three["lr_pair"] == lr) & (three["method"] == method)]
            if len(r):
                mat[i, j] = float(r["dss9_minus_healthy"].iloc[0])

    vmax = np.nanmax(np.abs(mat))
    vmax = max(vmax, 0.12)
    im = ax.imshow(mat, cmap=DIVERGE, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7, rotation=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY_LR[x] for x in rows], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("D9 minus D0", loc="left", fontsize=7.5, color=INK, pad=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
    cbar.outline.set_visible(False)


def build_figure():
    setup_style()
    traj = pd.read_csv(TRAJ)
    long_df = pd.read_csv(LONG)
    three = pd.read_csv(THREE)

    fig = plt.figure(figsize=(7.25, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 1.26],
        height_ratios=[0.72, 1.06, 0.88],
        left=0.08,
        right=0.97,
        top=0.94,
        bottom=0.11,
        wspace=0.48,
        hspace=0.66,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[2, 0])

    plot_timeline(ax_a)
    plot_spatialself_heatmap(ax_b, traj)
    plot_key_trajectories(ax_c, long_df, traj)
    plot_method_delta(ax_d, three)

    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUT / "mouse_colon_merfish_dss_story_clean"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
