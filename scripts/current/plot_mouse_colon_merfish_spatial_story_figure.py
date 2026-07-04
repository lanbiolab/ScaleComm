#!/usr/bin/env python
"""Spatially grounded MERFISH DSS colitis communication story figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
REPS = BASE / "multislice_replicates"
SPATIALSELF = BASE / "multislice_spatialself/predictions"
COMMOT = BASE / "commot_priority_baseline/predictions"
CELLNEST = BASE / "cellnest_priority_baseline"
TRAJ = BASE / "multislice_spatialself/summary/stage_lr_trajectories.csv"
LONG = BASE / "multislice_spatialself/summary/stage_lr_summary_long.csv"
THREE = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "figures/merfish_dss_spatial_story"

STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
LR_MAIN = "Cxcl10-Cxcr3"
LR_PANEL = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2", "Cxcl9-Cxcr3", "Cxcl12-Cxcr4", "Cxcl5-Cxcr2"]
KEY_TRAJ = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]

INK = "#1F1F1F"
MUTED = "#8A8A8A"
PALE = "#D9D6CF"
GRID = "#E9E6E0"
BG = "#FFFFFF"
OUR = "#C44536"
COMMOT_C = "#4F7EA7"
CELLNEST_C = "#A08A4C"
COMPARTMENT_COLORS = {
    "Epithelial": "#D99142",
    "Immune": "#B8517A",
    "Stromal": "#4D9A88",
    "Vascular": "#5B86B8",
    "Neural": "#8B75B7",
    "Other": "#CFCBC2",
}
HOT = LinearSegmentedColormap.from_list("spatial_hot", ["#EEEAE2", "#E4B07E", "#C85A3E", "#6F1D1B"])
HEAT = LinearSegmentedColormap.from_list("program_heat", ["#F7F2EA", "#E2B98E", "#BE6747", "#782B27"])
DIVERGE = LinearSegmentedColormap.from_list("delta_div", ["#4E6E8F", "#F5F1EA", "#AA453B"])


def style():
    mpl.rcParams.update(
        {
            "font.size": 6.8,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def label(ax, letter, x=-0.04, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="bottom", color=INK)


def rep_dir(stage):
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    row = manifest.loc[manifest["stage"].eq(stage)].sort_values("stage_replicate").iloc[0]
    return ROOT / row["replicate_dir"]


def pred_path(stage, rep_name, method):
    if method == "SpatialSelf":
        return SPATIALSELF / stage / "spatialself_wavelet_adaptive" / f"{rep_name}__spatialself_wavelet_adaptive_predictions.csv"
    if method == "COMMOT":
        return COMMOT / f"{rep_name}__commot_priority_predictions.csv"
    if method == "CellNEST":
        return CELLNEST / f"{rep_name}__cellnest_predictions.csv"
    raise ValueError(method)


def compartment(cell_type):
    s = str(cell_type).lower()
    if any(k in s for k in ["colonocyte", "epithelial", "goblet", "tuft", "stem", "ta", "iae"]):
        return "Epithelial"
    if any(k in s for k in ["t cell", "b cell", "dc", "mac", "neutrophil", "mast", "plasma", "lymph"]):
        return "Immune"
    if any(k in s for k in ["fibro", "iaf", "frc", "smc", "telocyte"]):
        return "Stromal"
    if any(k in s for k in ["ec", "endothelial", "arterial", "venous", "capillarial", "lyve"]):
        return "Vascular"
    if any(k in s for k in ["neuron", "glia"]):
        return "Neural"
    return "Other"


def read_stage(stage):
    d = rep_dir(stage)
    return d.name, pd.read_csv(d / "cells.csv"), pd.read_csv(d / "candidate_interactions.csv")


def cell_scores(stage, lr_pair, method="SpatialSelf"):
    rep_name, cells, cand = read_stage(stage)
    lig, rec = lr_pair.split("-")
    sub = cand.loc[(cand["ligand"] == lig) & (cand["receptor"] == rec)].copy()
    if sub.empty:
        cells = cells.copy()
        cells["incoming_score"] = 0.0
        return cells
    p = pred_path(stage, rep_name, method)
    pred = pd.read_csv(p)
    sub = sub.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    incoming = sub.groupby("receiver_id", as_index=False)["score"].max().rename(
        columns={"receiver_id": "cell_id", "score": "incoming_score"}
    )
    cells = cells.merge(incoming, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    return cells


def format_spatial(ax, cells):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    pad_x = (x.max() - x.min()) * 0.045
    pad_y = (y.max() - y.min()) * 0.045
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)


def plot_compartment(ax, stage, title=None):
    _, cells, _ = read_stage(stage)
    cells = cells.copy()
    cells["compartment"] = cells["cell_type"].map(compartment)
    for comp, col in COMPARTMENT_COLORS.items():
        sub = cells.loc[cells["compartment"].eq(comp)]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], s=3.6, c=col, alpha=0.86, linewidths=0, rasterized=True)
    format_spatial(ax, cells)
    if title:
        ax.set_title(title, fontsize=7, pad=2.5, color=INK)


def plot_hotspot(ax, stage, lr_pair, method="SpatialSelf", vmax=None, title=None):
    cells = cell_scores(stage, lr_pair, method)
    vals = cells["incoming_score"].to_numpy()
    if vmax is None:
        vmax = np.nanquantile(vals[vals > 0], 0.98) if np.any(vals > 0) else 1.0
    ax.scatter(cells["x"], cells["y"], s=2.6, c="#D8D5CE", alpha=0.42, linewidths=0, rasterized=True)
    positive = cells.loc[cells["incoming_score"] > 0].copy()
    if len(positive):
        positive = positive.sort_values("incoming_score")
        ax.scatter(
            positive["x"],
            positive["y"],
            s=5.0,
            c=np.clip(positive["incoming_score"], 0, vmax),
            cmap=HOT,
            vmin=0,
            vmax=vmax,
            alpha=0.96,
            linewidths=0,
            rasterized=True,
        )
    format_spatial(ax, cells)
    if title:
        ax.set_title(title, fontsize=7, pad=2.5, color=INK)
    return vmax


def zscore_rows(df):
    arr = df.to_numpy(float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return np.clip((arr - mu) / sd, -1.7, 1.7)


def plot_program_heat(ax):
    traj = pd.read_csv(TRAJ).set_index("lr_pair")
    rows = [x for x in LR_PANEL if x in traj.index]
    mat = pd.DataFrame(zscore_rows(traj.loc[rows, STAGES]), index=rows, columns=STAGE_LABELS)
    im = ax.imshow(mat.to_numpy(), cmap=HEAT, vmin=-1.7, vmax=1.7, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def plot_method_delta(ax):
    three = pd.read_csv(THREE)
    methods = ["SpatialSelf", "COMMOT", "CellNEST"]
    mat = np.full((len(KEY_TRAJ), len(methods)), np.nan)
    for i, lr in enumerate(KEY_TRAJ):
        for j, method in enumerate(methods):
            r = three.loc[(three["lr_pair"] == lr) & (three["method"] == method)]
            if len(r):
                mat[i, j] = r["dss9_minus_healthy"].iloc[0]
    vmax = max(np.nanmax(np.abs(mat)), 0.16)
    im = ax.imshow(mat, cmap=DIVERGE, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=6.4)
    ax.set_yticks(range(len(KEY_TRAJ)))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def plot_trajectory(ax):
    long_df = pd.read_csv(LONG)
    traj = pd.read_csv(TRAJ)
    xs = np.arange(4)
    colors = {"Cxcl10-Cxcr3": OUR, "Ccl5-Ccr5": "#D07B3F", "Ccl2-Ccr2": "#8B5B3B"}
    for lr in KEY_TRAJ:
        rep = long_df.loc[long_df["lr_pair"].eq(lr)].copy()
        means = traj.loc[traj["lr_pair"].eq(lr), STAGES].iloc[0].to_numpy(float)
        c = colors[lr]
        for i, st in enumerate(STAGES):
            vals = rep.loc[rep["stage"].eq(st), "score_q95"].to_numpy(float)
            if len(vals):
                ax.scatter(i + np.linspace(-0.025, 0.025, len(vals)), vals, s=9, facecolor=BG, edgecolor=c, lw=0.7, zorder=3)
        ax.plot(xs, means, lw=1.25, marker="o", ms=2.5, color=c, label=lr)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.5)
    ax.set_ylabel("score", fontsize=6.5)
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(0.72, 1.01)
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=5.4, handlelength=1.35, columnspacing=0.75)


def add_compartment_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgewidth=0, markersize=4.2, label=k)
        for k, c in COMPARTMENT_COLORS.items()
        if k != "Other"
    ]
    ax.legend(handles=handles, loc="center", ncol=3, fontsize=5.8, handletextpad=0.25, columnspacing=0.9)
    ax.set_axis_off()


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)

    # Shared scale for the stage-wise SpatialSelf hotspot maps.
    all_vals = []
    for st in STAGES:
        all_vals.append(cell_scores(st, LR_MAIN, "SpatialSelf")["incoming_score"].to_numpy())
    vmax_hot = np.nanquantile(np.concatenate(all_vals), 0.985)
    vmax_hot = max(float(vmax_hot), 1e-6)

    fig = plt.figure(figsize=(7.65, 6.75))
    gs = fig.add_gridspec(
        4,
        8,
        height_ratios=[0.92, 0.92, 0.96, 0.98],
        width_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
        left=0.055,
        right=0.97,
        top=0.965,
        bottom=0.075,
        wspace=0.05,
        hspace=0.24,
    )

    atlas_axes = [fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]) for i in range(4)]
    hot_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(4)]
    method_axes = [fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4:6])]
    leg_ax = fig.add_subplot(gs[2, 6:8])
    ax_heat = fig.add_subplot(gs[3, 0:3])
    ax_traj = fig.add_subplot(gs[3, 3:6])
    ax_delta = fig.add_subplot(gs[3, 6:8])

    label(atlas_axes[0], "a", x=-0.10)
    for ax, stage, lab in zip(atlas_axes, STAGES, STAGE_LABELS):
        plot_compartment(ax, stage, lab)
    add_compartment_legend(leg_ax)

    label(hot_axes[0], "b", x=-0.10)
    for ax, stage, lab in zip(hot_axes, STAGES, STAGE_LABELS):
        plot_hotspot(ax, stage, LR_MAIN, "SpatialSelf", vmax=vmax_hot, title=lab)

    label(method_axes[0], "c", x=-0.10)
    method_titles = ["SpatialSelf", "COMMOT", "CellNEST"]
    for ax, method in zip(method_axes, method_titles):
        plot_hotspot(ax, "DSS9", LR_MAIN, method, title=method)

    label(ax_heat, "d", x=-0.08)
    plot_program_heat(ax_heat)

    label(ax_traj, "e", x=-0.13)
    plot_trajectory(ax_traj)

    label(ax_delta, "f", x=-0.13)
    im_delta = plot_method_delta(ax_delta)
    cbar2 = fig.colorbar(im_delta, ax=ax_delta, fraction=0.052, pad=0.035)
    cbar2.set_ticks([-0.2, 0, 0.2])
    cbar2.ax.tick_params(labelsize=5.8, length=1.8, width=0.45)
    cbar2.outline.set_visible(False)

    stem = OUT / "mouse_colon_merfish_dss_spatial_story"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=650, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=650, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
