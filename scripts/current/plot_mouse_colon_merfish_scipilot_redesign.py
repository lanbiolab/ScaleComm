#!/usr/bin/env python
"""NC-style spatial story figure for mouse colon MERFISH CCC."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
REPS = BASE / "multislice_replicates"
PRED = BASE / "multislice_spatialself/predictions"
COMMOT = BASE / "commot_priority_baseline/predictions"
CELLNEST = BASE / "cellnest_priority_baseline"
TRAJ = BASE / "multislice_spatialself/summary/stage_lr_trajectories.csv"
LONG = BASE / "multislice_spatialself/summary/stage_lr_summary_long.csv"
THREE = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "figures/merfish_dss_scipilot_redesign"

STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
LR_MAIN = "Cxcl10-Cxcr3"
LR_ROWS = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2", "Cxcl9-Cxcr3", "Cxcl12-Cxcr4", "Cxcl5-Cxcr2"]
KEY_TRAJ = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]

INK = "#222222"
MUTED = "#8E8A83"
TISSUE = "#D6D2C8"
GRID = "#EAE6DE"
BG = "#FFFFFF"
RED = "#B33A2E"
RED_DARK = "#6F1D1B"
ORANGE = "#D98A32"
BLUE = "#4E6F8F"
BROWN = "#8A5A38"

COMPARTMENT = {
    "Epithelial": "#D99142",
    "Immune": "#B84D73",
    "Stromal": "#4C9A88",
    "Vascular": "#587FA9",
    "Neural": "#8B74B5",
    "Other": "#C9C5BC",
}
HOT = LinearSegmentedColormap.from_list("hot_clay", ["#F8F4EC", "#E6BE93", "#C65E43", "#6D1D1B"])
HEAT = LinearSegmentedColormap.from_list("program_clay", ["#F7F2EA", "#E0B588", "#BF6245", "#782B27"])
DIV = LinearSegmentedColormap.from_list("delta_div", ["#4C6D8C", "#F6F1EA", "#A94439"])


def style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 6.6,
            "axes.linewidth": 0.62,
            "axes.edgecolor": INK,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)


def stage_dir(stage: str, replicate: int = 1) -> Path:
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    row = manifest.loc[(manifest["stage"] == stage) & (manifest["stage_replicate"] == replicate)].iloc[0]
    return ROOT / row["replicate_dir"]


def pred_path(stage: str, rep_name: str, method: str) -> Path:
    if method == "SpatialSelf":
        return PRED / stage / "spatialself_wavelet_adaptive" / f"{rep_name}__spatialself_wavelet_adaptive_predictions.csv"
    if method == "COMMOT":
        return COMMOT / f"{rep_name}__commot_priority_predictions.csv"
    if method == "CellNEST":
        return CELLNEST / f"{rep_name}__cellnest_predictions.csv"
    raise ValueError(method)


def classify_cell_type(cell_type: str) -> str:
    s = str(cell_type).lower()
    if any(k in s for k in ["colonocyte", "epithelial", "goblet", "stem", "tuft", "iae", "ta"]):
        return "Epithelial"
    if any(k in s for k in ["macrophage", "neutrophil", "t cell", "b cell", "dc", "treg", "mast", "plasma", "lymph"]):
        return "Immune"
    if any(k in s for k in ["fibro", "iaf", "frc", "smc", "telocyte"]):
        return "Stromal"
    if any(k in s for k in ["ec", "endothelial", "arterial", "venous", "capillarial", "lymphatic"]):
        return "Vascular"
    if any(k in s for k in ["neuron", "glia"]):
        return "Neural"
    return "Other"


def load_slice(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    d = stage_dir(stage, replicate)
    return d.name, pd.read_csv(d / "cells.csv"), pd.read_csv(d / "candidate_interactions.csv")


def edge_table(stage: str, lr_pair: str, method: str = "SpatialSelf", replicate: int = 1) -> pd.DataFrame:
    rep, cells, cand = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    sub = cand.loc[(cand["ligand"] == ligand) & (cand["receptor"] == receptor)].copy()
    pred = pd.read_csv(pred_path(stage, rep, method))
    sub = sub.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    if sub.empty:
        return sub
    send = cells[["cell_id", "x", "y", "cell_type"]].rename(
        columns={"cell_id": "sender_id", "x": "sender_x", "y": "sender_y", "cell_type": "sender_type"}
    )
    recv = cells[["cell_id", "x", "y", "cell_type"]].rename(
        columns={"cell_id": "receiver_id", "x": "receiver_x", "y": "receiver_y", "cell_type": "receiver_type"}
    )
    return sub.merge(send, on="sender_id", how="left").merge(recv, on="receiver_id", how="left")


def receiver_scores(stage: str, lr_pair: str, method: str = "SpatialSelf", replicate: int = 1, rank_norm: bool = False) -> pd.DataFrame:
    _, cells, _ = load_slice(stage, replicate)
    edges = edge_table(stage, lr_pair, method, replicate)
    cells = cells.copy()
    if edges.empty:
        cells["incoming_score"] = 0.0
        return cells
    inc = edges.groupby("receiver_id", as_index=False)["score"].max().rename(
        columns={"receiver_id": "cell_id", "score": "incoming_score"}
    )
    cells = cells.merge(inc, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    if rank_norm and (cells["incoming_score"] > 0).any():
        pos = cells["incoming_score"] > 0
        cells.loc[pos, "incoming_score"] = cells.loc[pos, "incoming_score"].rank(pct=True)
    return cells


def set_spatial_axis(ax, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    dx = (x.max() - x.min()) * 0.045
    dy = (y.max() - y.min()) * 0.045
    ax.set_xlim(x.min() - dx, x.max() + dx)
    ax.set_ylim(y.min() - dy, y.max() + dy)


def plot_compartment_map(ax, stage: str, title: str) -> None:
    _, cells, _ = load_slice(stage)
    cells["compartment"] = cells["cell_type"].map(classify_cell_type)
    for comp in ["Other", "Epithelial", "Stromal", "Immune", "Vascular", "Neural"]:
        sub = cells.loc[cells["compartment"] == comp]
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], s=5.0, color=COMPARTMENT[comp], alpha=0.84, linewidths=0)
    set_spatial_axis(ax, cells)
    ax.set_title(title, fontsize=7.2, color=INK, pad=2.5)


def density_array(cells: pd.DataFrame, bins: int = 150, sigma: float = 3.2) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    w = cells["incoming_score"].to_numpy(float)
    pad_x = (x.max() - x.min()) * 0.045
    pad_y = (y.max() - y.min()) * 0.045
    extent = (x.min() - pad_x, x.max() + pad_x, y.min() - pad_y, y.max() + pad_y)
    h, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]], weights=w)
    z = gaussian_filter(h.T, sigma=sigma)
    if z.max() > 0:
        z = z / z.max()
    return z, extent


def plot_density_map(ax, cells: pd.DataFrame, title: str, show_top: bool = True) -> None:
    ax.scatter(cells["x"], cells["y"], s=3.4, color=TISSUE, alpha=0.46, linewidths=0)
    z, extent = density_array(cells)
    alpha = np.clip(z, 0, 1) ** 0.62 * 0.92
    alpha[z < 0.03] = 0
    ax.imshow(z, origin="lower", extent=extent, cmap=HOT, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear")
    if show_top and (cells["incoming_score"] > 0).any():
        pos = cells.loc[cells["incoming_score"] > 0].copy()
        cutoff = pos["incoming_score"].quantile(0.82)
        top = pos.loc[pos["incoming_score"] >= cutoff]
        ax.scatter(top["x"], top["y"], s=8.0, color=RED_DARK, alpha=0.96, linewidths=0)
    set_spatial_axis(ax, cells)
    ax.set_title(title, fontsize=7.2, color=INK, pad=2.5)


def plot_edge_hero(ax) -> None:
    rep, cells, _ = load_slice("DSS9")
    edges = edge_table("DSS9", LR_MAIN, "SpatialSelf").sort_values("score", ascending=False)
    top = edges.head(32)
    ax.scatter(cells["x"], cells["y"], s=4.2, color="#D8D5CE", alpha=0.42, linewidths=0, zorder=1)
    ax.scatter(top["sender_x"], top["sender_y"], s=16, color=ORANGE, alpha=0.92, linewidths=0, zorder=4)
    ax.scatter(top["receiver_x"], top["receiver_y"], s=18, color=RED_DARK, alpha=0.96, linewidths=0, zorder=5)
    for k, (_, row) in enumerate(top.iterrows()):
        rad = 0.12 if k % 2 == 0 else -0.12
        lw = 0.28 + 0.72 * ((row["score"] - top["score"].min()) / max(top["score"].max() - top["score"].min(), 1e-6))
        patch = FancyArrowPatch(
            (row["sender_x"], row["sender_y"]),
            (row["receiver_x"], row["receiver_y"]),
            arrowstyle="-|>",
            mutation_scale=5.0,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw,
            color=RED,
            alpha=0.42,
            zorder=3,
        )
        ax.add_patch(patch)
    set_spatial_axis(ax, cells)
    ax.set_title("D9 Cxcl10-Cxcr3", fontsize=7.4, color=INK, pad=2.5)


def zscore_rows(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return np.clip((arr - mu) / sd, -1.7, 1.7)


def plot_program_heatmap(ax) -> mpl.image.AxesImage:
    traj = pd.read_csv(TRAJ).set_index("lr_pair")
    rows = [x for x in LR_ROWS if x in traj.index]
    mat = pd.DataFrame(zscore_rows(traj.loc[rows, STAGES]), index=rows, columns=STAGE_LABELS)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=HEAT, vmin=-1.7, vmax=1.7)
    ax.set_xticks(np.arange(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.4)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def plot_trajectory(ax) -> None:
    long_df = pd.read_csv(LONG)
    traj = pd.read_csv(TRAJ)
    colors = {"Cxcl10-Cxcr3": RED, "Ccl5-Ccr5": ORANGE, "Ccl2-Ccr2": BROWN}
    xs = np.arange(len(STAGES))
    for lr in KEY_TRAJ:
        means = traj.loc[traj["lr_pair"] == lr, STAGES].iloc[0].to_numpy(float)
        rep = long_df.loc[long_df["lr_pair"] == lr].copy()
        color = colors[lr]
        ax.plot(xs, means, color=color, lw=1.25, marker="o", ms=2.5, label=lr, zorder=3)
        for i, st in enumerate(STAGES):
            vals = rep.loc[rep["stage"] == st, "score_q95"].to_numpy(float)
            if len(vals):
                ax.scatter(i + np.linspace(-0.025, 0.025, len(vals)), vals, s=9, facecolor=BG, edgecolor=color, lw=0.65, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.4)
    ax.set_ylim(0.72, 1.01)
    ax.set_ylabel("score", fontsize=6.4)
    ax.grid(axis="y", color=GRID, lw=0.48)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=5.2, handlelength=1.2, columnspacing=0.72)


def plot_delta_heatmap(ax) -> mpl.image.AxesImage:
    three = pd.read_csv(THREE)
    methods = ["SpatialSelf", "COMMOT", "CellNEST"]
    mat = np.full((len(KEY_TRAJ), len(methods)), np.nan)
    for i, lr in enumerate(KEY_TRAJ):
        for j, method in enumerate(methods):
            r = three.loc[(three["lr_pair"] == lr) & (three["method"] == method)]
            if len(r):
                mat[i, j] = float(r["dss9_minus_healthy"].iloc[0])
    vmax = max(float(np.nanmax(np.abs(mat))), 0.16)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=DIV, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, fontsize=6.2, rotation=0)
    ax.set_yticks(np.arange(len(KEY_TRAJ)))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def add_compartment_legend(ax) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COMPARTMENT[k], markeredgewidth=0, markersize=4.1, label=k)
        for k in ["Epithelial", "Immune", "Stromal", "Vascular", "Neural"]
    ]
    ax.legend(handles=handles, loc="center", ncol=3, fontsize=5.8, handletextpad=0.24, columnspacing=0.78)
    ax.set_axis_off()


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.35, 7.55))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.05, 1.48, 1.05], left=0.055, right=0.972, top=0.968, bottom=0.072, hspace=0.25)

    gs_a = outer[0].subgridspec(1, 4, wspace=0.035)
    gs_b = outer[1].subgridspec(1, 4, wspace=0.035)
    gs_cd = outer[2].subgridspec(1, 2, width_ratios=[1.12, 1.28], wspace=0.16)
    gs_d = gs_cd[1].subgridspec(1, 3, wspace=0.055)
    gs_efg = outer[3].subgridspec(1, 3, width_ratios=[1.05, 1.1, 0.76], wspace=0.16)

    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(4)]
    ax_c = fig.add_subplot(gs_cd[0, 0])
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(3)]
    ax_e = fig.add_subplot(gs_efg[0, 0])
    ax_f = fig.add_subplot(gs_efg[0, 1])
    ax_g = fig.add_subplot(gs_efg[0, 2])

    panel_label(axes_a[0], "a", x=-0.09)
    for ax, st, lab in zip(axes_a, STAGES, STAGE_LABELS):
        plot_compartment_map(ax, st, lab)

    panel_label(axes_b[0], "b", x=-0.09)
    for ax, st, lab in zip(axes_b, STAGES, STAGE_LABELS):
        cells = receiver_scores(st, LR_MAIN, "SpatialSelf")
        plot_density_map(ax, cells, lab)

    panel_label(ax_c, "c", x=-0.08)
    plot_edge_hero(ax_c)

    panel_label(axes_d[0], "d", x=-0.10)
    for ax, method in zip(axes_d, ["SpatialSelf", "COMMOT", "CellNEST"]):
        cells = receiver_scores("DSS9", LR_MAIN, method, rank_norm=True)
        plot_density_map(ax, cells, method, show_top=True)

    panel_label(ax_e, "e", x=-0.10)
    plot_program_heatmap(ax_e)

    panel_label(ax_f, "f", x=-0.10)
    plot_trajectory(ax_f)

    panel_label(ax_g, "g", x=-0.14)
    im_delta = plot_delta_heatmap(ax_g)
    cbar = fig.colorbar(im_delta, ax=ax_g, fraction=0.055, pad=0.045)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.ax.tick_params(labelsize=5.6, length=1.7, width=0.45)
    cbar.outline.set_visible(False)

    leg_ax = fig.add_axes([0.70, 0.565, 0.23, 0.045])
    add_compartment_legend(leg_ax)

    stem = OUT / "mouse_colon_merfish_dss_scipilot_redesign"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=650, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=650, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
