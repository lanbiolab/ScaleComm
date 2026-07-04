#!/usr/bin/env python
"""CellNEST-style DSS MERFISH biological story figure for SpatialSelf.

This figure is intentionally a biological discovery case study: it does not
include baseline comparison or downstream validation panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (
    ACTIVITY_CMAP,
    BASE,
    COMPARTMENT_COLORS,
    PROGRAM_CMAP,
    PROGRAMS,
    STAGE_LABELS,
    STAGES,
    broad_compartment,
    classify_program,
    load_slice,
    minmax,
    set_spatial_limits,
    setup_publication_style,
)


MODEL_NAME = "spatialself_wavelet_scale"
PRED_ROOT = BASE / "multislice_spatialself_scale_v21/predictions"
SUMMARY_ROOT = BASE / "multislice_spatialself_scale_v21/summary"
OUT = BASE / "figures/merfish_dss_cellnest_style_story_v21_scale_baseline"
PANEL_OUT = OUT / "panels"
TABLE_OUT = OUT / "source_tables"

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
FIG_BG = "#FFFFFF"
PANEL_FACE = "#FFFFFF"
TISSUE_BG = "#BDB8AE"
EDGE = "#237A78"
EDGE_LIGHT = "#B7C5BD"
D9 = "#C94F48"
D21 = "#237A78"
EARLY = "#D9992B"
OTHER = "#8A8177"
SOFT_NEUTRAL = "#BDB8AE"
SHOW_PANEL_TITLES = False
COMPARTMENT_ORDER = ["Epithelial", "Immune", "Mesenchymal", "Vascular", "Neural"]

PROGRAM_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "cellnest_program_warm",
    ["#FFF5E3", "#F1D28F", "#DD8D56", "#A94B43"],
)

PRISM_HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "prism_yellow_blue_activity",
    ["#E6C06A", "#F7F4EC", "#2F78A8"],
)

STAGE_ACCENTS = {
    "Healthy_D0": OTHER,
    "DSS3": EARLY,
    "DSS9": D9,
    "DSS21": D21,
}

HIGHLIGHT_LR = {
    "DSS9": {"Cxcl10-Cxcr3"},
    "DSS21": {"Tgfb3-Acvrl1"},
}

TRAJECTORY_LR = ["Rspo3-Lgr5", "Il1b-Il1r2", "Cxcl10-Cxcr3", "Tgfb3-Acvrl1"]
TRAJECTORY_COLORS = {
    "Rspo3-Lgr5": OTHER,
    "Il1b-Il1r2": EARLY,
    "Cxcl10-Cxcr3": D9,
    "Tgfb3-Acvrl1": D21,
}

TRAJECTORY_STYLE = {
    "Rspo3-Lgr5": {"alpha": 0.76, "lw": 1.12},
    "Il1b-Il1r2": {"alpha": 0.78, "lw": 1.14},
    "Cxcl10-Cxcr3": {"alpha": 1.0, "lw": 1.38},
    "Tgfb3-Acvrl1": {"alpha": 1.0, "lw": 1.38},
}


def prediction_path(stage: str, replicate_name: str, method: str = "SpatialSelf") -> Path:
    if method != "SpatialSelf":
        raise ValueError("v21 story figure only redraws SpatialSelf panels with the scale-aware baseline")
    return PRED_ROOT / stage / MODEL_NAME / f"{replicate_name}__{MODEL_NAME}_predictions.csv"


def setup_style() -> None:
    setup_publication_style()
    mpl.rcParams.update(
        {
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.55,
            "axes.facecolor": PANEL_FACE,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
        }
    )


def with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mpl.colors.to_rgba(color, alpha)


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.048, dy: float = 0.026) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=INK)


def panel_title(fig: plt.Figure, ax: plt.Axes, title: str, dy: float = 0.026, fontsize: float = 7.4) -> None:
    if not SHOW_PANEL_TITLES:
        return
    box = ax.get_position()
    fig.text(box.x0, box.y1 + dy, title, ha="left", va="bottom", fontsize=fontsize, color=INK)


def clean_quant_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_linewidth(0.52)
    ax.spines["bottom"].set_linewidth(0.52)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.34)


def read_predictions(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    rep, cells, cand = load_slice(stage, replicate)
    pred = pd.read_csv(prediction_path(stage, rep))
    tab = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    tab["lr_pair"] = tab["ligand"].astype(str) + "-" + tab["receptor"].astype(str)
    xy = cells.set_index("cell_id")[["x", "y", "cell_type"]]
    tab = tab.join(xy.rename(columns={"x": "sender_x", "y": "sender_y", "cell_type": "sender_type"}), on="sender_id")
    tab = tab.join(xy.rename(columns={"x": "receiver_x", "y": "receiver_y", "cell_type": "receiver_type"}), on="receiver_id")
    return rep, cells, tab.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y"])


def stage_top_edges(stage: str, replicate: int = 1, fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, cells, tab = read_predictions(stage, replicate)
    n = max(20, int(np.floor(fraction * len(tab))))
    return cells, tab.nlargest(n, "score").copy()


def draw_tissue(ax: plt.Axes, cells: pd.DataFrame, *, size: float = 1.25, alpha: float = 0.22) -> None:
    ax.set_facecolor(FIG_BG)
    ax.scatter(cells["x"], cells["y"], s=size, color=TISSUE_BG, alpha=alpha, linewidths=0, zorder=1)


def draw_cell_type_map(ax: plt.Axes, cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    cells["compartment"] = cells["cell_type"].map(broad_compartment)
    counts = (
        cells.groupby("compartment", as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
        .sort_values("n_cells", ascending=False)
    )
    counts["fraction"] = counts["n_cells"] / max(len(cells), 1)
    for comp in COMPARTMENT_ORDER + ["Other"]:
        sub = cells.loc[cells["compartment"] == comp]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=2.45,
            color=COMPARTMENT_COLORS.get(comp, SOFT_NEUTRAL),
            alpha=0.82 if comp != "Other" else 0.50,
            linewidths=0,
            zorder=2,
        )
    set_spatial_limits(ax, cells)
    return counts


def draw_top_edge_region(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame, color: str = EDGE, max_edges: int = 120) -> None:
    draw_tissue(ax, cells, size=1.20, alpha=0.30)
    top = edges.nlargest(min(max_edges, len(edges)), "score")
    if not top.empty:
        vmax = max(float(top["score"].max()), 1e-9)
        for row in top.itertuples(index=False):
            val = float(row.score) / vmax
            ax.plot(
                [row.sender_x, row.receiver_x],
                [row.sender_y, row.receiver_y],
                color=color,
                lw=0.12 + 0.30 * val,
                alpha=0.26 + 0.32 * val,
                zorder=2,
            )
        cells_in_edges = pd.unique(pd.concat([top["sender_id"], top["receiver_id"]], ignore_index=True))
        spot = cells.loc[cells["cell_id"].isin(cells_in_edges)]
        ax.scatter(spot["x"], spot["y"], s=5.3, color=color, alpha=0.90, edgecolors=PANEL_FACE, linewidths=0.10, zorder=3)
    set_spatial_limits(ax, cells)


def top_lr_table(stage: str, replicate: int = 1, n_lr: int = 6) -> pd.DataFrame:
    cells, top = stage_top_edges(stage, replicate, fraction=0.20)
    count = (
        top.groupby("lr_pair", as_index=False)
        .agg(edge_count=("score", "size"), score_sum=("score", "sum"), score_mean=("score", "mean"))
        .sort_values(["edge_count", "score_sum"], ascending=False)
        .head(n_lr)
    )
    count["stage"] = stage
    count["stage_label"] = STAGE_LABELS[STAGES.index(stage)]
    return count


def draw_lr_bars(ax: plt.Axes, df: pd.DataFrame, stage: str, xlabel: str = "abundance (n)", style: str = "stage") -> None:
    df = df.sort_values("edge_count", ascending=True)
    highlight = HIGHLIGHT_LR.get(stage, set())
    accent = STAGE_ACCENTS.get(stage, D21)
    if style == "local":
        colors = [with_alpha(accent, 0.98) if lr in highlight else with_alpha(SOFT_NEUTRAL, 0.92) for lr in df["lr_pair"]]
    else:
        colors = [with_alpha(accent, 0.98) if lr in highlight else with_alpha(accent, 0.76) for lr in df["lr_pair"]]
    y = np.arange(len(df))
    ax.barh(y, df["edge_count"], color=colors, height=0.58, edgecolor="none")
    xmax = float(df["edge_count"].max()) * 1.18 if len(df) else 1.0
    ax.set_xlim(0, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(df["lr_pair"])
    for label_obj, lr in zip(ax.get_yticklabels(), df["lr_pair"]):
        label_obj.set_color(TEXT)
        if lr in highlight:
            label_obj.set_color(accent)
            label_obj.set_fontweight("bold")
    for yi, (lr, value) in enumerate(zip(df["lr_pair"], df["edge_count"])):
        if lr in highlight:
            ax.scatter(
                float(value) + xmax * 0.040,
                yi,
                marker="v",
                s=20,
                facecolors=PANEL_FACE,
                edgecolors=accent,
                linewidths=0.78,
                zorder=5,
                clip_on=False,
            )
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.tick_params(axis="x", length=2.0, pad=1.0)
    ax.set_xlabel(xlabel, fontsize=5.45, labelpad=1.0, color=MUTED)
    clean_quant_axis(ax)
    ax.spines["left"].set_visible(False)


def draw_low_high_bar(ax: plt.Axes, cmap, label: str, where: str = "bottom") -> None:
    if where == "right":
        cax = ax.inset_axes([1.045, 0.18, 0.055, 0.58])
        orientation = "vertical"
    else:
        cax = ax.inset_axes([0.56, -0.255, 0.34, 0.052])
        orientation = "horizontal"
    for i in range(96):
        if orientation == "horizontal":
            cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=cmap(i / 95), edgecolor="none"))
        else:
            cax.add_patch(Rectangle((0, i / 96), 1, 1 / 96, facecolor=cmap(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    if orientation == "horizontal":
        cax.set_xticks([0, 1])
        cax.set_xticklabels(["low", "high"])
        cax.set_yticks([])
        cax.set_xlabel(label, fontsize=5.4, labelpad=0.8, color=MUTED)
    else:
        cax.set_xticks([])
        cax.set_yticks([0, 1])
        cax.set_yticklabels(["low", "high"])
        cax.yaxis.tick_right()
        cax.set_ylabel(label, fontsize=5.25, labelpad=2.0, color=MUTED, rotation=270, va="bottom")
    cax.tick_params(length=0, pad=0.8, labelsize=5.3)
    for spine in cax.spines.values():
        spine.set_visible(False)


def draw_matrix_rects(ax: plt.Axes, matrix: np.ndarray, cmap, vmin: float = 0.0, vmax: float = 1.0) -> None:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    nrow, ncol = matrix.shape
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(float(matrix[i, j]))),
                    edgecolor="#FFFDF8",
                    linewidth=0.42,
                )
            )
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            ncol,
            nrow,
            fill=False,
            edgecolor="#E7DCCE",
            linewidth=0.46,
            zorder=5,
        )
    )


def draw_prism_matrix_rects(ax: plt.Axes, matrix: np.ndarray, cmap, vmin: float = 0.0, vmax: float = 1.0) -> None:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    nrow, ncol = matrix.shape
    ax.set_facecolor("#FFFFFF")
    for i in range(nrow):
        for j in range(ncol):
            value = float(matrix[i, j])
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(value)),
                    edgecolor="#1E1E1E",
                    linewidth=0.34,
                )
            )
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "#FFFFFF" if luminance < 0.42 else "#171717"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=3.35, color=text_color, zorder=6)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            ncol,
            nrow,
            fill=False,
            edgecolor="#111111",
            linewidth=0.72,
            zorder=5,
        )
    )


def draw_prism_colorbar(ax: plt.Axes, cmap, label: str) -> None:
    cax = ax.inset_axes([1.065, 0.06, 0.050, 0.88])
    for i in range(160):
        v0 = i / 160
        cax.add_patch(Rectangle((0, v0), 1, 1 / 160, facecolor=cmap(v0), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([0, 0.5, 1])
    cax.set_yticklabels(["0", "0.5", "1"])
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", length=2.0, width=0.52, pad=1.2, labelsize=5.1, colors="#111111")
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.56)


def program_activity_table() -> pd.DataFrame:
    traj = pd.read_csv(SUMMARY_ROOT / "stage_lr_trajectories.csv")
    traj["program"] = traj.apply(classify_program, axis=1)
    rows = []
    for program in PROGRAMS:
        sub = traj.loc[traj["program"] == program]
        for stage, label in zip(STAGES, STAGE_LABELS):
            rows.append(
                {
                    "program": program,
                    "stage": stage,
                    "stage_label": label,
                    "activity": float(sub[stage].mean(skipna=True)),
                    "n_lr": int(len(sub)),
                }
            )
    out = pd.DataFrame(rows)
    out["relative_activity"] = out.groupby("program")["activity"].transform(lambda x: minmax(x.to_numpy()))
    out.to_csv(TABLE_OUT / "panel_d_program_activity.csv", index=False)
    return out


def trajectory_table() -> pd.DataFrame:
    p = SUMMARY_ROOT / "stage_lr_trajectories.csv"
    df = pd.read_csv(p)
    return df.loc[df["lr_pair"].isin(TRAJECTORY_LR)].copy()


def local_bbox_from_lr(stage: str, lr_pair: str, replicate: int = 1, half_width: float = 520.0) -> tuple[float, float, float, float]:
    _, cells, tab = read_predictions(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    sub = tab.loc[(tab["ligand"] == ligand) & (tab["receptor"] == receptor)].nlargest(80, "score").copy()
    if sub.empty:
        x = cells["x"].median()
        y = cells["y"].median()
    else:
        mid_x = (sub["sender_x"].to_numpy(float) + sub["receiver_x"].to_numpy(float)) / 2
        mid_y = (sub["sender_y"].to_numpy(float) + sub["receiver_y"].to_numpy(float)) / 2
        score = sub["score"].to_numpy(float)
        coords = np.column_stack([mid_x, mid_y])
        dist2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        radius2 = (half_width * 0.58) ** 2
        density = (dist2 <= radius2).astype(float) @ np.maximum(score, 1e-6)
        ix = int(np.argmax(density))
        near = dist2[ix] <= radius2
        weights = np.maximum(score[near], 1e-6)
        x = float(np.average(mid_x[near], weights=weights))
        y = float(np.average(mid_y[near], weights=weights))
    bbox = (x - half_width, x + half_width, y - half_width, y + half_width)
    return clip_bbox_to_cells(bbox, cells)


def clip_bbox_to_cells(bbox: tuple[float, float, float, float], cells: pd.DataFrame) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = bbox
    xmin, xmax = float(cells["x"].min()), float(cells["x"].max())
    ymin, ymax = float(cells["y"].min()), float(cells["y"].max())
    width = x1 - x0
    height = y1 - y0
    if x0 < xmin:
        x1 += xmin - x0
        x0 = xmin
    if x1 > xmax:
        x0 -= x1 - xmax
        x1 = xmax
    if y0 < ymin:
        y1 += ymin - y0
        y0 = ymin
    if y1 > ymax:
        y0 -= y1 - ymax
        y1 = ymax
    x0 = max(x0, xmin)
    x1 = min(x1, xmax)
    y0 = max(y0, ymin)
    y1 = min(y1, ymax)
    if (x1 - x0) < 0.45 * width:
        cx = float(np.clip((x0 + x1) / 2, xmin + width / 2, xmax - width / 2))
        x0, x1 = cx - width / 2, cx + width / 2
    if (y1 - y0) < 0.45 * height:
        cy = float(np.clip((y0 + y1) / 2, ymin + height / 2, ymax - height / 2))
        y0, y1 = cy - height / 2, cy + height / 2
    return x0, x1, y0, y1


def local_edges(stage: str, bbox: tuple[float, float, float, float], replicate: int = 1, fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, cells, tab = read_predictions(stage, replicate)
    top_n = max(20, int(np.floor(fraction * len(tab))))
    top = tab.nlargest(top_n, "score").copy()
    x0, x1, y0, y1 = bbox
    mask_sender = top["sender_x"].between(x0, x1) & top["sender_y"].between(y0, y1)
    mask_receiver = top["receiver_x"].between(x0, x1) & top["receiver_y"].between(y0, y1)
    local = top.loc[mask_sender | mask_receiver].copy()
    return cells, local


def draw_zoom_panel(
    ax_context: plt.Axes,
    ax_zoom: plt.Axes,
    stage: str,
    lr_pair: str,
    bbox: tuple[float, float, float, float],
    color: str,
    title: str,
) -> pd.DataFrame:
    cells, edges = local_edges(stage, bbox)
    _, all_cells, tab = read_predictions(stage, 1)
    ligand, receptor = lr_pair.split("-")
    focal = tab.loc[(tab["ligand"] == ligand) & (tab["receptor"] == receptor)].nlargest(36, "score").copy()
    x0, x1, y0, y1 = bbox

    draw_tissue(ax_context, cells, size=1.15, alpha=0.30)
    ax_context.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=0.86))
    set_spatial_limits(ax_context, cells)

    zoom_cells = cells.loc[cells["x"].between(x0, x1) & cells["y"].between(y0, y1)].copy()
    ax_zoom.set_facecolor(FIG_BG)
    ax_zoom.scatter(zoom_cells["x"], zoom_cells["y"], s=9.2, color=TISSUE_BG, alpha=0.30, linewidths=0, zorder=1)
    if not edges.empty:
        top = edges.nlargest(min(len(edges), 90), "score")
        vmax = max(float(top["score"].max()), 1e-9)
        for row in top.itertuples(index=False):
            val = float(row.score) / vmax
            ax_zoom.plot(
                [row.sender_x, row.receiver_x],
                [row.sender_y, row.receiver_y],
                color=EDGE_LIGHT,
                lw=0.16 + 0.28 * val,
                alpha=0.18 + 0.24 * val,
                zorder=2,
            )
    focal = focal.loc[
        (focal["sender_x"].between(x0, x1) & focal["sender_y"].between(y0, y1))
        | (focal["receiver_x"].between(x0, x1) & focal["receiver_y"].between(y0, y1))
    ]
    if not focal.empty:
        context_ids = pd.unique(pd.concat([focal["sender_id"], focal["receiver_id"]], ignore_index=True))
        context_spots = cells.loc[cells["cell_id"].isin(context_ids)]
        ax_context.scatter(
            context_spots["x"],
            context_spots["y"],
            s=5.2,
            color=color,
            edgecolors=PANEL_FACE,
            linewidths=0.08,
            alpha=0.88,
            zorder=4,
        )
        vmax = max(float(focal["score"].max()), 1e-9)
        for row in focal.itertuples(index=False):
            val = float(row.score) / vmax
            ax_zoom.plot(
                [row.sender_x, row.receiver_x],
                [row.sender_y, row.receiver_y],
                color=color,
                lw=0.52 + 0.58 * val,
                alpha=0.46 + 0.20 * val,
                zorder=4,
            )
        ids = pd.unique(pd.concat([focal["sender_id"], focal["receiver_id"]], ignore_index=True))
        spots = cells.loc[cells["cell_id"].isin(ids)]
        ax_zoom.scatter(spots["x"], spots["y"], s=24, color=color, edgecolors=PANEL_FACE, linewidths=0.20, alpha=0.96, zorder=5)
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y0, y1)
    ax_zoom.set_aspect("equal")
    for ax in [ax_context, ax_zoom]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return edges


def local_lr_count(edges: pd.DataFrame, stage: str, n_lr: int = 7) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["stage", "lr_pair", "edge_count", "score_sum", "score_mean"])
    out = (
        edges.groupby("lr_pair", as_index=False)
        .agg(edge_count=("score", "size"), score_sum=("score", "sum"), score_mean=("score", "mean"))
        .sort_values(["edge_count", "score_sum"], ascending=False)
        .head(n_lr)
    )
    out["stage"] = stage
    return out


def replicate_consistency_table(n_lr: int = 30) -> pd.DataFrame:
    rows = []
    for stage, label in zip(STAGES, STAGE_LABELS):
        sets = []
        for rep in [1, 2]:
            try:
                tbl = top_lr_table(stage, rep, n_lr=n_lr)
                sets.append(set(tbl["lr_pair"]))
            except Exception:
                sets.append(set())
        inter = len(sets[0] & sets[1])
        union = len(sets[0] | sets[1])
        rows.append({"stage": stage, "stage_label": label, "top_lr_jaccard": inter / union if union else np.nan, "overlap": inter, "union": union})
    return pd.DataFrame(rows)


def panel_a_cell_type_maps(fig: plt.Figure, gs, label: str = "a") -> list[plt.Axes]:
    axes = []
    composition_rows = []
    for i, (stage, stage_label) in enumerate(zip(STAGES, STAGE_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        _, cells, _ = load_slice(stage, 1)
        counts = draw_cell_type_map(ax, cells)
        counts["stage"] = stage
        counts["stage_label"] = stage_label
        composition_rows.append(counts)
        ax.set_title(stage_label, fontsize=6.9, color=TEXT, pad=1.0)
        axes.append(ax)
    pd.concat(composition_rows, ignore_index=True).to_csv(TABLE_OUT / "panel_a_cell_type_composition.csv", index=False)
    panel_label(fig, axes[0], label)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.1, markerfacecolor=COMPARTMENT_COLORS[comp], markeredgewidth=0, label=comp)
        for comp in COMPARTMENT_ORDER
    ]
    boxes = [ax.get_position() for ax in axes]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    bottom = min(box.y0 for box in boxes)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=((left + right) / 2, bottom - 0.010),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.22,
        columnspacing=0.64,
        borderaxespad=0.0,
        fontsize=5.25,
    )
    return axes


def panel_b_communication_regions(fig: plt.Figure, gs, label: str = "b") -> list[plt.Axes]:
    axes = []
    source_rows = []
    for i, (stage, stage_label) in enumerate(zip(STAGES, STAGE_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        cells, top = stage_top_edges(stage, 1, fraction=0.20)
        draw_top_edge_region(ax, cells, top, color=STAGE_ACCENTS[stage], max_edges=105)
        ax.set_title(stage_label, fontsize=6.9, color=TEXT, pad=1.0)
        axes.append(ax)
        source_rows.append({"stage": stage, "stage_label": stage_label, "n_top_edges": len(top), "top_score_cutoff": float(top["score"].min())})
    pd.DataFrame(source_rows).to_csv(TABLE_OUT / "panel_b_communication_region_summary.csv", index=False)
    panel_label(fig, axes[0], label)
    return axes


def panel_b_top_lr_histograms(fig: plt.Figure, gs, label: str = "c") -> list[plt.Axes]:
    axes = []
    all_rows = []
    for i, (stage, stage_label) in enumerate(zip(STAGES, STAGE_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        df = top_lr_table(stage, 1, n_lr=6)
        all_rows.append(df)
        draw_lr_bars(ax, df, stage)
        ax.set_title(stage_label, fontsize=6.9, color=TEXT, pad=1.7)
        if i > 0:
            ax.set_ylabel("")
        axes.append(ax)
    pd.concat(all_rows, ignore_index=True).to_csv(TABLE_OUT / f"panel_{label}_top_lr_histograms.csv", index=False)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "Dominant LR pairs within high-confidence edges")
    return axes


def panel_c_program_heatmap(fig: plt.Figure, ax: plt.Axes, label: str = "d", *, compact: bool = False) -> plt.Axes:
    activity = program_activity_table()
    matrix = (
        activity.pivot(index="program", columns="stage_label", values="relative_activity")
        .loc[PROGRAMS, STAGE_LABELS]
        .to_numpy(float)
    )
    draw_prism_matrix_rects(ax, matrix, PRISM_HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_yticks(range(len(PROGRAMS)))
    ax.set_yticklabels(["homeostasis", "early\nresponse", "inflammatory\npeak", "repair /\nremodeling"])
    ax.tick_params(length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(fig, ax, label, dx=-0.060)
    panel_title(fig, ax, "Communication programs")
    draw_prism_colorbar(ax, PRISM_HEATMAP_CMAP, "")
    return ax


def panel_d_lr_trajectories(fig: plt.Figure, ax: plt.Axes, label: str = "e") -> plt.Axes:
    df = trajectory_table()
    stage_order = {s: i for i, s in enumerate(STAGES)}
    x = np.array([0, 3, 9, 21], dtype=float)
    ax.axvspan(9, 21, color="#D9D9D9", alpha=0.78, zorder=0)
    ax.axvline(9, color="#111111", lw=0.78, ls=(0, (2.2, 2.2)), zorder=1)
    handles = []
    for lr in TRAJECTORY_LR:
        rec = df.loc[df["lr_pair"] == lr]
        if rec.empty:
            continue
        if set(STAGES).issubset(rec.columns):
            values = rec.iloc[0][STAGES].to_numpy(float)
            std_cols = [f"{s}_std" for s in STAGES]
            if set(std_cols).issubset(rec.columns):
                raw_err = rec.iloc[0][std_cols].to_numpy(float)
            else:
                raw_err = np.zeros_like(values)
        else:
            values = rec.sort_values("stage", key=lambda x: x.map(stage_order))["score"].to_numpy(float)
            raw_err = np.zeros_like(values)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        rel = (values - lo) / (hi - lo + 1e-9)
        err = np.nan_to_num(raw_err / (hi - lo + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        err = np.minimum(err, 0.12)
        style = TRAJECTORY_STYLE.get(lr, {"alpha": 0.86, "lw": 1.20})
        ax.errorbar(
            x,
            rel,
            yerr=err,
            fmt="-o",
            lw=style["lw"],
            ms=3.55,
            color=TRAJECTORY_COLORS[lr],
            alpha=style["alpha"],
            ecolor="#111111",
            elinewidth=0.50,
            capsize=1.8,
            capthick=0.50,
            markerfacecolor=TRAJECTORY_COLORS[lr],
            markeredgecolor="#111111",
            markeredgewidth=0.42,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
        handles.append(Patch(facecolor=TRAJECTORY_COLORS[lr], edgecolor="#111111", linewidth=0.55, label=lr))
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        handlelength=1.05,
        handleheight=0.70,
        columnspacing=0.86,
        handletextpad=0.30,
        borderaxespad=0.0,
        fontsize=4.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["D0", "D3", "D9", "D21"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-0.35, 21.75)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlabel("DSS day", labelpad=1.0)
    ax.set_ylabel("per-LR normalized activity")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(0.72)
    ax.spines["bottom"].set_linewidth(0.72)
    ax.tick_params(axis="both", colors="#111111", width=0.70, length=3.0, pad=1.2)
    panel_label(fig, ax, label)
    return ax


def panel_niche(fig: plt.Figure, gs, *, stage: str, lr_pair: str, color: str, title: str, label: str, half_width: float) -> tuple[list[plt.Axes], pd.DataFrame]:
    ax_context = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    bbox = local_bbox_from_lr(stage, lr_pair, 1, half_width=half_width)
    edges = draw_zoom_panel(ax_context, ax_zoom, stage, lr_pair, bbox, color, title)
    panel_label(fig, ax_context, label)
    panel_title(fig, ax_context, f"{STAGE_LABELS[STAGES.index(stage)]} {lr_pair} communication niche")
    return [ax_context, ax_zoom], edges


def panel_local_lr_hist(fig: plt.Figure, ax: plt.Axes, edges: pd.DataFrame, stage: str, label: str) -> plt.Axes:
    df = local_lr_count(edges, stage, n_lr=7)
    if df.empty:
        ax.text(0.5, 0.5, "no local high-confidence edges", ha="center", va="center", color=MUTED, fontsize=6)
        ax.axis("off")
        return ax
    df.to_csv(TABLE_OUT / f"panel_{label}_local_lr_{stage}.csv", index=False)
    draw_lr_bars(ax, df, stage, xlabel="local abundance (n)", style="local")
    panel_label(fig, ax, label)
    panel_title(fig, ax, "Local LR spectrum")
    return ax


def panel_h_replicate_consistency(fig: plt.Figure, ax: plt.Axes, label: str = "j") -> plt.Axes:
    df = replicate_consistency_table(n_lr=30)
    df.to_csv(TABLE_OUT / f"panel_{label}_replicate_consistency.csv", index=False)
    y = np.arange(len(df))
    colors = [with_alpha(STAGE_ACCENTS[s], 0.76) for s in df["stage"]]
    ax.barh(y, df["top_lr_jaccard"], color=colors, height=0.50, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["stage_label"])
    for label_obj, stage in zip(ax.get_yticklabels(), df["stage"]):
        label_obj.set_color(TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.60)
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_xlabel("top LR overlap Jaccard", fontsize=5.45, labelpad=1.0, color=MUTED)
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.tick_params(axis="x", length=2.0, pad=1.0)
    clean_quant_axis(ax)
    ax.spines["left"].set_visible(False)
    panel_label(fig, ax, label)
    panel_title(fig, ax, "Replicate consistency")
    return ax


def render_single_panels() -> None:
    fig = plt.figure(figsize=(7.10, 1.72))
    gs = fig.add_gridspec(1, 4, left=0.045, right=0.985, bottom=0.265, top=0.815, wspace=0.045)
    panel_a_cell_type_maps(fig, gs)
    export(fig, PANEL_OUT / "panel_a_cell_type_maps")

    fig = plt.figure(figsize=(7.10, 1.56))
    gs = fig.add_gridspec(1, 4, left=0.045, right=0.985, bottom=0.065, top=0.800, wspace=0.045)
    panel_b_communication_regions(fig, gs)
    export(fig, PANEL_OUT / "panel_b_communication_regions")

    fig = plt.figure(figsize=(7.18, 2.28))
    gs = fig.add_gridspec(1, 4, left=0.145, right=0.985, bottom=0.210, top=0.710, wspace=0.780)
    panel_b_top_lr_histograms(fig, gs, "c")
    export(fig, PANEL_OUT / "panel_c_top_lr_histograms")

    fig, ax = plt.subplots(figsize=(2.82, 2.05), constrained_layout=False)
    fig.subplots_adjust(left=0.290, right=0.855, bottom=0.260, top=0.775)
    panel_c_program_heatmap(fig, ax, "d")
    export(fig, PANEL_OUT / "panel_d_program_heatmap")

    fig, ax = plt.subplots(figsize=(3.45, 2.05), constrained_layout=False)
    fig.subplots_adjust(left=0.160, right=0.760, bottom=0.175, top=0.775)
    panel_d_lr_trajectories(fig, ax, "e")
    export(fig, PANEL_OUT / "panel_e_lr_trajectories")

    fig = plt.figure(figsize=(4.75, 2.30))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], left=0.050, right=0.985, bottom=0.070, top=0.785, wspace=0.070)
    _, d9_edges = panel_niche(fig, gs, stage="DSS9", lr_pair="Cxcl10-Cxcr3", color=D9, title="Cxcl10-Cxcr3", label="f", half_width=560)
    export(fig, PANEL_OUT / "panel_f_d9_inflammatory_niche")

    fig, ax = plt.subplots(figsize=(2.55, 2.30), constrained_layout=False)
    fig.subplots_adjust(left=0.390, right=0.965, bottom=0.205, top=0.775)
    panel_local_lr_hist(fig, ax, d9_edges, "DSS9", "g")
    export(fig, PANEL_OUT / "panel_g_d9_local_lr_histogram")

    fig = plt.figure(figsize=(4.75, 2.30))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], left=0.050, right=0.985, bottom=0.070, top=0.785, wspace=0.070)
    _, d21_edges = panel_niche(fig, gs, stage="DSS21", lr_pair="Tgfb3-Acvrl1", color=D21, title="Tgfb3-Acvrl1", label="h", half_width=620)
    export(fig, PANEL_OUT / "panel_h_d21_repair_niche")

    fig, ax = plt.subplots(figsize=(2.55, 2.30), constrained_layout=False)
    fig.subplots_adjust(left=0.390, right=0.965, bottom=0.205, top=0.775)
    panel_local_lr_hist(fig, ax, d21_edges, "DSS21", "i")
    export(fig, PANEL_OUT / "panel_i_d21_local_lr_histogram")

    fig, ax = plt.subplots(figsize=(2.35, 2.16), constrained_layout=False)
    fig.subplots_adjust(left=0.315, right=0.955, bottom=0.195, top=0.770)
    panel_h_replicate_consistency(fig, ax, "j")
    export(fig, PANEL_OUT / "panel_j_replicate_consistency")


def render_composite() -> None:
    fig = plt.figure(figsize=(7.20, 9.18))
    outer = fig.add_gridspec(
        6,
        1,
        height_ratios=[1.05, 0.96, 1.12, 0.98, 1.15, 1.15],
        left=0.055,
        right=0.970,
        bottom=0.045,
        top=0.960,
        hspace=0.520,
    )
    panel_a_cell_type_maps(fig, outer[0, 0].subgridspec(1, 4, wspace=0.045))
    panel_b_communication_regions(fig, outer[1, 0].subgridspec(1, 4, wspace=0.045))
    panel_b_top_lr_histograms(fig, outer[2, 0].subgridspec(1, 4, wspace=0.700), "c")
    mid = outer[3, 0].subgridspec(1, 3, width_ratios=[1.03, 1.58, 0.86], wspace=0.500)
    panel_c_program_heatmap(fig, fig.add_subplot(mid[0, 0]), "d", compact=True)
    panel_d_lr_trajectories(fig, fig.add_subplot(mid[0, 1]), "e")
    panel_h_replicate_consistency(fig, fig.add_subplot(mid[0, 2]), "j")
    row5 = outer[4, 0].subgridspec(1, 3, width_ratios=[1.58, 0.88, 0.02], wspace=0.315)
    _, d9_edges = panel_niche(fig, row5[0, 0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.065), stage="DSS9", lr_pair="Cxcl10-Cxcr3", color=D9, title="Cxcl10-Cxcr3", label="f", half_width=560)
    panel_local_lr_hist(fig, fig.add_subplot(row5[0, 1]), d9_edges, "DSS9", "g")
    row6 = outer[5, 0].subgridspec(1, 3, width_ratios=[1.58, 0.88, 0.02], wspace=0.315)
    _, d21_edges = panel_niche(fig, row6[0, 0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.065), stage="DSS21", lr_pair="Tgfb3-Acvrl1", color=D21, title="Tgfb3-Acvrl1", label="h", half_width=620)
    panel_local_lr_hist(fig, fig.add_subplot(row6[0, 1]), d21_edges, "DSS21", "i")
    export(fig, OUT / "mouse_colon_merfish_cellnest_style_story_v21_scale_baseline")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    render_single_panels()
    render_composite()
    svg = OUT / "mouse_colon_merfish_cellnest_style_story_v21_scale_baseline.svg"
    print(f"{OUT} image_tags={svg.read_text(errors='ignore').count('<image')}")


if __name__ == "__main__":
    main()
