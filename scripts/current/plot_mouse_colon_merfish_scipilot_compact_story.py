#!/usr/bin/env python
"""SciPilot compact story figure for DSS colitis MERFISH communication rewiring."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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
THREE = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "figures/merfish_dss_scipilot_compact_story"
TABLE_OUT = BASE / "progression_story"

STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
STAGE_PROGRAMS = ["homeostasis", "early", "inflammation", "repair"]
LR_CASE = "Cxcl10-Cxcr3"
PROGRAMS = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]

INK = "#202020"
MUTED = "#827D75"
BG = "#FFFFFF"
TISSUE = "#D9D5CC"
GRID = "#E9E4DA"
GREEN = "#5B8A72"
GOLD = "#D59A3B"
RED = "#B54034"
BLUE = "#5578A6"
PROGRAM_COLORS = {
    "homeostasis": GREEN,
    "early response": GOLD,
    "inflammatory peak": RED,
    "repair/remodeling": BLUE,
}
FLOW_COLORS = {
    "Epithelial -> Epithelial": "#D8903D",
    "Stromal -> Epithelial": "#66A391",
    "Immune -> Epithelial": "#B84D73",
    "Immune -> Stromal": "#8E5C90",
    "Stromal -> Immune": "#5B7FA8",
    "Other": "#C9C3B8",
}
PROGRAM_CMAP = LinearSegmentedColormap.from_list("program", ["#F6F1E8", "#E7BF8F", "#BD6246", "#762B25"])
HOT_CMAP = LinearSegmentedColormap.from_list("hot", ["#F8F4EC", "#E9BE91", "#C95E42", "#6D1D1B"])
DELTA_CMAP = LinearSegmentedColormap.from_list("delta", ["#4E6D8A", "#F6F1EA", "#A94439"])


def style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 6.8,
            "axes.linewidth": 0.62,
            "axes.edgecolor": INK,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="bottom", color=INK)


def classify_program(row: pd.Series) -> str:
    if row["trajectory_class"] == "flat_or_decreased" and row["peak_stage"] == "Healthy_D0":
        return "homeostasis"
    if row["trajectory_class"] == "early_induction" or row["peak_stage"] == "DSS3":
        return "early response"
    if row["peak_stage"] == "DSS9" and row["biological_category"] in {"Chemokine", "IL", "TNF"}:
        return "inflammatory peak"
    if row["trajectory_class"] == "late_repair_or_persistent" or row["peak_stage"] == "DSS21":
        return "repair/remodeling"
    return "other"


def broad_compartment(cell_type: str) -> str:
    s = str(cell_type).lower()
    if any(k in s for k in ["colonocyte", "epithelial", "goblet", "stem", "tuft", "iae", "ta"]):
        return "Epithelial"
    if any(k in s for k in ["macrophage", "neutrophil", "t cell", "b cell", "dc", "treg", "mast", "plasma", "lymph", "nk", "tfh"]):
        return "Immune"
    if any(k in s for k in ["fibro", "iaf", "frc", "smc", "pericyte", "icc"]):
        return "Stromal"
    if any(k in s for k in ["ec", "endothelial", "arterial", "venous", "capillarial", "lymphatic"]):
        return "Vascular"
    return "Other"


def rep_dir(stage: str, replicate: int = 1) -> Path:
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


def load_slice(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    d = rep_dir(stage, replicate)
    return d.name, pd.read_csv(d / "cells.csv"), pd.read_csv(d / "candidate_interactions.csv")


def edge_table(stage: str, lr_pair: str, method: str = "SpatialSelf", replicate: int = 1) -> pd.DataFrame:
    rep, cells, cand = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    sub = cand.loc[(cand["ligand"] == ligand) & (cand["receptor"] == receptor)].copy()
    pred = pd.read_csv(pred_path(stage, rep, method))
    sub = sub.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    if sub.empty:
        return sub
    recv = cells[["cell_id", "x", "y"]].rename(columns={"cell_id": "receiver_id", "x": "receiver_x", "y": "receiver_y"})
    return sub.merge(recv, on="receiver_id", how="left")


def receiver_scores(stage: str, lr_pair: str, method: str = "SpatialSelf", rank_norm: bool = False) -> pd.DataFrame:
    _, cells, _ = load_slice(stage)
    edges = edge_table(stage, lr_pair, method)
    cells = cells.copy()
    if edges.empty:
        cells["incoming_score"] = 0.0
        return cells
    inc = edges.groupby("receiver_id", as_index=False)["score"].max().rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
    cells = cells.merge(inc, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    if rank_norm and (cells["incoming_score"] > 0).any():
        pos = cells["incoming_score"] > 0
        cells.loc[pos, "incoming_score"] = cells.loc[pos, "incoming_score"].rank(pct=True)
    return cells


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi == lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def build_program_activity() -> pd.DataFrame:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(TRAJ)
    traj["program"] = traj.apply(classify_program, axis=1)
    rows = []
    for program in PROGRAMS:
        sub = traj.loc[traj["program"] == program]
        for stage, stage_label in zip(STAGES, STAGE_LABELS):
            rows.append(
                {
                    "program": program,
                    "stage": stage,
                    "stage_label": stage_label,
                    "activity": float(sub[stage].mean(skipna=True)) if len(sub) else np.nan,
                    "n_lr": int(len(sub)),
                }
            )
    out = pd.DataFrame(rows)
    out["activity_scaled"] = out.groupby("program")["activity"].transform(lambda x: minmax(x.to_numpy()))
    out.to_csv(TABLE_OUT / "scipilot_compact_program_activity.csv", index=False)
    traj.to_csv(TABLE_OUT / "scipilot_compact_lr_program_assignments.csv", index=False)
    return out


def build_flow_composition() -> pd.DataFrame:
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    rows = []
    for _, meta in manifest.iterrows():
        stage = meta["stage"]
        rep_name = Path(meta["replicate_dir"]).name
        d = ROOT / meta["replicate_dir"]
        cand = pd.read_csv(d / "candidate_interactions.csv")
        pred = pd.read_csv(pred_path(stage, rep_name, "SpatialSelf"))
        merged = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        cutoff = merged["score"].quantile(0.95)
        high = merged.loc[merged["score"] >= cutoff].copy()
        high["sender_compartment"] = high["sender_cell_type"].map(broad_compartment)
        high["receiver_compartment"] = high["receiver_cell_type"].map(broad_compartment)
        high["flow"] = high["sender_compartment"] + " -> " + high["receiver_compartment"]
        grouped = high.groupby("flow", as_index=False)["score"].sum()
        grouped["stage"] = stage
        grouped["stage_label"] = STAGE_LABELS[STAGES.index(stage)]
        rows.append(grouped)
    flow = pd.concat(rows, ignore_index=True)
    summary = flow.groupby(["stage", "stage_label", "flow"], as_index=False)["score"].sum()
    top_flows = summary.groupby("flow")["score"].sum().sort_values(ascending=False).head(5).index.tolist()
    summary["flow_plot"] = np.where(summary["flow"].isin(top_flows), summary["flow"], "Other")
    summary = summary.groupby(["stage", "stage_label", "flow_plot"], as_index=False)["score"].sum()
    summary["fraction"] = summary["score"] / summary.groupby("stage")["score"].transform("sum")
    summary.to_csv(TABLE_OUT / "scipilot_compact_flow_composition.csv", index=False)
    return summary


def set_spatial_axis(ax, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    dx = (x.max() - x.min()) * 0.045
    dy = (y.max() - y.min()) * 0.045
    ax.set_xlim(x.min() - dx, x.max() + dx)
    ax.set_ylim(y.min() - dy, y.max() + dy)


def density_grid(cells: pd.DataFrame, bins: int = 142, sigma: float = 3.0) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    w = cells["incoming_score"].to_numpy(float)
    dx = (x.max() - x.min()) * 0.045
    dy = (y.max() - y.min()) * 0.045
    extent = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]], weights=w)
    z = gaussian_filter(h.T, sigma=sigma)
    return z, extent


def plot_stage_strip(ax) -> None:
    ax.set_axis_off()
    xs = np.linspace(0.11, 0.89, 4)
    ax.plot([xs[0], xs[-1]], [0.55, 0.55], color=INK, lw=0.8)
    colors = [GREEN, GOLD, RED, BLUE]
    for x, stage, program, color in zip(xs, STAGE_LABELS, STAGE_PROGRAMS, colors):
        ax.scatter([x], [0.55], s=92, color=color, edgecolor=INK, lw=0.55, zorder=3)
        ax.text(x, 0.76, stage, ha="center", va="center", fontsize=7.8, color=INK)
        ax.text(x, 0.32, program, ha="center", va="center", fontsize=6.2, color=INK)
    ax.annotate("", xy=(xs[2] - 0.02, 0.55), xytext=(xs[1] + 0.02, 0.55), arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_program_heatmap(ax, activity: pd.DataFrame) -> None:
    panel_label(ax, "a")
    matrix = activity.pivot(index="program", columns="stage_label", values="activity_scaled").loc[PROGRAMS, STAGE_LABELS]
    ax.imshow(matrix.to_numpy(), cmap=PROGRAM_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.8)
    ax.set_yticks(range(len(PROGRAMS)))
    ax.set_yticklabels(["homeostasis", "early response", "inflammatory peak", "repair/remodeling"], fontsize=6.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("stage-specific communication programs", loc="left", fontsize=7.4, pad=5)


def plot_flow_bars(ax, flow: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.06)
    flow_order = [f for f in FLOW_COLORS if f in set(flow["flow_plot"])]
    y = np.arange(len(STAGE_LABELS))
    left = np.zeros(len(STAGE_LABELS))
    for fl in flow_order:
        vals = []
        for stage in STAGES:
            rec = flow.loc[(flow["stage"] == stage) & (flow["flow_plot"] == fl), "fraction"]
            vals.append(float(rec.iloc[0]) if len(rec) else 0.0)
        ax.barh(y, vals, left=left, color=FLOW_COLORS[fl], edgecolor="white", lw=0.45, height=0.58, label=fl)
        left += np.array(vals)
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS, fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of high-scoring edges", fontsize=6.4)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=5.6, handlelength=1.1, columnspacing=0.7)
    ax.set_title("sender-receiver flow composition", loc="left", fontsize=7.4, pad=5)


def plot_spatial_series(axes: list[plt.Axes]) -> None:
    panel_label(axes[0], "c", x=-0.13)
    cells_by_stage = [receiver_scores(stage, LR_CASE, "SpatialSelf") for stage in STAGES]
    grids = [density_grid(cells) for cells in cells_by_stage]
    global_vmax = max(float(z.max()) for z, _ in grids)
    global_vmax = max(global_vmax, 1e-6)
    for ax, stage_label, cells, (z, extent), stage in zip(axes, STAGE_LABELS, cells_by_stage, grids, STAGES):
        ax.scatter(cells["x"], cells["y"], s=3.2, color=TISSUE, alpha=0.46, linewidths=0)
        norm = np.clip(z / global_vmax, 0, 1)
        alpha = norm**0.62 * 0.92
        alpha[norm < 0.035] = 0
        ax.imshow(norm, origin="lower", extent=extent, cmap=HOT_CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear")
        if (cells["incoming_score"] > 0).any():
            pos = cells.loc[cells["incoming_score"] > 0]
            cutoff = pos["incoming_score"].quantile(0.88)
            top = pos.loc[pos["incoming_score"] >= cutoff]
            ax.scatter(top["x"], top["y"], s=6.4, color="#6E1D1B", alpha=0.85, linewidths=0)
        set_spatial_axis(ax, cells)
        ax.set_title(stage_label, fontsize=7.2, pad=2.5)
    axes[0].text(0, 1.15, "Cxcl10-Cxcr3 spatial activity", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=7.4, color=INK)


def plot_method_delta(ax) -> None:
    panel_label(ax, "d", x=-0.11)
    df = pd.read_csv(THREE)
    methods = ["SpatialSelf", "COMMOT", "CellNEST"]
    lrs = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
    mat = np.full((len(lrs), len(methods)), np.nan)
    for i, lr in enumerate(lrs):
        for j, method in enumerate(methods):
            rec = df.loc[(df["lr_pair"] == lr) & (df["method"] == method)]
            if len(rec):
                mat[i, j] = rec["dss9_minus_healthy"].iloc[0]
    vmax = max(0.16, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap=DELTA_CMAP, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=6.2)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels(lrs, fontsize=6.2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("D9-D0 response", loc="left", fontsize=7.4, pad=5)
    return im


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    activity = build_program_activity()
    flow = build_flow_composition()

    fig = plt.figure(figsize=(7.25, 5.75))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.58, 1.25, 1.25],
        width_ratios=[1.05, 1.05, 1.05, 1.05],
        left=0.075,
        right=0.965,
        top=0.94,
        bottom=0.12,
        wspace=0.28,
        hspace=0.54,
    )
    ax_strip = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0:2])
    ax_b = fig.add_subplot(gs[1, 2:4])
    spatial_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]
    plot_stage_strip(ax_strip)
    plot_program_heatmap(ax_a, activity)
    plot_flow_bars(ax_b, flow)
    plot_spatial_series(spatial_axes)

    # Inset method panel: compact and separated from the main disease-story axis.
    inset = fig.add_axes([0.685, 0.135, 0.255, 0.22])
    im = plot_method_delta(inset)
    cbar = fig.colorbar(im, ax=inset, fraction=0.050, pad=0.035)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.ax.tick_params(labelsize=5.4, length=1.6, width=0.45)
    cbar.outline.set_visible(False)

    stem = OUT / "mouse_colon_merfish_scipilot_compact_story"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=650, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=650, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
