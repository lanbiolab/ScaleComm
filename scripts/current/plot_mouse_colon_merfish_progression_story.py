#!/usr/bin/env python
"""Build a DSS colitis progression-centered MERFISH communication story figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
REPS = BASE / "multislice_replicates"
SPATIALSELF = BASE / "multislice_spatialself/predictions"
COMMOT = BASE / "commot_priority_baseline/predictions"
CELLNEST = BASE / "cellnest_priority_baseline"
TRAJ_PATH = BASE / "multislice_spatialself/summary/stage_lr_trajectories.csv"
LONG_PATH = BASE / "multislice_spatialself/summary/stage_lr_summary_long.csv"
THREE_PATH = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "progression_story"
FIG = BASE / "figures/merfish_dss_progression_story"

STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
STAGE_PROGRAM_LABELS = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]
LR_CASE = "Cxcl10-Cxcr3"
REPRESENTATIVE_AXES = {
    "homeostasis": "Rspo3-Lgr5",
    "early response": "Il1b-Il1r2",
    "inflammatory peak": "Cxcl10-Cxcr3",
    "repair/remodeling": "Wnt1-Fzd2",
}

INK = "#222222"
MUTED = "#8C887F"
PALE = "#D8D3CA"
GRID = "#E9E4DA"
BG = "#FFFFFF"
HOME = "#5E8C74"
EARLY = "#D49B43"
INFLAM = "#B63A2D"
REPAIR = "#5B78A6"
PROGRAM_COLORS = {
    "homeostasis": HOME,
    "early response": EARLY,
    "inflammatory peak": INFLAM,
    "repair/remodeling": REPAIR,
}
COMPARTMENT_COLORS = {
    "Epithelial": "#D8903D",
    "Immune": "#B84D73",
    "Stromal": "#4E9A88",
    "Vascular": "#587FA9",
    "Neural": "#8A73B4",
    "Other": "#C8C4BB",
}
PROGRAM_CMAP = LinearSegmentedColormap.from_list("program_warm", ["#F6F1E8", "#E0B989", "#BB6043", "#762A24"])
FLOW_CMAP = LinearSegmentedColormap.from_list("flow_clay", ["#F7F2EA", "#E5B883", "#C96646", "#762A24"])
HOT_CMAP = LinearSegmentedColormap.from_list("hotspot", ["#F8F4EC", "#E7BC8D", "#C75B40", "#711E1C"])
DELTA_CMAP = LinearSegmentedColormap.from_list("delta", ["#4D6D8B", "#F6F1EA", "#A94539"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 6.7,
            "axes.linewidth": 0.62,
            "axes.edgecolor": INK,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)


def classify_program(row: pd.Series) -> str:
    if row["trajectory_class"] == "flat_or_decreased" and row["peak_stage"] == "Healthy_D0":
        return "homeostasis"
    if row["trajectory_class"] == "early_induction" or row["peak_stage"] == "DSS3":
        return "early response"
    if (
        row["peak_stage"] == "DSS9"
        and row["biological_category"] in {"Chemokine", "IL", "TNF"}
        and row["trajectory_class"] in {"persistent_inflammation", "inflammation_peak_recovery", "early_induction"}
    ):
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
    if any(k in s for k in ["fibro", "iaf", "frc", "smc", "telocyte", "pericyte", "icc"]):
        return "Stromal"
    if any(k in s for k in ["ec", "endothelial", "arterial", "venous", "capillarial", "lymphatic"]):
        return "Vascular"
    if any(k in s for k in ["neuron", "glia"]):
        return "Neural"
    return "Other"


def rep_dir(stage: str, replicate: int = 1) -> Path:
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    row = manifest.loc[(manifest["stage"] == stage) & (manifest["stage_replicate"] == replicate)].iloc[0]
    return ROOT / row["replicate_dir"]


def pred_path(stage: str, rep_name: str, method: str) -> Path:
    if method == "SpatialSelf":
        return SPATIALSELF / stage / "spatialself_wavelet_adaptive" / f"{rep_name}__spatialself_wavelet_adaptive_predictions.csv"
    if method == "COMMOT":
        return COMMOT / f"{rep_name}__commot_priority_predictions.csv"
    if method == "CellNEST":
        return CELLNEST / f"{rep_name}__cellnest_predictions.csv"
    raise ValueError(method)


def load_slice(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    d = rep_dir(stage, replicate)
    return d.name, pd.read_csv(d / "cells.csv"), pd.read_csv(d / "candidate_interactions.csv")


def edge_table(stage: str, lr_pair: str, method: str = "SpatialSelf", replicate: int = 1) -> pd.DataFrame:
    rep_name, cells, cand = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    sub = cand.loc[(cand["ligand"] == ligand) & (cand["receptor"] == receptor)].copy()
    pred = pd.read_csv(pred_path(stage, rep_name, method))
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
    incoming = edges.groupby("receiver_id", as_index=False)["score"].max().rename(
        columns={"receiver_id": "cell_id", "score": "incoming_score"}
    )
    cells = cells.merge(incoming, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    if rank_norm and (cells["incoming_score"] > 0).any():
        pos = cells["incoming_score"] > 0
        cells.loc[pos, "incoming_score"] = cells.loc[pos, "incoming_score"].rank(pct=True)
    return cells


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_program_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(TRAJ_PATH)
    traj["program"] = traj.apply(classify_program, axis=1)
    traj.to_csv(OUT / "lr_program_assignments.csv", index=False)

    programs = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]
    rows = []
    for program in programs:
        sub = traj.loc[traj["program"] == program]
        for stage in STAGES:
            rows.append(
                {
                    "program": program,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[STAGES.index(stage)],
                    "n_lr": len(sub),
                    "activity": float(sub[stage].mean(skipna=True)) if len(sub) else np.nan,
                    "median_activity": float(sub[stage].median(skipna=True)) if len(sub) else np.nan,
                }
            )
    program_activity = pd.DataFrame(rows)
    program_activity["activity_scaled"] = program_activity.groupby("program")["activity"].transform(lambda x: minmax(x.to_numpy()))
    program_activity.to_csv(OUT / "program_activity_by_stage.csv", index=False)

    rep_rows = []
    for program, lr_pair in REPRESENTATIVE_AXES.items():
        if lr_pair not in set(traj["lr_pair"]):
            continue
        rec = traj.loc[traj["lr_pair"] == lr_pair].iloc[0]
        for stage, label in zip(STAGES, STAGE_LABELS):
            rep_rows.append(
                {
                    "program": program,
                    "lr_pair": lr_pair,
                    "stage": stage,
                    "stage_label": label,
                    "score": float(rec[stage]) if pd.notna(rec[stage]) else np.nan,
                    "biological_category": rec["biological_category"],
                    "trajectory_class": rec["trajectory_class"],
                    "top_cell_type_pair": rec["top_cell_type_pair"],
                }
            )
    representative = pd.DataFrame(rep_rows)
    representative.to_csv(OUT / "representative_lr_trajectories.csv", index=False)

    flow = build_flow_table(traj[["lr_pair", "program"]])
    return program_activity, representative, flow


def build_flow_table(program_map: pd.DataFrame) -> pd.DataFrame:
    flow_rows = []
    program_lookup = dict(zip(program_map["lr_pair"], program_map["program"]))
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    for _, row in manifest.iterrows():
        stage = row["stage"]
        rep_name = Path(row["replicate_dir"]).name
        d = ROOT / row["replicate_dir"]
        cand = pd.read_csv(d / "candidate_interactions.csv")
        pred = pd.read_csv(pred_path(stage, rep_name, "SpatialSelf"))
        merged = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        if merged.empty:
            continue
        cutoff = merged["score"].quantile(0.95)
        high = merged.loc[merged["score"] >= cutoff].copy()
        high["lr_pair"] = high["ligand"] + "-" + high["receptor"]
        high["program"] = high["lr_pair"].map(program_lookup).fillna("other")
        high["sender_compartment"] = high["sender_cell_type"].map(broad_compartment)
        high["receiver_compartment"] = high["receiver_cell_type"].map(broad_compartment)
        high["flow"] = high["sender_compartment"] + " -> " + high["receiver_compartment"]
        grp = (
            high.groupby(["program", "sender_compartment", "receiver_compartment", "flow"], as_index=False)
            .agg(edge_count=("candidate_id", "count"), score_sum=("score", "sum"), score_mean=("score", "mean"))
        )
        grp["stage"] = stage
        grp["stage_label"] = STAGE_LABELS[STAGES.index(stage)]
        grp["stage_replicate"] = row["stage_replicate"]
        flow_rows.append(grp)
    flow = pd.concat(flow_rows, ignore_index=True)
    summary = (
        flow.groupby(["stage", "stage_label", "flow"], as_index=False)
        .agg(edge_count=("edge_count", "sum"), score_sum=("score_sum", "sum"), score_mean=("score_mean", "mean"))
    )
    total = summary.groupby("stage")["score_sum"].transform("sum")
    summary["stage_fraction"] = summary["score_sum"] / total
    summary.to_csv(OUT / "dominant_compartment_flows_by_stage.csv", index=False)
    flow.to_csv(OUT / "dominant_compartment_flows_by_stage_replicate_program.csv", index=False)
    return summary


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


def plot_progression_axis(ax) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.025, y=0.98)
    xs = np.linspace(0.08, 0.92, 4)
    y = 0.58
    ax.plot([xs[0], xs[-1]], [y, y], color=INK, lw=0.8, solid_capstyle="round")
    for i, (x, stage, program) in enumerate(zip(xs, STAGE_LABELS, STAGE_PROGRAM_LABELS)):
        color = list(PROGRAM_COLORS.values())[i]
        ax.scatter([x], [y], s=130, color=color, edgecolor=INK, lw=0.55, zorder=3)
        ax.text(x, y + 0.17, stage, ha="center", va="center", fontsize=8.0, color=INK)
        ax.text(x, y - 0.18, program, ha="center", va="center", fontsize=6.2, color=INK)
    ax.annotate("", xy=(0.77, 0.58), xytext=(0.23, 0.58), arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_program_activity(ax, program_activity: pd.DataFrame) -> mpl.image.AxesImage:
    panel_label(ax, "b", x=-0.10)
    programs = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]
    mat = (
        program_activity.pivot(index="program", columns="stage_label", values="activity_scaled")
        .loc[programs, STAGE_LABELS]
        .to_numpy(float)
    )
    im = ax.imshow(mat, cmap=PROGRAM_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.6)
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(programs, fontsize=6.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("communication programs", loc="left", fontsize=7.2, pad=4)
    return im


def plot_flow_bubbles(ax, flow: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.08)
    totals = flow.groupby("flow")["score_sum"].sum().sort_values(ascending=False)
    selected = [x for x in totals.index if "Other" not in x][:8]
    sub = flow.loc[flow["flow"].isin(selected)].copy()
    # Make within-row temporal change visible while retaining size as total contribution.
    pivot = sub.pivot_table(index="flow", columns="stage_label", values="stage_fraction", aggfunc="sum").reindex(selected)
    pivot = pivot[STAGE_LABELS].fillna(0)
    arr = pivot.to_numpy(float)
    row_scaled = np.vstack([minmax(row) for row in arr])
    for i, flow_name in enumerate(selected):
        for j, label in enumerate(STAGE_LABELS):
            val = pivot.iloc[i, j]
            ax.scatter(
                j,
                i,
                s=32 + 820 * np.sqrt(max(val, 0)),
                color=FLOW_CMAP(row_scaled[i, j]),
                edgecolor="white",
                lw=0.45,
                zorder=3,
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.6)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected, fontsize=6.0)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(len(selected) - 0.45, -0.55)
    ax.grid(axis="x", color=GRID, lw=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("dominant sender-receiver flows", loc="left", fontsize=7.2, pad=4)


def plot_representative_trajectories(ax, representative: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.10)
    for program, lr_pair in REPRESENTATIVE_AXES.items():
        sub = representative.loc[representative["lr_pair"] == lr_pair].set_index("stage").reindex(STAGES)
        color = PROGRAM_COLORS[program]
        vals = sub["score"].to_numpy(float)
        ax.plot(range(4), vals, color=color, lw=1.3, marker="o", ms=2.8, label=lr_pair)
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS, fontsize=6.5)
    ax.set_ylim(0.05, 1.03)
    ax.set_ylabel("score", fontsize=6.4)
    ax.set_xlim(-0.15, 3.15)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.52), ncol=1, fontsize=5.5, handlelength=1.2, borderaxespad=0.2)
    ax.set_title("representative LR axes", loc="left", fontsize=7.2, pad=4)


def plot_edge_hero(ax) -> None:
    panel_label(ax, "e", x=-0.08)
    _, cells, _ = load_slice("DSS9")
    edges = edge_table("DSS9", LR_CASE, "SpatialSelf").sort_values("score", ascending=False)
    top = edges.head(34)
    ax.scatter(cells["x"], cells["y"], s=4.2, color=PALE, alpha=0.42, linewidths=0, zorder=1)
    ax.scatter(top["sender_x"], top["sender_y"], s=16, color="#DE922F", alpha=0.95, linewidths=0, zorder=4)
    ax.scatter(top["receiver_x"], top["receiver_y"], s=18, color="#6E1D1B", alpha=0.97, linewidths=0, zorder=5)
    lo, hi = top["score"].min(), top["score"].max()
    for k, (_, row) in enumerate(top.iterrows()):
        rad = 0.12 if k % 2 == 0 else -0.12
        lw = 0.25 + 0.7 * ((row["score"] - lo) / max(hi - lo, 1e-6))
        patch = FancyArrowPatch(
            (row["sender_x"], row["sender_y"]),
            (row["receiver_x"], row["receiver_y"]),
            arrowstyle="-|>",
            mutation_scale=5.2,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw,
            color=INFLAM,
            alpha=0.42,
            zorder=3,
        )
        ax.add_patch(patch)
    set_spatial_axis(ax, cells)
    ax.set_title("D9 Cxcl10-Cxcr3 edges", fontsize=7.4, pad=3)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#DE922F", markeredgewidth=0, markersize=4, label="sender"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E1D1B", markeredgewidth=0, markersize=4, label="receiver"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.7, handletextpad=0.3, borderpad=0.1)


def density_image(cells: pd.DataFrame, bins: int = 142, sigma: float = 3.1) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = cells["x"].to_numpy()
    y = cells["y"].to_numpy()
    w = cells["incoming_score"].to_numpy(float)
    dx = (x.max() - x.min()) * 0.045
    dy = (y.max() - y.min()) * 0.045
    extent = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]], weights=w)
    z = gaussian_filter(h.T, sigma=sigma)
    if z.max() > 0:
        z = z / z.max()
    return z, extent


def plot_density_map(ax, stage: str, method: str, title: str) -> None:
    cells = receiver_scores(stage, LR_CASE, method=method, rank_norm=(method != "SpatialSelf"))
    ax.scatter(cells["x"], cells["y"], s=3.0, color=PALE, alpha=0.42, linewidths=0)
    z, extent = density_image(cells)
    alpha = np.clip(z, 0, 1) ** 0.64 * 0.90
    alpha[z < 0.035] = 0
    ax.imshow(z, origin="lower", extent=extent, cmap=HOT_CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear")
    if (cells["incoming_score"] > 0).any():
        pos = cells.loc[cells["incoming_score"] > 0].copy()
        cutoff = pos["incoming_score"].quantile(0.84)
        top = pos.loc[pos["incoming_score"] >= cutoff]
        ax.scatter(top["x"], top["y"], s=7.2, color="#6E1D1B", alpha=0.88, linewidths=0)
    set_spatial_axis(ax, cells)
    ax.set_title(title, fontsize=6.9, pad=2.5)


def plot_method_delta(ax) -> mpl.image.AxesImage:
    panel_label(ax, "g", x=-0.16)
    three = pd.read_csv(THREE_PATH)
    methods = ["SpatialSelf", "COMMOT", "CellNEST"]
    lrs = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
    mat = np.full((len(lrs), len(methods)), np.nan)
    for i, lr_pair in enumerate(lrs):
        for j, method in enumerate(methods):
            rec = three.loc[(three["lr_pair"] == lr_pair) & (three["method"] == method)]
            if len(rec):
                mat[i, j] = rec["dss9_minus_healthy"].iloc[0]
    vmax = max(0.16, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap=DELTA_CMAP, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=6.1)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels(lrs, fontsize=6.1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("D9-D0", loc="left", fontsize=7.2, pad=4)
    return im


def build_figure() -> None:
    setup_style()
    FIG.mkdir(parents=True, exist_ok=True)
    program_activity, representative, flow = build_program_tables()

    fig = plt.figure(figsize=(7.35, 7.1))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.62, 1.42, 1.48, 1.08],
        left=0.07,
        right=0.97,
        top=0.965,
        bottom=0.075,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(outer[0])
    mid = outer[1].subgridspec(1, 3, width_ratios=[0.95, 1.55, 1.25], wspace=0.34)
    ax_b = fig.add_subplot(mid[0, 0])
    ax_c = fig.add_subplot(mid[0, 1])
    ax_d = fig.add_subplot(mid[0, 2])
    bottom_spatial = outer[2].subgridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.08)
    ax_e = fig.add_subplot(bottom_spatial[0, 0])
    method_axes = [fig.add_subplot(bottom_spatial[0, i]) for i in range(1, 4)]
    final = outer[3].subgridspec(1, 2, width_ratios=[1.7, 0.85], wspace=0.32)
    ax_f = fig.add_subplot(final[0, 0])
    ax_g = fig.add_subplot(final[0, 1])

    plot_progression_axis(ax_a)
    plot_program_activity(ax_b, program_activity)
    plot_flow_bubbles(ax_c, flow)
    plot_representative_trajectories(ax_d, representative)
    plot_edge_hero(ax_e)

    panel_label(method_axes[0], "f", x=-0.12)
    for ax, method in zip(method_axes, ["SpatialSelf", "COMMOT", "CellNEST"]):
        plot_density_map(ax, "DSS9", method, method)
    im = plot_method_delta(ax_g)
    cbar = fig.colorbar(im, ax=ax_g, fraction=0.055, pad=0.045)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.ax.tick_params(labelsize=5.7, length=1.7, width=0.45)
    cbar.outline.set_visible(False)

    ax_f.set_axis_off()
    ax_f.text(0.0, 0.82, "DSS colitis progression", fontsize=8.2, color=INK, fontweight="bold")
    ax_f.text(
        0.0,
        0.56,
        "SpatialSelf links disease stage, sender-receiver flow and LR-specific spatial activity.",
        fontsize=7.0,
        color=INK,
    )
    ax_f.text(
        0.0,
        0.30,
        "The example axis Cxcl10-Cxcr3 highlights the D9 inflammatory communication network.",
        fontsize=7.0,
        color=MUTED,
    )

    stem = FIG / "mouse_colon_merfish_dss_progression_story"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=650, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=650, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    build_figure()
