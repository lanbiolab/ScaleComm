#!/usr/bin/env python
"""Publication-style DSS colitis MERFISH communication rewiring figure.

This script follows the scipilot-figure-skill workflow: compact multi-panel
figure, one disease-progression story, restrained colors, vector exports, and
PNG preview for visual QA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
REPS = BASE / "multislice_replicates"
PRED = BASE / "multislice_spatialself/predictions"
COMMOT = BASE / "commot_priority_baseline/predictions"
CELLNEST = BASE / "cellnest_priority_baseline"
TRAJ = BASE / "multislice_spatialself/summary/stage_lr_trajectories.csv"
THREE_METHOD = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"
OUT = BASE / "figures/merfish_dss_scipilot_story_v2"
TABLE_OUT = BASE / "progression_story"

SKILL_SCRIPTS = Path.home() / ".codex/skills/scipilot-figure-skill/scripts"
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))

try:
    from setup_style import setup_style
except Exception:  # pragma: no cover - fallback only for portability
    setup_style = None


STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
SPATIAL_STAGES = ["Healthy_D0", "DSS9", "DSS21"]
SPATIAL_STAGE_LABELS = ["D0", "D9", "D21"]
PROGRAMS = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]
LR_CASE = "Cxcl10-Cxcr3"
INFLAMMATORY_AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
CHEMOKINE_LIGANDS = ["Cxcl10", "Ccl5", "Ccl2"]
CHEMOKINE_RECEPTORS = ["Cxcr3", "Ccr5", "Ccr2"]
METHODS = ["SpatialSelf", "COMMOT", "CellNEST"]

INK = "#222222"
MUTED = "#6F6A61"
LIGHT = "#E7E1D8"
PANEL_BG = "#FBFAF7"

COMPARTMENT_COLORS = {
    "Epithelial": "#D9953D",
    "Immune": "#B54C67",
    "Mesenchymal": "#4F9B8B",
    "Vascular": "#5E7FA4",
    "Neural": "#8A7AA3",
    "Other": "#BEB8AD",
}
METHOD_COLORS = {
    "SpatialSelf": "#A33A32",
    "COMMOT": "#377C89",
    "CellNEST": "#6F6A75",
}

PROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "program_activity",
    ["#F4EFE7", "#E5C68F", "#CC7A4B", "#7E2F2B"],
)
ACTIVITY_CMAP = LinearSegmentedColormap.from_list(
    "activity",
    ["#FFF8EE", "#EEB980", "#C85D45", "#641C1C"],
)
LIGAND_CMAP = LinearSegmentedColormap.from_list(
    "ligand_field",
    ["#FFF8EE", "#E7BE7E", "#C76540", "#6E211F"],
)
RECEPTOR_CMAP = LinearSegmentedColormap.from_list(
    "receptor_field",
    ["#F6F3EA", "#BFD0C7", "#5D9A8D", "#1F5D62"],
)
COLOC_CMAP = LinearSegmentedColormap.from_list(
    "colocalization",
    ["#FBF6ED", "#D8B979", "#B3685D", "#55304A"],
)
SHIFT_CMAP = LinearSegmentedColormap.from_list(
    "flow_shift",
    ["#2F6F73", "#F7F2EA", "#9D3C38"],
)


def setup_publication_style() -> None:
    if setup_style is not None:
        try:
            setup_style(journal="nature", lang="en", use_sciplots=False)
        except TypeError:
            setup_style(journal="nature", lang="en")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.7,
            "axes.labelsize": 6.9,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.026,
        box.y1 + 0.012,
        label,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    if any(k in s for k in ["colonocyte", "epithelial", "goblet", "stem", "tuft", "ta", "iae", "eec", "clu"]):
        return "Epithelial"
    if any(k in s for k in ["macrophage", "neutrophil", "monocyte", "t ", "t(", "b cell", "plasma", "dc", "treg", "mast", "nk", "tfh", "ilc"]):
        return "Immune"
    if any(k in s for k in ["fibro", "iaf", "frc", "smc", "iasmc", "pericyte", "icc", "adipose", "repair associated"]):
        return "Mesenchymal"
    if any(k in s for k in ["ec", "endothelial", "arterial", "venous", "capillarial", "lymphatic"]):
        return "Vascular"
    if any(k in s for k in ["neuron", "glia"]):
        return "Neural"
    return "Other"


def stage_replicate_dir(stage: str, replicate: int = 1) -> Path:
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    row = manifest.loc[(manifest["stage"] == stage) & (manifest["stage_replicate"] == replicate)].iloc[0]
    return ROOT / row["replicate_dir"]


def load_slice(stage: str, replicate: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    d = stage_replicate_dir(stage, replicate)
    cells = pd.read_csv(d / "cells.csv")
    cand = pd.read_csv(d / "candidate_interactions.csv")
    return d.name, cells, cand


def load_expression_slice(stage: str, replicate: int = 1) -> pd.DataFrame:
    d = stage_replicate_dir(stage, replicate)
    cells = pd.read_csv(d / "cells.csv")
    expr = pd.read_csv(d / "expression.csv")
    return cells.merge(expr, on="cell_id", how="left")


def prediction_path(stage: str, replicate_name: str, method: str = "SpatialSelf") -> Path:
    if method == "SpatialSelf":
        return PRED / stage / "spatialself_wavelet_adaptive" / f"{replicate_name}__spatialself_wavelet_adaptive_predictions.csv"
    if method == "COMMOT":
        return COMMOT / f"{replicate_name}__commot_priority_predictions.csv"
    if method == "CellNEST":
        return CELLNEST / f"{replicate_name}__cellnest_predictions.csv"
    raise ValueError(method)


def normalize_extent(cells: pd.DataFrame) -> tuple[float, float, float, float]:
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    pad_x = (x.max() - x.min()) * 0.045
    pad_y = (y.max() - y.min()) * 0.045
    return x.min() - pad_x, x.max() + pad_x, y.min() - pad_y, y.max() + pad_y


def set_spatial_limits(ax: plt.Axes, cells: pd.DataFrame) -> None:
    x0, x1, y0, y1 = normalize_extent(cells)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    clean_axis(ax)


def spatial_activity_grid(
    cells: pd.DataFrame,
    score_col: str = "incoming_score",
    bins: int = 185,
    sigma: float = 2.2,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return smoothed score-density grid and density mask on tissue coordinates."""
    x0, x1, y0, y1 = normalize_extent(cells)
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    score = cells[score_col].to_numpy(float)
    density, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]])
    weighted, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]], weights=score)
    density_s = gaussian_filter(density.T, sigma=sigma)
    weighted_s = gaussian_filter(weighted.T, sigma=sigma)
    tissue = density_s / (density_s.max() if density_s.max() else 1.0)
    tissue[tissue < 0.020] = 0
    return weighted_s, tissue, (x0, x1, y0, y1)


def expression_field_grid(
    cells: pd.DataFrame,
    score_col: str,
    bins: int = 190,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x0, x1, y0, y1 = normalize_extent(cells)
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    score = cells[score_col].to_numpy(float)
    density, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]])
    weighted, _, _ = np.histogram2d(x, y, bins=bins, range=[[x0, x1], [y0, y1]], weights=score)
    density_s = gaussian_filter(density.T, sigma=sigma)
    weighted_s = gaussian_filter(weighted.T, sigma=sigma)
    field = weighted_s / (density_s + 1e-8)
    tissue = density_s / (density_s.max() if density_s.max() else 1.0)
    tissue[tissue < 0.018] = 0
    field[tissue == 0] = np.nan
    return field, tissue, (x0, x1, y0, y1)


def draw_expression_field(
    ax: plt.Axes,
    cells: pd.DataFrame,
    field: np.ndarray,
    tissue: np.ndarray,
    extent: tuple[float, float, float, float],
    vmax: float,
    cmap,
) -> None:
    ax.scatter(cells["x"], cells["y"], s=0.9, color="#D7D2C9", alpha=0.11, linewidths=0, rasterized=True)
    clipped = np.clip(field / max(vmax, 1e-9), 0, 1)
    alpha = np.where(np.isfinite(clipped), np.clip((clipped ** 0.46) * 0.98, 0, 0.98), 0)
    alpha[clipped < 0.025] = 0
    alpha[tissue <= 0] = 0
    ax.imshow(
        clipped,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=alpha,
        interpolation="bilinear",
    )
    set_spatial_limits(ax, cells)


def draw_spatial_heatmap(
    ax: plt.Axes,
    cells: pd.DataFrame,
    vmax: float,
    cmap,
    background: bool = True,
    bins: int = 220,
    sigma: float = 2.60,
) -> None:
    grid, tissue, extent = spatial_activity_grid(cells, bins=bins, sigma=sigma)
    if background:
        ax.scatter(cells["x"], cells["y"], s=1.1, color="#D7D2C9", alpha=0.12, linewidths=0, rasterized=True)
    norm_grid = np.clip(grid / max(vmax, 1e-9), 0, 1)
    alpha = np.clip((norm_grid ** 0.34) * 0.98, 0, 0.98)
    alpha[norm_grid < 0.014] = 0
    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=alpha,
        interpolation="bilinear",
    )
    set_spatial_limits(ax, cells)


def receiver_scores(stage: str, lr_pair: str, method: str = "SpatialSelf", replicate: int = 1) -> pd.DataFrame:
    rep, cells, cand = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    pred = pd.read_csv(prediction_path(stage, rep, method))
    sub = cand.loc[(cand["ligand"] == ligand) & (cand["receptor"] == receptor)].copy()
    sub = sub.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    cells = cells.copy()
    if sub.empty:
        cells["incoming_score"] = 0.0
        return cells
    inc = (
        sub.groupby("receiver_id", as_index=False)["score"]
        .max()
        .rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
    )
    cells = cells.merge(inc, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    return cells


def aggregate_receiver_scores(stage: str, lr_pairs: list[str], method: str = "SpatialSelf", replicate: int = 1) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for lr_pair in lr_pairs:
        cells = receiver_scores(stage, lr_pair, method, replicate)
        if base is None:
            base = cells[["cell_id", "x", "y", "cell_type"]].copy()
            base["incoming_score"] = 0.0
        base = base.merge(cells[["cell_id", "incoming_score"]], on="cell_id", how="left", suffixes=("", "_lr"))
        base["incoming_score"] = base["incoming_score"] + base["incoming_score_lr"].fillna(0.0)
        base = base.drop(columns=["incoming_score_lr"])
    if base is None:
        _, cells, _ = load_slice(stage, replicate)
        base = cells[["cell_id", "x", "y", "cell_type"]].copy()
        base["incoming_score"] = 0.0
    return base


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_program_activity() -> pd.DataFrame:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(TRAJ)
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
    out.to_csv(TABLE_OUT / "scipilot_story_v2_program_activity.csv", index=False)
    return out


def build_flow_shift() -> pd.DataFrame:
    manifest = pd.read_csv(REPS / "multislice_replicate_manifest.csv")
    rows = []
    for _, meta in manifest.iterrows():
        stage = meta["stage"]
        rep_name = Path(meta["replicate_dir"]).name
        d = ROOT / meta["replicate_dir"]
        cand = pd.read_csv(d / "candidate_interactions.csv")
        pred = pd.read_csv(prediction_path(stage, rep_name))
        merged = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        if merged.empty:
            continue
        cutoff = merged["score"].quantile(0.95)
        high = merged.loc[merged["score"] >= cutoff].copy()
        high["sender_compartment"] = high["sender_cell_type"].map(broad_compartment)
        high["receiver_compartment"] = high["receiver_cell_type"].map(broad_compartment)
        grouped = high.groupby(["sender_compartment", "receiver_compartment"], as_index=False)["score"].sum()
        grouped["stage"] = stage
        rows.append(grouped)
    flow = pd.concat(rows, ignore_index=True)
    summary = flow.groupby(["stage", "sender_compartment", "receiver_compartment"], as_index=False)["score"].sum()
    summary["fraction"] = summary["score"] / summary.groupby("stage")["score"].transform("sum")

    d0 = summary.loc[summary["stage"] == "Healthy_D0"].copy()
    d9 = summary.loc[summary["stage"] == "DSS9"].copy()
    shift = d9.merge(
        d0,
        on=["sender_compartment", "receiver_compartment"],
        how="outer",
        suffixes=("_d9", "_d0"),
    )
    shift["fraction_d9"] = shift["fraction_d9"].fillna(0)
    shift["fraction_d0"] = shift["fraction_d0"].fillna(0)
    shift["d9_minus_d0"] = shift["fraction_d9"] - shift["fraction_d0"]
    shift.to_csv(TABLE_OUT / "scipilot_story_v2_flow_shift_d9_minus_d0.csv", index=False)
    return shift


def plot_compartment_maps(fig: plt.Figure, gs) -> None:
    axes = []
    for i, (stage, label) in enumerate(zip(STAGES, STAGE_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        _, cells, _ = load_slice(stage)
        cells = cells.copy()
        cells["compartment"] = cells["cell_type"].map(broad_compartment)
        for comp, color in COMPARTMENT_COLORS.items():
            sub = cells.loc[cells["compartment"] == comp]
            if len(sub):
                ax.scatter(sub["x"], sub["y"], s=3.0, color=color, alpha=0.82, linewidths=0, rasterized=True)
        set_spatial_limits(ax, cells)
        ax.set_title(label, fontsize=7.2, pad=2.0, color=INK)
        axes.append(ax)

    add_panel_label(fig, axes[0], "a")
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.034, "MERFISH tissue context", fontsize=7.8, color=INK)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markersize=3.4, color=color, label=comp)
        for comp, color in COMPARTMENT_COLORS.items()
        if comp != "Other"
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0].get_position().x0 + 0.270, axes[0].get_position().y1 + 0.060),
        ncol=5,
        handletextpad=0.25,
        columnspacing=0.72,
        borderaxespad=0.0,
        frameon=False,
        fontsize=5.9,
    )


def plot_program_heatmap(fig: plt.Figure, ax: plt.Axes, activity: pd.DataFrame) -> None:
    matrix = activity.pivot(index="program", columns="stage_label", values="relative_activity").loc[PROGRAMS, STAGE_LABELS]
    im = ax.imshow(matrix.to_numpy(), cmap=PROGRAM_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_yticks(np.arange(len(PROGRAMS)))
    ax.set_yticklabels(["homeostasis", "early response", "inflammatory peak", "repair/remodeling"])
    ax.tick_params(length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Communication programs", loc="left", pad=5.5)
    add_panel_label(fig, ax, "b")
    cax = ax.inset_axes([0.02, -0.18, 0.44, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(length=0, pad=1.2, labelsize=5.7)
    cb.set_label("relative activity", fontsize=5.8, labelpad=1.0, color=MUTED)
    cb.outline.set_visible(False)


def plot_lr_spatial_maps(fig: plt.Figure, gs) -> None:
    stage_cells = [aggregate_receiver_scores(stage, INFLAMMATORY_AXES, "SpatialSelf") for stage in SPATIAL_STAGES]
    grids = [spatial_activity_grid(c, bins=220, sigma=2.60)[0] for c in stage_cells]
    vmax = max(float(np.nanquantile(g[g > 0], 0.955)) if np.any(g > 0) else 0.0 for g in grids)
    vmax = max(vmax, 1e-6)
    axes = []
    for i, (cells, label) in enumerate(zip(stage_cells, SPATIAL_STAGE_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        draw_spatial_heatmap(ax, cells, vmax=vmax, cmap=ACTIVITY_CMAP, bins=220, sigma=2.60)
        ax.set_title(label, fontsize=7.8, pad=2.0, color=INK)
        axes.append(ax)
    add_panel_label(fig, axes[0], "c")
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.030, "SpatialSelf inflammatory chemokine activity", fontsize=7.8, color=INK)
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=ACTIVITY_CMAP)
    cax = axes[-1].inset_axes([0.62, -0.13, 0.34, 0.055])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["0", "high"])
    cb.ax.tick_params(length=0, pad=1.1, labelsize=5.7)
    cb.outline.set_visible(False)


def plot_method_stage_matrix(fig: plt.Figure, gs) -> None:
    axes: list[plt.Axes] = []
    for row, method in enumerate(METHODS):
        cells_by_stage = [aggregate_receiver_scores(stage, INFLAMMATORY_AXES, method) for stage in STAGES]
        grids = [spatial_activity_grid(c, bins=150, sigma=2.0)[0] for c in cells_by_stage]
        vmax = max(float(np.nanquantile(g[g > 0], 0.970)) if np.any(g > 0) else 0.0 for g in grids)
        vmax = max(vmax, 1e-6)
        for col, (stage_label, cells) in enumerate(zip(STAGE_LABELS, cells_by_stage)):
            ax = fig.add_subplot(gs[row, col])
            draw_spatial_heatmap(ax, cells, vmax=vmax, cmap=ACTIVITY_CMAP, bins=150, sigma=2.0)
            if row == 0:
                ax.set_title(stage_label, fontsize=6.7, pad=1.6, color=INK)
            if col == 0:
                ax.text(
                    -0.11,
                    0.50,
                    method,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.8,
                    color=METHOD_COLORS[method],
                )
            axes.append(ax)
    ax_bar = fig.add_subplot(gs[:, 4])
    df = pd.read_csv(THREE_METHOD)
    specificity = []
    for method in METHODS:
        sub = df.loc[(df["method"] == method) & (df["lr_pair"].isin(INFLAMMATORY_AXES))]
        means = sub[STAGES].mean(axis=0)
        specificity.append(float(means["DSS9"] - means.drop("DSS9").max()))
    y = np.arange(len(METHODS))
    colors = [METHOD_COLORS[m] if v >= 0 else "#B9B3AA" for m, v in zip(METHODS, specificity)]
    ax_bar.barh(y, specificity, color=colors, height=0.46, edgecolor="none")
    ax_bar.axvline(0, color=INK, lw=0.55)
    ax_bar.set_xlim(-0.38, 0.08)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_title("D9 -\nbest other", fontsize=6.2, pad=2.0)
    ax_bar.set_xlabel("score", fontsize=5.8, labelpad=1.0)
    ax_bar.tick_params(axis="x", labelsize=5.4, length=2.0, pad=1.0)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.grid(axis="x", color="#ECE7DE", lw=0.40)
    add_panel_label(fig, axes[0], "d")
    fig.text(
        axes[0].get_position().x0,
        axes[0].get_position().y1 + 0.028,
        "Inflammatory chemokine spatial specificity by method",
        fontsize=7.8,
        color=INK,
    )
    fig.text(
        axes[-1].get_position().x1 - 0.003,
        axes[-1].get_position().y0 - 0.018,
        "color scale normalized within each method",
        fontsize=5.6,
        color=MUTED,
        ha="right",
    )


def plot_colocalization_heatmaps(fig: plt.Figure, gs) -> None:
    panels = []
    for stage in SPATIAL_STAGES:
        cells = load_expression_slice(stage)
        ligand_genes = [g for g in CHEMOKINE_LIGANDS if g in cells.columns]
        receptor_genes = [g for g in CHEMOKINE_RECEPTORS if g in cells.columns]
        cells = cells.copy()
        cells["ligand_program"] = np.log1p(cells[ligand_genes].sum(axis=1)) if ligand_genes else 0.0
        cells["receptor_program"] = np.log1p(cells[receptor_genes].sum(axis=1)) if receptor_genes else 0.0
        ligand_field, tissue, extent = expression_field_grid(cells, "ligand_program", bins=220, sigma=4.8)
        receptor_field, _, _ = expression_field_grid(cells, "receptor_program", bins=220, sigma=4.8)
        panels.append(
            {
                "stage": stage,
                "cells": cells,
                "ligand": ligand_field,
                "receptor": receptor_field,
                "tissue": tissue,
                "extent": extent,
            }
        )

    ligand_values = np.concatenate([p["ligand"][np.isfinite(p["ligand"])] for p in panels])
    receptor_values = np.concatenate([p["receptor"][np.isfinite(p["receptor"])] for p in panels])
    ligand_vmax = max(float(np.nanquantile(ligand_values, 0.992)), 1e-6)
    receptor_vmax = max(float(np.nanquantile(receptor_values, 0.992)), 1e-6)
    for p in panels:
        ligand_norm = np.clip(p["ligand"] / ligand_vmax, 0, 1)
        receptor_norm = np.clip(p["receptor"] / receptor_vmax, 0, 1)
        p["coloc"] = np.sqrt(ligand_norm * receptor_norm)
    coloc_values = np.concatenate([p["coloc"][np.isfinite(p["coloc"])] for p in panels])
    coloc_vmax = max(float(np.nanquantile(coloc_values, 0.985)), 1e-6)

    row_info = [
        ("ligands\nCxcl10/Ccl5/Ccl2", "ligand", LIGAND_CMAP, 1.0),
        ("receptors\nCxcr3/Ccr5/Ccr2", "receptor", RECEPTOR_CMAP, 1.0),
        ("co-localization\nligand x receptor", "coloc", COLOC_CMAP, 1.0),
    ]
    axes: list[plt.Axes] = []
    for row, (label, key, cmap, vmax) in enumerate(row_info):
        for col, (stage_label, p) in enumerate(zip(SPATIAL_STAGE_LABELS, panels)):
            ax = fig.add_subplot(gs[row, col])
            if key == "ligand":
                field = np.clip(p[key] / ligand_vmax, 0, 1)
            elif key == "receptor":
                field = np.clip(p[key] / receptor_vmax, 0, 1)
            else:
                field = np.clip(p[key] / coloc_vmax, 0, 1)
            draw_expression_field(ax, p["cells"], field, p["tissue"], p["extent"], vmax, cmap)
            if row == 0:
                ax.set_title(stage_label, fontsize=6.8, pad=1.7, color=INK)
            if col == 0:
                ax.text(
                    -0.075,
                    0.50,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.6,
                    color=INK,
                )
            axes.append(ax)
    add_panel_label(fig, axes[0], "d")
    fig.text(
        axes[0].get_position().x0,
        axes[0].get_position().y1 + 0.028,
        "Spatial co-localization of inflammatory chemokine ligand and receptor programs",
        fontsize=7.8,
        color=INK,
    )


def plot_method_trajectories(fig: plt.Figure, gs) -> None:
    df = pd.read_csv(THREE_METHOD)
    axes = []
    for i, lr_pair in enumerate(INFLAMMATORY_AXES):
        ax = fig.add_subplot(gs[0, i])
        for method in METHODS:
            rec = df.loc[(df["method"] == method) & (df["lr_pair"] == lr_pair)]
            if rec.empty:
                continue
            values = rec[STAGES].iloc[0].to_numpy(float)
            ax.plot(
                np.arange(len(STAGE_LABELS)),
                values,
                color=METHOD_COLORS[method],
                lw=1.15,
                marker="o",
                ms=2.5,
                label=method,
            )
        ax.set_title(lr_pair, fontsize=6.7, pad=2.2)
        ax.set_xticks(np.arange(len(STAGE_LABELS)))
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_ylim(-0.02, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#ECE7DE", lw=0.42)
        if i == 0:
            ax.set_ylabel("stage score")
        else:
            ax.set_yticklabels([])
        axes.append(ax)
    add_panel_label(fig, axes[0], "e")
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.035, "Inflammatory-axis trajectories", fontsize=7.8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(axes[-1].get_position().x1 - 0.210, axes[0].get_position().y1 + 0.060),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.75,
        fontsize=5.9,
    )


def plot_flow_shift(fig: plt.Figure, ax: plt.Axes, shift: pd.DataFrame) -> None:
    compartments = ["Epithelial", "Immune", "Mesenchymal", "Vascular", "Neural"]
    matrix = np.zeros((len(compartments), len(compartments)))
    for i, sender in enumerate(compartments):
        for j, receiver in enumerate(compartments):
            rec = shift.loc[(shift["sender_compartment"] == sender) & (shift["receiver_compartment"] == receiver), "d9_minus_d0"]
            matrix[i, j] = float(rec.iloc[0]) if len(rec) else 0.0
    max_abs = max(0.04, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap=SHIFT_CMAP, norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs), interpolation="nearest")
    ax.set_xticks(np.arange(len(compartments)))
    ax.set_yticks(np.arange(len(compartments)))
    ax.set_xticklabels(["Epi", "Imm", "Mes", "Vas", "Neu"], rotation=0)
    ax.set_yticklabels(["Epi", "Imm", "Mes", "Vas", "Neu"])
    ax.set_xlabel("receiver")
    ax.set_ylabel("sender")
    ax.tick_params(length=0, pad=2.0)
    ax.set_title("Inflammatory-peak flow shift", loc="left", pad=5.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(compartments), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(compartments), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    add_panel_label(fig, ax, "f")
    cax = ax.inset_axes([1.055, 0.13, 0.045, 0.68])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks([-max_abs, 0, max_abs])
    cb.set_ticklabels(["D0", "0", "D9"])
    cb.ax.tick_params(length=0, pad=1.1, labelsize=5.7)
    cb.set_label("D9-D0", fontsize=5.8, labelpad=2.0, color=MUTED)
    cb.outline.set_visible(False)


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)

    activity = build_program_activity()
    shift = build_flow_shift()

    fig = plt.figure(figsize=(7.20, 9.65), constrained_layout=False)
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.25, 1.95, 3.15, 1.38],
        left=0.055,
        right=0.965,
        bottom=0.065,
        top=0.940,
        hspace=0.48,
    )

    top = outer[0, 0].subgridspec(1, 2, width_ratios=[4.55, 2.10], wspace=0.34)
    map_top = top[0, 0].subgridspec(1, 4, wspace=0.030)
    ax_b = fig.add_subplot(top[0, 1])

    spatial_stage = outer[1, 0].subgridspec(1, 3, wspace=0.055)
    spatial_methods = outer[2, 0].subgridspec(3, 3, wspace=0.055, hspace=0.075)

    bottom = outer[3, 0].subgridspec(1, 2, width_ratios=[4.65, 2.05], wspace=0.34)
    traj_gs = bottom[0, 0].subgridspec(1, 3, wspace=0.34)
    ax_f = fig.add_subplot(bottom[0, 1])

    plot_compartment_maps(fig, map_top)
    plot_program_heatmap(fig, ax_b, activity)
    plot_lr_spatial_maps(fig, spatial_stage)
    plot_colocalization_heatmaps(fig, spatial_methods)
    plot_method_trajectories(fig, traj_gs)
    plot_flow_shift(fig, ax_f, shift)

    stem = OUT / "mouse_colon_merfish_dss_scipilot_story_v2"
    fig.savefig(f"{stem}.png", dpi=650, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=650, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
