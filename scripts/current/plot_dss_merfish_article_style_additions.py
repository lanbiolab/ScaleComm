#!/usr/bin/env python
"""Article-style DSS MERFISH communication evidence additions.

Adds CytoSignal/article-figure-inspired panels to strengthen the DSS story:
- differential LR landscape (DSS9 vs D0; effect-size, not formal FDR)
- stage-wise LR activity heatmap across representative and top rewired axes
- replicate spatial activity grid for inflammatory/recovery LR programs
- cell-compartment flow shift across stages
- top stage-specific LR program counts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (  # noqa: E402
    BASE,
    COMPARTMENT_COLORS,
    STAGE_LABELS,
    STAGES,
    broad_compartment,
    load_slice,
    set_spatial_limits,
    setup_publication_style,
)
from scripts.current.plot_mouse_colon_merfish_cellnest_style_story_v21 import (  # noqa: E402
    MODEL_NAME,
    PRED_ROOT,
    prediction_path,
    read_predictions,
)

OUT = ROOT / "outputs/article_style_figure_additions/dss_merfish_article_style"
SOURCE = OUT / "source_data"
SUMMARY = BASE / "multislice_spatialself_scale_v21/summary"
TRAJ = SUMMARY / "stage_lr_trajectories.csv"
FLOW = BASE / "progression_story/dominant_compartment_flows_by_stage.csv"

INK = "#171717"
TEXT = "#2D2A27"
MUTED = "#6E665B"
GRID = "#E7E1D8"
BG = "#FFFFFF"
RED = "#C94F48"
GREEN = "#4E9B73"
AMBER = "#D6A64A"
TEAL = "#237A78"
TAUPE = "#8A8177"
PALE = "#F7F4EC"
BLUE = "#2F78A8"

STAGE_COLORS = {"D0": TAUPE, "D3": AMBER, "D9": RED, "D21": TEAL}
CLASS_COLORS = {
    "flat_or_decreased": TAUPE,
    "early_induction": AMBER,
    "inflammation_peak_recovery": RED,
    "persistent_inflammation": "#B36A55",
    "late_repair_or_persistent": TEAL,
}
MECH_COLORS = {
    "long_range_chemokine_niche": "#78A6C8",
    "short_medium_paracrine": "#D7B56D",
}

ACTIVITY_CMAP = LinearSegmentedColormap.from_list("activity_article", ["#F4F1E8", "#E8B66D", "#C94F48", "#641C1C"])
DIFF_CMAP = LinearSegmentedColormap.from_list("diff_article", ["#2F78A8", "#F7F4EC", "#C94F48"])
HEAT_CMAP = LinearSegmentedColormap.from_list("heat_article", ["#F7F4EC", "#E2BF72", "#C94F48"])

REPRESENTATIVE_LR = ["Rspo3-Lgr5", "Il5-Il5ra", "Cxcl10-Cxcr3", "Il1b-Il1r2", "Tgfb3-Acvrl1", "Wnt1-Wif1"]
STAGE_ORDER = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
STAGE_LABEL_MAP = dict(zip(STAGE_ORDER, ["D0", "D3", "D9", "D21"]))


def setup_style() -> None:
    setup_publication_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.030, dy: float = 0.014) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.45, length=2.0, pad=1.4, colors=TEXT)
    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.35, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.35, zorder=0)
    ax.set_axisbelow(True)


def robust_norm(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo, hi = arr.quantile(0.02), arr.quantile(0.98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = arr.min(), arr.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def build_volcano(traj: pd.DataFrame) -> pd.DataFrame:
    df = traj.copy()
    df["effect_d9_d0"] = df["DSS9"].astype(float) - df["Healthy_D0"].astype(float)
    df["effect_d21_d0"] = df["DSS21"].astype(float) - df["Healthy_D0"].astype(float)
    df["abs_effect"] = df["effect_d9_d0"].abs()
    # No formal replicate-level FDR here; encode effect size and replicate stability as evidence.
    df["evidence"] = df["abs_effect"] * df["replicate_stability_score"].fillna(0.5)
    df["evidence_plot"] = -np.log10(np.clip(1.0 - robust_norm(df["evidence"]), 1e-4, 1.0))
    df["highlight"] = df["lr_pair"].isin(["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Il1b-Il1r2", "Tgfb3-Acvrl1"])
    return df


def select_heatmap_lrs(traj: pd.DataFrame) -> pd.DataFrame:
    df = traj.copy()
    explicit = [lr for lr in REPRESENTATIVE_LR if lr in set(df["lr_pair"])]
    top = (
        df.sort_values(["story_score", "replicate_stability_score"], ascending=False)
        .loc[~df["lr_pair"].isin(explicit)]
        .head(8)["lr_pair"]
        .tolist()
    )
    selected = explicit + top
    out = df[df["lr_pair"].isin(selected)].copy()
    order = explicit + [lr for lr in top if lr in set(out["lr_pair"])]
    out["plot_order"] = out["lr_pair"].map({lr: i for i, lr in enumerate(order)})
    return out.sort_values("plot_order")


def spatial_activity_grid(cells: pd.DataFrame, bins: int = 180, sigma: float = 2.25) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)
    score = cells["incoming_score"].to_numpy(float)
    pad_x = (x.max() - x.min()) * 0.045
    pad_y = (y.max() - y.min()) * 0.045
    extent = (x.min() - pad_x, x.max() + pad_x, y.min() - pad_y, y.max() + pad_y)
    density, _, _ = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    weighted, _, _ = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]], weights=score)
    density_s = gaussian_filter(density.T, sigma=sigma)
    weighted_s = gaussian_filter(weighted.T, sigma=sigma)
    tissue = density_s / max(float(density_s.max()), 1e-9)
    tissue[tissue < 0.020] = 0
    return weighted_s, tissue, extent


def receiver_scores(stage: str, lr_pair: str, replicate: int = 1) -> pd.DataFrame:
    rep, cells, cand = load_slice(stage, replicate)
    pred = pd.read_csv(prediction_path(stage, rep))
    ligand, receptor = lr_pair.split("-")
    sub = cand[(cand["ligand"].eq(ligand)) & (cand["receptor"].eq(receptor))].copy()
    if sub.empty:
        cells = cells.copy()
        cells["incoming_score"] = 0.0
        return cells
    sub = sub.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    inc = sub.groupby("receiver_id", as_index=False)["score"].max().rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
    cells = cells.merge(inc, on="cell_id", how="left")
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    return cells


def aggregate_receiver_scores(stage: str, lr_pairs: list[str], replicate: int = 1) -> pd.DataFrame:
    base = None
    for lr in lr_pairs:
        cells = receiver_scores(stage, lr, replicate)
        if base is None:
            base = cells[["cell_id", "x", "y", "cell_type"]].copy()
            base["incoming_score"] = 0.0
        base = base.merge(cells[["cell_id", "incoming_score"]], on="cell_id", how="left", suffixes=("", "_lr"))
        base["incoming_score"] = base["incoming_score"] + base["incoming_score_lr"].fillna(0.0)
        base = base.drop(columns=["incoming_score_lr"])
    assert base is not None
    return base


def plot_volcano(ax: plt.Axes, volcano: pd.DataFrame) -> None:
    colors = volcano["mechanism"].map(MECH_COLORS).fillna("#BDB8AE")
    sizes = 10 + 46 * robust_norm(volcano["replicate_stability_score"].fillna(0.0))
    ax.scatter(volcano["effect_d9_d0"], volcano["evidence_plot"], s=sizes, c=colors, alpha=0.72, linewidths=0, zorder=2)
    ax.axvline(0, color="#222222", lw=0.55, ls=(0, (2, 2)))
    ax.axvline(0.25, color="#222222", lw=0.55, ls=(0, (2, 2)))
    ax.axvline(-0.25, color="#222222", lw=0.55, ls=(0, (2, 2)))
    ax.axhline(0.55, color="#222222", lw=0.55, ls=(0, (2, 2)))
    hit = volcano[volcano["highlight"]].copy()
    ax.scatter(hit["effect_d9_d0"], hit["evidence_plot"], s=62, facecolors=BG, edgecolors=RED, linewidths=0.85, zorder=4)
    label_offsets = {
        "Tgfb3-Acvrl1": (-0.030, 0.120, "right"),
        "Ccl5-Ccr5": (0.035, 0.045, "left"),
        "Cxcl10-Cxcr3": (0.035, 0.080, "left"),
        "Il1b-Il1r2": (0.030, 0.020, "left"),
    }
    for row in hit.sort_values("effect_d9_d0").itertuples(index=False):
        dx, dy, ha = label_offsets.get(row.lr_pair, (0.025, 0.035, "left"))
        ax.plot([row.effect_d9_d0, row.effect_d9_d0 + dx * 0.72], [row.evidence_plot, row.evidence_plot + dy * 0.72], color=MUTED, lw=0.30, alpha=0.65, zorder=3)
        ax.text(row.effect_d9_d0 + dx, row.evidence_plot + dy, row.lr_pair, fontsize=5.25, color=INK, ha=ha, va="center")
    ax.set_xlabel("DSS9 - D0 communication activity")
    ax.set_ylabel("effect-size evidence")
    ax.set_title("Differential signaling landscape", pad=4)
    clean_axis(ax, "y")
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=MECH_COLORS["long_range_chemokine_niche"], markersize=4.5, label="chemokine niche"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=MECH_COLORS["short_medium_paracrine"], markersize=4.5, label="paracrine"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, handletextpad=0.25, borderaxespad=0.2)


def plot_stage_heatmap(ax: plt.Axes, selected: pd.DataFrame) -> None:
    mat = selected[["Healthy_D0", "DSS3", "DSS9", "DSS21"]].astype(float).to_numpy()
    mat_norm = np.vstack([robust_norm(pd.Series(row)).to_numpy() for row in mat])
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=HEAT_CMAP(mat_norm[i, j]), edgecolor="#FFFFFF", linewidth=0.42))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(mat_norm.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["D0", "D3", "D9", "D21"])
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected["lr_pair"].tolist(), fontsize=5.2)
    for tick, cls in zip(ax.get_yticklabels(), selected["trajectory_class"]):
        tick.set_color(CLASS_COLORS.get(cls, TEXT))
    ax.tick_params(length=0, pad=1.6)
    ax.set_title("Stage-wise LR programs", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.60, -0.16, 0.34, 0.045])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=HEAT_CMAP, aspect="auto")
    cax.set_xticks([0, 127]); cax.set_xticklabels(["low", "high"]); cax.set_yticks([])
    cax.tick_params(length=0, labelsize=5.0, pad=0.8)
    for s in cax.spines.values():
        s.set_visible(False)


def plot_replicate_spatial_grid(fig: plt.Figure, outer_ax: plt.Axes, source_rows: list[pd.DataFrame]) -> None:
    outer_ax.set_axis_off()
    outer_ax.set_title("Replicated inflammatory receiver activity", pad=5)
    lr_pairs = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Il1b-Il1r2"]
    stage = "DSS9"
    cells_list = []
    for rep in [1, 2]:
        cells = aggregate_receiver_scores(stage, lr_pairs, replicate=rep)
        cells["replicate"] = rep
        cells_list.append(cells)
    all_scores = pd.concat([c["incoming_score"] for c in cells_list], ignore_index=True)
    vmax = float(all_scores[all_scores > 0].quantile(0.985)) if (all_scores > 0).any() else 1.0
    vmax = max(vmax, 1e-6)
    for i, (rep, cells) in enumerate(zip([1, 2], cells_list)):
        ax = outer_ax.inset_axes([0.028 + i * 0.49, 0.13, 0.44, 0.74])
        cells = cells.copy()
        cells["activity_norm"] = np.clip(cells["incoming_score"].astype(float) / vmax, 0, 1)
        grid, _, extent = spatial_activity_grid(cells, bins=180, sigma=1.55)
        grid_norm = np.clip(grid / vmax, 0, 1)
        alpha = np.clip(grid_norm ** 0.55, 0, 0.68)
        alpha[grid_norm < 0.010] = 0
        ax.imshow(grid_norm, origin="lower", extent=extent, cmap=ACTIVITY_CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear", zorder=1)
        ax.scatter(cells["x"], cells["y"], s=1.08, color="#D8D3CC", alpha=0.22, linewidths=0, rasterized=True, zorder=2)
        active = cells[cells["activity_norm"].gt(0)].sort_values("activity_norm")
        ax.scatter(active["x"], active["y"], s=2.05, c=active["activity_norm"], cmap=ACTIVITY_CMAP, norm=Normalize(0, 1), alpha=0.88, edgecolors=BG, linewidths=0.05, rasterized=True, zorder=3)
        set_spatial_limits(ax, cells)
        ax.set_title(f"DSS9 rep {rep}", fontsize=6.2, pad=1.4)
        source_rows.append(cells)
    cax = outer_ax.inset_axes([0.70, 0.035, 0.23, 0.035])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=ACTIVITY_CMAP, aspect="auto")
    cax.set_xticks([0, 127]); cax.set_xticklabels(["low", "high"]); cax.set_yticks([])
    cax.tick_params(length=0, labelsize=5.0, pad=0.8)
    for s in cax.spines.values():
        s.set_visible(False)


def plot_flow_shift(ax: plt.Axes, flow: pd.DataFrame) -> None:
    df = flow.copy()
    pivot = df.pivot_table(index="flow", columns="stage_label", values="stage_fraction", fill_value=0.0)
    for col in ["D0", "D3", "D9", "D21"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["D9_minus_D0"] = pivot["D9"] - pivot["D0"]
    top = pivot.reindex(pivot["D9_minus_D0"].abs().sort_values(ascending=False).head(9).index)
    vals = top["D9_minus_D0"].sort_values()
    colors = [RED if v > 0 else BLUE for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.58, edgecolor="none", alpha=0.86)
    ax.axvline(0, color=INK, lw=0.6)
    abbrev = {"Epithelial": "Epi", "Immune": "Imm", "Stromal": "Str", "Mesenchymal": "Mes", "Vascular": "Vas", "Neural": "Neu"}
    def short_flow(label: str) -> str:
        left, right = label.split(" -> ")
        return f"{abbrev.get(left, left[:3])} to {abbrev.get(right, right[:3])}"
    ax.set_yticks(y)
    ax.set_yticklabels([short_flow(label) for label in vals.index], fontsize=5.1)
    ax.set_xlabel("stage fraction shift (DSS9 - D0)")
    ax.set_title("Cell-compartment flow rewiring", pad=4)
    clean_axis(ax, "x")


def plot_top_stage_bubbles(ax: plt.Axes, traj: pd.DataFrame) -> None:
    df = traj.copy()
    top = df.sort_values("story_score", ascending=False).head(55).copy()
    counts = top.groupby(["peak_stage", "biological_category"], as_index=False).size().rename(columns={"size": "n_lr"})
    stage_labels = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
    cats = [c for c in ["Chemokine", "IL", "TGFb", "Wnt", "TNF"] if c in counts["biological_category"].unique()]
    x_map = {s: i for i, s in enumerate(stage_labels)}
    y_map = {c: i for i, c in enumerate(cats)}
    for row in counts.itertuples(index=False):
        if row.peak_stage not in x_map or row.biological_category not in y_map:
            continue
        color = STAGE_COLORS[STAGE_LABEL_MAP[row.peak_stage]]
        ax.scatter(x_map[row.peak_stage], y_map[row.biological_category], s=18 + 22 * row.n_lr, color=color, alpha=0.76, edgecolor=INK, linewidth=0.35)
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels([STAGE_LABEL_MAP[s] for s in stage_labels])
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlim(-0.55, len(stage_labels) - 0.45)
    ax.set_ylim(len(cats) - 0.45, -0.55)
    ax.set_title("Stage-specific LR families", pad=4)
    ax.grid(color=GRID, lw=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=1.4)


def plot_trajectory_lines(ax: plt.Axes, selected: pd.DataFrame) -> None:
    focused = ["Rspo3-Lgr5", "Il5-Il5ra", "Cxcl10-Cxcr3", "Tgfb3-Acvrl1"]
    plot_lrs = [lr for lr in focused if lr in set(selected["lr_pair"])]
    df = selected[selected["lr_pair"].isin(plot_lrs)].copy()
    x = np.arange(4)
    label_pos = {
        "Cxcl10-Cxcr3": (2.12, 1.000, "left"),
        "Tgfb3-Acvrl1": (3.10, 0.965, "left"),
        "Rspo3-Lgr5": (3.08, 0.270, "left"),
        "Il5-Il5ra": (3.08, 0.060, "left"),
    }
    for row in df.itertuples(index=False):
        vals = np.array([row.Healthy_D0, row.DSS3, row.DSS9, row.DSS21], dtype=float)
        color = CLASS_COLORS.get(row.trajectory_class, TEXT)
        alpha = 1.0 if row.lr_pair in {"Cxcl10-Cxcr3", "Tgfb3-Acvrl1"} else 0.78
        lw = 1.22 if row.lr_pair in {"Cxcl10-Cxcr3", "Tgfb3-Acvrl1"} else 1.02
        ax.plot(x, vals, color=color, lw=lw, marker="o", ms=2.7, alpha=alpha, mec=BG, mew=0.35)
        lx, ly, ha = label_pos.get(row.lr_pair, (3.08, vals[-1], "left"))
        ax.text(lx, ly, row.lr_pair, fontsize=4.95, va="center", ha=ha, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["D0", "D3", "D9", "D21"])
    ax.set_xlim(-0.15, 4.25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("LR activity")
    ax.set_title("Representative trajectories", pad=4)
    clean_axis(ax, "y")


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(7.55, 5.10), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        6,
        left=0.075,
        right=0.985,
        bottom=0.110,
        top=0.925,
        width_ratios=[1.05, 1.05, 1.00, 1.00, 1.05, 1.05],
        height_ratios=[1.05, 1.00],
        wspace=0.55,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    plot_volcano(ax_a, source["volcano"])
    plot_stage_heatmap(ax_b, source["heatmap_lrs"])
    plot_trajectory_lines(ax_c, source["heatmap_lrs"])
    spatial_sources: list[pd.DataFrame] = []
    plot_replicate_spatial_grid(fig, ax_d, spatial_sources)
    if spatial_sources:
        pd.concat(spatial_sources, ignore_index=True).to_csv(SOURCE / "panel_d_replicate_spatial_activity.csv", index=False)
    plot_flow_shift(ax_e, source["flow"])

    for label, ax in zip(list("abcde"), [ax_a, ax_b, ax_c, ax_d, ax_e]):
        panel_label(fig, ax, label)
    return fig


def plot_mechanism_counts(ax: plt.Axes, traj: pd.DataFrame) -> None:
    df = traj.copy()
    top = df.sort_values("story_score", ascending=False).head(70)
    counts = top.groupby(["peak_stage", "mechanism"], as_index=False).size().rename(columns={"size": "n_lr"})
    pivot = counts.pivot_table(index="mechanism", columns="peak_stage", values="n_lr", fill_value=0)
    pivot = pivot.reindex(index=[m for m in MECH_COLORS if m in pivot.index], columns=STAGE_ORDER, fill_value=0)
    x = np.arange(len(STAGE_ORDER))
    bottom = np.zeros(len(STAGE_ORDER))
    for mech in pivot.index:
        vals = pivot.loc[mech].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=MECH_COLORS.get(mech, "#BDB8AE"), width=0.62, edgecolor=BG, linewidth=0.35, label=mech.replace("_", " "))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL_MAP[s] for s in STAGE_ORDER])
    ax.set_ylabel("top LR count")
    ax.set_title("Mechanism composition", pad=4)
    clean_axis(ax, "y")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=4.9, handlelength=0.9, handletextpad=0.25)


def plot_replicate_support(ax: plt.Axes, traj: pd.DataFrame) -> None:
    df = traj.copy()
    order = ["flat_or_decreased", "early_induction", "inflammation_peak_recovery", "persistent_inflammation", "late_repair_or_persistent"]
    data = [df.loc[df["trajectory_class"].eq(cls), "replicate_stability_score"].dropna().to_numpy(float) for cls in order]
    x = np.arange(len(order))
    parts = ax.violinplot(data, positions=x, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, cls in zip(parts["bodies"], order):
        body.set_facecolor(CLASS_COLORS.get(cls, "#BDB8AE"))
        body.set_edgecolor("none")
        body.set_alpha(0.62)
    med = [np.nanmedian(v) if len(v) else np.nan for v in data]
    ax.scatter(x, med, s=15, color=INK, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["D0", "early", "D9", "persist", "D21"], rotation=25, ha="right")
    ax.set_ylabel("replicate stability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Replicate support", pad=4)
    clean_axis(ax, "y")


def build_source() -> dict[str, pd.DataFrame]:
    traj = pd.read_csv(TRAJ)
    volcano = build_volcano(traj)
    heatmap_lrs = select_heatmap_lrs(traj)
    flow = pd.read_csv(FLOW)
    source = {"trajectories": traj, "volcano": volcano, "heatmap_lrs": heatmap_lrs, "flow": flow}
    return source


def export(fig: plt.Figure, stem: Path, dpi: int = 850) -> None:
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    for stale in SOURCE.glob("*.csv"):
        stale.unlink()
    setup_style()
    source = build_source()
    for name, table in source.items():
        table.to_csv(SOURCE / f"{name}.csv", index=False)
    fig = plot_figure(source)
    export(fig, OUT / "dss_merfish_article_style_additions")
    (OUT / "README.md").write_text(
        "DSS MERFISH article-style evidence additions. Panels are inspired by CytoSignal article_figures Fig5d/e/g but use ScaleComm source data. "
        "The volcano-like panel uses effect-size evidence from existing trajectory summaries, not formal FDR.\n",
        encoding="utf-8",
    )
    print(OUT / "dss_merfish_article_style_additions.pdf")


if __name__ == "__main__":
    main()
