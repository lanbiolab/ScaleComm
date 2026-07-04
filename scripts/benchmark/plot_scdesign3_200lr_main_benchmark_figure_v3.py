#!/usr/bin/env python
"""Publication-style main benchmark figure for scDesign3 200-LR results."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
METRIC_DIR = BENCH / "extended_metrics"
DATASET_DIR = BENCH / "datasets"
OUT = BENCH / "main_figure_v3"
SOURCE = OUT / "source_data"

METHOD_ORDER = [
    "ScaleComm",
    "COMMOT",
    "CellPhoneDB",
    "SpaOTsc",
    "CellChat",
    "NICHES",
    "SpaTalk",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
]

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = ["Contact", "Short", "Medium", "Long niche"]

TEXT = "#252321"
MUTED = "#6F6861"
GRID = "#E8E2D6"
PALE = "#F8F5EE"
PAPER = "#FFFFFF"
RED = "#B85450"
RED_DARK = "#77312E"
BLUE = "#2F78A8"
TEAL = "#237A78"
GOLD = "#E6C06A"
AMBER = "#D9992B"
NEUTRAL = "#A49C93"
LIGHT_NEUTRAL = "#D8D0C7"

METHOD_COLORS = {
    "ScaleComm": RED,
    "COMMOT": BLUE,
    "CellPhoneDB": AMBER,
    "SpaOTsc": TEAL,
    "CellChat": "#83A95C",
    "NICHES": "#8E78AA",
    "SpaTalk": "#B69B45",
    "CellNEST": "#8A8177",
    "DeepTalk": "#C47C9E",
    "CytoSignal": "#768DB8",
}

CT_COLORS = {
    "B": BLUE,
    "T": RED,
    "Other": NEUTRAL,
    "DC": TEAL,
    "Macrophage": AMBER,
    "stromal": GOLD,
}

GOOD_CMAP = LinearSegmentedColormap.from_list("good_scale", [PALE, "#E8BF78", RED])
BAD_CMAP = LinearSegmentedColormap.from_list("bad_scale", [PALE, "#F1B87B", RED_DARK])
OVERLAP_CMAP = LinearSegmentedColormap.from_list("overlap_scale", [PALE, "#B9E5FA", BLUE])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    setup_style()
    method_metrics, summary, mechanism, overlap, cells = load_data()
    write_source(method_metrics, summary, mechanism, overlap)
    fig = plot_figure(method_metrics, summary, mechanism, overlap, cells)
    export(fig, OUT / "scdesign3_200lr_main_benchmark_figure_v3")
    print(f"Wrote {OUT / 'scdesign3_200lr_main_benchmark_figure_v3.pdf'}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "axes.edgecolor": TEXT,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    method_metrics = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_method_metrics.csv")
    summary = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_method_summary.csv")
    mechanism = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_by_mechanism.csv")
    overlap = pd.read_csv(METRIC_DIR / "scdesign3_200lr_method_overlap_pairwise.csv")
    cells = {
        "LN": pd.read_csv(DATASET_DIR / "reference_lymph_node_scdesign3_200lr_replicate_00/cells.csv"),
        "Melanoma": pd.read_csv(DATASET_DIR / "reference_melanoma_scdesign3_200lr_replicate_00/cells.csv"),
    }
    return method_metrics, summary, mechanism, overlap, cells


def write_source(method_metrics: pd.DataFrame, summary: pd.DataFrame, mechanism: pd.DataFrame, overlap: pd.DataFrame) -> None:
    method_metrics.to_csv(SOURCE / "method_metrics.csv", index=False)
    summary.to_csv(SOURCE / "method_summary.csv", index=False)
    mechanism.to_csv(SOURCE / "mechanism_metrics.csv", index=False)
    overlap.to_csv(SOURCE / "method_overlap_pairwise.csv", index=False)


def plot_figure(
    method_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    overlap: pd.DataFrame,
    cells: dict[str, pd.DataFrame],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 10.85), constrained_layout=False)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.68, 1.62, 1.34, 1.30, 1.34],
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=0.960,
        hspace=0.58,
    )
    panel_a(fig, outer[0, 0], cells)
    panel_b_standard_metrics(fig, outer[1, 0], method_metrics)
    row3 = outer[2, 0].subgridspec(1, 2, width_ratios=[1.02, 1.08], wspace=0.34)
    panel_c_ranking_heatmap(fig, fig.add_subplot(row3[0, 0]), summary)
    panel_d_mechanism_heatmap(fig, fig.add_subplot(row3[0, 1]), mechanism)
    row4 = outer[3, 0].subgridspec(1, 2, width_ratios=[1.08, 1.00], wspace=0.27)
    panel_e_noise(fig, fig.add_subplot(row4[0, 0]), method_metrics)
    panel_f_detection(fig, fig.add_subplot(row4[0, 1]), method_metrics, summary)
    row5 = outer[4, 0].subgridspec(1, 2, width_ratios=[1.04, 1.02], wspace=0.27)
    panel_g_overlap(fig, fig.add_subplot(row5[0, 0]), overlap)
    panel_h_error_modes(fig, fig.add_subplot(row5[0, 1]), mechanism)
    return fig


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.026, dy: float = 0.010) -> None:
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, label, ha="left", va="bottom", fontsize=12.0, fontweight="bold", color="#111111")


def panel_a(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, cells: dict[str, pd.DataFrame]) -> None:
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    panel_label(fig, ax, "a", dx=-0.006, dy=0.002)
    ax.text(0.50, 0.995, "Reference-based LR range benchmark", ha="center", va="top", fontsize=10.8, color=TEXT, transform=ax.transAxes)
    sub = spec.subgridspec(1, 5, wspace=0.08)
    cards = [
        fig.add_subplot(sub[0, 0]),
        fig.add_subplot(sub[0, 1]),
        fig.add_subplot(sub[0, 2]),
        fig.add_subplot(sub[0, 3]),
        fig.add_subplot(sub[0, 4]),
    ]
    titles = ["Spatial references", "Reference expression", "LR range truth", "Stress tests", "Unified readout"]
    subtitles = ["LN + melanoma", "scDesign3-like counts", "200 LR pairs", "noise + wrong range", "exact top 5%"]
    drawers = [
        lambda a: draw_reference_card(a, cells),
        draw_expression_card,
        draw_range_card,
        draw_stress_card,
        draw_readout_card,
    ]
    for card, title, subtitle, drawer in zip(cards, titles, subtitles, drawers, strict=True):
        style_card(card)
        card.text(0.5, 0.980, title, ha="center", va="top", fontsize=7.15, color=TEXT, transform=card.transAxes)
        card.text(0.5, 0.875, subtitle, ha="center", va="top", fontsize=5.20, color=MUTED, transform=card.transAxes)
        drawer(card)

    for left, right in zip(cards[:-1], cards[1:], strict=True):
        bb1, bb2 = left.get_position(), right.get_position()
        y = (bb1.y0 + bb1.y1) / 2
        arr = FancyArrowPatch(
            (bb1.x1 + 0.005, y),
            (bb2.x0 - 0.005, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10.5,
            lw=0.9,
            color="#8C8780",
        )
        fig.patches.append(arr)


def style_card(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.075),
            0.95,
            0.78,
            boxstyle="round,pad=0.012,rounding_size=0.030",
            fc="#FBFAF7",
            ec="#D8D0C7",
            lw=0.65,
            zorder=0,
        )
    )


def draw_reference_card(ax: plt.Axes, cells: dict[str, pd.DataFrame]) -> None:
    for i, key in enumerate(["LN", "Melanoma"]):
        data = cells[key].copy()
        data = data.sample(min(360, len(data)), random_state=3 if key == "LN" else 5)
        yoff = 0.60 if i == 0 else 0.30
        x = 0.13 + 0.74 * norm01(data["x"].to_numpy(float))
        y = yoff - 0.105 + 0.21 * norm01(data["y"].to_numpy(float))
        colors = [CT_COLORS.get(str(t), NEUTRAL) for t in data["cell_type"]]
        ax.scatter(x, y, s=2.2, c=colors, alpha=0.82, lw=0)
        ax.text(0.07, yoff, key if key == "LN" else "MEL", ha="left", va="center", fontsize=5.2, color=MUTED)
        ax.add_patch(Rectangle((0.12, yoff - 0.125), 0.76, 0.25, fill=False, ec="#DFD8CE", lw=0.55))
    ax.text(0.50, 0.145, "real coordinates\ncell composition retained", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_expression_card(ax: plt.Axes) -> None:
    rng = np.random.default_rng(2)
    mat = rng.beta(1.2, 2.5, size=(10, 13))
    x0, y0 = 0.17, 0.43
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(Rectangle((x0 + j * 0.036, y0 + i * 0.026), 0.029, 0.020, fc=GOOD_CMAP(mat[i, j]), ec="white", lw=0.18))
    for x, y, c in [(0.29, 0.72, RED), (0.52, 0.68, BLUE), (0.47, 0.32, TEAL)]:
        ax.scatter([x], [y], s=145, c=c, alpha=0.38, edgecolor="#222222", lw=0.75)
        ax.scatter([x + 0.018], [y + 0.010], s=34, c=c, edgecolor="#222222", lw=0.45)
    ax.text(0.50, 0.145, "library size, sparsity\nand gene variance matched", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_range_card(ax: plt.Axes) -> None:
    yvals = [0.69, 0.56, 0.43, 0.30]
    labels = ["contact", "short", "medium", "long"]
    cols = [RED, AMBER, BLUE, TEAL]
    lengths = [0.11, 0.20, 0.32, 0.48]
    for y, lab, col, dist in zip(yvals, labels, cols, lengths, strict=True):
        ax.scatter([0.22], [y], s=125, c=col, alpha=0.40, edgecolor="#222222", lw=0.65)
        ax.scatter([0.22 + dist], [y], s=70, c=col, alpha=0.40, edgecolor="#222222", lw=0.55)
        ax.plot([0.25, 0.20 + dist], [y, y], color=col, lw=1.0, alpha=0.80)
        ax.text(0.78, y, lab, ha="left", va="center", fontsize=5.0, color=MUTED)
    ax.text(0.50, 0.145, "true positives constrained\nby LR-specific range", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_stress_card(ax: plt.Axes) -> None:
    rng = np.random.default_rng(8)
    x = 0.18 + 0.64 * rng.random(55)
    y = 0.30 + 0.42 * rng.random(55)
    c = np.where(np.arange(55) % 4 == 0, RED, np.where(np.arange(55) % 4 == 1, BLUE, LIGHT_NEUTRAL))
    ax.scatter(x, y, s=np.where(np.arange(55) % 6 == 0, 18, 7), c=c, alpha=0.75, lw=0)
    ax.plot([0.18, 0.82], [0.29, 0.73], color=RED_DARK, lw=1.0, alpha=0.72)
    ax.plot([0.18, 0.82], [0.73, 0.29], color=RED_DARK, lw=1.0, alpha=0.72)
    ax.text(0.50, 0.145, "dropout, expression decoys\nand wrong-range negatives", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_readout_card(ax: plt.Axes) -> None:
    items = [("F1", RED, 0.82), ("AUPR", BLUE, 0.78), ("FDR", AMBER, 0.55), ("range", TEAL, 0.74)]
    for i, (lab, col, width) in enumerate(items):
        y = 0.68 - i * 0.115
        ax.text(0.20, y, lab, ha="left", va="center", fontsize=6.0, color=TEXT)
        ax.add_patch(Rectangle((0.47, y - 0.018), 0.34, 0.036, fc="#E9E3DB", ec="none"))
        ax.add_patch(Rectangle((0.47, y - 0.018), 0.34 * width, 0.036, fc=col, ec="none", alpha=0.88))
    ax.text(0.50, 0.145, "all methods select\nexact top 5% candidates", ha="center", va="center", fontsize=5.1, color=MUTED)


def panel_b_standard_metrics(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, method_metrics: pd.DataFrame) -> None:
    grid = spec.subgridspec(2, 1, height_ratios=[0.28, 0.72], hspace=0.02)
    holder = fig.add_subplot(grid[0, 0])
    holder.set_axis_off()
    panel_label(fig, holder, "b")
    holder.text(0.0, 0.58, "Standard benchmark metrics at a fixed detection burden", ha="left", va="center", fontsize=7.6, color=TEXT, transform=holder.transAxes)
    sub = grid[1, 0].subgridspec(1, 4, wspace=0.09)
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity")]
    for i, (col, title) in enumerate(metrics):
        ax = fig.add_subplot(sub[0, i])
        metric_dot_panel(ax, method_metrics, col, title, show_y=(i == 0))


def metric_dot_panel(ax: plt.Axes, data: pd.DataFrame, col: str, title: str, show_y: bool) -> None:
    y = np.arange(len(METHOD_ORDER))
    for yi, method in enumerate(METHOD_ORDER):
        vals = data[data["method_label"].eq(method)][col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.125, 0.125, vals.size)
        color = METHOD_COLORS[method] if method == "ScaleComm" else "#A39B92"
        ax.scatter(vals, yi + jitter, s=9.5, c=color, alpha=0.70, edgecolor="white", lw=0.25, zorder=2)
        ax.scatter([vals.mean()], [yi], s=42 if method == "ScaleComm" else 24, c=color, edgecolor="#222222" if method == "ScaleComm" else "white", lw=0.45, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=6.6, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.50)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(METHOD_ORDER)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "ScaleComm":
                lab.set_color(RED)
                lab.set_fontweight("bold")
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, pad=1, colors=MUTED)
    despine(ax)


def panel_c_ranking_heatmap(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_label(fig, ax, "c")
    table = summary.set_index("method_label").reindex(METHOD_ORDER)
    rows = [
        ("aupr_mean", "AUPR", True),
        ("auroc_mean", "AUROC", True),
        ("top_k_precision_mean", "Top-k precision", True),
        ("empirical_fdr_mean", "1 - FDR", False),
        ("wrong_range_rate_mean", "1 - wrong-range FP", False),
    ]
    mat = []
    for col, _label, higher in rows:
        vals = table[col].to_numpy(float)
        mat.append(vals if higher else 1 - vals)
    mat = np.vstack(mat)
    im = ax.imshow(mat, cmap=GOOD_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.2)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=5.6)
    highlight_scalecomm(ax, axis="x")
    ax.add_patch(Rectangle((-0.5, -0.5), 1, len(rows), fill=False, ec=RED, lw=1.0))
    ax.set_title("Ranking accuracy and false-positive suppression", loc="left", pad=5, fontsize=7.3)
    heatmap_grid(ax, len(rows), len(METHOD_ORDER))
    cbar(ax, im, "higher is better", x=1.018)


def panel_d_mechanism_heatmap(fig: plt.Figure, ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    panel_label(fig, ax, "d")
    sub = mechanism[mechanism["range_class"].isin(RANGE_ORDER)].copy()
    mean = (
        sub.groupby(["method_label", "range_class"], observed=True)["f1"]
        .mean()
        .reset_index()
        .pivot(index="method_label", columns="range_class", values="f1")
        .reindex(index=METHOD_ORDER, columns=RANGE_ORDER)
    )
    im = ax.imshow(mean.to_numpy(float), cmap=GOOD_CMAP, vmin=0, vmax=np.nanquantile(mean.to_numpy(float), 0.98), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels(RANGE_LABELS, fontsize=5.6)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.6)
    highlight_scalecomm(ax, axis="y")
    ax.add_patch(Rectangle((-0.5, -0.5), len(RANGE_ORDER), 1, fill=False, ec=RED, lw=1.0))
    ax.set_title("Recovery across LR range classes", loc="left", pad=5, fontsize=7.3)
    heatmap_grid(ax, len(METHOD_ORDER), len(RANGE_ORDER))
    cbar(ax, im, "F1", x=1.020)


def panel_e_noise(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "e")
    noise_order = ["clean", "low", "high"]
    mean = method_metrics.groupby(["method_label", "noise"], observed=True)["aupr"].mean().reset_index()
    x = np.arange(len(noise_order))
    for method in METHOD_ORDER:
        vals = mean[mean["method_label"].eq(method)].set_index("noise").reindex(noise_order)["aupr"].to_numpy(float)
        if method == "ScaleComm":
            ax.plot(x, vals, color=RED, lw=1.75, marker="o", ms=3.8, mec="white", mew=0.55, zorder=4)
        elif method in {"COMMOT", "CellPhoneDB", "SpaOTsc"}:
            ax.plot(x, vals, color=METHOD_COLORS[method], lw=1.05, marker="o", ms=2.8, mec="white", mew=0.35, alpha=0.86, zorder=3)
        else:
            ax.plot(x, vals, color="#B7B1AA", lw=0.75, alpha=0.62, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Low", "High"])
    ax.set_ylabel("AUPR")
    ax.set_title("Robustness under expression noise", loc="left", pad=5, fontsize=7.3)
    ax.grid(axis="y", color=GRID, lw=0.50)
    despine(ax)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", lw=1.3, ms=3, label=m)
        for m in ["ScaleComm", "COMMOT", "CellPhoneDB", "SpaOTsc"]
    ]
    handles.append(Line2D([0], [0], color="#B7B1AA", lw=1.0, label="other baselines"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.01, 1.05), ncol=2, handlelength=1.5, columnspacing=0.8)


def panel_f_detection(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    panel_label(fig, ax, "f")
    frac = method_metrics.groupby("method_label", observed=True)["detected_candidate_fraction"].mean().reindex(METHOD_ORDER) * 100
    lr_count = summary.set_index("method_label").reindex(METHOD_ORDER)["detected_lr_pair_count_mean"]
    y = np.arange(len(METHOD_ORDER))
    ax.barh(y, frac, height=0.42, color="#D8D0C7", edgecolor="none", label="candidate fraction")
    ax.set_xlim(0, 5.8)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.4)
    highlight_scalecomm(ax, axis="y")
    ax.invert_yaxis()
    ax.set_xlabel("detected candidates (%)")
    ax.set_title("Exact top 5% selection", loc="left", pad=5, fontsize=7.3)
    ax.grid(axis="x", color=GRID, lw=0.48)
    ax2 = ax.twiny()
    colors = [RED if m == "ScaleComm" else "#918A82" for m in METHOD_ORDER]
    ax2.scatter(lr_count, y, s=22, c=colors, zorder=3, edgecolor="white", lw=0.4)
    ax2.set_xlim(0, max(205, float(lr_count.max()) * 1.08))
    ax2.set_xlabel("detected LR pairs", color=MUTED)
    ax2.tick_params(axis="x", length=2, pad=1, labelsize=5.2, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    despine(ax)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, fc="#D8D0C7", ec="none", label="fixed candidate burden"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=4.5, label="LR-pair coverage"),
    ], loc="lower right", bbox_to_anchor=(1.00, -0.03))


def panel_g_overlap(fig: plt.Figure, ax: plt.Axes, overlap: pd.DataFrame) -> None:
    panel_label(fig, ax, "g")
    mat = overlap_matrix(overlap, level="lr_pair")
    im = ax.imshow(mat.to_numpy(float), cmap=OVERLAP_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.0)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.0)
    highlight_scalecomm(ax, axis="both")
    ax.set_title("Overlap of detected LR pairs", loc="left", pad=5, fontsize=7.3)
    heatmap_grid(ax, len(METHOD_ORDER), len(METHOD_ORDER))
    cbar(ax, im, "Jaccard", x=1.018)


def panel_h_error_modes(fig: plt.Figure, ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    panel_label(fig, ax, "h")
    sub = mechanism[mechanism["range_class"].isin(RANGE_ORDER)].copy()
    mean = (
        sub.groupby(["method_label", "range_class"], observed=True)["wrong_range_rate"]
        .mean()
        .reset_index()
        .pivot(index="method_label", columns="range_class", values="wrong_range_rate")
        .reindex(index=METHOD_ORDER, columns=RANGE_ORDER)
    )
    vmax = float(np.nanquantile(mean.to_numpy(float), 0.96))
    im = ax.imshow(mean.to_numpy(float), cmap=BAD_CMAP, vmin=0, vmax=max(vmax, 0.01), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels(RANGE_LABELS, fontsize=5.6)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.6)
    highlight_scalecomm(ax, axis="y")
    ax.add_patch(Rectangle((-0.5, -0.5), len(RANGE_ORDER), 1, fill=False, ec=RED, lw=1.0))
    ax.set_title("Wrong-range false-positive modes", loc="left", pad=5, fontsize=7.3)
    heatmap_grid(ax, len(METHOD_ORDER), len(RANGE_ORDER))
    cbar(ax, im, "FP rate", x=1.020)


def overlap_matrix(overlap: pd.DataFrame, level: str) -> pd.DataFrame:
    sub = overlap[overlap["overlap_level"].eq(level)]
    key_to_label = {
        "SpatialSelf": "ScaleComm",
        "COMMOT": "COMMOT",
        "CellPhoneDB": "CellPhoneDB",
        "SpaOTsc": "SpaOTsc",
        "CellChat": "CellChat",
        "NICHES": "NICHES",
        "SpaTalk": "SpaTalk",
        "CellNEST": "CellNEST",
        "DeepTalk": "DeepTalk",
        "CytoSignal": "CytoSignal",
    }
    mat = pd.DataFrame(np.nan, index=METHOD_ORDER, columns=METHOD_ORDER)
    mean = sub.groupby(["method_a", "method_b"], sort=False)["jaccard"].mean().reset_index()
    for _, row in mean.iterrows():
        a = key_to_label.get(str(row["method_a"]), str(row["method_a"]))
        b = key_to_label.get(str(row["method_b"]), str(row["method_b"]))
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = float(row["jaccard"])
            mat.loc[b, a] = float(row["jaccard"])
    return mat


def heatmap_grid(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.70)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def cbar(ax: plt.Axes, im: mpl.image.AxesImage, label: str, x: float = 1.02, inside: bool = False) -> None:
    if inside:
        cax = ax.inset_axes([x, 0.18, 0.018, 0.58])
    else:
        cax = ax.inset_axes([x, 0.18, 0.022, 0.58])
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.0, colors=MUTED, pad=1)
    cb.set_label(label, fontsize=5.0, color=MUTED, rotation=270, labelpad=8)


def highlight_scalecomm(ax: plt.Axes, axis: str) -> None:
    if axis in {"y", "both"}:
        for lab in ax.get_yticklabels():
            if lab.get_text() == "ScaleComm":
                lab.set_color(RED)
                lab.set_fontweight("bold")
    if axis in {"x", "both"}:
        for lab in ax.get_xticklabels():
            if lab.get_text() == "ScaleComm":
                lab.set_color(RED)
                lab.set_fontweight("bold")


def norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.50)
    ax.spines["bottom"].set_linewidth(0.50)
    ax.spines["left"].set_color("#5C5752")
    ax.spines["bottom"].set_color("#5C5752")


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
