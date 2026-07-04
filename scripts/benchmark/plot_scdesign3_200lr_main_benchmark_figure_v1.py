#!/usr/bin/env python
"""Main benchmark figure for the scDesign3 200-LR mechanism benchmark."""

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
METRIC_DIR = BENCH / "extended_metrics"
OUT = BENCH / "main_figure_v2"
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

METHOD_KEY = {
    "ScaleComm": "SpatialSelf",
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

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}

TEXT = "#292624"
MUTED = "#6E665F"
GRID = "#E8E2D6"
NEUTRAL = "#B9C1D1"
NEUTRAL_DARK = "#746E68"
RED = "#B85450"
RED_DARK = "#793330"
BLUE = "#2F78A8"
BLUE_DARK = "#1F5677"
GOLD = "#E6C06A"
AMBER = "#D9992B"
TEAL = "#237A78"
GREEN = "#88B378"
PALE = "#F7F4EC"
FIG_BG = "#FFFFFF"

METHOD_COLORS = {
    "ScaleComm": RED,
    "COMMOT": BLUE,
    "CellPhoneDB": AMBER,
    "SpaOTsc": TEAL,
    "CellChat": GREEN,
    "NICHES": "#8C6FAE",
    "SpaTalk": "#B39B45",
    "CellNEST": "#8A8177",
    "DeepTalk": "#C7779C",
    "CytoSignal": "#7089B5",
}

RANGE_COLORS = {
    "contact": RED,
    "short": AMBER,
    "medium": BLUE,
    "long_niche": TEAL,
}

HEAT = LinearSegmentedColormap.from_list("scalecomm_benchmark_heat", [PALE, "#E8BE76", RED])
BLUE_HEAT = LinearSegmentedColormap.from_list("scalecomm_overlap", [PALE, "#B9E5FA", BLUE])
DIVERGE = LinearSegmentedColormap.from_list("scalecomm_diverge", [GOLD, PALE, BLUE])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    setup_style()
    method_metrics, summary, mechanism, overlap = load_data()
    source_tables(method_metrics, summary, mechanism, overlap)
    fig = plot_figure(method_metrics, summary, mechanism, overlap)
    export(fig, OUT / "scdesign3_200lr_main_benchmark_figure_v2")
    print(f"Wrote {OUT / 'scdesign3_200lr_main_benchmark_figure_v2.pdf'}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "axes.titlesize": 7.3,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.55,
            "axes.edgecolor": TEXT,
            "axes.facecolor": FIG_BG,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_metrics = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_method_metrics.csv")
    summary = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_method_summary.csv")
    mechanism = pd.read_csv(METRIC_DIR / "scdesign3_200lr_extended_by_mechanism.csv")
    overlap = pd.read_csv(METRIC_DIR / "scdesign3_200lr_method_overlap_pairwise.csv")
    for df in (method_metrics, summary, mechanism):
        df["method_label"] = pd.Categorical(df["method_label"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method_label").copy()
    return method_metrics, summary, mechanism, overlap


def source_tables(
    method_metrics: pd.DataFrame, summary: pd.DataFrame, mechanism: pd.DataFrame, overlap: pd.DataFrame
) -> None:
    method_metrics.to_csv(SOURCE / "panel_cdfg_method_metrics.csv", index=False)
    summary.to_csv(SOURCE / "panel_b_summary_metrics.csv", index=False)
    mechanism.to_csv(SOURCE / "panel_e_mechanism_metrics.csv", index=False)
    overlap.to_csv(SOURCE / "panel_h_overlap_pairwise.csv", index=False)


def plot_figure(
    method_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    overlap: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 10.65), constrained_layout=False)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.78, 1.30, 1.42, 1.50, 1.24],
        left=0.050,
        right=0.985,
        bottom=0.045,
        top=0.965,
        hspace=0.62,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    panel_a_schematic(fig, ax_a)

    row2 = outer[1, 0].subgridspec(1, 2, width_ratios=[0.92, 1.55], wspace=0.28)
    ax_b = fig.add_subplot(row2[0, 0], projection="polar")
    ax_c = fig.add_subplot(row2[0, 1])
    panel_b_radar(fig, ax_b, summary, method_metrics)
    panel_c_standard_metrics(fig, ax_c, method_metrics)

    row3 = outer[2, 0].subgridspec(1, 2, width_ratios=[1.18, 1.24], wspace=0.30)
    ax_d = fig.add_subplot(row3[0, 0])
    ax_e = fig.add_subplot(row3[0, 1])
    panel_d_ranking_false_positive(fig, ax_d, summary)
    panel_e_mechanism_bubbles(fig, ax_e, mechanism)

    row4 = outer[3, 0].subgridspec(1, 2, width_ratios=[1.20, 1.28], wspace=0.32)
    ax_f = fig.add_subplot(row4[0, 0])
    ax_g = fig.add_subplot(row4[0, 1])
    panel_f_noise_robustness(fig, ax_f, method_metrics)
    panel_g_detection_burden(fig, ax_g, method_metrics, summary)

    ax_h = fig.add_subplot(outer[4, 0])
    panel_h_overlap_heatmap(fig, ax_h, overlap)
    return fig


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.032, dy: float = 0.012) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
    )


def panel_a_schematic(fig: plt.Figure, ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(fig, ax, "a", dx=-0.006, dy=0.002)

    ax.text(
        0.50,
        0.985,
        "Reference-based LR range benchmark",
        ha="center",
        va="top",
        fontsize=10.0,
        color=TEXT,
    )

    centers = [0.105, 0.305, 0.505, 0.705, 0.900]
    titles = [
        "Spatial references",
        "scDesign3 expression",
        "LR range truth",
        "Noise + decoys",
        "Fair evaluation",
    ]
    subtitles = [
        "LN + melanoma",
        "200 LR pairs",
        "contact to niche",
        "dropout / hard negatives",
        "exact top 5%",
    ]
    for cx, title, subtitle in zip(centers, titles, subtitles, strict=True):
        ax.text(cx, 0.815, title, ha="center", va="center", fontsize=7.8, color=TEXT)
        ax.text(cx, 0.760, subtitle, ha="center", va="center", fontsize=5.7, color=MUTED)
    for x0, x1 in zip(centers[:-1], centers[1:], strict=True):
        arrow = FancyArrowPatch(
            (x0 + 0.075, 0.510),
            (x1 - 0.075, 0.510),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.0,
            color="#8E8A85",
            alpha=0.90,
        )
        ax.add_patch(arrow)

    draw_reference_cloud(ax, centers[0], 0.505)
    draw_scdesign3_module(ax, centers[1], 0.505)
    draw_range_truth(ax, centers[2], 0.505)
    draw_noise_decoys(ax, centers[3], 0.505)
    draw_evaluation_module(ax, centers[4], 0.505)

    ax.text(0.105, 0.145, "real coordinates\nand cell composition", ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(0.305, 0.145, "reference-like\ncounts + sparsity", ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(0.505, 0.145, "distance-specific\npositives", ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(0.705, 0.145, "wrong-range\nfalse positives", ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(0.900, 0.145, "precision, FDR,\nrange violation", ha="center", va="top", fontsize=5.3, color=MUTED)


def draw_reference_cloud(ax: plt.Axes, cx: float, cy: float) -> None:
    rng = np.random.default_rng(7)
    colors = [RED, BLUE, TEAL, AMBER, "#8C6FAE"]
    for k, (sx, sy, rot) in enumerate([(0.042, 0.020, 0.5), (0.033, 0.030, -0.9)]):
        pts = rng.normal(size=(210, 2))
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        r = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        pts = pts @ r.T
        pts[:, 0] += cx
        pts[:, 1] += cy + (0.045 if k == 0 else -0.055)
        cc = np.take(colors, np.arange(len(pts)) % len(colors))
        ax.scatter(pts[:, 0], pts[:, 1], s=1.8, c=cc, alpha=0.78, lw=0, zorder=3)
        ax.add_patch(Circle((cx, cy + (0.045 if k == 0 else -0.055)), 0.070, fill=False, ec="#D4CEC5", lw=0.65))
    ax.text(cx - 0.068, cy + 0.125, "LN", ha="center", va="center", fontsize=5.2, color=MUTED)
    ax.text(cx - 0.082, cy - 0.125, "MEL", ha="center", va="center", fontsize=5.2, color=MUTED)


def draw_scdesign3_module(ax: plt.Axes, cx: float, cy: float) -> None:
    mat_x, mat_y = cx - 0.055, cy - 0.050
    rng = np.random.default_rng(4)
    vals = rng.random((8, 9))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = mpl.colors.to_hex(HEAT(vals[i, j] * 0.75 + 0.08))
            ax.add_patch(Rectangle((mat_x + j * 0.011, mat_y + i * 0.011), 0.010, 0.010, fc=color, ec="white", lw=0.20))
    for x, y, c in [
        (cx + 0.050, cy + 0.075, RED),
        (cx + 0.075, cy + 0.010, BLUE),
        (cx + 0.037, cy - 0.072, TEAL),
    ]:
        draw_cell(ax, x, y, c, scale=0.017)
    ax.text(cx, cy - 0.132, "expression matrix", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_range_truth(ax: plt.Axes, cx: float, cy: float) -> None:
    x0 = cx - 0.067
    y0 = cy + 0.078
    specs = [
        ("contact", 0.018),
        ("short", 0.035),
        ("medium", 0.053),
        ("long_niche", 0.075),
    ]
    for idx, (key, radius) in enumerate(specs):
        x = x0 + idx * 0.045
        y = y0 - idx * 0.040
        draw_cell(ax, x, y, RANGE_COLORS[key], scale=0.016)
        ax.add_patch(Circle((x + radius * 0.65, y), radius, fill=False, ec=RANGE_COLORS[key], lw=0.9, alpha=0.88))
        draw_cell(ax, x + radius * 1.28, y, RANGE_COLORS[key], scale=0.011)
        ax.plot([x + 0.013, x + radius * 1.10], [y, y], color=RANGE_COLORS[key], lw=1.0, alpha=0.85)
    ax.text(cx, cy - 0.128, "contact / short / medium / long", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_noise_decoys(ax: plt.Axes, cx: float, cy: float) -> None:
    rng = np.random.default_rng(2)
    for i in range(38):
        x = cx - 0.060 + rng.random() * 0.120
        y = cy - 0.075 + rng.random() * 0.165
        s = 5 if i % 5 else 12
        c = [NEUTRAL, GOLD, RED, BLUE][i % 4]
        ax.scatter(x, y, s=s, c=c, alpha=0.72, lw=0)
    ax.plot([cx - 0.060, cx + 0.060], [cy - 0.100, cy + 0.100], color=RED_DARK, lw=0.9, alpha=0.75)
    ax.plot([cx - 0.060, cx + 0.060], [cy + 0.100, cy - 0.100], color=RED_DARK, lw=0.9, alpha=0.75)
    ax.add_patch(FancyBboxPatch((cx - 0.070, cy - 0.113), 0.140, 0.226, boxstyle="round,pad=0.006,rounding_size=0.018", fc="none", ec="#D7D0C8", lw=0.7))
    ax.text(cx, cy - 0.132, "same LR, wrong range", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_evaluation_module(ax: plt.Axes, cx: float, cy: float) -> None:
    card = FancyBboxPatch((cx - 0.079, cy - 0.112), 0.158, 0.224, boxstyle="round,pad=0.007,rounding_size=0.018", fc="#FBFAF7", ec="#D7D0C8", lw=0.70)
    ax.add_patch(card)
    for i, (label, val, col) in enumerate(
        [
            ("top 5%", 0.94, RED),
            ("AUPR", 0.82, BLUE),
            ("FDR", 0.68, AMBER),
            ("range", 0.88, TEAL),
        ]
    ):
        y = cy + 0.071 - i * 0.046
        ax.text(cx - 0.058, y, label, ha="left", va="center", fontsize=5.5, color=TEXT)
        ax.add_patch(Rectangle((cx - 0.004, y - 0.008), 0.060, 0.016, fc="#ECE7E1", ec="none"))
        ax.add_patch(Rectangle((cx - 0.004, y - 0.008), 0.060 * val, 0.016, fc=col, ec="none", alpha=0.88))
    ax.text(cx, cy - 0.132, "fixed detection burden", ha="center", va="center", fontsize=5.1, color=MUTED)


def draw_cell(ax: plt.Axes, x: float, y: float, color: str, scale: float = 0.015) -> None:
    ax.add_patch(Circle((x, y), scale, fc=mpl.colors.to_rgba(color, 0.70), ec="#1B1B1B", lw=0.65))
    ax.add_patch(Circle((x + scale * 0.20, y + scale * 0.10), scale * 0.43, fc=mpl.colors.to_rgba(color, 0.96), ec="#1B1B1B", lw=0.45))


def panel_b_radar(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "b", dx=-0.024, dy=0.012)
    plot_methods = ["ScaleComm", "COMMOT", "CellPhoneDB", "SpaOTsc", "CellNEST"]
    high = method_metrics.groupby(["method_label", "noise"], observed=True)["aupr"].mean().reset_index()
    clean = high[high["noise"].eq("clean")].set_index("method_label")["aupr"]
    high_noise = high[high["noise"].eq("high")].set_index("method_label")["aupr"]
    robustness = (high_noise / clean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data = summary.set_index("method_label")
    radar = pd.DataFrame(
        {
            "AUPR": data["aupr_mean"],
            "precision": data["precision_mean"],
            "recall": data["recall_mean"],
            "specificity": data["specificity_mean"],
            "F1": data["f1_mean"],
            "1 - FDR": 1.0 - data["empirical_fdr_mean"],
            "1 - range FP": 1.0 - data["wrong_range_rate_mean"],
            "noise": robustness,
        }
    ).loc[plot_methods]
    normalized = radar.copy()
    for col in normalized.columns:
        lo, hi = float(normalized[col].min()), float(normalized[col].max())
        normalized[col] = 0.5 if hi <= lo else (normalized[col] - lo) / (hi - lo)
    angles = np.linspace(0, 2 * np.pi, len(normalized.columns), endpoint=False)
    angles = np.r_[angles, angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels([])
    ax.grid(color=GRID, lw=0.55)
    ax.spines["polar"].set_color("#D6D0C8")
    ax.spines["polar"].set_linewidth(0.65)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(normalized.columns, fontsize=5.4, color=MUTED)
    for tick in ax.get_xticklabels():
        tick.set_y(tick.get_position()[1] - 0.02)
    for method in reversed(plot_methods):
        values = normalized.loc[method].to_numpy(float)
        values = np.r_[values, values[0]]
        color = METHOD_COLORS[method]
        lw = 1.45 if method == "ScaleComm" else 0.90
        alpha = 0.95 if method == "ScaleComm" else 0.48
        ax.plot(angles, values, color=color, lw=lw, alpha=alpha)
        ax.fill(angles, values, color=color, alpha=0.10 if method == "ScaleComm" else 0.035)
    ax.set_title("Multi-metric benchmark profile", pad=12, fontsize=7.5, color=TEXT)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=1.5 if m == "ScaleComm" else 1.0, label=m)
        for m in plot_methods
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1.4, columnspacing=0.9, borderaxespad=0.0)


def panel_c_standard_metrics(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "c")
    metrics = [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
    ]
    plot_metric_small_multiples(ax, method_metrics, metrics, "Standard binary benchmark metrics", higher_better=True)


def panel_d_ranking_false_positive(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_label(fig, ax, "d")
    plot = summary.set_index("method_label").loc[METHOD_ORDER].reset_index()
    metrics = [
        ("aupr_mean", "AUPR", True),
        ("auroc_mean", "AUROC", True),
        ("top_k_precision_mean", "Top-k", True),
        ("empirical_fdr_mean", "FDR", False),
        ("wrong_range_rate_mean", "Wrong range", False),
    ]
    x = np.arange(len(metrics))
    y = np.arange(len(METHOD_ORDER))
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylim(-0.65, len(METHOD_ORDER) - 0.35)
    ax.invert_yaxis()
    for i, (col, _label, higher) in enumerate(metrics):
        vals = plot[col].to_numpy(float)
        display = vals if higher else 1.0 - vals
        sizes = 24 + 122 * minmax(display)
        colors = [METHOD_COLORS[m] if m == "ScaleComm" else mpl.colors.to_rgba(NEUTRAL_DARK, 0.58) for m in plot["method_label"]]
        ax.scatter(np.full_like(y, i, dtype=float), y, s=sizes, c=colors, edgecolor="white", lw=0.55, zorder=3)
        best_idx = int(np.nanargmax(display))
        ax.scatter([i], [best_idx], s=sizes[best_idx] + 22, facecolors="none", edgecolors=RED, lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=5.9)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.8)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_title("Ranking accuracy and false-positive control", loc="left", pad=5, fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.text(-0.52, -1.08, "circle area = higher-is-better transformed value", ha="left", va="center", fontsize=4.9, color=MUTED, transform=ax.transData)


def panel_e_mechanism_bubbles(fig: plt.Figure, ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    panel_label(fig, ax, "e")
    sub = mechanism[mechanism["range_class"].isin(RANGE_ORDER)].copy()
    mean = (
        sub.groupby(["method_label", "range_class"], observed=True)
        .agg(aupr=("aupr", "mean"), wrong_range=("wrong_range_rate", "mean"))
        .reset_index()
    )
    mean["specificity"] = 1.0 - mean["wrong_range"].clip(0, 1)
    mean = mean[np.isfinite(mean["aupr"].to_numpy(float)) & np.isfinite(mean["specificity"].to_numpy(float))].copy()
    method_to_y = {m: i for i, m in enumerate(METHOD_ORDER)}
    range_to_x = {r: i for i, r in enumerate(RANGE_ORDER)}
    vmax = float(mean["aupr"].quantile(0.98))
    for _, row in mean.iterrows():
        x = range_to_x[str(row["range_class"])]
        y = method_to_y[str(row["method_label"])]
        size = 24 + 130 * float(row["specificity"])
        ax.scatter(x, y, s=size, c=[row["aupr"]], cmap=HEAT, vmin=0, vmax=vmax, edgecolor="white", lw=0.55)
    ax.set_xticks(range(len(RANGE_ORDER)))
    ax.set_xticklabels([RANGE_LABELS[r] for r in RANGE_ORDER], fontsize=5.8)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.8)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_xlim(-0.55, len(RANGE_ORDER) - 0.45)
    ax.set_ylim(-0.65, len(METHOD_ORDER) - 0.35)
    ax.invert_yaxis()
    ax.set_title("Mechanism-specific recovery", loc="left", pad=5, fontsize=7.5)
    ax.grid(color=GRID, lw=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cax = ax.inset_axes([1.025, 0.18, 0.025, 0.55])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=HEAT)
    cb = plt.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.0, colors=MUTED, pad=1)
    cb.set_label("AUPR", fontsize=5.2, color=MUTED, rotation=270, labelpad=8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D8D0C7", markeredgecolor="white", markersize=5.6, label="lower range FP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D8D0C7", markeredgecolor="white", markersize=8.4, label="higher range specificity"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.00, -0.105), ncol=2, handletextpad=0.45, columnspacing=0.85, labelspacing=0.20, borderaxespad=0.0)


def panel_f_noise_robustness(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "f")
    noise_order = ["clean", "low", "high"]
    x = np.arange(len(noise_order))
    mean = method_metrics.groupby(["method_label", "noise"], observed=True)["aupr"].mean().reset_index()
    for method in METHOD_ORDER:
        sub = mean[mean["method_label"].astype(str).eq(method)].set_index("noise").reindex(noise_order)
        vals = sub["aupr"].to_numpy(float)
        color = METHOD_COLORS[method]
        if method == "ScaleComm":
            ax.plot(x, vals, color=color, lw=1.65, marker="o", ms=3.8, mec="white", mew=0.55, zorder=5)
        elif method in {"COMMOT", "CellPhoneDB", "SpaOTsc"}:
            ax.plot(x, vals, color=color, lw=0.95, marker="o", ms=2.8, mec="white", mew=0.35, alpha=0.82, zorder=3)
        else:
            ax.plot(x, vals, color="#AFA9A1", lw=0.70, alpha=0.62, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Low", "High"], fontsize=5.9)
    ax.set_ylabel("AUPR", fontsize=6.0, labelpad=1.5)
    ax.set_title("Noise robustness", loc="left", pad=5, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    despine(ax)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=1.4 if m == "ScaleComm" else 1.0, marker="o", ms=3, label=m)
        for m in ["ScaleComm", "COMMOT", "CellPhoneDB", "SpaOTsc"]
    ]
    handles.append(Line2D([0], [0], color="#AFA9A1", lw=0.9, label="other baselines"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.01, 1.15), frameon=False, handlelength=1.6, ncol=2, columnspacing=0.9)


def panel_g_detection_burden(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    panel_label(fig, ax, "g")
    frac = method_metrics.groupby("method_label", observed=True)["detected_candidate_fraction"].mean().reindex(METHOD_ORDER)
    lr_count = summary.set_index("method_label").reindex(METHOD_ORDER)["detected_lr_pair_count_mean"]
    y = np.arange(len(METHOD_ORDER))
    ax.barh(y - 0.16, frac.to_numpy(float) * 100, height=0.26, color="#D8D0C7", edgecolor="none", label="candidate fraction")
    ax2 = ax.twiny()
    colors = [METHOD_COLORS[m] if m == "ScaleComm" else "#9F978F" for m in METHOD_ORDER]
    ax2.barh(y + 0.16, lr_count.to_numpy(float), height=0.26, color=colors, edgecolor="none", alpha=0.92, label="LR pairs")
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.8)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("detected candidates (%)", fontsize=5.8, color=MUTED, labelpad=2)
    ax2.set_xlabel("detected LR pairs", fontsize=5.8, color=MUTED, labelpad=2)
    ax.set_xlim(0, 5.8)
    ax2.set_xlim(0, max(205, float(lr_count.max()) * 1.08))
    ax.set_title("Fixed detection burden", loc="left", pad=5, fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.5)
    despine(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=5.2, length=2, colors=MUTED, pad=1)
    ax.tick_params(axis="x", labelsize=5.2, length=2, colors=MUTED, pad=1)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fc="#D8D0C7", ec="none", label="exact top 5%"),
            Rectangle((0, 0), 1, 1, fc=RED, ec="none", label="LR-pair coverage"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.03),
    )


def panel_h_overlap_heatmap(fig: plt.Figure, ax: plt.Axes, overlap: pd.DataFrame) -> None:
    panel_label(fig, ax, "h")
    sub = overlap[overlap["overlap_level"].eq("lr_pair")].copy()
    mean = sub.groupby(["method_a", "method_b"], sort=False)["jaccard"].mean().reset_index()
    key_to_label = {v: k for k, v in METHOD_KEY.items()}
    label_to_key = {k: v for k, v in METHOD_KEY.items()}
    matrix = pd.DataFrame(np.nan, index=METHOD_ORDER, columns=METHOD_ORDER, dtype=float)
    for _, row in mean.iterrows():
        a = key_to_label.get(str(row["method_a"]), str(row["method_a"]))
        b = key_to_label.get(str(row["method_b"]), str(row["method_b"]))
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] = float(row["jaccard"])
            matrix.loc[b, a] = float(row["jaccard"])
    cmap = BLUE_HEAT.copy()
    cmap.set_bad("#F6F3ED")
    im = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.4)
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_title("Method overlap across detected LR pairs", loc="left", pad=5, fontsize=7.5)
    for i in range(len(METHOD_ORDER)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="#D8D0C7", lw=0.55))
    ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.70)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.012, 0.13, 0.014, 0.70])
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.0, colors=MUTED, pad=1)
    cb.set_label("Jaccard", fontsize=5.2, color=MUTED, rotation=270, labelpad=7)


def plot_metric_small_multiples(
    ax: plt.Axes,
    data: pd.DataFrame,
    metrics: list[tuple[str, str]],
    title: str,
    *,
    higher_better: bool,
) -> None:
    ax.set_axis_off()
    bbox = ax.get_position()
    fig = ax.figure
    left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
    fig.text(left, bbox.y1 + 0.008, title, ha="left", va="bottom", fontsize=7.5, color=TEXT)
    sub_axes = []
    gap = 0.018
    single_w = (width - gap * (len(metrics) - 1)) / len(metrics)
    for i, (metric, label) in enumerate(metrics):
        sub_axes.append(fig.add_axes([left + i * (single_w + gap), bottom, single_w, height * 0.90]))
        draw_metric_points(sub_axes[-1], data, metric, label, show_y=(i == 0))


def draw_metric_points(ax: plt.Axes, data: pd.DataFrame, metric: str, label: str, show_y: bool = False) -> None:
    y = np.arange(len(METHOD_ORDER))
    for yi, method in enumerate(METHOD_ORDER):
        vals = data[data["method_label"].astype(str).eq(method)][metric].to_numpy(float)
        jitter = np.linspace(-0.11, 0.11, len(vals)) if len(vals) > 1 else np.array([0])
        color = METHOD_COLORS[method] if method == "ScaleComm" else "#9F978F"
        ax.scatter(vals, yi + jitter, s=10 if method == "ScaleComm" else 7.5, c=color, alpha=0.86, edgecolor="white", lw=0.25, zorder=3)
        ax.scatter(np.nanmean(vals), yi, s=32 if method == "ScaleComm" else 18, c=color, edgecolor="#222222" if method == "ScaleComm" else "white", lw=0.45, zorder=4)
    ax.set_ylim(-0.65, len(METHOD_ORDER) - 0.35)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(label, fontsize=6.4, color=TEXT, pad=3)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=5.0, color=MUTED)
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(METHOD_ORDER, fontsize=5.4)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "ScaleComm":
                lab.set_color(RED)
                lab.set_fontweight("bold")
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, pad=1)
    despine(ax)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.ones_like(values) * 0.5
    return (values - low) / (high - low)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.50)
    ax.spines["bottom"].set_linewidth(0.50)
    ax.spines["left"].set_color("#5A5652")
    ax.spines["bottom"].set_color("#5A5652")


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
