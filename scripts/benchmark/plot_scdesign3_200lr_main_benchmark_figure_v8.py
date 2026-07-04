#!/usr/bin/env python
"""Publication-style main benchmark figure for scDesign3 200-LR results.

v8 applies the final benchmark audit:
CytoSignal is removed because only an adapter smoke run was available; panel a
shows only the four LR communication range types; quantitative panels follow
the user-provided reference figure styles.
"""

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
OUT = BENCH / "main_figure_v8"
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
]

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = ["Contact", "Short", "Medium", "Long niche"]

TEXT = "#252321"
MUTED = "#6F6861"
GRID = "#E8E2D6"
PALE = "#F8F5EE"
PAPER = "#FFFFFF"
RED = "#B85450"
RED_DARK = "#7C332F"
BLUE = "#2F78A8"
TEAL = "#237A78"
GOLD = "#E6C06A"
AMBER = "#D9992B"
ORANGE = "#E87722"
GREEN = "#4C8C6B"
PURPLE = "#8E6AAE"
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
}

CT_COLORS = {
    "B": BLUE,
    "T": RED,
    "Other": NEUTRAL,
    "DC": TEAL,
    "Macrophage": AMBER,
    "stromal": GOLD,
}

GOOD_CMAP = LinearSegmentedColormap.from_list("good_scale", ["#FFF1A8", "#F48A22", "#9B2C83"])
BAD_CMAP = LinearSegmentedColormap.from_list("bad_scale", ["#FFF1A8", "#F48A22", "#9B2C2C"])
OVERLAP_CMAP = LinearSegmentedColormap.from_list("overlap_scale", ["#F6F3EE", "#94C6DE", "#1F5E92"])
FOREST_COLORS = ["#6A51A3", "#0C775E", "#005B96", "#A50034", "#9DA611", "#5DA5DA", "#F28E2B", "#D05A9B"]
RADAR_COLORS = {
    "AUPR": ORANGE,
    "AUROC": GREEN,
    "F1": RED,
    "Precision": TEAL,
    "Recall": BLUE,
    "Specificity": PURPLE,
    "1-FDR": AMBER,
    "1-range FP": "#A65858",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    setup_style()
    method_metrics, summary, mechanism, overlap, cells = load_data()
    method_metrics, summary, mechanism, overlap = filter_methods(method_metrics, summary, mechanism, overlap)
    write_source(method_metrics, summary, mechanism, overlap)
    fig = plot_figure(method_metrics, summary, mechanism, overlap, cells)
    export(fig, OUT / "scdesign3_200lr_main_benchmark_figure_v8")
    print(f"Wrote {OUT / 'scdesign3_200lr_main_benchmark_figure_v8.pdf'}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.65,
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


def filter_methods(
    method_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    overlap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = set(METHOD_ORDER)
    method_metrics = method_metrics[method_metrics["method_label"].isin(keep)].copy()
    summary = summary[summary["method_label"].isin(keep)].copy()
    mechanism = mechanism[mechanism["method_label"].isin(keep)].copy()
    overlap = overlap[
        overlap["method_a"].map(method_key_to_label).isin(keep)
        & overlap["method_b"].map(method_key_to_label).isin(keep)
    ].copy()
    return method_metrics, summary, mechanism, overlap


def method_key_to_label(method: object) -> str:
    mapping = {"SpatialSelf": "ScaleComm"}
    return mapping.get(str(method), str(method))


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
    fig = plt.figure(figsize=(7.45, 12.85), constrained_layout=False)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.22, 3.34, 1.56, 1.45, 1.42],
        left=0.067,
        right=0.970,
        bottom=0.052,
        top=0.968,
        hspace=0.50,
    )
    panel_a(fig, outer[0, 0], cells)
    panel_b_radar(fig, outer[1, 0], summary)
    row3 = outer[2, 0].subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.30)
    panel_c_metric_distributions(fig, row3[0, 0], method_metrics)
    panel_d_detected_overlap(fig, row3[0, 1], summary, overlap)
    row4 = outer[3, 0].subgridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.32)
    panel_e_mechanism_heatmap(fig, fig.add_subplot(row4[0, 0]), mechanism)
    panel_f_error_composition(fig, fig.add_subplot(row4[0, 1]), method_metrics)
    row5 = outer[4, 0].subgridspec(1, 2, width_ratios=[1.00, 1.00], wspace=0.32)
    panel_g_noise(fig, fig.add_subplot(row5[0, 0]), method_metrics)
    panel_h_error_modes(fig, fig.add_subplot(row5[0, 1]), mechanism)
    return fig


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.026, dy: float = 0.010) -> None:
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, label, ha="left", va="bottom", fontsize=12.0, fontweight="bold", color="#111111")


def panel_a(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, cells: dict[str, pd.DataFrame]) -> None:
    ax = fig.add_subplot(spec)
    panel_label(fig, ax, "a", dx=-0.035, dy=0.005)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.005, 0.980, "LR-specific communication range types", ha="left", va="top", fontsize=10.0, color=TEXT, transform=ax.transAxes)
    card_specs = [
        (0.012, 0.080, 0.224, 0.740, "Contact", "juxtacrine", RED, 0.15),
        (0.268, 0.080, 0.224, 0.740, "Short-range", "local paracrine", AMBER, 0.34),
        (0.524, 0.080, 0.224, 0.740, "Medium-range", "diffusive paracrine", BLUE, 0.54),
        (0.780, 0.080, 0.208, 0.740, "Long-niche", "chemokine-like niche", TEAL, 0.78),
    ]
    for x, y, w, h, title, subtitle, color, span in card_specs:
        draw_range_type_card(ax, x, y, w, h, title, subtitle, color, span)


def draw_range_type_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    color: str,
    span: float,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.009",
            fc="#FFFFFF",
            ec="#D7D0C7",
            lw=0.72,
            zorder=0,
        )
    )
    ax.text(x + 0.5 * w, y + h - 0.050, title, ha="center", va="center", fontsize=7.4, color=TEXT, fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.5 * w, y + h - 0.103, subtitle, ha="center", va="center", fontsize=5.2, color=MUTED, transform=ax.transAxes)
    inset = ax.inset_axes([x + 0.08 * w, y + 0.118 * h, 0.84 * w, 0.640 * h])
    inset.set_axis_off()
    inset.set_aspect("equal")
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    draw_range_scene(inset, color, span)


def draw_range_scene(ax: plt.Axes, color: str, span: float) -> None:
    center = np.array([0.32, 0.50])
    rng = np.random.default_rng(int(span * 1000))
    background = np.array(
        [
            [0.18, 0.28],
            [0.20, 0.72],
            [0.43, 0.22],
            [0.50, 0.72],
            [0.70, 0.30],
            [0.82, 0.62],
            [0.64, 0.80],
            [0.82, 0.18],
        ]
    )
    background += rng.normal(0, 0.014, background.shape)
    for pt in background:
        draw_cell(ax, pt[0], pt[1], "#E5DFD7", "#5E5952", size=0.050, alpha=0.88)
    radius = 0.17 + span * 0.34
    circle = plt.Circle(center, radius, fill=False, ls=(0, (2, 2)), lw=0.85, ec="#4C4742", alpha=0.78)
    ax.add_patch(circle)
    draw_cell(ax, center[0], center[1], color, "#24211F", size=0.072, alpha=0.62)
    receiver = center + np.array([0.08 + span * 0.45, 0.02 + 0.10 * np.sin(span * np.pi)])
    receiver[0] = min(receiver[0], 0.86)
    receiver[1] = min(max(receiver[1], 0.22), 0.78)
    ax.plot([center[0], receiver[0]], [center[1], receiver[1]], color=color, lw=1.75, alpha=0.92, solid_capstyle="round")
    draw_cell(ax, receiver[0], receiver[1], color, "#24211F", size=0.062, alpha=0.48)
    if span > 0.6:
        for pt in [[0.55, 0.38], [0.66, 0.47], [0.75, 0.56]]:
            ax.scatter([pt[0]], [pt[1]], s=10, c=color, alpha=0.55, lw=0)
    elif span > 0.3:
        ax.scatter([0.50, 0.58], [0.50, 0.54], s=10, c=color, alpha=0.55, lw=0)
    ax.text(center[0], center[1] - 0.135, "L", ha="center", va="center", fontsize=5.7, color=TEXT, fontweight="bold")
    ax.text(receiver[0], receiver[1] + 0.125, "R", ha="center", va="center", fontsize=5.7, color=TEXT, fontweight="bold")


def draw_cell(ax: plt.Axes, x: float, y: float, color: str, edge: str, *, size: float, alpha: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    r = size * (1 + 0.12 * np.sin(3 * theta + x * 7))
    xs = x + r * np.cos(theta)
    ys = y + r * np.sin(theta)
    ax.fill(xs, ys, facecolor=color, edgecolor=edge, lw=0.75, alpha=alpha, zorder=2)
    ax.add_patch(plt.Circle((x + size * 0.18, y + size * 0.06), size * 0.28, fc=edge, ec="none", alpha=0.22, zorder=3))


def draw_module_title(ax: plt.Axes, x: float, y: float, title: str, subtitle: str) -> None:
    ax.text(x, y, title, ha="center", va="center", fontsize=7.2, color=TEXT, fontweight="bold", transform=ax.transAxes)
    ax.text(x, y - 0.060, subtitle, ha="center", va="center", fontsize=5.2, color=MUTED, transform=ax.transAxes)


def draw_large_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#77716A", shrinkA=0, shrinkB=0, mutation_scale=12),
    )


def draw_spatial_reference_module(ax: plt.Axes, cells: dict[str, pd.DataFrame]) -> None:
    ax.add_patch(FancyBboxPatch((0.015, 0.135), 0.272, 0.505, transform=ax.transAxes, boxstyle="round,pad=0.010,rounding_size=0.010", fc="#FBFAF7", ec="#D8D0C7", lw=0.70))
    draw_equal_spot_inset(ax, [0.041, 0.318, 0.106, 0.248], cells["LN"], "LN", 7)
    draw_equal_spot_inset(ax, [0.163, 0.318, 0.106, 0.248], cells["Melanoma"], "MEL", 11)
    draw_expression_mini(ax, 0.060, 0.185, 0.180, 0.092)
    ax.text(0.151, 0.145, "reference-based counts preserve sparsity", ha="center", va="center", fontsize=5.0, color=MUTED, transform=ax.transAxes)


def draw_equal_spot_inset(parent: plt.Axes, rect: list[float], data: pd.DataFrame, label: str, seed: int) -> None:
    inset = parent.inset_axes(rect)
    inset.set_axis_off()
    inset.set_aspect("equal")
    data = data.sample(min(520, len(data)), random_state=seed)
    x, y = data["x"].to_numpy(float), data["y"].to_numpy(float)
    sx, sy = scale_xy_equal(x, y, 0.06, 0.08, 0.88, 0.78)
    colors = [CT_COLORS.get(str(t), LIGHT_NEUTRAL) for t in data["cell_type"]]
    inset.scatter(sx, sy, s=3.0, c=colors, alpha=0.78, lw=0)
    inset.text(0.02, 0.95, label, ha="left", va="top", fontsize=5.5, color=MUTED, transform=inset.transAxes)
    inset.add_patch(Rectangle((0.005, 0.005), 0.99, 0.99, fill=False, ec="#D8D0C7", lw=0.55, transform=inset.transAxes))
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)


def draw_expression_mini(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    rng = np.random.default_rng(23)
    mat = rng.beta(1.4, 3.0, size=(5, 14))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(
                Rectangle(
                    (x0 + w * j / mat.shape[1], y0 + h * i / mat.shape[0]),
                    w / mat.shape[1] * 0.88,
                    h / mat.shape[0] * 0.78,
                    transform=ax.transAxes,
                    fc=GOOD_CMAP(mat[i, j]),
                    ec="none",
                )
            )


def draw_range_truth_module(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch((0.356, 0.135), 0.275, 0.505, transform=ax.transAxes, boxstyle="round,pad=0.010,rounding_size=0.010", fc="#FBFAF7", ec="#D8D0C7", lw=0.70))
    inset = ax.inset_axes([0.382, 0.205, 0.172, 0.365])
    inset.set_axis_off()
    inset.set_aspect("equal")
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    sender = (0.32, 0.50)
    ranges = [
        ("contact", RED, (0.43, 0.52)),
        ("short", AMBER, (0.58, 0.62)),
        ("medium", BLUE, (0.75, 0.43)),
        ("long niche", TEAL, (0.88, 0.58)),
    ]
    inset.scatter([sender[0]], [sender[1]], s=160, c=RED, alpha=0.30, edgecolor=TEXT, lw=0.65, zorder=3)
    inset.text(sender[0], sender[1] - 0.18, "sender", ha="center", va="center", fontsize=5.0, color=MUTED)
    for label, col, recv in ranges:
        inset.plot([sender[0], recv[0]], [sender[1], recv[1]], color=col, lw=1.35, alpha=0.88)
        inset.scatter([recv[0]], [recv[1]], s=115, c=col, alpha=0.30, edgecolor=TEXT, lw=0.55, zorder=4)
    for i, (label, col, _recv) in enumerate(ranges):
        yy = 0.540 - i * 0.060
        ax.add_patch(Rectangle((0.555, yy - 0.008), 0.025, 0.016, transform=ax.transAxes, fc=col, ec="none", alpha=0.85))
        ax.text(0.586, yy, label, ha="left", va="center", fontsize=5.0, color=MUTED, transform=ax.transAxes)
    inset.plot([0.18, 0.92], [0.18, 0.85], color=RED_DARK, lw=0.8, ls=(0, (2, 2)), alpha=0.75)
    ax.text(0.495, 0.168, "wrong-range decoys", ha="center", va="center", fontsize=5.0, color=MUTED, transform=ax.transAxes)


def draw_evaluation_module(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch((0.712, 0.135), 0.268, 0.505, transform=ax.transAxes, boxstyle="round,pad=0.010,rounding_size=0.010", fc="#FBFAF7", ec="#D8D0C7", lw=0.70))
    metrics = [("F1", 0.72, RED), ("AUPR", 0.66, BLUE), ("1-FDR", 0.58, AMBER), ("1-range FP", 0.82, TEAL)]
    for i, (name, value, col) in enumerate(metrics):
        y = 0.548 - i * 0.080
        ax.text(0.742, y, name, ha="left", va="center", fontsize=5.8, color=TEXT, transform=ax.transAxes)
        ax.add_patch(Rectangle((0.818, y - 0.012), 0.115, 0.024, fc="#E7E0D7", ec="none", transform=ax.transAxes))
        ax.add_patch(Rectangle((0.818, y - 0.012), 0.115 * value, 0.024, fc=col, ec="none", transform=ax.transAxes, alpha=0.94))
    ax.add_patch(Rectangle((0.758, 0.185), 0.168, 0.050, fc="#FFFFFF", ec="#D8D0C7", lw=0.50, transform=ax.transAxes))
    ax.text(0.842, 0.210, "exact top 5%", ha="center", va="center", fontsize=5.3, color=MUTED, transform=ax.transAxes)


def scale_xy_equal(x: np.ndarray, y: np.ndarray, left: float, bottom: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xr = max(float(xmax - xmin), 1e-9)
    yr = max(float(ymax - ymin), 1e-9)
    scale = min(width / xr, height / yr)
    sw, sh = xr * scale, yr * scale
    x0 = left + (width - sw) / 2
    y0 = bottom + (height - sh) / 2
    return x0 + (x - xmin) * scale, y0 + (y - ymin) * scale


def draw_workflow_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            fc="#FBFAF7",
            ec="#D9D1C7",
            lw=0.70,
            zorder=0,
        )
    )
    ax.text(x + 0.5 * w, y + h - 0.050, title, ha="center", va="center", fontsize=6.7, color=TEXT, transform=ax.transAxes)
    ax.text(x + 0.5 * w, y + h - 0.112, subtitle, ha="center", va="center", fontsize=5.0, color=MUTED, transform=ax.transAxes)


def draw_reference_workflow(ax: plt.Axes, card: tuple[float, float, float, float], cells: dict[str, pd.DataFrame]) -> None:
    x, y, w, h = card
    rng = np.random.default_rng(3)
    for i, key in enumerate(["LN", "Melanoma"]):
        data = cells[key]
        data = data.sample(min(520, len(data)), random_state=12 + i)
        rx, ry = x + 0.045 * w, y + (0.402 - 0.205 * i) * h
        rw, rh = 0.905 * w, 0.180 * h
        ax.add_patch(Rectangle((rx, ry), rw, rh, transform=ax.transAxes, fc="#FFFFFF", ec="#E2DCD3", lw=0.45))
        px = rx + rw * (0.06 + 0.88 * norm01(data["x"].to_numpy(float)))
        py = ry + rh * (0.10 + 0.80 * norm01(data["y"].to_numpy(float)))
        jitter = rng.normal(0, 0.0015, len(px))
        colors = [CT_COLORS.get(str(t), LIGHT_NEUTRAL) for t in data["cell_type"]]
        ax.scatter(px + jitter, py, s=2.0, c=colors, alpha=0.78, lw=0, transform=ax.transAxes, zorder=3)
        ax.text(rx + 0.012 * rw, ry + 0.78 * rh, "LN" if key == "LN" else "MEL", ha="left", va="center", fontsize=5.0, color=MUTED, transform=ax.transAxes)
    ax.text(x + 0.5 * w, y + 0.110 * h, "tissue geometry retained", ha="center", va="center", fontsize=5.1, color=MUTED, transform=ax.transAxes)


def draw_expression_workflow(ax: plt.Axes, card: tuple[float, float, float, float]) -> None:
    x, y, w, h = card
    rng = np.random.default_rng(2)
    mat = rng.beta(1.4, 3.2, size=(9, 10))
    x0, y0 = x + 0.195 * w, y + 0.360 * h
    cw, ch = 0.056 * w, 0.037 * h
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(
                Rectangle(
                    (x0 + j * cw * 1.18, y0 + i * ch * 1.18),
                    cw,
                    ch,
                    transform=ax.transAxes,
                    fc=GOOD_CMAP(mat[i, j]),
                    ec="#FFFFFF",
                    lw=0.18,
                )
            )
    for cx, cy, col in [(0.34, 0.685, RED), (0.67, 0.635, BLUE), (0.55, 0.270, TEAL)]:
        ax.scatter([x + cx * w], [y + cy * h], s=118, c=col, alpha=0.32, edgecolor=TEXT, lw=0.50, transform=ax.transAxes, zorder=3)
        ax.scatter([x + (cx + 0.030) * w], [y + (cy + 0.020) * h], s=27, c=col, edgecolor=TEXT, lw=0.38, transform=ax.transAxes, zorder=4)
    ax.text(x + 0.5 * w, y + 0.110 * h, "library size and sparsity matched", ha="center", va="center", fontsize=5.1, color=MUTED, transform=ax.transAxes)


def draw_range_workflow(ax: plt.Axes, card: tuple[float, float, float, float]) -> None:
    x, y, w, h = card
    items = [
        ("contact", RED, 0.14, 0.660),
        ("short", AMBER, 0.26, 0.535),
        ("medium", BLUE, 0.41, 0.410),
        ("long niche", TEAL, 0.58, 0.285),
    ]
    for label, col, span, yy in items:
        x1 = x + 0.185 * w
        x2 = x + (0.185 + span) * w
        yy_abs = y + yy * h
        ax.plot([x1, x2], [yy_abs, yy_abs], color=col, lw=1.2, alpha=0.85, transform=ax.transAxes)
        ax.scatter([x1, x2], [yy_abs, yy_abs], s=[92, 58], c=col, alpha=0.34, edgecolor=TEXT, lw=0.45, transform=ax.transAxes, zorder=3)
        ax.text(x + 0.740 * w, yy_abs, label, ha="left", va="center", fontsize=5.2, color=MUTED, transform=ax.transAxes)
    ax.plot([x + 0.15 * w, x + 0.70 * w], [y + 0.170 * h, y + 0.740 * h], color="#9C5D57", lw=0.75, ls=(0, (2, 2)), transform=ax.transAxes)
    ax.text(x + 0.5 * w, y + 0.110 * h, "wrong-range negatives injected", ha="center", va="center", fontsize=5.1, color=MUTED, transform=ax.transAxes)


def draw_readout_workflow(ax: plt.Axes, card: tuple[float, float, float, float]) -> None:
    x, y, w, h = card
    rows = [("F1", 0.78, RED), ("AUPR", 0.71, BLUE), ("FDR", 0.43, AMBER), ("range FP", 0.26, TEAL)]
    for i, (label, value, col) in enumerate(rows):
        yy = y + (0.665 - i * 0.118) * h
        ax.text(x + 0.160 * w, yy, label, ha="left", va="center", fontsize=5.6, color=TEXT, transform=ax.transAxes)
        ax.add_patch(Rectangle((x + 0.480 * w, yy - 0.015 * h), 0.355 * w, 0.030 * h, transform=ax.transAxes, fc="#E7E0D7", ec="none"))
        ax.add_patch(Rectangle((x + 0.480 * w, yy - 0.015 * h), 0.355 * w * value, 0.030 * h, transform=ax.transAxes, fc=col, ec="none", alpha=0.90))
    ax.add_patch(Rectangle((x + 0.190 * w, y + 0.185 * h), 0.620 * w, 0.060 * h, transform=ax.transAxes, fc="#FFFFFF", ec="#D8D0C7", lw=0.45))
    ax.text(x + 0.500 * w, y + 0.215 * h, "rank exact top 5%", ha="center", va="center", fontsize=5.1, color=MUTED, transform=ax.transAxes)


def panel_b_radar(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, summary: pd.DataFrame) -> None:
    holder_grid = spec.subgridspec(2, 1, height_ratios=[0.16, 0.84], hspace=0.03)
    holder = fig.add_subplot(holder_grid[0, 0])
    holder.set_axis_off()
    panel_label(fig, holder, "b", dx=-0.035, dy=-0.005)
    holder.text(
        0.0,
        0.72,
        "Normalized performance profiles",
        ha="left",
        va="center",
        fontsize=7.7,
        color=TEXT,
        transform=holder.transAxes,
    )
    sub = holder_grid[1, 0].subgridspec(2, 4, wspace=0.42, hspace=0.96)
    metrics = [
        ("aupr_mean", "AUPR", "AUPR", True),
        ("auroc_mean", "AUROC", "AUROC", True),
        ("f1_mean", "F1", "F1 score"),
        ("precision_mean", "Precision", "Precision", True),
        ("recall_mean", "Recall", "Recall", True),
        ("specificity_mean", "Specificity", "Specificity zoom", True, "zoom"),
        ("empirical_fdr_mean", "1-FDR", "1 - FDR", False),
        ("wrong_range_rate_mean", "1-range FP", "1 - range FP", False),
    ]
    for i, item in enumerate(metrics):
        if len(item) == 3:
            col, key, title = item
            higher = True
            scale_mode = "relative_max"
        elif len(item) == 4:
            col, key, title, higher = item
            scale_mode = "relative_max"
        else:
            col, key, title, higher, scale_mode = item
        ax = fig.add_subplot(sub[i // 4, i % 4], projection="polar")
        radar_panel(ax, summary, col, title, RADAR_COLORS[key], higher_is_better=higher, scale_mode=scale_mode)


def radar_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric_col: str,
    title: str,
    color: str,
    *,
    higher_is_better: bool = True,
    scale_mode: str = "relative_max",
) -> None:
    labels = {
        "ScaleComm": "ScaleComm",
        "COMMOT": "COMMOT",
        "CellPhoneDB": "CPDB",
        "SpaOTsc": "SpaOTsc",
        "CellChat": "CellChat",
        "NICHES": "NICHES",
        "SpaTalk": "SpaTalk",
        "CellNEST": "CellNEST",
        "DeepTalk": "DeepTalk",
    }
    table = summary.set_index("method_label").reindex(METHOD_ORDER)
    values = table[metric_col].to_numpy(float)
    if not higher_is_better:
        values = 1 - values
    if scale_mode == "zoom":
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        values = (values - lo) / (hi - lo) if hi > lo else np.ones_like(values)
    else:
        denom = float(np.nanmax(values))
        values = values / denom if denom > 0 else values
    angles = np.linspace(0, 2 * np.pi, len(METHOD_ORDER), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    closed_values = np.r_[values, values[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(closed_angles, closed_values, color=color, lw=1.08, marker="o", ms=1.95, mec="white", mew=0.32, zorder=4)
    ax.fill(closed_angles, closed_values, color=color, alpha=0.13, zorder=2)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#DCD5CC", lw=0.45)
    ax.xaxis.grid(True, color="#E8E1D8", lw=0.35)
    ax.spines["polar"].set_color("#3A3530")
    ax.spines["polar"].set_linewidth(0.56)
    ax.set_xticks(angles)
    ax.set_xticklabels([labels[m] for m in METHOD_ORDER], fontsize=3.25)
    for lab, method in zip(ax.get_xticklabels(), METHOD_ORDER, strict=True):
        if method == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
        else:
            lab.set_color(TEXT)
    ax.set_title(title, fontsize=5.9, color=color, pad=2.5, fontweight="bold")
    if scale_mode == "zoom":
        pass
    else:
        ax.text(np.deg2rad(83), 1.02, "1.0", ha="center", va="center", fontsize=3.7, color=MUTED)
        ax.text(np.deg2rad(83), 0.52, "0.5", ha="center", va="center", fontsize=3.6, color=MUTED)


def panel_c_metric_distributions(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, method_metrics: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    panel_label(fig, ax, "c", dx=-0.050, dy=0.000)
    forest_metric_advantage(ax, method_metrics)


def forest_metric_advantage(ax: plt.Axes, data: pd.DataFrame) -> None:
    metrics = [
        ("AUPR", "aupr", True, FOREST_COLORS[0]),
        ("AUROC", "auroc", True, FOREST_COLORS[1]),
        ("F1", "f1", True, FOREST_COLORS[2]),
        ("Precision", "precision", True, FOREST_COLORS[3]),
        ("Recall", "recall", True, FOREST_COLORS[4]),
        ("Specificity", "specificity", True, FOREST_COLORS[5]),
        ("1 - FDR", "empirical_fdr", False, FOREST_COLORS[6]),
        ("1 - range FP", "wrong_range_rate", False, FOREST_COLORS[7]),
    ]
    rows = []
    for label, col, higher, color in metrics:
        tmp = data[["dataset_id", "method_label", col]].copy()
        tmp["value"] = tmp[col] if higher else 1 - tmp[col]
        for dataset_id, group in tmp.groupby("dataset_id", observed=True):
            scale = group[group["method_label"].eq("ScaleComm")]["value"]
            other = group[~group["method_label"].eq("ScaleComm")]["value"]
            if scale.empty or other.empty:
                continue
            rows.append({"metric": label, "delta": float(scale.iloc[0] - other.max()), "color": color})
    df = pd.DataFrame(rows)
    y = np.arange(len(metrics))
    metric_labels = [m[0] for m in metrics]
    colors = [m[3] for m in metrics]
    for yi, label, color in zip(y, metric_labels, colors, strict=True):
        vals = df[df["metric"].eq(label)]["delta"].to_numpy(float)
        if vals.size == 0:
            continue
        mean = float(np.mean(vals))
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        ax.hlines(yi, lo, hi, color=color, lw=1.65, zorder=2)
        ax.scatter([mean], [yi], s=72, c=color, edgecolor="white", lw=0.70, zorder=4)
    ax.axvline(0, color="#777777", lw=0.82, ls=(0, (2.5, 2.5)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels, fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlabel("ScaleComm advantage over best non-ScaleComm baseline", fontsize=6.5, fontweight="bold")
    ax.set_title("Effect-size summary", loc="left", fontsize=7.4, pad=5)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, colors=MUTED)
    xlim = max(0.18, float(np.nanmax(np.abs(df["delta"]))) * 1.22 if not df.empty else 0.2)
    ax.set_xlim(-xlim, xlim)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_d_detected_overlap(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, summary: pd.DataFrame, overlap: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    panel_label(fig, ax, "d", dx=-0.060, dy=0.000)
    triangular_overlap_heatmap(ax, overlap)


def lr_count_bar(ax: plt.Axes, summary: pd.DataFrame) -> None:
    counts = summary.set_index("method_label").reindex(METHOD_ORDER)["detected_lr_pair_count_mean"]
    y = np.arange(len(METHOD_ORDER))
    colors = [RED if m == "ScaleComm" else "#BFB7AD" for m in METHOD_ORDER]
    ax.barh(y, counts.to_numpy(float), color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=4.7)
    highlight_scalecomm(ax, "y")
    ax.invert_yaxis()
    ax.set_xlabel("detected LR pairs", fontsize=5.4, color=MUTED, labelpad=1)
    ax.grid(axis="x", color=GRID, lw=0.40)
    ax.tick_params(axis="x", length=2, pad=1, labelsize=4.9, colors=MUTED)
    ax.tick_params(axis="y", length=0, pad=1)
    despine(ax)


def overlap_heatmap(ax: plt.Axes, overlap: pd.DataFrame) -> None:
    mat = overlap_matrix(overlap, level="lr_pair")
    im = ax.imshow(mat.to_numpy(float), cmap=OVERLAP_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(short_method_labels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=4.4)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(short_method_labels(), fontsize=4.4)
    highlight_scalecomm(ax, "both")
    heatmap_grid(ax, len(METHOD_ORDER), len(METHOD_ORDER), lw=0.42)
    cbar(ax, im, "Jaccard", x=1.020)


def triangular_overlap_heatmap(ax: plt.Axes, overlap: pd.DataFrame) -> None:
    mat = overlap_matrix(overlap, level="lr_pair").reindex(index=METHOD_ORDER, columns=METHOD_ORDER)
    arr = mat.to_numpy(float)
    mask = np.triu(np.ones_like(arr, dtype=bool), k=0)
    arr_masked = np.ma.array(arr, mask=mask)
    # Drop the empty first row and last column after lower-triangle masking.
    arr_plot = arr_masked[1:, :-1]
    row_labels = METHOD_ORDER[1:]
    col_labels = METHOD_ORDER[:-1]
    cmap = OVERLAP_CMAP
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    n_rows, n_cols = arr_plot.shape
    ax.set_aspect("equal")
    for i in range(n_rows):
        for j in range(n_cols):
            if np.ma.is_masked(arr_plot[i, j]) or not np.isfinite(float(arr_plot[i, j])):
                continue
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(float(arr_plot[i, j]))),
                    edgecolor="#303030",
                    linewidth=0.38,
                )
            )
    im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([short_label(m) for m in col_labels], rotation=45, ha="right", rotation_mode="anchor", fontsize=5.2)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([short_label(m) for m in row_labels], fontsize=5.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() == "ScaleComm":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    ax.set_title("Detected LR-pair overlap", loc="left", fontsize=7.4, pad=5)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar(ax, im, "Jaccard", x=1.015)


def panel_e_mechanism_heatmap(fig: plt.Figure, ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    panel_label(fig, ax, "e")
    sub = mechanism[mechanism["range_class"].isin(RANGE_ORDER)].copy()
    mean = (
        sub.groupby(["method_label", "range_class"], observed=True)["f1"]
        .mean()
        .reset_index()
        .pivot(index="method_label", columns="range_class", values="f1")
        .reindex(index=METHOD_ORDER, columns=RANGE_ORDER)
    )
    vmax = float(np.nanquantile(mean.to_numpy(float), 0.98))
    im = ax.imshow(mean.to_numpy(float), cmap=GOOD_CMAP, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    add_nan_hatching(ax, mean)
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels(RANGE_LABELS, fontsize=5.4)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.25)
    highlight_scalecomm(ax, axis="y")
    ax.add_patch(Rectangle((-0.5, -0.5), len(RANGE_ORDER), 1, fill=False, ec=RED_DARK, lw=1.05))
    ax.set_title("LR range recovery", loc="left", pad=5, fontsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#2B2826")
    cbar(ax, im, "F1", x=1.014)


def panel_f_error_composition(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "f")
    grouped = (
        method_metrics.groupby("method_label", observed=True)[
            ["tp", "fp", "wrong_range_fp_count", "detected_candidate_count"]
        ]
        .mean()
        .reindex(METHOD_ORDER)
    )
    denom = grouped["detected_candidate_count"].replace(0, np.nan)
    tp = grouped["tp"] / denom
    wrong = grouped["wrong_range_fp_count"] / denom
    other = (grouped["fp"] - grouped["wrong_range_fp_count"]).clip(lower=0) / denom
    x = np.arange(len(METHOD_ORDER))
    width = 0.62
    ax.set_facecolor("#F0E9F1")
    tp_pct = (tp * 100).fillna(0).to_numpy(float)
    wrong_pct = (wrong * 100).fillna(0).to_numpy(float)
    other_pct = (other * 100).fillna(0).to_numpy(float)
    ax.bar(x, tp_pct, width=width, color="#5578B5", edgecolor="#111111", lw=0.38, label="TP", zorder=3)
    ax.bar(x, wrong_pct, width=width, bottom=tp_pct, color="#F47C22", edgecolor="#111111", lw=0.38, label="wrong-range FP", zorder=3)
    ax.bar(x, other_pct, width=width, bottom=tp_pct + wrong_pct, color="#D23B73", edgecolor="#111111", lw=0.38, label="other FP", zorder=3)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.62, len(METHOD_ORDER) - 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(m) for m in METHOD_ORDER], rotation=44, ha="right", rotation_mode="anchor", fontsize=5.0)
    highlight_scalecomm(ax, axis="x")
    ax.set_ylabel("Percentage")
    ax.set_title("Selected-call composition", loc="left", pad=5, fontsize=7.5)
    ax.grid(False)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", length=2, colors=TEXT)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.34), ncol=3, handlelength=1.15, columnspacing=0.75, borderpad=0.35)
    leg.get_frame().set_facecolor("#ECECEC")
    leg.get_frame().set_edgecolor("none")


def panel_g_noise(fig: plt.Figure, ax: plt.Axes, method_metrics: pd.DataFrame) -> None:
    panel_label(fig, ax, "g")
    noise_order = ["clean", "low", "high"]
    stats = method_metrics.groupby(["method_label", "noise"], observed=True)["aupr"].agg(["mean", "std"]).reset_index()
    x = np.arange(len(noise_order))
    ax.axvspan(0.5, 2.5, color="#D9D9D9", alpha=0.55, zorder=0)
    ax.axvline(0.5, color="#111111", lw=0.85, ls=(0, (3, 3)), zorder=1)
    for method in METHOD_ORDER:
        rows = stats[stats["method_label"].eq(method)].set_index("noise").reindex(noise_order)
        vals = rows["mean"].to_numpy(float)
        errs = rows["std"].fillna(0).to_numpy(float)
        if method == "ScaleComm":
            ax.errorbar(x, vals, yerr=errs, color=RED, lw=1.80, marker="o", ms=4.2, mec="#111111", mew=0.55, capsize=2.2, elinewidth=0.80, zorder=5)
        elif method in {"COMMOT", "CellPhoneDB", "SpaOTsc"}:
            ax.errorbar(x, vals, yerr=errs, color=METHOD_COLORS[method], lw=1.18, marker="o", ms=3.4, mec="#111111", mew=0.42, capsize=1.8, elinewidth=0.65, alpha=0.92, zorder=3)
        else:
            ax.plot(x, vals, color="#B7B1AA", lw=0.82, alpha=0.45, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Low", "High"])
    ax.set_ylabel("AUPR")
    ax.set_title("Noise robustness", loc="left", pad=5, fontsize=7.5)
    ax.grid(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", lw=1.3, ms=3, label=m)
        for m in ["ScaleComm", "COMMOT", "CellPhoneDB", "SpaOTsc"]
    ]
    handles.append(Line2D([0], [0], color="#B7B1AA", lw=1.0, label="other baselines"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.01, 1.03), ncol=2, handlelength=1.45, columnspacing=0.75)


def short_method_labels() -> list[str]:
    return [short_label(m) for m in METHOD_ORDER]


def short_label(method: str) -> str:
    return {"CellPhoneDB": "CPDB"}.get(method, method)


def panel_b_standard_metrics(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, method_metrics: pd.DataFrame) -> None:
    grid = spec.subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.10)
    holder = fig.add_subplot(grid[0, 0])
    holder.set_axis_off()
    panel_label(fig, holder, "b", dx=-0.035, dy=0.000)
    holder.text(0.0, 0.52, "Standard benchmark metrics at fixed detection burden", ha="left", va="center", fontsize=7.8, color=TEXT, transform=holder.transAxes)
    sub = grid[1, 0].subgridspec(1, 4, wspace=0.075)
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity")]
    for i, (col, title) in enumerate(metrics):
        ax = fig.add_subplot(sub[0, i])
        metric_forest_panel(ax, method_metrics, col, title, show_y=(i == 0))


def metric_forest_panel(ax: plt.Axes, data: pd.DataFrame, col: str, title: str, show_y: bool) -> None:
    y = np.arange(len(METHOD_ORDER))
    for yi, method in enumerate(METHOD_ORDER):
        vals = data[data["method_label"].eq(method)][col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.155, 0.155, vals.size)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        color = METHOD_COLORS[method] if method == "ScaleComm" else "#9D968F"
        ax.scatter(vals, yi + jitter, s=7.4, c=color, alpha=0.35 if method != "ScaleComm" else 0.55, edgecolor="none", zorder=2)
        ax.errorbar(
            mean,
            yi,
            xerr=sd,
            fmt="o",
            ms=4.8 if method == "ScaleComm" else 3.6,
            mfc=color,
            mec="#2A2725" if method == "ScaleComm" else "#FFFFFF",
            mew=0.55 if method == "ScaleComm" else 0.35,
            ecolor=color,
            elinewidth=1.05 if method == "ScaleComm" else 0.72,
            capsize=0,
            alpha=0.95,
            zorder=4 if method == "ScaleComm" else 3,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=6.7, pad=5)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.axvline(0.5, color="#CFC7BE", lw=0.55, zorder=0)
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
    cols = [
        ("aupr_mean", "AUPR", True),
        ("auroc_mean", "AUROC", True),
        ("top_k_precision_mean", "Top-k", True),
        ("precision_mean", "Precision", True),
        ("recall_mean", "Recall", True),
        ("specificity_mean", "Specificity", True),
        ("f1_mean", "F1", True),
        ("empirical_fdr_mean", "1 - FDR", False),
        ("wrong_range_rate_mean", "1 - range FP", False),
        ("high_expression_fp_rate_mean", "1 - expr FP", False),
    ]
    raw = []
    for col, _label, higher in cols:
        vals = table[col].to_numpy(float)
        vals = vals if higher else 1 - vals
        raw.append(vals)
    raw = np.vstack(raw).T
    mat = np.zeros_like(raw, dtype=float)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        mat[:, j] = 0.5 if hi <= lo else (col - lo) / (hi - lo)
    im = ax.imshow(mat, cmap=GOOD_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], rotation=38, ha="right", rotation_mode="anchor", fontsize=5.15)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.35)
    highlight_scalecomm(ax, axis="y")
    ax.add_patch(Rectangle((-0.5, -0.5), len(cols), 1, fill=False, ec=RED, lw=1.05))
    ax.set_title("Multi-metric scorecard", loc="left", pad=5, fontsize=7.3)
    heatmap_grid(ax, len(METHOD_ORDER), len(cols), lw=0.48)
    cbar(ax, im, "relative best", x=1.012)


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
    grouped = (
        method_metrics.groupby("method_label", observed=True)[
            ["tp", "fp", "wrong_range_fp_count", "detected_candidate_count"]
        ]
        .mean()
        .reindex(METHOD_ORDER)
    )
    denom = grouped["detected_candidate_count"].replace(0, np.nan)
    tp = grouped["tp"] / denom
    wrong = grouped["wrong_range_fp_count"] / denom
    other = (grouped["fp"] - grouped["wrong_range_fp_count"]).clip(lower=0) / denom
    y = np.arange(len(METHOD_ORDER))
    left = np.zeros(len(METHOD_ORDER))
    ax.barh(y, tp.to_numpy(float), height=0.54, color=RED, alpha=0.90, edgecolor="white", lw=0.30, label="true positive")
    left += tp.fillna(0).to_numpy(float)
    ax.barh(y, wrong.to_numpy(float), left=left, height=0.54, color=AMBER, alpha=0.95, edgecolor="white", lw=0.30, label="wrong-range FP")
    left += wrong.fillna(0).to_numpy(float)
    ax.barh(y, other.to_numpy(float), left=left, height=0.54, color="#D8D0C7", alpha=0.96, edgecolor="white", lw=0.30, label="other FP")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(METHOD_ORDER) - 0.45, -1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.4)
    highlight_scalecomm(ax, axis="y")
    ax.set_xlabel("fraction of exact top 5% calls")
    ax.set_title("Composition of selected calls", loc="left", pad=5, fontsize=7.3)
    ax.grid(axis="x", color=GRID, lw=0.48)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    ax.tick_params(axis="y", length=0)
    despine(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.00, 0.995), ncol=3, handlelength=1.1, columnspacing=0.82, borderaxespad=0)


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
    add_nan_hatching(ax, mean)
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels(RANGE_LABELS, fontsize=5.6)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.6)
    highlight_scalecomm(ax, axis="y")
    ax.add_patch(Rectangle((-0.5, -0.5), len(RANGE_ORDER), 1, fill=False, ec=RED, lw=1.0))
    ax.set_title("Wrong-range FP modes", loc="left", pad=5, fontsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#2B2826")
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


def heatmap_grid(ax: plt.Axes, n_rows: int, n_cols: int, lw: float = 0.62) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=lw)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_nan_hatching(ax: plt.Axes, table: pd.DataFrame) -> None:
    values = table.to_numpy(float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                continue
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="#F0F0F0",
                    edgecolor="#B8B8B8",
                    linewidth=0.32,
                    hatch="////",
                    zorder=5,
                )
            )


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
