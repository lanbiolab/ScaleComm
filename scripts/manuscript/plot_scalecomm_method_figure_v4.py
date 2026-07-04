#!/usr/bin/env python
"""ScaleComm method overview figure in a CytoSignal-like schematic style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/manuscript_method_figure_scalecomm_v4"

TEXT = "#222222"
MUTED = "#6E6A65"
LIGHT = "#F7F5EF"
EDGE = "#2A2725"
GRID = "#E2DAD0"
BLUE = "#4F93BF"
TEAL = "#31908A"
GREEN = "#69B58C"
CORAL = "#D96A63"
ORANGE = "#E6A23A"
PURPLE = "#A46BB4"
YELLOW = "#F0C767"
GREY = "#CFCBC4"
PALETTE = [CORAL, BLUE, GREEN, PURPLE, ORANGE, TEAL]
HEAT = LinearSegmentedColormap.from_list("scalecomm_heat", ["#F6F1E6", "#E8B85E", "#C95A55", "#6B4E8E"])
TEALMAP = LinearSegmentedColormap.from_list("scalecomm_teal", ["#F4F1E8", "#9BC7C0", "#237A78"])


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.4, 7.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.020, 0.965, "a", fontsize=14, fontweight="bold", ha="left", va="top", color="#111111")
    ax.text(
        0.500,
        0.955,
        "ScaleComm: learning LR-specific spatial scale and context for cell-cell communication",
        fontsize=11.2,
        ha="center",
        va="top",
        color=TEXT,
    )

    draw_top_row(ax)
    draw_bottom_row(ax)

    stem = OUT / "scalecomm_method_figure_cytosignal_style_v4"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}")


def draw_top_row(ax: plt.Axes) -> None:
    y0, y1 = 0.555, 0.900
    modules = [
        (0.030, 0.170, "Spatial transcriptomics"),
        (0.230, 0.385, "Spatial graph"),
        (0.438, 0.592, "Graph wavelet context"),
        (0.640, 0.790, "LR-specific gates"),
        (0.842, 0.970, "Edge score"),
    ]
    for x0, x1, title in modules:
        ax.text((x0 + x1) / 2, y1 + 0.008, title, ha="center", va="bottom", fontsize=8.7, color=TEXT)
    draw_input_module(ax, modules[0][0], y0, modules[0][1] - modules[0][0], y1 - y0)
    draw_spatial_graph_module(ax, modules[1][0], y0, modules[1][1] - modules[1][0], y1 - y0)
    draw_wavelet_module(ax, modules[2][0], y0, modules[2][1] - modules[2][0], y1 - y0)
    draw_gate_module(ax, modules[3][0], y0, modules[3][1] - modules[3][0], y1 - y0)
    draw_score_module(ax, modules[4][0], y0, modules[4][1] - modules[4][0], y1 - y0)
    for x in [0.190, 0.408, 0.600, 0.812]:
        arrow(ax, x, 0.720, x + 0.036, 0.720, lw=1.8, color="#85817B", scale=18)


def draw_bottom_row(ax: plt.Axes) -> None:
    ax.text(0.020, 0.505, "b", fontsize=14, fontweight="bold", ha="left", va="top", color="#111111")
    ax.text(
        0.500,
        0.505,
        "Model readouts connect significance, spatial maps, interpretability and downstream response",
        fontsize=9.5,
        ha="center",
        va="top",
        color=TEXT,
    )
    modules = [
        (0.045, 0.230, "Spatial null and FDR"),
        (0.285, 0.455, "LR communication map"),
        (0.515, 0.690, "Sender and receiver cells"),
        (0.745, 0.940, "Scale programs and response"),
    ]
    for x0, x1, title in modules:
        ax.text((x0 + x1) / 2, 0.458, title, ha="center", va="bottom", fontsize=8.8, color=TEXT)
    draw_null_module(ax, modules[0][0], 0.120, modules[0][1] - modules[0][0], 0.320)
    draw_map_module(ax, modules[1][0], 0.110, modules[1][1] - modules[1][0], 0.330)
    draw_sender_receiver_module(ax, modules[2][0], 0.105, modules[2][1] - modules[2][0], 0.335)
    draw_program_response_module(ax, modules[3][0], 0.108, modules[3][1] - modules[3][0], 0.332)


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str = "#FFFFFF") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            fc=fc,
            ec="#E7DFD6",
            lw=0.55,
            zorder=0,
        )
    )


def draw_input_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    # Stylized expression matrix.
    ax.text(x + 0.020, y + h * 0.820, "Expression", fontsize=7.2, ha="left", va="center", color=MUTED)
    x0, y0 = x + 0.026, y + h * 0.600
    rng = np.random.default_rng(2)
    mat = rng.beta(1.2, 2.4, size=(8, 12))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(
                Rectangle(
                    (x0 + j * w * 0.0118, y0 + i * h * 0.027),
                    w * 0.0100,
                    h * 0.021,
                    fc=HEAT(mat[i, j]),
                    ec="none",
                    zorder=2,
                )
            )
    # Spatial positions.
    ax.text(x + 0.020, y + h * 0.420, "Coordinates", fontsize=7.2, ha="left", va="center", color=MUTED)
    pts = make_tissue_points(210, seed=4)
    px = x + w * (0.18 + 0.36 * pts[:, 0])
    py = y + h * (0.115 + 0.240 * pts[:, 1])
    c = np.where(pts[:, 0] + pts[:, 1] > 1.08, BLUE, np.where(pts[:, 0] < 0.45, GREEN, CORAL))
    ax.scatter(px, py, s=3.0, c=c, alpha=0.78, lw=0, zorder=2)
    # LR list.
    ax.text(x + w * 0.665, y + h * 0.730, "LR pairs", fontsize=7.2, ha="left", va="center", color=MUTED)
    for k, (txt, col) in enumerate([("L1-R1", CORAL), ("L2-R2", BLUE), ("L3-R3", GREEN), ("L4-R4", ORANGE)]):
        yy = y + h * (0.650 - 0.085 * k)
        ax.add_patch(Rectangle((x + w * 0.665, yy - h * 0.018), w * 0.030, h * 0.036, fc=col, ec=EDGE, lw=0.5))
        ax.text(x + w * 0.710, yy, txt, ha="left", va="center", fontsize=6.6, color=TEXT)


def draw_spatial_graph_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    centers = np.array(
        [
            [0.20, 0.30],
            [0.30, 0.66],
            [0.43, 0.44],
            [0.58, 0.69],
            [0.72, 0.32],
            [0.82, 0.58],
            [0.52, 0.23],
        ]
    )
    cols = [GREEN, CORAL, BLUE, PURPLE, ORANGE, TEAL, GREEN]
    for i, j in [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (2, 6)]:
        ax.plot(
            [x + w * centers[i, 0], x + w * centers[j, 0]],
            [y + h * centers[i, 1], y + h * centers[j, 1]],
            color="#B6B1AA",
            lw=0.9,
            zorder=1,
        )
    ax.add_patch(Ellipse((x + w * 0.49, y + h * 0.48), w * 0.70, h * 0.78, fc="#F3F8F6", ec="#333333", lw=0.9, ls=(0, (4, 3))))
    for p, col, angle in zip(centers, cols, [-20, 15, -8, 18, -22, 12, 8], strict=True):
        draw_cell(ax, x + w * p[0], y + h * p[1], col, angle=angle, size=0.026)
    # Candidate LR edge.
    arrow(ax, x + w * 0.30, y + h * 0.66, x + w * 0.43, y + h * 0.44, lw=1.4, color=CORAL, scale=12)
    arrow(ax, x + w * 0.43, y + h * 0.44, x + w * 0.72, y + h * 0.32, lw=1.4, color=BLUE, scale=12)
    ax.text(x + w * 0.58, y + h * 0.055, "candidate sender-receiver edges", ha="center", va="center", fontsize=6.5, color=MUTED)


def draw_wavelet_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    cx, cy = x + w * 0.47, y + h * 0.50
    draw_cell(ax, cx, cy, BLUE, angle=-10, size=0.030)
    for r, col, label in [(0.16, CORAL, "local"), (0.28, ORANGE, "neighbor"), (0.40, TEAL, "niche")]:
        ax.add_patch(Ellipse((cx, cy), w * r * 1.60, h * r * 1.85, fc=col, ec=col, lw=1.25, alpha=0.08, ls=(0, (3, 2))))
        ax.add_patch(Ellipse((cx, cy), w * r * 1.60, h * r * 1.85, fc="none", ec=col, lw=1.1, alpha=0.86, ls=(0, (3, 2))))
        ax.text(cx + w * (0.50 * r + 0.05), cy + h * (0.55 * r), label, fontsize=5.8, color=col, ha="left", va="center")
    ring_pts = np.array([[0.26, 0.70], [0.73, 0.69], [0.25, 0.28], [0.74, 0.30], [0.50, 0.78], [0.50, 0.20]])
    for k, p in enumerate(ring_pts):
        draw_cell(ax, x + w * p[0], y + h * p[1], PALETTE[k % len(PALETTE)], angle=20 - 8 * k, size=0.020)
    # Wavelet coefficients.
    bx, by = x + w * 0.700, y + h * 0.155
    for i, col in enumerate([CORAL, ORANGE, TEAL, BLUE]):
        ax.add_patch(Rectangle((bx + i * w * 0.042, by), w * 0.030, h * (0.10 + 0.045 * i), fc=col, ec=EDGE, lw=0.4, alpha=0.80))
        ax.text(bx + i * w * 0.042 + w * 0.015, by - h * 0.040, f"s{i}", ha="center", va="center", fontsize=5.5, color=MUTED)
    ax.text(x + w * 0.50, y + h * 0.085, "multi-scale context embedding", ha="center", va="center", fontsize=6.5, color=MUTED)


def draw_gate_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    # Scale gate heatmap.
    ax.text(x + w * 0.260, y + h * 0.800, "scale gate", fontsize=7.0, ha="center", va="center", color=TEXT)
    rng = np.random.default_rng(8)
    mat = np.array(
        [
            [0.88, 0.22, 0.08, 0.04],
            [0.18, 0.62, 0.33, 0.06],
            [0.08, 0.28, 0.72, 0.36],
            [0.02, 0.14, 0.42, 0.92],
        ]
    )
    mat += rng.normal(0, 0.025, mat.shape)
    x0, y0 = x + w * 0.088, y + h * 0.510
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x0 + j * w * 0.073, y0 - i * h * 0.060), w * 0.060, h * 0.048, fc=TEALMAP(np.clip(mat[i, j], 0, 1)), ec="#FFFFFF", lw=0.7))
    for j, lab in enumerate(["S0", "S1", "S2", "S3"]):
        ax.text(x0 + j * w * 0.073 + w * 0.030, y0 + h * 0.065, lab, ha="center", va="center", fontsize=5.3, color=MUTED)
    for i, lab in enumerate(["contact", "short", "medium", "niche"]):
        ax.text(x0 - w * 0.020, y0 - i * h * 0.060 + h * 0.024, lab, ha="right", va="center", fontsize=5.2, color=MUTED)
    # Context gate.
    ax.text(x + w * 0.735, y + h * 0.800, "context gate", fontsize=7.0, ha="center", va="center", color=TEXT)
    for k, yy in enumerate([0.610, 0.435]):
        col = TEAL if k == 0 else GREY
        alpha = 0.95 if k == 0 else 0.38
        draw_cell(ax, x + w * 0.620, y + h * yy, CORAL, angle=16, size=0.021, alpha=alpha)
        draw_cell(ax, x + w * 0.830, y + h * yy, BLUE, angle=-10, size=0.021, alpha=alpha)
        arrow(ax, x + w * 0.652, y + h * yy, x + w * 0.798, y + h * yy, lw=1.3, color=col, scale=12, alpha=alpha)
        txt = "kept" if k == 0 else "suppressed"
        ax.text(x + w * 0.735, y + h * (yy - 0.080), txt, ha="center", va="center", fontsize=5.8, color=col if k == 0 else MUTED)
    ax.text(x + w * 0.50, y + h * 0.085, "LR identity controls scale and context", ha="center", va="center", fontsize=6.5, color=MUTED)


def draw_score_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    draw_ligand_receptor(ax, x + w * 0.38, y + h * 0.580, scale=1.0)
    ax.text(
        x + w * 0.52,
        y + h * 0.300,
        "score(i,j,l) = f(sender, receiver,\nLR embedding, distance,\nscale gate, context gate)",
        ha="center",
        va="center",
        fontsize=6.7,
        color=TEXT,
        linespacing=1.4,
    )
    ax.add_patch(Rectangle((x + w * 0.170, y + h * 0.120), w * 0.660, h * 0.055, fc="#F1EEE8", ec="none"))
    ax.add_patch(Rectangle((x + w * 0.170, y + h * 0.120), w * 0.500, h * 0.055, fc=TEAL, ec="none", alpha=0.90))
    ax.text(x + w * 0.505, y + h * 0.080, "ranked LR edges", ha="center", va="center", fontsize=6.2, color=MUTED)


def draw_null_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    card(ax, x, y, w, h)
    for offset, title in [(0.28, "real"), (0.72, "null")]:
        ax.text(x + w * offset, y + h * 0.830, title, ha="center", va="center", fontsize=6.4, color=MUTED)
        cx = x + w * offset
        cy = y + h * 0.520
        for k, ang in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
            px = cx + w * 0.145 * np.cos(ang)
            py = cy + h * 0.195 * np.sin(ang)
            draw_cell(ax, px, py, PALETTE[k % len(PALETTE)], angle=18 * k, size=0.014)
        ax.add_patch(Ellipse((cx, cy), w * 0.355, h * 0.450, fc="none", ec="#333333", lw=0.8, ls=(0, (3, 3))))
    arrow(ax, x + w * 0.410, y + h * 0.520, x + w * 0.575, y + h * 0.520, lw=1.0, color="#77736D", scale=10)
    ax.text(x + w * 0.492, y + h * 0.585, "permute", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.text(x + w * 0.50, y + h * 0.155, "spatially constrained null", ha="center", va="center", fontsize=6.2, color=MUTED)


def draw_map_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    card(ax, x, y, w, h)
    pts = make_tissue_points(520, seed=12)
    strength = np.exp(-((pts[:, 0] - 0.58) ** 2 / 0.040 + (pts[:, 1] - 0.46) ** 2 / 0.030))
    px = x + w * (0.08 + 0.84 * pts[:, 0])
    py = y + h * (0.12 + 0.70 * pts[:, 1])
    bg = np.clip(0.10 + 0.90 * strength, 0, 1)
    ax.scatter(px, py, s=5.1, c=TEALMAP(bg), lw=0, alpha=0.96)
    ax.add_patch(Ellipse((x + w * 0.580, y + h * 0.445), w * 0.420, h * 0.355, fc=CORAL, ec="none", alpha=0.035))
    ax.add_patch(Ellipse((x + w * 0.580, y + h * 0.445), w * 0.420, h * 0.355, fc="none", ec=CORAL, lw=1.25, ls=(0, (4, 3))))
    ax.text(x + w * 0.500, y + h * 0.085, "LR-specific communication hotspot", ha="center", va="center", fontsize=6.2, color=MUTED)


def draw_sender_receiver_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    card(ax, x, y, w, h)
    sender_y = y + h * 0.650
    receiver_y = y + h * 0.330
    sender_x = [x + w * v for v in [0.18, 0.34, 0.50, 0.66, 0.82]]
    recv_x = [x + w * v for v in [0.22, 0.42, 0.62, 0.78]]
    for sx in sender_x:
        draw_cell(ax, sx, sender_y + np.random.default_rng(int(sx * 1000)).normal(0, h * 0.018), CORAL, angle=10, size=0.016)
    for rx in recv_x:
        draw_cell(ax, rx, receiver_y + np.random.default_rng(int(rx * 1000)).normal(0, h * 0.015), BLUE, angle=-10, size=0.017)
    for sx, rx in [(sender_x[0], recv_x[0]), (sender_x[1], recv_x[1]), (sender_x[2], recv_x[1]), (sender_x[3], recv_x[2]), (sender_x[4], recv_x[3])]:
        ax.plot([sx, rx], [sender_y - h * 0.030, receiver_y + h * 0.040], color=TEAL, lw=1.0, alpha=0.55)
    ax.text(x + w * 0.150, sender_y + h * 0.115, "sender", ha="center", va="center", fontsize=6.1, color=CORAL, zorder=10)
    ax.text(x + w * 0.155, receiver_y - h * 0.115, "receiver", ha="center", va="center", fontsize=6.1, color=BLUE, zorder=10)
    ax.text(x + w * 0.50, y + h * 0.085, "inferred sending and receiving cells", ha="center", va="center", fontsize=6.2, color=MUTED)


def draw_program_response_module(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    card(ax, x, y, w, h)
    # Learned scale program heatmap.
    hm_x, hm_y = x + w * 0.060, y + h * 0.540
    scale_mat = np.array(
        [
            [0.85, 0.24, 0.09, 0.05],
            [0.28, 0.66, 0.36, 0.10],
            [0.08, 0.32, 0.76, 0.42],
            [0.03, 0.13, 0.48, 0.91],
        ]
    )
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((hm_x + j * w * 0.048, hm_y - i * h * 0.058), w * 0.040, h * 0.048, fc=TEALMAP(scale_mat[i, j]), ec="white", lw=0.6))
    ax.text(hm_x + w * 0.095, y + h * 0.830, "scale weights", ha="center", va="center", fontsize=6.2, color=MUTED)
    # Response signature mini regression.
    bx = x + w * 0.360
    by = y + h * 0.580
    ax.text(bx + w * 0.200, y + h * 0.830, "receiver response", ha="center", va="center", fontsize=6.2, color=MUTED)
    rng = np.random.default_rng(14)
    vals = np.linspace(0.10, 0.92, 22)
    response = vals * 0.78 + rng.normal(0, 0.070, len(vals)) + 0.08
    yy = by - h * 0.315 + h * 0.285 * np.clip(response, 0, 1)
    ax.scatter(bx + w * 0.330 * vals, yy, s=11, c=TEAL, alpha=0.72, edgecolors="white", linewidths=0.25)
    ax.plot([bx + w * 0.030, bx + w * 0.330], [by - h * 0.285, by - h * 0.055], color=CORAL, lw=1.1)
    ax.plot([bx, bx], [by - h * 0.330, by], color="#333333", lw=0.6)
    ax.plot([bx, bx + w * 0.360], [by - h * 0.330, by - h * 0.330], color="#333333", lw=0.6)
    ax.text(x + w * 0.50, y + h * 0.085, "interpretable scales linked to target signatures", ha="center", va="center", fontsize=6.2, color=MUTED)


def draw_ligand_receptor(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    s = 0.018 * scale
    # Ligand.
    ax.plot([x - 2.0 * s, x - 2.0 * s, x - 1.05 * s], [y - 1.3 * s, y + 1.1 * s, y + 1.1 * s], color=ORANGE, lw=5.0 * scale, solid_capstyle="round")
    ax.plot([x - 3.0 * s, x - 3.0 * s], [y - 1.2 * s, y + 0.4 * s], color=ORANGE, lw=5.0 * scale, solid_capstyle="round")
    ax.plot([x - 1.0 * s, x - 1.0 * s], [y - 1.2 * s, y + 0.4 * s], color=ORANGE, lw=5.0 * scale, solid_capstyle="round")
    ax.text(x - 2.0 * s, y + 2.25 * s, "L", color=ORANGE, fontsize=7.0, ha="center", va="bottom", fontweight="bold")
    # Receptor.
    ax.plot([x + 1.0 * s, x + 1.0 * s, x + 2.0 * s, x + 2.0 * s], [y - 1.2 * s, y + 1.1 * s, y + 1.1 * s, y - 1.2 * s], color=BLUE, lw=5.0 * scale, solid_capstyle="round")
    ax.text(x + 1.55 * s, y + 2.25 * s, "R", color=BLUE, fontsize=7.0, ha="center", va="bottom", fontweight="bold")


def draw_cell(ax: plt.Axes, x: float, y: float, color: str, *, angle: float = 0, size: float = 0.020, alpha: float = 1.0) -> None:
    ax.add_patch(Ellipse((x, y), size * 1.45, size * 2.05, angle=angle, fc=color, ec=EDGE, lw=1.0, alpha=alpha, zorder=3))
    ax.add_patch(Ellipse((x + size * 0.12, y + size * 0.04), size * 0.55, size * 0.34, angle=angle, fc="#F3EDE4", ec=EDGE, lw=0.45, alpha=0.90 * alpha, zorder=4))


def arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    lw: float = 1.0,
    color: str = EDGE,
    scale: float = 12,
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def make_tissue_points(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.random(2)
        dx = (p[0] - 0.50) / 0.52
        dy = (p[1] - 0.48) / 0.42
        notch = p[1] < 0.25 and p[0] < 0.33
        if dx * dx + dy * dy < 1.0 and not notch:
            pts.append(p)
    return np.array(pts)


if __name__ == "__main__":
    main()
