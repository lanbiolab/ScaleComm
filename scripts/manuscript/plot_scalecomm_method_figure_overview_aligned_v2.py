#!/usr/bin/env python
"""Draw ScaleComm method overview aligned to overview.png.

The canvas uses the same 1691 x 930 coordinate system as the provided
reference. The figure is redrawn with matplotlib vector objects rather than
embedding the reference image into the manuscript-ready output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/manuscript_method_figure_overview_aligned_v2"
REFERENCE = ROOT / "overview.png"

W, H = 1691, 930
TEXT = "#111111"
MUTED = "#555555"
LIGHT_TEXT = "#777777"
GRID = "#8D8D8D"
EDGE = "#222222"
RED = "#E46C68"
BLUE = "#77A9E5"
TEAL = "#138A7A"
TEAL2 = "#61BDB2"
GREEN = "#B8C8AA"
PURPLE = "#C59ACD"
ORANGE = "#E9B061"
PINK = "#DD7FA5"
GREY = "#D5D5D5"
CMAP = LinearSegmentedColormap.from_list("scalecomm_teal", ["#F5F5F2", "#B7DDD7", "#118A79"])
RED_BLUE = LinearSegmentedColormap.from_list("expr", ["#4E78B8", "#F7F7F7", "#D95B5B"])


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.91, 9.30))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    draw_method_overview(ax)
    scale_all_text(ax, 0.63)
    export(fig, OUT / "scalecomm_method_overview_aligned_v2")

    # Exact raster reference package for direct visual comparison only.
    if REFERENCE.exists():
        img = plt.imread(REFERENCE)
        ref_fig, ref_ax = plt.subplots(figsize=(16.91, 9.30))
        ref_ax.imshow(img)
        ref_ax.axis("off")
        export(ref_fig, OUT / "overview_exact_reference")

    print(f"Wrote {OUT / 'scalecomm_method_overview_aligned_v2.pdf'}")


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_method_overview(ax: plt.Axes) -> None:
    # Panel a header.
    ax.text(15, 35, "a", fontsize=26, fontweight="bold", ha="left", va="center", color=TEXT)
    ax.text(60, 31, "ScaleComm learns LR-specific spatial scale and context", fontsize=22, fontweight="bold", ha="left", va="center", color=TEXT)

    top_titles = [
        (52, "1", "Spatial transcriptomics input"),
        (384, "2", "Spatial graph"),
        (642, "3", "Graph wavelet context"),
        (994, "4", "LR-specific scale gate"),
        (1258, "5", "Context gate"),
        (1502, "6", "Edge score"),
    ]
    for x, num, title in top_titles:
        ax.text(x, 80, num, fontsize=17, fontweight="bold", ha="left", va="center", color=TEXT)
        ax.text(x + 28, 80, title, fontsize=15, ha="left", va="center", color=TEXT)

    draw_input(ax)
    grey_arrow(ax, 292, 306, 340, 306)
    draw_spatial_graph(ax)
    grey_arrow(ax, 582, 306, 622, 306)
    draw_wavelet(ax)
    grey_arrow(ax, 890, 306, 932, 306)
    draw_scale_gate(ax)
    grey_arrow(ax, 1222, 306, 1260, 306)
    draw_context_gate(ax)
    grey_arrow(ax, 1430, 306, 1470, 306)
    draw_edge_score(ax)

    # Panel b header.
    ax.text(15, 522, "b", fontsize=26, fontweight="bold", ha="left", va="center", color=TEXT)
    ax.text(60, 522, "Interpretable communication readouts", fontsize=22, fontweight="bold", ha="left", va="center", color=TEXT)
    bottom_titles = [
        (60, "1", "Spatial null / FDR"),
        (490, "2", "LR communication map"),
        (850, "3", "Sender and receiver cells"),
        (1245, "4", "Scale programs and response"),
    ]
    for x, num, title in bottom_titles:
        ax.text(x, 572, num, fontsize=17, fontweight="bold", ha="left", va="center", color=TEXT)
        ax.text(x + 28, 572, title, fontsize=15, ha="left", va="center", color=TEXT)

    draw_null_fdr(ax)
    draw_comm_map(ax)
    draw_sender_receiver(ax)
    draw_program_response(ax)


def draw_input(ax: plt.Axes) -> None:
    ax.text(150, 116, "Spatial map", fontsize=13, ha="center", va="center")
    pts = tissue_cloud(430, seed=1)
    x = 42 + 235 * pts[:, 0]
    y = 137 + 125 * pts[:, 1]
    # Five broad regions.
    colors = np.where(
        pts[:, 0] < 0.25,
        "#F1B46A",
        np.where(pts[:, 1] > 0.55, "#B484B9", np.where(pts[:, 0] > 0.72, "#B9A0C5", np.where(pts[:, 1] < 0.28, "#A8C083", "#6FA0D3"))),
    )
    ax.scatter(x, y, s=26, c=colors, edgecolors="#746B72", linewidths=0.32, alpha=0.86, zorder=3)

    ax.text(55, 315, "Expression matrix\n(genes x spots)", fontsize=11, ha="left", va="top")
    rng = np.random.default_rng(2)
    mat = rng.normal(0, 0.8, (7, 7))
    hm_x, hm_y, cell = 55, 340, 14
    for i in range(7):
        for j in range(7):
            ax.add_patch(Rectangle((hm_x + j * cell, hm_y + i * cell), cell, cell, fc=RED_BLUE((mat[i, j] + 2) / 4), ec="white", lw=0.55))
    ax.text(23, hm_y + 13, r"$G_1$", fontsize=11)
    ax.text(23, hm_y + 37, r"$G_2$", fontsize=11)
    ax.text(23, hm_y + 61, r"$G_3$", fontsize=11)
    ax.text(25, hm_y + 86, "...", fontsize=12)
    ax.text(23, hm_y + 108, r"$G_n$", fontsize=11)
    grad = np.linspace(1, 0, 80).reshape(-1, 1)
    ax.imshow(grad, cmap=RED_BLUE, extent=(164, 172, 340, 428), aspect="auto")
    ax.text(176, 350, "High", fontsize=10, va="center")
    ax.text(176, 420, "Low", fontsize=10, va="center")

    ax.text(228, 315, "LR list", fontsize=12, ha="left", va="top")
    for i, lr in enumerate([r"$L_1$–$R_1$", r"$L_2$–$R_2$", r"$L_3$–$R_3$", r"$L_4$–$R_4$", "..."]):
        ax.text(228, 348 + i * 28, lr, fontsize=13, ha="left", va="center")


def draw_spatial_graph(ax: plt.Axes) -> None:
    nodes = np.array(
        [
            [365, 205],
            [410, 165],
            [475, 185],
            [535, 178],
            [406, 286],
            [475, 284],
            [535, 287],
            [362, 385],
            [420, 405],
            [487, 387],
            [548, 358],
            [405, 240],
        ],
        dtype=float,
    )
    cols = [PURPLE, RED, GREEN, TEAL2, PINK, BLUE, PURPLE, BLUE, GREEN, ORANGE, PURPLE, "#F5A7A7"]
    edges = [(0, 1), (0, 7), (1, 4), (1, 11), (4, 5), (4, 8), (5, 6), (5, 9), (6, 3), (6, 10), (7, 8), (8, 9), (9, 10), (11, 1), (11, 5), (2, 5), (3, 6)]
    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color="#B8B8B8", lw=1.0, zorder=1)
    for idx, (p, c) in enumerate(zip(nodes, cols, strict=True)):
        if idx in {0, 8, 10}:
            draw_star_cell(ax, p[0], p[1], c, r=16)
        else:
            draw_cell(ax, p[0], p[1], c, r=13, angle=(idx * 23) % 70 - 35)
    ax.text(432, 150, r"Sender ($i$)", fontsize=11, color="#D33532", ha="left")
    ax.text(505, 258, r"Receiver ($j$)", fontsize=11, color="#2C61B7", ha="left")
    arrow(ax, 418, 175, 477, 270, color=TEAL, lw=2.0, scale=14)
    arrow(ax, 405, 245, 472, 278, color=TEAL, lw=2.0, scale=14)


def draw_wavelet(ax: plt.Axes) -> None:
    cx, cy = 725, 272
    for r in [37, 79, 121]:
        ax.add_patch(Circle((cx, cy), r, fc="none", ec="#777777", lw=1.1, ls=(0, (4, 3))))
    ax.add_patch(Circle((cx, cy), 151, fc="none", ec="#777777", lw=1.0, ls=(0, (4, 3))))
    ring_nodes = [
        (655, 170, GREEN),
        (746, 152, RED),
        (812, 186, PURPLE),
        (638, 255, TEAL2),
        (760, 240, PURPLE),
        (673, 340, RED),
        (735, 333, GREEN),
        (815, 322, TEAL2),
        (703, 276, BLUE),
    ]
    for i, (x, y, c) in enumerate(ring_nodes):
        if i in {1, 4}:
            draw_cell(ax, x, y, c, r=10, angle=i * 15)
        else:
            draw_star_cell(ax, x, y, c, r=12)
    draw_cell(ax, cx, cy, BLUE, r=17)
    for tx, ty, label in [(840, 160, "Local\n($s_0$)"), (840, 245, "Neighbor\n($s_1$)"), (840, 320, "Niche\n($s_2$)")]:
        ax.plot([cx + 25, tx - 8], [cy, ty], color="#B0B0B0", lw=1.0)
        ax.text(tx, ty, label, fontsize=11, ha="left", va="center")
    # Bar chart.
    base_x, base_y = 615, 437
    ax.plot([base_x, base_x + 240], [base_y, base_y], color="#777777", lw=0.8)
    heights = [58, 38, 25, 17]
    bar_cols = [TEAL, "#58B1A7", "#9CCBC6", "#D5E4E1"]
    for i, (h, c) in enumerate(zip(heights, bar_cols, strict=True)):
        bx = base_x + 20 + i * 61
        ax.add_patch(Rectangle((bx, base_y - h), 26, h, fc=c, ec="#44736E", lw=0.6))
        ax.text(bx + 13, base_y + 20, rf"$s_{i}$", fontsize=10, ha="center")
    ax.text(base_x + 115, base_y + 45, "Wavelet coefficients", fontsize=10, ha="center")


def draw_scale_gate(ax: plt.Axes) -> None:
    ax.text(1030, 126, r"$s_0$", fontsize=10, ha="center")
    ax.text(1030, 147, "(Local)", fontsize=8, ha="center")
    for x, s, lab in [(1082, r"$s_1$", "(Neighbor)"), (1135, r"$s_2$", "(Intermediate)"), (1189, r"$s_3$", "(Niche)")]:
        ax.text(x, 126, s, fontsize=10, ha="center")
        ax.text(x, 147, lab, fontsize=7.5, ha="center")
    row_labels = ["Contact\n($L_1$–$R_1$)", "Short-range\n($L_2$–$R_2$)", "Medium-range\n($L_3$–$R_3$)", "Long-niche\n($L_4$–$R_4$)"]
    values = np.eye(4) * 0.82 + np.array([[0.10, 0, 0, 0], [0, 0.04, 0, 0], [0, 0, 0.05, 0], [0, 0, 0, 0.10]])
    x0, y0, cw, ch = 1008, 165, 51, 54
    for i, lab in enumerate(row_labels):
        ax.text(995, y0 + i * ch + ch / 2, lab, fontsize=10, ha="right", va="center")
        for j in range(4):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * ch), cw, ch, fc=CMAP(values[i, j]), ec="#707070", lw=0.8))
    # Colorbar.
    grad = np.linspace(0, 1, 120).reshape(1, -1)
    ax.imshow(grad, cmap=CMAP, extent=(1027, 1169, 417, 430), aspect="auto")
    ax.text(1002, 425, "Low", fontsize=10, ha="left", va="center")
    ax.text(1175, 425, "High", fontsize=10, ha="left", va="center")


def draw_context_gate(ax: plt.Axes) -> None:
    ax.text(1253, 135, "Compatible local niche (accepted)", fontsize=11, color=TEAL, ha="left")
    ax.text(1260, 321, "Wrong context (suppressed)", fontsize=11, color="#D84B45", ha="left")
    # Accepted.
    for x, y, c in [(1296, 179, PURPLE), (1262, 218, RED), (1348, 218, BLUE), (1395, 188, GREEN), (1287, 274, PURPLE), (1398, 274, GREEN)]:
        draw_cell(ax, x, y, c, r=12, angle=20)
    ax.add_patch(Ellipse((1347, 218), 105, 72, fc="none", ec=TEAL, lw=1.1, ls=(0, (4, 3))))
    arrow(ax, 1278, 220, 1335, 220, color=TEAL, lw=1.8, scale=13)
    ax.plot([1426, 1445, 1480], [225, 248, 198], color=TEAL, lw=3)
    # Suppressed.
    for x, y, c in [(1296, 357, GREY), (1262, 389, RED), (1353, 389, PURPLE), (1412, 360, PURPLE), (1288, 442, PURPLE), (1414, 432, GREY)]:
        draw_cell(ax, x, y, c, r=12, angle=-20, alpha=0.62)
    ax.add_patch(Ellipse((1346, 390), 105, 72, fc="none", ec="#BDBDBD", lw=1.0, ls=(0, (4, 3))))
    arrow(ax, 1278, 390, 1335, 390, color="#E0A2A0", lw=1.3, scale=12, alpha=0.65)
    ax.plot([1427, 1446], [390, 408], color="#F05B54", lw=2.2)
    ax.plot([1446, 1427], [390, 408], color="#F05B54", lw=2.2)


def draw_edge_score(ax: plt.Axes) -> None:
    ax.text(1488, 128, "Ligand (L)", fontsize=10, color="#D74B46", ha="left")
    ax.text(1580, 128, "Receptor (R)", fontsize=10, color="#3264C8", ha="left")
    draw_ligand_icon(ax, 1518, 165, RED)
    draw_receptor_icon(ax, 1623, 165, BLUE)
    ax.text(1560, 208, r"score($i$, $j$, $l$) =", fontsize=14, ha="center")
    ax.text(1560, 239, "f(sender, receiver, LR,\ndistance, scale gate,\ncontext gate)", fontsize=11, ha="center", va="center", linespacing=1.35)
    ax.text(1490, 324, r"Top LR edges (to receiver $j$)", fontsize=10, ha="left")
    lrs = [r"$L_1$–$R_1$", r"$L_3$–$R_3$", r"$L_2$–$R_2$", r"$L_4$–$R_4$", "..."]
    vals = [132, 94, 73, 68]
    for i, lr in enumerate(lrs):
        yy = 356 + i * 26
        ax.text(1490, yy, lr, fontsize=11, ha="left", va="center")
        if i < 4:
            ax.add_patch(Rectangle((1532, yy - 8), vals[i], 12, fc=TEAL, ec="none"))
    ax.plot([1532, 1665], [446, 446], color="#555555", lw=0.8)
    ax.text(1530, 465, "0", fontsize=10, ha="center")
    ax.text(1666, 465, "1", fontsize=10, ha="center")
    ax.text(1600, 465, "Score", fontsize=11, ha="center")


def draw_null_fdr(ax: plt.Axes) -> None:
    ax.text(62, 615, "Real neighborhood", fontsize=11, ha="left")
    ax.text(302, 615, "Null (permuted)", fontsize=11, ha="left")
    # Real neighborhood.
    center = (132, 724)
    real = [(70, 680, PURPLE), (87, 625, RED), (130, 650, RED), (180, 671, PINK), (210, 724, BLUE), (145, 775, ORANGE), (82, 735, PURPLE)]
    for x, y, c in real:
        draw_cell(ax, x, y, c, r=14, angle=20)
        arrow(ax, x + 7, y + 4, center[0] - 8, center[1], color=TEAL, lw=1.5, scale=10)
    draw_cell(ax, *center, BLUE, r=15)
    # Permuted.
    null = [(317, 661, BLUE), (377, 661, ORANGE), (353, 735, GREEN), (270, 723, GREEN), (437, 723, PURPLE), (430, 775, ORANGE), (354, 790, GREEN), (390, 724, RED)]
    for x, y, c in null:
        draw_cell(ax, x, y, c, r=13, angle=-25)
    grey_arrow(ax, 205, 724, 253, 724)
    grey_arrow(ax, 253, 724, 205, 724)
    ax.text(228, 745, "Permute\nneighbors", fontsize=10, ha="center")
    ax.add_patch(Rectangle((79, 805), 276, 45, fc="#F7F7F7", ec="#C8C8C8", lw=1.0))
    ax.text(217, 833, r"$FDR \;=\; P_{\mathrm{null}}(\mathrm{score}\geq \mathrm{score}_{\mathrm{real}})$", fontsize=13, ha="center", va="center")


def draw_comm_map(ax: plt.Axes) -> None:
    pts = tissue_cloud(780, seed=7)
    x = 486 + 235 * pts[:, 0]
    y = 612 + 235 * pts[:, 1]
    strength = np.exp(-((pts[:, 0] - 0.62) ** 2 / 0.050 + (pts[:, 1] - 0.54) ** 2 / 0.040))
    ax.scatter(x, y, s=10, c=[CMAP(0.12 + 0.85 * s) if s > 0.05 else "#D7D7D7" for s in strength], edgecolors="#9B9B9B", linewidths=0.22, alpha=0.95)
    ax.add_patch(Ellipse((637, 736), 105, 145, angle=5, fc="none", ec="#F05B54", lw=1.9, ls=(0, (5, 4))))
    ax.imshow(np.linspace(1, 0, 100).reshape(-1, 1), cmap=CMAP, extent=(740, 758, 663, 803), aspect="auto")
    ax.text(765, 665, "High", fontsize=10, va="center")
    ax.text(765, 800, "Low", fontsize=10, va="center")


def draw_sender_receiver(ax: plt.Axes) -> None:
    sender = [(840, 655, RED), (902, 660, RED), (950, 650, RED), (1006, 660, RED), (1065, 653, RED)]
    receiver = [(835, 790, BLUE), (899, 783, BLUE), (958, 800, TEAL2), (1030, 790, BLUE), (1090, 783, BLUE)]
    for x1, y1, _ in sender:
        for x2, y2, _ in receiver:
            ax.plot([x1, x2], [y1 + 16, y2 - 16], color="#CFCFCF", lw=1.0, zorder=1)
    for i, (x, y, c) in enumerate(sender):
        if i in {1, 3}:
            draw_star_cell(ax, x, y, c, r=16)
        else:
            draw_cell(ax, x, y, c, r=15, angle=i * 18)
    for i, (x, y, c) in enumerate(receiver):
        if i == 1:
            draw_star_cell(ax, x, y, c, r=17)
        else:
            draw_cell(ax, x, y, c, r=16, angle=-25 + 10 * i)
    ax.text(928, 617, "Senders", fontsize=11, color="#D84B45", ha="center")
    ax.text(950, 848, "Receivers", fontsize=13, color="#2C61B7", ha="center")


def draw_program_response(ax: plt.Axes) -> None:
    ax.text(1210, 615, "LR-by-scale program (learned)", fontsize=11, ha="left")
    row_labs = [r"$L_1$–$R_1$", r"$L_2$–$R_2$", r"$L_3$–$R_3$", r"$L_4$–$R_4$", "..."]
    x0, y0, cw, ch = 1210, 632, 48, 45
    vals = np.eye(4) * 0.82
    for i in range(4):
        ax.text(1200, y0 + i * ch + ch / 2, row_labs[i], fontsize=10, ha="right", va="center")
        for j in range(4):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * ch), cw, ch, fc=CMAP(vals[i, j]), ec="#6C6C6C", lw=0.8))
    ax.text(1200, y0 + 4 * ch + ch / 2, row_labs[-1], fontsize=10, ha="right", va="center")
    for j, lab in enumerate([r"$s_0$", r"$s_1$", r"$s_2$", r"$s_3$"]):
        ax.text(x0 + j * cw + cw / 2, y0 + 4 * ch + 23, lab, fontsize=10, ha="center")
    ax.text(x0 + 24, y0 + 4 * ch + 45, "(Local)", fontsize=7, ha="center")
    ax.text(x0 + 72, y0 + 4 * ch + 45, "(Neighbor)", fontsize=7, ha="center")
    ax.text(x0 + 120, y0 + 4 * ch + 45, "(Intermediate)", fontsize=7, ha="center")
    ax.text(x0 + 168, y0 + 4 * ch + 45, "(Niche)", fontsize=7, ha="center")
    ax.imshow(np.linspace(0, 1, 110).reshape(1, -1), cmap=CMAP, extent=(1225, 1355, 860, 872), aspect="auto")
    ax.text(1215, 872, "Low", fontsize=9, ha="right", va="center")
    ax.text(1360, 872, "High", fontsize=9, ha="left", va="center")

    ax.text(1508, 615, "Communication score\nvs. receiver response", fontsize=11, ha="center")
    ox, oy = 1455, 805
    ax.plot([ox, ox], [oy, 655], color="#777777", lw=1.0)
    ax.plot([ox, 1665], [oy, oy], color="#777777", lw=1.0)
    arrow(ax, ox, 655, ox, 645, color="#777777", lw=1.0, scale=8)
    arrow(ax, 1665, oy, 1675, oy, color="#777777", lw=1.0, scale=8)
    rng = np.random.default_rng(8)
    xx = np.linspace(0.05, 0.95, 55)
    yy = 0.20 + 0.62 * xx + rng.normal(0, 0.14, xx.size)
    yy = np.clip(yy, 0.05, 0.95)
    ax.scatter(ox + 20 + xx * 165, oy - 15 - yy * 130, s=16, c=TEAL, alpha=0.85, edgecolors="white", linewidths=0.25)
    ax.plot([ox + 20, ox + 185], [oy - 43, oy - 150], color="#F05B54", lw=1.7)
    ax.text(1430, 730, "Communication score", fontsize=11, rotation=90, ha="center", va="center")
    ax.text(1567, 850, "Receiver target response", fontsize=11, ha="center")
    ax.text(1600, 778, r"$R=0.63$" + "\n" + r"$P<1e$-6", fontsize=10, ha="left", va="top")


def tissue_cloud(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.random(2)
        x, y = p[0], p[1]
        # Kidney-like cloud, close to the overview reference.
        dx = (x - 0.52) / 0.54
        dy = (y - 0.50) / 0.42
        left = ((x - 0.22) / 0.28) ** 2 + ((y - 0.52) / 0.38) ** 2 < 1
        body = dx * dx + dy * dy < 1.0
        notch = x < 0.24 and y < 0.30
        if (left or body) and not notch:
            pts.append(p)
    return np.array(pts)


def draw_cell(ax: plt.Axes, x: float, y: float, color: str, r: float = 12, angle: float = 0, alpha: float = 1.0) -> None:
    ax.add_patch(Ellipse((x, y), 1.65 * r, 2.25 * r, angle=angle, fc=color, ec=EDGE, lw=1.35, alpha=alpha, zorder=4))
    ax.add_patch(Ellipse((x + 0.12 * r, y), 0.68 * r, 0.40 * r, angle=angle, fc="#EFE8E0", ec=EDGE, lw=0.55, alpha=0.92 * alpha, zorder=5))


def draw_star_cell(ax: plt.Axes, x: float, y: float, color: str, r: float = 14, alpha: float = 1.0) -> None:
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radii = np.where(np.arange(12) % 2 == 0, r * 1.15, r * 0.72)
    pts = np.column_stack([x + radii * np.cos(angles), y + radii * np.sin(angles)])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=EDGE, lw=1.25, alpha=alpha, zorder=4))
    ax.add_patch(Circle((x + 1, y), r * 0.30, fc="#EFE8E0", ec=EDGE, lw=0.50, alpha=0.92 * alpha, zorder=5))


def draw_ligand_icon(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.add_patch(Circle((x, y), 13, fc=color, ec="#8A2F2F", lw=1.2))
    ax.plot([x, x], [y + 13, y + 43], color=color, lw=6, solid_capstyle="round")
    ax.plot([x - 8, x + 8], [y + 30, y + 30], color=color, lw=4, solid_capstyle="round")


def draw_receptor_icon(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.plot([x - 11, x - 11, x, x + 11, x + 11], [y - 12, y + 8, y + 18, y + 8, y - 12], color=color, lw=5, solid_capstyle="round")
    ax.add_patch(Circle((x, y + 18), 5, fc=color, ec="#315B96", lw=1))


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, *, color: str = EDGE, lw: float = 1.2, scale: float = 12, alpha: float = 1.0) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=scale, color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0))


def grey_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow(ax, x0, y0, x1, y1, color="#888888", lw=2.0, scale=24)


def scale_all_text(ax: plt.Axes, factor: float) -> None:
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() * factor)


if __name__ == "__main__":
    main()
