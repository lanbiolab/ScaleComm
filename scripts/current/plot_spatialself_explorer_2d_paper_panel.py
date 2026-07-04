#!/usr/bin/env python
"""Render a static 2D paper panel for the ScaleComm Explorer.

This figure is not a mock-up. It uses the same compact payload as the web
explorer and renders a publication-style static panel that is easier to place
in a manuscript than a WebGL screenshot.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np

from scripts.current.build_spatialself_explorer_static import build_payload


OUT_DIR = ROOT / "outputs/current_web_ui"
LR = "Cxcl12-Cxcr4"
TOP_FRACTION = 0.20
SENDER = "#237A78"
RECEIVER = "#B85450"
EDGE = "#29353A"
BG = "#CBD3D8"
INK = "#172126"
MUTED = "#67757D"
LINE = "#DCE5E8"
ACCENT = "#6C9E68"
CELL_HIGHLIGHT = "#93CC82"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def visible_indices(edges: dict, top_fraction: float = TOP_FRACTION) -> np.ndarray:
    scores = np.asarray(edges["score"], dtype=float)
    order = np.argsort(-scores)
    n = min(len(order), max(20, int(np.ceil(len(order) * top_fraction))))
    return order[:n]


def curved_segments(edges: dict, idx: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    scores = np.asarray(edges["score"], dtype=float)[idx]
    if len(scores) == 0:
        return [], np.array([])
    lo, hi = float(np.min(scores)), float(np.max(scores))
    span = max(hi - lo, 1e-12)
    segments: list[np.ndarray] = []
    edge_strength = []
    for j, edge_idx in enumerate(idx):
        x0 = float(edges["sx"][edge_idx])
        y0 = float(edges["sy"][edge_idx])
        x1 = float(edges["rx"][edge_idx])
        y1 = float(edges["ry"][edge_idx])
        dx, dy = x1 - x0, y1 - y0
        length = max(float(np.hypot(dx, dy)), 1e-9)
        px, py = -dy / length, dx / length
        sign = 1 if j % 2 == 0 else -1
        bend = sign * (0.010 + 0.014 * ((j % 11) / 10))
        t = np.linspace(0, 1, 28)
        arc = np.sin(np.pi * t)
        xs = (1 - t) * x0 + t * x1 + bend * px * arc
        ys = (1 - t) * y0 + t * y1 + bend * py * arc
        segments.append(np.column_stack([xs, ys]))
        edge_strength.append((float(edges["score"][edge_idx]) - lo) / span)
    return segments, np.asarray(edge_strength)


def draw_spatial_panel(ax: plt.Axes, payload: dict, lr_data: dict, idx: np.ndarray) -> None:
    bg = payload["background"]
    x = np.asarray(bg["x"], dtype=float)
    y = np.asarray(bg["y"], dtype=float)
    cell_type = np.asarray(bg["cell_type"], dtype=object)

    unannotated = cell_type == "unannotated"
    ax.scatter(x[unannotated], y[unannotated], s=0.9, c=BG, alpha=0.12, linewidths=0, rasterized=True)
    annotated = ~unannotated
    ax.scatter(x[annotated], y[annotated], s=1.45, c="#B9C1D1", alpha=0.14, linewidths=0, rasterized=True)

    edges = lr_data["edges"]
    involved = np.zeros(len(x), dtype=bool)
    involved_types = set()
    for i in idx:
        involved_types.add(str(edges["sender_type"][i]))
        involved_types.add(str(edges["receiver_type"][i]))
    for t in involved_types:
        if t != "unannotated":
            involved |= cell_type == t
    ax.scatter(x[involved], y[involved], s=1.75, c=CELL_HIGHLIGHT, alpha=0.23, linewidths=0, rasterized=True)

    segments, strength = curved_segments(edges, idx)
    if segments:
        lc = LineCollection(
            segments,
            colors=EDGE,
            linewidths=0.18 + 0.42 * strength,
            alpha=0.13 + 0.32 * strength,
            zorder=5,
        )
        ax.add_collection(lc)

    sx = np.asarray(edges["sx"], dtype=float)[idx]
    sy = np.asarray(edges["sy"], dtype=float)[idx]
    rx = np.asarray(edges["rx"], dtype=float)[idx]
    ry = np.asarray(edges["ry"], dtype=float)[idx]
    score = np.asarray(edges["score"], dtype=float)[idx]
    score_norm = (score - score.min()) / max(float(score.max() - score.min()), 1e-12)
    ax.scatter(sx, sy, s=2.0 + 3.2 * score_norm, c=SENDER, alpha=0.86, linewidths=0.12, edgecolors="white", zorder=8, rasterized=True)
    ax.scatter(rx, ry, s=2.2 + 3.4 * score_norm, c=RECEIVER, alpha=0.86, linewidths=0.12, edgecolors="white", zorder=9, rasterized=True)

    ax.set_aspect("equal")
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.035, 1.035)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LINE)
        spine.set_linewidth(0.55)
    ax.set_title("ScaleComm communication explorer", loc="left", fontsize=7.8, fontweight="bold", color=INK, pad=5)

    legend_items = [
        ("cell / bead context", BG, "o"),
        ("sender", SENDER, "o"),
        ("receiver", RECEIVER, "o"),
        ("ScaleComm event", EDGE, "_"),
    ]
    handles = []
    labels = []
    for label, color, marker in legend_items:
        if marker == "_":
            h = mpl.lines.Line2D([0], [0], color=color, lw=1.0, alpha=0.75)
        else:
            h = mpl.lines.Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgewidth=0, markersize=4.2, alpha=0.9)
        handles.append(h)
        labels.append(label)
    leg = ax.legend(handles, labels, loc="lower left", frameon=True, fontsize=4.8, ncol=4, handlelength=0.95, columnspacing=0.58, borderpad=0.30, labelspacing=0.32)
    leg.get_frame().set_edgecolor(LINE)
    leg.get_frame().set_linewidth(0.45)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.92)


def draw_summary(ax: plt.Axes, payload: dict, lr_data: dict, idx: np.ndarray) -> None:
    ax.axis("off")
    visible = len(idx)
    pool = int(lr_data["embedded_edges"])
    scale = np.asarray(lr_data["scale"], dtype=float)
    scale_names = np.array(["local", "near", "mid", "wide", "broad", "global"])
    dominant = scale_names[int(np.argmax(scale))]
    ax.text(0.00, 0.98, "Communication axis", transform=ax.transAxes, fontsize=6.7, fontweight="bold", color=INK, va="top")
    ax.text(0.00, 0.78, LR, transform=ax.transAxes, fontsize=9.1, fontweight="bold", color=INK, va="top")
    ax.text(0.00, 0.58, f"{lr_data['mechanism']} | rank {lr_data['rank_q95']}", transform=ax.transAxes, fontsize=5.4, color=MUTED, va="top")
    ax.text(0.00, 0.40, f"{visible:,} / {pool:,} top events", transform=ax.transAxes, fontsize=6.4, fontweight="bold", color=SENDER, va="top")
    ax.text(0.00, 0.22, f"dominant scale: {dominant}", transform=ax.transAxes, fontsize=5.8, color=INK, va="top")


def draw_score(ax: plt.Axes, lr_data: dict, idx: np.ndarray) -> None:
    values = np.asarray(lr_data["edges"]["score"], dtype=float)
    shown = values[idx]
    bins = np.linspace(values.min(), values.max(), 26)
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges) * 0.86
    ax.bar(centers, counts, width=width, color="#B7C2C7", alpha=0.72, edgecolor="none")
    cutoff = float(np.min(shown))
    ax.axvline(cutoff, color=ACCENT, lw=1.0)
    ax.set_title("Displayed event pool", loc="left", fontsize=6.7, fontweight="bold", color=INK, pad=3)
    ax.text(0.99, 0.90, f"top 20% events\nn={len(shown):,}", transform=ax.transAxes, ha="right", va="top", fontsize=5.1, color=MUTED)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.spines["bottom"].set_linewidth(0.5)


def draw_scale(ax: plt.Axes, lr_data: dict) -> None:
    names = ["local", "near", "mid", "wide", "broad", "global"]
    vals = np.asarray(lr_data["scale"], dtype=float)
    y = np.arange(len(vals))
    ax.barh(y, vals, color="#E8EEF0", height=0.52, edgecolor="none")
    ax.barh(y, vals, color=[SENDER if i == int(np.argmax(vals)) else "#AFC6C3" for i in y], height=0.52, edgecolor="none", alpha=0.85)
    ax.set_yticks(y, names, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.85, vals.max() * 1.08))
    ax.set_title("Learned spatial scale", loc="left", fontsize=6.7, fontweight="bold", color=INK, pad=3)
    ax.set_xticks([])
    for yi, val in zip(y, vals):
        ax.text(max(0.015, val + 0.018), yi, f"{val:.2f}", va="center", fontsize=5.2, color=INK, fontweight="bold" if yi == int(np.argmax(vals)) else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_pairs(ax: plt.Axes, lr_data: dict) -> None:
    rows = lr_data["pair_summary"][:6]
    pairs = [r["pair"].replace("_", " ").replace("microglia macrophage", "microglia").replace("dentate neuron", "dentate") for r in rows]
    counts = np.asarray([r["n_edges"] for r in rows], dtype=float)
    y = np.arange(len(rows))
    ax.barh(y, counts, color="#E8EEF0", height=0.54, edgecolor="none")
    ax.barh(y, counts, color="#AEB8C8", height=0.54, edgecolor="none", alpha=0.86)
    ax.set_yticks(y, pairs, fontsize=5.0)
    ax.invert_yaxis()
    ax.set_title("Dominant cell-type interactions", loc="left", fontsize=6.7, fontweight="bold", color=INK, pad=3)
    ax.set_xticks([])
    xmax = max(counts) * 1.18
    ax.set_xlim(0, xmax)
    for yi, val in zip(y, counts):
        ax.text(xmax, yi, f"{int(val):,}", va="center", ha="right", fontsize=5.2, color=INK, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure() -> tuple[plt.Figure, dict]:
    setup_style()
    payload = build_payload()
    lr_data = payload["lrs"][LR]
    idx = visible_indices(lr_data["edges"], TOP_FRACTION)

    fig = plt.figure(figsize=(7.2, 3.65), dpi=300)
    ax_map = fig.add_axes([0.025, 0.080, 0.672, 0.845])
    draw_spatial_panel(ax_map, payload, lr_data, idx)

    ax_summary = fig.add_axes([0.735, 0.755, 0.245, 0.175])
    draw_summary(ax_summary, payload, lr_data, idx)
    ax_score = fig.add_axes([0.735, 0.538, 0.245, 0.145])
    draw_score(ax_score, lr_data, idx)
    ax_scale = fig.add_axes([0.735, 0.308, 0.245, 0.165])
    draw_scale(ax_scale, lr_data)
    ax_pairs = fig.add_axes([0.735, 0.075, 0.245, 0.168])
    draw_pairs(ax_pairs, lr_data)
    manifest = {
        "dataset": payload["dataset"],
        "lr": LR,
        "top_fraction": TOP_FRACTION,
        "visible_edges": int(len(idx)),
        "filter_pool": int(lr_data["embedded_edges"]),
        "annotated_pool": int(lr_data["annotated_edges"]),
        "rank_q95": lr_data["rank_q95"],
        "dominant_scale": ["local", "near", "mid", "wide", "broad", "global"][int(np.argmax(lr_data["scale"]))],
        "scale": lr_data["scale"],
        "top_pair": lr_data["pair_summary"][0] if lr_data["pair_summary"] else None,
    }
    return fig, manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, manifest = build_figure()
    base = OUT_DIR / "spatialself_explorer_2d_paper_panel"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=450 if ext == "png" else None)
    plt.close(fig)
    manifest_path = base.with_name(base.name + "_manifest.json")
    manifest["png"] = str(base.with_suffix(".png"))
    manifest["pdf"] = str(base.with_suffix(".pdf"))
    manifest["svg"] = str(base.with_suffix(".svg"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
