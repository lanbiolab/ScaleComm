#!/usr/bin/env python
"""DSS MERFISH method comparison plus receiver-response validation figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (
    BASE,
    METHOD_COLORS as OLD_METHOD_COLORS,
    METHODS,
    STAGES,
    STAGE_LABELS,
    aggregate_receiver_scores,
    broad_compartment,
    load_expression_slice,
    load_slice,
    normalize_extent,
    prediction_path,
    receiver_scores,
    set_spatial_limits,
)
from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import CURATED_SIGNATURES


OUT = BASE / "figures/merfish_dss_method_downstream_v1"
PANEL_OUT = OUT / "panels"

LR_CASE = "Tnf-Tnfrsf1b"
SIGNATURE = "Tnf_nfkb_response"
INFLAMMATORY_AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
SPATIAL_STAGES = ["Healthy_D0", "DSS9", "DSS21"]
SPATIAL_LABELS = ["D0", "D9", "D21"]

INK = "#232323"
MUTED = "#6A665E"
GRID = "#EAE5DB"
BACKGROUND = "#D7D1C8"

METHOD_COLORS = {
    "SpatialSelf": "#9E332E",
    "COMMOT": "#2F7782",
    "CellNEST": "#6D6375",
}

COMPARTMENT_COLORS = {
    "Epithelial": "#D5963D",
    "Immune": "#B94E67",
    "Mesenchymal": "#4F9A8B",
    "Vascular": "#5D7FA7",
    "Neural": "#8B7BA4",
    "Other": "#BDB7AE",
}

ACTIVITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tnf_activity",
    ["#FCF7EF", "#E8BE83", "#C56246", "#6C1E1E"],
)
RESPONSE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "nfkb_response",
    ["#F7F3EC", "#B9D1C9", "#559A91", "#164F58"],
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.9,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.58,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.027, dy: float = 0.010) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)


def panel_title(fig: plt.Figure, ax: plt.Axes, title: str, dy: float = 0.033) -> None:
    box = ax.get_position()
    fig.text(box.x0, box.y1 + dy, title, ha="left", va="bottom", fontsize=7.8, color=INK)


def draw_context(ax: plt.Axes, cells: pd.DataFrame, size: float = 1.35, alpha: float = 0.22) -> None:
    ax.scatter(cells["x"], cells["y"], s=size, color=BACKGROUND, alpha=alpha, linewidths=0)


def add_low_high_bar(ax: plt.Axes, cmap, label: str) -> None:
    cax = ax.inset_axes([0.58, -0.112, 0.34, 0.055])
    for i in range(96):
        cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=cmap(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.8, labelsize=5.6)
    cax.set_xlabel(label, fontsize=5.7, labelpad=0.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)


def incoming_score_table(stage: str, lr_pair: str, method: str, replicate: int = 1) -> pd.DataFrame:
    cells = receiver_scores(stage, lr_pair, method=method, replicate=replicate)
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    return cells


def edge_table(stage: str, lr_pair: str, method: str, replicate: int = 1) -> pd.DataFrame:
    rep, cells, candidates = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    predictions = pd.read_csv(prediction_path(stage, rep, method))
    edges = candidates.loc[(candidates["ligand"] == ligand) & (candidates["receptor"] == receptor)].copy()
    edges = edges.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    xy = cells.set_index("cell_id")[["x", "y"]]
    edges = edges.join(xy.rename(columns={"x": "sender_x", "y": "sender_y"}), on="sender_id")
    edges = edges.join(xy.rename(columns={"x": "receiver_x", "y": "receiver_y"}), on="receiver_id")
    return edges.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y"])


def signature_cells(stage: str, replicate: int = 1) -> pd.DataFrame:
    cells = load_expression_slice(stage, replicate).copy()
    genes = [gene for gene in CURATED_SIGNATURES[SIGNATURE] if gene in cells.columns]
    cells["signature_score"] = np.log1p(cells[genes].to_numpy(float)).mean(axis=1)
    return cells


def draw_score_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    vmax: float,
    *,
    cmap=ACTIVITY_CMAP,
    low_clip: float = 0.055,
    title: str | None = None,
) -> None:
    draw_context(ax, cells, size=1.25, alpha=0.25)
    score = cells["incoming_score"].to_numpy(float)
    norm = np.clip(score / max(vmax, 1e-9), 0, 1)
    mask = norm > low_clip
    if mask.any():
        sizes = 3.2 + 15.5 * np.power(norm[mask], 0.62)
        ax.scatter(
            cells.loc[mask, "x"],
            cells.loc[mask, "y"],
            c=norm[mask],
            s=sizes,
            cmap=cmap,
            vmin=0,
            vmax=1,
            marker="o",
            alpha=0.94,
            edgecolors="white",
            linewidths=0.11,
            zorder=3,
        )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def draw_signature_map(ax: plt.Axes, cells: pd.DataFrame, vmax: float, title: str | None = None) -> None:
    score = cells["signature_score"].to_numpy(float)
    norm = np.clip(score / max(vmax, 1e-9), 0, 1)
    ax.scatter(cells["x"], cells["y"], s=1.15, color=BACKGROUND, alpha=0.20, linewidths=0)
    mask = norm > 0.020
    ax.scatter(
        cells.loc[mask, "x"],
        cells.loc[mask, "y"],
        c=norm[mask],
        s=2.8 + 6.5 * np.power(norm[mask], 0.75),
        cmap=RESPONSE_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.88,
        marker="o",
        edgecolors="none",
        zorder=3,
    )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def draw_commot_arrows(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame, title: str | None = None) -> None:
    draw_context(ax, cells, size=1.20, alpha=0.25)
    if not edges.empty:
        selected = edges.sort_values("score", ascending=False).head(24).copy()
        vmax = max(float(selected["score"].max()), 1e-9)
        for row in selected.itertuples(index=False):
            val = float(row.score) / vmax
            color = METHOD_COLORS["COMMOT"]
            ax.annotate(
                "",
                xy=(row.receiver_x, row.receiver_y),
                xytext=(row.sender_x, row.sender_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 0.35 + 0.75 * val,
                    "color": color,
                    "alpha": 0.18 + 0.58 * val,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                    "mutation_scale": 4.6 + 2.5 * val,
                },
                zorder=3,
            )
        ax.scatter(
            selected["receiver_x"],
            selected["receiver_y"],
            s=5.0,
            color=METHOD_COLORS["COMMOT"],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.10,
            zorder=4,
        )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def top_bottom_effects(stage: str = "DSS9", lr_pair: str = LR_CASE) -> pd.DataFrame:
    rows = []
    genes = CURATED_SIGNATURES[SIGNATURE]
    ligand, receptor = lr_pair.split("-")
    for replicate in [1, 2]:
        rep, _, candidates = load_slice(stage, replicate)
        expr_cells = load_expression_slice(stage, replicate).set_index("cell_id")
        present = [gene for gene in genes if gene in expr_cells.columns]
        signature = pd.Series(
            np.log1p(expr_cells[present].to_numpy(float)).mean(axis=1),
            index=expr_cells.index,
        )
        for method in METHODS:
            predictions = pd.read_csv(prediction_path(stage, rep, method))
            group = candidates.loc[(candidates["ligand"] == ligand) & (candidates["receptor"] == receptor)]
            group = group.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
            receiver_score = group.groupby("receiver_id")["score"].mean()
            common = signature.index.intersection(receiver_score.index)
            if len(common) < 10:
                continue
            ranked = receiver_score.loc[common].sort_values(ascending=False)
            k = max(5, int(np.floor(0.20 * len(ranked))))
            top_cells = ranked.head(k).index
            bottom_cells = ranked.tail(k).index
            rows.append(
                {
                    "stage": stage,
                    "replicate": replicate,
                    "method": method,
                    "lr_pair": lr_pair,
                    "signature": SIGNATURE,
                    "n_receiver_cells": int(len(common)),
                    "k": int(k),
                    "top_signature_mean": float(signature.loc[top_cells].mean()),
                    "bottom_signature_mean": float(signature.loc[bottom_cells].mean()),
                    "signature_effect": float(signature.loc[top_cells].mean() - signature.loc[bottom_cells].mean()),
                    "top_score_mean": float(ranked.head(k).mean()),
                    "bottom_score_mean": float(ranked.tail(k).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_panel_a(fig: plt.Figure, gs, label: str = "a") -> list[plt.Axes]:
    axes = []
    for i, (stage, stage_label) in enumerate(zip(SPATIAL_STAGES, SPATIAL_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        _, cells, _ = load_slice(stage)
        cells = cells.copy()
        cells["compartment"] = cells["cell_type"].map(broad_compartment)
        for comp, color in COMPARTMENT_COLORS.items():
            sub = cells.loc[cells["compartment"] == comp]
            if not sub.empty:
                ax.scatter(sub["x"], sub["y"], s=3.9, color=color, alpha=0.88, linewidths=0)
        set_spatial_limits(ax, cells)
        ax.set_title(stage_label, fontsize=7.5, pad=1.8)
        axes.append(ax)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "DSS MERFISH tissue context")
    handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=3.5, color=COMPARTMENT_COLORS[k], label=k)
        for k in ["Epithelial", "Immune", "Mesenchymal", "Vascular", "Neural"]
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0].get_position().x0 + 0.190, axes[0].get_position().y1 + 0.061),
        ncol=5,
        frameon=False,
        handletextpad=0.22,
        columnspacing=0.72,
        borderaxespad=0,
        fontsize=5.9,
    )
    return axes


def plot_panel_b(fig: plt.Figure, gs, label: str = "b") -> list[plt.Axes]:
    stage_cells = [aggregate_receiver_scores(stage, INFLAMMATORY_AXES, "SpatialSelf") for stage in SPATIAL_STAGES]
    vals = np.concatenate([df["incoming_score"].to_numpy(float) for df in stage_cells])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    vmax = max(float(np.quantile(vals, 0.982)) if len(vals) else 0.0, 1e-6)
    axes = []
    for i, (stage_label, cells) in enumerate(zip(SPATIAL_LABELS, stage_cells)):
        ax = fig.add_subplot(gs[0, i])
        draw_score_map(ax, cells, vmax, low_clip=0.060, title=stage_label)
        axes.append(ax)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "SpatialSelf inflammatory chemokine activity across DSS progression")
    add_low_high_bar(axes[-1], ACTIVITY_CMAP, "incoming score")
    return axes


def plot_panel_c(fig: plt.Figure, gs, label: str = "c") -> list[plt.Axes]:
    axes = []
    method_cells = {method: incoming_score_table("DSS9", LR_CASE, method) for method in METHODS}
    vals = np.concatenate([df["incoming_score"].to_numpy(float) for df in method_cells.values()])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    vmax = max(float(np.quantile(vals, 0.980)) if len(vals) else 0.0, 1e-6)
    for i, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, i])
        if method == "COMMOT":
            _, cells, _ = load_slice("DSS9")
            draw_commot_arrows(ax, cells, edge_table("DSS9", LR_CASE, method), title=method)
        else:
            draw_score_map(ax, method_cells[method], vmax, low_clip=0.040, title=method)
        axes.append(ax)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], f"DSS9 {LR_CASE} prediction patterns by method", dy=0.048)
    return axes


def plot_panel_d(fig: plt.Figure, gs, label: str = "d") -> tuple[plt.Axes, plt.Axes]:
    ax_map = fig.add_subplot(gs[0, 0])
    ax_quant = fig.add_subplot(gs[0, 1])
    sig_cells = signature_cells("DSS9")
    sig_vmax = max(float(sig_cells["signature_score"].quantile(0.985)), 1e-6)
    draw_signature_map(ax_map, sig_cells, sig_vmax, title="DSS9 NF-kB response")
    add_low_high_bar(ax_map, RESPONSE_CMAP, "signature")

    effects = top_bottom_effects()
    effects.to_csv(OUT / "method_downstream_effects.csv", index=False)
    x = np.arange(len(METHODS))
    for i, method in enumerate(METHODS):
        sub = effects.loc[effects["method"] == method].copy()
        color = METHOD_COLORS.get(method, OLD_METHOD_COLORS.get(method, "#666666"))
        jitter = np.linspace(-0.045, 0.045, len(sub)) if len(sub) > 1 else np.array([0.0])
        ax_quant.scatter(
            np.full(len(sub), x[i]) + jitter,
            sub["signature_effect"],
            s=20,
            color=color,
            edgecolors="white",
            linewidths=0.30,
            zorder=3,
        )
        if not sub.empty:
            mean = float(sub["signature_effect"].mean())
            ax_quant.plot([x[i] - 0.18, x[i] + 0.18], [mean, mean], color=color, lw=1.25, solid_capstyle="round")
    ax_quant.axhline(0, color="#8F8A82", lw=0.60, linestyle=(0, (2.2, 2.2)))
    ax_quant.set_xticks(x)
    ax_quant.set_xticklabels(["SpatialSelf", "COMMOT", "CellNEST"], rotation=28, ha="right")
    ax_quant.set_ylabel("top - bottom\nNF-kB response")
    ax_quant.set_xlim(-0.50, len(METHODS) - 0.50)
    ax_quant.set_ylim(-0.022, max(0.062, float(effects["signature_effect"].max()) + 0.014))
    ax_quant.grid(axis="y", color=GRID, lw=0.45)
    ax_quant.spines["top"].set_visible(False)
    ax_quant.spines["right"].set_visible(False)
    ax_quant.tick_params(axis="x", length=0, pad=1.5)
    ax_quant.tick_params(axis="y", length=2.0, pad=1.5)
    ax_quant.set_title("hotspot receiver enrichment", fontsize=7.4, pad=1.9)

    panel_label(fig, ax_map, label)
    panel_title(fig, ax_map, "Receiver response validation", dy=0.048)
    return ax_map, ax_quant


def render_single_panels() -> None:
    fig = plt.figure(figsize=(6.10, 2.02))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.070, top=0.835, wspace=0.040)
    plot_panel_a(fig, gs)
    export(fig, PANEL_OUT / "panel_a_tissue_context")

    fig = plt.figure(figsize=(6.10, 2.22))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.100, top=0.815, wspace=0.050)
    plot_panel_b(fig, gs)
    export(fig, PANEL_OUT / "panel_b_spatialself_tnf_progression")

    fig = plt.figure(figsize=(6.10, 2.22))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.075, top=0.815, wspace=0.050)
    plot_panel_c(fig, gs)
    export(fig, PANEL_OUT / "panel_c_method_prediction_patterns")

    fig = plt.figure(figsize=(5.95, 2.26))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.00], left=0.055, right=0.985, bottom=0.170, top=0.805, wspace=0.300)
    plot_panel_d(fig, gs)
    export(fig, PANEL_OUT / "panel_d_downstream_validation")


def render_composite() -> None:
    fig = plt.figure(figsize=(7.20, 6.65))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.20, 1.44, 1.44, 1.48],
        left=0.055,
        right=0.968,
        bottom=0.065,
        top=0.925,
        hspace=0.525,
    )
    plot_panel_a(fig, outer[0, 0].subgridspec(1, 3, wspace=0.038))
    plot_panel_b(fig, outer[1, 0].subgridspec(1, 3, wspace=0.052))
    plot_panel_c(fig, outer[2, 0].subgridspec(1, 3, wspace=0.052))
    plot_panel_d(fig, outer[3, 0].subgridspec(1, 2, width_ratios=[1.35, 1.00], wspace=0.300))
    export(fig, OUT / "mouse_colon_merfish_method_downstream_v1")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    render_single_panels()
    render_composite()
    svg = OUT / "mouse_colon_merfish_method_downstream_v1.svg"
    if svg.exists():
        text = svg.read_text(errors="ignore")
        print(f"{OUT} image_tags={text.count('<image')}")
    else:
        print(OUT)


if __name__ == "__main__":
    main()
