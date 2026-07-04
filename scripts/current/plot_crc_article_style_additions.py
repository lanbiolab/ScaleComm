#!/usr/bin/env python
"""Article-style CRC Visium HD evidence additions for TGFB1-TGFBR2.

Panels mirror useful CytoSignal article_figure patterns:
- LR expression / LR activity / significant receiver triplet
- sender and receiver layered communication view
- TGF-beta target gene heatmap across receiver groups
- target-hotspot alignment benchmark summary
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset"
SRC = ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/source_data"
AUC_SRC = ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cytosignal_style_response_auc_v1/source_data"
OUT = ROOT / "outputs/article_style_figure_additions/crc_visium_hd_article_style"
SOURCE = OUT / "source_data"
LR_ID = "visium_hd_crc_lr_003_TGFB1_TGFBR2"
LR_LABEL = "TGFB1-TGFBR2"

INK = "#171717"
TEXT = "#2D2A27"
MUTED = "#6E665B"
GRID = "#E7E1D8"
BG = "#FFFFFF"
RED = "#C94F48"
BLUE = "#2F78A8"
TEAL = "#237A78"
TAUPE = "#BDB8AE"
AMBER = "#D7B56D"
PURPLE = "#7E6C9A"

METHOD_COLORS = {"SpatialSelf": RED, "COMMOT": BLUE, "CellNEST": PURPLE}
CELL_COLORS = {
    "tumor_proxy": "#D8D2C8",
    "caf_proxy": "#C94F48",
    "myeloid_proxy": "#6F9F9A",
    "endothelial_proxy": "#6D88A8",
    "t_cell_proxy": "#8E79AA",
    "epithelial_proxy": "#B88455",
}
EXPR_CMAP = LinearSegmentedColormap.from_list("expr_crc", ["#F7F4EC", "#E0B46C", "#B85450"])
ACTIVITY_CMAP = LinearSegmentedColormap.from_list("activity_crc", ["#F7F4EC", "#E0B46C", "#C94F48", "#641C1C"])
TARGET_CMAP = LinearSegmentedColormap.from_list("target_crc", ["#F7F4EC", "#BFD0C7", "#2F78A8"])
HEAT_CMAP = LinearSegmentedColormap.from_list("target_heat", ["#F7F4EC", "#D9B66F", "#B85450"])

TARGET_GENES = ["SERPINE1", "COL1A1", "COL1A2", "COL3A1", "COL6A1", "ACTA2", "TAGLN", "FN1", "MMP2", "MMP9", "JUN", "FOS", "SMAD7"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.026, dy: float = 0.014) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)


def add_spatial_style(ax: plt.Axes, frame: pd.DataFrame, pad_fraction: float = 0.018) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    px = (frame["x_plot"].max() - frame["x_plot"].min()) * pad_fraction
    py = (frame["y_plot"].max() - frame["y_plot"].min()) * pad_fraction
    ax.set_xlim(frame["x_plot"].min() - px, frame["x_plot"].max() + px)
    ax.set_ylim(frame["y_plot"].min() - py, frame["y_plot"].max() + py)
    for spine in ax.spines.values():
        spine.set_visible(False)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333"); ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.55); ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.45, length=2.0, pad=1.4, colors=TEXT)
    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.35, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.35, zorder=0)
    ax.set_axisbelow(True)


def robust_norm(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo, hi = arr.quantile(0.02), arr.quantile(0.985)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = arr.min(), arr.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def receiver_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one canonical cell_id column for receiver-level scores."""
    out = df.copy()
    if "receiver_id" in out.columns:
        if "cell_id" in out.columns:
            out = out.drop(columns=["cell_id"])
        out = out.rename(columns={"receiver_id": "cell_id"})
    return out


def build_expression_maps(cells: pd.DataFrame, expr: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    base = cells.copy()
    base["x_plot"] = base["x"].astype(float)
    base["y_plot"] = -base["y"].astype(float)
    expr_ix = expr.set_index("cell_id")
    rows = []
    for gene in ["TGFB1", "TGFBR2"]:
        frame = base[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].copy()
        vals = np.log1p(expr_ix[gene].reindex(frame["cell_id"]).fillna(0).astype(float).to_numpy())
        frame["feature"] = gene
        frame["value"] = vals
        frame["value_norm"] = robust_norm(pd.Series(vals)).to_numpy()
        rows.append(frame)
    spatial = receiver_table(incoming[incoming["method_label"].eq("SpatialSelf")])
    frame = base[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].merge(spatial[["cell_id", "incoming_score"]], on="cell_id", how="left")
    frame["incoming_score"] = frame["incoming_score"].fillna(0.0)
    frame["feature"] = "ScaleComm activity"
    frame["value"] = frame["incoming_score"]
    frame["value_norm"] = robust_norm(frame["incoming_score"]).to_numpy()
    rows.append(frame[["cell_id", "x_plot", "y_plot", "cell_type", "niche", "feature", "value", "value_norm"]])
    return pd.concat(rows, ignore_index=True)


def build_sender_receiver_layers(cells: pd.DataFrame, candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    xy = cells.set_index("cell_id")[["x", "y", "cell_type", "niche"]]
    edges = candidates[candidates["lr_id"].eq(LR_ID)].merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    top = edges.sort_values(["score", "candidate_id"], ascending=[False, True]).head(160).copy()
    senders = top[["sender_id", "score"]].rename(columns={"sender_id": "cell_id"})
    receivers = top[["receiver_id", "score"]].rename(columns={"receiver_id": "cell_id"})
    senders["role"] = "sender"
    receivers["role"] = "receiver"
    role = pd.concat([senders, receivers], ignore_index=True).groupby(["role", "cell_id"], as_index=False).agg(score=("score", "max"), n=("score", "size"))
    role["x"] = role["cell_id"].map(xy["x"])
    role["y"] = role["cell_id"].map(xy["y"])
    role["cell_type"] = role["cell_id"].map(xy["cell_type"])
    role["niche"] = role["cell_id"].map(xy["niche"])
    role["x_plot"] = role["x"].astype(float)
    role["y_plot"] = -role["y"].astype(float)
    return role.dropna(subset=["x_plot", "y_plot"])


def build_top_edges(cells: pd.DataFrame, candidates: pd.DataFrame, predictions: pd.DataFrame, n_edges: int = 180) -> pd.DataFrame:
    xy = cells.set_index("cell_id")[["x", "y", "cell_type", "niche"]]
    edges = candidates[candidates["lr_id"].eq(LR_ID)].merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    top = edges.sort_values(["score", "candidate_id"], ascending=[False, True]).head(n_edges).copy()
    for role, key in [("sender", "sender_id"), ("receiver", "receiver_id")]:
        top[f"{role}_x"] = top[key].map(xy["x"])
        top[f"{role}_y"] = -top[key].map(xy["y"])
        top[f"{role}_cell_type"] = top[key].map(xy["cell_type"])
        top[f"{role}_niche"] = top[key].map(xy["niche"])
    return top.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y"])


def build_target_gene_heatmap(expr: pd.DataFrame, target_map: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    expr_ix = expr.set_index("cell_id")
    spatial = receiver_table(incoming[incoming["method_label"].eq("SpatialSelf")])
    spatial["score_pct"] = spatial["incoming_score"].rank(pct=True)
    target_cols = ["cell_id"] + (["target_hotspot"] if "target_hotspot" in target_map.columns else [])
    target = target_map[target_cols].copy()
    meta = spatial.merge(target, on="cell_id", how="left", suffixes=("", "_target"))
    if "target_hotspot_target" in meta.columns:
        hotspot = meta["target_hotspot_target"].fillna(meta.get("target_hotspot", False))
    elif "target_hotspot" in meta.columns:
        hotspot = meta["target_hotspot"]
    else:
        hotspot = pd.Series(False, index=meta.index)
    meta["group"] = np.where(hotspot.fillna(False), "response hotspot", np.where(meta["score_pct"].ge(0.90), "ScaleComm top", "other"))
    genes = [g for g in TARGET_GENES if g in expr_ix.columns]
    rows = []
    for gene in genes:
        vals = np.log1p(expr_ix[gene].reindex(meta["cell_id"]).fillna(0).astype(float))
        tmp = meta[["cell_id", "group"]].copy()
        tmp["gene"] = gene
        tmp["expr"] = vals.to_numpy()
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    summary = long.groupby(["gene", "group"], as_index=False).agg(mean_expr=("expr", "mean"))
    all_gene_mean = summary.groupby("gene")["mean_expr"].transform("mean")
    all_gene_std = summary.groupby("gene")["mean_expr"].transform("std").replace(0, np.nan)
    summary["z"] = ((summary["mean_expr"] - all_gene_mean) / all_gene_std).fillna(0).clip(-2, 2)
    return summary


def plot_spatial_strip(fig: plt.Figure, outer_ax: plt.Axes, maps: pd.DataFrame, target: pd.DataFrame) -> None:
    outer_ax.set_axis_off()
    outer_ax.set_title("TGFB1-TGFBR2 spatial signal", pad=4, fontsize=7.2)
    target_map = target[["cell_id", "x_plot", "y_plot", "cell_type", "niche", "target_activity"]].copy()
    target_map["feature"] = "Target response"
    target_map["value"] = target_map["target_activity"].astype(float)
    target_map["value_norm"] = robust_norm(target_map["value"]).to_numpy()
    features = [
        ("TGFB1", maps[maps["feature"].eq("TGFB1")], EXPR_CMAP, 0.035),
        ("TGFBR2", maps[maps["feature"].eq("TGFBR2")], EXPR_CMAP, 0.035),
        ("ScaleComm", maps[maps["feature"].eq("ScaleComm activity")], ACTIVITY_CMAP, 0.00),
        ("Target response", target_map, TARGET_CMAP, 0.035),
    ]
    for i, (title, sub, cmap, active_cutoff) in enumerate(features):
        ax = outer_ax.inset_axes([0.012 + i * 0.247, 0.12, 0.225, 0.76])
        sub = sub.copy()
        if title == "ScaleComm":
            sub["plot_value"] = sub["value"].rank(pct=True).to_numpy()
        else:
            sub["plot_value"] = sub["value_norm"].to_numpy()
        sub = sub.sort_values("plot_value")
        ax.scatter(sub["x_plot"], sub["y_plot"], s=1.45, color="#D8D3CC", alpha=0.28, linewidths=0, rasterized=True)
        active = sub[sub["plot_value"].ge(active_cutoff)]
        ax.scatter(active["x_plot"], active["y_plot"], s=2.25, c=active["plot_value"], cmap=cmap, norm=Normalize(0, 1), alpha=0.90, linewidths=0, rasterized=True)
        add_spatial_style(ax, sub)
        ax.set_title(title, fontsize=6.2, pad=1.0, color=TEXT)
    cax = outer_ax.inset_axes([0.72, 0.00, 0.22, 0.035])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=ACTIVITY_CMAP, aspect="auto")
    cax.set_xticks([0, 127]); cax.set_xticklabels(["low", "high"]); cax.set_yticks([])
    cax.tick_params(length=0, labelsize=5.0, pad=0.8)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _project_layer(frame: pd.DataFrame, x_col: str, y_col: str, base: pd.DataFrame, x0: float, y0: float, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = float(base["x_plot"].min()), float(base["x_plot"].max())
    ymin, ymax = float(base["y_plot"].min()), float(base["y_plot"].max())
    xn = (frame[x_col].astype(float).to_numpy() - xmin) / max(xmax - xmin, 1e-9)
    yn = (frame[y_col].astype(float).to_numpy() - ymin) / max(ymax - ymin, 1e-9)
    return x0 + w * xn, y0 + h * yn


def plot_sender_receiver(fig: plt.Figure, outer_ax: plt.Axes, cells_plot: pd.DataFrame, layers: pd.DataFrame, top_edges: pd.DataFrame) -> None:
    outer_ax.set_axis_off()
    outer_ax.set_title("Inferred sending and receiving cells", pad=4, fontsize=7.2)
    base = cells_plot.copy()
    bands = {"receiver": (0.13, 0.55, 0.77, 0.34), "sender": (0.13, 0.08, 0.77, 0.34)}
    for role, (x0, y0, w, h) in bands.items():
        bx, by = _project_layer(base, "x_plot", "y_plot", base, x0, y0, w, h)
        outer_ax.scatter(bx, by, s=1.1, color="#D8D3CC", alpha=0.28, linewidths=0, transform=outer_ax.transAxes, rasterized=True, zorder=1)
    edge_plot = top_edges.head(95)
    sx, sy = _project_layer(edge_plot, "sender_x", "sender_y", base, *bands["sender"])
    rx, ry = _project_layer(edge_plot, "receiver_x", "receiver_y", base, *bands["receiver"])
    for x1, y1, x2, y2 in zip(sx, sy, rx, ry):
        outer_ax.plot([x1, x2], [y1, y2], color="#BFB8B0", lw=0.24, alpha=0.28, transform=outer_ax.transAxes, zorder=2)
    for role, color in [("sender", TEAL), ("receiver", RED)]:
        sub = layers[layers["role"].eq(role)].copy().sort_values("score")
        x0, y0, w, h = bands[role]
        px, py = _project_layer(sub, "x_plot", "y_plot", base, x0, y0, w, h)
        outer_ax.scatter(
            px,
            py,
            s=7.0 + 8.0 * robust_norm(sub["score"]),
            color=color,
            alpha=0.88,
            edgecolors=BG,
            linewidths=0.12,
            transform=outer_ax.transAxes,
            rasterized=True,
            zorder=4,
        )
    outer_ax.text(0.055, 0.72, "Receivers", rotation=90, ha="center", va="center", fontsize=6.0, color=INK, bbox=dict(boxstyle="square,pad=0.18", fc=BG, ec=INK, lw=0.45), transform=outer_ax.transAxes)
    outer_ax.text(0.055, 0.25, "Senders", rotation=90, ha="center", va="center", fontsize=6.0, color=INK, bbox=dict(boxstyle="square,pad=0.18", fc=BG, ec=INK, lw=0.45), transform=outer_ax.transAxes)


def plot_target_map(ax: plt.Axes, target: pd.DataFrame) -> None:
    sub = target.copy().sort_values("target_activity")
    vals = robust_norm(sub["target_activity"])
    ax.scatter(sub["x_plot"], sub["y_plot"], s=1.4, color=TAUPE, alpha=0.20, linewidths=0, rasterized=True)
    active = sub[vals.ge(0.04)].copy()
    active["norm"] = vals.loc[active.index]
    ax.scatter(active["x_plot"], active["y_plot"], s=2.8, c=active["norm"], cmap=TARGET_CMAP, norm=Normalize(0, 1), alpha=0.92, linewidths=0, rasterized=True)
    top = sub[sub["target_hotspot"].fillna(False)]
    ax.scatter(top["x_plot"], top["y_plot"], s=8, facecolors="none", edgecolors=INK, linewidths=0.25, alpha=0.42)
    ax.set_title("TGF-beta response", pad=4)
    add_spatial_style(ax, sub)


def plot_gene_heatmap(ax: plt.Axes, heat: pd.DataFrame) -> None:
    groups = ["other", "ScaleComm top", "response hotspot"]
    genes = [g for g in TARGET_GENES if g in set(heat["gene"])]
    pivot = heat.pivot_table(index="gene", columns="group", values="z", fill_value=0).reindex(index=genes, columns=groups, fill_value=0)
    norm = Normalize(-1.8, 1.8)
    for i, gene in enumerate(pivot.index):
        for j, group in enumerate(pivot.columns):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=HEAT_CMAP(norm(float(pivot.loc[gene, group]))), edgecolor="#8F8B85", linewidth=0.22))
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["other", "ScaleComm\ntop", "response\nhotspot"], rotation=0)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=5.2)
    ax.tick_params(length=0, pad=1.3)
    ax.set_title("Receiver target genes", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_alignment(ax: plt.Axes, auc: pd.DataFrame) -> None:
    order = ["SpatialSelf", "COMMOT", "CellNEST"]
    df = auc.set_index("method_label").reindex(order).reset_index()
    x = np.arange(len(df))
    colors = [METHOD_COLORS[m] for m in df["method_label"]]
    ax.bar(x, df["auc"], color=colors, width=0.56, alpha=0.84, edgecolor=INK, linewidth=0.30)
    ax.axhline(0.5, color=INK, lw=0.55, ls=(0, (2, 2)))
    for xi, row in zip(x, df.itertuples(index=False)):
        label = f"P={row.p:.1e}" if row.p < 0.01 else "n.s."
        ax.text(xi, row.auc + 0.009, label, ha="center", va="bottom", fontsize=4.9, color=METHOD_COLORS[row.method_label] if row.method_label == "SpatialSelf" else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylim(0.47, 0.635)
    ax.set_ylabel("target-hotspot AUC", labelpad=1.5)
    ax.set_title("Response alignment", pad=4)
    clean_axis(ax, "y")


def plot_lr_abundance(ax: plt.Axes, abundance: pd.DataFrame) -> None:
    df = abundance.sort_values("top_edge_count", ascending=True).copy()
    colors = [RED if name == LR_LABEL else "#D8D2C8" for name in df["control_name"]]
    ax.barh(np.arange(len(df)), df["top_edge_count"], color=colors, edgecolor=INK, linewidth=0.32, height=0.58)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["control_name"], fontsize=5.2)
    for tick, name in zip(ax.get_yticklabels(), df["control_name"]):
        if name == LR_LABEL:
            tick.set_color(RED)
            tick.set_fontweight("bold")
    ax.set_xlabel("top-edge count")
    ax.set_title("Tumor-stromal LR ranking", pad=4)
    clean_axis(ax, "x")


def build_source() -> dict[str, pd.DataFrame]:
    cells = pd.read_csv(DATASET / "cells.csv")
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    expr = pd.read_csv(DATASET / "expression.csv")
    candidates = pd.read_csv(DATASET / "candidate_interactions.csv")
    pred = pd.read_csv(ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_spatialself_scale_v21/spatialself_wavelet_scale/visium_hd_crc_p2crc_rich_targets_subset__spatialself_wavelet_scale_predictions.csv")
    incoming = pd.read_csv(SRC / "panel_g_target_hotspot_alignment.csv")
    target = pd.read_csv(SRC / "panel_f_tgfb_response_map.csv")
    abundance = pd.read_csv(SRC / "panel_b_region_lr_abundance.csv")
    auc = pd.read_csv(AUC_SRC / "target_hotspot_auc_summary.csv")
    maps = build_expression_maps(cells, expr, incoming)
    layers = build_sender_receiver_layers(cells, candidates, pred)
    top_edges = build_top_edges(cells, candidates, pred)
    target_heat = build_target_gene_heatmap(expr, target, incoming)
    return {
        "cells": cells,
        "expression_maps": maps,
        "sender_receiver_layers": layers,
        "top_edges": top_edges,
        "target_map": target,
        "target_gene_heatmap": target_heat,
        "alignment_auc": auc,
        "lr_abundance": abundance,
    }


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(9.25, 5.20), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        12,
        left=0.052,
        right=0.985,
        bottom=0.108,
        top=0.920,
        height_ratios=[1.02, 1.00],
        wspace=0.92,
        hspace=0.54,
    )
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:10])
    ax_c = fig.add_subplot(gs[0, 10:12])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:7])
    ax_f = fig.add_subplot(gs[1, 7:9])
    ax_g = fig.add_subplot(gs[1, 9:12])

    plot_spatial_strip(fig, ax_a, source["expression_maps"], source["target_map"])
    plot_sender_receiver(fig, ax_b, source["cells"], source["sender_receiver_layers"], source["top_edges"])
    plot_lr_abundance(ax_c, source["lr_abundance"])
    plot_target_map(ax_d, source["target_map"])
    plot_gene_heatmap(ax_e, source["target_gene_heatmap"])
    plot_alignment(ax_f, source["alignment_auc"])
    plot_celltype_summary(ax_g, source["sender_receiver_layers"])

    for label, ax in zip(list("abcdefg"), [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g]):
        panel_label(fig, ax, label)
    return fig


def plot_celltype_summary(ax: plt.Axes, layers: pd.DataFrame) -> None:
    df = layers.groupby(["role", "cell_type"], as_index=False).size().rename(columns={"size": "n"})
    focus = df[df["cell_type"].isin(["caf_proxy", "myeloid_proxy", "tumor_proxy", "endothelial_proxy"])]
    pivot = focus.pivot_table(index="cell_type", columns="role", values="n", fill_value=0)
    order = pivot.sum(axis=1).sort_values().index.tolist()
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for role, color in [("sender", TEAL), ("receiver", RED)]:
        vals = pivot.reindex(order).get(role, pd.Series(0, index=order)).to_numpy(float)
        ax.barh(y, vals, left=left, color=color, height=0.55, alpha=0.82, label=role)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_proxy", "") for s in order], fontsize=5.2)
    ax.set_xlabel("top-edge cells")
    ax.set_title("Cell-type roles", pad=4)
    clean_axis(ax, "x")
    ax.legend(frameon=False, loc="lower right", fontsize=5.0, handlelength=0.8)


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
    export(fig, OUT / "crc_visium_hd_article_style_additions")
    (OUT / "README.md").write_text(
        "CRC Visium HD article-style evidence additions for TGFB1-TGFBR2. Panels are inspired by CytoSignal article_figures Fig2a/b/k and Fig4-style validation summaries.\n",
        encoding="utf-8",
    )
    print(OUT / "crc_visium_hd_article_style_additions.pdf")


if __name__ == "__main__":
    main()
