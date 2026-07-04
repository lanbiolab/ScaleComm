#!/usr/bin/env python
"""Article-style SPAMOSAIC tonsil CXCL13-CXCR5 evidence additions."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story/source_data"
OUT = ROOT / "outputs/article_style_figure_additions/tonsil_article_style"
SOURCE_OUT = OUT / "source_data"

INK = "#171717"
TEXT = "#2D2A27"
MUTED = "#6E665B"
GRID = "#E7E1D8"
BG = "#FFFFFF"
RED = "#B85450"
BLUE = "#2F78A8"
TEAL = "#237A78"
AMBER = "#D7B56D"
TAUPE = "#BDB8AE"
PALE = "#F7F4EC"

PAIR = "CXCL13-CXCR5"
NICHE_COLORS = {
    "germinal_center_proxy": RED,
    "T_zone_proxy": "#66A9E8",
    "myeloid_stromal_proxy": "#9BC8C1",
    "epithelial_proxy": "#D6D1C9",
}
NICHE_LABELS = {
    "germinal_center_proxy": "GC proxy",
    "T_zone_proxy": "T zone",
    "myeloid_stromal_proxy": "myeloid/stromal",
    "epithelial_proxy": "epithelial",
}
EXPR_CMAP = LinearSegmentedColormap.from_list("tonsil_expr", ["#F7F4EC", "#E7C6C2", "#B85450"])
TARGET_CMAP = LinearSegmentedColormap.from_list("tonsil_target", ["#F7F4EC", "#BFD8EC", "#66A9E8"])
HOT_CMAP = LinearSegmentedColormap.from_list("tonsil_hot", ["#F7F4EC", "#E7C6C2", "#B85450", "#7A2D2A"])
SCALE_CMAP = LinearSegmentedColormap.from_list("tonsil_scale", ["#F7F4EC", "#D9B66F", "#B85450"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.1,
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
    lo, hi = arr.quantile(0.02), arr.quantile(0.985)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = arr.min(), arr.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def spatial_axis(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    px = (frame["x_plot"].max() - frame["x_plot"].min()) * 0.018
    py = (frame["y_plot"].max() - frame["y_plot"].min()) * 0.018
    ax.set_xlim(frame["x_plot"].min() - px, frame["x_plot"].max() + px)
    ax.set_ylim(frame["y_plot"].min() - py, frame["y_plot"].max() + py)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_source() -> dict[str, pd.DataFrame]:
    return {
        "context": pd.read_csv(SOURCE / "panel_a_tonsil_context.csv"),
        "expression": pd.read_csv(SOURCE / "panel_b_ligand_receptor_expression.csv"),
        "hotspot": pd.read_csv(SOURCE / "panel_c_scalecomm_receiver_hotspot.csv"),
        "target": pd.read_csv(SOURCE / "panel_e_target_response_map.csv"),
        "composition": pd.read_csv(SOURCE / "panel_d_top_receiver_composition.csv"),
        "replicate": pd.read_csv(SOURCE / "panel_f_replicate_support.csv"),
        "gc_rank": pd.read_csv(SOURCE / "panel_g_gc_top20_lr_hotspot_counts.csv"),
        "scale": pd.read_csv(SOURCE / "source_lr_scale_distribution.csv"),
        "sender_receiver": pd.read_csv(SOURCE / "source_sender_receiver_summary.csv"),
    }


def plot_context(ax: plt.Axes, context: pd.DataFrame) -> None:
    for niche, color in NICHE_COLORS.items():
        sub = context[context["niche"].eq(niche)]
        if sub.empty:
            continue
        ax.scatter(sub["x_plot"], sub["y_plot"], s=4.0, color=color, alpha=0.78, linewidths=0, rasterized=True)
    spatial_axis(ax, context)
    ax.set_title("SPAMOSAIC tonsil context", pad=4)
    handles = [
        Line2D([], [], marker="o", ls="", color=color, markersize=4.0, label=NICHE_LABELS[niche])
        for niche, color in NICHE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.00, -0.05), ncol=2, frameon=False, handletextpad=0.20, columnspacing=0.55)


def plot_spatial_strip(fig: plt.Figure, outer_ax: plt.Axes, expression: pd.DataFrame, hotspot: pd.DataFrame, target: pd.DataFrame) -> None:
    outer_ax.set_axis_off()
    outer_ax.set_title("CXCL13-CXCR5 spatial program", pad=4, fontsize=7.2)
    cxcl13 = expression[expression["gene"].eq("CXCL13")].copy()
    cxcr5 = expression[expression["gene"].eq("CXCR5")].copy()
    features = [
        ("CXCL13", cxcl13, "expression_norm", EXPR_CMAP),
        ("CXCR5", cxcr5, "expression_norm", EXPR_CMAP),
        ("ScaleComm", hotspot, "incoming_norm", HOT_CMAP),
        ("GC response", target, "target_score_norm", TARGET_CMAP),
    ]
    for i, (title, frame, value_col, cmap) in enumerate(features):
        ax = outer_ax.inset_axes([0.015 + i * 0.247, 0.13, 0.220, 0.74])
        sub = frame.copy().sort_values(value_col)
        ax.scatter(sub["x_plot"], sub["y_plot"], s=3.6, color="#D8D2C8", alpha=0.25, linewidths=0, rasterized=True)
        active = sub[sub[value_col].astype(float).ge(0.06)].sort_values(value_col)
        ax.scatter(active["x_plot"], active["y_plot"], s=5.4, c=active[value_col], cmap=cmap, norm=Normalize(0, 1), alpha=0.75, linewidths=0, rasterized=True)
        if "top_receiver" in sub.columns and title == "ScaleComm":
            top = sub[sub["top_receiver"].fillna(False)]
            ax.scatter(top["x_plot"], top["y_plot"], s=12, facecolors="none", edgecolors=RED, linewidths=0.30, alpha=0.45)
        spatial_axis(ax, sub)
        ax.set_title(title, fontsize=6.2, pad=1.0, color=TEXT)
    cax = outer_ax.inset_axes([0.75, 0.00, 0.20, 0.035])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=HOT_CMAP, aspect="auto")
    cax.set_xticks([0, 127]); cax.set_xticklabels(["low", "high"]); cax.set_yticks([])
    cax.tick_params(length=0, labelsize=5.0, pad=0.8)
    for s in cax.spines.values():
        s.set_visible(False)


def plot_sender_receiver(ax: plt.Axes, summary: pd.DataFrame) -> None:
    dat = summary[summary["lr_pair"].eq(PAIR)].copy().head(20)
    pivot = dat.pivot_table(index="sender_cell_type", columns="receiver_cell_type", values="n_edges", aggfunc="sum", fill_value=0)
    senders = pivot.sum(axis=1).sort_values(ascending=False).head(7).index.tolist()
    receivers = pivot.sum(axis=0).sort_values(ascending=False).head(6).index.tolist()
    pivot = pivot.reindex(index=senders, columns=receivers, fill_value=0)
    vmax = max(float(pivot.to_numpy().max()), 1.0)
    for i, sender in enumerate(pivot.index):
        for j, receiver in enumerate(pivot.columns):
            val = float(pivot.loc[sender, receiver])
            face = HOT_CMAP(val / vmax)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="#8F8B85", linewidth=0.22))
    ax.set_xlim(-0.5, len(pivot.columns) - 0.5)
    ax.set_ylim(len(pivot.index) - 0.5, -0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([s.replace("_proxy", "").replace("_", " ") for s in pivot.columns], rotation=35, ha="right", fontsize=5.0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.replace("_proxy", "").replace("_", " ") for s in pivot.index], fontsize=5.0)
    ax.tick_params(length=0, pad=1.3)
    ax.set_title("Sender-receiver cell pairs", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_gc_enrichment(ax: plt.Axes, comp: pd.DataFrame) -> None:
    dat = comp[comp["feature"].eq("niche")].sort_values("fold_enrichment", ascending=True)
    colors = [NICHE_COLORS.get(n, TAUPE) for n in dat["label"]]
    y = np.arange(len(dat))
    ax.barh(y, dat["fold_enrichment"], color=colors, alpha=0.84, edgecolor=INK, linewidth=0.28, height=0.56)
    ax.axvline(1, color=INK, lw=0.55, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(dat["display_label"].tolist())
    ax.set_xlabel("top receiver enrichment")
    ax.set_title("Top receivers enrich in GC", pad=4)
    clean_axis(ax, "x")


def plot_replicate(ax: plt.Axes, replicate: pd.DataFrame) -> None:
    metrics = [
        ("target_lift_top_vs_background", "target lift"),
        ("spearman_incoming_target", "rho"),
        ("gc_fold_enrichment", "GC fold"),
    ]
    x = np.arange(len(metrics))
    width = 0.30
    for offset, section, color in [(-width / 2, "A1", RED), (width / 2, "D1", BLUE)]:
        row = replicate[replicate["section"].eq(section)]
        vals = [float(row.iloc[0][m]) if len(row) else np.nan for m, _ in metrics]
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.84, edgecolor=INK, linewidth=0.28, label=section)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("support value")
    ax.set_title("A1/D1 replicate support", pad=4)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.55, 1.03), ncol=2, handlelength=0.9, columnspacing=0.5)
    clean_axis(ax, "y")


def plot_gc_rank(ax: plt.Axes, rank: pd.DataFrame) -> None:
    dat = rank[rank["section"].eq("A1")].sort_values("top20_gc_n", ascending=True)
    y = np.arange(len(dat))
    colors = [RED if pair == PAIR else "#D8D2C8" for pair in dat["lr_pair"]]
    labels = [("" if p == PAIR else p.replace("TNFSF13B-TNFRSF13C", "BAFF-TNFRSF13C")) for p in dat["lr_pair"]]
    ax.barh(y, dat["top20_gc_n"], color=colors, alpha=0.86, edgecolor=INK, linewidth=0.28, height=0.56)
    selected = dat[dat["lr_pair"].eq(PAIR)]
    if not selected.empty:
        yi = int(np.where(dat["lr_pair"].to_numpy() == PAIR)[0][0])
        ax.text(
            min(float(selected.iloc[0]["top20_gc_n"]) - 8, float(dat["top20_gc_n"].max()) * 0.90),
            yi,
            PAIR,
            ha="right",
            va="center",
            fontsize=5.4,
            color=BG,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.1)
    ax.set_xlabel("GC cells in LR top 20%")
    ax.set_title("GC-localized LR ranking", pad=4)
    clean_axis(ax, "x")


def plot_scale(ax: plt.Axes, scale: pd.DataFrame) -> None:
    scale_cols = [f"scale_{i}" for i in range(6)]
    keep = ["CXCL13-CXCR5", "CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "TNFSF13B-TNFRSF13C", "ACTB-EGFR", "MS4A1-CD3E"]
    dat = scale[(scale["section"].eq("A1")) & (scale["lr_pair"].isin(keep))].copy()
    dat["order"] = dat["lr_pair"].map({lr: i for i, lr in enumerate(keep)})
    dat = dat.sort_values("order")
    mat = dat[scale_cols].astype(float).to_numpy()
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SCALE_CMAP(mat[i, j]), edgecolor="#8F8B85", linewidth=0.20))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_xticks([0, 2.5, 5])
    ax.set_xticklabels(["local", "mid", "broad"])
    ax.set_yticks(range(len(dat)))
    ax.set_yticklabels(["" if lr == PAIR else lr for lr in dat["lr_pair"]], fontsize=5.1)
    for tick, lr in zip(ax.get_yticklabels(), dat["lr_pair"], strict=False):
        if lr == PAIR:
            tick.set_color(RED)
            tick.set_fontweight("bold")
    if PAIR in set(dat["lr_pair"]):
        yi = int(np.where(dat["lr_pair"].to_numpy() == PAIR)[0][0])
        ax.text(0.10, yi, PAIR, ha="left", va="center", fontsize=5.2, color=BG, fontweight="bold")
    ax.tick_params(length=0, pad=1.4)
    ax.set_title("LR scale profiles in tonsil", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(9.20, 5.35), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        12,
        left=0.060,
        right=0.985,
        bottom=0.110,
        top=0.920,
        wspace=0.90,
        hspace=0.60,
        height_ratios=[1.02, 1.00],
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:9])
    ax_c = fig.add_subplot(gs[0, 9:12])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    ax_f = fig.add_subplot(gs[1, 6:9])
    ax_g = fig.add_subplot(gs[1, 9:12])

    plot_context(ax_a, source["context"])
    plot_spatial_strip(fig, ax_b, source["expression"], source["hotspot"], source["target"])
    plot_sender_receiver(ax_c, source["sender_receiver"])
    plot_gc_enrichment(ax_d, source["composition"])
    plot_replicate(ax_e, source["replicate"])
    plot_gc_rank(ax_f, source["gc_rank"])
    plot_scale(ax_g, source["scale"])

    for label, ax in zip(list("abcdefg"), [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g]):
        panel_label(fig, ax, label)
    return fig


def export(fig: plt.Figure, stem: Path, dpi: int = 850) -> None:
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    for stale in SOURCE_OUT.glob("*.csv"):
        stale.unlink()
    setup_style()
    source = load_source()
    for name, table in source.items():
        table.to_csv(SOURCE_OUT / f"{name}.csv", index=False)
    fig = plot_figure(source)
    export(fig, OUT / "tonsil_article_style_additions")
    (OUT / "README.md").write_text(
        "SPAMOSAIC tonsil article-style evidence additions for CXCL13-CXCR5. "
        "Panels summarize GC context, spatial ligand/receptor/activity/response, sender-receiver pairs, GC enrichment, replicate support, and scale profiles.\n",
        encoding="utf-8",
    )
    print(OUT / "tonsil_article_style_additions.pdf")


if __name__ == "__main__":
    main()
