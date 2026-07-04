#!/usr/bin/env python
"""CRC TGFB1-TGFBR2 method alignment against target-response proxy."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/source_data/panel_g_target_hotspot_alignment.csv"
OUT_DIR = PROJECT_ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cytosignal_style_response_auc_v1"

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
BG = "#FFFFFF"
COLORS = {"SpatialSelf": "#C94F48", "COMMOT": "#2F78A8", "CellNEST": "#7E6C9A"}
ORDER = ["SpatialSelf", "COMMOT", "CellNEST"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()
    configure()
    data = pd.read_csv(DATA)
    summary = (
        data.groupby("method_label", as_index=False)
        .agg(
            auc=("target_hotspot_auc", "first"),
            p=("mannwhitney_greater_p", "first"),
            target_hotspot_n=("target_hotspot_n", "first"),
            background_n=("background_n", "first"),
        )
        .assign(method_label=lambda d: pd.Categorical(d["method_label"], ORDER, ordered=True))
        .sort_values("method_label")
    )
    data.to_csv(source_dir / "target_hotspot_alignment_long.csv", index=False)
    summary.to_csv(source_dir / "target_hotspot_auc_summary.csv", index=False)
    fig = plot_auc(data, summary)
    out = OUT_DIR / "crc_tgfb1_tgfbr2_method_response_auc"
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 1000
        fig.savefig(f"{out}.{ext}", **kwargs)
    plt.close(fig)
    print(f"{out}.png")
    print(f"{out}.pdf")


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.58,
            "axes.edgecolor": "#6A625B",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def plot_auc(data: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(5.55, 2.65), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, bottom=0.18, top=0.86, wspace=0.36, width_ratios=[1.15, 0.82])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_hotspot_percentiles(ax_a, data)
    plot_auc_bar(ax_b, summary)
    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


def plot_hotspot_percentiles(ax: plt.Axes, data: pd.DataFrame) -> None:
    hotspot = data[data["target_hotspot"].fillna(False)].copy()
    rng = np.random.default_rng(12)
    positions = np.arange(len(ORDER))
    for i, method in enumerate(ORDER):
        vals = hotspot.loc[hotspot["method_label"].eq(method), "score_percentile"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        parts = ax.violinplot(vals, positions=[i], widths=0.62, showmeans=False, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("none")
            body.set_alpha(0.28 if method != "SpatialSelf" else 0.40)
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=4.0, color=COLORS[method], alpha=0.30, linewidths=0, rasterized=True)
        med = np.nanmedian(vals)
        ax.plot([i - 0.23, i + 0.23], [med, med], color=INK, lw=0.65)
    ax.axhline(0.5, color="#9E968D", lw=0.55, ls=(0, (2, 2)))
    ax.set_xticks(positions)
    ax.set_xticklabels(ORDER, fontsize=5.7)
    ax.set_ylabel("prediction percentile in\nresponse-hotspot bins", fontsize=5.8)
    ax.set_title("target-hotspot alignment", fontsize=6.8, pad=2.0)
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color=GRID, lw=0.42)
    ax.tick_params(axis="y", labelsize=5.2, length=2.0, pad=1)
    ax.tick_params(axis="x", length=0, pad=2)


def plot_auc_bar(ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = summary.copy()
    y = np.arange(len(table))[::-1]
    auc = table["auc"].to_numpy(dtype=float)
    colors = [COLORS[str(m)] for m in table["method_label"]]
    ax.barh(y, auc, color=colors, height=0.52, edgecolor="none", alpha=0.92)
    ax.axvline(0.5, color="#8A8177", lw=0.58, ls=(0, (2, 2)))
    for yy, aa, m in zip(y, auc, table["method_label"], strict=False):
        ax.text(aa + 0.012, yy, f"{aa:.2f}", va="center", ha="left", fontsize=5.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(table["method_label"], fontsize=5.8)
    ax.set_xlim(0.45, max(0.66, float(auc.max()) + 0.06))
    ax.set_xlabel("AUC", fontsize=5.8)
    ax.set_title("response proxy AUC", fontsize=6.8, pad=2.0)
    ax.grid(axis="x", color=GRID, lw=0.42)
    ax.tick_params(axis="x", labelsize=5.2, length=2.0, pad=1)
    ax.tick_params(axis="y", length=0, pad=1.2)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.035, box.y1 + 0.014, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
