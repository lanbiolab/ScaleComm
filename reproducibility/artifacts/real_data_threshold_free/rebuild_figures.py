#!/usr/bin/env python3
"""Render the reviewer-requested tonsil full-ranking receiver audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
FIGURES = ROOT / "figures"
ARIAL = Path("/home/xd/x/YOLOFuse-main/Arial.ttf")
if ARIAL.exists():
    font_manager.fontManager.addfont(str(ARIAL))
    FONT = font_manager.FontProperties(fname=str(ARIAL)).get_name()
else:
    FONT = "Liberation Sans"


mpl.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 8.4,
        "axes.linewidth": 0.85,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


TEXT = "#171717"
AXIS = "#171717"
GRID = "#D4DBDE"
CASE_COLORS = {
    "Tonsil A1": "#167C91",
    "Tonsil D1": "#45A9A5",
    "CRC TGFB1-TGFBR2": "#D17A43",
    "NSCLC CXCL12-CXCR4": "#C95C62",
    "DSS9 Cxcl10-Cxcr3 replicate 1": "#6D5A9E",
    "DSS9 Cxcl10-Cxcr3 replicate 2": "#9A86B9",
    "DSS21 Cxcl12-Cxcr4 replicate 1": "#8E647B",
    "DSS21 Cxcl12-Cxcr4 replicate 2": "#C08A9C",
}
CASE_ORDER = list(CASE_COLORS)
CASE_LABELS = {
    "Tonsil A1": "Tonsil A1",
    "Tonsil D1": "Tonsil D1",
    "CRC TGFB1-TGFBR2": "CRC",
    "NSCLC CXCL12-CXCR4": "NSCLC",
    "DSS9 Cxcl10-Cxcr3 replicate 1": "D9 r1",
    "DSS9 Cxcl10-Cxcr3 replicate 2": "D9 r2",
    "DSS21 Cxcl12-Cxcr4 replicate 1": "D21 r1",
    "DSS21 Cxcl12-Cxcr4 replicate 2": "D21 r2",
}


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", length=4, width=0.8, color=AXIS, pad=4)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def curve_legend() -> list[Line2D]:
    return [Line2D([0], [0], color=CASE_COLORS[c], linewidth=2.0, label=CASE_LABELS[c]) for c in CASE_ORDER]


def draw_curves(curves: pd.DataFrame, curve_name: str, xlabel: str, ylabel: str, title: str, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.35, 5.05))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.27, top=0.88)
    style_axis(ax)
    for case in CASE_ORDER:
        q = curves[(curves["case"] == case) & (curves["curve"] == curve_name)]
        if q.empty:
            continue
        ax.plot(q["x"], q["y"], color=CASE_COLORS[case], linewidth=1.65, alpha=0.95, solid_capstyle="round", zorder=3)
    if curve_name == "ROC":
        ax.plot([0, 1], [0, 1], color="#9EA8AC", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel, labelpad=7, fontweight="bold")
    ax.set_ylabel(ylabel, labelpad=7, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT, pad=8)
    ax.legend(handles=curve_legend(), loc="upper center", bbox_to_anchor=(0.50, -0.18), ncol=2,
              frameon=False, fontsize=7.0, handlelength=1.7, columnspacing=1.3, handletextpad=0.45)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".png"), dpi=700, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_auc(summary: pd.DataFrame, output_stem: Path) -> None:
    ordered = summary.set_index("case").reindex(CASE_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(5.75, 4.65))
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.18, top=0.88)
    style_axis(ax)
    y = list(range(len(ordered) - 1, -1, -1))
    ax.set_xlim(0.48, 0.86)
    ax.set_ylim(-0.65, len(ordered) - 0.35)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_yticks(y)
    ax.set_yticklabels([CASE_LABELS[c] for c in ordered["case"]])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("ROC-AUC", labelpad=7, fontweight="bold")
    ax.set_title("c  Full-ranking AUC", loc="left", fontsize=11, fontweight="bold", color=TEXT, pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)

    ax.axvspan(0.813, 0.822, color="#167C91", alpha=0.10, zorder=0)
    ax.axvline(0.8175, color="#167C91", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.75)
    ax.text(0.8175, len(ordered) - 0.40, "Tonsil A1/D1", ha="center", va="top", fontsize=6.7, color="#167C91")
    for yi, (_, row) in zip(y, ordered.iterrows()):
        color = CASE_COLORS[row["case"]]
        ax.plot(row["roc_auc"], yi, marker="o", markersize=8.0, color=color, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        ax.text(min(float(row["roc_auc"]) + 0.014, 0.845), yi, f"{row['roc_auc']:.3f}", va="center", fontsize=7.0, color=TEXT)

    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".png"), dpi=700, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(SOURCE / "w12_comment9_threshold_free_curves.csv")
    summary = pd.read_csv(SOURCE / "w12_comment9_threshold_free_summary.csv")
    draw_curves(curves, "ROC", "False-positive rate", "True-positive rate", "a  ROC", FIGURES / "R13a_ROC")
    draw_curves(curves, "PR", "Recall", "Precision", "b  Precision-recall", FIGURES / "R13b_PrecisionRecall")
    draw_auc(summary, FIGURES / "R13c_AUC")


if __name__ == "__main__":
    main()
