#!/usr/bin/env python
"""Render Response Fig. R5a--R5d as restrained Nature-style benchmark dot plots.

Each output contains the six setting-level observations for all nine methods.
Large coloured points show the arithmetic mean across the six settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PANDAS_USE_BOTTLENECK"] = "0"
os.environ["PANDAS_USE_NUMEXPR"] = "0"
sys.modules["bottleneck"] = None
sys.modules["numexpr"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = OUT / "source_data"
FIGURES = OUT / "figures"
DATA = SOURCE / "wp5_per_setting_exact_budget_metrics.csv"
ARIAL_PATH = Path("/home/xd/x/YOLOFuse-main/Arial.ttf")

if ARIAL_PATH.exists():
    font_manager.fontManager.addfont(str(ARIAL_PATH))
    FONT = font_manager.FontProperties(fname=str(ARIAL_PATH)).get_name()
else:
    FONT = "Liberation Sans"

mpl.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)

METHOD_ORDER = [
    "ScaleComm",
    "COMMOT",
    "SpaOTsc",
    "CellPhoneDB",
    "CellNEST",
    "CellChat",
    "NICHES",
    "DeepTalk",
    "SpaTalk",
]

# Muted method colors follow the supplied bubble-swarm reference while giving
# ScaleComm a clear visual anchor. Individual observations are neutral gray.
METHOD_COLORS = {
    "ScaleComm": "#21618C",
    "COMMOT": "#6F98B8",
    "SpaOTsc": "#78A78F",
    "CellPhoneDB": "#C88755",
    "CellNEST": "#8D7EAA",
    "CellChat": "#B9818E",
    "NICHES": "#9EAC70",
    "DeepTalk": "#858E96",
    "SpaTalk": "#B79A68",
}

METRICS = {
    "F1": {
        "column": "overall_f1_at_global_top5pct",
        "letter": "a",
        "stem": "R5a_F1",
        "ylim": (0.0, 0.60),
        "yticks": np.arange(0.0, 0.61, 0.10),
    },
    "Precision": {
        "column": "overall_precision_at_global_top5pct",
        "letter": "b",
        "stem": "R5b_Precision",
        "ylim": (0.0, 0.90),
        "yticks": np.arange(0.0, 0.91, 0.15),
    },
    "Recall": {
        "column": "overall_recall_at_global_top5pct",
        "letter": "c",
        "stem": "R5c_Recall",
        "ylim": (0.0, 0.55),
        "yticks": np.arange(0.0, 0.56, 0.10),
    },
    "False-discovery proportion": {
        "column": "overall_empirical_fdr_at_global_top5pct",
        "letter": "d",
        "stem": "R5d_FDP",
        "ylim": (0.0, 1.00),
        "yticks": np.arange(0.0, 1.01, 0.20),
    },
}


def _deterministic_offsets(values: np.ndarray, width: float = 0.16) -> np.ndarray:
    """Separate exact ties without encoding any setting identity."""
    offsets = np.zeros(values.size, dtype=float)
    for value in np.unique(values):
        positions = np.flatnonzero(np.isclose(values, value, rtol=0.0, atol=1e-12))
        if positions.size > 1:
            offsets[positions] = np.linspace(-width, width, positions.size)
    return offsets


def _style_axis(axis: plt.Axes, metric: dict) -> None:
    axis.set_ylim(*metric["ylim"])
    axis.set_yticks(metric["yticks"])
    axis.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
    axis.set_xticks(np.arange(len(METHOD_ORDER)))
    axis.set_xticklabels(METHOD_ORDER, rotation=38, ha="right", rotation_mode="anchor")
    axis.set_ylabel(metric["stem"].replace("R5a_", "").replace("R5b_", "").replace("R5c_", "").replace("R5d_", ""), labelpad=6)
    axis.set_title(f"{metric['letter']}  {metric['stem'].split('_', 1)[1]}", loc="left", fontsize=9.2, fontweight="bold", color="#111111", pad=9)
    axis.grid(False)
    axis.tick_params(axis="x", length=0, pad=5, labelsize=6.5, colors="#111111")
    axis.tick_params(axis="y", length=3.2, width=0.75, labelsize=6.8, colors="#111111")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#111111")
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_color("#111111")
    axis.spines["bottom"].set_linewidth(0.8)


def render_metric(table: pd.DataFrame, metric_name: str) -> None:
    metric = METRICS[metric_name]
    # Compact square point-cloud layout matching the supplied bubble-swarm reference.
    fig, axis = plt.subplots(figsize=(4.25, 4.05))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.28, top=0.84)
    _style_axis(axis, metric)

    for x, method in enumerate(METHOD_ORDER):
        values = table.loc[table["method"].eq(method), metric["column"]].to_numpy(dtype=float)
        if values.size != 6:
            raise ValueError(f"{method} has {values.size} values; expected exactly six settings")
        order = np.argsort(values, kind="stable")
        offsets = np.zeros(values.size, dtype=float)
        offsets[order] = _deterministic_offsets(values[order], width=0.13)

        axis.scatter(
            np.full(values.size, x, dtype=float) + offsets,
            values,
            s=25,
            marker="o",
            facecolor="#BFC5C8",
            edgecolor="#17232B",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )
        mean = float(np.mean(values))
        axis.scatter(
            [x],
            [mean],
            s=112,
            marker="o",
            facecolor=METHOD_COLORS[method],
            edgecolor="#111111",
            linewidth=1.0,
            zorder=5,
        )

    # Mean point is the only coloured summary; no numeric column or error bars.
    axis.set_xlabel("")
    fig.savefig(FIGURES / f"{metric['stem']}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{metric['stem']}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{metric['stem']}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA)
    table = table[table["method"].isin(METHOD_ORDER)].copy()
    if set(table["method"]) != set(METHOD_ORDER):
        raise ValueError("Source data do not contain all requested methods")
    for metric_name in METRICS:
        render_metric(table, metric_name)


if __name__ == "__main__":
    main()
