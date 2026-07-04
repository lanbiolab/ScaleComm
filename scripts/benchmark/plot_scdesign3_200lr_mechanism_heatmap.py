#!/usr/bin/env python
"""Plot mechanism-specific performance for the 200-LR scDesign3 benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CLEAN = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/method_runs/scdesign3_200lr_by_mechanism.csv"
INPUT_NOISY = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/noisy_method_runs/scdesign3_200lr_by_mechanism.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/figures"
SOURCE_DIR = OUTPUT_DIR / "source_data"


METHOD_ORDER = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellPhoneDB",
    "CellChat",
    "NICHES",
    "SpaTalk",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
]

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long-niche",
}

METRICS = [
    ("aupr", "AUPR", False),
    ("auroc", "AUROC", False),
    ("top100_precision", "Top-100 precision", False),
    ("fdr_top5pct", "1 - FDR", True),
    ("wrong_range_rate_top5pct", "1 - range violation", True),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    summary = _summarize(data)
    summary.to_csv(SOURCE_DIR / "scdesign3_200lr_mechanism_heatmap_source.csv", index=False)

    fig = _plot_heatmaps(summary)
    out = OUTPUT_DIR / "scdesign3_200lr_mechanism_heatmap"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {out}.svg/.pdf/.png/.tiff")


def _load_data() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN).assign(noise_level="clean")
    noisy = pd.read_csv(INPUT_NOISY).assign(noise_level="noisy")
    data = pd.concat([clean, noisy], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER) & data["range_class"].isin(RANGE_ORDER)].copy()
    data["method"] = pd.Categorical(data["method"], METHOD_ORDER, ordered=True)
    data["range_class"] = pd.Categorical(data["range_class"], RANGE_ORDER, ordered=True)
    return data.sort_values(["method", "range_class", "dataset_id"])


def _summarize(data: pd.DataFrame) -> pd.DataFrame:
    metric_names = [metric for metric, _, _ in METRICS]
    summary = (
        data.groupby(["method", "range_class"], observed=True)[metric_names]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["range_label"] = summary["range_class"].map(RANGE_LABELS)
    return summary


def _plot_heatmaps(summary: pd.DataFrame) -> plt.Figure:
    _set_style()
    cmap = LinearSegmentedColormap.from_list(
        "column_rank_blue",
        ["#f4f7f8", "#cbdfe8", "#7fb6ca", "#2d819f", "#084c68"],
    )
    na_color = "#e6e6e6"
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.15, 6.25), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.105,
        right=0.900,
        bottom=0.105,
        top=0.955,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.30,
        hspace=0.38,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.add_subplot(grid[1, 2]).axis("off")

    for idx, ((metric, label, lower_is_better), ax) in enumerate(zip(METRICS, axes, strict=True)):
        matrix = _metric_matrix(summary, metric)
        score_matrix = 1.0 - matrix if lower_is_better else matrix
        color_matrix = _column_normalize(score_matrix)
        _draw_panel(
            ax,
            matrix,
            color_matrix,
            cmap=cmap,
            norm=norm,
            na_color=na_color,
            panel_label=chr(ord("a") + idx),
            metric_label=label,
            show_y=idx in {0, 3},
            show_x=idx >= 3,
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.918, 0.18, 0.014, 0.68])
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.0, length=2, width=0.5, pad=1.5)
    cb.set_label("Column-normalized score", fontsize=6.5, labelpad=4)
    return fig


def _metric_matrix(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    matrix = summary.pivot_table(
        index="method",
        columns="range_class",
        values=metric,
        observed=False,
        aggfunc="mean",
    )
    return matrix.reindex(index=METHOD_ORDER, columns=RANGE_ORDER)


def _column_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    normalized = matrix.copy().astype(float)
    for column in normalized.columns:
        values = normalized[column]
        max_value = values.max(skipna=True)
        if not np.isfinite(max_value) or max_value <= 0:
            normalized[column] = np.nan
        else:
            normalized[column] = values / max_value
    return normalized.clip(lower=0.0, upper=1.0)


def _draw_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    *,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    na_color: str,
    panel_label: str,
    metric_label: str,
    show_y: bool,
    show_x: bool,
) -> None:
    values = color_matrix.to_numpy(dtype=float)
    rgba = cmap(norm(np.nan_to_num(values, nan=norm.vmin)))
    rgba[np.isnan(values)] = mpl.colors.to_rgba(na_color)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    n_rows, n_cols = matrix.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(np.arange(n_cols))
    if show_x:
        ax.set_xticklabels([RANGE_LABELS[x] for x in RANGE_ORDER], fontsize=6.8, rotation=35, ha="right")
        ax.tick_params(axis="x", length=0, pad=1.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0, pad=0)

    if show_y:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(matrix.index, fontsize=7.0)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "SpatialSelf":
                tick.set_fontweight("bold")
                tick.set_color("#d62728")
    else:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)

    ax.text(
        -0.13,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_title(metric_label, loc="left", fontsize=8.2, pad=4, fontweight="normal")

    for x in np.arange(-0.5, n_cols + 0.5, 1):
        ax.axvline(x, color="white", linewidth=0.65)
    for y in np.arange(-0.5, n_rows + 0.5, 1):
        ax.axhline(y, color="white", linewidth=0.55)

    raw = matrix.to_numpy(dtype=float)
    for row in range(n_rows):
        for col in range(n_cols):
            if np.isnan(raw[row, col]):
                ax.text(col, row, "NA", ha="center", va="center", fontsize=5.6, color="#777777")

    for spine in ax.spines.values():
        spine.set_visible(False)


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


if __name__ == "__main__":
    main()
