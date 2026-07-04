#!/usr/bin/env python
"""Plot the 200-LR scDesign3 mechanism benchmark as publication heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CLEAN = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/method_runs/scdesign3_200lr_method_summary.csv"
INPUT_NOISY = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/noisy_method_runs/scdesign3_200lr_method_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/figures"


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

CONDITION_ORDER = [
    "Lymph\nclean",
    "Lymph\nlow",
    "Lymph\nhigh",
    "Melanoma\nclean",
    "Melanoma\nlow",
    "Melanoma\nhigh",
]

CONDITION_DISPLAY = [
    "LN ref. clean",
    "LN ref. low",
    "LN ref. high",
    "Mel. ref. clean",
    "Mel. ref. low",
    "Mel. ref. high",
]

METRICS = [
    ("aupr", "AUPR", False),
    ("auroc", "AUROC", False),
    ("top_k_precision", "Top-k precision", False),
    ("empirical_fdr_at_threshold", "1 - FDR", True),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    fig = _plot_heatmaps(data)
    out = OUTPUT_DIR / "scdesign3_200lr_benchmark_heatmap"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {out}.svg/.pdf/.png/.tiff")


def _load_data() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN)
    noisy = pd.read_csv(INPUT_NOISY)

    clean = clean.copy()
    clean["condition"] = clean["dataset_id"].map(
        {
            "lymph_scdesign3_200lr": "Lymph\nclean",
            "melanoma_scdesign3_200lr": "Melanoma\nclean",
        }
    )

    noisy = noisy.copy()
    noisy["condition"] = noisy["dataset_id"].map(
        {
            "lymph_scdesign3_200lr_low_noise": "Lymph\nlow",
            "lymph_scdesign3_200lr_high_noise": "Lymph\nhigh",
            "melanoma_scdesign3_200lr_low_noise": "Melanoma\nlow",
            "melanoma_scdesign3_200lr_high_noise": "Melanoma\nhigh",
        }
    )

    data = pd.concat([clean, noisy], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER) & data["condition"].notna()].copy()
    data["method"] = pd.Categorical(data["method"], METHOD_ORDER, ordered=True)
    data["condition"] = pd.Categorical(data["condition"], CONDITION_ORDER, ordered=True)
    return data.sort_values(["method", "condition"])


def _plot_heatmaps(data: pd.DataFrame) -> plt.Figure:
    _set_style()
    cmap = LinearSegmentedColormap.from_list(
        "column_rank_blue",
        ["#f3f6f8", "#c9dce6", "#7fb2c8", "#2d7f9f", "#0b4f6c"],
    )
    na_color = "#e5e5e5"

    fig = plt.figure(figsize=(7.15, 5.95), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.900,
        bottom=0.125,
        top=0.965,
        wspace=0.28,
        hspace=0.50,
    )

    for idx, (metric, label, lower_is_better) in enumerate(METRICS):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        matrix = _metric_matrix(data, metric)
        score_matrix = 1.0 - matrix if lower_is_better else matrix
        color_matrix = _column_normalize(score_matrix)
        norm = Normalize(vmin=0.0, vmax=1.0)
        _draw_panel(
            ax,
            matrix,
            color_matrix,
            cmap=cmap,
            norm=norm,
            na_color=na_color,
            panel_label=chr(ord("a") + idx),
            metric_label=label,
            show_y=idx % 2 == 0,
            show_x=idx >= 2,
        )

    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cax = fig.add_axes([0.918, 0.18, 0.014, 0.70])
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.0, length=2, width=0.5, pad=1.5)
    cb.set_label("Column-normalized score", fontsize=6.5, labelpad=4)

    return fig


def _metric_matrix(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    matrix = data.pivot_table(
        index="method",
        columns="condition",
        values=metric,
        observed=False,
        aggfunc="mean",
    )
    return matrix.reindex(index=METHOD_ORDER, columns=CONDITION_ORDER)


def _column_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    """Scale each condition so the best method in that column is darkest."""
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
        ax.set_xticklabels(CONDITION_DISPLAY, fontsize=6.0, rotation=46, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=1.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0, pad=0)
    ax.tick_params(axis="y", length=0, pad=3)

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

    ax.text(
        -0.12,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    for x in np.arange(-0.5, n_cols + 0.5, 1):
        ax.axvline(x, color="white", linewidth=0.55)
    for y in np.arange(-0.5, n_rows + 0.5, 1):
        ax.axhline(y, color="white", linewidth=0.55)
    ax.axvline(2.5, color="#3f3f3f", linewidth=0.85)

    raw = matrix.to_numpy(dtype=float)
    for row in range(n_rows):
        for col in range(n_cols):
            value = raw[row, col]
            if np.isnan(value):
                ax.text(col, row, "NA", ha="center", va="center", fontsize=5.6, color="#777777")
                continue

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(metric_label, loc="left", fontsize=8.2, pad=4, fontweight="normal")


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
