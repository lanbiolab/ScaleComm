#!/usr/bin/env python
"""Combine robustness and mechanism-specific heatmaps for the 200-LR benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
INPUT_CLEAN_SUMMARY = ROOT / "method_runs/scdesign3_200lr_method_summary.csv"
INPUT_NOISY_SUMMARY = ROOT / "noisy_method_runs/scdesign3_200lr_method_summary.csv"
INPUT_CLEAN_MECH = ROOT / "method_runs/scdesign3_200lr_by_mechanism.csv"
INPUT_NOISY_MECH = ROOT / "noisy_method_runs/scdesign3_200lr_by_mechanism.csv"
OUTPUT_DIR = ROOT / "figures"
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

CONDITION_ORDER = [
    "Lymph\nclean",
    "Lymph\nlow",
    "Lymph\nhigh",
    "Melanoma\nclean",
    "Melanoma\nlow",
    "Melanoma\nhigh",
]
CONDITION_DISPLAY = ["LN clean", "LN low", "LN high", "Mel. clean", "Mel. low", "Mel. high"]

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_DISPLAY = ["Contact", "Short", "Medium", "Long-niche"]

LEFT_METRICS = [
    ("aupr", "AUPR", False),
    ("empirical_fdr_at_threshold", "1 - FDR", True),
    ("top_k_precision", "Top-k precision", False),
]
RIGHT_METRICS = [
    ("aupr", "AUPR", False),
    ("fdr_top5pct", "1 - FDR", True),
    ("wrong_range_rate_top5pct", "1 - range violation", True),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    robustness = _load_robustness()
    mechanism = _load_mechanism()
    _write_source_tables(robustness, mechanism)

    fig = _plot(robustness, mechanism)
    out = OUTPUT_DIR / "scdesign3_200lr_combined_benchmark_heatmap"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {out}.svg/.pdf/.png/.tiff")


def _load_robustness() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN_SUMMARY)
    noisy = pd.read_csv(INPUT_NOISY_SUMMARY)

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


def _load_mechanism() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN_MECH).assign(noise_level="clean")
    noisy = pd.read_csv(INPUT_NOISY_MECH).assign(noise_level="noisy")
    data = pd.concat([clean, noisy], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER) & data["range_class"].isin(RANGE_ORDER)].copy()
    metric_names = [metric for metric, _, _ in RIGHT_METRICS]
    summary = (
        data.groupby(["method", "range_class"], observed=True)[metric_names]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary["range_class"] = pd.Categorical(summary["range_class"], RANGE_ORDER, ordered=True)
    return summary.sort_values(["method", "range_class"])


def _write_source_tables(robustness: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    robustness.to_csv(SOURCE_DIR / "scdesign3_200lr_combined_robustness_source.csv", index=False)
    mechanism.to_csv(SOURCE_DIR / "scdesign3_200lr_combined_mechanism_source.csv", index=False)


def _plot(robustness: pd.DataFrame, mechanism: pd.DataFrame) -> plt.Figure:
    _set_style()
    cmap = LinearSegmentedColormap.from_list(
        "warm_column_rank",
        ["#f8f6f2", "#efc6a4", "#dc7463", "#a12d4b", "#4b0c24"],
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    na_color = "#e4e4e4"

    fig = plt.figure(figsize=(7.15, 6.95), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.105,
        right=0.900,
        bottom=0.095,
        top=0.925,
        wspace=0.30,
        hspace=0.25,
    )

    left_axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    right_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]

    for row, ((metric, label, lower_is_better), ax) in enumerate(zip(LEFT_METRICS, left_axes, strict=True)):
        matrix = _robustness_matrix(robustness, metric)
        score_matrix = 1.0 - matrix if lower_is_better else matrix
        _draw_panel(
            ax,
            matrix,
            _column_normalize(score_matrix),
            cmap=cmap,
            norm=norm,
            na_color=na_color,
            panel_label=chr(ord("a") + row * 2),
            title=label,
            xlabels=CONDITION_DISPLAY,
            show_x=row == 2,
            show_y=True,
        )

    for row, ((metric, label, lower_is_better), ax) in enumerate(zip(RIGHT_METRICS, right_axes, strict=True)):
        matrix = _mechanism_matrix(mechanism, metric)
        score_matrix = 1.0 - matrix if lower_is_better else matrix
        _draw_panel(
            ax,
            matrix,
            _column_normalize(score_matrix),
            cmap=cmap,
            norm=norm,
            na_color=na_color,
            panel_label=chr(ord("b") + row * 2),
            title=label,
            xlabels=RANGE_DISPLAY,
            show_x=row == 2,
            show_y=False,
        )

    fig.text(0.105, 0.965, "Reference and noise robustness", ha="left", va="bottom", fontsize=8.5)
    fig.text(0.525, 0.965, "Mechanism-specific recovery", ha="left", va="bottom", fontsize=8.5)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.918, 0.18, 0.014, 0.66])
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.0, length=2, width=0.5, pad=1.5)
    cb.set_label("Column-normalized score", fontsize=6.5, labelpad=4)
    return fig


def _robustness_matrix(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    matrix = data.pivot_table(
        index="method",
        columns="condition",
        values=metric,
        observed=False,
        aggfunc="mean",
    )
    return matrix.reindex(index=METHOD_ORDER, columns=CONDITION_ORDER)


def _mechanism_matrix(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    matrix = data.pivot_table(
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
    raw_matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    *,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    na_color: str,
    panel_label: str,
    title: str,
    xlabels: list[str],
    show_x: bool,
    show_y: bool,
) -> None:
    values = color_matrix.to_numpy(dtype=float)
    rgba = cmap(norm(np.nan_to_num(values, nan=norm.vmin)))
    rgba[np.isnan(values)] = mpl.colors.to_rgba(na_color)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    n_rows, n_cols = raw_matrix.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(np.arange(n_cols))
    if show_x:
        rotation = 43 if n_cols > 4 else 34
        ax.set_xticklabels(xlabels, fontsize=6.4, rotation=rotation, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=1.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0, pad=0)

    ax.set_yticks(np.arange(n_rows))
    if show_y:
        ax.set_yticklabels(raw_matrix.index, fontsize=6.9)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "SpatialSelf":
                tick.set_fontweight("bold")
                tick.set_color("#b21f3a")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)

    ax.text(
        -0.13,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_title(title, loc="left", fontsize=8.1, pad=4, fontweight="normal")

    for x in np.arange(-0.5, n_cols + 0.5, 1):
        ax.axvline(x, color="white", linewidth=0.60)
    for y in np.arange(-0.5, n_rows + 0.5, 1):
        ax.axhline(y, color="white", linewidth=0.52)
    if n_cols == 6:
        ax.axvline(2.5, color="#4c4c4c", linewidth=0.80)

    raw = raw_matrix.to_numpy(dtype=float)
    for row in range(n_rows):
        for col in range(n_cols):
            if np.isnan(raw[row, col]):
                ax.text(col, row, "NA", ha="center", va="center", fontsize=5.4, color="#777777")

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
