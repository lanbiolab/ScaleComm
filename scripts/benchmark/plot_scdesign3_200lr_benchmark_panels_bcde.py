#!/usr/bin/env python
"""Plot benchmark panels b-e for the 200-LR scDesign3 mechanism benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
TOP_METHODS = METHOD_ORDER
RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long-niche",
}
NOISE_ORDER = ["clean", "low", "high"]
NOISE_LABELS = {"clean": "Clean", "low": "Low noise", "high": "High noise"}

ACCENT = "#9E2F3D"
ACCENT_DARK = "#5C1526"
LIGHT_GRID = "#ECE7E1"
AXIS = "#403D3A"
TEXT = "#252525"

METHOD_COLORS = {
    "SpatialSelf": "#9E2F3D",
    "COMMOT": "#2B6C8A",
    "SpaOTsc": "#2C8C7A",
    "CellPhoneDB": "#D28C34",
    "CellChat": "#6F9E4E",
    "NICHES": "#8C5E9F",
    "SpaTalk": "#B49A3A",
    "CellNEST": "#7F6A4A",
    "DeepTalk": "#C86C8C",
    "CytoSignal": "#4E79A7",
}

METHOD_LIGHT = {
    method: mpl.colors.to_rgba(color, 0.28) for method, color in METHOD_COLORS.items()
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    summary = _load_summary()
    mechanism = _load_mechanism()
    leaderboard = _make_leaderboard(summary, mechanism)
    mechanism_summary = _make_mechanism_summary(mechanism)
    robustness = _make_robustness(summary)
    tradeoff = _make_tradeoff(mechanism)

    leaderboard.to_csv(SOURCE_DIR / "scdesign3_200lr_panels_bcde_leaderboard.csv", index=False)
    mechanism_summary.to_csv(SOURCE_DIR / "scdesign3_200lr_panels_bcde_mechanism.csv", index=False)
    robustness.to_csv(SOURCE_DIR / "scdesign3_200lr_panels_bcde_robustness.csv", index=False)
    tradeoff.to_csv(SOURCE_DIR / "scdesign3_200lr_panels_bcde_tradeoff.csv", index=False)

    fig = _plot(leaderboard, mechanism_summary, robustness, tradeoff)
    out = OUTPUT_DIR / "scdesign3_200lr_benchmark_panels_bcde"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"Wrote {out}.svg/.pdf/.png/.tiff")


def _load_summary() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN_SUMMARY).copy()
    clean["noise"] = "clean"
    clean["reference"] = clean["dataset_id"].map(
        {
            "lymph_scdesign3_200lr": "LN",
            "melanoma_scdesign3_200lr": "Melanoma",
        }
    )

    noisy = pd.read_csv(INPUT_NOISY_SUMMARY).copy()
    noisy["noise"] = noisy["dataset_id"].str.extract(r"200lr_(low|high)_noise", expand=False)
    noisy["reference"] = np.where(noisy["dataset_id"].str.startswith("lymph"), "LN", "Melanoma")

    data = pd.concat([clean, noisy], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER)].copy()
    data["method"] = pd.Categorical(data["method"], METHOD_ORDER, ordered=True)
    data["noise"] = pd.Categorical(data["noise"], NOISE_ORDER, ordered=True)
    return data.sort_values(["method", "reference", "noise"])


def _load_mechanism() -> pd.DataFrame:
    clean = pd.read_csv(INPUT_CLEAN_MECH).assign(noise="clean")
    noisy = pd.read_csv(INPUT_NOISY_MECH).copy()
    noisy["noise"] = noisy["dataset_id"].str.extract(r"200lr_(low|high)_noise", expand=False)
    data = pd.concat([clean, noisy], ignore_index=True)
    data = data[data["method"].isin(METHOD_ORDER) & data["range_class"].isin(RANGE_ORDER)].copy()
    data["method"] = pd.Categorical(data["method"], METHOD_ORDER, ordered=True)
    data["range_class"] = pd.Categorical(data["range_class"], RANGE_ORDER, ordered=True)
    data["noise"] = pd.Categorical(data["noise"], NOISE_ORDER, ordered=True)
    return data.sort_values(["method", "range_class", "noise"])


def _make_leaderboard(summary: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    overall = (
        summary.groupby("method", observed=True)
        .agg(
            aupr=("aupr", "mean"),
            auroc=("auroc", "mean"),
            topk=("top_k_precision", "mean"),
            fdr=("empirical_fdr_at_threshold", "mean"),
        )
        .reset_index()
    )
    mech = (
        mechanism.groupby("method", observed=True)
        .agg(range_violation=("wrong_range_rate_top5pct", "mean"))
        .reset_index()
    )
    table = overall.merge(mech, on="method", how="left")
    for source, target in [
        ("aupr", "aupr_norm"),
        ("auroc", "auroc_norm"),
        ("topk", "topk_norm"),
    ]:
        table[target] = _minmax(table[source])
    table["fdr_norm"] = _minmax(1.0 - table["fdr"])
    table["range_norm"] = _minmax(1.0 - table["range_violation"])
    table["composite_score"] = table[["aupr_norm", "auroc_norm", "topk_norm", "fdr_norm", "range_norm"]].mean(axis=1)
    table["score_low"] = table[["aupr_norm", "auroc_norm", "topk_norm", "fdr_norm", "range_norm"]].min(axis=1)
    table["score_high"] = table[["aupr_norm", "auroc_norm", "topk_norm", "fdr_norm", "range_norm"]].max(axis=1)
    return table.sort_values("composite_score", ascending=True)


def _make_mechanism_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mechanism.groupby(["method", "range_class"], observed=True)
        .agg(
            aupr=("aupr", "mean"),
            fdr=("fdr_top5pct", "mean"),
            range_violation=("wrong_range_rate_top5pct", "mean"),
        )
        .reset_index()
    )
    summary["specificity"] = 1.0 - summary["range_violation"].clip(0, 1)
    summary["precision_support"] = 1.0 - summary["fdr"].clip(0, 1)
    return summary


def _make_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["method", "noise"], observed=True)
        .agg(aupr=("aupr", "mean"))
        .reset_index()
        .sort_values(["method", "noise"])
    )


def _make_tradeoff(mechanism: pd.DataFrame) -> pd.DataFrame:
    return (
        mechanism.groupby(["method", "range_class"], observed=True)
        .agg(
            aupr=("aupr", "mean"),
            fdr=("fdr_top5pct", "mean"),
            range_violation=("wrong_range_rate_top5pct", "mean"),
        )
        .reset_index()
    )


def _plot(
    leaderboard: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> plt.Figure:
    _set_style()
    fig = plt.figure(figsize=(7.15, 6.45), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.145,
        top=0.945,
        wspace=0.33,
        hspace=0.42,
    )
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])

    _panel_leaderboard(ax_b, leaderboard)
    _panel_mechanism_bubbles(ax_c, mechanism_summary)
    _panel_robustness(ax_d, robustness)
    _panel_tradeoff(ax_e, tradeoff)

    _panel_label(ax_b, "b")
    _panel_label(ax_c, "c")
    _panel_label(ax_d, "d")
    _panel_label(ax_e, "e")
    _method_legend(fig)
    return fig


def _panel_leaderboard(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    colors = [METHOD_COLORS.get(str(method), "#777777") for method in data["method"]]
    ax.scatter(data["composite_score"], y, s=34, color=colors, edgecolor="white", linewidth=0.65, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(data["method"], fontsize=7)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "SpatialSelf":
            tick.set_fontweight("bold")
            tick.set_color(ACCENT)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Composite benchmark score", fontsize=7)
    ax.set_title("Aggregate benchmark score", loc="left", fontsize=8.2, pad=5)
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.6)
    _despine(ax)
    _highlight_spatialself_tick(ax)


def _panel_mechanism_bubbles(ax: plt.Axes, data: pd.DataFrame) -> None:
    keep = [m for m in TOP_METHODS if m in set(data["method"].astype(str))]
    subset = data[data["method"].isin(keep)].copy()
    method_to_y = {method: i for i, method in enumerate(reversed(keep))}
    range_to_x = {range_class: i for i, range_class in enumerate(RANGE_ORDER)}
    max_size = 185
    min_size = 18
    aupr_min = float(subset["aupr"].min())
    aupr_max = float(subset["aupr"].max())
    for _, row in subset.iterrows():
        x = range_to_x[str(row["range_class"])]
        y = method_to_y[str(row["method"])]
        aupr_value = float(row["aupr"]) if np.isfinite(row["aupr"]) else aupr_min
        fdr_value = float(row["precision_support"]) if np.isfinite(row["precision_support"]) else 0.0
        aupr_scaled = (aupr_value - aupr_min) / max(aupr_max - aupr_min, 1e-9)
        fdr_scaled = float(np.clip(fdr_value, 0, 1))
        alpha = float(np.clip(0.55 + 0.38 * fdr_scaled, 0.0, 1.0))
        size = min_size + max_size * aupr_scaled
        method = str(row["method"])
        color = METHOD_COLORS.get(method, "#777777")
        ax.scatter(
            x,
            y,
            s=size,
            color=mpl.colors.to_rgba(color, alpha),
            edgecolor=color,
            linewidth=0.35 + 1.05 * fdr_scaled,
        )

    ax.set_xticks(range(len(RANGE_ORDER)))
    ax.set_xticklabels([RANGE_LABELS[x] for x in RANGE_ORDER], rotation=28, ha="right", fontsize=7)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(list(reversed(keep)), fontsize=7)
    ax.set_xlim(-0.55, len(RANGE_ORDER) - 0.45)
    ax.set_ylim(-0.55, len(keep) - 0.45)
    ax.set_title("Mechanism-wise recovery", loc="left", fontsize=8.2, pad=5)
    _bubble_size_legend(ax)
    _despine(ax)
    _highlight_spatialself_tick(ax)


def _panel_robustness(ax: plt.Axes, data: pd.DataFrame) -> None:
    subset = data[data["method"].isin(TOP_METHODS)].copy()
    x = np.arange(len(NOISE_ORDER))
    for method in TOP_METHODS:
        group = subset[subset["method"] == method].set_index("noise").reindex(NOISE_ORDER)
        values = group["aupr"].to_numpy(dtype=float)
        color = METHOD_COLORS.get(method, "#777777")
        lw = 2.05 if method == "SpatialSelf" else 0.9
        alpha = 1.0 if method == "SpatialSelf" else 0.82
        ax.plot(x, values, marker="o", markersize=3.4, linewidth=lw, color=color, alpha=alpha)
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[x] for x in NOISE_ORDER], fontsize=7)
    ax.set_ylabel("AUPR", fontsize=7)
    ax.set_title("Noise robustness", loc="left", fontsize=8.2, pad=5)
    ax.set_xlim(-0.12, 2.20)
    ax.set_ylim(0, max(0.76, float(np.nanmax(subset["aupr"])) + 0.06))
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)
    _despine(ax)


def _panel_tradeoff(ax: plt.Axes, data: pd.DataFrame) -> None:
    subset = data[data["method"].isin(TOP_METHODS)].copy()
    ax.axvspan(0.55, 0.88, color="#F3E9DA", alpha=0.55, linewidth=0)
    ax.axhspan(0.70, 1.04, color="#F1DFE4", alpha=0.46, linewidth=0)
    for method in TOP_METHODS:
        group = subset[subset["method"] == method]
        color = METHOD_COLORS.get(method, "#777777")
        edge = ACCENT_DARK if method == "SpatialSelf" else "white"
        zorder = 3 if method == "SpatialSelf" else 2
        size = 50 if method == "SpatialSelf" else 34
        ax.scatter(
            group["aupr"],
            1.0 - group["fdr"],
            s=size,
            color=color,
            edgecolor=edge,
            linewidth=0.5,
            alpha=0.94 if method == "SpatialSelf" else 0.78,
            zorder=zorder,
        )
    ax.set_xlabel("AUPR", fontsize=7)
    ax.set_ylabel("1 - FDR", fontsize=7)
    ax.set_title("Recovery-FDR trade-off", loc="left", fontsize=8.2, pad=5)
    ax.set_xlim(0, min(1.0, float(subset["aupr"].max()) + 0.10))
    ax.set_ylim(0, 1.04)
    ax.grid(color=LIGHT_GRID, linewidth=0.6)
    _despine(ax)


def _minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom", color="black")


def _method_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.535, 0.012),
        ncol=5,
        frameon=False,
        fontsize=5.8,
        handletextpad=0.25,
        columnspacing=0.85,
    )


def _bubble_size_legend(ax: plt.Axes) -> None:
    legend_y = 1.135
    ax.text(
        0.72,
        legend_y + 0.016,
        "AUPR",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="left",
        va="center",
        color=TEXT,
        clip_on=False,
    )
    ax.scatter(
        [0.83, 0.93],
        [legend_y, legend_y],
        s=[26, 86],
        facecolor="white",
        edgecolor=AXIS,
        linewidth=0.75,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        0.83,
        legend_y - 0.043,
        "low",
        transform=ax.transAxes,
        fontsize=5.5,
        ha="center",
        va="top",
        color=TEXT,
        clip_on=False,
    )
    ax.text(
        0.93,
        legend_y - 0.043,
        "high",
        transform=ax.transAxes,
        fontsize=5.5,
        ha="center",
        va="top",
        color=TEXT,
        clip_on=False,
    )
def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(width=0.55, length=2.0, labelsize=7)


def _highlight_spatialself_tick(ax: plt.Axes) -> None:
    for tick in ax.get_yticklabels():
        if tick.get_text() == "SpatialSelf":
            tick.set_fontweight("bold")
            tick.set_color(ACCENT)


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


if __name__ == "__main__":
    main()
