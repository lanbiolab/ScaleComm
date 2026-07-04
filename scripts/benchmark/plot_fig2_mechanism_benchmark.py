#!/usr/bin/env python
"""Draw Fig. 2: mechanism-aware scDesign3 benchmark design and robustness."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/reference_based_benchmark/fig2_mechanism_benchmark"
MAIN_SUMMARY = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_benchmark/mechanism_benchmark_summary.csv"
MAIN_MECH = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_benchmark/mechanism_benchmark_range_class_summary.csv"
NOISY_SUMMARY = PROJECT_ROOT / "outputs/reference_based_benchmark/noisy_scdesign3_mechanism/method_runs_complete/noisy_method_summary.csv"
NOISY_MECH = PROJECT_ROOT / "outputs/reference_based_benchmark/noisy_scdesign3_mechanism/method_runs_complete/noisy_method_mechanism_summary.csv"


METHOD_ORDER_CLEAN = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellChat",
    "CellPhoneDB",
    "NICHES",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
    "SpaTalk",
]
METHOD_ORDER_NOISY = METHOD_ORDER_CLEAN
METHOD_RENAME = {
    "spatialself_range_context": "SpatialSelf",
    "commot": "COMMOT",
    "spaotsc": "SpaOTsc",
    "cellchat": "CellChat",
    "cellnest": "CellNEST",
}
RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABEL = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}

COLORS = {
    "ink": "#252525",
    "muted": "#777777",
    "soft": "#F4F5F4",
    "line": "#B9B9B9",
    "cell_sender": "#4C9C99",
    "cell_receiver": "#6E86BC",
    "truth": "#157A7F",
    "decoy": "#B66A43",
    "niche": "#E3EFEC",
    "niche2": "#EEE7D7",
    "grid": "#FFFFFF",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    fig = plt.figure(figsize=(7.2, 8.9), constrained_layout=False)
    outer = gridspec.GridSpec(
        3,
        1,
        height_ratios=[0.70, 1.05, 1.12],
        hspace=0.30,
        figure=fig,
    )
    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.13)
    middle = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.30)
    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.30)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(middle[0, 0])
    ax_e = fig.add_subplot(middle[0, 1])
    ax_f = fig.add_subplot(bottom[0, 0])
    ax_g = fig.add_subplot(bottom[0, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    im_score = draw_clean_score_heatmap(ax_d)
    im_fdr = draw_clean_fdr_heatmap(ax_e)
    draw_noisy_score_heatmap(ax_f)
    draw_noisy_fdr_heatmap(ax_g)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d, "d", x=-0.16)
    add_panel_label(ax_e, "e", x=-0.14)
    add_panel_label(ax_f, "f", x=-0.16)
    add_panel_label(ax_g, "g", x=-0.14)

    cax = fig.add_axes([0.925, 0.365, 0.012, 0.20])
    cb = plt.colorbar(im_score, cax=cax)
    cb.set_label("AUPR", fontsize=6.3, color=COLORS["muted"], labelpad=4)
    cb.ax.tick_params(labelsize=5.8, length=2, width=0.6)
    cb.outline.set_visible(False)
    cax = fig.add_axes([0.925, 0.115, 0.012, 0.20])
    cb = plt.colorbar(im_fdr, cax=cax)
    cb.set_label("FDR", fontsize=6.3, color=COLORS["muted"], labelpad=4)
    cb.ax.tick_params(labelsize=5.8, length=2, width=0.6)
    cb.outline.set_visible(False)

    for ext in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if ext in {"png", "tiff"} else None
        fig.savefig(OUT_DIR / f"fig2_mechanism_benchmark.{ext}", bbox_inches="tight", dpi=dpi)
    write_source_tables()
    print(f"Wrote Fig. 2 outputs to {OUT_DIR}")


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=COLORS["ink"])


def setup_schematic_ax(ax, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    ax.text(0.02, 0.98, title, ha="left", va="top", fontsize=7.2, color=COLORS["ink"], fontweight="bold")


def draw_cell(ax, xy, color, radius=0.035, edge="#FFFFFF", lw=0.7, alpha=1.0) -> None:
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor=edge, lw=lw, alpha=alpha))


def draw_arrow(ax, start, end, color, lw=1.2, rad=0.0, style="-|>", alpha=1.0, ls="-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=7,
            lw=lw,
            color=color,
            alpha=alpha,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_panel_a(ax) -> None:
    setup_schematic_ax(ax, "Reference-derived tissue")
    ax.add_patch(Rectangle((0.08, 0.15), 0.84, 0.66, facecolor=COLORS["soft"], edgecolor="none"))
    ax.add_patch(Rectangle((0.08, 0.15), 0.40, 0.66, facecolor=COLORS["niche"], edgecolor="none"))
    ax.add_patch(Rectangle((0.48, 0.15), 0.44, 0.66, facecolor=COLORS["niche2"], edgecolor="none"))
    ax.plot([0.12, 0.87], [0.23, 0.69], color="white", lw=1.3, alpha=0.95)
    pts = [
        (0.18, 0.30, "s"), (0.25, 0.48, "r"), (0.34, 0.63, "r"), (0.42, 0.36, "s"),
        (0.58, 0.28, "r"), (0.66, 0.54, "s"), (0.77, 0.66, "r"), (0.84, 0.40, "s"),
        (0.30, 0.24, "s"), (0.72, 0.32, "r"),
    ]
    for x, y, kind in pts:
        draw_cell(ax, (x, y), COLORS["cell_sender"] if kind == "s" else COLORS["cell_receiver"], radius=0.032)
    ax.text(0.13, 0.75, "domain 1", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.66, 0.75, "domain 2", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.50, 0.075, "real ST geometry + reference expression", fontsize=5.9, color=COLORS["ink"], ha="center")


def draw_panel_b(ax) -> None:
    setup_schematic_ax(ax, "Mechanism-specific positives")
    rows = [
        ("Contact", 0.69, 0.23, 0.37),
        ("Short", 0.50, 0.23, 0.52),
        ("Medium", 0.31, 0.23, 0.70),
    ]
    for label, y, x0, x1 in rows:
        ax.text(0.05, y, label, fontsize=6.2, color=COLORS["ink"], ha="left", va="center")
        draw_cell(ax, (x0, y), COLORS["cell_sender"], radius=0.027)
        draw_cell(ax, (x1, y), COLORS["cell_receiver"], radius=0.027)
        draw_arrow(ax, (x0 + 0.035, y), (x1 - 0.035, y), COLORS["truth"], lw=1.1)
        ax.plot([x0, x1], [y - 0.055, y - 0.055], color=COLORS["line"], lw=0.8)
    ax.text(0.05, 0.15, "Long niche", fontsize=6.2, color=COLORS["ink"], ha="left", va="center")
    ax.add_patch(Rectangle((0.52, 0.08), 0.31, 0.14, facecolor=COLORS["niche"], edgecolor="none"))
    draw_cell(ax, (0.23, 0.15), COLORS["cell_sender"], radius=0.027)
    draw_cell(ax, (0.75, 0.15), COLORS["cell_receiver"], radius=0.027, alpha=0.95)
    draw_arrow(ax, (0.265, 0.15), (0.715, 0.15), COLORS["truth"], lw=1.1)
    ax.text(0.55, 0.235, "allowed niche", fontsize=5.6, color=COLORS["muted"])


def draw_panel_c(ax) -> None:
    setup_schematic_ax(ax, "Matched negatives")
    y_true, y_range, y_niche = 0.68, 0.46, 0.23
    draw_cell(ax, (0.16, y_true), COLORS["cell_sender"], radius=0.028)
    draw_cell(ax, (0.55, y_true), COLORS["cell_receiver"], radius=0.028)
    draw_arrow(ax, (0.195, y_true), (0.515, y_true), COLORS["truth"], lw=1.1)
    ax.text(0.64, y_true, "valid LR edge", fontsize=6.1, va="center", color=COLORS["truth"])

    draw_cell(ax, (0.16, y_range), COLORS["cell_sender"], radius=0.028)
    draw_cell(ax, (0.33, y_range), COLORS["cell_receiver"], radius=0.028)
    draw_arrow(ax, (0.195, y_range), (0.295, y_range), COLORS["decoy"], lw=1.0, ls="--")
    ax.text(0.64, y_range, "wrong distance", fontsize=6.1, va="center", color=COLORS["decoy"])

    ax.add_patch(Rectangle((0.10, y_niche - 0.06), 0.26, 0.12, facecolor=COLORS["niche"], edgecolor="none"))
    ax.add_patch(Rectangle((0.43, y_niche - 0.06), 0.26, 0.12, facecolor=COLORS["niche2"], edgecolor="none"))
    draw_cell(ax, (0.18, y_niche), COLORS["cell_sender"], radius=0.024)
    draw_cell(ax, (0.51, y_niche), COLORS["cell_receiver"], radius=0.024)
    draw_arrow(ax, (0.21, y_niche), (0.48, y_niche), COLORS["decoy"], lw=1.0, ls="--")
    ax.text(0.64, y_niche, "wrong niche", fontsize=6.1, va="center", color=COLORS["decoy"])

    ax.plot([0.13, 0.70], [0.81, 0.81], color=COLORS["line"], lw=0.8)
    for x in [0.18, 0.31, 0.44, 0.57, 0.70]:
        ax.plot([x, x], [0.785, 0.835], color=COLORS["line"], lw=0.8)
    ax.text(0.13, 0.855, "dropout / expression noise", fontsize=5.7, color=COLORS["muted"], ha="left")


def draw_clean_score_heatmap(ax):
    table = pd.read_csv(MAIN_MECH)
    table = table[table["method"].isin(METHOD_ORDER_CLEAN)].copy()
    table["range_class"] = pd.Categorical(table["range_class"], RANGE_ORDER, ordered=True)
    matrix = (
        table.pivot(index="method", columns="range_class", values="aupr")
        .reindex(METHOD_ORDER_CLEAN)
        .reindex(columns=RANGE_ORDER)
    )
    return draw_heatmap(
        ax,
        matrix,
        title="Clean benchmark: detection",
        xlabel="LR mechanism",
        xlabels=[RANGE_LABEL[r] for r in RANGE_ORDER],
        metric="score",
    )


def draw_clean_fdr_heatmap(ax):
    table = pd.read_csv(MAIN_MECH)
    table = table[table["method"].isin(METHOD_ORDER_CLEAN)].copy()
    table["range_class"] = pd.Categorical(table["range_class"], RANGE_ORDER, ordered=True)
    matrix = (
        table.pivot(index="method", columns="range_class", values="fdr_top5pct")
        .reindex(METHOD_ORDER_CLEAN)
        .reindex(columns=RANGE_ORDER)
    )
    return draw_heatmap(
        ax,
        matrix,
        title="Clean benchmark: false positives",
        xlabel="LR mechanism",
        xlabels=[RANGE_LABEL[r] for r in RANGE_ORDER],
        metric="fdr",
    )


def draw_noisy_score_heatmap(ax):
    table = pd.read_csv(NOISY_MECH)
    table["method"] = table["method"].map(METHOD_RENAME).fillna(table["method"])
    table = table[table["method"].isin(METHOD_ORDER_NOISY)].copy()
    table = _mask_noisy_unavailable(table, "aupr")
    table["noise_label"] = table["noise_level"].map({"low_noise": "Low", "high_noise": "High"})
    table["col"] = table["noise_label"] + "\n" + table["range_class"].map(RANGE_LABEL)
    columns = [f"{noise}\n{RANGE_LABEL[r]}" for noise in ["Low", "High"] for r in RANGE_ORDER]
    matrix = (
        table.pivot_table(index="method", columns="col", values="aupr", aggfunc="mean")
        .reindex(METHOD_ORDER_NOISY)
        .reindex(columns=columns)
    )
    xlabels = ["Contact", "Short", "Medium", "Long\nniche"] * 2
    im = draw_heatmap(
        ax,
        matrix,
        title="Noise perturbation: detection",
        xlabel="",
        xlabels=xlabels,
        x_fontsize=5.2,
        metric="score",
    )
    ax.axvline(3.5, color="white", lw=1.4)
    add_noise_group_labels(ax)
    return im


def draw_noisy_fdr_heatmap(ax):
    table = pd.read_csv(NOISY_MECH)
    table["method"] = table["method"].map(METHOD_RENAME).fillna(table["method"])
    table = table[table["method"].isin(METHOD_ORDER_NOISY)].copy()
    table = _mask_noisy_unavailable(table, "fdr_top5pct")
    table["noise_label"] = table["noise_level"].map({"low_noise": "Low", "high_noise": "High"})
    table["col"] = table["noise_label"] + "\n" + table["range_class"].map(RANGE_LABEL)
    columns = [f"{noise}\n{RANGE_LABEL[r]}" for noise in ["Low", "High"] for r in RANGE_ORDER]
    matrix = (
        table.pivot_table(index="method", columns="col", values="fdr_top5pct", aggfunc="mean")
        .reindex(METHOD_ORDER_NOISY)
        .reindex(columns=columns)
    )
    xlabels = ["Contact", "Short", "Medium", "Long\nniche"] * 2
    im = draw_heatmap(
        ax,
        matrix,
        title="Noise perturbation: false positives",
        xlabel="",
        xlabels=xlabels,
        x_fontsize=5.2,
        metric="fdr",
    )
    ax.axvline(3.5, color="white", lw=1.4)
    add_noise_group_labels(ax)
    return im


def add_noise_group_labels(ax) -> None:
    ax.text(0.245, -0.17, "low noise", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.755, -0.17, "high noise", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=COLORS["muted"])
    ax.plot([0.05, 0.44], [-0.12, -0.12], transform=ax.transAxes, color=COLORS["line"], lw=0.7, clip_on=False)
    ax.plot([0.56, 0.95], [-0.12, -0.12], transform=ax.transAxes, color=COLORS["line"], lw=0.7, clip_on=False)


def _mask_noisy_unavailable(table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    summary = pd.read_csv(NOISY_SUMMARY)
    unavailable = summary.loc[summary["aupr"].isna(), ["noise_level", "method"]].drop_duplicates()
    if unavailable.empty:
        return table
    out = table.merge(unavailable.assign(_unavailable=True), on=["noise_level", "method"], how="left")
    unavailable_mask = out["_unavailable"].eq(True)
    out.loc[unavailable_mask, value_column] = np.nan
    return out.drop(columns=["_unavailable"])


def draw_heatmap(ax, matrix: pd.DataFrame, title: str, xlabel: str, xlabels: list[str], x_fontsize: float = 5.8, metric: str = "score"):
    if metric == "fdr":
        cmap = LinearSegmentedColormap.from_list("spatialself_fdr", ["#EDF5F3", "#EEDBC8", "#C77A52", "#7A332A"])
    else:
        cmap = LinearSegmentedColormap.from_list("spatialself_heat", ["#F3F4F4", "#D5E6E3", "#8EC1BC", "#3E8E8B", "#125C69"])
    cmap.set_bad("#F7F7F7")
    data = matrix.to_numpy(dtype=float)
    vmax = 1.0 if metric == "fdr" else 0.85
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=7.4, loc="left", pad=5, color=COLORS["ink"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=6.2, labelpad=5, color=COLORS["muted"])
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(xlabels, fontsize=x_fontsize)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=5.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "SpatialSelf":
            tick.set_fontweight("bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.6, color="#9A9A9A")
                continue
            if metric == "fdr":
                color = "white" if value >= 0.62 else COLORS["ink"]
            else:
                color = "white" if value >= 0.48 else COLORS["ink"]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.0, color=color)
    return im


def write_source_tables() -> None:
    main_mech = pd.read_csv(MAIN_MECH)
    main_summary = pd.read_csv(MAIN_SUMMARY)
    noisy_mech = pd.read_csv(NOISY_MECH)
    noisy_summary = pd.read_csv(NOISY_SUMMARY)
    main_mech.to_csv(OUT_DIR / "fig2_panel_d_source.csv", index=False)
    main_mech.to_csv(OUT_DIR / "fig2_panel_e_source.csv", index=False)
    main_summary.to_csv(OUT_DIR / "fig2_clean_overall_metrics_source.csv", index=False)
    noisy_mech.to_csv(OUT_DIR / "fig2_panel_f_source.csv", index=False)
    noisy_mech.to_csv(OUT_DIR / "fig2_panel_g_source.csv", index=False)
    noisy_mech.to_csv(OUT_DIR / "fig2_panels_f_g_source.csv", index=False)
    noisy_summary.to_csv(OUT_DIR / "fig2_noisy_overall_metrics_source.csv", index=False)


if __name__ == "__main__":
    main()
