#!/usr/bin/env python
"""Plot a first-pass Slide-seqV2 extension-platform SpatialSelf figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_extension/standardized_inputs/slideseqv2_mouse_cerebellum_subset"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_extension/spatialself_scale_smoke_e60"
FIG_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_extension/figures/slideseqv2_extension_platform_v2"


CELL_COLORS = {
    "glial_cluster": "#93cc82",
    "oligodendrocyte_cluster": "#b9c1d1",
    "granule_cluster": "#b9e5fa",
    "purkinje_cluster": "#fbc9c4",
    "interneuron_cluster": "#f3d78a",
    "neuronal_cluster": "#ffeebc",
    "unassigned_cluster": "#d6d2ca",
}
ACCENT = "#2F78A8"
WARM = "#C86B53"
TEXT = "#282522"
GRID = "#E6E1D7"
BG = "#F4F1E8"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    cells = pd.read_csv(DATA_ROOT / "cells.csv")
    lr_summary = pd.read_csv(RESULT_ROOT / "lr_ranking_summary.csv")
    merged = pd.read_csv(RESULT_ROOT / "merged_predictions.csv")
    scale = pd.read_csv(
        RESULT_ROOT
        / "slideseqv2_mouse_cerebellum_subset__spatialself_wavelet_scale_lr_scale_distribution.csv"
    )

    selected_lrs = select_lrs(lr_summary)
    hotspot_tables = {lr: receiver_hotspot(cells, merged, lr) for lr in selected_lrs[:3]}

    fig = plt.figure(figsize=(8.8, 6.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=7,
        height_ratios=[1.16, 1.0, 0.92],
        width_ratios=[1.0, 1.0, 1.0, 0.24, 0.88, 0.88, 0.88],
        left=0.06,
        right=0.985,
        top=0.955,
        bottom=0.07,
        wspace=0.34,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0:2, 0:3])
    ax_b = fig.add_subplot(gs[0, 4:7])
    ax_c = fig.add_subplot(gs[1, 4])
    ax_d = fig.add_subplot(gs[1, 5])
    ax_e = fig.add_subplot(gs[1, 6])
    ax_f = fig.add_subplot(gs[2, 0:7])

    plot_cell_map(ax_a, cells)
    plot_lr_ranking(ax_b, lr_summary, selected_lrs)
    for ax, lr in zip([ax_c, ax_d, ax_e], selected_lrs[:3], strict=False):
        plot_hotspot(ax, hotspot_tables[lr], lr)
    plot_scale_heatmap(ax_f, scale, selected_lrs)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d, "d")
    add_panel_label(ax_e, "e")
    add_panel_label(ax_f, "f")

    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG_DIR / f"slideseqv2_extension_platform_v2.{ext}", **kwargs)
    plt.close(fig)

    source_dir = FIG_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    cells.to_csv(source_dir / "panel_a_cells.csv", index=False)
    lr_summary.to_csv(source_dir / "panel_b_lr_ranking.csv", index=False)
    for lr, table in hotspot_tables.items():
        table.to_csv(source_dir / f"panel_hotspot_{clean_name(lr)}.csv", index=False)
    scale.to_csv(source_dir / "panel_f_scale_distribution.csv", index=False)
    print(FIG_DIR / "slideseqv2_extension_platform_v2.png")
    print(FIG_DIR / "slideseqv2_extension_platform_v2.pdf")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#6A625B",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
        }
    )


def select_lrs(lr_summary: pd.DataFrame) -> list[str]:
    preferred = ["Nrg2-Erbb4", "Reln-Vldlr", "Ptn-Sdc3", "Gdf10-Bmpr1a", "Apoe-Lrp1"]
    available = {f"{r.ligand}-{r.receptor}" for r in lr_summary.itertuples(index=False)}
    chosen = [lr for lr in preferred if lr in available]
    for row in lr_summary.itertuples(index=False):
        name = f"{row.ligand}-{row.receptor}"
        if name not in chosen and not str(row.mechanism).startswith("negative"):
            chosen.append(name)
        if len(chosen) >= 6:
            break
    return chosen


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    order = sorted(cells["cell_type"].unique())
    for cell_type in order:
        sub = cells[cells["cell_type"] == cell_type]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=4.2,
            c=CELL_COLORS.get(cell_type, "#d6d2ca"),
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            label=pretty_cell(cell_type),
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Slide-seqV2 mouse cerebellum", fontsize=8.5, pad=5, loc="left")
    ax.text(
        0.015,
        0.025,
        "5,000-bead spatial subset",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#5A524C",
        ha="left",
        va="bottom",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.035),
        ncol=3,
        fontsize=6.2,
        handletextpad=0.25,
        columnspacing=0.9,
        markerscale=2.0,
    )


def plot_lr_ranking(ax: plt.Axes, lr_summary: pd.DataFrame, selected_lrs: list[str]) -> None:
    table = lr_summary.copy()
    table["lr_pair"] = table["ligand"] + "-" + table["receptor"]
    table = table.sort_values("q95_score", ascending=True).tail(10)
    colors = []
    for row in table.itertuples(index=False):
        if row.lr_pair in selected_lrs[:3]:
            colors.append(ACCENT)
        elif str(row.mechanism).startswith("negative"):
            colors.append("#C8C1B7")
        else:
            colors.append("#BFD7D6")
    ax.barh(np.arange(len(table)), table["q95_score"], color=colors, height=0.68, edgecolor="none")
    ax.set_yticks(np.arange(len(table)))
    labels = list(table["lr_pair"])
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlabel("top-edge score (q95)", fontsize=7.2)
    ax.set_xlim(0, max(1.0, float(table["q95_score"].max()) * 1.04))
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Prioritized LR programs", fontsize=8.2, pad=4, loc="left")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.55)


def receiver_hotspot(cells: pd.DataFrame, merged: pd.DataFrame, lr_pair: str) -> pd.DataFrame:
    ligand, receptor = lr_pair.split("-")
    sub = merged[(merged["ligand"] == ligand) & (merged["receptor"] == receptor)].copy()
    cutoff = float(sub["score"].quantile(0.80))
    sub = sub[sub["score"] >= cutoff]
    hot = sub.groupby("receiver_id", as_index=False).agg(
        hotspot_score=("score", "mean"),
        n_edges=("score", "size"),
    )
    out = cells[["cell_id", "x", "y", "cell_type", "niche"]].merge(hot, left_on="cell_id", right_on="receiver_id", how="left")
    out["hotspot_score"] = out["hotspot_score"].fillna(0.0)
    if out["hotspot_score"].max() > 0:
        out["hotspot_scaled"] = out["hotspot_score"] / out["hotspot_score"].max()
    else:
        out["hotspot_scaled"] = 0.0
    out["n_edges"] = out["n_edges"].fillna(0).astype(int)
    return out


def plot_hotspot(ax: plt.Axes, table: pd.DataFrame, lr_pair: str) -> None:
    ax.set_aspect("equal")
    ax.scatter(table["x"], table["y"], s=2.0, c="#D7D1C7", alpha=0.38, linewidths=0, rasterized=True)
    active = table[table["hotspot_scaled"] > 0].copy()
    if not active.empty:
        active = active.sort_values("hotspot_scaled")
        cmap = LinearSegmentedColormap.from_list("hotspot", [BG, "#E6C06A", ACCENT])
        ax.scatter(
            active["x"],
            active["y"],
            s=4.2,
            c=active["hotspot_scaled"],
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0,
            alpha=0.92,
            rasterized=True,
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(lr_pair, fontsize=7.6, pad=3.5)


def plot_scale_heatmap(ax: plt.Axes, scale: pd.DataFrame, selected_lrs: list[str]) -> None:
    scale = scale.copy()
    scale["lr_pair"] = scale["ligand"] + "-" + scale["receptor"]
    wanted = selected_lrs + [
        lr
        for lr in scale["lr_pair"].tolist()
        if lr not in selected_lrs and "negative_control" not in str(scale.loc[scale["lr_pair"] == lr, "mechanism"].iloc[0])
    ]
    wanted = wanted[:10]
    table = scale.set_index("lr_pair").loc[wanted].reset_index()
    value_cols = [c for c in table.columns if c.startswith("scale_")]
    values = table[value_cols].to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=LinearSegmentedColormap.from_list("scale", [BG, "#B9E5FA", ACCENT]), vmin=0, vmax=max(0.75, values.max()))
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["lr_pair"], fontsize=6.5)
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels([f"S{i}" for i in range(len(value_cols))], fontsize=6.5)
    ax.set_xlabel("learned graph-wavelet scale", fontsize=7.2)
    ax.set_title("LR-specific scale preferences", fontsize=8.2, pad=4, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.014)
    cbar.ax.tick_params(labelsize=6.2, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("weight", fontsize=6.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def pretty_cell(label: str) -> str:
    return label.replace("_cluster", "").replace("_", " ")


def clean_name(name: str) -> str:
    return name.replace("-", "_").replace("/", "_")


if __name__ == "__main__":
    main()
