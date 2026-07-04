#!/usr/bin/env python
"""Spatial maps of mechanism-specific LR activity in Slide-seqV2 hippocampus."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_mechanism_activity_maps_v1"

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
TISSUE = "#D8D2C8"
BG = "#FFFFFF"
CONTACT = "#8A8177"
PARACRINE = "#C86B53"
LONG = "#2F78A8"
GRID = "#E8E2D6"

MECH_LABELS = {
    "juxtacrine_contact": "contact",
    "short_medium_paracrine": "paracrine",
    "long_range_chemokine_niche": "long-range",
}
MECH_COLORS = {"contact": CONTACT, "paracrine": PARACRINE, "long-range": LONG}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()
    configure()
    cells = pd.read_csv(DATA_ROOT / "cells.csv")
    pred = pd.read_csv(RESULT_ROOT / "merged_predictions.csv", usecols=["ligand", "receptor", "receiver_id", "mechanism", "score"])
    source = build_source(cells, pred)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)
    fig = plot_maps(source)
    out = OUT_DIR / "slideseqv2_mechanism_activity_maps"
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


def build_source(cells: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    annotated = cells[cells["cell_type"].ne("unannotated")][["cell_id", "x_plot", "y_plot", "cell_type"]].copy()
    pred = pred[~pred["mechanism"].astype(str).str.startswith("negative")].copy()
    pred["mechanism_label"] = pred["mechanism"].map(MECH_LABELS)
    rows = []
    for mechanism, sub in pred.dropna(subset=["mechanism_label"]).groupby("mechanism_label"):
        cutoff = float(sub["score"].quantile(0.99))
        top = sub[sub["score"].ge(cutoff)]
        count = top.groupby("receiver_id", as_index=False).agg(activity=( "score", "size"), mean_score=("score", "mean"))
        count["mechanism_label"] = mechanism
        rows.append(count)
    activity = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["receiver_id", "activity", "mean_score", "mechanism_label"])
    wide = activity.pivot_table(index="receiver_id", columns="mechanism_label", values="activity", aggfunc="sum", fill_value=0).reset_index()
    for label in ["contact", "paracrine", "long-range"]:
        if label not in wide.columns:
            wide[label] = 0
    maps = annotated.merge(wide, left_on="cell_id", right_on="receiver_id", how="left")
    for label in ["contact", "paracrine", "long-range"]:
        maps[label] = maps[label].fillna(0).astype(float)
        vmax = max(float(np.nanquantile(maps[label], 0.995)), 1.0)
        maps[f"{label}_scaled"] = np.clip(maps[label] / vmax, 0, 1)
    values = maps[["contact", "paracrine", "long-range"]]
    maps["dominant"] = values.idxmax(axis=1)
    maps.loc[values.max(axis=1).eq(0), "dominant"] = "none"
    summary = maps["dominant"].value_counts().rename_axis("dominant").reset_index(name="n_beads")
    return {"mechanism_maps": maps, "mechanism_activity_long": activity, "dominant_summary": summary}


def plot_maps(source: dict[str, pd.DataFrame]) -> plt.Figure:
    maps = source["mechanism_maps"]
    fig = plt.figure(figsize=(7.35, 2.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, left=0.04, right=0.99, bottom=0.13, top=0.86, wspace=0.17)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    plot_activity(axes[0], maps, "contact", CONTACT, "contact LR activity")
    plot_activity(axes[1], maps, "paracrine", PARACRINE, "paracrine LR activity")
    plot_activity(axes[2], maps, "long-range", LONG, "long-range LR activity")
    plot_dominant(axes[3], maps)
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        panel_label(fig, ax, label)
    return fig


def plot_activity(ax: plt.Axes, maps: pd.DataFrame, label: str, color: str, title: str) -> None:
    scaled = f"{label}_scaled"
    cmap = LinearSegmentedColormap.from_list(label, ["#F3EFE7", "#DAD4CA", color])
    frame = maps.sort_values(scaled)
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.0, c=TISSUE, alpha=0.22, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    active = frame[frame[label].gt(0)]
    ax.scatter(active["x_plot"], active["y_plot"], s=6.3, c=active[scaled], cmap=cmap, vmin=0, vmax=1, alpha=0.98, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    set_title(ax, title)
    add_spatial_style(ax, maps)
    ax.text(0.04, 0.055, f"n={len(active):,}", transform=ax.transAxes, fontsize=4.9, color=MUTED, ha="left", va="bottom")


def plot_dominant(ax: plt.Axes, maps: pd.DataFrame) -> None:
    ax.scatter(maps["x_plot"], maps["y_plot"], s=3.0, c=TISSUE, alpha=0.18, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    for label, color in MECH_COLORS.items():
        sub = maps[maps["dominant"].eq(label)]
        ax.scatter(sub["x_plot"], sub["y_plot"], s=5.6, c=color, alpha=0.88, linewidths=0, edgecolors="none", antialiaseds=False, rasterized=False)
    set_title(ax, "dominant activity mode")
    add_spatial_style(ax, maps)
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in MECH_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=4.8, handlelength=0.8, handletextpad=0.25, borderaxespad=0.08)


def add_spatial_style(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    xspan = float(data["x_plot"].max() - data["x_plot"].min())
    yspan = float(data["y_plot"].max() - data["y_plot"].min())
    pad = max(xspan, yspan) * 0.018
    ax.set_xlim(float(data["x_plot"].min()) - pad, float(data["x_plot"].max()) + pad)
    ax.set_ylim(float(data["y_plot"].min()) - pad, float(data["y_plot"].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#7A716A")
        ax.spines[side].set_linewidth(0.54)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=6.8, pad=2.0, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.020, box.y1 + 0.014, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
