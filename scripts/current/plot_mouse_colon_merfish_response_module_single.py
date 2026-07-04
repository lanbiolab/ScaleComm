#!/usr/bin/env python
"""Single-map DSS9 response-module view for the Cxcl10-Cxcr3 MERFISH story."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_method_downstream_v3 import (
    BASE,
    GENE_CMAP,
    LR_CASE,
    MUTED,
    RESPONSE_GENE_PANEL,
    _receiver_scores_for_method,
    load_expression_slice,
    set_spatial_limits,
    setup_style,
)


OUT = BASE / "figures/merfish_dss_response_module_single"


def robust_gene_module(cells: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    present = [gene for gene in genes if gene in cells.columns]
    if not present:
        raise ValueError("None of the requested module genes are present in expression table.")
    scaled = []
    for gene in present:
        x = np.log1p(cells[gene].to_numpy(float))
        positive = x[x > 0]
        vmax = float(np.quantile(positive, 0.98)) if len(positive) else 1.0
        vmax = max(vmax, 1e-6)
        scaled.append(np.clip(x / vmax, 0, 1))
    out = cells[["cell_id", "x", "y"]].copy()
    out["response_module"] = np.vstack(scaled).mean(axis=0)
    out["n_module_genes"] = len(present)
    return out


def spatialself_hotspots(stage: str = "DSS9", lr_pair: str = LR_CASE, replicate: int = 1) -> pd.DataFrame:
    scores, _ = _receiver_scores_for_method(stage, lr_pair, "SpatialSelf", replicate)
    ranked = scores.dropna().sort_values(ascending=False)
    k = max(5, int(np.floor(0.20 * len(ranked)))) if len(ranked) else 0
    return pd.DataFrame({"cell_id": ranked.index, "spatialself_score": ranked.values, "is_hotspot": np.arange(len(ranked)) < k})


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_low_high_bar(ax: plt.Axes) -> None:
    cax = ax.inset_axes([0.685, 0.060, 0.210, 0.018])
    for i in range(128):
        cax.add_patch(Rectangle((i / 128, 0), 1 / 128, 1, facecolor=GENE_CMAP(i / 127), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.6, labelsize=5.2)
    cax.set_xlabel("module score", fontsize=5.2, labelpad=0.5, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)


def plot_single_response_module() -> None:
    setup_style()
    mpl.rcParams.update({"font.size": 7.0, "axes.titlesize": 8.2})
    OUT.mkdir(parents=True, exist_ok=True)

    cells = load_expression_slice("DSS9", replicate=1).copy()
    module = robust_gene_module(cells, RESPONSE_GENE_PANEL)
    hotspot_scores = spatialself_hotspots()
    module = module.merge(hotspot_scores, on="cell_id", how="left")
    module["is_hotspot"] = module["is_hotspot"].astype("boolean").fillna(False).astype(bool)
    module["spatialself_score"] = module["spatialself_score"].fillna(0.0)
    module.to_csv(OUT / "dss9_cxcl10_cxcr3_response_module_cells.csv", index=False)

    score = module["response_module"].to_numpy(float)
    vmax = max(float(np.quantile(score[score > 0], 0.985)) if (score > 0).any() else 1.0, 1e-6)
    norm = np.clip(score / vmax, 0, 1)
    visible = norm > 0.025
    hotspot = module.loc[module["is_hotspot"]]

    fig, ax = plt.subplots(figsize=(3.35, 3.22))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.045, top=0.905)

    ax.scatter(module["x"], module["y"], s=1.20, color="#DCD7CE", alpha=0.22, linewidths=0, zorder=1)
    ax.scatter(
        module.loc[visible, "x"],
        module.loc[visible, "y"],
        c=norm[visible],
        s=3.2 + 18.0 * np.power(norm[visible], 0.78),
        cmap=GENE_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.93,
        marker="o",
        edgecolors="white",
        linewidths=0.08,
        zorder=2,
    )
    ax.scatter(
        hotspot["x"],
        hotspot["y"],
        s=42,
        facecolors="none",
        edgecolors="#265F5B",
        linewidths=0.82,
        alpha=0.96,
        zorder=5,
    )

    set_spatial_limits(ax, module)
    ax.set_title(f"DSS9 {LR_CASE}", loc="left", pad=1.9, fontsize=7.6)

    handles = [
        Line2D([0], [0], marker="o", lw=0, color=GENE_CMAP(0.82), markersize=4.2, label="response module"),
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            markerfacecolor="none",
            markeredgecolor="#265F5B",
            markeredgewidth=0.95,
            color="#265F5B",
            markersize=5.2,
            label="SpatialSelf hotspot",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.010, 0.985),
        frameon=False,
        fontsize=5.8,
        handletextpad=0.25,
        labelspacing=0.22,
        borderaxespad=0,
    )
    ax.text(
        0.015,
        0.030,
        "6-gene response module",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#5D574F",
    )
    draw_low_high_bar(ax)
    export(fig, OUT / "dss9_cxcl10_cxcr3_response_module_single")

    svg = OUT / "dss9_cxcl10_cxcr3_response_module_single.svg"
    print(f"{OUT} image_tags={svg.read_text(errors='ignore').count('<image')}")


if __name__ == "__main__":
    plot_single_response_module()
