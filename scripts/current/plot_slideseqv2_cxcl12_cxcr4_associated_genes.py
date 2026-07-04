#!/usr/bin/env python
"""Associated marker genes for Cxcl12-Cxcr4 SpatialSelf activity in Slide-seqV2 hippocampus."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_cxcl12_cxcr4_associated_genes_v1"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
SPATIAL = "#2F78A8"
RESPONSE = "#C86B53"
BG = "#FFFFFF"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()
    configure()
    cells = pd.read_csv(DATA_ROOT / "cells.csv", usecols=["cell_id", "cell_type"])
    expr = pd.read_csv(DATA_ROOT / "expression.csv")
    pred = pd.read_csv(RESULT_ROOT / "merged_predictions.csv", usecols=["ligand", "receptor", "receiver_id", "score"])
    source = build_source(cells, expr, pred)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)
    fig = plot_associated(source)
    out = OUT_DIR / "slideseqv2_cxcl12_cxcr4_associated_genes"
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


def build_source(cells: pd.DataFrame, expr: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    annotated_ids = set(cells.loc[cells["cell_type"].ne("unannotated"), "cell_id"].astype(str))
    expr = expr[expr["cell_id"].astype(str).isin(annotated_ids)].copy()
    sub = pred[(pred["ligand"].eq(LIGAND)) & (pred["receptor"].eq(RECEPTOR))].copy()
    receiver = sub.groupby("receiver_id", as_index=False).agg(score=("score", "mean"))
    table = expr.merge(receiver, left_on="cell_id", right_on="receiver_id", how="left")
    table["score"] = table["score"].fillna(0.0)
    pos = table.loc[table["score"].gt(0), "score"]
    cutoff = float(pos.quantile(0.985)) if len(pos) else 1.0
    table["hotspot"] = table["score"].ge(cutoff)
    gene_cols = [c for c in table.columns if c not in {"cell_id", "receiver_id", "score", "hotspot"}]
    X = table[gene_cols].fillna(0).astype(float)
    y = table["score"].to_numpy(dtype=float)
    rows = []
    for gene in gene_cols:
        v = X[gene].to_numpy(dtype=float)
        if np.nanstd(v) == 0 or np.nanstd(y) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(v, y)[0, 1])
        hot = v[table["hotspot"].to_numpy()]
        bg = v[~table["hotspot"].to_numpy()]
        delta = float(np.nanmean(hot) - np.nanmean(bg)) if len(hot) else 0.0
        rows.append({"gene": gene, "pearson_r": corr, "hotspot_minus_bg": delta, "mean_hotspot": float(np.nanmean(hot)) if len(hot) else 0.0, "mean_background": float(np.nanmean(bg)) if len(bg) else 0.0})
    assoc = pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)
    top_genes = assoc.head(12)["gene"].tolist()
    heat = table[["hotspot"] + top_genes].copy()
    heat["group"] = np.where(heat["hotspot"], "hotspot receivers", "other beads")
    summary = heat.groupby("group")[top_genes].mean().T.reset_index().rename(columns={"index": "gene"})
    for col in ["hotspot receivers", "other beads"]:
        if col not in summary.columns:
            summary[col] = 0.0
    summary["delta"] = summary["hotspot receivers"] - summary["other beads"]
    return {"associated_genes": assoc, "hotspot_expression_summary": summary, "receiver_gene_table": table[["cell_id", "score", "hotspot"] + top_genes]}


def plot_associated(source: dict[str, pd.DataFrame]) -> plt.Figure:
    assoc = source["associated_genes"].head(12).copy()
    summary = source["hotspot_expression_summary"].copy()
    fig = plt.figure(figsize=(5.45, 2.75), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.09, right=0.985, bottom=0.17, top=0.86, wspace=0.36, width_ratios=[1.0, 0.88])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_corr_rank(ax_a, assoc)
    plot_hotspot_heatmap(ax_b, summary)
    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


def plot_corr_rank(ax: plt.Axes, assoc: pd.DataFrame) -> None:
    table = assoc.sort_values("pearson_r", ascending=True)
    y = np.arange(len(table))
    colors = [RESPONSE if g in {"Cxcr4", "Gfap", "Aqp4", "Slc1a2", "Pdgfra"} else SPATIAL for g in table["gene"]]
    ax.barh(y, table["pearson_r"], color=colors, height=0.58, edgecolor="none", alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(table["gene"], fontsize=5.8)
    ax.axvline(0, color="#8A8177", lw=0.52)
    ax.set_xlabel("correlation with SpatialSelf score", fontsize=6.0)
    ax.set_title("score-associated genes", fontsize=6.8, pad=2.0)
    ax.grid(axis="x", color=GRID, lw=0.42)
    ax.tick_params(axis="x", labelsize=5.3, length=2.0, pad=1.0)
    ax.tick_params(axis="y", length=0, pad=1.5)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)


def plot_hotspot_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = summary.copy()
    values = table[["other beads", "hotspot receivers"]].to_numpy(dtype=float)
    z = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-9)
    cmap = LinearSegmentedColormap.from_list("assoc_heat", ["#E6C06A", "#F7F4EC", "#2F78A8"])
    im = ax.imshow(z, aspect="auto", cmap=cmap, vmin=-1.2, vmax=1.2, interpolation="nearest")
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["gene"], fontsize=5.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nbeads", "hotspot\nreceivers"], fontsize=5.6)
    ax.set_title("hotspot expression pattern", fontsize=6.8, pad=2.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.18, -0.18, 0.62, 0.045])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.35)
    cb.set_ticks([-1, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.6, length=1.8, width=0.35, pad=1)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.036, box.y1 + 0.016, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
