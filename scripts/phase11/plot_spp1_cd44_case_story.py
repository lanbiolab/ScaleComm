#!/usr/bin/env python
"""Plot detailed SPP1-CD44 biological case story for Visium HD CRC."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

TEXT = "#202020"
GRID = "#D8D8D8"
NICHE_COLORS = {
    "tumor_myeloid_boundary": "#4C78A8",
    "tumor_immune_boundary": "#72B7B2",
    "tumor_stromal_boundary": "#59A14F",
    "vascular_stromal_boundary": "#B07A4B",
    "tumor_core": "#B8B8B8",
    "epithelial_region": "#C17DA1",
}


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    spatial = pd.read_csv(RESULT_DIR / "spp1_cd44_case_spatial_features.csv")
    niche = pd.read_csv(RESULT_DIR / "spp1_cd44_case_niche_summary.csv")
    targets = pd.read_csv(RESULT_DIR / "spp1_cd44_case_target_gene_summary.csv")
    controls = pd.read_csv(RESULT_DIR / "spp1_cd44_case_control_summary.csv")
    edges = pd.read_csv(RESULT_DIR / "spp1_cd44_case_edge_summary.csv")

    source = build_source(spatial, niche, targets, controls, edges)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"spp1_cd44_case_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.35), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1, 1, 1.05, 1.1],
        height_ratios=[1.0, 1.0, 0.85],
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.12,
        wspace=0.44,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[0:2, 2])
    ax_f = fig.add_subplot(gs[0:2, 3])
    ax_g = fig.add_subplot(gs[2, 0:2])
    ax_h = fig.add_subplot(gs[2, 2:4])

    plot_spatial(ax_a, spatial, "spatialself_spp1_cd44_incoming", "SPP1-CD44 incoming", cmap="YlGn")
    plot_spatial(ax_b, spatial, "SPP1", "SPP1 ligand", cmap="YlOrBr")
    plot_spatial(ax_c, spatial, "CD44", "CD44 receptor", cmap="PuBuGn")
    plot_spatial(ax_d, spatial, "target_signature", "Target signature", cmap="YlGnBu")
    plot_niche_summary(ax_e, source["niche_summary"])
    plot_target_genes(ax_f, source["target_summary"])
    plot_controls(ax_g, source["controls"])
    plot_edge_flow(ax_h, source["edge_flow"])

    for label, ax in zip(["a", "b", "c", "d", "e", "f", "g", "h"], [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]):
        add_panel_label(ax, label)

    fig.suptitle(
        "SPP1-CD44 links tumor-myeloid boundary signaling to receiver target programs",
        x=0.075,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.928,
        "SpatialSelf SPP1-CD44 incoming localizes to tumor-myeloid niches and tracks myeloid/remodeling target signatures while negative controls are suppressed.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "This is observational spatial and transcriptomic consistency, not perturbational proof. Target signature uses curated SPP1-CD44-associated receiver genes present in the full-transcriptome matrix.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_spp1_cd44_case_story")
    plt.close(fig)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_source(
    spatial: pd.DataFrame,
    niche: pd.DataFrame,
    targets: pd.DataFrame,
    controls: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    target_summary = targets[targets["feature"].ne("target_signature")].head(10).copy()
    niche_order = niche.sort_values("mean_spatialself_spp1_cd44_incoming", ascending=False)["niche"].tolist()
    niche["niche"] = pd.Categorical(niche["niche"], categories=niche_order, ordered=True)
    controls = controls[controls["control_type"].isin(["observed", "shuffled_receiver", "shuffled_lr_identity", "wrong_niche_masked", "wrong_receiver_masked", "wrong_type_masked"])].copy()
    edge_flow = edges.head(8).copy()
    return {
        "spatial_features": spatial,
        "niche_summary": niche.sort_values("niche"),
        "target_summary": target_summary,
        "controls": controls,
        "edge_flow": edge_flow,
    }


def plot_spatial(ax: plt.Axes, table: pd.DataFrame, column: str, title: str, cmap: str) -> None:
    values = pd.to_numeric(table[column], errors="coerce").fillna(0).to_numpy(dtype=float)
    vmax = np.nanquantile(values, 0.985)
    vmin = np.nanquantile(values, 0.02) if values.min() < 0 else 0
    sc = ax.scatter(table["x"], -table["y"], c=values, s=3.8, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-8), linewidths=0, rasterized=True)
    ax.set_title(title, loc="left", pad=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.8, length=2)


def plot_niche_summary(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.sort_values("mean_spatialself_spp1_cd44_incoming", ascending=True)
    y = np.arange(len(plot))
    colors = [NICHE_COLORS.get(n, "#B8B8B8") for n in plot["niche"].astype(str)]
    ax.barh(y - 0.18, plot["mean_spatialself_spp1_cd44_incoming"], height=0.32, color=colors, edgecolor="none", label="incoming")
    ax.barh(y + 0.18, plot["mean_target_signature"], height=0.32, color="#333333", alpha=0.55, edgecolor="none", label="target signature")
    ax.set_yticks(y)
    ax.set_yticklabels([short_niche(x) for x in plot["niche"].astype(str)], fontsize=5.8)
    ax.set_title("Niche enrichment", loc="left", pad=4)
    ax.set_xlabel("mean value")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.4)


def plot_target_genes(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.sort_values("spearman_with_spatialself_incoming", ascending=True)
    y = np.arange(len(plot))
    ax.barh(y, plot["spearman_with_spatialself_incoming"], color="#4C78A8", edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["feature"], fontsize=5.8)
    ax.set_title("Receiver target genes", loc="left", pad=4)
    ax.set_xlabel("Spearman with incoming")
    ax.grid(axis="x", color=GRID, linewidth=0.45)


def plot_controls(ax: plt.Axes, table: pd.DataFrame) -> None:
    order = ["observed", "shuffled_receiver", "shuffled_lr_identity", "wrong_niche_masked", "wrong_receiver_masked", "wrong_type_masked"]
    labels = ["observed", "shuffle recv.", "shuffle LR", "wrong niche", "wrong recv.", "wrong type"]
    plot = table.set_index("control_type").reindex(order).reset_index()
    x = np.arange(len(plot))
    colors = ["#2F6B4F"] + ["#B8B8B8"] * (len(plot) - 1)
    ax.bar(x - 0.18, plot["target_overlap_top_n"].fillna(0), width=0.34, color=colors, edgecolor="none", label="target overlap")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, plot["delta_cv_r2"].fillna(0), color="#456C9B", marker="o", markersize=3, linewidth=1.0, label="delta R2")
    ax.set_title("Control specificity", loc="left", pad=4)
    ax.set_ylabel("top-target overlap")
    ax2.set_ylabel("delta R2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=5.5)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax2.spines["top"].set_visible(False)


def plot_edge_flow(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.sort_values("total_score", ascending=True)
    labels = [f"{short_niche(s)} -> {short_niche(r)}" for s, r in zip(plot["sender_niche"], plot["receiver_niche"])]
    y = np.arange(len(plot))
    colors = ["#2F6B4F" if "myeloid" in str(s) else "#B8B8B8" for s in plot["sender_niche"]]
    ax.barh(y, plot["total_score"], color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.set_title("Top SPP1-CD44 niche flows", loc="left", pad=4)
    ax.set_xlabel("total edge score")
    ax.grid(axis="x", color=GRID, linewidth=0.45)


def short_niche(value: str) -> str:
    return (
        value.replace("tumor_", "")
        .replace("_boundary", "")
        .replace("vascular_", "vascular ")
        .replace("epithelial_region", "epithelial")
        .replace("_", " ")
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
