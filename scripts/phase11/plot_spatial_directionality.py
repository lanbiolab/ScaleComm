#!/usr/bin/env python
"""Plot COMMOT-style spatial directionality summaries."""

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

METHOD_ORDER = ["spatialself", "commot", "cellchat", "cellnest"]
METHOD_LABEL = {
    "spatialself": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "cellnest": "CellNEST",
}
METHOD_COLORS = {
    "spatialself": "#2F6B4F",
    "commot": "#456C9B",
    "cellchat": "#7D668E",
    "cellnest": "#B07A4B",
}
TEXT = "#202020"
GRID = "#D7D7D7"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(RESULT_DIR / "directionality_lr_summary.csv")
    niche = pd.read_csv(RESULT_DIR / "directionality_niche_flow.csv")
    vectors = pd.read_csv(RESULT_DIR / "directionality_receiver_vectors.csv")
    edge = pd.read_csv(RESULT_DIR / "directionality_edges.csv")

    source = build_source(summary, niche, vectors, edge)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"directionality_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.05, 1.05],
        height_ratios=[1.1, 0.95],
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.12,
        wspace=0.42,
        hspace=0.52,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])

    plot_spatial_vectors(ax_a, source["spatial_vectors"])
    plot_expected_fraction(ax_b, source["expected_fraction"])
    plot_best_flow(ax_c, source["best_flow"])
    plot_niche_flow_heatmap(ax_d, source["niche_heatmap"])

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Spatial signaling directionality and niche flow in Visium HD CRC",
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
        0.925,
        "Candidate sender-to-receiver edges define weighted flow vectors; SPP1-CD44 is dominated by local tumor-myeloid flow, while expected niche flow varies by LR and method.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "Directionality is inferred from scored spatial edges, not time-resolved transport. Expected flows are pre-specified by LR/niche context.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_spatial_directionality")
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
            "axes.linewidth": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_source(summary: pd.DataFrame, niche: pd.DataFrame, vectors: pd.DataFrame, edge: pd.DataFrame) -> dict[str, pd.DataFrame]:
    focus = "SPP1-CD44"
    spatial = vectors[(vectors["method"].isin(["spatialself", "commot"])) & (vectors["lr_pair"].eq(focus))].copy()
    spatial = spatial[spatial["incoming_score"].gt(0)].copy()
    spatial["rank"] = spatial.groupby("method")["incoming_score"].rank(ascending=False, method="first")
    spatial = spatial[spatial["rank"].le(500)].copy()
    spatial = spatial.sort_values(["method", "rank"])

    expected = summary[summary["method"].isin(METHOD_ORDER) & summary["lr_pair"].isin(["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR"])].copy()
    expected["method_label"] = expected["method"].map(METHOD_LABEL)

    best = summary[summary["method"].isin(["spatialself", "commot"])].copy()
    best = best[best["lr_pair"].isin(["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2"])]
    best["method_label"] = best["method"].map(METHOD_LABEL)

    heat = niche[
        niche["method"].eq("spatialself")
        & niche["lr_pair"].eq(focus)
        & niche["sender_niche"].isin(["tumor_myeloid_boundary", "tumor_immune_boundary", "tumor_stromal_boundary", "tumor_core"])
        & niche["receiver_niche"].isin(["tumor_myeloid_boundary", "tumor_immune_boundary", "tumor_stromal_boundary", "tumor_core"])
    ].copy()
    return {
        "spatial_vectors": spatial,
        "expected_fraction": expected,
        "best_flow": best,
        "niche_heatmap": heat,
        "edge_subset": edge[(edge["method"].isin(["spatialself", "commot"])) & (edge["lr_pair"].eq(focus))].head(5000),
    }


def plot_spatial_vectors(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("SPP1-CD44 receiver flow field", loc="left", pad=4)
    for method in ["spatialself", "commot"]:
        sub = table[table["method"].eq(method)].copy()
        if sub.empty:
            continue
        alpha = 0.52 if method == "spatialself" else 0.4
        ax.scatter(sub["receiver_x"], -sub["receiver_y"], s=4, color=METHOD_COLORS[method], alpha=alpha, linewidths=0, label=METHOD_LABEL[method])
        step = max(len(sub) // 90, 1)
        q = sub.iloc[::step]
        scale = 420
        ax.quiver(
            q["receiver_x"],
            -q["receiver_y"],
            q["vector_dx"] * scale,
            -q["vector_dy"] * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0022,
            color=METHOD_COLORS[method],
            alpha=0.62,
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.2)


def plot_expected_fraction(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR"]
    methods = ["spatialself", "commot", "cellnest", "cellchat"]
    x = np.arange(len(pairs))
    width = 0.18
    for i, method in enumerate(methods):
        vals = table[table["method"].eq(method)].set_index("lr_pair").reindex(pairs)["expected_flow_score_fraction"].fillna(0)
        ax.bar(x + (i - 1.5) * width, vals, width, color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABEL[method])
    ax.set_title("Pre-specified niche-flow support", loc="left", pad=4)
    ax.set_ylabel("score fraction")
    ax.set_ylim(0, 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_best_flow(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table[table["lr_pair"].isin(["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2"])].copy()
    plot["pair_method"] = plot["lr_pair"] + "\n" + plot["method_label"]
    plot = plot.sort_values(["lr_pair", "method"])
    y = np.arange(len(plot))
    colors = [METHOD_COLORS[m] if exp else "#B8B8B8" for m, exp in zip(plot["method"], plot["best_flow_is_expected"])]
    ax.barh(y, plot["best_flow_score"], color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["pair_method"], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_title("Top niche-to-niche flow", loc="left", pad=4)
    ax.set_xlabel("flow score")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.text(0.98, 0.04, "grey: top flow not pre-specified", transform=ax.transAxes, ha="right", fontsize=5.5, color="#555555")


def plot_niche_flow_heatmap(ax: plt.Axes, table: pd.DataFrame) -> None:
    senders = ["tumor_myeloid_boundary", "tumor_immune_boundary", "tumor_stromal_boundary", "tumor_core"]
    receivers = ["tumor_myeloid_boundary", "tumor_immune_boundary", "tumor_stromal_boundary", "tumor_core"]
    matrix = table.pivot_table(index="sender_niche", columns="receiver_niche", values="total_score", aggfunc="sum").reindex(index=senders, columns=receivers).fillna(0)
    arr = np.log1p(matrix.to_numpy(dtype=float))
    im = ax.imshow(arr, cmap="YlGn", aspect="auto")
    ax.set_title("SpatialSelf SPP1-CD44 niche flow", loc="left", pad=4)
    ax.set_xticks(np.arange(len(receivers)))
    ax.set_xticklabels([short_niche(x) for x in receivers], rotation=25, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(senders)))
    ax.set_yticklabels([short_niche(x) for x in senders], fontsize=6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.0f}", ha="center", va="center", fontsize=5.2, color="#1f1f1f")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("log1p flow", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4, length=2)


def short_niche(value: str) -> str:
    return (
        value.replace("tumor_", "")
        .replace("_boundary", "")
        .replace("_", " ")
        .replace("stromal", "stromal")
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
