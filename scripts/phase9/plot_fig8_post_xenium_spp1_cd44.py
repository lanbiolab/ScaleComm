#!/usr/bin/env python
"""Create a Nature-style Fig. 8 for the post-Xenium colon SPP1-CD44 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
)
STD_ROOT = DATA_ROOT / "standardized_inputs/post_xenium_human_colon_cancer_subset"
FIG_ROOT = PROJECT_ROOT / "outputs/phase9/fig8_post_xenium_spp1_cd44"

PATHS = {
    "rank": DATA_ROOT / "post_xenium_human_colon_cancer_method_lr_rank_table.csv",
    "local_fdr": DATA_ROOT
    / "post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_fdr_all.csv",
    "downstream_specificity": FIG_ROOT / "downstream_specificity/fig8_downstream_specificity.csv",
    "cells": STD_ROOT / "cells.csv",
    "candidates": STD_ROOT / "candidate_interactions.csv",
    "summary": STD_ROOT / "standardization_summary.csv",
    "pred_spatialself": DATA_ROOT
    / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
}

METHOD_ORDER = ["spatialself_calibrated", "commot", "cellchat", "cellnest", "spatalk"]
METHOD_LABELS = {
    "spatialself_calibrated": "SpatialSelf",
    "spatialself": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "cellnest": "CellNEST",
    "spatalk": "SpaTalk",
}
LR_LABELS = {
    "SPP1-CD44": "SPP1-CD44",
    "VEGFA-KDR": "VEGFA-KDR",
    "EGF-EGFR": "EGF-EGFR",
    "TGFB1-TGFBR2": "TGFB1-TGFBR2",
    "CXCL12-CXCR4": "CXCL12-CXCR4",
    "ACTB-KRT8": "ACTB-KRT8",
    "COL1A1-EPCAM": "COL1A1-EPCAM",
}
NULL_LABELS = {
    "within_region_permutation": "within\nregion",
    "cell_type_preserving_permutation": "cell type\npreserved",
    "lr_identity_permutation": "LR identity\nshuffle",
}

COLORS = {
    "ink": "#202733",
    "muted": "#667085",
    "axis": "#374151",
    "grid": "#E6E8EC",
    "hairline": "#D3D7DE",
    "signal": "#B85A4A",
    "signal_dark": "#84392F",
    "signal_soft": "#E8B8AD",
    "other": "#A9C9C5",
    "negative": "#9AA1AA",
    "negative_dark": "#606875",
    "blue": "#527AA3",
    "panel_tint": "#F7F8FA",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.7,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def canonical_pair(df: pd.DataFrame) -> pd.Series:
    return df["ligand"].astype(str) + "-" + df["receptor"].astype(str)


def read_inputs() -> dict[str, pd.DataFrame]:
    missing = [name for name, path in PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Fig. 8 source files: {missing}")
    return {name: pd.read_csv(path) for name, path in PATHS.items()}


def build_plot_data(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | dict[str, object]]:
    rank = inputs["rank"].copy()
    rank["pair"] = canonical_pair(rank)
    rank["is_negative"] = rank["false_positive_high_expression"].astype(str).str.lower().eq("true")

    spatialself_rank = (
        rank[rank["method"].eq("spatialself_calibrated")]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    baseline = rank[rank["method"].isin(METHOD_ORDER)].copy()
    spp1_rank = baseline[baseline["pair"].eq("SPP1-CD44")][["method", "rank"]].rename(
        columns={"rank": "spp1_rank"}
    )
    best_neg = (
        baseline[baseline["is_negative"]]
        .sort_values(["method", "rank"])
        .groupby("method", as_index=False)
        .first()[["method", "pair", "rank"]]
        .rename(columns={"pair": "best_negative", "rank": "best_negative_rank"})
    )
    method_ranks = pd.merge(spp1_rank, best_neg, on="method", how="outer")
    method_ranks["method"] = pd.Categorical(method_ranks["method"], METHOD_ORDER, ordered=True)
    method_ranks = method_ranks.sort_values("method")

    local_fdr = inputs["local_fdr"].copy()
    local_fdr["pair"] = canonical_pair(local_fdr)
    local_fdr = local_fdr[
        local_fdr["pair"].isin(["SPP1-CD44", "ACTB-KRT8", "COL1A1-EPCAM"])
    ].copy()

    specificity = inputs["downstream_specificity"].copy()
    specificity["method"] = specificity["method"].replace({"spatialself": "spatialself_calibrated"})
    specificity["method"] = pd.Categorical(specificity["method"], METHOD_ORDER, ordered=True)
    specificity = specificity.sort_values("method")

    spatial = build_spatial_table(inputs)

    return {
        "spatialself_rank": spatialself_rank,
        "method_ranks": method_ranks,
        "local_fdr": local_fdr,
        "specificity": specificity,
        "spatial": spatial,
        "dataset_summary": inputs["summary"].iloc[0].to_dict(),
    }


def build_spatial_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cells = inputs["cells"].copy()
    candidates = inputs["candidates"].copy()
    pred = inputs["pred_spatialself"].copy()

    spp1_candidates = candidates[candidates["lr_id"].eq("post_xenium_crc_lr_001_SPP1_CD44")].copy()
    merged = spp1_candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
    receiver = (
        merged.groupby("receiver_id", as_index=False)
        .agg(
            spp1_incoming=("score", "mean"),
            top_edge=("score", "max"),
            n_edges=("score", "size"),
        )
        .rename(columns={"receiver_id": "cell_id"})
    )
    spatial = cells.merge(receiver, on="cell_id", how="left")
    spatial["spp1_incoming"] = pd.to_numeric(spatial["spp1_incoming"], errors="coerce").fillna(0.0)
    spatial["top_edge"] = pd.to_numeric(spatial["top_edge"], errors="coerce").fillna(0.0)
    spatial["is_top_decile"] = spatial["spp1_incoming"] >= spatial["spp1_incoming"].quantile(0.90)
    return spatial


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.045) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def clean(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_spatial(ax: mpl.axes.Axes, spatial: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.02, y=1.025)
    cmap = LinearSegmentedColormap.from_list(
        "spp1_colon_signal", ["#F2F4F6", "#F0D6CB", COLORS["signal"], COLORS["signal_dark"]]
    )
    background = spatial[spatial["spp1_incoming"] <= spatial["spp1_incoming"].quantile(0.02)]
    signal = spatial[spatial["spp1_incoming"] > 0].copy()
    top = signal[signal["is_top_decile"]]

    ax.scatter(background["x"], -background["y"], s=2.2, c="#DDE2E7", linewidths=0, alpha=0.42)
    sc = ax.scatter(
        signal["x"],
        -signal["y"],
        s=3.9,
        c=signal["spp1_incoming"],
        cmap=cmap,
        linewidths=0,
        alpha=0.88,
    )
    ax.scatter(
        top["x"],
        -top["y"],
        s=8.6,
        facecolors="none",
        edgecolors="#7C2E25",
        linewidths=0.35,
        alpha=0.9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("SPP1-CD44 incoming activity", loc="left", fontsize=8.0, pad=4, fontweight="bold")
    ax.text(
        0.01,
        0.015,
        "outline: top-decile receiver cells",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.008)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=5.5, length=1.8, width=0.45)
    cbar.set_label("incoming score", fontsize=5.8, labelpad=2)


def draw_rank(ax: mpl.axes.Axes, rank: pd.DataFrame) -> None:
    panel_label(ax, "b")
    data = rank.sort_values("rank", ascending=False)
    y = np.arange(len(data))
    colors = [
        COLORS["signal"] if row["pair"] == "SPP1-CD44" else COLORS["negative"] if row["is_negative"] else "#B8D1CD"
        for _, row in data.iterrows()
    ]
    ax.barh(y, data["mean_score"], color=colors, height=0.58, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([LR_LABELS.get(p, p) for p in data["pair"]], fontsize=6.4)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("mean SpatialSelf score", fontsize=6.8, labelpad=2)
    ax.set_title("LR ranking", loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    clean(ax)
    for yi, row in zip(y, data.itertuples(index=False)):
        ax.text(float(row.mean_score) + 0.015, yi, str(int(row.rank)), va="center", ha="left", fontsize=5.8, color=COLORS["muted"])
    ax.text(
        0.98,
        1.02,
        "rank 2",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=COLORS["signal_dark"],
    )


def draw_method_separation(ax: mpl.axes.Axes, ranks: pd.DataFrame) -> None:
    panel_label(ax, "c")
    data = ranks.copy().reset_index(drop=True)
    y = np.arange(len(data))
    ax.set_facecolor("white")
    for i, row in data.iterrows():
        method = str(row["method"])
        if method == "spatalk":
            ax.hlines(i, 1, 7, color=COLORS["hairline"], linewidth=1.0)
            ax.text(4.0, i, "constant scores", ha="center", va="center", fontsize=5.8, color=COLORS["muted"])
            continue
        ax.plot(
            [row["spp1_rank"], row["best_negative_rank"]],
            [i, i],
            color=COLORS["hairline"],
            linewidth=1.2,
            zorder=1,
        )
        ax.scatter(row["best_negative_rank"], i, s=26, color=COLORS["negative"], zorder=3)
        ax.scatter(row["spp1_rank"], i, s=34, color=COLORS["signal"], zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(str(m), str(m)) for m in data["method"]], fontsize=6.4)
    ax.set_xlim(7.35, 0.65)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xlabel("LR rank (lower is better)", fontsize=6.8, labelpad=2)
    ax.set_title("Rank separation", loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.invert_yaxis()
    clean(ax)


def draw_local_fdr(ax: mpl.axes.Axes, local_fdr: pd.DataFrame) -> None:
    panel_label(ax, "d")
    null_order = ["within_region_permutation", "cell_type_preserving_permutation", "lr_identity_permutation"]
    spp1 = local_fdr[local_fdr["pair"].eq("SPP1-CD44")].set_index("null_model").loc[null_order]
    neg = (
        local_fdr[local_fdr["pair"].isin(["ACTB-KRT8", "COL1A1-EPCAM"])]
        .groupby("null_model")["local_contrast_bh_fdr"]
        .min()
        .loc[null_order]
    )
    x = np.arange(len(null_order))
    spp1_y = -np.log10(spp1["local_contrast_bh_fdr"].clip(lower=1e-6))
    neg_y = -np.log10(neg.clip(lower=1e-6))
    for xi, y1, y0 in zip(x, spp1_y, neg_y):
        ax.plot([xi, xi], [y0, y1], color=COLORS["hairline"], linewidth=1.3, zorder=1)
    ax.scatter(x, neg_y, s=25, color=COLORS["negative"], zorder=3, label="negative controls")
    ax.scatter(x, spp1_y, s=34, color=COLORS["signal"], zorder=4, label="SPP1-CD44")
    ax.axhline(-np.log10(0.05), color=COLORS["signal_dark"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(2.45, -np.log10(0.05) + 0.045, "FDR 0.05", fontsize=5.5, color=COLORS["signal_dark"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([NULL_LABELS[n] for n in null_order], fontsize=5.9)
    ax.set_ylim(0, 2.55)
    ax.set_ylabel(r"$-\log_{10}$ local FDR", fontsize=6.8)
    ax.set_title("Local-background FDR", loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    clean(ax)
    ax.legend(loc="upper left", fontsize=5.7, handletextpad=0.3, borderaxespad=0.2)


def draw_downstream(ax: mpl.axes.Axes, specificity: pd.DataFrame) -> None:
    panel_label(ax, "e")
    data = specificity.copy().reset_index(drop=True)
    y = np.arange(len(data))
    for i, row in data.iterrows():
        status = str(row["status"])
        method = str(row["method"])
        if status != "tested":
            ax.hlines(i, -0.62, 0.3, color=COLORS["hairline"], linewidth=1.0)
            ax.text(0.0, i, "constant scores", ha="center", va="center", fontsize=5.8, color=COLORS["muted"])
            continue
        delta = float(row["delta_rho"])
        color = COLORS["signal"] if delta > 0 else COLORS["negative"]
        ax.plot([0, delta], [i, i], color=color, linewidth=3.2, solid_capstyle="round")
        ax.scatter(delta, i, s=28, color=color, zorder=3)
        label = f"{delta:+.2f}"
        if method == "spatialself_calibrated":
            label = f"{delta:+.2f}\nFDR {float(row['spp1_fdr']):.3f}"
        ax.text(delta + (0.03 if delta >= 0 else -0.025), i, label, ha="left" if delta >= 0 else "right", va="center", fontsize=5.5, color=COLORS["ink"], linespacing=0.95)
    ax.axvline(0, color=COLORS["axis"], linewidth=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(str(m), str(m)) for m in data["method"]], fontsize=6.4)
    ax.set_xlim(-0.66, 0.52)
    ax.set_xlabel(r"$\rho$(SPP1-CD44) - $\rho$(best negative)", fontsize=6.8, labelpad=2)
    ax.set_title("Target-response specificity", loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.invert_yaxis()
    clean(ax)


def draw_evidence_strip(ax: mpl.axes.Axes, summary: dict[str, object]) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.82,
        "post-Xenium colon subset",
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["ink"],
    )
    entries = [
        ("2,400", "cells/spots"),
        ("335.9k", "edges"),
        ("7", "LR pairs"),
        ("3", "nulls"),
    ]
    x = 0.0
    for value, label in entries:
        ax.text(
            x,
            0.43,
            value,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            ha="left",
            va="baseline",
            color=COLORS["ink"],
        )
        ax.text(
            x,
            0.16,
            label,
            transform=ax.transAxes,
            fontsize=5.7,
            ha="left",
            va="baseline",
            color=COLORS["muted"],
        )
        x += 0.245
    ax.plot([0, 0.93], [0.02, 0.02], color=COLORS["hairline"], linewidth=0.7, transform=ax.transAxes, clip_on=False)


def make_figure(plot_data: dict[str, pd.DataFrame | dict[str, object]], output_base: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.25, 4.95), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 0.96],
        width_ratios=[1.40, 1.40, 1.0, 1.0, 1.0, 1.0],
        hspace=0.60,
        wspace=0.78,
    )

    ax_spatial = fig.add_subplot(gs[:, 0:2])
    ax_rank = fig.add_subplot(gs[0, 2:4])
    ax_sep = fig.add_subplot(gs[0, 4:6])
    ax_fdr = fig.add_subplot(gs[1, 2:4])
    ax_down = fig.add_subplot(gs[1, 4:6])

    draw_spatial(ax_spatial, plot_data["spatial"])
    draw_rank(ax_rank, plot_data["spatialself_rank"])
    draw_method_separation(ax_sep, plot_data["method_ranks"])
    draw_local_fdr(ax_fdr, plot_data["local_fdr"])
    draw_downstream(ax_down, plot_data["specificity"])

    fig.text(
        0.055,
        0.012,
        "Post-Xenium colon subset: 2,400 cells/spots, 335,900 candidate edges, 7 LR pairs and 3 matched nulls. Local FDR is matched-background FDR; target response uses selected receiver-state genes.",
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.095)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=520)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def write_source_data(plot_data: dict[str, pd.DataFrame | dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in plot_data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"fig8_source_{key}.csv", index=False)
        else:
            (output_dir / f"fig8_source_{key}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase9/fig8_post_xenium_spp1_cd44")
    parser.add_argument("--name", default="fig8_post_xenium_spp1_cd44")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    inputs = read_inputs()
    plot_data = build_plot_data(inputs)
    write_source_data(plot_data, output_dir / "source_data")
    make_figure(plot_data, output_dir / args.name)


if __name__ == "__main__":
    main()
