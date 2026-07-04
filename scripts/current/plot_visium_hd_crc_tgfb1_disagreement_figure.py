#!/usr/bin/env python
"""Plot Phase 15 Visium HD CRC TGFB1-TGFBR2 disagreement validation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE15_DIR = PROJECT_ROOT / "outputs/current_visium_hd_crc_disagreement_validation"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    "standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset"
)
LR_ID = "visium_hd_crc_lr_003_TGFB1_TGFBR2"
LR_LABEL = "TGFB1-TGFBR2"


METHOD_COLORS = {
    "spatialself": "#B85450",
    "commot": "#2F78A8",
    "cellnest": "#7E6C9A",
}

NICHE_COLORS = {
    "tumor_core": "#D6D0C7",
    "tumor_stromal_boundary": "#B88455",
    "tumor_immune_boundary": "#6F9F9A",
    "tumor_myeloid_boundary": "#C66C5C",
    "vascular_stromal_boundary": "#6D88A8",
    "epithelial_region": "#9B8FB6",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase15-dir", default=str(DEFAULT_PHASE15_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_PHASE15_DIR / "figures"))
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    phase15_dir = _resolve(args.phase15_dir)
    dataset_dir = _resolve(args.dataset_dir)
    output_dir = _resolve(args.output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    receiver = pd.read_csv(phase15_dir / "method_receiver_incoming_scores.csv")
    target = pd.read_csv(phase15_dir / "receiver_target_signature_activity.csv")
    corr = pd.read_csv(phase15_dir / "downstream_correlation_by_method.csv")
    overlap = pd.read_csv(phase15_dir / "hotspot_overlap_with_target_signature.csv")
    rank = pd.read_csv(phase15_dir / "method_lr_rank_table.csv")
    scan = pd.read_csv(phase15_dir / "candidate_axis_scan.csv")

    source_tables = build_source_tables(cells, receiver, target, corr, overlap, rank, scan, int(args.top_n))
    for name, table in source_tables.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_figure(source_tables, top_n=int(args.top_n))
    out_base = output_dir / "phase15_visium_hd_crc_tgfb1_tgfbr2_disagreement"
    save_figure(fig, out_base)
    plt.close(fig)
    print(f"Wrote figure to {out_base}.pdf/.svg/.png/.tiff")
    print(f"Wrote source data to {source_dir}")


def build_source_tables(
    cells: pd.DataFrame,
    receiver: pd.DataFrame,
    target: pd.DataFrame,
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
    rank: pd.DataFrame,
    scan: pd.DataFrame,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    cells = cells.copy()
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)

    target_lr = target[target["lr_id"].eq(LR_ID)].copy()
    target_lr["x_plot"] = target_lr["x"].astype(float)
    target_lr["y_plot"] = -target_lr["y"].astype(float)
    top_target_ids = set(
        target_lr.sort_values(["target_activity", "cell_id"], ascending=[False, True])
        .head(top_n)["cell_id"]
        .astype(str)
    )
    target_lr["target_top_hotspot"] = target_lr["cell_id"].astype(str).isin(top_target_ids)

    map_rows = []
    for method in ["spatialself", "commot", "cellnest"]:
        sub = receiver[(receiver["method"].eq(method)) & (receiver["lr_id"].eq(LR_ID))].copy()
        sub["x_plot"] = sub["x"].astype(float)
        sub["y_plot"] = -sub["y"].astype(float)
        top_ids = set(
            sub.sort_values(["incoming_score", "receiver_id"], ascending=[False, True])
            .head(top_n)["receiver_id"]
            .astype(str)
        )
        sub["method_top_hotspot"] = sub["receiver_id"].astype(str).isin(top_ids)
        sub = sub.merge(
            target_lr[["cell_id", "target_activity", "target_top_hotspot"]],
            left_on="receiver_id",
            right_on="cell_id",
            how="left",
        )
        score = sub["incoming_score"].astype(float)
        lo, hi = np.nanquantile(score, [0.02, 0.995])
        if hi <= lo:
            sub["score_scaled"] = 0.0
        else:
            sub["score_scaled"] = np.clip((score - lo) / (hi - lo), 0, 1)
        map_rows.append(sub)
    method_maps = pd.concat(map_rows, ignore_index=True)

    evidence = (
        corr[corr["lr_id"].eq(LR_ID)][["method", "method_label", "spearman_r", "bh_fdr"]]
        .merge(
            overlap[overlap["lr_id"].eq(LR_ID)][
                [
                    "method",
                    "hotspot_target_jaccard",
                    "pred_hotspot_active_niche_fold",
                    "pred_hotspot_target_activity_fold",
                    "top_pred_niche",
                ]
            ],
            on="method",
            how="left",
        )
        .merge(
            rank[rank["lr_id"].eq(LR_ID)][["method", "rank_by_mean_score", "mean_score", "top100_edges"]],
            on="method",
            how="left",
        )
    )
    negative = (
        corr[corr["lr_id"].astype(str).str.contains("negative")][["method", "lr_id", "spearman_r"]]
        .groupby("method", as_index=False)["spearman_r"]
        .max()
        .rename(columns={"spearman_r": "max_negative_spearman"})
    )
    evidence = evidence.merge(negative, on="method", how="left")
    method_order = pd.Categorical(evidence["method"], categories=["spatialself", "commot", "cellnest"], ordered=True)
    evidence = evidence.assign(method_order=method_order).sort_values("method_order").drop(columns="method_order")

    scan_lr = scan[scan["lr_id"].eq(LR_ID)].copy()
    return {
        "panel_a_niche_map": cells,
        "panel_bcd_method_maps": method_maps,
        "panel_e_target_signature": target_lr,
        "panel_f_evidence_summary": evidence,
        "candidate_scan_tgfb1_tgfbr2": scan_lr,
    }


def plot_figure(source: dict[str, pd.DataFrame], *, top_n: int) -> plt.Figure:
    configure_matplotlib()
    cells = source["panel_a_niche_map"]
    method_maps = source["panel_bcd_method_maps"]
    target = source["panel_e_target_signature"]
    evidence = source["panel_f_evidence_summary"]

    fig = plt.figure(figsize=(7.1, 4.95), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.05, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.95],
        left=0.045,
        right=0.99,
        bottom=0.085,
        top=0.96,
        wspace=0.11,
        hspace=0.20,
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[0, 3]),
        "e": fig.add_subplot(gs[1, 0]),
        "f": fig.add_subplot(gs[1, 1:]),
    }

    plot_niche_map(axes["a"], cells)
    plot_method_map(axes["b"], method_maps, "spatialself", "SpatialSelf")
    plot_method_map(axes["c"], method_maps, "commot", "COMMOT")
    plot_method_map(axes["d"], method_maps, "cellnest", "CellNEST")
    plot_target_map(axes["e"], target)
    plot_evidence_summary(axes["f"], evidence)

    labels = {
        "a": "a",
        "b": "b",
        "c": "c",
        "d": "d",
        "e": "e",
        "f": "f",
    }
    for key, ax in axes.items():
        ax.text(
            -0.095,
            1.055,
            labels[key],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#252525",
        )
    fig.text(
        0.045,
        0.012,
        f"{LR_LABEL}; filled points are method top {top_n} receiver bins, black outlines are top {top_n} target-response bins. Target signature excludes ligand and receptor genes.",
        fontsize=6.5,
        color="#55504A",
        ha="left",
        va="bottom",
    )
    return fig


def plot_niche_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    for niche, color in NICHE_COLORS.items():
        sub = cells[cells["niche"].eq(niche)]
        ax.scatter(sub["x_plot"], sub["y_plot"], s=2.8, c=color, alpha=0.88, linewidths=0)
    ax.set_title("CRC niches", loc="left", fontsize=8.0, pad=3.5, color="#2B2826")
    add_spatial_style(ax, cells)
    labels = [
        ("tumor", "tumor_core"),
        ("stromal bd.", "tumor_stromal_boundary"),
        ("immune bd.", "tumor_immune_boundary"),
        ("myeloid bd.", "tumor_myeloid_boundary"),
        ("vascular", "vascular_stromal_boundary"),
        ("epithelial", "epithelial_region"),
    ]
    handles = [
        mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor=NICHE_COLORS[n], markeredgewidth=0, markersize=4, label=lab)
        for lab, n in labels
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.00, -0.02),
        frameon=False,
        fontsize=5.7,
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0,
    )


def plot_method_map(ax: plt.Axes, method_maps: pd.DataFrame, method: str, label: str) -> None:
    sub = method_maps[method_maps["method"].eq(method)].copy()
    color = METHOD_COLORS[method]
    ax.scatter(sub["x_plot"], sub["y_plot"], s=2.0, c="#D8D6D0", alpha=0.34, linewidths=0)
    top = sub[sub["method_top_hotspot"]].copy()
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=9.0,
        c=color,
        alpha=0.92,
        linewidths=0.16,
        edgecolors="white",
    )
    target_top = sub[sub["target_top_hotspot"]]
    ax.scatter(
        target_top["x_plot"],
        target_top["y_plot"],
        s=13.0,
        facecolors="none",
        edgecolors="#2A2928",
        alpha=0.38,
        linewidths=0.35,
    )
    ax.set_title(label, loc="left", fontsize=8.0, pad=3.5, color="#2B2826")
    add_spatial_style(ax, sub)


def plot_target_map(ax: plt.Axes, target: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("tgfb_target", ["#ECE8DE", "#D6B06A", "#B85450"])
    values = target["target_activity"].astype(float)
    vmax = np.nanquantile(values, 0.985)
    norm = Normalize(vmin=0.0, vmax=vmax)
    ax.scatter(
        target["x_plot"],
        target["y_plot"],
        s=3.2,
        c=values,
        cmap=cmap,
        norm=norm,
        alpha=0.92,
        linewidths=0,
    )
    top = target[target["target_top_hotspot"]]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=9.0,
        facecolors="none",
        edgecolors="#2A2928",
        alpha=0.42,
        linewidths=0.35,
    )
    ax.set_title("TGF-beta/ECM response", loc="left", fontsize=8.0, pad=3.5, color="#2B2826")
    add_spatial_style(ax, target)
    cax = ax.inset_axes([0.06, -0.075, 0.52, 0.025])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4, colors="#5A554F", pad=1)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["low", "high"])


def plot_evidence_summary(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    method_order = ["spatialself", "commot", "cellnest"]
    labels = {"spatialself": "SpatialSelf", "commot": "COMMOT", "cellnest": "CellNEST"}
    metrics = [
        ("rank_by_mean_score", "LR rank", "lower"),
        ("spearman_r", "downstream\nSpearman", "higher"),
        ("hotspot_target_jaccard", "hotspot-target\nJaccard", "higher"),
        ("pred_hotspot_active_niche_fold", "stromal boundary\nfold", "higher"),
        ("max_negative_spearman", "negative-control\nSpearman", "lower"),
    ]
    table = evidence.set_index("method")
    x = np.arange(len(metrics))
    offsets = np.array([-0.24, 0.0, 0.24])
    width = 0.20
    for i, method in enumerate(method_order):
        row = table.loc[method]
        heights = []
        for col, _, direction in metrics:
            value = float(row[col])
            heights.append(_metric_height(value, col, direction, evidence))
        ax.bar(
            x + offsets[i],
            heights,
            width=width,
            color=METHOD_COLORS[method],
            alpha=0.92,
            linewidth=0,
            label=labels[method],
        )
        for j, (col, _, _) in enumerate(metrics):
            value = float(row[col])
            ax.text(
                x[j] + offsets[i],
                heights[j] + 0.035,
                _metric_label(value, col),
                ha="center",
                va="bottom",
                fontsize=5.4,
                color="#3B3733",
                rotation=0,
            )
    ax.axhline(0, color="#BEB8AE", lw=0.45)
    ax.set_ylim(0, 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=6.4, color="#37322E")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "best"], fontsize=6.0, color="#5A554F")
    ax.set_ylabel("relative support", fontsize=6.7, color="#37322E")
    ax.set_title("Matched evidence for TGFB1-TGFBR2", loc="left", fontsize=8.0, pad=3.5, color="#2B2826")
    ax.grid(axis="y", color="#E5E0D7", lw=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BEB8AE")
    ax.spines["bottom"].set_color("#BEB8AE")
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2, width=0.4, colors="#5A554F")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 1.00),
        ncol=3,
        fontsize=6.1,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.8,
    )


def _metric_height(value: float, col: str, direction: str, evidence: pd.DataFrame) -> float:
    vals = pd.to_numeric(evidence[col], errors="coerce").to_numpy(dtype=float)
    if col == "rank_by_mean_score":
        max_rank = np.nanmax(vals)
        if not np.isfinite(max_rank) or max_rank <= 1:
            return 0.0
        return float((max_rank - value) / (max_rank - 1))
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return 0.5
    scaled = (value - vmin) / (vmax - vmin)
    if direction == "lower":
        scaled = 1.0 - scaled
    return float(np.clip(scaled, 0.03, 1.0))


def _metric_label(value: float, col: str) -> str:
    if col == "rank_by_mean_score":
        return f"#{int(value)}"
    if col == "pred_hotspot_active_niche_fold":
        return f"{value:.1f}x"
    return f"{value:.2f}"


def add_spatial_style(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    pad_x = (frame["x_plot"].max() - frame["x_plot"].min()) * 0.025
    pad_y = (frame["y_plot"].max() - frame["y_plot"].min()) * 0.025
    ax.set_xlim(frame["x_plot"].min() - pad_x, frame["x_plot"].max() + pad_x)
    ax.set_ylim(frame["y_plot"].min() - pad_y, frame["y_plot"].max() + pad_y)
    for spine in ax.spines.values():
        spine.set_visible(False)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
