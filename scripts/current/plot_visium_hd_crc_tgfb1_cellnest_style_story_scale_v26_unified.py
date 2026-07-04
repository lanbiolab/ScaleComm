#!/usr/bin/env python
"""CellNEST Fig.2-style CRC communication story for TGFB1-TGFBR2."""

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
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import scalecomm_figure_style as scf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL7_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DEFAULT_DATASET_DIR = GOAL7_DIR / "standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset"
DEFAULT_PHASE15_DIR = PROJECT_ROOT / "outputs/current_visium_hd_crc_disagreement_validation_scale_v21"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/manuscript_experiment_figures_unified_v5/fig2_crc_visium_hd"
ARTICLE_CRC_SOURCE = PROJECT_ROOT / "outputs/article_style_figure_additions/crc_visium_hd_article_style/source_data"
ARTICLE_EXTRA_SOURCE = PROJECT_ROOT / "outputs/article_style_figure_additions/high_priority_remaining/source_data"

LR_ID = "visium_hd_crc_lr_003_TGFB1_TGFBR2"
LR_LABEL = "TGFB1-TGFBR2"
TARGET_NICHE = "tumor_stromal_boundary"
TARGET_NICHE_LABEL = "tumor-stromal boundary"
STORY_CELL_TYPE = "caf_proxy"
STORY_CELL_TYPE_LABEL = "CAF / stromal"

PREDICTIONS = {
    "spatialself": GOAL7_DIR
    / "visium_hd_crc_p2crc_spatialself_scale_v21/spatialself_wavelet_scale/"
    "visium_hd_crc_p2crc_rich_targets_subset__spatialself_wavelet_scale_predictions.csv",
    "commot": GOAL7_DIR / "visium_hd_crc_p2crc_baseline_panel/commot/visium_hd_crc_p2crc_subset__commot_predictions.csv",
    "cellnest": GOAL7_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellnest/visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
}

METHOD_COLORS = {
    "spatialself": scf.METHOD_COLORS["scalecomm"],
    "commot": scf.METHOD_COLORS["commot"],
    "cellnest": scf.METHOD_COLORS["cellnest"],
}

INK = scf.INK
BLACK = "#111111"
TEXT = scf.TEXT
MUTED = scf.MUTED
GRID = scf.GRID
FIG_BG = scf.FIG_BG
PANEL_FACE = scf.PANEL_FACE
TISSUE_BG = scf.TISSUE_BG
SOFT_NEUTRAL = scf.NEUTRAL
WARM_SIGNAL = scf.SOURCE_GOLD
BAR_SIGNAL = "#D7BA7D"
BORDER = scf.BORDER

NICHE_COLORS = scf.CRC_NICHE_COLORS

CELL_CONTEXT_COLOR = scf.TISSUE_BG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--phase15-dir", default=str(DEFAULT_PHASE15_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--abundance-quantile", type=float, default=0.8)
    parser.add_argument("--hotspot-n", type=int, default=200)
    args = parser.parse_args()

    dataset_dir = _resolve(args.dataset_dir)
    phase15_dir = _resolve(args.phase15_dir)
    output_dir = _resolve(args.output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()

    cells = pd.read_csv(dataset_dir / "cells.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    receiver = pd.read_csv(phase15_dir / "method_receiver_incoming_scores.csv")
    target = pd.read_csv(phase15_dir / "receiver_target_signature_activity.csv")
    pred = {method: pd.read_csv(path) for method, path in PREDICTIONS.items()}

    source = build_source_tables(
        cells=cells,
        candidates=candidates,
        lr_pairs=lr_pairs,
        receiver=receiver,
        target=target,
        predictions=pred,
        abundance_quantile=float(args.abundance_quantile),
        hotspot_n=int(args.hotspot_n),
    )
    source.update(load_crc_addition_tables())
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_cellnest_style_figure(source, hotspot_n=int(args.hotspot_n), abundance_quantile=float(args.abundance_quantile))
    out_base = output_dir / "fig2_crc_visium_hd_tgfb1_tgfbr2_v5_unified"
    save_figure(fig, out_base)
    plt.close(fig)
    print(f"Wrote {out_base}.pdf/.svg/.png/.tiff")
    print(f"Wrote source data to {source_dir}")


def build_source_tables(
    *,
    cells: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    receiver: pd.DataFrame,
    target: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    abundance_quantile: float,
    hotspot_n: int,
) -> dict[str, pd.DataFrame]:
    cells_plot = cells.copy()
    cells_plot["x_plot"] = cells_plot["x"].astype(float)
    cells_plot["y_plot"] = -cells_plot["y"].astype(float)
    cells_plot["target_niche"] = cells_plot["niche"].eq(TARGET_NICHE)

    lr_meta = lr_pairs[["lr_id", "control_name", "true_lr", "false_positive_high_expression"]]
    spatial_edges = (
        candidates.drop(columns=[col for col in ["true_lr", "false_positive_high_expression"] if col in candidates.columns])
        .merge(predictions["spatialself"], on="candidate_id", how="left", validate="one_to_one")
        .merge(lr_meta, on="lr_id", how="left")
        .copy()
    )
    stromal_receiver = spatial_edges[spatial_edges["receiver_niche"].eq(TARGET_NICHE)].copy()
    threshold = stromal_receiver["score"].quantile(abundance_quantile)
    top_region = stromal_receiver[stromal_receiver["score"] >= threshold].copy()
    abundance = (
        top_region.groupby(["lr_id", "control_name", "true_lr", "false_positive_high_expression"], as_index=False)
        .size()
        .rename(columns={"size": "top_edge_count"})
        .sort_values("top_edge_count", ascending=True)
    )
    abundance["rank_desc"] = abundance["top_edge_count"].rank(method="min", ascending=False).astype(int)
    abundance["highlight"] = np.where(abundance["lr_id"].eq(LR_ID), "TGFB1-TGFBR2", "other")
    abundance["region"] = TARGET_NICHE
    abundance["abundance_quantile"] = abundance_quantile

    score_density = stromal_receiver[["lr_id", "control_name", "score", "coexpression_distance_score"]].copy()
    score_density["group"] = np.where(score_density["lr_id"].eq(LR_ID), LR_LABEL, "all other LR/control edges")
    prior_summary = (
        stromal_receiver.groupby(["lr_id", "control_name", "true_lr", "false_positive_high_expression"], as_index=False)
        .agg(
            mean_prior=("coexpression_distance_score", "mean"),
            max_prior=("coexpression_distance_score", "max"),
            nonzero_prior_fraction=("coexpression_distance_score", lambda values: float((values > 0).mean())),
        )
        .sort_values("mean_prior", ascending=True)
    )
    prior_summary["highlight"] = np.where(prior_summary["lr_id"].eq(LR_ID), "TGFB1-TGFBR2", "other")

    target_lr = target[target["lr_id"].eq(LR_ID)].copy()
    target_lr["x_plot"] = target_lr["x"].astype(float)
    target_lr["y_plot"] = -target_lr["y"].astype(float)
    top_target_ids = set(
        target_lr.sort_values(["target_activity", "cell_id"], ascending=[False, True])
        .head(hotspot_n)["cell_id"]
        .astype(str)
    )
    target_lr["target_hotspot"] = target_lr["cell_id"].astype(str).isin(top_target_ids)

    spatial_receiver = receiver[(receiver["method"].eq("spatialself")) & (receiver["lr_id"].eq(LR_ID))].copy()
    spatial_receiver = spatial_receiver.merge(
        target_lr[["cell_id", "target_activity", "target_hotspot"]],
        left_on="receiver_id",
        right_on="cell_id",
        how="left",
    )
    spatial_receiver["x_plot"] = spatial_receiver["x"].astype(float)
    spatial_receiver["y_plot"] = -spatial_receiver["y"].astype(float)
    ranker = spatial_receiver.sort_values(["incoming_score", "receiver_id"], ascending=[False, True]).reset_index(drop=True)
    top_sets = {}
    for n in [100, 300, 500]:
        ids = set(ranker.head(n)["receiver_id"].astype(str))
        sub = spatial_receiver.copy()
        sub["threshold_label"] = f"top {n}"
        sub["top_edge_receiver"] = sub["receiver_id"].astype(str).isin(ids)
        top_sets[n] = sub
    threshold_maps = pd.concat(top_sets.values(), ignore_index=True)

    validation_rows = []
    for method in ["spatialself", "commot", "cellnest"]:
        sub = receiver[(receiver["method"].eq(method)) & (receiver["lr_id"].eq(LR_ID))].copy()
        sub = sub.merge(
            target_lr[["cell_id", "target_activity", "target_hotspot"]],
            left_on="receiver_id",
            right_on="cell_id",
            how="left",
        )
        sub["score_percentile"] = sub["incoming_score"].rank(method="average", pct=True)
        hotspot = sub.loc[sub["target_hotspot"].fillna(False), "score_percentile"].dropna()
        other = sub.loc[~sub["target_hotspot"].fillna(False), "score_percentile"].dropna()
        if len(hotspot) and len(other):
            stat = stats.mannwhitneyu(hotspot, other, alternative="greater")
            p_value = float(stat.pvalue)
            auc = float(stat.statistic / (len(hotspot) * len(other)))
        else:
            p_value = np.nan
            auc = np.nan
        sub["mannwhitney_greater_p"] = p_value
        sub["target_hotspot_auc"] = auc
        sub["target_hotspot_n"] = int(len(hotspot))
        sub["background_n"] = int(len(other))
        validation_rows.append(sub)
    response_alignment = pd.concat(validation_rows, ignore_index=True)

    baseline_rows = []
    for method in ["commot", "cellnest"]:
        sub = receiver[(receiver["method"].eq(method)) & (receiver["lr_id"].eq(LR_ID))].copy()
        sub["x_plot"] = sub["x"].astype(float)
        sub["y_plot"] = -sub["y"].astype(float)
        ids = set(sub.sort_values(["incoming_score", "receiver_id"], ascending=[False, True]).head(hotspot_n)["receiver_id"].astype(str))
        sub["top_edge_receiver"] = sub["receiver_id"].astype(str).isin(ids)
        sub = sub.merge(
            target_lr[["cell_id", "target_hotspot"]],
            left_on="receiver_id",
            right_on="cell_id",
            how="left",
        )
        baseline_rows.append(sub)
    baseline_maps = pd.concat(baseline_rows, ignore_index=True)

    commot_arrow_edges = build_method_edge_table(cells, candidates, predictions["commot"], lr_meta)
    cellnest_heat = make_regional_heat_map(baseline_maps[baseline_maps["method"].eq("cellnest")].copy())

    return {
        "panel_a_niche_map": cells_plot,
        "panel_b_region_lr_abundance": abundance,
        "panel_c_spatialself_score_density": score_density,
        "panel_d_expression_distance_prior_summary": prior_summary,
        "panel_e_spatialself_top_edge_maps": threshold_maps,
        "panel_f_tgfb_response_map": target_lr,
        "panel_g_target_hotspot_alignment": response_alignment,
        "panel_hi_baseline_maps": baseline_maps,
        "panel_h_commot_arrow_edges": commot_arrow_edges,
        "panel_i_cellnest_heat_map": cellnest_heat,
    }


def load_crc_addition_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    paths = {
        "panel_j_target_gene_heatmap": ARTICLE_CRC_SOURCE / "target_gene_heatmap.csv",
        "panel_k_sender_receiver_layers": ARTICLE_CRC_SOURCE / "sender_receiver_layers.csv",
        "panel_l_celltype_pair_matrix": ARTICLE_EXTRA_SOURCE / "realdata_crc_tgfb1_tgfbr2_celltype_pairs.csv",
    }
    for key, path in paths.items():
        tables[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def build_method_edge_table(cells: pd.DataFrame, candidates: pd.DataFrame, prediction: pd.DataFrame, lr_meta: pd.DataFrame) -> pd.DataFrame:
    xy = cells.set_index("cell_id")[["x", "y"]]
    edges = (
        candidates[candidates["lr_id"].eq(LR_ID)]
        .merge(prediction, on="candidate_id", how="left", validate="one_to_one")
        .merge(lr_meta, on="lr_id", how="left")
        .copy()
    )
    edges["sender_x"] = edges["sender_id"].map(xy["x"]).astype(float)
    edges["sender_y"] = edges["sender_id"].map(xy["y"]).astype(float)
    edges["receiver_x"] = edges["receiver_id"].map(xy["x"]).astype(float)
    edges["receiver_y"] = edges["receiver_id"].map(xy["y"]).astype(float)
    edges["sender_x_plot"] = edges["sender_x"]
    edges["sender_y_plot"] = -edges["sender_y"]
    edges["receiver_x_plot"] = edges["receiver_x"]
    edges["receiver_y_plot"] = -edges["receiver_y"]
    return edges.dropna(subset=["score", "sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot"])


def make_regional_heat_map(frame: pd.DataFrame, bandwidth_factor: float = 4.0) -> pd.DataFrame:
    heat = frame.copy()
    coords = heat[["x_plot", "y_plot"]].to_numpy(dtype=float)
    score = heat["incoming_score"].fillna(0).to_numpy(dtype=float)
    if len(heat) == 0 or np.nanmax(score) <= 0:
        heat["regional_heat"] = 0.0
        heat["heat_bandwidth"] = np.nan
        return heat
    tree = cKDTree(coords)
    nn_dist, _ = tree.query(coords, k=2)
    median_nn = float(np.nanmedian(nn_dist[:, 1]))
    bandwidth = max(median_nn * bandwidth_factor, 1e-6)
    radius = bandwidth * 3.0
    regional = np.zeros(len(heat), dtype=float)
    for idx, point in enumerate(coords):
        neighbors = tree.query_ball_point(point, r=radius)
        if not neighbors:
            continue
        neighbor_coords = coords[neighbors]
        dist2 = np.sum((neighbor_coords - point) ** 2, axis=1)
        weights = np.exp(-dist2 / (2.0 * bandwidth**2))
        regional[idx] = float(np.dot(score[neighbors], weights))
    max_heat = float(np.nanmax(regional))
    if max_heat > 0:
        regional = regional / max_heat
    heat["regional_heat"] = regional
    heat["heat_bandwidth"] = bandwidth
    return heat


def plot_cellnest_style_figure(source: dict[str, pd.DataFrame], *, hotspot_n: int, abundance_quantile: float) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.25, 8.25), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        3,
        left=0.052,
        right=0.988,
        bottom=0.060,
        top=0.962,
        wspace=0.25,
        hspace=0.42,
        height_ratios=[1.0, 1.03, 0.95, 0.92],
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[1, 0]),
        "e": fig.add_subplot(gs[1, 1]),
        "f": fig.add_subplot(gs[1, 2]),
        "g": fig.add_subplot(gs[2, 0]),
        "h": fig.add_subplot(gs[2, 1]),
        "i": fig.add_subplot(gs[2, 2]),
        "j": fig.add_subplot(gs[3, 0]),
        "k": fig.add_subplot(gs[3, 1]),
        "l": fig.add_subplot(gs[3, 2]),
    }

    plot_niche_map(axes["a"], source["panel_a_niche_map"])
    plot_region_abundance(axes["b"], source["panel_b_region_lr_abundance"], abundance_quantile)
    plot_density(axes["c"], source["panel_c_spatialself_score_density"], "score", "ScaleComm score")
    plot_prior_summary(axes["d"], source["panel_d_expression_distance_prior_summary"])
    plot_top_edge_series(fig, axes["e"], source["panel_e_spatialself_top_edge_maps"])
    plot_target_response_map(axes["f"], source["panel_f_tgfb_response_map"])
    plot_target_hotspot_alignment(axes["g"], source["panel_g_target_hotspot_alignment"], hotspot_n)
    plot_commot_arrow_map(
        axes["h"],
        source["panel_hi_baseline_maps"],
        source["panel_h_commot_arrow_edges"],
        "COMMOT",
    )
    plot_cellnest_heat_map(axes["i"], source["panel_i_cellnest_heat_map"])
    plot_target_gene_heatmap(axes["j"], source.get("panel_j_target_gene_heatmap", pd.DataFrame()))
    plot_sender_receiver_layers(
        axes["k"],
        source["panel_a_niche_map"],
        source.get("panel_k_sender_receiver_layers", pd.DataFrame()),
    )
    plot_celltype_pair_matrix(axes["l"], source.get("panel_l_celltype_pair_matrix", pd.DataFrame()))

    for key, ax in axes.items():
        panel_label(fig, ax, key)
    return fig


def with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mpl.colors.to_rgba(color, alpha)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.040, dy: float = 0.020) -> None:
    scf.panel_label(fig, ax, label, dx=dx, dy=dy)


def set_panel_title(ax: plt.Axes, title: str) -> None:
    scf.short_panel_title(ax, title, pad=2.0)


def plot_niche_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    background = cells[~cells["cell_type"].eq(STORY_CELL_TYPE)]
    story = cells[cells["cell_type"].eq(STORY_CELL_TYPE)]
    ax.scatter(background["x_plot"], background["y_plot"], s=2.0, c=CELL_CONTEXT_COLOR, alpha=0.48, linewidths=0)
    ax.scatter(
        story["x_plot"],
        story["y_plot"],
        s=3.9,
        c=METHOD_COLORS["spatialself"],
        alpha=0.92,
        edgecolors=BLACK,
        linewidths=0.055,
    )
    set_panel_title(ax, "Cell context")
    add_spatial_style(ax, cells)
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=METHOD_COLORS["spatialself"], markersize=3.8, label=STORY_CELL_TYPE_LABEL),
        mpl.lines.Line2D([], [], marker="o", ls="", color=CELL_CONTEXT_COLOR, markersize=3.8, label="other cells"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.075),
        ncol=1,
        fontsize=5.2,
        handletextpad=0.25,
        columnspacing=0.62,
        labelcolor=TEXT,
    )


def plot_region_abundance(ax: plt.Axes, abundance: pd.DataFrame, abundance_quantile: float) -> None:
    table = abundance.sort_values("top_edge_count", ascending=True).copy()
    draw_crc_barh(ax, table, "top_edge_count", "top-edge count")
    set_panel_title(ax, "LR abundance")
    style_bar_axis(ax)


def plot_prior_summary(ax: plt.Axes, prior: pd.DataFrame) -> None:
    table = prior.sort_values("mean_prior", ascending=True).copy()
    draw_crc_barh(ax, table, "mean_prior", "mean prior")
    set_panel_title(ax, "Prior score")
    style_bar_axis(ax)


def draw_crc_barh(ax: plt.Axes, table: pd.DataFrame, value_col: str, xlabel: str) -> None:
    y = np.arange(len(table))
    for yi, row in enumerate(table.itertuples(index=False)):
        lr = getattr(row, "lr_id")
        neg = bool(getattr(row, "false_positive_high_expression"))
        value = float(getattr(row, value_col))
        if lr == LR_ID:
            face = with_alpha(METHOD_COLORS["spatialself"], 0.98)
            edge = with_alpha(BLACK, 0.92)
            lw = 0.42
        elif neg:
            face = with_alpha(SOFT_NEUTRAL, 0.90)
            edge = with_alpha("#8F887F", 0.26)
            lw = 0.28
        else:
            face = with_alpha(BAR_SIGNAL, 0.86)
            edge = with_alpha("#8A6A2E", 0.22)
            lw = 0.28
        ax.barh(yi, value, color=face, height=0.58, edgecolor=edge, linewidth=lw)
    xmax = float(table[value_col].max()) * 1.18 if len(table) else 1.0
    if xmax <= 0:
        xmax = 1.0
    ax.set_xlim(0, xmax)
    labels = table["control_name"].str.replace("-", "\n", regex=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.55, color=TEXT)
    for tick, lr in zip(ax.get_yticklabels(), table["lr_id"]):
        if lr == LR_ID:
            tick.set_color(METHOD_COLORS["spatialself"])
            tick.set_fontweight("bold")
    for yi, (lr, value) in enumerate(zip(table["lr_id"], table[value_col])):
        if lr == LR_ID:
            ax.scatter(
                float(value) + xmax * 0.035,
                yi,
                marker="v",
                s=20,
                facecolors=PANEL_FACE,
                edgecolors=METHOD_COLORS["spatialself"],
                linewidths=0.78,
                zorder=5,
                clip_on=False,
            )
    ax.set_xlabel(xlabel, fontsize=5.45, color=MUTED, labelpad=1.0)
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.tick_params(axis="x", length=2.0, pad=1.0)


def plot_density(ax: plt.Axes, data: pd.DataFrame, column: str, title: str) -> None:
    other = data[data["group"].ne(LR_LABEL)][column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    focus = data[data["group"].eq(LR_LABEL)][column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if column == "coexpression_distance_score":
        other = np.log1p(other)
        focus = np.log1p(focus)
        xlabel = "log1p prior"
    else:
        xlabel = "score"
    hi = np.nanquantile(pd.concat([pd.Series(other), pd.Series(focus)]), 0.995)
    bins = np.linspace(0, max(float(hi), 1e-6), 36)
    ax.hist(other, bins=bins, density=True, color="#D9D5CE", alpha=0.80, edgecolor=with_alpha(BLACK, 0.18), linewidth=0.22, label="other")
    ax.hist(
        focus,
        bins=bins,
        density=True,
        color=METHOD_COLORS["spatialself"],
        alpha=0.70,
        edgecolor=with_alpha(BLACK, 0.32),
        linewidth=0.24,
        label=LR_LABEL,
    )
    set_panel_title(ax, title)
    ax.set_xlabel(xlabel, fontsize=5.9, color=MUTED, labelpad=1.0)
    ax.set_ylabel("density", fontsize=5.9, color=MUTED, labelpad=1.0)
    handles = [
        Patch(facecolor="#D9D5CE", edgecolor=with_alpha(BLACK, 0.25), linewidth=0.35, label="other"),
        Patch(facecolor=with_alpha(METHOD_COLORS["spatialself"], 0.70), edgecolor=with_alpha(BLACK, 0.36), linewidth=0.35, label=LR_LABEL),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=5.35, handlelength=0.9, labelcolor=TEXT)
    style_quant_axis(ax)
    ax.grid(False)


def plot_top_edge_series(fig: plt.Figure, outer_ax: plt.Axes, maps: pd.DataFrame) -> None:
    outer_ax.axis("off")
    set_panel_title(outer_ax, "ScaleComm")
    inset_layout = [
        (0.040, 0.505, 0.430, 0.410),
        (0.530, 0.505, 0.430, 0.410),
        (0.285, 0.045, 0.430, 0.410),
    ]
    for idx, label in enumerate(["top 100", "top 300", "top 500"]):
        ax = outer_ax.inset_axes(inset_layout[idx])
        frame = maps[maps["threshold_label"].eq(label)].copy()
        ax.scatter(frame["x_plot"], frame["y_plot"], s=1.15, c=TISSUE_BG, alpha=0.18, linewidths=0)
        top = frame[frame["top_edge_receiver"]]
        ax.scatter(
            top["x_plot"],
            top["y_plot"],
            s=3.75,
            c=METHOD_COLORS["spatialself"],
            alpha=0.92,
            edgecolors=BLACK,
            linewidths=0.070,
        )
        target_top = frame[frame["target_hotspot"]]
        ax.scatter(target_top["x_plot"], target_top["y_plot"], s=6.6, facecolors="none", edgecolors=BLACK, alpha=0.36, linewidths=0.28)
        add_spatial_style(ax, frame, pad_fraction=0.004)
        ax.text(0.035, 0.965, label, transform=ax.transAxes, ha="left", va="top", fontsize=5.3, color=TEXT)


def plot_target_response_map(ax: plt.Axes, target: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("response", ["#F7F4EC", scf.SOURCE_GOLD_SOFT, METHOD_COLORS["spatialself"]])
    plot_target = target.sort_values(["target_activity", "cell_id"], ascending=[True, True]).copy()
    values = plot_target["target_activity"].astype(float)
    norm = Normalize(vmin=0, vmax=float(np.nanquantile(values, 0.985)))
    ax.scatter(plot_target["x_plot"], plot_target["y_plot"], s=2.65, c=values, cmap=cmap, norm=norm, alpha=0.94, linewidths=0)
    top = target[target["target_hotspot"]]
    ax.scatter(top["x_plot"], top["y_plot"], s=7.4, facecolors="none", edgecolors=BLACK, alpha=0.42, linewidths=0.34)
    set_panel_title(ax, "Target response")
    add_spatial_style(ax, target)
    cax = ax.inset_axes([0.13, -0.112, 0.54, 0.040])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.45)
    cb.set_ticks([0, norm.vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=5.05, length=2.0, width=0.45, colors=BLACK, pad=1.0)


def plot_target_hotspot_alignment(ax: plt.Axes, alignment: pd.DataFrame, hotspot_n: int) -> None:
    method_order = ["spatialself", "commot", "cellnest"]
    method_labels = ["ScaleComm", "COMMOT", "CellNEST"]
    positions = np.arange(len(method_order), dtype=float)

    hotspot = alignment[alignment["target_hotspot"].fillna(False)].copy()
    summary = []
    rng = np.random.default_rng(23)
    for method in method_order:
        vals = hotspot.loc[hotspot["method"].eq(method), "score_percentile"].dropna().to_numpy(dtype=float)
        effect = vals - 0.5
        mean_val = float(np.nanmean(effect)) if len(vals) else np.nan
        if len(vals):
            boots = np.array([np.nanmean(rng.choice(effect, size=len(effect), replace=True)) for _ in range(2000)])
            lo, hi = np.nanquantile(boots, [0.025, 0.975])
        else:
            lo, hi = np.nan, np.nan
        summary.append({"method": method, "mean": mean_val, "lo": float(lo), "hi": float(hi)})
    summary_df = pd.DataFrame(summary).set_index("method")

    ax.axhline(0.0, color=with_alpha(BLACK, 0.54), lw=0.62, ls=(0, (2.0, 2.0)), zorder=0)
    ax.text(2.58, 0.002, "random", ha="right", va="bottom", fontsize=5.25, color=MUTED)

    for pos, method in zip(positions, method_order):
        mean_val = float(summary_df.loc[method, "mean"])
        lo = float(summary_df.loc[method, "lo"])
        hi = float(summary_df.loc[method, "hi"])
        color = METHOD_COLORS[method]
        ax.plot([pos, pos], [0.0, mean_val], color=with_alpha(color, 0.62), lw=4.6, solid_capstyle="round", zorder=1)
        ax.plot([pos, pos], [lo, hi], color=BLACK, lw=0.80, solid_capstyle="round", zorder=3)
        ax.plot([pos - 0.055, pos + 0.055], [lo, lo], color=BLACK, lw=0.72, zorder=3)
        ax.plot([pos - 0.055, pos + 0.055], [hi, hi], color=BLACK, lw=0.72, zorder=3)
        ax.scatter(
            [pos],
            [mean_val],
            s=31.0,
            marker="D",
            facecolors=color,
            edgecolors=BLACK,
            linewidths=0.42,
            zorder=5,
        )

    stats_rows = (
        alignment[["method", "mannwhitney_greater_p", "target_hotspot_auc", "target_hotspot_n"]]
        .drop_duplicates("method")
        .set_index("method")
    )
    for pos, method in zip(positions, method_order):
        p_value = float(stats_rows.loc[method, "mannwhitney_greater_p"])
        label = f"P={p_value:.1e}" if p_value < 0.01 else "n.s."
        color = METHOD_COLORS[method] if method == "spatialself" else MUTED
        y_text = 0.125 if method == "spatialself" else 0.030
        ax.text(pos, y_text, label, ha="center", va="bottom", fontsize=5.45, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels(method_labels, fontsize=5.65, color=TEXT)
    ax.set_ylabel("Alignment above random\n(mean percentile - 0.5)", fontsize=6.05, color=BLACK, labelpad=1.6, fontweight="bold")
    set_panel_title(ax, "Target-hotspot alignment")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.025, 0.145)
    style_reference_box_axis(ax)
    ax.spines["left"].set_linewidth(0.68)
    ax.spines["bottom"].set_linewidth(0.68)
    ax.tick_params(axis="x", length=0, pad=1.6)
    ax.tick_params(axis="y", length=2.6, pad=1.2)
    ax.set_yticks([0.00, 0.05, 0.10])


def plot_baseline_map(ax: plt.Axes, maps: pd.DataFrame, method: str, label: str, edges: pd.DataFrame | None = None) -> None:
    sub = maps[maps["method"].eq(method)].copy()
    ax.scatter(sub["x_plot"], sub["y_plot"], s=1.25, c=TISSUE_BG, alpha=0.20, linewidths=0)
    if edges is not None:
        draw_cellnest_edge_lines(ax, edges)
    top = sub[sub["top_edge_receiver"]]
    ax.scatter(
        top["x_plot"],
        top["y_plot"],
        s=5.6,
        c=METHOD_COLORS[method],
        alpha=0.92,
        edgecolors=BLACK,
        linewidths=0.085,
        zorder=4,
    )
    target_top = sub[sub["target_hotspot"]]
    ax.scatter(target_top["x_plot"], target_top["y_plot"], s=7.8, facecolors="none", edgecolors=BLACK, alpha=0.38, linewidths=0.30, zorder=5)
    set_panel_title(ax, label)
    add_spatial_style(ax, sub)


def plot_cellnest_heat_map(ax: plt.Axes, heat: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("cellnest_heat", ["#F7F4EC", "#B8A7CA", METHOD_COLORS["cellnest"]])
    plot_heat = heat.sort_values(["regional_heat", "receiver_id"], ascending=[True, True]).copy()
    values = plot_heat["regional_heat"].astype(float)
    vmax = float(np.nanquantile(values, 0.985))
    if vmax <= 0:
        vmax = 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    ax.scatter(
        plot_heat["x_plot"],
        plot_heat["y_plot"],
        s=2.55,
        c=values,
        cmap=cmap,
        norm=norm,
        alpha=0.94,
        linewidths=0,
        zorder=1,
    )
    top = heat[heat["target_hotspot"]]
    ax.scatter(top["x_plot"], top["y_plot"], s=7.6, facecolors="none", edgecolors=BLACK, alpha=0.39, linewidths=0.31, zorder=3)
    set_panel_title(ax, "CellNEST")
    add_spatial_style(ax, heat)
    cax = ax.inset_axes([0.13, -0.112, 0.54, 0.040])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.45)
    cb.set_ticks([0, norm.vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=5.05, length=2.0, width=0.45, colors=BLACK, pad=1.0)


def plot_target_gene_heatmap(ax: plt.Axes, heatmap: pd.DataFrame) -> None:
    if heatmap.empty:
        ax.axis("off")
        return
    group_order = ["other", "ScaleComm top", "response hotspot"]
    gene_order = (
        heatmap[heatmap["group"].eq("response hotspot")]
        .sort_values("z", ascending=True)["gene"]
        .drop_duplicates()
        .tolist()
    )
    if not gene_order:
        gene_order = sorted(heatmap["gene"].dropna().unique())
    mat = (
        heatmap.pivot_table(index="gene", columns="group", values="z", aggfunc="mean")
        .reindex(index=gene_order, columns=group_order)
        .fillna(0.0)
    )
    values = mat.to_numpy(dtype=float)
    vmax = max(0.2, float(np.nanmax(np.abs(values))))
    cmap = scf.CMAPS["diverging"]
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(["other", "ScaleComm\ntop", "response\nhotspot"], fontsize=5.2, color=TEXT)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=4.85, color=TEXT)
    ax.tick_params(length=0, pad=1.0)
    set_panel_title(ax, "Target genes")
    ax.set_xticks(np.arange(-0.5, len(group_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(0.48)
    cax = ax.inset_axes([1.035, 0.16, 0.038, 0.66])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=-vmax, vmax=vmax), orientation="vertical")
    cb.outline.set_linewidth(0.42)
    cb.outline.set_edgecolor(BORDER)
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels(["low", "0", "high"])
    cb.ax.tick_params(labelsize=4.7, length=0, colors=MUTED, pad=1.0)


def plot_sender_receiver_layers(ax: plt.Axes, cells: pd.DataFrame, layers: pd.DataFrame) -> None:
    if layers.empty:
        ax.axis("off")
        return
    ax.scatter(cells["x_plot"], cells["y_plot"], s=1.05, c=TISSUE_BG, alpha=0.17, linewidths=0, zorder=1)
    for role, color in [("sender", WARM_SIGNAL), ("receiver", METHOD_COLORS["spatialself"])]:
        sub = layers[layers["role"].eq(role)].copy()
        if sub.empty:
            continue
        sizes = 5.0 + 4.5 * np.sqrt(np.clip(sub["n"].astype(float), 1, None))
        ax.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=sizes,
            c=color,
            alpha=0.88 if role == "receiver" else 0.74,
            edgecolors=BLACK,
            linewidths=0.10,
            zorder=3 if role == "receiver" else 2,
        )
    set_panel_title(ax, "Sender-receiver layers")
    add_spatial_style(ax, cells)
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=WARM_SIGNAL, markeredgecolor=BLACK, markeredgewidth=0.25, markersize=3.8, label="sender"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=METHOD_COLORS["spatialself"], markeredgecolor=BLACK, markeredgewidth=0.25, markersize=3.8, label="receiver"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.075),
        ncol=2,
        fontsize=5.2,
        handletextpad=0.22,
        columnspacing=0.55,
    )


def plot_celltype_pair_matrix(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    if matrix.empty:
        ax.axis("off")
        return
    table = matrix.set_index(matrix.columns[0]).astype(float)
    row_order = table.sum(axis=1).sort_values(ascending=False).index.tolist()
    col_order = table.sum(axis=0).sort_values(ascending=False).index.tolist()
    table = table.loc[row_order, col_order]
    values = table.to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(values)))
    cmap = LinearSegmentedColormap.from_list("crc_pair_matrix", ["#F7F4EC", scf.SOURCE_GOLD_SOFT, METHOD_COLORS["spatialself"]])
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right", rotation_mode="anchor", fontsize=4.85, color=TEXT)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=4.85, color=TEXT)
    ax.tick_params(length=0, pad=1.0)
    ax.set_xlabel("receiver cell type", fontsize=5.45, color=MUTED, labelpad=0.8)
    ax.set_ylabel("sender cell type", fontsize=5.45, color=MUTED, labelpad=0.8)
    set_panel_title(ax, "Cell-type pairs")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(0.48)
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.50)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_cellnest_edge_lines(ax: plt.Axes, edges: pd.DataFrame, max_edges: int = 36) -> None:
    top = edges[edges["score"].astype(float) > 0].sort_values(["score", "candidate_id"], ascending=[False, True]).head(max_edges).copy()
    if top.empty:
        top = edges.sort_values(["score", "candidate_id"], ascending=[False, True]).head(max_edges).copy()
    if top.empty:
        return
    vmax = max(float(top["score"].max()), 1e-9)
    for row in top.itertuples(index=False):
        val = float(row.score) / vmax
        ax.plot(
            [float(row.sender_x_plot), float(row.receiver_x_plot)],
            [float(row.sender_y_plot), float(row.receiver_y_plot)],
            color=BLACK,
            lw=0.60 + 0.48 * val,
            alpha=0.58 + 0.32 * val,
            solid_capstyle="round",
            zorder=6,
        )


def plot_commot_arrow_map(ax: plt.Axes, maps: pd.DataFrame, arrows: pd.DataFrame, label: str, max_arrows: int = 54) -> None:
    sub = maps[maps["method"].eq("commot")].copy()
    ax.scatter(sub["x_plot"], sub["y_plot"], s=1.22, c=TISSUE_BG, alpha=0.19, linewidths=0, zorder=1)
    target_top = sub[sub["target_hotspot"]]
    ax.scatter(
        target_top["x_plot"],
        target_top["y_plot"],
        s=7.4,
        facecolors="none",
        edgecolors=BLACK,
        alpha=0.38,
        linewidths=0.30,
        zorder=4,
    )

    top = arrows[arrows["score"].astype(float) > 0].sort_values(["score", "candidate_id"], ascending=[False, True]).head(max_arrows).copy()
    if not top.empty:
        dx = top["receiver_x_plot"].to_numpy(dtype=float) - top["sender_x_plot"].to_numpy(dtype=float)
        dy = top["receiver_y_plot"].to_numpy(dtype=float) - top["sender_y_plot"].to_numpy(dtype=float)
        length = np.sqrt(dx**2 + dy**2)
        keep = length > 0
        top = top.loc[keep].copy()
        dx = dx[keep]
        dy = dy[keep]
        length = length[keep]
        if len(top):
            x_span = float(sub["x_plot"].max() - sub["x_plot"].min())
            y_span = float(sub["y_plot"].max() - sub["y_plot"].min())
            visual_length = max(x_span, y_span) * 0.052
            ux = dx / length * visual_length
            uy = dy / length * visual_length
            mid_x = (top["sender_x_plot"].to_numpy(dtype=float) + top["receiver_x_plot"].to_numpy(dtype=float)) / 2
            mid_y = (top["sender_y_plot"].to_numpy(dtype=float) + top["receiver_y_plot"].to_numpy(dtype=float)) / 2
            scores = top["score"].to_numpy(dtype=float)
            vmax = max(float(np.nanmax(scores)), 1e-9)
            for x0, y0, u, v, score in zip(mid_x, mid_y, ux, uy, scores):
                val = float(score) / vmax
                start = (float(x0 - 0.50 * u), float(y0 - 0.50 * v))
                end = (float(x0 + 0.50 * u), float(y0 + 0.50 * v))
                patch = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=7.6,
                    linewidth=0.52 + 0.30 * val,
                    color=METHOD_COLORS["commot"],
                    alpha=0.58 + 0.32 * val,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    zorder=3,
                )
                ax.add_patch(patch)
    set_panel_title(ax, label)
    add_spatial_style(ax, sub)


def add_spatial_style(ax: plt.Axes, frame: pd.DataFrame, pad_fraction: float = 0.025) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    pad_x = (frame["x_plot"].max() - frame["x_plot"].min()) * pad_fraction
    pad_y = (frame["y_plot"].max() - frame["y_plot"].min()) * pad_fraction
    ax.set_xlim(frame["x_plot"].min() - pad_x, frame["x_plot"].max() + pad_x)
    ax.set_ylim(frame["y_plot"].min() - pad_y, frame["y_plot"].max() + pad_y)
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_quant_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, lw=0.30)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.62)
    ax.spines["bottom"].set_linewidth(0.62)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=5.55, colors=MUTED, width=0.50, length=2.0, pad=1.2)


def style_bar_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, lw=0.34)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["bottom"].set_linewidth(0.62)
    ax.tick_params(axis="x", labelsize=5.55, colors=MUTED, width=0.50, length=2.0, pad=1.0)
    ax.tick_params(axis="y", labelsize=5.55, colors=TEXT, width=0.0, length=0, pad=1.0)


def style_reference_box_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#111111")
        ax.spines[spine].set_linewidth(0.82)
    ax.tick_params(axis="both", colors="#111111", width=0.72, length=3.0, pad=1.4, labelsize=6.0)


def configure_matplotlib() -> None:
    scf.configure_matplotlib(font_size=6.5)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=900, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=900, bbox_inches="tight")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
