#!/usr/bin/env python
"""Build Fig. 5 candidate: high-resolution tonsil and cross-platform validation.

The figure is deliberately scoped as supportive real-data validation. It does
not claim native ground truth in tonsil; the main defensible signal is cleaner
positive/negative control prioritization with receiver-state support.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL_DIR / "standardized_inputs/spamosaic_tonsil_A1_subset"
RICH_DATASET_DIR = GOAL_DIR / "standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset"
OUT = ROOT / "outputs/current_fig5_cross_platform_tonsil_story"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"

SELECTED_LR = "spamosaic_tonsil_lr_000_CXCL13_CXCR5"
SELECTED_PAIR = "CXCL13-CXCR5"
TOP_N_RECEIVERS = 300

PREDICTIONS = {
    "SpatialSelf": GOAL_DIR
    / "spamosaic_tonsil_A1_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
    "COMMOT": GOAL_DIR / "spamosaic_tonsil_A1_commot/replicate_00__commot_predictions.csv",
    "CellChat": GOAL_DIR / "spamosaic_tonsil_A1_cellchat/replicate_00__cellchat_predictions.csv",
    "CellNEST": GOAL_DIR / "spamosaic_tonsil_A1_cellnest/replicate_00__cellnest_predictions.csv",
}

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "CellNEST", "SpaTalk"]
MAP_METHODS = ["SpatialSelf", "COMMOT", "CellChat", "CellNEST"]

METHOD_COLORS = {
    "SpatialSelf": "#C94F48",
    "COMMOT": "#2F78A8",
    "CellChat": "#D6A24A",
    "CellNEST": "#7E6C9A",
    "SpaTalk": "#8A8177",
}

CELL_TYPE_COLORS = {
    "B_cell_proxy": "#B85450",
    "FDC_proxy": "#D6B06A",
    "T_cell_proxy": "#2F78A8",
    "Tfh_proxy": "#7E6C9A",
    "stromal_proxy": "#9A8F80",
    "myeloid_proxy": "#6F9F9A",
    "epithelial_proxy": "#C8C3B8",
}

NICHE_COLORS = {
    "germinal_center_proxy": "#B85450",
    "T_zone_proxy": "#2F78A8",
    "myeloid_stromal_proxy": "#6F9F9A",
    "epithelial_proxy": "#C8C3B8",
}

NICHE_LABELS = {
    "germinal_center_proxy": "germinal center",
    "T_zone_proxy": "T zone",
    "myeloid_stromal_proxy": "myeloid/stromal",
    "epithelial_proxy": "epithelial",
}

STATUS_COLORS = {
    "complete": "#2F78A8",
    "supportive": "#C94F48",
    "partial": "#D6A24A",
    "pending": "#C8C3B8",
    "not_primary": "#E4DDD2",
}

INK = "#2B2826"
TEXT = "#37322E"
MUTED = "#6E665B"
GRID = "#E8E2D6"
LIGHT_GRID = "#F0EBE2"
FIG_BG = "#FFFFFF"
TISSUE_BG = "#C8C3B8"

ACTIVITY_CMAP = LinearSegmentedColormap.from_list(
    "tonsil_activity_warm",
    ["#F7F4EC", "#E8C987", "#C94F48"],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--rich-dataset-dir", default=str(RICH_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--top-n-receivers", type=int, default=TOP_N_RECEIVERS)
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    figure_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    source = build_source_tables(
        dataset_dir=_resolve(args.dataset_dir),
        rich_dataset_dir=_resolve(args.rich_dataset_dir),
        top_n=int(args.top_n_receivers),
    )
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)

    write_manifest(source, output_dir / "fig5_tonsil_cross_platform_manifest.json")

    setup_style()
    fig = plot_figure(source, top_n=int(args.top_n_receivers))
    stem = figure_dir / "fig5_tonsil_cross_platform_validation"
    export(fig, stem)
    print(f"Wrote {stem}.png/.pdf/.svg/.tiff")
    print(f"Wrote source tables to {source_dir}")


def build_source_tables(dataset_dir: Path, rich_dataset_dir: Path, top_n: int) -> dict[str, pd.DataFrame]:
    cells = pd.read_csv(dataset_dir / "cells.csv")
    cells_plot = cells.copy()
    cells_plot["x_plot"] = cells_plot["x"].astype(float)
    cells_plot["y_plot"] = -cells_plot["y"].astype(float)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    expression = pd.read_csv(rich_dataset_dir / "expression.csv")
    target_signatures = pd.read_csv(rich_dataset_dir / "target_signatures.csv")

    rank_table = pd.read_csv(GOAL_DIR / "spamosaic_tonsil_A1_method_lr_rank_table.csv")
    control_summary = pd.read_csv(GOAL_DIR / "spamosaic_tonsil_A1_method_control_summary.csv")
    replicate_summary = load_replicate_control_summary()

    predictions = load_predictions()
    receiver_maps = build_receiver_maps(
        cells=cells_plot,
        candidates=candidates,
        predictions=predictions,
        selected_lr=SELECTED_LR,
        target_scores=target_score_for_pair(expression, target_signatures, "CXCL13", "CXCR5"),
        top_n=top_n,
    )
    target_lift = build_target_response_lift(
        candidates=candidates,
        lr_pairs=lr_pairs,
        expression=expression,
        target_signatures=target_signatures,
        predictions=predictions,
        top_n=top_n,
    )
    cross_platform = build_cross_platform_status()

    return {
        "panel_a_tonsil_context": cells_plot,
        "panel_b_method_rank_summary": control_summary,
        "panel_c_replicate_control_summary": replicate_summary,
        "panel_d_receiver_maps": receiver_maps,
        "panel_e_target_response_lift": target_lift,
        "panel_f_cross_platform_status": cross_platform,
        "source_lr_rank_table": rank_table,
    }


def load_predictions() -> dict[str, pd.DataFrame]:
    out = {}
    for method, path in PREDICTIONS.items():
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        out[method] = pred[["candidate_id", "score"]].copy()
    return out


def load_replicate_control_summary() -> pd.DataFrame:
    rows = []
    for section in ["A1", "D1"]:
        path = GOAL_DIR / f"spamosaic_tonsil_{section}_method_control_summary.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table["section"] = section
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table["method_label"] = pd.Categorical(table["method_label"], METHOD_ORDER, ordered=True)
    return table.sort_values(["method_label", "section"]).reset_index(drop=True)


def target_score_for_pair(
    expression: pd.DataFrame,
    target_signatures: pd.DataFrame,
    ligand: str,
    receptor: str,
) -> pd.DataFrame:
    row = target_signatures[
        target_signatures["ligand"].astype(str).eq(ligand)
        & target_signatures["receptor"].astype(str).eq(receptor)
    ]
    if row.empty:
        return pd.DataFrame({"cell_id": expression["cell_id"], "target_score": np.nan})
    genes = [g.strip() for g in str(row.iloc[0]["targets"]).split(",") if g.strip()]
    genes = [g for g in genes if g in expression.columns and g != receptor]
    if not genes:
        return pd.DataFrame({"cell_id": expression["cell_id"], "target_score": np.nan})
    mat = expression.set_index("cell_id")[genes].astype(float)
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0).replace(0, np.nan)
    score = z.mean(axis=1).rename("target_score").reset_index()
    score["target_genes_used"] = ",".join(genes)
    score["n_target_genes_used"] = len(genes)
    score["ligand"] = ligand
    score["receptor"] = receptor
    return score


def build_receiver_maps(
    cells: pd.DataFrame,
    candidates: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    selected_lr: str,
    target_scores: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    base = candidates.loc[candidates["lr_id"].eq(selected_lr), ["candidate_id", "receiver_id", "lr_id"]].copy()
    cell_xy = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].copy()
    rows = []
    for method in MAP_METHODS:
        pred = predictions.get(method)
        if pred is None:
            continue
        merged = base.merge(pred, on="candidate_id", how="left")
        incoming = (
            merged.groupby("receiver_id", as_index=False)["score"]
            .mean()
            .rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
        )
        incoming = cell_xy.merge(incoming, on="cell_id", how="left").merge(
            target_scores[["cell_id", "target_score"]], on="cell_id", how="left"
        )
        incoming["method"] = method
        incoming["rank"] = incoming["incoming_score"].rank(method="first", ascending=False)
        incoming["top_receiver"] = incoming["rank"].le(top_n)
        incoming["lr_pair"] = SELECTED_PAIR
        incoming["incoming_norm"] = robust_norm(incoming["incoming_score"])
        rows.append(incoming)
    return pd.concat(rows, ignore_index=True)


def build_target_response_lift(
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    expression: pd.DataFrame,
    target_signatures: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for lr in lr_pairs.itertuples(index=False):
        score = target_score_for_pair(expression, target_signatures, lr.ligand, lr.receptor)
        if "target_score" not in score.columns or score["target_score"].isna().all():
            continue
        edge_base = candidates.loc[candidates["lr_id"].eq(lr.lr_id), ["candidate_id", "receiver_id"]].copy()
        for method in MAP_METHODS:
            pred = predictions.get(method)
            if pred is None:
                continue
            merged = edge_base.merge(pred, on="candidate_id", how="left")
            incoming = (
                merged.groupby("receiver_id", as_index=False)["score"]
                .mean()
                .rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
                .merge(score[["cell_id", "target_score"]], on="cell_id", how="left")
            )
            incoming = incoming.dropna(subset=["incoming_score", "target_score"])
            if incoming.empty:
                continue
            top_ids = set(incoming.sort_values("incoming_score", ascending=False).head(top_n)["cell_id"])
            hot = incoming.loc[incoming["cell_id"].isin(top_ids), "target_score"].dropna()
            bg = incoming.loc[~incoming["cell_id"].isin(top_ids), "target_score"].dropna()
            if len(hot) and len(bg):
                p_value = mannwhitneyu(hot, bg, alternative="greater").pvalue
                lift = float(hot.mean() - bg.mean())
            else:
                p_value = np.nan
                lift = np.nan
            rows.append(
                {
                    "method": method,
                    "lr_id": lr.lr_id,
                    "lr_pair": lr.control_name,
                    "control_class": lr.control_class,
                    "true_lr": bool(lr.true_lr),
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "target_lift_top_vs_background": lift,
                    "hotspot_mean_target_score": float(hot.mean()) if len(hot) else np.nan,
                    "background_mean_target_score": float(bg.mean()) if len(bg) else np.nan,
                    "mannwhitney_greater_p": p_value,
                    "n_hotspot": int(len(hot)),
                    "n_background": int(len(bg)),
                    "top_n": int(top_n),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["lr_group"] = np.where(table["false_positive_high_expression"], "High-expression negative", "Positive/control prior")
    summary = (
        table.groupby(["method", "lr_group"], as_index=False)
        .agg(
            mean_lift=("target_lift_top_vs_background", "mean"),
            median_lift=("target_lift_top_vs_background", "median"),
            n_lr=("lr_pair", "nunique"),
        )
    )
    return table.merge(summary, on=["method", "lr_group"], how="left")


def build_cross_platform_status() -> pd.DataFrame:
    rows = [
        {
            "dataset": "CRC Visium HD",
            "platform": "Visium HD",
            "context": "tumor-stroma",
            "ranking_controls": "complete",
            "localization": "supportive",
            "baseline_panel": "complete",
            "response_layer": "supportive",
        },
        {
            "dataset": "DSS MERFISH",
            "platform": "MERFISH",
            "context": "inflammation time course",
            "ranking_controls": "partial",
            "localization": "supportive",
            "baseline_panel": "partial",
            "response_layer": "partial",
        },
        {
            "dataset": "Tonsil Slide-tags",
            "platform": "SPAMOSAIC / Slide-tags",
            "context": "immune zones",
            "ranking_controls": "complete",
            "localization": "supportive",
            "baseline_panel": "complete",
            "response_layer": "supportive",
        },
        {
            "dataset": "Xenium breast",
            "platform": "Xenium",
            "context": "tumor-stroma",
            "ranking_controls": "pending",
            "localization": "pending",
            "baseline_panel": "pending",
            "response_layer": "pending",
        },
        {
            "dataset": "CosMx NSCLC",
            "platform": "CosMx SMI",
            "context": "tumor-immune",
            "ranking_controls": "pending",
            "localization": "pending",
            "baseline_panel": "pending",
            "response_layer": "pending",
        },
    ]
    return pd.DataFrame(rows)


def robust_norm(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    if arr.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    lo = float(arr.quantile(0.02))
    hi = float(arr.quantile(0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        lo = float(arr.min())
        hi = float(arr.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(arr)), index=values.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "axes.facecolor": FIG_BG,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(source: dict[str, pd.DataFrame], top_n: int) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 6.8), facecolor=FIG_BG)
    gs = fig.add_gridspec(
        3,
        6,
        width_ratios=[1.02, 1.02, 1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.08, 1.02],
        left=0.055,
        right=0.985,
        top=0.97,
        bottom=0.07,
        hspace=0.52,
        wspace=0.58,
    )

    ax_a = fig.add_subplot(gs[0:2, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    maps_grid = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 2:6], wspace=0.04)
    map_axes = [fig.add_subplot(maps_grid[0, i]) for i in range(4)]
    ax_e = fig.add_subplot(gs[2, 0:3])
    ax_f = fig.add_subplot(gs[2, 3:6])

    draw_context_map(ax_a, source["panel_a_tonsil_context"])
    draw_rank_panel(ax_b, source["panel_b_method_rank_summary"])
    draw_replicate_panel(ax_c, source["panel_c_replicate_control_summary"])
    draw_method_maps(map_axes, source["panel_d_receiver_maps"])
    draw_target_lift_panel(ax_e, source["panel_e_target_response_lift"])
    draw_cross_platform_panel(ax_f, source["panel_f_cross_platform_status"])

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], [ax_a, ax_b, ax_c, map_axes[0], ax_e, ax_f]):
        panel_label(fig, ax, label)

    return fig


def draw_context_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    for niche, color in NICHE_COLORS.items():
        sub = cells[cells["niche"].eq(niche)]
        if sub.empty:
            continue
        ax.scatter(sub["x_plot"], sub["y_plot"], s=3.2, color=color, alpha=0.78, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.02, "SPAMOSAIC tonsil A1", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color=INK)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4.0, color=color, label=NICHE_LABELS[label])
        for label, color in NICHE_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.075),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.75,
        borderaxespad=0,
        frameon=False,
    )


def draw_rank_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = summary.copy()
    table["method_label"] = table["method_label"].astype(str)
    table = table[table["method_label"].isin(METHOD_ORDER)]
    table["y"] = table["method_label"].map({m: i for i, m in enumerate(METHOD_ORDER[::-1])})
    ax.axvspan(0.5, 3.5, color="#F6EFE5", zorder=0)
    for _, row in table.iterrows():
        y = row["y"]
        ax.plot(
            [row["best_positive_rank"], row["best_negative_rank"]],
            [y, y],
            color="#D8D0C4",
            lw=0.8,
            zorder=1,
        )
        ax.scatter(
            row["best_positive_rank"],
            y,
            s=28,
            color=METHOD_COLORS.get(row["method_label"], "#C94F48"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            row["best_negative_rank"],
            y,
            s=28,
            facecolor="#F7F4EC",
            edgecolor="#7D756C",
            linewidth=0.85,
            zorder=4,
        )
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER[::-1])
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xlabel("LR rank (1 = highest)")
    ax.set_title("Positive vs high-expression control", pad=4, loc="center")
    clean_axis(ax, xgrid=True)
    ax.text(0.98, 0.19, "positive", transform=ax.transAxes, ha="right", va="center", fontsize=5.8, color="#C94F48")
    ax.text(0.98, 0.07, "negative", transform=ax.transAxes, ha="right", va="center", fontsize=5.8, color="#7D756C")


def draw_replicate_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        ax.set_axis_off()
        return
    table = summary.copy()
    table["method_label"] = table["method_label"].astype(str)
    table = table[table["method_label"].isin(METHOD_ORDER)]
    row_order = METHOD_ORDER[::-1]
    col_defs = [("A1", "positive_top3"), ("A1", "negative_top3"), ("D1", "positive_top3"), ("D1", "negative_top3")]
    for yi, method in enumerate(row_order):
        for xi, (section, metric) in enumerate(col_defs):
            sub = table[(table["method_label"].eq(method)) & (table["section"].eq(section))]
            value = float(sub.iloc[0][metric]) if len(sub) else np.nan
            color = METHOD_COLORS.get(method, "#C94F48") if "positive" in metric else "#7D756C"
            size = 12 + 18 * max(value, 0)
            ax.scatter(xi, yi, s=size, color=color, alpha=0.88 if value > 0 else 0.18, edgecolor="white", linewidth=0.45)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, len(row_order) - 0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["A1\npos", "A1\nneg", "D1\npos", "D1\nneg"])
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("Two-section control audit", pad=4, loc="center")
    clean_axis(ax, xgrid=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for x in [1.5]:
        ax.axvline(x, color=LIGHT_GRID, lw=0.7, zorder=0)


def draw_method_maps(axes: list[plt.Axes], maps: pd.DataFrame) -> None:
    for ax, method in zip(axes, MAP_METHODS):
        sub = maps[maps["method"].eq(method)].copy()
        ax.set_aspect("equal")
        ax.scatter(sub["x_plot"], sub["y_plot"], s=1.5, color=TISSUE_BG, alpha=0.26, linewidths=0, rasterized=True)
        top = sub[sub["top_receiver"]]
        ax.scatter(
            top["x_plot"],
            top["y_plot"],
            s=7.5,
            facecolors=METHOD_COLORS.get(method, "#C94F48"),
            edgecolors="white",
            linewidths=0.22,
            alpha=0.82,
            rasterized=True,
        )
        ax.text(0.50, 1.018, method, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].text(0.0, 1.10, f"{SELECTED_PAIR} top receiver hotspots", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=7.0, color=INK)


def draw_target_lift_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    table = table[table["method"].isin(MAP_METHODS)].copy()
    groups = ["Positive/control prior", "High-expression negative"]
    offsets = {"Positive/control prior": -0.14, "High-expression negative": 0.14}
    colors = {"Positive/control prior": "#C94F48", "High-expression negative": "#8A8177"}
    xbase = {method: i for i, method in enumerate(MAP_METHODS)}
    for group in groups:
        sub = table[table["lr_group"].eq(group)]
        for method in MAP_METHODS:
            vals = sub.loc[sub["method"].eq(method), "target_lift_top_vs_background"].dropna().astype(float)
            if vals.empty:
                continue
            x = xbase[method] + offsets[group]
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=11,
                color=colors[group],
                alpha=0.52,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            ax.plot([x - 0.075, x + 0.075], [vals.mean(), vals.mean()], color=colors[group], lw=1.2, zorder=4)
    ax.axhline(0, color=GRID, lw=0.7, zorder=0)
    ax.set_xticks(range(len(MAP_METHODS)))
    ax.set_xticklabels(MAP_METHODS, rotation=0)
    ax.set_ylabel("target-response lift")
    ax.set_title(f"Hotspot receiver response (top {TOP_N_RECEIVERS})", pad=4, loc="center")
    clean_axis(ax, ygrid=True)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[g], markersize=3.8, label=g.replace("/control prior", " LR"))
        for g in groups
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.35)


def draw_cross_platform_panel(ax: plt.Axes, status: pd.DataFrame) -> None:
    metrics = ["ranking_controls", "localization", "baseline_panel", "response_layer"]
    labels = ["rank", "space", "baseline", "response"]
    y_order = list(status["dataset"])[::-1]
    dataset_to_y = {dataset: i for i, dataset in enumerate(y_order)}
    for _, row in status.iterrows():
        y = dataset_to_y[row["dataset"]]
        for x, metric in enumerate(metrics):
            state = row[metric]
            ax.scatter(
                x,
                y,
                s=52,
                color=STATUS_COLORS.get(state, "#C8C3B8"),
                edgecolor="white",
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylim(-0.6, len(y_order) - 0.4)
    ax.set_title("Cross-platform validation status", pad=4, loc="center")
    clean_axis(ax, xgrid=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, markersize=4.2, label=state)
        for state, color in [("complete", STATUS_COLORS["complete"]), ("supportive", STATUS_COLORS["supportive"]), ("partial", STATUS_COLORS["partial"]), ("pending", STATUS_COLORS["pending"])]
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, columnspacing=0.65, handletextpad=0.25)


def clean_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_linewidth(0.52)
    ax.spines["bottom"].set_linewidth(0.52)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.35)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.35)
    ax.set_axisbelow(True)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.030, box.y1 + 0.016, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_manifest(source: dict[str, pd.DataFrame], path: Path) -> None:
    manifest = {
        "core_conclusion": (
            "In an independent high-resolution tonsil dataset, SpatialSelf prioritizes known immune-zone LR axes "
            "while suppressing high-expression negative controls, and the predicted hotspots show receiver-state support."
        ),
        "claim_boundary": (
            "SPAMOSAIC tonsil labels are ADT marker-rule proxies and not native CCC ground truth. "
            "Target-response lift is supportive, not a strict proof of causality."
        ),
        "selected_lr": SELECTED_PAIR,
        "top_n_receivers": TOP_N_RECEIVERS,
        "source_tables": {name: int(len(table)) for name, table in source.items()},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


if __name__ == "__main__":
    main()
