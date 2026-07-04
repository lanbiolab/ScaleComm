#!/usr/bin/env python
"""Build a ScaleComm tonsil CXCL13-CXCR5 biological story figure.

This figure uses fresh `spatialself_wavelet_scale` reruns, not the older
SpatialSelf tonsil outputs used by earlier draft figures.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs"
RERUN_ROOT = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_rerun"
OUT = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
UI_SCREENSHOT = FIG_DIR / "ScaleComm (2).png"

DATASETS = {
    "A1": DATA_ROOT / "spamosaic_tonsil_A1_rich_targets_subset",
    "D1": DATA_ROOT / "spamosaic_tonsil_D1_rich_targets_subset",
}
PREDICTIONS = {
    "A1": RERUN_ROOT
    / "a1_rich_spatialself_wavelet_scale_e200/spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_predictions.csv",
    "D1": RERUN_ROOT
    / "d1_rich_spatialself_wavelet_scale_e200/spamosaic_tonsil_D1_rich_targets_subset__spatialself_wavelet_scale_predictions.csv",
}
SCALE_TABLES = {
    "A1": RERUN_ROOT
    / "a1_rich_spatialself_wavelet_scale_e200/spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_lr_scale_distribution.csv",
    "D1": RERUN_ROOT
    / "d1_rich_spatialself_wavelet_scale_e200/spamosaic_tonsil_D1_rich_targets_subset__spatialself_wavelet_scale_lr_scale_distribution.csv",
}

SELECTED_PAIR = "CXCL13-CXCR5"
SELECTED_LIGAND = "CXCL13"
SELECTED_RECEPTOR = "CXCR5"
TOP_N = 300

COL = {
    "ink": "#000000",
    "text": "#202020",
    "muted": "#4D4D4D",
    "grid": "#E6E0D8",
    "light": "#F4F1E8",
    "bg": "#FFFFFF",
    "blue": "#66A9E8",
    "blue_light": "#BFD8EC",
    "red": "#B85450",
    "red_soft": "#E7C6C2",
    "red_dark": "#7A2D2A",
    "neutral": "#C8C1B7",
    "neutral_dark": "#827B72",
    "gc": "#B85450",
    "bcell": "#B85450",
    "fdc": "#D9A6A2",
    "tzone": "#66A9E8",
    "tfh": "#BFD8EC",
    "myeloid": "#9BC8C1",
    "stromal": "#B9B0A6",
    "epithelial": "#D6D1C9",
    "hot": "#B85450",
    "hot_dark": "#7A2D2A",
    "target": "#66A9E8",
    "grey": "#D8D2C8",
}

NICHE_COLORS = {
    "germinal_center_proxy": COL["gc"],
    "T_zone_proxy": COL["tzone"],
    "myeloid_stromal_proxy": COL["myeloid"],
    "epithelial_proxy": COL["epithelial"],
}
NICHE_LABELS = {
    "germinal_center_proxy": "GC proxy",
    "T_zone_proxy": "T zone",
    "myeloid_stromal_proxy": "myeloid/stromal",
    "epithelial_proxy": "epithelial",
}
CELL_COLORS = {
    "B_cell_proxy": COL["bcell"],
    "FDC_proxy": COL["fdc"],
    "T_cell_proxy": COL["tzone"],
    "Tfh_proxy": COL["tfh"],
    "myeloid_proxy": COL["myeloid"],
    "stromal_proxy": COL["stromal"],
    "epithelial_proxy": COL["epithelial"],
}
CELL_LABELS = {
    "B_cell_proxy": "B cell",
    "FDC_proxy": "FDC",
    "T_cell_proxy": "T cell",
    "Tfh_proxy": "Tfh",
    "myeloid_proxy": "myeloid",
    "stromal_proxy": "stromal",
    "epithelial_proxy": "epithelial",
}

EXPR_CMAP = LinearSegmentedColormap.from_list("expr_red", ["#F8F5EE", "#E7C6C2", "#B85450"])
TARGET_CMAP = LinearSegmentedColormap.from_list("target_blue", ["#F7F4EC", "#BFD8EC", "#66A9E8"])
HOT_CMAP = LinearSegmentedColormap.from_list("hotspot_red", ["#F8F5EE", "#E7C6C2", "#B85450"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    source = build_source_tables()
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"{name}.csv", index=False)

    write_manifest(source, OUT / "tonsil_cxcl13_cxcr5_scalecomm_manifest.json")
    setup_style()
    fig = plot_figure(source)
    export(fig, FIG_DIR / "tonsil_cxcl13_cxcr5_scalecomm_story")
    ui_fig = plot_ui_screenshot_figure(UI_SCREENSHOT)
    export(ui_fig, FIG_DIR / "tonsil_scalecomm_ui_full_resolution_panel")
    rank_fig = plot_gc_lr_rank(source["source_gc_region_lr_activity_rank"])
    export(rank_fig, FIG_DIR / "tonsil_gc_region_lr_activity_rank")
    print(f"Wrote {FIG_DIR / 'tonsil_cxcl13_cxcr5_scalecomm_story'}.[pdf/svg/png/tiff]")
    print(f"Wrote {FIG_DIR / 'tonsil_scalecomm_ui_full_resolution_panel'}.[pdf/svg/png/tiff]")
    print(f"Wrote {FIG_DIR / 'tonsil_gc_region_lr_activity_rank'}.[pdf/svg/png/tiff]")
    print(f"Wrote source data to {SOURCE_DIR}")


def build_source_tables() -> dict[str, pd.DataFrame]:
    dataset = load_dataset("A1")
    cells = dataset["cells"]
    candidates = dataset["candidates"]
    prediction = load_prediction("A1")
    selected = selected_edges(candidates, prediction)

    incoming = receiver_incoming(cells, selected, TOP_N)
    expression_maps = expression_map_table(dataset, incoming)
    target_map = target_response_map(dataset, incoming)
    composition = top_receiver_composition(cells, incoming)
    rank = lr_rank_table(dataset, prediction)
    replicate = replicate_support_table()
    sender_receiver = sender_receiver_summary(candidates, selected)
    scale = scale_table()
    gc_lr_rank = gc_region_lr_activity_rank()
    gc_top20 = gc_top20_lr_hotspot_counts()

    return {
        "panel_a_tonsil_context": cells,
        "panel_b_ligand_receptor_expression": expression_maps,
        "panel_c_scalecomm_receiver_hotspot": incoming,
        "panel_d_top_receiver_composition": composition,
        "panel_e_target_response_map": target_map,
        "panel_f_replicate_support": replicate,
        "source_lr_rank_table": rank,
        "source_sender_receiver_summary": sender_receiver,
        "source_lr_scale_distribution": scale,
        "source_gc_region_lr_activity_rank": gc_lr_rank,
        "panel_g_gc_top20_lr_hotspot_counts": gc_top20,
    }


def load_dataset(section: str) -> dict[str, pd.DataFrame]:
    dataset_dir = DATASETS[section]
    cells = pd.read_csv(dataset_dir / "cells.csv")
    cells = cells.copy()
    cells["section"] = section
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    expression = pd.read_csv(dataset_dir / "expression.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    targets = pd.read_csv(dataset_dir / "target_signatures.csv")
    return {
        "cells": cells,
        "expression": expression,
        "candidates": candidates,
        "lr_pairs": lr_pairs,
        "targets": targets,
    }


def load_prediction(section: str) -> pd.DataFrame:
    path = PREDICTIONS[section]
    if not path.exists():
        raise FileNotFoundError(f"Missing current ScaleComm prediction for {section}: {path}")
    pred = pd.read_csv(path)
    return pred[["candidate_id", "score"]].copy()


def selected_edges(candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(prediction, on="candidate_id", how="inner")
    table["lr_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
    table = table[table["lr_pair"].eq(SELECTED_PAIR)].copy()
    if table.empty:
        raise ValueError(f"{SELECTED_PAIR} was not found in candidate table.")
    return table


def receiver_incoming(cells: pd.DataFrame, edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    incoming = (
        edges.groupby("receiver_id", as_index=False)["score"]
        .mean()
        .rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
    )
    table = cells.merge(incoming, on="cell_id", how="left")
    table["incoming_score"] = table["incoming_score"].fillna(0.0)
    table["incoming_norm"] = robust_norm(table["incoming_score"])
    table["receiver_rank"] = table["incoming_score"].rank(method="first", ascending=False).astype(int)
    table["top_receiver"] = table["receiver_rank"].le(int(top_n))
    table["lr_pair"] = SELECTED_PAIR
    table["top_n"] = int(top_n)
    return table


def expression_map_table(dataset: dict[str, pd.DataFrame], incoming: pd.DataFrame) -> pd.DataFrame:
    expression = dataset["expression"].set_index("cell_id")
    rows = []
    for gene in [SELECTED_LIGAND, SELECTED_RECEPTOR, "MS4A1", "CR2", "BCL6"]:
        if gene not in expression.columns:
            continue
        values = np.log1p(expression[gene].astype(float))
        frame = incoming[["cell_id", "x_plot", "y_plot", "cell_type", "niche", "top_receiver"]].copy()
        frame["gene"] = gene
        frame["expression"] = values.reindex(frame["cell_id"]).to_numpy(dtype=float)
        frame["expression_norm"] = robust_norm(frame["expression"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def target_response_map(dataset: dict[str, pd.DataFrame], incoming: pd.DataFrame) -> pd.DataFrame:
    target = target_score_for_pair(dataset["expression"], dataset["targets"])
    table = incoming.merge(target, on="cell_id", how="left")
    table["target_score_norm"] = robust_norm(table["target_score"])
    table["target_gene_short"] = table["target_genes_used"].fillna("")
    return table


def target_score_for_pair(expression: pd.DataFrame, target_signatures: pd.DataFrame) -> pd.DataFrame:
    row = target_signatures[
        target_signatures["ligand"].astype(str).eq(SELECTED_LIGAND)
        & target_signatures["receptor"].astype(str).eq(SELECTED_RECEPTOR)
    ]
    if row.empty:
        raise ValueError("Target signature for selected LR pair was not found.")
    genes = [gene.strip() for gene in str(row.iloc[0]["targets"]).split(",") if gene.strip()]
    genes = [gene for gene in genes if gene in expression.columns and gene != SELECTED_RECEPTOR]
    if not genes:
        raise ValueError("No target genes available for selected LR pair.")
    mat = np.log1p(expression.set_index("cell_id")[genes].astype(float))
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0).replace(0, np.nan)
    score = z.mean(axis=1).rename("target_score").reset_index()
    score["target_genes_used"] = ",".join(genes)
    score["n_target_genes_used"] = len(genes)
    return score


def top_receiver_composition(cells: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    top = incoming[incoming["top_receiver"]].copy()
    rows = []
    for variable in ["niche", "cell_type"]:
        bg_counts = cells[variable].value_counts()
        top_counts = top[variable].value_counts()
        for label in sorted(bg_counts.index.astype(str)):
            top_n = int(top_counts.get(label, 0))
            bg_n = int(bg_counts.get(label, 0))
            top_other = int(len(top) - top_n)
            bg_other = int(len(cells) - bg_n)
            odds, p_value = fisher_exact([[top_n, top_other], [bg_n, bg_other]], alternative="greater")
            bg_fraction = bg_n / max(len(cells), 1)
            top_fraction = top_n / max(len(top), 1)
            rows.append(
                {
                    "feature": variable,
                    "label": label,
                    "display_label": display_label(variable, label),
                    "top_count": top_n,
                    "background_count": bg_n,
                    "top_fraction": top_fraction,
                    "background_fraction": bg_fraction,
                    "fold_enrichment": top_fraction / max(bg_fraction, 1e-12),
                    "fisher_greater_p": p_value,
                    "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                    "top_n": int(len(top)),
                }
            )
    return pd.DataFrame(rows)


def display_label(variable: str, label: str) -> str:
    if variable == "niche":
        return NICHE_LABELS.get(label, label.replace("_proxy", ""))
    return CELL_LABELS.get(label, label.replace("_proxy", ""))


def lr_rank_table(dataset: dict[str, pd.DataFrame], prediction: pd.DataFrame) -> pd.DataFrame:
    candidates = dataset["candidates"].merge(dataset["lr_pairs"], on="lr_id", how="left", suffixes=("", "_lr"))
    table = candidates.merge(prediction, on="candidate_id", how="inner")
    table["lr_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
    rows = []
    for lr_pair, group in table.groupby("lr_pair", sort=False):
        rows.append(
            {
                "section": "A1",
                "lr_pair": lr_pair,
                "mean_score": float(group["score"].mean()),
                "max_score": float(group["score"].max()),
                "q95_score": float(group["score"].quantile(0.95)),
                "q99_score": float(group["score"].quantile(0.99)),
                "global_top100_edges": int(group.index.isin(table.sort_values("score", ascending=False).head(100).index).sum()),
                "control_class": first_existing(group, "control_class_lr", "control_class"),
                "true_lr": bool(first_existing(group, "true_lr", default=False)),
                "false_positive_high_expression": bool(
                    first_existing(group, "false_positive_high_expression_lr", "false_positive_high_expression", default=False)
                ),
            }
        )
    rank = pd.DataFrame(rows)
    for metric in ["mean_score", "q95_score", "q99_score", "max_score"]:
        rank[f"{metric}_rank"] = rank[metric].rank(ascending=False, method="min").astype(int)
    return rank.sort_values("mean_score_rank").reset_index(drop=True)


def first_existing(frame: pd.DataFrame, *columns: str, default=None):
    for column in columns:
        if column in frame.columns and len(frame[column]):
            value = frame[column].iloc[0]
            if not pd.isna(value):
                return value
    return default


def sender_receiver_summary(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    top_edges = selected.sort_values("score", ascending=False).head(500).copy()
    summary = (
        top_edges.groupby(["sender_cell_type", "receiver_cell_type"], as_index=False)
        .agg(n_edges=("candidate_id", "size"), mean_score=("score", "mean"), max_score=("score", "max"))
        .sort_values(["n_edges", "mean_score"], ascending=False)
    )
    summary["lr_pair"] = SELECTED_PAIR
    return summary


def replicate_support_table() -> pd.DataFrame:
    rows = []
    for section, dataset_dir in DATASETS.items():
        pred_path = PREDICTIONS[section]
        if not pred_path.exists():
            rows.append({"section": section, "status": "missing_prediction"})
            continue
        dataset = load_dataset(section)
        prediction = load_prediction(section)
        selected = selected_edges(dataset["candidates"], prediction)
        incoming = receiver_incoming(dataset["cells"], selected, TOP_N)
        target = target_response_map(dataset, incoming)
        tested = target.dropna(subset=["incoming_score", "target_score"]).copy()
        hot = tested[tested["top_receiver"]]
        bg = tested[~tested["top_receiver"]]
        if hot.empty or bg.empty:
            rows.append({"section": section, "status": "too_few_cells"})
            continue
        lift = float(hot["target_score"].mean() - bg["target_score"].mean())
        p_value = float(mannwhitneyu(hot["target_score"], bg["target_score"], alternative="greater").pvalue)
        rho = float(spearmanr(tested["incoming_score"], tested["target_score"]).statistic)
        gc_fraction = float(hot["niche"].eq("germinal_center_proxy").mean())
        background_gc = float(tested["niche"].eq("germinal_center_proxy").mean())
        rows.append(
            {
                "section": section,
                "status": "tested",
                "lr_pair": SELECTED_PAIR,
                "top_n": TOP_N,
                "target_lift_top_vs_background": lift,
                "mannwhitney_greater_p": p_value,
                "spearman_incoming_target": rho,
                "top_receiver_gc_fraction": gc_fraction,
                "background_gc_fraction": background_gc,
                "gc_fold_enrichment": gc_fraction / max(background_gc, 1e-12),
                "n_hotspot": int(len(hot)),
                "n_background": int(len(bg)),
                "n_target_genes_used": int(hot["n_target_genes_used"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def gc_region_lr_activity_rank() -> pd.DataFrame:
    rows = []
    for section in ["A1", "D1"]:
        dataset = load_dataset(section)
        prediction = load_prediction(section)
        cells = dataset["cells"][["cell_id", "niche", "cell_type"]].copy()
        candidates = dataset["candidates"].merge(prediction, on="candidate_id", how="inner")
        candidates["lr_pair"] = candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)
        candidates = candidates.merge(
            dataset["lr_pairs"][
                [
                    "lr_id",
                    "control_class",
                    "true_lr",
                    "false_positive_high_expression",
                    "mechanism",
                    "spatial_range",
                    "active_niche",
                ]
            ],
            on="lr_id",
            how="left",
            suffixes=("", "_lr"),
        )
        for lr_pair, group in candidates.groupby("lr_pair", sort=False):
            incoming = (
                group.groupby("receiver_id", as_index=False)["score"]
                .mean()
                .rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
                .merge(cells, on="cell_id", how="left")
            )
            gc = incoming[incoming["niche"].eq("germinal_center_proxy")].copy()
            non_gc = incoming[~incoming["niche"].eq("germinal_center_proxy")].copy()
            if gc.empty:
                continue
            rows.append(
                {
                    "section": section,
                    "lr_pair": lr_pair,
                    "mean_incoming_gc": float(gc["incoming_score"].mean()),
                    "median_incoming_gc": float(gc["incoming_score"].median()),
                    "q95_incoming_gc": float(gc["incoming_score"].quantile(0.95)),
                    "q99_incoming_gc": float(gc["incoming_score"].quantile(0.99)),
                    "max_incoming_gc": float(gc["incoming_score"].max()),
                    "mean_incoming_non_gc": float(non_gc["incoming_score"].mean()) if len(non_gc) else np.nan,
                    "gc_vs_non_gc_mean_lift": float(gc["incoming_score"].mean() - non_gc["incoming_score"].mean())
                    if len(non_gc)
                    else np.nan,
                    "n_gc_receivers": int(len(gc)),
                    "n_non_gc_receivers": int(len(non_gc)),
                    "control_class": first_existing(group, "control_class_lr", "control_class"),
                    "true_lr": bool(first_existing(group, "true_lr", default=False)),
                    "false_positive_high_expression": bool(
                        first_existing(group, "false_positive_high_expression_lr", "false_positive_high_expression", default=False)
                    ),
                    "mechanism": first_existing(group, "mechanism_lr", "mechanism"),
                    "spatial_range": first_existing(group, "spatial_range_lr", "spatial_range"),
                    "active_niche": first_existing(group, "active_niche_lr", "active_niche"),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for metric in ["mean_incoming_gc", "q95_incoming_gc", "q99_incoming_gc", "gc_vs_non_gc_mean_lift"]:
        table[f"{metric}_rank"] = table.groupby("section")[metric].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["section", "mean_incoming_gc_rank", "q95_incoming_gc_rank"]).reset_index(drop=True)


def gc_top20_lr_hotspot_counts() -> pd.DataFrame:
    rows = []
    for section in ["A1", "D1"]:
        dataset = load_dataset(section)
        prediction = load_prediction(section)
        cells = dataset["cells"][["cell_id", "niche", "cell_type"]].copy()
        candidates = dataset["candidates"].merge(prediction, on="candidate_id", how="inner")
        candidates["lr_pair"] = candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)
        for pair, group in candidates.groupby("lr_pair", sort=False):
            incoming = (
                group.groupby("receiver_id", as_index=False)["score"]
                .mean()
                .rename(columns={"receiver_id": "cell_id", "score": "incoming"})
                .merge(cells, on="cell_id", how="left")
            )
            threshold = incoming["incoming"].quantile(0.80)
            top = incoming[incoming["incoming"].ge(threshold)].copy()
            rows.append(
                {
                    "section": section,
                    "lr_pair": pair,
                    "top20_threshold": float(threshold),
                    "top20_total_n": int(len(top)),
                    "top20_gc_n": int(top["niche"].eq("germinal_center_proxy").sum()),
                    "top20_gc_fraction": float(top["niche"].eq("germinal_center_proxy").mean()),
                    "top20_b_cell_n": int(top["cell_type"].eq("B_cell_proxy").sum()),
                    "top20_fdc_n": int(top["cell_type"].eq("FDC_proxy").sum()),
                    "top20_tfh_n": int(top["cell_type"].eq("Tfh_proxy").sum()),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for metric in ["top20_gc_n", "top20_gc_fraction"]:
        table[f"{metric}_rank"] = table.groupby("section")[metric].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["section", "top20_gc_n_rank", "top20_gc_fraction_rank"]).reset_index(drop=True)


def scale_table() -> pd.DataFrame:
    rows = []
    for section, path in SCALE_TABLES.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table["section"] = section
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def robust_norm(values: pd.Series | np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> pd.Series:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    if arr.notna().sum() == 0:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    lo = float(arr.quantile(lo_q))
    hi = float(arr.quantile(hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        lo = float(arr.min())
        hi = float(arr.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": COL["bg"],
            "savefig.facecolor": COL["bg"],
            "axes.facecolor": COL["bg"],
            "text.color": COL["ink"],
            "axes.labelcolor": COL["text"],
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "image.interpolation": "none",
            "image.resample": False,
        }
    )


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(9.60, 10.40), facecolor=COL["bg"])
    gs = fig.add_gridspec(
        3,
        16,
        width_ratios=[1, 1, 1, 0.18, 1, 1, 1, 0.18, 1, 1, 1, 0.18, 1, 1, 1, 0.04],
        height_ratios=[0.80, 0.72, 2.78],
        left=0.046,
        right=0.992,
        top=0.942,
        bottom=0.030,
        hspace=0.24,
        wspace=0.42,
    )

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 4:7])
    ax_c = fig.add_subplot(gs[0, 8:11])
    ax_d = fig.add_subplot(gs[0, 12:15])
    ax_e = fig.add_subplot(gs[1, 0:4])
    ax_f = fig.add_subplot(gs[1, 5:8])
    ax_g = fig.add_subplot(gs[1, 9:15])
    ax_h = fig.add_subplot(gs[2, 0:16])

    draw_context_map(ax_a, source["panel_a_tonsil_context"])
    draw_availability_map(ax_b, source["panel_b_ligand_receptor_expression"])
    draw_hotspot_map(ax_c, source["panel_c_scalecomm_receiver_hotspot"])
    draw_target_map(ax_d, source["panel_e_target_response_map"])
    draw_composition_panel(ax_e, source["panel_d_top_receiver_composition"])
    draw_response_panel(ax_f, source["panel_e_target_response_map"], source["panel_f_replicate_support"])
    draw_gc_top20_rank_panel(ax_g, source["panel_g_gc_top20_lr_hotspot_counts"])
    draw_ui_screenshot_panel(ax_h, UI_SCREENSHOT)

    panel_label(fig, ax_a, "a", xoff=-0.020, yoff=0.020)
    panel_label(fig, ax_b, "b", xoff=-0.018, yoff=0.026)
    panel_label(fig, ax_c, "c", xoff=-0.018, yoff=0.026)
    panel_label(fig, ax_d, "d", xoff=-0.018, yoff=0.026)
    panel_label(fig, ax_e, "e", xoff=-0.020, yoff=0.018)
    panel_label(fig, ax_f, "f", xoff=-0.018, yoff=0.018)
    panel_label(fig, ax_g, "g", xoff=-0.018, yoff=0.018)
    panel_label(fig, ax_h, "h", xoff=-0.018, yoff=0.018)
    return fig


def draw_context_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    for niche, color in NICHE_COLORS.items():
        sub = cells[cells["niche"].eq(niche)]
        alpha = 0.82 if niche == "germinal_center_proxy" else 0.72
        ax.scatter(sub["x_plot"], sub["y_plot"], s=4.4, facecolor=color, edgecolor="#000000", alpha=alpha, linewidths=0.09)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("SPAMOSAIC tonsil context", pad=4)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4.4, markerfacecolor=color, markeredgecolor="#000000", markeredgewidth=0.25, label=NICHE_LABELS[niche])
        for niche, color in NICHE_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.040),
        ncol=2,
        handletextpad=0.30,
        columnspacing=0.70,
        frameon=False,
    )


def draw_availability_map(ax: plt.Axes, expression_maps: pd.DataFrame) -> None:
    ligand = expression_maps[expression_maps["gene"].eq(SELECTED_LIGAND)].copy()
    receptor = expression_maps[expression_maps["gene"].eq(SELECTED_RECEPTOR)][["cell_id", "expression_norm"]].copy()
    receptor = receptor.rename(columns={"expression_norm": "receptor_norm"})
    table = ligand.merge(receptor, on="cell_id", how="left")
    table["availability"] = np.sqrt(
        np.clip(table["expression_norm"].to_numpy(float), 0, 1)
        * np.clip(table["receptor_norm"].fillna(0).to_numpy(float), 0, 1)
    )
    table["availability"] = robust_norm(table["availability"])

    ax.set_aspect("equal")
    ax.scatter(table["x_plot"], table["y_plot"], s=4.8, facecolor="#D8D2C8", alpha=0.28, linewidths=0)
    active = table[table["availability"].ge(0.08)].sort_values("availability")
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=6.8,
        c=active["availability"],
        cmap=EXPR_CMAP,
        norm=Normalize(0, 1),
        alpha=0.66,
        edgecolors="none",
        linewidths=0,
    )
    ax.set_title("CXCL13-CXCR5 availability", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_vertical_colorbar(ax, EXPR_CMAP, "availability", x=1.08, y=0.20, height=0.34)


def draw_hotspot_map(ax: plt.Axes, incoming: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.scatter(incoming["x_plot"], incoming["y_plot"], s=4.8, facecolor="#D8D2C8", alpha=0.28, linewidths=0)
    active = incoming[incoming["incoming_norm"].ge(0.08)].sort_values("incoming_norm")
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=6.8,
        c=active["incoming_norm"],
        cmap=HOT_CMAP,
        norm=Normalize(0, 1),
        alpha=0.66,
        edgecolors="none",
        linewidths=0,
    )
    ax.set_title("ScaleComm receiver hotspot", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_vertical_colorbar(ax, HOT_CMAP, "activity", x=1.08, y=0.20, height=0.34)


def draw_composition_panel(ax: plt.Axes, comp: pd.DataFrame) -> None:
    niche = comp[comp["feature"].eq("niche")].copy()
    niche = niche.sort_values("fold_enrichment", ascending=True)
    colors = [NICHE_COLORS.get(label, COL["grey"]) for label in niche["label"]]
    ax.barh(
        np.arange(len(niche)),
        niche["fold_enrichment"],
        color=colors,
        alpha=0.84,
        height=0.58,
        edgecolor="#000000",
        linewidth=0.35,
    )
    ax.axvline(1, color="#000000", lw=0.72, ls=(0, (2, 2)))
    ax.set_yticks(np.arange(len(niche)))
    ax.set_yticklabels(niche["display_label"])
    ax.set_xlabel("top receiver enrichment")
    ax.set_title("GC enrichment of top receivers", pad=4)
    clean_axis(ax, grid_x=True)


def draw_target_map(ax: plt.Axes, target: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.scatter(target["x_plot"], target["y_plot"], s=4.8, facecolor="#D8D2C8", alpha=0.28, linewidths=0)
    active = target[target["target_score_norm"].ge(0.12)].sort_values("target_score_norm")
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=7.0,
        c=active["target_score_norm"],
        cmap=TARGET_CMAP,
        norm=Normalize(0, 1),
        alpha=0.64,
        edgecolors="none",
        linewidths=0,
    )
    ax.set_title("B-cell/GC response signature", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_vertical_colorbar(ax, TARGET_CMAP, "response", x=1.08, y=0.20, height=0.34)


def draw_response_panel(ax: plt.Axes, target: pd.DataFrame, replicate: pd.DataFrame) -> None:
    tested = target.dropna(subset=["target_score"]).copy()
    groups = [
        tested[~tested["top_receiver"]]["target_score"].to_numpy(float),
        tested[tested["top_receiver"]]["target_score"].to_numpy(float),
    ]
    groups = [vals[np.isfinite(vals)] for vals in groups]
    fill_colors = ["#BFD8EC", "#F5D0D3"]
    point_colors = ["#66A9E8", "#FF1E18"]

    box = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#000000", "linewidth": 0.75},
        whiskerprops={"color": "#000000", "linewidth": 0.70},
        capprops={"color": "#000000", "linewidth": 0.70},
        boxprops={"edgecolor": "#000000", "linewidth": 0.70},
        zorder=3,
    )
    for patch, color in zip(box["boxes"], fill_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    rng = np.random.default_rng(17)
    max_points = [420, 300]
    for i, vals in enumerate(groups):
        if len(vals) == 0:
            continue
        sample = vals
        if len(vals) > max_points[i]:
            sample = rng.choice(vals, size=max_points[i], replace=False)
        x = np.full(len(sample), i) + rng.normal(0, 0.043, size=len(sample))
        ax.scatter(
            x,
            sample,
            s=18.0,
            facecolor=point_colors[i],
            edgecolor="#000000",
            linewidth=0.35,
            alpha=0.92,
            zorder=4,
        )
    a1 = replicate[(replicate["section"].eq("A1")) & (replicate["status"].eq("tested"))]
    if len(a1):
        p = float(a1.iloc[0]["mannwhitney_greater_p"])
        lift = float(a1.iloc[0]["target_lift_top_vs_background"])
        finite = np.concatenate([vals for vals in groups if len(vals)])
        y = 1.70
        ax.plot([0 - 0.16, 1 + 0.16], [y, y], color="#000000", lw=1.05, clip_on=False)
        ax.text(0.5, y + 0.10, f"{p:.1e}", ha="center", va="bottom", fontsize=8.8, color="#000000")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nreceivers", "top\nreceivers"])
    ax.set_ylabel("Response signature", fontweight="bold", fontsize=8.3, color="#000000", labelpad=5.0)
    ax.set_title("")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-1.08, 1.98)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", colors="#000000", labelsize=8.8, length=4.2, width=1.05, pad=2.2)
    ax.set_yticks([-1.0, 0.0, 1.0, 2.0])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=fill_colors[0], edgecolor="#000000", linewidth=0.7, label="Other"),
        plt.Rectangle((0, 0), 1, 1, facecolor=fill_colors[1], edgecolor="#000000", linewidth=0.7, label="Top receiver"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
        handlelength=1.15,
        columnspacing=0.75,
        fontsize=7.4,
    )


def draw_replicate_panel(ax: plt.Axes, replicate: pd.DataFrame) -> None:
    tested = replicate[replicate["status"].eq("tested")].copy()
    metrics = [
        ("target_lift_top_vs_background", "target lift"),
        ("spearman_incoming_target", "incoming-target rho"),
        ("gc_fold_enrichment", "GC enrichment"),
    ]
    ax.set_axis_off()
    ax.text(0.5, 1.04, "A1/D1 support", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4, color=COL["ink"])
    positions = [(0.02, 0.12, 0.28, 0.78), (0.37, 0.12, 0.28, 0.78), (0.72, 0.12, 0.28, 0.78)]
    legend_drawn = False
    for panel_idx, ((metric, label), bounds) in enumerate(zip(metrics, positions)):
        child = ax.inset_axes(bounds)
        values = {}
        for section in ["A1", "D1"]:
            row = tested[tested["section"].eq(section)]
            if row.empty:
                continue
            values[section] = float(row.iloc[0][metric])
        if len(values) == 2:
            child.plot(
                [0, 1],
                [values["A1"], values["D1"]],
                color=COL["neutral_dark"],
                lw=0.90,
                alpha=0.62,
                zorder=1,
            )
        for x, section in enumerate(["A1", "D1"]):
            if section not in values:
                continue
            color = COL["red"] if section == "A1" else COL["blue"]
            child.scatter(
                x,
                values[section],
                s=34,
                color=color,
                edgecolor="#000000",
                linewidth=0.35,
                zorder=3,
                label=section if not legend_drawn else None,
            )
        legend_drawn = True
        child.set_xlim(-0.45, 1.45)
        ymax = max(values.values()) * 1.22 if values else 1.0
        if metric == "spearman_incoming_target":
            child.set_ylim(0, max(0.75, ymax))
            child.set_yticks([0, 0.4, 0.8])
        elif metric == "gc_fold_enrichment":
            child.set_ylim(0, max(2.7, ymax))
            child.set_yticks([0, 1, 2])
        else:
            child.set_ylim(0, max(1.05, ymax))
            child.set_yticks([0, 0.5, 1.0])
        child.set_xticks([0, 1])
        child.set_xticklabels(["A1", "D1"])
        child.set_title(label, pad=2, fontsize=6.25, color=COL["text"])
        child.grid(axis="y", color=COL["grid"], lw=0.42)
        child.spines["top"].set_visible(False)
        child.spines["right"].set_visible(False)
        child.spines["left"].set_color("#000000")
        child.spines["bottom"].set_color("#000000")
        child.spines["bottom"].set_linewidth(0.72)
        child.spines["left"].set_linewidth(0.72)
        child.tick_params(labelsize=5.8, length=2.0, width=0.65, pad=1.2, colors="#000000")
        if panel_idx > 0:
            child.set_ylabel("")
        else:
            child.set_ylabel("value", labelpad=1)


def draw_gc_top20_rank_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    sub = table[table["section"].eq("A1")].copy()
    sub = sub.sort_values("top20_gc_n", ascending=True)
    label_map = {
        "TNFSF13B-TNFRSF13C": "BAFF-TNFRSF13C",
        "CXCL13-CXCR5": "CXCL13-CXCR5",
        "CD40LG-CD40": "CD40LG-CD40",
        "IL21-IL21R": "IL21-IL21R",
        "CCL21-CCR7": "CCL21-CCR7",
        "MS4A1-CD3E": "MS4A1-CD3E",
        "ACTB-EGFR": "ACTB-EGFR",
    }
    colors = [COL["red"] if pair == SELECTED_PAIR else "#D8D2C8" for pair in sub["lr_pair"]]
    ax.barh(
        np.arange(len(sub)),
        sub["top20_gc_n"],
        color=colors,
        edgecolor="#000000",
        linewidth=0.50,
        height=0.62,
        zorder=3,
    )
    selected = sub[sub["lr_pair"].eq(SELECTED_PAIR)]
    if len(selected):
        row = selected.iloc[0]
        y = int(np.where(sub["lr_pair"].to_numpy() == SELECTED_PAIR)[0][0])
        ax.text(
            float(row["top20_gc_n"]) + 8,
            y,
            f"rank {int(row['top20_gc_n_rank'])}",
            ha="left",
            va="center",
            fontsize=6.3,
            color="#000000",
        )
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels([label_map.get(pair, pair) for pair in sub["lr_pair"]], fontsize=5.55)
    ax.set_xlabel("GC cells in LR top 20%")
    ax.set_title("GC-localized hotspot rank", pad=4)
    ax.set_xlim(0, max(500, float(sub["top20_gc_n"].max()) * 1.14))
    ax.grid(axis="x", color=COL["grid"], lw=0.42, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", colors="#000000", labelsize=6.0, length=2.4, width=0.72, pad=1.5)


def draw_ui_screenshot_panel(ax: plt.Axes, image_path: Path) -> None:
    ax.set_axis_off()
    ax.set_title("interactive LR browser", pad=3)
    if not image_path.exists():
        ax.text(0.5, 0.5, "UI screenshot\nnot available", ha="center", va="center", fontsize=7.0, color=COL["muted"])
        return
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = Image.open(image_path).convert("RGBA")
        arr = np.asarray(image)
    except Exception as exc:
        ax.text(0.5, 0.5, f"UI screenshot\nread failed", ha="center", va="center", fontsize=7.0, color=COL["muted"])
        return

    ax.imshow(arr, interpolation="none", resample=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D7D2CA")
        spine.set_linewidth(0.55)


def plot_ui_screenshot_figure(image_path: Path) -> plt.Figure:
    image = Image.open(image_path).convert("RGBA")
    width_px, height_px = image.size
    dpi = 300
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=COL["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(image), interpolation="none", resample=False)
    ax.set_axis_off()
    return fig


def add_vertical_colorbar(ax: plt.Axes, cmap: LinearSegmentedColormap, label: str, *, x: float, y: float, height: float) -> None:
    cax = ax.inset_axes([x, y, 0.040, height])
    sm = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1.4, labelsize=4.7, colors=COL["muted"])
    cb.ax.yaxis.set_ticks_position("right")
    cb.set_label(label, rotation=270, labelpad=8.0, fontsize=4.9, color=COL["muted"])


def plot_gc_lr_rank(table: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(4.4, 2.55), facecolor=COL["bg"])
    gs = fig.add_gridspec(1, 2, left=0.18, right=0.985, top=0.84, bottom=0.20, wspace=0.46)
    for idx, section in enumerate(["A1", "D1"]):
        ax = fig.add_subplot(gs[0, idx])
        sub = table[table["section"].eq(section)].copy()
        sub = sub.sort_values("mean_incoming_gc", ascending=True)
        colors = [COL["red"] if pair == SELECTED_PAIR else "#D8D2C8" for pair in sub["lr_pair"]]
        ax.barh(np.arange(len(sub)), sub["mean_incoming_gc"], color=colors, edgecolor="#000000", linewidth=0.45, height=0.58)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["lr_pair"], fontsize=6.0)
        ax.set_xlabel("mean incoming in GC")
        ax.set_title(section, pad=4, fontsize=7.5)
        clean_axis(ax, grid_x=True)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(axis="both", colors="#000000", length=2.4, width=0.65, pad=1.5)
    fig.text(0.18, 0.95, "GC-region LR activity ranking", ha="left", va="top", fontsize=8.6, color="#000000")
    return fig


def clean_axis(ax: plt.Axes, *, grid_x: bool = False, grid_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["muted"])
    ax.spines["bottom"].set_color(COL["muted"])
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(length=2.0, width=0.45, pad=1.5)
    if grid_x:
        ax.grid(axis="x", color=COL["grid"], lw=0.38)
    if grid_y:
        ax.grid(axis="y", color=COL["grid"], lw=0.38)
    ax.set_axisbelow(True)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, *, xoff: float = -0.026, yoff: float = 0.014) -> None:
    box = ax.get_position()
    fig.text(box.x0 + xoff, box.y1 + yoff, label, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=COL["ink"])


def export(fig: plt.Figure, stem: Path, dpi: int = 1200) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_manifest(source: dict[str, pd.DataFrame], path: Path) -> None:
    replicate = source["panel_f_replicate_support"]
    manifest = {
        "core_conclusion": (
            "Fresh ScaleComm reruns localize a CXCL13-CXCR5 receiver program to germinal-center/FDC-B-cell "
            "regions in SPAMOSAIC tonsil, with receiver target-response support."
        ),
        "claim_boundary": (
            "SPAMOSAIC tonsil has real spatial coordinates and RNA/ADT measurements, but no native CCC ground truth. "
            "This figure supports spatial localization and receiver-state consistency, not causal LR proof. "
            "The current rerun does not support claiming global CXCL13-CXCR5 rank 1 or broad negative-control suppression."
        ),
        "method_label": "ScaleComm",
        "internal_model_name": "spatialself_wavelet_scale",
        "selected_lr": SELECTED_PAIR,
        "top_n_receivers": TOP_N,
        "rerun_prediction_paths": {section: str(path.relative_to(ROOT)) for section, path in PREDICTIONS.items() if path.exists()},
        "replicate_support": replicate.to_dict(orient="records"),
        "source_tables": {name: int(len(table)) for name, table in source.items()},
    }
    path.write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
