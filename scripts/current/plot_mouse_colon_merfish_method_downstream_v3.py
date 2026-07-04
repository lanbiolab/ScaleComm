#!/usr/bin/env python
"""DSS MERFISH method comparison plus receiver-response validation figure."""

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

from scripts.current.plot_mouse_colon_merfish_scipilot_story_v2 import (
    BASE,
    METHOD_COLORS as OLD_METHOD_COLORS,
    METHODS,
    STAGES,
    STAGE_LABELS,
    aggregate_receiver_scores,
    broad_compartment,
    load_expression_slice,
    load_slice,
    normalize_extent,
    prediction_path,
    receiver_scores,
    set_spatial_limits,
)
from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import CURATED_SIGNATURES


OUT = BASE / "figures/merfish_dss_method_downstream_v3"
PANEL_OUT = OUT / "panels"

LR_CASE = "Cxcl10-Cxcr3"
SIGNATURE = "Cxcr3_ifng_response"
INFLAMMATORY_AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2"]
SPATIAL_STAGES = ["Healthy_D0", "DSS9", "DSS21"]
SPATIAL_LABELS = ["D0", "D9", "D21"]
PROGRAM_ORDER = ["homeostasis", "early response", "inflammatory peak", "repair/remodeling"]
PROGRAM_LABELS = ["homeostasis", "early\nresponse", "inflammatory\npeak", "repair /\nremodeling"]
PROGRAM_TABLE = BASE / "progression_story/scipilot_story_v2_program_activity.csv"
THREE_METHOD_TABLE = BASE / "three_method_priority_story/three_method_priority_trajectory_long.csv"

INK = "#232323"
MUTED = "#6A665E"
GRID = "#EAE5DB"
BACKGROUND = "#D7D1C8"

METHOD_COLORS = {
    "SpatialSelf": "#9E332E",
    "COMMOT": "#2F7782",
    "CellNEST": "#6D6375",
}

COMPARTMENT_COLORS = {
    "Epithelial": "#D5963D",
    "Immune": "#B94E67",
    "Mesenchymal": "#4F9A8B",
    "Vascular": "#5D7FA7",
    "Neural": "#8B7BA4",
    "Other": "#BDB7AE",
}

ACTIVITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tnf_activity",
    ["#FCF7EF", "#E8BE83", "#C56246", "#6C1E1E"],
)
RESPONSE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "nfkb_response",
    ["#F7F3EC", "#B9D1C9", "#559A91", "#164F58"],
)
GENE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "hotspot_gene_expression",
    ["#F8F1E7", "#EBC783", "#C86A4A", "#612623"],
)
PROGRAM_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "program_activity",
    ["#F7F2EA", "#E5C68F", "#CA7849", "#7C2D2A"],
)

RESPONSE_GENE_PANEL = ["Vsir", "Irgm1", "Areg", "Egr1", "Itgb2", "Nfkbie"]
RESPONSE_GENE_NOTES = {
    "Vsir": "immunoregulatory checkpoint",
    "Irgm1": "interferon-linked response",
    "Areg": "epithelial repair response",
    "Egr1": "stress/inflammatory immediate response",
    "Itgb2": "leukocyte adhesion/migration",
    "Nfkbie": "NF-kB inflammatory regulation",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.9,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.58,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.027, dy: float = 0.010) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)


def panel_title(fig: plt.Figure, ax: plt.Axes, title: str, dy: float = 0.033) -> None:
    box = ax.get_position()
    fig.text(box.x0, box.y1 + dy, title, ha="left", va="bottom", fontsize=7.8, color=INK)


def draw_context(ax: plt.Axes, cells: pd.DataFrame, size: float = 1.35, alpha: float = 0.22) -> None:
    ax.scatter(cells["x"], cells["y"], s=size, color=BACKGROUND, alpha=alpha, linewidths=0)


def add_low_high_bar(ax: plt.Axes, cmap, label: str) -> None:
    cax = ax.inset_axes([0.58, -0.112, 0.34, 0.055])
    for i in range(96):
        cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=cmap(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.8, labelsize=5.6)
    cax.set_xlabel(label, fontsize=5.7, labelpad=0.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)


def draw_matrix_rects(ax: plt.Axes, matrix: np.ndarray, cmap, vmin: float = 0.0, vmax: float = 1.0) -> None:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    values = np.asarray(matrix, dtype=float)
    nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(values[i, j])),
                    edgecolor="white",
                    linewidth=0.72,
                )
            )
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)


def incoming_score_table(stage: str, lr_pair: str, method: str, replicate: int = 1) -> pd.DataFrame:
    cells = receiver_scores(stage, lr_pair, method=method, replicate=replicate)
    cells["incoming_score"] = cells["incoming_score"].fillna(0.0)
    return cells


def edge_table(stage: str, lr_pair: str, method: str, replicate: int = 1) -> pd.DataFrame:
    rep, cells, candidates = load_slice(stage, replicate)
    ligand, receptor = lr_pair.split("-")
    predictions = pd.read_csv(prediction_path(stage, rep, method))
    edges = candidates.loc[(candidates["ligand"] == ligand) & (candidates["receptor"] == receptor)].copy()
    edges = edges.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    xy = cells.set_index("cell_id")[["x", "y"]]
    edges = edges.join(xy.rename(columns={"x": "sender_x", "y": "sender_y"}), on="sender_id")
    edges = edges.join(xy.rename(columns={"x": "receiver_x", "y": "receiver_y"}), on="receiver_id")
    return edges.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y"])


def signature_cells(stage: str, replicate: int = 1) -> pd.DataFrame:
    cells = load_expression_slice(stage, replicate).copy()
    genes = [gene for gene in CURATED_SIGNATURES[SIGNATURE] if gene in cells.columns]
    cells["signature_score"] = np.log1p(cells[genes].to_numpy(float)).mean(axis=1)
    return cells


def draw_score_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    vmax: float,
    *,
    cmap=ACTIVITY_CMAP,
    low_clip: float = 0.055,
    title: str | None = None,
) -> None:
    draw_context(ax, cells, size=1.25, alpha=0.25)
    score = cells["incoming_score"].to_numpy(float)
    norm = np.clip(score / max(vmax, 1e-9), 0, 1)
    mask = norm > low_clip
    if mask.any():
        sizes = 3.2 + 15.5 * np.power(norm[mask], 0.62)
        ax.scatter(
            cells.loc[mask, "x"],
            cells.loc[mask, "y"],
            c=norm[mask],
            s=sizes,
            cmap=cmap,
            vmin=0,
            vmax=1,
            marker="o",
            alpha=0.94,
            edgecolors="white",
            linewidths=0.11,
            zorder=3,
        )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def draw_signature_map(ax: plt.Axes, cells: pd.DataFrame, vmax: float, title: str | None = None) -> None:
    score = cells["signature_score"].to_numpy(float)
    norm = np.clip(score / max(vmax, 1e-9), 0, 1)
    ax.scatter(cells["x"], cells["y"], s=1.15, color=BACKGROUND, alpha=0.20, linewidths=0)
    mask = norm > 0.020
    ax.scatter(
        cells.loc[mask, "x"],
        cells.loc[mask, "y"],
        c=norm[mask],
        s=2.8 + 6.5 * np.power(norm[mask], 0.75),
        cmap=RESPONSE_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.88,
        marker="o",
        edgecolors="none",
        zorder=3,
    )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def spatialself_hotspot_ids(stage: str = "DSS9", lr_pair: str = LR_CASE, replicate: int = 1) -> list[str]:
    receiver_score, _ = _receiver_scores_for_method(stage, lr_pair, "SpatialSelf", replicate)
    ranked = receiver_score.dropna().sort_values(ascending=False)
    if ranked.empty:
        return []
    k = max(5, int(np.floor(0.20 * len(ranked))))
    return ranked.head(k).index.tolist()


def selected_response_gene_table(summary: pd.DataFrame, expr_cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in RESPONSE_GENE_PANEL:
        if gene not in expr_cells.columns:
            continue
        rec = summary.loc[summary["gene"] == gene].head(1)
        if rec.empty:
            continue
        row = rec.iloc[0].to_dict()
        row["note"] = RESPONSE_GENE_NOTES.get(gene, "")
        row["panel_order"] = len(rows) + 1
        rows.append(row)
    selected = pd.DataFrame(rows)
    if not selected.empty:
        selected.to_csv(OUT / "cxcl10_cxcr3_selected_spatial_response_genes.csv", index=False)
    return selected


def draw_gene_expression_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    gene: str,
    hotspot_ids: list[str],
    *,
    title: str | None = None,
) -> None:
    values = np.log1p(cells[gene].to_numpy(float))
    positive = values > 0
    vmax = float(np.quantile(values[positive], 0.985)) if positive.any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = np.clip(values / vmax, 0, 1)

    ax.scatter(cells["x"], cells["y"], s=1.05, color="#D8D3CB", alpha=0.20, linewidths=0, zorder=1)
    if positive.any():
        ax.scatter(
            cells.loc[positive, "x"],
            cells.loc[positive, "y"],
            c=norm[positive],
            s=2.4 + 8.8 * np.power(norm[positive], 0.70),
            cmap=GENE_CMAP,
            vmin=0,
            vmax=1,
            alpha=0.92,
            marker="o",
            edgecolors="none",
            zorder=2,
        )

    hotspot = cells.loc[cells["cell_id"].isin(hotspot_ids)]
    if not hotspot.empty:
        ax.scatter(
            hotspot["x"],
            hotspot["y"],
            s=25,
            facecolors="none",
            edgecolors="#255C59",
            linewidths=0.68,
            alpha=0.92,
            zorder=4,
        )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.1, pad=1.7)


def draw_commot_arrows(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame, title: str | None = None) -> None:
    draw_context(ax, cells, size=1.20, alpha=0.25)
    if not edges.empty:
        selected = edges.sort_values("score", ascending=False).head(24).copy()
        vmax = max(float(selected["score"].max()), 1e-9)
        for row in selected.itertuples(index=False):
            val = float(row.score) / vmax
            color = METHOD_COLORS["COMMOT"]
            ax.annotate(
                "",
                xy=(row.receiver_x, row.receiver_y),
                xytext=(row.sender_x, row.sender_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 0.35 + 0.75 * val,
                    "color": color,
                    "alpha": 0.18 + 0.58 * val,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                    "mutation_scale": 4.6 + 2.5 * val,
                },
                zorder=3,
            )
        ax.scatter(
            selected["receiver_x"],
            selected["receiver_y"],
            s=5.0,
            color=METHOD_COLORS["COMMOT"],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.10,
            zorder=4,
        )
    set_spatial_limits(ax, cells)
    if title:
        ax.set_title(title, fontsize=7.4, pad=1.9)


def _receiver_scores_for_method(stage: str, lr_pair: str, method: str, replicate: int) -> tuple[pd.Series, pd.DataFrame]:
    ligand, receptor = lr_pair.split("-")
    rep, _, candidates = load_slice(stage, replicate)
    predictions = pd.read_csv(prediction_path(stage, rep, method))
    group = candidates.loc[(candidates["ligand"] == ligand) & (candidates["receptor"] == receptor)]
    group = group.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    return group.groupby("receiver_id")["score"].mean(), group


def gene_level_response_screen(stage: str = "DSS9", lr_pair: str = LR_CASE, top_n: int = 20) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    blocked_prefixes = ("Blank",)
    ligand, receptor = lr_pair.split("-")
    for replicate in [1, 2]:
        expr_cells = load_expression_slice(stage, replicate).set_index("cell_id")
        receiver_score, _ = _receiver_scores_for_method(stage, lr_pair, "SpatialSelf", replicate)
        common = expr_cells.index.intersection(receiver_score.index)
        if len(common) < 10:
            continue
        ranked = receiver_score.loc[common].sort_values(ascending=False)
        k = max(5, int(np.floor(0.20 * len(ranked))))
        top_cells = ranked.head(k).index
        bottom_cells = ranked.tail(k).index
        metadata_columns = {"x", "y", "library_size", "stage_day"}
        genes = [
            gene
            for gene in expr_cells.columns
            if pd.api.types.is_numeric_dtype(expr_cells[gene])
            and gene not in metadata_columns
            and not gene.startswith(blocked_prefixes)
            and gene not in {ligand, receptor}
        ]
        top = np.log1p(expr_cells.loc[top_cells, genes].to_numpy(float))
        bottom = np.log1p(expr_cells.loc[bottom_cells, genes].to_numpy(float))
        top_mean = top.mean(axis=0)
        bottom_mean = bottom.mean(axis=0)
        top_detect = (expr_cells.loc[top_cells, genes].to_numpy(float) > 0).mean(axis=0)
        bottom_detect = (expr_cells.loc[bottom_cells, genes].to_numpy(float) > 0).mean(axis=0)
        for gene, tm, bm, td, bd in zip(genes, top_mean, bottom_mean, top_detect, bottom_detect):
            rows.append(
                {
                    "stage": stage,
                    "replicate": replicate,
                    "lr_pair": lr_pair,
                    "gene": gene,
                    "n_receiver_cells": int(len(ranked)),
                    "k": int(k),
                    "top_mean_log1p": float(tm),
                    "bottom_mean_log1p": float(bm),
                    "effect_log1p": float(tm - bm),
                    "top_detection": float(td),
                    "bottom_detection": float(bd),
                    "detection_delta": float(td - bd),
                }
            )
    screen = pd.DataFrame(rows)
    if screen.empty:
        return screen, []
    summary = (
        screen.groupby(["lr_pair", "gene"], as_index=False)
        .agg(
            mean_effect_log1p=("effect_log1p", "mean"),
            min_effect_log1p=("effect_log1p", "min"),
            mean_detection_delta=("detection_delta", "mean"),
            n_positive_replicates=("effect_log1p", lambda x: int((x > 0).sum())),
            n_replicates=("replicate", "nunique"),
        )
        .sort_values(["n_positive_replicates", "mean_effect_log1p"], ascending=False)
    )
    selected = summary.loc[
        (summary["n_positive_replicates"] >= 1) & (summary["mean_effect_log1p"] > 0),
        "gene",
    ].head(top_n).tolist()
    screen.to_csv(OUT / "cxcl10_cxcr3_gene_level_response_screen_by_replicate.csv", index=False)
    summary.to_csv(OUT / "cxcl10_cxcr3_gene_level_response_screen_summary.csv", index=False)
    return summary, selected


def module_effects(stage: str = "DSS9", lr_pair: str = LR_CASE) -> tuple[pd.DataFrame, list[str]]:
    _, module_genes = gene_level_response_screen(stage=stage, lr_pair=lr_pair, top_n=20)
    if not module_genes:
        module_genes = [gene for gene in CURATED_SIGNATURES[SIGNATURE]]
    rows = []
    for replicate in [1, 2]:
        expr_cells = load_expression_slice(stage, replicate).set_index("cell_id")
        present = [gene for gene in module_genes if gene in expr_cells.columns]
        if not present:
            continue
        module_score = pd.Series(np.log1p(expr_cells[present].to_numpy(float)).mean(axis=1), index=expr_cells.index)
        for method in METHODS:
            receiver_score, _ = _receiver_scores_for_method(stage, lr_pair, method, replicate)
            common = module_score.index.intersection(receiver_score.index)
            if len(common) < 10:
                continue
            ranked = receiver_score.loc[common].sort_values(ascending=False)
            k = max(5, int(np.floor(0.20 * len(ranked))))
            top_cells = ranked.head(k).index
            bottom_cells = ranked.tail(k).index
            rows.append(
                {
                    "stage": stage,
                    "replicate": replicate,
                    "method": method,
                    "lr_pair": lr_pair,
                    "module": "SpatialSelf-associated top response genes",
                    "module_genes": ";".join(present),
                    "n_module_genes": len(present),
                    "n_receiver_cells": int(len(common)),
                    "k": int(k),
                    "top_module_mean": float(module_score.loc[top_cells].mean()),
                    "bottom_module_mean": float(module_score.loc[bottom_cells].mean()),
                    "module_effect": float(module_score.loc[top_cells].mean() - module_score.loc[bottom_cells].mean()),
                    "top_score_mean": float(ranked.head(k).mean()),
                    "bottom_score_mean": float(ranked.tail(k).mean()),
                }
            )
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / "cxcl10_cxcr3_top_response_module_effects.csv", index=False)
    return effects, module_genes


def plot_panel_a(fig: plt.Figure, gs, label: str = "a") -> list[plt.Axes]:
    axes = []
    for i, (stage, stage_label) in enumerate(zip(SPATIAL_STAGES, SPATIAL_LABELS)):
        ax = fig.add_subplot(gs[0, i])
        _, cells, _ = load_slice(stage)
        cells = cells.copy()
        cells["compartment"] = cells["cell_type"].map(broad_compartment)
        for comp, color in COMPARTMENT_COLORS.items():
            sub = cells.loc[cells["compartment"] == comp]
            if not sub.empty:
                ax.scatter(sub["x"], sub["y"], s=3.9, color=color, alpha=0.88, linewidths=0)
        set_spatial_limits(ax, cells)
        ax.set_title(stage_label, fontsize=7.5, pad=1.8)
        axes.append(ax)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "DSS MERFISH tissue context")
    handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=3.5, color=COMPARTMENT_COLORS[k], label=k)
        for k in ["Epithelial", "Immune", "Mesenchymal", "Vascular", "Neural"]
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0].get_position().x0 + 0.190, axes[0].get_position().y1 + 0.061),
        ncol=5,
        frameon=False,
        handletextpad=0.22,
        columnspacing=0.72,
        borderaxespad=0,
        fontsize=5.9,
    )
    return axes


def plot_panel_b(fig: plt.Figure, ax: plt.Axes, label: str = "b") -> plt.Axes:
    activity = pd.read_csv(PROGRAM_TABLE)
    matrix = (
        activity.pivot(index="program", columns="stage_label", values="relative_activity")
        .loc[PROGRAM_ORDER, STAGE_LABELS]
        .to_numpy(float)
    )
    draw_matrix_rects(ax, matrix, PROGRAM_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_yticks(range(len(PROGRAM_LABELS)))
    ax.set_yticklabels(PROGRAM_LABELS)
    ax.tick_params(length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(fig, ax, label, dx=-0.034)
    panel_title(fig, ax, "Spatial communication programs")
    cax = ax.inset_axes([0.58, -0.155, 0.34, 0.055])
    for i in range(96):
        cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=PROGRAM_CMAP(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.8, labelsize=5.5)
    cax.set_xlabel("relative activity", fontsize=5.6, labelpad=0.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)
    return ax


def plot_panel_c_trajectories(fig: plt.Figure, gs, label: str = "c") -> list[plt.Axes]:
    df = pd.read_csv(THREE_METHOD_TABLE)
    axes = []
    for i, lr_pair in enumerate(INFLAMMATORY_AXES):
        ax = fig.add_subplot(gs[0, i])
        endpoint_y = {}
        for method in METHODS:
            rec = df.loc[(df["method"] == method) & (df["lr_pair"] == lr_pair)]
            if rec.empty:
                continue
            values = rec[STAGES].iloc[0].to_numpy(float)
            ax.plot(
                np.arange(len(STAGE_LABELS)),
                values,
                color=METHOD_COLORS[method],
                lw=1.20,
                marker="o",
                ms=2.6,
                label=method,
            )
            endpoint_y[method] = float(values[-1])
        ax.set_title(lr_pair, fontsize=7.0, pad=2.0)
        ax.set_xticks(np.arange(len(STAGE_LABELS)))
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.15, 3.12)
        ax.grid(axis="y", color=GRID, lw=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_ylabel("stage score")
        else:
            ax.set_yticklabels([])
        axes.append(ax)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker="o", lw=1.2, markersize=2.8, label=method)
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0].get_position().x0 + 0.470, axes[0].get_position().y1 + 0.064),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0,
        fontsize=5.9,
    )
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "Inflammatory-axis trajectories by method")
    return axes


def plot_panel_d_predictions(fig: plt.Figure, gs, label: str = "d") -> list[plt.Axes]:
    axes = []
    method_cells = {method: incoming_score_table("DSS9", LR_CASE, method) for method in METHODS}
    vals = np.concatenate([df["incoming_score"].to_numpy(float) for df in method_cells.values()])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    vmax = max(float(np.quantile(vals, 0.980)) if len(vals) else 0.0, 1e-6)
    for i, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, i])
        if method == "COMMOT":
            _, cells, _ = load_slice("DSS9")
            draw_commot_arrows(ax, cells, edge_table("DSS9", LR_CASE, method), title=method)
        else:
            draw_score_map(ax, method_cells[method], vmax, low_clip=0.040, title=method)
        axes.append(ax)
    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], f"DSS9 {LR_CASE} prediction patterns by method", dy=0.048)
    return axes


def plot_panel_e_downstream(fig: plt.Figure, gs, label: str = "e") -> list[plt.Axes]:
    summary, _ = gene_level_response_screen()
    expr_cells = load_expression_slice("DSS9", replicate=1).copy()
    selected = selected_response_gene_table(summary, expr_cells)
    genes = selected["gene"].tolist() if not selected.empty else [gene for gene in RESPONSE_GENE_PANEL if gene in expr_cells.columns]
    hotspot_ids = spatialself_hotspot_ids("DSS9", LR_CASE, replicate=1)

    axes = []
    for i, gene in enumerate(genes[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        draw_gene_expression_map(ax, expr_cells, gene, hotspot_ids, title=gene)
        axes.append(ax)

    panel_label(fig, axes[0], label)
    panel_title(fig, axes[0], "DSS9 hotspot-associated response genes", dy=0.050)

    handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=3.4, color=GENE_CMAP(0.82), label="gene expression"),
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            markersize=4.2,
            markerfacecolor="none",
            markeredgewidth=0.7,
            markeredgecolor="#255C59",
            color="#255C59",
            label="SpatialSelf hotspot",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0].get_position().x0 + 0.300, axes[0].get_position().y1 + 0.052),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.72,
        borderaxespad=0,
        fontsize=5.8,
    )
    cax = axes[-1].inset_axes([0.52, -0.182, 0.38, 0.070])
    for i in range(96):
        cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=GENE_CMAP(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.8, labelsize=5.4)
    cax.set_xlabel("log1p expression", fontsize=5.5, labelpad=0.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)
    return axes


def render_single_panels() -> None:
    fig = plt.figure(figsize=(6.10, 2.02))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.070, top=0.835, wspace=0.040)
    plot_panel_a(fig, gs)
    export(fig, PANEL_OUT / "panel_a_tissue_context")

    fig, ax = plt.subplots(figsize=(3.05, 2.10))
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.22, top=0.80)
    plot_panel_b(fig, ax)
    export(fig, PANEL_OUT / "panel_b_program_heatmap")

    fig = plt.figure(figsize=(5.95, 2.05))
    gs = fig.add_gridspec(1, 3, left=0.095, right=0.985, bottom=0.190, top=0.745, wspace=0.340)
    plot_panel_c_trajectories(fig, gs)
    export(fig, PANEL_OUT / "panel_c_method_trajectories")

    fig = plt.figure(figsize=(6.10, 2.22))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, bottom=0.075, top=0.815, wspace=0.050)
    plot_panel_d_predictions(fig, gs)
    export(fig, PANEL_OUT / "panel_d_method_prediction_patterns")

    fig = plt.figure(figsize=(6.10, 3.48))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, bottom=0.105, top=0.845, wspace=0.040, hspace=0.185)
    plot_panel_e_downstream(fig, gs)
    export(fig, PANEL_OUT / "panel_e_downstream_validation")


def render_composite() -> None:
    fig = plt.figure(figsize=(7.20, 9.05))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.18, 1.28, 1.42, 2.20],
        left=0.055,
        right=0.968,
        bottom=0.065,
        top=0.925,
        hspace=0.640,
    )
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[4.40, 2.00], wspace=0.36)
    plot_panel_a(fig, top[0, 0].subgridspec(1, 3, wspace=0.038))
    ax_b = fig.add_subplot(top[0, 1])
    plot_panel_b(fig, ax_b)
    plot_panel_c_trajectories(fig, outer[1, 0].subgridspec(1, 3, wspace=0.340))
    plot_panel_d_predictions(fig, outer[2, 0].subgridspec(1, 3, wspace=0.052))
    plot_panel_e_downstream(fig, outer[3, 0].subgridspec(2, 3, wspace=0.050, hspace=0.190))
    export(fig, OUT / "mouse_colon_merfish_method_downstream_v3")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    render_single_panels()
    render_composite()
    svg = OUT / "mouse_colon_merfish_method_downstream_v3.svg"
    if svg.exists():
        text = svg.read_text(errors="ignore")
        print(f"{OUT} image_tags={text.count('<image')}")
    else:
        print(OUT)


if __name__ == "__main__":
    main()
