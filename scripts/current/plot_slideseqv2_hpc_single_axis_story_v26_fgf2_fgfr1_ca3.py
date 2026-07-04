#!/usr/bin/env python
"""CellNEST Fig.2-aligned Slide-seqV2 hippocampus regional story for Fgf2-Fgfr1 in CA3 neurons."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
PREV_SOURCE = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v19_cytosignal_full_interactions/source_data"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_fgf2_fgfr1_ca3_v26_rank_panel"

LIGAND = "Fgf2"
RECEPTOR = "Fgfr1"
LR_PAIR = f"{LIGAND}-{RECEPTOR}"
RESPONSE_GENES = ["Fgfr1", "Tbr1", "Slc17a7", "Fgf2"]
REGION_CELL_TYPE = "CA3_neuron"
REGION_LABEL = "CA3 neuron"

INK = "#222222"
TEXT = "#3B3B3B"
MUTED = "#6B7178"
GRID = "#ECE7DE"
TISSUE = "#DDE4EF"
TISSUE_DARK = "#C8D1DE"
BLUE = "#223D73"
BLUE_SOFT = "#AFC5DD"
GREEN = "#63B35D"
GREEN_DARK = "#3E8B44"
AMBER = "#E6C06A"
ORANGE = "#D9903D"
ROSE = "#8F496F"
BG = "#FFFFFF"
BLACK = "#111111"

CELL_COLORS = {
    "astrocyte": "#93CC82",
    "oligodendrocyte": "#B9C1D1",
    "CA1_neuron": "#B9E5FA",
    "CA3_neuron": "#8EBBD8",
    "dentate_neuron": "#F3D78A",
    "interneuron": "#FBC9C4",
    "endothelial": "#9CC8BA",
    "microglia_macrophage": "#C9B3D7",
    "polydendrocyte": "#B9C1D1",
    "entorhinal_neuron": "#C9CDD6",
}
DISPLAY_TYPES = [
    "astrocyte",
    "dentate_neuron",
    "CA1_neuron",
    "CA3_neuron",
    "interneuron",
    "oligodendrocyte",
    "microglia_macrophage",
    "endothelial",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    for stale in source_dir.glob("*.csv"):
        stale.unlink()

    configure()
    source = build_source()
    for name, table in source.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(source_dir / f"{name}.csv", index=False)

    fig = plot_figure(source)
    out = OUT_DIR / "slideseqv2_hpc_fgf2_fgfr1_ca3_single_axis_story"
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "facecolor": BG}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 900
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
            "font.size": 6.1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#706860",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def build_source() -> dict[str, pd.DataFrame]:
    cells_raw = pd.read_csv(DATA_ROOT / "cells.csv")
    expr = pd.read_csv(DATA_ROOT / "expression.csv")
    cells = cells_raw[cells_raw["cell_type"].ne("unannotated")].copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["x_plot"] = cells["x"].astype(float)
    cells["y_plot"] = -cells["y"].astype(float)
    expr["cell_id"] = expr["cell_id"].astype(str)
    genes = sorted(set([LIGAND, RECEPTOR, *RESPONSE_GENES]))
    cells = cells[["cell_id", "x_plot", "y_plot", "cell_type"]].merge(expr[["cell_id", *genes]], on="cell_id", how="left")
    for gene in genes:
        cells[gene] = cells[gene].fillna(0.0).astype(float)

    merged = read_annotated_predictions(cells)
    target = merged[merged["lr_pair"].eq(LR_PAIR)].copy()
    other = merged[~merged["lr_pair"].eq(LR_PAIR)].copy()
    regional_target = target[target["receiver_cell_type"].eq(REGION_CELL_TYPE)].copy()
    score_cells = build_receiver_score_cells(cells, regional_target, region_only=True)
    response, validation = build_response_validation(score_cells)
    stats_table = pd.DataFrame([response_stats(validation)])
    abundance = build_abundance(merged)
    regional_summary = build_regional_summary(merged)
    regional_priority = build_regional_priority(merged)
    density = build_density_tables(regional_target, merged[(merged["receiver_cell_type"].eq(REGION_CELL_TYPE)) & (~merged["lr_pair"].eq(LR_PAIR))].copy())
    edge_series = build_edge_series(regional_target)

    summary = pd.DataFrame(
        [
            {"metric": "annotated_beads", "value": len(cells)},
            {"metric": "annotated_candidate_edges", "value": len(merged)},
            {"metric": "target_edges", "value": len(target)},
            {"metric": "regional_target_edges", "value": len(regional_target)},
            {"metric": "target_rank_q95", "value": float(rank_target_q95(merged))},
            {"metric": "regional_q95_rank", "value": float(regional_summary.loc[regional_summary["metric"].eq("q95 score"), "rank"].iloc[0])},
            {"metric": "target_abundance_rank_top20", "value": float(abundance.loc[abundance["lr_pair"].eq(LR_PAIR), "rank"].iloc[0])},
            {"metric": "receiver_response_delta_z", "value": float(stats_table["delta_mean"].iloc[0])},
            {"metric": "receiver_response_p_mannwhitney", "value": float(stats_table["p_mannwhitney"].iloc[0])},
        ]
    )
    return {
        "panel_a_cells": cells,
        "panel_b_score": score_cells,
        "panel_c_abundance": abundance,
        "panel_c_regional_summary": regional_summary,
        "panel_c_priority": regional_priority,
        "panel_d_density": density,
        "panel_e_edges": edge_series,
        "panel_f_response": response,
        "panel_f_validation": validation,
        "panel_f_stats": stats_table,
        "summary": summary,
    }


def read_annotated_predictions(cells: pd.DataFrame) -> pd.DataFrame:
    annotated = cells[["cell_id", "x_plot", "y_plot", "cell_type"]].copy()
    annotated["cell_id"] = annotated["cell_id"].astype(str)
    annotated_ids = set(annotated["cell_id"])
    usecols = [
        "candidate_id",
        "sender_id",
        "receiver_id",
        "ligand",
        "receptor",
        "score",
        "ligand_expr",
        "receptor_expr",
        "coexpression_distance_score",
        "distance",
    ]
    df = pd.read_csv(RESULT_ROOT / "merged_predictions.csv", usecols=usecols)
    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df = df[df["sender_id"].isin(annotated_ids) & df["receiver_id"].isin(annotated_ids)].copy()
    df["lr_pair"] = df["ligand"] + "-" + df["receptor"]
    sender = annotated.add_prefix("sender_")
    receiver = annotated.add_prefix("receiver_")
    df = df.merge(sender, left_on="sender_id", right_on="sender_cell_id", how="left")
    df = df.merge(receiver, left_on="receiver_id", right_on="receiver_cell_id", how="left")
    return df


def build_receiver_score_cells(cells: pd.DataFrame, target: pd.DataFrame, region_only: bool = False) -> pd.DataFrame:
    receiver_score = target.groupby("receiver_id", as_index=False).agg(
        scalecomm_score=("score", "mean"),
        n_edges=("score", "size"),
        receptor_expr_mean=("receptor_expr", "mean"),
    )
    out = cells.merge(receiver_score, left_on="cell_id", right_on="receiver_id", how="left")
    out["scalecomm_score"] = out["scalecomm_score"].fillna(0.0)
    out["n_edges"] = out["n_edges"].fillna(0).astype(int)
    if region_only:
        out["in_region"] = out["cell_type"].eq(REGION_CELL_TYPE)
    else:
        out["in_region"] = True
    pos = out.loc[out["scalecomm_score"].gt(0) & out["in_region"], "scalecomm_score"]
    cutoff = float(pos.quantile(0.985)) if len(pos) else 1.0
    out["hotspot"] = out["scalecomm_score"].ge(cutoff) & out["in_region"]
    return out


def build_response_validation(score_cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = score_cells.copy()
    response = response[response["in_region"]].copy()
    genes = [gene for gene in RESPONSE_GENES if gene in response.columns]
    values = response[genes].fillna(0.0).astype(float)
    z = (values - values.mean(axis=0)) / (values.std(axis=0, ddof=0) + 1e-9)
    response["response_score"] = z.mean(axis=1)
    validation = response[["cell_id", "hotspot", "response_score"]].copy()
    validation["group"] = np.where(validation["hotspot"], "ScaleComm receiver", "other beads")
    return response, validation


def build_regional_summary(merged: pd.DataFrame) -> pd.DataFrame:
    reg = merged[merged["receiver_cell_type"].eq(REGION_CELL_TYPE)].copy()
    q95 = reg.groupby("lr_pair")["score"].quantile(0.95).sort_values(ascending=False)
    top20 = reg[reg["score"].ge(reg["score"].quantile(0.80))]
    abundance = top20["lr_pair"].value_counts()
    rows = []
    specs = [
        ("q95 score", q95.loc[LR_PAIR], q95.index.get_loc(LR_PAIR) + 1, None),
        ("top20 edge count", float(abundance.get(LR_PAIR, 0)), abundance.index.get_loc(LR_PAIR) + 1 if LR_PAIR in abundance.index else 999, None),
    ]
    top = reg[reg["lr_pair"].eq(LR_PAIR)].sort_values("score", ascending=False).head(500)
    specs.extend(
        [
            ("receptor+ top500", float((top["receptor_expr"] > 0).mean()), None, "fraction"),
            ("ligand+ top500", float((top["ligand_expr"] > 0).mean()), None, "fraction"),
            ("coexpr+ top500", float((top["coexpression_distance_score"] > 0).mean()), None, "fraction"),
        ]
    )
    for metric, value, rank, kind in specs:
        rows.append({"metric": metric, "value": value, "rank": rank if rank is not None else np.nan, "kind": kind or "rank"})
    return pd.DataFrame(rows)


def build_regional_priority(merged: pd.DataFrame) -> pd.DataFrame:
    reg = merged[merged["receiver_cell_type"].eq(REGION_CELL_TYPE)].copy()
    if reg.empty:
        return pd.DataFrame(
            columns=[
                "lr_pair",
                "q95_score",
                "mean_score",
                "top20_edge_count",
                "q95_rank",
                "top20_rank",
                "is_target",
            ]
        )
    cutoff = float(reg["score"].quantile(0.80))
    q95 = reg.groupby("lr_pair")["score"].quantile(0.95).rename("q95_score")
    mean = reg.groupby("lr_pair")["score"].mean().rename("mean_score")
    top20 = reg[reg["score"].ge(cutoff)]["lr_pair"].value_counts().rename("top20_edge_count")
    table = pd.concat([q95, mean, top20], axis=1).fillna({"top20_edge_count": 0}).reset_index()
    table["top20_edge_count"] = table["top20_edge_count"].astype(int)
    table["q95_rank"] = table["q95_score"].rank(method="min", ascending=False).astype(int)
    table["top20_rank"] = table["top20_edge_count"].rank(method="min", ascending=False).astype(int)
    table["is_target"] = table["lr_pair"].eq(LR_PAIR)
    table = table.sort_values(["is_target", "q95_score", "top20_edge_count"], ascending=[True, False, False])
    return table


def build_abundance(merged: pd.DataFrame) -> pd.DataFrame:
    cutoff = float(merged["score"].quantile(0.80))
    top = merged[merged["score"].ge(cutoff)].copy()
    table = top["lr_pair"].value_counts().rename_axis("lr_pair").reset_index(name="top20_edge_count")
    table["rank"] = np.arange(1, len(table) + 1)
    table["is_target"] = table["lr_pair"].eq(LR_PAIR)
    return table


def build_density_tables(target: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n_other = min(220_000, len(other))
    other_sample = other.sample(n_other, random_state=21) if len(other) > n_other else other.copy()
    rows = []
    for metric, label in [("score", "ScaleComm score"), ("coexpression_distance_score", "coexpression score")]:
        vals_target = target[metric].to_numpy(dtype=float)
        vals_other = other_sample[metric].to_numpy(dtype=float)
        if metric == "coexpression_distance_score":
            vals_target = np.log10(vals_target + 1e-6)
            vals_other = np.log10(vals_other + 1e-6)
        for group, vals in [(LR_PAIR, vals_target), ("other LR pairs", vals_other)]:
            vals = vals[np.isfinite(vals)]
            if len(vals) > 120_000:
                vals = rng.choice(vals, 120_000, replace=False)
            grid, density = kde_curve(vals, metric)
            for x, y in zip(grid, density, strict=False):
                rows.append({"metric": label, "group": group, "x": x, "density": y})
    return pd.DataFrame(rows)


def kde_curve(values: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "coexpression_distance_score":
        grid = np.linspace(-6.0, -0.6, 180)
    else:
        grid = np.linspace(0.0, 1.0, 180)
    if len(values) < 5 or np.nanstd(values) < 1e-10:
        density = np.zeros_like(grid)
        return grid, density
    kde = stats.gaussian_kde(values)
    density = kde(grid)
    if density.max() > 0:
        density = density / density.max()
    return grid, density


def build_edge_series(target: pd.DataFrame) -> pd.DataFrame:
    if target.empty:
        return target.copy()
    target = target.sort_values("score", ascending=False).copy()
    rows = []
    for keep_n, label in [(1500, "top 1,500"), (750, "top 750"), (250, "top 250")]:
        sub = target.head(min(keep_n, len(target))).copy()
        sub["edge_set"] = label
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def rank_target_q95(merged: pd.DataFrame) -> int:
    q95 = merged.groupby("lr_pair")["score"].quantile(0.95).sort_values(ascending=False)
    return int(q95.index.get_loc(LR_PAIR) + 1)


def response_stats(validation: pd.DataFrame) -> dict[str, float]:
    hot = validation.loc[validation["group"].eq("ScaleComm receiver"), "response_score"].dropna().to_numpy(dtype=float)
    bg = validation.loc[validation["group"].eq("other beads"), "response_score"].dropna().to_numpy(dtype=float)
    p = float(stats.mannwhitneyu(hot, bg, alternative="greater").pvalue) if len(hot) and len(bg) else np.nan
    return {
        "n_hot": int(len(hot)),
        "n_other": int(len(bg)),
        "delta_mean": float(np.nanmean(hot) - np.nanmean(bg)) if len(hot) and len(bg) else np.nan,
        "p_mannwhitney": p,
    }


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(7.35, 6.85), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        left=0.052,
        right=0.985,
        bottom=0.085,
        top=0.958,
        wspace=0.230,
        hspace=0.260,
        width_ratios=[1.15, 1.0, 1.0],
        height_ratios=[1.05, 1.02, 1.24],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    bsub = gs[0, 1:3].subgridspec(1, 2, wspace=0.090)
    ax_b1 = fig.add_subplot(bsub[0, 0])
    ax_b2 = fig.add_subplot(bsub[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, 0:3])

    plot_cell_map(ax_a, source["panel_a_cells"])
    expr_cmap = LinearSegmentedColormap.from_list("expr", ["#F8F7F1", AMBER, ORANGE, ROSE])
    plot_expression(ax_b1, source["panel_b_score"], LIGAND, expr_cmap, "Fgf2")
    plot_expression(ax_b2, source["panel_b_score"], RECEPTOR, expr_cmap, "Fgfr1")
    plot_ca3_lr_rank(ax_c, source["panel_c_priority"])
    plot_density_pair(ax_d, source["panel_d_density"])
    plot_top_edge_series(fig, ax_e, source["panel_a_cells"], source["panel_e_edges"])
    plot_response_support(ax_f, source["panel_f_validation"], source["panel_f_stats"])

    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b1, "b")
    panel_label(fig, ax_c, "c")
    panel_label(fig, ax_d, "d")
    panel_label(fig, ax_f, "f")
    panel_label(fig, ax_e, "e", dx=0.014, dy=0.012)
    return fig


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    other = cells[~cells["cell_type"].isin(DISPLAY_TYPES)]
    ax.scatter(other["x_plot"], other["y_plot"], s=1.35, c=TISSUE_DARK, alpha=0.28, linewidths=0, rasterized=False)
    for cell_type in DISPLAY_TYPES:
        sub = cells[cells["cell_type"].eq(cell_type)]
        ax.scatter(sub["x_plot"], sub["y_plot"], s=2.0, c=CELL_COLORS.get(cell_type, TISSUE), alpha=0.86, linewidths=0, rasterized=False)
    set_title(ax, "Slide-seqV2 hippocampus")
    add_spatial_frame(ax, cells, pad_frac=0.022)
    add_celltype_key(ax)


def add_celltype_key(ax: plt.Axes) -> None:
    labels = [
        ("astrocyte", CELL_COLORS["astrocyte"]),
        ("dentate", CELL_COLORS["dentate_neuron"]),
        ("CA1", CELL_COLORS["CA1_neuron"]),
        ("CA3", CELL_COLORS["CA3_neuron"]),
        ("interneuron", CELL_COLORS["interneuron"]),
        ("oligo", CELL_COLORS["oligodendrocyte"]),
        ("microglia", CELL_COLORS["microglia_macrophage"]),
        ("other", TISSUE_DARK),
    ]
    x0, y0 = 0.050, 0.050
    dx, dy = 0.245, 0.044
    for i, (label, color) in enumerate(labels):
        x = x0 + (i % 4) * dx
        y = y0 + (1 - i // 4) * dy
        ax.scatter([x], [y], s=8.8, c=color, transform=ax.transAxes, linewidths=0, clip_on=False, zorder=20)
        ax.text(x + 0.020, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.6, color=MUTED, zorder=20)


def plot_expression(ax: plt.Axes, data: pd.DataFrame, gene: str, cmap: LinearSegmentedColormap, title: str) -> None:
    frame = data.sort_values([gene, "cell_id"], ascending=[True, True]).copy()
    values = frame[gene].to_numpy(dtype=float)
    vmax = float(np.nanquantile(values[values > 0], 0.99)) if np.any(values > 0) else 1.0
    vmax = max(vmax, 1e-9)
    ax.scatter(frame["x_plot"], frame["y_plot"], s=1.55, c=TISSUE, alpha=0.30, linewidths=0, rasterized=False)
    pos = frame[frame[gene].gt(0)]
    ax.scatter(
        pos["x_plot"],
        pos["y_plot"],
        s=4.0 if gene == RECEPTOR else 3.0,
        c=np.clip(pos[gene].to_numpy(dtype=float), 0, vmax),
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=vmax),
        alpha=0.98,
        linewidths=0,
        rasterized=False,
    )
    set_title(ax, title)
    add_spatial_frame(ax, data)
    add_colorbar(ax, cmap, Normalize(0, vmax), width=0.42)


def plot_abundance(ax: plt.Axes, abundance: pd.DataFrame) -> None:
    top = abundance.head(18).copy()
    colors = [BLUE if pair == LR_PAIR else "#D8D3CB" for pair in top["lr_pair"]]
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["top20_edge_count"], height=0.62, color=colors, edgecolor="none")
    ax.set_yticks(y)
    labels = [pair if pair == LR_PAIR else pair.replace("-", "-") for pair in top["lr_pair"]]
    ax.set_yticklabels(labels, fontsize=4.75)
    ax.set_xlabel("edges in top 20%", fontsize=5.2, color=MUTED, labelpad=1.5)
    ax.set_title("LR abundance in high-score edges", fontsize=5.8, pad=1.2, color=INK)
    target_row = top[top["lr_pair"].eq(LR_PAIR)]
    if len(target_row):
        yy = y[target_row.index[0]]
        xx = float(target_row["top20_edge_count"].iloc[0])
        ax.text(xx * 1.01, yy, f"rank {int(target_row['rank'].iloc[0])}", ha="left", va="center", fontsize=4.6, color=BLUE)
    style_quant_axis(ax)
    ax.grid(axis="x", color=GRID, lw=0.35)
    ax.set_axisbelow(True)


def plot_regional_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    order = ["q95 score", "top20 edge count", "receptor+ top500", "ligand+ top500", "coexpr+ top500"]
    table = summary.set_index("metric").loc[order].reset_index()
    values = []
    labels = []
    annotations = []
    for row in table.itertuples(index=False):
        if row.metric in {"q95 score", "top20 edge count"}:
            scaled = max(0.05, (24 - float(row.rank)) / 23.0)
            ann = f"rank {int(row.rank)}"
        else:
            scaled = float(row.value)
            ann = f"{100 * float(row.value):.1f}%"
        values.append(scaled)
        labels.append(row.metric.replace("top500", "top 500").replace("top20", "top 20%"))
        annotations.append(ann)
    y = np.arange(len(labels))[::-1]
    colors = [BLUE, BLUE, "#6E8FAE", "#8BA47D", "#B99B56"]
    ax.barh(y, values, color=colors, height=0.58, edgecolor="none", alpha=0.92)
    for yy, value, ann in zip(y, values, annotations, strict=False):
        ax.text(min(value + 0.035, 1.02), yy, ann, ha="left", va="center", fontsize=4.7, color=BLUE if "rank" in ann else MUTED)
    ax.set_xlim(0, 1.08)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.85)
    ax.set_xlabel("regional support", fontsize=5.2, color=MUTED, labelpad=1.5)
    ax.set_title(f"{REGION_LABEL} receiver-region evidence", fontsize=5.8, pad=1.2, color=INK)
    style_quant_axis(ax)
    ax.grid(axis="x", color=GRID, lw=0.35)
    ax.set_axisbelow(True)


def plot_lr_priority(ax: plt.Axes, priority: pd.DataFrame) -> None:
    if priority.empty:
        ax.set_axis_off()
        return
    other = priority[~priority["is_target"]].copy()
    target = priority[priority["is_target"]].copy()
    ax.scatter(
        other["q95_score"],
        other["top20_edge_count"],
        s=13,
        c="#D6D1C8",
        alpha=0.68,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        target["q95_score"],
        target["top20_edge_count"],
        s=34,
        c=BLUE,
        alpha=0.96,
        edgecolors=BLACK,
        linewidths=0.34,
        zorder=4,
    )
    if len(target):
        row = target.iloc[0]
        ax.axvline(row["q95_score"], color=mpl.colors.to_rgba(BLUE, 0.20), lw=0.58, zorder=1)
        ax.axhline(row["top20_edge_count"], color=mpl.colors.to_rgba(BLUE, 0.20), lw=0.58, zorder=1)
        ax.text(
            row["q95_score"] - 0.018,
            row["top20_edge_count"] + max(18, priority["top20_edge_count"].max() * 0.025),
            f"{LR_PAIR}\nrank {int(row['q95_rank'])} / {int(row['top20_rank'])}",
            ha="right",
            va="bottom",
            fontsize=4.9,
            color=BLUE,
            linespacing=0.92,
        )
    x_min = max(0.0, float(priority["q95_score"].min()) - 0.035)
    x_max = min(1.02, float(priority["q95_score"].max()) + 0.018)
    y_max = float(priority["top20_edge_count"].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max * 0.035, y_max * 1.10)
    ax.set_xlabel("CA3 receiver q95 score", fontsize=5.2, color=MUTED, labelpad=1.5)
    ax.set_ylabel("high-score CA3 edges", fontsize=5.2, color=MUTED, labelpad=1.5)
    set_title(ax, "CA3-region LR prioritization")
    style_quant_axis(ax)
    ax.grid(color=GRID, lw=0.32)
    ax.set_axisbelow(True)
    ax.text(
        0.025,
        0.965,
        "each dot = one LR pair",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.55,
        color=MUTED,
    )


def plot_ca3_lr_rank(ax: plt.Axes, priority: pd.DataFrame) -> None:
    if priority.empty:
        ax.set_axis_off()
        return
    top = priority.sort_values("q95_score", ascending=False).head(12).copy()
    top = top.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))
    colors = np.where(top["is_target"], BLUE, "#D8D3CB")
    ax.barh(y, top["q95_score"], height=0.58, color=colors, edgecolor="none", alpha=0.96)
    max_edges = max(float(top["top20_edge_count"].max()), 1.0)
    edge_sizes = 8.0 + 28.0 * np.sqrt(top["top20_edge_count"].astype(float) / max_edges)
    ax.scatter(
        np.repeat(1.006, len(top)),
        y,
        s=edge_sizes,
        c=np.where(top["is_target"], BLUE, "#B5B0A8"),
        edgecolors=BLACK,
        linewidths=np.where(top["is_target"], 0.28, 0.18),
        clip_on=False,
        zorder=4,
    )
    labels = [pair if pair == LR_PAIR else pair for pair in top["lr_pair"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.95)
    for tick, is_target in zip(ax.get_yticklabels(), top["is_target"], strict=False):
        if is_target:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
        else:
            tick.set_color(TEXT)
    if top["is_target"].any():
        row = top[top["is_target"]].iloc[0]
        yy = int(top.index[top["is_target"]][0])
        ax.text(
            0.955,
            yy,
            "rank 4",
            ha="right",
            va="center",
            fontsize=4.8,
            color=BG,
            fontweight="bold",
        )
        ax.text(
            1.028,
            yy,
            "edge\nrank 7",
            ha="left",
            va="center",
            fontsize=4.25,
            color=BLUE,
            linespacing=0.88,
            clip_on=False,
        )
    ax.set_xlim(0.945, 1.014)
    ax.set_xlabel("CA3 receiver q95 score", fontsize=5.2, color=MUTED, labelpad=1.5)
    set_title(ax, "CA3-region LR ranking")
    style_quant_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.set_xticks([0.95, 0.98, 1.00])
    ax.grid(axis="x", color=GRID, lw=0.32)
    ax.set_axisbelow(True)
    ax.text(
        1.004,
        len(top) - 0.25,
        "edge count",
        ha="left",
        va="bottom",
        fontsize=4.35,
        color=MUTED,
        clip_on=False,
    )


def plot_density_pair(ax: plt.Axes, density: pd.DataFrame) -> None:
    ax.set_axis_off()
    sub = ax.get_subplotspec().subgridspec(2, 1, hspace=0.46)
    ax1 = ax.figure.add_subplot(sub[0, 0])
    ax2 = ax.figure.add_subplot(sub[1, 0])
    plot_density(ax1, density[density["metric"].eq("ScaleComm score")], "ScaleComm score", xlabel="score")
    plot_density(ax2, density[density["metric"].eq("coexpression score")], "coexpression control", xlabel="log10 coexpression")


def plot_density(ax: plt.Axes, data: pd.DataFrame, title: str, xlabel: str) -> None:
    for group, color, alpha, lw in [("other LR pairs", "#BFC5CE", 0.90, 1.05), (LR_PAIR, BLUE, 0.95, 1.25)]:
        sub = data[data["group"].eq(group)]
        ax.fill_between(sub["x"], sub["density"], color=mpl.colors.to_rgba(color, 0.18 if group == LR_PAIR else 0.25), linewidth=0)
        ax.plot(sub["x"], sub["density"], color=color, alpha=alpha, lw=lw)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "max"], fontsize=4.5)
    ax.set_xlabel(xlabel, fontsize=4.8, color=MUTED, labelpad=1.0)
    ax.set_title(title, fontsize=5.25, pad=1.0, color=INK)
    style_quant_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.30)
    if title.startswith("ScaleComm"):
        ax.text(0.97, 0.80, LR_PAIR, transform=ax.transAxes, ha="right", va="center", fontsize=4.6, color=BLUE)
        ax.text(0.97, 0.60, "other LR", transform=ax.transAxes, ha="right", va="center", fontsize=4.5, color="#7D8490")
    else:
        ax.text(0.97, 0.79, "most edges have\nzero coexpression", transform=ax.transAxes, ha="right", va="center", fontsize=4.2, color=MUTED)


def plot_top_edge_series(fig: plt.Figure, host: plt.Axes, cells: pd.DataFrame, edge_series: pd.DataFrame) -> None:
    host.set_axis_off()
    sub = host.get_subplotspec().subgridspec(1, 3, wspace=0.070)
    labels = ["top 1,500", "top 750", "top 250"]
    for i, label in enumerate(labels):
        ax = fig.add_subplot(sub[0, i])
        plot_edge_panel(ax, cells, edge_series[edge_series["edge_set"].eq(label)], label)
    host.set_title(f"{LR_PAIR} top CA3-receiver edge convergence", fontsize=5.8, pad=2.2, color=INK)


def plot_edge_panel(ax: plt.Axes, cells: pd.DataFrame, edges: pd.DataFrame, label: str) -> None:
    ax.scatter(cells["x_plot"], cells["y_plot"], s=1.35, c=TISSUE, alpha=0.26, linewidths=0, rasterized=False)
    if not edges.empty:
        n_lines = min(120, len(edges))
        line_edges = edges.head(n_lines)
        for row in line_edges.itertuples(index=False):
            ax.plot(
                [row.sender_x_plot, row.receiver_x_plot],
                [row.sender_y_plot, row.receiver_y_plot],
                color=mpl.colors.to_rgba(BLUE, 0.035),
                lw=0.12,
                solid_capstyle="round",
                zorder=2,
            )
        recv = edges[["receiver_id", "receiver_x_plot", "receiver_y_plot"]].drop_duplicates("receiver_id")
        send = edges[["sender_id", "sender_x_plot", "sender_y_plot"]].drop_duplicates("sender_id")
        ax.scatter(send["sender_x_plot"], send["sender_y_plot"], s=1.9, c=GREEN, alpha=0.075, linewidths=0, rasterized=False, zorder=3)
        ax.scatter(recv["receiver_x_plot"], recv["receiver_y_plot"], s=3.0, c=BLUE, alpha=0.34, linewidths=0, rasterized=False, zorder=4)
    ax.text(0.050, 0.060, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=4.75, color=BLUE)
    add_spatial_frame(ax, cells)


def plot_response_support(ax: plt.Axes, validation: pd.DataFrame, stats_table: pd.DataFrame) -> None:
    rng = np.random.default_rng(20)
    groups = ["other beads", "ScaleComm receiver"]
    colors = ["#D8D3CB", BLUE]
    data = [validation.loc[validation["group"].eq(group), "response_score"].dropna().to_numpy(dtype=float) for group in groups]
    bp = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": BLACK, "linewidth": 0.64},
        whiskerprops={"color": BLACK, "linewidth": 0.50},
        capprops={"color": BLACK, "linewidth": 0.50},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set(facecolor=mpl.colors.to_rgba(color, 0.24), edgecolor=BLACK, linewidth=0.42)
    for idx, vals in enumerate(data):
        sample = vals if len(vals) <= 220 else rng.choice(vals, 220, replace=False)
        x = idx + rng.normal(0, 0.040, len(sample))
        ax.scatter(x, sample, s=4.2, c=colors[idx], alpha=0.34 if idx == 0 else 0.58, edgecolors="none", linewidths=0, zorder=3)
        ax.scatter([idx], [np.nanmean(vals)], s=13, marker="D", c=colors[idx], edgecolors=BLACK, linewidths=0.22, zorder=5)
    stat = stats_table.iloc[0].to_dict() if len(stats_table) else {}
    delta = float(stat.get("delta_mean", np.nan))
    p = float(stat.get("p_mannwhitney", np.nan))
    p_text = "p<1e-4" if np.isfinite(p) and p < 1e-4 else (f"p={p:.2g}" if np.isfinite(p) else "p=NA")
    ax.text(0.50, 0.965, f"Delta z={delta:.2f}; {p_text}", transform=ax.transAxes, ha="center", va="top", fontsize=4.75, color=BLUE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nbeads", "receiver\nbeads"], fontsize=5.0)
    ax.set_ylabel("receiver response\n(z-score)", fontsize=5.25, color=MUTED)
    set_title(ax, "receiver response support")
    style_quant_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.35)


def add_spatial_frame(ax: plt.Axes, data: pd.DataFrame, pad_frac: float = 0.018) -> None:
    ax.set_aspect("equal")
    if len(data):
        xspan = float(data["x_plot"].max() - data["x_plot"].min())
        yspan = float(data["y_plot"].max() - data["y_plot"].min())
        pad = max(xspan, yspan) * pad_frac
        ax.set_xlim(float(data["x_plot"].min()) - pad, float(data["x_plot"].max()) + pad)
        ax.set_ylim(float(data["y_plot"].min()) - pad, float(data["y_plot"].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_colorbar(ax: plt.Axes, cmap: LinearSegmentedColormap, norm: Normalize, width: float = 0.46) -> None:
    x0 = (1.0 - width) / 2.0
    cax = ax.inset_axes([x0, -0.070, width, 0.030])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.outline.set_edgecolor(BLACK)
    cb.outline.set_linewidth(0.30)
    cb.set_ticks([norm.vmin, norm.vmax])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=3.9, length=1.0, width=0.25, pad=0.5)


def style_quant_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.spines["left"].set_color("#7A746C")
    ax.spines["bottom"].set_color("#7A746C")
    ax.tick_params(axis="both", labelsize=5.0, length=1.8, width=0.45, pad=1.0)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=5.85, pad=1.0, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = 0.018, dy: float = 0.008) -> None:
    box = ax.get_position()
    fig.text(box.x0 - dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=8.4, fontweight="bold", color=INK)


if __name__ == "__main__":
    main()
