#!/usr/bin/env python
"""CellNEST-style Slide-seqV2 hippocampus story for Cxcl12-Cxcr4.

Focused evidence chain:
cell-type map -> ligand/receptor availability -> ScaleComm-localized niche ->
sender/receiver layers -> cell-type interaction composition -> receiver response support.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
PREV_SOURCE = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v19_cytosignal_full_interactions/source_data"
OUT_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_single_axis_cxcl12_cxcr4_v20_scalecomm_cellnest_logic"

LIGAND = "Cxcl12"
RECEPTOR = "Cxcr4"
LR_PAIR = f"{LIGAND}-{RECEPTOR}"
RESPONSE_GENES = ["Gfap", "Aqp4", "Slc1a2", "Pdgfra"]

INK = "#222222"
TEXT = "#3B3B3B"
MUTED = "#6C7179"
GRID = "#E7E2D8"
TISSUE = "#DDE4EF"
TISSUE_DARK = "#C8D1DE"
BLUE = "#2F5F8F"
BLUE_DARK = "#223D73"
GREEN = "#63B35D"
GREEN_DARK = "#3E8B44"
AMBER = "#E6C06A"
ROSE = "#B95C6A"
ORANGE = "#D9903D"
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
    "cajal_retzius": "#E8C7A6",
    "neurogenic": "#BFD8B8",
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
    out = OUT_DIR / "slideseqv2_hpc_cxcl12_cxcr4_single_axis_story"
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
    cells = pd.read_csv(PREV_SOURCE / "panel_a_cells.csv")
    cells["cell_id"] = cells["cell_id"].astype(str)
    score = pd.read_csv(PREV_SOURCE / "panel_b_expression_score.csv")
    response = pd.read_csv(PREV_SOURCE / "panel_e_response.csv")
    validation = pd.read_csv(PREV_SOURCE / "panel_f_validation.csv")

    top_edges = read_top20_edges(cells)
    if top_edges.empty:
        top_edges = pd.read_csv(PREV_SOURCE / "panel_e_full_edges.csv")

    receiver_count = top_edges.groupby("receiver_id", as_index=False).agg(
        receiver_events=("score", "size"),
        receiver_mean_score=("score", "mean"),
    )
    sender_count = top_edges.groupby("sender_id", as_index=False).agg(
        sender_events=("score", "size"),
        sender_mean_score=("score", "mean"),
    )
    score = score.drop(columns=[c for c in ["receiver_events", "receiver_mean_score"] if c in score.columns], errors="ignore")
    score = score.merge(receiver_count, left_on="cell_id", right_on="receiver_id", how="left")
    score["receiver_events"] = score["receiver_events"].fillna(0).astype(int)
    score["receiver_event_scaled"] = scaled_positive(score["receiver_events"].astype(float), 1, 99)

    sender_layer = cells.merge(sender_count, left_on="cell_id", right_on="sender_id", how="left")
    receiver_layer = cells.merge(receiver_count, left_on="cell_id", right_on="receiver_id", how="left")
    sender_layer["sender_events"] = sender_layer["sender_events"].fillna(0).astype(int)
    receiver_layer["receiver_events"] = receiver_layer["receiver_events"].fillna(0).astype(int)
    sender_layer["sender_scaled"] = scaled_count_signal(sender_layer["sender_events"].astype(float))
    receiver_layer["receiver_scaled"] = scaled_count_signal(receiver_layer["receiver_events"].astype(float))

    comp = interaction_composition(top_edges)
    response = response.copy()
    validation_high = response[["cell_id", "hotspot", "response_score"]].copy()
    validation_high["group"] = np.where(validation_high["hotspot"], "ScaleComm receiver", "other beads")

    stat = response_stats(validation_high)
    stat_table = pd.DataFrame([stat])
    summary = pd.DataFrame(
        [
            {"metric": "annotated_beads", "value": len(cells)},
            {"metric": "top20_edges", "value": len(top_edges)},
            {"metric": "top20_sender_beads", "value": int(top_edges["sender_id"].nunique())},
            {"metric": "top20_receiver_beads", "value": int(top_edges["receiver_id"].nunique())},
            {"metric": "receiver_response_delta_z", "value": stat["delta_mean"]},
            {"metric": "receiver_response_p_mannwhitney", "value": stat["p_mannwhitney"]},
        ]
    )

    return {
        "panel_a_cells": cells,
        "panel_b_score": score,
        "panel_c_edges_top20": top_edges,
        "panel_d_sender_layer": sender_layer,
        "panel_d_receiver_layer": receiver_layer,
        "panel_e_interaction_composition": comp,
        "panel_f_response": response,
        "panel_f_validation": validation_high,
        "panel_f_stats": stat_table,
        "summary": summary,
    }


def read_top20_edges(cells: pd.DataFrame) -> pd.DataFrame:
    path = RESULT_ROOT / "merged_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    annotated = cells[["cell_id", "x_plot", "y_plot", "cell_type"]].copy()
    annotated["cell_id"] = annotated["cell_id"].astype(str)
    annotated_ids = set(annotated["cell_id"])
    usecols = ["candidate_id", "sender_id", "receiver_id", "ligand", "receptor", "score", "distance"]
    merged = pd.read_csv(path, usecols=usecols)
    sub = merged[merged["ligand"].eq(LIGAND) & merged["receptor"].eq(RECEPTOR)].copy()
    sub["sender_id"] = sub["sender_id"].astype(str)
    sub["receiver_id"] = sub["receiver_id"].astype(str)
    sub = sub[sub["sender_id"].isin(annotated_ids) & sub["receiver_id"].isin(annotated_ids)].copy()
    if sub.empty:
        return sub
    cutoff = float(sub["score"].quantile(0.80))
    sub = sub[sub["score"].ge(cutoff)].sort_values("score", ascending=False).copy()
    sender = annotated.add_prefix("sender_")
    receiver = annotated.add_prefix("receiver_")
    sub = sub.merge(sender, left_on="sender_id", right_on="sender_cell_id", how="left")
    sub = sub.merge(receiver, left_on="receiver_id", right_on="receiver_cell_id", how="left")
    return sub


def scaled_positive(values: pd.Series, lo_pct: float = 5, hi_pct: float = 99.5) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.nanpercentile(pos, [lo_pct, hi_pct])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def scaled_count_signal(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    out = np.zeros_like(arr, dtype=float)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return out
    hi = max(float(np.nanpercentile(pos, 99)), 1.0)
    out[arr > 0] = np.clip(arr[arr > 0] / hi, 0.36, 1.0)
    return out


def interaction_composition(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["sender", "receiver", "events", "fraction", "sender_label", "receiver_label"])
    table = (
        edges.groupby(["sender_cell_type", "receiver_cell_type"], as_index=False)
        .agg(events=("score", "size"), mean_score=("score", "mean"), mean_distance=("distance", "mean"))
        .rename(columns={"sender_cell_type": "sender", "receiver_cell_type": "receiver"})
    )
    table["fraction"] = table["events"] / table["events"].sum()
    table = table.sort_values(["events", "mean_score"], ascending=False)
    keep_types = set(DISPLAY_TYPES)
    top_pairs = set(map(tuple, table.head(14)[["sender", "receiver"]].to_numpy()))
    table = table[(table["sender"].isin(keep_types) & table["receiver"].isin(keep_types)) | table[["sender", "receiver"]].apply(tuple, axis=1).isin(top_pairs)].copy()
    table["sender_label"] = table["sender"].map(pretty_cell)
    table["receiver_label"] = table["receiver"].map(pretty_cell)
    return table


def response_stats(validation: pd.DataFrame) -> dict[str, float]:
    hot = validation.loc[validation["group"].eq("ScaleComm receiver"), "response_score"].dropna().to_numpy(dtype=float)
    bg = validation.loc[validation["group"].eq("other beads"), "response_score"].dropna().to_numpy(dtype=float)
    if len(hot) == 0 or len(bg) == 0:
        return {"n_hot": len(hot), "n_other": len(bg), "delta_mean": np.nan, "p_mannwhitney": np.nan}
    p = float(stats.mannwhitneyu(hot, bg, alternative="greater").pvalue)
    return {
        "n_hot": int(len(hot)),
        "n_other": int(len(bg)),
        "delta_mean": float(np.nanmean(hot) - np.nanmean(bg)),
        "p_mannwhitney": p,
        "hot_mean": float(np.nanmean(hot)),
        "other_mean": float(np.nanmean(bg)),
    }


def plot_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.95), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        3,
        left=0.035,
        right=0.990,
        bottom=0.060,
        top=0.968,
        wspace=0.165,
        hspace=0.285,
        width_ratios=[1.28, 1.02, 1.02],
        height_ratios=[1.05, 1.0, 0.94],
    )
    ax_a = fig.add_subplot(outer[0:2, 0])
    b_sub = outer[0, 1:3].subgridspec(1, 2, wspace=0.075)
    ax_b1 = fig.add_subplot(b_sub[0, 0])
    ax_b2 = fig.add_subplot(b_sub[0, 1])
    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[1, 2])
    ax_e = fig.add_subplot(outer[2, 0:2])
    ax_f = fig.add_subplot(outer[2, 2])

    plot_cell_map(ax_a, source["panel_a_cells"])
    expr_cmap = LinearSegmentedColormap.from_list("expr", ["#F8F7F1", "#E6C06A", "#D9903D", "#7A517D"])
    plot_expression(ax_b1, source["panel_b_score"], LIGAND, expr_cmap, "Cxcl12")
    plot_expression(ax_b2, source["panel_b_score"], RECEPTOR, expr_cmap, "Cxcr4")
    plot_niche_map(ax_c, source["panel_b_score"], source["panel_c_edges_top20"])
    plot_sender_receiver_layers(ax_d, source["panel_d_sender_layer"], source["panel_d_receiver_layer"])
    plot_interaction_composition(ax_e, source["panel_e_interaction_composition"])
    plot_response_support(ax_f, source["panel_f_response"], source["panel_f_validation"], source["panel_f_stats"])

    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b1, "b")
    panel_label(fig, ax_c, "c")
    panel_label(fig, ax_d, "d")
    panel_label(fig, ax_e, "e", dx=0.014, dy=0.030)
    panel_label(fig, ax_f, "f")
    return fig


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    cells = cells.copy()
    major = cells[cells["cell_type"].isin(DISPLAY_TYPES)].copy()
    other = cells[~cells["cell_type"].isin(DISPLAY_TYPES)].copy()
    ax.scatter(other["x_plot"], other["y_plot"], s=1.35, c=TISSUE_DARK, alpha=0.28, linewidths=0, rasterized=False)
    for cell_type in DISPLAY_TYPES:
        sub = major[major["cell_type"].eq(cell_type)]
        if sub.empty:
            continue
        ax.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=2.05,
            c=CELL_COLORS.get(cell_type, TISSUE),
            alpha=0.86,
            linewidths=0,
            rasterized=False,
        )
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
        ax.scatter([x], [y], s=9.0, c=color, transform=ax.transAxes, linewidths=0, clip_on=False, zorder=20)
        ax.text(x + 0.020, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.65, color=MUTED, zorder=20)


def plot_expression(ax: plt.Axes, data: pd.DataFrame, gene: str, cmap: LinearSegmentedColormap, title: str) -> None:
    frame = data.sort_values([gene, "cell_id"], ascending=[True, True]).copy()
    values = frame[gene].to_numpy(dtype=float)
    vmax = float(np.nanquantile(values[values > 0], 0.99)) if np.any(values > 0) else 1.0
    vmax = max(vmax, 1e-9)
    ax.scatter(frame["x_plot"], frame["y_plot"], s=1.55, c=TISSUE, alpha=0.30, linewidths=0, rasterized=False)
    pos = frame[frame[gene].gt(0)].copy()
    ax.scatter(
        pos["x_plot"],
        pos["y_plot"],
        s=4.1 if gene == RECEPTOR else 3.1,
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


def plot_niche_map(ax: plt.Axes, data: pd.DataFrame, edges: pd.DataFrame) -> None:
    ax.scatter(data["x_plot"], data["y_plot"], s=1.55, c=TISSUE, alpha=0.30, linewidths=0, rasterized=False)
    if not edges.empty:
        recv = edges[["receiver_id", "receiver_x_plot", "receiver_y_plot"]].drop_duplicates("receiver_id").copy()
        send = edges[["sender_id", "sender_x_plot", "sender_y_plot"]].drop_duplicates("sender_id").copy()
        high_cut = float(edges["score"].quantile(0.95))
        high_edges = edges[edges["score"].ge(high_cut)].copy()
        high_recv = high_edges[["receiver_id", "receiver_x_plot", "receiver_y_plot"]].drop_duplicates("receiver_id")
        high_send = high_edges[["sender_id", "sender_x_plot", "sender_y_plot"]].drop_duplicates("sender_id")
        ax.scatter(send["sender_x_plot"], send["sender_y_plot"], s=2.1, c=GREEN, alpha=0.13, linewidths=0, rasterized=False)
        ax.scatter(recv["receiver_x_plot"], recv["receiver_y_plot"], s=2.5, c=BLUE_DARK, alpha=0.18, linewidths=0, rasterized=False)
        for row in high_edges.sort_values("score", ascending=False).head(95).itertuples(index=False):
            ax.plot(
                [row.sender_x_plot, row.receiver_x_plot],
                [row.sender_y_plot, row.receiver_y_plot],
                color=mpl.colors.to_rgba(BLUE_DARK, 0.24),
                lw=0.22,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(high_send["sender_x_plot"], high_send["sender_y_plot"], s=4.8, c=GREEN_DARK, alpha=0.56, linewidths=0, rasterized=False, zorder=4)
        ax.scatter(high_recv["receiver_x_plot"], high_recv["receiver_y_plot"], s=5.6, c=BLUE_DARK, alpha=0.82, linewidths=0, rasterized=False, zorder=5)
        ax.text(
            0.045,
            0.045,
            "top 20% context; dark = top 5%",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=4.55,
            color=MUTED,
        )
        legend_edge(ax)
    set_title(ax, "ScaleComm niche")
    add_spatial_frame(ax, data)


def legend_edge(ax: plt.Axes) -> None:
    x0, y0 = 0.045, 0.125
    ax.scatter([x0], [y0], s=9, c=GREEN, alpha=0.70, transform=ax.transAxes, clip_on=False)
    ax.text(x0 + 0.030, y0, "sender", transform=ax.transAxes, va="center", ha="left", fontsize=4.4, color=MUTED)
    ax.scatter([x0 + 0.23], [y0], s=9, c=BLUE_DARK, alpha=0.80, transform=ax.transAxes, clip_on=False)
    ax.text(x0 + 0.260, y0, "receiver", transform=ax.transAxes, va="center", ha="left", fontsize=4.4, color=MUTED)


def plot_sender_receiver_layers(ax: plt.Axes, sender: pd.DataFrame, receiver: pd.DataFrame) -> None:
    ax.set_axis_off()
    data = sender.copy()
    xmin, xmax = float(data["x_plot"].min()), float(data["x_plot"].max())
    ymin, ymax = float(data["y_plot"].min()), float(data["y_plot"].max())
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    def nx(v: pd.Series) -> np.ndarray:
        return (v.to_numpy(dtype=float) - xmin) / xspan

    def ny(v: pd.Series, offset: float) -> np.ndarray:
        return offset + 0.245 * ((v.to_numpy(dtype=float) - ymin) / yspan - 0.5)

    ax.scatter(nx(data["x_plot"]), ny(data["y_plot"], 0.72), s=1.10, c=TISSUE, alpha=0.12, linewidths=0)
    ax.scatter(nx(data["x_plot"]), ny(data["y_plot"], 0.30), s=1.10, c=TISSUE, alpha=0.12, linewidths=0)

    s_active = sender[sender["sender_events"].gt(0)].copy()
    r_active = receiver[receiver["receiver_events"].gt(0)].copy()
    s_cmap = LinearSegmentedColormap.from_list("sender_layer", ["#F7F7F2", "#BFD9B5", GREEN_DARK])
    r_cmap = LinearSegmentedColormap.from_list("receiver_layer", ["#F7F7F2", "#B9CDE2", BLUE_DARK])
    ax.scatter(nx(s_active["x_plot"]), ny(s_active["y_plot"], 0.72), s=4.7, c=s_active["sender_scaled"], cmap=s_cmap, vmin=0, vmax=1, alpha=0.96, linewidths=0)
    ax.scatter(nx(r_active["x_plot"]), ny(r_active["y_plot"], 0.30), s=4.7, c=r_active["receiver_scaled"], cmap=r_cmap, vmin=0, vmax=1, alpha=0.96, linewidths=0)
    ax.text(0.030, 0.935, "sending layer", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=GREEN_DARK)
    ax.text(0.030, 0.505, "receiving layer", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=BLUE_DARK)
    ax.text(0.970, 0.055, f"senders {len(s_active):,} / receivers {len(r_active):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.35, color=MUTED)
    set_title(ax, "signal layers")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.06, 0.95)


def plot_interaction_composition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    if comp.empty:
        ax.set_axis_off()
        return
    top_senders = comp.groupby("sender_label")["events"].sum().sort_values(ascending=False).head(7).index.tolist()
    top_receivers = comp.groupby("receiver_label")["events"].sum().sort_values(ascending=False).head(7).index.tolist()
    sub = comp[comp["sender_label"].isin(top_senders) & comp["receiver_label"].isin(top_receivers)].copy()
    sub = sub.sort_values(["events", "mean_score"], ascending=False).head(24).copy()
    send_order = top_senders[::-1]
    recv_order = top_receivers
    xmap = {r: i for i, r in enumerate(recv_order)}
    ymap = {s: i for i, s in enumerate(send_order)}
    sub["x"] = sub["receiver_label"].map(xmap)
    sub["y"] = sub["sender_label"].map(ymap)
    max_events = max(float(sub["events"].max()), 1.0)
    log_events = np.log1p(sub["events"].to_numpy(dtype=float))
    log_max = max(float(np.log1p(max_events)), 1e-9)
    sizes = 18 + 178 * (log_events / log_max) ** 1.16
    colors = log_events / log_max
    cmap = LinearSegmentedColormap.from_list("composition", ["#F4EFE6", "#D9C181", "#7C9E7B", BLUE_DARK])
    for x in range(len(recv_order)):
        ax.axvline(x, color="#F1EEE8", lw=0.45, zorder=0)
    for y in range(len(send_order)):
        ax.axhline(y, color="#F1EEE8", lw=0.45, zorder=0)
    sc = ax.scatter(sub["x"], sub["y"], s=sizes, c=colors, cmap=cmap, vmin=0, vmax=1, edgecolors=mpl.colors.to_rgba(BLACK, 0.28), linewidths=0.22, zorder=3)
    ax.set_xticks(range(len(recv_order)))
    ax.set_xticklabels(recv_order, fontsize=5.0, rotation=0)
    ax.set_yticks(range(len(send_order)))
    ax.set_yticklabels(send_order, fontsize=5.0)
    ax.set_xlabel("receiver cell type", fontsize=5.3, labelpad=2.0, color=MUTED)
    ax.set_ylabel("sender cell type", fontsize=5.3, labelpad=2.0, color=MUTED)
    set_title(ax, "cell-type interaction composition")
    ax.set_xlim(-0.55, len(recv_order) - 0.45)
    ax.set_ylim(-0.55, len(send_order) - 0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=1.8)
    add_size_legend(ax, max_events)
    cax = ax.inset_axes([0.730, 1.040, 0.175, 0.035])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.25)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=3.8, length=1.0, width=0.25, pad=0.5)


def add_size_legend(ax: plt.Axes, max_events: float) -> None:
    vals = [max(1, round(max_events * f)) for f in [0.20, 0.55, 1.0]]
    vals = sorted(set(vals))
    x0 = 0.060
    y0 = 1.062
    for i, val in enumerate(vals):
        s = 18 + 178 * (np.log1p(val) / max(float(np.log1p(max_events)), 1e-9)) ** 1.16
        x = x0 + i * 0.090
        y = y0
        ax.scatter([x], [y], s=s, transform=ax.transAxes, facecolors="#E9D9A5", edgecolors=mpl.colors.to_rgba(BLACK, 0.30), linewidths=0.22, clip_on=False)
        ax.text(x + 0.035, y, str(int(val)), transform=ax.transAxes, ha="left", va="center", fontsize=3.9, color=MUTED)
    ax.text(x0, y0 + 0.055, "events", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.05, color=MUTED)


def plot_response_support(ax: plt.Axes, response: pd.DataFrame, validation: pd.DataFrame, stats_table: pd.DataFrame) -> None:
    rng = np.random.default_rng(20)
    groups = ["other beads", "ScaleComm receiver"]
    colors = ["#D8D3CB", BLUE_DARK]
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
        ax.scatter(x, sample, s=4.3, c=colors[idx], alpha=0.34 if idx == 0 else 0.58, edgecolors="none", linewidths=0, zorder=3)
        ax.scatter([idx], [np.nanmean(vals)], s=13, marker="D", c=colors[idx], edgecolors=BLACK, linewidths=0.22, zorder=5)
    stat = stats_table.iloc[0].to_dict() if len(stats_table) else {}
    delta = float(stat.get("delta_mean", np.nan))
    p = float(stat.get("p_mannwhitney", np.nan))
    p_text = "p<1e-4" if np.isfinite(p) and p < 1e-4 else (f"p={p:.2g}" if np.isfinite(p) else "p=NA")
    ax.text(0.50, 0.965, f"Delta z={delta:.2f}; {p_text}", transform=ax.transAxes, ha="center", va="top", fontsize=4.75, color=BLUE_DARK)
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
    ax.tick_params(axis="both", labelsize=5.05, length=1.8, width=0.45, pad=1.0)


def set_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="center", fontsize=5.9, pad=1.0, color=INK)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = 0.018, dy: float = 0.008) -> None:
    box = ax.get_position()
    fig.text(box.x0 - dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=8.4, fontweight="bold", color=INK)


def pretty_cell(label: str) -> str:
    out = label.replace("_neuron", "").replace("microglia_macrophage", "microglia").replace("oligodendrocyte", "oligo")
    return out.replace("_", " ")


if __name__ == "__main__":
    main()
