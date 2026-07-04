#!/usr/bin/env python
"""Plot a first-pass Slide-seqV2 extension-platform SpatialSelf figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/standardized_inputs/slideseqv2_mouse_hpc_full"
RESULT_ROOT = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80"
FIG_DIR = PROJECT_ROOT / "outputs/current_slideseqv2_hpc_full/figures/slideseqv2_hpc_full_platform_v10_biological_story"


CELL_COLORS = {
    "unannotated": "#d6d2ca",
    "astrocyte": "#93cc82",
    "oligodendrocyte": "#b9c1d1",
    "CA1_neuron": "#b9e5fa",
    "CA3_neuron": "#8abedc",
    "dentate_neuron": "#f3d78a",
    "interneuron": "#fbc9c4",
    "endothelial": "#9fc7bd",
    "microglia_macrophage": "#cab2d6",
    "polydendrocyte": "#b9c1d1",
    "neurogenic": "#e6c06a",
}
MECH_COLORS = {
    "long_range_chemokine_niche": "#2F78A8",
    "short_medium_paracrine": "#C86B53",
    "juxtacrine_contact": "#8A8177",
    "negative_control_high_expression": "#B8B0A6",
}
ACCENT = "#2F78A8"
WARM = "#C86B53"
TEXT = "#282522"
GRID = "#E6E1D7"
BG = "#F4F1E8"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    cells = pd.read_csv(DATA_ROOT / "cells.csv")
    lr_summary = pd.read_csv(RESULT_ROOT / "lr_ranking_summary.csv")
    merged = pd.read_csv(RESULT_ROOT / "merged_predictions.csv")
    expression = pd.read_csv(DATA_ROOT / "expression.csv")

    selected_lrs = select_lrs(lr_summary)
    hotspot_tables = {lr: receiver_hotspot(cells, merged, lr) for lr in selected_lrs[:3]}
    response_tables, response_summary = build_response_support(cells, expression, hotspot_tables)
    program_summary = build_program_summary(lr_summary, response_summary, selected_lrs)

    fig = plt.figure(figsize=(11.1, 8.15), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=9,
        height_ratios=[1.1, 0.95, 0.96, 0.96],
        width_ratios=[1.04, 1.04, 1.04, 0.13, 0.82, 0.82, 0.82, 0.82, 0.82],
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.065,
        wspace=0.30,
        hspace=0.56,
    )
    ax_a = fig.add_subplot(gs[0:2, 0:3])
    ax_b = fig.add_subplot(gs[0, 4:9])
    ax_c = fig.add_subplot(gs[1, 4])
    ax_d = fig.add_subplot(gs[1, 5])
    ax_e = fig.add_subplot(gs[1, 6])
    ax_i = fig.add_subplot(gs[1, 7:9])
    ax_f = fig.add_subplot(gs[2:4, 0:3])
    ax_g = fig.add_subplot(gs[2:4, 3:6])
    ax_h = fig.add_subplot(gs[2:4, 6:9])

    plot_cell_map(ax_a, cells)
    plot_program_evidence(ax_b, program_summary)
    for ax, lr in zip([ax_c, ax_d, ax_e], selected_lrs[:3], strict=False):
        plot_hotspot(ax, hotspot_tables[lr], lr)
    plot_response_enrichment(ax_i, response_summary)
    for ax, lr in zip([ax_f, ax_g, ax_h], selected_lrs[:3], strict=False):
        plot_response_map(ax, response_tables[lr], lr)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d, "d")
    add_panel_label(ax_e, "e")
    add_panel_label(ax_f, "f")
    add_panel_label(ax_g, "g")
    add_panel_label(ax_h, "h")
    add_panel_label(ax_i, "i")

    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 1200
        fig.savefig(FIG_DIR / f"slideseqv2_hpc_full_platform_v10_biological_story.{ext}", **kwargs)
    plt.close(fig)

    save_standalone_panel_a(cells)

    source_dir = FIG_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    cells.to_csv(source_dir / "panel_a_cells.csv", index=False)
    lr_summary.to_csv(source_dir / "panel_b_lr_ranking.csv", index=False)
    for lr, table in hotspot_tables.items():
        table.to_csv(source_dir / f"panel_hotspot_{clean_name(lr)}.csv", index=False)
    response_summary.to_csv(source_dir / "panel_f_response_enrichment.csv", index=False)
    program_summary.to_csv(source_dir / "panel_b_program_evidence.csv", index=False)
    for lr, table in response_tables.items():
        table.to_csv(source_dir / f"panel_response_{clean_name(lr)}.csv", index=False)
    print(FIG_DIR / "slideseqv2_hpc_full_platform_v10_biological_story.png")
    print(FIG_DIR / "slideseqv2_hpc_full_platform_v10_biological_story.pdf")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#6A625B",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
        }
    )


def select_lrs(lr_summary: pd.DataFrame) -> list[str]:
    preferred = ["Cxcl12-Cxcr4", "Nrg1-Erbb4", "Vegfa-Flt1", "Reln-Lrp8", "Sema3a-Nrp2"]
    available = {f"{r.ligand}-{r.receptor}" for r in lr_summary.itertuples(index=False)}
    chosen = [lr for lr in preferred if lr in available]
    for row in lr_summary.itertuples(index=False):
        name = f"{row.ligand}-{row.receptor}"
        if name not in chosen and not str(row.mechanism).startswith("negative"):
            chosen.append(name)
        if len(chosen) >= 6:
            break
    return chosen


def plot_cell_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    cells = cells[cells["cell_type"] != "unannotated"].copy()
    counts = cells["cell_type"].value_counts()
    order = [idx for idx in counts.index if idx != "unannotated"][:10]
    for cell_type in order:
        sub = cells[cells["cell_type"] == cell_type]
        count = int(counts[cell_type])
        point_size = 3.7 if count >= 1500 else 4.6 if count >= 500 else 5.8
        ax.scatter(
            sub["x"],
            sub["y"],
            s=point_size,
            c=CELL_COLORS.get(cell_type, "#d6d2ca"),
            alpha=1.0,
            linewidths=0,
            edgecolors="none",
            antialiaseds=False,
            rasterized=False,
            label=pretty_cell(cell_type),
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Slide-seqV2 mouse hippocampus", fontsize=8.5, pad=5, loc="left")
    ax.text(
        0.015,
        0.025,
        "15,003 annotated beads shown",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#5A524C",
        ha="left",
        va="bottom",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.03),
        ncol=5,
        fontsize=5.9,
        handletextpad=0.25,
        columnspacing=0.72,
        borderaxespad=0.0,
        labelspacing=0.34,
        markerscale=1.7,
    )


def save_standalone_panel_a(cells: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.75), constrained_layout=False)
    plot_cell_map(ax, cells)
    add_panel_label(ax, "a")
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 1400
        fig.savefig(FIG_DIR / f"panel_a_annotated_sharp.{ext}", **kwargs)
    plt.close(fig)


def build_program_summary(
    lr_summary: pd.DataFrame,
    response_summary: pd.DataFrame,
    selected_lrs: list[str],
) -> pd.DataFrame:
    table = lr_summary.copy()
    table["lr_pair"] = table["ligand"] + "-" + table["receptor"]
    table = table[table["lr_pair"].isin(selected_lrs[:6])].copy()
    table = table.merge(response_summary, on="lr_pair", how="left")
    table["response_delta_z"] = table["response_delta_z"].fillna(0.0)
    table["n_hot_receivers"] = table["n_hot_receivers"].fillna(0).astype(int)
    table["program_score"] = table["mean_score"].rank(pct=True) + table["response_delta_z"].rank(pct=True)
    order = selected_lrs[:6]
    table["lr_pair"] = pd.Categorical(table["lr_pair"], categories=order, ordered=True)
    return table.sort_values("lr_pair").reset_index(drop=True)


def plot_program_evidence(ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = summary.copy()
    y = np.arange(len(table))[::-1]
    x = table["mean_score"].to_numpy(dtype=float)
    sizes = 42 + np.sqrt(np.maximum(table["n_hot_receivers"].to_numpy(dtype=float), 1)) * 8.5
    colors = [MECH_COLORS.get(str(m), "#BFD7D6") for m in table["mechanism"]]
    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.96,
        zorder=3,
    )
    for xx, yy, row in zip(x, y, table.itertuples(index=False), strict=False):
        if str(row.lr_pair) in {"Cxcl12-Cxcr4", "Nrg1-Erbb4", "Vegfa-Flt1"}:
            ax.text(
                xx + 0.012,
                yy,
                f"+{row.response_delta_z:.2f}",
                fontsize=6.2,
                va="center",
                ha="left",
                color=TEXT,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([str(v) for v in table["lr_pair"]], fontsize=6.6)
    ax.set_xlabel("mean SpatialSelf activity", fontsize=7.2)
    ax.set_xlim(0, max(0.82, float(np.nanmax(x)) + 0.09))
    ax.set_ylim(-0.75, len(table) - 0.25)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_title("LR programs prioritized for hippocampal niches", fontsize=8.2, pad=4, loc="left")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.55)


def receiver_hotspot(cells: pd.DataFrame, merged: pd.DataFrame, lr_pair: str) -> pd.DataFrame:
    ligand, receptor = lr_pair.split("-")
    sub = merged[(merged["ligand"] == ligand) & (merged["receptor"] == receptor)].copy()
    cutoff = float(sub["score"].quantile(0.985))
    sub = sub[sub["score"] >= cutoff]
    hot = sub.groupby("receiver_id", as_index=False).agg(
        hotspot_score=("score", "mean"),
        n_edges=("score", "size"),
    )
    if len(hot) > 1200:
        hot = hot.sort_values("hotspot_score", ascending=False).head(1200)
    annotated = cells[cells["cell_type"] != "unannotated"].copy()
    out = annotated[["cell_id", "x", "y", "cell_type", "niche"]].merge(hot, left_on="cell_id", right_on="receiver_id", how="left")
    out["hotspot_score"] = out["hotspot_score"].fillna(0.0)
    if out["hotspot_score"].max() > 0:
        out["hotspot_scaled"] = out["hotspot_score"] / out["hotspot_score"].max()
    else:
        out["hotspot_scaled"] = 0.0
    out["n_edges"] = out["n_edges"].fillna(0).astype(int)
    return out


def plot_hotspot(ax: plt.Axes, table: pd.DataFrame, lr_pair: str) -> None:
    ax.set_aspect("equal")
    ax.scatter(table["x"], table["y"], s=3.6, c="#D7D1C7", alpha=0.58, linewidths=0, rasterized=True)
    active = table[table["hotspot_scaled"] > 0].copy()
    if not active.empty:
        active = active.sort_values("hotspot_scaled")
        cmap = LinearSegmentedColormap.from_list("hotspot", [BG, "#E6C06A", ACCENT])
        ax.scatter(
            active["x"],
            active["y"],
            s=8.4,
            c=active["hotspot_scaled"],
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0,
            alpha=0.98,
            rasterized=True,
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(lr_pair, fontsize=7.6, pad=3.5)


def build_response_support(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    hotspot_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    response_genes = {
        "Cxcl12-Cxcr4": ["Cxcr4", "Gfap", "Aqp4", "Slc1a2", "Pdgfra"],
        "Nrg1-Erbb4": ["Erbb4", "Gad1", "Gad2"],
        "Vegfa-Flt1": ["Flt1", "Kdr", "Cldn5", "Pecam1"],
        "Reln-Lrp8": ["Lrp8", "Vldlr", "Reln", "Calb1", "Prox1", "Tbr1", "Slc17a7"],
        "Sema3a-Nrp2": ["Nrp2", "Nrp1", "Pdgfra"],
        "Fgf2-Fgfr1": ["Fgfr1", "Pdgfra", "Gfap", "Aqp4"],
    }
    annotated = cells[cells["cell_type"] != "unannotated"][
        ["cell_id", "x", "y", "cell_type", "niche"]
    ].copy()
    expr = annotated.merge(expression, on="cell_id", how="left")
    available = set(expr.columns)
    response_tables: dict[str, pd.DataFrame] = {}
    rows = []
    for lr_pair, hot in hotspot_tables.items():
        genes = [gene for gene in response_genes.get(lr_pair, []) if gene in available]
        if not genes:
            genes = [lr_pair.split("-")[1]]
        values = expr[genes].fillna(0.0).astype(float)
        z = (values - values.mean(axis=0)) / (values.std(axis=0, ddof=0) + 1e-9)
        table = expr[["cell_id", "x", "y", "cell_type", "niche"]].copy()
        table = table.merge(hot[["cell_id", "hotspot_scaled"]], on="cell_id", how="left")
        table["hotspot_scaled"] = table["hotspot_scaled"].fillna(0.0)
        table["response_score"] = z.mean(axis=1)
        lo, hi = np.nanpercentile(table["response_score"], [1, 99])
        table["response_scaled"] = np.clip((table["response_score"] - lo) / (hi - lo + 1e-9), 0, 1)
        active_ids = set(hot.loc[hot["hotspot_scaled"] > 0, "cell_id"])
        table["predicted_receiver"] = table["cell_id"].isin(active_ids)
        response_tables[lr_pair] = table

        active = table["predicted_receiver"]
        mean_active = float(table.loc[active, "response_score"].mean()) if active.any() else np.nan
        mean_bg = float(table.loc[~active, "response_score"].mean()) if (~active).any() else np.nan
        rows.append(
            {
                "lr_pair": lr_pair,
                "signature": ", ".join(genes),
                "n_hot_receivers": int(active.sum()),
                "mean_hotspot_response": mean_active,
                "mean_background_response": mean_bg,
                "response_delta_z": mean_active - mean_bg,
            }
        )
    return response_tables, pd.DataFrame(rows)


def plot_response_enrichment(ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = summary.copy()
    table["lr_pair"] = pd.Categorical(
        table["lr_pair"],
        categories=["Cxcl12-Cxcr4", "Nrg1-Erbb4", "Vegfa-Flt1"],
        ordered=True,
    )
    table = table.sort_values("lr_pair")
    y = np.arange(len(table))
    bg = table["mean_background_response"].to_numpy(dtype=float)
    hot = table["mean_hotspot_response"].to_numpy(dtype=float)
    ax.hlines(y, bg, hot, color="#B8B0A6", linewidth=1.2, zorder=1)
    ax.scatter(bg, y, s=28, color="#D5CFC5", edgecolors="white", linewidths=0.5, zorder=2, label="background")
    ax.scatter(hot, y, s=34, color=ACCENT, edgecolors="white", linewidths=0.5, zorder=3, label="predicted receiver")
    ax.set_yticks(y)
    ax.set_yticklabels([str(v).replace("-", "-\n") for v in table["lr_pair"]], fontsize=5.9)
    xmin = min(-0.12, float(np.nanmin([bg.min(), hot.min()])) - 0.08)
    xmax = max(0.35, float(np.nanmax([bg.max(), hot.max()])) + 0.12)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("response signature (z-score)", fontsize=6.8)
    ax.set_title("response support", fontsize=7.2, pad=3.5, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="lower right",
        fontsize=5.2,
        handletextpad=0.18,
        borderaxespad=0.15,
        labelspacing=0.2,
        markerscale=0.9,
    )
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.55)


def plot_response_map(ax: plt.Axes, table: pd.DataFrame, lr_pair: str) -> None:
    ax.set_aspect("equal")
    ax.scatter(table["x"], table["y"], s=3.2, c="#D7D1C7", alpha=0.42, linewidths=0, rasterized=True)
    active = table[table["response_scaled"] >= table["response_scaled"].quantile(0.82)].copy()
    active = active.sort_values("response_scaled")
    cmap = LinearSegmentedColormap.from_list("response", [BG, "#E6C06A", WARM])
    ax.scatter(
        active["x"],
        active["y"],
        s=7.6,
        c=active["response_scaled"],
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        alpha=0.94,
        rasterized=True,
    )
    predicted = table[table["predicted_receiver"]].copy()
    if not predicted.empty:
        predicted = predicted.sort_values("hotspot_scaled", ascending=False).head(220)
        ax.scatter(
            predicted["x"],
            predicted["y"],
            s=8.0,
            facecolors="none",
            edgecolors=ACCENT,
            linewidths=0.24,
            alpha=0.45,
            rasterized=True,
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{lr_pair} response", fontsize=7.4, pad=3.5)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def pretty_cell(label: str) -> str:
    return label.replace("_neuron", "").replace("_", " ")


def clean_name(name: str) -> str:
    return name.replace("-", "_").replace("/", "_")


if __name__ == "__main__":
    main()
