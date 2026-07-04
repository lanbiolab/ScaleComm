#!/usr/bin/env python
"""Plot native lymph node full-transcriptome COMMOT-style validation panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

METHOD_ORDER = ["spatialself", "commot"]
METHOD_LABEL = {"spatialself": "SpatialSelf", "commot": "COMMOT"}
METHOD_COLORS = {"spatialself": "#2F6B4F", "commot": "#456C9B"}
NICHE_COLORS = {
    "T_cell_zone_proxy": "#5B8CC0",
    "B_follicle_proxy": "#D8B365",
    "germinal_center_proxy": "#8CB369",
    "stromal_follicle_proxy": "#9C8FBC",
    "myeloid_proxy": "#B77B9B",
    "unknown": "#B8B8B8",
}
PROGRAM_COLORS = {
    "chemokine recruitment": "#4C78A8",
    "contact-dependent immune signaling": "#72B7B2",
    "high-expression negative": "#D55E5E",
}
TEXT = "#202020"
SUBTLE = "#666666"
GRID = "#D9D9D9"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    vectors = pd.read_csv(RESULT_DIR / "lymph_node_full_directionality_receiver_vectors.csv")
    flow_summary = pd.read_csv(RESULT_DIR / "lymph_node_full_directionality_lr_summary.csv")
    impact = pd.read_csv(RESULT_DIR / "lymph_node_full_lr_level_impact_summary.csv")
    modules = pd.read_csv(RESULT_DIR / "lymph_node_full_modules_summary.csv")
    assignments = pd.read_csv(RESULT_DIR / "lymph_node_full_modules_lr_assignments.csv")

    source = build_source(cells, vectors, flow_summary, impact, modules, assignments)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"lymph_node_full_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.4, 5.95), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.35, 0.95, 1.05],
        height_ratios=[1.0, 0.92, 0.98],
        left=0.07,
        right=0.985,
        top=0.885,
        bottom=0.105,
        wspace=0.42,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[2, 1:])

    plot_spatial_flow(ax_a, source["spatial_flow"])
    plot_delta_r2(ax_b, source["impact_focus"])
    plot_expected_flow(ax_c, source["expected_flow"])
    plot_negative_gap(ax_d, source["method_gap"])
    plot_module_composition(ax_e, source["module_composition"])
    plot_module_table(ax_f, source["module_summary"])

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Native lymph node Visium validates CCR7 flow while exposing receptor-shared false positives",
        x=0.07,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.07,
        0.928,
        "A first-pass 36,601-gene human lymph node Visium analysis supports CCL19/CCL21-CCR7 target response and T-zone-local flow; CCR7 high-expression negatives remain competitive.",
        fontsize=6.3,
        color=SUBTLE,
    )
    fig.text(
        0.07,
        0.035,
        "Niches are marker-rule proxies; directionality is inferred from weighted sender-to-receiver candidate edges and is not causal transport. Delta R2 uses target expression residualized by library size and spatial coordinates.",
        fontsize=5.8,
        color=SUBTLE,
    )
    save_figure(fig, FIG_DIR / "phase11_lymph_node_full_commot_style_story")
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
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.frameon": False,
        }
    )


def build_source(
    cells: pd.DataFrame,
    vectors: pd.DataFrame,
    flow_summary: pd.DataFrame,
    impact: pd.DataFrame,
    modules: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    spatial = cells[["cell_id", "x", "y", "niche", "cell_type"]].copy()
    spatial["panel_role"] = "background"
    flow = vectors[
        vectors["method"].isin(["spatialself", "commot"])
        & vectors["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7"])
        & vectors["incoming_score"].gt(0)
    ].copy()
    flow["rank"] = flow.groupby(["method", "lr_pair"])["incoming_score"].rank(ascending=False, method="first")
    flow = flow[flow["rank"].le(260)].copy()

    focus_lr = [
        "ln_full_lr_000_CCL19_CCR7",
        "ln_full_lr_001_CCL21_CCR7",
        "ln_full_lr_002_CXCL13_CXCR5",
        "ln_full_lr_008_EEF1A1_CCR7_negative",
    ]
    impact_focus = impact[impact["lr_id"].isin(focus_lr)].copy()
    impact_focus["lr_label"] = impact_focus["ligand"].astype(str) + "-" + impact_focus["receptor"].astype(str)
    impact_focus.loc[impact_focus["false_positive_high_expression"].astype(bool), "lr_label"] += " neg."

    expected_flow = flow_summary[
        flow_summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4"])
    ].copy()

    method_gap = (
        impact.groupby("method", as_index=False)
        .agg(
            mean_priority_delta_r2=("mean_delta_r2", lambda s: s[~impact.loc[s.index, "false_positive_high_expression"].astype(bool)].mean()),
            max_negative_delta_r2=("mean_delta_r2", lambda s: s[impact.loc[s.index, "false_positive_high_expression"].astype(bool)].max()),
        )
    )
    method_gap["priority_minus_max_negative"] = method_gap["mean_priority_delta_r2"] - method_gap["max_negative_delta_r2"]

    spa_modules = modules[modules["method"].eq("spatialself")].copy()
    spa_assign = assignments[assignments["method"].eq("spatialself")].copy()
    comp = (
        spa_assign.groupby(["module", "program_prior"], as_index=False)
        .size()
        .rename(columns={"size": "n_lr"})
    )
    return {
        "spatial_background": spatial,
        "spatial_flow": flow.merge(spatial, left_on="receiver_id", right_on="cell_id", how="left", suffixes=("", "_cell")),
        "impact_focus": impact_focus,
        "expected_flow": expected_flow,
        "method_gap": method_gap,
        "module_composition": comp,
        "module_summary": spa_modules,
    }


def plot_spatial_flow(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("CCR7 receiver flow in native lymph node", loc="left", pad=4)
    bg = table.drop_duplicates("receiver_id")
    for niche, color in NICHE_COLORS.items():
        sub = bg[bg["receiver_niche"].eq(niche)]
        if not sub.empty:
            ax.scatter(sub["receiver_x"], -sub["receiver_y"], s=5.5, color=color, alpha=0.28, linewidths=0, label=clean_niche(niche))
    for method in ["spatialself", "commot"]:
        sub = table[(table["method"].eq(method)) & (table["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7"]))].copy()
        sub = sub.sort_values("incoming_score", ascending=False)
        if sub.empty:
            continue
        step = max(len(sub) // 80, 1)
        q = sub.iloc[::step]
        ax.quiver(
            q["receiver_x"],
            -q["receiver_y"],
            q["vector_dx"] * 0.42,
            -q["vector_dy"] * 0.42,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002 if method == "spatialself" else 0.0016,
            color=METHOD_COLORS[method],
            alpha=0.48 if method == "spatialself" else 0.42,
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.legend(loc="lower left", fontsize=4.9, markerscale=1.1, handletextpad=0.2, borderpad=0.1)
    ax.text(
        0.02,
        0.98,
        "SpatialSelf and COMMOT arrows both concentrate\nwithin proxy T-cell-zone CCR7 receiver fields",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=SUBTLE,
    )


def plot_delta_r2(ax: plt.Axes, table: pd.DataFrame) -> None:
    order = ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "EEF1A1-CCR7 neg."]
    x = np.arange(len(order))
    width = 0.34
    for i, method in enumerate(METHOD_ORDER):
        vals = table[table["method"].eq(method)].set_index("lr_label").reindex(order)["mean_delta_r2"].fillna(0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABEL[method])
    ax.set_title("Receiver target impact", loc="left", pad=4)
    ax.set_ylabel("mean delta R2")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5.6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(loc="upper left", fontsize=5.2)


def plot_expected_flow(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4"]
    x = np.arange(len(pairs))
    width = 0.34
    for i, method in enumerate(METHOD_ORDER):
        vals = table[table["method"].eq(method)].set_index("lr_pair").reindex(pairs)["expected_flow_score_fraction"].fillna(0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=METHOD_COLORS[method], edgecolor="none")
    ax.set_title("Expected niche-flow support", loc="left", pad=4)
    ax.set_ylabel("score fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=5.6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.axhline(0.5, color="#999999", linewidth=0.55, linestyle=":")


def plot_negative_gap(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.set_index("method").reindex(METHOD_ORDER).reset_index()
    y = np.arange(len(plot))
    ax.barh(y, plot["mean_priority_delta_r2"], color="#C9D8CF", edgecolor="none", label="positive mean")
    ax.scatter(plot["max_negative_delta_r2"], y, color="#D55E5E", s=28, zorder=3, label="max negative")
    for yi, row in zip(y, plot.itertuples()):
        ax.plot([row.mean_priority_delta_r2, row.max_negative_delta_r2], [yi, yi], color="#777777", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[m] for m in plot["method"]], fontsize=6)
    ax.set_title("Negative-control stress", loc="left", pad=4)
    ax.set_xlabel("mean delta R2")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.1)


def plot_module_composition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    modules = sorted(comp["module"].unique())
    left = np.zeros(len(modules))
    for prior, color in PROGRAM_COLORS.items():
        vals = comp[comp["program_prior"].eq(prior)].set_index("module").reindex(modules)["n_lr"].fillna(0).to_numpy()
        ax.barh(modules, vals, left=left, color=color, edgecolor="white", linewidth=0.35, label=prior)
        left += vals
    ax.set_title("SpatialSelf modules", loc="left", pad=4)
    ax.set_xlabel("LR pairs")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=4.9, handlelength=0.8)


def plot_module_table(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.axis("off")
    plot = table[table["method"].eq("spatialself")].copy().sort_values("module")
    rows = []
    for row in plot.itertuples():
        lr_pairs = str(row.lr_pairs).replace(";", "\n")
        rows.append(
            [
                row.module,
                lr_pairs,
                int(row.n_high_expression_negative),
                f"{row.mean_downstream_delta_r2:.3f}" if np.isfinite(row.mean_downstream_delta_r2) else "NA",
                f"{row.mean_expected_flow_fraction:.2f}" if np.isfinite(row.mean_expected_flow_fraction) else "NA",
            ]
        )
    tbl = ax.table(
        cellText=rows,
        colLabels=["Module", "LR pairs", "Neg.", "delta R2", "flow"],
        colWidths=[0.12, 0.46, 0.11, 0.15, 0.12],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.3)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#D6D6D6")
        if r == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold", color=TEXT)
        elif c == 2 and rows[r - 1][2] > 0:
            cell.set_facecolor("#F4D7D7")
        else:
            cell.set_facecolor("white")
    ax.set_title("SpatialSelf module interpretation", loc="left", pad=6)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def clean_niche(value: str) -> str:
    return (
        str(value)
        .replace("_proxy", "")
        .replace("_", " ")
        .replace("T cell zone", "T-cell zone")
        .replace("B follicle", "B follicle")
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
