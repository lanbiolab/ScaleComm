#!/usr/bin/env python
"""Plot lymph node communication modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/lymph_node_phase6_expanded"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

PROGRAM_COLORS = {
    "chemokine recruitment": "#4C78A8",
    "contact-dependent immune signaling": "#72B7B2",
    "inflammatory/paracrine remodeling": "#59A14F",
    "background LR-like decoy": "#B8B8B8",
    "high-expression negative": "#E15759",
}
METHOD_COLORS = {
    "spatialself": "#2F6B4F",
    "commot": "#456C9B",
    "cellchat": "#7D668E",
    "cellnest": "#B07A4B",
    "spatalk": "#6E6E6E",
}
TEXT = "#202020"
GRID = "#D8D8D8"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(RESULT_DIR / "lymph_node_modules_lr_assignments.csv")
    summary = pd.read_csv(RESULT_DIR / "lymph_node_modules_summary.csv")
    activity = pd.read_csv(RESULT_DIR / "lymph_node_modules_activity_by_niche.csv")
    similarity = pd.read_csv(RESULT_DIR / "lymph_node_modules_lr_similarity.csv")
    enrichment = pd.read_csv(RESULT_DIR / "lymph_node_modules_mechanism_enrichment.csv")

    source = build_source(assignments, summary, activity, similarity, enrichment)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"lymph_node_modules_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.55), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 1.0, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.885,
        bottom=0.12,
        wspace=0.48,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])

    plot_similarity_matrix(ax_a, source["spatialself_similarity"], source["spatialself_assignments"])
    plot_module_composition(ax_b, source["spatialself_composition"])
    plot_module_scores(ax_c, source["module_scores"])
    plot_module_activity(ax_d, source["module_activity"])

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Expanded lymph node LR profiles resolve communication programs",
        x=0.075,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.928,
        "SpatialSelf clusters 25 LR profiles into follicular chemokine, T-zone/contact, paracrine and background/high-expression modules.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "Modules are derived from receiver incoming activity profiles in an expanded benchmark panel; this is broader than the CRC 7-LR proof-of-concept but not a whole-database atlas.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_lymph_node_communication_modules")
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_source(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    activity: pd.DataFrame,
    similarity: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    spa_assign = assignments[assignments["method"].eq("spatialself")].copy()
    spa_sim = similarity[similarity["method"].eq("spatialself")].copy()
    order = (
        spa_sim[["lr_pair_a", "cluster_order_a"]]
        .drop_duplicates()
        .sort_values("cluster_order_a")["lr_pair_a"]
        .tolist()
    )
    spa_assign["cluster_order"] = spa_assign["lr_id"].map({lr: i for i, lr in enumerate(order)})
    spa_assign = spa_assign.sort_values("cluster_order")
    comp = (
        spa_assign.groupby(["module", "program_prior"], as_index=False)
        .size()
        .rename(columns={"size": "n_lr"})
    )
    module_scores = summary[summary["method"].eq("spatialself")].copy()
    module_scores = module_scores.sort_values("module")
    act = activity[
        activity["method"].eq("spatialself")
        & activity["module"].isin(module_scores["module"])
        & activity["niche"].isin(["B follicle", "Germinal center", "T cell zone", "Mixed"])
    ].copy()
    act = act.groupby(["module", "niche"], as_index=False)["mean_module_activity"].mean()
    return {
        "spatialself_assignments": spa_assign,
        "spatialself_similarity": spa_sim,
        "spatialself_composition": comp,
        "module_scores": module_scores,
        "module_activity": act,
        "mechanism_enrichment": enrichment[enrichment["method"].eq("spatialself")].copy(),
    }


def plot_similarity_matrix(ax: plt.Axes, similarity: pd.DataFrame, assignments: pd.DataFrame) -> None:
    order = assignments["lr_id"].tolist()
    matrix = similarity.pivot_table(index="lr_pair_a", columns="lr_pair_b", values="spearman_similarity").reindex(index=order, columns=order).fillna(0)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [short_lr(x) for x in order]
    ax.set_title("SpatialSelf LR activity similarity", loc="left", pad=4)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=4.2)
    for i, row in enumerate(assignments.itertuples()):
        ax.add_patch(plt.Rectangle((-0.9, i - 0.5), 0.28, 1.0, color=PROGRAM_COLORS.get(row.program_prior, "#B8B8B8"), clip_on=False, linewidth=0))
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Spearman", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=2)


def plot_module_composition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    modules = sorted(comp["module"].unique())
    left = np.zeros(len(modules))
    for prior, color in PROGRAM_COLORS.items():
        vals = comp[comp["program_prior"].eq(prior)].set_index("module").reindex(modules)["n_lr"].fillna(0).to_numpy()
        ax.barh(modules, vals, left=left, color=color, edgecolor="white", linewidth=0.4, label=prior)
        left += vals
    ax.set_title("Module composition", loc="left", pad=4)
    ax.set_xlabel("LR pairs")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=4.8, handlelength=0.8)


def plot_module_scores(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.sort_values("mean_downstream_delta_r2", ascending=True)
    y = np.arange(len(plot))
    colors = [PROGRAM_COLORS.get(x, "#B8B8B8") for x in plot["top_enriched_program_prior"]]
    ax.barh(y, plot["mean_downstream_delta_r2"].fillna(0), color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["module"], fontsize=6)
    ax.set_title("Module target impact", loc="left", pad=4)
    ax.set_xlabel("mean delta R2")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    for yi, row in zip(y, plot.itertuples()):
        label = row.top_enriched_program_prior.replace("high-expression ", "high-exp. ")
        ax.text(max(row.mean_downstream_delta_r2 if np.isfinite(row.mean_downstream_delta_r2) else 0, 0) + 0.002, yi, label, va="center", fontsize=5.2, color="#555555")


def plot_module_activity(ax: plt.Axes, table: pd.DataFrame) -> None:
    modules = sorted(table["module"].unique())
    niches = ["B follicle", "Germinal center", "T cell zone", "Mixed"]
    matrix = table.pivot_table(index="module", columns="niche", values="mean_module_activity").reindex(index=modules, columns=niches).fillna(0)
    arr = matrix.to_numpy(dtype=float)
    arr = arr / np.maximum(arr.max(axis=1, keepdims=True), 1e-9)
    im = ax.imshow(arr, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Niche activity patterns", loc="left", pad=4)
    ax.set_xticks(np.arange(len(niches)))
    ax.set_xticklabels([short_niche(x) for x in niches], rotation=25, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels(modules, fontsize=6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center", fontsize=5.1, color="#202020")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("row-scaled", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=2)


def short_lr(lr_id: str) -> str:
    parts = lr_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return lr_id


def short_niche(value: str) -> str:
    return value.replace("Germinal center", "GC").replace("B follicle", "B foll.").replace("T cell zone", "T zone")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
