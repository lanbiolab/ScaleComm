#!/usr/bin/env python
"""Plot Phase 10 full-transcriptome downstream validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
RESULT_DIR: Path
CELL_PATH: Path
FIG_DIR: Path
METHOD_ORDER: list[str]
DATASET_LABEL: str
DATASET_NOTE: str

BASE_METHOD_ORDER = ["spatialself", "commot", "cellchat", "cellnest", "spatalk"]
METHOD_LABEL = {
    "spatialself": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "cellnest": "CellNEST",
    "spatalk": "SpaTalk",
}
METHOD_COLORS = {
    "spatialself": "#2F6B4F",
    "commot": "#3F6C99",
    "cellchat": "#7A5C8E",
    "cellnest": "#B57945",
    "spatalk": "#8A8A8A",
}
NEGATIVE_COLOR = "#B9B9B9"
TEXT = "#202020"
GRID = "#D8D8D8"


def configure_dataset(dataset: str) -> None:
    global RESULT_DIR
    global CELL_PATH
    global FIG_DIR
    global METHOD_ORDER
    global DATASET_LABEL
    global DATASET_NOTE

    if dataset == "post_xenium":
        slug = "post_xenium_human_colon_cancer_subset"
        standardized = "post_xenium_human_colon_cancer_subset"
        DATASET_LABEL = "Post-Xenium human colon cancer Visium subset"
        DATASET_NOTE = (
            "SpatialSelf shows target enrichment and target-rank specificity over high-expression negative controls; "
            "communication-impact support is partial."
        )
        METHOD_ORDER = BASE_METHOD_ORDER.copy()
    elif dataset == "visium_hd_crc":
        slug = "visium_hd_crc_p2crc_subset"
        standardized = "visium_hd_crc_p2crc_subset"
        DATASET_LABEL = "Visium HD CRC P2CRC 8 um bin subset"
        DATASET_NOTE = (
            "SpatialSelf shows SPP1-CD44 target enrichment, target-rank specificity and communication-impact support; "
            "baselines show enrichment with stronger negative-control risk."
        )
        METHOD_ORDER = ["spatialself", "commot", "cellchat", "cellnest"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    RESULT_DIR = PROJECT_ROOT / f"outputs/phase10/full_transcriptome_commot_style/{slug}/results"
    CELL_PATH = GOAL5_DIR / f"standardized_inputs/{standardized}/cells.csv"
    FIG_DIR = PROJECT_ROOT / f"outputs/phase10/full_transcriptome_commot_style/{slug}/figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["post_xenium", "visium_hd_crc"], default="post_xenium")
    args = parser.parse_args()

    configure_dataset(args.dataset)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    cells = pd.read_csv(CELL_PATH)
    incoming = pd.read_csv(RESULT_DIR / "receiver_level_incoming_scores.csv")
    scan = pd.read_csv(RESULT_DIR / "gene_association_top300_by_method_lr.csv")
    enrichment = pd.read_csv(RESULT_DIR / "target_enrichment_summary.csv")
    specificity = pd.read_csv(RESULT_DIR / "downstream_specificity_summary.csv")
    impact = pd.read_csv(RESULT_DIR / "communication_impact_summary.csv")
    story = pd.read_csv(RESULT_DIR / "phase10_story_summary.csv")

    source = build_source_tables(cells, incoming, scan, enrichment, specificity, impact, story)
    write_source_tables(source)

    fig = plt.figure(figsize=(7.15, 6.35), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.05, 1.25],
        height_ratios=[1.0, 0.92, 0.68],
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.12,
        wspace=0.34,
        hspace=0.62,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_spatial_map(ax_a, source["spatial_map"])
    plot_gene_rank(ax_b, source["gene_rank"])
    plot_target_enrichment(ax_c, source["target_enrichment"])
    plot_specificity(ax_d, source["specificity"])
    plot_impact(ax_e, source["impact"])

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d, "d")
    add_panel_label(ax_e, "e")

    fig.suptitle(
        "Full-transcriptome validation of SPP1-CD44 receiver response",
        x=0.075,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.94,
        DATASET_NOTE,
        fontsize=6.2,
        color="#555555",
    )
    fig.text(
        0.075,
        0.025,
        f"{DATASET_LABEL}; associations are observational. "
        "Negative controls are high-expression LR-like pairs.",
        fontsize=6,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase10_full_transcriptome_downstream")
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
            "axes.linewidth": 0.7,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_source_tables(
    cells: pd.DataFrame,
    incoming: pd.DataFrame,
    scan: pd.DataFrame,
    enrichment: pd.DataFrame,
    specificity: pd.DataFrame,
    impact: pd.DataFrame,
    story: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    spatial = incoming[incoming["method"].eq("spatialself")][["cell_id", "SPP1-CD44"]].merge(
        cells[["cell_id", "x", "y", "region", "cell_type"]], on="cell_id", how="left"
    )
    spatial = spatial.rename(columns={"SPP1-CD44": "incoming_score"})

    gene_rank = scan[(scan["method"].eq("spatialself")) & (scan["lr_pair"].eq("SPP1-CD44"))].copy()
    gene_rank = gene_rank.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).reset_index(drop=True)
    targets = {"LYZ", "CD68", "CD14", "C1QA", "C1QB", "C1QC", "LST1", "FCGR3A", "MSR1", "MRC1", "COL1A1", "COL1A2", "FN1", "LGALS3", "CTSS", "APOE", "TREM2"}
    gene_rank["rank"] = np.arange(1, len(gene_rank) + 1)
    gene_rank["is_curated_target"] = gene_rank["gene"].isin(targets)
    top_genes = gene_rank.head(14)
    top_targets = gene_rank[gene_rank["is_curated_target"]].head(6)
    gene_rank = (
        pd.concat([top_genes, top_targets], ignore_index=True)
        .drop_duplicates("gene")
        .sort_values("rank")
        .head(20)
    )

    target = enrichment[
        enrichment["lr_pair"].isin(["SPP1-CD44", "ACTB-KRT8", "COL1A1-EPCAM"])
        & enrichment["target_source"].eq("curated")
    ].copy()
    target["method_label"] = target["method"].map(METHOD_LABEL)
    target["pair_class"] = np.where(target["lr_pair"].eq("SPP1-CD44"), "SPP1-CD44", "high-expression negative")
    target_multi_source = enrichment[enrichment["lr_pair"].isin(["SPP1-CD44", "ACTB-KRT8", "COL1A1-EPCAM"])].copy()
    target_multi_source["method_label"] = target_multi_source["method"].map(METHOD_LABEL)

    spec = story[story["method"].isin(METHOD_ORDER)].copy()
    spec["method_label"] = spec["method"].map(METHOD_LABEL)
    spec = spec.merge(
        specificity[specificity["positive_lr"].eq("SPP1-CD44")][
            ["method", "best_negative_enrichment_lr", "best_negative_rank_lr", "best_negative_impact_lr"]
        ],
        on="method",
        how="left",
    )
    spec = spec.merge(
        impact[impact["lr_pair"].eq("SPP1-CD44")][["method", "rf_comm_importance_rank"]],
        on="method",
        how="left",
    )
    impact_plot = build_impact_source(impact)
    return {
        "spatial_map": spatial,
        "gene_rank": gene_rank,
        "target_enrichment": target,
        "target_enrichment_all_sources": target_multi_source,
        "specificity": spec,
        "impact": impact_plot,
    }


def build_impact_source(impact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        method_table = impact[impact["method"].eq(method)]
        spp1 = method_table[method_table["lr_pair"].eq("SPP1-CD44")]
        neg = (
            method_table[method_table["lr_status"].eq("negative_control")]
            .sort_values("delta_cv_r2", ascending=False)
            .head(1)
        )
        if not spp1.empty:
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "pair_class": "SPP1-CD44",
                    "lr_pair": "SPP1-CD44",
                    "delta_cv_r2": spp1.iloc[0]["delta_cv_r2"],
                    "rf_comm_importance": spp1.iloc[0]["rf_comm_importance"],
                }
            )
        if not neg.empty:
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "pair_class": "best high-expression negative",
                    "lr_pair": neg.iloc[0]["lr_pair"],
                    "delta_cv_r2": neg.iloc[0]["delta_cv_r2"],
                    "rf_comm_importance": neg.iloc[0]["rf_comm_importance"],
                }
            )
    return pd.DataFrame(rows)


def write_source_tables(source: dict[str, pd.DataFrame]) -> None:
    source_dir = FIG_DIR / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, table in source.items():
        table.to_csv(source_dir / f"{name}.csv", index=False)


def plot_spatial_map(ax: plt.Axes, table: pd.DataFrame) -> None:
    score = table["incoming_score"].to_numpy(dtype=float)
    vmax = np.nanquantile(score, 0.98)
    sc = ax.scatter(
        table["x"],
        -table["y"],
        c=score,
        s=4.8,
        cmap="YlGn",
        vmin=0,
        vmax=max(vmax, 1e-9),
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("SpatialSelf SPP1-CD44 incoming", loc="left", pad=4)
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("incoming score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2)


def plot_gene_rank(ax: plt.Axes, table: pd.DataFrame) -> None:
    table = table.sort_values("spearman_r", ascending=True)
    y = np.arange(len(table))
    colors = np.where(table["is_curated_target"], METHOD_COLORS["spatialself"], "#C7C7C7")
    ax.barh(y, table["spearman_r"], color=colors, height=0.72, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(table["gene"], fontsize=5.6)
    for tick, is_target in zip(ax.get_yticklabels(), table["is_curated_target"]):
        tick.set_color(METHOD_COLORS["spatialself"] if is_target else "#555555")
        tick.set_fontweight("bold" if is_target else "normal")
    ax.set_title("Receiver genes associated with SPP1-CD44", loc="left", pad=4)
    ax.set_xlabel("Spearman r")
    ax.set_xlim(0, max(0.34, table["spearman_r"].max() + 0.035))
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.text(
        0.98,
        0.04,
        "green labels: curated targets",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#555555",
        ha="right",
    )


def plot_target_enrichment(ax: plt.Axes, table: pd.DataFrame) -> None:
    table = table[table["method"].isin(METHOD_ORDER)].copy()
    table["method"] = pd.Categorical(table["method"], METHOD_ORDER, ordered=True)
    table = table.sort_values(["method", "lr_pair"])
    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    spp1 = table[table["lr_pair"].eq("SPP1-CD44")].set_index("method").reindex(METHOD_ORDER)
    neg = (
        table[~table["lr_pair"].eq("SPP1-CD44")]
        .sort_values("target_enrichment_fold", ascending=False)
        .groupby("method", as_index=False, observed=False)
        .head(1)
        .set_index("method")
        .reindex(METHOD_ORDER)
    )
    ax.bar(x - width / 2, spp1["target_enrichment_fold"], width, color=[METHOD_COLORS[m] for m in METHOD_ORDER], edgecolor="none", label="SPP1-CD44")
    ax.bar(x + width / 2, neg["target_enrichment_fold"], width, color=NEGATIVE_COLOR, edgecolor="none", label="best high-expression negative")
    for idx, method in enumerate(METHOD_ORDER):
        p = spp1.loc[method, "target_enrichment_p"] if method in spp1.index else np.nan
        if pd.notna(p) and p <= 0.05:
            ax.text(idx - width / 2, spp1.loc[method, "target_enrichment_fold"] + 1.5, "*", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_title("Known-target enrichment", loc="left", pad=4)
    ax.set_ylabel("target enrichment")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=28, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.4)


def plot_specificity(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table[table["method"].isin(METHOD_ORDER)].copy()
    plot["method"] = pd.Categorical(plot["method"], METHOD_ORDER, ordered=True)
    plot = plot.sort_values("method")
    x = np.arange(len(plot))
    colors = [METHOD_COLORS.get(m, "#777777") for m in plot["method"]]
    ax.axhline(0, color="#646464", linewidth=0.7)
    ax.bar(
        x,
        plot["spp1_cd44_target_rank_specificity"],
        color=colors,
        edgecolor="none",
        width=0.62,
    )
    for idx, row in plot.reset_index(drop=True).iterrows():
        label = support_label(row["support_class"])
        va = "bottom" if row["spp1_cd44_target_rank_specificity"] >= 0 else "top"
        offset = 260 if va == "bottom" else -260
        if pd.notna(row["spp1_cd44_target_rank_specificity"]):
            ax.text(idx, row["spp1_cd44_target_rank_specificity"] + offset, label, ha="center", va=va, fontsize=5.6, color=TEXT)
    ax.set_title("Specificity over high-expression controls", loc="left", pad=4)
    ax.set_ylabel("target-rank specificity")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["method_label"], rotation=28, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def support_label(support_class: str) -> str:
    if support_class == "strong_specific_downstream_support":
        return "strong"
    if support_class == "partial_specific_downstream_support":
        return "partial"
    if support_class == "target_enrichment_with_negative_control_risk":
        return "risk"
    if support_class == "limited_full_transcriptome_downstream_support":
        return "n/e"
    return "mixed"


def plot_impact(ax: plt.Axes, table: pd.DataFrame) -> None:
    table = table[table["method"].isin(METHOD_ORDER)].copy()
    table["method"] = pd.Categorical(table["method"], METHOD_ORDER, ordered=True)
    spp1 = table[table["pair_class"].eq("SPP1-CD44")].set_index("method").reindex(METHOD_ORDER)
    neg = table[table["pair_class"].eq("best high-expression negative")].set_index("method").reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    ax.axhline(0, color="#646464", linewidth=0.7)
    ax.bar(
        x - width / 2,
        spp1["delta_cv_r2"],
        width,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="none",
        label="SPP1-CD44",
    )
    ax.bar(
        x + width / 2,
        neg["delta_cv_r2"],
        width,
        color=NEGATIVE_COLOR,
        edgecolor="none",
        label="best high-expression negative",
    )
    ax.set_title("Communication impact on curated receiver targets", loc="left", pad=4)
    ax.set_ylabel("delta CV R2")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=18, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(loc="upper left", ncol=2, fontsize=5.6, handlelength=1.4)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color=TEXT,
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
