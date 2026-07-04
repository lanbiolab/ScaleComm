#!/usr/bin/env python
"""Render Phase 12 Figure 2: post-Xenium colon SPP1-CD44 validation story."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "phase12" / "manuscript_figure_package"
OUTDIR = ROOT / "outputs" / "phase12" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


COL = {
    "spatialself": "#0F4D92",
    "commot": "#767676",
    "cellchat": "#9A9A9A",
    "spatalk": "#B5B5B5",
    "cellnest": "#5F6F8F",
    "spp1": "#2E9E44",
    "positive": "#66A8D9",
    "negative": "#B64342",
    "neutral": "#4D4D4D",
    "light": "#EDEDED",
    "grid": "#D8D8D8",
    "text": "#272727",
}


METHOD_ORDER = [
    "spatialself_calibrated",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
]

METHOD_LABEL = {
    "spatialself_calibrated": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}

METHOD_COLOR = {
    "spatialself_calibrated": COL["spatialself"],
    "commot": "#6E6E6E",
    "cellchat": "#9A9A9A",
    "spatalk": "#B8B8B8",
    "cellnest": "#7584A8",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COL["text"],
    )


def clean_label(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "rank": pd.read_csv(SOURCE / "figure2_crc_method_lr_rank_table.csv"),
        "fdr": pd.read_csv(SOURCE / "figure2_crc_local_contrast_fdr.csv"),
        "downstream": pd.read_csv(SOURCE / "figure2_crc_whole_transcriptome_downstream.csv"),
        "program": pd.read_csv(SOURCE / "figure2_crc_gene_program_specificity.csv"),
        "pathway": pd.read_csv(SOURCE / "figure2_crc_pathway_specific_programs.csv"),
        "impact": pd.read_csv(SOURCE / "figure2_crc_commot_style_impact.csv"),
    }


def draw_rank_panel(ax: plt.Axes, rank: pd.DataFrame) -> None:
    df = rank[rank["method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "rank"])

    y_lookup = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    ax.axvspan(0.5, 3.5, color="#F4F8FB", zorder=0)

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        y = y_lookup[method]
        ax.hlines(y, 0.8, 7.2, color="#E3E3E3", lw=1.0, zorder=0)
        for _, row in sub.iterrows():
            name = row["control_name"]
            x = float(row["rank"])
            if bool(row["false_positive_high_expression"]):
                marker, color, size, z = "v", COL["negative"], 42, 3
                alpha = 0.9
            elif name == "SPP1-CD44":
                marker, color, size, z = "o", COL["spp1"], 76, 5
                alpha = 1.0
            else:
                marker, color, size, z = "o", COL["positive"], 38, 2
                alpha = 0.72
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=alpha,
                zorder=z,
            )
        # Put a subtle outline on the SpatialSelf row.
        if method == "spatialself_calibrated":
            ax.hlines(y, 0.8, 7.2, color=COL["spatialself"], lw=1.1, zorder=1, alpha=0.55)

    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("LR rank within control panel (lower is better)", labelpad=3)
    ax.set_yticks([y_lookup[m] for m in METHOD_ORDER])
    ax.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_title("SPP1-CD44 is retained while hard negatives are demoted", loc="left", pad=8)
    panel_label(ax, "a")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["spp1"], markeredgecolor="white", markersize=6.5, label="SPP1-CD44"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["positive"], markeredgecolor="white", markersize=5.5, label="other positive LR"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=COL["negative"], markeredgecolor="white", markersize=6.0, label="high-expression negative"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, -0.18),
        ncol=3,
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=0.9,
    )


def draw_local_fdr_panel(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    order = ["SPP1-CD44", "CXCL12-CXCR4", "TGFB1-TGFBR2", "VEGFA-KDR", "ACTB-KRT8", "COL1A1-EPCAM"]
    pivot = (
        fdr.pivot_table(
            index="control_name",
            columns="null_model",
            values="local_contrast_bh_fdr",
            aggfunc="min",
        )
        .reindex(order)
        .rename(
            columns={
                "within_region_permutation": "region",
                "cell_type_preserving_permutation": "cell type",
                "lr_identity_permutation": "LR label",
            }
        )
    )
    logq = -np.log10(pivot.clip(lower=1e-4).values)
    im = ax.imshow(logq, cmap="Blues", vmin=0, vmax=2.2, aspect="auto")

    for i, lr in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            q = pivot.loc[lr, col]
            txt = "q<0.01" if q < 0.01 else f"{q:.2f}"
            color = "white" if logq[i, j] > 1.3 else COL["text"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.1, color=color)

    for i, lr in enumerate(pivot.index):
        if lr in {"ACTB-KRT8", "COL1A1-EPCAM"}:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(pivot.columns), 1, fill=False, edgecolor=COL["negative"], linewidth=0.9))
        elif lr == "SPP1-CD44":
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(pivot.columns), 1, fill=False, edgecolor=COL["spp1"], linewidth=1.2))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=6.8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6.8)
    ax.tick_params(length=0)
    ax.set_title("Local-contrast FDR supports spatially restricted signal", loc="left", pad=8)
    panel_label(ax, "b")

    cbar = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.025)
    cbar.set_label("-log10(q)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6, length=2)


def draw_downstream_panel(ax: plt.Axes, downstream: pd.DataFrame, program: pd.DataFrame, pathway: pd.DataFrame) -> None:
    df = downstream.copy()
    prog = program[program["true_lr"] == True].set_index("lr_pair")
    order = ["SPP1-CD44", "CXCL12-CXCR4", "TGFB1-TGFBR2", "VEGFA-KDR"]
    df = df.set_index("lr_pair").reindex(order).reset_index()
    df["specific_terms"] = df["lr_pair"].map(prog["n_specific_significant_terms_fdr_0_05"])
    df["specific_fraction"] = df["lr_pair"].map(prog["specific_fraction_of_significant"])
    path = pathway.set_index("lr_pair") if not pathway.empty else pd.DataFrame()
    if not path.empty:
        df["pathway_call"] = df["lr_pair"].map(path["pathway_specific_call"])
        df["best_specific_term"] = df["lr_pair"].map(path["best_specific_term"])
    else:
        df["pathway_call"] = ""
        df["best_specific_term"] = ""

    y = np.arange(len(df))[::-1]
    colors = [COL["spp1"] if lr == "SPP1-CD44" else "#6EA6CF" for lr in df["lr_pair"]]
    ax.barh(
        y,
        df["target_top300_enrichment"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.58,
    )
    ax.axvline(1.0, color="#8F8F8F", linestyle=(0, (3, 2)), lw=0.8)
    for yi, (_, row) in zip(y, df.iterrows()):
        q = row["target_top300_enrichment_bh_fdr"]
        star = "**" if q < 0.01 else "*" if q < 0.05 else ""
        clean = row["pathway_call"] == "clean_specific_support"
        term_count = int(row["specific_terms"]) if pd.notna(row["specific_terms"]) else 0
        label = f"{row['n_targets_top300']} targets {star}"
        if clean:
            label += f" | {term_count} clean terms"
        elif term_count:
            label += f" | {term_count} terms"
        ax.text(
            row["target_top300_enrichment"] + 0.07,
            yi,
            label,
            ha="left",
            va="center",
            fontsize=6.2,
            color=COL["text"],
        )
        if clean:
            ax.scatter(
                0.16,
                yi,
                marker="s",
                s=28,
                facecolor=COL["spp1"],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["lr_pair"], fontsize=6.8)
    ax.set_xlim(0, max(4.05, float(df["target_top300_enrichment"].max()) + 1.25))
    ax.set_xlabel("target enrichment in top 300 receiver genes", labelpad=3)
    ax.set_title("Receiver transcriptome supports\ntarget genes and ECM/stroma programs", loc="left", pad=8)
    panel_label(ax, "c")


def draw_impact_panel(ax: plt.Axes, impact: pd.DataFrame) -> None:
    df = impact.copy()
    df["label"] = df["method"].map({"spatialself": "SpatialSelf", "commot": "COMMOT"})
    df = df.set_index("method").reindex(["spatialself", "commot"]).reset_index()

    x = np.arange(len(df))
    width = 0.34
    pos = df["mean_positive_rf_importance"].astype(float).values
    neg = df["max_negative_rf_importance"].astype(float).values
    ax.bar(
        x - width / 2,
        pos,
        width=width,
        color=[COL["spatialself"], "#9A9A9A"],
        edgecolor="white",
        linewidth=0.8,
        label="positive LR mean",
    )
    ax.bar(
        x + width / 2,
        neg,
        width=width,
        color=[COL["negative"], COL["negative"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        label="worst negative control",
    )
    ax.patches[2].set_alpha(0.55)
    ax.patches[3].set_alpha(0.85)
    for xi, p, n in zip(x, pos, neg):
        ax.plot([xi - width / 2, xi + width / 2], [p, n], color="#777777", lw=0.7, alpha=0.45)
        ax.text(xi - width / 2, p + 0.025, f"{p:.2f}", ha="center", va="bottom", fontsize=6.0)
        ax.text(xi + width / 2, n + 0.025, f"{n:.2f}", ha="center", va="bottom", fontsize=6.0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7)
    ax.set_ylabel("RF importance in receiver-gene model", labelpad=3)
    ax.set_ylim(0, 1.08)
    ax.set_title("Whole-transcriptome impact audit\nflags negative leakage", loc="left", pad=8)
    panel_label(ax, "d")
    ax.legend(loc="upper left", fontsize=6.2)
    ax.text(
        0.02,
        -0.26,
        "Interpretation: observational association, not perturbation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#666666",
    )


def render() -> None:
    apply_style()
    data = load_data()

    fig = plt.figure(figsize=(7.2, 6.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.28, 1.0],
        left=0.085,
        right=0.985,
        top=0.845,
        bottom=0.12,
        wspace=0.42,
        hspace=0.7,
    )

    fig.text(
        0.085,
        0.982,
        "SPP1-CD44 is prioritized in post-Xenium colon with spatial and transcriptome support",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COL["text"],
    )
    fig.text(
        0.085,
        0.948,
        "Control-panel ranking, local permutation contrast and receiver-gene audits are shown; claims remain non-causal.",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#5A5A5A",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_rank_panel(ax_a, data["rank"])
    draw_local_fdr_panel(ax_b, data["fdr"])
    draw_downstream_panel(ax_c, data["downstream"], data["program"], data["pathway"])
    draw_impact_panel(ax_d, data["impact"])

    for ax in [ax_a, ax_c, ax_d]:
        ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
        ax.set_axisbelow(True)

    basename = OUTDIR / "phase12_figure2_crc_spp1_cd44_story"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = pd.DataFrame(
        [
            {
                "figure": "Figure 2",
                "output_stem": str(basename),
                "panel": "a",
                "source_data": "figure2_crc_method_lr_rank_table.csv",
                "message": "SpatialSelf retains SPP1-CD44 near the top while calibrated hard negatives rank below positives.",
            },
            {
                "figure": "Figure 2",
                "output_stem": str(basename),
                "panel": "b",
                "source_data": "figure2_crc_local_contrast_fdr.csv",
                "message": "SPP1-CD44 passes local-contrast FDR across required nulls; high-expression negatives do not.",
            },
            {
                "figure": "Figure 2",
                "output_stem": str(basename),
                "panel": "c",
                "source_data": "figure2_crc_whole_transcriptome_downstream.csv; figure2_crc_gene_program_specificity.csv; figure2_crc_pathway_specific_programs.csv",
                "message": "Receiver transcriptome target enrichment supports SPP1-CD44 and related CRC LRs, while the pathway-specific program layer remains leakage-caveated.",
            },
            {
                "figure": "Figure 2",
                "output_stem": str(basename),
                "panel": "d",
                "source_data": "figure2_crc_commot_style_impact.csv",
                "message": "Whole-transcriptome impact audit quantifies positive support and negative leakage.",
            },
        ]
    )
    manifest.to_csv(OUTDIR / "phase12_figure2_crc_spp1_cd44_story_manifest.csv", index=False)


if __name__ == "__main__":
    render()
