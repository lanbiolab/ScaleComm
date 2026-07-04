#!/usr/bin/env python
"""Phase 14 redesigned Figure 3: post-Xenium colon biological anchor."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from phase14_style import PALETTE, METHOD_COLORS, apply_style, panel_label, clean_axes, save_figure, add_figure_title


ROOT = Path(__file__).resolve().parents[2]
P13 = ROOT / "outputs" / "phase13" / "source_data"
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"

FOCAL = "SPP1-CD44"
POSITIVE = ["SPP1-CD44", "TGFB1-TGFBR2", "CXCL12-CXCR4", "VEGFA-KDR"]
NEGATIVE = ["ACTB-KRT8", "COL1A1-EPCAM"]
METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "SpaTalk", "CellNEST"]
LR_COLORS = {
    "SPP1-CD44": PALETTE["signal"],
    "TGFB1-TGFBR2": PALETTE["lr"],
    "CXCL12-CXCR4": PALETTE["spatialself"],
    "VEGFA-KDR": PALETTE["weak"],
    "ACTB-KRT8": PALETTE["risk"],
    "COL1A1-EPCAM": "#8F4A48",
}


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    data = load_data()
    tables = build_source_tables(data)
    export_source(tables)

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=False)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.48, 1.0, 1.0],
        height_ratios=[1.0, 0.88, 0.78],
        left=0.06,
        right=0.985,
        bottom=0.08,
        top=0.88,
        wspace=0.42,
        hspace=0.55,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rank = fig.add_subplot(gs[0, 1:])
    ax_fdr = fig.add_subplot(gs[1, 1])
    ax_down = fig.add_subplot(gs[1, 2])
    ax_guard = fig.add_subplot(gs[2, 1:])

    draw_spatial_hero(ax_map, tables["spatial_map"], data["cells"])
    draw_rank_strip(ax_rank, tables["rank_summary"])
    draw_fdr_split(ax_fdr, tables["fdr_summary"])
    draw_target_response(ax_down, tables["downstream_summary"])
    draw_guardrail(ax_guard, tables["guardrail_summary"])

    add_figure_title(
        fig,
        "SPP1-CD44 is prioritized as a context-specific colon communication candidate",
        "Phase 14 redesigned Figure 3: spatial localization, baseline rank, local contrast, receiver response and hard-negative suppression form one validation flow.",
    )
    stem = FIG / "phase14_figure3_post_xenium_colon_redesign"
    save_figure(fig, stem)
    write_manifest(stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "spatial_map": P13 / "figure3_spatial_map.csv",
        "cells": P13 / "figure3_cells.csv",
        "rank": P13 / "figure3_figure2_crc_method_lr_rank_table.csv",
        "fdr": P13 / "figure3_figure2_crc_local_contrast_fdr.csv",
        "downstream": P13 / "figure3_figure2_crc_whole_transcriptome_downstream.csv",
        "audit": P13 / "figure3_figure2_crc_local_contrast_audit.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 14 Figure 3 sources: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in files.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rank = data["rank"].copy()
    rank["lr_pair"] = rank["ligand"].astype(str) + "-" + rank["receptor"].astype(str)
    rank["method_label"] = rank["method_label"].replace({"SpatialSelf calibrated": "SpatialSelf"})
    rank_summary = rank[rank["lr_pair"].isin(POSITIVE + NEGATIVE)].copy()
    rank_summary = rank_summary[rank_summary["method_label"].isin(METHOD_ORDER)].copy()

    fdr = data["fdr"].copy()
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    fdr_summary = fdr[fdr["null_model"].eq("lr_identity_permutation") & fdr["lr_pair"].isin(POSITIVE + NEGATIVE)].copy()
    fdr_summary["minus_log10_q"] = -np.log10(fdr_summary["local_contrast_bh_fdr"].astype(float).clip(lower=1e-4))
    fdr_summary["class"] = np.where(fdr_summary["false_positive_high_expression"].astype(bool), "high-expression negative", "biological prior")

    downstream = data["downstream"].copy()
    downstream = downstream[downstream["lr_pair"].isin(POSITIVE)].copy()
    downstream["target_support_label"] = downstream["target_top300_enrichment"].map(lambda x: f"{float(x):.1f}x")

    spatial = data["spatial_map"].copy()
    spatial = spatial.nlargest(min(650, len(spatial)), "score").copy()

    guard_rows = []
    for lr in [FOCAL] + NEGATIVE:
        r = rank_summary[(rank_summary["lr_pair"].eq(lr)) & (rank_summary["method_label"].eq("SpatialSelf"))]
        q = fdr_summary[fdr_summary["lr_pair"].eq(lr)]
        guard_rows.append(
            {
                "lr_pair": lr,
                "class": "focal positive" if lr == FOCAL else "high-expression negative",
                "spatialself_rank": float(r["rank"].iloc[0]) if not r.empty else np.nan,
                "local_contrast_q": float(q["local_contrast_bh_fdr"].iloc[0]) if not q.empty else np.nan,
                "passes_q_0_05": bool(float(q["local_contrast_bh_fdr"].iloc[0]) < 0.05) if not q.empty else False,
            }
        )
    guard = pd.DataFrame(guard_rows)

    return {
        "spatial_map": spatial,
        "rank_summary": rank_summary,
        "fdr_summary": fdr_summary,
        "downstream_summary": downstream,
        "guardrail_summary": guard,
    }


def export_source(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(SOURCE / f"figure3_{name}.csv", index=False)


def draw_spatial_hero(ax: plt.Axes, spatial: pd.DataFrame, cells: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.035, y=1.02)
    ax.set_title("SPP1-CD44 communication hotspots in post-Xenium colon", loc="left", pad=8)
    sample = cells.sample(min(len(cells), 5000), random_state=14)
    type_colors = {
        "epithelial_proxy": "#D5DAE0",
        "myeloid_proxy": "#C8E2D0",
        "fibroblast_proxy": "#DDD2EB",
        "unknown": "#E6E8EA",
    }
    for ctype, sub in sample.groupby("cell_type", dropna=False):
        ax.scatter(sub["x"], sub["y"], s=2.4, color=type_colors.get(str(ctype), "#E2E4E6"), alpha=0.55, linewidths=0, rasterized=True)
    top = spatial.nlargest(min(420, len(spatial)), "score")
    sizes = np.interp(top["score"], (top["score"].min(), top["score"].max()), (8, 34))
    ax.scatter(top["mid_x"], top["mid_y"], s=sizes, c=top["score"], cmap="Greens", edgecolor="none", alpha=0.88, rasterized=True)
    ax.text(
        0.04,
        0.04,
        "green points: top SPP1-CD44 edges\nbackground: cells/spots",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["neutral_soft"], alpha=0.92),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_rank_strip(ax: plt.Axes, rank: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.05, y=1.04)
    ax.set_title("Rank strip: positive LR stay ahead of hard negatives", loc="left", pad=8)
    y_lookup = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    ax.axvspan(0.5, 3.5, color=PALETTE["signal_soft"], alpha=0.75, zorder=0)
    ax.axvspan(3.5, 7.5, color=PALETTE["soft_bg"], alpha=0.8, zorder=0)
    for method in METHOD_ORDER:
        sub = rank[rank["method_label"].eq(method)]
        y = y_lookup[method]
        ax.hlines(y, 0.75, 7.25, color=PALETTE["line"], lw=0.8, zorder=1)
        for _, row in sub.iterrows():
            lr = row["lr_pair"]
            neg = bool(row["false_positive_high_expression"])
            marker = "v" if neg else "o"
            size = 55 if lr == FOCAL else 42 if not neg else 46
            ax.scatter(row["rank"], y, s=size, marker=marker, color=LR_COLORS.get(lr, PALETTE["neutral"]), edgecolor="white", linewidth=0.65, zorder=3)
    ax.set_yticks([y_lookup[m] for m in METHOD_ORDER])
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("rank in matched control panel")
    ax.tick_params(axis="y", length=0)
    clean_axes(ax, grid="x")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LR_COLORS[FOCAL], markeredgecolor="white", markersize=6, label=FOCAL),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["neutral"], markeredgecolor="white", markersize=6, label="other biological priors"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=PALETTE["risk"], markeredgecolor="white", markersize=6, label="high-expression negatives"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=3, columnspacing=0.8, handletextpad=0.35)


def draw_fdr_split(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.12, y=1.04)
    ax.set_title("Local contrast supports SPP1-CD44", loc="left", pad=8)
    order = [FOCAL, "TGFB1-TGFBR2", "CXCL12-CXCR4", "VEGFA-KDR", "ACTB-KRT8", "COL1A1-EPCAM"]
    sub = fdr.set_index("lr_pair").reindex(order).reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [LR_COLORS.get(lr, PALETTE["neutral"]) for lr in sub["lr_pair"]]
    ax.barh(y, sub["minus_log10_q"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(-np.log10(0.05), color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(sub["lr_pair"])
    ax.set_xlabel("-log10 local q")
    ax.set_xlim(0, max(2.2, float(sub["minus_log10_q"].max()) + 0.4))
    clean_axes(ax, grid="x")


def draw_target_response(ax: plt.Axes, downstream: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.12, y=1.04)
    ax.set_title("Receiver target response is enriched", loc="left", pad=8)
    sub = downstream.set_index("lr_pair").reindex(POSITIVE).dropna(subset=["target_top300_enrichment"]).reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [LR_COLORS.get(lr, PALETTE["neutral"]) for lr in sub["lr_pair"]]
    ax.barh(y, sub["target_top300_enrichment"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(1.0, color=PALETTE["muted"], lw=0.8, linestyle=(0, (3, 2)))
    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.text(row["target_top300_enrichment"] + 0.08, yi, f"q={float(row['target_top300_enrichment_bh_fdr']):.2g}", ha="left", va="center", fontsize=6.1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["lr_pair"])
    ax.set_xlabel("target enrichment")
    ax.set_xlim(0, max(3.6, float(sub["target_top300_enrichment"].max()) + 0.75))
    clean_axes(ax, grid="x")


def draw_guardrail(ax: plt.Axes, guard: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.05, y=1.04)
    ax.set_title("Hard-negative guardrail: high-expression controls do not pass local contrast", loc="left", pad=8)
    ax.set_axis_off()
    x_positions = [0.16, 0.50, 0.82]
    for x, (_, row) in zip(x_positions, guard.iterrows()):
        is_focal = row["lr_pair"] == FOCAL
        face = PALETTE["signal_soft"] if is_focal else PALETTE["risk_soft"]
        edge = PALETTE["signal"] if is_focal else PALETTE["risk"]
        ax.add_patch(patches.FancyBboxPatch((x - 0.135, 0.30), 0.27, 0.42, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=face, edgecolor=edge, linewidth=0.9, transform=ax.transAxes))
        ax.text(x, 0.63, row["lr_pair"], transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color=PALETTE["ink"])
        ax.text(x, 0.51, f"rank {int(row['spatialself_rank'])}", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=PALETTE["ink"])
        ax.text(x, 0.40, f"q={float(row['local_contrast_q']):.2g}", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=PALETTE["ink"])
        ax.text(x, 0.25, "passes" if row["passes_q_0_05"] else "suppressed", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=edge, fontweight="bold")
    ax.text(0.02, 0.05, "Interpretation: selected local guardrail supports the SPP1-CD44 story; it is not a whole-atlas causal proof.", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=PALETTE["muted"])


def write_manifest(stem: Path) -> None:
    rows = [
        ("a", "SPP1-CD44 top communication edges are spatially localized in post-Xenium colon.", "outputs/phase14/source_data/figure3_spatial_map.csv"),
        ("b", "SpatialSelf ranks SPP1-CD44 and other biological priors ahead of high-expression controls.", "outputs/phase14/source_data/figure3_rank_summary.csv"),
        ("c", "Local-contrast FDR supports SPP1-CD44 while high-expression controls fail.", "outputs/phase14/source_data/figure3_fdr_summary.csv"),
        ("d", "Receiver target-gene enrichment supports focal biological priors.", "outputs/phase14/source_data/figure3_downstream_summary.csv"),
        ("e", "Hard-negative guardrail separates the focal LR from high-expression controls.", "outputs/phase14/source_data/figure3_guardrail_summary.csv"),
    ]
    pd.DataFrame(rows, columns=["panel", "claim", "source_data"]).to_csv(f"{stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
