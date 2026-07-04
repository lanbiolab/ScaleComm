#!/usr/bin/env python
"""Render Phase 13 Figure 3: post-Xenium colon biological case."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from phase13_figure_style import PALETTE, apply_style, panel_label, save_figure, soften_axes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/phase12/manuscript_figure_package"
REPLICATE = ROOT / "outputs/phase12/atlas_model_expanded_integrated_guarded_spatial_training/replicates/post_xenium_colon_full"
PREDICTION = (
    ROOT
    / "outputs/phase12/integrated_background_target_guarded_training_smoke/predictions/post_xenium_colon_full/"
    "spatialself_wavelet_integrated_background_target_guarded/post_xenium_colon_full__spatialself_wavelet_predictions.csv"
)
OUT_DIR = ROOT / "outputs/phase13/figures"
SOURCE_OUT = ROOT / "outputs/phase13/source_data"

FOCAL_LR = ["SPP1-CD44", "TGFB1-TGFBR2", "CXCL12-CXCR4"]
POSITIVE_LR = ["SPP1-CD44", "TGFB1-TGFBR2", "CXCL12-CXCR4", "VEGFA-KDR"]
NEGATIVE_LR = ["ACTB-KRT8", "COL1A1-EPCAM"]
METHOD_ORDER = ["spatialself_calibrated", "commot", "cellchat", "spatalk", "cellnest"]
METHOD_LABEL = {
    "spatialself_calibrated": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}
METHOD_COLOR = {
    "spatialself_calibrated": PALETTE["spatialself"],
    "commot": "#707070",
    "cellchat": "#9B9B9B",
    "spatalk": "#B8BDC4",
    "cellnest": "#7A86A8",
}
LR_COLOR = {
    "SPP1-CD44": "#3B8F5A",
    "TGFB1-TGFBR2": "#7E57A0",
    "CXCL12-CXCR4": "#2C6FBB",
    "VEGFA-KDR": "#D39B2A",
    "ACTB-KRT8": PALETTE["hard_negative"],
    "COL1A1-EPCAM": "#8D3B36",
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    source_tables = build_source_tables(data)
    export_source_tables(source_tables)

    fig = plt.figure(figsize=(7.2, 7.1), constrained_layout=False)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 0.86, 0.88],
        width_ratios=[1.12, 1.0, 1.0],
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.9,
        wspace=0.42,
        hspace=0.55,
    )

    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_rank = fig.add_subplot(gs[0, 1:])
    ax_fdr = fig.add_subplot(gs[1, 1])
    ax_down = fig.add_subplot(gs[1, 2])
    ax_neg = fig.add_subplot(gs[2, 0:2])
    ax_summary = fig.add_subplot(gs[2, 2])

    draw_spatial_map(ax_map, source_tables["spatial_map"])
    draw_rank_panel(ax_rank, data["rank"])
    draw_local_fdr_panel(ax_fdr, data["fdr"])
    draw_downstream_panel(ax_down, data["downstream"], data["program"])
    draw_negative_panel(ax_neg, data["rank"], data["fdr"])
    draw_summary_panel(ax_summary, data["audit"])

    fig.suptitle(
        "Post-Xenium colon: SpatialSelf prioritizes context-specific CCC with spatial and receiver-response support",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=10.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.07,
        0.952,
        "Main case figure draft. Local contrast and downstream target support are observational and non-causal; whole-atlas FDR remains handled in the guardrail figure.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    out_stem = OUT_DIR / "phase13_figure3_post_xenium_colon"
    save_figure(fig, out_stem)
    write_manifest(out_stem)
    plt.close(fig)
    print(f"wrote {out_stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "rank": SOURCE / "figure2_crc_method_lr_rank_table.csv",
        "fdr": SOURCE / "figure2_crc_local_contrast_fdr.csv",
        "downstream": SOURCE / "figure2_crc_whole_transcriptome_downstream.csv",
        "program": SOURCE / "figure2_crc_gene_program_specificity.csv",
        "audit": SOURCE / "figure2_crc_local_contrast_audit.csv",
        "cells": REPLICATE / "cells.csv",
        "candidates": REPLICATE / "candidate_interactions.csv",
        "predictions": PREDICTION,
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 3 source files: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in files.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cells = data["cells"][["cell_id", "x", "y", "cell_type", "region", "niche"]].copy()
    candidates = data["candidates"][
        ["candidate_id", "sender_id", "receiver_id", "ligand", "receptor", "lr_id", "true_lr", "false_positive_high_expression"]
    ].copy()
    pred = data["predictions"].copy()
    merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    merged["lr_pair"] = merged["ligand"].astype(str) + "-" + merged["receptor"].astype(str)
    spp1 = merged[merged["lr_pair"].eq("SPP1-CD44")].nlargest(950, "score").copy()
    coords = cells[["cell_id", "x", "y", "cell_type", "niche"]]
    spp1 = spp1.merge(coords.add_prefix("sender_"), left_on="sender_id", right_on="sender_cell_id", how="left")
    spp1 = spp1.merge(coords.add_prefix("receiver_"), left_on="receiver_id", right_on="receiver_cell_id", how="left")
    spp1["mid_x"] = (spp1["sender_x"] + spp1["receiver_x"]) / 2.0
    spp1["mid_y"] = (spp1["sender_y"] + spp1["receiver_y"]) / 2.0
    spatial_map = spp1[
        [
            "candidate_id",
            "lr_pair",
            "score",
            "mid_x",
            "mid_y",
            "sender_x",
            "sender_y",
            "receiver_x",
            "receiver_y",
            "sender_cell_type",
            "receiver_cell_type",
        ]
    ].copy()
    return {
        "spatial_map": spatial_map,
        "cells": cells,
    }


def export_source_tables(source_tables: dict[str, pd.DataFrame]) -> None:
    for key, table in source_tables.items():
        table.to_csv(SOURCE_OUT / f"figure3_{key}.csv", index=False)
    for name in [
        "figure2_crc_method_lr_rank_table.csv",
        "figure2_crc_local_contrast_fdr.csv",
        "figure2_crc_whole_transcriptome_downstream.csv",
        "figure2_crc_gene_program_specificity.csv",
        "figure2_crc_local_contrast_audit.csv",
    ]:
        pd.read_csv(SOURCE / name).to_csv(SOURCE_OUT / f"figure3_{name}", index=False)


def draw_spatial_map(ax: plt.Axes, spatial_map: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.04, y=1.02)
    ax.set_title("SPP1-CD44 communication localizes within the colon context", loc="left", pad=8)
    cells_path = SOURCE_OUT / "figure3_cells.csv"
    cells = pd.read_csv(cells_path) if cells_path.exists() else pd.DataFrame()
    if not cells.empty:
        sample = cells.sample(min(len(cells), 4500), random_state=13)
        type_color = {
            "epithelial_proxy": "#C7CFD8",
            "myeloid_proxy": "#BBD7C3",
            "fibroblast_proxy": "#D8C9E8",
            "unknown": "#E1E1E1",
        }
        for ctype, sub in sample.groupby("cell_type", dropna=False):
            ax.scatter(
                sub["x"],
                sub["y"],
                s=2.2,
                color=type_color.get(str(ctype), "#D8D8D8"),
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )

    top = spatial_map.nlargest(360, "score").copy()
    sizes = np.interp(top["score"], (top["score"].min(), top["score"].max()), (7, 26))
    ax.scatter(
        top["mid_x"],
        top["mid_y"],
        s=sizes,
        c=top["score"],
        cmap="YlGn",
        edgecolor="none",
        alpha=0.86,
        rasterized=True,
    )
    q = spatial_map["score"].quantile([0.50, 0.90, 0.99]).to_dict()
    ax.text(
        0.02,
        0.02,
        f"top SPP1-CD44 edges shown\nscore q50={q[0.5]:.2f}, q90={q[0.9]:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["neutral_soft"], alpha=0.9),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_rank_panel(ax: plt.Axes, rank: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.06, y=1.03)
    ax.set_title("SpatialSelf keeps positives ahead of high-expression controls", loc="left", pad=8)
    df = rank[rank["method"].isin(METHOD_ORDER)].copy()
    df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    y_lookup = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    ax.axvspan(0.5, 4.5, color="#F3F8FC", zorder=0)
    for method in METHOD_ORDER:
        sub = df[df["method"].eq(method)]
        y = y_lookup[method]
        ax.hlines(y, 0.8, 7.2, color=PALETTE["neutral_soft"], lw=1.0, zorder=0)
        for _, row in sub.iterrows():
            lr = row["lr_pair"]
            is_neg = bool(row["false_positive_high_expression"])
            if lr in FOCAL_LR:
                color, marker, size = LR_COLOR[lr], "o", 54
            elif is_neg:
                color, marker, size = PALETTE["hard_negative"], "v", 48
            elif bool(row["true_lr"]):
                color, marker, size = PALETTE["neutral"], "o", 32
            else:
                color, marker, size = PALETTE["neutral_soft"], "o", 24
            ax.scatter(float(row["rank"]), y, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.65, zorder=3)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(range(1, 8))
    ax.set_yticks([y_lookup[m] for m in METHOD_ORDER])
    ax.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_xlabel("rank in CRC control panel")
    ax.tick_params(axis="y", length=0)
    soften_axes(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LR_COLOR["SPP1-CD44"], markeredgecolor="white", markersize=6, label="SPP1-CD44"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LR_COLOR["TGFB1-TGFBR2"], markeredgecolor="white", markersize=6, label="TGFB1-TGFBR2"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LR_COLOR["CXCL12-CXCR4"], markeredgecolor="white", markersize=6, label="CXCL12-CXCR4"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=PALETTE["hard_negative"], markeredgecolor="white", markersize=6, label="high-expression negative"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=4, columnspacing=0.7, handletextpad=0.35)


def draw_local_fdr_panel(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.12, y=1.03)
    ax.set_title("Local contrast separates signal from controls", loc="left", pad=8)
    sub = fdr[fdr["null_model"].astype(str).eq("lr_identity_permutation")].copy()
    sub["lr_pair"] = sub["ligand"].astype(str) + "-" + sub["receptor"].astype(str)
    order = ["SPP1-CD44", "TGFB1-TGFBR2", "CXCL12-CXCR4", "VEGFA-KDR", "ACTB-KRT8", "COL1A1-EPCAM"]
    sub = sub.set_index("lr_pair").reindex(order).reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [LR_COLOR.get(lr, PALETTE["neutral"]) for lr in sub["lr_pair"]]
    vals = -np.log10(sub["local_contrast_bh_fdr"].clip(lower=1e-4).astype(float))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(-np.log10(0.05), color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    for yi, (_, row), val in zip(y, sub.iterrows(), vals):
        q = float(row["local_contrast_bh_fdr"])
        ax.text(val + 0.06, yi, f"q={q:.3g}", va="center", ha="left", fontsize=5.9, color=PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["lr_pair"])
    ax.set_xlabel("-log10 local-contrast q")
    ax.set_xlim(0, max(2.7, float(vals.max()) + 0.75))
    soften_axes(ax)
    ax.text(
        0.02,
        -0.32,
        "LR-label null shown; region and cell-type preserving nulls give the same SPP1-CD44 call.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=PALETTE["muted"],
    )


def draw_downstream_panel(ax: plt.Axes, downstream: pd.DataFrame, program: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.12, y=1.03)
    ax.set_title("Receiver transcriptome supports focal LR", loc="left", pad=8)
    df = downstream.copy()
    df = df[df["lr_pair"].isin(POSITIVE_LR)].copy()
    df["lr_pair"] = pd.Categorical(df["lr_pair"], POSITIVE_LR, ordered=True)
    df = df.sort_values("lr_pair")
    prog = program.set_index("lr_pair")
    df["specific_terms"] = df["lr_pair"].astype(str).map(prog["n_specific_significant_terms_fdr_0_05"])
    y = np.arange(len(df))[::-1]
    colors = [LR_COLOR.get(str(lr), PALETTE["spatialself"]) for lr in df["lr_pair"]]
    ax.barh(y, df["target_top300_enrichment"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(1.0, color=PALETTE["muted"], lw=0.8, linestyle=(0, (3, 2)))
    for yi, (_, row) in zip(y, df.iterrows()):
        q = float(row["target_top300_enrichment_bh_fdr"])
        terms = int(row["specific_terms"]) if pd.notna(row["specific_terms"]) else 0
        ax.text(
            row["target_top300_enrichment"] + 0.08,
            yi,
            f"q={q:.3g}; {terms} specific terms",
            ha="left",
            va="center",
            fontsize=5.9,
            color=PALETTE["ink"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df["lr_pair"].astype(str))
    ax.set_xlabel("target enrichment")
    ax.set_xlim(0, max(4.2, float(df["target_top300_enrichment"].max()) + 1.4))
    soften_axes(ax)
    ax.text(
        0.02,
        -0.32,
        "NicheNet/target-gene association; observational, not perturbational.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=PALETTE["muted"],
    )


def draw_negative_panel(ax: plt.Axes, rank: pd.DataFrame, fdr: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.04, y=1.03)
    ax.set_title("High-expression controls are demoted by SpatialSelf and fail local contrast", loc="left", pad=8)
    df = rank[rank["method"].isin(METHOD_ORDER)].copy()
    df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
    neg = df[df["lr_pair"].isin(NEGATIVE_LR)].copy()
    neg["method"] = pd.Categorical(neg["method"], METHOD_ORDER, ordered=True)
    pivot = neg.pivot_table(index="method", columns="lr_pair", values="rank", aggfunc="min").reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    for i, lr in enumerate(NEGATIVE_LR):
        vals = pivot[lr].astype(float).values
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=LR_COLOR[lr],
            alpha=0.82 if i == 0 else 0.66,
            edgecolor="white",
            linewidth=0.7,
            label=lr,
        )
    ax.invert_yaxis()
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=0)
    ax.set_ylabel("negative-control rank\n(lower is worse)")
    ax.set_ylim(7.6, 0.4)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    soften_axes(ax)

    fdr_lr = fdr[fdr["null_model"].astype(str).eq("lr_identity_permutation")].copy()
    fdr_lr["lr_pair"] = fdr_lr["ligand"].astype(str) + "-" + fdr_lr["receptor"].astype(str)
    q = fdr_lr[fdr_lr["lr_pair"].isin(NEGATIVE_LR)].set_index("lr_pair")["local_contrast_bh_fdr"]
    ax.text(
        0.02,
        0.06,
        "local contrast q: " + "; ".join([f"{lr}={float(q.get(lr, np.nan)):.2g}" for lr in NEGATIVE_LR]),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=PALETTE["muted"],
    )


def draw_summary_panel(ax: plt.Axes, audit: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.12, y=1.03)
    ax.set_axis_off()
    row = audit.iloc[0].to_dict() if not audit.empty else {}
    status = str(row.get("local_contrast_all_null_status", ""))
    required = str(row.get("null_models_required", "")).replace(";", "\n")
    passing = str(row.get("null_models_passing", "")).replace(";", "\n")
    cards = [
        ("local contrast", status.replace("_", " "), PALETTE["success"]),
        ("required nulls", required, PALETTE["neutral"]),
        ("passing nulls", passing, PALETTE["spatialself"]),
    ]
    y = 0.78
    for title, body, color in cards:
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.03, y - 0.14),
                0.92,
                0.17,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.94,
                transform=ax.transAxes,
            )
        )
        ax.text(0.07, y, title, transform=ax.transAxes, ha="left", va="center", fontsize=6.4, color="white", fontweight="bold")
        ax.text(0.44, y, wrap(str(body), 26), transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color="white", linespacing=1.05)
        y -= 0.24
    ax.text(
        0.03,
        0.03,
        "Claim boundary: supports a prioritized biological case; does not prove causal signaling or whole-atlas FDR closure.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["muted"],
        wrap=True,
    )


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def write_manifest(out_stem: Path) -> None:
    rows = [
        {
            "figure": "Figure 3",
            "panel": "a",
            "claim": "Top SPP1-CD44 communication edges are spatially localized in the post-Xenium colon background.",
            "source_data": "outputs/phase13/source_data/figure3_spatial_map.csv; figure3_cells.csv",
            "review_risk": "Map shows model-prioritized edges, not histological annotation or causal communication.",
        },
        {
            "figure": "Figure 3",
            "panel": "b",
            "claim": "SpatialSelf ranks focal positive LR ahead of high-expression negative controls in the CRC control panel.",
            "source_data": "outputs/phase13/source_data/figure3_figure2_crc_method_lr_rank_table.csv",
            "review_risk": "Control-panel rank is not unbiased whole-atlas FDR.",
        },
        {
            "figure": "Figure 3",
            "panel": "c",
            "claim": "SPP1-CD44 passes local-contrast FDR while high-expression controls do not.",
            "source_data": "outputs/phase13/source_data/figure3_figure2_crc_local_contrast_fdr.csv",
            "review_risk": "Local contrast validates spatial concentration, not causal signaling.",
        },
        {
            "figure": "Figure 3",
            "panel": "d",
            "claim": "Focal positive LR have receiver target-gene enrichment and gene-program support.",
            "source_data": "outputs/phase13/source_data/figure3_figure2_crc_whole_transcriptome_downstream.csv; figure3_figure2_crc_gene_program_specificity.csv",
            "review_risk": "Downstream association is observational; pathway-specific program table is currently empty and is not claimed as closed.",
        },
        {
            "figure": "Figure 3",
            "panel": "e",
            "claim": "High-expression negative controls are visually separated by rank and local-contrast q-values.",
            "source_data": "outputs/phase13/source_data/figure3_figure2_crc_method_lr_rank_table.csv; figure3_figure2_crc_local_contrast_fdr.csv",
            "review_risk": "Negative controls are selected hard controls, not all possible false positives.",
        },
        {
            "figure": "Figure 3",
            "panel": "f",
            "claim": "All required local-contrast null models support at least one positive LR and suppress high-expression controls.",
            "source_data": "outputs/phase13/source_data/figure3_figure2_crc_local_contrast_audit.csv",
            "review_risk": "This is a local story guardrail, not final whole-atlas FDR.",
        },
    ]
    pd.DataFrame(rows).to_csv(f"{out_stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
