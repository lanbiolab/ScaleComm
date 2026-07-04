#!/usr/bin/env python
"""Render Phase 13 Figure 5: FDR and claim guardrails."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from phase13_figure_style import PALETTE, apply_style, panel_label, save_figure, soften_axes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/phase12/manuscript_figure_package"
OUT_DIR = ROOT / "outputs/phase13/figures"
SOURCE_OUT = ROOT / "outputs/phase13/source_data"

SOURCE_FILES = {
    "permutation_dataset": SOURCE / "figure1_expanded_permutation_dataset_summary.csv",
    "permutation_topk": SOURCE / "figure1_expanded_permutation_topk_summary.csv",
    "target_decoy_dataset": SOURCE / "figure1_expanded_target_decoy_dataset_summary.csv",
    "target_decoy_topk": SOURCE / "figure1_expanded_target_decoy_topk_summary.csv",
    "targeted_dataset": SOURCE / "figure1_targeted_matched_null_dataset_summary.csv",
    "targeted_lr": SOURCE / "figure1_targeted_matched_null_supported_lr.csv",
    "hard_negative_dataset": SOURCE / "figure4_hard_negative_dataset_summary.csv",
    "edge_guardrail": SOURCE / "figure4_edge_hard_negative_dataset_summary.csv",
    "coordinate_guardrail": SOURCE / "figure4_coordinate_control_guardrail_dataset_summary.csv",
    "coordinate_smoke": SOURCE / "figure4_coordinate_model_rerun_smoke_summary.csv",
    "pathway_dataset": SOURCE / "figure4_pathway_specific_dataset_summary.csv",
    "claim_matrix": SOURCE / "figure4_claim_matrix_guardrails.csv",
}

DATASET_LABEL = {
    "human_lymph_node_visium_raw_full": "Lymph node\nVisium",
    "post_xenium_colon_full": "Post-Xenium\ncolon",
    "visium_hd_crc_full": "Visium HD\nCRC",
}
DATASET_COLOR = {
    "human_lymph_node_visium_raw_full": PALETTE["spatialself"],
    "post_xenium_colon_full": PALETTE["success"],
    "visium_hd_crc_full": PALETTE["weak"],
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    source_tables = build_source_tables(data)
    export_source_tables(source_tables)

    fig = plt.figure(figsize=(7.2, 7.05), constrained_layout=False)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.9, 0.96, 0.94],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.07,
        right=0.985,
        top=0.895,
        bottom=0.075,
        wspace=0.42,
        hspace=0.6,
    )

    ax_fdr = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[0, 1])
    ax_decoy = fig.add_subplot(gs[0, 2])
    ax_hard = fig.add_subplot(gs[1, 0])
    ax_edge = fig.add_subplot(gs[1, 1])
    ax_coord = fig.add_subplot(gs[1, 2])
    ax_path = fig.add_subplot(gs[2, 0])
    ax_claim = fig.add_subplot(gs[2, 1:])

    draw_fdr_layer(ax_fdr, source_tables["fdr_layer_compact"])
    draw_targeted_support(ax_target, source_tables["targeted_dataset"])
    draw_topk_control_fdr(ax_decoy, source_tables["target_decoy_topk_compact"])
    draw_hard_negative(ax_hard, source_tables["hard_negative_dataset"])
    draw_edge_guardrail(ax_edge, source_tables["edge_guardrail"])
    draw_coordinate_guardrail(ax_coord, source_tables["coordinate_guardrail"])
    draw_pathway_guardrail(ax_path, source_tables["pathway_dataset"])
    draw_claim_matrix(ax_claim, source_tables["claim_matrix_compact"])

    fig.suptitle(
        "FDR and guardrail audits define what SpatialSelf can and cannot claim",
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
        "Guardrail figure draft. Targeted, matched and control-null evidence supports selected stories; strict unbiased whole-atlas FDR remains conservatively caveated.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    out_stem = OUT_DIR / "phase13_figure5_fdr_guardrails"
    save_figure(fig, out_stem)
    write_manifest(out_stem)
    plt.close(fig)
    print(f"wrote {out_stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 5 source files: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in SOURCE_FILES.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    targeted = data["targeted_dataset"].copy()
    fdr_layer = targeted[
        [
            "dataset_key",
            "n_anchor_lr",
            "n_whole_atlas_fdr_0_1",
            "n_targeted_panel_q_0_1",
            "n_matched_control_q_0_1",
            "n_hard_negative_q_0_1",
            "n_strong_targeted_matched_support",
        ]
    ].copy()

    decoy = data["target_decoy_topk"].copy()
    decoy = decoy[decoy["top_k"].isin([10, 100])].copy()

    claims = data["claim_matrix"].copy()
    claim_topics = [
        "Whole-atlas permutation FDR",
        "Targeted matched-null validation",
        "Edge-level hard-negative leakage modes",
        "Coordinate model-rerun smoke",
        "Pathway-specific receiver programs",
        "Global directionality superiority",
        "COMMOT-style communication impact",
        "Directionality edge-level permutation audit",
    ]
    claim_compact = claims[claims["topic"].isin(claim_topics)].copy()
    claim_compact["claim_class"] = claim_compact["support_status"].map(classify_claim)

    return {
        "fdr_layer_compact": fdr_layer,
        "permutation_dataset": data["permutation_dataset"],
        "permutation_topk": data["permutation_topk"],
        "targeted_dataset": targeted,
        "targeted_lr": data["targeted_lr"],
        "target_decoy_dataset": data["target_decoy_dataset"],
        "target_decoy_topk_compact": decoy,
        "hard_negative_dataset": data["hard_negative_dataset"],
        "edge_guardrail": data["edge_guardrail"],
        "coordinate_guardrail": data["coordinate_guardrail"],
        "coordinate_smoke": data["coordinate_smoke"],
        "pathway_dataset": data["pathway_dataset"],
        "claim_matrix_compact": claim_compact,
    }


def classify_claim(status: object) -> str:
    text = str(status).lower()
    if "not_supported" in text or "not supported" in text:
        return "not supported"
    if "failure" in text or "caveat" in text or "smoke" in text or "guardrail" in text or "leakage" in text:
        return "guardrail"
    if "supported" in text:
        return "supported"
    return "guardrail"


def export_source_tables(source_tables: dict[str, pd.DataFrame]) -> None:
    for key, table in source_tables.items():
        table.to_csv(SOURCE_OUT / f"figure5_{key}.csv", index=False)


def draw_fdr_layer(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.1, y=1.03)
    ax.set_title("Strict whole-atlas FDR remains conservative", loc="left", pad=8)
    df = fdr.copy()
    y = np.arange(len(df))[::-1]
    cols = [
        ("n_whole_atlas_fdr_0_1", "whole atlas", PALETTE["neutral"]),
        ("n_targeted_panel_q_0_1", "targeted", PALETTE["spatialself"]),
        ("n_matched_control_q_0_1", "matched", PALETTE["success"]),
    ]
    left = np.zeros(len(df))
    for col, label, color in cols:
        vals = df[col].to_numpy(float)
        ax.barh(y, vals, left=left, height=0.52, color=color, edgecolor="white", linewidth=0.5, label=label)
        for yi, lft, val in zip(y, left, vals):
            if val > 0:
                ax.text(lft + val / 2, yi, f"{int(val)}", ha="center", va="center", fontsize=5.7, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]])
    ax.set_xlabel("Anchor LR passing q < 0.1")
    ax.set_xlim(0, max(left.max() * 1.18, 10))
    soften_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=5.6)


def draw_targeted_support(ax: plt.Axes, targeted: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.1, y=1.03)
    ax.set_title("Targeted/matched null supports selected anchors", loc="left", pad=8)
    df = targeted.copy()
    x = np.arange(len(df))
    width = 0.32
    ax.bar(x - width / 2, df["n_rank_top10"], width=width, color=PALETTE["neutral_soft"], edgecolor=PALETTE["line"], linewidth=0.7, label="rank top10")
    ax.bar(x + width / 2, df["n_strong_targeted_matched_support"], width=width, color=PALETTE["spatialself"], edgecolor="white", linewidth=0.7, label="strong support")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]], rotation=0)
    ax.set_ylabel("No. anchor LR")
    ax.set_ylim(0, max(8, float(df[["n_rank_top10", "n_strong_targeted_matched_support"]].max().max()) + 1))
    soften_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    for xi, lr in zip(x, df["best_supported_lr"]):
        ax.text(xi, 0.15, lr.replace("-", "\n"), ha="center", va="bottom", fontsize=5.3, color=PALETTE["muted"])


def draw_topk_control_fdr(ax: plt.Axes, decoy: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.1, y=1.03)
    ax.set_title("Control FDR improves deeper in ranked lists", loc="left", pad=8)
    for dataset, sub in decoy.groupby("dataset_key"):
        sub = sub.sort_values("top_k")
        ax.plot(
            sub["top_k"],
            sub["primary_control_prefix_fdr"],
            marker="o",
            lw=1.2,
            ms=4.2,
            color=DATASET_COLOR.get(dataset, PALETTE["neutral"]),
            label=DATASET_LABEL.get(dataset, dataset).replace("\n", " "),
        )
    ax.axhline(0.1, color=PALETTE["muted"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([10, 100])
    ax.set_xticklabels(["top10", "top100"])
    ax.set_ylabel("Control-prefix FDR")
    ax.set_ylim(0, 0.42)
    soften_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=5.3)


def draw_hard_negative(ax: plt.Axes, hard: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.1, y=1.03)
    ax.set_title("Dataset-level hard-negative stress", loc="left", pad=8)
    df = hard.copy()
    y = np.arange(len(df))[::-1]
    vals = df["negative_comparison_beat_fraction"].astype(float)
    colors = [DATASET_COLOR.get(k, PALETTE["neutral"]) for k in df["dataset_key"]]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]])
    ax.set_xlabel("Negative beats positive fraction")
    ax.set_xlim(0, max(0.5, vals.max() * 1.2))
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    for yy, val in zip(y, vals):
        ax.text(val + 0.015, yy, f"{val:.2f}", va="center", fontsize=5.6, color=PALETTE["muted"])


def draw_edge_guardrail(ax: plt.Axes, edge: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.1, y=1.03)
    ax.set_title("Edge guardrail reduces leakage but does not solve it", loc="left", pad=8)
    df = edge.copy()
    datasets = ["post_xenium_colon_full", "human_lymph_node_visium_raw_full"]
    x = np.arange(len(datasets))
    width = 0.34
    for j, mode in enumerate(["raw", "edge_guardrail"]):
        sub = df[df["score_mode"].eq(mode)].set_index("dataset_key").reindex(datasets)
        vals = sub["mean_leakage_fraction"].astype(float).to_numpy()
        ax.bar(
            x + (j - 0.5) * width,
            vals,
            width=width,
            color=PALETTE["hard_negative_soft"] if mode == "raw" else PALETTE["spatialself"],
            edgecolor="white",
            linewidth=0.6,
            label="raw" if mode == "raw" else "guardrail",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in datasets])
    ax.set_ylabel("Mean leakage fraction")
    ax.set_ylim(0, 0.65)
    soften_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)


def draw_coordinate_guardrail(ax: plt.Axes, coord: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.1, y=1.03)
    ax.set_title("Coordinate-control guardrail rescues story LR", loc="left", pad=8)
    df = coord.copy()
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["n_story_guardrail_true_lr"], color=PALETTE["success"], edgecolor="white", linewidth=0.6, label="story true LR")
    ax.barh(
        y,
        df["n_story_guardrail_high_expression_negative"],
        left=df["n_story_guardrail_true_lr"],
        color=PALETTE["hard_negative"],
        edgecolor="white",
        linewidth=0.6,
        label="story hard neg.",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]])
    ax.set_xlabel("Story guardrail discoveries")
    ax.set_xlim(0, max(12, int(df["n_story_guardrail_true_lr"].max()) + 2))
    soften_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=5.4)
    for yy, q in zip(y, df["negative_control_q95"]):
        ax.text(0.1, yy - 0.28, f"neg q95={q:.3g}", ha="left", va="center", fontsize=5.3, color=PALETTE["muted"])


def draw_pathway_guardrail(ax: plt.Axes, path: pd.DataFrame) -> None:
    panel_label(ax, "g", x=-0.1, y=1.03)
    ax.set_title("Pathway-specific closure remains leakage-confounded", loc="left", pad=8)
    df = path.copy()
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["n_negative_specific_leakage"], color=PALETTE["hard_negative_soft"], edgecolor="white", linewidth=0.6, label="negative-specific leakage")
    ax.barh(
        y,
        df["n_clean_specific_support"],
        left=df["n_negative_specific_leakage"],
        color=PALETTE["success"],
        edgecolor="white",
        linewidth=0.6,
        label="clean support",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]])
    ax.set_xlabel("No. pathway programs")
    ax.set_xlim(0, max(6, int(df[["n_negative_specific_leakage", "n_clean_specific_support"]].sum(axis=1).max()) + 1))
    soften_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=5.3)


def draw_claim_matrix(ax: plt.Axes, claims: pd.DataFrame) -> None:
    panel_label(ax, "h", x=-0.045, y=1.03)
    ax.set_title("Claim matrix separates support from forbidden extensions", loc="left", pad=8)
    ax.set_axis_off()
    rows = claims.copy()
    priority = {"supported": 0, "guardrail": 1, "not supported": 2}
    rows["_order"] = rows["claim_class"].map(priority).fillna(1)
    rows = rows.sort_values(["_order", "topic"]).head(8)
    color = {"supported": PALETTE["success"], "guardrail": PALETTE["weak"], "not supported": PALETTE["hard_negative"]}
    y0 = 0.86
    for i, row in enumerate(rows.itertuples(index=False)):
        y = y0 - i * 0.11
        c = color.get(row.claim_class, PALETTE["neutral"])
        ax.add_patch(patches.Circle((0.035, y), 0.018, transform=ax.transAxes, facecolor=c, edgecolor="white", lw=0.5))
        ax.text(0.07, y + 0.018, row.topic, transform=ax.transAxes, ha="left", va="center", fontsize=6.1, color=PALETTE["ink"], fontweight="bold")
        caveat = "\n".join(textwrap.wrap(str(row.caveat_or_forbidden_extension), width=88))
        ax.text(0.07, y - 0.028, caveat, transform=ax.transAxes, ha="left", va="center", fontsize=5.05, color=PALETTE["muted"], linespacing=1.02)
    ax.text(
        0.02,
        0.02,
        "Manuscript wording should track this table: selected-story support is allowed; global superiority/strict whole-atlas FDR closure is not.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["spatialself_dark"],
    )


def write_manifest(out_stem: Path) -> None:
    rows = [
        ("a", "FDR layers", "figure5_fdr_layer_compact.csv", "Strict whole-atlas FDR is conservative; targeted/matched nulls support selected anchors."),
        ("b", "Targeted support", "figure5_targeted_dataset.csv", "Selected anchor LR have targeted or matched support depending on dataset."),
        ("c", "Control FDR top-k", "figure5_target_decoy_topk_compact.csv", "Control-prefix FDR is quantified and improves at deeper top-k thresholds in several datasets."),
        ("d", "Hard-negative stress", "figure5_hard_negative_dataset.csv", "Hard negatives remain a measurable risk and are not hidden."),
        ("e", "Edge guardrail", "figure5_edge_guardrail.csv", "Diagnostic edge guardrail reduces leakage but is not a final FDR solution."),
        ("f", "Coordinate guardrail", "figure5_coordinate_guardrail.csv", "Coordinate-control guardrail rescues selected story LR while suppressing known high-expression controls."),
        ("g", "Pathway guardrail", "figure5_pathway_dataset.csv", "Pathway-specific receiver-program closure remains leakage-confounded."),
        ("h", "Claim matrix", "figure5_claim_matrix_compact.csv", "Supported statements and forbidden extensions are explicitly separated."),
    ]
    pd.DataFrame(
        [
            {
                "figure": "Figure 5",
                "panel": panel,
                "title": title,
                "source_data": f"outputs/phase13/source_data/{source}",
                "claim_supported": claim,
            }
            for panel, title, source, claim in rows
        ]
    ).to_csv(f"{out_stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
