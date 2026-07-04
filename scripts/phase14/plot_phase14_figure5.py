#!/usr/bin/env python
"""Phase 14 redesigned Figure 5: FDR evidence hierarchy and claim boundaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from phase14_style import PALETTE, apply_style, panel_label, clean_axes, save_figure, add_figure_title


ROOT = Path(__file__).resolve().parents[2]
P13 = ROOT / "outputs" / "phase13" / "source_data"
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"

DATASET_LABEL = {
    "human_lymph_node_visium_raw_full": "lymph node",
    "post_xenium_colon_full": "post-Xenium\ncolon",
    "visium_hd_crc_full": "Visium HD\nCRC",
}


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data = load_data()
    tables = build_source_tables(data)
    export_source(tables)

    fig = plt.figure(figsize=(7.2, 5.55), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1.12, 0.86],
        left=0.06,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.42,
        hspace=0.55,
    )
    ax_hier = fig.add_subplot(gs[:, 0])
    ax_target = fig.add_subplot(gs[0, 1])
    ax_decoy = fig.add_subplot(gs[0, 2])
    ax_hard = fig.add_subplot(gs[1, 1])
    ax_claim = fig.add_subplot(gs[1, 2])

    draw_evidence_hierarchy(ax_hier, tables["hierarchy"])
    draw_targeted(ax_target, tables["targeted"])
    draw_decoy(ax_decoy, tables["decoy"])
    draw_hard_negative(ax_hard, tables["hard_negative"])
    draw_claim_boundary(ax_claim, tables["claim_boundary"])

    add_figure_title(
        fig,
        "FDR guardrails define where SpatialSelf claims are strong and where they remain bounded",
        "Phase 14 redesigned Figure 5: a hierarchy separates whole-atlas limits from targeted and local guardrail evidence.",
    )
    stem = FIG / "phase14_figure5_fdr_guardrails_redesign"
    save_figure(fig, stem)
    write_manifest(stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "fdr_layer": P13 / "figure5_fdr_layer_compact.csv",
        "targeted": P13 / "figure5_targeted_dataset.csv",
        "decoy": P13 / "figure5_target_decoy_topk_compact.csv",
        "hard": P13 / "figure5_hard_negative_dataset.csv",
        "coord": P13 / "figure5_coordinate_guardrail.csv",
        "claim": P13 / "figure5_claim_matrix_compact.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 5 sources: " + "; ".join(missing))
    return {k: pd.read_csv(p) for k, p in files.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fdr = data["fdr_layer"].copy()
    hierarchy = pd.DataFrame(
        [
            {
                "layer": "whole-atlas permutation",
                "claim_strength": "limited",
                "message": "conservative; not solved",
                "risk_level": 3,
            },
            {
                "layer": "targeted panel",
                "claim_strength": "story support",
                "message": "selected anchors tested",
                "risk_level": 2,
            },
            {
                "layer": "matched null",
                "claim_strength": "stronger story support",
                "message": "matched controls reduce expression bias",
                "risk_level": 1,
            },
            {
                "layer": "target-decoy top-k",
                "claim_strength": "ranking guardrail",
                "message": "control-prefix FDR is audited",
                "risk_level": 2,
            },
            {
                "layer": "coordinate / hard-negative",
                "claim_strength": "failure-mode guardrail",
                "message": "spatial and high-expression artifacts audited",
                "risk_level": 1,
            },
        ]
    )

    targeted = fdr[["dataset_key", "n_whole_atlas_fdr_0_1", "n_targeted_panel_q_0_1", "n_matched_control_q_0_1", "n_strong_targeted_matched_support"]].copy()

    decoy = data["decoy"].copy()
    decoy = decoy[decoy["top_k"].isin([10, 100])].copy()

    hard = data["hard"].copy()
    keep_cols = [c for c in ["dataset_key", "negative_comparison_beat_fraction", "n_positive_lr", "n_negative_controls"] if c in hard.columns]
    hard = hard[keep_cols].copy() if keep_cols else hard.copy()

    claim = data["claim"].copy()
    claim_boundary = pd.DataFrame(
        [
            {
                "claim": "selected story LR",
                "status": "supported with targeted/local guardrails",
                "write_as": "observational prioritized CCC",
            },
            {
                "claim": "high-expression false positives",
                "status": "audited and reduced",
                "write_as": "major failure mode is suppressed",
            },
            {
                "claim": "strict whole-atlas FDR",
                "status": "not fully closed",
                "write_as": "conservatively bounded limitation",
            },
        ]
    )

    return {
        "hierarchy": hierarchy,
        "targeted": targeted,
        "decoy": decoy,
        "hard_negative": hard,
        "claim_boundary": claim_boundary,
        "claim_matrix_source": claim,
    }


def export_source(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(SOURCE / f"figure5_{name}.csv", index=False)


def draw_evidence_hierarchy(ax: plt.Axes, hierarchy: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.06, y=1.03)
    ax.set_axis_off()
    ax.set_title("Evidence hierarchy, not a single overclaim", loc="left", pad=8)
    y_positions = np.linspace(0.82, 0.18, len(hierarchy))
    colors = {1: PALETTE["signal"], 2: PALETTE["weak"], 3: PALETTE["risk"]}
    for y, (_, row) in zip(y_positions, hierarchy.iterrows()):
        risk = int(row["risk_level"])
        color = colors[risk]
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.08, y - 0.045),
                0.78,
                0.082,
                boxstyle="round,pad=0.014,rounding_size=0.02",
                facecolor=PALETTE["soft_bg"],
                edgecolor=color,
                linewidth=1.0,
                transform=ax.transAxes,
            )
        )
        ax.scatter(0.08, y, s=70, color=color, edgecolor="white", linewidth=0.8, transform=ax.transAxes, zorder=3)
        ax.text(0.15, y + 0.014, row["layer"], transform=ax.transAxes, ha="left", va="center", fontsize=6.7, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.15, y - 0.021, row["message"], transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color=PALETTE["muted"])
    ax.annotate("claim strength increases\nwhen nulls become matched/local", xy=(0.91, 0.78), xytext=(0.91, 0.22), xycoords=ax.transAxes, textcoords=ax.transAxes, ha="center", va="center", fontsize=5.8, color=PALETTE["muted"], arrowprops=dict(arrowstyle="<|-", color=PALETTE["line"], lw=0.9))


def draw_targeted(ax: plt.Axes, targeted: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.12, y=1.04)
    ax.set_title("Targeted and matched support", loc="left", pad=8)
    df = targeted.copy()
    y = np.arange(len(df))[::-1]
    cols = [
        ("n_whole_atlas_fdr_0_1", "whole", PALETTE["neutral"]),
        ("n_targeted_panel_q_0_1", "targeted", PALETTE["spatialself"]),
        ("n_matched_control_q_0_1", "matched", PALETTE["signal"]),
    ]
    left = np.zeros(len(df))
    for col, label, color in cols:
        vals = df[col].to_numpy(float)
        ax.barh(y, vals, left=left, height=0.5, color=color, edgecolor="white", linewidth=0.5, label=label)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]])
    ax.set_xlabel("anchor LR count")
    clean_axes(ax, grid="x")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right")


def draw_decoy(ax: plt.Axes, decoy: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.12, y=1.04)
    ax.set_title("Target-decoy top-k FDR", loc="left", pad=8)
    for dataset, sub in decoy.groupby("dataset_key"):
        sub = sub.sort_values("top_k")
        ax.plot(sub["top_k"], sub["primary_control_prefix_fdr"], marker="o", lw=1.3, label=DATASET_LABEL.get(dataset, dataset).replace("\n", " "))
    ax.axhline(0.1, color=PALETTE["line"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("top-k")
    ax.set_ylabel("control-prefix FDR")
    ax.set_ylim(0, max(0.42, float(decoy["primary_control_prefix_fdr"].max()) * 1.2))
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right")


def draw_hard_negative(ax: plt.Axes, hard: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.12, y=1.04)
    ax.set_title("Hard-negative stress remains explicit", loc="left", pad=8)
    df = hard.copy()
    if "negative_comparison_beat_fraction" not in df.columns:
        ax.text(0.5, 0.5, "hard-negative summary\nsource columns unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = np.arange(len(df))
    ax.bar(x, df["negative_comparison_beat_fraction"], color=PALETTE["risk_soft"], edgecolor=PALETTE["risk"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(k, k) for k in df["dataset_key"]], rotation=25, ha="right")
    ax.set_ylabel("negative beat fraction")
    ax.set_ylim(0, 1.05)
    clean_axes(ax)


def draw_claim_boundary(ax: plt.Axes, claim: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.12, y=1.04)
    ax.set_axis_off()
    ax.set_title("Claim boundary", loc="left", pad=8)
    y = 0.78
    for _, row in claim.iterrows():
        is_limit = "whole-atlas" in row["claim"]
        color = PALETTE["risk"] if is_limit else PALETTE["signal"]
        ax.scatter(0.06, y, s=55, color=color, edgecolor="white", linewidth=0.7, transform=ax.transAxes)
        ax.text(0.14, y + 0.025, row["claim"], transform=ax.transAxes, ha="left", va="center", fontsize=6.5, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.14, y - 0.025, row["write_as"], transform=ax.transAxes, ha="left", va="center", fontsize=5.7, color=PALETTE["muted"])
        y -= 0.25


def write_manifest(stem: Path) -> None:
    rows = [
        ("a", "FDR evidence is represented as a hierarchy rather than a single universal claim.", "outputs/phase14/source_data/figure5_hierarchy.csv"),
        ("b", "Targeted and matched-null support are separated from whole-atlas FDR.", "outputs/phase14/source_data/figure5_targeted.csv"),
        ("c", "Target-decoy control-prefix FDR audits top-k rankings.", "outputs/phase14/source_data/figure5_decoy.csv"),
        ("d", "Hard-negative stress is kept explicit instead of hidden.", "outputs/phase14/source_data/figure5_hard_negative.csv"),
        ("e", "Claim boundary states what can and cannot be written.", "outputs/phase14/source_data/figure5_claim_boundary.csv"),
    ]
    pd.DataFrame(rows, columns=["panel", "claim", "source_data"]).to_csv(f"{stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
