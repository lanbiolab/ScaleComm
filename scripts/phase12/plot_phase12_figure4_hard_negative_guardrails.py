#!/usr/bin/env python
"""Render Phase 12 Figure 4: hard-negative stress, modules, and claim guardrails."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    "blue": "#0F4D92",
    "blue_light": "#78A9CF",
    "green": "#2E9E44",
    "red": "#B64342",
    "red_light": "#E2AAA6",
    "grey": "#D9D9D9",
    "grey_mid": "#8F8F8F",
    "text": "#272727",
    "grid": "#D8D8D8",
    "gold": "#C69C3D",
}


DATASET_ORDER = [
    "post_xenium_colon_full",
    "human_lymph_node_visium_raw_full",
    "visium_hd_crc_full",
]

DATASET_LABEL = {
    "post_xenium_colon_full": "post-Xenium\ncolon",
    "human_lymph_node_visium_raw_full": "lymph node\nVisium",
    "visium_hd_crc_full": "Visium HD\nCRC",
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


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "dataset": pd.read_csv(SOURCE / "figure4_hard_negative_dataset_summary.csv"),
        "positive": pd.read_csv(SOURCE / "figure4_hard_negative_positive_summary.csv"),
        "comparisons": pd.read_csv(SOURCE / "figure4_hard_negative_comparisons.csv"),
        "edge_hard_dataset": pd.read_csv(SOURCE / "figure4_edge_hard_negative_dataset_summary.csv"),
        "edge_hard_class": pd.read_csv(SOURCE / "figure4_edge_hard_negative_class_summary.csv"),
        "modules": pd.read_csv(SOURCE / "figure4_communication_modules.csv"),
        "program": pd.read_csv(SOURCE / "figure4_program_specificity_all.csv"),
        "pathway_specific": pd.read_csv(SOURCE / "figure4_pathway_specific_dataset_summary.csv"),
        "readiness": pd.read_csv(SOURCE / "figure4_directionality_permutation_readiness.csv"),
        "direction_perm": pd.read_csv(SOURCE / "figure4_directionality_label_permutation_dataset_summary.csv"),
        "direction_edge": pd.read_csv(SOURCE / "figure4_directionality_edge_permutation_dataset_summary.csv"),
        "coordinate_smoke": pd.read_csv(SOURCE / "figure4_coordinate_model_rerun_smoke_summary.csv"),
        "coordinate_guardrail": pd.read_csv(SOURCE / "figure4_coordinate_control_guardrail_dataset_summary.csv"),
        "targeted": pd.read_csv(SOURCE / "figure4_targeted_matched_null_lr_audit.csv"),
        "claims": pd.read_csv(SOURCE / "figure4_claim_matrix_guardrails.csv"),
    }


def draw_dataset_stress(ax: plt.Axes, dataset: pd.DataFrame) -> None:
    df = dataset.set_index("dataset_key").reindex(DATASET_ORDER).reset_index()
    x = np.arange(len(df))
    beat = df["negative_comparison_beat_fraction"].astype(float).values
    rank_pass = df["n_positive_rank_score_pass"].astype(float).values / df["n_positive_controls"].astype(float).values
    downstream_pass = df["n_positive_downstream_pass"].astype(float).values / df["n_positive_controls"].astype(float).values

    ax.bar(x, beat, width=0.56, color=COL["red_light"], edgecolor="white", linewidth=0.8, label="negative beats positive")
    ax.plot(x, rank_pass, marker="o", color=COL["blue"], lw=1.3, label="rank-score pass")
    ax.plot(x, downstream_pass, marker="o", color=COL["green"], lw=1.3, label="downstream pass")
    for xi, b in zip(x, beat):
        ax.text(xi, b + 0.025, f"{b:.2f}", ha="center", va="bottom", fontsize=6.2)

    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], fontsize=6.6)
    ax.set_ylabel("fraction", labelpad=3)
    ax.grid(axis="y", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Hard-negative stress exposes dataset-specific leakage", loc="left", pad=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.01), fontsize=5.7)
    panel_label(ax, "a")


def draw_edge_guardrail(ax: plt.Axes, edge_dataset: pd.DataFrame) -> None:
    df = edge_dataset.copy()
    df = df[df["dataset_key"].isin(DATASET_ORDER)]
    order = [d for d in DATASET_ORDER if d in set(df["dataset_key"])]
    x = np.arange(len(order))
    width = 0.34
    raw = (
        df[df["score_mode"].eq("raw")]
        .set_index("dataset_key")
        .reindex(order)["mean_leakage_fraction"]
        .astype(float)
        .values
    )
    guarded = (
        df[df["score_mode"].eq("edge_guardrail")]
        .set_index("dataset_key")
        .reindex(order)["mean_leakage_fraction"]
        .astype(float)
        .values
    )
    raw_counts = df[df["score_mode"].eq("raw")].set_index("dataset_key").reindex(order)
    guarded_counts = df[df["score_mode"].eq("edge_guardrail")].set_index("dataset_key").reindex(order)

    ax.bar(x - width / 2, raw, width=width, color=COL["red_light"], edgecolor="white", linewidth=0.8, label="raw score")
    ax.bar(x + width / 2, guarded, width=width, color=COL["blue"], edgecolor="white", linewidth=0.8, label="edge guardrail")
    for xi, r, g, (_, rr), (_, gr) in zip(x, raw, guarded, raw_counts.iterrows(), guarded_counts.iterrows()):
        ax.plot([xi - width / 2, xi + width / 2], [r, g], color=COL["grey_mid"], lw=0.75, zorder=0)
        ax.text(xi - width / 2, r + 0.025, f"{int(rr.n_leakage_lr_class_tests)}/{int(rr.n_lr_class_tests)}", ha="center", va="bottom", fontsize=5.9)
        ax.text(xi + width / 2, g + 0.025, f"{int(gr.n_leakage_lr_class_tests)}/{int(gr.n_lr_class_tests)}", ha="center", va="bottom", fontsize=5.9)

    ax.set_ylim(0, 0.68)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in order], fontsize=6.6)
    ax.set_ylabel("mean leakage fraction", labelpad=3)
    ax.grid(axis="y", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Diagnostic edge guardrail reduces aggregate leakage", loc="left", pad=8)
    ax.legend(loc="upper right", fontsize=5.8)
    ax.text(
        0.0,
        -0.24,
        "Leaking LR-class tests / all tests; post-hoc diagnostic, not retraining.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#666666",
    )
    panel_label(ax, "b")


def draw_pathway_specificity(ax: plt.Axes, pathway_specific: pd.DataFrame) -> None:
    df = pathway_specific.set_index("dataset_key").reindex(DATASET_ORDER)
    classes = [
        ("n_clean_specific_support", "clean specific"),
        ("n_mixed_specific_support_broad_dominant", "broad-caveated"),
        ("n_negative_specific_leakage", "negative leakage"),
    ]
    colors = [COL["green"], COL["gold"], COL["red"]]
    x = np.arange(len(DATASET_ORDER))
    bottom = np.zeros(len(DATASET_ORDER))
    for column, label, color in zip([c[0] for c in classes], [c[1] for c in classes], colors):
        vals = pd.to_numeric(df[column], errors="coerce").fillna(0).values
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, width=0.62, label=label)
        for xi, val, base in zip(x, vals, bottom):
            if val > 0:
                ax.text(xi, base + val / 2, f"{int(val)}", ha="center", va="center", fontsize=6.2, color="white" if color == COL["red"] else COL["text"])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], fontsize=6.6)
    ax.set_ylabel("program calls", labelpad=3)
    ax.set_title("Pathway-specific downstream support\nis dataset-dependent", loc="left", pad=8)
    ax.grid(axis="y", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=5.7)
    panel_label(ax, "c")


def draw_guardrails(
    ax: plt.Axes,
    claims: pd.DataFrame,
    readiness: pd.DataFrame,
    direction_perm: pd.DataFrame,
    direction_edge: pd.DataFrame,
    coordinate_smoke: pd.DataFrame,
    coordinate_guardrail: pd.DataFrame,
    targeted: pd.DataFrame,
) -> None:
    df = claims.copy()

    def claim_group(status: str) -> str:
        s = str(status)
        if "not_supported" in s or "not_yet" in s or "do_not" in s or "insufficient" in s:
            return "not yet"
        if "caveat" in s or "mixed" in s or "partial" in s or "supportive" in s or "empirical" in s:
            return "qualified"
        return "supported"

    df["group"] = df["support_status"].map(claim_group)
    counts = df["group"].value_counts().reindex(["supported", "qualified", "not yet"]).fillna(0)
    colors = [COL["blue"], COL["gold"], COL["red"]]
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, color=colors, edgecolor="white", linewidth=0.8, height=0.58)
    for yi, val in zip(y, counts.values):
        ax.text(val + 0.35, yi, f"{int(val)} claims", va="center", ha="left", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=6.8)
    ax.set_xlim(0, max(12, counts.max() + 2.2))
    ax.set_xlabel("claim-matrix count", labelpad=3)
    ax.set_title("Claim guardrails", loc="left", pad=8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    label_audits = len(direction_perm)
    both_supported = int(direction_perm["n_both_fdr_0_1"].sum()) if "n_both_fdr_0_1" in direction_perm else 0
    edge_receiver = int(direction_edge["n_receiver_pairing_fdr_0_1"].sum()) if "n_receiver_pairing_fdr_0_1" in direction_edge else 0
    edge_vector = int(direction_edge["n_edge_vector_fdr_0_1"].sum()) if "n_edge_vector_fdr_0_1" in direction_edge else 0
    coordinate_datasets = len(coordinate_smoke)
    coordinate_reruns = int(pd.to_numeric(coordinate_smoke["n_completed_permutations"], errors="coerce").fillna(0).sum()) if not coordinate_smoke.empty else 0
    coordinate_min_p = (
        pd.to_numeric(coordinate_smoke[["best_positive_p", "best_negative_p"]].stack(), errors="coerce").min()
        if not coordinate_smoke.empty
        else np.nan
    )
    coordinate_q = int(
        pd.to_numeric(coordinate_smoke["n_positive_p_le_0_1"], errors="coerce").fillna(0).sum()
        + pd.to_numeric(coordinate_smoke["n_negative_p_le_0_1"], errors="coerce").fillna(0).sum()
    ) if not coordinate_smoke.empty else 0
    story_guardrail = (
        int(pd.to_numeric(coordinate_guardrail["n_story_guardrail_discoveries"], errors="coerce").fillna(0).sum())
        if not coordinate_guardrail.empty
        else 0
    )
    story_guardrail_high = (
        int(
            pd.to_numeric(
                coordinate_guardrail["n_story_guardrail_high_expression_negative"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        if not coordinate_guardrail.empty
        else 0
    )
    if not coordinate_guardrail.empty and "n_control_guardrail_broad_background" in coordinate_guardrail:
        broad_background_leakage = int(
            pd.to_numeric(coordinate_guardrail["n_control_guardrail_broad_background"], errors="coerce").fillna(0).sum()
        )
    else:
        broad_background_leakage = 0
    targeted_supported = int(
        targeted["targeted_validation_call"].astype(str).isin(
            ["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"]
        ).sum()
    )
    strong_targeted = int(targeted["targeted_validation_call"].astype(str).eq("strong_targeted_matched_support").sum())
    whole_supported = int(pd.to_numeric(targeted["whole_atlas_bh_fdr"], errors="coerce").le(0.1).sum())
    ax.text(
        0.02,
        -0.28,
        f"Label null: both q<=0.1 LR={both_supported}; fixed-score edge null: receiver={edge_receiver}, vector={edge_vector}.\n"
        f"Coordinate rerun: {coordinate_datasets} datasets/{coordinate_reruns} reruns; nominal hits contaminated.\n"
        f"Story guardrail: {story_guardrail} LR, highNeg={story_guardrail_high}; broad-bg leakage={broad_background_leakage}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.65,
        color="#666666",
    )
    panel_label(ax, "d")


def render() -> None:
    apply_style()
    data = load_data()
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.105,
        right=0.985,
        top=0.84,
        bottom=0.15,
        wspace=0.52,
        hspace=0.72,
    )
    fig.text(
        0.105,
        0.982,
        "Hard-negative audits define where SpatialSelf is supported and where it remains vulnerable",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COL["text"],
    )
    fig.text(
        0.105,
        0.948,
        "Phase 12 explicitly quantifies receptor/shared-expression artifacts, program leakage, label-permutation limits, and unsupported claim boundaries.",
        ha="left",
        va="top",
        fontsize=7.1,
        color="#5A5A5A",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_dataset_stress(ax_a, data["dataset"])
    draw_edge_guardrail(ax_b, data["edge_hard_dataset"])
    draw_pathway_specificity(ax_c, data["pathway_specific"])
    draw_guardrails(
        ax_d,
        data["claims"],
        data["readiness"],
        data["direction_perm"],
        data["direction_edge"],
        data["coordinate_smoke"],
        data["coordinate_guardrail"],
        data["targeted"],
    )

    basename = OUTDIR / "phase12_figure4_hard_negative_guardrails"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {"figure": "Figure 4", "panel": "a", "source_data": "figure4_hard_negative_dataset_summary.csv", "message": "Dataset-level hard-negative leakage and positive-control pass rates are quantified."},
            {"figure": "Figure 4", "panel": "b", "source_data": "figure4_edge_hard_negative_dataset_summary.csv; figure4_edge_hard_negative_class_summary.csv", "message": "A diagnostic edge guardrail reduces aggregate leakage but remains post-hoc and class-specific."},
            {"figure": "Figure 4", "panel": "c", "source_data": "figure4_pathway_specific_dataset_summary.csv; figure4_pathway_specific_candidate_programs.csv", "message": "Pathway-specific downstream support is clean in post-Xenium colon, broad-caveated in lymph node, and leakage-confounded in Visium HD CRC."},
            {"figure": "Figure 4", "panel": "d", "source_data": "figure4_claim_matrix_guardrails.csv; figure4_directionality_permutation_readiness.csv; figure4_directionality_label_permutation_dataset_summary.csv; figure4_directionality_edge_permutation_dataset_summary.csv; figure4_coordinate_model_rerun_smoke_summary.csv; figure4_coordinate_control_guardrail_dataset_summary.csv; figure4_targeted_matched_null_lr_audit.csv", "message": "Claim guardrails separate targeted validation, label/edge permutation checks, coordinate model-rerun smoke, and post-hoc high-expression-control guardrails."},
        ]
    ).to_csv(OUTDIR / "phase12_figure4_hard_negative_guardrails_manifest.csv", index=False)


if __name__ == "__main__":
    render()
