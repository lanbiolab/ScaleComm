#!/usr/bin/env python
"""Render Phase 12 Figure 3: second-tissue validation and directionality audit."""

from __future__ import annotations

from pathlib import Path

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
    "positive": "#2E9E44",
    "positive_soft": "#6EA6CF",
    "negative": "#B64342",
    "gold": "#C69C3D",
    "grey": "#D9D9D9",
    "grey_mid": "#8F8F8F",
    "text": "#272727",
    "grid": "#D8D8D8",
}


METHOD_ORDER = [
    "spatialself_wavelet_edge_prior",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
]

METHOD_LABEL = {
    "spatialself_wavelet_edge_prior": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
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
        "ln_story": pd.read_csv(SOURCE / "figure3_lymph_node_second_story_evidence.csv"),
        "strict_story": pd.read_csv(SOURCE / "figure3_strict_story_candidate_evidence.csv"),
        "ln_dir": pd.read_csv(SOURCE / "figure3_lymph_node_directionality_lr_statistics.csv"),
        "ln_dir_perm": pd.read_csv(SOURCE / "figure3_lymph_node_directionality_label_permutation.csv"),
        "ln_dir_edge": pd.read_csv(SOURCE / "figure3_lymph_node_directionality_edge_permutation.csv"),
        "ln_program": pd.read_csv(SOURCE / "figure3_lymph_node_program_specificity.csv"),
        "platform": pd.read_csv(SOURCE / "figure3_cross_platform_second_tissue_audit.csv"),
        "tonsil_rank": pd.read_csv(SOURCE / "figure3_tonsil_A1_method_lr_rank.csv"),
        "tonsil_fdr": pd.read_csv(SOURCE / "figure3_tonsil_A1_local_contrast_fdr.csv"),
    }


def draw_story_triage(ax: plt.Axes, strict_story: pd.DataFrame) -> None:
    keep_classes = [
        "A_main_story_ready",
        "B_biological_consistency_with_stat_support",
        "C_targeted_story_candidate_missing_functional_closure",
    ]
    df = strict_story[strict_story["strict_story_class"].isin(keep_classes)].copy()
    class_rank = {name: rank for rank, name in enumerate(keep_classes)}
    df["_class_rank"] = df["strict_story_class"].map(class_rank).fillna(99)
    df["_score"] = pd.to_numeric(df["evidence_score"], errors="coerce").fillna(0)
    df = df.sort_values(["_class_rank", "_score", "coordinate_score"], ascending=[True, False, False])
    df["display_label"] = df.apply(
        lambda row: f"{row['lr_pair']} ({'colon' if row['dataset_key'] == 'post_xenium_colon_full' else 'LN'})",
        axis=1,
    )
    y = np.arange(len(df))[::-1]

    class_color = {
        "A_main_story_ready": COL["positive"],
        "B_biological_consistency_with_stat_support": COL["spatialself"],
        "C_targeted_story_candidate_missing_functional_closure": COL["gold"],
        "D_supportive_but_not_main_story": COL["grey_mid"],
        "E_not_story_ready": COL["negative"],
    }
    colors = [class_color.get(str(cls), COL["grey_mid"]) for cls in df["strict_story_class"]]
    scores = df["_score"]
    ax.barh(y, scores, height=0.54, color=colors, edgecolor="white", linewidth=0.8)

    for yi, score_val, row in zip(y, scores.values, df.itertuples(index=False)):
        dataset_label = "colon" if row.dataset_key == "post_xenium_colon_full" else "lymph node"
        target_q = pd.to_numeric(pd.Series([row.downstream_target_fdr]), errors="coerce").iloc[0]
        dir_q = pd.to_numeric(pd.Series([row.direction_receiver_pairing_q]), errors="coerce").iloc[0]
        suffix = []
        if np.isfinite(target_q):
            suffix.append(f"target q={target_q:.3g}")
        if np.isfinite(dir_q):
            suffix.append(f"edge q={dir_q:.3g}")
        suffix_text = "; ".join(suffix) if suffix else "functional gap"
        ax.text(
            score_val + 0.12,
            yi,
            f"{dataset_label}; SS r{int(row.spatialself_rank)}; base<=10 {int(row.baseline_top10_count)}; {suffix_text}",
            va="center",
            ha="left",
            fontsize=5.35,
            color="#555555",
        )

    ax.set_xlim(0, 8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display_label"], fontsize=6.8)
    ax.set_xlabel("integrated evidence score", labelpad=3)
    ax.set_title("Strict triage separates main and support stories", loc="left", pad=8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "a")
    ax.text(
        0.0,
        -0.24,
        "Green=A main; blue=B consistency; gold=C incomplete. B-class is not SpatialSelf-exclusive.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#666666",
    )


def draw_directionality(ax: plt.Axes, direction: pd.DataFrame, permutation: pd.DataFrame, edge_perm: pd.DataFrame) -> None:
    df = direction[direction["dataset_key"] == "lymph_node_full"].copy()
    perm = permutation[permutation["dataset_key"] == "lymph_node_full"].copy()
    edge = edge_perm[edge_perm["dataset_key"] == "lymph_node_full"].copy()
    keep = ["CCL21-CCR7", "CCL19-CCR7", "CD48-CD244", "CXCL13-CXCR5"]
    df = df[df["lr_pair"].isin(keep)]
    pivot = df.pivot(index="lr_pair", columns="method", values="expected_flow_score_fraction").reindex(keep)
    q = df.pivot(index="lr_pair", columns="method", values="expected_flow_bh_fdr").reindex(keep)
    q_perm = perm.pivot(index="lr_pair", columns="method", values="niche_label_permutation_bh_fdr").reindex(keep)
    q_edge = edge.pivot(index="lr_pair", columns="method", values="receiver_pairing_bh_fdr").reindex(keep) if not edge.empty else pd.DataFrame()
    q_vec = edge.pivot(index="lr_pair", columns="method", values="edge_vector_bh_fdr").reindex(keep) if not edge.empty else pd.DataFrame()
    y = np.arange(len(keep))[::-1]
    height = 0.32
    ax.barh(y + height / 2, pivot["spatialself"], height=height, color=COL["spatialself"], label="SpatialSelf")
    ax.barh(y - height / 2, pivot["commot"], height=height, color=COL["commot"], label="COMMOT")
    for yi, lr in zip(y, keep):
        for xval, dy, method in [
            (pivot.loc[lr, "spatialself"], height / 2, "spatialself"),
            (pivot.loc[lr, "commot"], -height / 2, "commot"),
        ]:
            qq = q.loc[lr, method]
            qn = q_perm.loc[lr, method] if method in q_perm.columns and lr in q_perm.index else np.nan
            qe = q_edge.loc[lr, method] if method in q_edge.columns and lr in q_edge.index else np.nan
            qv = q_vec.loc[lr, method] if method in q_vec.columns and lr in q_vec.index else np.nan
            label = f"q={qq:.2f}"
            if np.isfinite(qn) and qn <= 0.1:
                label += f" | perm {qn:.2f}"
            if np.isfinite(qe) and qe <= 0.1:
                label += f" | edge {qe:.2f}"
            if np.isfinite(qv) and qv <= 0.1:
                label += f" | vec {qv:.2f}"
            ax.text(xval + 0.025, yi + dy, label, ha="left", va="center", fontsize=5.6, color="#555555")

    ax.set_xlim(0, 1.12)
    ax.set_yticks(y)
    ax.set_yticklabels(keep, fontsize=6.8)
    ax.set_xlabel("expected source-sink flow fraction", labelpad=3)
    ax.set_title("Directionality has empirical, label and edge-level nulls", loc="left", pad=8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=5.8)
    ax.text(
        0.0,
        -0.24,
        "perm: aggregated niche-label q; edge: fixed-score receiver-pairing q; vec: edge-vector q.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#666666",
    )
    panel_label(ax, "b")


def draw_program_specificity(ax: plt.Axes, program: pd.DataFrame) -> None:
    order = ["CXCL13-CXCR5", "CCL21-CCR7", "CCL19-CCR7", "CD48-CD244"]
    df = program.set_index("lr_pair").reindex(order).reset_index()
    y = np.arange(len(df))[::-1]
    specific = df["n_specific_significant_terms_fdr_0_05"].astype(float).values
    broad = df["n_broad_significant_terms_fdr_0_05"].astype(float).values
    ax.barh(y, broad, height=0.58, color="#D2D2D2", edgecolor="white", label="broad housekeeping")
    ax.barh(y, specific, height=0.58, color="#6EA6CF", edgecolor="white", label="specific program")
    ax.set_yticks(y)
    ax.set_yticklabels(df["lr_pair"], fontsize=6.8)
    ax.set_xlabel("significant enrichment terms", labelpad=3)
    ax.set_title("Receiver program specificity", loc="left", pad=8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=5.8)
    panel_label(ax, "c")


def draw_cross_platform_audit(ax: plt.Axes, platform: pd.DataFrame) -> None:
    keep = ["spamosaic_tonsil_A1", "spamosaic_tonsil_D1", "xenium_human_lymph_node"]
    df = platform[platform["dataset_key"].isin(keep)].set_index("dataset_key").reindex(keep).reset_index()
    labels = ["tonsil A1", "tonsil D1", "Xenium LN"]
    y = np.arange(len(df))[::-1]

    positive = pd.to_numeric(df["positive_top3"], errors="coerce").fillna(0).values / 3.0
    negative = pd.to_numeric(df["negative_top3"], errors="coerce").fillna(0).values / 3.0
    targeted_leak = pd.to_numeric(df["n_targeted_negative_leakage"], errors="coerce").fillna(0).values / 3.0

    ax.barh(y + 0.22, positive, height=0.18, color=COL["positive"], label="positive top3")
    ax.barh(y, negative, height=0.18, color=COL["negative"], label="negative top3")
    ax.barh(y - 0.22, targeted_leak, height=0.18, color="#D6A13A", label="targeted leakage")
    for yi, row in zip(y, df.itertuples(index=False)):
        best_pos = row.best_positive_control if isinstance(row.best_positive_control, str) else "NA"
        best_neg = row.best_negative_control if isinstance(row.best_negative_control, str) else "NA"
        score = float(row.story_score) if pd.notna(row.story_score) else 0.0
        ax.text(
            1.04,
            yi,
            f"{best_pos} vs {best_neg}; score {score:.0f}",
            va="center",
            ha="left",
            fontsize=5.5,
            color="#555555",
        )

    ax.set_xlim(0, 1.48)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("fraction of three controls", labelpad=3)
    ax.set_title("Cross-platform support remains caveated", loc="left", pad=8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=3, fontsize=5.6, columnspacing=0.7)
    ax.text(
        0.0,
        -0.24,
        "Tonsil supports ranking/local FDR but has targeted negative leakage; Xenium LN remains mixed.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#666666",
    )
    panel_label(ax, "d")


def render() -> None:
    apply_style()
    data = load_data()

    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.05],
        left=0.105,
        right=0.985,
        top=0.84,
        bottom=0.11,
        wspace=0.52,
        hspace=0.58,
    )
    fig.text(
        0.105,
        0.982,
        "Second-tissue validation supports biological consistency with explicit caveats",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COL["text"],
    )
    fig.text(
        0.105,
        0.948,
        "Strict triage benchmarks lymph-node and tonsil support against the A-class colon story, with permutation checks and explicit negative-competition caveats.",
        ha="left",
        va="top",
        fontsize=7.1,
        color="#5A5A5A",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_story_triage(ax_a, data["strict_story"])
    draw_directionality(ax_b, data["ln_dir"], data["ln_dir_perm"], data["ln_dir_edge"])
    draw_program_specificity(ax_c, data["ln_program"])
    draw_cross_platform_audit(ax_d, data["platform"])

    basename = OUTDIR / "phase12_figure3_second_tissue_validation"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {"figure": "Figure 3", "panel": "a", "source_data": "figure3_lymph_node_second_story_evidence.csv; figure3_strict_story_candidate_evidence.csv; figure3_strict_story_candidate_dataset_summary.csv", "message": "Strict story triage separates the A-class SPP1-CD44 main story from B-class lymph-node biological-consistency candidates and C-class incomplete candidates."},
            {"figure": "Figure 3", "panel": "b", "source_data": "figure3_lymph_node_directionality_lr_statistics.csv; figure3_lymph_node_directionality_label_permutation.csv; figure3_lymph_node_directionality_edge_permutation.csv", "message": "Directionality is quantified with empirical FDR, aggregated label-permutation checks and edge-level fixed-score receiver-pairing nulls, but does not establish global superiority over COMMOT."},
            {"figure": "Figure 3", "panel": "c", "source_data": "figure3_lymph_node_program_specificity.csv", "message": "Program specificity is mixed and includes broad transcriptional signatures."},
            {"figure": "Figure 3", "panel": "d", "source_data": "figure3_cross_platform_second_tissue_audit.csv", "message": "Tonsil A1/D1 support cross-platform ranking but retain targeted negative leakage; Xenium lymph node remains mixed."},
        ]
    ).to_csv(OUTDIR / "phase12_figure3_second_tissue_validation_manifest.csv", index=False)


if __name__ == "__main__":
    render()
