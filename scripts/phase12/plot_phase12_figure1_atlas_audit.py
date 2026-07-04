#!/usr/bin/env python
"""Render Phase 12 Figure 1: expanded whole-LR atlas audit."""

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
    "signal": "#0F4D92",
    "signal_soft": "#AFCBE3",
    "risk": "#B64342",
    "risk_soft": "#E7B3AD",
    "neutral": "#B8B8B8",
    "neutral_soft": "#E9E9E9",
    "neutral_faint": "#F6F6F6",
    "text": "#272727",
    "muted": "#666666",
    "grid": "#D8D8D8",
    "gold": "#C69C3D",
    "teal": "#42949E",
}


DATASET_ORDER = [
    "human_lymph_node_visium_raw_full",
    "post_xenium_colon_full",
]

DATASET_LABEL = {
    "human_lymph_node_visium_raw_full": "lymph node\nVisium",
    "post_xenium_colon_full": "post-Xenium\ncolon",
}

METHOD_ORDER = [
    "spatialself_calibrated",
    "rule_expression_distance",
    "coexpression_distance_mean",
    "coexpression_distance_q95",
    "coexpression_only",
    "expression_support_fraction",
    "high_expression_naive",
    "distance_type_rule_prior",
    "spatialself_raw_mean",
]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.1, y: float = 1.04) -> None:
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


def clean_label(text: str, width: int = 23) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "calibrated_dataset": pd.read_csv(SOURCE / "figure1_expanded_calibrated_dataset_summary.csv"),
        "calibrated_topk": pd.read_csv(SOURCE / "figure1_expanded_calibrated_topk_summary.csv"),
        "calibrated_ranking": pd.read_csv(SOURCE / "figure1_expanded_calibrated_lr_ranking_top25.csv"),
        "permutation_dataset": pd.read_csv(SOURCE / "figure1_expanded_permutation_dataset_summary.csv"),
        "permutation_topk": pd.read_csv(SOURCE / "figure1_expanded_permutation_topk_summary.csv"),
        "target_decoy_dataset": pd.read_csv(SOURCE / "figure1_expanded_target_decoy_dataset_summary.csv"),
        "targeted_dataset": pd.read_csv(SOURCE / "figure1_targeted_matched_null_dataset_summary.csv"),
        "targeted_lr": pd.read_csv(SOURCE / "figure1_targeted_matched_null_supported_lr.csv"),
        "baseline_method": pd.read_csv(SOURCE / "figure1_expanded_same_universe_baseline_method_summary.csv"),
        "official_adapter_method": pd.read_csv(SOURCE / "figure1_expanded_official_adapter_method_summary.csv"),
        "official_status": pd.read_csv(SOURCE / "figure1_expanded_official_baseline_status.csv"),
    }


def draw_metric_strip(ax: plt.Axes, summary: pd.DataFrame) -> None:
    df = summary.set_index("dataset_key").reindex(DATASET_ORDER).reset_index()
    metrics = [
        ("n_lr", "LR\nuniverse", "count", 1000, "neutral"),
        ("top10_true_lr", "true LR\nin top 10", "int", 10, "signal"),
        ("top10_high_expression_negative", "high-expr\nnegatives", "int", 10, "risk"),
        ("top10_zero_expr_lr", "zero-expr\nLR", "int", 10, "risk"),
        ("top10_negative_or_risk_fraction", "top-10\nrisk", "frac", 1, "risk"),
        ("hard_negative_best_comparison_beat_fraction", "hard-neg\nbeat rate", "frac", 1, "risk"),
    ]

    ax.set_xlim(-0.65, len(metrics) - 0.35)
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_facecolor("white")

    for y, row in enumerate(df.itertuples(index=False)):
        for x, (column, _label, fmt, vmax, family) in enumerate(metrics):
            value = float(getattr(row, column))
            if family == "signal":
                shade = min(value / vmax, 1.0)
                color = mpl.colors.to_hex(
                    np.array(mpl.colors.to_rgb(COL["signal_soft"])) * (1 - shade)
                    + np.array(mpl.colors.to_rgb(COL["signal"])) * shade
                )
            elif family == "risk":
                shade = min(value / vmax, 1.0)
                color = mpl.colors.to_hex(
                    np.array(mpl.colors.to_rgb(COL["neutral_faint"])) * (1 - shade)
                    + np.array(mpl.colors.to_rgb(COL["risk"])) * shade
                )
            else:
                color = COL["neutral_soft"]

            rect = plt.Rectangle(
                (x - 0.43, y - 0.33),
                0.86,
                0.66,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
            )
            ax.add_patch(rect)
            if fmt == "frac":
                label = f"{value:.2f}"
            elif fmt == "int":
                label = f"{int(value)}"
            else:
                label = f"{int(value):,}"
            text_color = "white" if family in {"signal", "risk"} and value / vmax > 0.45 else COL["text"]
            ax.text(x, y, label, ha="center", va="center", fontsize=7.0, color=text_color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=6.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in df["dataset_key"]], fontsize=6.8)
    ax.tick_params(length=0)
    ax.set_title("Expanded atlas calibration suppresses top-rank artifacts", loc="left", pad=8)
    panel_label(ax, "a", x=-0.12)
    ax.text(
        0,
        -0.27,
        "Two real-background atlases; values are calibrated rank statistics, not final permutation FDR.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=COL["muted"],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_topk_composition(ax: plt.Axes, topk: pd.DataFrame) -> None:
    available = sorted(topk["top_k"].dropna().astype(int).unique().tolist())
    selected = [k for k in [10, 50, 100] if k in available]
    if len(selected) < 3:
        selected = available[:3]

    df = topk[topk["top_k"].isin(selected)].copy()
    x = np.arange(len(DATASET_ORDER))
    width = 0.22
    offsets = np.linspace(-width, width, len(selected))

    for offset, k in zip(offsets, selected):
        sub = df[df["top_k"].eq(k)].set_index("dataset_key").reindex(DATASET_ORDER)
        xs = x + offset
        true = sub["true_lr_fraction"].astype(float).values
        risk = sub["negative_or_risk_fraction"].astype(float).values
        high = sub["high_expression_negative_fraction"].astype(float).values
        zero = sub["zero_expr_lr_fraction"].astype(float).values
        risk_core = np.maximum(risk - high - zero, 0)
        other = np.clip(1 - true - risk, 0, 1)

        ax.bar(xs, true, width=width * 0.92, color=COL["signal"], edgecolor="white", linewidth=0.6)
        ax.bar(xs, risk_core, width=width * 0.92, bottom=true, color=COL["risk_soft"], edgecolor="white", linewidth=0.6)
        ax.bar(xs, high + zero, width=width * 0.92, bottom=true + risk_core, color=COL["risk"], edgecolor="white", linewidth=0.6)
        ax.bar(xs, other, width=width * 0.92, bottom=true + risk, color=COL["neutral_soft"], edgecolor="white", linewidth=0.6)
        for x0, top in zip(xs, true + risk + other):
            ax.text(x0, top + 0.025, f"{k}", ha="center", va="bottom", fontsize=5.7, color=COL["muted"])

    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.58, len(DATASET_ORDER) - 0.42)
    ax.set_ylabel("fraction of top-k LR", labelpad=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], fontsize=6.8)
    ax.grid(axis="y", color=COL["grid"], lw=0.45, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Top-k composition after calibration", loc="left", pad=8)
    panel_label(ax, "b", x=-0.12)
    handles = [
        Patch(facecolor=COL["signal"], edgecolor="white", label="true/anchor LR"),
        Patch(facecolor=COL["risk"], edgecolor="white", label="high/zero-expression negative"),
        Patch(facecolor=COL["risk_soft"], edgecolor="white", label="shared/broad risk"),
        Patch(facecolor=COL["neutral_soft"], edgecolor="white", label="other/background"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=5.8, columnspacing=0.9)


def draw_permutation_boundary(
    ax: plt.Axes,
    perm: pd.DataFrame,
    target_decoy: pd.DataFrame,
    targeted_dataset: pd.DataFrame,
    targeted_lr: pd.DataFrame,
) -> None:
    df = perm.set_index("dataset_key").reindex(DATASET_ORDER).reset_index()
    controls = target_decoy.set_index("dataset_key").reindex(DATASET_ORDER).reset_index()
    targeted = targeted_dataset.set_index("dataset_key").reindex(DATASET_ORDER).reset_index()
    y = np.arange(len(df))[::-1]
    whole_best = df["best_true_lr_fdr"].astype(float).values
    whole_median = df["top10_median_conservative_fdr"].astype(float).values
    control_top10 = controls["top10_primary_control_prefix_fdr"].astype(float).values
    control_top100 = controls["top100_primary_control_prefix_fdr"].astype(float).values

    best_panel_q = []
    best_matched_q = []
    best_lr = []
    n_supported = []
    for dataset_key in df["dataset_key"].astype(str):
        sub = targeted_lr[targeted_lr["dataset_key"].astype(str).eq(dataset_key)].copy()
        if sub.empty:
            best_panel_q.append(np.nan)
            best_matched_q.append(np.nan)
            best_lr.append("")
            n_supported.append(0)
            continue
        sub = sub.sort_values(["targeted_panel_bh_q", "matched_control_bh_q", "spatialself_rank"])
        best_panel_q.append(float(sub.iloc[0]["targeted_panel_bh_q"]))
        best_matched_q.append(float(sub.iloc[0]["matched_control_bh_q"]))
        best_lr.append(str(sub.iloc[0]["lr_pair"]))
        n_supported.append(int(len(sub)))
    best_panel_q = np.asarray(best_panel_q, dtype=float)
    best_matched_q = np.asarray(best_matched_q, dtype=float)

    ax.axvspan(0, 0.1, color="#F8E5E2", zorder=0)
    ax.axvline(0.1, color=COL["risk"], lw=0.9, ls="--", zorder=1)
    for i, row in enumerate(df.itertuples(index=False)):
        yi = y[i]
        ax.hlines(yi + 0.22, 0, whole_median[i], color=COL["grid"], lw=1.2, zorder=1)
        ax.scatter(whole_best[i], yi + 0.22, s=46, color=COL["neutral"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(whole_median[i], yi + 0.22, s=46, color="#CFCFCF", edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(yi, control_top10[i], control_top100[i], color=COL["gold"], lw=1.35, zorder=2)
        ax.scatter(control_top10[i], yi, s=56, marker="s", color=COL["gold"], edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter(control_top100[i], yi, s=44, marker="s", color="#E6C978", edgecolor="white", linewidth=0.8, zorder=4)
        if np.isfinite(best_panel_q[i]):
            ax.hlines(yi - 0.22, best_panel_q[i], best_matched_q[i], color=COL["teal"], lw=1.35, zorder=2)
            ax.scatter(best_panel_q[i], yi - 0.22, s=58, color=COL["signal"], edgecolor="white", linewidth=0.8, zorder=4)
            ax.scatter(best_matched_q[i], yi - 0.22, s=58, color=COL["teal"], edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(
                min(max(best_panel_q[i], best_matched_q[i]) + 0.025, 0.89),
                yi - 0.22,
                f"{best_lr[i]} ({n_supported[i]} anchors)",
                ha="left",
                va="center",
                fontsize=5.8,
                color=COL["text"],
            )
        ax.text(
            min(whole_best[i] + 0.025, 0.89),
            yi + 0.22,
            f"whole-atlas best: {row.best_true_lr}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=COL["muted"],
        )
        ax.text(
            min(max(control_top10[i], control_top100[i]) + 0.025, 0.89),
            yi,
            f"control FDR top10/top100: {control_top10[i]:.2g}/{control_top100[i]:.2g}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=COL["muted"],
        )
        ax.text(
            0.98,
            yi - 0.46,
            f"whole q<=0.1: {int(row.all_lr_conservative_fdr_0_1)} | targeted q<=0.1: {int(targeted.iloc[i].n_targeted_panel_q_0_1)}",
            ha="right",
            va="top",
            fontsize=5.8,
            color=COL["muted"],
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.62, len(df) - 0.38)
    ax.set_xlabel("BH q-value", labelpad=3)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABEL[d] for d in df["dataset_key"]], fontsize=6.8)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_title("Null tests separate discovery FDR from story support", loc="left", pad=8)
    panel_label(ax, "c", x=-0.12)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["neutral"], markeredgecolor="white", markersize=6, label="whole-atlas best"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL["gold"], markeredgecolor="white", markersize=6, label="control FDR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["signal"], markeredgecolor="white", markersize=6, label="targeted panel"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["teal"], markeredgecolor="white", markersize=6, label="matched control"),
        Line2D([0], [0], color=COL["risk"], lw=0.9, ls="--", label="q=0.1"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3, fontsize=5.7, columnspacing=0.8)


def draw_same_universe_baselines(ax: plt.Axes, methods: pd.DataFrame, official: pd.DataFrame) -> None:
    spatialself = methods[methods["method"].eq("spatialself_calibrated")].copy()
    official = official.copy()
    keep_cols = [
        "method",
        "method_label",
        "mean_top10_true_lr",
        "mean_top10_high_expression_negative",
        "mean_top10_negative_or_risk_fraction",
        "mean_hard_negative_beat_fraction",
    ]
    df = pd.concat([spatialself[keep_cols], official[keep_cols]], ignore_index=True)
    order = ["spatialself_calibrated", "commot", "cellchat", "spatalk", "cellnest"]
    df["method"] = pd.Categorical(df["method"], order, ordered=True)
    df = df.sort_values("method").dropna(subset=["method"])
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))

    true_vals = df["mean_top10_true_lr"].astype(float).values
    high_neg = df["mean_top10_high_expression_negative"].astype(float).values
    hard_beat = df["mean_hard_negative_beat_fraction"].astype(float).values
    labels = [clean_label(x, 25) for x in df["method_label"].astype(str)]

    ax.barh(y + 0.16, true_vals, height=0.28, color=COL["signal"], edgecolor="white", linewidth=0.6, label="true LR in top 10")
    ax.barh(y - 0.16, high_neg, height=0.28, color=COL["risk"], edgecolor="white", linewidth=0.6, label="high-expression negatives")
    ax2 = ax.twiny()
    ax2.scatter(hard_beat, y, s=24, color=COL["gold"], edgecolor="white", linewidth=0.5, zorder=4, label="hard-neg beat rate")

    ax.set_xlim(0, max(7, np.nanmax([true_vals.max(), high_neg.max()]) + 0.7))
    ax2.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.7)
    ax.set_xlabel("mean count in top 10", labelpad=3)
    ax2.set_xlabel("hard-negative beat rate", labelpad=3, fontsize=6.4, color=COL["muted"])
    ax2.tick_params(axis="x", labelsize=5.8, colors=COL["muted"], length=2)
    ax.grid(axis="x", color=COL["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Adapter-light baselines expose false-positive modes", loc="left", pad=8)
    panel_label(ax, "d", x=-0.16)

    ax.text(
        0.0,
        -0.20,
        "Blue: true LR in top 10; red: high-expression negatives; gold dot: hard-negative beat rate. Adapter-light/projected runs.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=COL["muted"],
    )


def render() -> None:
    apply_style()
    data = load_data()
    fig = plt.figure(figsize=(7.25, 6.05), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.18],
        width_ratios=[1.04, 1.0],
        left=0.105,
        right=0.985,
        top=0.84,
        bottom=0.11,
        wspace=0.42,
        hspace=0.62,
    )

    fig.text(
        0.105,
        0.982,
        "Expanded atlas audit separates calibrated LR recovery from unresolved FDR proof",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=COL["text"],
    )
    fig.text(
        0.105,
        0.948,
        "SpatialSelf was evaluated on 886/957-LR real-background atlases; target-decoy control FDR quantifies hard-control contamination, while conservative permutation discovery FDR remains caveated.",
        ha="left",
        va="top",
        fontsize=7.05,
        color=COL["muted"],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_metric_strip(ax_a, data["calibrated_dataset"])
    draw_topk_composition(ax_b, data["calibrated_topk"])
    draw_permutation_boundary(ax_c, data["permutation_dataset"], data["target_decoy_dataset"], data["targeted_dataset"], data["targeted_lr"])
    draw_same_universe_baselines(ax_d, data["baseline_method"], data["official_adapter_method"])

    basename = OUTDIR / "phase12_figure1_expanded_atlas_audit"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "output_stem": str(basename),
                "panel": "a",
                "source_data": "figure1_expanded_calibrated_dataset_summary.csv",
                "message": "Expanded calibrated atlas scoring gives clean top-10 rankings but is not final permutation FDR.",
            },
            {
                "figure": "Figure 1",
                "output_stem": str(basename),
                "panel": "b",
                "source_data": "figure1_expanded_calibrated_topk_summary.csv",
                "message": "Top-k composition separates true LR, risk, high/zero-expression negatives, and unknown/background LR.",
            },
            {
                "figure": "Figure 1",
                "output_stem": str(basename),
                "panel": "c",
                "source_data": "figure1_expanded_permutation_dataset_summary.csv; figure1_expanded_target_decoy_dataset_summary.csv; figure1_targeted_matched_null_dataset_summary.csv; figure1_targeted_matched_null_supported_lr.csv",
                "message": "Permutation discovery FDR remains caveated; target-decoy/control FDR quantifies hard-control contamination; targeted anchor-panel and matched-control validation supports pre-specified stories.",
            },
            {
                "figure": "Figure 1",
                "output_stem": str(basename),
                "panel": "d",
                "source_data": "figure1_expanded_same_universe_baseline_method_summary.csv; figure1_expanded_official_adapter_method_summary.csv",
                "message": "Calibrated SpatialSelf is cleaner than same-universe heuristics and official adapter-light baselines in top-k contamination.",
            },
        ]
    )
    manifest.to_csv(OUTDIR / "phase12_figure1_expanded_atlas_audit_manifest.csv", index=False)


if __name__ == "__main__":
    render()
