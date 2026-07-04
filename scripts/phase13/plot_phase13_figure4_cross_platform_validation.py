#!/usr/bin/env python
"""Render Phase 13 Figure 4: cross-platform immune communication validation."""

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
    "strict_evidence": SOURCE / "figure3_strict_story_candidate_evidence.csv",
    "strict_dataset": SOURCE / "figure3_strict_story_candidate_dataset_summary.csv",
    "lymph_positive": SOURCE / "figure3_lymph_node_atlas_positive_controls.csv",
    "lymph_second": SOURCE / "figure3_lymph_node_second_story_evidence.csv",
    "lymph_direction": SOURCE / "figure3_lymph_node_directionality_edge_permutation.csv",
    "lymph_program": SOURCE / "figure3_lymph_node_program_specificity.csv",
    "cross_platform": SOURCE / "figure3_cross_platform_second_tissue_audit.csv",
    "tonsil_a1_rank": SOURCE / "figure3_tonsil_A1_method_lr_rank.csv",
    "tonsil_d1_rank": SOURCE / "figure3_tonsil_D1_method_lr_rank.csv",
    "tonsil_a1_fdr": SOURCE / "figure3_tonsil_A1_local_contrast_fdr.csv",
}

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "SpaTalk", "CellNEST"]
METHOD_COLOR = {
    "SpatialSelf": PALETTE["spatialself"],
    "COMMOT": "#626262",
    "CellChat": "#9A9A9A",
    "SpaTalk": "#C58E3C",
    "CellNEST": "#8A6AAE",
}
LR_COLOR = {
    "CCL19-CCR7": "#2C6FBB",
    "CCL21-CCR7": "#4CA3A3",
    "CXCL13-CXCR5": "#D39B2A",
    "CD48-CD244": "#7E57A0",
    "CD40LG-CD40": "#4C8A63",
    "IL21-IL21R": "#B96C42",
    "SPP1-CD44": "#3B8F5A",
    "ACTB-EGFR": PALETTE["hard_negative"],
    "MS4A1-CD3E": "#8D3B36",
    "CD19-CD3E": "#8D3B36",
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
        height_ratios=[0.85, 1.0, 0.88],
        width_ratios=[1.02, 1.02, 0.96],
        left=0.07,
        right=0.985,
        top=0.895,
        bottom=0.075,
        wspace=0.42,
        hspace=0.58,
    )

    ax_triage = fig.add_subplot(gs[0, 0])
    ax_tonsil_rank = fig.add_subplot(gs[0, 1:])
    ax_lymph_dir = fig.add_subplot(gs[1, 0:2])
    ax_tonsil_fdr = fig.add_subplot(gs[1, 2])
    ax_program = fig.add_subplot(gs[2, 0])
    ax_platform = fig.add_subplot(gs[2, 1:])

    draw_story_triage(ax_triage, source_tables["strict_dataset"])
    draw_tonsil_rank(ax_tonsil_rank, source_tables["tonsil_rank_compact"])
    draw_lymph_direction(ax_lymph_dir, source_tables["lymph_direction_compact"])
    draw_tonsil_fdr(ax_tonsil_fdr, source_tables["tonsil_a1_fdr_compact"])
    draw_program_specificity(ax_program, source_tables["lymph_program_compact"])
    draw_cross_platform_audit(ax_platform, source_tables["cross_platform_compact"])

    fig.suptitle(
        "Cross-platform validation supports immune CCC while preserving claim guardrails",
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
        "Main validation figure draft. Lymph-node and tonsil panels are supportive biological consistency evidence; they are not claimed as SpatialSelf-exclusive causal discoveries.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    out_stem = OUT_DIR / "phase13_figure4_cross_platform_validation"
    save_figure(fig, out_stem)
    write_manifest(out_stem)
    plt.close(fig)
    print(f"wrote {out_stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 source files: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in SOURCE_FILES.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    strict_dataset = data["strict_dataset"].copy()

    tonsil_frames = []
    for tissue, key in [("SpaMosaic tonsil A1", "tonsil_a1_rank"), ("SpaMosaic tonsil D1", "tonsil_d1_rank")]:
        df = data[key].copy()
        df["dataset_label"] = tissue
        df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
        tonsil_frames.append(df)
    tonsil = pd.concat(tonsil_frames, ignore_index=True)
    tonsil = tonsil[tonsil["method_label"].isin(METHOD_ORDER)].copy()
    tonsil_compact = tonsil[
        [
            "dataset_label",
            "method",
            "method_label",
            "lr_pair",
            "control_name",
            "true_lr",
            "false_positive_high_expression",
            "rank",
            "mean_score",
            "top100_edges",
            "top_decile_edges",
        ]
    ].copy()

    lymph_dir = data["lymph_direction"].copy()
    lymph_dir = lymph_dir[
        lymph_dir["dataset_key"].eq("lymph_node_full") & lymph_dir["method"].isin(["spatialself", "commot"])
    ].copy()
    lymph_dir["method_label"] = lymph_dir["method"].map({"spatialself": "SpatialSelf", "commot": "COMMOT"})
    lymph_dir_compact = lymph_dir[
        [
            "method_label",
            "lr_pair",
            "high_expression_negative",
            "observed_expected_edge_score_fraction",
            "receiver_pairing_bh_fdr",
            "edge_vector_bh_fdr",
            "edge_permutation_call",
        ]
    ].copy()

    program = data["lymph_program"].copy()
    program_compact = program[
        [
            "lr_pair",
            "program_specificity_class",
            "specific_fraction_of_significant",
            "n_specific_significant_terms_fdr_0_05",
            "best_specific_term",
            "best_specific_term_fdr",
        ]
    ].copy()

    fdr = data["tonsil_a1_fdr"].copy()
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    fdr_compact = fdr[
        [
            "null_model",
            "lr_pair",
            "true_lr",
            "false_positive_high_expression",
            "local_contrast_statistic",
            "local_contrast_bh_fdr",
        ]
    ].copy()

    cross = data["cross_platform"].copy()
    cross_compact = cross[
        [
            "dataset_key",
            "platform",
            "rank_validation_status",
            "best_positive_control",
            "best_positive_rank",
            "best_negative_control",
            "best_negative_rank",
            "positive_top3",
            "negative_top3",
            "local_contrast_status",
            "targeted_fdr_status",
            "downstream_claim_status",
            "story_score",
            "story_class",
            "recommended_use",
        ]
    ].copy()

    return {
        "strict_dataset": strict_dataset,
        "strict_evidence": data["strict_evidence"],
        "tonsil_rank_compact": tonsil_compact,
        "tonsil_a1_fdr_compact": fdr_compact,
        "lymph_positive": data["lymph_positive"],
        "lymph_second": data["lymph_second"],
        "lymph_direction_compact": lymph_dir_compact,
        "lymph_program_compact": program_compact,
        "cross_platform_compact": cross_compact,
    }


def export_source_tables(source_tables: dict[str, pd.DataFrame]) -> None:
    for key, table in source_tables.items():
        table.to_csv(SOURCE_OUT / f"figure4_{key}.csv", index=False)


def draw_story_triage(ax: plt.Axes, dataset_summary: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.09, y=1.03)
    ax.set_title("Story triage keeps claims bounded", loc="left", pad=8)
    ax.set_axis_off()
    rows = dataset_summary.copy()
    labels = {
        "post_xenium_colon_full": "Post-Xenium\ncolon",
        "human_lymph_node_visium_raw_full": "Human lymph\nnode Visium",
    }
    categories = [
        ("n_main_story_ready", "main story", PALETTE["success"]),
        ("n_biological_consistency_with_stat_support", "supportive", PALETTE["spatialself"]),
        ("n_targeted_missing_functional_closure", "needs closure", PALETTE["weak"]),
        ("n_supportive_not_main", "supplement", PALETTE["neutral"]),
    ]
    y0 = 0.77
    for i, row in enumerate(rows.itertuples(index=False)):
        y = y0 - i * 0.27
        ax.text(0.02, y + 0.07, labels.get(row.dataset_key, row.dataset_key), transform=ax.transAxes, ha="left", va="center", fontsize=6.6, fontweight="bold", color=PALETTE["ink"])
        x0 = 0.38
        total = max(float(row.n_story_guardrail_candidates), 1.0)
        for col, label, color in categories:
            val = float(getattr(row, col))
            width = 0.50 * val / total
            ax.add_patch(patches.Rectangle((x0, y), width, 0.08, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.5))
            if val > 0:
                ax.text(x0 + width / 2, y + 0.04, f"{int(val)}", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color="white")
            x0 += width
        ax.text(0.38, y - 0.035, f"top: {row.top_candidate}", transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=PALETTE["muted"])
    legend_y = 0.08
    x = 0.02
    for _col, label, color in categories:
        ax.add_patch(patches.Rectangle((x, legend_y), 0.035, 0.035, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(x + 0.045, legend_y + 0.017, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.3, color=PALETTE["muted"])
        x += 0.24


def draw_tonsil_rank(ax: plt.Axes, tonsil: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.045, y=1.03)
    ax.set_title("Tonsil controls: positives prioritized, high-expression controls suppressed", loc="left", pad=8)
    focal = tonsil[tonsil["dataset_label"].eq("SpaMosaic tonsil A1")].copy()
    focal = focal[focal["lr_pair"].isin(["CXCL13-CXCR5", "CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "ACTB-EGFR", "MS4A1-CD3E"])]
    y_lookup = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    ax.axvspan(0.5, 3.5, color="#F2F7FC", zorder=0)
    for method in METHOD_ORDER:
        sub = focal[focal["method_label"].eq(method)]
        y = y_lookup[method]
        ax.hlines(y, 0.8, 7.2, color=PALETTE["neutral_soft"], lw=0.8, zorder=0)
        for _, row in sub.iterrows():
            lr = row["lr_pair"]
            is_neg = bool(row["false_positive_high_expression"])
            ax.scatter(
                row["rank"],
                y,
                s=55 if not is_neg else 48,
                marker="v" if is_neg else "o",
                color=LR_COLOR.get(lr, PALETTE["neutral"]),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-0.7, len(METHOD_ORDER) - 0.3)
    ax.invert_xaxis()
    ax.set_xlabel("LR rank within control panel (left = better)")
    ax.set_yticks(list(y_lookup.values()))
    ax.set_yticklabels(METHOD_ORDER[::-1])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    handles = [
        plt.Line2D([0], [0], marker="o", ls="None", color="w", markerfacecolor=LR_COLOR["CXCL13-CXCR5"], markeredgecolor="white", markersize=6, label="positive LR"),
        plt.Line2D([0], [0], marker="v", ls="None", color="w", markerfacecolor=PALETTE["hard_negative"], markeredgecolor="white", markersize=6, label="high-expression control"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.7)
    ax.text(0.01, 0.02, "SpaMosaic tonsil A1; D1 shows the same qualitative pattern in source data.", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color=PALETTE["muted"])


def draw_lymph_direction(ax: plt.Axes, direction: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.045, y=1.03)
    ax.set_title("Lymph-node source-sink consistency has edge-level support for selected LR", loc="left", pad=8)
    df = direction[direction["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "CXCL13-CXCR5"])].copy()
    x_lookup = {lr: i for i, lr in enumerate(["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "CXCL13-CXCR5"])}
    offsets = {"SpatialSelf": -0.13, "COMMOT": 0.13}
    for _, row in df.iterrows():
        x = x_lookup[row["lr_pair"]] + offsets[row["method_label"]]
        q = row["receiver_pairing_bh_fdr"]
        supported = q <= 0.1
        ax.scatter(
            x,
            row["observed_expected_edge_score_fraction"],
            s=72,
            color=METHOD_COLOR[row["method_label"]],
            marker="o" if supported else "s",
            edgecolor=PALETTE["ink"] if row["method_label"] == "SpatialSelf" else "white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(x, row["observed_expected_edge_score_fraction"] + 0.045, f"q={q:.2f}", ha="center", va="bottom", fontsize=5.4, color=PALETTE["muted"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["CCL19\nCCR7", "CCL21\nCCR7", "CD48\nCD244", "CXCL13\nCXCR5"])
    ax.set_ylabel("Expected receiver-pairing\nscore fraction")
    ax.set_ylim(-0.04, 1.12)
    soften_axes(ax)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLOR["SpatialSelf"], markeredgecolor=PALETTE["ink"], markersize=6, label="SpatialSelf"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLOR["COMMOT"], markeredgecolor="white", markersize=6, label="COMMOT"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["neutral"], markeredgecolor="white", markersize=6, label="q > 0.1"),
        ],
        frameon=False,
        loc="lower left",
        fontsize=5.7,
    )


def draw_tonsil_fdr(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.09, y=1.03)
    ax.set_title("Tonsil local contrast under null models", loc="left", pad=8)
    sub = fdr[fdr["null_model"].eq("receiver_permutation")].copy()
    order = ["CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "CXCL13-CXCR5", "ACTB-EGFR", "MS4A1-CD3E"]
    sub["lr_pair"] = pd.Categorical(sub["lr_pair"], order, ordered=True)
    sub = sub.sort_values("lr_pair")
    y = np.arange(len(sub))[::-1]
    vals = -np.log10(sub["local_contrast_bh_fdr"].astype(float).clip(lower=1e-4))
    colors = [LR_COLOR.get(str(lr), PALETTE["neutral"]) if bool(t) else PALETTE["hard_negative"] for lr, t in zip(sub["lr_pair"], sub["true_lr"])]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(-np.log10(0.05), color=PALETTE["muted"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([str(x).replace("-", "\n") for x in sub["lr_pair"]])
    ax.set_xlabel("-log10 BH q")
    ax.set_xlim(0, max(2.5, float(vals.max()) * 1.15))
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.text(0.98, 0.03, "receiver permutation", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=PALETTE["muted"])


def draw_program_specificity(ax: plt.Axes, program: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.09, y=1.03)
    ax.set_title("Receiver-program support is mixed", loc="left", pad=8)
    df = program.copy()
    y = np.arange(len(df))[::-1]
    vals = df["specific_fraction_of_significant"].fillna(0).astype(float)
    colors = [LR_COLOR.get(lr, PALETTE["neutral"]) for lr in df["lr_pair"]]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([x.replace("-", "\n") for x in df["lr_pair"]])
    ax.set_xlabel("Specific terms / all significant terms")
    ax.set_xlim(0, max(0.22, vals.max() * 1.25))
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    for yy, row in zip(y, df.itertuples(index=False)):
        label = "specific" if "specific" in str(row.program_specificity_class) and "mixed" not in str(row.program_specificity_class) else "mixed"
        ax.text(vals.iloc[len(df) - 1 - yy] + 0.008, yy, label, ha="left", va="center", fontsize=5.5, color=PALETTE["muted"])


def draw_cross_platform_audit(ax: plt.Axes, audit: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.045, y=1.03)
    ax.set_title("Cross-platform evidence is supportive, not overclaimed", loc="left", pad=8)
    ax.set_axis_off()
    rows = audit.copy()
    y0 = 0.78
    for i, row in enumerate(rows.itertuples(index=False)):
        y = y0 - i * 0.27
        color = PALETTE["success"] if row.story_score >= 5 else (PALETTE["weak"] if row.story_score >= 4 else PALETTE["neutral"])
        ax.add_patch(patches.FancyBboxPatch((0.02, y - 0.07), 0.14, 0.14, boxstyle="round,pad=0.01,rounding_size=0.015", transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.7))
        ax.text(0.09, y, f"{row.story_score:.0f}", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.19, y + 0.045, row.platform, transform=ax.transAxes, ha="left", va="center", fontsize=6.6, color=PALETTE["ink"], fontweight="bold")
        detail = f"best +: {row.best_positive_control} rank {row.best_positive_rank:.0f}; best -: {row.best_negative_control} rank {row.best_negative_rank:.0f}"
        ax.text(0.19, y + 0.003, detail, transform=ax.transAxes, ha="left", va="center", fontsize=5.7, color=PALETTE["muted"])
        rec = "\n".join(textwrap.wrap(row.recommended_use, width=70))
        ax.text(0.19, y - 0.052, rec, transform=ax.transAxes, ha="left", va="center", fontsize=5.35, color=PALETTE["muted"], linespacing=1.05)
    ax.text(
        0.02,
        0.02,
        "Allowed claim: cross-platform biological consistency with explicit caveats; not a clean second main story.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["spatialself_dark"],
    )


def write_manifest(out_stem: Path) -> None:
    rows = [
        ("a", "Story triage", "figure4_strict_dataset.csv", "Only post-Xenium colon is currently A-class; lymph node is supportive biological consistency."),
        ("b", "Tonsil control ranking", "figure4_tonsil_rank_compact.csv", "SpatialSelf ranks tonsil positive controls ahead of high-expression controls in A1/D1 source tables."),
        ("c", "Lymph-node directionality", "figure4_lymph_direction_compact.csv", "Selected lymph-node LR show receiver-pairing support, but this is not SpatialSelf-exclusive."),
        ("d", "Tonsil local contrast FDR", "figure4_tonsil_a1_fdr_compact.csv", "Tonsil A1 positives pass receiver-permutation local contrast while negative controls do not."),
        ("e", "Receiver-program specificity", "figure4_lymph_program_compact.csv", "Lymph-node receiver programs include specific immune terms but many signals remain broad/mixed."),
        ("f", "Cross-platform audit", "figure4_cross_platform_compact.csv", "Tonsil supports cross-platform validation; Xenium lymph node remains exploratory/mixed."),
    ]
    pd.DataFrame(
        [
            {
                "figure": "Figure 4",
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
