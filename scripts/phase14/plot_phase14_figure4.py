#!/usr/bin/env python
"""Phase 14 redesigned Figure 4: cross-platform immune support."""

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

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "SpaTalk", "CellNEST"]
TONSIL_LR = ["CXCL13-CXCR5", "IL21-IL21R", "CD40LG-CD40", "CCL21-CCR7", "ACTB-EGFR", "MS4A1-CD3E"]
LYMPH_LR = ["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "ACTB-CD44", "RPLP0-CXCR4", "EEF1A1-CCR7"]
LR_COLORS = {
    "CXCL13-CXCR5": PALETTE["spatialself"],
    "IL21-IL21R": PALETTE["lr"],
    "CD40LG-CD40": PALETTE["weak"],
    "CCL21-CCR7": PALETTE["signal"],
    "CCL19-CCR7": PALETTE["signal"],
    "CD48-CD244": "#4F7DBA",
    "ACTB-EGFR": PALETTE["risk"],
    "MS4A1-CD3E": "#8F4A48",
    "ACTB-CD44": PALETTE["risk"],
    "RPLP0-CXCR4": "#8F4A48",
    "EEF1A1-CCR7": "#9A6A6A",
}


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data = load_data()
    tables = build_source_tables(data)
    export_source(tables)

    fig = plt.figure(figsize=(7.2, 5.45), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.12, 1.25, 1.0],
        height_ratios=[1.0, 0.92],
        left=0.06,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.42,
        hspace=0.55,
    )
    ax_overview = fig.add_subplot(gs[:, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_fdr = fig.add_subplot(gs[0, 2])
    ax_lymph = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    draw_platform_overview(ax_overview, tables["cross_platform"])
    draw_tonsil_rank(ax_rank, tables["tonsil_rank"])
    draw_tonsil_fdr(ax_fdr, tables["tonsil_fdr"])
    draw_lymph_direction(ax_lymph, tables["lymph_direction"])
    draw_support_summary(ax_summary, tables["support_summary"])

    add_figure_title(
        fig,
        "Immune communication programs are supported across tissue and platform contexts",
        "Phase 14 redesigned Figure 4: cross-platform evidence is shown as supportive validation with explicit caveats.",
    )
    stem = FIG / "phase14_figure4_cross_platform_redesign"
    save_figure(fig, stem)
    write_manifest(stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "cross": P13 / "figure4_cross_platform_compact.csv",
        "tonsil_rank": P13 / "figure4_tonsil_rank_compact.csv",
        "tonsil_fdr": P13 / "figure4_tonsil_a1_fdr_compact.csv",
        "lymph_direction": P13 / "figure4_lymph_direction_compact.csv",
        "lymph_program": P13 / "figure4_lymph_program_compact.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 sources: " + "; ".join(missing))
    return {k: pd.read_csv(p) for k, p in files.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cross = data["cross"].copy()
    cross["display_platform"] = cross["platform"].astype(str).str.replace("SpaMosaic tonsil", "tonsil", regex=False)
    cross["support_level"] = cross["recommended_use"].map(
        lambda x: "supportive" if "supportive" in str(x).lower() else "exploratory"
    )

    tonsil_rank = data["tonsil_rank"].copy()
    tonsil_rank = tonsil_rank[tonsil_rank["lr_pair"].isin(TONSIL_LR)].copy()
    tonsil_rank["method_label"] = tonsil_rank["method_label"].replace({"SpatialSelf calibrated": "SpatialSelf"})

    tonsil_fdr = data["tonsil_fdr"].copy()
    tonsil_fdr = tonsil_fdr[tonsil_fdr["null_model"].eq("receiver_permutation") & tonsil_fdr["lr_pair"].isin(TONSIL_LR)].copy()
    tonsil_fdr["minus_log10_q"] = -np.log10(tonsil_fdr["local_contrast_bh_fdr"].astype(float).clip(lower=1e-4))

    lymph = data["lymph_direction"].copy()
    lymph = lymph[lymph["method_label"].isin(["SpatialSelf", "COMMOT"]) & lymph["lr_pair"].isin(LYMPH_LR)].copy()
    lymph["minus_log10_pairing_q"] = -np.log10(lymph["receiver_pairing_bh_fdr"].astype(float).clip(lower=1e-4))

    program = data["lymph_program"].copy()
    support = pd.DataFrame(
        [
            {
                "context": "tonsil A1",
                "primary_signal": "CXCL13-CXCR5 / CCL21-CCR7",
                "support": "positive ranks + local contrast",
                "caveat": "targeted negative leakage disclosed",
            },
            {
                "context": "tonsil D1",
                "primary_signal": "CD40LG-CD40",
                "support": "positive ranks",
                "caveat": "local contrast missing",
            },
            {
                "context": "lymph node",
                "primary_signal": "CCL19/CCL21-CCR7; CD48-CD244",
                "support": "receiver pairing / program support",
                "caveat": "mixed; not SpatialSelf-exclusive",
            },
        ]
    )
    return {
        "cross_platform": cross,
        "tonsil_rank": tonsil_rank,
        "tonsil_fdr": tonsil_fdr,
        "lymph_direction": lymph,
        "lymph_program": program,
        "support_summary": support,
    }


def export_source(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(SOURCE / f"figure4_{name}.csv", index=False)


def draw_platform_overview(ax: plt.Axes, cross: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.07, y=1.03)
    ax.set_axis_off()
    ax.set_title("Cross-platform support is caveated by design", loc="left", pad=8)
    y = 0.80
    for _, row in cross.iterrows():
        supportive = row["support_level"] == "supportive"
        color = PALETTE["signal"] if supportive else PALETTE["weak"]
        face = PALETTE["signal_soft"] if supportive else PALETTE["weak_soft"]
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.06, y - 0.075),
                0.84,
                0.12,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=face,
                edgecolor=color,
                linewidth=0.9,
                transform=ax.transAxes,
            )
        )
        ax.text(0.10, y + 0.016, str(row["display_platform"]), transform=ax.transAxes, ha="left", va="center", fontsize=6.7, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.10, y - 0.026, f"best positive: {row['best_positive_control']} rank {int(row['best_positive_rank'])}", transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color=PALETTE["muted"])
        ax.text(0.84, y, "supportive" if supportive else "exploratory", transform=ax.transAxes, ha="right", va="center", fontsize=5.8, color=color, fontweight="bold")
        y -= 0.20
    ax.text(0.06, 0.08, "Message: cross-platform panels support biological consistency; they are not a second causal discovery claim.", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=PALETTE["muted"], wrap=True)


def draw_tonsil_rank(ax: plt.Axes, rank: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.1, y=1.04)
    ax.set_title("Tonsil A1 rank strip", loc="left", pad=8)
    df = rank[rank["dataset_label"].eq("SpaMosaic tonsil A1") & rank["method_label"].isin(METHOD_ORDER)].copy()
    y_lookup = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    ax.axvspan(0.5, 4.5, color=PALETTE["signal_soft"], alpha=0.55, zorder=0)
    for method in METHOD_ORDER:
        sub = df[df["method_label"].eq(method)]
        y = y_lookup[method]
        ax.hlines(y, 0.7, 7.3, color=PALETTE["line"], lw=0.75)
        for _, row in sub.iterrows():
            lr = row["lr_pair"]
            neg = bool(row["false_positive_high_expression"])
            ax.scatter(row["rank"], y, s=52 if not neg else 46, marker="v" if neg else "o", color=LR_COLORS.get(lr, PALETTE["neutral"]), edgecolor="white", linewidth=0.65, zorder=3)
    ax.set_yticks([y_lookup[m] for m in METHOD_ORDER])
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("rank")
    ax.tick_params(axis="y", length=0)
    clean_axes(ax, grid="x")


def draw_tonsil_fdr(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.12, y=1.04)
    ax.set_title("Tonsil local contrast", loc="left", pad=8)
    order = ["CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "CXCL13-CXCR5", "ACTB-EGFR", "MS4A1-CD3E"]
    sub = fdr.set_index("lr_pair").reindex(order).dropna(subset=["minus_log10_q"]).reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [LR_COLORS.get(lr, PALETTE["neutral"]) for lr in sub["lr_pair"]]
    ax.barh(y, sub["minus_log10_q"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(-np.log10(0.05), color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(sub["lr_pair"])
    ax.set_xlabel("-log10 q")
    ax.set_xlim(0, max(2.4, float(sub["minus_log10_q"].max()) + 0.35))
    clean_axes(ax, grid="x")


def draw_lymph_direction(ax: plt.Axes, lymph: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.1, y=1.04)
    ax.set_title("Lymph-node receiver-pairing support", loc="left", pad=8)
    sub = lymph[lymph["method_label"].eq("SpatialSelf")].copy()
    order = ["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "ACTB-CD44", "RPLP0-CXCR4", "EEF1A1-CCR7"]
    sub = sub.set_index("lr_pair").reindex(order).dropna(subset=["minus_log10_pairing_q"]).reset_index()
    y = np.arange(len(sub))[::-1]
    colors = [LR_COLORS.get(lr, PALETTE["neutral"]) for lr in sub["lr_pair"]]
    ax.barh(y, sub["minus_log10_pairing_q"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(-np.log10(0.05), color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(sub["lr_pair"])
    ax.set_xlabel("-log10 receiver-pairing q")
    clean_axes(ax, grid="x")


def draw_support_summary(ax: plt.Axes, support: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.12, y=1.04)
    ax.set_axis_off()
    ax.set_title("How to write this figure", loc="left", pad=8)
    y = 0.78
    for _, row in support.iterrows():
        ax.scatter(0.06, y, s=52, color=PALETTE["signal"] if "tonsil A1" in row["context"] else PALETTE["weak"], transform=ax.transAxes)
        ax.text(0.14, y + 0.032, row["context"], transform=ax.transAxes, ha="left", va="center", fontsize=6.4, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.14, y - 0.006, row["support"], transform=ax.transAxes, ha="left", va="center", fontsize=5.7, color=PALETTE["ink"])
        ax.text(0.14, y - 0.044, row["caveat"], transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=PALETTE["muted"])
        y -= 0.25


def write_manifest(stem: Path) -> None:
    rows = [
        ("a", "Cross-platform validation is framed as supportive and caveated.", "outputs/phase14/source_data/figure4_cross_platform.csv"),
        ("b", "Tonsil A1 ranks positive immune LR ahead of hard negatives.", "outputs/phase14/source_data/figure4_tonsil_rank.csv"),
        ("c", "Tonsil local contrast supports selected immune LR and suppresses high-expression controls.", "outputs/phase14/source_data/figure4_tonsil_fdr.csv"),
        ("d", "Lymph-node receiver-pairing supports selected chemokine and immune LR but remains mixed.", "outputs/phase14/source_data/figure4_lymph_direction.csv"),
        ("e", "Writing guardrail keeps the figure as cross-platform support rather than causal discovery.", "outputs/phase14/source_data/figure4_support_summary.csv"),
    ]
    pd.DataFrame(rows, columns=["panel", "claim", "source_data"]).to_csv(f"{stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
