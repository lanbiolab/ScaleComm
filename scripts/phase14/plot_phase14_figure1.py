#!/usr/bin/env python
"""Phase 14 redesigned Figure 1: method front-door figure."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from phase14_style import PALETTE, apply_style, panel_label, clean_axes, save_figure, add_figure_title


ROOT = Path(__file__).resolve().parents[2]
P13 = ROOT / "outputs" / "phase13" / "source_data"
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    weak = pd.read_csv(P13 / "figure1_weak_summary.csv")
    ablation = pd.read_csv(P13 / "figure1_component_ablation.csv")
    arch = pd.read_csv(P13 / "figure1_architecture_schematic.csv")
    source_tables = build_source_tables(weak, ablation, arch)
    export_source(source_tables)

    fig = plt.figure(figsize=(7.2, 5.25), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.58, 1.0, 1.0],
        height_ratios=[1.08, 0.86],
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.88,
        wspace=0.42,
        hspace=0.52,
    )
    ax_arch = fig.add_subplot(gs[:, 0])
    ax_bench = fig.add_subplot(gs[0, 1])
    ax_perf = fig.add_subplot(gs[0, 2])
    ax_ablate = fig.add_subplot(gs[1, 1:])

    draw_architecture(ax_arch)
    draw_benchmark_cartoon(ax_bench)
    draw_performance(ax_perf, source_tables["performance_summary"])
    draw_ablation_delta(ax_ablate, source_tables["ablation_delta"])

    add_figure_title(
        fig,
        "SpatialSelf learns LR-specific and multi-scale spatial communication under realistic stressors",
        "Phase 14 redesigned Figure 1: a schematic-led method figure with compact benchmark and component evidence.",
    )
    stem = FIG / "phase14_figure1_method_benchmark_redesign"
    save_figure(fig, stem)
    write_manifest(stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png")


def build_source_tables(weak: pd.DataFrame, ablation: pd.DataFrame, arch: pd.DataFrame) -> dict[str, pd.DataFrame]:
    final = weak.sort_values("weak_signal_recall_at_threshold_mean", ascending=False).iloc[0].copy()
    ref = weak[weak["display_label"].eq("wavelet default")].iloc[0].copy()
    performance = pd.DataFrame(
        [
            {
                "metric": "AUPR",
                "reference": float(ref["aupr_mean"]),
                "phase14_model": float(final["aupr_mean"]),
                "delta": float(final["aupr_mean"] - ref["aupr_mean"]),
                "display": "AUPR",
            },
            {
                "metric": "weak_signal_recall",
                "reference": float(ref["weak_signal_recall_at_threshold_mean"]),
                "phase14_model": float(final["weak_signal_recall_at_threshold_mean"]),
                "delta": float(final["weak_signal_recall_at_threshold_mean"] - ref["weak_signal_recall_at_threshold_mean"]),
                "display": "weak recall",
            },
            {
                "metric": "high_expression_fp_rate",
                "reference": float(ref["false_positive_high_expression_rate_at_threshold_mean"]),
                "phase14_model": float(final["false_positive_high_expression_rate_at_threshold_mean"]),
                "delta": float(final["false_positive_high_expression_rate_at_threshold_mean"] - ref["false_positive_high_expression_rate_at_threshold_mean"]),
                "display": "high-expr FP",
            },
        ]
    )

    keep = ["no LR emb", "fixed random LR", "no graph", "no distance", "no expression", "MLP"]
    ref_row = ablation[ablation["display_label"].eq("wavelet ref")].iloc[0]
    sub = ablation[ablation["display_label"].isin(keep)].copy()
    sub["delta_aupr_vs_ref"] = sub["aupr_mean"].astype(float) - float(ref_row["aupr_mean"])
    sub["delta_weak_vs_ref"] = sub["weak_signal_recall_at_threshold_mean"].astype(float) - float(ref_row["weak_signal_recall_at_threshold_mean"])
    sub["delta_fdr_vs_ref"] = sub["empirical_fdr_at_threshold_mean"].astype(float) - float(ref_row["empirical_fdr_at_threshold_mean"])
    sub["display_label"] = pd.Categorical(sub["display_label"], categories=keep, ordered=True)
    sub = sub.sort_values("display_label")

    return {
        "architecture_schematic": arch,
        "performance_summary": performance,
        "ablation_delta": sub,
    }


def export_source(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(SOURCE / f"figure1_{name}.csv", index=False)


def draw_architecture(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.02, y=1.02)
    ax.text(0.02, 0.965, "Identity-aware multi-scale CCC inference", transform=ax.transAxes, ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.02, 0.915, "One LR pair is scored in its spatial context; hard negatives teach the model what high-expression artifacts look like.", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=PALETTE["muted"], wrap=True)

    # Tissue mini-scene
    rng = np.random.default_rng(14)
    centers = np.array([[0.18, 0.68], [0.26, 0.55], [0.16, 0.43], [0.30, 0.34]])
    colors = [PALETTE["signal_soft"], PALETTE["neutral_soft"], "#E9E0F3", "#EDE7D5"]
    for i in range(80):
        c = centers[rng.integers(0, len(centers))]
        x, y = c + rng.normal(0, 0.045, 2)
        ax.scatter(x, y, s=10, color=colors[i % len(colors)], edgecolor="white", linewidth=0.25, transform=ax.transAxes, zorder=1)
    for xy, label, color in [((0.17, 0.68), "sender\nligand", PALETTE["signal"]), ((0.31, 0.50), "receiver\nreceptor", PALETTE["spatialself"])]:
        ax.scatter(*xy, s=115, color=color, edgecolor="white", linewidth=0.9, transform=ax.transAxes, zorder=3)
        ax.text(xy[0], xy[1] - 0.075, label, transform=ax.transAxes, ha="center", va="top", fontsize=5.9, color=PALETTE["ink"])

    # Main pathway
    blocks = [
        (0.52, 0.74, "LR identity", "pair-specific\nembedding", PALETTE["lr"]),
        (0.52, 0.53, "graph wavelet", "short + long\nspatial scales", PALETTE["wavelet"]),
        (0.52, 0.32, "hard negatives", "high expression\nwrong context", PALETTE["risk"]),
        (0.82, 0.53, "CCC score", "ranked, mapped,\nvalidated", PALETTE["spatialself"]),
    ]
    for x, y, head, body, color in blocks:
        w, h = (0.24, 0.135) if head != "CCC score" else (0.22, 0.18)
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.018",
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                transform=ax.transAxes,
            )
        )
        ax.text(x, y + 0.024, head, transform=ax.transAxes, ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
        ax.text(x, y - 0.032, body, transform=ax.transAxes, ha="center", va="center", fontsize=5.7, color="white", linespacing=1.05)

    for start, end in [((0.34, 0.58), (0.40, 0.74)), ((0.34, 0.55), (0.40, 0.53)), ((0.34, 0.51), (0.40, 0.32)), ((0.64, 0.74), (0.71, 0.57)), ((0.64, 0.53), (0.71, 0.53)), ((0.64, 0.32), (0.71, 0.49))]:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["line"]))

    ax.text(0.04, 0.12, "Design point", transform=ax.transAxes, ha="left", va="center", fontsize=6.4, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.22, 0.12, "separate real weak/context CCC from high-expression artifacts", transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color=PALETTE["muted"])


def draw_benchmark_cartoon(ax: plt.Axes) -> None:
    panel_label(ax, "b", x=-0.1, y=1.04)
    ax.set_axis_off()
    ax.set_title("Benchmark stressors are spatially distinct", loc="left", pad=8)
    scenarios = [
        ("contact", 0.2, PALETTE["signal"], 0.055),
        ("paracrine", 0.5, PALETTE["weak"], 0.12),
        ("chemokine\nniche", 0.8, PALETTE["spatialself"], 0.19),
    ]
    for label, x, color, radius in scenarios:
        ax.add_patch(patches.Circle((x, 0.55), radius, transform=ax.transAxes, facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.0))
        ax.scatter([x - radius / 2, x + radius / 2], [0.55, 0.55], s=[42, 42], color=[color, PALETTE["neutral_dark"]], edgecolor="white", linewidth=0.6, transform=ax.transAxes, zorder=3)
        ax.text(x, 0.23, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.6, color=PALETTE["ink"])
    ax.add_patch(patches.FancyBboxPatch((0.12, 0.02), 0.76, 0.11, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=PALETTE["risk_soft"], edgecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.075, "explicit high-expression false positives", transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color=PALETTE["risk"])


def draw_performance(ax: plt.Axes, perf: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.1, y=1.04)
    ax.set_title("Compact benchmark readout", loc="left", pad=8)
    x = np.arange(len(perf))
    vals = perf["phase14_model"].to_numpy(float)
    ref = perf["reference"].to_numpy(float)
    colors = [PALETTE["spatialself"], PALETTE["weak"], PALETTE["risk"]]
    ax.bar(x, vals, width=0.55, color=colors, edgecolor="white", linewidth=0.7)
    ax.scatter(x, ref, s=28, color=PALETTE["ink"], marker="_", linewidth=1.6, zorder=4, label="reference")
    for i, row in perf.iterrows():
        ax.text(i, vals[i] + 0.018, f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=6.3, color=PALETTE["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(perf["display"])
    ax.set_ylim(0, max(0.32, float(vals.max()) * 1.28))
    ax.set_ylabel("score / rate")
    clean_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=5.8)


def draw_ablation_delta(ax: plt.Axes, ablation: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.04, y=1.04)
    ax.set_title("Ablation is shown as a clean delta from the wavelet reference", loc="left", pad=8)
    labels = [str(x) for x in ablation["display_label"]]
    y = np.arange(len(labels))[::-1]
    delta = ablation["delta_weak_vs_ref"].to_numpy(float)
    colors = [PALETTE["signal"] if v >= 0 else PALETTE["risk"] for v in delta]
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.barh(y, delta, color=colors, edgecolor="white", linewidth=0.7, height=0.64)
    for yi, val in zip(y, delta):
        ax.text(val + (0.004 if val >= 0 else -0.004), yi, f"{val:+.2f}", ha="left" if val >= 0 else "right", va="center", fontsize=6.1, color=PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("delta weak-signal recall vs wavelet reference")
    ax.set_xlim(min(-0.03, float(delta.min()) - 0.02), max(0.06, float(delta.max()) + 0.03))
    clean_axes(ax, grid="x")


def write_manifest(stem: Path) -> None:
    rows = [
        ("a", "SpatialSelf combines LR identity, graph-wavelet context and hard-negative self-supervision.", "outputs/phase14/source_data/figure1_architecture_schematic.csv"),
        ("b", "The benchmark includes contact, paracrine, chemokine-like and high-expression false-positive stressors.", "outputs/phase13/source_data/figure1_mechanism_coverage.csv"),
        ("c", "A compact benchmark readout reports AUPR, weak recall and high-expression false-positive behavior.", "outputs/phase14/source_data/figure1_performance_summary.csv"),
        ("d", "Ablation evidence is summarized as delta weak-signal recall relative to the wavelet reference.", "outputs/phase14/source_data/figure1_ablation_delta.csv"),
    ]
    pd.DataFrame(rows, columns=["panel", "claim", "source_data"]).to_csv(f"{stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
