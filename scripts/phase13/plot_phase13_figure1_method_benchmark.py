#!/usr/bin/env python
"""Render Phase 13 Figure 1: method and realistic benchmark."""

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
OUT_DIR = ROOT / "outputs" / "phase13" / "figures"
SOURCE_OUT = ROOT / "outputs" / "phase13" / "source_data"

SOURCE_FILES = {
    "mechanism_weak": ROOT / "outputs/phase2/goal6_paper_figures/source_tables/goal1_mechanism_weak_signal.csv",
    "weak_summary": ROOT / "outputs/phase2/goal6_paper_figures/source_tables/goal1_controlled_weak_signal.csv",
    "component_ablation": ROOT / "outputs/phase2/goal6_paper_figures/source_tables/goal6_component_ablation.csv",
    "mechanism_coverage": ROOT / "outputs/phase2_revised/goal34_benchmark_quality_report/goal34_mechanism_coverage_matrix.csv",
}

MECHANISM_LABEL = {
    "juxtacrine_contact": "contact",
    "short_medium_paracrine": "short/medium\nparacrine",
    "long_range_chemokine_niche": "long-range\nchemokine",
    "false_positive_high_expression": "high-expression\nfalse positive",
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    export_source_tables(data)

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.12, 1.0],
        height_ratios=[1.05, 0.82, 0.86],
        wspace=0.34,
        hspace=0.48,
    )

    ax_arch = fig.add_subplot(gs[0:2, 0])
    ax_design = fig.add_subplot(gs[0, 1])
    ax_perf = fig.add_subplot(gs[1, 1])
    ax_ablate = fig.add_subplot(gs[2, :])

    draw_architecture(ax_arch)
    draw_benchmark_design(ax_design, data["mechanism_coverage"])
    draw_weak_signal_performance(ax_perf, data["weak_summary"])
    draw_component_audit(ax_ablate, data["component_ablation"])

    fig.suptitle(
        "SpatialSelf learns LR-specific, multi-scale spatial communication for realistic CCC benchmarks",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.03,
        0.952,
        "Figure 1 draft. Mechanistic benchmark and component-audit evidence; component ablation is early controlled evidence, not final whole-atlas FDR.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )
    fig.subplots_adjust(top=0.9, left=0.055, right=0.985, bottom=0.075)

    out_stem = OUT_DIR / "phase13_figure1_method_benchmark"
    save_figure(fig, out_stem)
    write_manifest(out_stem)
    plt.close(fig)
    print(f"wrote {out_stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source tables: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in SOURCE_FILES.items()}


def export_source_tables(data: dict[str, pd.DataFrame]) -> None:
    for key, table in data.items():
        table.to_csv(SOURCE_OUT / f"figure1_{key}.csv", index=False)
    architecture_source_table().to_csv(SOURCE_OUT / "figure1_architecture_schematic.csv", index=False)


def architecture_source_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "Sender and receiver expression",
                "role_in_schematic": "cell/spot-level ligand, receptor and covariate inputs",
                "claim_supported": "SpatialSelf scores directed sender-receiver LR candidates.",
                "claim_boundary": "Schematic does not prove causal communication.",
            },
            {
                "component": "LR identity embedding",
                "role_in_schematic": "pair-specific latent representation for LR identity",
                "claim_supported": "Model includes an identity-aware component beyond expression and distance.",
                "claim_boundary": "Embedding is interpreted as latent structure, not direct biochemical mechanism.",
            },
            {
                "component": "Graph-wavelet multi-scale context",
                "role_in_schematic": "spatial graph context with multi-scale neighborhood propagation",
                "claim_supported": "Model can represent short- and longer-range spatial context.",
                "claim_boundary": "Range behavior is evaluated empirically; it is not guaranteed for all LR pairs.",
            },
            {
                "component": "Hard-negative self-supervision",
                "role_in_schematic": "training contrast against high-expression or wrong-context negatives",
                "claim_supported": "Training explicitly exposes the model to hard false-positive modes.",
                "claim_boundary": "Hard-negative suppression is benchmark- and guardrail-dependent.",
            },
            {
                "component": "Calibrated CCC score",
                "role_in_schematic": "final LR-specific communication score used for ranking and validation",
                "claim_supported": "Outputs are used for LR ranking, spatial maps and downstream validation.",
                "claim_boundary": "Scores require null/guardrail analysis before FDR-level claims.",
            },
        ]
    )


def draw_architecture(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.02, y=1.01)
    ax.text(
        0.02,
        0.965,
        "Identity-aware multi-scale CCC model",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    nodes = {
        "sender": (0.12, 0.72, "Sender\ncell/spot", PALETTE["spatialself"]),
        "receiver": (0.12, 0.38, "Receiver\ncell/spot", PALETTE["success"]),
        "lr": (0.42, 0.72, "LR identity\nembedding", PALETTE["lr"]),
        "wavelet": (0.42, 0.38, "Graph wavelet\nmulti-scale context", PALETTE["wavelet"]),
        "ssl": (0.71, 0.58, "Self-supervised\ncontrastive training", PALETTE["spatialself_dark"]),
        "output": (0.88, 0.58, "Calibrated\nCCC score", PALETTE["weak"]),
    }
    for key, (x, y, label, color) in nodes.items():
        w = 0.19 if key not in {"ssl", "output"} else 0.22
        h = 0.14
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.8,
            color="white",
            fontweight="bold" if key == "output" else "normal",
            linespacing=1.1,
        )

    arrows = [
        ("sender", "lr"),
        ("receiver", "wavelet"),
        ("lr", "ssl"),
        ("wavelet", "ssl"),
        ("ssl", "output"),
        ("sender", "wavelet"),
        ("receiver", "lr"),
    ]
    for a, b in arrows:
        x1, y1 = nodes[a][0], nodes[a][1]
        x2, y2 = nodes[b][0], nodes[b][1]
        ax.annotate(
            "",
            xy=(x2 - 0.11 if x2 > x1 else x2 + 0.11, y2),
            xytext=(x1 + 0.10 if x2 > x1 else x1 - 0.10, y1),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["line"], shrinkA=2, shrinkB=2),
        )

    feature_rows = [
        ("LR-specific ranges", "contact / paracrine / chemokine-like"),
        ("Hard negatives", "high expression, wrong range, wrong niche"),
        ("Receiver response", "target-program support used for validation"),
    ]
    y0 = 0.16
    for i, (head, body) in enumerate(feature_rows):
        y = y0 - i * 0.07
        ax.add_patch(
            patches.Circle((0.065, y), 0.012, transform=ax.transAxes, facecolor=PALETTE["weak"], edgecolor="none")
        )
        ax.text(0.09, y, head, transform=ax.transAxes, ha="left", va="center", fontsize=6.7, color=PALETTE["ink"], fontweight="bold")
        ax.text(0.33, y, body, transform=ax.transAxes, ha="left", va="center", fontsize=6.5, color=PALETTE["muted"])

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.02),
            0.94,
            0.88,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="none",
            edgecolor=PALETTE["neutral_soft"],
            linewidth=0.9,
            transform=ax.transAxes,
        )
    )


def draw_benchmark_design(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    panel_label(ax, "b")
    ax.set_title("Benchmark covers multiple CCC mechanisms", loc="left", pad=8)
    mechanisms = [
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    ]
    datasets = [
        ("controlled_simulator_v2", "controlled\nsimulation"),
        ("real_background_semisynthetic_v2", "real-background\nsemi-synthetic"),
    ]
    mat = []
    for bench, _label in datasets:
        sub = coverage[coverage["benchmark"].astype(str).eq(bench)]
        row = []
        for mech in mechanisms:
            present = sub[sub["mechanism"].astype(str).eq(mech)]["present"]
            row.append(bool(present.astype(bool).any()) if not present.empty else False)
        mat.append(row)
    mat = np.asarray(mat, dtype=int)
    ax.imshow(mat, cmap=discrete_cmap(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels([MECHANISM_LABEL[m] for m in mechanisms], rotation=0, ha="center")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([label for _bench, label in datasets])
    ax.tick_params(length=0)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            ax.text(x, y, "✓" if mat[y, x] else "–", ha="center", va="center", fontsize=9, color="white" if mat[y, x] else PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0,
        -0.27,
        "Ground truth includes range-specific mechanisms and explicit high-expression false positives.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
    )


def discrete_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap([PALETTE["neutral_soft"], PALETTE["spatialself"]])


def draw_weak_signal_performance(ax: plt.Axes, weak: pd.DataFrame) -> None:
    panel_label(ax, "c")
    ax.set_title("Weak-signal recovery improves with mechanism-aware tuning", loc="left", pad=8)
    df = weak.copy()
    order = [
        "wavelet default",
        "edge prior",
        "weak mining",
        "focal weak mining",
    ]
    df["display_label"] = pd.Categorical(df["display_label"], categories=order, ordered=True)
    df = df.sort_values("display_label")
    x = np.arange(len(df))
    vals = df["weak_signal_recall_at_threshold_mean"].astype(float).values
    err = df["weak_signal_recall_at_threshold_std"].astype(float).fillna(0).values
    colors = [PALETTE["neutral"], PALETTE["wavelet"], PALETTE["weak_soft"], PALETTE["weak"]]
    ax.bar(x, vals, yerr=err, color=colors[: len(x)], edgecolor="white", linewidth=0.8, capsize=2.5)
    ax.set_ylabel("weak-signal recall")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(str(v), 11) for v in df["display_label"]])
    ax.set_ylim(0, max(0.22, float(np.nanmax(vals + err)) * 1.18))
    soften_axes(ax)
    best = df.iloc[int(np.nanargmax(vals))]
    ax.text(
        0.02,
        0.96,
        f"best: {best['display_label']} ({float(best['weak_signal_recall_at_threshold_mean']):.2f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["ink"],
    )
    ax.text(
        0,
        -0.29,
        "Controlled benchmark summary; high FDR at this operating point is audited elsewhere.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
    )


def draw_component_audit(ax: plt.Axes, ablation: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.035, y=1.02)
    ax.set_title("Component audit separates graph, LR identity and expression/distance contributions", loc="left", pad=8)
    keep = [
        "wavelet ref",
        "GAT",
        "GCN",
        "MLP",
        "no LR emb",
        "one-hot LR",
        "no graph",
        "no distance",
        "no expression",
    ]
    df = ablation[ablation["display_label"].isin(keep)].copy()
    df["display_label"] = pd.Categorical(df["display_label"], categories=keep, ordered=True)
    df = df.sort_values("display_label")
    x = np.arange(len(df))
    aupr = df["aupr_mean"].astype(float).values
    weak = df["weak_signal_recall_at_threshold_mean"].astype(float).values
    fdr = df["empirical_fdr_at_threshold_mean"].astype(float).values

    ax2 = ax.twinx()
    bars = ax.bar(x - 0.16, aupr, width=0.32, color=PALETTE["spatialself"], edgecolor="white", linewidth=0.7, label="AUPR")
    ax2.plot(x + 0.16, weak, color=PALETTE["weak"], marker="o", markersize=3.4, linewidth=1.4, label="weak recall")
    ax2.plot(x + 0.16, fdr, color=PALETTE["hard_negative"], marker="s", markersize=3.0, linewidth=1.0, alpha=0.72, label="empirical FDR")
    ax.set_ylabel("AUPR")
    ax2.set_ylabel("weak recall / empirical FDR")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(str(v), 10) for v in df["display_label"]])
    ax.set_ylim(0, max(0.13, float(np.nanmax(aupr)) * 1.22))
    ax2.set_ylim(0, 1.05)
    soften_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(PALETTE["line"])
    ax2.tick_params(axis="y", colors=PALETTE["muted"], width=0.6)
    handles = [bars, ax2.lines[0], ax2.lines[1]]
    labels = ["AUPR", "weak recall", "empirical FDR"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False)
    ax.text(
        0,
        -0.28,
        "Early controlled component audit; use for mechanism support, not final atlas-level model selection.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
    )


def wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def write_manifest(out_stem: Path) -> None:
    rows = [
        {
            "figure": "Figure 1",
            "panel": "a",
            "claim": "SpatialSelf combines LR identity, graph-wavelet spatial context and hard-negative self-supervision.",
            "source_data": "outputs/phase13/source_data/figure1_architecture_schematic.csv",
            "review_risk": "Architecture schematic must match implemented model variants and should not imply final FDR closure.",
        },
        {
            "figure": "Figure 1",
            "panel": "b",
            "claim": "Benchmarks include contact, paracrine, long-range chemokine-like and high-expression false-positive settings.",
            "source_data": "outputs/phase13/source_data/figure1_mechanism_coverage.csv",
            "review_risk": "Coverage table proves benchmark design coverage, not performance.",
        },
        {
            "figure": "Figure 1",
            "panel": "c",
            "claim": "Mechanism-aware tuning improves weak-signal recall in the controlled benchmark.",
            "source_data": "outputs/phase13/source_data/figure1_weak_summary.csv",
            "review_risk": "Weak-recall operating point has high empirical FDR and needs paired FDR guardrail panels elsewhere.",
        },
        {
            "figure": "Figure 1",
            "panel": "d",
            "claim": "Controlled component audit separates graph, LR identity and feature contributions.",
            "source_data": "outputs/phase13/source_data/figure1_component_ablation.csv",
            "review_risk": "Component audit is early controlled evidence, not final whole-atlas ablation.",
        },
    ]
    pd.DataFrame(rows).to_csv(f"{out_stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
