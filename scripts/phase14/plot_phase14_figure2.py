#!/usr/bin/env python
"""Phase 14 redesigned Figure 2: baseline tradeoff figure."""

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
        width_ratios=[1.05, 1.38, 1.0],
        height_ratios=[1.08, 0.9],
        left=0.06,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.42,
        hspace=0.55,
    )
    ax_land = fig.add_subplot(gs[0, 0])
    ax_trade = fig.add_subplot(gs[0, 1:])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_runtime = fig.add_subplot(gs[1, 1])
    ax_deep = fig.add_subplot(gs[1, 2])

    draw_method_landscape(ax_land, tables["method_landscape"])
    draw_tradeoff_hero(ax_trade, tables["tradeoff"])
    draw_metric_pair(ax_metrics, tables["metric_pair"])
    draw_runtime(ax_runtime, tables["runtime"])
    draw_deep_status(ax_deep, tables["deep_status"])

    add_figure_title(
        fig,
        "SpatialSelf improves the practical tradeoff between weak CCC recovery and hard false positives",
        "Phase 14 redesigned Figure 2: baseline comparison is organized around tradeoff, not table density.",
    )
    stem = FIG / "phase14_figure2_baseline_tradeoff_redesign"
    save_figure(fig, stem)
    write_manifest(stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "formal": P13 / "figure2_formal_summary_compact.csv",
        "tradeoff": P13 / "figure2_weak_tradeoff_gate.csv",
        "runtime": P13 / "figure2_runtime_compact.csv",
        "deep": P13 / "figure2_deep_status_compact.csv",
        "status": P13 / "figure2_status_compact.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 sources: " + "; ".join(missing))
    return {k: pd.read_csv(p) for k, p in files.items()}


def normalize_method_label(label: object) -> str:
    text = str(label).replace("\n", " ").strip()
    if text.startswith("SpatialSelf"):
        return "SpatialSelf"
    return text


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    trade = data["tradeoff"].copy()
    trade["method_clean"] = trade["method_label"].map(normalize_method_label)
    trade = trade[trade["method_clean"].isin(METHOD_ORDER)].copy()
    trade = trade.sort_values("method_clean")

    formal = data["formal"].copy()
    formal["method_clean"] = formal["method_label"].map(normalize_method_label)
    formal = formal[formal["method_clean"].isin(METHOD_ORDER)].copy()
    real = formal[formal["dataset"].eq("real_background_v2_formal")].copy()
    metric_pair = real.groupby("method_clean", as_index=False).agg(
        aupr=("aupr_mean", "mean"),
        high_expression_fp=("false_positive_high_expression_rate_at_threshold_mean", "mean"),
    )
    metric_pair["method_clean"] = pd.Categorical(metric_pair["method_clean"], METHOD_ORDER, ordered=True)
    metric_pair = metric_pair.sort_values("method_clean")

    runtime = data["runtime"].copy()
    runtime["method_clean"] = runtime["method_label"].map(normalize_method_label)
    runtime = runtime[runtime["method_clean"].isin(METHOD_ORDER + ["DeepTalk"])].copy()
    runtime["method_clean"] = pd.Categorical(runtime["method_clean"], METHOD_ORDER + ["DeepTalk"], ordered=True)
    runtime = runtime.sort_values("method_clean")

    status = data["status"].copy()
    status["method_clean"] = status["method_label"].map(normalize_method_label)
    method_landscape = status[status["method_clean"].isin(METHOD_ORDER + ["DeepTalk"])][
        ["method_clean", "evidence_level", "n_datasets"]
    ].drop_duplicates()

    deep = data["deep"].copy()
    deep["usable_for_rank"] = deep["lr_pair_ranking_support"].astype(str).eq("usable")
    deep_status = deep[["method_label", "lr_pair_ranking_support", "manuscript_use", "limitation"]].copy()

    return {
        "tradeoff": trade,
        "metric_pair": metric_pair,
        "runtime": runtime,
        "method_landscape": method_landscape,
        "deep_status": deep_status,
    }


def export_source(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(SOURCE / f"figure2_{name}.csv", index=False)


def draw_method_landscape(ax: plt.Axes, landscape: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.1, y=1.04)
    ax.set_axis_off()
    ax.set_title("Comparison contract", loc="left", pad=8)
    levels = {
        "candidate-edge": (PALETTE["signal"], "formal score"),
        "adapter-light": (PALETTE["weak"], "adapter-light"),
        "status-only": (PALETTE["neutral"], "status-only"),
    }
    y = 0.82
    for method in ["SpatialSelf", "COMMOT", "CellChat", "SpaTalk", "CellNEST", "DeepTalk"]:
        row = landscape[landscape["method_clean"].eq(method)]
        level = str(row["evidence_level"].iloc[0]) if not row.empty else "status-only"
        color, label = levels.get(level, (PALETTE["neutral"], level))
        ax.add_patch(patches.FancyBboxPatch((0.03, y - 0.055), 0.26, 0.08, boxstyle="round,pad=0.01,rounding_size=0.018", facecolor=METHOD_COLORS.get(method, PALETTE["neutral"]), edgecolor="white", transform=ax.transAxes))
        ax.text(0.16, y - 0.015, method, transform=ax.transAxes, ha="center", va="center", fontsize=5.9, color="white", fontweight="bold" if method == "SpatialSelf" else "normal")
        ax.scatter(0.39, y - 0.015, s=42, color=color, transform=ax.transAxes)
        ax.text(0.47, y - 0.015, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.1, color=PALETTE["ink"])
        y -= 0.115
    ax.text(0.03, 0.05, "Deep methods without comparable LR-pair ranking stay status-only.", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=PALETTE["muted"], wrap=True)


def draw_tradeoff_hero(ax: plt.Axes, trade: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.055, y=1.04)
    ax.set_title("Hero readout: weak-signal recovery must be interpreted with FDR", loc="left", pad=8)
    for _, row in trade.iterrows():
        method = row["method_clean"]
        color = METHOD_COLORS.get(method, PALETTE["neutral"])
        is_ss = method == "SpatialSelf"
        ax.scatter(
            row["empirical_fdr_at_threshold_mean"],
            row["weak_signal_recall_at_threshold_mean"],
            s=180 if is_ss else 105,
            color=color,
            edgecolor=PALETTE["ink"] if is_ss else "white",
            linewidth=0.9,
            zorder=4 if is_ss else 3,
        )
        ax.text(row["empirical_fdr_at_threshold_mean"] + 0.018, row["weak_signal_recall_at_threshold_mean"] + 0.018, method, fontsize=6.3, color=PALETTE["ink"])
    ax.axvspan(0, 0.25, color=PALETTE["signal_soft"], alpha=0.55, zorder=0)
    ax.axvline(0.5, color=PALETTE["line"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("empirical FDR")
    ax.set_ylabel("weak-signal recall")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.04, 1.08)
    clean_axes(ax)
    ax.text(0.03, 0.93, "desirable region:\nrecover weak signal\nwithout high FDR", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=PALETTE["signal"])


def draw_metric_pair(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.12, y=1.04)
    ax.set_title("Real-background benchmark", loc="left", pad=8)
    x = np.arange(len(metrics))
    width = 0.34
    colors = [METHOD_COLORS.get(m, PALETTE["neutral"]) for m in metrics["method_clean"]]
    ax.bar(x - width / 2, metrics["aupr"], width=width, color=colors, edgecolor="white", linewidth=0.6, label="AUPR")
    ax.bar(x + width / 2, metrics["high_expression_fp"], width=width, color=PALETTE["risk_soft"], edgecolor=PALETTE["risk"], linewidth=0.6, label="high-expr FP")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["method_clean"], rotation=35, ha="right")
    ax.set_ylim(0, max(0.42, float(max(metrics["aupr"].max(), metrics["high_expression_fp"].max())) * 1.25))
    ax.set_ylabel("score / rate")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")


def draw_runtime(ax: plt.Axes, runtime: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.08, y=1.04)
    ax.set_title("Runtime remains practical in retained runs", loc="left", pad=8)
    run = runtime[runtime["status"].eq("completed")].copy()
    x = np.arange(len(run))
    ax.bar(x, run["wall_clock_seconds"], color=[METHOD_COLORS.get(m, PALETTE["neutral"]) for m in run["method_clean"]], edgecolor="white", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(run["method_clean"], rotation=25, ha="right")
    ax.set_ylabel("wall-clock seconds")
    clean_axes(ax)


def draw_deep_status(ax: plt.Axes, deep: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.12, y=1.04)
    ax.set_axis_off()
    ax.set_title("Deep baseline closure", loc="left", pad=8)
    methods = ["DeepTalk", "HoloNet", "stMLnet", "CytoSignal", "NICHES"]
    y = 0.82
    for method in methods:
        row = deep[deep["method_label"].astype(str).eq(method)]
        status = str(row["lr_pair_ranking_support"].iloc[0]) if not row.empty else "not audited"
        usable = "usable" in status.lower()
        color = PALETTE["signal"] if usable else PALETTE["neutral"]
        ax.scatter(0.08, y, s=54, color=color, transform=ax.transAxes)
        ax.text(0.17, y, method, transform=ax.transAxes, ha="left", va="center", fontsize=6.4, color=PALETTE["ink"], fontweight="bold" if method == "DeepTalk" else "normal")
        ax.text(0.17, y - 0.055, "LR-rank output not comparable" if not usable else "comparable", transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=PALETTE["muted"])
        y -= 0.16
    ax.text(0.02, 0.03, "Shown as status evidence, not scored wins/losses.", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, color=PALETTE["muted"], wrap=True)


def write_manifest(stem: Path) -> None:
    rows = [
        ("a", "Comparable and status-only baselines are visually separated.", "outputs/phase14/source_data/figure2_method_landscape.csv"),
        ("b", "SpatialSelf is positioned by weak-signal recall and empirical FDR rather than raw recall alone.", "outputs/phase14/source_data/figure2_tradeoff.csv"),
        ("c", "Real-background benchmark summarizes AUPR and high-expression false-positive behavior.", "outputs/phase14/source_data/figure2_metric_pair.csv"),
        ("d", "Runtime remains practical in retained runnable baselines.", "outputs/phase14/source_data/figure2_runtime.csv"),
        ("e", "Deep baseline closure remains status-only when LR-pair ranking output is not comparable.", "outputs/phase14/source_data/figure2_deep_status.csv"),
    ]
    pd.DataFrame(rows, columns=["panel", "claim", "source_data"]).to_csv(f"{stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
