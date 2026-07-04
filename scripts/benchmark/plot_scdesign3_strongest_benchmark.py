#!/usr/bin/env python
"""Publication-style figure for the locked scDesign3 benchmark result."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs" / "reference_based_benchmark" / "full_scdesign3_strongest"
TABLES = BENCH / "final_tables"
OUT = BENCH / "figures"
SOURCE = BENCH / "source_data"


PALETTE = {
    "ink": "#20252B",
    "muted": "#68727D",
    "line": "#C9D0D8",
    "grid": "#E8EBEF",
    "bg": "#F7F8FA",
    "spatialself": "#1F6F8B",
    "commot": "#6E7781",
    "cellchat": "#A8B0B8",
    "spatalk": "#B9914B",
    "cellnest": "#8874A3",
    "deeptalk": "#C1C7CE",
    "risk": "#B95E5A",
    "good": "#2F8C62",
}

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "DeepTalk", "SpaTalk", "CellNEST"]
METHOD_COLORS = {
    "SpatialSelf": PALETTE["spatialself"],
    "COMMOT": PALETTE["commot"],
    "CellChat": PALETTE["cellchat"],
    "DeepTalk": PALETTE["deeptalk"],
    "SpaTalk": PALETTE["spatalk"],
    "CellNEST": PALETTE["cellnest"],
}
DATASET_LABELS = {
    "lymph_scdesign3": "Lymph node",
    "melanoma_scdesign3": "Melanoma",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid: str | None = "x") -> None:
    if grid:
        ax.grid(axis=grid, color=PALETTE["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(TABLES / "scdesign3_strongest_comparison_summary.csv")
    detail = pd.read_csv(TABLES / "scdesign3_strongest_comparison_detail.csv")
    summary["method_label"] = pd.Categorical(summary["method_label"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method_label")
    detail["method_label"] = pd.Categorical(detail["method_label"], METHOD_ORDER, ordered=True)
    detail = detail.sort_values(["dataset", "method_label"])
    return summary, detail


def save_outputs(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def plot_benchmark_scope(ax: plt.Axes, detail: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.05, y=1.07)
    ax.set_axis_off()
    ax.set_title("Reference-based benchmark scope", loc="left", pad=8)

    completed = detail["aupr"].notna().sum()
    total = len(detail)
    n_candidates = (
        detail[["dataset", "n_candidates"]]
        .drop_duplicates()
        .assign(n_candidates=lambda x: pd.to_numeric(x["n_candidates"], errors="coerce"))
    )
    total_candidates = int(n_candidates["n_candidates"].sum())

    cards = [
        ("2", "datasets"),
        ("6", "methods"),
        (f"{total_candidates:,}", "candidates"),
        (f"{completed}/{total}", "scored"),
    ]
    for i, (value, label) in enumerate(cards):
        x0 = 0.015 + i * 0.238
        ax.add_patch(
            plt.Rectangle(
                (x0, 0.70),
                0.205,
                0.19,
                transform=ax.transAxes,
                facecolor=PALETTE["bg"],
                edgecolor=PALETTE["line"],
                linewidth=0.6,
            )
        )
        ax.text(x0 + 0.1025, 0.815, value, transform=ax.transAxes, ha="center", va="center", fontsize=11.8, fontweight="bold", color=PALETTE["ink"])
        ax.text(x0 + 0.1025, 0.74, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.3, color=PALETTE["muted"], linespacing=1.05)

    rows = METHOD_ORDER
    cols = ["lymph_scdesign3", "melanoma_scdesign3"]
    x_l, x_m, x_note = 0.30, 0.49, 0.685
    y0 = 0.55
    dy = 0.063
    ax.text(0.02, y0 + 0.055, "method", transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=PALETTE["muted"])
    for j, col in enumerate(cols):
        ax.text([x_l, x_m][j], y0 + 0.055, DATASET_LABELS[col], transform=ax.transAxes, ha="center", va="center", fontsize=6.8, color=PALETTE["muted"])
    ax.text(x_note, y0 + 0.055, "status note", transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=PALETTE["muted"])
    for i, method in enumerate(rows):
        y = y0 - i * dy
        ax.text(0.02, y, method, transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=PALETTE["ink"])
        for j, col in enumerate(cols):
            sub = detail[(detail["dataset"].eq(col)) & (detail["method_label"].astype(str).eq(method))]
            has_metric = bool(len(sub) and pd.notna(sub.iloc[0]["aupr"]))
            status = str(sub.iloc[0]["status"]) if len(sub) else "missing"
            if has_metric:
                color, marker = PALETTE["good"], "o"
            elif "failed" in status:
                color, marker = PALETTE["risk"], "x"
            else:
                color, marker = PALETTE["line"], "o"
            ax.scatter([x_l, x_m][j], y, s=28, color=color, marker=marker, linewidth=0.9, transform=ax.transAxes, zorder=3)
        note = ""
        if method == "CellNEST":
            note = "lymph failed"
        elif method == "DeepTalk":
            note = "official adapter"
        elif method == "SpaTalk":
            note = "fallback score"
        ax.text(x_note, y, note, transform=ax.transAxes, ha="left", va="center", fontsize=6.0, color=PALETTE["muted"])

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["good"], markeredgecolor=PALETTE["good"], markersize=4.3, label="scored"),
        Line2D([0], [0], marker="x", color=PALETTE["risk"], markersize=4.5, label="failed"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.015, 0.025), ncol=2, handletextpad=0.4, columnspacing=1.1)


def plot_overall_metrics(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.11, y=1.07)
    ax.set_title("SpatialSelf has the strongest overall benchmark profile", loc="left", pad=8)
    metrics = [
        ("aupr", "AUPR", "higher"),
        ("top_k_precision", "top-k\nprecision", "higher"),
        ("empirical_fdr", "empirical\nFDR", "lower"),
    ]
    xbase = np.arange(len(metrics))
    offsets = np.linspace(-0.32, 0.32, len(METHOD_ORDER))
    width = 0.095
    for k, method in enumerate(METHOD_ORDER):
        row = summary[summary["method_label"].astype(str).eq(method)]
        if row.empty:
            continue
        vals = [float(row.iloc[0][m[0]]) for m in metrics]
        ax.bar(
            xbase + offsets[k],
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            label=method,
            zorder=3 if method == "SpatialSelf" else 2,
        )
    ax.set_xticks(xbase)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("mean across datasets")
    clean_axes(ax, grid="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, handlelength=1.0, columnspacing=0.9)

    ss = summary[summary["method_label"].astype(str).eq("SpatialSelf")].iloc[0]
    commot = summary[summary["method_label"].astype(str).eq("COMMOT")].iloc[0]
    txt = f"+{ss['aupr'] - commot['aupr']:.2f} AUPR vs COMMOT\n-{commot['empirical_fdr'] - ss['empirical_fdr']:.2f} FDR"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=PALETTE["spatialself"])


def plot_dataset_aupr(ax: plt.Axes, detail: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.11, y=1.07)
    ax.set_title("AUPR advantage is reproduced in both tissue backgrounds", loc="left", pad=8)
    scored = detail[pd.notna(detail["aupr"])].copy()
    y_positions = {m: i for i, m in enumerate(METHOD_ORDER[::-1])}
    x_positions = {"lymph_scdesign3": 0, "melanoma_scdesign3": 1}

    for method in METHOD_ORDER:
        sub = scored[scored["method_label"].astype(str).eq(method)]
        if len(sub) > 1:
            sub = sub.sort_values("dataset")
            ax.plot([x_positions[d] for d in sub["dataset"]], [sub["aupr"].iloc[0], sub["aupr"].iloc[1]], color=METHOD_COLORS[method], lw=1.0 if method == "SpatialSelf" else 0.65, alpha=0.9 if method == "SpatialSelf" else 0.55)
        for _, row in sub.iterrows():
            ax.scatter(
                x_positions[row["dataset"]],
                row["aupr"],
                s=45 if method == "SpatialSelf" else 25,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.55,
                zorder=4 if method == "SpatialSelf" else 3,
            )

    for dataset, x in x_positions.items():
        ss = scored[(scored["dataset"].eq(dataset)) & (scored["method_label"].astype(str).eq("SpatialSelf"))].iloc[0]
        ax.text(x + 0.03, ss["aupr"] + 0.025, f"{ss['aupr']:.2f}", ha="left", va="bottom", fontsize=6.4, color=PALETTE["spatialself"])

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(0, 0.70)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([DATASET_LABELS["lymph_scdesign3"], DATASET_LABELS["melanoma_scdesign3"]])
    ax.set_ylabel("AUPR")
    clean_axes(ax, grid="y")

    handles = [
        Line2D([0], [0], marker="o", color=METHOD_COLORS[m], markerfacecolor=METHOD_COLORS[m], markeredgecolor="white", lw=0.8, markersize=4.2, label=m)
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.4)


def plot_precision_fdr(ax: plt.Axes, detail: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.11, y=1.07)
    ax.set_title("Top predictions remain cleaner at comparable recall pressure", loc="left", pad=8)
    scored = detail[pd.notna(detail["top_k_precision"]) & pd.notna(detail["empirical_fdr"])].copy()
    markers = {"lymph_scdesign3": "o", "melanoma_scdesign3": "s"}
    for _, row in scored.iterrows():
        method = str(row["method_label"])
        ax.scatter(
            row["empirical_fdr"],
            row["top_k_precision"],
            s=55 if method == "SpatialSelf" else 32,
            color=METHOD_COLORS[method],
            marker=markers[row["dataset"]],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.98 if method in {"SpatialSelf", "COMMOT", "CellChat"} else 0.75,
            zorder=4 if method == "SpatialSelf" else 3,
        )

    for method in ["SpatialSelf", "COMMOT", "CellChat"]:
        sub = scored[scored["method_label"].astype(str).eq(method)]
        if sub.empty:
            continue
        x = float(sub["empirical_fdr"].mean())
        y = float(sub["top_k_precision"].mean())
        ax.text(x + 0.018, y + 0.012, method, ha="left", va="center", fontsize=6.2, color=METHOD_COLORS[method])

    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(-0.03, 0.88)
    ax.set_xlabel("empirical FDR at threshold (lower is better)")
    ax.set_ylabel("top-k precision (higher is better)")
    clean_axes(ax)
    dataset_handles = [
        Line2D([0], [0], marker=markers[d], color="none", markerfacecolor=PALETTE["muted"], markeredgecolor="white", markersize=4.5, label=DATASET_LABELS[d])
        for d in ["lymph_scdesign3", "melanoma_scdesign3"]
    ]
    ax.legend(handles=dataset_handles, loc="lower left", frameon=False, handletextpad=0.4)


def export_source_data(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE / "scdesign3_strongest_figure_summary.csv", index=False)
    detail.to_csv(SOURCE / "scdesign3_strongest_figure_detail.csv", index=False)
    manifest = pd.DataFrame(
        [
            ("a", "benchmark scope and run status", "scdesign3_strongest_figure_detail.csv"),
            ("b", "mean benchmark metrics across datasets", "scdesign3_strongest_figure_summary.csv"),
            ("c", "dataset-level AUPR", "scdesign3_strongest_figure_detail.csv"),
            ("d", "top-k precision versus empirical FDR", "scdesign3_strongest_figure_detail.csv"),
        ],
        columns=["panel", "content", "source_data"],
    )
    manifest.to_csv(SOURCE / "scdesign3_strongest_figure_manifest.csv", index=False)


def main() -> None:
    apply_style()
    summary, detail = load_data()
    export_source_data(summary, detail)

    fig = plt.figure(figsize=(7.35, 5.05), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.935,
        wspace=0.34,
        hspace=0.60,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_benchmark_scope(ax_a, detail)
    plot_overall_metrics(ax_b, summary)
    plot_dataset_aupr(ax_c, detail)
    plot_precision_fdr(ax_d, detail)

    stem = OUT / "scdesign3_strongest_benchmark_figure"
    save_outputs(fig, stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
