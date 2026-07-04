#!/usr/bin/env python
"""Nature-style benchmark figure for the locked scDesign3 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs" / "reference_based_benchmark" / "full_scdesign3_strongest"
TABLES = BENCH / "final_tables"
OUT = BENCH / "figures"
SOURCE = BENCH / "source_data"

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "DeepTalk", "SpaTalk", "CellNEST"]
METHOD_DIRS = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "CellChat": "cellchat",
    "DeepTalk": "deeptalk",
    "SpaTalk": "spatalk",
    "CellNEST": "cellnest",
}
DATASET_DIRS = {"lymph_scdesign3": "lymph", "melanoma_scdesign3": "melanoma"}
MECHANISMS = [
    ("juxtacrine_contact", "Contact"),
    ("short_range_paracrine", "Short"),
    ("medium_range_paracrine", "Medium"),
    ("long_range_chemokine_niche", "Niche"),
]

COL = {
    "ink": "#22272E",
    "muted": "#69737D",
    "axis": "#C9D0D8",
    "grid": "#EBEEF2",
    "paper": "#FFFFFF",
    "soft": "#F6F8FA",
    "spatialself": "#177E89",
    "spatialself_light": "#DDEFF0",
    "commot": "#5D6873",
    "cellchat": "#AEB7C2",
    "other": "#D5DAE0",
    "accent": "#C9973A",
    "risk": "#B85C5A",
}

METHOD_COLORS = {
    "SpatialSelf": COL["spatialself"],
    "COMMOT": COL["commot"],
    "CellChat": COL["cellchat"],
    "DeepTalk": COL["other"],
    "SpaTalk": "#C9AA67",
    "CellNEST": "#B9A8C8",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.06, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COL["ink"],
    )


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(TABLES / "scdesign3_strongest_comparison_summary.csv")
    summary["method_label"] = pd.Categorical(summary["method_label"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method_label").copy()
    summary["discovery_purity"] = 1.0 - summary["empirical_fdr"]
    return summary


def load_mechanism_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, dataset_dir in DATASET_DIRS.items():
        for method, method_dir in METHOD_DIRS.items():
            files = sorted((BENCH / dataset_dir / method_dir).glob("*_metrics.json"))
            metrics: dict[str, object] = {}
            if files:
                with files[0].open() as fh:
                    metrics = json.load(fh)
            for mech_key, mech_label in MECHANISMS:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "mechanism": mech_key,
                        "mechanism_label": mech_label,
                        "aupr": metrics.get(f"aupr__{mech_key}", np.nan),
                    }
                )
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df["mechanism_label"] = pd.Categorical(df["mechanism_label"], [m[1] for m in MECHANISMS], ordered=True)
    return df.sort_values(["dataset", "method", "mechanism_label"])


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.8,
            color=COL["axis"],
        )
    )


def draw_schematic(ax: plt.Axes) -> None:
    panel_label(ax, "a", x=-0.025, y=1.02)
    ax.set_axis_off()
    ax.set_title("Reference-based benchmark design", loc="left", pad=7)

    blocks = [
        (0.04, 0.39, 0.18, 0.30, "Real ST", "coordinates\ncell types\nsparsity"),
        (0.30, 0.39, 0.18, 0.30, "scDesign3", "reference-based\nexpression"),
        (0.56, 0.39, 0.23, 0.30, "LR injection", ""),
        (0.86, 0.39, 0.10, 0.30, "Ranking", "AUPR\ntop-k\nFDR"),
    ]
    for x, y, w, h, title, body in blocks:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor=COL["soft"],
                edgecolor=COL["axis"],
                linewidth=0.65,
            )
        )
        ax.text(x + w / 2, y + h * 0.66, title, transform=ax.transAxes, ha="center", va="center", fontsize=7.4, color=COL["ink"], fontweight="bold")
        ax.text(x + w / 2, y + h * 0.32, body, transform=ax.transAxes, ha="center", va="center", fontsize=5.45, color=COL["muted"], linespacing=1.05)
    for x0, x1 in [(0.22, 0.30), (0.48, 0.56), (0.79, 0.86)]:
        add_arrow(ax, (x0 + 0.015, 0.53), (x1 - 0.015, 0.53))

    # Mechanism glyphs inside injection block.
    glyph_y = 0.50
    glyph_x = [0.595, 0.675, 0.755]
    for gx, radius, label in zip(glyph_x, [0.025, 0.045, 0.065], ["contact", "local", "niche"], strict=True):
        ax.add_patch(Circle((gx, glyph_y), radius, transform=ax.transAxes, facecolor="none", edgecolor=COL["spatialself"], linewidth=0.9, alpha=0.8))
        ax.scatter([gx - radius * 0.45, gx + radius * 0.45], [glyph_y, glyph_y], s=[18, 18], transform=ax.transAxes, color=[COL["spatialself"], COL["accent"]], edgecolors="white", linewidths=0.35, zorder=3)
        ax.text(gx, glyph_y - 0.105, label, transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=COL["muted"])


def draw_overall(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.09, y=1.04)
    ax.set_title("Overall benchmark profile", loc="left", pad=7)
    metrics = [
        ("aupr", "AUPR"),
        ("top_k_precision", "top-k precision"),
        ("discovery_purity", "1 - empirical FDR"),
    ]
    y = np.arange(len(METHOD_ORDER))[::-1]
    offsets = [0.23, 0.0, -0.23]
    markers = ["o", "s", "^"]
    for (metric, label), off, marker in zip(metrics, offsets, markers, strict=True):
        vals = summary.set_index("method_label").loc[METHOD_ORDER, metric].to_numpy(float)
        ax.scatter(vals, y + off, s=28, marker=marker, color=[METHOD_COLORS[m] for m in METHOD_ORDER], edgecolors="white", linewidths=0.45, zorder=3, label=label)
        for v, yy, method in zip(vals, y + off, METHOD_ORDER, strict=True):
            if method == "SpatialSelf":
                ax.plot([0, v], [yy, yy], color=COL["spatialself"], linewidth=1.15, alpha=0.75, zorder=2)
            else:
                ax.plot([0, v], [yy, yy], color=COL["axis"], linewidth=0.55, alpha=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("score")
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COL["axis"])
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.35, borderpad=0.1)
    ss = summary[summary["method_label"].astype(str).eq("SpatialSelf")].iloc[0]
    commot = summary[summary["method_label"].astype(str).eq("COMMOT")].iloc[0]
    ax.text(
        0.80,
        y[0] + 0.35,
        f"+{ss['aupr'] - commot['aupr']:.2f} AUPR vs COMMOT",
        ha="right",
        va="center",
        fontsize=6.3,
        color=COL["spatialself"],
    )


def draw_mechanism(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.07, y=1.04)
    ax.set_title("Mechanism-specific AUPR", loc="left", pad=7)
    keep = ["SpatialSelf", "COMMOT", "CellChat"]
    mean = (
        metrics[metrics["method"].astype(str).isin(keep)]
        .groupby(["method", "mechanism_label"], observed=True)["aupr"]
        .mean()
        .reset_index()
    )
    pivot = mean.pivot(index="method", columns="mechanism_label", values="aupr").reindex(index=keep, columns=[m[1] for m in MECHANISMS])
    arr = pivot.to_numpy(float)
    cmap = mpl.colormaps["YlGnBu"].copy()
    im = ax.imshow(arr, vmin=0, vmax=0.85, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.1, color="white" if val > 0.43 else COL["ink"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COL["axis"])
        spine.set_linewidth(0.6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.033, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.set_label("AUPR", rotation=270, labelpad=9)


def export_source(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE / "scdesign3_benchmark_nature_overall.csv", index=False)
    metrics.to_csv(SOURCE / "scdesign3_benchmark_nature_mechanism.csv", index=False)
    (
        metrics.groupby(["method", "mechanism_label"], observed=True)["aupr"]
        .mean()
        .reset_index()
        .to_csv(SOURCE / "scdesign3_benchmark_nature_mechanism_mean.csv", index=False)
    )
    pd.DataFrame(
        [
            ("a", "reference-based benchmark construction", "schematic"),
            ("b", "overall benchmark metrics", "scdesign3_benchmark_nature_overall.csv"),
            ("c", "mechanism-specific AUPR for key methods", "scdesign3_benchmark_nature_mechanism_mean.csv"),
        ],
        columns=["panel", "content", "source_data"],
    ).to_csv(SOURCE / "scdesign3_benchmark_nature_manifest.csv", index=False)


def save_outputs(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def main() -> None:
    apply_style()
    summary = load_summary()
    metrics = load_mechanism_metrics()
    export_source(summary, metrics)

    fig = plt.figure(figsize=(7.25, 4.65), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.88, 1.0],
        width_ratios=[0.95, 1.15],
        left=0.075,
        right=0.975,
        bottom=0.11,
        top=0.93,
        hspace=0.52,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_schematic(ax_a)
    draw_overall(ax_b, summary)
    draw_mechanism(ax_c, metrics)

    stem = OUT / "scdesign3_benchmark_nature"
    save_outputs(fig, stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
