#!/usr/bin/env python
"""Publication-style five-method native lymph node validation/stress-test panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

METHOD_ORDER = ["spatialself", "commot", "cellchat", "cellnest", "spatalk"]
METHOD_LABEL = {
    "spatialself": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "cellnest": "CellNEST",
    "spatalk": "SpaTalk",
}
METHOD_COLORS = {
    "spatialself": "#0F4D92",
    "commot": "#7884B4",
    "cellchat": "#B4C0E4",
    "cellnest": "#E4CCD8",
    "spatalk": "#CFCECE",
}
NEG = "#B64342"
POS = "#2F6B4F"
NEUTRAL = "#767676"
TEXT = "#202020"
SUBTLE = "#666666"
GRID = "#D9D9D9"
T_ZONE = "#5B8CC0"
OTHER_ZONE = "#D7D7D7"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    vectors = pd.read_csv(RESULT_DIR / "lymph_node_full_directionality_receiver_vectors.csv")
    flow = pd.read_csv(RESULT_DIR / "lymph_node_full_directionality_lr_summary.csv")
    impact = pd.read_csv(RESULT_DIR / "lymph_node_full_lr_level_impact_summary.csv")
    modules = pd.read_csv(RESULT_DIR / "lymph_node_full_modules_summary.csv")
    five = pd.read_csv(RESULT_DIR / "lymph_node_full_five_method_panel_summary.csv")

    source = build_source(cells, vectors, flow, impact, modules, five)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"lymph_node_full_five_method_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.42, 0.9, 0.98, 0.82],
        height_ratios=[1.02, 0.92, 1.05],
        left=0.065,
        right=0.985,
        top=0.875,
        bottom=0.09,
        wspace=0.48,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])
    ax_c = fig.add_subplot(gs[0, 3])
    ax_d = fig.add_subplot(gs[1, 1:3])
    ax_e = fig.add_subplot(gs[1, 3])
    ax_f = fig.add_subplot(gs[2, 1:])

    plot_spatial_ccr7_flow(ax_a, source["spatial_flow"])
    plot_downstream_stress(ax_b, source["method_summary"])
    plot_raw_top_lr(ax_c, source["method_summary"])
    plot_expected_flow_heatmap(ax_d, source["flow_heatmap"])
    plot_module_cleanliness(ax_e, source["module_cleanliness"])
    plot_interpretation_table(ax_f, source["interpretation_table"])

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Native lymph node Visium supports CCR7 flow but exposes receptor-shared false positives",
        x=0.065,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.065,
        0.925,
        "Five-method analysis on a native 36,601-gene human lymph node Visium source: CCL19/CCL21-CCR7 shows T-zone-local flow, while high-expression controls remain a stress-test risk.",
        fontsize=6.3,
        color=SUBTLE,
    )
    fig.text(
        0.065,
        0.035,
        "Real ST has no native CCC ground truth. Directionality is inferred from weighted sender-to-receiver edges; niches are marker-rule proxies. Downstream delta R2 is residualized by library size and spatial coordinates.",
        fontsize=5.75,
        color=SUBTLE,
    )
    save_figure(fig, FIG_DIR / "phase11_lymph_node_full_five_method_panel")
    plt.close(fig)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.frameon": False,
        }
    )


def build_source(
    cells: pd.DataFrame,
    vectors: pd.DataFrame,
    flow: pd.DataFrame,
    impact: pd.DataFrame,
    modules: pd.DataFrame,
    five: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ccr7 = vectors[
        vectors["method"].isin(["spatialself", "commot", "cellchat", "cellnest"])
        & vectors["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7"])
        & vectors["incoming_score"].gt(0)
    ].copy()
    ccr7["rank"] = ccr7.groupby(["method", "lr_pair"])["incoming_score"].rank(ascending=False, method="first")
    ccr7 = ccr7[ccr7["rank"].le(220)].copy()
    ccr7 = ccr7.merge(cells[["cell_id", "niche"]], left_on="receiver_id", right_on="cell_id", how="left", suffixes=("", "_cell"))

    method_summary = five[five["method"].isin(METHOD_ORDER)].copy()
    method_summary["method_label"] = method_summary["method"].map(METHOD_LABEL)
    method_summary["top_lr_short"] = method_summary["baseline_top_lr_by_mean_score"].map(short_lr_id)

    flow_focus = flow[flow["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CD48-CD244"])].copy()
    flow_focus = flow_focus[flow_focus["method"].isin(METHOD_ORDER)]
    flow_focus["method_label"] = flow_focus["method"].map(METHOD_LABEL)

    clean = []
    for method in METHOD_ORDER:
        sub = modules[modules["method"].eq(method)]
        clean.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "clean_true_modules": int(((sub["n_true_lr"].gt(0)) & (sub["n_high_expression_negative"].eq(0))).sum()),
                "negative_containing_modules": int(sub["n_high_expression_negative"].gt(0).sum()),
                "max_module_delta_r2": float(sub["mean_downstream_delta_r2"].max()) if len(sub) else np.nan,
            }
        )
    module_cleanliness = pd.DataFrame(clean)

    interpretation = method_summary[
        [
            "method",
            "method_label",
            "best_priority_lr",
            "priority_minus_max_negative",
            "ccl19_ccr7_expected_flow_fraction",
            "ccl21_ccr7_expected_flow_fraction",
            "interpretation",
        ]
    ].copy()
    interpretation["best_priority_lr"] = interpretation["best_priority_lr"].map(short_lr_id)
    interpretation["ccr7_mean_flow"] = interpretation[
        ["ccl19_ccr7_expected_flow_fraction", "ccl21_ccr7_expected_flow_fraction"]
    ].mean(axis=1, skipna=True)

    return {
        "spatial_flow": ccr7,
        "method_summary": method_summary,
        "flow_heatmap": flow_focus,
        "module_cleanliness": module_cleanliness,
        "interpretation_table": interpretation,
    }


def plot_spatial_ccr7_flow(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("CCR7 flow converges on proxy T-cell zone", loc="left", pad=5, fontsize=7.6)
    bg = table.drop_duplicates("receiver_id")
    colors = np.where(bg["receiver_niche"].eq("T_cell_zone_proxy"), T_ZONE, OTHER_ZONE)
    ax.scatter(bg["receiver_x"], -bg["receiver_y"], s=7.2, c=colors, alpha=0.42, linewidths=0)
    for method in ["spatialself", "commot"]:
        sub = table[table["method"].eq(method)].sort_values("incoming_score", ascending=False)
        if sub.empty:
            continue
        step = max(len(sub) // 100, 1)
        q = sub.iloc[::step]
        ax.quiver(
            q["receiver_x"],
            -q["receiver_y"],
            q["vector_dx"] * 0.38,
            -q["vector_dy"] * 0.38,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002 if method == "spatialself" else 0.0015,
            color=METHOD_COLORS[method],
            alpha=0.55,
            label=METHOD_LABEL[method],
        )
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.legend(loc="lower left", fontsize=5.3, handlelength=1.3)
    ax.text(
        0.02,
        0.98,
        "blue background: proxy T-cell zone\narrows: top CCR7 receiver vectors",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=SUBTLE,
    )


def plot_downstream_stress(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.set_index("method").reindex(METHOD_ORDER).reset_index()
    y = np.arange(len(plot))
    ax.barh(
        y,
        plot["mean_priority_delta_r2"],
        color=[METHOD_COLORS[m] for m in plot["method"]],
        edgecolor="none",
        height=0.58,
        label="positive LR mean",
    )
    ax.scatter(plot["max_negative_delta_r2"], y, color=NEG, s=26, zorder=4, label="max high-expression negative")
    for yi, row in zip(y, plot.itertuples(index=False)):
        if np.isfinite(row.mean_priority_delta_r2) and np.isfinite(row.max_negative_delta_r2):
            ax.plot([row.mean_priority_delta_r2, row.max_negative_delta_r2], [yi, yi], color="#858585", linewidth=0.75, zorder=2)
            gap = row.priority_minus_max_negative
            ax.text(
                max(row.mean_priority_delta_r2, row.max_negative_delta_r2) + 0.01,
                yi,
                f"{gap:+.3f}",
                va="center",
                ha="left",
                fontsize=5.4,
                color=POS if gap > 0 else NEG,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["method_label"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("target impact delta R2")
    ax.set_title("Downstream impact and negative-control stress", loc="left", pad=5, fontsize=7.4)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.1, handletextpad=0.4)


def plot_raw_top_lr(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.set_index("method").reindex(METHOD_ORDER).reset_index()
    is_neg = plot["baseline_top_lr_is_high_expression_negative"].astype(bool).fillna(False)
    y = np.arange(len(plot))
    ax.scatter(np.where(is_neg, 1, 0), y, c=np.where(is_neg, NEG, POS), s=42, zorder=3)
    for yi, row in zip(y, plot.itertuples(index=False)):
        ax.text(1.08 if row.baseline_top_lr_is_high_expression_negative else 0.08, yi, row.top_lr_short, va="center", fontsize=5.2, color=SUBTLE)
    ax.set_xlim(-0.18, 1.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true/\npriority", "high-expression\nnegative"], fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["method_label"], fontsize=5.6)
    ax.invert_yaxis()
    ax.set_title("Top LR by mean score", loc="left", pad=5, fontsize=7.4)
    ax.axvline(1, color=NEG, linewidth=0.7, alpha=0.35)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_expected_flow_heatmap(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["CCL19-CCR7", "CCL21-CCR7", "CD48-CD244", "CXCL13-CXCR5"]
    matrix = (
        table.pivot_table(index="method", columns="lr_pair", values="expected_flow_score_fraction", aggfunc="first")
        .reindex(METHOD_ORDER)
        .reindex(columns=pairs)
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", norm=Normalize(0, 1), aspect="auto")
    for i, method in enumerate(matrix.index):
        for j, pair in enumerate(matrix.columns):
            val = matrix.loc[method, pair]
            label = "NA" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=5.5, color="white" if pd.notna(val) and val > 0.62 else TEXT)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(pairs, rotation=28, ha="right", fontsize=5.8)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=5.8)
    ax.set_title("Expected niche-flow fraction", loc="left", pad=5, fontsize=7.4)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.1, length=2)


def plot_module_cleanliness(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.set_index("method").reindex(METHOD_ORDER).reset_index()
    y = np.arange(len(plot))
    ax.barh(y, plot["clean_true_modules"], color=POS, height=0.52, label="true modules\nwithout negatives")
    ax.barh(y, -plot["negative_containing_modules"], color=NEG, height=0.52, label="modules with\nnegative controls")
    ax.axvline(0, color=TEXT, linewidth=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["method_label"], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["3", "2", "1", "0", "1", "2", "3"], fontsize=5.2)
    ax.set_xlabel("module count")
    ax.set_title("Module-level control leakage", loc="left", pad=5, fontsize=7.4)
    ax.text(
        0.76,
        -0.26,
        "clean true modules",
        transform=ax.transAxes,
        color=POS,
        fontsize=4.9,
        ha="center",
        va="top",
    )
    ax.text(
        0.22,
        -0.26,
        "modules with negatives",
        transform=ax.transAxes,
        color=NEG,
        fontsize=4.9,
        ha="center",
        va="top",
    )


def plot_interpretation_table(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.axis("off")
    plot = table.set_index("method").reindex(METHOD_ORDER).reset_index()
    rows = []
    for row in plot.itertuples(index=False):
        flow = "NA" if pd.isna(row.ccr7_mean_flow) else f"{row.ccr7_mean_flow:.2f}"
        gap = "NA" if pd.isna(row.priority_minus_max_negative) else f"{row.priority_minus_max_negative:+.3f}"
        rows.append([row.method_label, row.best_priority_lr, flow, gap, compress_interpretation(row.interpretation)])
    tbl = ax.table(
        cellText=rows,
        colLabels=["Method", "best priority LR", "CCR7 flow", "impact gap", "take-home"],
        colWidths=[0.13, 0.18, 0.12, 0.12, 0.43],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.2)
    tbl.scale(1.0, 1.42)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.34)
        cell.set_edgecolor("#D6D6D6")
        if r == 0:
            cell.set_facecolor("#ECEFF3")
            cell.set_text_props(weight="bold", color=TEXT)
        elif c == 3:
            text = cell.get_text().get_text()
            cell.set_facecolor("#E6F1E8" if text.startswith("+") else "#F5E0E0" if text.startswith("-") else "white")
        else:
            cell.set_facecolor("white")
    ax.set_title("What this native full-transcriptome panel supports", loc="left", pad=6)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )


def short_lr_id(value: object) -> str:
    text = str(value)
    if not text or text == "nan":
        return ""
    for prefix in ["ln_full_lr_000_", "ln_full_lr_001_", "ln_full_lr_002_", "ln_full_lr_003_", "ln_full_lr_004_", "ln_full_lr_005_", "ln_full_lr_006_", "ln_full_lr_007_", "ln_full_lr_008_"]:
        text = text.replace(prefix, "")
    text = text.replace("_negative", " neg.").replace("_", "-")
    return text


def compress_interpretation(text: object) -> str:
    value = str(text)
    if "no informative" in value:
        return "runs, but score output is not informative here"
    if "negatives are suppressed" in value:
        return "downstream negatives suppressed; raw top LR still negative"
    if "receptor-shared negatives remain competitive" in value:
        return "CCR7 flow supported; receptor-shared negatives compete"
    if "High-expression negative controls" in value:
        return "high-expression negatives dominate ranking"
    return value[:80]


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
