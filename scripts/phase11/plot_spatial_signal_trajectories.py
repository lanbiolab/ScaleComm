#!/usr/bin/env python
"""Plot source-to-sink trajectory summaries for Phase 11 directionality evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/trajectory_summary"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

TEXT = "#202020"
SUBTLE = "#666666"
GRID = "#D9D9D9"
METHOD_COLORS = {"spatialself": "#0F4D92", "commot": "#7884B4"}
METHOD_LABEL = {"spatialself": "SpatialSelf", "commot": "COMMOT"}

FOCUS_ROWS = [
    ("visium_hd_crc", "SPP1-CD44"),
    ("lymph_node_expanded", "CXCL13-CXCR5"),
    ("lymph_node_full", "CCL19-CCR7"),
    ("lymph_node_full", "CCL21-CCR7"),
]
CASE_LABELS = {
    "visium_hd_crc": "Visium HD CRC",
    "lymph_node_expanded": "Lymph node expanded benchmark",
    "lymph_node_full": "Native human lymph node Visium",
}


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(OUT_DIR / "phase11_signal_trajectory_summary.csv")
    source = build_source(trajectory)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"phase11_signal_trajectory_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.25, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 1.05, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.13,
        wspace=0.48,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])

    plot_trajectory_arrows(ax_a, source["focus"])
    plot_expected_flow_heatmap(ax_b, source["focus"])
    plot_niche_rule(ax_c, source["focus"])
    plot_summary_table(ax_d, source["table"])

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Source-to-sink trajectories separate local and structured spatial communication",
        x=0.075,
        y=0.97,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.91,
        "Weighted sender and receiver centroids summarize COMMOT-style flow: SPP1-CD44 is local/niche-retained in CRC, whereas lymph-node chemokines align with follicular or T-zone structures.",
        fontsize=6.2,
        color=SUBTLE,
    )
    fig.text(
        0.075,
        0.045,
        "Trajectories are descriptive summaries of top-scoring sender-to-receiver edges, not time-resolved transport or causal ligand movement.",
        fontsize=5.7,
        color=SUBTLE,
    )
    save_figure(fig, FIG_DIR / "phase11_spatial_signal_trajectories")
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
            "legend.frameon": False,
        }
    )


def build_source(trajectory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keep = []
    for dataset_key, lr_pair in FOCUS_ROWS:
        sub = trajectory[trajectory["dataset_key"].eq(dataset_key) & trajectory["lr_pair"].eq(lr_pair)]
        keep.append(sub)
    focus = pd.concat(keep, ignore_index=True)
    focus = focus[focus["method"].isin(["spatialself", "commot"])].copy()
    focus["case_label"] = focus["dataset"].astype(str) + "\n" + focus["lr_pair"].astype(str)
    focus["method_label"] = focus["method"].map(METHOD_LABEL)
    focus["best_flow_expected_any"] = focus.apply(best_flow_expected, axis=1)
    table = focus[
        [
            "dataset",
            "method",
            "lr_pair",
            "centroid_flow_label",
            "top_niche_flow_label",
            "expected_flow_score_fraction",
            "top_niche_rule_fraction",
            "trajectory_length",
            "best_flow_expected_any",
        ]
    ].copy()
    return {"focus": focus, "table": table}


def plot_trajectory_arrows(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("Weighted source-to-sink centroids", loc="left", pad=5)
    cases = list(dict.fromkeys(table["case_label"]))
    y_positions = {case: i for i, case in enumerate(cases[::-1])}
    for case in cases:
        ax.axhline(y_positions[case], color="#EFEFEF", linewidth=0.8, zorder=0)
    for row in table.itertuples(index=False):
        y = y_positions[row.case_label] + (-0.12 if row.method == "spatialself" else 0.12)
        length = np.log1p(row.trajectory_length)
        x0 = 0.08
        x1 = min(0.88, 0.08 + 0.18 + 0.09 * length)
        arrow = FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.4 if row.method == "spatialself" else 1.1,
            color=METHOD_COLORS[row.method],
            alpha=0.9,
        )
        ax.add_patch(arrow)
        ax.scatter([x0], [y], s=18, color=METHOD_COLORS[row.method], edgecolor="white", linewidth=0.35, zorder=3)
        ax.text(x1 + 0.02, y, row.method_label, va="center", fontsize=5.5, color=METHOD_COLORS[row.method])
    ax.set_yticks([y_positions[c] for c in cases])
    ax.set_yticklabels(cases, fontsize=5.6)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.55, len(cases) - 0.45)
    ax.set_xticks([])
    ax.set_xlabel("shorter to longer centroid displacement")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_expected_flow_heatmap(ax: plt.Axes, table: pd.DataFrame) -> None:
    matrix = table.pivot_table(index="case_label", columns="method", values="expected_flow_score_fraction", aggfunc="first")
    matrix = matrix.reindex([CASE_LABELS[d] + "\n" + lr for d, lr in FOCUS_ROWS]).reindex(columns=["spatialself", "commot"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", norm=Normalize(0, 1), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=5.8, color="white" if pd.notna(val) and val > 0.6 else TEXT)
    ax.set_title("Expected-flow support", loc="left", pad=5)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["SpatialSelf", "COMMOT"], rotation=25, ha="right", fontsize=5.8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([label.split("\n")[-1] for label in matrix.index], fontsize=5.7)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5, length=2)


def plot_niche_rule(ax: plt.Axes, table: pd.DataFrame) -> None:
    plot = table.copy()
    plot["case_short"] = plot["dataset_key"].map({"visium_hd_crc": "CRC", "lymph_node_expanded": "LN bench", "lymph_node_full": "LN full"}) + "\n" + plot["lr_pair"]
    xlabels = list(dict.fromkeys(plot["case_short"]))
    x = np.arange(len(xlabels))
    width = 0.35
    for offset, method in [(-0.18, "spatialself"), (0.18, "commot")]:
        vals = plot[plot["method"].eq(method)].set_index("case_short").reindex(xlabels)["top_niche_rule_fraction"].fillna(0)
        ax.bar(x + offset, vals, width=width, color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABEL[method])
    ax.set_title("Top-edge niche consistency", loc="left", pad=5)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=5.3)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(loc="upper right", fontsize=5.1)


def plot_summary_table(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.axis("off")
    rows = []
    focus = table[table["method"].eq("spatialself")].copy()
    focus = focus.sort_values(["dataset", "lr_pair"])
    for row in focus.itertuples(index=False):
        rows.append(
            [
                short_dataset(row.dataset),
                row.lr_pair,
                short_flow(row.top_niche_flow_label),
                f"{row.expected_flow_score_fraction:.2f}" if pd.notna(row.expected_flow_score_fraction) else "NA",
                "yes" if row.best_flow_expected_any else "no",
            ]
        )
    tbl = ax.table(
        cellText=rows,
        colLabels=["Dataset", "LR", "top SpatialSelf flow", "expected\nfraction", "expected\ntop flow"],
        colWidths=[0.2, 0.18, 0.36, 0.13, 0.13],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.4)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.34)
        cell.set_edgecolor("#D6D6D6")
        if r == 0:
            cell.set_facecolor("#ECEFF3")
            cell.set_text_props(weight="bold", color=TEXT)
        elif c == 4:
            cell.set_facecolor("#E5F1E8" if cell.get_text().get_text() == "yes" else "#F3E0E0")
        else:
            cell.set_facecolor("white")
    ax.set_title("SpatialSelf trajectory interpretation", loc="left", pad=5)


def best_flow_expected(row: pd.Series) -> bool:
    for col in ["best_flow_is_expected", "best_flow_expected", "top_niche_flow_expected"]:
        if col in row and pd.notna(row[col]):
            return bool(row[col])
    return False


def short_dataset(value: str) -> str:
    return str(value).replace("Native human lymph node Visium", "LN full").replace("Lymph node expanded benchmark", "LN benchmark").replace("Visium HD CRC", "CRC")


def short_flow(value: object) -> str:
    text = str(value)
    text = text.replace("tumor_myeloid_boundary", "tumor-myeloid")
    text = text.replace("tumor_stromal_boundary", "tumor-stromal")
    text = text.replace("T_cell_zone_proxy", "T zone")
    text = text.replace("B_follicle_proxy", "B follicle")
    text = text.replace("_proxy", "")
    text = text.replace("_", " ")
    return text


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", color=TEXT, ha="left", va="top")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
