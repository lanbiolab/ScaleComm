#!/usr/bin/env python
"""Plot lymph node chemokine directionality panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/lymph_node_phase6_expanded"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
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
    "spatialself": "#2F6B4F",
    "commot": "#456C9B",
    "cellchat": "#7D668E",
    "cellnest": "#B07A4B",
    "spatalk": "#6E6E6E",
}
NICHE_COLORS = {
    "B follicle": "#D8B365",
    "Germinal center": "#8CB369",
    "T cell zone": "#5B8CC0",
    "Mixed": "#B77B9B",
}
TEXT = "#202020"
GRID = "#D8D8D8"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(RESULT_DIR / "lymph_node_directionality_lr_summary.csv")
    vectors = pd.read_csv(RESULT_DIR / "lymph_node_directionality_receiver_vectors.csv")
    flows = pd.read_csv(RESULT_DIR / "lymph_node_directionality_niche_flow.csv")

    source = build_source(summary, vectors, flows)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"lymph_node_directionality_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.2, 1.05, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.885,
        bottom=0.12,
        wspace=0.44,
        hspace=0.56,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    plot_vector_overlay(ax_a, source["vector_overlay"])
    plot_expected_fraction(ax_b, source["expected_fraction"])
    plot_top_truth(ax_c, source["top_truth"])
    plot_niche_heatmap(ax_d, source["cxcl13_flow"], "CXCL13-CXCR5")
    plot_niche_heatmap(ax_e, source["ccl19_flow"], "CCL19-CCR7")

    for label, ax in zip(["a", "b", "c", "d", "e"], [ax_a, ax_b, ax_c, ax_d, ax_e]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Lymph node chemokine flow recovers follicular and T-zone structure",
        x=0.075,
        y=0.975,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.928,
        "Weighted sender-to-receiver edges summarize SpatialSelf and baseline niche flow; CXCL13-CXCR5 is strongly follicular, whereas CCL19-CCR7 directionality is method-dependent.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "Directionality is inferred from scored spatial candidate edges, not causal transport. Expected flows are defined from the injected lymph-node niche rules.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_lymph_node_directionality")
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_source(summary: pd.DataFrame, vectors: pd.DataFrame, flows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overlay = vectors[
        vectors["method"].isin(["spatialself", "commot"])
        & vectors["lr_pair"].isin(["CXCL13-CXCR5", "CCL19-CCR7"])
        & vectors["incoming_score"].gt(0)
    ].copy()
    overlay["rank"] = overlay.groupby(["method", "lr_pair"])["incoming_score"].rank(ascending=False, method="first")
    overlay = overlay[overlay["rank"].le(220)].copy()

    expected = summary[
        summary["method"].isin(METHOD_ORDER)
        & summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4", "CD48-CD244"])
    ].copy()

    top_truth = expected.copy()
    top_truth["method_label"] = top_truth["method"].map(METHOD_LABEL)

    cxcl13 = flows[(flows["method"].eq("spatialself")) & (flows["lr_pair"].eq("CXCL13-CXCR5"))].copy()
    ccl19 = flows[(flows["method"].eq("spatialself")) & (flows["lr_pair"].eq("CCL19-CCR7"))].copy()
    return {
        "vector_overlay": overlay,
        "expected_fraction": expected,
        "top_truth": top_truth,
        "cxcl13_flow": cxcl13,
        "ccl19_flow": ccl19,
    }


def plot_vector_overlay(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_title("Receiver flow fields", loc="left", pad=4)
    pair_style = {
        "CXCL13-CXCR5": {"marker": "o", "alpha": 0.52},
        "CCL19-CCR7": {"marker": "^", "alpha": 0.42},
    }
    for niche, color in NICHE_COLORS.items():
        sub = table[(table["method"].eq("spatialself")) & (table["receiver_niche"].eq(niche))]
        if not sub.empty:
            ax.scatter(sub["receiver_x"], -sub["receiver_y"], s=6, color=color, alpha=0.3, linewidths=0, label=niche)
    for method in ["spatialself", "commot"]:
        for pair in ["CXCL13-CXCR5", "CCL19-CCR7"]:
            sub = table[(table["method"].eq(method)) & (table["lr_pair"].eq(pair))].copy()
            if sub.empty:
                continue
            step = max(len(sub) // 70, 1)
            q = sub.iloc[::step]
            scale = 0.45 if method == "spatialself" else 0.35
            color = METHOD_COLORS[method]
            ax.quiver(
                q["receiver_x"],
                -q["receiver_y"],
                q["vector_dx"] * scale,
                -q["vector_dy"] * scale,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0022 if pair == "CXCL13-CXCR5" else 0.0018,
                color=color,
                alpha=pair_style[pair]["alpha"],
            )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.text(0.02, 0.98, "green/blue arrows: SpatialSelf/COMMOT", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color="#555555")
    ax.text(0.02, 0.91, "circle-like field: CXCL13-CXCR5; triangle field: CCL19-CCR7", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color="#555555")


def plot_expected_fraction(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4"]
    x = np.arange(len(pairs))
    width = 0.15
    for i, method in enumerate(METHOD_ORDER):
        vals = table[table["method"].eq(method)].set_index("lr_pair").reindex(pairs)["expected_flow_score_fraction"]
        ax.bar(x + (i - 2) * width, vals.fillna(0), width, color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABEL[method])
    ax.set_title("Expected niche-flow support", loc="left", pad=4)
    ax.set_ylabel("score fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=5.8)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(loc="upper left", fontsize=5.2, ncol=1, handlelength=1.0)


def plot_top_truth(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4"]
    methods = ["spatialself", "commot", "cellchat"]
    x = np.arange(len(pairs))
    width = 0.21
    for i, method in enumerate(methods):
        vals = table[table["method"].eq(method)].set_index("lr_pair").reindex(pairs)["top_ground_truth_fraction"]
        ax.bar(x + (i - 1) * width, vals.fillna(0), width, color=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABEL[method])
    ax.set_title("Top-edge rule recovery", loc="left", pad=4)
    ax.set_ylabel("ground-truth fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=5.8)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_niche_heatmap(ax: plt.Axes, table: pd.DataFrame, pair: str) -> None:
    niches = ["B follicle", "Germinal center", "T cell zone", "Mixed"]
    matrix = table.pivot_table(index="sender_niche", columns="receiver_niche", values="total_score", aggfunc="sum").reindex(index=niches, columns=niches).fillna(0)
    arr = np.log1p(matrix.to_numpy(dtype=float))
    im = ax.imshow(arr, cmap="YlGn", aspect="auto")
    ax.set_title(f"SpatialSelf {pair}", loc="left", pad=4)
    ax.set_xticks(np.arange(len(niches)))
    ax.set_xticklabels([short_niche(x) for x in niches], rotation=30, ha="right", fontsize=5.8)
    ax.set_yticks(np.arange(len(niches)))
    ax.set_yticklabels([short_niche(x) for x in niches], fontsize=5.8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.0f}", ha="center", va="center", fontsize=5.1, color="#202020")
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label("log1p flow", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=2)


def short_niche(value: str) -> str:
    return value.replace("Germinal center", "GC").replace("B follicle", "B foll.").replace("T cell zone", "T zone")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
