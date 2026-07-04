#!/usr/bin/env python
"""Plot Phase 11 COMMOT-style story panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

METHOD_COLORS = {
    "spatialself": "#2F6B4F",
    "commot": "#456C9B",
    "cellchat": "#7D668E",
    "cellnest": "#B07A4B",
}
TEXT = "#202020"
GRID = "#D7D7D7"


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    direction = pd.read_csv(RESULT_DIR / "directionality_lr_summary.csv")
    modules = pd.read_csv(RESULT_DIR / "communication_modules_summary.csv")
    controls = pd.read_csv(PHASE10_DIR / "results/spp1_cd44_control_specificity_summary.csv")
    gene_rank = pd.read_csv(PHASE10_DIR / "results/gene_association_top300_by_method_lr.csv")
    cells = pd.read_csv(
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        / "standardized_inputs/visium_hd_crc_p2crc_subset/cells.csv"
    )
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    impact = load_optional_impact()

    source = build_source(direction, modules, controls, gene_rank, cells, incoming, impact)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"commot_story_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.05, 1.15],
        height_ratios=[1.0, 0.9, 0.8],
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.12,
        wspace=0.35,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_case_map(ax_a, source["case_map"])
    plot_direction(ax_b, source["direction"])
    plot_modules(ax_c, source["modules"])
    plot_controls(ax_d, source["controls"])
    plot_impact(ax_e, source["impact"])

    for label, ax in zip(["a", "b", "c", "d", "e"], [ax_a, ax_b, ax_c, ax_d, ax_e]):
        add_panel_label(ax, label)

    fig.suptitle(
        "COMMOT-style spatial communication story for SPP1-CD44 in Visium HD CRC",
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
        0.935,
        "SpatialSelf links SPP1-CD44 incoming activity to receiver programs, niche context, negative-control suppression and target-gene impact.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "Analyses are observational. Module analysis is based on the current small LR panel and should be expanded before strong module-level claims.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_commot_style_story_panels")
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
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_optional_impact() -> pd.DataFrame:
    path = RESULT_DIR / "downstream_lr_level_impact_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["method", "lr_pair", "mean_delta_cv_r2", "mean_delta_specificity_vs_best_negative", "n_impact_fdr_0_1"])


def build_source(
    direction: pd.DataFrame,
    modules: pd.DataFrame,
    controls: pd.DataFrame,
    gene_rank: pd.DataFrame,
    cells: pd.DataFrame,
    incoming: pd.DataFrame,
    impact: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    spatial = incoming[incoming["method"].eq("spatialself")][["cell_id", "SPP1-CD44"]].merge(
        cells[["cell_id", "x", "y", "niche", "cell_type"]], on="cell_id", how="left"
    )
    spatial = spatial.rename(columns={"SPP1-CD44": "incoming_score"})
    top_targets = gene_rank[(gene_rank["method"].eq("spatialself")) & (gene_rank["lr_pair"].eq("SPP1-CD44"))].copy()
    top_targets = top_targets.sort_values("spearman_r", ascending=False).head(30)
    spatial["target_signature_proxy"] = spatial["incoming_score"]

    direction_source = direction[direction["lr_pair"].isin(["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR"])].copy()
    modules_source = modules[modules["method"].eq("spatialself")].copy()
    controls_source = controls.copy()
    impact_source = impact[impact["lr_pair"].isin(["SPP1-CD44", "ACTB-KRT8", "COL1A1-EPCAM"])].copy()
    return {
        "case_map": spatial,
        "top_target_genes": top_targets,
        "direction": direction_source,
        "modules": modules_source,
        "controls": controls_source,
        "impact": impact_source,
    }


def plot_case_map(ax: plt.Axes, table: pd.DataFrame) -> None:
    score = table["incoming_score"].to_numpy(dtype=float)
    vmax = np.quantile(score, 0.98)
    sc = ax.scatter(table["x"], -table["y"], c=score, s=4.4, cmap="YlGn", vmin=0, vmax=max(vmax, 1e-8), linewidths=0, rasterized=True)
    ax.set_title("SpatialSelf SPP1-CD44 incoming", loc="left", pad=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.02)
    cb.set_label("incoming", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.3, length=2)


def plot_direction(ax: plt.Axes, table: pd.DataFrame) -> None:
    pairs = ["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR"]
    methods = ["spatialself", "commot"]
    x = np.arange(len(pairs))
    width = 0.32
    for i, method in enumerate(methods):
        vals = table[table["method"].eq(method)].set_index("lr_pair").reindex(pairs)["expected_flow_score_fraction"].fillna(0)
        ax.bar(x + (i - 0.5) * width, vals, width, color=METHOD_COLORS[method], edgecolor="none", label=method.capitalize() if method != "spatialself" else "SpatialSelf")
    ax.set_title("Expected niche-flow support", loc="left", pad=4)
    ax.set_ylabel("score fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=28, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.legend(fontsize=5.6)


def plot_modules(ax: plt.Axes, table: pd.DataFrame) -> None:
    table = table.sort_values("module")
    y = np.arange(len(table))
    vals = table["top_niche_mean_activity"].astype(float)
    colors = ["#4D7C62" if "high-expression negative" not in x else "#B9B9B9" for x in table["program_priors"]]
    ax.barh(y, vals, color=colors, edgecolor="none")
    labels = [f"{m}: {pairs.replace(';', ', ')}" for m, pairs in zip(table["module"], table["lr_pairs"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_title("SpatialSelf communication modules", loc="left", pad=4)
    ax.set_xlabel("top niche mean activity")
    ax.grid(axis="x", color=GRID, linewidth=0.45)


def plot_controls(ax: plt.Axes, table: pd.DataFrame) -> None:
    order = ["observed", "shuffled_lr_identity", "shuffled_receiver", "wrong_niche_masked", "wrong_receiver_masked", "wrong_type_masked"]
    labels = ["observed", "shuffled LR", "shuffled receiver", "wrong niche", "wrong receiver", "wrong type"]
    plot = table.set_index("control_type").reindex(order).reset_index()
    x = np.arange(len(plot))
    colors = ["#2F6B4F"] + ["#B9B9B9"] * (len(plot) - 1)
    ax.bar(x, plot["target_overlap_top_n"], color=colors, edgecolor="none")
    ax.set_title("Negative-control suppression", loc="left", pad=4)
    ax.set_ylabel("target overlap top 200")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=5.6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_impact(ax: plt.Axes, table: pd.DataFrame) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "Impact benchmark running or unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return
    methods = ["spatialself", "commot", "cellchat", "cellnest"]
    plot = table[table["lr_pair"].eq("SPP1-CD44")].set_index("method").reindex(methods)
    x = np.arange(len(methods))
    vals = plot["mean_delta_cv_r2"].fillna(0)
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.bar(x, vals, color=[METHOD_COLORS[m] for m in methods], edgecolor="none")
    ax.set_title("Gene-level downstream impact", loc="left", pad=4)
    ax.set_ylabel("mean target-gene delta CV R2")
    ax.set_xticks(x)
    ax.set_xticklabels(["SpatialSelf", "COMMOT", "CellChat", "CellNEST"], rotation=22, ha="right", fontsize=6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
