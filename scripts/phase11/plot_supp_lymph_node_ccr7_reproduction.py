#!/usr/bin/env python
"""Plot supplementary lymph-node CCR7-axis reproduction figure."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import commot as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/official_method_reproduction/lymph_node_ccr7_supplement_figure"

CELLS = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset/cells.csv"
CANDIDATES = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset/candidate_interactions.csv"
SPATIALSELF_PRED = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/baselines/spatialself_range_context_mechniche/dataset__spatialself_range_context_predictions.csv"
CELLNEST_ANNOTATION = Path("/home/xd/y/BenchmarkBaselines/CellNEST/data/V1_Human_Lymph_Node_spatial_annotation.csv")
CELLNEST_TCELL_HIST = PROJECT_ROOT / "outputs/phase11/official_method_reproduction/cellnest_human_lymph_node/cellnest_official_tcell_histogram_top20percent.csv"
CELLNEST_ALLCCC = Path("/home/xd/y/BenchmarkBaselines/CellNEST/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_allCCC.csv")
COMMOT_H5AD = PROJECT_ROOT / "outputs/phase11/official_method_reproduction/commot_human_lymph_node/commot_official_human_lymph_node.h5ad"
COMMOT_SUMMARY = PROJECT_ROOT / "outputs/phase11/official_method_reproduction/commot_human_lymph_node/commot_official_summary.json"
EXPRESSION = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset/expression.csv"
TARGET_SIGNATURES = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset/target_signatures.csv"


COLORS = {
    "bg": "#D9DEE4",
    "bg_edge": "#CBD2DA",
    "tissue": "#EEF1F4",
    "dark_grey": "#4F5661",
    "light_text": "#6B7280",
    "grid": "#E7EAEE",
    "tcell": "#0072B2",
    "signal": "#C75D2C",
    "signal_dark": "#7F3B1D",
    "signal_light": "#E7B08B",
    "top_receiver": "#C75D2C",
    "cellnest": "#C75D2C",
    "neutral": "#C7CED6",
    "commot": "#7F3B1D",
    "downstream": "#C75D2C",
    "black": "#1F2328",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.handletextpad": 0.35,
            "legend.borderpad": 0.15,
            "figure.facecolor": "white",
        }
    )


def load_cells() -> pd.DataFrame:
    cells = pd.read_csv(CELLS)
    ann = pd.read_csv(CELLNEST_ANNOTATION)
    ann.columns = ["cell_id", "official_region"]
    region = ann.set_index("cell_id")["official_region"].astype(str).to_dict()
    cells["official_region"] = cells["cell_id"].map(region).fillna("unannotated")
    cells["is_official_tcell"] = cells["official_region"].eq("T-cell")
    cells["plot_y"] = -cells["y"]
    return cells


def load_spatialself_scores(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidates = pd.read_csv(
        CANDIDATES,
        usecols=["candidate_id", "receiver_id", "ligand", "receptor"],
    )
    predictions = pd.read_csv(SPATIALSELF_PRED, usecols=["candidate_id", "score"])
    table = candidates.merge(predictions, on="candidate_id", how="inner")
    table = table[table["ligand"].eq("CCL19") & table["receptor"].eq("CCR7")].copy()
    incoming = (
        table.groupby("receiver_id", as_index=False)
        .agg(spatialself_score=("score", "sum"), n_spatialself_edges=("score", "size"))
        .rename(columns={"receiver_id": "cell_id"})
    )
    out = cells.merge(incoming, on="cell_id", how="left")
    out["spatialself_score"] = out["spatialself_score"].fillna(0.0)
    out["n_spatialself_edges"] = out["n_spatialself_edges"].fillna(0).astype(int)
    t_vals = out.loc[out["is_official_tcell"], "spatialself_score"].to_numpy(float)
    o_vals = out.loc[~out["is_official_tcell"], "spatialself_score"].to_numpy(float)
    p_value = float(mannwhitneyu(t_vals, o_vals, alternative="greater").pvalue)
    stats = {
        "n_tcell_spots": int(out["is_official_tcell"].sum()),
        "n_other_spots": int((~out["is_official_tcell"]).sum()),
        "tcell_mean": float(np.mean(t_vals)),
        "other_mean": float(np.mean(o_vals)),
        "mean_ratio": float(np.mean(t_vals) / np.mean(o_vals)),
        "mannwhitneyu_greater_p": p_value,
    }
    return out, table, stats


def add_cellnest_receiver_scores(cells: pd.DataFrame) -> pd.DataFrame:
    allccc = pd.read_csv(CELLNEST_ALLCCC, usecols=["to_cell", "ligand", "receptor", "attention_score"])
    subset = allccc[allccc["ligand"].eq("CCL19") & allccc["receptor"].eq("CCR7")].copy()
    receiver = (
        subset.groupby("to_cell", as_index=False)
        .agg(cellnest_score=("attention_score", "sum"), n_cellnest_edges=("attention_score", "size"))
        .rename(columns={"to_cell": "cell_id"})
    )
    out = cells.merge(receiver, on="cell_id", how="left")
    out["cellnest_score"] = out["cellnest_score"].fillna(0.0)
    out["n_cellnest_edges"] = out["n_cellnest_edges"].fillna(0).astype(int)
    return out


def add_downstream_signature_scores(cells: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expression = pd.read_csv(EXPRESSION)
    targets = pd.read_csv(TARGET_SIGNATURES)
    genes = (
        targets.loc[
            targets["lr_id"].eq("ln_full_lr_000_CCL19_CCR7") & targets["present_in_expression"],
            "target_gene",
        ]
        .drop_duplicates()
        .tolist()
    )
    genes = [gene for gene in genes if gene in expression.columns]
    values = np.log1p(expression[genes].to_numpy(float))
    z = (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, ddof=1, keepdims=True), 1e-12)
    expression["downstream_signature"] = z.mean(axis=1)
    out = cells.merge(expression[["cell_id", "downstream_signature"]], on="cell_id", how="left")
    out["downstream_signature"] = out["downstream_signature"].fillna(0.0)
    return out, genes


def add_commot_receiver_scores(cells: pd.DataFrame, adata) -> pd.DataFrame:
    scores = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "commot_score": np.asarray(adata.obsm["commot-CellChat-sum-receiver"]["r-CCL19-CCR7"], dtype=float),
        }
    )
    out = cells.merge(scores, on="cell_id", how="left")
    out["commot_score"] = out["commot_score"].fillna(0.0)
    return out


def enrichment_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, column in [
        ("SpatialSelf", "spatialself_score"),
        ("CellNEST", "cellnest_score"),
        ("COMMOT", "commot_score"),
    ]:
        t_vals = cells.loc[cells["is_official_tcell"], column].to_numpy(float)
        o_vals = cells.loc[~cells["is_official_tcell"], column].to_numpy(float)
        rows.append(
            {
                "method": method,
                "tcell_mean": float(np.mean(t_vals)),
                "other_mean": float(np.mean(o_vals)),
                "enrichment": float(np.mean(t_vals) / np.maximum(np.mean(o_vals), 1e-12)),
                "p_value": float(mannwhitneyu(t_vals, o_vals, alternative="greater").pvalue),
            }
        )
    return pd.DataFrame(rows)


def ensure_commot_direction_fields(adata, database_name: str = "CellChat") -> None:
    for pair in [("CCL19", "CCR7")]:
        pair_key = f"{pair[0]}-{pair[1]}"
        sender_key = f"commot_sender_vf-{database_name}-{pair_key}"
        receiver_key = f"commot_receiver_vf-{database_name}-{pair_key}"
        if sender_key not in adata.obsm or receiver_key not in adata.obsm:
            ct.tl.communication_direction(adata, database_name=database_name, lr_pair=pair, k=5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.075, 1.015, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")


def format_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panel_a(ax: plt.Axes, cells: pd.DataFrame) -> None:
    other = cells[~cells["is_official_tcell"]]
    tcell = cells[cells["is_official_tcell"]]
    ax.scatter(other["x"], other["plot_y"], s=5.2, c=COLORS["bg"], lw=0, alpha=0.80)
    ax.scatter(tcell["x"], tcell["plot_y"], s=10.5, c=COLORS["tcell"], lw=0, alpha=0.96)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["tcell"], markeredgewidth=0, markersize=4.2, label="T-cell"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["bg"], markeredgewidth=0, markersize=4.2, label="Other"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.03),
        fontsize=7,
        handlelength=0.8,
    )
    format_spatial_axis(ax)
    panel_label(ax, "a")


def method_high_mask(scores: pd.Series | np.ndarray, keep_quantile: float = 0.80) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    positive = arr > 0
    cutoff = np.quantile(arr[positive], keep_quantile) if positive.any() else np.inf
    return positive & (arr >= cutoff)


def add_panel_tag(ax: plt.Axes, label: str) -> None:
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", ha="left", fontsize=8.0, color=COLORS["dark_grey"])


def plot_score_map(
    ax: plt.Axes,
    cells: pd.DataFrame,
    score_col: str,
    label: str,
    color: str,
    panel: str,
    show_tcell: bool = True,
    keep_quantile: float = 0.80,
) -> None:
    scores = cells[score_col].to_numpy(float)
    high = method_high_mask(scores, keep_quantile)
    vmin, vmax = np.quantile(scores[high], [0.02, 0.98]) if high.any() else (0.0, 1.0)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{score_col}_cmap", ["#EAC5AE", color])
    tcell = cells[cells["is_official_tcell"]]
    ax.scatter(cells["x"], cells["plot_y"], s=4.8, c=COLORS["bg"], lw=0, alpha=0.66)
    if show_tcell:
        ax.scatter(tcell["x"], tcell["plot_y"], s=8.0, c=COLORS["tcell"], lw=0, alpha=0.22)
    ax.scatter(
        cells.loc[high, "x"],
        cells.loc[high, "plot_y"],
        s=8.5,
        c=scores[high],
        cmap=cmap,
        norm=norm,
        lw=0,
        alpha=0.92,
    )
    format_spatial_axis(ax)
    add_panel_tag(ax, label)
    panel_label(ax, panel)


def estimate_spot_spacing(xy: np.ndarray) -> float:
    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=2)
    return float(np.nanmedian(distances[:, 1]))


def interpolate_vectors(
    xy: np.ndarray,
    vectors: np.ndarray,
    scores: np.ndarray,
    grid_nx: int = 12,
    knn: int = 90,
    bandwidth_scale: float = 1.55,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = estimate_spot_spacing(xy)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    pad = spacing * 0.55
    xs = np.linspace(xmin - pad, xmax + pad, grid_nx)
    grid_step = xs[1] - xs[0]
    grid_ny = max(8, int(round((ymax - ymin + 2 * pad) / grid_step)))
    ys = np.linspace(ymin - pad, ymax + pad, grid_ny)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(xy)
    distances, idx = tree.query(grid, k=knn)
    weights = np.exp(-0.5 * (distances / (spacing * bandwidth_scale)) ** 2)
    weight_sum = weights.sum(axis=1)
    v_grid = (vectors[idx] * weights[:, :, None]).sum(axis=1) / np.maximum(weight_sum[:, None], 1e-12)
    tissue = weight_sum > np.percentile(weight_sum, 57)
    magnitude = np.linalg.norm(v_grid, axis=1)
    keep = tissue & (magnitude > np.percentile(magnitude, 38))
    return grid[keep], v_grid[keep]


def plot_commot_receiver(ax: plt.Axes, adata, pair: tuple[str, str], label: str) -> None:
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    pair_key = f"{pair[0]}-{pair[1]}"
    vectors = np.asarray(adata.obsm[f"commot_receiver_vf-CellChat-{pair_key}"], dtype=float)[:, :2]
    scores = np.asarray(adata.obsm["commot-CellChat-sum-receiver"][f"r-{pair_key}"], dtype=float)
    score_cut = np.quantile(scores, 0.90)
    ax.scatter(xy[:, 0], -xy[:, 1], s=4.4, color=COLORS["bg"], linewidth=0, alpha=0.66)
    high = scores >= score_cut
    ax.scatter(xy[high, 0], -xy[high, 1], s=6.6, color=COLORS["signal_light"], linewidth=0, alpha=0.68)
    grid, v_grid = interpolate_vectors(xy, vectors, scores)
    norm = np.linalg.norm(v_grid, axis=1)
    valid = norm > 0
    grid = grid[valid]
    v_grid = v_grid[valid] / norm[valid, None]
    arrow_length = estimate_spot_spacing(xy) * 2.35
    ax.quiver(
        grid[:, 0],
        -grid[:, 1],
        v_grid[:, 0] * arrow_length,
        -v_grid[:, 1] * arrow_length,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=COLORS["signal_dark"],
        width=0.0046,
        headwidth=3.6,
        headlength=4.6,
        headaxislength=4.0,
        alpha=0.95,
        pivot="middle",
    )
    format_spatial_axis(ax)
    add_panel_tag(ax, label)


def plot_panel_e(ax: plt.Axes, cells: pd.DataFrame) -> pd.DataFrame:
    method_cols = ["spatialself_score", "cellnest_score", "commot_score"]
    support = np.zeros(len(cells), dtype=int)
    for col in method_cols:
        support += method_high_mask(cells[col], 0.80).astype(int)
    out = cells[["cell_id", "x", "y", "plot_y", "is_official_tcell"]].copy()
    out["method_support_count"] = support
    consensus = out["method_support_count"].ge(2)
    ax.scatter(cells["x"], cells["plot_y"], s=4.8, c=COLORS["bg"], lw=0, alpha=0.66)
    tcell = cells[cells["is_official_tcell"]]
    ax.scatter(tcell["x"], tcell["plot_y"], s=8.0, c=COLORS["tcell"], lw=0, alpha=0.18)
    ax.scatter(
        out.loc[consensus, "x"],
        out.loc[consensus, "plot_y"],
        s=9.5,
        c=COLORS["signal_dark"],
        lw=0,
        alpha=0.92,
    )
    format_spatial_axis(ax)
    add_panel_tag(ax, "2+ method support")
    panel_label(ax, "e")
    return out


def plot_panel_f(ax: plt.Axes, cells: pd.DataFrame) -> None:
    scores = cells["downstream_signature"].to_numpy(float)
    cutoff = np.quantile(scores, 0.70)
    high = scores >= cutoff
    vmin, vmax = np.quantile(scores[high], [0.02, 0.98])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("downstream_cmap", ["#F0D3BE", COLORS["signal"]])
    ax.scatter(cells["x"], cells["plot_y"], s=4.8, c=COLORS["bg"], lw=0, alpha=0.62)
    ax.scatter(cells.loc[high, "x"], cells.loc[high, "plot_y"], s=8.2, c=scores[high], cmap=cmap, norm=norm, lw=0, alpha=0.92)
    format_spatial_axis(ax)
    add_panel_tag(ax, "Target response")
    panel_label(ax, "f")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = load_cells()
    adata = ad.read_h5ad(COMMOT_H5AD)
    ensure_commot_direction_fields(adata)
    cells_scored, spatialself_edges, stats = load_spatialself_scores(cells)
    cells_scored = add_cellnest_receiver_scores(cells_scored)
    cells_scored = add_commot_receiver_scores(cells_scored, adata)
    cells_scored, downstream_genes = add_downstream_signature_scores(cells_scored)
    enrich = enrichment_table(cells_scored)

    fig = plt.figure(figsize=(11.2, 7.35), constrained_layout=False)
    outer = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.26,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])
    ax_d = fig.add_subplot(outer[1, 0])
    ax_e = fig.add_subplot(outer[1, 1])
    ax_f = fig.add_subplot(outer[1, 2])

    plot_panel_a(ax_a, cells)
    plot_score_map(ax_b, cells_scored, "spatialself_score", "SpatialSelf", COLORS["signal"], "b")
    plot_score_map(ax_c, cells_scored, "cellnest_score", "CellNEST", COLORS["signal"], "c")
    plot_commot_receiver(ax_d, adata, ("CCL19", "CCR7"), "COMMOT")
    panel_label(ax_d, "d")
    consensus = plot_panel_e(ax_e, cells_scored)
    plot_panel_f(ax_f, cells_scored)
    fig.subplots_adjust(top=0.985, bottom=0.075, left=0.06, right=0.985)

    base = OUT_DIR / "supp_lymph_node_ccr7_official_reproduction"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    source_dir = OUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    cells_scored[
        [
            "cell_id",
            "x",
            "y",
            "official_region",
            "is_official_tcell",
            "spatialself_score",
            "n_spatialself_edges",
            "cellnest_score",
            "n_cellnest_edges",
            "commot_score",
            "downstream_signature",
        ]
    ].to_csv(
        source_dir / "panel_abcdef_spot_scores.csv",
        index=False,
    )
    enrich.to_csv(source_dir / "method_tcell_enrichment_summary.csv", index=False)
    consensus.to_csv(source_dir / "panel_e_method_consensus_support.csv", index=False)
    spatialself_edges.to_csv(source_dir / "panel_b_spatialself_ccl19_ccr7_candidate_scores.csv", index=False)
    commot_summary = json.loads(COMMOT_SUMMARY.read_text(encoding="utf-8"))
    manifest = {
        "figure": str(base),
        "panels": {
            "a": "10x human lymph-node Visium spots with official CellNEST T-cell annotation.",
            "b": "SpatialSelf CCL19-CCR7 receiver-level score map.",
            "c": "CellNEST official workflow CCL19-CCR7 receiver attention map.",
            "d": "COMMOT official workflow CCL19-CCR7 receiver communication direction field.",
            "e": "Spatial consensus of high CCL19-CCR7 receiver support across SpatialSelf, CellNEST and COMMOT.",
            "f": "CCL19-CCR7 downstream target-response signature map.",
        },
        "spatialself_stats": stats,
        "cellnest_score_source": str(CELLNEST_ALLCCC),
        "downstream_signature_genes": downstream_genes,
        "method_tcell_enrichment": enrich.to_dict(orient="records"),
        "commot_summary": {
            "ccl19_ccr7_total_score_rank": commot_summary.get("ccl19_ccr7_total_score_rank"),
            "ccl21_ccr7_total_score_rank": commot_summary.get("ccl21_ccr7_total_score_rank"),
            "ccl19_ccr7_tzone_receiver_score_fraction": commot_summary.get("ccl19_ccr7_tzone_receiver_score_fraction"),
            "ccl21_ccr7_tzone_receiver_score_fraction": commot_summary.get("ccl21_ccr7_tzone_receiver_score_fraction"),
        },
        "claim_boundary": (
            "This is a positive-control reproduction figure. It demonstrates consistency with known lymph-node CCR7-axis biology "
            "and official baseline workflows, not a SpatialSelf-specific discovery claim."
        ),
    }
    (OUT_DIR / "supp_lymph_node_ccr7_official_reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
