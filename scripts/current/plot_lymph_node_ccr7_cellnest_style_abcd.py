#!/usr/bin/env python
"""CellNEST-Fig2-style lymph-node CCR7 reproduction panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/datasets/human_lymph_node_visium_phase6_expanded_replicate_00"
RUN_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded"


def project_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


OUT_DIR = project_path_from_env("LYMPH_CCR7_OUT_DIR", PROJECT_ROOT / "outputs/current_lymph_node_ccr7_cellnest_style")
FIG_DIR = OUT_DIR / "figures"
SRC_DIR = OUT_DIR / "source_data"

CELLS = DATA_DIR / "cells.csv"
CANDIDATES = DATA_DIR / "candidate_interactions.csv"
PREDICTIONS = {
    "SpatialSelf": project_path_from_env(
        "LYMPH_CCR7_SPATIALSELF_PREDICTION",
        RUN_DIR / "spatialself_wavelet_edge_prior/human_lymph_node_visium_phase6_expanded_replicate_00__spatialself_wavelet_predictions.csv",
    ),
    "CellNEST": RUN_DIR / "cellnest/human_lymph_node_visium_phase6_expanded_replicate_00__cellnest_predictions.csv",
    "COMMOT": RUN_DIR / "commot/human_lymph_node_visium_phase6_expanded_replicate_00__commot_predictions.csv",
}

PAIR = ("CCL19", "CCR7")
PAIR_LABEL = "CCL19-CCR7"

COL = {
    "ink": "#25211F",
    "muted": "#6B625B",
    "grid": "#E8E2D8",
    "background": "#D8D2C8",
    "tzone": "#C94F48",
    "bzone": "#D6B98C",
    "gc": "#7BA6A3",
    "mixed": "#B8B0A7",
    "signal_light": "#F3D9C7",
    "signal": "#C94F48",
    "signal_dark": "#82352F",
    "commot": "#2F78A8",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "legend.frameon": False,
        }
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 1.0
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    cells = pd.read_csv(CELLS)
    cells["plot_y"] = -cells["y"].to_numpy(float)
    cand_cols = [
        "candidate_id",
        "sender_id",
        "receiver_id",
        "lr_id",
        "ligand",
        "receptor",
        "sender_niche",
        "receiver_niche",
        "distance",
        "ground_truth",
        "ligand_expr",
        "receptor_expr",
    ]
    candidates = pd.read_csv(CANDIDATES, usecols=cand_cols)
    pair_edges = candidates[(candidates["ligand"].eq(PAIR[0])) & (candidates["receptor"].eq(PAIR[1]))].copy()
    if pair_edges.empty:
        raise RuntimeError(f"No candidate edges found for {PAIR_LABEL}")

    scored: dict[str, pd.DataFrame] = {}
    receiver_maps = cells[["cell_id", "x", "y", "plot_y", "region", "niche", "cell_type", "dominant_cell_type"]].copy()
    for method, path in PREDICTIONS.items():
        pred = pd.read_csv(path, usecols=["candidate_id", "score"])
        merged = pair_edges.merge(pred, on="candidate_id", how="left")
        merged["score"] = merged["score"].fillna(0.0).astype(float)
        scored[method] = merged
        incoming = (
            merged.groupby("receiver_id", as_index=False)
            .agg(score=("score", "sum"), n_edges=("score", "size"), n_positive_edges=("score", lambda x: int(np.sum(np.asarray(x) > 0))))
            .rename(columns={"receiver_id": "cell_id", "score": f"{method}_incoming", "n_edges": f"{method}_n_edges", "n_positive_edges": f"{method}_n_positive_edges"})
        )
        receiver_maps = receiver_maps.merge(incoming, on="cell_id", how="left")
        receiver_maps[f"{method}_incoming"] = receiver_maps[f"{method}_incoming"].fillna(0.0)
        receiver_maps[f"{method}_n_edges"] = receiver_maps[f"{method}_n_edges"].fillna(0).astype(int)
        receiver_maps[f"{method}_n_positive_edges"] = receiver_maps[f"{method}_n_positive_edges"].fillna(0).astype(int)

    return cells, pair_edges, scored, receiver_maps


def format_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.035, 1.015, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=COL["ink"])


def plot_region_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    region_color = {
        "B follicle": COL["bzone"],
        "Germinal center": COL["gc"],
        "Mixed": COL["mixed"],
        "T cell zone": COL["tzone"],
    }
    order = ["B follicle", "Germinal center", "Mixed", "T cell zone"]
    for region in order:
        sub = cells[cells["region"].eq(region)]
        if sub.empty:
            continue
        size = 10.8 if region == "T cell zone" else 8.4
        alpha = 0.94 if region == "T cell zone" else 0.72
        ax.scatter(sub["x"], sub["plot_y"], s=size, c=region_color[region], lw=0, alpha=alpha, rasterized=True)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=region_color[r], markeredgewidth=0, markersize=4.2, label=r)
        for r in order
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.00, -0.03), ncol=2, fontsize=6.3, handletextpad=0.25, columnspacing=0.8)
    ax.set_title("T-cell-zone annotation", fontsize=8.6, pad=3.0)
    format_spatial_axis(ax)


def plot_receiver_hotspot(ax: plt.Axes, receiver: pd.DataFrame, method: str) -> None:
    score = receiver[f"{method}_incoming"].to_numpy(float)
    positive = score > 0
    if positive.any():
        cutoff = float(np.quantile(score[positive], 0.82))
        hot = positive & (score >= cutoff)
        hi_vals = score[hot]
        lo, hi = np.quantile(hi_vals, [0.05, 0.98]) if len(hi_vals) > 1 else (float(hi_vals.min()), float(hi_vals.max() + 1e-9))
    else:
        hot = np.zeros(len(receiver), dtype=bool)
        lo, hi = 0.0, 1.0
    tzone = receiver["region"].eq("T cell zone").to_numpy()
    ax.scatter(receiver["x"], receiver["plot_y"], s=6.2, c=COL["background"], lw=0, alpha=0.42, rasterized=True)
    ax.scatter(receiver.loc[tzone, "x"], receiver.loc[tzone, "plot_y"], s=8.6, c=COL["tzone"], lw=0, alpha=0.16, rasterized=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{method}_signal", [COL["signal_light"], COL["signal"]])
    norm = mpl.colors.Normalize(vmin=lo, vmax=max(hi, lo + 1e-9))
    ax.scatter(
        receiver.loc[hot, "x"],
        receiver.loc[hot, "plot_y"],
        s=14.0 if method == "CellNEST" else 11.0,
        c=score[hot],
        cmap=cmap,
        norm=norm,
        lw=0.18,
        edgecolors="white",
        alpha=0.94,
        rasterized=True,
    )
    ax.set_title(method, fontsize=8.6, pad=3.0)
    format_spatial_axis(ax)


def receiver_vectors(edges: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    coord = cells.set_index("cell_id")[["x", "plot_y", "region"]]
    sender = coord.rename(columns={"x": "sender_x", "plot_y": "sender_y", "region": "sender_region"})
    receiver = coord.rename(columns={"x": "receiver_x", "plot_y": "receiver_y", "region": "receiver_region"})
    table = edges.merge(sender, left_on="sender_id", right_index=True, how="left").merge(receiver, left_on="receiver_id", right_index=True, how="left")
    table = table.dropna(subset=["sender_x", "sender_y", "receiver_x", "receiver_y"]).copy()
    table["dx"] = table["receiver_x"] - table["sender_x"]
    table["dy"] = table["receiver_y"] - table["sender_y"]
    table["w_dx"] = table["score"] * table["dx"]
    table["w_dy"] = table["score"] * table["dy"]
    grouped = (
        table.groupby("receiver_id", as_index=False)
        .agg(
            x=("receiver_x", "first"),
            plot_y=("receiver_y", "first"),
            region=("receiver_region", "first"),
            score=("score", "sum"),
            dx=("w_dx", "sum"),
            dy=("w_dy", "sum"),
        )
    )
    grouped["dx"] = grouped["dx"] / np.maximum(grouped["score"], 1e-12)
    grouped["dy"] = grouped["dy"] / np.maximum(grouped["score"], 1e-12)
    return grouped


def interpolate_arrow_field(vectors: pd.DataFrame, grid_nx: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = vectors[["x", "plot_y"]].to_numpy(float)
    vec = vectors[["dx", "dy"]].to_numpy(float)
    score = vectors["score"].to_numpy(float)
    if len(vectors) < 5:
        return xy, vec, score
    tree = cKDTree(xy)
    spacing = np.median(tree.query(xy, k=min(2, len(xy)))[0][:, -1])
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    pad = spacing * 1.2
    xs = np.linspace(xmin - pad, xmax + pad, grid_nx)
    step = xs[1] - xs[0] if len(xs) > 1 else 0.08
    ys = np.linspace(ymin - pad, ymax + pad, max(7, int(round((ymax - ymin + 2 * pad) / max(step, 1e-6)))))
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    k = min(45, len(xy))
    dist, idx = tree.query(grid, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    bandwidth = max(spacing * 2.2, 1e-6)
    weights = np.exp(-0.5 * (dist / bandwidth) ** 2) * (0.10 + minmax(score[idx]))
    weight_sum = weights.sum(axis=1)
    v_grid = (vec[idx] * weights[:, :, None]).sum(axis=1) / np.maximum(weight_sum[:, None], 1e-12)
    s_grid = (score[idx] * weights).sum(axis=1) / np.maximum(weight_sum, 1e-12)
    tissue = weight_sum > np.quantile(weight_sum, 0.45)
    mag = np.linalg.norm(v_grid, axis=1)
    keep = tissue & (mag > np.quantile(mag[tissue], 0.35))
    return grid[keep], v_grid[keep], s_grid[keep]


def plot_commot_arrows(ax: plt.Axes, cells: pd.DataFrame, commot_edges: pd.DataFrame) -> pd.DataFrame:
    vectors = receiver_vectors(commot_edges, cells)
    score = vectors["score"].to_numpy(float)
    hot = score >= np.quantile(score[score > 0], 0.80) if np.any(score > 0) else np.zeros(len(score), dtype=bool)
    tzone = cells["region"].eq("T cell zone")
    ax.scatter(cells["x"], cells["plot_y"], s=6.0, c=COL["background"], lw=0, alpha=0.40, rasterized=True)
    ax.scatter(cells.loc[tzone, "x"], cells.loc[tzone, "plot_y"], s=8.5, c=COL["tzone"], lw=0, alpha=0.16, rasterized=True)
    ax.scatter(vectors.loc[hot, "x"], vectors.loc[hot, "plot_y"], s=9.0, c=COL["signal_light"], lw=0, alpha=0.65, rasterized=True)
    grid, v_grid, s_grid = interpolate_arrow_field(vectors[vectors["score"].gt(0)].copy())
    mag = np.linalg.norm(v_grid, axis=1)
    keep = mag > 0
    grid = grid[keep]
    v_grid = v_grid[keep] / np.maximum(mag[keep, None], 1e-12)
    s_grid = s_grid[keep]
    if len(grid):
        tree = cKDTree(cells[["x", "plot_y"]].to_numpy(float))
        spacing = np.median(tree.query(cells[["x", "plot_y"]].to_numpy(float), k=2)[0][:, 1])
        length = spacing * 2.4
        alpha = 0.35 + 0.55 * minmax(s_grid)
        for (x, y), (dx, dy), a in zip(grid, v_grid, alpha, strict=False):
            ax.arrow(
                x,
                y,
                dx * length,
                dy * length,
                width=0.0017,
                head_width=0.018,
                head_length=0.020,
                length_includes_head=True,
                color=COL["commot"],
                alpha=float(a),
                linewidth=0,
                zorder=5,
            )
    ax.set_title("COMMOT", fontsize=8.6, pad=3.0)
    format_spatial_axis(ax)
    return vectors


def write_source(cells: pd.DataFrame, pair_edges: pd.DataFrame, scored: dict[str, pd.DataFrame], receiver: pd.DataFrame, commot_vectors: pd.DataFrame) -> dict:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    receiver.to_csv(SRC_DIR / "panel_abcd_receiver_scores.csv", index=False)
    pair_edges.to_csv(SRC_DIR / "panel_pair_candidate_edges.csv", index=False)
    commot_vectors.to_csv(SRC_DIR / "panel_d_commot_receiver_vectors.csv", index=False)
    for method, table in scored.items():
        table[["candidate_id", "sender_id", "receiver_id", "lr_id", "ligand", "receptor", "sender_niche", "receiver_niche", "distance", "score"]].to_csv(
            SRC_DIR / f"{method.lower()}_{PAIR[0].lower()}_{PAIR[1].lower()}_edge_scores.csv",
            index=False,
        )

    summaries = []
    for method in scored:
        col = f"{method}_incoming"
        vals = receiver[col].to_numpy(float)
        t_vals = receiver.loc[receiver["region"].eq("T cell zone"), col].to_numpy(float)
        o_vals = receiver.loc[~receiver["region"].eq("T cell zone"), col].to_numpy(float)
        summaries.append(
            {
                "method": method,
                "pair": PAIR_LABEL,
                "n_positive_edges": int((scored[method]["score"] > 0).sum()),
                "n_positive_receivers": int((vals > 0).sum()),
                "top_receiver_region": str(receiver.loc[vals.argmax(), "region"]) if len(vals) else "",
                "t_zone_mean_incoming": float(np.mean(t_vals)) if len(t_vals) else np.nan,
                "other_mean_incoming": float(np.mean(o_vals)) if len(o_vals) else np.nan,
                "t_zone_enrichment": float(np.mean(t_vals) / max(np.mean(o_vals), 1e-12)) if len(t_vals) and len(o_vals) else np.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(SRC_DIR / "method_tzone_summary.csv", index=False)
    return {"method_tzone_summary": summaries}


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cells, pair_edges, scored, receiver = load_data()

    fig = plt.figure(figsize=(7.05, 6.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.985,
        top=0.955,
        bottom=0.065,
        wspace=0.045,
        hspace=0.100,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    plot_region_map(axes[0], cells)
    plot_receiver_hotspot(axes[1], receiver, "SpatialSelf")
    plot_receiver_hotspot(axes[2], receiver, "CellNEST")
    commot_vectors = plot_commot_arrows(axes[3], cells, scored["COMMOT"])
    for ax, lab in zip(axes, ["a", "b", "c", "d"], strict=False):
        panel_label(ax, lab)

    manifest = write_source(cells, pair_edges, scored, receiver, commot_vectors)
    manifest.update(
        {
            "figure_claim": "Known lymph-node CCL19-CCR7 activity is concentrated around the T-cell-zone context across SpatialSelf and baseline reproductions.",
            "pair": PAIR_LABEL,
            "dataset": str(DATA_DIR.relative_to(PROJECT_ROOT)),
            "predictions": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in PREDICTIONS.items()},
            "claim_boundary": "This is a positive-control reproduction and visualization panel, not a SpatialSelf-specific discovery claim.",
        }
    )
    (OUT_DIR / "lymph_node_ccr7_cellnest_style_abcd_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base = FIG_DIR / "lymph_node_ccr7_cellnest_style_abcd"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figure": str(base), **manifest}, indent=2))


if __name__ == "__main__":
    main()
