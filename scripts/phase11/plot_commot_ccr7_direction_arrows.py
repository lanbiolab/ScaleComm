#!/usr/bin/env python
"""Plot COMMOT CCR7-axis communication direction arrows for human lymph node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import commot as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_direction_fields(adata, database_name: str, pairs: list[tuple[str, str]], k: int) -> None:
    for pair in pairs:
        key_sender = f"commot_sender_vf-{database_name}-{pair[0]}-{pair[1]}"
        key_receiver = f"commot_receiver_vf-{database_name}-{pair[0]}-{pair[1]}"
        if key_sender not in adata.obsm or key_receiver not in adata.obsm:
            ct.tl.communication_direction(adata, database_name=database_name, lr_pair=pair, k=k)


def estimate_spot_spacing(xy: np.ndarray) -> float:
    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=2)
    return float(np.nanmedian(distances[:, 1]))


def interpolate_vectors(
    xy: np.ndarray,
    vectors: np.ndarray,
    scores: np.ndarray,
    grid_nx: int = 19,
    knn: int = 80,
    bandwidth_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = estimate_spot_spacing(xy)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    pad = spacing * 0.7
    xs = np.linspace(xmin - pad, xmax + pad, grid_nx)
    grid_step = xs[1] - xs[0]
    grid_ny = max(8, int(round((ymax - ymin + 2 * pad) / grid_step)))
    ys = np.linspace(ymin - pad, ymax + pad, grid_ny)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    tree = cKDTree(xy)
    distances, idx = tree.query(grid, k=knn)
    bandwidth = spacing * bandwidth_scale
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weight_sum = weights.sum(axis=1)
    tissue_threshold = np.percentile(weight_sum, 58)
    score_grid = (scores[idx] * weights).sum(axis=1) / np.maximum(weight_sum, 1e-12)
    v_grid = (vectors[idx] * weights[:, :, None]).sum(axis=1) / np.maximum(weight_sum[:, None], 1e-12)
    keep = weight_sum > tissue_threshold
    keep &= np.linalg.norm(v_grid, axis=1) > np.percentile(np.linalg.norm(v_grid, axis=1), 35)
    return grid[keep], v_grid[keep], score_grid[keep]


def plot_panel(ax, adata, database_name: str, pair: tuple[str, str], summary: str, title: str) -> None:
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    pair_key = f"{pair[0]}-{pair[1]}"
    if summary == "sender":
        vectors = np.asarray(adata.obsm[f"commot_sender_vf-{database_name}-{pair_key}"], dtype=float)
        scores = np.asarray(adata.obsm[f"commot-{database_name}-sum-sender"][f"s-{pair_key}"], dtype=float)
        arrow_color = "#1f2937"
    else:
        vectors = np.asarray(adata.obsm[f"commot_receiver_vf-{database_name}-{pair_key}"], dtype=float)
        scores = np.asarray(adata.obsm[f"commot-{database_name}-sum-receiver"][f"r-{pair_key}"], dtype=float)
        arrow_color = "#8b2f2b"

    vmin, vmax = np.quantile(scores, [0.02, 0.98])
    ax.scatter(xy[:, 0], xy[:, 1], s=3.0, c=scores, cmap="viridis", vmin=vmin, vmax=vmax, linewidth=0, alpha=0.82)

    grid, v_grid, _ = interpolate_vectors(xy, vectors[:, :2], scores)
    norm = np.linalg.norm(v_grid, axis=1)
    valid = norm > 0
    grid = grid[valid]
    v_grid = v_grid[valid] / norm[valid, None]
    arrow_length = estimate_spot_spacing(xy) * 1.05
    u = v_grid[:, 0] * arrow_length
    v = v_grid[:, 1] * arrow_length
    ax.quiver(
        grid[:, 0],
        grid[:, 1],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=arrow_color,
        width=0.005,
        headwidth=4.2,
        headlength=5.4,
        headaxislength=4.8,
        alpha=0.92,
        pivot="middle",
    )
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()


def save_all(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adata",
        default="outputs/phase11/official_method_reproduction/commot_human_lymph_node/commot_official_human_lymph_node.h5ad",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/phase11/official_method_reproduction/commot_human_lymph_node/commot_official_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase11/official_method_reproduction/commot_human_lymph_node/figures_ccr7_direction",
    )
    parser.add_argument("--database-name", default="CellChat")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--write-adata-with-vectors", action="store_true")
    args = parser.parse_args()

    set_pub_style()
    adata_path = PROJECT_ROOT / args.adata
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((PROJECT_ROOT / args.summary_json).read_text(encoding="utf-8"))

    adata = ad.read_h5ad(adata_path)
    pairs = [("CCL21", "CCR7"), ("CCL19", "CCR7")]
    ensure_direction_fields(adata, args.database_name, pairs, args.k)

    rank_lookup = {
        ("CCL21", "CCR7"): summary.get("ccl21_ccr7_total_score_rank"),
        ("CCL19", "CCR7"): summary.get("ccl19_ccr7_total_score_rank"),
    }
    frac_lookup = {
        ("CCL21", "CCR7"): summary.get("ccl21_ccr7_tzone_receiver_score_fraction"),
        ("CCL19", "CCR7"): summary.get("ccl19_ccr7_tzone_receiver_score_fraction"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0), constrained_layout=True)
    panel_specs = [
        (("CCL21", "CCR7"), "sender", "CCL21-CCR7 sender field"),
        (("CCL21", "CCR7"), "receiver", "CCL21-CCR7 receiver field"),
        (("CCL19", "CCR7"), "sender", "CCL19-CCR7 sender field"),
        (("CCL19", "CCR7"), "receiver", "CCL19-CCR7 receiver field"),
    ]
    for ax, (pair, role, title) in zip(axes.ravel(), panel_specs):
        extra = f"rank {rank_lookup[pair]}, T-zone receiver {frac_lookup[pair]:.3f}" if role == "receiver" else f"rank {rank_lookup[pair]}"
        plot_panel(ax, adata, args.database_name, pair, role, f"{title}\n{extra}")

    fig.suptitle("COMMOT official CCR7-axis communication direction in human lymph node", fontsize=9, y=1.02)
    save_all(fig, output_dir / "commot_ccr7_direction_4panel")
    plt.close(fig)

    for pair in pairs:
        for role in ["sender", "receiver"]:
            fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
            title = f"{pair[0]}-{pair[1]} {role} direction"
            plot_panel(ax, adata, args.database_name, pair, role, title)
            save_all(fig, output_dir / f"commot_{pair[0].lower()}_{pair[1].lower()}_{role}_direction")
            plt.close(fig)

    if args.write_adata_with_vectors:
        adata.write_h5ad(output_dir / "commot_official_human_lymph_node_with_ccr7_directions.h5ad", compression="gzip")

    manifest = {
        "source_adata": str(adata_path),
        "output_dir": str(output_dir),
        "pairs": [f"{a}-{b}" for a, b in pairs],
        "database_name": args.database_name,
        "direction_k": args.k,
        "plot_method": "COMMOT ct.tl.communication_direction + matplotlib quiver grid rendering",
        "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    }
    (output_dir / "commot_ccr7_direction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
