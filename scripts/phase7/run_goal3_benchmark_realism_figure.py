#!/usr/bin/env python
"""Build Phase 7 Goal 3 benchmark realism statistics and figure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_LABELS = {
    "lymph_node": "Lymph node",
    "tonsil": "Tonsil",
    "melanoma": "Melanoma",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        default="outputs/phase2_revised/goal4_real_background_v2/formal_datasets",
    )
    parser.add_argument("--max-moran-genes", type=int, default=350)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    goal_dir = PROJECT_ROOT / config["goals"]["goal3_benchmark_realism_figure"]["output_dir"]
    source_dir = goal_dir / "source_data"
    figure_dir = goal_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = PROJECT_ROOT / args.dataset_root
    replicate_dirs = sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    if not replicate_dirs:
        raise FileNotFoundError(f"No replicate directories found under {dataset_root}")

    metric_rows: list[dict[str, Any]] = []
    distribution_frames: dict[str, list[pd.DataFrame]] = {
        "cell_library_sparsity": [],
        "gene_mean_variance": [],
        "moran": [],
        "lr_expression": [],
        "pca": [],
        "spatial_density": [],
    }

    for rep_dir in replicate_dirs:
        result = _analyze_replicate(rep_dir, max_moran_genes=args.max_moran_genes)
        metric_rows.extend(result["metrics"])
        for key, frame in result["distributions"].items():
            distribution_frames[key].append(frame)

    metrics = pd.DataFrame(metric_rows)
    metrics_path = goal_dir / "phase7_benchmark_realism_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = _build_summary(metrics)
    summary_path = goal_dir / "phase7_benchmark_realism_summary.csv"
    summary.to_csv(summary_path, index=False)

    source_paths = {}
    for key, frames in distribution_frames.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = source_dir / f"phase7_realism_{key}.csv"
        frame.to_csv(path, index=False)
        source_paths[key] = _rel(path)

    _plot_realism_figure(metrics, distribution_frames, figure_dir)
    manifest = {
        "config": _rel(config_path),
        "dataset_root": _rel(dataset_root),
        "metrics": _rel(metrics_path),
        "summary": _rel(summary_path),
        "source_data": source_paths,
        "figure": {
            "pdf": _rel(figure_dir / "fig_phase7_benchmark_realism.pdf"),
            "svg": _rel(figure_dir / "fig_phase7_benchmark_realism.svg"),
            "tiff": _rel(figure_dir / "fig_phase7_benchmark_realism.tiff"),
        },
        "core_conclusion": (
            "Semi-synthetic benchmarks preserve real spatial layout and broad expression statistics "
            "while LR-injected genes show expected, localized distribution changes."
        ),
        "limitations": [
            "Library-size and sparsity comparisons use the selected benchmark gene panel, while cells.csv preserves the original full-library metadata.",
            "Realism statistics assess distributional similarity and do not provide native real-tissue CCC ground truth.",
            "Injected LR genes are expected to deviate from the real background because they encode known benchmark truth.",
        ],
    }
    (goal_dir / "phase7_goal3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote Goal 3 outputs to {goal_dir}")


def _analyze_replicate(rep_dir: Path, *, max_moran_genes: int) -> dict[str, Any]:
    cells = pd.read_csv(rep_dir / "cells.csv")
    semi_expression = pd.read_csv(rep_dir / "expression.csv")
    lr_pairs = pd.read_csv(rep_dir / "lr_pairs.csv")
    metadata = json.loads((rep_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset_key = _dataset_key(rep_dir.name)
    dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)

    real_expression = _load_real_expression_for_replicate(metadata, semi_expression)
    semi_matrix = _expression_matrix(semi_expression)
    real_matrix = _expression_matrix(real_expression)
    genes = [col for col in semi_expression.columns if col != "cell_id"]

    if real_matrix.shape != semi_matrix.shape:
        raise ValueError(f"{rep_dir}: real/semi matrix shape mismatch {real_matrix.shape} vs {semi_matrix.shape}")

    coords = cells[["x", "y"]].to_numpy(dtype=float)
    lr_genes = sorted(
        set(lr_pairs["ligand"].astype(str)).union(set(lr_pairs["receptor"].astype(str))).intersection(genes)
    )
    background_genes = [gene for gene in genes if gene not in set(lr_genes)]

    cell_stats = _cell_stats(dataset_key, dataset_label, rep_dir.name, cells, real_matrix, semi_matrix)
    gene_stats = _gene_stats(dataset_key, dataset_label, rep_dir.name, genes, lr_genes, real_matrix, semi_matrix)
    moran = _moran_stats(
        dataset_key,
        dataset_label,
        rep_dir.name,
        genes,
        lr_genes,
        background_genes,
        coords,
        real_matrix,
        semi_matrix,
        max_genes=max_moran_genes,
    )
    lr_expression = _lr_expression_long(
        dataset_key,
        dataset_label,
        rep_dir.name,
        lr_genes,
        real_expression,
        semi_expression,
    )
    pca = _pca_overlap_source(dataset_key, dataset_label, rep_dir.name, real_matrix, semi_matrix)
    spatial_density = _spatial_density_source(dataset_key, dataset_label, rep_dir.name, coords)

    metrics = []
    metrics.extend(_cell_metric_rows(dataset_key, dataset_label, rep_dir.name, cells, cell_stats, spatial_density))
    metrics.extend(_gene_metric_rows(dataset_key, dataset_label, rep_dir.name, gene_stats))
    metrics.extend(_moran_metric_rows(dataset_key, dataset_label, rep_dir.name, moran))
    metrics.extend(_pca_metric_rows(dataset_key, dataset_label, rep_dir.name, pca))
    metrics.extend(_lr_metric_rows(dataset_key, dataset_label, rep_dir.name, lr_expression))

    return {
        "metrics": metrics,
        "distributions": {
            "cell_library_sparsity": cell_stats,
            "gene_mean_variance": gene_stats,
            "moran": moran,
            "lr_expression": lr_expression,
            "pca": pca,
            "spatial_density": spatial_density,
        },
    }


def _load_real_expression_for_replicate(metadata: dict[str, Any], semi_expression: pd.DataFrame) -> pd.DataFrame:
    config = metadata["config"]
    fmt = str(config.get("background_format", "h5ad")).lower()
    cell_ids = semi_expression["cell_id"].astype(str).tolist()
    genes = [col for col in semi_expression.columns if col != "cell_id"]
    if fmt == "h5ad":
        return _load_h5ad_expression(config, cell_ids, genes)
    if fmt in {"matrix_market", "mtx"}:
        return _load_matrix_market_expression(config, cell_ids, genes)
    raise ValueError(f"Unsupported background_format={fmt!r}")


def _load_h5ad_expression(config: dict[str, Any], cell_ids: list[str], genes: list[str]) -> pd.DataFrame:
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(PROJECT_ROOT / config["h5ad_path"])
    obs_index = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(_h5ad_gene_symbols(adata).astype(str))
    obs_lookup = {cell_id: idx for idx, cell_id in enumerate(obs_index)}
    gene_lookup = _first_index_lookup(var_names)
    obs_indices = [obs_lookup[cell_id] for cell_id in cell_ids]
    gene_indices = [gene_lookup.get(gene) for gene in genes]
    missing = [gene for gene, idx in zip(genes, gene_indices, strict=True) if idx is None]
    if missing:
        raise KeyError(f"h5ad missing genes from benchmark expression: {missing[:8]}")
    matrix = adata.X[np.ix_(obs_indices, np.asarray(gene_indices, dtype=int))]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    return pd.DataFrame(matrix, columns=genes).assign(cell_id=cell_ids)[["cell_id", *genes]]


def _load_matrix_market_expression(config: dict[str, Any], cell_ids: list[str], genes: list[str]) -> pd.DataFrame:
    from scipy.io import mmread

    matrix_dir = PROJECT_ROOT / config["matrix_dir"]
    features = pd.read_csv(matrix_dir / "features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(matrix_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    matrix = mmread(matrix_dir / "matrix.mtx.gz").tocsr()
    if matrix.shape[0] == len(features) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()
    gene_symbols = features[1].astype(str) if features.shape[1] > 1 else features[0].astype(str)
    barcode_lookup = {barcode: idx for idx, barcode in enumerate(barcodes)}
    gene_lookup = _first_index_lookup(pd.Index(gene_symbols.astype(str)))
    obs_indices = [barcode_lookup[cell_id] for cell_id in cell_ids]
    gene_indices = [gene_lookup.get(gene) for gene in genes]
    missing = [gene for gene, idx in zip(genes, gene_indices, strict=True) if idx is None]
    if missing:
        raise KeyError(f"MatrixMarket background missing genes from benchmark expression: {missing[:8]}")
    selected = matrix[np.ix_(obs_indices, np.asarray(gene_indices, dtype=int))].toarray().astype(float)
    return pd.DataFrame(selected, columns=genes).assign(cell_id=cell_ids)[["cell_id", *genes]]


def _h5ad_gene_symbols(adata: Any) -> pd.Series:
    for column in ("gene_symbols", "SYMBOL", "symbol", "feature_name", "gene_ids"):
        if column in adata.var.columns:
            values = adata.var[column].astype(str)
            if values.notna().any():
                return values
    return pd.Series(adata.var_names.astype(str))


def _first_index_lookup(values: pd.Index) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, value in enumerate(values.astype(str)):
        lookup.setdefault(value, idx)
    return lookup


def _expression_matrix(expression: pd.DataFrame) -> np.ndarray:
    return expression.drop(columns=["cell_id"]).to_numpy(dtype=float)


def _cell_stats(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    cells: pd.DataFrame,
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
) -> pd.DataFrame:
    real_lib = real_matrix.sum(axis=1)
    semi_lib = semi_matrix.sum(axis=1)
    real_sparsity = (real_matrix <= 0).mean(axis=1)
    semi_sparsity = (semi_matrix <= 0).mean(axis=1)
    return pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "replicate": replicate,
            "cell_id": cells["cell_id"].astype(str),
            "region": cells["region"].astype(str),
            "cell_type": cells["cell_type"].astype(str),
            "real_selected_library_size": real_lib,
            "semi_selected_library_size": semi_lib,
            "real_selected_sparsity": real_sparsity,
            "semi_selected_sparsity": semi_sparsity,
            "preserved_full_library_size": pd.to_numeric(cells["library_size"], errors="coerce"),
            "preserved_observed_sparsity": pd.to_numeric(cells["observed_sparsity"], errors="coerce"),
        }
    )


def _gene_stats(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    genes: list[str],
    lr_genes: list[str],
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "replicate": replicate,
            "gene": genes,
            "is_lr_gene": [gene in set(lr_genes) for gene in genes],
            "real_mean": real_matrix.mean(axis=0),
            "semi_mean": semi_matrix.mean(axis=0),
            "real_variance": real_matrix.var(axis=0),
            "semi_variance": semi_matrix.var(axis=0),
            "real_sparsity": (real_matrix <= 0).mean(axis=0),
            "semi_sparsity": (semi_matrix <= 0).mean(axis=0),
        }
    )


def _moran_stats(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    genes: list[str],
    lr_genes: list[str],
    background_genes: list[str],
    coords: np.ndarray,
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
    *,
    max_genes: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(7137)
    if len(background_genes) > max_genes:
        background_genes = sorted(rng.choice(background_genes, size=max_genes, replace=False).tolist())
    selected_genes = lr_genes + [gene for gene in background_genes if gene not in set(lr_genes)]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    weights = _knn_weights(coords, k=8)
    rows = []
    for gene in selected_genes:
        idx = gene_to_idx[gene]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "replicate": replicate,
                "gene": gene,
                "is_lr_gene": gene in set(lr_genes),
                "real_morans_i": _morans_i(real_matrix[:, idx], weights),
                "semi_morans_i": _morans_i(semi_matrix[:, idx], weights),
            }
        )
    return pd.DataFrame(rows)


def _lr_expression_long(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    lr_genes: list[str],
    real_expression: pd.DataFrame,
    semi_expression: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for gene in lr_genes:
        for source, expression in (("real", real_expression), ("semi", semi_expression)):
            values = pd.to_numeric(expression[gene], errors="coerce").to_numpy(dtype=float)
            sampled = _deterministic_sample(values, max_n=500, seed=abs(hash((replicate, gene, source))) % (2**32))
            rows.extend(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "replicate": replicate,
                    "gene": gene,
                    "source": source,
                    "expression": float(value),
                    "log1p_expression": float(np.log1p(value)),
                }
                for value in sampled
            )
    return pd.DataFrame(rows)


def _pca_overlap_source(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = real_matrix.shape[0]
    stacked = np.vstack([np.log1p(real_matrix), np.log1p(semi_matrix)])
    if stacked.shape[1] > 600:
        rng = np.random.default_rng(77)
        keep = np.sort(rng.choice(stacked.shape[1], size=600, replace=False))
        stacked = stacked[:, keep]
    scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(stacked)
    coords = PCA(n_components=2, random_state=77).fit_transform(scaled)
    return pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "replicate": replicate,
            "source": ["real"] * n + ["semi"] * n,
            "matched_cell_index": list(range(n)) + list(range(n)),
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        }
    )


def _spatial_density_source(dataset_key: str, dataset_label: str, replicate: str, coords: np.ndarray) -> pd.DataFrame:
    distances = _nearest_distances(coords)
    local_density = _local_density(coords, k=8)
    return pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "replicate": replicate,
            "cell_index": np.arange(len(coords)),
            "nn_distance": distances,
            "local_density": local_density,
        }
    )


def _cell_metric_rows(
    dataset_key: str,
    dataset_label: str,
    replicate: str,
    cells: pd.DataFrame,
    cell_stats: pd.DataFrame,
    spatial_density: pd.DataFrame,
) -> list[dict[str, Any]]:
    real_lib = cell_stats["real_selected_library_size"].to_numpy(dtype=float)
    semi_lib = cell_stats["semi_selected_library_size"].to_numpy(dtype=float)
    real_sparse = cell_stats["real_selected_sparsity"].to_numpy(dtype=float)
    semi_sparse = cell_stats["semi_selected_sparsity"].to_numpy(dtype=float)
    rows = [
        _metric(dataset_key, dataset_label, replicate, "selected_library_size_spearman", _corr(real_lib, semi_lib, "spearman"), "cell_distribution"),
        _metric(dataset_key, dataset_label, replicate, "selected_library_size_relative_mean_delta", _relative_delta(real_lib, semi_lib), "cell_distribution"),
        _metric(dataset_key, dataset_label, replicate, "selected_sparsity_spearman", _corr(real_sparse, semi_sparse, "spearman"), "cell_distribution"),
        _metric(dataset_key, dataset_label, replicate, "selected_sparsity_mean_delta", float(np.nanmean(semi_sparse - real_sparse)), "cell_distribution"),
        _metric(dataset_key, dataset_label, replicate, "coordinate_span_x", float(cells["x"].max() - cells["x"].min()), "spatial_geometry"),
        _metric(dataset_key, dataset_label, replicate, "coordinate_span_y", float(cells["y"].max() - cells["y"].min()), "spatial_geometry"),
        _metric(dataset_key, dataset_label, replicate, "median_nn_distance", float(spatial_density["nn_distance"].median()), "spatial_geometry"),
        _metric(dataset_key, dataset_label, replicate, "median_local_density", float(spatial_density["local_density"].median()), "spatial_geometry"),
        _metric(dataset_key, dataset_label, replicate, "n_regions", float(cells["region"].nunique()), "composition"),
        _metric(dataset_key, dataset_label, replicate, "n_cell_types", float(cells["cell_type"].nunique()), "composition"),
    ]
    return rows


def _gene_metric_rows(dataset_key: str, dataset_label: str, replicate: str, gene_stats: pd.DataFrame) -> list[dict[str, Any]]:
    bg = gene_stats[~gene_stats["is_lr_gene"].astype(bool)]
    lr = gene_stats[gene_stats["is_lr_gene"].astype(bool)]
    return [
        _metric(dataset_key, dataset_label, replicate, "background_gene_mean_spearman", _corr(bg["real_mean"], bg["semi_mean"], "spearman"), "gene_distribution"),
        _metric(dataset_key, dataset_label, replicate, "background_gene_variance_spearman", _corr(bg["real_variance"], bg["semi_variance"], "spearman"), "gene_distribution"),
        _metric(dataset_key, dataset_label, replicate, "background_gene_sparsity_spearman", _corr(bg["real_sparsity"], bg["semi_sparsity"], "spearman"), "gene_distribution"),
        _metric(dataset_key, dataset_label, replicate, "lr_gene_median_mean_delta", float((lr["semi_mean"] - lr["real_mean"]).median()), "injection_effect"),
        _metric(dataset_key, dataset_label, replicate, "lr_gene_median_sparsity_delta", float((lr["semi_sparsity"] - lr["real_sparsity"]).median()), "injection_effect"),
    ]


def _moran_metric_rows(dataset_key: str, dataset_label: str, replicate: str, moran: pd.DataFrame) -> list[dict[str, Any]]:
    bg = moran[~moran["is_lr_gene"].astype(bool)]
    lr = moran[moran["is_lr_gene"].astype(bool)]
    return [
        _metric(dataset_key, dataset_label, replicate, "background_morans_i_spearman", _corr(bg["real_morans_i"], bg["semi_morans_i"], "spearman"), "spatial_autocorrelation"),
        _metric(dataset_key, dataset_label, replicate, "background_morans_i_median_abs_delta", float((bg["semi_morans_i"] - bg["real_morans_i"]).abs().median()), "spatial_autocorrelation"),
        _metric(dataset_key, dataset_label, replicate, "lr_morans_i_median_delta", float((lr["semi_morans_i"] - lr["real_morans_i"]).median()), "injection_effect"),
    ]


def _pca_metric_rows(dataset_key: str, dataset_label: str, replicate: str, pca: pd.DataFrame) -> list[dict[str, Any]]:
    real = pca[pca["source"] == "real"].sort_values("matched_cell_index")
    semi = pca[pca["source"] == "semi"].sort_values("matched_cell_index")
    matched_distance = np.sqrt((real[["pc1", "pc2"]].to_numpy() - semi[["pc1", "pc2"]].to_numpy()) ** 2).sum(axis=1)
    all_scale = float(np.sqrt(pca["pc1"].var() + pca["pc2"].var()))
    return [
        _metric(dataset_key, dataset_label, replicate, "pca_matched_distance_median", float(np.median(matched_distance)), "pca_overlap"),
        _metric(dataset_key, dataset_label, replicate, "pca_matched_distance_norm", float(np.median(matched_distance) / max(all_scale, 1e-12)), "pca_overlap"),
    ]


def _lr_metric_rows(dataset_key: str, dataset_label: str, replicate: str, lr_expression: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for gene, group in lr_expression.groupby("gene", sort=False):
        real = group.loc[group["source"] == "real", "log1p_expression"].to_numpy(dtype=float)
        semi = group.loc[group["source"] == "semi", "log1p_expression"].to_numpy(dtype=float)
        rows.append(_metric(dataset_key, dataset_label, replicate, f"lr_expression_log1p_median_delta__{gene}", float(np.median(semi) - np.median(real)), "lr_expression"))
    return rows


def _metric(dataset_key: str, dataset_label: str, replicate: str, metric: str, value: float, category: str) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "replicate": replicate,
        "metric": metric,
        "value": value,
        "category": category,
    }


def _build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, group in metrics.groupby("dataset_key", sort=False):
        lookup = group.set_index("metric")["value"].to_dict()
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
                "selected_library_size_spearman": lookup.get("selected_library_size_spearman", np.nan),
                "selected_sparsity_spearman": lookup.get("selected_sparsity_spearman", np.nan),
                "background_gene_mean_spearman": lookup.get("background_gene_mean_spearman", np.nan),
                "background_gene_variance_spearman": lookup.get("background_gene_variance_spearman", np.nan),
                "background_gene_sparsity_spearman": lookup.get("background_gene_sparsity_spearman", np.nan),
                "background_morans_i_spearman": lookup.get("background_morans_i_spearman", np.nan),
                "background_morans_i_median_abs_delta": lookup.get("background_morans_i_median_abs_delta", np.nan),
                "pca_matched_distance_norm": lookup.get("pca_matched_distance_norm", np.nan),
                "median_nn_distance": lookup.get("median_nn_distance", np.nan),
                "median_local_density": lookup.get("median_local_density", np.nan),
                "n_regions": lookup.get("n_regions", np.nan),
                "n_cell_types": lookup.get("n_cell_types", np.nan),
                "interpretation": _summary_interpretation(lookup),
            }
        )
    return pd.DataFrame(rows)


def _summary_interpretation(lookup: dict[str, float]) -> str:
    preserved = (
        lookup.get("background_gene_mean_spearman", 0) >= 0.95
        and lookup.get("background_gene_sparsity_spearman", 0) >= 0.95
        and lookup.get("background_morans_i_spearman", 0) >= 0.90
    )
    if preserved:
        return "background_distribution_preserved_with_targeted_lr_injection"
    return "partial_realism_requires_review"


def _plot_realism_figure(
    metrics: pd.DataFrame,
    distribution_frames: dict[str, list[pd.DataFrame]],
    figure_dir: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    palette = {
        "real": "#4E79A7",
        "semi": "#E15759",
        "lymph_node": "#4E79A7",
        "tonsil": "#59A14F",
        "melanoma": "#B07AA1",
        "background": "#777777",
        "lr": "#D3722C",
    }
    cell_stats = pd.concat(distribution_frames["cell_library_sparsity"], ignore_index=True)
    gene_stats = pd.concat(distribution_frames["gene_mean_variance"], ignore_index=True)
    moran = pd.concat(distribution_frames["moran"], ignore_index=True)
    lr_expression = pd.concat(distribution_frames["lr_expression"], ignore_index=True)
    pca = pd.concat(distribution_frames["pca"], ignore_index=True)
    spatial_density = pd.concat(distribution_frames["spatial_density"], ignore_index=True)

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    _panel_metric_heatmap(ax_a, metrics)
    _panel_library_distribution(ax_b, cell_stats, palette)
    _panel_sparsity_distribution(ax_c, cell_stats, palette)
    _panel_mean_variance(ax_d, gene_stats, palette)
    _panel_moran(ax_e, moran, palette)
    _panel_nn_density(ax_f, spatial_density, palette)
    _panel_lr_expression(ax_g, lr_expression, palette)
    _panel_pca(ax_h, pca, palette)
    _panel_injection_effect(ax_i, metrics, palette)

    for label, ax in zip("abcdefghi", [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i], strict=True):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    base = figure_dir / "fig_phase7_benchmark_realism"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel_metric_heatmap(ax: Any, metrics: pd.DataFrame) -> None:
    selected = [
        ("selected_library_size_spearman", "Library"),
        ("selected_sparsity_spearman", "Sparsity"),
        ("background_gene_mean_spearman", "Gene mean"),
        ("background_gene_variance_spearman", "Gene var."),
        ("background_gene_sparsity_spearman", "Gene zero"),
        ("background_morans_i_spearman", "Moran's I"),
    ]
    datasets = ["lymph_node", "tonsil", "melanoma"]
    matrix = []
    for metric, _label in selected:
        values = []
        for dataset in datasets:
            row = metrics[(metrics["dataset_key"] == dataset) & (metrics["metric"] == metric)]
            values.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        matrix.append(values)
    im = ax.imshow(np.asarray(matrix), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets], rotation=35, ha="right")
    ax.set_yticks(range(len(selected)), [label for _metric, label in selected])
    ax.set_title("Realism correlations")
    for i in range(len(selected)):
        for j in range(len(datasets)):
            value = matrix[i][j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > 0.65 else "black")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)


def _panel_library_distribution(ax: Any, cell_stats: pd.DataFrame, palette: dict[str, str]) -> None:
    rows = []
    for source, column in (("real", "real_selected_library_size"), ("semi", "semi_selected_library_size")):
        temp = cell_stats[["dataset_key", "dataset_label", column]].rename(columns={column: "value"}).copy()
        temp["source"] = source
        rows.append(temp)
    data = pd.concat(rows, ignore_index=True)
    _paired_boxplot(ax, data, value_col="value", palette=palette)
    ax.set_yscale("log")
    ax.set_ylabel("Selected-gene library")
    ax.set_title("Library size")


def _panel_sparsity_distribution(ax: Any, cell_stats: pd.DataFrame, palette: dict[str, str]) -> None:
    rows = []
    for source, column in (("real", "real_selected_sparsity"), ("semi", "semi_selected_sparsity")):
        temp = cell_stats[["dataset_key", "dataset_label", column]].rename(columns={column: "value"}).copy()
        temp["source"] = source
        rows.append(temp)
    data = pd.concat(rows, ignore_index=True)
    _paired_boxplot(ax, data, value_col="value", palette=palette)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Zero fraction")
    ax.set_title("Expression sparsity")


def _panel_mean_variance(ax: Any, gene_stats: pd.DataFrame, palette: dict[str, str]) -> None:
    for is_lr, color, label, alpha in ((False, palette["background"], "Background genes", 0.18), (True, palette["lr"], "LR genes", 0.85)):
        data = gene_stats[gene_stats["is_lr_gene"].astype(bool) == is_lr]
        ax.scatter(np.log1p(data["real_mean"]), np.log1p(data["semi_mean"]), s=8 if is_lr else 4, alpha=alpha, c=color, edgecolors="none", label=label)
    lim = _square_lim(ax)
    ax.plot(lim, lim, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real log1p mean")
    ax.set_ylabel("Semi log1p mean")
    ax.set_title("Mean expression")
    ax.legend(loc="lower right", fontsize=6)


def _panel_moran(ax: Any, moran: pd.DataFrame, palette: dict[str, str]) -> None:
    for is_lr, color, label, alpha in ((False, palette["background"], "Background", 0.25), (True, palette["lr"], "LR", 0.9)):
        data = moran[moran["is_lr_gene"].astype(bool) == is_lr]
        ax.scatter(data["real_morans_i"], data["semi_morans_i"], s=8 if is_lr else 5, alpha=alpha, c=color, edgecolors="none", label=label)
    lim = _square_lim(ax)
    ax.plot(lim, lim, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real Moran's I")
    ax.set_ylabel("Semi Moran's I")
    ax.set_title("Spatial autocorrelation")


def _panel_nn_density(ax: Any, spatial_density: pd.DataFrame, palette: dict[str, str]) -> None:
    datasets = ["lymph_node", "tonsil", "melanoma"]
    positions = np.arange(len(datasets))
    values = [spatial_density.loc[spatial_density["dataset_key"] == dataset, "nn_distance"].to_numpy(dtype=float) for dataset in datasets]
    box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, dataset in zip(box["boxes"], datasets, strict=True):
        patch.set_facecolor(palette[dataset])
        patch.set_alpha(0.65)
        patch.set_linewidth(0.8)
    ax.set_xticks(positions, [DATASET_LABELS[d] for d in datasets], rotation=35, ha="right")
    ax.set_ylabel("Nearest-neighbor distance")
    ax.set_title("Spatial density")


def _panel_lr_expression(ax: Any, lr_expression: pd.DataFrame, palette: dict[str, str]) -> None:
    data = lr_expression.copy()
    order = sorted(data["dataset_key"].unique())
    positions = []
    values = []
    colors = []
    labels = []
    pos = 0
    for dataset in order:
        for source in ("real", "semi"):
            subset = data[(data["dataset_key"] == dataset) & (data["source"] == source)]
            values.append(subset["log1p_expression"].to_numpy(dtype=float))
            positions.append(pos)
            colors.append(palette[source])
            labels.append(DATASET_LABELS.get(dataset, dataset) if source == "real" else "")
            pos += 0.75
        pos += 0.35
    box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_linewidth(0.8)
    tick_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
    ax.set_xticks(tick_positions, [DATASET_LABELS.get(dataset, dataset) for dataset in order], rotation=35, ha="right")
    ax.set_ylabel("LR gene log1p expression")
    ax.set_title("Injected LR expression")
    ax.plot([], [], color=palette["real"], lw=6, alpha=0.68, label="Real")
    ax.plot([], [], color=palette["semi"], lw=6, alpha=0.68, label="Semi")
    ax.legend(fontsize=6, loc="upper left")


def _panel_pca(ax: Any, pca: pd.DataFrame, palette: dict[str, str]) -> None:
    data = pca.copy()
    rng = np.random.default_rng(11)
    if len(data) > 1200:
        data = data.iloc[np.sort(rng.choice(len(data), size=1200, replace=False))]
    for source in ("real", "semi"):
        subset = data[data["source"] == source]
        ax.scatter(subset["pc1"], subset["pc2"], s=4, alpha=0.35, c=palette[source], edgecolors="none", label=source.capitalize())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA overlap")
    ax.legend(fontsize=6, loc="best")


def _panel_injection_effect(ax: Any, metrics: pd.DataFrame, palette: dict[str, str]) -> None:
    rows = metrics[metrics["metric"].str.startswith("lr_expression_log1p_median_delta__")].copy()
    rows["gene"] = rows["metric"].str.replace("lr_expression_log1p_median_delta__", "", regex=False)
    rows = rows.sort_values(["dataset_key", "value"], ascending=[True, False])
    top = rows.groupby("dataset_key", sort=False).head(4)
    y = np.arange(len(top))
    colors = [palette.get(dataset, "#999999") for dataset in top["dataset_key"]]
    ax.barh(y, top["value"], color=colors, alpha=0.75)
    ax.set_yticks(y, [f"{DATASET_LABELS.get(d, d)} {g}" for d, g in zip(top["dataset_key"], top["gene"], strict=True)], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Semi - real median log1p")
    ax.set_title("Targeted LR shifts")


def _paired_boxplot(ax: Any, data: pd.DataFrame, *, value_col: str, palette: dict[str, str]) -> None:
    datasets = ["lymph_node", "tonsil", "melanoma"]
    positions = []
    values = []
    colors = []
    pos = 0
    for dataset in datasets:
        for source in ("real", "semi"):
            subset = data[(data["dataset_key"] == dataset) & (data["source"] == source)]
            values.append(subset[value_col].to_numpy(dtype=float))
            positions.append(pos)
            colors.append(palette[source])
            pos += 0.7
        pos += 0.35
    box = ax.boxplot(values, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_linewidth(0.8)
    tick_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
    ax.set_xticks(tick_positions, [DATASET_LABELS[d] for d in datasets], rotation=35, ha="right")
    ax.plot([], [], color=palette["real"], lw=6, alpha=0.68, label="Real")
    ax.plot([], [], color=palette["semi"], lw=6, alpha=0.68, label="Semi")


def _knn_weights(coords: np.ndarray, *, k: int) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    kth = min(k, len(coords) - 1)
    neighbors = np.argsort(distances, axis=1)[:, :kth]
    weights = np.zeros((len(coords), len(coords)), dtype=float)
    row_idx = np.repeat(np.arange(len(coords)), kth)
    weights[row_idx, neighbors.reshape(-1)] = 1.0
    weights = np.maximum(weights, weights.T)
    return weights


def _morans_i(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 3 or np.nanvar(x[valid]) <= 0:
        return float("nan")
    x = x - np.nanmean(x)
    denom = float(np.nansum(x * x))
    w_sum = float(weights.sum())
    if denom <= 0 or w_sum <= 0:
        return float("nan")
    return float((len(x) / w_sum) * (weights * np.outer(x, x)).sum() / denom)


def _nearest_distances(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _local_density(coords: np.ndarray, *, k: int) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    kth = min(k, len(coords) - 1)
    radius = np.sort(distances, axis=1)[:, kth - 1]
    return kth / (math.pi * np.maximum(radius, 1e-12) ** 2)


def _corr(a: Any, b: Any, method: str) -> float:
    left = pd.Series(a, dtype="float64")
    right = pd.Series(b, dtype="float64")
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return float("nan")
    return float(left[valid].corr(right[valid], method=method))


def _relative_delta(real: np.ndarray, semi: np.ndarray) -> float:
    real_mean = float(np.nanmean(real))
    semi_mean = float(np.nanmean(semi))
    return float((semi_mean - real_mean) / max(abs(real_mean), 1e-12))


def _deterministic_sample(values: np.ndarray, *, max_n: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= max_n:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=max_n, replace=False))]


def _square_lim(ax: Any) -> tuple[float, float]:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _dataset_key(name: str) -> str:
    lowered = name.lower()
    if "lymph" in lowered:
        return "lymph_node"
    if "tonsil" in lowered:
        return "tonsil"
    if "melanoma" in lowered:
        return "melanoma"
    return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
