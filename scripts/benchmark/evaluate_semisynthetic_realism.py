#!/usr/bin/env python
"""Quantify realism of semi-synthetic ST backgrounds against the source h5ad."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_semisynthetic_realism(replicate_dir: Path, output_dir: Path) -> dict[str, Path]:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    metadata = json.loads((replicate_dir / "metadata.json").read_text(encoding="utf-8"))

    real_expression = _source_expression_for_output_genes(
        Path(metadata["source_h5ad"]),
        cells["cell_id"].astype(str).tolist(),
        [column for column in expression.columns if column != "cell_id"],
    )
    semi_matrix = expression.drop(columns=["cell_id"]).to_numpy(dtype=float)
    real_matrix = real_expression.drop(columns=["cell_id"]).to_numpy(dtype=float)
    genes = [column for column in expression.columns if column != "cell_id"]
    lr_genes = sorted(set(lr_pairs["ligand"].astype(str)) | set(lr_pairs["receptor"].astype(str)))

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_stats = _gene_stats(genes, real_matrix, semi_matrix, lr_genes)
    gene_stats_path = output_dir / "semisynthetic_realism_gene_stats.csv"
    gene_stats.to_csv(gene_stats_path, index=False)

    moran = _moran_panel(cells, genes, real_matrix, semi_matrix, lr_genes)
    moran_path = output_dir / "semisynthetic_realism_moran.csv"
    moran.to_csv(moran_path, index=False)

    composition = _composition_stats(cells)
    composition_path = output_dir / "semisynthetic_realism_composition_geometry.csv"
    composition.to_csv(composition_path, index=False)

    summary = _summary_table(cells, real_matrix, semi_matrix, gene_stats, moran, lr_genes)
    summary_path = output_dir / "semisynthetic_realism_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "replicate_dir": str(replicate_dir),
        "source_h5ad": metadata["source_h5ad"],
        "n_cells": int(len(cells)),
        "n_genes": int(len(genes)),
        "n_lr_genes": int(len(lr_genes)),
        "outputs": {
            "summary": str(summary_path),
            "gene_stats": str(gene_stats_path),
            "moran": str(moran_path),
            "composition_geometry": str(composition_path),
        },
    }
    manifest_path = output_dir / "semisynthetic_realism_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "summary": summary_path,
        "gene_stats": gene_stats_path,
        "moran": moran_path,
        "composition_geometry": composition_path,
        "manifest": manifest_path,
    }


def _source_expression_for_output_genes(
    h5ad_path: Path,
    cell_ids: list[str],
    genes: list[str],
) -> pd.DataFrame:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError("scanpy is required to evaluate semi-synthetic realism") from exc

    adata = sc.read_h5ad(h5ad_path)
    adata = adata[cell_ids].copy()
    symbol_to_index = _first_symbol_indices(adata)
    data: dict[str, np.ndarray] = {"cell_id": np.asarray(cell_ids, dtype=str)}
    for gene in genes:
        idx = symbol_to_index.get(str(gene))
        if idx is None:
            data[gene] = np.zeros(adata.n_obs, dtype=float)
            continue
        values = adata.X[:, int(idx)]
        if hasattr(values, "toarray"):
            values = values.toarray()
        data[gene] = np.asarray(values).reshape(-1).astype(float)
    return pd.DataFrame(data)


def _first_symbol_indices(adata) -> dict[str, int]:
    if "SYMBOL" in adata.var.columns:
        symbols = adata.var["SYMBOL"].astype(str).to_numpy()
    else:
        symbols = adata.var_names.astype(str)
    out: dict[str, int] = {}
    for idx, symbol in enumerate(symbols):
        if symbol not in out:
            out[symbol] = idx
    return out


def _gene_stats(
    genes: list[str],
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
    lr_genes: list[str],
) -> pd.DataFrame:
    rows = []
    lr_gene_set = set(lr_genes)
    for idx, gene in enumerate(genes):
        real = real_matrix[:, idx]
        semi = semi_matrix[:, idx]
        rows.append(
            {
                "gene": gene,
                "is_lr_gene": gene in lr_gene_set,
                "real_mean": float(real.mean()),
                "semi_mean": float(semi.mean()),
                "real_variance": float(real.var()),
                "semi_variance": float(semi.var()),
                "real_sparsity": float(np.mean(real == 0)),
                "semi_sparsity": float(np.mean(semi == 0)),
                "mean_delta": float(semi.mean() - real.mean()),
                "variance_delta": float(semi.var() - real.var()),
                "sparsity_delta": float(np.mean(semi == 0) - np.mean(real == 0)),
                "pearson_cell_expression": _corr(real, semi, method="pearson"),
                "spearman_cell_expression": _corr(real, semi, method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def _moran_panel(
    cells: pd.DataFrame,
    genes: list[str],
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
    lr_genes: list[str],
    *,
    k: int = 8,
    max_background_genes: int = 200,
) -> pd.DataFrame:
    lr_gene_set = set(lr_genes)
    variance = real_matrix.var(axis=0)
    background_indices = [
        idx for idx in np.argsort(variance)[::-1].tolist() if genes[idx] not in lr_gene_set
    ][:max_background_genes]
    lr_indices = [idx for idx, gene in enumerate(genes) if gene in lr_gene_set]
    selected = sorted(set(background_indices + lr_indices))
    neighbors = _knn_indices(cells[["x", "y"]].to_numpy(dtype=float), k=k)
    rows = []
    for idx in selected:
        real_i = _morans_i(real_matrix[:, idx], neighbors)
        semi_i = _morans_i(semi_matrix[:, idx], neighbors)
        rows.append(
            {
                "gene": genes[idx],
                "is_lr_gene": genes[idx] in lr_gene_set,
                "real_morans_i": real_i,
                "semi_morans_i": semi_i,
                "morans_i_delta": semi_i - real_i,
            }
        )
    return pd.DataFrame(rows)


def _composition_stats(cells: pd.DataFrame) -> pd.DataFrame:
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    nn = _nearest_neighbor_distances(coords)
    rows = [
        {"statistic": "n_cells", "value": float(len(cells))},
        {"statistic": "coordinate_span_x", "value": float(cells["x"].max() - cells["x"].min())},
        {"statistic": "coordinate_span_y", "value": float(cells["y"].max() - cells["y"].min())},
        {"statistic": "median_nearest_neighbor_distance", "value": float(np.median(nn))},
        {"statistic": "iqr_nearest_neighbor_distance", "value": _iqr(nn)},
        {"statistic": "n_regions", "value": float(cells["region"].nunique())},
        {"statistic": "n_cell_types", "value": float(cells["cell_type"].nunique())},
        {
            "statistic": "dominant_region_fraction",
            "value": float(cells["region"].value_counts(normalize=True).max()),
        },
        {
            "statistic": "dominant_cell_type_fraction",
            "value": float(cells["cell_type"].value_counts(normalize=True).max()),
        },
    ]
    for column in ("region", "cell_type"):
        fractions = cells[column].value_counts(normalize=True)
        for label, value in fractions.items():
            rows.append({"statistic": f"{column}_fraction:{label}", "value": float(value)})
    return pd.DataFrame(rows)


def _summary_table(
    cells: pd.DataFrame,
    real_matrix: np.ndarray,
    semi_matrix: np.ndarray,
    gene_stats: pd.DataFrame,
    moran: pd.DataFrame,
    lr_genes: list[str],
) -> pd.DataFrame:
    real_library = real_matrix.sum(axis=1)
    semi_library = semi_matrix.sum(axis=1)
    real_sparsity = np.mean(real_matrix == 0, axis=1)
    semi_sparsity = np.mean(semi_matrix == 0, axis=1)
    background_gene_stats = gene_stats[~gene_stats["is_lr_gene"]]
    lr_gene_stats = gene_stats[gene_stats["is_lr_gene"]]
    background_moran = moran[~moran["is_lr_gene"]]
    lr_moran = moran[moran["is_lr_gene"]]

    rows = [
        _metric("cells", len(cells), "count"),
        _metric("genes_compared", real_matrix.shape[1], "count"),
        _metric("lr_genes_compared", len(lr_genes), "count"),
        _metric("selected_library_size_pearson", _corr(real_library, semi_library), "cell_distribution"),
        _metric("selected_library_size_spearman", _corr(real_library, semi_library, method="spearman"), "cell_distribution"),
        _metric("selected_library_size_relative_mean_delta", _relative_delta(real_library.mean(), semi_library.mean()), "cell_distribution"),
        _metric("selected_library_size_wasserstein_norm", _wasserstein_norm(real_library, semi_library), "cell_distribution"),
        _metric("cell_sparsity_pearson", _corr(real_sparsity, semi_sparsity), "cell_distribution"),
        _metric("cell_sparsity_mean_delta", float(semi_sparsity.mean() - real_sparsity.mean()), "cell_distribution"),
        _metric("cell_sparsity_wasserstein_norm", _wasserstein_norm(real_sparsity, semi_sparsity), "cell_distribution"),
        _metric("gene_mean_pearson_all", _corr(gene_stats["real_mean"], gene_stats["semi_mean"]), "gene_distribution"),
        _metric("gene_variance_pearson_all", _corr(gene_stats["real_variance"], gene_stats["semi_variance"]), "gene_distribution"),
        _metric("gene_sparsity_pearson_all", _corr(gene_stats["real_sparsity"], gene_stats["semi_sparsity"]), "gene_distribution"),
        _metric("gene_mean_pearson_background", _corr(background_gene_stats["real_mean"], background_gene_stats["semi_mean"]), "gene_distribution"),
        _metric("gene_variance_pearson_background", _corr(background_gene_stats["real_variance"], background_gene_stats["semi_variance"]), "gene_distribution"),
        _metric("gene_sparsity_pearson_background", _corr(background_gene_stats["real_sparsity"], background_gene_stats["semi_sparsity"]), "gene_distribution"),
        _metric("median_abs_sparsity_delta_background", float(background_gene_stats["sparsity_delta"].abs().median()), "gene_distribution"),
        _metric("median_mean_delta_lr_genes", float(lr_gene_stats["mean_delta"].median()), "injection_effect"),
        _metric("max_mean_delta_lr_genes", float(lr_gene_stats["mean_delta"].max()), "injection_effect"),
        _metric("moran_pearson_background", _corr(background_moran["real_morans_i"], background_moran["semi_morans_i"]), "spatial_autocorrelation"),
        _metric("moran_median_abs_delta_background", float(background_moran["morans_i_delta"].abs().median()), "spatial_autocorrelation"),
        _metric("moran_median_delta_lr_genes", float(lr_moran["morans_i_delta"].median()), "injection_effect"),
        _metric("coordinate_span_x", float(cells["x"].max() - cells["x"].min()), "geometry"),
        _metric("coordinate_span_y", float(cells["y"].max() - cells["y"].min()), "geometry"),
        _metric("n_regions", float(cells["region"].nunique()), "composition"),
        _metric("n_cell_types", float(cells["cell_type"].nunique()), "composition"),
    ]
    return pd.DataFrame(rows)


def _metric(name: str, value: float, category: str) -> dict[str, object]:
    return {"metric": name, "value": float(value), "category": category}


def _knn_indices(coords: np.ndarray, *, k: int) -> list[np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for kNN spatial realism metrics") from exc
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(coords)))
    if idx.ndim == 1:
        idx = idx[:, None]
    return [row[row != i].astype(int) for i, row in enumerate(idx)]


def _morans_i(values: np.ndarray, neighbors: list[np.ndarray]) -> float:
    x = np.asarray(values, dtype=float)
    centered = x - x.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 0:
        return float("nan")
    numerator = 0.0
    weight_sum = 0
    for i, neigh in enumerate(neighbors):
        if len(neigh) == 0:
            continue
        numerator += float(np.sum(centered[i] * centered[neigh]))
        weight_sum += int(len(neigh))
    if weight_sum == 0:
        return float("nan")
    return float((len(x) / weight_sum) * (numerator / denom))


def _nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for nearest-neighbor realism metrics") from exc
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return np.asarray(distances[:, 1], dtype=float)


def _corr(x, y, *, method: str = "pearson") -> float:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def _iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _relative_delta(real: float, semi: float) -> float:
    return float((semi - real) / max(abs(real), 1e-12))


def _wasserstein_norm(real: np.ndarray, semi: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return float("nan")
    real = np.asarray(real, dtype=float)
    semi = np.asarray(semi, dtype=float)
    scale = _iqr(real)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(real))
    return float(wasserstein_distance(real, semi) / max(scale, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replicate_dir", help="Semi-synthetic replicate directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for realism reports.")
    args = parser.parse_args()

    outputs = evaluate_semisynthetic_realism(Path(args.replicate_dir), Path(args.output_dir))
    summary = pd.read_csv(outputs["summary"])
    print(summary.to_string(index=False))
    print(f"Wrote semi-synthetic realism outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
