#!/usr/bin/env python
"""Generate noisy versions of the scDesign3 mechanism benchmark.

The ground-truth candidate universe is kept fixed while observed expression is
perturbed. This follows the CellNEST-style controlled benchmark idea of testing
the same truth under increasing measurement noise.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "lymph_scdesign3": Path(
        "outputs/reference_based_benchmark/native_rerun/recovered_inputs/lymph/reference_lymph_node_scdesign3_replicate_00"
    ),
    "melanoma_scdesign3": Path(
        "outputs/reference_based_benchmark/native_rerun/recovered_inputs/melanoma/reference_melanoma_scdesign3_replicate_00"
    ),
}

NOISE_LEVELS = {
    "low_noise": {
        "dropout_rate": 0.05,
        "library_lognormal_sigma": 0.08,
        "gene_lognormal_sigma": 0.06,
        "additive_poisson_rate": 0.03,
    },
    "high_noise": {
        "dropout_rate": 0.18,
        "library_lognormal_sigma": 0.22,
        "gene_lognormal_sigma": 0.16,
        "additive_poisson_rate": 0.10,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/noisy_scdesign3_mechanism/datasets",
    )
    parser.add_argument(
        "--source-manifest",
        default="",
        help="Optional manifest with dataset_id and replicate_dir columns. Defaults to the retained 6-LR benchmark.",
    )
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    datasets = _source_datasets(args.source_manifest)
    for dataset_id, source_dir in datasets.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source benchmark directory: {source_dir}")
        for level_id, params in NOISE_LEVELS.items():
            noisy_dataset_id = f"{dataset_id}_{level_id}"
            rng = np.random.default_rng(args.seed + abs(hash((dataset_id, level_id))) % 1_000_000)
            target_dir = output_dir / noisy_dataset_id / source_dir.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)
            _apply_expression_noise(target_dir, params, rng)
            manifest_rows.append(
                {
                    "dataset_id": noisy_dataset_id,
                    "source_dataset_id": dataset_id,
                    "noise_level": level_id,
                    "source_dir": str(source_dir),
                    "replicate_dir": str(target_dir),
                    **params,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "noisy_scdesign3_manifest.csv", index=False)
    (output_dir / "noisy_scdesign3_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False))
    print(f"Wrote noisy benchmark datasets to {output_dir}")


def _source_datasets(source_manifest: str) -> dict[str, Path]:
    if not source_manifest:
        return DATASETS
    manifest_path = Path(source_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = pd.read_csv(manifest_path)
    required = {"dataset_id", "replicate_dir"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"source manifest missing columns: {sorted(missing)}")
    return {
        str(row.dataset_id): _resolve_project_path(str(row.replicate_dir))
        for row in manifest.itertuples(index=False)
    }


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _apply_expression_noise(target_dir: Path, params: dict, rng: np.random.Generator) -> None:
    expression = pd.read_csv(target_dir / "expression.csv")
    cells = pd.read_csv(target_dir / "cells.csv")
    candidates = pd.read_csv(target_dir / "candidate_interactions.csv")

    cell_ids = expression["cell_id"].astype(str)
    genes = [column for column in expression.columns if column != "cell_id"]
    matrix = expression[genes].to_numpy(dtype=float)

    dropout_rate = float(params["dropout_rate"])
    library_sigma = float(params["library_lognormal_sigma"])
    gene_sigma = float(params["gene_lognormal_sigma"])
    additive_rate = float(params["additive_poisson_rate"])

    library_scale = rng.lognormal(mean=-0.5 * library_sigma**2, sigma=library_sigma, size=(matrix.shape[0], 1))
    gene_scale = rng.lognormal(mean=-0.5 * gene_sigma**2, sigma=gene_sigma, size=(1, matrix.shape[1]))
    noisy = matrix * library_scale * gene_scale

    if additive_rate > 0:
        low_background = additive_rate * np.maximum(np.mean(matrix, axis=0, keepdims=True), 0.05)
        noisy = noisy + rng.poisson(low_background, size=noisy.shape)

    dropout = rng.random(noisy.shape) < dropout_rate
    noisy[dropout] = 0.0
    noisy = np.rint(np.clip(noisy, 0.0, None)).astype(int)

    expression_noisy = pd.DataFrame(noisy, columns=genes)
    expression_noisy.insert(0, "cell_id", cell_ids.to_numpy())
    expression_noisy.to_csv(target_dir / "expression.csv", index=False)

    cells = cells.copy()
    cells["library_size"] = noisy.sum(axis=1).astype(float)
    cells["observed_sparsity"] = (noisy <= 0).mean(axis=1).astype(float)
    cells.to_csv(target_dir / "cells.csv", index=False)

    expr_lookup = expression_noisy.set_index("cell_id")
    candidates = candidates.copy()
    candidates["ligand_expr"] = _lookup_expression(expr_lookup, candidates["sender_id"], candidates["ligand"])
    candidates["receptor_expr"] = _lookup_expression(expr_lookup, candidates["receiver_id"], candidates["receptor"])
    candidates["coexpression_distance_score"] = (
        np.log1p(candidates["ligand_expr"].to_numpy(dtype=float))
        * np.log1p(candidates["receptor_expr"].to_numpy(dtype=float))
        / (1.0 + candidates["distance"].to_numpy(dtype=float))
    )
    candidates.to_csv(target_dir / "candidate_interactions.csv", index=False)

    summary = {
        "dropout_rate": dropout_rate,
        "library_lognormal_sigma": library_sigma,
        "gene_lognormal_sigma": gene_sigma,
        "additive_poisson_rate": additive_rate,
        "mean_library_size": float(cells["library_size"].mean()),
        "mean_observed_sparsity": float(cells["observed_sparsity"].mean()),
        "n_candidates": int(len(candidates)),
        "n_positive": int(candidates["ground_truth"].astype(int).sum()),
    }
    (target_dir / "noise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _lookup_expression(expr: pd.DataFrame, cell_ids: pd.Series, genes: pd.Series) -> np.ndarray:
    values = np.zeros(len(cell_ids), dtype=float)
    cell_ids = cell_ids.astype(str).reset_index(drop=True)
    genes = genes.astype(str).reset_index(drop=True)
    for gene in sorted(set(genes)):
        mask = genes.eq(gene).to_numpy()
        if gene in expr.columns:
            values[mask] = expr.loc[cell_ids[mask], gene].to_numpy(dtype=float)
    return values


if __name__ == "__main__":
    main()
