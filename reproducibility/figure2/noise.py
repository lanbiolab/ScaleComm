"""Paper noise transformation on one generated replicate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

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
