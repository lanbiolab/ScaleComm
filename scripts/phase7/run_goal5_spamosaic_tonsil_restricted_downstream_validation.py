#!/usr/bin/env python
"""Receiver-restricted residual downstream validation for SPAMOSAIC tonsil."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import bh_adjust


GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

TARGET_RULES = {
    "CXCL13-CXCR5": {
        "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
        "receiver_niches": ["germinal_center_proxy"],
        "targets": ["MS4A1", "CD19", "CR2", "CD40", "TNFRSF13C", "AICDA", "BCL6", "PAX5", "CD79A", "CD79B", "BANK1"],
        "covariates": ["library_size", "CXCR5", "receiver_type"],
    },
    "CCL21-CCR7": {
        "receiver_cell_types": ["T_cell_proxy", "Tfh_proxy"],
        "receiver_niches": ["T_zone_proxy"],
        "targets": ["CD3D", "CD3E", "CD4", "CD8A", "IL7R", "LTB", "SELL", "TCF7", "LEF1", "MAL"],
        "covariates": ["library_size", "CCR7", "receiver_type"],
    },
    "CD40LG-CD40": {
        "receiver_cell_types": ["B_cell_proxy", "FDC_proxy"],
        "receiver_niches": ["germinal_center_proxy"],
        "targets": ["MS4A1", "CD19", "TNFRSF13C", "CR2", "NFKBIA", "TNFAIP3", "IRF4", "AICDA", "BCL6", "CD83", "CD86"],
        "covariates": ["library_size", "CD40", "receiver_type"],
    },
    "IL21-IL21R": {
        "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
        "receiver_niches": ["germinal_center_proxy"],
        "targets": ["MS4A1", "CD19", "CD40", "TNFRSF13C", "CR2", "PRDM1", "IRF4", "BCL6", "AICDA", "JCHAIN", "XBP1", "MZB1"],
        "covariates": ["library_size", "IL21R", "receiver_type"],
    },
    "TNFSF13B-TNFRSF13C": {
        "receiver_cell_types": ["B_cell_proxy"],
        "receiver_niches": ["germinal_center_proxy"],
        "targets": ["MS4A1", "CD19", "CD40", "CR2", "BANK1", "PAX5", "CD79A", "CD79B", "NFKBIA", "TNFAIP3"],
        "covariates": ["library_size", "TNFRSF13C", "receiver_type"],
    },
    "ACTB-EGFR": {
        "receiver_cell_types": ["epithelial_proxy", "myeloid_proxy"],
        "receiver_niches": ["epithelial_proxy", "myeloid_stromal_proxy"],
        "targets": ["EPCAM", "KRT8", "KRT18", "KRT19", "VIM", "ACTA2"],
        "covariates": ["library_size", "EGFR", "receiver_type"],
    },
    "MS4A1-CD3E": {
        "receiver_cell_types": ["T_cell_proxy", "Tfh_proxy"],
        "receiver_niches": ["T_zone_proxy", "germinal_center_proxy"],
        "targets": ["CD3D", "CD4", "CD8A", "IL7R", "LTB", "TCF7", "LEF1"],
        "covariates": ["library_size", "CD3E", "receiver_type"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section-id",
        choices=("A1", "D1"),
        default="A1",
        help="SPAMOSAIC tonsil section to validate.",
    )
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=202611)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    section_id = str(args.section_id)
    dataset_key = f"spamosaic_tonsil_{section_id}"
    dataset_dir = PROJECT_ROOT / args.dataset_dir if args.dataset_dir else GOAL_DIR / f"standardized_inputs/{dataset_key}_subset"
    output_dir = PROJECT_ROOT / args.output_dir if args.output_dir else GOAL_DIR / f"{dataset_key}_restricted_downstream_validation"
    predictions = _prediction_paths(dataset_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    cell_meta = cells.set_index("cell_id")
    expr = expression.set_index("cell_id")
    library_size = expr.sum(axis=1)

    rows = []
    for method, prediction_path in predictions.items():
        pred = pd.read_csv(prediction_path)
        table = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
        for lr in lr_pairs.itertuples(index=False):
            canonical = f"{lr.ligand}-{lr.receptor}"
            rule = TARGET_RULES.get(canonical)
            if rule is None:
                continue
            result = _test_lr(
                table[table["lr_id"].astype(str).eq(str(lr.lr_id))].copy(),
                cell_meta,
                expr,
                library_size,
                rule,
                method=method,
                lr_id=str(lr.lr_id),
                canonical_pair=canonical,
                false_positive_high_expression=bool(lr.false_positive_high_expression),
                n_permutations=int(args.n_permutations),
                rng=np.random.default_rng(int(args.random_seed) + len(rows)),
            )
            rows.append(result)

    result = pd.DataFrame(rows)
    tested = result["status"].eq("tested")
    result["bh_fdr"] = np.nan
    if tested.any():
        result.loc[tested, "bh_fdr"] = bh_adjust(result.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
    result.to_csv(output_dir / f"{dataset_key}_restricted_downstream_correlation.csv", index=False)
    method_summary = _method_summary(result)
    method_summary.to_csv(output_dir / f"{dataset_key}_restricted_downstream_method_summary.csv", index=False)
    claims = _claims(method_summary)
    claims.to_csv(output_dir / f"{dataset_key}_restricted_downstream_claims.csv", index=False)
    manifest = {
        "dataset": f"{dataset_key}_subset",
        "dataset_dir": _rel(dataset_dir),
        "output_dir": _rel(output_dir),
        "methods": sorted(predictions),
        "n_permutations": int(args.n_permutations),
        "validation_mode": "receiver_restricted_residual_target_signature",
        "claim_boundary": (
            "This is a stricter proxy validation that restricts receiver cells/niches and residualizes target "
            "signatures against library size, receptor abundance, and receiver cell-type indicators. It is still "
            "observational association, not perturbation evidence."
        ),
    }
    (output_dir / f"{dataset_key}_restricted_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(method_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _prediction_paths(dataset_key: str) -> dict[str, Path]:
    return {
        "spatialself_wavelet_edge_prior": GOAL_DIR
        / f"{dataset_key}_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
        "commot": GOAL_DIR / f"{dataset_key}_commot/replicate_00__commot_predictions.csv",
        "cellchat": GOAL_DIR / f"{dataset_key}_cellchat/replicate_00__cellchat_predictions.csv",
        "cellnest": GOAL_DIR / f"{dataset_key}_cellnest/replicate_00__cellnest_predictions.csv",
    }


def _test_lr(
    group: pd.DataFrame,
    cell_meta: pd.DataFrame,
    expr: pd.DataFrame,
    library_size: pd.Series,
    rule: dict[str, list[str]],
    *,
    method: str,
    lr_id: str,
    canonical_pair: str,
    false_positive_high_expression: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    receiver_meta = cell_meta.loc[group["receiver_id"]]
    mask = receiver_meta["cell_type"].isin(rule["receiver_cell_types"]).to_numpy()
    mask &= receiver_meta["niche"].isin(rule["receiver_niches"]).to_numpy()
    restricted = group.loc[mask].copy()
    row = {
        "method": method,
        "lr_id": lr_id,
        "canonical_pair": canonical_pair,
        "false_positive_high_expression": false_positive_high_expression,
        "n_edges": int(len(group)),
        "n_restricted_edges": int(len(restricted)),
        "n_receiver_cells": int(restricted["receiver_id"].nunique()) if len(restricted) else 0,
        "n_targets": 0,
        "spearman_r": np.nan,
        "empirical_p_value": np.nan,
        "status": "not_tested",
    }
    targets = [gene for gene in rule["targets"] if gene in expr.columns]
    row["n_targets"] = int(len(targets))
    if len(targets) < 3:
        row["status"] = "too_few_targets"
        return row
    if restricted.empty or restricted["receiver_id"].nunique() < 8:
        row["status"] = "too_few_receiver_cells"
        return row

    comm = restricted.groupby("receiver_id", sort=False)["score"].mean()
    common = expr.index.intersection(comm.index)
    if len(common) < 8:
        row["status"] = "too_few_receiver_cells"
        return row
    y = np.log1p(expr.loc[common, targets].to_numpy(dtype=float)).mean(axis=1)
    x = comm.loc[common].to_numpy(dtype=float)
    covariates = _covariate_matrix(common, expr, cell_meta, library_size, rule["covariates"])
    y_resid = _residualize(y, covariates)
    if np.nanstd(x) <= 1e-12 or np.nanstd(y_resid) <= 1e-12:
        row["status"] = "constant_vector"
        return row
    corr, _ = spearmanr(x, y_resid)
    exceed = 0
    for _ in range(int(n_permutations)):
        null_y = rng.permutation(y_resid)
        null_corr, _ = spearmanr(x, null_y)
        if abs(float(null_corr)) >= abs(float(corr)):
            exceed += 1
    row.update(
        {
            "spearman_r": float(corr),
            "empirical_p_value": float((exceed + 1.0) / (int(n_permutations) + 1.0)),
            "status": "tested",
        }
    )
    return row


def _covariate_matrix(
    cells: pd.Index,
    expr: pd.DataFrame,
    cell_meta: pd.DataFrame,
    library_size: pd.Series,
    covariates: list[str],
) -> np.ndarray:
    cols = []
    if "library_size" in covariates:
        cols.append(np.log1p(library_size.loc[cells].to_numpy(dtype=float)))
    for covariate in covariates:
        if covariate in {"library_size", "receiver_type"}:
            continue
        if covariate in expr.columns:
            cols.append(np.log1p(expr.loc[cells, covariate].to_numpy(dtype=float)))
    if "receiver_type" in covariates:
        dummies = pd.get_dummies(cell_meta.loc[cells, "cell_type"], drop_first=True)
        for column in dummies.columns:
            cols.append(dummies[column].to_numpy(dtype=float))
    if not cols:
        return np.ones((len(cells), 1), dtype=float)
    return np.column_stack(cols)


def _residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y), dtype=float), covariates])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _method_summary(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in result.groupby("method", sort=True):
        tested = group[group["status"].eq("tested")]
        positive = tested[~tested["false_positive_high_expression"].astype(bool)]
        negative = tested[tested["false_positive_high_expression"].astype(bool)]
        rows.append(
            {
                "method": method,
                "n_tested": int(tested["lr_id"].nunique()),
                "best_positive_lr": _best_lr(positive),
                "best_positive_spearman": _best_value(positive, "spearman_r"),
                "best_positive_bh_fdr": _best_value(positive, "bh_fdr"),
                "best_negative_lr": _best_lr(negative),
                "best_negative_spearman": _best_value(negative, "spearman_r"),
                "best_negative_bh_fdr": _best_value(negative, "bh_fdr"),
                "n_positive_fdr_0_1": int((pd.to_numeric(positive.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_negative_fdr_0_1": int((pd.to_numeric(negative.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        positive = float(getattr(row, "best_positive_spearman", 0) or 0) > 0 and float(
            getattr(row, "best_positive_bh_fdr", 1) or 1
        ) <= 0.1
        negative = int(getattr(row, "n_negative_fdr_0_1", 0) or 0) > 0
        if positive and not negative:
            status = "restricted_functional_support_without_negative_fdr_signal"
        elif positive and negative:
            status = "mixed_restricted_functional_support_negative_controls_also_correlate"
        else:
            status = "no_restricted_positive_functional_support"
        rows.append(
            {
                "method": row.method,
                "claim_status": status,
                "allowed_claim": (
                    "Receiver-restricted residual target association supports selected positive LR without negative-control leakage."
                    if status == "restricted_functional_support_without_negative_fdr_signal"
                    else "Do not claim strict functional validation under this restricted audit."
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_lr(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0]
    return str(row["canonical_pair"])


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    row = frame.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0]
    return float(row[column])


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
