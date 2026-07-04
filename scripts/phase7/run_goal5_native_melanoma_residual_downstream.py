#!/usr/bin/env python
"""Receiver-residual downstream validation for native HumanMelanomaMultiome."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/native_human_melanoma_multiome_subset"
BASELINE_DIR = GOAL5_DIR / "native_human_melanoma_multiome/baseline_panel"
OUTPUT_DIR = GOAL5_DIR / "native_human_melanoma_multiome/residual_downstream_200perm"
METHODS = ("spatialself_wavelet_edge_prior", "commot", "cellchat", "cellnest")

TARGET_PROGRAMS = [
    {
        "program": "cxcr3_interferon_t_cell_response",
        "target_genes": ("CXCR3", "STAT1", "IRF1", "IFNG", "GZMB", "LAG3", "HAVCR2", "TOX"),
        "positive_pairs": {"CXCL9-CXCR3", "CXCL10-CXCR3", "CXCL11-CXCR3"},
    },
    {
        "program": "spp1_cd44_matrix_inflammatory_response",
        "target_genes": ("CD44", "FOS", "JUN", "NFKBIA", "TNFAIP3", "ITGB1"),
        "positive_pairs": {"SPP1-CD44", "MIF-CD74"},
    },
    {
        "program": "checkpoint_ifn_exhaustion_response",
        "target_genes": ("CD274", "PDCD1", "STAT1", "IRF1", "LAG3", "HAVCR2", "TOX", "GZMB"),
        "positive_pairs": {"PDCD1-CD274"},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=202612)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    baseline_dir = PROJECT_ROOT / args.baseline_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv").set_index("cell_id")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")

    rows = []
    for method in METHODS:
        prediction_path = _prediction_path(baseline_dir, dataset_dir, method)
        if not prediction_path.exists():
            rows.append({"method": method, "status": "missing_prediction"})
            continue
        predictions = pd.read_csv(prediction_path)
        merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        rows.extend(
            _method_rows(
                method,
                merged,
                cells,
                expression,
                lr_pairs,
                n_permutations=int(args.n_permutations),
                seed=int(args.seed),
            )
        )

    result = pd.DataFrame(rows)
    tested = result["status"].astype(str).eq("tested") if "status" in result else pd.Series(dtype=bool)
    if tested.any():
        result.loc[tested, "bh_fdr"] = _bh(result.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
    summary = _summary(result)
    claims = _claims(summary)

    result.to_csv(output_dir / "native_melanoma_residual_downstream_correlation.csv", index=False)
    summary.to_csv(output_dir / "native_melanoma_residual_downstream_method_summary.csv", index=False)
    claims.to_csv(output_dir / "native_melanoma_residual_downstream_claims.csv", index=False)
    manifest = {
        "dataset": "native_human_melanoma_multiome_subset",
        "dataset_dir": _rel(dataset_dir),
        "baseline_dir": _rel(baseline_dir),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "covariates": [
            "receiver_cell_type",
            "receiver_niche",
            "log_library_size",
            "mean_receiver_receptor_expression",
        ],
        "claim_boundary": (
            "Residual downstream association controls receiver cell type, niche, library size, and receptor abundance. "
            "This reduces marker/cell-state confounding but is still observational, not perturbation evidence."
        ),
    }
    (output_dir / "native_melanoma_residual_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(claims.to_string(index=False))


def _method_rows(
    method: str,
    merged: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cell_meta = cells.set_index("cell_id")
    lr_meta = lr_pairs.set_index("lr_id").to_dict("index")
    rows = []
    for program in TARGET_PROGRAMS:
        targets = [gene for gene in program["target_genes"] if gene in expression.columns]
        for lr_id, group in merged.groupby("lr_id", sort=False):
            meta = lr_meta[str(lr_id)]
            row = {
                "method": method,
                "lr_id": lr_id,
                "control_name": meta.get("control_name", ""),
                "control_class": meta.get("control_class", ""),
                "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
                "true_lr": bool(meta.get("true_lr", False)),
                "program": program["program"],
                "expected_program": meta.get("control_name", "") in program["positive_pairs"],
                "n_targets_present": int(len(targets)),
                "n_receiver_cells": 0,
                "residual_spearman_r": np.nan,
                "empirical_p_value": np.nan,
                "status": "not_tested",
            }
            if len(targets) < 3:
                row["status"] = "too_few_targets"
                rows.append(row)
                continue
            receiver = (
                group.groupby("receiver_id", sort=False)
                .agg(
                    communication=("score", "mean"),
                    receptor_expr=("receptor_expr", "mean"),
                    receiver_cell_type=("receiver_cell_type", "first"),
                    receiver_niche=("receiver_niche", "first"),
                )
            )
            common = expression.index.intersection(receiver.index)
            if len(common) < 12:
                row["status"] = "too_few_receiver_cells"
                rows.append(row)
                continue
            receiver = receiver.loc[common].copy()
            signature = np.log1p(expression.loc[common, targets].to_numpy(dtype=float)).mean(axis=1)
            library = np.log1p(expression.loc[common].sum(axis=1).to_numpy(dtype=float))
            cov = _covariates(receiver, library)
            comm_resid = _residualize(receiver["communication"].to_numpy(dtype=float), cov)
            sig_resid = _residualize(signature, cov)
            if np.nanstd(comm_resid) <= 1e-12 or np.nanstd(sig_resid) <= 1e-12:
                row["status"] = "constant_residual"
                rows.append(row)
                continue
            corr, _ = spearmanr(comm_resid, sig_resid)
            exceed = 0
            strata = receiver["receiver_cell_type"].astype(str).to_numpy()
            for _ in range(n_permutations):
                null_sig = _permute_within_strata(sig_resid, strata, rng)
                null_corr, _ = spearmanr(comm_resid, null_sig)
                if abs(float(null_corr)) >= abs(float(corr)):
                    exceed += 1
            row.update(
                {
                    "n_receiver_cells": int(len(common)),
                    "residual_spearman_r": float(corr),
                    "empirical_p_value": float((exceed + 1.0) / (n_permutations + 1.0)),
                    "status": "tested",
                }
            )
            rows.append(row)
    return rows


def _covariates(receiver: pd.DataFrame, library: np.ndarray) -> np.ndarray:
    numeric = pd.DataFrame(
        {
            "log_library_size": library,
            "receptor_expr": np.log1p(receiver["receptor_expr"].to_numpy(dtype=float)),
        },
        index=receiver.index,
    )
    categorical = pd.get_dummies(
        receiver[["receiver_cell_type", "receiver_niche"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    x = pd.concat([numeric, categorical], axis=1).to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.column_stack([np.ones(len(x)), x])
    return x


def _residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    beta, *_ = np.linalg.lstsq(covariates, values, rcond=None)
    return values - covariates @ beta


def _permute_within_strata(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _summary(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in result.groupby("method", sort=False):
        tested = group[group["status"].astype(str).eq("tested")]
        expected = tested[tested["expected_program"].fillna(False).astype(bool)]
        unexpected = tested[~tested["expected_program"].fillna(False).astype(bool)]
        negatives = tested[tested["false_positive_high_expression"].fillna(False).astype(bool)]
        best_expected = _best(expected)
        best_negative = _best(negatives)
        rows.append(
            {
                "method": method,
                "status": "tested" if not tested.empty else "not_tested",
                "best_expected_lr": best_expected.get("control_name", ""),
                "best_expected_program": best_expected.get("program", ""),
                "best_expected_residual_spearman": best_expected.get("residual_spearman_r", ""),
                "best_expected_bh_fdr": best_expected.get("bh_fdr", ""),
                "n_expected_fdr_0_1": int((pd.to_numeric(expected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_unexpected_fdr_0_1": int((pd.to_numeric(unexpected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_negative_fdr_0_1": int((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "best_negative_lr": best_negative.get("control_name", ""),
                "best_negative_program": best_negative.get("program", ""),
                "best_negative_residual_spearman": best_negative.get("residual_spearman_r", ""),
                "best_negative_bh_fdr": best_negative.get("bh_fdr", ""),
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        positive = float(getattr(row, "best_expected_residual_spearman", 0) or 0) > 0 and float(
            getattr(row, "best_expected_bh_fdr", 1) or 1
        ) <= 0.1
        negative = int(getattr(row, "n_negative_fdr_0_1", 0) or 0) > 0
        unexpected = int(getattr(row, "n_unexpected_fdr_0_1", 0) or 0) > 4
        if positive and not negative and not unexpected:
            status = "residual_functional_support_without_negative_leakage"
        elif positive and negative:
            status = "mixed_residual_support_negative_leakage"
        elif positive:
            status = "mixed_residual_support_unexpected_program_leakage"
        else:
            status = "no_clean_residual_functional_support"
        rows.append({"method": row.method, "claim_status": status})
    return pd.DataFrame(rows)


def _best(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.sort_values(["bh_fdr", "residual_spearman_r"], ascending=[True, False]).iloc[0].to_dict()


def _prediction_path(baseline_dir: Path, dataset_dir: Path, method: str) -> Path:
    return baseline_dir / method / f"{dataset_dir.name}__{method}_predictions.csv"


def _bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
