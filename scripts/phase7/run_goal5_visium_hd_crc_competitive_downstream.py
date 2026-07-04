#!/usr/bin/env python
"""Competitive downstream validation for Visium HD CRC Goal 5.

This differs from the earlier per-LR target proxy validation: negative control
LR pairs do not get their own marker signatures. Instead, every LR is tested
against the same positive response programs, so high-expression invalid pairs
must compete with biologically plausible LR pairs for the same receiver
functional readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset"

PREDICTION_PATHS = {
    "spatialself_calibrated": GOAL_DIR
    / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
    "spatialself_default": GOAL_DIR
    / "visium_hd_crc_p2crc_baseline_panel/spatialself_wavelet_edge_prior/"
    "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL_DIR / "visium_hd_crc_p2crc_baseline_panel/commot/visium_hd_crc_p2crc_subset__commot_predictions.csv",
    "cellchat": GOAL_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellchat/visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
    "cellnest": GOAL_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellnest/visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
}

PROGRAMS = [
    {
        "program_id": "tgfb_ecm_response",
        "expected_lr": "visium_hd_crc_lr_003_TGFB1_TGFBR2",
        "genes": ["SERPINE1", "COL1A1", "COL1A2", "ACTA2", "TAGLN", "FN1"],
        "receiver_cell_types": ["caf_proxy", "tumor_proxy", "epithelial_proxy"],
        "receiver_niches": ["tumor_stromal_boundary", "tumor_core"],
    },
    {
        "program_id": "vegfa_endothelial_response",
        "expected_lr": "visium_hd_crc_lr_002_VEGFA_KDR",
        "genes": ["FLT1", "PECAM1", "VWF", "ENG", "ESAM", "CDH5"],
        "receiver_cell_types": ["endothelial_proxy"],
        "receiver_niches": ["vascular_stromal_boundary", "tumor_stromal_boundary"],
    },
    {
        "program_id": "spp1_inflammatory_matrix_response",
        "expected_lr": "visium_hd_crc_lr_000_SPP1_CD44",
        "genes": ["MMP9", "ITGAV", "ITGB1", "FN1", "LGALS3", "CXCL8"],
        "receiver_cell_types": ["caf_proxy", "myeloid_proxy", "tumor_proxy", "T_cell_proxy"],
        "receiver_niches": ["tumor_immune_boundary", "tumor_stromal_boundary", "tumor_core"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202618)
    parser.add_argument("--min-receivers", type=int, default=40)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
            "visium_hd_crc_p2crc_competitive_downstream_200perm"
        ),
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")

    program_table = _program_table(expression)
    program_table.to_csv(output_dir / "visium_hd_crc_competitive_downstream_programs.csv", index=False)

    rows = []
    status_rows = []
    for method, prediction_path in PREDICTION_PATHS.items():
        if not prediction_path.exists():
            status_rows.append({"method": method, "status": "missing_prediction", "prediction_path": _rel(prediction_path)})
            continue
        predictions = pd.read_csv(prediction_path)
        rows.extend(
            _score_method(
                method=method,
                predictions=predictions,
                candidates=candidates,
                lr_pairs=lr_pairs,
                cells=cells,
                expression=expression,
                n_permutations=int(args.n_permutations),
                random_seed=int(args.random_seed),
                min_receivers=int(args.min_receivers),
            )
        )
        status_rows.append({"method": method, "status": "completed", "prediction_path": _rel(prediction_path)})

    correlation = pd.DataFrame(rows)
    if not correlation.empty:
        tested = correlation["status"].astype(str).eq("tested")
        correlation.loc[tested, "bh_fdr_within_program_method"] = correlation[tested].groupby(
            ["method", "program_id"], group_keys=False
        )["empirical_p_value"].transform(_bh_adjust)
        correlation.loc[~tested, "bh_fdr_within_program_method"] = np.nan
    summary = _summary(correlation)
    claims = _claims(summary)
    status = pd.DataFrame(status_rows)

    correlation.to_csv(output_dir / "visium_hd_crc_competitive_downstream_correlation.csv", index=False)
    summary.to_csv(output_dir / "visium_hd_crc_competitive_downstream_method_summary.csv", index=False)
    claims.to_csv(output_dir / "visium_hd_crc_competitive_downstream_claims.csv", index=False)
    status.to_csv(output_dir / "visium_hd_crc_competitive_downstream_status.csv", index=False)

    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "dataset_dir": _rel(dataset_dir),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "min_receivers": int(args.min_receivers),
        "programs": PROGRAMS,
        "claim_boundary": (
            "Competitive downstream support only: positive LR pairs and high-expression negative controls are tested "
            "against the same positive response programs. This avoids assigning marker-like target signatures to "
            "negative LR pairs, but remains observational and not perturbation evidence."
        ),
        "outputs": {
            "correlation": _rel(output_dir / "visium_hd_crc_competitive_downstream_correlation.csv"),
            "method_summary": _rel(output_dir / "visium_hd_crc_competitive_downstream_method_summary.csv"),
            "claims": _rel(output_dir / "visium_hd_crc_competitive_downstream_claims.csv"),
            "programs": _rel(output_dir / "visium_hd_crc_competitive_downstream_programs.csv"),
            "status": _rel(output_dir / "visium_hd_crc_competitive_downstream_status.csv"),
        },
    }
    (output_dir / "visium_hd_crc_competitive_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _program_table(expression: pd.DataFrame) -> pd.DataFrame:
    genes_present = set(expression.columns) - {"cell_id"}
    rows = []
    for program in PROGRAMS:
        present = [gene for gene in program["genes"] if gene in genes_present]
        rows.append(
            {
                "program_id": program["program_id"],
                "expected_lr": program["expected_lr"],
                "target_genes": ",".join(present),
                "n_targets_present": len(present),
                "receiver_cell_types": ",".join(program["receiver_cell_types"]),
                "receiver_niches": ",".join(program["receiver_niches"]),
            }
        )
    return pd.DataFrame(rows)


def _score_method(
    *,
    method: str,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    n_permutations: int,
    random_seed: int,
    min_receivers: int,
) -> list[dict]:
    merged = candidates.merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    incoming = merged.groupby(["lr_id", "receiver_id"], sort=False)["score"].mean().reset_index(name="incoming_score")
    expr = expression.merge(cells[["cell_id", "cell_type", "niche"]], on="cell_id", how="left")
    lr_meta = lr_pairs.set_index("lr_id").to_dict(orient="index")
    rng = np.random.default_rng(random_seed + abs(hash(method)) % 10000)
    rows = []
    for program in PROGRAMS:
        genes = [gene for gene in program["genes"] if gene in expr.columns]
        receiver_mask = expr["cell_type"].isin(program["receiver_cell_types"]) | expr["niche"].isin(program["receiver_niches"])
        receiver_expr = expr.loc[receiver_mask, ["cell_id", "cell_type", "niche", *genes]].copy()
        if len(genes) < 3 or len(receiver_expr) < min_receivers:
            rows.extend(_empty_program_rows(method, program, lr_pairs, "too_few_targets_or_receivers", len(receiver_expr), len(genes)))
            continue
        target_activity = np.log1p(receiver_expr[genes].astype(float)).mean(axis=1)
        receiver_expr["target_activity"] = target_activity
        for lr_id, lr_incoming in incoming.groupby("lr_id", sort=False):
            test = lr_incoming.merge(
                receiver_expr[["cell_id", "target_activity"]],
                left_on="receiver_id",
                right_on="cell_id",
                how="inner",
            )
            test = test.dropna(subset=["incoming_score", "target_activity"])
            if len(test) < min_receivers or test["incoming_score"].nunique() < 2 or test["target_activity"].nunique() < 2:
                rows.append(_row(method, program, lr_id, lr_meta, "too_few_receivers_or_constant_vector", len(test), len(genes)))
                continue
            rho, p_value = stats.spearmanr(test["incoming_score"], test["target_activity"])
            if not np.isfinite(rho):
                rows.append(_row(method, program, lr_id, lr_meta, "nonfinite_correlation", len(test), len(genes)))
                continue
            exceed = 0
            target = test["target_activity"].to_numpy(dtype=float)
            score = test["incoming_score"].to_numpy(dtype=float)
            for _ in range(n_permutations):
                null_target = rng.permutation(target)
                null_rho, _ = stats.spearmanr(score, null_target)
                if np.isfinite(null_rho) and null_rho >= rho:
                    exceed += 1
            empirical = (exceed + 1.0) / (n_permutations + 1.0)
            out = _row(method, program, lr_id, lr_meta, "tested", len(test), len(genes))
            out.update(
                {
                    "spearman_r": float(rho),
                    "spearman_p_value": float(p_value),
                    "empirical_p_value": float(empirical),
                    "is_expected_lr": str(lr_id) == str(program["expected_lr"]),
                }
            )
            rows.append(out)
    return rows


def _empty_program_rows(method: str, program: dict, lr_pairs: pd.DataFrame, status: str, n_receivers: int, n_targets: int) -> list[dict]:
    lr_meta = lr_pairs.set_index("lr_id").to_dict(orient="index")
    return [_row(method, program, lr_id, lr_meta, status, n_receivers, n_targets) for lr_id in lr_pairs["lr_id"]]


def _row(method: str, program: dict, lr_id: str, lr_meta: dict, status: str, n_receivers: int, n_targets: int) -> dict:
    meta = lr_meta.get(str(lr_id), {})
    return {
        "method": method,
        "program_id": program["program_id"],
        "expected_lr": program["expected_lr"],
        "lr_id": lr_id,
        "ligand": meta.get("ligand", ""),
        "receptor": meta.get("receptor", ""),
        "control_name": meta.get("control_name", ""),
        "control_class": meta.get("control_class", ""),
        "true_lr": bool(meta.get("true_lr", False)),
        "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
        "is_expected_lr": str(lr_id) == str(program["expected_lr"]),
        "n_receiver_cells": int(n_receivers),
        "n_targets_present": int(n_targets),
        "spearman_r": np.nan,
        "spearman_p_value": np.nan,
        "empirical_p_value": np.nan,
        "status": status,
    }


def _summary(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if correlation.empty:
        return pd.DataFrame(rows)
    tested = correlation[correlation["status"].astype(str).eq("tested")].copy()
    for (method, program_id), group in tested.groupby(["method", "program_id"], sort=True):
        expected = group[group["is_expected_lr"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        positives = group[group["true_lr"].fillna(False).astype(bool)]
        sig = group[(group["bh_fdr_within_program_method"] <= 0.1) & (group["spearman_r"] > 0)]
        rows.append(
            {
                "method": method,
                "program_id": program_id,
                "expected_lr": str(group["expected_lr"].iloc[0]),
                "expected_spearman": _best_value(expected, "spearman_r"),
                "expected_bh_fdr": _best_value(expected, "bh_fdr_within_program_method"),
                "expected_rank_by_spearman": _rank_of_expected(group),
                "best_positive_lr": _best_lr(positives),
                "best_positive_spearman": _best_value(positives, "spearman_r"),
                "best_negative_lr": _best_lr(negatives),
                "best_negative_spearman": _best_value(negatives, "spearman_r"),
                "n_significant_positive_lr": int((sig["true_lr"].fillna(False).astype(bool)).sum()),
                "n_significant_negative_lr": int((sig["false_positive_high_expression"].fillna(False).astype(bool)).sum()),
                "best_positive_minus_best_negative": _best_value(positives, "spearman_r") - _best_value(negatives, "spearman_r"),
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame(rows)
    for method, group in summary.groupby("method", sort=True):
        clean = (
            (group["n_significant_positive_lr"] > 0)
            & (group["n_significant_negative_lr"] == 0)
            & (group["best_positive_minus_best_negative"] > 0)
        )
        mixed = (group["n_significant_negative_lr"] > 0).any()
        if bool(clean.any()) and not bool(mixed):
            status = "competitive_downstream_support_without_negative_leakage"
        elif bool(clean.any()) and bool(mixed):
            status = "mixed_competitive_downstream_support_negative_leakage"
        else:
            status = "no_clean_competitive_downstream_support"
        rows.append(
            {
                "method": method,
                "claim_status": status,
                "n_clean_programs": int(clean.sum()),
                "n_programs": int(len(group)),
                "n_programs_with_negative_leakage": int((group["n_significant_negative_lr"] > 0).sum()),
                "best_clean_program": _best_clean_program(group, clean),
                "allowed_claim": (
                    "Competitive target-response validation tests all LR pairs against the same positive response "
                    "programs. It supports functional specificity only when positive LR scores correlate with a "
                    "program and high-expression negative controls do not."
                ),
            }
        )
    return pd.DataFrame(rows)


def _bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked, dtype=float)
    running = 1.0
    n = len(ranked)
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _rank_of_expected(group: pd.DataFrame) -> float:
    ranked = group.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    matches = ranked[ranked["is_expected_lr"].fillna(False).astype(bool)]
    if matches.empty:
        return float("nan")
    return float(matches.index[0] + 1)


def _best_lr(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values("spearman_r", ascending=False).iloc[0]["lr_id"])


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return float("nan")
    idx = pd.to_numeric(frame["spearman_r"], errors="coerce").idxmax()
    return float(frame.loc[idx, column])


def _best_clean_program(group: pd.DataFrame, clean: pd.Series) -> str:
    clean_group = group.loc[clean]
    if clean_group.empty:
        return ""
    return str(clean_group.sort_values("best_positive_minus_best_negative", ascending=False).iloc[0]["program_id"])


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
