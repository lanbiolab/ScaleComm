#!/usr/bin/env python
"""Pathway-specific downstream validation for Phase 7 Goal 5.

This audit asks a stricter functional question than the broad receiver-marker
proxy checks: for a pre-specified positive LR axis, does that LR's incoming
score correlate with its own receiver target program more strongly than
high-expression negative-control LR scores do in the same receiver context?
"""

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

from src.spitialself.fdr import bh_adjust


GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "spamosaic_tonsil_A1": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/spamosaic_tonsil_A1_subset",
        "output_id": "spamosaic_tonsil_A1_pathway_specific_downstream",
        "prediction_paths": {
            "spatialself_wavelet_edge_prior": GOAL_DIR
            / "spamosaic_tonsil_A1_spatialself/spatialself_wavelet_edge_prior/"
            "replicate_00__spatialself_wavelet_predictions.csv",
            "commot": GOAL_DIR / "spamosaic_tonsil_A1_commot/replicate_00__commot_predictions.csv",
            "cellchat": GOAL_DIR / "spamosaic_tonsil_A1_cellchat/replicate_00__cellchat_predictions.csv",
            "cellnest": GOAL_DIR / "spamosaic_tonsil_A1_cellnest/replicate_00__cellnest_predictions.csv",
        },
        "targets": [
            {
                "target_pair": "IL21-IL21R",
                "target_genes": ["CD40", "TNFRSF13C", "CR2", "PRDM1", "IRF4", "BCL6", "AICDA", "JCHAIN", "XBP1", "MZB1"],
                "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "IL21R", "receiver_type"],
            },
            {
                "target_pair": "CD40LG-CD40",
                "target_genes": ["TNFRSF13C", "CR2", "MS4A1", "NFKBIA", "TNFAIP3", "IRF4", "AICDA", "BCL6", "CD83", "CD86"],
                "receiver_cell_types": ["B_cell_proxy", "FDC_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "CD40", "receiver_type"],
            },
            {
                "target_pair": "CXCL13-CXCR5",
                "target_genes": ["CR2", "CD40", "TNFRSF13C", "AICDA", "BCL6", "PAX5", "CD79A", "CD79B", "BANK1"],
                "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "CXCR5", "receiver_type"],
            },
        ],
    },
    "spamosaic_tonsil_D1": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/spamosaic_tonsil_D1_subset",
        "output_id": "spamosaic_tonsil_D1_pathway_specific_downstream",
        "prediction_paths": {
            "spatialself_wavelet_edge_prior": GOAL_DIR
            / "spamosaic_tonsil_D1_spatialself/spatialself_wavelet_edge_prior/"
            "replicate_00__spatialself_wavelet_predictions.csv",
            "commot": GOAL_DIR / "spamosaic_tonsil_D1_commot/replicate_00__commot_predictions.csv",
            "cellchat": GOAL_DIR / "spamosaic_tonsil_D1_cellchat/replicate_00__cellchat_predictions.csv",
            "cellnest": GOAL_DIR / "spamosaic_tonsil_D1_cellnest/replicate_00__cellnest_predictions.csv",
        },
        "targets": [
            {
                "target_pair": "IL21-IL21R",
                "target_genes": ["CD40", "TNFRSF13C", "CR2", "PRDM1", "IRF4", "BCL6", "AICDA", "JCHAIN", "XBP1", "MZB1"],
                "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "IL21R", "receiver_type"],
            },
            {
                "target_pair": "CD40LG-CD40",
                "target_genes": ["TNFRSF13C", "CR2", "MS4A1", "NFKBIA", "TNFAIP3", "IRF4", "AICDA", "BCL6", "CD83", "CD86"],
                "receiver_cell_types": ["B_cell_proxy", "FDC_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "CD40", "receiver_type"],
            },
            {
                "target_pair": "CXCL13-CXCR5",
                "target_genes": ["CR2", "CD40", "TNFRSF13C", "AICDA", "BCL6", "PAX5", "CD79A", "CD79B", "BANK1"],
                "receiver_cell_types": ["B_cell_proxy", "Tfh_proxy"],
                "receiver_niches": ["germinal_center_proxy"],
                "covariates": ["library_size", "CXCR5", "receiver_type"],
            },
        ],
    },
    "visium_hd_crc_p2crc": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset",
        "output_id": "visium_hd_crc_p2crc_pathway_specific_downstream",
        "prediction_paths": {
            "spatialself_calibrated": GOAL_DIR
            / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
            "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
            "spatialself_default": GOAL_DIR
            / "visium_hd_crc_p2crc_baseline_panel/spatialself_wavelet_edge_prior/"
            "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
            "commot": GOAL_DIR
            / "visium_hd_crc_p2crc_baseline_panel/commot/"
            "visium_hd_crc_p2crc_subset__commot_predictions.csv",
            "cellchat": GOAL_DIR
            / "visium_hd_crc_p2crc_baseline_panel/cellchat/"
            "visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
            "cellnest": GOAL_DIR
            / "visium_hd_crc_p2crc_baseline_panel/cellnest/"
            "visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
        },
        "targets": [
            {
                "target_pair": "TGFB1-TGFBR2",
                "target_genes": ["JUN", "FOS", "SERPINE1", "FN1", "MMP9"],
                "receiver_cell_types": ["stromal_proxy", "myeloid_proxy", "tumor_proxy"],
                "receiver_niches": ["tumor_stromal_boundary", "tumor_myeloid_boundary", "stromal_immune_proxy"],
                "covariates": ["library_size", "TGFBR2", "receiver_type"],
            },
            {
                "target_pair": "VEGFA-KDR",
                "target_genes": ["PECAM1", "VWF", "ENG", "FLT1", "ESAM"],
                "receiver_cell_types": ["vascular_proxy", "stromal_proxy"],
                "receiver_niches": ["vascular_stromal_boundary", "tumor_stromal_boundary"],
                "covariates": ["library_size", "KDR", "receiver_type"],
            },
            {
                "target_pair": "CD274-PDCD1",
                "target_genes": ["TOX", "LAG3", "HAVCR2", "CD3D", "CD8A"],
                "receiver_cell_types": ["T_cell_proxy", "immune_proxy"],
                "receiver_niches": ["tumor_immune_boundary", "immune_proxy"],
                "covariates": ["library_size", "PDCD1", "receiver_type"],
            },
        ],
    },
    "post_xenium_colon": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset",
        "output_id": "post_xenium_colon_pathway_specific_downstream",
        "prediction_paths": {
            "spatialself_calibrated": GOAL_DIR
            / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
            "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
            "spatialself_default": GOAL_DIR
            / "post_xenium_human_colon_cancer_baseline_panel/spatialself_wavelet_edge_prior/"
            "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
            "commot": GOAL_DIR
            / "post_xenium_human_colon_cancer_baseline_panel/commot/"
            "post_xenium_human_colon_cancer_subset__commot_predictions.csv",
        },
        "targets": [
            {
                "target_pair": "TGFB1-TGFBR2",
                "target_genes": ["COL1A2", "DCN", "LUM", "ACTA2"],
                "receiver_cell_types": ["stromal_proxy", "myeloid_proxy"],
                "receiver_niches": ["stromal_immune_proxy", "myeloid_stromal_proxy"],
                "covariates": ["library_size", "TGFBR2", "receiver_type"],
            },
            {
                "target_pair": "VEGFA-KDR",
                "target_genes": ["PECAM1", "VWF", "ACTA2", "COL1A2"],
                "receiver_cell_types": ["vascular_proxy", "stromal_proxy"],
                "receiver_niches": ["vascular_stromal_proxy", "stromal_immune_proxy"],
                "covariates": ["library_size", "KDR", "receiver_type"],
            },
        ],
    },
    "native_human_melanoma_multiome": {
        "dataset_dir": GOAL_DIR / "standardized_inputs/native_human_melanoma_multiome_subset",
        "output_id": "native_human_melanoma_multiome_pathway_specific_downstream",
        "prediction_paths": {
            "spatialself_wavelet_edge_prior": GOAL_DIR
            / "native_human_melanoma_multiome/baseline_panel/spatialself_wavelet_edge_prior/"
            "native_human_melanoma_multiome_subset__spatialself_wavelet_predictions.csv",
            "commot": GOAL_DIR
            / "native_human_melanoma_multiome/baseline_panel/commot/"
            "native_human_melanoma_multiome_subset__commot_predictions.csv",
            "cellchat": GOAL_DIR
            / "native_human_melanoma_multiome/baseline_panel/cellchat/"
            "native_human_melanoma_multiome_subset__cellchat_predictions.csv",
            "cellnest": GOAL_DIR
            / "native_human_melanoma_multiome/baseline_panel/cellnest/"
            "native_human_melanoma_multiome_subset__cellnest_predictions.csv",
        },
        "targets": [
            {
                "target_pair": "CXCL9-CXCR3",
                "target_genes": ["IFNG", "STAT1", "IRF1", "GZMB", "TOX"],
                "receiver_cell_types": ["T_CD8", "T_reg", "T_CD4"],
                "receiver_niches": ["tumor_immune_boundary", "tme_immune_context", "tumor_myeloid_boundary"],
                "covariates": ["library_size", "CXCR3", "receiver_type"],
            },
            {
                "target_pair": "CXCL10-CXCR3",
                "target_genes": ["IFNG", "STAT1", "IRF1", "GZMB", "TOX"],
                "receiver_cell_types": ["T_CD8", "T_reg", "T_CD4"],
                "receiver_niches": ["tumor_immune_boundary", "tme_immune_context", "tumor_myeloid_boundary"],
                "covariates": ["library_size", "CXCR3", "receiver_type"],
            },
            {
                "target_pair": "CXCL11-CXCR3",
                "target_genes": ["IFNG", "STAT1", "IRF1", "GZMB", "TOX"],
                "receiver_cell_types": ["T_CD8", "T_reg", "T_CD4"],
                "receiver_niches": ["tumor_immune_boundary", "tme_immune_context", "tumor_myeloid_boundary"],
                "covariates": ["library_size", "CXCR3", "receiver_type"],
            },
            {
                "target_pair": "PDCD1-CD274",
                "target_genes": ["TOX", "HAVCR2", "LAG3", "NFKBIA", "TNFAIP3"],
                "receiver_cell_types": ["tumor", "myeloid"],
                "receiver_niches": ["tumor_immune_boundary", "tumor_myeloid_boundary"],
                "covariates": ["library_size", "CD274", "receiver_type"],
            },
            {
                "target_pair": "SPP1-CD44",
                "target_genes": ["FOS", "JUN", "NFKBIA", "TNFAIP3", "HLA-DRA"],
                "receiver_cell_types": ["myeloid", "tumor"],
                "receiver_niches": ["tumor_myeloid_boundary", "myeloid_tumor_context"],
                "covariates": ["library_size", "CD44", "receiver_type"],
            },
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=202616)
    parser.add_argument("--min-receivers", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-id", default=None)
    parser.add_argument(
        "--residualize-communication",
        action="store_true",
        help="Residualize incoming communication scores with the same receiver covariates used for target signatures.",
    )
    parser.add_argument(
        "--clean-target-signatures",
        action="store_true",
        help=(
            "Remove ligand/receptor genes, high-expression negative-control genes, common lineage markers, "
            "and housekeeping genes from receiver target signatures before correlation."
        ),
    )
    args = parser.parse_args()

    config = DATASET_CONFIGS[str(args.dataset)]
    dataset_dir = PROJECT_ROOT / args.dataset_dir if args.dataset_dir else Path(config["dataset_dir"])
    output_dir = GOAL_DIR / str(args.output_id or config["output_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    target_rules = config["targets"]
    if args.clean_target_signatures:
        target_rules = _clean_target_rules(config["targets"], lr_pairs)
    cell_meta = cells.set_index("cell_id")
    expr = expression.set_index("cell_id")
    library_size = expr.sum(axis=1)
    lr_meta = _lr_metadata(lr_pairs)

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for method, prediction_path in config["prediction_paths"].items():
        prediction_path = Path(prediction_path)
        if not prediction_path.exists():
            status_rows.append({"method": method, "status": "missing_prediction", "prediction_path": _rel(prediction_path)})
            continue
        predictions = pd.read_csv(prediction_path)
        merged = candidates.merge(
            predictions[["candidate_id", "score"]],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
        for target_rule in target_rules:
            target_lr = _find_lr(lr_pairs, str(target_rule["target_pair"]))
            if not target_lr:
                continue
            for scored_lr in lr_pairs.itertuples(index=False):
                row = _test_target_rule(
                    merged[merged["lr_id"].astype(str).eq(str(scored_lr.lr_id))].copy(),
                    cell_meta,
                    expr,
                    library_size,
                    target_rule,
                    target_lr_id=str(target_lr["lr_id"]),
                    scored_lr_id=str(scored_lr.lr_id),
                    scored_pair=f"{scored_lr.ligand}-{scored_lr.receptor}",
                    scored_control_name=str(scored_lr.control_name),
                    scored_control_class=str(scored_lr.control_class),
                    scored_false_positive=bool(scored_lr.false_positive_high_expression),
                    method=str(method),
                    n_permutations=int(args.n_permutations),
                    min_receivers=int(args.min_receivers),
                    residualize_communication=bool(args.residualize_communication),
                    rng=np.random.default_rng(int(args.random_seed) + len(rows)),
                )
                rows.append(row)
        status_rows.append(
            {
                "method": method,
                "status": "completed",
                "prediction_path": _rel(prediction_path),
                "n_rows": len([r for r in rows if r["method"] == method]),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        tested = result["status"].astype(str).eq("tested")
        result["bh_fdr"] = np.nan
        if tested.any():
            result.loc[tested, "bh_fdr"] = bh_adjust(
                result.loc[tested, "empirical_p_value"].to_numpy(dtype=float)
            )
        result["scored_true_lr"] = result["scored_lr_id"].map(
            lambda lr_id: bool(lr_meta.get(str(lr_id), {}).get("true_lr", False))
        )
    summary = _summarize(result, alpha=float(args.alpha))
    claims = _claims(summary)
    status = pd.DataFrame(status_rows)

    result.to_csv(output_dir / f"{args.dataset}_pathway_specific_downstream_correlation.csv", index=False)
    summary.to_csv(output_dir / f"{args.dataset}_pathway_specific_downstream_summary.csv", index=False)
    claims.to_csv(output_dir / f"{args.dataset}_pathway_specific_downstream_claims.csv", index=False)
    status.to_csv(output_dir / f"{args.dataset}_pathway_specific_downstream_status.csv", index=False)
    manifest = {
        "dataset": str(args.dataset),
        "dataset_dir": _rel(dataset_dir),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "alpha": float(args.alpha),
        "min_receivers": int(args.min_receivers),
        "validation_mode": "pathway_specific_target_response_competition",
        "residualize_communication": bool(args.residualize_communication),
        "clean_target_signatures": bool(args.clean_target_signatures),
        "target_rules": target_rules,
        "claim_boundary": (
            "This is an observational functional specificity audit. It tests whether each pre-specified "
            "positive LR score aligns with its own receiver target program better than high-expression "
            "negative-control LR scores in the same receiver context. It does not replace strict all-null "
            "model-rerun FDR or perturbation evidence."
        ),
    }
    (output_dir / f"{args.dataset}_pathway_specific_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _clean_target_rules(target_rules: list[dict[str, Any]], lr_pairs: pd.DataFrame) -> list[dict[str, Any]]:
    lr_genes = set(lr_pairs["ligand"].astype(str)) | set(lr_pairs["receptor"].astype(str))
    negative_genes = set(
        lr_pairs.loc[
            lr_pairs.get("false_positive_high_expression", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool),
            ["ligand", "receptor"],
        ]
        .astype(str)
        .to_numpy()
        .ravel()
        .tolist()
    )
    banned = lr_genes | negative_genes | _lineage_marker_genes() | _housekeeping_genes()
    cleaned = []
    for rule in target_rules:
        rule_copy = dict(rule)
        ligand, receptor = str(rule["target_pair"]).split("-", 1)
        local_banned = banned | {ligand, receptor}
        targets = [gene for gene in rule.get("target_genes", []) if str(gene) not in local_banned]
        rule_copy["target_genes"] = targets
        rule_copy["cleaned_target_signature"] = True
        rule_copy["removed_target_genes"] = sorted(set(rule.get("target_genes", [])) - set(targets))
        cleaned.append(rule_copy)
    return cleaned


def _lineage_marker_genes() -> set[str]:
    return {
        "MS4A1",
        "CD19",
        "CD79A",
        "CD79B",
        "BANK1",
        "PAX5",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD27",
        "CD28",
        "PTPRC",
        "LYZ",
        "CD14",
        "CD68",
        "C1QA",
        "C1QB",
        "EPCAM",
        "KRT8",
        "KRT19",
        "MUC1",
        "PECAM1",
        "VWF",
        "ENG",
        "ACTA2",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL6A1",
        "DCN",
        "LUM",
    }


def _housekeeping_genes() -> set[str]:
    return {
        "ACTB",
        "GAPDH",
        "B2M",
        "MALAT1",
        "RPLP0",
        "RPL13A",
        "RPS18",
        "RPS19",
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DPB1",
    }


def _test_target_rule(
    scored_edges: pd.DataFrame,
    cell_meta: pd.DataFrame,
    expr: pd.DataFrame,
    library_size: pd.Series,
    target_rule: dict[str, Any],
    *,
    target_lr_id: str,
    scored_lr_id: str,
    scored_pair: str,
    scored_control_name: str,
    scored_control_class: str,
    scored_false_positive: bool,
    method: str,
    n_permutations: int,
    min_receivers: int,
    residualize_communication: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "target_pair": str(target_rule["target_pair"]),
        "target_lr_id": target_lr_id,
        "scored_lr_id": scored_lr_id,
        "scored_pair": scored_pair,
        "scored_control_name": scored_control_name,
        "scored_control_class": scored_control_class,
        "scored_false_positive_high_expression": bool(scored_false_positive),
        "scored_is_target_lr": bool(scored_lr_id == target_lr_id),
        "n_edges": int(len(scored_edges)),
        "n_restricted_edges": 0,
        "n_receiver_cells": 0,
        "n_targets": 0,
        "target_genes_present": "",
        "spearman_r": np.nan,
        "empirical_p_value": np.nan,
        "status": "not_tested",
    }
    targets = [gene for gene in target_rule["target_genes"] if gene in expr.columns]
    row["n_targets"] = int(len(targets))
    row["target_genes_present"] = ",".join(targets)
    if len(targets) < 2:
        row["status"] = "too_few_targets"
        return row
    if scored_edges.empty:
        row["status"] = "no_edges"
        return row

    receiver_meta = cell_meta.loc[scored_edges["receiver_id"]]
    mask = receiver_meta["cell_type"].isin(target_rule["receiver_cell_types"]).to_numpy()
    mask &= receiver_meta["niche"].isin(target_rule["receiver_niches"]).to_numpy()
    restricted = scored_edges.loc[mask].copy()
    row["n_restricted_edges"] = int(len(restricted))
    row["n_receiver_cells"] = int(restricted["receiver_id"].nunique()) if len(restricted) else 0
    if restricted["receiver_id"].nunique() < min_receivers:
        row["status"] = "too_few_receiver_cells"
        return row

    incoming = restricted.groupby("receiver_id", sort=False)["score"].mean()
    common = expr.index.intersection(incoming.index)
    if len(common) < min_receivers:
        row["status"] = "too_few_receiver_cells"
        return row

    x = incoming.loc[common].to_numpy(dtype=float)
    y = np.log1p(expr.loc[common, targets].to_numpy(dtype=float)).mean(axis=1)
    cov = _covariate_matrix(common, expr, cell_meta, library_size, target_rule.get("covariates", []))
    x_resid = _residualize(x, cov) if residualize_communication else x
    y_resid = _residualize(y, cov)
    if np.nanstd(x_resid) <= 1e-12 or np.nanstd(y_resid) <= 1e-12:
        row["status"] = "constant_vector"
        return row

    rho, _ = spearmanr(x_resid, y_resid)
    if not np.isfinite(rho):
        row["status"] = "nonfinite_correlation"
        return row
    exceed = 0
    receiver_types = cell_meta.loc[common, "cell_type"].astype(str).to_numpy()
    for _ in range(int(n_permutations)):
        null_y = _permute_within_labels(y_resid, receiver_types, rng)
        null_rho, _ = spearmanr(x_resid, null_y)
        if abs(float(null_rho)) >= abs(float(rho)):
            exceed += 1
    row.update(
        {
            "spearman_r": float(rho),
            "empirical_p_value": float((exceed + 1.0) / (int(n_permutations) + 1.0)),
            "status": "tested",
        }
    )
    return row


def _summarize(result: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    if result.empty:
        return pd.DataFrame()
    for (method, target_pair), group in result.groupby(["method", "target_pair"], sort=False):
        tested = group[group["status"].astype(str).eq("tested")].copy()
        target = tested[tested["scored_is_target_lr"].fillna(False).astype(bool)]
        negatives = tested[tested["scored_false_positive_high_expression"].fillna(False).astype(bool)]
        positives = tested[
            tested["scored_control_class"].astype(str).str.contains("positive|context|prior", case=False, regex=True)
            & ~tested["scored_false_positive_high_expression"].fillna(False).astype(bool)
        ]
        best_target = _best_row(target)
        best_negative = _best_row(negatives)
        rows.append(
            {
                "method": method,
                "target_pair": target_pair,
                "n_tested": int(len(tested)),
                "target_lr_spearman": best_target.get("spearman_r", np.nan),
                "target_lr_bh_fdr": best_target.get("bh_fdr", np.nan),
                "target_lr_status": best_target.get("status", ""),
                "best_negative_pair": best_negative.get("scored_pair", ""),
                "best_negative_spearman": best_negative.get("spearman_r", np.nan),
                "best_negative_bh_fdr": best_negative.get("bh_fdr", np.nan),
                "n_negative_fdr_alpha": int((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_positive_fdr_alpha": int((pd.to_numeric(positives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "target_minus_best_negative_r": _safe_float(best_target.get("spearman_r", np.nan))
                - _safe_float(best_negative.get("spearman_r", np.nan)),
                "pathway_specific_success": bool(
                    _safe_float(best_target.get("spearman_r", 0)) > 0
                    and _safe_float(best_target.get("bh_fdr", 1)) <= alpha
                    and int((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()) == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for method, group in summary.groupby("method", sort=False):
        successes = group[group["pathway_specific_success"].fillna(False).astype(bool)]
        if successes.empty:
            status = "no_clean_pathway_specific_success"
            best = group.sort_values(["target_lr_bh_fdr", "target_lr_spearman"], ascending=[True, False]).iloc[0]
        else:
            status = "pathway_specific_functional_success"
            best = successes.sort_values(["target_lr_bh_fdr", "target_lr_spearman"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "method": method,
                "claim_status": status,
                "n_pathway_specific_successes": int(len(successes)),
                "best_target_pair": best.get("target_pair", ""),
                "best_target_lr_spearman": best.get("target_lr_spearman", ""),
                "best_target_lr_bh_fdr": best.get("target_lr_bh_fdr", ""),
                "best_negative_pair": best.get("best_negative_pair", ""),
                "best_negative_bh_fdr": best.get("best_negative_bh_fdr", ""),
                "allowed_claim": (
                    "Pre-specified target-response specificity is supported without high-expression negative-control leakage."
                    if not successes.empty
                    else "No clean pathway-specific downstream claim under this target panel."
                ),
            }
        )
    return pd.DataFrame(rows)


def _covariate_matrix(
    cells: pd.Index,
    expr: pd.DataFrame,
    cell_meta: pd.DataFrame,
    library_size: pd.Series,
    covariates: list[str],
) -> np.ndarray:
    columns = []
    if "library_size" in covariates:
        columns.append(np.log1p(library_size.loc[cells].to_numpy(dtype=float)))
    for covariate in covariates:
        if covariate in {"library_size", "receiver_type"}:
            continue
        if covariate in expr.columns:
            columns.append(np.log1p(expr.loc[cells, covariate].to_numpy(dtype=float)))
    if "receiver_type" in covariates:
        dummies = pd.get_dummies(cell_meta.loc[cells, "cell_type"], drop_first=True)
        for column in dummies.columns:
            columns.append(dummies[column].to_numpy(dtype=float))
    if not columns:
        return np.ones((len(cells), 1), dtype=float)
    return np.column_stack(columns)


def _residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y), dtype=float), covariates])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _permute_within_labels(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.array(values, copy=True)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > 1:
            shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def _find_lr(lr_pairs: pd.DataFrame, canonical_pair: str) -> dict[str, Any]:
    canonical = lr_pairs["ligand"].astype(str) + "-" + lr_pairs["receptor"].astype(str)
    subset = lr_pairs[canonical.eq(canonical_pair)]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _lr_metadata(lr_pairs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row.lr_id): row._asdict() for row in lr_pairs.itertuples(index=False)}


def _best_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return (
        frame.assign(abs_r=frame["spearman_r"].abs())
        .sort_values(["bh_fdr", "spearman_r", "abs_r"], ascending=[True, False, False])
        .iloc[0]
        .to_dict()
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
