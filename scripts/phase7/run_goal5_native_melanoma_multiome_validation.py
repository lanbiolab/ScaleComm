#!/usr/bin/env python
"""Run native HumanMelanomaMultiome validation for Phase 7 Goal 5."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix, issparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellchat import run_cellchat_on_replicate  # noqa: E402
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate  # noqa: E402
from scripts.benchmark.run_commot import run_commot_on_replicate  # noqa: E402
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
from src.spitialself.fdr import PermutationFDRConfig, permutation_group_fdr  # noqa: E402


RAW_DIR = PROJECT_ROOT / "HumanMelannomaMultiome/data"
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
OUTPUT_DIR = GOAL5_DIR / "native_human_melanoma_multiome"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/native_human_melanoma_multiome_subset"
BASELINE_DIR = OUTPUT_DIR / "baseline_panel"
DOWNSTREAM_DIR = OUTPUT_DIR / "competitive_downstream_200perm"
FDR_DIR = OUTPUT_DIR / "targeted_context_fdr_200perm"

NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")
METHODS = ("spatialself_wavelet_edge_prior", "commot", "cellchat", "cellnest")

LR_PANEL = [
    {
        "lr_id": "native_melanoma_lr_000_CXCL9_CXCR3",
        "ligand": "CXCL9",
        "receptor": "CXCR3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 650.0,
        "active_niche": "tme_immune_context",
        "control_name": "CXCL9-CXCR3",
        "control_class": "melanoma_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": True,
    },
    {
        "lr_id": "native_melanoma_lr_001_CXCL10_CXCR3",
        "ligand": "CXCL10",
        "receptor": "CXCR3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 650.0,
        "active_niche": "tme_immune_context",
        "control_name": "CXCL10-CXCR3",
        "control_class": "melanoma_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": True,
    },
    {
        "lr_id": "native_melanoma_lr_002_CXCL11_CXCR3",
        "ligand": "CXCL11",
        "receptor": "CXCR3",
        "mechanism": "long_range_chemokine_niche",
        "spatial_range": 650.0,
        "active_niche": "tme_immune_context",
        "control_name": "CXCL11-CXCR3",
        "control_class": "melanoma_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": True,
    },
    {
        "lr_id": "native_melanoma_lr_003_SPP1_CD44",
        "ligand": "SPP1",
        "receptor": "CD44",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 360.0,
        "active_niche": "tumor_myeloid_boundary",
        "control_name": "SPP1-CD44",
        "control_class": "melanoma_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": False,
    },
    {
        "lr_id": "native_melanoma_lr_004_PDCD1_CD274",
        "ligand": "PDCD1",
        "receptor": "CD274",
        "mechanism": "juxtacrine_contact",
        "spatial_range": 170.0,
        "active_niche": "tumor_immune_boundary",
        "control_name": "PDCD1-CD274",
        "control_class": "melanoma_positive_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": True,
    },
    {
        "lr_id": "native_melanoma_lr_005_MIF_CD74",
        "ligand": "MIF",
        "receptor": "CD74",
        "mechanism": "short_medium_paracrine",
        "spatial_range": 360.0,
        "active_niche": "myeloid_tumor_context",
        "control_name": "MIF-CD74",
        "control_class": "melanoma_context_prior",
        "false_positive_high_expression": False,
        "true_lr": True,
        "weak_signal": False,
    },
    {
        "lr_id": "native_melanoma_lr_006_ACTB_EGFR_negative",
        "ligand": "ACTB",
        "receptor": "EGFR",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 650.0,
        "active_niche": "any",
        "control_name": "ACTB-EGFR",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
        "weak_signal": False,
    },
    {
        "lr_id": "native_melanoma_lr_007_GAPDH_ITGB1_negative",
        "ligand": "GAPDH",
        "receptor": "ITGB1",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 650.0,
        "active_niche": "any",
        "control_name": "GAPDH-ITGB1",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
        "weak_signal": False,
    },
    {
        "lr_id": "native_melanoma_lr_008_MS4A1_CD3E_negative",
        "ligand": "MS4A1",
        "receptor": "CD3E",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 650.0,
        "active_niche": "any",
        "control_name": "MS4A1-CD3E",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
        "weak_signal": False,
    },
    {
        "lr_id": "native_melanoma_lr_009_B2M_CD44_negative",
        "ligand": "B2M",
        "receptor": "CD44",
        "mechanism": "negative_control_high_expression",
        "spatial_range": 650.0,
        "active_niche": "any",
        "control_name": "B2M-CD44",
        "control_class": "negative_high_expression_control",
        "false_positive_high_expression": True,
        "true_lr": False,
        "weak_signal": False,
    },
]

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

EXTRA_EXPRESSION_GENES = {
    "CD3D",
    "CD3E",
    "CD4",
    "CD8A",
    "MS4A1",
    "CD68",
    "CD14",
    "HLA-DRA",
    "HLA-DRB1",
    "MALAT1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--max-neighbors", type=int, default=36)
    parser.add_argument("--max-candidates-per-lr", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=202611)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-standardize", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_standardize:
        standardization = _standardize_dataset(
            raw_dir,
            dataset_dir,
            max_neighbors=int(args.max_neighbors),
            max_candidates_per_lr=int(args.max_candidates_per_lr),
            seed=int(args.seed),
        )
    else:
        standardization = _load_json(dataset_dir / "metadata.json")

    baseline_status = []
    if not args.skip_baselines:
        baseline_status = _run_baselines(dataset_dir, BASELINE_DIR, epochs=int(args.epochs), seed=int(args.seed))
    else:
        baseline_status = _existing_prediction_status(BASELINE_DIR, dataset_dir)
    baseline_status_table = pd.DataFrame(baseline_status)
    baseline_status_table.to_csv(output_dir / "native_melanoma_baseline_status.csv", index=False)

    downstream = _run_competitive_downstream(dataset_dir, BASELINE_DIR, DOWNSTREAM_DIR, int(args.n_permutations), int(args.seed))
    fdr_audit = _run_targeted_fdr(dataset_dir, BASELINE_DIR, FDR_DIR, int(args.n_permutations), int(args.seed))
    ranking = _rank_controls(dataset_dir, BASELINE_DIR)
    claim = _claim_table(ranking, downstream, fdr_audit)

    ranking.to_csv(output_dir / "native_melanoma_control_ranking.csv", index=False)
    downstream.to_csv(output_dir / "native_melanoma_competitive_downstream_summary.csv", index=False)
    fdr_audit.to_csv(output_dir / "native_melanoma_targeted_fdr_audit.csv", index=False)
    claim.to_csv(output_dir / "native_melanoma_claim_audit.csv", index=False)

    manifest = {
        "dataset": "native_human_melanoma_multiome_subset",
        "platform": "Slide-tags / 10x Multiome human melanoma lymph-node metastasis",
        "raw_dir": _rel(raw_dir),
        "dataset_dir": _rel(dataset_dir),
        "output_dir": _rel(output_dir),
        "baseline_status": _rel(output_dir / "native_melanoma_baseline_status.csv"),
        "ranking": _rel(output_dir / "native_melanoma_control_ranking.csv"),
        "competitive_downstream": _rel(output_dir / "native_melanoma_competitive_downstream_summary.csv"),
        "targeted_fdr": _rel(output_dir / "native_melanoma_targeted_fdr_audit.csv"),
        "claim_audit": _rel(output_dir / "native_melanoma_claim_audit.csv"),
        "n_permutations": int(args.n_permutations),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": _peak_rss_mb(),
        "standardization": standardization,
        "claim_boundary": (
            "Native real-sample validation with real cells, coordinates, cell types, and expression. "
            "There is no native CCC ground truth; success requires positive ranking/context, competitive downstream "
            "support, targeted FDR, and negative-control suppression."
        ),
    }
    (output_dir / "native_melanoma_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(claim.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _standardize_dataset(
    raw_dir: Path,
    dataset_dir: Path,
    *,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
) -> dict[str, Any]:
    lr_pairs = pd.DataFrame(LR_PANEL)
    genes = sorted(
        set(lr_pairs["ligand"])
        | set(lr_pairs["receptor"])
        | EXTRA_EXPRESSION_GENES
        | {gene for program in TARGET_PROGRAMS for gene in program["target_genes"]}
    )
    cells = _load_cells(raw_dir)
    expression = _load_expression(raw_dir, list(cells["cell_id"]), genes)
    cells = _annotate_niches(cells, expression)
    candidates = _candidate_interactions(
        cells,
        expression,
        lr_pairs,
        max_neighbors=max_neighbors,
        max_candidates_per_lr=max_candidates_per_lr,
        seed=seed,
    )

    cells.to_csv(dataset_dir / "cells.csv", index=False)
    expression.to_csv(dataset_dir / "expression.csv", index=False)
    lr_pairs.to_csv(dataset_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(dataset_dir / "candidate_interactions.csv", index=False)
    metadata = {
        "dataset_id": "native_human_melanoma_multiome_subset",
        "platform": "Slide-tags / 10x Multiome human melanoma lymph-node metastasis",
        "source": _rel(raw_dir),
        "n_cells": int(len(cells)),
        "n_expression_genes": int(expression.shape[1] - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_interactions": int(len(candidates)),
        "cell_type_counts": cells["cell_type"].value_counts().to_dict(),
        "niche_counts": cells["niche"].value_counts().to_dict(),
        "candidate_counts_by_lr": candidates["lr_id"].value_counts().to_dict(),
        "max_neighbors": int(max_neighbors),
        "max_candidates_per_lr": int(max_candidates_per_lr),
        "claim_boundary": (
            "Real raw cells, coordinates, cell types, and expression are preserved. Niche labels are derived from "
            "cell-type composition and local tumor/immune neighborhoods; they are not native CCC ground truth."
        ),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([metadata]).to_csv(dataset_dir / "standardization_summary.csv", index=False)
    return metadata


def _load_cells(raw_dir: Path) -> pd.DataFrame:
    spatial = pd.read_csv(raw_dir / "HumanMelanomaMultiome_spatial.csv").iloc[1:].copy()
    spatial["x"] = pd.to_numeric(spatial["X"], errors="coerce")
    spatial["y"] = pd.to_numeric(spatial["Y"], errors="coerce")
    spatial = spatial.rename(columns={"NAME": "cell_id"})
    cells = spatial[["cell_id", "x", "y", "cell_type"]].dropna().copy()
    cells["cell_type"] = cells["cell_type"].map(_normalize_cell_type)
    return cells.reset_index(drop=True)


def _normalize_cell_type(value: Any) -> str:
    text = str(value)
    if text in {"T_CD8", "T_CD4", "T_reg"}:
        return text
    if text in {"mono-mac", "myeloid", "mDC", "pDC"}:
        return "myeloid"
    if text in {"tumour_1", "tumour_2"}:
        return "tumor"
    if text == "plasma":
        return "plasma"
    return "other"


def _load_expression(raw_dir: Path, cell_ids: list[str], genes: list[str]) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(raw_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).tolist()
    feature_gene = features[1].astype(str).to_numpy()
    matrix = mmread(raw_dir / "matrix.mtx.gz")
    matrix = csc_matrix(matrix) if issparse(matrix) else csc_matrix(matrix)
    barcode_to_index = {barcode: idx for idx, barcode in enumerate(barcodes)}
    selected_barcodes = [cell_id for cell_id in cell_ids if cell_id in barcode_to_index]
    barcode_idx = np.array([barcode_to_index[cell_id] for cell_id in selected_barcodes], dtype=int)
    rows = {"cell_id": selected_barcodes}
    for gene in genes:
        matches = np.where(feature_gene == gene)[0]
        if len(matches) == 0:
            rows[gene] = np.zeros(len(selected_barcodes), dtype=float)
            continue
        values = np.asarray(matrix[int(matches[0]), barcode_idx].todense()).ravel().astype(float)
        rows[gene] = values
    return pd.DataFrame(rows)


def _annotate_niches(cells: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    out = cells.merge(expression[["cell_id", "CXCR3", "CXCL9", "CXCL10", "CXCL11", "SPP1", "CD44"]], on="cell_id", how="left")
    coords = out[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    immune = out["cell_type"].isin(["T_CD8", "T_CD4", "T_reg", "myeloid"]).to_numpy()
    tumor = out["cell_type"].eq("tumor").to_numpy()
    myeloid = out["cell_type"].eq("myeloid").to_numpy()
    radius = 420.0
    local_immune = np.zeros(len(out), dtype=int)
    local_tumor = np.zeros(len(out), dtype=int)
    local_myeloid = np.zeros(len(out), dtype=int)
    for idx, neighbors in enumerate(tree.query_ball_point(coords, r=radius)):
        local_immune[idx] = int(immune[neighbors].sum())
        local_tumor[idx] = int(tumor[neighbors].sum())
        local_myeloid[idx] = int(myeloid[neighbors].sum())
    out["local_immune_count"] = local_immune
    out["local_tumor_count"] = local_tumor
    out["local_myeloid_count"] = local_myeloid
    out["region"] = "other_context"
    tme = (local_immune >= 6) & (local_tumor >= 3)
    tumor_myeloid = (local_myeloid >= 3) & (local_tumor >= 3)
    immune_context = (local_immune >= 8) & (local_myeloid >= 2)
    out.loc[tme, "region"] = "tumor_immune_boundary"
    out.loc[tumor_myeloid, "region"] = "tumor_myeloid_boundary"
    out.loc[immune_context & ~tme, "region"] = "tme_immune_context"
    out["niche"] = out["region"]
    return out.drop(columns=["CXCR3", "CXCL9", "CXCL10", "CXCL11", "SPP1", "CD44"])


def _candidate_interactions(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    expr = expression.set_index("cell_id")
    rows = []
    for lr in lr_pairs.itertuples(index=False):
        pairs = []
        for sender_idx, neighbors in enumerate(tree.query_ball_point(coords, r=float(lr.spatial_range))):
            neighbors = [idx for idx in neighbors if idx != sender_idx]
            if len(neighbors) > max_neighbors:
                distances = np.linalg.norm(coords[neighbors] - coords[sender_idx], axis=1)
                neighbors = [neighbors[int(i)] for i in np.argsort(distances)[:max_neighbors]]
            pairs.extend((sender_idx, receiver_idx) for receiver_idx in neighbors)
        if len(pairs) > max_candidates_per_lr:
            keep = rng.choice(np.arange(len(pairs)), size=int(max_candidates_per_lr), replace=False)
            pairs = [pairs[int(i)] for i in np.sort(keep)]
        for sender_idx, receiver_idx in pairs:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            distance = float(np.linalg.norm(coords[int(sender_idx)] - coords[int(receiver_idx)]))
            ligand_expr = float(expr.loc[sender["cell_id"], lr.ligand])
            receptor_expr = float(expr.loc[receiver["cell_id"], lr.receptor])
            rows.append(
                {
                    "sender_id": sender["cell_id"],
                    "receiver_id": receiver["cell_id"],
                    "lr_id": lr.lr_id,
                    "ligand": lr.ligand,
                    "receptor": lr.receptor,
                    "mechanism": lr.mechanism,
                    "distance": distance,
                    "sender_cell_type": sender["cell_type"],
                    "receiver_cell_type": receiver["cell_type"],
                    "sender_niche": sender["niche"],
                    "receiver_niche": receiver["niche"],
                    "weak_signal": bool(lr.weak_signal),
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "true_lr": bool(lr.true_lr),
                    "ground_truth": int(bool(lr.true_lr) and _context_rule(str(lr.control_name), sender, receiver, distance)),
                    "type_rule_satisfied": bool(_type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))),
                    "range_rule_satisfied": bool(distance <= float(lr.spatial_range)),
                    "niche_rule_satisfied": bool(_niche_rule(str(lr.active_niche), sender, receiver)),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates.insert(0, "candidate_id", [f"native_melanoma_cand_{i:08d}" for i in range(len(candidates))])
    return candidates


def _type_rule(control_name: str, sender_type: str, receiver_type: str) -> bool:
    if control_name in {"CXCL9-CXCR3", "CXCL10-CXCR3", "CXCL11-CXCR3"}:
        return sender_type in {"myeloid", "tumor"} and receiver_type in {"T_CD8", "T_CD4", "T_reg"}
    if control_name == "SPP1-CD44":
        return sender_type in {"myeloid", "tumor"} and receiver_type in {"T_CD8", "T_CD4", "T_reg", "tumor", "myeloid"}
    if control_name == "PDCD1-CD274":
        return sender_type in {"T_CD8", "T_CD4", "T_reg"} and receiver_type in {"tumor", "myeloid"}
    if control_name == "MIF-CD74":
        return sender_type in {"tumor", "myeloid"} and receiver_type in {"myeloid", "T_CD8", "T_CD4", "T_reg"}
    return False


def _niche_rule(active_niche: str, sender: pd.Series, receiver: pd.Series) -> bool:
    if active_niche in {"", "any", "nan"}:
        return True
    sender_niche = str(sender["niche"])
    receiver_niche = str(receiver["niche"])
    if active_niche == "myeloid_tumor_context":
        return sender_niche in {"tumor_myeloid_boundary", "tumor_immune_boundary"} or receiver_niche in {
            "tumor_myeloid_boundary",
            "tumor_immune_boundary",
        }
    return sender_niche == active_niche or receiver_niche == active_niche


def _context_rule(control_name: str, sender: pd.Series, receiver: pd.Series, distance: float) -> bool:
    return _type_rule(control_name, str(sender["cell_type"]), str(receiver["cell_type"])) and bool(
        str(sender["niche"]) != "other_context" or str(receiver["niche"]) != "other_context"
    ) and distance > 0


def _run_baselines(dataset_dir: Path, baseline_dir: Path, *, epochs: int, seed: int) -> list[dict[str, Any]]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    status = []
    runners = [
        (
            "spatialself_wavelet_edge_prior",
            lambda: run_spatialself_on_replicate(
                dataset_dir,
                baseline_dir / "spatialself_wavelet_edge_prior",
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=epochs,
                positive_quantile=0.88,
                negative_quantile=0.25,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.2,
                weak_positive_label=0.75,
                weak_positive_mode="long_range",
                ranking_loss_weight=0.2,
                feasibility_loss_weight=0.15,
                hard_negative_loss_weight=0.12,
                context_contrastive_loss_weight=0.15,
                null_candidate_loss_weight=0.08,
                lr_cap_positive=70,
                lr_cap_negative=220,
                use_edge_prior_features=True,
                max_train_samples=22000,
                seed=seed,
            ),
        ),
        ("commot", lambda: run_commot_on_replicate(dataset_dir, baseline_dir / "commot")),
        ("cellchat", lambda: run_cellchat_on_replicate(dataset_dir, baseline_dir / "cellchat", nboot=10)),
        ("cellnest", lambda: run_cellnest_on_replicate(dataset_dir, baseline_dir / "cellnest")),
    ]
    for method, runner in runners:
        started = time.perf_counter()
        try:
            metrics = runner()
            if method == "spatialself_wavelet_edge_prior":
                _alias_spatialself_prediction(dataset_dir, baseline_dir / "spatialself_wavelet_edge_prior")
            status.append(
                {
                    "method": method,
                    "status": "completed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "prediction_path": _rel(Path(metrics.get("prediction_path", ""))),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status.append(
                {
                    "method": method,
                    "status": "failed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "prediction_path": "",
                    "error": str(exc)[:1000],
                }
            )
    return status


def _existing_prediction_status(baseline_dir: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        path = _prediction_path(baseline_dir, dataset_dir, method)
        rows.append(
            {
                "method": method,
                "status": "completed" if path.exists() else "missing",
                "runtime_seconds": "",
                "prediction_path": _rel(path) if path.exists() else "",
                "error": "",
            }
        )
    return rows


def _alias_spatialself_prediction(dataset_dir: Path, output_dir: Path) -> None:
    source = output_dir / f"{dataset_dir.name}__spatialself_wavelet_predictions.csv"
    target = output_dir / f"{dataset_dir.name}__spatialself_wavelet_edge_prior_predictions.csv"
    source_metrics = output_dir / f"{dataset_dir.name}__spatialself_wavelet_metrics.json"
    target_metrics = output_dir / f"{dataset_dir.name}__spatialself_wavelet_edge_prior_metrics.json"
    if source.exists():
        pred = pd.read_csv(source)
        pred["method"] = "spatialself_wavelet_edge_prior"
        pred.to_csv(target, index=False)
    if source_metrics.exists():
        metrics = _load_json(source_metrics)
        metrics["method"] = "spatialself_wavelet_edge_prior"
        metrics["model_name"] = "spatialself_wavelet"
        metrics["prediction_path"] = str(target)
        target_metrics.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")


def _rank_controls(dataset_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    rows = []
    for method in METHODS:
        path = _prediction_path(baseline_dir, dataset_dir, method)
        if not path.exists():
            rows.append({"method": method, "status": "missing_prediction"})
            continue
        predictions = pd.read_csv(path)
        merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        summary = (
            merged.groupby("lr_id", sort=False)
            .agg(score_mean=("score", "mean"), score_q95=("score", lambda x: float(np.quantile(x, 0.95))), n_edges=("score", "size"))
            .reset_index()
            .merge(lr_pairs, on="lr_id", how="left")
        )
        summary["rank_by_score_q95"] = summary["score_q95"].rank(ascending=False, method="min")
        positive = summary[summary["true_lr"].fillna(False).astype(bool)]
        negative = summary[summary["false_positive_high_expression"].fillna(False).astype(bool)]
        best_pos = _best_rank_row(positive)
        best_neg = _best_rank_row(negative)
        rows.append(
            {
                "method": method,
                "status": "tested",
                "best_positive": best_pos.get("control_name", ""),
                "best_positive_rank": best_pos.get("rank_by_score_q95", ""),
                "best_positive_score_q95": best_pos.get("score_q95", ""),
                "best_negative": best_neg.get("control_name", ""),
                "best_negative_rank": best_neg.get("rank_by_score_q95", ""),
                "best_negative_score_q95": best_neg.get("score_q95", ""),
                "positive_top3": int((pd.to_numeric(positive["rank_by_score_q95"], errors="coerce") <= 3).sum()),
                "negative_top3": int((pd.to_numeric(negative["rank_by_score_q95"], errors="coerce") <= 3).sum()),
                "positive_top10": int((pd.to_numeric(positive["rank_by_score_q95"], errors="coerce") <= 10).sum()),
                "negative_top10": int((pd.to_numeric(negative["rank_by_score_q95"], errors="coerce") <= 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def _run_competitive_downstream(
    dataset_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv").set_index("cell_id")
    rows = []
    for method in METHODS:
        path = _prediction_path(baseline_dir, dataset_dir, method)
        if not path.exists():
            rows.append({"method": method, "status": "missing_prediction"})
            continue
        predictions = pd.read_csv(path)
        merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        rows.extend(_competitive_downstream_rows(method, merged, lr_pairs, expression, n_permutations, seed))
    out = pd.DataFrame(rows)
    if not out.empty and "empirical_p_value" in out:
        tested = out["status"].astype(str).eq("tested")
        out.loc[tested, "bh_fdr"] = _bh(out.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
    summary = _downstream_summary(out)
    out.to_csv(output_dir / "native_melanoma_competitive_downstream_all.csv", index=False)
    summary.to_csv(output_dir / "native_melanoma_competitive_downstream_summary.csv", index=False)
    return summary


def _competitive_downstream_rows(
    method: str,
    merged: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    expression: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    pair_meta = lr_pairs.set_index("lr_id").to_dict("index")
    for program in TARGET_PROGRAMS:
        targets = [gene for gene in program["target_genes"] if gene in expression.columns]
        for lr_id, group in merged.groupby("lr_id", sort=False):
            meta = pair_meta[str(lr_id)]
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
                "spearman_r": np.nan,
                "empirical_p_value": np.nan,
                "status": "not_tested",
            }
            if len(targets) < 3:
                row["status"] = "too_few_targets"
                rows.append(row)
                continue
            comm = group.groupby("receiver_id", sort=False)["score"].mean()
            common = expression.index.intersection(comm.index)
            if len(common) < 8:
                row["status"] = "too_few_receiver_cells"
                rows.append(row)
                continue
            communication = comm.loc[common].to_numpy(dtype=float)
            signature = np.log1p(expression.loc[common, targets].to_numpy(dtype=float)).mean(axis=1)
            if np.nanstd(communication) <= 1e-12 or np.nanstd(signature) <= 1e-12:
                row["status"] = "constant_vector"
                rows.append(row)
                continue
            corr, _ = spearmanr(communication, signature)
            exceed = 0
            for _ in range(int(n_permutations)):
                null_corr, _ = spearmanr(communication, rng.permutation(signature))
                if abs(float(null_corr)) >= abs(float(corr)):
                    exceed += 1
            row.update(
                {
                    "spearman_r": float(corr),
                    "empirical_p_value": float((exceed + 1.0) / (n_permutations + 1.0)),
                    "status": "tested",
                }
            )
            rows.append(row)
    return rows


def _downstream_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in all_rows.groupby("method", sort=False):
        tested = group[group["status"].astype(str).eq("tested")]
        expected = tested[tested["expected_program"].fillna(False).astype(bool)]
        unexpected = tested[~tested["expected_program"].fillna(False).astype(bool)]
        negatives = tested[tested["false_positive_high_expression"].fillna(False).astype(bool)]
        best_expected = _best_downstream_row(expected)
        best_negative = _best_downstream_row(negatives)
        rows.append(
            {
                "method": method,
                "status": "tested" if not tested.empty else "not_tested",
                "best_expected_lr": best_expected.get("control_name", ""),
                "best_expected_program": best_expected.get("program", ""),
                "best_expected_spearman": best_expected.get("spearman_r", ""),
                "best_expected_bh_fdr": best_expected.get("bh_fdr", ""),
                "n_expected_fdr_0_1": int((pd.to_numeric(expected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_unexpected_fdr_0_1": int((pd.to_numeric(unexpected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_negative_fdr_0_1": int((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "best_negative_lr": best_negative.get("control_name", ""),
                "best_negative_program": best_negative.get("program", ""),
                "best_negative_spearman": best_negative.get("spearman_r", ""),
                "best_negative_bh_fdr": best_negative.get("bh_fdr", ""),
                "competitive_downstream_status": _competitive_downstream_status(expected, negatives, unexpected),
            }
        )
    return pd.DataFrame(rows)


def _run_targeted_fdr(
    dataset_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    targeted = _targeted_candidates(candidates)
    targeted.to_csv(output_dir / "native_melanoma_targeted_context_candidates.csv", index=False)
    rows = []
    config = PermutationFDRConfig(n_permutations=n_permutations, random_seed=seed, score_column="score")
    for method in METHODS:
        path = _prediction_path(baseline_dir, dataset_dir, method)
        if not path.exists():
            rows.append({"method": method, "targeted_fdr_status": "missing_prediction"})
            continue
        predictions = pd.read_csv(path)
        method_rows = []
        for null_model in NULL_MODELS:
            table = permutation_group_fdr(
                targeted,
                predictions,
                null_model=null_model,
                group_columns=["lr_id", "ligand", "receptor", "mechanism"],
                config=config,
                top_fraction=0.1,
            ).merge(lr_pairs, on=["lr_id", "ligand", "receptor", "mechanism"], how="left")
            table["method"] = method
            table["null_model"] = null_model
            table.to_csv(output_dir / f"{method}__{null_model}__targeted_lr_fdr.csv", index=False)
            method_rows.append(table)
        combined = pd.concat(method_rows, ignore_index=True)
        combined.to_csv(output_dir / f"{method}__targeted_lr_fdr_all.csv", index=False)
        rows.extend(_targeted_fdr_audit_rows(combined, method))
    return pd.DataFrame(rows)


def _targeted_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    positive = (
        candidates["type_rule_satisfied"].fillna(False).astype(bool)
        & candidates["range_rule_satisfied"].fillna(False).astype(bool)
        & candidates["niche_rule_satisfied"].fillna(False).astype(bool)
        & ~candidates["false_positive_high_expression"].fillna(False).astype(bool)
    )
    negative = candidates["false_positive_high_expression"].fillna(False).astype(bool)
    targeted = candidates[positive | negative].copy()
    targeted["targeted_context_reason"] = "positive_context_rule"
    targeted.loc[negative.loc[targeted.index].to_numpy(dtype=bool), "targeted_context_reason"] = "negative_control_all_edges"
    return targeted


def _targeted_fdr_audit_rows(fdr: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    rows = []
    for lr_id, group in fdr.groupby("lr_id", sort=False):
        meta = group.iloc[0].to_dict()
        by_null = group.groupby("null_model")["group_bh_fdr"].min()
        all_null = set(by_null.index) >= set(NULL_MODELS) and bool((pd.to_numeric(by_null, errors="coerce") <= 0.05).all())
        lr_identity = float(by_null.get("lr_identity_permutation", 1.0)) <= 0.05
        is_negative = bool(meta.get("false_positive_high_expression", False))
        if is_negative and (all_null or lr_identity):
            status = "negative_control_leakage"
        elif (not is_negative) and all_null:
            status = "targeted_positive_all_null_success"
        elif (not is_negative) and lr_identity:
            status = "targeted_positive_lr_identity_success"
        else:
            status = "not_targeted_fdr_significant"
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                "control_name": meta.get("control_name", ""),
                "control_class": meta.get("control_class", ""),
                "false_positive_high_expression": is_negative,
                "targeted_spatial_bh_fdr": by_null.get("spatial_permutation", ""),
                "targeted_receiver_bh_fdr": by_null.get("receiver_permutation", ""),
                "targeted_lr_identity_bh_fdr": by_null.get("lr_identity_permutation", ""),
                "targeted_all_null_fdr05": bool(all_null),
                "targeted_lr_identity_fdr05": bool(lr_identity),
                "targeted_fdr_status": status,
            }
        )
    return rows


def _claim_table(ranking: pd.DataFrame, downstream: pd.DataFrame, fdr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        rank = _match(ranking, "method", method)
        ds = _match(downstream, "method", method)
        fdr_m = fdr[fdr.get("method", pd.Series(dtype=str)).astype(str).eq(method)]
        positive_ranking = _safe_int(rank.get("positive_top3", 0), default=0) > 0 and _safe_int(
            rank.get("negative_top3", 999), default=999
        ) == 0
        downstream_pass = (
            str(ds.get("competitive_downstream_status", ""))
            == "expected_program_support_without_negative_leakage"
        )
        all_null_positive = bool((fdr_m.get("targeted_fdr_status", pd.Series(dtype=str)).astype(str) == "targeted_positive_all_null_success").any())
        lr_identity_positive = bool(
            (fdr_m.get("targeted_fdr_status", pd.Series(dtype=str)).astype(str) == "targeted_positive_lr_identity_success").any()
        )
        negative_leakage = bool((fdr_m.get("targeted_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any())
        if positive_ranking and downstream_pass and all_null_positive and not negative_leakage:
            status = "strict_native_real_success"
        elif positive_ranking and downstream_pass and lr_identity_positive and not negative_leakage:
            status = "supportive_native_real_lr_identity_success"
        elif downstream_pass and not negative_leakage:
            status = "supportive_downstream_only"
        else:
            status = "not_clean_native_real_success"
        rows.append(
            {
                "dataset": "native_human_melanoma_multiome_subset",
                "method": method,
                "claim_status": status,
                "positive_ranking_pass": bool(positive_ranking),
                "competitive_downstream_pass": bool(downstream_pass),
                "targeted_all_null_fdr_success": bool(all_null_positive),
                "targeted_lr_identity_fdr_success": bool(lr_identity_positive),
                "targeted_negative_leakage": bool(negative_leakage),
                "best_positive": rank.get("best_positive", ""),
                "best_positive_rank": rank.get("best_positive_rank", ""),
                "best_negative": rank.get("best_negative", ""),
                "best_negative_rank": rank.get("best_negative_rank", ""),
                "best_expected_downstream": ds.get("best_expected_lr", ""),
                "best_expected_downstream_fdr": ds.get("best_expected_bh_fdr", ""),
                "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
                "allowed_claim": _allowed_claim(status),
            }
        )
    return pd.DataFrame(rows)


def _prediction_path(baseline_dir: Path, dataset_dir: Path, method: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        return baseline_dir / method / f"{dataset_dir.name}__{method}_predictions.csv"
    return baseline_dir / method / f"{dataset_dir.name}__{method}_predictions.csv"


def _best_rank_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.sort_values(["rank_by_score_q95", "score_q95"], ascending=[True, False]).iloc[0].to_dict()


def _best_downstream_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0].to_dict()


def _competitive_downstream_status(expected: pd.DataFrame, negatives: pd.DataFrame, unexpected: pd.DataFrame) -> str:
    expected_hit = bool(
        (
            (pd.to_numeric(expected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1)
            & (pd.to_numeric(expected.get("spearman_r", pd.Series(dtype=float)), errors="coerce") > 0)
        ).any()
    )
    negative_leak = bool((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).any())
    unexpected_leak = int((pd.to_numeric(unexpected.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()) > 4
    if expected_hit and not negative_leak and not unexpected_leak:
        return "expected_program_support_without_negative_leakage"
    if expected_hit and negative_leak:
        return "mixed_expected_support_negative_leakage"
    if expected_hit:
        return "mixed_expected_support_unexpected_program_leakage"
    return "no_clean_expected_program_support"


def _allowed_claim(status: str) -> str:
    if status == "strict_native_real_success":
        return "Native melanoma real-sample evidence supports weak/context CCC with downstream response and all-null targeted FDR."
    if status == "supportive_native_real_lr_identity_success":
        return "Supportive native melanoma evidence; LR-identity FDR works but spatial/receiver nulls remain conservative."
    if status == "supportive_downstream_only":
        return "Downstream support only; do not use as FDR closure."
    return "Do not claim native real-sample functional/FDR success under the current control panel."


def _match(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    hit = frame[frame[column].astype(str).eq(str(value))]
    return hit.iloc[0].to_dict() if not hit.empty else {}


def _bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _safe_int(value: Any, *, default: int) -> int:
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return int(default)
    return int(numeric)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
