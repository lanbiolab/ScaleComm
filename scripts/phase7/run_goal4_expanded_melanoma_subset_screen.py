#!/usr/bin/env python
"""Build and screen an expanded melanoma LR panel for Phase 7 Goal 4."""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellchat import run_cellchat_on_replicate  # noqa: E402
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate  # noqa: E402
from scripts.benchmark.run_commot import run_commot_on_replicate  # noqa: E402
from scripts.benchmark.run_spatalk import run_spatalk_on_replicate  # noqa: E402
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
from src.spitialself.fdr import FormalFDRPanelConfig, bh_adjust, run_formal_fdr_panel  # noqa: E402


SOURCE_DATASET = (
    PROJECT_ROOT
    / "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/"
    "human_melanoma_multiome_replicate_00"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/"
    "goal4_expanded_melanoma_subset_screen"
)
DATASET_DIR = OUTPUT_DIR / "constructed_replicate"
BASELINE_DIR = OUTPUT_DIR / "baseline_panel"
FDR_DIR = OUTPUT_DIR / "fdr_screening_50perm"

METHODS = (
    "spatialself_wavelet_edge_prior",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)
ACTIVE_BASELINES = ("commot", "cellchat", "cellnest")
STATUS_BASELINES = ("spatalk",)
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


@dataclass(frozen=True)
class ControlPrior:
    control_name: str
    control_class: str
    ligand: str
    receptor: str
    mechanism: str
    active_niche: str
    spatial_range: float
    expected_senders: frozenset[str]
    expected_receivers: frozenset[str]
    weak_signal: bool = False
    false_positive_high_expression: bool = False
    target_genes: tuple[str, ...] = ()


MELANOMA_CONTROLS = (
    ControlPrior(
        "CXCL9-CXCR3",
        "melanoma_positive_prior",
        "CXCL9",
        "CXCR3",
        "long_range_chemokine_niche",
        "TME_immune",
        0.42,
        frozenset({"Macrophage", "Myeloid", "T"}),
        frozenset({"T"}),
        weak_signal=True,
        target_genes=("CXCR3", "STAT1", "IRF1", "GBP1", "GBP2", "GBP4", "IFITM3", "CXCL10"),
    ),
    ControlPrior(
        "CXCL10-CXCR3",
        "melanoma_positive_prior",
        "CXCL10",
        "CXCR3",
        "long_range_chemokine_niche",
        "TME_immune",
        0.42,
        frozenset({"Macrophage", "Myeloid", "T", "Tumor"}),
        frozenset({"T"}),
        weak_signal=True,
        target_genes=("CXCR3", "STAT1", "IRF1", "GBP1", "GBP2", "GBP4", "IFITM3", "CXCL9"),
    ),
    ControlPrior(
        "CXCL11-CXCR3",
        "melanoma_positive_prior",
        "CXCL11",
        "CXCR3",
        "long_range_chemokine_niche",
        "TME_immune",
        0.42,
        frozenset({"Macrophage", "Myeloid", "T", "Tumor"}),
        frozenset({"T"}),
        weak_signal=True,
        target_genes=("CXCR3", "STAT1", "IRF1", "GBP1", "GBP2", "GBP4", "IFITM3"),
    ),
    ControlPrior(
        "SPP1-CD44",
        "melanoma_positive_prior",
        "SPP1",
        "CD44",
        "short_medium_paracrine",
        "tumor_immune_boundary",
        0.24,
        frozenset({"Macrophage", "Myeloid", "Tumor"}),
        frozenset({"T", "Tumor", "Macrophage", "Other"}),
        target_genes=("CD44", "SPP1", "FOS", "JUNB", "NFKBIA", "TNFAIP3", "STAT1", "CXCR3"),
    ),
    ControlPrior(
        "PDCD1-CD274",
        "melanoma_positive_prior",
        "PDCD1",
        "CD274",
        "juxtacrine_contact",
        "tumor_immune_boundary",
        0.11,
        frozenset({"T"}),
        frozenset({"Tumor", "Macrophage", "Myeloid"}),
        target_genes=("CD274", "PDCD1", "STAT1", "IRF1", "IDO1", "HLA-A", "HLA-B", "HLA-C"),
    ),
    ControlPrior(
        "MIF-CD74",
        "melanoma_context_prior",
        "MIF",
        "CD74",
        "short_medium_paracrine",
        "myeloid_tumor",
        0.24,
        frozenset({"Tumor", "Macrophage", "Myeloid", "Other"}),
        frozenset({"Macrophage", "Myeloid", "T"}),
        target_genes=("CD74", "HLA-DRA", "HLA-DRB1", "STAT1", "IRF1", "NFKBIA", "TNFAIP3"),
    ),
    ControlPrior(
        "EGFR-MIF",
        "melanoma_context_prior",
        "EGFR",
        "MIF",
        "short_medium_paracrine",
        "myeloid_tumor",
        0.24,
        frozenset({"Tumor", "Other"}),
        frozenset({"Macrophage", "Myeloid", "Tumor"}),
        target_genes=("MIF", "EGFR", "FOS", "JUNB", "NFKBIA", "TNFAIP3"),
    ),
    ControlPrior(
        "ACTB-EGFR",
        "housekeeping_negative_control",
        "ACTB",
        "EGFR",
        "false_positive_high_expression",
        "any",
        0.42,
        frozenset(),
        frozenset(),
        false_positive_high_expression=True,
        target_genes=("EGFR", "FOS", "JUNB", "NFKBIA", "TNFAIP3"),
    ),
    ControlPrior(
        "GAPDH-ITGB1",
        "housekeeping_negative_control",
        "GAPDH",
        "ITGB1",
        "false_positive_high_expression",
        "any",
        0.42,
        frozenset(),
        frozenset(),
        false_positive_high_expression=True,
        target_genes=("ITGB1", "FOS", "JUNB", "NFKBIA", "TNFAIP3"),
    ),
    ControlPrior(
        "MALAT1-EGFR",
        "housekeeping_negative_control",
        "MALAT1",
        "EGFR",
        "false_positive_high_expression",
        "any",
        0.42,
        frozenset(),
        frozenset(),
        false_positive_high_expression=True,
        target_genes=("EGFR", "FOS", "JUNB", "NFKBIA", "TNFAIP3"),
    ),
    ControlPrior(
        "B2M-CD44",
        "housekeeping_negative_control",
        "B2M",
        "CD44",
        "false_positive_high_expression",
        "any",
        0.42,
        frozenset(),
        frozenset(),
        false_positive_high_expression=True,
        target_genes=("CD44", "FOS", "JUNB", "NFKBIA", "TNFAIP3"),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", default=str(SOURCE_DATASET.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--n-background-lr", type=int, default=38)
    parser.add_argument("--max-neighbors", type=int, default=44)
    parser.add_argument("--max-candidates-per-lr", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--run-fdr", action="store_true")
    parser.add_argument("--fdr-permutations", type=int, default=50)
    args = parser.parse_args()

    started = time.perf_counter()
    source = PROJECT_ROOT / args.source_dataset
    output_dir = PROJECT_ROOT / args.output_dir
    dataset_dir = output_dir / "constructed_replicate"
    baseline_dir = output_dir / "baseline_panel"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    build_profile = _build_dataset(
        source,
        dataset_dir,
        n_background_lr=int(args.n_background_lr),
        max_neighbors=int(args.max_neighbors),
        max_candidates_per_lr=int(args.max_candidates_per_lr),
        seed=int(args.seed),
    )
    if not args.skip_baselines:
        _run_baselines(dataset_dir, baseline_dir)

    rank_table = _build_rank_table(dataset_dir, baseline_dir)
    method_status = _build_method_status(baseline_dir)
    candidate_table = _screen_candidates(rank_table, method_status)
    validation = _validate_candidates(dataset_dir, baseline_dir, candidate_table)

    fdr_table = pd.DataFrame()
    if args.run_fdr:
        fdr_dir = output_dir / f"fdr_screening_{int(args.fdr_permutations)}perm"
        _run_fdr(dataset_dir, baseline_dir, fdr_dir, n_permutations=int(args.fdr_permutations))
        fdr_table = _collect_fdr(candidate_table, fdr_dir)
    else:
        fdr_table = _collect_fdr(candidate_table, output_dir / f"fdr_screening_{int(args.fdr_permutations)}perm")

    audit = _build_audit(candidate_table, validation, fdr_table, method_status)

    rank_table.to_csv(output_dir / "phase7_goal4_expanded_melanoma_method_lr_rank_table.csv", index=False)
    method_status.to_csv(output_dir / "phase7_goal4_expanded_melanoma_method_status.csv", index=False)
    candidate_table.to_csv(output_dir / "phase7_goal4_expanded_melanoma_spatialself_specific_candidates.csv", index=False)
    validation.to_csv(output_dir / "phase7_goal4_expanded_melanoma_candidate_validation.csv", index=False)
    fdr_table.to_csv(output_dir / "phase7_goal4_expanded_melanoma_fdr_status.csv", index=False)
    audit.to_csv(output_dir / "phase7_goal4_expanded_melanoma_audit.csv", index=False)

    manifest = {
        "script": _rel(Path(__file__)),
        "source_dataset": _rel(source),
        "output_dir": _rel(output_dir),
        "constructed_replicate": _rel(dataset_dir),
        "baseline_dir": _rel(baseline_dir),
        "n_lr_pairs": int(build_profile["n_lr_pairs"]),
        "n_candidate_edges": int(build_profile["n_candidate_edges"]),
        "strict_success": bool((audit["audit_status"] == "strict_success").any()),
        "available_baseline_candidate_found": bool(
            (candidate_table["candidate_status"] == "available_baseline_candidate").any()
        ),
        "fdr_run": bool(args.run_fdr),
        "wall_clock_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": _peak_rss_mb(),
        "outputs": {
            "rank_table": _rel(output_dir / "phase7_goal4_expanded_melanoma_method_lr_rank_table.csv"),
            "method_status": _rel(output_dir / "phase7_goal4_expanded_melanoma_method_status.csv"),
            "candidates": _rel(output_dir / "phase7_goal4_expanded_melanoma_spatialself_specific_candidates.csv"),
            "validation": _rel(output_dir / "phase7_goal4_expanded_melanoma_candidate_validation.csv"),
            "fdr": _rel(output_dir / "phase7_goal4_expanded_melanoma_fdr_status.csv"),
            "audit": _rel(output_dir / "phase7_goal4_expanded_melanoma_audit.csv"),
        },
        "limitations": [
            "Expanded melanoma panel mixes LR-table-derived pairs with explicitly labeled synthetic/control priors.",
            "This is a real-background discovery/control screen, not native melanoma CCC ground truth.",
            "Strict Goal 4 success requires ranking, niche/context support, downstream target support, and all three null FDR.",
        ],
    }
    (output_dir / "phase7_goal4_expanded_melanoma_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    if not candidate_table.empty:
        print(candidate_table.head(12).to_string(index=False))


def _build_dataset(
    source: Path,
    dataset_dir: Path,
    *,
    n_background_lr: int,
    max_neighbors: int,
    max_candidates_per_lr: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells = pd.read_csv(source / "cells.csv")
    expression = pd.read_csv(source / "expression.csv")
    lr_source = pd.read_csv(PROJECT_ROOT / "high_spatial/visium_lr_benchmark/lr_table.csv")
    lr_pairs = _select_lr_pairs(lr_source, expression, n_background_lr=n_background_lr)
    candidates = _build_candidates(
        cells,
        expression,
        lr_pairs,
        max_neighbors=max_neighbors,
        max_candidates_per_lr=max_candidates_per_lr,
        rng=rng,
    )
    cells.to_csv(dataset_dir / "cells.csv", index=False)
    expression.to_csv(dataset_dir / "expression.csv", index=False)
    lr_pairs.to_csv(dataset_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(dataset_dir / "candidate_interactions.csv", index=False)
    target_table = _target_signature_table(lr_pairs)
    target_table.to_csv(dataset_dir / "melanoma_curated_target_signatures.csv", index=False)
    profile = {
        "dataset": "human_melanoma_multiome_expanded_phase7_goal4",
        "n_cells": int(len(cells)),
        "n_genes": int(len(expression.columns) - 1),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_edges": int(len(candidates)),
        "n_positive_or_context_controls": int(
            lr_pairs["control_class"].astype(str).isin({"melanoma_positive_prior", "melanoma_context_prior"}).sum()
        ),
        "n_negative_controls": int((lr_pairs["control_class"] == "housekeeping_negative_control").sum()),
    }
    pd.DataFrame([profile]).to_csv(dataset_dir / "dataset_profile.csv", index=False)
    return profile


def _select_lr_pairs(lr_source: pd.DataFrame, expression: pd.DataFrame, *, n_background_lr: int) -> pd.DataFrame:
    genes = set(expression.columns) - {"cell_id"}
    expr = expression.drop(columns=["cell_id"])
    means = expr.mean(axis=0)
    detection = (expr > 0).mean(axis=0)
    source = lr_source[
        lr_source["ligand"].astype(str).isin(genes) & lr_source["receptor"].astype(str).isin(genes)
    ].copy()
    source["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in source.itertuples(index=False)]
    source["expression_detectability_score"] = [
        float(
            np.log1p(means[row.ligand])
            * np.log1p(means[row.receptor])
            * (0.5 + detection[row.ligand])
            * (0.5 + detection[row.receptor])
        )
        for row in source.itertuples(index=False)
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for control in MELANOMA_CONTROLS:
        if control.ligand not in genes or control.receptor not in genes:
            continue
        canonical = _canonical_pair(control.ligand, control.receptor)
        match = source[source["canonical_pair"] == canonical]
        if match.empty:
            row = {
                "source_lr_id": f"synthetic_control_{control.control_name}",
                "ligand": control.ligand,
                "receptor": control.receptor,
                "source_type": "synthetic_or_control_prior",
            }
        else:
            row = match.sort_values("expression_detectability_score", ascending=False).iloc[0].to_dict()
            row["source_lr_id"] = row.get("lr_id", "")
            row["source_type"] = "lr_table_control_prior"
        row.update(_metadata_from_control(control))
        selected.append(row)
        seen.add(canonical)
    background = source[~source["canonical_pair"].isin(seen)].sort_values(
        "expression_detectability_score", ascending=False
    )
    for row in background.head(n_background_lr).to_dict(orient="records"):
        row["source_lr_id"] = row.get("lr_id", "")
        row["source_type"] = "lr_table_background"
        row.update(_infer_lr_metadata(row["ligand"], row["receptor"]))
        selected.append(row)
        seen.add(row["canonical_pair"])

    rows = []
    for idx, row in enumerate(selected):
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        rows.append(
            {
                "lr_id": f"melanoma_lr_{idx:03d}_{ligand}_{receptor}",
                "source_lr_id": row.get("source_lr_id", row.get("lr_id", "")),
                "source_type": row.get("source_type", "unknown"),
                "ligand": ligand,
                "receptor": receptor,
                "mechanism": row.get("mechanism", "short_medium_paracrine"),
                "spatial_range": float(row.get("spatial_range", 0.24)),
                "active_niche": row.get("active_niche", "any"),
                "control_name": row.get("control_name", ""),
                "control_class": row.get("control_class", "background_lr"),
                "weak_signal": bool(row.get("weak_signal", False)),
                "false_positive_high_expression": bool(row.get("false_positive_high_expression", False)),
                "true_lr": not bool(row.get("false_positive_high_expression", False)),
                "target_genes": row.get("target_genes", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_candidates(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    max_neighbors: int,
    max_candidates_per_lr: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    k = min(max_neighbors + 1, len(cells))
    dist, neigh = tree.query(coords, k=k)
    dist = np.asarray(dist)[:, 1:]
    neigh = np.asarray(neigh)[:, 1:]
    max_observed = float(np.nanmax(dist))
    edge_cache = {
        "juxtacrine_contact": _edge_frame_from_neighbors(dist, neigh, k=min(10, dist.shape[1]), max_distance=0.12),
        "short_medium_paracrine": _edge_frame_from_neighbors(dist, neigh, k=min(24, dist.shape[1]), max_distance=0.26),
        "long_range_chemokine_niche": _edge_frame_from_neighbors(dist, neigh, k=min(max_neighbors, dist.shape[1]), max_distance=0.46),
        "false_positive_high_expression": _edge_frame_from_neighbors(dist, neigh, k=min(32, dist.shape[1]), max_distance=max_observed),
    }
    cell_id = cells["cell_id"].to_numpy()
    cell_type = cells["cell_type"].to_numpy()
    niche = cells["niche"].to_numpy()
    expression_index = expression.set_index("cell_id")
    parts = []
    counter = 0
    for lr in lr_pairs.itertuples(index=False):
        edges = edge_cache.get(lr.mechanism, edge_cache["short_medium_paracrine"]).copy()
        if len(edges) > max_candidates_per_lr:
            take = rng.choice(len(edges), size=max_candidates_per_lr, replace=False)
            edges = edges.iloc[np.sort(take)].copy()
        src = edges["src_idx"].to_numpy(dtype=int)
        dst = edges["dst_idx"].to_numpy(dtype=int)
        ligand_expr = expression_index.loc[cell_id[src], lr.ligand].to_numpy(dtype=float)
        receptor_expr = expression_index.loc[cell_id[dst], lr.receptor].to_numpy(dtype=float)
        type_rule = _type_rule(lr, cell_type[src], cell_type[dst])
        range_rule = edges["distance"].to_numpy(dtype=float) <= float(lr.spatial_range)
        niche_rule = _niche_rule(str(lr.active_niche), niche[src], niche[dst])
        coexp = np.log1p(ligand_expr) * np.log1p(receptor_expr)
        bandwidth = max(float(np.median(edges["distance"])), 1e-6)
        score = coexp * np.exp(-edges["distance"].to_numpy(dtype=float) / bandwidth)
        n = len(edges)
        part = pd.DataFrame(
            {
                "candidate_id": [f"melanoma_expanded_cand_{counter + i:08d}" for i in range(n)],
                "sender_id": cell_id[src],
                "receiver_id": cell_id[dst],
                "lr_id": lr.lr_id,
                "ligand": lr.ligand,
                "receptor": lr.receptor,
                "mechanism": lr.mechanism,
                "distance": edges["distance"].to_numpy(dtype=float),
                "sender_cell_type": cell_type[src],
                "receiver_cell_type": cell_type[dst],
                "sender_niche": niche[src],
                "receiver_niche": niche[dst],
                "weak_signal": bool(lr.weak_signal),
                "false_positive_high_expression": bool(lr.false_positive_high_expression),
                "type_rule_satisfied": type_rule,
                "range_rule_satisfied": range_rule,
                "niche_rule_satisfied": niche_rule,
                "ligand_expr": ligand_expr,
                "receptor_expr": receptor_expr,
                "coexpression_distance_score": score,
            }
        )
        counter += n
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _run_baselines(dataset_dir: Path, baseline_dir: Path) -> None:
    status_rows = []
    metrics_rows = []
    for method in METHODS:
        started = time.perf_counter()
        method_dir = baseline_dir / method
        try:
            if method == "spatialself_wavelet_edge_prior":
                result = run_spatialself_on_replicate(
                    dataset_dir,
                    method_dir,
                    model_name="spatialself_wavelet",
                    epochs=8,
                    k_nn=12,
                    max_train_samples=10000,
                    pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                    ranking_loss_weight=0.35,
                    ranking_margin=0.5,
                    lr_cap_positive=50,
                    lr_cap_negative=200,
                    use_edge_prior_features=True,
                    seed=20260707,
                )
                _rewrite_prediction_method(Path(result["prediction_path"]), method)
                result["method"] = method
            elif method == "commot":
                result = run_commot_on_replicate(dataset_dir, method_dir, cot_nitermax=1000)
            elif method == "cellchat":
                result = run_cellchat_on_replicate(dataset_dir, method_dir, nboot=10)
            elif method == "spatalk":
                result = run_spatalk_on_replicate(dataset_dir, method_dir, per_num=50, min_pairs=1)
            elif method == "cellnest":
                result = run_cellnest_on_replicate(dataset_dir, method_dir, num_epoch=2, hidden=32)
            else:
                raise ValueError(method)
            result["wall_clock_seconds"] = float(time.perf_counter() - started)
            metrics_rows.append(result)
            status_rows.append(
                {
                    "method": method,
                    "status": "completed",
                    "detail": "baseline adapter completed",
                    "prediction_path": _rel(Path(result["prediction_path"])),
                    "wall_clock_seconds": float(time.perf_counter() - started),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "method": method,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "prediction_path": "",
                    "wall_clock_seconds": float(time.perf_counter() - started),
                }
            )
    pd.DataFrame(status_rows).to_csv(baseline_dir / "baseline_status.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(baseline_dir / "baseline_metrics.csv", index=False)


def _build_rank_table(dataset_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    cells = pd.read_csv(dataset_dir / "cells.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv").set_index("lr_id").to_dict("index")
    rows = []
    for method in METHODS:
        pred_path = _prediction_path(baseline_dir, dataset_dir.name, method)
        if not pred_path.exists():
            continue
        predictions = pd.read_csv(pred_path)
        if "score" not in predictions.columns or "candidate_id" not in predictions.columns:
            continue
        annotated = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        lr_summary = _summarize_lr(annotated, cells, operating_quantile=0.94)
        lr_summary.to_csv(baseline_dir / method / f"{method}__lr_summary.csv", index=False)
        for row in lr_summary.itertuples(index=False):
            meta = lr_pairs.get(row.lr_id, {})
            rows.append(
                {
                    "dataset": "melanoma_expanded_45lr_subset",
                    "method": method,
                    "lr_id": row.lr_id,
                    "canonical_pair": row.canonical_pair,
                    "ligand": row.ligand,
                    "receptor": row.receptor,
                    "mechanism": row.mechanism,
                    "source_type": meta.get("source_type", ""),
                    "control_name": meta.get("control_name", ""),
                    "control_class": meta.get("control_class", "background_lr"),
                    "active_niche": meta.get("active_niche", ""),
                    "weak_signal": bool(meta.get("weak_signal", False)),
                    "true_lr": bool(meta.get("true_lr", False)),
                    "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
                    "score_mean": float(row.score_mean),
                    "score_max": float(row.score_max),
                    "top_edge_count": int(row.top_edge_count),
                    "rank_by_score_mean": int(row.rank_by_score_mean),
                    "rank_by_top_edge_count": int(row.rank_by_top_edge_count),
                    "dominant_cell_type_pair": row.dominant_cell_type_pair,
                    "receiver_spatial_concentration": float(row.receiver_spatial_concentration),
                    "prediction_path": _rel(pred_path),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["method", "rank_by_score_mean", "lr_id"])


def _build_method_status(baseline_dir: Path) -> pd.DataFrame:
    status_path = baseline_dir / "baseline_status.csv"
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    rows = []
    for method in METHODS:
        row = status[status["method"] == method] if not status.empty and "method" in status else pd.DataFrame()
        replicate_names = sorted(
            {
                path.name.split("__", 1)[0]
                for path in baseline_dir.glob(f"{method}/*__*_predictions.csv")
            }
        )
        replicate_name = replicate_names[0] if replicate_names else DATASET_DIR.name
        pred_path = _prediction_path(baseline_dir, replicate_name, method)
        n_unique = ""
        n_nonzero = ""
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            n_unique = int(pred["score"].nunique()) if "score" in pred else 0
            n_nonzero = int((pd.to_numeric(pred["score"], errors="coerce").fillna(0.0) != 0).sum()) if "score" in pred else 0
        if row.empty:
            stat = "missing"
            detail = "method row missing"
        else:
            stat = str(row.iloc[0]["status"])
            detail = str(row.iloc[0].get("detail", ""))
        if method == "spatalk" and pred_path.exists() and int(n_unique or 0) <= 1:
            stat = "status_only_constant_or_noninformative"
            detail = "SpaTalk returned constant/noninformative LR scores for this subset."
        elif stat == "completed":
            stat = "rank_usable"
        rows.append(
            {
                "method": method,
                "status_for_goal4_screen": stat,
                "n_unique_scores": n_unique,
                "n_nonzero_scores": n_nonzero,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def _screen_candidates(rank_table: pd.DataFrame, method_status: pd.DataFrame) -> pd.DataFrame:
    if rank_table.empty:
        return pd.DataFrame()
    rank_table = rank_table.copy()
    for column in ["source_type", "control_name", "control_class", "active_niche"]:
        if column in rank_table:
            rank_table[column] = rank_table[column].fillna("")
    pivot = rank_table.pivot_table(
        index=[
            "lr_id",
            "canonical_pair",
            "ligand",
            "receptor",
            "mechanism",
            "source_type",
            "control_name",
            "control_class",
            "active_niche",
            "weak_signal",
            "true_lr",
            "false_positive_high_expression",
        ],
        columns="method",
        values="rank_by_score_mean",
        aggfunc="first",
    ).reset_index()
    for method in METHODS:
        if method not in pivot.columns:
            pivot[method] = np.nan
    status = method_status.set_index("method")["status_for_goal4_screen"].to_dict()
    rows = []
    for row in pivot.itertuples(index=False):
        spatial_rank = _maybe_int(getattr(row, "spatialself_wavelet_edge_prior"))
        commot_rank = _maybe_int(getattr(row, "commot"))
        cellchat_rank = _maybe_int(getattr(row, "cellchat"))
        spatalk_rank = _maybe_int(getattr(row, "spatalk"))
        cellnest_rank = _maybe_int(getattr(row, "cellnest"))
        weak_or_context = (
            bool(row.weak_signal)
            or str(row.control_class) in {"melanoma_positive_prior", "melanoma_context_prior"}
            or str(row.active_niche) not in {"", "any", "nan"}
        )
        available_pass = (
            spatial_rank != ""
            and int(spatial_rank) <= 10
            and commot_rank != ""
            and int(commot_rank) > 10
            and cellchat_rank != ""
            and int(cellchat_rank) > 10
            and cellnest_rank != ""
            and int(cellnest_rank) > 10
            and not bool(row.false_positive_high_expression)
            and weak_or_context
        )
        reasons = []
        if spatial_rank == "" or int(spatial_rank) > 10:
            reasons.append("spatialself_rank_gt10_or_missing")
        if commot_rank == "" or int(commot_rank) <= 10:
            reasons.append("commot_not_gt10")
        if cellchat_rank == "" or int(cellchat_rank) <= 10:
            reasons.append("cellchat_not_gt10")
        if cellnest_rank == "" or int(cellnest_rank) <= 10:
            reasons.append("cellnest_not_gt10")
        if bool(row.false_positive_high_expression):
            reasons.append("high_expression_negative_control")
        if not weak_or_context:
            reasons.append("not_weak_or_context_specific")
        rows.append(
            {
                "dataset": "melanoma_expanded_45lr_subset",
                "lr_id": row.lr_id,
                "canonical_pair": row.canonical_pair,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": row.mechanism,
                "source_type": row.source_type,
                "control_name": row.control_name,
                "control_class": row.control_class,
                "active_niche": row.active_niche,
                "weak_signal": bool(row.weak_signal),
                "true_lr": bool(row.true_lr),
                "false_positive_high_expression": bool(row.false_positive_high_expression),
                "spatialself_rank": spatial_rank,
                "commot_rank": commot_rank,
                "cellchat_rank": cellchat_rank,
                "spatalk_rank": spatalk_rank,
                "cellnest_rank": cellnest_rank,
                "spatalk_status": status.get("spatalk", ""),
                "cellnest_status": status.get("cellnest", ""),
                "passes_available_baselines_gt10": bool(available_pass),
                "candidate_status": "available_baseline_candidate" if available_pass else "not_candidate",
                "failure_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["spatialself_rank", "lr_id"])


def _validate_candidates(dataset_dir: Path, baseline_dir: Path, candidate_table: pd.DataFrame) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame()
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    predictions = pd.read_csv(_prediction_path(baseline_dir, dataset_dir.name, "spatialself_wavelet_edge_prior"))
    scored = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="left")
    downstream = _run_downstream(dataset_dir, scored, predictions)
    rows = []
    for row in candidate_table.itertuples(index=False):
        edges = scored[scored["lr_id"] == row.lr_id].copy()
        top = edges.nlargest(max(1, int(np.ceil(len(edges) * 0.1))), "score")
        receiver_niche, receiver_fc = _top_enrichment(top, edges, "receiver_niche")
        sender_niche, sender_fc = _top_enrichment(top, edges, "sender_niche")
        context, context_fc = _top_enrichment(top, edges, ["sender_cell_type", "receiver_cell_type"])
        ds = downstream.get(row.lr_id, {})
        ds_r = _to_float(ds.get("spearman_r", np.nan))
        ds_fdr = _to_float(ds.get("bh_fdr", np.nan))
        has_ds = ds.get("status") == "tested" and ds_fdr <= 0.1 and ds_r > 0
        niche_context = receiver_fc > 1.0 and context_fc > 1.0 and float(top["niche_rule_satisfied"].mean()) > 0.1
        rows.append(
            {
                "dataset": row.dataset,
                "lr_id": row.lr_id,
                "canonical_pair": row.canonical_pair,
                "candidate_status": row.candidate_status,
                "receiver_niche_enrichment": receiver_niche,
                "receiver_niche_fold_change": receiver_fc,
                "sender_niche_enrichment": sender_niche,
                "sender_niche_fold_change": sender_fc,
                "dominant_sender_receiver_context": context,
                "dominant_context_fold_change": context_fc,
                "top_decile_range_rule_satisfied_rate": float(top["range_rule_satisfied"].mean()) if len(top) else np.nan,
                "top_decile_niche_rule_satisfied_rate": float(top["niche_rule_satisfied"].mean()) if len(top) else np.nan,
                "downstream_status": ds.get("status", "not_run_for_this_lr"),
                "downstream_target_genes": ds.get("target_genes", ""),
                "downstream_n_target_genes": ds.get("n_target_genes", ""),
                "downstream_spearman_r": ds.get("spearman_r", ""),
                "downstream_empirical_p_value": ds.get("empirical_p_value", ""),
                "downstream_bh_fdr": ds.get("bh_fdr", ""),
                "has_downstream_support_fdr10": bool(has_ds),
                "validation_status": (
                    "rank_niche_downstream_supported"
                    if row.candidate_status == "available_baseline_candidate" and niche_context and has_ds
                    else "rank_niche_only_or_not_candidate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_downstream(dataset_dir: Path, scored: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    expression = pd.read_csv(dataset_dir / "expression.csv").set_index("cell_id")
    target_table = pd.read_csv(dataset_dir / "melanoma_curated_target_signatures.csv")
    targets = {
        row.lr_id: [gene for gene in str(row.target_genes).split(";") if gene]
        for row in target_table.itertuples(index=False)
    }
    rng = np.random.default_rng(20260707)
    rows = []
    for lr_id, group in scored.groupby("lr_id", sort=False):
        genes = [gene for gene in targets.get(lr_id, []) if gene in expression.columns]
        row = {
            "lr_id": lr_id,
            "target_genes": ";".join(genes),
            "n_target_genes": int(len(genes)),
            "spearman_r": np.nan,
            "empirical_p_value": np.nan,
            "status": "too_few_targets",
        }
        if len(genes) >= 3:
            comm = group.groupby("receiver_id")["score"].mean()
            common = expression.index.intersection(comm.index)
            if len(common) >= 4:
                from scipy.stats import spearmanr

                communication = comm.loc[common].to_numpy(dtype=float)
                signature = np.log1p(expression.loc[common, genes].to_numpy(dtype=float)).mean(axis=1)
                if np.nanstd(communication) > 1e-12 and np.nanstd(signature) > 1e-12:
                    corr, _ = spearmanr(communication, signature)
                    exceed = 0
                    for _ in range(100):
                        null_corr, _ = spearmanr(communication, rng.permutation(signature))
                        if abs(float(null_corr)) >= abs(float(corr)):
                            exceed += 1
                    row.update(
                        {
                            "spearman_r": float(corr),
                            "empirical_p_value": float((exceed + 1) / 101.0),
                            "status": "tested",
                        }
                    )
                else:
                    row["status"] = "constant_vector"
            else:
                row["status"] = "too_few_receiver_cells"
        rows.append(row)
    frame = pd.DataFrame(rows)
    tested = frame["status"] == "tested"
    frame["bh_fdr"] = np.nan
    if tested.any():
        frame.loc[tested, "bh_fdr"] = bh_adjust(frame.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
    frame.to_csv(OUTPUT_DIR / "phase7_goal4_expanded_melanoma_downstream_target_validation.csv", index=False)
    return frame.set_index("lr_id").to_dict("index")


def _run_fdr(dataset_dir: Path, baseline_dir: Path, fdr_dir: Path, *, n_permutations: int) -> None:
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    prediction_path = _prediction_path(baseline_dir, dataset_dir.name, "spatialself_wavelet_edge_prior")
    predictions = pd.read_csv(prediction_path)
    config = FormalFDRPanelConfig(
        null_models=NULL_MODELS,
        n_permutations=n_permutations,
        random_seed=20260707,
        score_column="score",
        alpha=0.05,
        direct_group_top_fraction=0.1,
        panel_metadata={"dataset": "melanoma_expanded_45lr_subset", "replicate": dataset_dir.name},
        claim_level=f"phase7_goal4_melanoma_{n_permutations}perm_screen",
    )
    run_formal_fdr_panel(
        candidates,
        predictions,
        output_dir=fdr_dir,
        candidate_path=dataset_dir / "candidate_interactions.csv",
        prediction_path=prediction_path,
        config=config,
    )


def _collect_fdr(candidate_table: pd.DataFrame, fdr_dir: Path) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame()
    prediction_stem = f"{DATASET_DIR.name}__spatialself_wavelet_predictions"
    rows = []
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for null in NULL_MODELS:
        path = fdr_dir / f"{prediction_stem}__{null}__lr_pair_direct_fdr.csv"
        maps[null] = pd.read_csv(path).set_index("lr_id").to_dict("index") if path.exists() else {}
    for row in candidate_table.itertuples(index=False):
        out: dict[str, Any] = {
            "dataset": row.dataset,
            "lr_id": row.lr_id,
            "canonical_pair": row.canonical_pair,
            "candidate_status": row.candidate_status,
            "fdr_dir": _rel(fdr_dir) if fdr_dir.exists() else "",
            "fdr_depth": "50_permutation_screening" if fdr_dir.exists() else "not_run",
        }
        all_present = True
        for null in NULL_MODELS:
            item = maps.get(null, {}).get(row.lr_id, {})
            if not item:
                all_present = False
            out[f"{null}_p_value"] = item.get("group_p_value", "")
            out[f"{null}_bh_fdr"] = item.get("group_bh_fdr", "")
        out["all_null_models_present"] = all_present
        out["all_three_nulls_fdr05"] = all(
            _to_float(maps.get(null, {}).get(row.lr_id, {}).get("group_bh_fdr", np.nan)) <= 0.05
            for null in NULL_MODELS
        )
        out["any_fdr05"] = any(
            _to_float(maps.get(null, {}).get(row.lr_id, {}).get("group_bh_fdr", np.nan)) <= 0.05
            for null in NULL_MODELS
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _build_audit(
    candidate_table: pd.DataFrame,
    validation: pd.DataFrame,
    fdr_table: pd.DataFrame,
    method_status: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_table.empty:
        status = "no_rank_table"
        reason = "No usable rank table was generated."
        available = pd.DataFrame()
        strict = pd.DataFrame()
    else:
        merged = candidate_table.merge(
            validation[["lr_id", "validation_status", "has_downstream_support_fdr10"]]
            if not validation.empty
            else pd.DataFrame(columns=["lr_id", "validation_status", "has_downstream_support_fdr10"]),
            on="lr_id",
            how="left",
        ).merge(
            fdr_table[["lr_id", "all_three_nulls_fdr05"]]
            if not fdr_table.empty
            else pd.DataFrame(columns=["lr_id", "all_three_nulls_fdr05"]),
            on="lr_id",
            how="left",
        )
        available = merged[merged["candidate_status"] == "available_baseline_candidate"].copy()
        strict = available[
            available["validation_status"].eq("rank_niche_downstream_supported")
            & available["all_three_nulls_fdr05"].fillna(False).astype(bool)
        ]
        if len(strict):
            status = "strict_success"
            reason = "At least one expanded melanoma candidate has ranking, validation, and three-null FDR support."
        elif len(available):
            status = "candidate_pending_functional_or_final_fdr"
            reason = (
                "Expanded melanoma candidate(s) satisfy the rank screen, but strict Goal 4 still needs "
                "downstream and/or all-three-null FDR support."
            )
        else:
            status = "no_available_baseline_candidate"
            reason = "No LR passed SpatialSelf top10 and COMMOT/CellChat/CellNEST rank >10 screen."
    top_candidates = (
        available.sort_values("spatialself_rank")["canonical_pair"].head(5).tolist()
        if not available.empty
        else []
    )
    return pd.DataFrame(
        [
            {
                "goal_id": "goal4_expanded_melanoma_subset_screen",
                "audit_status": status,
                "audit_reason": reason,
                "n_lr_pairs_screened": int(candidate_table["lr_id"].nunique()) if not candidate_table.empty else 0,
                "n_available_baseline_candidates": int(len(available)),
                "n_strict_candidates": int(len(strict)),
                "top_available_candidates": ";".join(top_candidates),
                "spatalk_status": _status(method_status, "spatalk"),
                "cellnest_status": _status(method_status, "cellnest"),
            }
        ]
    )


def _summarize_lr(annotated: pd.DataFrame, cells: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(annotated["score"].quantile(operating_quantile))
    top = annotated[annotated["score"] >= threshold].copy()
    top["cell_type_pair"] = top["sender_cell_type"].astype(str) + "->" + top["receiver_cell_type"].astype(str)
    all_dispersion = _coordinate_dispersion(cells["x"].to_numpy(), cells["y"].to_numpy())
    rows = []
    for lr_id, group in annotated.groupby("lr_id", sort=False):
        top_group = top[top["lr_id"] == lr_id]
        pair_counts = top_group["cell_type_pair"].value_counts()
        receiver_dispersion = _receiver_dispersion(top_group, cells)
        rows.append(
            {
                "lr_id": lr_id,
                "ligand": group["ligand"].iloc[0],
                "receptor": group["receptor"].iloc[0],
                "mechanism": group["mechanism"].iloc[0],
                "n_candidate_edges": int(len(group)),
                "score_mean": float(group["score"].mean()),
                "score_max": float(group["score"].max()),
                "score_p95": float(group["score"].quantile(0.95)),
                "top_edge_count": int(len(top_group)),
                "top_score_mean": float(top_group["score"].mean()) if len(top_group) else 0.0,
                "top_distance_median": float(top_group["distance"].median()) if len(top_group) else 0.0,
                "dominant_cell_type_pair": str(pair_counts.index[0]) if len(pair_counts) else "",
                "dominant_cell_type_pair_fraction": float(pair_counts.iloc[0] / len(top_group)) if len(pair_counts) else 0.0,
                "receiver_spatial_concentration": float(1.0 - receiver_dispersion / all_dispersion)
                if all_dispersion > 0 and math.isfinite(receiver_dispersion)
                else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    summary["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in summary.itertuples(index=False)]
    summary["rank_by_score_mean"] = summary["score_mean"].rank(method="min", ascending=False).astype(int)
    summary["rank_by_top_edge_count"] = summary["top_edge_count"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank_by_score_mean", "lr_id"]).reset_index(drop=True)


def _edge_frame_from_neighbors(dist: np.ndarray, neigh: np.ndarray, *, k: int, max_distance: float) -> pd.DataFrame:
    keep_dist = dist[:, :k]
    keep_neigh = neigh[:, :k]
    src = np.repeat(np.arange(dist.shape[0]), k)
    dst = keep_neigh.reshape(-1)
    flat_dist = keep_dist.reshape(-1)
    mask = np.isfinite(flat_dist) & (flat_dist > 0) & (flat_dist <= float(max_distance))
    return pd.DataFrame({"src_idx": src[mask].astype(np.int64), "dst_idx": dst[mask].astype(np.int64), "distance": flat_dist[mask]})


def _type_rule(lr: object, sender_types: np.ndarray, receiver_types: np.ndarray) -> np.ndarray:
    control = _find_control(str(lr.ligand), str(lr.receptor))
    if bool(lr.false_positive_high_expression):
        return np.zeros(len(sender_types), dtype=bool)
    if control is None or not control.expected_senders or not control.expected_receivers:
        return np.ones(len(sender_types), dtype=bool)
    forward = np.isin(sender_types, list(control.expected_senders)) & np.isin(receiver_types, list(control.expected_receivers))
    reverse = np.isin(sender_types, list(control.expected_receivers)) & np.isin(receiver_types, list(control.expected_senders))
    return forward | reverse


def _niche_rule(active_niche: str, sender_niche: np.ndarray, receiver_niche: np.ndarray) -> np.ndarray:
    if active_niche in {"", "any", "nan", "None"}:
        return np.ones(len(sender_niche), dtype=bool)
    if active_niche == "TME_immune":
        allowed = {"mono-mac", "T_CD8", "T_reg", "T_CD4", "tumor", "malignant"}
        return np.isin(sender_niche, list(allowed)) & np.isin(receiver_niche, list(allowed))
    if active_niche == "tumor_immune_boundary":
        immune = {"mono-mac", "T_CD8", "T_reg", "T_CD4"}
        tumor = {"tumor", "tumour_1", "tumour_2", "malignant"}
        sender_immune_receiver_tumor = np.isin(sender_niche, list(immune)) & np.isin(receiver_niche, list(tumor))
        sender_tumor_receiver_immune = np.isin(sender_niche, list(tumor)) & np.isin(receiver_niche, list(immune))
        within_immune = np.isin(sender_niche, list(immune)) & np.isin(receiver_niche, list(immune))
        return sender_immune_receiver_tumor | sender_tumor_receiver_immune | within_immune
    if active_niche == "myeloid_tumor":
        allowed = {"mono-mac", "tumor", "tumour_1", "tumour_2", "malignant"}
        return np.isin(sender_niche, list(allowed)) & np.isin(receiver_niche, list(allowed))
    allowed = set(active_niche.split("|"))
    return np.isin(sender_niche, list(allowed)) & np.isin(receiver_niche, list(allowed))


def _target_signature_table(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    control_targets = {
        _canonical_pair(c.ligand, c.receptor): ";".join(c.target_genes)
        for c in MELANOMA_CONTROLS
    }
    fallback = {
        "CXCR3": "CXCR3;STAT1;IRF1;GBP1;GBP2;GBP4;IFITM3",
        "CD44": "CD44;FOS;JUNB;NFKBIA;TNFAIP3;STAT1",
        "CD274": "CD274;STAT1;IRF1;IDO1;HLA-A;HLA-B;HLA-C",
        "CD74": "CD74;HLA-DRA;HLA-DRB1;STAT1;IRF1;NFKBIA",
        "MIF": "MIF;FOS;JUNB;NFKBIA;TNFAIP3",
        "EGFR": "EGFR;FOS;JUNB;NFKBIA;TNFAIP3",
        "ITGB1": "ITGB1;FOS;JUNB;NFKBIA;TNFAIP3",
    }
    rows = []
    for row in lr_pairs.itertuples(index=False):
        canonical = _canonical_pair(row.ligand, row.receptor)
        targets = control_targets.get(canonical, "")
        if not targets:
            targets = fallback.get(str(row.receptor), fallback.get(str(row.ligand), ""))
        rows.append(
            {
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "target_genes": targets,
                "target_source": "phase7_curated_selected_signature" if targets else "none",
            }
        )
    return pd.DataFrame(rows)


def _infer_lr_metadata(ligand: str, receptor: str) -> dict[str, Any]:
    ligand = str(ligand)
    receptor = str(receptor)
    chemokine_prefixes = ("CCL", "CXCL", "XCL", "CX3CL")
    contact_markers = ("NOTCH", "DLL", "JAG", "CDH", "ICAM", "VCAM", "ITG", "SELE", "SELL", "SELPLG", "HLA")
    if ligand.startswith(chemokine_prefixes) or receptor.startswith(("CCR", "CXCR", "XCR", "CX3CR")):
        return {"mechanism": "long_range_chemokine_niche", "spatial_range": 0.42, "active_niche": "any", "control_name": "", "control_class": "background_lr", "weak_signal": False, "false_positive_high_expression": False}
    if any(token in ligand or token in receptor for token in contact_markers):
        return {"mechanism": "juxtacrine_contact", "spatial_range": 0.11, "active_niche": "any", "control_name": "", "control_class": "background_lr", "weak_signal": False, "false_positive_high_expression": False}
    return {"mechanism": "short_medium_paracrine", "spatial_range": 0.24, "active_niche": "any", "control_name": "", "control_class": "background_lr", "weak_signal": False, "false_positive_high_expression": False}


def _metadata_from_control(control: ControlPrior) -> dict[str, Any]:
    return {
        "mechanism": control.mechanism,
        "spatial_range": control.spatial_range,
        "active_niche": control.active_niche,
        "control_name": control.control_name,
        "control_class": control.control_class,
        "weak_signal": control.weak_signal,
        "false_positive_high_expression": control.false_positive_high_expression,
        "target_genes": ";".join(control.target_genes),
    }


def _find_control(ligand: str, receptor: str) -> ControlPrior | None:
    canonical = _canonical_pair(ligand, receptor)
    for control in MELANOMA_CONTROLS:
        if _canonical_pair(control.ligand, control.receptor) == canonical:
            return control
    return None


def _prediction_path(baseline_dir: Path, replicate_name: str, method: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        return baseline_dir / method / f"{replicate_name}__spatialself_wavelet_predictions.csv"
    return baseline_dir / method / f"{replicate_name}__{method}_predictions.csv"


def _rewrite_prediction_method(prediction_path: Path, method: str) -> None:
    prediction = pd.read_csv(prediction_path)
    prediction["method"] = method
    prediction.to_csv(prediction_path, index=False)


def _receiver_dispersion(top_group: pd.DataFrame, cells: pd.DataFrame) -> float:
    if top_group.empty:
        return math.nan
    coords = cells.set_index("cell_id").loc[top_group["receiver_id"], ["x", "y"]].to_numpy(dtype=float)
    return _coordinate_dispersion(coords[:, 0], coords[:, 1])


def _coordinate_dispersion(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return math.nan
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    return float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))


def _top_enrichment(top: pd.DataFrame, all_edges: pd.DataFrame, columns: str | list[str]) -> tuple[str, float]:
    if len(top) == 0 or len(all_edges) == 0:
        return "", np.nan
    if isinstance(columns, str):
        top_key = top[columns].astype(str)
        all_key = all_edges[columns].astype(str)
    else:
        top_key = top[columns].astype(str).agg("->".join, axis=1)
        all_key = all_edges[columns].astype(str).agg("->".join, axis=1)
    best = top_key.value_counts().index[0]
    top_rate = float((top_key == best).mean())
    all_rate = float((all_key == best).mean())
    return str(best), top_rate / max(all_rate, 1e-12)


def _status(method_status: pd.DataFrame, method: str) -> str:
    row = method_status[method_status["method"] == method]
    if row.empty:
        return ""
    return str(row.iloc[0]["status_for_goal4_screen"])


def _maybe_int(value: Any) -> int | str:
    if pd.isna(value):
        return ""
    return int(value)


def _to_float(value: Any) -> float:
    try:
        if value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if sys.platform == "darwin":
        peak /= 1024.0
    return float(peak)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
