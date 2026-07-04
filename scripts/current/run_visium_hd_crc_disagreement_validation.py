#!/usr/bin/env python
"""Phase 15 real-data disagreement validation on Visium HD CRC.

The goal is not to declare ground truth in a real tissue. Instead, this script
compares whether SpatialSelf, COMMOT, and CellNEST localize each LR axis to
receiver regions that are more consistent with curated downstream response
signatures and the annotated CRC microenvironment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL7_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DEFAULT_DATASET_DIR = GOAL7_DIR / "standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/current_visium_hd_crc_disagreement_validation"


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path

METHOD_SPECS = {
    "spatialself": {
        "label": os.environ.get("CRC_SPATIALSELF_LABEL", "SpatialSelf"),
        "prediction": _path_from_env(
            "CRC_SPATIALSELF_PREDICTION",
            GOAL7_DIR
            / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
            "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
        ),
        "role": "primary_method",
    },
    "commot": {
        "label": "COMMOT",
        "prediction": GOAL7_DIR
        / "visium_hd_crc_p2crc_baseline_panel/commot/visium_hd_crc_p2crc_subset__commot_predictions.csv",
        "role": "baseline",
    },
    "cellnest": {
        "label": "CellNEST",
        "prediction": GOAL7_DIR
        / "visium_hd_crc_p2crc_baseline_panel/cellnest/visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
        "role": "baseline",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202622)
    parser.add_argument("--min-target-genes", type=int, default=3)
    args = parser.parse_args()

    dataset_dir = _resolve_path(args.dataset_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    target_signatures = pd.read_csv(dataset_dir / "target_signatures.csv")

    _validate_dataset(cells, expression, lr_pairs, candidates, target_signatures)

    prediction_tables, method_status = _load_predictions(candidates)
    method_status.to_csv(output_dir / "method_input_status.csv", index=False)

    target_table, target_status = _build_target_activity_table(
        cells=cells,
        expression=expression,
        lr_pairs=lr_pairs,
        target_signatures=target_signatures,
        min_target_genes=int(args.min_target_genes),
    )
    target_table.to_csv(output_dir / "receiver_target_signature_activity.csv", index=False)
    target_status.to_csv(output_dir / "target_signature_status.csv", index=False)

    receiver_scores = _build_receiver_scores(cells, candidates, prediction_tables)
    receiver_scores.to_csv(output_dir / "method_receiver_incoming_scores.csv", index=False)

    rank_table = _rank_lr_pairs(candidates, lr_pairs, prediction_tables)
    rank_table.to_csv(output_dir / "method_lr_rank_table.csv", index=False)

    correlation = _downstream_correlation(
        receiver_scores=receiver_scores,
        target_table=target_table,
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
    )
    correlation.to_csv(output_dir / "downstream_correlation_by_method.csv", index=False)

    overlap = _hotspot_overlap_and_context(
        cells=cells,
        receiver_scores=receiver_scores,
        target_table=target_table,
        lr_pairs=lr_pairs,
        top_n=int(args.top_n),
    )
    overlap.to_csv(output_dir / "hotspot_overlap_with_target_signature.csv", index=False)

    candidate_scan = _candidate_axis_scan(
        lr_pairs=lr_pairs,
        rank_table=rank_table,
        correlation=correlation,
        overlap=overlap,
    )
    candidate_scan.to_csv(output_dir / "candidate_axis_scan.csv", index=False)
    candidate_scan.sort_values(
        ["recommendation_tier", "priority_score"],
        ascending=[True, False],
    ).to_csv(output_dir / "candidate_axis_recommendations.csv", index=False)

    summary = _write_status_markdown(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        cells=cells,
        candidates=candidates,
        lr_pairs=lr_pairs,
        method_status=method_status,
        candidate_scan=candidate_scan,
        top_n=int(args.top_n),
        n_permutations=int(args.n_permutations),
    )

    manifest = {
        "phase": "Phase 15",
        "objective": (
            "Non-lymph-node real-data validation: compare SpatialSelf, COMMOT, and CellNEST on Visium HD CRC "
            "and identify LR axes where spatial localization is better supported by receiver downstream response."
        ),
        "dataset_dir": _rel(dataset_dir),
        "output_dir": _rel(output_dir),
        "methods": {method: {**spec, "prediction": _rel(spec["prediction"])} for method, spec in METHOD_SPECS.items()},
        "top_n_hotspots": int(args.top_n),
        "n_permutations": int(args.n_permutations),
        "outputs": {
            "method_input_status": _rel(output_dir / "method_input_status.csv"),
            "target_signature_status": _rel(output_dir / "target_signature_status.csv"),
            "receiver_target_signature_activity": _rel(output_dir / "receiver_target_signature_activity.csv"),
            "method_receiver_incoming_scores": _rel(output_dir / "method_receiver_incoming_scores.csv"),
            "method_lr_rank_table": _rel(output_dir / "method_lr_rank_table.csv"),
            "downstream_correlation_by_method": _rel(output_dir / "downstream_correlation_by_method.csv"),
            "hotspot_overlap_with_target_signature": _rel(output_dir / "hotspot_overlap_with_target_signature.csv"),
            "candidate_axis_scan": _rel(output_dir / "candidate_axis_scan.csv"),
            "candidate_axis_recommendations": _rel(output_dir / "candidate_axis_recommendations.csv"),
            "status_markdown": _rel(output_dir / "phase15_visium_hd_crc_disagreement_status.md"),
        },
        "claim_boundary": (
            "This is observational validation on real Visium HD CRC data. It can support claims about consistency with "
            "spatial niche context and receiver downstream signatures, but it is not perturbation ground truth."
        ),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print((output_dir / "phase15_visium_hd_crc_disagreement_status.md").read_text(encoding="utf-8"))


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_dataset(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    target_signatures: pd.DataFrame,
) -> None:
    required_cells = {"cell_id", "x", "y", "cell_type", "niche"}
    required_candidates = {"candidate_id", "sender_id", "receiver_id", "lr_id", "distance"}
    required_lr = {"lr_id", "ligand", "receptor", "control_name", "true_lr", "false_positive_high_expression"}
    required_targets = {"ligand", "receptor"}
    missing = {
        "cells": sorted(required_cells - set(cells.columns)),
        "candidate_interactions": sorted(required_candidates - set(candidates.columns)),
        "lr_pairs": sorted(required_lr - set(lr_pairs.columns)),
        "target_signatures": sorted(required_targets - set(target_signatures.columns)),
    }
    missing = {name: cols for name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not cells["cell_id"].is_unique:
        raise ValueError("cells.csv contains duplicated cell_id values")
    if not candidates["candidate_id"].is_unique:
        raise ValueError("candidate_interactions.csv contains duplicated candidate_id values")


def _load_predictions(candidates: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prediction_tables: dict[str, pd.DataFrame] = {}
    rows = []
    candidate_ids = candidates["candidate_id"].astype(str)
    candidate_id_set = set(candidate_ids)
    for method, spec in METHOD_SPECS.items():
        pred_path = Path(spec["prediction"])
        if not pred_path.exists():
            rows.append(
                {
                    "method": method,
                    "method_label": spec["label"],
                    "status": "missing_prediction",
                    "prediction_path": _rel(pred_path),
                    "n_predictions": 0,
                    "candidate_id_order_identical": False,
                    "candidate_id_set_mismatch": len(candidate_id_set),
                    "detail": "prediction file not found",
                }
            )
            continue
        pred = pd.read_csv(pred_path)
        if not {"candidate_id", "score"}.issubset(pred.columns):
            rows.append(
                {
                    "method": method,
                    "method_label": spec["label"],
                    "status": "invalid_prediction_schema",
                    "prediction_path": _rel(pred_path),
                    "n_predictions": len(pred),
                    "candidate_id_order_identical": False,
                    "candidate_id_set_mismatch": np.nan,
                    "detail": "prediction must contain candidate_id and score",
                }
            )
            continue
        pred = pred[["candidate_id", "score"]].copy()
        pred["candidate_id"] = pred["candidate_id"].astype(str)
        pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
        mismatch = len(candidate_id_set.symmetric_difference(set(pred["candidate_id"])))
        order_identical = len(pred) == len(candidates) and candidate_ids.equals(pred["candidate_id"])
        if mismatch:
            status = "candidate_id_mismatch"
            detail = "prediction candidate_ids do not match dataset candidate_ids"
        elif pred["score"].nunique(dropna=True) <= 1:
            status = "completed_constant_score"
            detail = "prediction file loaded but score vector is constant"
        else:
            status = "completed"
            detail = "prediction loaded and aligned by candidate_id"
            prediction_tables[method] = pred
        rows.append(
            {
                "method": method,
                "method_label": spec["label"],
                "status": status,
                "prediction_path": _rel(pred_path),
                "n_predictions": len(pred),
                "candidate_id_order_identical": bool(order_identical),
                "candidate_id_set_mismatch": int(mismatch),
                "score_min": _finite_or_nan(pred["score"].min()),
                "score_mean": _finite_or_nan(pred["score"].mean()),
                "score_max": _finite_or_nan(pred["score"].max()),
                "detail": detail,
            }
        )
    if not prediction_tables:
        raise RuntimeError("No usable prediction tables were loaded.")
    return prediction_tables, pd.DataFrame(rows)


def _build_target_activity_table(
    *,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    target_signatures: pd.DataFrame,
    min_target_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = expression.copy()
    expr["cell_id"] = expr["cell_id"].astype(str)
    cells_meta = cells[["cell_id", "cell_type", "niche", "x", "y"]].copy()
    cells_meta["cell_id"] = cells_meta["cell_id"].astype(str)
    expr = cells_meta.merge(expr, on="cell_id", how="left")
    gene_columns = set(expression.columns) - {"cell_id"}

    target_by_pair = {}
    for row in target_signatures.itertuples(index=False):
        ligand = str(getattr(row, "ligand"))
        receptor = str(getattr(row, "receptor"))
        raw_targets = getattr(row, "target_genes", None)
        if raw_targets is None:
            raw_targets = getattr(row, "targets", "")
        target_by_pair[(ligand, receptor)] = _split_genes(raw_targets)

    activity_rows = []
    status_rows = []
    for lr in lr_pairs.itertuples(index=False):
        lr_id = str(lr.lr_id)
        ligand = str(lr.ligand)
        receptor = str(lr.receptor)
        raw_targets = target_by_pair.get((ligand, receptor), [])
        present = [gene for gene in raw_targets if gene in gene_columns]
        no_lr = [gene for gene in present if gene not in {ligand, receptor}]
        if len(no_lr) >= min_target_genes:
            genes = no_lr
            target_mode = "downstream_no_ligand_receptor"
        else:
            genes = present
            target_mode = "all_present_targets"
        status = "tested" if len(genes) >= min_target_genes else "too_few_target_genes"
        status_rows.append(
            {
                "lr_id": lr_id,
                "control_name": lr.control_name,
                "ligand": ligand,
                "receptor": receptor,
                "status": status,
                "target_mode": target_mode,
                "n_raw_targets": len(raw_targets),
                "n_present_targets": len(present),
                "n_used_targets": len(genes),
                "used_target_genes": ",".join(genes),
            }
        )
        if status != "tested":
            activity = pd.Series(np.nan, index=expr.index)
        else:
            activity = np.log1p(expr[genes].fillna(0.0).astype(float)).mean(axis=1)
        table = expr[["cell_id", "cell_type", "niche", "x", "y"]].copy()
        table.insert(0, "lr_id", lr_id)
        table.insert(1, "control_name", lr.control_name)
        table["target_activity"] = activity
        table["target_status"] = status
        table["target_mode"] = target_mode
        table["n_used_targets"] = len(genes)
        table["used_target_genes"] = ",".join(genes)
        activity_rows.append(table)
    return pd.concat(activity_rows, ignore_index=True), pd.DataFrame(status_rows)


def _build_receiver_scores(
    cells: pd.DataFrame,
    candidates: pd.DataFrame,
    prediction_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cell_index = cells[["cell_id", "x", "y", "cell_type", "niche"]].copy()
    cell_index["cell_id"] = cell_index["cell_id"].astype(str)
    lr_ids = candidates["lr_id"].drop_duplicates().astype(str).tolist()
    receiver_grid = pd.MultiIndex.from_product(
        [lr_ids, cell_index["cell_id"].tolist()],
        names=["lr_id", "receiver_id"],
    ).to_frame(index=False)
    receiver_grid = receiver_grid.merge(
        cell_index,
        left_on="receiver_id",
        right_on="cell_id",
        how="left",
    ).drop(columns=["cell_id"])

    base = candidates[["candidate_id", "lr_id", "receiver_id"]].copy()
    base["candidate_id"] = base["candidate_id"].astype(str)
    base["receiver_id"] = base["receiver_id"].astype(str)
    rows = []
    for method, pred in prediction_tables.items():
        merged = base.merge(pred, on="candidate_id", how="left", validate="one_to_one")
        incoming = (
            merged.groupby(["lr_id", "receiver_id"], sort=False)["score"]
            .mean()
            .reset_index(name="incoming_score")
        )
        method_table = receiver_grid.merge(incoming, on=["lr_id", "receiver_id"], how="left")
        method_table["incoming_score"] = method_table["incoming_score"].fillna(0.0)
        method_table.insert(0, "method", method)
        method_table.insert(1, "method_label", METHOD_SPECS[method]["label"])
        rows.append(method_table)
    return pd.concat(rows, ignore_index=True)


def _rank_lr_pairs(
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    prediction_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    base = candidates[["candidate_id", "lr_id"]].copy()
    base["candidate_id"] = base["candidate_id"].astype(str)
    lr_meta = lr_pairs.copy()
    for method, pred in prediction_tables.items():
        merged = base.merge(pred, on="candidate_id", how="left", validate="one_to_one")
        scored = (
            merged.groupby("lr_id", sort=False)["score"]
            .agg(mean_score="mean", median_score="median", max_score="max", n_edges="count")
            .reset_index()
        )
        table = lr_meta.merge(scored, on="lr_id", how="left")
        table["rank_by_mean_score"] = table["mean_score"].rank(method="min", ascending=False).astype(int)
        table["rank_by_max_score"] = table["max_score"].rank(method="min", ascending=False).astype(int)
        top100 = merged.sort_values("score", ascending=False).head(100).groupby("lr_id").size()
        top1pct_n = max(int(math.ceil(len(merged) * 0.01)), 1)
        top1pct = merged.sort_values("score", ascending=False).head(top1pct_n).groupby("lr_id").size()
        table["top100_edges"] = table["lr_id"].map(top100).fillna(0).astype(int)
        table["top1pct_edges"] = table["lr_id"].map(top1pct).fillna(0).astype(int)
        table.insert(0, "method", method)
        table.insert(1, "method_label", METHOD_SPECS[method]["label"])
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _downstream_correlation(
    *,
    receiver_scores: pd.DataFrame,
    target_table: pd.DataFrame,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    target = target_table.rename(columns={"cell_id": "receiver_id"})
    merged = receiver_scores.merge(
        target[["lr_id", "receiver_id", "target_activity", "target_status", "target_mode", "n_used_targets"]],
        on=["lr_id", "receiver_id"],
        how="left",
    )
    rng = np.random.default_rng(random_seed)
    rows = []
    for (method, lr_id), group in merged.groupby(["method", "lr_id"], sort=False):
        tested = group[group["target_status"].eq("tested")].dropna(subset=["target_activity", "incoming_score"])
        row = {
            "method": method,
            "method_label": METHOD_SPECS[method]["label"],
            "lr_id": lr_id,
            "status": "tested",
            "n_receivers": int(len(tested)),
            "target_mode": tested["target_mode"].iloc[0] if len(tested) else "",
            "n_used_targets": int(tested["n_used_targets"].iloc[0]) if len(tested) else 0,
        }
        if len(tested) < 30:
            row.update({"status": "too_few_receivers", "spearman_r": np.nan, "spearman_p": np.nan, "empirical_p": np.nan})
            rows.append(row)
            continue
        score = tested["incoming_score"].to_numpy(dtype=float)
        target_values = tested["target_activity"].to_numpy(dtype=float)
        if np.unique(score).size < 2 or np.unique(target_values).size < 2:
            row.update({"status": "constant_vector", "spearman_r": np.nan, "spearman_p": np.nan, "empirical_p": np.nan})
            rows.append(row)
            continue
        rho, p_value = stats.spearmanr(score, target_values)
        if not np.isfinite(rho):
            row.update({"status": "nonfinite_correlation", "spearman_r": np.nan, "spearman_p": np.nan, "empirical_p": np.nan})
            rows.append(row)
            continue
        exceed = 0
        for _ in range(int(n_permutations)):
            null_target = rng.permutation(target_values)
            null_rho, _ = stats.spearmanr(score, null_target)
            if np.isfinite(null_rho) and null_rho >= rho:
                exceed += 1
        row.update(
            {
                "spearman_r": float(rho),
                "spearman_p": float(p_value),
                "empirical_p": float((exceed + 1.0) / (n_permutations + 1.0)),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    tested_mask = out["status"].eq("tested")
    out.loc[tested_mask, "bh_fdr"] = _bh_adjust(out.loc[tested_mask, "empirical_p"].to_numpy(dtype=float))
    out.loc[~tested_mask, "bh_fdr"] = np.nan
    return out


def _hotspot_overlap_and_context(
    *,
    cells: pd.DataFrame,
    receiver_scores: pd.DataFrame,
    target_table: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    cell_niche = cells.set_index("cell_id")["niche"].astype(str)
    lr_meta = lr_pairs.set_index("lr_id").to_dict(orient="index")
    target = target_table.rename(columns={"cell_id": "receiver_id"})
    merged = receiver_scores.merge(
        target[["lr_id", "receiver_id", "target_activity", "target_status"]],
        on=["lr_id", "receiver_id"],
        how="left",
    )
    rows = []
    for (method, lr_id), group in merged.groupby(["method", "lr_id"], sort=False):
        group = group.dropna(subset=["incoming_score"]).copy()
        meta = lr_meta.get(lr_id, {})
        control_name = meta.get("control_name", lr_id)
        active_niche = str(meta.get("active_niche", "any"))
        top_pred = group.sort_values(["incoming_score", "receiver_id"], ascending=[False, True]).head(top_n)
        target_tested = group[group["target_status"].eq("tested")].dropna(subset=["target_activity"])
        top_target = target_tested.sort_values(["target_activity", "receiver_id"], ascending=[False, True]).head(top_n)
        pred_ids = set(top_pred["receiver_id"])
        target_ids = set(top_target["receiver_id"])
        overlap = len(pred_ids & target_ids)
        union = len(pred_ids | target_ids)
        global_target_mean = float(target_tested["target_activity"].mean()) if len(target_tested) else np.nan
        pred_target_mean = float(top_pred["target_activity"].mean()) if len(top_pred) else np.nan
        target_enrichment_delta = pred_target_mean - global_target_mean
        target_enrichment_fold = _safe_div(pred_target_mean, global_target_mean)
        if active_niche and active_niche != "any":
            global_active_fraction = float((cell_niche == active_niche).mean())
            pred_active_fraction = float(top_pred["niche"].astype(str).eq(active_niche).mean()) if len(top_pred) else np.nan
            active_niche_fold = _safe_div(pred_active_fraction, global_active_fraction)
        else:
            global_active_fraction = np.nan
            pred_active_fraction = np.nan
            active_niche_fold = np.nan
        niche_counts = top_pred["niche"].astype(str).value_counts()
        top_pred_niche = str(niche_counts.index[0]) if len(niche_counts) else ""
        top_pred_niche_fraction = float(niche_counts.iloc[0] / len(top_pred)) if len(top_pred) else np.nan
        rows.append(
            {
                "method": method,
                "method_label": METHOD_SPECS[method]["label"],
                "lr_id": lr_id,
                "control_name": control_name,
                "top_n": int(top_n),
                "target_status": target_tested["target_status"].iloc[0] if len(target_tested) else "not_tested",
                "hotspot_target_overlap_count": int(overlap),
                "hotspot_target_jaccard": _safe_div(overlap, union),
                "hotspot_target_precision": _safe_div(overlap, len(pred_ids)),
                "hotspot_target_recall": _safe_div(overlap, len(target_ids)),
                "pred_hotspot_target_activity_mean": pred_target_mean,
                "global_target_activity_mean": global_target_mean,
                "pred_hotspot_target_activity_delta": target_enrichment_delta,
                "pred_hotspot_target_activity_fold": target_enrichment_fold,
                "active_niche": active_niche,
                "pred_hotspot_active_niche_fraction": pred_active_fraction,
                "global_active_niche_fraction": global_active_fraction,
                "pred_hotspot_active_niche_fold": active_niche_fold,
                "top_pred_niche": top_pred_niche,
                "top_pred_niche_fraction": top_pred_niche_fraction,
            }
        )
    return pd.DataFrame(rows)


def _candidate_axis_scan(
    *,
    lr_pairs: pd.DataFrame,
    rank_table: pd.DataFrame,
    correlation: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    rank_cols = ["method", "lr_id", "rank_by_mean_score", "mean_score", "top100_edges", "top1pct_edges"]
    corr_cols = ["method", "lr_id", "spearman_r", "empirical_p", "bh_fdr"]
    overlap_cols = [
        "method",
        "lr_id",
        "hotspot_target_jaccard",
        "hotspot_target_precision",
        "pred_hotspot_target_activity_delta",
        "pred_hotspot_active_niche_fold",
        "top_pred_niche",
    ]
    merged = (
        rank_table[rank_cols]
        .merge(correlation[corr_cols], on=["method", "lr_id"], how="left")
        .merge(overlap[overlap_cols], on=["method", "lr_id"], how="left")
    )
    wide = {}
    for method in METHOD_SPECS:
        sub = merged[merged["method"].eq(method)].set_index("lr_id")
        wide[method] = sub

    rows = []
    positives = lr_pairs[lr_pairs["true_lr"].astype(bool)].copy()
    negatives = lr_pairs[lr_pairs["false_positive_high_expression"].astype(bool)].copy()
    for lr in positives.itertuples(index=False):
        lr_id = str(lr.lr_id)
        spatial = _row_or_empty(wide.get("spatialself"), lr_id)
        commot = _row_or_empty(wide.get("commot"), lr_id)
        cellnest = _row_or_empty(wide.get("cellnest"), lr_id)
        baseline_ranks = [commot.get("rank_by_mean_score", np.nan), cellnest.get("rank_by_mean_score", np.nan)]
        baseline_corrs = [commot.get("spearman_r", np.nan), cellnest.get("spearman_r", np.nan)]
        baseline_jaccards = [commot.get("hotspot_target_jaccard", np.nan), cellnest.get("hotspot_target_jaccard", np.nan)]
        baseline_niche_folds = [commot.get("pred_hotspot_active_niche_fold", np.nan), cellnest.get("pred_hotspot_active_niche_fold", np.nan)]

        best_baseline_rank = _nanmin(baseline_ranks)
        best_baseline_corr = _nanmax(baseline_corrs)
        best_baseline_jaccard = _nanmax(baseline_jaccards)
        best_baseline_niche_fold = _nanmax(baseline_niche_folds)

        neg_rows = []
        for neg in negatives.itertuples(index=False):
            neg_id = str(neg.lr_id)
            neg_spatial = _row_or_empty(wide.get("spatialself"), neg_id)
            neg_rows.append(
                {
                    "lr_id": neg_id,
                    "mean_score": neg_spatial.get("mean_score", np.nan),
                    "spearman_r": neg_spatial.get("spearman_r", np.nan),
                    "jaccard": neg_spatial.get("hotspot_target_jaccard", np.nan),
                }
            )
        max_negative_score = _nanmax([row["mean_score"] for row in neg_rows])
        max_negative_corr = _nanmax([row["spearman_r"] for row in neg_rows])
        max_negative_jaccard = _nanmax([row["jaccard"] for row in neg_rows])

        rank_advantage = best_baseline_rank - spatial.get("rank_by_mean_score", np.nan)
        corr_advantage = spatial.get("spearman_r", np.nan) - best_baseline_corr
        jaccard_advantage = spatial.get("hotspot_target_jaccard", np.nan) - best_baseline_jaccard
        niche_advantage = spatial.get("pred_hotspot_active_niche_fold", np.nan) - best_baseline_niche_fold
        negative_score_margin = spatial.get("mean_score", np.nan) - max_negative_score
        negative_corr_margin = spatial.get("spearman_r", np.nan) - max_negative_corr
        negative_jaccard_margin = spatial.get("hotspot_target_jaccard", np.nan) - max_negative_jaccard

        spatial_fdr = spatial.get("bh_fdr", np.nan)
        spatial_corr = spatial.get("spearman_r", np.nan)
        spatial_rank = spatial.get("rank_by_mean_score", np.nan)
        spatial_jaccard = spatial.get("hotspot_target_jaccard", np.nan)

        pass_rank = _is_finite(spatial_rank) and spatial_rank <= 3 and (
            not _is_finite(best_baseline_rank) or spatial_rank < best_baseline_rank
        )
        pass_function = _is_finite(spatial_corr) and spatial_corr > 0 and _is_finite(spatial_fdr) and spatial_fdr <= 0.1
        pass_location = _is_finite(jaccard_advantage) and jaccard_advantage > 0
        pass_negative = (
            (_is_finite(negative_score_margin) and negative_score_margin > 0)
            or (_is_finite(negative_corr_margin) and negative_corr_margin > 0)
            or (_is_finite(negative_jaccard_margin) and negative_jaccard_margin > 0)
        )

        priority_score = 0.0
        priority_score += _score_component(rank_advantage, scale=3.0)
        priority_score += _score_component(corr_advantage, scale=0.1)
        priority_score += _score_component(jaccard_advantage, scale=0.05)
        priority_score += 0.5 if pass_function else 0.0
        priority_score += 0.25 if pass_negative else 0.0
        priority_score += _score_component(niche_advantage, scale=1.0, cap=0.5)

        if pass_rank and pass_function and pass_location and pass_negative:
            tier = "1_strong_spatialself_specific_candidate"
        elif pass_function and (pass_location or pass_negative):
            tier = "2_functionally_supported_disagreement_candidate"
        elif pass_function:
            tier = "3_functionally_supported_but_not_specific"
        else:
            tier = "4_not_ready_for_story"

        rows.append(
            {
                "lr_id": lr_id,
                "control_name": lr.control_name,
                "mechanism": getattr(lr, "mechanism", ""),
                "active_niche": getattr(lr, "active_niche", ""),
                "spatialself_rank": spatial_rank,
                "commot_rank": commot.get("rank_by_mean_score", np.nan),
                "cellnest_rank": cellnest.get("rank_by_mean_score", np.nan),
                "best_baseline_rank": best_baseline_rank,
                "rank_advantage": rank_advantage,
                "spatialself_spearman": spatial_corr,
                "commot_spearman": commot.get("spearman_r", np.nan),
                "cellnest_spearman": cellnest.get("spearman_r", np.nan),
                "best_baseline_spearman": best_baseline_corr,
                "spearman_advantage": corr_advantage,
                "spatialself_bh_fdr": spatial_fdr,
                "spatialself_hotspot_target_jaccard": spatial_jaccard,
                "commot_hotspot_target_jaccard": commot.get("hotspot_target_jaccard", np.nan),
                "cellnest_hotspot_target_jaccard": cellnest.get("hotspot_target_jaccard", np.nan),
                "best_baseline_hotspot_target_jaccard": best_baseline_jaccard,
                "hotspot_jaccard_advantage": jaccard_advantage,
                "spatialself_active_niche_fold": spatial.get("pred_hotspot_active_niche_fold", np.nan),
                "best_baseline_active_niche_fold": best_baseline_niche_fold,
                "active_niche_fold_advantage": niche_advantage,
                "spatialself_top_pred_niche": spatial.get("top_pred_niche", ""),
                "max_spatialself_negative_score": max_negative_score,
                "max_spatialself_negative_spearman": max_negative_corr,
                "max_spatialself_negative_jaccard": max_negative_jaccard,
                "negative_score_margin": negative_score_margin,
                "negative_spearman_margin": negative_corr_margin,
                "negative_jaccard_margin": negative_jaccard_margin,
                "pass_rank_disagreement": bool(pass_rank),
                "pass_downstream_function": bool(pass_function),
                "pass_spatial_location": bool(pass_location),
                "pass_negative_control": bool(pass_negative),
                "priority_score": float(priority_score),
                "recommendation_tier": tier,
            }
        )
    return pd.DataFrame(rows)


def _write_status_markdown(
    *,
    output_dir: Path,
    dataset_dir: Path,
    cells: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    method_status: pd.DataFrame,
    candidate_scan: pd.DataFrame,
    top_n: int,
    n_permutations: int,
) -> dict:
    completed_methods = method_status[method_status["status"].isin(["completed", "completed_constant_score"])]
    sorted_candidates = candidate_scan.sort_values(["recommendation_tier", "priority_score"], ascending=[True, False])
    top_rows = sorted_candidates.head(5)
    best = top_rows.iloc[0].to_dict() if len(top_rows) else {}
    status_cols = ["method", "method_label", "status", "candidate_id_order_identical", "candidate_id_set_mismatch"]
    lines = [
        "# Phase 15 Visium HD CRC Disagreement Validation",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{_rel(dataset_dir)}`",
        f"- Platform: 10x Visium HD human colorectal cancer P2CRC, not human lymph node.",
        f"- Spots/bins: {len(cells):,}",
        f"- Candidate interactions: {len(candidates):,}",
        f"- LR/control pairs: {len(lr_pairs):,}",
        f"- Hotspot definition: top {top_n} receiver bins per method and LR.",
        f"- Downstream test: Spearman correlation with receiver target signature plus {n_permutations} target-label permutations.",
        "",
        "## Method Input Status",
        "",
        _markdown_table(method_status[status_cols]),
        "",
        "## Candidate Selection Rule",
        "",
        "- Scan all positive LR axes before selecting a story case.",
        "- Require evidence from rank or localization disagreement, downstream target response, and context/niche enrichment.",
        "- Treat all signals as observational support, not ground truth.",
        "",
        "## Top Candidate Axes",
        "",
    ]
    if len(top_rows):
        display_cols = [
            "control_name",
            "recommendation_tier",
            "priority_score",
            "spatialself_rank",
            "commot_rank",
            "cellnest_rank",
            "spatialself_spearman",
            "best_baseline_spearman",
            "spatialself_hotspot_target_jaccard",
            "best_baseline_hotspot_target_jaccard",
            "spatialself_bh_fdr",
        ]
        lines.append(_markdown_table(top_rows[display_cols], float_fmt=".4g"))
    else:
        lines.append("No candidate rows were generated.")
    lines.extend(
        [
            "",
            "## Current Readout",
            "",
            _interpret_best_candidate(best),
            "",
            "## Output Tables",
            "",
            "- `method_lr_rank_table.csv`",
            "- `downstream_correlation_by_method.csv`",
            "- `hotspot_overlap_with_target_signature.csv`",
            "- `candidate_axis_scan.csv`",
            "- `candidate_axis_recommendations.csv`",
            "",
            "## Claim Boundary",
            "",
            "This real Visium HD CRC sample has no perturbation-defined CCC ground truth. The valid claim is that one method's spatial localization is more consistent with receiver downstream response and CRC niche context.",
            "",
        ]
    )
    (output_dir / "phase15_visium_hd_crc_disagreement_status.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "n_completed_methods": int(len(completed_methods)),
        "best_candidate": str(best.get("control_name", "")),
        "best_candidate_tier": str(best.get("recommendation_tier", "")),
        "best_candidate_priority_score": _finite_or_nan(best.get("priority_score", np.nan)),
    }


def _interpret_best_candidate(row: dict) -> str:
    if not row:
        return "No interpretable candidate yet."
    name = row.get("control_name", "")
    tier = row.get("recommendation_tier", "")
    sp_rank = row.get("spatialself_rank", np.nan)
    commot_rank = row.get("commot_rank", np.nan)
    cellnest_rank = row.get("cellnest_rank", np.nan)
    sp_corr = row.get("spatialself_spearman", np.nan)
    base_corr = row.get("best_baseline_spearman", np.nan)
    sp_fdr = row.get("spatialself_bh_fdr", np.nan)
    return (
        f"Best current candidate is `{name}` ({tier}). SpatialSelf rank={_fmt(sp_rank)}, "
        f"COMMOT rank={_fmt(commot_rank)}, CellNEST rank={_fmt(cellnest_rank)}; "
        f"SpatialSelf downstream Spearman={_fmt(sp_corr)}, best baseline Spearman={_fmt(base_corr)}, "
        f"SpatialSelf BH-FDR={_fmt(sp_fdr)}. This is a candidate for manual biological review and figure design, not a final claim yet."
    )


def _markdown_table(table: pd.DataFrame, *, float_fmt: str = ".4g") -> str:
    if table.empty:
        return ""
    display = table.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), float_fmt))
    display = display.fillna("")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=False)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=False)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=False)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def _split_genes(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    text = str(raw).replace(";", ",").replace("|", ",")
    return [gene.strip() for gene in text.split(",") if gene.strip()]


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    p_finite = p[finite]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[np.where(finite)[0]] = restored
    return out


def _row_or_empty(frame: pd.DataFrame | None, lr_id: str) -> dict:
    if frame is None or lr_id not in frame.index:
        return {}
    return frame.loc[lr_id].to_dict()


def _safe_div(a: object, b: object) -> float:
    try:
        a_float = float(a)
        b_float = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(a_float) or not np.isfinite(b_float) or abs(b_float) < 1e-12:
        return float("nan")
    return float(a_float / b_float)


def _nanmin(values: list[object]) -> float:
    finite = [float(v) for v in values if _is_finite(v)]
    return float(min(finite)) if finite else float("nan")


def _nanmax(values: list[object]) -> float:
    finite = [float(v) for v in values if _is_finite(v)]
    return float(max(finite)) if finite else float("nan")


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _score_component(value: object, *, scale: float, cap: float = 1.0) -> float:
    if not _is_finite(value):
        return 0.0
    return float(np.clip(float(value) / float(scale), -cap, cap))


def _finite_or_nan(value: object) -> float:
    return float(value) if _is_finite(value) else float("nan")


def _fmt(value: object) -> str:
    if not _is_finite(value):
        return "NA"
    return f"{float(value):.3g}"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
