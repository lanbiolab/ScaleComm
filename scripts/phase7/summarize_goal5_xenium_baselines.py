#!/usr/bin/env python
"""Summarize Xenium Phase 7 Goal 5 method ranking and control behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOAL_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
)
DEFAULT_DATASET_DIR = DEFAULT_GOAL_DIR / "standardized_inputs/xenium_human_lymph_node_subset"


METHOD_SPECS = {
    "spatialself_wavelet_edge_prior": {
        "label": "SpatialSelf",
        "prediction": "xenium_human_lymph_node_baseline_panel/spatialself_wavelet_edge_prior/xenium_human_lymph_node_subset__spatialself_wavelet_predictions.csv",
        "time_log": "xenium_human_lymph_node_baseline_panel_time.log",
    },
    "commot": {
        "label": "COMMOT",
        "prediction": "xenium_human_lymph_node_commot/xenium_human_lymph_node_replicate_00__commot_predictions.csv",
        "time_log": "xenium_human_lymph_node_commot_time.log",
    },
    "cellchat": {
        "label": "CellChat",
        "prediction": "xenium_human_lymph_node_cellchat/cellchat/xenium_human_lymph_node_subset__cellchat_predictions.csv",
        "time_log": "xenium_human_lymph_node_cellchat_time.log",
    },
    "spatalk": {
        "label": "SpaTalk",
        "prediction": "xenium_human_lymph_node_spatalk/spatalk/xenium_human_lymph_node_subset__spatalk_predictions.csv",
        "time_log": "xenium_human_lymph_node_spatalk_time.log",
    },
    "cellnest": {
        "label": "CellNEST",
        "prediction": "xenium_human_lymph_node_cellnest/cellnest/xenium_human_lymph_node_subset__cellnest_predictions.csv",
        "time_log": "xenium_human_lymph_node_cellnest_time.log",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(DEFAULT_GOAL_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")

    rank_table, status_table = _rank_methods(goal_dir, candidates, lr_pairs)
    control_summary = _control_summary(rank_table)
    runtime_summary = _runtime_summary(goal_dir, status_table)

    rank_table.to_csv(goal_dir / "xenium_human_lymph_node_method_lr_rank_table.csv", index=False)
    control_summary.to_csv(goal_dir / "xenium_human_lymph_node_method_control_summary.csv", index=False)
    runtime_summary.to_csv(goal_dir / "xenium_human_lymph_node_method_runtime_summary.csv", index=False)
    status_table.to_csv(goal_dir / "xenium_human_lymph_node_method_status_summary.csv", index=False)

    manifest = {
        "dataset_dir": _rel(dataset_dir),
        "goal_dir": _rel(goal_dir),
        "n_methods_completed": int((status_table["status"] == "completed").sum()),
        "n_methods_missing": int((status_table["status"] != "completed").sum()),
        "outputs": {
            "rank_table": _rel(goal_dir / "xenium_human_lymph_node_method_lr_rank_table.csv"),
            "control_summary": _rel(goal_dir / "xenium_human_lymph_node_method_control_summary.csv"),
            "runtime_summary": _rel(goal_dir / "xenium_human_lymph_node_method_runtime_summary.csv"),
            "status_summary": _rel(goal_dir / "xenium_human_lymph_node_method_status_summary.csv"),
        },
        "claim_boundary": (
            "Xenium has no native CCC ground truth. These tables summarize LR/control ranking and runtime/status, "
            "not AUROC/AUPR benchmark truth."
        ),
    }
    (goal_dir / "xenium_human_lymph_node_method_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(control_summary.to_string(index=False))


def _rank_methods(goal_dir: Path, candidates: pd.DataFrame, lr_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    status_rows = []
    candidate_meta = candidates[["candidate_id", "lr_id"]].copy()
    for method, spec in METHOD_SPECS.items():
        pred_path = goal_dir / spec["prediction"]
        status_detail = "prediction file not found"
        if not pred_path.exists():
            status_rows.append(
                {
                    "method": method,
                    "method_label": spec["label"],
                    "status": "prediction_missing",
                    "prediction_path": _rel(pred_path),
                    "status_detail": status_detail,
                }
            )
            method_scores = _empty_method_scores(lr_pairs)
        else:
            pred = pd.read_csv(pred_path)
            score_col = "score" if "score" in pred.columns else ""
            if not score_col:
                status = "score_column_missing"
                method_scores = _empty_method_scores(lr_pairs)
            else:
                status = "completed"
                merged = candidate_meta.merge(
                    pred[["candidate_id", score_col]],
                    on="candidate_id",
                    how="left",
                )
                method_scores = (
                    merged.groupby("lr_id", sort=False)[score_col]
                    .agg(["mean", "max", "count"])
                    .reset_index()
                    .rename(columns={"mean": "mean_score", "max": "max_score", "count": "n_scored_candidates"})
                )
            status_detail = _method_status_detail(pred)
            status_rows.append(
                {
                    "method": method,
                    "method_label": spec["label"],
                    "status": status,
                    "prediction_path": _rel(pred_path),
                    "status_detail": status_detail,
                }
            )
        method_table = lr_pairs.merge(method_scores, on="lr_id", how="left")
        method_table["mean_score"] = pd.to_numeric(method_table["mean_score"], errors="coerce")
        method_table["max_score"] = pd.to_numeric(method_table["max_score"], errors="coerce")
        method_table["n_candidates"] = method_table["lr_id"].map(candidates.groupby("lr_id").size()).fillna(0).astype(int)
        method_table["n_scored_candidates"] = method_table["n_scored_candidates"].fillna(0).astype(int)
        method_table["top_decile_edges"] = _top_decile_edges(candidates, pred_path)
        method_table["rank"] = _rank_scores(method_table["mean_score"], method_table["lr_id"])
        method_table.insert(0, "method", method)
        method_table.insert(1, "method_label", spec["label"])
        method_table["method_status"] = status_rows[-1]["status"]
        method_table["method_status_detail"] = status_detail
        rows.append(method_table)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(status_rows)


def _empty_method_scores(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lr_id": lr_pairs["lr_id"],
            "mean_score": np.nan,
            "max_score": np.nan,
            "n_scored_candidates": 0,
        }
    )


def _method_status_detail(pred: pd.DataFrame) -> str:
    if "spatalk_status" in pred.columns:
        values = sorted(set(pred["spatalk_status"].dropna().astype(str)))
        if values:
            return ";".join(values[:3])
    if "score" not in pred.columns:
        return "score column missing"
    if pred["score"].nunique(dropna=True) <= 1:
        return f"constant_score={pred['score'].dropna().iloc[0] if pred['score'].notna().any() else 'NA'}"
    return "candidate_scores_available"


def _rank_scores(scores: pd.Series, lr_ids: pd.Series) -> pd.Series:
    table = pd.DataFrame({"lr_id": lr_ids, "score": scores})
    scored = table.dropna(subset=["score"]).sort_values(["score", "lr_id"], ascending=[False, True])
    ranks = {lr_id: rank for rank, lr_id in enumerate(scored["lr_id"], start=1)}
    absent_rank = len(table) + 1
    return table["lr_id"].map(lambda value: ranks.get(value, absent_rank)).astype(int)


def _top_decile_edges(candidates: pd.DataFrame, prediction_path: Path) -> pd.Series:
    if not prediction_path.exists():
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    pred = pd.read_csv(prediction_path)
    if "score" not in pred.columns or pred.empty:
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    merged = candidates[["candidate_id", "lr_id"]].merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
    if merged["score"].nunique(dropna=True) <= 1:
        counts = pd.Series(0, index=sorted(candidates["lr_id"].unique()))
    else:
        threshold = merged["score"].quantile(0.9)
        counts = merged.loc[merged["score"] >= threshold].groupby("lr_id").size()
    return candidates["lr_id"].drop_duplicates().map(counts).fillna(0).astype(int).reset_index(drop=True)


def _control_summary(rank_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in rank_table.groupby("method", sort=False):
        positives = group[group["true_lr"].astype(bool)].sort_values(["rank", "control_name"])
        negatives = group[group["false_positive_high_expression"].astype(bool)].sort_values(["rank", "control_name"])
        best_pos = positives.iloc[0].to_dict() if len(positives) else {}
        best_neg = negatives.iloc[0].to_dict() if len(negatives) else {}
        rows.append(
            {
                "method": method,
                "method_label": group["method_label"].iloc[0],
                "method_status": group["method_status"].iloc[0],
                "method_status_detail": group["method_status_detail"].iloc[0],
                "best_positive_control": best_pos.get("control_name", ""),
                "best_positive_rank": best_pos.get("rank", np.nan),
                "best_positive_mean_score": best_pos.get("mean_score", np.nan),
                "best_negative_control": best_neg.get("control_name", ""),
                "best_negative_rank": best_neg.get("rank", np.nan),
                "best_negative_mean_score": best_neg.get("mean_score", np.nan),
                "negative_top3": int((negatives["rank"] <= 3).sum()) if len(negatives) else 0,
                "positive_top3": int((positives["rank"] <= 3).sum()) if len(positives) else 0,
                "negative_top1": int((negatives["rank"] <= 1).sum()) if len(negatives) else 0,
                "positive_top1": int((positives["rank"] <= 1).sum()) if len(positives) else 0,
                "rank_validation_status": _rank_validation_status(
                    positives,
                    negatives,
                    str(group["method_status_detail"].iloc[0]),
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_validation_status(positives: pd.DataFrame, negatives: pd.DataFrame, status_detail: str) -> str:
    if status_detail.startswith("constant_score=") or status_detail.startswith("no_spatalk_scores:"):
        return "noninformative_constant_or_status_only_scores"
    if positives.empty:
        return "no_positive_controls"
    if negatives.empty:
        return "no_negative_controls"
    best_pos = float(positives["rank"].min())
    best_neg = float(negatives["rank"].min())
    if best_pos < best_neg and int((negatives["rank"] <= 3).sum()) == 0:
        return "positive_prioritized_negative_suppressed"
    if best_pos < best_neg:
        return "positive_prioritized_but_negative_top3"
    return "negative_control_ranks_as_high_or_higher"


def _runtime_summary(goal_dir: Path, status_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, spec in METHOD_SPECS.items():
        rows.append(
            {
                "method": method,
                "method_label": spec["label"],
                "time_log": _rel(goal_dir / spec["time_log"]),
                "runtime": _runtime_from_time_log(goal_dir / spec["time_log"]),
                "status": status_table.loc[status_table["method"] == method, "status"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _runtime_from_time_log(path: Path) -> str:
    if not path.exists():
        return "not_yet_run"
    wall = ""
    rss = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Elapsed (wall clock) time" in line:
            wall = line.split("):", 1)[-1].strip() if "):" in line else line.rsplit(":", 1)[-1].strip()
        if "Maximum resident set size" in line:
            rss = line.split(":", 1)[-1].strip()
    return f"wall_clock={wall};max_rss_kb={rss}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
