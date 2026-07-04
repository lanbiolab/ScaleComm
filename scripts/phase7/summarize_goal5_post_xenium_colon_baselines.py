#!/usr/bin/env python
"""Summarize post-Xenium colon cancer Goal 5 method/control behavior."""

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
DEFAULT_DATASET_DIR = DEFAULT_GOAL_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"

METHOD_SPECS = {
    "spatialself_default": {
        "label": "SpatialSelf default",
        "prediction": "post_xenium_human_colon_cancer_baseline_panel/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_baseline_panel_time.log",
        "status_table": "post_xenium_human_colon_cancer_baseline_panel/baseline_status.csv",
    },
    "commot": {
        "label": "COMMOT",
        "prediction": "post_xenium_human_colon_cancer_baseline_panel/commot/post_xenium_human_colon_cancer_subset__commot_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_baseline_panel_time.log",
        "status_table": "post_xenium_human_colon_cancer_baseline_panel/baseline_status.csv",
    },
    "cellchat": {
        "label": "CellChat",
        "prediction": "post_xenium_human_colon_cancer_extended_baselines/cellchat/post_xenium_human_colon_cancer_subset__cellchat_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_extended_baselines_time.log",
        "status_table": "post_xenium_human_colon_cancer_extended_baselines/baseline_status.csv",
    },
    "spatalk": {
        "label": "SpaTalk",
        "prediction": "post_xenium_human_colon_cancer_extended_baselines/spatalk/post_xenium_human_colon_cancer_subset__spatalk_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_extended_baselines_time.log",
        "status_table": "post_xenium_human_colon_cancer_extended_baselines/baseline_status.csv",
    },
    "cellnest": {
        "label": "CellNEST",
        "prediction": "post_xenium_human_colon_cancer_extended_baselines/cellnest/post_xenium_human_colon_cancer_subset__cellnest_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_extended_baselines_time.log",
        "status_table": "post_xenium_human_colon_cancer_extended_baselines/baseline_status.csv",
    },
    "spatialself_calibrated": {
        "label": "SpatialSelf calibrated",
        "prediction": "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "time_log": "post_xenium_human_colon_cancer_spatialself_calibrated_time.log",
        "status_table": "post_xenium_human_colon_cancer_spatialself_calibrated/baseline_status.csv",
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

    rank_table.to_csv(goal_dir / "post_xenium_human_colon_cancer_method_lr_rank_table.csv", index=False)
    control_summary.to_csv(goal_dir / "post_xenium_human_colon_cancer_method_control_summary.csv", index=False)
    runtime_summary.to_csv(goal_dir / "post_xenium_human_colon_cancer_method_runtime_summary.csv", index=False)
    status_table.to_csv(goal_dir / "post_xenium_human_colon_cancer_method_status_summary.csv", index=False)

    manifest = {
        "dataset_dir": _rel(dataset_dir),
        "goal_dir": _rel(goal_dir),
        "n_methods_completed": int((status_table["status"] == "completed").sum()),
        "n_methods_missing": int((status_table["status"] != "completed").sum()),
        "outputs": {
            "rank_table": _rel(goal_dir / "post_xenium_human_colon_cancer_method_lr_rank_table.csv"),
            "control_summary": _rel(goal_dir / "post_xenium_human_colon_cancer_method_control_summary.csv"),
            "runtime_summary": _rel(goal_dir / "post_xenium_human_colon_cancer_method_runtime_summary.csv"),
            "status_summary": _rel(goal_dir / "post_xenium_human_colon_cancer_method_status_summary.csv"),
        },
        "claim_boundary": (
            "Post-Xenium colon cancer has no native CCC ground truth. The table compares LR/control ranking under "
            "real 10x Visium CytAssist coordinates and expression. The calibrated SpatialSelf row is a stress-test "
            "configuration using LR-prior negatives plus mechanism/niche pseudo-labeling; it needs downstream and FDR "
            "before being used as positive biological validation."
        ),
    }
    (goal_dir / "post_xenium_human_colon_cancer_method_summary_manifest.json").write_text(
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
        if not pred_path.exists():
            status = "prediction_missing"
            status_detail = "prediction file not found"
            method_scores = _empty_method_scores(lr_pairs)
        else:
            pred = pd.read_csv(pred_path)
            if "score" not in pred.columns:
                status = "score_column_missing"
                status_detail = "score column missing"
                method_scores = _empty_method_scores(lr_pairs)
            else:
                status = "completed"
                status_detail = _method_status_detail(pred)
                merged = candidate_meta.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
                method_scores = (
                    merged.groupby("lr_id", sort=False)["score"]
                    .agg(["mean", "max", "count"])
                    .reset_index()
                    .rename(columns={"mean": "mean_score", "max": "max_score", "count": "n_scored_candidates"})
                )
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
        method_table["top100_edges"] = _top_n_edges(candidates, pred_path, 100)
        method_table["rank"] = _rank_scores(method_table["mean_score"], method_table["lr_id"])
        method_table.insert(0, "method", method)
        method_table.insert(1, "method_label", spec["label"])
        method_table["method_status"] = status
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
    return _top_edges(candidates, prediction_path, quantile=0.9)


def _top_n_edges(candidates: pd.DataFrame, prediction_path: Path, n_edges: int) -> pd.Series:
    if not prediction_path.exists():
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    pred = pd.read_csv(prediction_path)
    if "score" not in pred.columns or pred.empty:
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    merged = candidates[["candidate_id", "lr_id"]].merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
    top = merged.sort_values("score", ascending=False).head(int(n_edges))
    counts = top.groupby("lr_id").size()
    return candidates["lr_id"].drop_duplicates().map(counts).fillna(0).astype(int).reset_index(drop=True)


def _top_edges(candidates: pd.DataFrame, prediction_path: Path, *, quantile: float) -> pd.Series:
    if not prediction_path.exists():
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    pred = pd.read_csv(prediction_path)
    if "score" not in pred.columns or pred.empty:
        return pd.Series(0, index=range(candidates["lr_id"].nunique()))
    merged = candidates[["candidate_id", "lr_id"]].merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
    if merged["score"].nunique(dropna=True) <= 1:
        counts = pd.Series(0, index=sorted(candidates["lr_id"].unique()))
    else:
        threshold = merged["score"].quantile(float(quantile))
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
                "positive_top3": int((positives["rank"] <= 3).sum()) if len(positives) else 0,
                "negative_top3": int((negatives["rank"] <= 3).sum()) if len(negatives) else 0,
                "positive_top100_edges": int(group.loc[group["true_lr"].astype(bool), "top100_edges"].sum()),
                "negative_top100_edges": int(group.loc[group["false_positive_high_expression"].astype(bool), "top100_edges"].sum()),
                "rank_validation_status": _rank_validation_status(positives, negatives),
            }
        )
    return pd.DataFrame(rows)


def _rank_validation_status(positives: pd.DataFrame, negatives: pd.DataFrame) -> str:
    status_detail = ""
    if "method_status_detail" in positives.columns and len(positives):
        status_detail = str(positives["method_status_detail"].iloc[0])
    elif "method_status_detail" in negatives.columns and len(negatives):
        status_detail = str(negatives["method_status_detail"].iloc[0])
    if status_detail.startswith("constant_score="):
        return "noninformative_constant_or_status_only_scores"
    if positives.empty:
        return "no_positive_controls"
    if negatives.empty:
        return "no_negative_controls"
    best_pos = float(positives["rank"].min())
    best_neg = float(negatives["rank"].min())
    negative_top3 = int((negatives["rank"] <= 3).sum())
    if best_pos < best_neg and negative_top3 == 0:
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
                "status": _find_status(status_table, method),
                "time_log": _rel(goal_dir / spec["time_log"]),
                "runtime": _runtime_from_time_log(goal_dir / spec["time_log"]),
            }
        )
    return pd.DataFrame(rows)


def _find_status(status_table: pd.DataFrame, method: str) -> str:
    row = status_table[status_table["method"] == method]
    if row.empty:
        return "not_available"
    return str(row.iloc[0]["status"])


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
