#!/usr/bin/env python
"""Summarize SPAMOSAIC tonsil Goal 5 method/control behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOAL_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
)
DEFAULT_SECTION_ID = "spamosaic_tonsil_A1"
DEFAULT_DATASET_DIR = DEFAULT_GOAL_DIR / f"standardized_inputs/{DEFAULT_SECTION_ID}_subset"

METHOD_TEMPLATES = {
    "spatialself_wavelet_edge_prior": {
        "label": "SpatialSelf",
        "prediction": "{section_id}_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
        "time_log": "{section_id}_spatialself_time.log",
    },
    "commot": {
        "label": "COMMOT",
        "prediction": "{section_id}_commot/replicate_00__commot_predictions.csv",
        "time_log": "{section_id}_commot_time.log",
    },
    "cellchat": {
        "label": "CellChat",
        "prediction": "{section_id}_cellchat/replicate_00__cellchat_predictions.csv",
        "time_log": "{section_id}_cellchat_time.log",
    },
    "spatalk": {
        "label": "SpaTalk",
        "prediction": "{section_id}_spatalk/replicate_00__spatalk_predictions.csv",
        "time_log": "{section_id}_spatalk_time.log",
    },
    "cellnest": {
        "label": "CellNEST",
        "prediction": "{section_id}_cellnest/replicate_00__cellnest_predictions.csv",
        "time_log": "{section_id}_cellnest_time.log",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(DEFAULT_GOAL_DIR))
    parser.add_argument("--section-id", default=DEFAULT_SECTION_ID)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    section_id = str(args.section_id)
    method_specs = _method_specs(section_id)
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")

    rank_table, status_table = _rank_methods(goal_dir, candidates, lr_pairs, method_specs)
    control_summary = _control_summary(rank_table)
    runtime_summary = _runtime_summary(goal_dir, status_table, method_specs)

    rank_path = goal_dir / f"{section_id}_method_lr_rank_table.csv"
    control_path = goal_dir / f"{section_id}_method_control_summary.csv"
    runtime_path = goal_dir / f"{section_id}_method_runtime_summary.csv"
    status_path = goal_dir / f"{section_id}_method_status_summary.csv"
    rank_table.to_csv(rank_path, index=False)
    control_summary.to_csv(control_path, index=False)
    runtime_summary.to_csv(runtime_path, index=False)
    status_table.to_csv(status_path, index=False)

    manifest = {
        "dataset_dir": _rel(dataset_dir),
        "section_id": section_id,
        "goal_dir": _rel(goal_dir),
        "n_methods_completed": int((status_table["status"] == "completed").sum()),
        "n_methods_missing": int((status_table["status"] != "completed").sum()),
        "outputs": {
            "rank_table": _rel(rank_path),
            "control_summary": _rel(control_path),
            "runtime_summary": _rel(runtime_path),
            "status_summary": _rel(status_path),
        },
        "claim_boundary": (
            "SPAMOSAIC tonsil has real RNA/ADT-derived coordinates and expression. Cell-type labels are ADT "
            "marker-rule proxies, not curated histology. This table compares LR/control ranking and high-expression "
            "negative-control behavior, not native CCC ground truth."
        ),
    }
    manifest_path = goal_dir / f"{section_id}_method_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(control_summary.to_string(index=False))


def _method_specs(section_id: str) -> dict[str, dict[str, str]]:
    return {
        method: {
            key: value.format(section_id=section_id) if isinstance(value, str) else value
            for key, value in spec.items()
        }
        for method, spec in METHOD_TEMPLATES.items()
    }


def _rank_methods(
    goal_dir: Path,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    method_specs: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    status_rows = []
    candidate_meta = candidates[["candidate_id", "lr_id"]].copy()
    for method, spec in method_specs.items():
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
        value = pred["score"].dropna().iloc[0] if pred["score"].notna().any() else "NA"
        return f"constant_score={value}"
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
                "weak_context_lr_rank": _rank_for(positives, "IL21-IL21R"),
                "rank_validation_status": _rank_validation_status(positives, negatives),
            }
        )
    return pd.DataFrame(rows)


def _rank_for(frame: pd.DataFrame, control_name: str) -> int | str:
    subset = frame[frame["control_name"].astype(str).eq(control_name)]
    if subset.empty:
        return ""
    return int(subset.iloc[0]["rank"])


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


def _runtime_summary(
    goal_dir: Path,
    status_table: pd.DataFrame,
    method_specs: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for method, spec in method_specs.items():
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
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
