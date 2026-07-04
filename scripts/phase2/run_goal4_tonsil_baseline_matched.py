#!/usr/bin/env python
"""Run baseline-matched control ranking on the fresh tonsil subset."""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellnest import run_cellnest_on_replicate  # noqa: E402
from scripts.benchmark.run_cellchat import R_RUNNER as CELLCHAT_R_RUNNER  # noqa: E402
from scripts.benchmark.run_spatalk import R_RUNNER as SPATALK_R_RUNNER  # noqa: E402
from scripts.phase2.run_goal4_fresh_tonsil_subset import (  # noqa: E402
    _summarize_cell_type_pairs,
    _summarize_controls,
    _summarize_lr,
)
from src.spitialself.baselines import score_commot  # noqa: E402


DEFAULT_METHODS = (
    "spatialself_wavelet_calibrated",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh-root",
        default="outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset",
    )
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--operating-quantile", type=float, default=0.94)
    parser.add_argument("--commot-nitermax", type=int, default=1000)
    parser.add_argument("--cellchat-nboot", type=int, default=10)
    parser.add_argument("--spatalk-per-num", type=int, default=50)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--cellnest-num-epoch", type=int, default=2)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    fresh_root = PROJECT_ROOT / args.fresh_root
    replicate_dir = fresh_root / "constructed_replicate"
    output_dir = PROJECT_ROOT / args.output_dir
    prediction_root = output_dir / "predictions"
    summary_root = output_dir / "control_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    methods = tuple(args.methods) if args.methods else DEFAULT_METHODS
    _require_replicate(replicate_dir)
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    cells = pd.read_csv(replicate_dir / "cells.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")

    status_records: list[dict[str, object]] = []
    method_records: list[dict[str, object]] = []
    control_tables: list[pd.DataFrame] = []
    for method in methods:
        method_started = time.perf_counter()
        method_dir = prediction_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            prediction_path, run_status = _run_or_load_prediction(
                method,
                replicate_dir=replicate_dir,
                fresh_root=fresh_root,
                method_dir=method_dir,
                args=args,
            )
            prediction = pd.read_csv(prediction_path)
            _validate_prediction(prediction, candidates, method)
            annotated = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner")
            lr_summary = _summarize_lr(annotated, cells, operating_quantile=args.operating_quantile)
            cell_type_summary = _summarize_cell_type_pairs(
                annotated,
                operating_quantile=args.operating_quantile,
            )
            control_summary = _summarize_controls(
                lr_summary,
                annotated,
                operating_quantile=args.operating_quantile,
            )
            lr_summary.to_csv(summary_root / f"{method}__lr_summary.csv", index=False)
            cell_type_summary.to_csv(summary_root / f"{method}__cell_type_pair_summary.csv", index=False)
            control_summary.to_csv(summary_root / f"{method}__control_summary.csv", index=False)
            control_summary.insert(0, "method", method)
            control_tables.append(control_summary)
            method_records.append(
                _method_summary_row(
                    method=method,
                    prediction=prediction,
                    control_summary=control_summary,
                    cell_type_summary=cell_type_summary,
                    method_seconds=time.perf_counter() - method_started,
                    prediction_path=prediction_path,
                    status=run_status,
                )
            )
            status_records.append(
                _status_row(
                    method,
                    "completed",
                    run_status,
                    prediction_path,
                    time.perf_counter() - method_started,
                    prediction,
                )
            )
        except Exception as exc:  # noqa: BLE001 - external baseline failures are evidence.
            status_records.append(
                {
                    "method": method,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "prediction_path": "",
                    "n_prediction_rows": 0,
                    "n_unique_scores": 0,
                    "n_nonzero_scores": 0,
                    "wall_clock_seconds": float(time.perf_counter() - method_started),
                    "claim_eligible": False,
                }
            )

    method_summary = pd.DataFrame(method_records)
    status_table = pd.DataFrame(status_records)
    control_lr_summary = pd.concat(control_tables, ignore_index=True) if control_tables else pd.DataFrame()
    overview = _overview_summary(
        method_summary,
        status_table,
        replicate_dir=replicate_dir,
        started=started,
        output_dir=output_dir,
    )
    method_summary.to_csv(output_dir / "tonsil_baseline_matched_method_summary.csv", index=False)
    status_table.to_csv(output_dir / "tonsil_baseline_matched_status.csv", index=False)
    control_lr_summary.to_csv(output_dir / "tonsil_baseline_matched_control_lr_summary.csv", index=False)
    overview.to_csv(output_dir / "tonsil_baseline_matched_summary.csv", index=False)
    manifest_path = output_dir / "tonsil_baseline_matched_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "goal_id": "4",
                "analysis_label": "tonsil_baseline_matched_fresh_subset",
                "evidence_status": "fresh_subset_baseline_matched_initial",
                "fresh_root": str(fresh_root.relative_to(PROJECT_ROOT)),
                "replicate_dir": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "methods_requested": list(methods),
                "operating_quantile": float(args.operating_quantile),
                "method_summary": str(
                    (output_dir / "tonsil_baseline_matched_method_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "status_table": str(
                    (output_dir / "tonsil_baseline_matched_status.csv").relative_to(PROJECT_ROOT)
                ),
                "control_lr_summary": str(
                    (output_dir / "tonsil_baseline_matched_control_lr_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "overview_summary": str((output_dir / "tonsil_baseline_matched_summary.csv").relative_to(PROJECT_ROOT)),
                "limitations": [
                    "Single fresh tonsil subset with biological positive/negative controls, not ground-truth AUROC/AUPR.",
                    "CellChat and SpaTalk are cell-type-pair scores projected to candidate edges.",
                    "CellNEST depends on the retained local CellNEST installation and records failure status if unavailable.",
                    "Use this as high-resolution subset validation support, not final baseline superiority evidence.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(method_summary.to_string(index=False) if not method_summary.empty else "No completed methods")
    print(status_table.to_string(index=False))
    print(f"Wrote tonsil baseline-matched outputs to {output_dir}")


def _require_replicate(replicate_dir: Path) -> None:
    missing = [
        name
        for name in ["cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"]
        if not (replicate_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Fresh tonsil replicate is missing files: {missing}")


def _run_or_load_prediction(
    method: str,
    *,
    replicate_dir: Path,
    fresh_root: Path,
    method_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if method == "spatialself_wavelet_calibrated":
        source = fresh_root / "tonsil_fresh_spatialself_wavelet_predictions.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        target = method_dir / f"{replicate_dir.name}__spatialself_wavelet_calibrated_predictions.csv"
        prediction = pd.read_csv(source)
        prediction["method"] = method
        prediction.to_csv(target, index=False)
        return target, "reused_fresh_phase2_default_prediction"
    if method == "commot":
        cells = pd.read_csv(replicate_dir / "cells.csv")
        expression = pd.read_csv(replicate_dir / "expression.csv")
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        prediction = score_commot(
            cells,
            expression,
            lr_pairs,
            candidates,
            cot_nitermax=args.commot_nitermax,
        )
        prediction_path = method_dir / f"{replicate_dir.name}__commot_predictions.csv"
        prediction.to_csv(prediction_path, index=False)
        return prediction_path, "candidate_edge_commot_adapter_no_ground_truth_metrics"
    if method == "cellchat":
        prediction_path = method_dir / f"{replicate_dir.name}__cellchat_predictions.csv"
        _run_r_prediction(
            [
                str(CELLCHAT_R_RUNNER),
                str(replicate_dir / "cells.csv"),
                str(replicate_dir / "expression.csv"),
                str(replicate_dir / "lr_pairs.csv"),
                str(replicate_dir / "candidate_interactions.csv"),
                str(prediction_path),
                str(int(args.cellchat_nboot)),
            ],
            conda_env="cellchat_env",
            log_path=method_dir / f"{replicate_dir.name}__cellchat.log",
        )
        return prediction_path, "cell_type_pair_projected_cellchat_adapter_no_ground_truth_metrics"
    if method == "spatalk":
        prediction_path = method_dir / f"{replicate_dir.name}__spatalk_predictions.csv"
        _run_r_prediction(
            [
                str(SPATALK_R_RUNNER),
                str(replicate_dir / "cells.csv"),
                str(replicate_dir / "expression.csv"),
                str(replicate_dir / "lr_pairs.csv"),
                str(replicate_dir / "candidate_interactions.csv"),
                str(prediction_path),
                str(int(args.spatalk_per_num)),
                str(int(args.spatalk_min_pairs)),
                "FALSE",
            ],
            conda_env="spatalk_env",
            log_path=method_dir / f"{replicate_dir.name}__spatalk.log",
        )
        return prediction_path, "cell_type_pair_projected_spatalk_adapter_no_ground_truth_metrics"
    if method == "cellnest":
        result = run_cellnest_on_replicate(
            replicate_dir,
            method_dir,
            num_epoch=args.cellnest_num_epoch,
            hidden=args.cellnest_hidden,
        )
        return Path(result["prediction_path"]), "cellnest_projected_candidate_edge_adapter"
    raise ValueError(f"Unknown method: {method}")


def _run_r_prediction(args: list[str], *, conda_env: str, log_path: Path) -> None:
    cmd = ["conda", "run", "-n", conda_env, "Rscript", *args]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, text=True, capture_output=True)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(cmd)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"R prediction failed with exit code {completed.returncode}; see {log_path}")


def _validate_prediction(prediction: pd.DataFrame, candidates: pd.DataFrame, method: str) -> None:
    missing = {"candidate_id", "score"} - set(prediction.columns)
    if missing:
        raise ValueError(f"{method} prediction missing columns: {sorted(missing)}")
    if len(prediction) != len(candidates):
        raise ValueError(f"{method} prediction row count mismatch: {len(prediction)} != {len(candidates)}")
    overlap = prediction["candidate_id"].isin(candidates["candidate_id"]).sum()
    if int(overlap) != len(candidates):
        raise ValueError(f"{method} prediction candidate overlap mismatch: {overlap} != {len(candidates)}")


def _method_summary_row(
    *,
    method: str,
    prediction: pd.DataFrame,
    control_summary: pd.DataFrame,
    cell_type_summary: pd.DataFrame,
    method_seconds: float,
    prediction_path: Path,
    status: str,
) -> dict[str, object]:
    positives = control_summary[control_summary["control_class"] == "tonsil_positive_prior"]
    negatives = control_summary[control_summary["control_class"] != "tonsil_positive_prior"]
    pos_top_decile = positives[positives["rank_percentile_by_score_mean"] <= 0.10]
    neg_top_decile = negatives[negatives["rank_percentile_by_score_mean"] <= 0.10]
    best_positive = positives.sort_values("rank_by_score_mean").iloc[0].to_dict() if not positives.empty else {}
    best_negative = negatives.sort_values("rank_by_score_mean").iloc[0].to_dict() if not negatives.empty else {}
    top_pairs = "; ".join(
        f"{row.sender_cell_type}->{row.receiver_cell_type}:{int(row.n_top_edges)}"
        for row in cell_type_summary.head(3).itertuples(index=False)
    )
    return {
        "method": method,
        "status": "completed",
        "method_scope": status,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "n_prediction_rows": int(len(prediction)),
        "n_unique_scores": int(prediction["score"].nunique()),
        "n_nonzero_scores": int((prediction["score"].astype(float) != 0).sum()),
        "n_positive_controls_top_decile": int(len(pos_top_decile)),
        "n_negative_controls_top_decile": int(len(neg_top_decile)),
        "best_positive_control": best_positive.get("control_name", ""),
        "best_positive_control_rank": best_positive.get("rank_by_score_mean", math.nan),
        "best_positive_rank_percentile": best_positive.get("rank_percentile_by_score_mean", math.nan),
        "best_negative_control": best_negative.get("control_name", ""),
        "best_negative_control_rank": best_negative.get("rank_by_score_mean", math.nan),
        "best_negative_rank_percentile": best_negative.get("rank_percentile_by_score_mean", math.nan),
        "median_positive_rank_percentile": float(positives["rank_percentile_by_score_mean"].median())
        if not positives.empty
        else math.nan,
        "median_negative_rank_percentile": float(negatives["rank_percentile_by_score_mean"].median())
        if not negatives.empty
        else math.nan,
        "positive_controls_rank_ahead_of_negatives": bool(
            not positives.empty
            and not negatives.empty
            and float(positives["rank_by_score_mean"].max()) < float(negatives["rank_by_score_mean"].min())
        ),
        "negative_top3_count": int((negatives["rank_by_score_mean"].astype(float) <= 3).sum())
        if not negatives.empty
        else 0,
        "top_cell_type_pairs": top_pairs,
        "wall_clock_seconds": float(method_seconds),
        "rank_validation_pass": bool(len(pos_top_decile) >= 1 and len(neg_top_decile) == 0),
    }


def _status_row(
    method: str,
    status: str,
    detail: str,
    prediction_path: Path,
    seconds: float,
    prediction: pd.DataFrame,
) -> dict[str, object]:
    return {
        "method": method,
        "status": status,
        "detail": detail,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "n_prediction_rows": int(len(prediction)),
        "n_unique_scores": int(prediction["score"].nunique()),
        "n_nonzero_scores": int((prediction["score"].astype(float) != 0).sum()),
        "wall_clock_seconds": float(seconds),
        "claim_eligible": bool(status == "completed" and prediction["score"].nunique() > 1),
    }


def _overview_summary(
    method_summary: pd.DataFrame,
    status_table: pd.DataFrame,
    *,
    replicate_dir: Path,
    started: float,
    output_dir: Path,
) -> pd.DataFrame:
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    completed = method_summary if not method_summary.empty else pd.DataFrame()
    spatialself = (
        completed[completed["method"] == "spatialself_wavelet_calibrated"]
        if not completed.empty
        else pd.DataFrame()
    )
    best = (
        completed.sort_values(
            ["rank_validation_pass", "n_positive_controls_top_decile", "n_negative_controls_top_decile"],
            ascending=[False, False, True],
        ).iloc[0]
        if not completed.empty
        else {}
    )
    return pd.DataFrame(
        [
            {
                "goal_id": "4",
                "analysis_label": "tonsil_baseline_matched_fresh_subset",
                "evidence_status": "fresh_subset_baseline_matched_initial",
                "dataset_context": "human_tonsil_slide_tags_fresh_subset_controls",
                "n_cells": int(pd.read_csv(replicate_dir / "cells.csv", usecols=["cell_id"]).shape[0]),
                "n_lr_pairs": int(pd.read_csv(replicate_dir / "lr_pairs.csv", usecols=["lr_id"]).shape[0]),
                "n_candidate_edges": int(
                    pd.read_csv(replicate_dir / "candidate_interactions.csv", usecols=["candidate_id"]).shape[0]
                ),
                "n_methods_requested": int(len(status_table)),
                "n_methods_completed": int((status_table["status"] == "completed").sum()) if not status_table.empty else 0,
                "n_methods_failed": int((status_table["status"] == "failed").sum()) if not status_table.empty else 0,
                "n_methods_rank_validation_pass": int(completed["rank_validation_pass"].sum())
                if not completed.empty
                else 0,
                "best_rank_validation_method": best.get("method", "NA"),
                "spatialself_rank_validation_pass": bool(spatialself["rank_validation_pass"].iloc[0])
                if not spatialself.empty
                else False,
                "spatialself_positive_controls_top_decile": int(spatialself["n_positive_controls_top_decile"].iloc[0])
                if not spatialself.empty
                else 0,
                "spatialself_negative_controls_top_decile": int(spatialself["n_negative_controls_top_decile"].iloc[0])
                if not spatialself.empty
                else 0,
                "wall_clock_seconds": float(time.perf_counter() - started),
                "peak_rss_mb": float(peak_rss_mb),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "interpretation": (
                    "Initial baseline-matched fresh tonsil subset control-ranking evidence; "
                    "not a ground-truth AUROC/AUPR benchmark and not final baseline superiority evidence."
                ),
            }
        ]
    )


if __name__ == "__main__":
    main()
