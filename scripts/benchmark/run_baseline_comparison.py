#!/usr/bin/env python
"""Run available baseline adapters on generated benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_commot import run_commot_on_replicate
from scripts.benchmark.run_cellchat import run_cellchat_on_replicate
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.benchmark.run_deeptalk import run_deeptalk_projection_on_replicate
from scripts.benchmark.run_spatalk import run_spatalk_on_replicate
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


AVAILABLE_METHODS = (
    "spatialself_wavelet_scale",
    "spatialself_wavelet",
    "spatialself_wavelet_adaptive",
    "spatialself_range_context",
    "spatialself_wavelet_edge_prior",
    "spatialself_gat",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
    "deeptalk",
)
PENDING_METHODS = {
    "nichenet": "installed; routed to downstream ligand-target validation via scripts/benchmark/extract_nichenet_targets.py, not a candidate-edge scorer",
    "niches": "optional blocked; package install failed in local R env",
}


def run_baselines_on_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    cot_nitermax: int,
    cellchat_nboot: int,
    spatialself_epochs: int,
    spatialself_k_nn: int,
    spatialself_max_train_samples: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    spatialself_feasibility_loss_weight: float,
    spatialself_hard_negative_loss_weight: float,
    spatialself_context_contrastive_loss_weight: float,
    spatialself_context_contrastive_margin: float,
    spatialself_context_contrastive_negative_mode: str,
    spatialself_null_candidate_loss_weight: float,
    spatialself_mechanism_guard_loss_weight: float,
    spatialself_lr_cap_positive: int,
    spatialself_lr_cap_negative: int,
    spatialself_no_lr_embedding: bool,
    spatialself_no_distance: bool,
    spatialself_no_spatial_graph: bool,
    spatialself_no_expression_features: bool,
    cellnest_num_epoch: int,
    cellnest_hidden: int,
    deeptalk_input_root: Path | None,
    spatalk_per_num: int,
    spatalk_min_pairs: int,
    spatalk_use_synthetic_pathways: bool,
) -> pd.DataFrame:
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    status = []
    for method in methods:
        if method in {
            "spatialself_wavelet_scale",
            "spatialself_wavelet",
            "spatialself_wavelet_adaptive",
            "spatialself_range_context",
            "spatialself_wavelet_edge_prior",
            "spatialself_gat",
        }:
            method_dir = output_dir / method
            is_edge_prior_candidate = method == "spatialself_wavelet_edge_prior"
            is_scale_baseline = method == "spatialself_wavelet_scale"
            model_name = "spatialself_wavelet" if is_edge_prior_candidate else method
            for replicate_dir in replicate_dirs:
                result = run_spatialself_on_replicate(
                    replicate_dir,
                    method_dir,
                    model_name=model_name,
                    k_nn=spatialself_k_nn,
                    epochs=spatialself_epochs,
                    max_train_samples=spatialself_max_train_samples,
                    pseudo_label_strategy=spatialself_pseudo_label_strategy,
                    ranking_loss_weight=spatialself_ranking_loss_weight,
                    ranking_margin=spatialself_ranking_margin,
                    feasibility_loss_weight=spatialself_feasibility_loss_weight,
                    hard_negative_loss_weight=spatialself_hard_negative_loss_weight,
                    context_contrastive_loss_weight=spatialself_context_contrastive_loss_weight,
                    context_contrastive_margin=spatialself_context_contrastive_margin,
                    context_contrastive_negative_mode=spatialself_context_contrastive_negative_mode,
                    null_candidate_loss_weight=spatialself_null_candidate_loss_weight,
                    mechanism_guard_loss_weight=spatialself_mechanism_guard_loss_weight,
                    lr_cap_positive=spatialself_lr_cap_positive,
                    lr_cap_negative=spatialself_lr_cap_negative,
                    use_lr_embedding=not spatialself_no_lr_embedding,
                    use_distance=not spatialself_no_distance,
                    use_spatial_graph=not spatialself_no_spatial_graph,
                    use_expression_features=not spatialself_no_expression_features,
                    use_edge_prior_features=is_edge_prior_candidate or is_scale_baseline,
                )
                if is_edge_prior_candidate:
                    result["method"] = method
                    result["model_name"] = model_name
                    _rewrite_prediction_method(Path(result["prediction_path"]), method)
                metrics.append(result)
            if is_scale_baseline:
                detail = "candidate-level SpatialSelf scale-aware wavelet baseline ran"
            elif is_edge_prior_candidate:
                detail = "candidate-level SpatialSelf edge-prior scorer candidate ran"
            else:
                detail = "candidate-level SpatialSelf adapter ran"
            status.append(_status(method, "completed", detail))
        elif method == "commot":
            method_dir = output_dir / "commot"
            for replicate_dir in replicate_dirs:
                metrics.append(
                    run_commot_on_replicate(
                        replicate_dir,
                        method_dir,
                        cot_nitermax=cot_nitermax,
                    )
            )
            status.append(_status(method, "completed", "candidate-level adapter ran"))
        elif method == "cellchat":
            method_dir = output_dir / "cellchat"
            for replicate_dir in replicate_dirs:
                metrics.append(
                    run_cellchat_on_replicate(
                        replicate_dir,
                        method_dir,
                        nboot=cellchat_nboot,
                    )
                )
            status.append(
                _status(
                    method,
                    "completed",
                    "cell-type pair CellChat scores projected to candidate edges",
                )
            )
        elif method == "spatalk":
            method_dir = output_dir / "spatalk"
            for replicate_dir in replicate_dirs:
                metrics.append(
                    run_spatalk_on_replicate(
                        replicate_dir,
                        method_dir,
                        per_num=spatalk_per_num,
                        min_pairs=spatalk_min_pairs,
                        use_synthetic_pathways=spatalk_use_synthetic_pathways,
                    )
                )
            status.append(
                _status(
                    method,
                    "completed",
                    "cell-type pair SpaTalk LR scores projected to candidate edges",
                )
            )
        elif method == "cellnest":
            method_dir = output_dir / "cellnest"
            for replicate_dir in replicate_dirs:
                metrics.append(
                    run_cellnest_on_replicate(
                        replicate_dir,
                        method_dir,
                        num_epoch=cellnest_num_epoch,
                        hidden=cellnest_hidden,
                    )
                )
            status.append(
                _status(
                    method,
                    "completed",
                    "CellNEST preprocess/train/postprocess output projected to candidate edges",
                )
            )
        elif method == "deeptalk":
            method_dir = output_dir / "deeptalk"
            for replicate_dir in replicate_dirs:
                metrics.append(
                    run_deeptalk_projection_on_replicate(
                        replicate_dir,
                        method_dir,
                        deeptalk_input_root=deeptalk_input_root,
                    )
                )
            status.append(
                _status(
                    method,
                    "completed_or_model_required",
                    "DeepTalk predict_ccc outputs projected when present; otherwise LR-specific model requirement recorded",
                )
            )
        elif method in PENDING_METHODS:
            if method == "nichenet":
                status_name = "downstream_validation_only"
            elif method == "niches":
                status_name = "optional_blocked"
            else:
                status_name = "pending_adapter"
            status.append(_status(method, status_name, PENDING_METHODS[method]))
        else:
            status.append(_status(method, "unknown_method", "method is not registered"))

    metrics_table = pd.DataFrame(metrics)
    if not metrics_table.empty:
        metrics_table.to_csv(output_dir / "baseline_metrics.csv", index=False)
        _summarize_metrics(metrics_table).to_csv(output_dir / "baseline_metrics_summary.csv", index=False)

    status_table = pd.DataFrame(status)
    status_table.to_csv(output_dir / "baseline_status.csv", index=False)
    with (output_dir / "baseline_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=True)
    return metrics_table


def _summarize_metrics(metrics_table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metric_columns if column in metrics_table.columns]
    if metrics_table.empty or "method" not in metrics_table or not available:
        return pd.DataFrame()
    summary = metrics_table.groupby("method", sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _status(method: str, status: str, detail: str) -> dict:
    return {"method": method, "status": status, "detail": detail}


def _rewrite_prediction_method(prediction_path: Path, method: str) -> None:
    prediction = pd.read_csv(prediction_path)
    if "method" in prediction.columns:
        prediction["method"] = method
        prediction.to_csv(prediction_path, index=False)


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    required_files = {
        "cells.csv",
        "expression.csv",
        "lr_pairs.csv",
        "candidate_interactions.csv",
    }
    if dataset_root.is_dir() and required_files.issubset({path.name for path in dataset_root.iterdir() if path.is_file()}):
        return [dataset_root]
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison",
        help="Directory for combined baseline outputs.",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method to run or record. Repeatable. Defaults to available adapters.",
    )
    parser.add_argument("--cot-nitermax", type=int, default=500)
    parser.add_argument("--cellchat-nboot", type=int, default=20)
    parser.add_argument("--spatialself-epochs", type=int, default=60)
    parser.add_argument("--spatialself-k-nn", type=int, default=12)
    parser.add_argument("--spatialself-max-train-samples", type=int, default=20000)
    parser.add_argument("--spatialself-pseudo-label-strategy", default="expression_distance")
    parser.add_argument("--spatialself-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-margin", type=float, default=1.0)
    parser.add_argument("--spatialself-feasibility-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-context-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-context-contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--spatialself-context-contrastive-negative-mode",
        choices=("random_receiver", "hard_receiver"),
        default="random_receiver",
    )
    parser.add_argument("--spatialself-null-candidate-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-mechanism-guard-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-lr-cap-positive", type=int, default=50)
    parser.add_argument("--spatialself-lr-cap-negative", type=int, default=200)
    parser.add_argument("--spatialself-no-lr-embedding", action="store_true")
    parser.add_argument("--spatialself-no-distance", action="store_true")
    parser.add_argument("--spatialself-no-spatial-graph", action="store_true")
    parser.add_argument("--spatialself-no-expression-features", action="store_true")
    parser.add_argument("--cellnest-num-epoch", type=int, default=2)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    parser.add_argument("--deeptalk-input-root", default="")
    parser.add_argument("--spatalk-per-num", type=int, default=100)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    args = parser.parse_args()

    methods = tuple(args.methods) if args.methods else AVAILABLE_METHODS
    metrics = run_baselines_on_dataset(
        Path(args.dataset_root),
        Path(args.output_dir),
        methods=methods,
        cot_nitermax=args.cot_nitermax,
        cellchat_nboot=args.cellchat_nboot,
        spatialself_epochs=args.spatialself_epochs,
        spatialself_k_nn=args.spatialself_k_nn,
        spatialself_max_train_samples=args.spatialself_max_train_samples,
        spatialself_pseudo_label_strategy=args.spatialself_pseudo_label_strategy,
        spatialself_ranking_loss_weight=args.spatialself_ranking_loss_weight,
        spatialself_ranking_margin=args.spatialself_ranking_margin,
        spatialself_feasibility_loss_weight=args.spatialself_feasibility_loss_weight,
        spatialself_hard_negative_loss_weight=args.spatialself_hard_negative_loss_weight,
        spatialself_context_contrastive_loss_weight=args.spatialself_context_contrastive_loss_weight,
        spatialself_context_contrastive_margin=args.spatialself_context_contrastive_margin,
        spatialself_context_contrastive_negative_mode=args.spatialself_context_contrastive_negative_mode,
        spatialself_null_candidate_loss_weight=args.spatialself_null_candidate_loss_weight,
        spatialself_mechanism_guard_loss_weight=args.spatialself_mechanism_guard_loss_weight,
        spatialself_lr_cap_positive=args.spatialself_lr_cap_positive,
        spatialself_lr_cap_negative=args.spatialself_lr_cap_negative,
        spatialself_no_lr_embedding=args.spatialself_no_lr_embedding,
        spatialself_no_distance=args.spatialself_no_distance,
        spatialself_no_spatial_graph=args.spatialself_no_spatial_graph,
        spatialself_no_expression_features=args.spatialself_no_expression_features,
        cellnest_num_epoch=args.cellnest_num_epoch,
        cellnest_hidden=args.cellnest_hidden,
        deeptalk_input_root=Path(args.deeptalk_input_root) if args.deeptalk_input_root else None,
        spatalk_per_num=args.spatalk_per_num,
        spatalk_min_pairs=args.spatalk_min_pairs,
        spatalk_use_synthetic_pathways=args.spatalk_use_synthetic_pathways,
    )
    if metrics.empty:
        print("No runnable baseline adapters selected.")
    else:
        columns = [
            "replicate",
            "method",
            "auroc",
            "aupr",
            "top_k_precision",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
        ]
        print(metrics[columns].to_string(index=False))
    print(f"Wrote baseline comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
