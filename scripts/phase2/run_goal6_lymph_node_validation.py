#!/usr/bin/env python
"""Run Phase 2 lymph-node real-background injected-control validation."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_downstream_target_correlation import run_downstream_target_correlation
from scripts.benchmark.run_region_enrichment import run_region_enrichment
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
from scripts.benchmark.summarize_real_validation_controls import summarize_real_validation_controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate-dir",
        default=(
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/"
            "human_lymph_node_visium_replicate_00"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--lr-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--negative-quantile", type=float, default=0.20)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--region-permutations", type=int, default=100)
    parser.add_argument("--downstream-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--method-label", default="spatialself_wavelet_edge_prior")
    parser.add_argument(
        "--pathway-csv",
        default="outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_nichenet_targets/lr_pairs__nichenet_targets.csv",
    )
    parser.add_argument(
        "--reactome-tf-ppr-csv",
        default="",
    )
    parser.add_argument("--trrust-dir", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    replicate_dir = PROJECT_ROOT / args.replicate_dir
    output_dir = PROJECT_ROOT / args.output_dir
    prediction_dir = output_dir / "prediction_inputs"
    region_dir = output_dir / "region_enrichment"
    downstream_dir = output_dir / "downstream_target"
    control_dir = output_dir / "control_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_spatialself_on_replicate(
        replicate_dir,
        prediction_dir,
        model_name="spatialself_wavelet",
        k_nn=args.k_nn,
        h_dim=args.h_dim,
        lr_dim=args.lr_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
        pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
        ranking_loss_weight=0.35,
        ranking_margin=0.5,
        lr_cap_positive=50,
        lr_cap_negative=200,
        use_edge_prior_features=True,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    raw_prediction_path = Path(metrics["prediction_path"])
    prediction_path = _relabel_prediction(raw_prediction_path, args.method_label)
    metrics["method"] = args.method_label
    metrics["prediction_path"] = str(prediction_path)
    metrics["raw_prediction_path"] = str(raw_prediction_path)
    metrics["use_edge_prior_features"] = True
    pd.DataFrame([metrics]).to_csv(output_dir / "lymph_node_spatialself_metrics.csv", index=False)

    candidate_path = replicate_dir / "candidate_interactions.csv"
    cells_path = replicate_dir / "cells.csv"
    expression_path = replicate_dir / "expression.csv"
    lr_pairs_path = replicate_dir / "lr_pairs.csv"

    region_enrichment, lr_ranking = run_region_enrichment(
        candidate_path,
        prediction_path,
        cells_path,
        region_dir,
        region_column="region",
        receiver_region=True,
        n_permutations=args.region_permutations,
        random_seed=args.random_seed,
    )
    downstream = run_downstream_target_correlation(
        candidate_path,
        prediction_path,
        expression_path,
        downstream_dir,
        pathway_csv=PROJECT_ROOT / args.pathway_csv if args.pathway_csv else None,
        reactome_tf_ppr_csv=PROJECT_ROOT / args.reactome_tf_ppr_csv if args.reactome_tf_ppr_csv else None,
        trrust_dir=PROJECT_ROOT / args.trrust_dir if args.trrust_dir else None,
        n_permutations=args.downstream_permutations,
        random_seed=args.random_seed,
    )
    control_lr, control_method = summarize_real_validation_controls(
        lr_pairs_path,
        [region_dir / f"{prediction_path.stem}__lr_global_ranking.csv"],
        [region_dir / f"{prediction_path.stem}__region_enrichment.csv"],
        [downstream_dir / f"{prediction_path.stem}__downstream_target_correlation.csv"],
        control_dir,
    )
    summary = _summary_table(
        replicate_dir=replicate_dir,
        output_dir=output_dir,
        metrics=metrics,
        control_method=control_method,
        downstream=downstream,
        lr_ranking=lr_ranking,
        region_enrichment=region_enrichment,
        started=started,
    )
    summary_path = output_dir / "lymph_node_validation_summary.csv"
    manifest_path = output_dir / "lymph_node_validation_manifest.json"
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "analysis_label": "phase2_goal6_lymph_node_validation",
                "evidence_status": "phase2_initial_lymph_node_validation",
                "replicate_dir": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "metrics_path": str((output_dir / "lymph_node_spatialself_metrics.csv").relative_to(PROJECT_ROOT)),
                "region_enrichment_path": str(
                    (region_dir / f"{prediction_path.stem}__region_enrichment.csv").relative_to(PROJECT_ROOT)
                ),
                "lr_global_ranking_path": str(
                    (region_dir / f"{prediction_path.stem}__lr_global_ranking.csv").relative_to(PROJECT_ROOT)
                ),
                "downstream_target_path": str(
                    (downstream_dir / f"{prediction_path.stem}__downstream_target_correlation.csv").relative_to(
                        PROJECT_ROOT
                    )
                ),
                "control_lr_summary_path": str(
                    (control_dir / "real_validation_control_lr_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "control_method_summary_path": str(
                    (control_dir / "real_validation_control_method_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
                "model_parameters": {
                    "model_name": "spatialself_wavelet",
                    "method_label": args.method_label,
                    "k_nn": args.k_nn,
                    "h_dim": args.h_dim,
                    "lr_dim": args.lr_dim,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "positive_quantile": args.positive_quantile,
                    "negative_quantile": args.negative_quantile,
                    "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
                    "ranking_loss_weight": 0.35,
                    "ranking_margin": 0.5,
                    "use_edge_prior_features": True,
                    "max_train_samples": args.max_train_samples,
                    "seed": args.seed,
                },
                "validation_parameters": {
                    "region_permutations": args.region_permutations,
                    "downstream_permutations": args.downstream_permutations,
                    "random_seed": args.random_seed,
                },
                "limitations": [
                    "Full-slide sparse real-background injected-control validation, not unlabeled real-data ground truth.",
                    "Only one human lymph-node semi-synthetic replicate is run in this Phase 2 initial validation.",
                    "High-expression negative controls have no downstream target sets and are assessed by rank/region controls.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote Phase 2 lymph-node validation outputs to {output_dir}")


def _relabel_prediction(raw_prediction_path: Path, method_label: str) -> Path:
    prediction = pd.read_csv(raw_prediction_path)
    prediction["method"] = method_label
    out_path = raw_prediction_path.with_name(raw_prediction_path.name.replace("__spatialself_wavelet_", f"__{method_label}_"))
    prediction.to_csv(out_path, index=False)
    return out_path


def _summary_table(
    *,
    replicate_dir: Path,
    output_dir: Path,
    metrics: dict,
    control_method: pd.DataFrame,
    downstream: pd.DataFrame,
    lr_ranking: pd.DataFrame,
    region_enrichment: pd.DataFrame,
    started: float,
) -> pd.DataFrame:
    method = control_method.iloc[0] if not control_method.empty else {}
    tested_downstream = downstream[downstream["status"] == "tested"] if not downstream.empty else pd.DataFrame()
    positives = lr_ranking[lr_ranking["lr_id"].isin(_positive_lr_ids(replicate_dir))]
    negatives = lr_ranking[lr_ranking["lr_id"].isin(_negative_lr_ids(replicate_dir))]
    best_region_fdr = (
        float(region_enrichment["bh_fdr"].min())
        if not region_enrichment.empty and "bh_fdr" in region_enrichment
        else float("nan")
    )
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return pd.DataFrame(
        [
            {
                "analysis_label": "phase2_goal6_lymph_node_validation",
                "evidence_status": "phase2_initial_lymph_node_validation",
                "dataset_context": "human_lymph_node_visium_full_slide_sparse_real_background_injected_controls",
                "replicate": replicate_dir.name,
                "method": metrics.get("method", "spatialself_wavelet_edge_prior"),
                "n_cells": int(pd.read_csv(replicate_dir / "cells.csv", usecols=["cell_id"]).shape[0]),
                "n_lr_pairs": int(pd.read_csv(replicate_dir / "lr_pairs.csv", usecols=["lr_id"]).shape[0]),
                "n_candidate_edges": int(pd.read_csv(replicate_dir / "candidate_interactions.csv", usecols=["candidate_id"]).shape[0]),
                "auroc": metrics.get("auroc"),
                "aupr": metrics.get("aupr"),
                "weak_signal_recall_at_threshold": metrics.get("weak_signal_recall_at_threshold"),
                "empirical_fdr_at_threshold": metrics.get("empirical_fdr_at_threshold"),
                "false_positive_high_expression_rate_at_threshold": metrics.get(
                    "false_positive_high_expression_rate_at_threshold"
                ),
                "mean_positive_rank": method.get("mean_positive_rank", float("nan")),
                "mean_negative_rank": method.get("mean_negative_rank", float("nan")),
                "best_negative_rank": method.get("best_negative_rank", float("nan")),
                "negative_top3_count": method.get("negative_top3_count", float("nan")),
                "mean_positive_downstream_spearman": method.get(
                    "mean_positive_downstream_spearman", float("nan")
                ),
                "mean_positive_region_enrichment": method.get("mean_positive_region_enrichment", float("nan")),
                "mean_negative_region_enrichment": method.get("mean_negative_region_enrichment", float("nan")),
                "n_downstream_tested": int(len(tested_downstream)),
                "n_downstream_bh_fdr_0_05": int((tested_downstream["bh_fdr"].astype(float) <= 0.05).sum())
                if not tested_downstream.empty
                else 0,
                "best_region_bh_fdr": best_region_fdr,
                "positive_controls_rank_ahead_of_negatives": bool(
                    len(positives)
                    and len(negatives)
                    and float(positives["global_rank"].max()) < float(negatives["global_rank"].min())
                ),
                "wall_clock_seconds": time.perf_counter() - started,
                "peak_rss_mb": peak_rss_mb,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            }
        ]
    )


def _positive_lr_ids(replicate_dir: Path) -> set[str]:
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    return set(lr_pairs.loc[lr_pairs["true_lr"].astype(bool), "lr_id"].astype(str))


def _negative_lr_ids(replicate_dir: Path) -> set[str]:
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    return set(lr_pairs.loc[lr_pairs["false_positive_high_expression"].astype(bool), "lr_id"].astype(str))


if __name__ == "__main__":
    main()
