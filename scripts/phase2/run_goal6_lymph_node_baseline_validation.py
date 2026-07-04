#!/usr/bin/env python
"""Run baseline-matched lymph-node injected-control validation."""

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
from scripts.benchmark.summarize_real_validation_controls import summarize_real_validation_controls


DEFAULT_METHODS = (
    "spatialself_wavelet_edge_prior",
    "spatialself_wavelet",
    "spatialself_gat",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)


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
        "--baseline-dir",
        default="outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained",
    )
    parser.add_argument(
        "--spatialself-phase2-dir",
        default="outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse",
    )
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--region-permutations", type=int, default=100)
    parser.add_argument("--downstream-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument(
        "--pathway-csv",
        default="outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_nichenet_targets/lr_pairs__nichenet_targets.csv",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    replicate_dir = PROJECT_ROOT / args.replicate_dir
    baseline_dir = PROJECT_ROOT / args.baseline_dir
    spatialself_phase2_dir = PROJECT_ROOT / args.spatialself_phase2_dir
    output_dir = PROJECT_ROOT / args.output_dir
    region_root = output_dir / "region_enrichment"
    downstream_root = output_dir / "downstream_target"
    control_root = output_dir / "control_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_status = _load_baseline_status(baseline_dir)
    methods = tuple(args.methods) if args.methods else DEFAULT_METHODS
    prediction_records = []
    status_records = []
    for method in methods:
        prediction_path = _prediction_path(
            method,
            baseline_dir=baseline_dir,
            spatialself_phase2_dir=spatialself_phase2_dir,
        )
        status = baseline_status.get(method, {})
        if not prediction_path.exists():
            status_records.append(
                {
                    "method": method,
                    "status": "missing_prediction",
                    "detail": f"prediction not found: {prediction_path}",
                    "claim_eligible": False,
                }
            )
            continue
        prediction = pd.read_csv(prediction_path, usecols=["candidate_id", "method", "score"])
        nonzero = int((prediction["score"] != 0).sum())
        unique_scores = int(prediction["score"].nunique())
        claim_eligible = (
            str(status.get("status", "")) == "completed"
            and unique_scores > 1
            and nonzero > 0
            and method != "deeptalk"
        )
        status_records.append(
            {
                "method": method,
                "status": status.get("status", "completed" if claim_eligible else "unknown"),
                "detail": status.get("detail", ""),
                "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "n_prediction_rows": int(len(prediction)),
                "n_unique_scores": unique_scores,
                "n_nonzero_scores": nonzero,
                "claim_eligible": bool(claim_eligible),
            }
        )
        if not claim_eligible:
            continue
        region_dir = region_root / method
        downstream_dir = downstream_root / method
        enrichment, ranking = run_region_enrichment(
            replicate_dir / "candidate_interactions.csv",
            prediction_path,
            replicate_dir / "cells.csv",
            region_dir,
            region_column="region",
            receiver_region=True,
            n_permutations=args.region_permutations,
            random_seed=args.random_seed,
        )
        downstream = run_downstream_target_correlation(
            replicate_dir / "candidate_interactions.csv",
            prediction_path,
            replicate_dir / "expression.csv",
            downstream_dir,
            pathway_csv=PROJECT_ROOT / args.pathway_csv if args.pathway_csv else None,
            reactome_tf_ppr_csv=None,
            trrust_dir=None,
            n_permutations=args.downstream_permutations,
            random_seed=args.random_seed,
        )
        control_lr, control_method = summarize_real_validation_controls(
            replicate_dir / "lr_pairs.csv",
            [region_dir / f"{prediction_path.stem}__lr_global_ranking.csv"],
            [region_dir / f"{prediction_path.stem}__region_enrichment.csv"],
            [downstream_dir / f"{prediction_path.stem}__downstream_target_correlation.csv"],
            control_root / method,
        )
        prediction_records.append(
            _method_summary_row(
                method=method,
                prediction_path=prediction_path,
                control_method=control_method,
                control_lr=control_lr,
                downstream=downstream,
                enrichment=enrichment,
                ranking=ranking,
                baseline_status=status,
            )
        )

    method_summary = pd.DataFrame(prediction_records)
    status_table = pd.DataFrame(status_records)
    control_lr_summary = _concat_control_tables(control_root, methods)
    overview = _overview_summary(
        method_summary,
        status_table,
        replicate_dir=replicate_dir,
        started=started,
        output_dir=output_dir,
    )

    method_summary.to_csv(output_dir / "baseline_matched_method_summary.csv", index=False)
    status_table.to_csv(output_dir / "baseline_matched_status.csv", index=False)
    control_lr_summary.to_csv(output_dir / "baseline_matched_control_lr_summary.csv", index=False)
    overview.to_csv(output_dir / "baseline_matched_validation_summary.csv", index=False)
    manifest_path = output_dir / "baseline_matched_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "analysis_label": "phase2_goal6_lymph_node_baseline_matched_validation",
                "evidence_status": "phase2_initial_baseline_matched_lymph_node_validation",
                "replicate_dir": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "baseline_dir": str(baseline_dir.relative_to(PROJECT_ROOT)),
                "spatialself_phase2_dir": str(spatialself_phase2_dir.relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "methods_requested": list(methods),
                "method_summary": str((output_dir / "baseline_matched_method_summary.csv").relative_to(PROJECT_ROOT)),
                "status_table": str((output_dir / "baseline_matched_status.csv").relative_to(PROJECT_ROOT)),
                "control_lr_summary": str(
                    (output_dir / "baseline_matched_control_lr_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "overview_summary": str(
                    (output_dir / "baseline_matched_validation_summary.csv").relative_to(PROJECT_ROOT)
                ),
                "region_permutations": args.region_permutations,
                "downstream_permutations": args.downstream_permutations,
                "limitations": [
                    "Single full-slide sparse real-background injected-control replicate.",
                    "CellChat and SpaTalk are projected cell-type-pair/LR scores, not native candidate-edge models.",
                    "CellNEST is projected from retained CellNEST output; DeepTalk is status-only when prediction is all zero.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(method_summary.to_string(index=False))
    print(status_table.to_string(index=False))
    print(f"Wrote baseline-matched lymph-node validation outputs to {output_dir}")


def _load_baseline_status(baseline_dir: Path) -> dict[str, dict[str, object]]:
    path = baseline_dir / "baseline_status.csv"
    if not path.exists():
        return {}
    return {
        str(row["method"]): row
        for row in pd.read_csv(path).to_dict(orient="records")
    }


def _prediction_path(method: str, *, baseline_dir: Path, spatialself_phase2_dir: Path) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        phase2 = (
            spatialself_phase2_dir
            / "prediction_inputs/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions.csv"
        )
        if phase2.exists():
            return phase2
        return (
            baseline_dir
            / "spatialself_wavelet_edge_prior/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv"
        )
    return baseline_dir / method / f"human_lymph_node_visium_replicate_00__{method}_predictions.csv"


def _method_summary_row(
    *,
    method: str,
    prediction_path: Path,
    control_method: pd.DataFrame,
    control_lr: pd.DataFrame,
    downstream: pd.DataFrame,
    enrichment: pd.DataFrame,
    ranking: pd.DataFrame,
    baseline_status: dict[str, object],
) -> dict[str, object]:
    control = control_method.iloc[0] if not control_method.empty else {}
    tested = downstream[downstream["status"] == "tested"] if not downstream.empty else pd.DataFrame()
    positives = control_lr[control_lr["true_lr"].astype(bool)] if not control_lr.empty else pd.DataFrame()
    negatives = (
        control_lr[control_lr["false_positive_high_expression"].astype(bool)]
        if not control_lr.empty
        else pd.DataFrame()
    )
    return {
        "method": method,
        "status": baseline_status.get("status", "completed"),
        "detail": baseline_status.get("detail", ""),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "n_lr_pairs_ranked": int(ranking["lr_id"].nunique()) if not ranking.empty else 0,
        "mean_positive_rank": control.get("mean_positive_rank", float("nan")),
        "mean_negative_rank": control.get("mean_negative_rank", float("nan")),
        "best_negative_rank": control.get("best_negative_rank", float("nan")),
        "negative_top3_count": control.get("negative_top3_count", float("nan")),
        "positive_controls_rank_ahead_of_negatives": bool(
            len(positives)
            and len(negatives)
            and float(positives["global_rank"].max()) < float(negatives["global_rank"].min())
        ),
        "mean_positive_downstream_spearman": control.get(
            "mean_positive_downstream_spearman", float("nan")
        ),
        "mean_positive_region_enrichment": control.get("mean_positive_region_enrichment", float("nan")),
        "mean_negative_region_enrichment": control.get("mean_negative_region_enrichment", float("nan")),
        "n_downstream_tested": int(len(tested)),
        "n_downstream_bh_fdr_0_05": int((tested["bh_fdr"].astype(float) <= 0.05).sum())
        if not tested.empty
        else 0,
        "best_region_bh_fdr": float(enrichment["bh_fdr"].min()) if not enrichment.empty else float("nan"),
        "rank_validation_pass": bool(
            len(positives)
            and len(negatives)
            and float(positives["global_rank"].max()) < float(negatives["global_rank"].min())
            and float(control.get("negative_top3_count", 999)) == 0
        ),
    }


def _concat_control_tables(control_root: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    tables = []
    for method in methods:
        path = control_root / method / "real_validation_control_lr_summary.csv"
        if path.exists():
            table = pd.read_csv(path)
            table["baseline_validation_scope"] = "phase2_baseline_matched_lymph_node"
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _overview_summary(
    method_summary: pd.DataFrame,
    status_table: pd.DataFrame,
    *,
    replicate_dir: Path,
    started: float,
    output_dir: Path,
) -> pd.DataFrame:
    claim_eligible = status_table[status_table["claim_eligible"].astype(bool)] if not status_table.empty else pd.DataFrame()
    spatialself = method_summary[method_summary["method"] == "spatialself_wavelet_edge_prior"]
    best_rank = (
        method_summary.sort_values(["rank_validation_pass", "mean_positive_rank"], ascending=[False, True]).iloc[0]
        if not method_summary.empty
        else {}
    )
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return pd.DataFrame(
        [
            {
                "analysis_label": "phase2_goal6_lymph_node_baseline_matched_validation",
                "evidence_status": "phase2_initial_baseline_matched_lymph_node_validation",
                "dataset_context": "human_lymph_node_visium_full_slide_sparse_real_background_injected_controls",
                "n_cells": int(pd.read_csv(replicate_dir / "cells.csv", usecols=["cell_id"]).shape[0]),
                "n_lr_pairs": int(pd.read_csv(replicate_dir / "lr_pairs.csv", usecols=["lr_id"]).shape[0]),
                "n_candidate_edges": int(
                    pd.read_csv(replicate_dir / "candidate_interactions.csv", usecols=["candidate_id"]).shape[0]
                ),
                "n_methods_requested": int(len(status_table)),
                "n_claim_eligible_methods": int(len(claim_eligible)),
                "n_methods_rank_validation_pass": int(method_summary["rank_validation_pass"].sum())
                if not method_summary.empty
                else 0,
                "best_rank_validation_method": best_rank.get("method", "NA"),
                "spatialself_edge_prior_rank_validation_pass": bool(spatialself["rank_validation_pass"].iloc[0])
                if not spatialself.empty
                else False,
                "spatialself_edge_prior_mean_positive_rank": float(spatialself["mean_positive_rank"].iloc[0])
                if not spatialself.empty
                else float("nan"),
                "spatialself_edge_prior_mean_negative_rank": float(spatialself["mean_negative_rank"].iloc[0])
                if not spatialself.empty
                else float("nan"),
                "wall_clock_seconds": time.perf_counter() - started,
                "peak_rss_mb": peak_rss_mb,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            }
        ]
    )


if __name__ == "__main__":
    main()
