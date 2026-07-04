#!/usr/bin/env python
"""Phase 1 experiment runner placeholder.

The registry and configs are now fixed. Real execution functions will be wired
in experiment by experiment so each benchmark remains auditable.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.experiments import get_phase1_experiment
from scripts.benchmark.generate_controlled_ccc import _config_from_yaml as _controlled_config_from_yaml
from scripts.benchmark.generate_semisynthetic_real_background import (
    _config_from_yaml as _semisynthetic_config_from_yaml,
)
from scripts.benchmark.extract_nichenet_targets import DEFAULT_HUMAN_MATRIX, extract_nichenet_targets
from scripts.benchmark.run_baseline_comparison import run_baselines_on_dataset
from scripts.benchmark.run_downstream_target_correlation import run_downstream_target_correlation
from scripts.benchmark.run_permutation_fdr import run_permutation_fdr
from scripts.benchmark.run_region_enrichment import run_region_enrichment
from scripts.benchmark.run_runtime_scaling import run_runtime_scaling
from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
from scripts.benchmark.run_spatialself_ablation import ABLATIONS, run_ablation
from scripts.benchmark.run_weak_signal_false_positive_dataset import run_stress_dataset
from scripts.benchmark.summarize_real_validation_controls import summarize_real_validation_controls
from scripts.benchmark.validate_controlled_ccc import (
    summarize_lr_pairs,
    summarize_mechanisms,
    validate_replicate,
)
from scripts.benchmark.validate_semisynthetic_real_background import (
    summarize_injections as summarize_semisynthetic_injections,
    summarize_mechanisms as summarize_semisynthetic_mechanisms,
    validate_replicate as validate_semisynthetic_replicate,
)
from src.spitialself.benchmarks import generate_controlled_ccc_benchmark
from src.spitialself.benchmarks import generate_semisynthetic_real_background


def main():
    parser = argparse.ArgumentParser(description="Run a Phase 1 SpatialSelf experiment.")
    parser.add_argument("exp_id", help="Experiment id, e.g. exp01_controlled_benchmark")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned config/output paths without running computation.",
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run a smaller smoke-scale variant where supported.",
    )
    args = parser.parse_args()

    exp = get_phase1_experiment(args.exp_id)
    print(f"Experiment: {exp.exp_id}")
    print(f"Title:      {exp.title}")
    print(f"Objective:  {exp.objective}")
    print(f"Config:     {exp.config_path}")
    print(f"Output:     {exp.output_dir}")

    if args.dry_run:
        return

    if exp.exp_id == "exp01_controlled_benchmark":
        config = _controlled_config_from_yaml(PROJECT_ROOT / exp.config_path)
        if args.quick_smoke:
            config = config.__class__(
                **{
                    **config.__dict__,
                    "n_cells": 120,
                    "replicates": 1,
                    "output_dir": "outputs/phase1/exp01_controlled_benchmark/smoke_datasets",
                }
            )
        replicate_dirs = generate_controlled_ccc_benchmark(config)
        print(f"Generated {len(replicate_dirs)} controlled benchmark replicate(s)")
        print(f"Dataset root: {config.output_dir}")

        validation = [validate_replicate(path) for path in replicate_dirs]
        validation_path = Path(config.output_dir) / "validation_summary.csv"
        pd.DataFrame(validation).to_csv(validation_path, index=False)
        print(f"Validation summary: {validation_path}")
        mechanism_summary_path = Path(config.output_dir) / "controlled_mechanism_summary.csv"
        lr_summary_path = Path(config.output_dir) / "controlled_lr_summary.csv"
        pd.concat([summarize_mechanisms(path) for path in replicate_dirs], ignore_index=True).to_csv(
            mechanism_summary_path,
            index=False,
        )
        pd.concat([summarize_lr_pairs(path) for path in replicate_dirs], ignore_index=True).to_csv(
            lr_summary_path,
            index=False,
        )
        print(f"Mechanism summary: {mechanism_summary_path}")
        print(f"LR summary: {lr_summary_path}")

        return

    if exp.exp_id == "exp02_semisynthetic_real_background":
        config = _semisynthetic_config_from_yaml(PROJECT_ROOT / exp.config_path)
        if args.quick_smoke:
            config = config.__class__(
                **{
                    **config.__dict__,
                    "n_spots": 300,
                    "output_dir": "outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets",
                }
            )
        replicate_dir = generate_semisynthetic_real_background(config)
        print(f"Generated semi-synthetic benchmark: {replicate_dir}")

        validation = [validate_semisynthetic_replicate(replicate_dir)]
        validation_path = Path(config.output_dir) / "validation_summary.csv"
        pd.DataFrame(validation).to_csv(validation_path, index=False)
        print(f"Validation summary: {validation_path}")
        injection_summary_path = Path(config.output_dir) / "semisynthetic_injection_summary.csv"
        mechanism_summary_path = Path(config.output_dir) / "semisynthetic_mechanism_summary.csv"
        summarize_semisynthetic_injections(replicate_dir).to_csv(injection_summary_path, index=False)
        summarize_semisynthetic_mechanisms(replicate_dir).to_csv(mechanism_summary_path, index=False)
        print(f"Injection summary: {injection_summary_path}")
        print(f"Mechanism summary: {mechanism_summary_path}")

        return

    if exp.exp_id == "exp03_baseline_comparison":
        dataset_root = _controlled_dataset_root(quick_smoke=args.quick_smoke)
        output_dir = PROJECT_ROOT / exp.output_dir
        if args.quick_smoke:
            output_dir = output_dir / "smoke_controlled/phase_runner"
        methods = (
            "spatialself_wavelet",
            "spatialself_wavelet_edge_prior",
            "spatialself_gat",
            "commot",
            "cellchat",
            "spatalk",
            "cellnest",
            "deeptalk",
            "nichenet",
            "niches",
        )
        if args.quick_smoke:
            methods = (
                "spatialself_wavelet",
                "spatialself_wavelet_edge_prior",
                "commot",
                "deeptalk",
                "nichenet",
                "niches",
            )
        metrics = run_baselines_on_dataset(
            dataset_root,
            output_dir,
            methods=methods,
            cot_nitermax=50 if args.quick_smoke else 500,
            cellchat_nboot=5 if args.quick_smoke else 20,
            spatialself_epochs=3 if args.quick_smoke else 20,
            spatialself_k_nn=12,
            spatialself_max_train_samples=2000 if args.quick_smoke else 20000,
            spatialself_pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
            spatialself_weak_positive_fraction=0.0,
            spatialself_ranking_loss_weight=0.35,
            spatialself_ranking_margin=0.5,
            spatialself_lr_cap_positive=50,
            spatialself_lr_cap_negative=200,
            spatialself_no_lr_embedding=False,
            spatialself_no_distance=False,
            spatialself_no_spatial_graph=False,
            spatialself_no_expression_features=False,
            cellnest_num_epoch=1 if args.quick_smoke else 2,
            cellnest_hidden=32,
            deeptalk_input_root=None,
            spatalk_per_num=20 if args.quick_smoke else 100,
            spatalk_min_pairs=1,
            spatalk_use_synthetic_pathways=args.quick_smoke,
        )
        print(f"Baseline comparison output: {output_dir}")
        if not metrics.empty:
            print(
                metrics[
                    [
                        "replicate",
                        "method",
                        "auroc",
                        "aupr",
                        "weak_signal_recall_at_threshold",
                        "empirical_fdr_at_threshold",
                    ]
                ].to_string(index=False)
            )
        return

    if exp.exp_id == "exp04_model_ablation":
        dataset_root = _controlled_dataset_root(quick_smoke=args.quick_smoke)
        output_dir = PROJECT_ROOT / exp.output_dir
        if args.quick_smoke:
            output_dir = output_dir / "smoke_controlled/phase_runner"
        ablations = (
            "gat_hard_expression_distance",
            "wavelet_hard_expression_distance",
            "wavelet_mechanism_niche_lr_capped",
            "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
            "wavelet_mechanism_niche_weak_augmented",
            "wavelet_mechanism_niche_soft_weak_augmented",
            "wavelet_mechanism_niche_soft_weak_augmented_ranking",
            "wavelet_mechanism_niche_edge_prior",
            "wavelet_mechanism_niche_edge_prior_soft_weak",
            "wavelet_mechanism_niche_edge_prior_ranking",
            "wavelet_no_lr_embedding",
            "wavelet_fixed_random_lr_embedding",
            "wavelet_one_hot_lr_identity",
            "wavelet_no_distance",
            "wavelet_no_spatial_graph",
            "gcn_hard_expression_distance",
            "mlp_hard_expression_distance",
        )
        if not args.quick_smoke:
            ablations = tuple(ABLATIONS)
        metrics = run_ablation(
            dataset_root,
            output_dir,
            ablations=ablations,
            epochs=3 if args.quick_smoke else 20,
            k_nn=12,
            max_train_samples=2000 if args.quick_smoke else 20000,
        )
        print(f"Ablation output: {output_dir}")
        if not metrics.empty:
            print(
                metrics[
                    [
                        "ablation",
                        "replicate",
                        "auroc",
                        "aupr",
                        "weak_signal_recall_at_threshold",
                        "empirical_fdr_at_threshold",
                    ]
                ].to_string(index=False)
            )
        return

    if exp.exp_id == "exp05_weak_signal_false_positive":
        dataset_root = _controlled_dataset_root(quick_smoke=args.quick_smoke)
        output_dir = PROJECT_ROOT / exp.output_dir
        if args.quick_smoke:
            output_dir = output_dir / "smoke_controlled/phase_runner"
        metrics = run_stress_dataset(
            dataset_root,
            output_dir,
            methods=("spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"),
            spatialself_epochs=3 if args.quick_smoke else 10,
            spatialself_max_train_samples=2000 if args.quick_smoke else 10000,
            spatialself_k_nn=12,
            spatialself_pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
            spatialself_weak_positive_fraction=0.0,
            spatialself_ranking_loss_weight=0.35,
            spatialself_ranking_margin=0.5,
            spatialself_lr_cap_positive=50,
            spatialself_lr_cap_negative=200,
        )
        print(f"Weak-signal/false-positive stress output: {output_dir}")
        if not metrics.empty:
            print(
                metrics[
                    [
                        "replicate",
                        "method",
                        "auroc",
                        "aupr",
                        "weak_signal_recall_at_threshold",
                        "empirical_fdr_at_threshold",
                        "false_positive_high_expression_rate_at_threshold",
                    ]
                ].to_string(index=False)
            )
        return

    if exp.exp_id == "exp06_permutation_fdr":
        dataset_root = _controlled_dataset_root(quick_smoke=args.quick_smoke)
        replicate_dir = _first_replicate_dir(dataset_root)
        prediction_dir = PROJECT_ROOT / exp.output_dir / (
            "smoke_prediction_inputs" if args.quick_smoke else "prediction_inputs"
        )
        fdr_method = "spatialself_wavelet_edge_prior" if args.quick_smoke else "spatialself_wavelet"
        prediction_metrics = run_spatialself_on_replicate(
            replicate_dir,
            prediction_dir / fdr_method,
            model_name="spatialself_wavelet",
            epochs=3 if args.quick_smoke else 20,
            max_train_samples=2000 if args.quick_smoke else 20000,
            pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
            weak_positive_fraction=0.0,
            ranking_loss_weight=0.35,
            ranking_margin=0.5,
            use_edge_prior_features=args.quick_smoke,
        )
        prediction_path = Path(prediction_metrics["prediction_path"])
        if args.quick_smoke:
            prediction = pd.read_csv(prediction_path)
            prediction["method"] = fdr_method
            prediction.to_csv(prediction_path, index=False)
        output_dir = PROJECT_ROOT / exp.output_dir
        if args.quick_smoke:
            output_dir = output_dir / "smoke_spatialself_wavelet_edge_prior_phase_runner"
        summary = run_permutation_fdr(
            replicate_dir / "candidate_interactions.csv",
            prediction_path,
            output_dir,
            null_models=("spatial_permutation", "receiver_permutation", "lr_identity_permutation"),
            n_permutations=20 if args.quick_smoke else 100,
            random_seed=2025,
            score_column="score",
            alpha=0.05,
        )
        print(f"Permutation/FDR output: {output_dir}")
        print(summary.to_string(index=False))
        return

    if exp.exp_id == "exp07_lymph_node_global_validation":
        dataset_root = _semisynthetic_dataset_root(quick_smoke=args.quick_smoke)
        replicate_dir = _first_replicate_dir(dataset_root)
        output_dir = PROJECT_ROOT / exp.output_dir
        if args.quick_smoke:
            output_dir = output_dir / "smoke_phase_runner"
        nichenet_targets = extract_nichenet_targets(
            replicate_dir / "lr_pairs.csv",
            output_dir / "nichenet_targets",
            ligand_target_matrix=Path(DEFAULT_HUMAN_MATRIX),
            top_n=20 if args.quick_smoke else 200,
        )
        validation_methods = (
            ("spatialself_wavelet", False),
            ("spatialself_wavelet_edge_prior", True),
        )
        enrichments = []
        rankings = []
        downstream_tables = []
        ranking_paths = []
        enrichment_paths = []
        downstream_paths = []
        for method_name, use_edge_prior in validation_methods:
            prediction_metrics = run_spatialself_on_replicate(
                replicate_dir,
                output_dir / f"prediction_inputs/{method_name}",
                model_name="spatialself_wavelet",
                epochs=3 if args.quick_smoke else 20,
                max_train_samples=4000 if args.quick_smoke else 20000,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                weak_positive_fraction=0.0,
                ranking_loss_weight=0.35,
                ranking_margin=0.5,
                use_edge_prior_features=use_edge_prior,
            )
            prediction_path = Path(prediction_metrics["prediction_path"])
            if use_edge_prior:
                prediction = pd.read_csv(prediction_path)
                prediction["method"] = method_name
                aliased_prediction_path = prediction_path.with_name(
                    prediction_path.name.replace(
                        "__spatialself_wavelet_predictions.csv",
                        "__spatialself_wavelet_edge_prior_predictions.csv",
                    )
                )
                prediction.to_csv(aliased_prediction_path, index=False)
                prediction_path = aliased_prediction_path

            enrichment, ranking = run_region_enrichment(
                replicate_dir / "candidate_interactions.csv",
                prediction_path,
                replicate_dir / "cells.csv",
                output_dir / "region_enrichment",
                region_column="region",
                receiver_region=True,
                n_permutations=20 if args.quick_smoke else 100,
                random_seed=2025,
            )
            downstream = run_downstream_target_correlation(
                replicate_dir / "candidate_interactions.csv",
                prediction_path,
                replicate_dir / "expression.csv",
                output_dir / "downstream_target",
                pathway_csv=nichenet_targets,
                reactome_tf_ppr_csv=None,
                trrust_dir=None,
                min_targets=3,
                max_targets=200,
                n_permutations=20 if args.quick_smoke else 100,
                random_seed=2025,
            )
            enrichments.append(enrichment)
            rankings.append(ranking)
            downstream_tables.append(downstream)
            prediction_stem = prediction_path.stem
            ranking_paths.append(output_dir / "region_enrichment" / f"{prediction_stem}__lr_global_ranking.csv")
            enrichment_paths.append(output_dir / "region_enrichment" / f"{prediction_stem}__region_enrichment.csv")
            downstream_paths.append(output_dir / "downstream_target" / f"{prediction_stem}__downstream_target_correlation.csv")
        control_table, control_summary = summarize_real_validation_controls(
            replicate_dir / "lr_pairs.csv",
            ranking_paths,
            enrichment_paths,
            downstream_paths,
            output_dir / "control_summary",
        )
        print(f"Lymph-node global validation output: {output_dir}")
        print("Top region enrichment:")
        print(pd.concat(enrichments, ignore_index=True).head(12).to_string(index=False))
        print("Downstream target correlation:")
        downstream_all = pd.concat(downstream_tables, ignore_index=True)
        print(
            downstream_all[
                [
                    "lr_id",
                    "method",
                    "n_targets_present",
                    "spearman_r",
                    "empirical_p_value",
                    "bh_fdr",
                    "status",
                ]
            ].to_string(index=False)
        )
        print("Top global LR ranking:")
        print(pd.concat(rankings, ignore_index=True).head(12).to_string(index=False))
        print("Control summary:")
        print(control_summary.to_string(index=False))
        return

    if exp.exp_id == "exp08_runtime_memory":
        output_dir = PROJECT_ROOT / exp.output_dir
        scales = (120, 240) if args.quick_smoke else (1000, 5000, 10000)
        if args.quick_smoke:
            output_dir = output_dir / "smoke_scaling_controlled"
        scaling = run_runtime_scaling(
            output_dir,
            scales=scales,
            methods=("spatialself_wavelet", "spatialself_gat", "commot")
            if not args.quick_smoke
            else ("spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"),
            replicates=1,
            seed=2026,
            conda_env="Benchmark",
            cot_nitermax=100 if args.quick_smoke else 500,
            cellchat_nboot=5 if args.quick_smoke else 20,
            spatialself_epochs=3 if args.quick_smoke else 10,
            spatialself_k_nn=12,
            spatialself_max_train_samples=2000 if args.quick_smoke else 10000,
            spatialself_pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
            spatialself_weak_positive_fraction=0.0,
            spatialself_ranking_loss_weight=0.35,
            spatialself_ranking_margin=0.5,
            cellnest_num_epoch=2,
            cellnest_hidden=32,
            deeptalk_input_root="",
            spatalk_per_num=20 if args.quick_smoke else 100,
            spatalk_min_pairs=1,
            spatalk_use_synthetic_pathways=args.quick_smoke,
        )
        summary_path = output_dir / "runtime_memory_scaling.csv"
        print(f"Runtime scaling summary: {summary_path}")
        print(
            scaling[
                [
                    "scale_n_cells",
                    "method",
                    "status",
                    "n_candidate_edges",
                    "wall_clock_seconds",
                    "peak_cpu_memory_mb",
                    "peak_gpu_memory_mb",
                ]
            ].to_string(index=False)
        )
        return

    raise NotImplementedError(
        "Phase 1 runner wiring is not implemented yet. Start with --dry-run, "
        "then attach a concrete runner for this experiment."
    )


def _controlled_dataset_root(*, quick_smoke: bool) -> Path:
    dataset_root = PROJECT_ROOT / (
        "outputs/phase1/exp01_controlled_benchmark/smoke_datasets"
        if quick_smoke
        else "outputs/phase1/exp01_controlled_benchmark/datasets"
    )
    if _replicate_dirs(dataset_root):
        return dataset_root
    config = _controlled_config_from_yaml(PROJECT_ROOT / "configs/experiments/exp01_controlled_benchmark.yaml")
    if quick_smoke:
        config = config.__class__(
            **{
                **config.__dict__,
                "n_cells": 120,
                "replicates": 1,
                "output_dir": str(dataset_root),
            }
        )
    generate_controlled_ccc_benchmark(config)
    return dataset_root


def _semisynthetic_dataset_root(*, quick_smoke: bool) -> Path:
    dataset_root = PROJECT_ROOT / (
        "outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets"
        if quick_smoke
        else "outputs/phase1/exp02_semisynthetic_real_background/datasets"
    )
    if _replicate_dirs(dataset_root):
        return dataset_root
    config = _semisynthetic_config_from_yaml(
        PROJECT_ROOT / "configs/experiments/exp02_semisynthetic_real_background.yaml"
    )
    if quick_smoke:
        config = config.__class__(
            **{
                **config.__dict__,
                "n_spots": 300,
                "output_dir": str(dataset_root),
            }
        )
    generate_semisynthetic_real_background(config)
    return dataset_root


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _first_replicate_dir(dataset_root: Path) -> Path:
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")
    return replicate_dirs[0]


if __name__ == "__main__":
    main()
