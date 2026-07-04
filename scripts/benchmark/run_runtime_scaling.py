#!/usr/bin/env python
"""Generate small controlled scales and profile benchmark method runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.profile_runtime_memory import RUNNABLE_METHODS, profile_methods
from src.spitialself.benchmarks import ControlledCCCConfig, generate_controlled_ccc_benchmark


DEFAULT_METHODS = ("spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat", "commot")
DEFAULT_SMOKE_SCALES = (120, 240)
DEFAULT_FULL_SCALES = (1000, 5000, 10000)
MAX_CONTROLLED_SCALE = 10000
MAX_SPARSE_CONTROLLED_SCALE = 100000


def run_runtime_scaling(
    output_dir: Path,
    *,
    scales: tuple[int, ...],
    methods: tuple[str, ...],
    replicates: int,
    seed: int,
    conda_env: str,
    cot_nitermax: int,
    cellchat_nboot: int,
    spatialself_epochs: int,
    spatialself_k_nn: int,
    spatialself_max_train_samples: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_weak_positive_fraction: float,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    cellnest_num_epoch: int,
    cellnest_hidden: int,
    deeptalk_input_root: str,
    spatalk_per_num: int,
    spatalk_min_pairs: int,
    spatalk_use_synthetic_pathways: bool,
    candidate_generation: str,
    max_candidates_per_lr: int | None,
    max_candidates_per_sender_lr: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_profiles: list[pd.DataFrame] = []
    manifests = []

    for scale in scales:
        scale_cap = (
            MAX_SPARSE_CONTROLLED_SCALE
            if candidate_generation == "sparse_sampled"
            else MAX_CONTROLLED_SCALE
        )
        if scale > scale_cap:
            raise ValueError(
                f"Controlled scaling generation is capped at {scale_cap} cells for "
                f"candidate_generation={candidate_generation!r}."
            )
        scale_dir = output_dir / f"scale_{scale:06d}"
        dataset_root = scale_dir / "dataset"
        profile_root = scale_dir / "profile"

        config = ControlledCCCConfig(
            n_cells=int(scale),
            n_background_genes=32,
            random_seed=int(seed),
            replicates=int(replicates),
            candidate_generation=candidate_generation,
            max_candidates_per_lr=max_candidates_per_lr,
            max_candidates_per_sender_lr=max_candidates_per_sender_lr,
            output_dir=str(dataset_root),
        )
        replicate_dirs = generate_controlled_ccc_benchmark(config)
        manifests.append(
            {
                "scale_n_cells": int(scale),
                "dataset_root": str(dataset_root),
                "profile_root": str(profile_root),
                "replicates": len(replicate_dirs),
                "methods": list(methods),
                "candidate_generation": candidate_generation,
                "max_candidates_per_lr": max_candidates_per_lr,
                "max_candidates_per_sender_lr": int(max_candidates_per_sender_lr),
            }
        )

        profile = profile_methods(
            dataset_root,
            profile_root,
            methods=methods,
            conda_env=conda_env,
            cot_nitermax=cot_nitermax,
            cellchat_nboot=cellchat_nboot,
            spatialself_epochs=spatialself_epochs,
            spatialself_k_nn=spatialself_k_nn,
            spatialself_max_train_samples=spatialself_max_train_samples,
            spatialself_pseudo_label_strategy=spatialself_pseudo_label_strategy,
            spatialself_weak_positive_fraction=spatialself_weak_positive_fraction,
            spatialself_ranking_loss_weight=spatialself_ranking_loss_weight,
            spatialself_ranking_margin=spatialself_ranking_margin,
            spatialself_no_lr_embedding=False,
            spatialself_no_distance=False,
            spatialself_no_spatial_graph=False,
            spatialself_no_expression_features=False,
            cellnest_num_epoch=cellnest_num_epoch,
            cellnest_hidden=cellnest_hidden,
            deeptalk_input_root=deeptalk_input_root,
            spatalk_per_num=spatalk_per_num,
            spatalk_min_pairs=spatalk_min_pairs,
            spatalk_use_synthetic_pathways=spatalk_use_synthetic_pathways,
        )
        profile.insert(0, "scale_n_cells", int(scale))
        all_profiles.append(profile)

    scaling = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    scaling_path = output_dir / "runtime_memory_scaling.csv"
    scaling.to_csv(scaling_path, index=False)
    with (output_dir / "runtime_scaling_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)
    return scaling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp08_runtime_memory/scaling_controlled",
        help="Directory for generated scale datasets and summary tables.",
    )
    parser.add_argument(
        "--scale",
        action="append",
        type=int,
        dest="scales",
        help="Controlled benchmark scale to generate. Repeatable.",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=RUNNABLE_METHODS,
        help="Method to profile. Repeatable. Defaults to SpatialSelf wavelet/GAT and COMMOT.",
    )
    parser.add_argument("--quick-smoke", action="store_true", help="Use 120/240-cell smoke scales.")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--conda-env", default="Benchmark")
    parser.add_argument("--cot-nitermax", type=int, default=100)
    parser.add_argument("--cellchat-nboot", type=int, default=5)
    parser.add_argument("--spatialself-epochs", type=int, default=10)
    parser.add_argument("--spatialself-k-nn", type=int, default=12)
    parser.add_argument("--spatialself-max-train-samples", type=int, default=10000)
    parser.add_argument(
        "--spatialself-pseudo-label-strategy",
        default="lr_prior_lr_capped_mechanism_niche_expression_distance",
    )
    parser.add_argument("--spatialself-weak-positive-fraction", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-loss-weight", type=float, default=0.35)
    parser.add_argument("--spatialself-ranking-margin", type=float, default=0.5)
    parser.add_argument("--cellnest-num-epoch", type=int, default=2)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    parser.add_argument("--deeptalk-input-root", default="")
    parser.add_argument("--spatalk-per-num", type=int, default=20)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    parser.add_argument(
        "--candidate-generation",
        choices=("full_radius", "sparse_sampled"),
        default="full_radius",
        help="Controlled benchmark candidate generation mode.",
    )
    parser.add_argument(
        "--max-candidates-per-lr",
        type=int,
        help="Sparse mode cap per LR pair. Leave unset for uncapped sparse radius sampling.",
    )
    parser.add_argument("--max-candidates-per-sender-lr", type=int, default=48)
    args = parser.parse_args()

    scales = tuple(args.scales or (DEFAULT_SMOKE_SCALES if args.quick_smoke else DEFAULT_FULL_SCALES))
    methods = tuple(args.methods or DEFAULT_METHODS)
    scaling = run_runtime_scaling(
        Path(args.output_dir),
        scales=scales,
        methods=methods,
        replicates=args.replicates,
        seed=args.seed,
        conda_env=args.conda_env,
        cot_nitermax=args.cot_nitermax,
        cellchat_nboot=args.cellchat_nboot,
        spatialself_epochs=args.spatialself_epochs,
        spatialself_k_nn=args.spatialself_k_nn,
        spatialself_max_train_samples=args.spatialself_max_train_samples,
        spatialself_pseudo_label_strategy=args.spatialself_pseudo_label_strategy,
        spatialself_weak_positive_fraction=args.spatialself_weak_positive_fraction,
        spatialself_ranking_loss_weight=args.spatialself_ranking_loss_weight,
        spatialself_ranking_margin=args.spatialself_ranking_margin,
        cellnest_num_epoch=args.cellnest_num_epoch,
        cellnest_hidden=args.cellnest_hidden,
        deeptalk_input_root=args.deeptalk_input_root,
        spatalk_per_num=args.spatalk_per_num,
        spatalk_min_pairs=args.spatalk_min_pairs,
        spatalk_use_synthetic_pathways=args.spatalk_use_synthetic_pathways,
        candidate_generation=args.candidate_generation,
        max_candidates_per_lr=args.max_candidates_per_lr,
        max_candidates_per_sender_lr=args.max_candidates_per_sender_lr,
    )
    columns = [
        "scale_n_cells",
        "method",
        "status",
        "n_candidate_edges",
        "n_lr_pairs",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "peak_gpu_memory_mb",
    ]
    print(scaling[columns].to_string(index=False))
    print(f"Wrote runtime scaling summary to {Path(args.output_dir) / 'runtime_memory_scaling.csv'}")


if __name__ == "__main__":
    main()
