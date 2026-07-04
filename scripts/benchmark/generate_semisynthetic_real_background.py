#!/usr/bin/env python
"""Generate a semi-synthetic CCC benchmark from a real ST background."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks import SemiSyntheticConfig, generate_semisynthetic_real_background


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load benchmark config files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _config_from_yaml(path: Path) -> SemiSyntheticConfig:
    raw = _load_yaml(path)
    generator = raw.get("semisynthetic_generator", {})
    return SemiSyntheticConfig(
        background_format=str(generator.get("background_format", "h5ad")),
        background_simulator=str(generator.get("background_simulator", "real")),
        h5ad_path=str(generator.get("h5ad_path", "Human_Lymph_Node.h5ad")),
        simulator_config_json=str(generator.get("simulator_config_json", "")),
        simulator_r_script=str(generator.get("simulator_r_script", "")),
        simulator_output_dir=str(generator.get("simulator_output_dir", "")),
        region_csv=str(generator.get("region_csv", "visium_lr_benchmark/spot_region_from_c2l.csv")),
        matrix_dir=str(generator.get("matrix_dir", "")),
        spatial_csv=str(generator.get("spatial_csv", "")),
        metadata_csv=str(generator.get("metadata_csv", "")),
        cell_id_col=str(generator.get("cell_id_col", "NAME")),
        x_col=str(generator.get("x_col", "X")),
        y_col=str(generator.get("y_col", "Y")),
        cell_type_col=str(generator.get("cell_type_col", "cell_type")),
        metadata_cell_type_col=str(generator.get("metadata_cell_type_col", "cluster")),
        region_col=str(generator.get("region_col", "")),
        dataset_name=str(generator.get("dataset_name", "human_lymph_node_visium")),
        injection_set=str(generator.get("injection_set", "lymph_node")),
        output_dir=str(generator.get("output_dir", raw.get("output_dir", SemiSyntheticConfig.output_dir))),
        n_spots=generator.get("n_spots"),
        random_seed=int(raw.get("random_seed", generator.get("random_seed", 2025))),
        n_background_genes=int(generator.get("n_background_genes", 2000)),
        background_gene_selection=str(
            generator.get("background_gene_selection", "mean_variance_random")
        ),
        required_gene_csvs=tuple(generator.get("required_gene_csvs") or ()),
        max_candidate_range_quantile=float(generator.get("max_candidate_range_quantile", 0.08)),
        negative_edge_fraction=float(generator.get("negative_edge_fraction", 0.25)),
        candidate_generation=str(generator.get("candidate_generation", "full_radius")),
        max_candidates_per_lr=generator.get("max_candidates_per_lr"),
        max_candidates_per_sender_lr=int(generator.get("max_candidates_per_sender_lr", 64)),
        expression_boost_strong=float(generator.get("expression_boost_strong", 8.0)),
        expression_boost_weak=float(generator.get("expression_boost_weak", 3.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/exp02_semisynthetic_real_background.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--n-spots", type=int)
    parser.add_argument("--n-background-genes", type=int)
    parser.add_argument("--injection-set", choices=("lymph_node", "tonsil", "melanoma"))
    parser.add_argument("--background-format", choices=("h5ad", "matrix_market"))
    parser.add_argument("--background-simulator", choices=("real", "srtsim", "scdesign3"))
    parser.add_argument("--simulator-config-json")
    parser.add_argument("--simulator-r-script")
    parser.add_argument("--simulator-output-dir")
    parser.add_argument("--required-gene-csv", action="append", dest="required_gene_csvs")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--candidate-generation",
        choices=("full_radius", "sparse_sampled"),
        help="Override candidate generation mode.",
    )
    parser.add_argument("--max-candidates-per-lr", type=int)
    parser.add_argument("--max-candidates-per-sender-lr", type=int)
    args = parser.parse_args()

    config = _config_from_yaml(Path(args.config))
    updates = {}
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.n_spots is not None:
        updates["n_spots"] = args.n_spots
    if args.n_background_genes is not None:
        updates["n_background_genes"] = args.n_background_genes
    if args.injection_set:
        updates["injection_set"] = args.injection_set
    if args.background_format:
        updates["background_format"] = args.background_format
    if args.background_simulator:
        updates["background_simulator"] = args.background_simulator
    if args.simulator_config_json:
        updates["simulator_config_json"] = args.simulator_config_json
    if args.simulator_r_script:
        updates["simulator_r_script"] = args.simulator_r_script
    if args.simulator_output_dir:
        updates["simulator_output_dir"] = args.simulator_output_dir
    if args.required_gene_csvs:
        updates["required_gene_csvs"] = tuple(args.required_gene_csvs)
    if args.seed is not None:
        updates["random_seed"] = args.seed
    if args.candidate_generation is not None:
        updates["candidate_generation"] = args.candidate_generation
    if args.max_candidates_per_lr is not None:
        updates["max_candidates_per_lr"] = args.max_candidates_per_lr
    if args.max_candidates_per_sender_lr is not None:
        updates["max_candidates_per_sender_lr"] = args.max_candidates_per_sender_lr
    if updates:
        config = SemiSyntheticConfig(**{**config.__dict__, **updates})

    replicate_dir = generate_semisynthetic_real_background(config)
    print(f"Generated semi-synthetic benchmark: {replicate_dir}")


if __name__ == "__main__":
    main()
