#!/usr/bin/env python
"""Generate the fair CCC benchmark v2 on real ST backgrounds.

This entry point intentionally keeps CCC truth injection separate from model
training tricks. It builds real-background semi-synthetic datasets with the
same schema across tissue contexts so all methods can be evaluated on the same
candidate-edge task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks import SemiSyntheticConfig, generate_semisynthetic_real_background


DATASET_PRESETS = {
    "range_lymph_node": {
        "dataset_name": "range_v2_lymph_node",
        "injection_set": "range_lymph_node",
        "background_format": "h5ad",
        "h5ad_path": "Human_Lymph_Node.h5ad",
        "region_csv": "visium_lr_benchmark/spot_region_from_c2l.csv",
        "n_spots": 1200,
        "n_background_genes": 1200,
        "max_candidate_range_quantile": 0.08,
        "max_candidates_per_sender_lr": 64,
        "candidate_generation": "range_stress",
        "negative_edge_fraction": 0.45,
    },
    "lymph_node": {
        "dataset_name": "fair_v2_lymph_node",
        "injection_set": "lymph_node",
        "background_format": "h5ad",
        "h5ad_path": "Human_Lymph_Node.h5ad",
        "region_csv": "visium_lr_benchmark/spot_region_from_c2l.csv",
        "n_spots": 1200,
        "n_background_genes": 1200,
        "max_candidate_range_quantile": 0.08,
        "max_candidates_per_sender_lr": 48,
    },
    "tonsil": {
        "dataset_name": "fair_v2_tonsil",
        "injection_set": "tonsil",
        "background_format": "matrix_market",
        "matrix_dir": "data/raw/spamosaic_tonsil_A1",
        "spatial_csv": "data/raw/spamosaic_tonsil_A1/spatial.csv",
        "metadata_csv": "data/raw/spamosaic_tonsil_A1/metadata.csv",
        "cell_id_col": "NAME",
        "x_col": "X",
        "y_col": "Y",
        "cell_type_col": "cell_type",
        "metadata_cell_type_col": "cluster",
        "n_spots": 1200,
        "n_background_genes": 1200,
        "max_candidate_range_quantile": 0.05,
        "max_candidates_per_sender_lr": 48,
    },
    "melanoma": {
        "dataset_name": "fair_v2_melanoma",
        "injection_set": "melanoma",
        "background_format": "matrix_market",
        "matrix_dir": "HumanMelannomaMultiome/data",
        "spatial_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_spatial.csv",
        "metadata_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_metadata.csv",
        "cell_id_col": "NAME",
        "x_col": "X",
        "y_col": "Y",
        "cell_type_col": "cell_type",
        "metadata_cell_type_col": "cluster",
        "n_spots": 1200,
        "n_background_genes": 1200,
        "max_candidate_range_quantile": 0.05,
        "max_candidates_per_sender_lr": 48,
    },
    "range_melanoma": {
        "dataset_name": "range_v2_melanoma",
        "injection_set": "range_melanoma",
        "background_format": "matrix_market",
        "matrix_dir": "HumanMelannomaMultiome/data",
        "spatial_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_spatial.csv",
        "metadata_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_metadata.csv",
        "cell_id_col": "NAME",
        "x_col": "X",
        "y_col": "Y",
        "cell_type_col": "cell_type",
        "metadata_cell_type_col": "cluster",
        "n_spots": 1200,
        "n_background_genes": 1200,
        "max_candidate_range_quantile": 0.05,
        "max_candidates_per_sender_lr": 64,
        "candidate_generation": "range_stress",
        "negative_edge_fraction": 0.45,
    },
}


def _existing_dataset_keys(keys: list[str]) -> list[str]:
    available = []
    for key in keys:
        preset = DATASET_PRESETS[key]
        if preset["background_format"] == "h5ad":
            if Path(preset["h5ad_path"]).exists():
                available.append(key)
        elif Path(preset["matrix_dir"]).exists() and Path(preset["spatial_csv"]).exists():
            available.append(key)
    return available


def _make_config(key: str, output_dir: Path, seed: int, replicate_id: int) -> SemiSyntheticConfig:
    fields = dict(DATASET_PRESETS[key])
    fields["output_dir"] = str(output_dir)
    fields["random_seed"] = int(seed)
    fields["replicate_id"] = int(replicate_id)
    return SemiSyntheticConfig(**fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/fair_benchmark_v2/datasets",
        help="Directory receiving generated replicate folders.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_PRESETS),
        help="Dataset preset to generate. Repeatable. Defaults to available presets.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--n-spots", type=int)
    parser.add_argument("--n-background-genes", type=int)
    parser.add_argument("--max-candidates-per-sender-lr", type=int)
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Try requested presets even if their local raw files are not detected.",
    )
    args = parser.parse_args()

    requested = args.dataset or list(DATASET_PRESETS)
    datasets = requested if args.include_missing else _existing_dataset_keys(requested)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for key in datasets:
        for replicate_id in range(int(args.replicates)):
            config = _make_config(key, output_dir, seed=args.seed + replicate_id, replicate_id=replicate_id)
            updates = {}
            if args.n_spots is not None:
                updates["n_spots"] = int(args.n_spots)
            if args.n_background_genes is not None:
                updates["n_background_genes"] = int(args.n_background_genes)
            if args.max_candidates_per_sender_lr is not None:
                updates["max_candidates_per_sender_lr"] = int(args.max_candidates_per_sender_lr)
            if updates:
                config = SemiSyntheticConfig(**{**config.__dict__, **updates})
            rep_dir = generate_semisynthetic_real_background(config)
            records.append(
                {
                    "dataset": key,
                    "replicate": replicate_id,
                    "path": str(rep_dir),
                    "background_format": config.background_format,
                    "injection_set": config.injection_set,
                    "n_spots": config.n_spots,
                    "n_background_genes": config.n_background_genes,
                }
            )

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "fair_benchmark_v2_manifest.csv", index=False)
    metadata = {
        "benchmark": "fair_ccc_benchmark_v2",
        "purpose": "real-background CCC recovery with candidate-edge fairness across methods",
        "truth_design": [
            "contact-dependent juxtacrine",
            "short-range paracrine",
            "medium-range paracrine",
            "longer-range niche-restricted chemokine-like",
            "optional high-expression false-positive negative controls in legacy presets",
        ],
        "training_fairness_note": (
            "Weak-positive augmentation and weak-signal reweighting are not used by "
            "the clean SpatialSelf runner. Weak labels remain evaluation annotations only."
        ),
        "datasets_requested": requested,
        "datasets_generated": datasets,
        "records": records,
    }
    (output_dir / "fair_benchmark_v2_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False) if len(manifest) else "No datasets generated.")
    print(f"Wrote fair benchmark v2 manifest to {output_dir}")


if __name__ == "__main__":
    main()
