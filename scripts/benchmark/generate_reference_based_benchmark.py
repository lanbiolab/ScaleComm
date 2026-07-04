#!/usr/bin/env python
"""Generate a reference-simulator-backed CCC benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks import SemiSyntheticConfig, generate_semisynthetic_real_background


DATASET_PRESETS = {
    "lymph_node": {
        "dataset_name": "reference_lymph_node",
        "background_format": "h5ad",
        "h5ad_path": "Human_Lymph_Node.h5ad",
        "region_csv": "visium_lr_benchmark/spot_region_from_c2l.csv",
        "injection_set": "range_lymph_node",
        "n_spots": 800,
        "n_background_genes": 600,
        "candidate_generation": "range_stress",
        "negative_edge_fraction": 0.45,
        "max_candidate_range_quantile": 0.08,
        "max_candidates_per_sender_lr": 48,
    },
    "melanoma": {
        "dataset_name": "reference_melanoma",
        "background_format": "matrix_market",
        "matrix_dir": "HumanMelannomaMultiome/data",
        "spatial_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_spatial.csv",
        "metadata_csv": "HumanMelannomaMultiome/data/HumanMelanomaMultiome_metadata.csv",
        "cell_id_col": "NAME",
        "x_col": "X",
        "y_col": "Y",
        "cell_type_col": "cell_type",
        "metadata_cell_type_col": "cluster",
        "injection_set": "range_melanoma",
        "n_spots": 800,
        "n_background_genes": 600,
        "candidate_generation": "range_stress",
        "negative_edge_fraction": 0.45,
        "max_candidate_range_quantile": 0.05,
        "max_candidates_per_sender_lr": 48,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_PRESETS), default="lymph_node")
    parser.add_argument("--simulator", choices=("srtsim", "scdesign3"), default="scdesign3")
    parser.add_argument("--output-dir", default="outputs/reference_based_benchmark/datasets")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replicate-id", type=int, default=0)
    parser.add_argument("--n-spots", type=int)
    parser.add_argument("--n-background-genes", type=int)
    parser.add_argument("--simulator-output-dir")
    parser.add_argument("--simulator-config-json")
    parser.add_argument("--simulator-r-script")
    args = parser.parse_args()

    fields = dict(DATASET_PRESETS[args.dataset])
    fields["dataset_name"] = f"{fields['dataset_name']}_{args.simulator}"
    fields["background_simulator"] = args.simulator
    fields["output_dir"] = str(Path(args.output_dir))
    fields["random_seed"] = int(args.seed)
    fields["replicate_id"] = int(args.replicate_id)
    if args.n_spots is not None:
        fields["n_spots"] = int(args.n_spots)
    if args.n_background_genes is not None:
        fields["n_background_genes"] = int(args.n_background_genes)
    if args.simulator_output_dir:
        fields["simulator_output_dir"] = args.simulator_output_dir
    if args.simulator_config_json:
        fields["simulator_config_json"] = args.simulator_config_json
    if args.simulator_r_script:
        fields["simulator_r_script"] = args.simulator_r_script

    config = SemiSyntheticConfig(**fields)
    replicate_dir = generate_semisynthetic_real_background(config)
    manifest = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "simulator": args.simulator,
                "replicate_id": int(args.replicate_id),
                "path": str(replicate_dir),
            }
        ]
    )
    manifest_path = Path(args.output_dir) / "reference_based_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(manifest.to_string(index=False))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
