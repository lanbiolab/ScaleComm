#!/usr/bin/env python
"""Generate the Phase 1 controlled ground-truth CCC benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks import ControlledCCCConfig, generate_controlled_ccc_benchmark


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load benchmark config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def _config_from_yaml(path: Path) -> ControlledCCCConfig:
    raw = _load_yaml(path)
    simulation = raw.get("simulation", {})
    controlled = simulation.get("controlled_generator", {})
    return ControlledCCCConfig(
        n_cells=int(controlled.get("n_cells", 720)),
        n_background_genes=int(controlled.get("n_background_genes", 32)),
        random_seed=int(raw.get("random_seed", controlled.get("random_seed", 2025))),
        replicates=int(simulation.get("replicates", controlled.get("replicates", 5))),
        coordinate_noise=float(controlled.get("coordinate_noise", 0.012)),
        contact_range=float(controlled.get("contact_range", 0.055)),
        paracrine_range=float(controlled.get("paracrine_range", 0.18)),
        chemokine_range=float(controlled.get("chemokine_range", 0.42)),
        max_candidate_range=float(controlled.get("max_candidate_range", 0.45)),
        negative_edge_fraction=float(controlled.get("negative_edge_fraction", 0.35)),
        candidate_generation=str(controlled.get("candidate_generation", "full_radius")),
        max_candidates_per_lr=controlled.get("max_candidates_per_lr"),
        max_candidates_per_sender_lr=int(controlled.get("max_candidates_per_sender_lr", 48)),
        output_dir=str(controlled.get("output_dir", raw.get("output_dir", ControlledCCCConfig.output_dir))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/exp01_controlled_benchmark.yaml",
        help="Experiment YAML config.",
    )
    parser.add_argument("--output-dir", help="Override generated dataset root.")
    parser.add_argument("--replicates", type=int, help="Override number of replicates.")
    parser.add_argument("--n-cells", type=int, help="Override number of simulated cells/spots.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
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
    if args.replicates is not None:
        updates["replicates"] = args.replicates
    if args.n_cells is not None:
        updates["n_cells"] = args.n_cells
    if args.seed is not None:
        updates["random_seed"] = args.seed
    if args.candidate_generation is not None:
        updates["candidate_generation"] = args.candidate_generation
    if args.max_candidates_per_lr is not None:
        updates["max_candidates_per_lr"] = args.max_candidates_per_lr
    if args.max_candidates_per_sender_lr is not None:
        updates["max_candidates_per_sender_lr"] = args.max_candidates_per_sender_lr
    if updates:
        config = ControlledCCCConfig(**{**config.__dict__, **updates})

    replicate_dirs = generate_controlled_ccc_benchmark(config)
    print(f"Generated {len(replicate_dirs)} controlled benchmark replicate(s)")
    print(f"Dataset root: {config.output_dir}")
    for path in replicate_dirs:
        print(path)


if __name__ == "__main__":
    main()
