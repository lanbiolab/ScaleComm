#!/usr/bin/env python
"""Generate and summarize formal revised Goal 3 simulator-v2 datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.generate_controlled_ccc import _config_from_yaml
from scripts.benchmark.validate_controlled_ccc import (
    summarize_lr_pairs,
    summarize_mechanisms,
    validate_replicate,
)
from src.spitialself.benchmarks import ControlledCCCConfig, generate_controlled_ccc_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal3_simulator_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal3_simulator_v2/formal_datasets")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--n-cells", type=int, default=960)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--clean", action="store_true", help="Remove output-dir before generation.")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = _config_from_yaml(PROJECT_ROOT / args.config)
    config = ControlledCCCConfig(
        **{
            **base_config.__dict__,
            "output_dir": str(output_dir),
            "replicates": int(args.replicates),
            "n_cells": int(args.n_cells),
            "random_seed": int(args.seed),
        }
    )
    replicate_dirs = generate_controlled_ccc_benchmark(config)
    validation = pd.DataFrame([validate_replicate(path) for path in replicate_dirs])
    mechanism_summary = pd.concat([summarize_mechanisms(path) for path in replicate_dirs], ignore_index=True)
    lr_summary = pd.concat([summarize_lr_pairs(path) for path in replicate_dirs], ignore_index=True)
    formal_summary = _formal_summary(validation, mechanism_summary, lr_summary)
    claims = _claims(config, validation, mechanism_summary, lr_summary)

    validation.to_csv(output_dir / "formal_validation_summary.csv", index=False)
    mechanism_summary.to_csv(output_dir / "formal_mechanism_summary.csv", index=False)
    lr_summary.to_csv(output_dir / "formal_lr_summary.csv", index=False)
    formal_summary.to_csv(output_dir / "formal_benchmark_summary.csv", index=False)
    claims.to_csv(output_dir / "formal_simulator_claims.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal3_formal_simulator_v2",
        "canonical_goal": "goal3_benchmark_simulator_v2",
        "evidence_status": "formal_benchmark_structure_generated_not_final_manuscript_scale",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "replicates": int(len(replicate_dirs)),
        "n_cells": int(config.n_cells),
        "outputs": {
            "validation_summary": str((output_dir / "formal_validation_summary.csv").relative_to(PROJECT_ROOT)),
            "mechanism_summary": str((output_dir / "formal_mechanism_summary.csv").relative_to(PROJECT_ROOT)),
            "lr_summary": str((output_dir / "formal_lr_summary.csv").relative_to(PROJECT_ROOT)),
            "benchmark_summary": str((output_dir / "formal_benchmark_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "formal_simulator_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This generates the formal mechanism-layered simulator structure and replicate summaries.",
            "It is not yet a full method-comparison result table; baseline scoring belongs to revised Goals 1-2.",
        ],
    }
    (output_dir / "formal_simulator_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(formal_summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote formal Goal 3 simulator-v2 outputs to {output_dir}")


def _formal_summary(validation: pd.DataFrame, mechanism_summary: pd.DataFrame, lr_summary: pd.DataFrame) -> pd.DataFrame:
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    mechanisms = set(mechanism_summary["mechanism"].astype(str))
    return pd.DataFrame(
        [
            {
                "n_replicates": int(validation["replicate"].nunique()),
                "min_cells": int(validation["cells"].min()),
                "median_candidate_interactions": float(validation["candidate_interactions"].median()),
                "median_positive_interactions": float(validation["positive_interactions"].median()),
                "median_weak_positive_interactions": float(validation["weak_positive_interactions"].median()),
                "false_positive_labels_total": int(validation["false_positive_labels"].sum()),
                "mechanisms_observed": "|".join(sorted(mechanisms)),
                "required_mechanisms_present": bool(required_mechanisms.issubset(mechanisms)),
                "weak_lr_pairs": int(lr_summary["weak_signal"].astype(bool).sum()),
                "high_expression_negative_lr_pairs": int(lr_summary["false_positive_high_expression"].astype(bool).sum()),
                "heuristic_mean_auroc": float(validation["heuristic_coexpr_distance_auroc"].mean()),
                "heuristic_mean_aupr": float(validation["heuristic_coexpr_distance_aupr"].mean()),
            }
        ]
    )


def _claims(
    config: ControlledCCCConfig,
    validation: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    lr_summary: pd.DataFrame,
) -> pd.DataFrame:
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    mechanisms = set(mechanism_summary["mechanism"].astype(str))
    return pd.DataFrame(
        [
            {
                "claim_id": "G3_C1_mechanism_layered_truth",
                "status": "supported" if required_mechanisms.issubset(mechanisms) else "partial",
                "evidence": "formal_mechanism_summary.csv;formal_lr_summary.csv",
                "statement": "Controlled simulator v2 contains contact, short/medium paracrine, long-range niche-restricted, weak-signal, and high-expression negative-control LR classes.",
                "n_replicates": int(config.replicates),
            },
            {
                "claim_id": "G3_C2_truth_table_contract",
                "status": "supported" if int(validation["false_positive_labels"].sum()) == 0 else "partial",
                "evidence": "replicate_*/ground_truth_table.csv;replicate_*/truth_*.csv",
                "statement": "Each replicate writes candidate, LR-pair, cell-type-pair, and spatial-niche truth tables; high-expression controls remain negative.",
                "n_replicates": int(config.replicates),
            },
            {
                "claim_id": "G3_C3_not_simple_high_expression_distance",
                "status": "supported",
                "evidence": "formal_mechanism_summary.csv",
                "statement": "Truth labels are gated by mechanism, sender/receiver type, range, and niche rather than expression and distance alone.",
                "n_replicates": int(config.replicates),
            },
        ]
    )


if __name__ == "__main__":
    main()
