#!/usr/bin/env python
"""Generate and summarize formal revised Goal 4 real-background benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.validate_semisynthetic_real_background import (
    summarize_injections,
    summarize_mechanisms,
    validate_replicate,
)
from src.spitialself.benchmarks.semisynthetic import (
    SemiSyntheticConfig,
    generate_semisynthetic_real_background,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal4_real_background_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal4_real_background_v2/formal_datasets")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n-spots", type=int, default=900)
    parser.add_argument("--seed", type=int, default=4026)
    parser.add_argument("--datasets", nargs="+", default=["lymph_node", "tonsil", "melanoma"])
    parser.add_argument("--clean", action="store_true", help="Remove output-dir before generation.")
    args = parser.parse_args()

    raw = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / args.output_dir
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rep_dirs: list[Path] = []
    dataset_status: list[dict[str, Any]] = []
    for dataset_key in args.datasets:
        try:
            dataset_config = _dataset_config(
                raw,
                dataset_key,
                output_dir=output_dir,
                replicates=args.replicates,
                n_spots=args.n_spots,
                seed=args.seed,
                dataset_offset=len(dataset_status),
            )
            for config in dataset_config:
                rep_dir = generate_semisynthetic_real_background(config)
                all_rep_dirs.append(rep_dir)
                dataset_status.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_name": config.dataset_name,
                        "replicate": int(config.replicate_id),
                        "status": "generated",
                        "path": str(rep_dir),
                        "error": "",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - status table should record dataset-specific failures.
            dataset_status.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_name": "",
                    "replicate": "",
                    "status": "failed",
                    "path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not all_rep_dirs:
        pd.DataFrame(dataset_status).to_csv(output_dir / "formal_dataset_status.csv", index=False)
        raise SystemExit(f"No real-background datasets generated. See {output_dir / 'formal_dataset_status.csv'}")

    validation = pd.DataFrame([validate_replicate(path) | {"dataset_key": _dataset_key_from_path(path)} for path in all_rep_dirs])
    injection_summary = pd.concat(
        [summarize_injections(path).assign(dataset_key=_dataset_key_from_path(path)) for path in all_rep_dirs],
        ignore_index=True,
    )
    mechanism_summary = pd.concat(
        [summarize_mechanisms(path).assign(dataset_key=_dataset_key_from_path(path)) for path in all_rep_dirs],
        ignore_index=True,
    )
    benchmark_summary = _benchmark_summary(validation, injection_summary, mechanism_summary)
    claims = _claims(validation, injection_summary, mechanism_summary, dataset_status)

    pd.DataFrame(dataset_status).to_csv(output_dir / "formal_dataset_status.csv", index=False)
    validation.to_csv(output_dir / "formal_validation_summary.csv", index=False)
    injection_summary.to_csv(output_dir / "formal_injection_summary.csv", index=False)
    mechanism_summary.to_csv(output_dir / "formal_mechanism_summary.csv", index=False)
    benchmark_summary.to_csv(output_dir / "formal_real_background_summary.csv", index=False)
    claims.to_csv(output_dir / "formal_real_background_claims.csv", index=False)
    pd.DataFrame(
        {
            "replicate": list(range(len(all_rep_dirs))),
            "path": [str(path) for path in all_rep_dirs],
            "dataset_key": [_dataset_key_from_path(path) for path in all_rep_dirs],
        }
    ).to_csv(output_dir / "manifest.csv", index=False)

    manifest = {
        "analysis_label": "phase2_revised_goal4_formal_real_background_v2",
        "canonical_goal": "goal4_real_background_semisynthetic_benchmark",
        "evidence_status": "formal_real_background_structure_generated_not_final_manuscript_scale",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets_requested": args.datasets,
        "replicates_per_dataset": int(args.replicates),
        "n_spots": int(args.n_spots),
        "outputs": {
            "dataset_status": str((output_dir / "formal_dataset_status.csv").relative_to(PROJECT_ROOT)),
            "validation_summary": str((output_dir / "formal_validation_summary.csv").relative_to(PROJECT_ROOT)),
            "injection_summary": str((output_dir / "formal_injection_summary.csv").relative_to(PROJECT_ROOT)),
            "mechanism_summary": str((output_dir / "formal_mechanism_summary.csv").relative_to(PROJECT_ROOT)),
            "benchmark_summary": str((output_dir / "formal_real_background_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "formal_real_background_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This formalizes real-background benchmark generation and truth-table summaries.",
            "It is not yet a full tuned baseline comparison or curated downstream target validation.",
        ],
    }
    (output_dir / "formal_real_background_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(benchmark_summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote formal Goal 4 real-background outputs to {output_dir}")


def _dataset_config(
    raw: dict,
    dataset_key: str,
    *,
    output_dir: Path,
    replicates: int,
    n_spots: int,
    seed: int,
    dataset_offset: int,
) -> list[SemiSyntheticConfig]:
    common = raw.get("semisynthetic_generator", {})
    datasets = raw.get("background_datasets", {})
    if dataset_key not in datasets:
        raise KeyError(f"Unknown dataset key {dataset_key!r}; available={sorted(datasets)}")
    merged = {**common, **datasets[dataset_key]}
    merged["output_dir"] = str(output_dir)
    merged["n_spots"] = int(n_spots)
    configs: list[SemiSyntheticConfig] = []
    valid_fields = set(SemiSyntheticConfig.__dataclass_fields__)
    for replicate in range(int(replicates)):
        fields = {key: value for key, value in merged.items() if key in valid_fields}
        fields["random_seed"] = int(seed) + int(dataset_offset) * 1000 + replicate
        fields["replicate_id"] = int(replicate)
        configs.append(SemiSyntheticConfig(**fields))
    return configs


def _benchmark_summary(
    validation: pd.DataFrame,
    injection_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    required_properties = [
        "coordinate_span_x",
        "coordinate_span_y",
        "median_library_size",
        "mean_observed_sparsity",
        "median_local_density",
        "regions",
        "cell_types",
    ]
    for dataset_key, group in validation.groupby("dataset_key", sort=False):
        injections = injection_summary[injection_summary["dataset_key"] == dataset_key]
        mechanisms = mechanism_summary[mechanism_summary["dataset_key"] == dataset_key]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_replicates": int(group["replicate"].nunique()),
                "min_cells": int(group["cells"].min()),
                "median_candidate_interactions": float(group["candidate_interactions"].median()),
                "median_positive_interactions": float(group["positive_interactions"].median()),
                "median_weak_positive_interactions": float(group["weak_positive_interactions"].median()),
                "false_positive_labels_total": int(group["false_positive_labels"].sum()),
                "mechanisms_observed": "|".join(sorted(set(mechanisms["mechanism"].astype(str)))),
                "lr_pairs": int(injections["lr_id"].nunique()),
                "weak_lr_pairs": int(injections.drop_duplicates(["replicate", "lr_id"])["weak_signal"].astype(bool).sum()),
                "high_expression_negative_lr_pairs": int(
                    injections.drop_duplicates(["replicate", "lr_id"])["false_positive_high_expression"].astype(bool).sum()
                ),
                "preserved_property_fields_present": bool(all(column in group.columns for column in required_properties)),
                "median_library_size": float(group["median_library_size"].median()),
                "mean_observed_sparsity": float(group["mean_observed_sparsity"].mean()),
                "median_local_density": float(group["median_local_density"].median()),
                "median_regions": float(group["regions"].median()),
                "median_cell_types": float(group["cell_types"].median()),
                "heuristic_mean_auroc": float(group["heuristic_coexpr_distance_auroc"].mean()),
                "heuristic_mean_aupr": float(group["heuristic_coexpr_distance_aupr"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _claims(
    validation: pd.DataFrame,
    injection_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    dataset_status: list[dict[str, Any]],
) -> pd.DataFrame:
    completed = {row["dataset_key"] for row in dataset_status if row["status"] == "generated"}
    requested = {row["dataset_key"] for row in dataset_status}
    required_datasets = {"lymph_node", "tonsil", "melanoma"}
    false_positive_labels = int(validation["false_positive_labels"].sum()) if len(validation) else -1
    has_weak = bool((validation["weak_positive_interactions"] > 0).all()) if len(validation) else False
    mechanisms = set(mechanism_summary["mechanism"].astype(str)) if len(mechanism_summary) else set()
    return pd.DataFrame(
        [
            {
                "claim_id": "G4_C1_real_background_triplet",
                "status": "supported" if required_datasets.issubset(completed) else "partial",
                "evidence": "formal_dataset_status.csv;formal_validation_summary.csv",
                "statement": "Formal real-background benchmark covers lymph node, tonsil, and melanoma backgrounds when local data are available.",
                "requested": "|".join(sorted(requested)),
                "completed": "|".join(sorted(completed)),
            },
            {
                "claim_id": "G4_C2_preserved_st_properties",
                "status": "supported" if _preserved_properties_ok(validation) else "partial",
                "evidence": "formal_validation_summary.csv",
                "statement": "Generated benchmarks preserve coordinates, library size, sparsity, local density, region/domain structure, and cell-type composition as measured fields.",
                "requested": "|".join(sorted(requested)),
                "completed": "|".join(sorted(completed)),
            },
            {
                "claim_id": "G4_C3_injected_truth_controls",
                "status": "supported" if false_positive_labels == 0 and has_weak else "partial",
                "evidence": "formal_injection_summary.csv;replicate_*/ground_truth_table.csv",
                "statement": "Injected LR signals include weak positives and high-expression negative controls that remain ground-truth negative.",
                "requested": "|".join(sorted(requested)),
                "completed": "|".join(sorted(completed)),
            },
            {
                "claim_id": "G4_C4_mechanism_diversity_under_real_background",
                "status": "supported" if {"juxtacrine_contact", "short_medium_paracrine", "long_range_chemokine_niche", "false_positive_high_expression"}.issubset(mechanisms) else "partial",
                "evidence": "formal_mechanism_summary.csv",
                "statement": "Across real-background datasets, injected truth spans contact, short/medium paracrine, long-range niche-restricted, and high-expression negative-control mechanisms.",
                "requested": "|".join(sorted(requested)),
                "completed": "|".join(sorted(completed)),
            },
        ]
    )


def _preserved_properties_ok(validation: pd.DataFrame) -> bool:
    if validation.empty:
        return False
    return bool(
        (validation["coordinate_span_x"] > 0).all()
        and (validation["coordinate_span_y"] > 0).all()
        and (validation["median_library_size"] > 0).all()
        and (validation["mean_observed_sparsity"].between(0, 1)).all()
        and (validation["median_local_density"] > 0).all()
        and (validation["regions"] >= 2).all()
        and (validation["cell_types"] >= 2).all()
    )


def _dataset_key_from_path(path: Path) -> str:
    name = path.name
    if "lymph" in name:
        return "lymph_node"
    if "tonsil" in name:
        return "tonsil"
    if "melanoma" in name:
        return "melanoma"
    return "unknown"


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
