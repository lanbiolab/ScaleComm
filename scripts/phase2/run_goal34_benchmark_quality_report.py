#!/usr/bin/env python
"""Assemble Goal 3/4 benchmark quality and realism evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_MECHANISMS = (
    "juxtacrine_contact",
    "short_medium_paracrine",
    "long_range_chemokine_niche",
    "false_positive_high_expression",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal34_benchmark_quality_report")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    g3_root = PROJECT_ROOT / "outputs/phase2_revised/goal3_simulator_v2/formal_datasets"
    g4_root = PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/formal_datasets"
    g3_summary = pd.read_csv(g3_root / "formal_benchmark_summary.csv")
    g3_validation = pd.read_csv(g3_root / "formal_validation_summary.csv")
    g3_mechanism = pd.read_csv(g3_root / "formal_mechanism_summary.csv")
    g3_lr = pd.read_csv(g3_root / "formal_lr_summary.csv")
    g4_summary = pd.read_csv(g4_root / "formal_real_background_summary.csv")
    g4_validation = pd.read_csv(g4_root / "formal_validation_summary.csv")
    g4_mechanism = pd.read_csv(g4_root / "formal_mechanism_summary.csv")
    g4_injection = pd.read_csv(g4_root / "formal_injection_summary.csv")

    quality = _quality_summary(g3_summary, g3_validation, g4_summary, g4_validation)
    mechanism_matrix = _mechanism_matrix(g3_mechanism, g4_mechanism)
    real_background = _real_background_properties(g4_validation)
    truth_contract = _truth_contract(g3_summary, g3_validation, g3_lr, g4_summary, g4_validation, g4_injection)
    claims = _claims(quality, mechanism_matrix, real_background, truth_contract)

    quality.to_csv(output_dir / "goal34_benchmark_quality_summary.csv", index=False)
    mechanism_matrix.to_csv(output_dir / "goal34_mechanism_coverage_matrix.csv", index=False)
    real_background.to_csv(output_dir / "goal34_real_background_property_summary.csv", index=False)
    truth_contract.to_csv(output_dir / "goal34_truth_contract_summary.csv", index=False)
    claims.to_csv(output_dir / "goal34_benchmark_quality_claims.csv", index=False)

    manifest = {
        "analysis_label": "phase2_revised_goal34_benchmark_quality_report",
        "canonical_goals": ["goal3_benchmark_simulator_v2", "goal4_real_background_semisynthetic_benchmark"],
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "claim_level": "benchmark_quality_realism_summary_not_final_external_realism_test",
        "inputs": {
            "goal3_summary": str((g3_root / "formal_benchmark_summary.csv").relative_to(PROJECT_ROOT)),
            "goal3_validation": str((g3_root / "formal_validation_summary.csv").relative_to(PROJECT_ROOT)),
            "goal3_mechanism": str((g3_root / "formal_mechanism_summary.csv").relative_to(PROJECT_ROOT)),
            "goal4_summary": str((g4_root / "formal_real_background_summary.csv").relative_to(PROJECT_ROOT)),
            "goal4_validation": str((g4_root / "formal_validation_summary.csv").relative_to(PROJECT_ROOT)),
            "goal4_mechanism": str((g4_root / "formal_mechanism_summary.csv").relative_to(PROJECT_ROOT)),
        },
        "outputs": {
            "quality_summary": str((output_dir / "goal34_benchmark_quality_summary.csv").relative_to(PROJECT_ROOT)),
            "mechanism_coverage": str((output_dir / "goal34_mechanism_coverage_matrix.csv").relative_to(PROJECT_ROOT)),
            "real_background_properties": str((output_dir / "goal34_real_background_property_summary.csv").relative_to(PROJECT_ROOT)),
            "truth_contract": str((output_dir / "goal34_truth_contract_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal34_benchmark_quality_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This report audits generated benchmark structure and retained real-background properties.",
            "It is not a full external realism test against held-out ST datasets.",
            "Controlled simulator realism is mechanism/truth-contract evidence, not a claim that the simulator reproduces all ST distributions.",
        ],
    }
    (output_dir / "goal34_benchmark_quality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(quality.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote Goal3/4 benchmark quality report to {output_dir}")


def _quality_summary(
    g3_summary: pd.DataFrame,
    g3_validation: pd.DataFrame,
    g4_summary: pd.DataFrame,
    g4_validation: pd.DataFrame,
) -> pd.DataFrame:
    g3 = g3_summary.iloc[0]
    rows = [
        {
            "benchmark": "controlled_simulator_v2",
            "dataset_key": "controlled",
            "n_replicates": int(g3["n_replicates"]),
            "min_cells": int(g3["min_cells"]),
            "median_candidate_interactions": float(g3["median_candidate_interactions"]),
            "median_positive_interactions": float(g3["median_positive_interactions"]),
            "median_weak_positive_interactions": float(g3["median_weak_positive_interactions"]),
            "false_positive_labels_total": int(g3["false_positive_labels_total"]),
            "required_mechanisms_present": bool(g3["required_mechanisms_present"]),
            "preserved_property_fields_present": False,
            "heuristic_mean_auroc": float(g3["heuristic_mean_auroc"]),
            "heuristic_mean_aupr": float(g3["heuristic_mean_aupr"]),
            "quality_scope": "mechanism_layered_truth_contract",
        }
    ]
    for row in g4_summary.itertuples(index=False):
        rows.append(
            {
                "benchmark": "real_background_semisynthetic_v2",
                "dataset_key": row.dataset_key,
                "n_replicates": int(row.n_replicates),
                "min_cells": int(row.min_cells),
                "median_candidate_interactions": float(row.median_candidate_interactions),
                "median_positive_interactions": float(row.median_positive_interactions),
                "median_weak_positive_interactions": float(row.median_weak_positive_interactions),
                "false_positive_labels_total": int(row.false_positive_labels_total),
                "required_mechanisms_present": _has_required_real_background_mechanisms(str(row.mechanisms_observed)),
                "preserved_property_fields_present": bool(row.preserved_property_fields_present),
                "heuristic_mean_auroc": float(row.heuristic_mean_auroc),
                "heuristic_mean_aupr": float(row.heuristic_mean_aupr),
                "quality_scope": "real_st_background_preservation_plus_injected_truth",
            }
        )
    table = pd.DataFrame(rows)
    table["heuristic_not_trivial"] = table["heuristic_mean_auroc"].between(0.25, 0.9) & table["heuristic_mean_aupr"].between(0.01, 0.6)
    table["truth_contract_ok"] = (
        (table["median_positive_interactions"] > 0)
        & (table["median_weak_positive_interactions"] > 0)
        & (table["false_positive_labels_total"] == 0)
    )
    table["benchmark_quality_pass"] = table["truth_contract_ok"] & table["heuristic_not_trivial"]
    table.loc[table["benchmark"] == "controlled_simulator_v2", "benchmark_quality_pass"] &= table.loc[
        table["benchmark"] == "controlled_simulator_v2", "required_mechanisms_present"
    ]
    table.loc[table["benchmark"] == "real_background_semisynthetic_v2", "benchmark_quality_pass"] &= table.loc[
        table["benchmark"] == "real_background_semisynthetic_v2", "preserved_property_fields_present"
    ]
    return table


def _mechanism_matrix(g3_mechanism: pd.DataFrame, g4_mechanism: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in REQUIRED_MECHANISMS:
        g3_rows = g3_mechanism[g3_mechanism["mechanism"].astype(str) == mechanism]
        rows.append(
            {
                "benchmark": "controlled_simulator_v2",
                "dataset_key": "controlled",
                "mechanism": mechanism,
                "present": not g3_rows.empty,
                "candidate_edges": int(g3_rows["n_candidates"].sum()) if "n_candidates" in g3_rows else int(g3_rows["candidate_edges"].sum()) if "candidate_edges" in g3_rows else 0,
                "positive_edges": int(g3_rows["n_positive"].sum()) if "n_positive" in g3_rows else int(g3_rows["positive_edges"].sum()) if "positive_edges" in g3_rows else 0,
            }
        )
    for dataset_key, dataset_group in g4_mechanism.groupby("dataset_key", sort=False):
        for mechanism in REQUIRED_MECHANISMS:
            rows_for_mech = dataset_group[dataset_group["mechanism"].astype(str) == mechanism]
            rows.append(
                {
                    "benchmark": "real_background_semisynthetic_v2",
                    "dataset_key": dataset_key,
                    "mechanism": mechanism,
                    "present": not rows_for_mech.empty,
                    "candidate_edges": int(_sum_first_available(rows_for_mech, ["n_candidates", "candidate_edges", "candidate_interactions"])),
                    "positive_edges": int(_sum_first_available(rows_for_mech, ["n_positive", "positive_edges", "positive_interactions"])),
                }
            )
    return pd.DataFrame(rows)


def _real_background_properties(g4_validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in g4_validation.itertuples(index=False):
        rows.append(
            {
                "dataset_key": row.dataset_key,
                "replicate": row.replicate,
                "cells": int(row.cells),
                "expression_genes": int(row.expression_genes),
                "mean_observed_sparsity": float(row.mean_observed_sparsity),
                "median_library_size": float(row.median_library_size),
                "library_size_iqr": float(row.library_size_iqr),
                "coordinate_span_x": float(row.coordinate_span_x),
                "coordinate_span_y": float(row.coordinate_span_y),
                "median_nn_distance": float(row.median_nn_distance),
                "median_local_density": float(row.median_local_density),
                "regions": int(row.regions),
                "cell_types": int(row.cell_types),
                "dominant_region_fraction": float(row.dominant_region_fraction),
                "dominant_cell_type_fraction": float(row.dominant_cell_type_fraction),
                "property_pass": bool(
                    row.cells > 0
                    and row.expression_genes >= 1000
                    and 0.0 <= row.mean_observed_sparsity <= 1.0
                    and row.median_library_size > 0
                    and row.coordinate_span_x > 0
                    and row.coordinate_span_y > 0
                    and row.median_local_density > 0
                    and row.regions >= 2
                    and row.cell_types >= 2
                ),
            }
        )
    return pd.DataFrame(rows)


def _truth_contract(
    g3_summary: pd.DataFrame,
    g3_validation: pd.DataFrame,
    g3_lr: pd.DataFrame,
    g4_summary: pd.DataFrame,
    g4_validation: pd.DataFrame,
    g4_injection: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    g3 = g3_summary.iloc[0]
    rows.append(
        {
            "benchmark": "controlled_simulator_v2",
            "dataset_key": "controlled",
            "replicates": int(g3["n_replicates"]),
            "lr_pairs": int(g3_lr["lr_id"].nunique()),
            "weak_lr_pairs": int(g3["weak_lr_pairs"]),
            "high_expression_negative_lr_pairs": int(g3["high_expression_negative_lr_pairs"]),
            "truth_edges": int(g3_validation["positive_interactions"].sum()),
            "weak_truth_edges": int(g3_validation["weak_positive_interactions"].sum()),
            "false_positive_truth_edges": int(g3_validation["false_positive_labels"].sum()),
            "truth_contract_pass": bool(
                int(g3_validation["positive_interactions"].sum()) > 0
                and int(g3_validation["weak_positive_interactions"].sum()) > 0
                and int(g3_validation["false_positive_labels"].sum()) == 0
            ),
        }
    )
    for row in g4_summary.itertuples(index=False):
        validation = g4_validation[g4_validation["dataset_key"].astype(str) == str(row.dataset_key)]
        injections = g4_injection[g4_injection["dataset_key"].astype(str) == str(row.dataset_key)]
        rows.append(
            {
                "benchmark": "real_background_semisynthetic_v2",
                "dataset_key": row.dataset_key,
                "replicates": int(row.n_replicates),
                "lr_pairs": int(row.lr_pairs),
                "weak_lr_pairs": int(row.weak_lr_pairs),
                "high_expression_negative_lr_pairs": int(row.high_expression_negative_lr_pairs),
                "truth_edges": int(validation["positive_interactions"].sum()),
                "weak_truth_edges": int(validation["weak_positive_interactions"].sum()),
                "false_positive_truth_edges": int(validation["false_positive_labels"].sum()),
                "truth_contract_pass": bool(
                    int(validation["positive_interactions"].sum()) > 0
                    and int(validation["weak_positive_interactions"].sum()) > 0
                    and int(validation["false_positive_labels"].sum()) == 0
                    and int(injections["false_positive_high_expression"].astype(bool).sum()) > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _claims(
    quality: pd.DataFrame,
    mechanism_matrix: pd.DataFrame,
    real_background: pd.DataFrame,
    truth_contract: pd.DataFrame,
) -> pd.DataFrame:
    real_quality = quality[quality["benchmark"] == "real_background_semisynthetic_v2"]
    controlled_quality = quality[quality["benchmark"] == "controlled_simulator_v2"]
    required_real_datasets = {"lymph_node", "tonsil", "melanoma"}
    real_datasets = set(real_quality["dataset_key"].astype(str))
    mechanism_pairs = set(zip(mechanism_matrix["dataset_key"].astype(str), mechanism_matrix["mechanism"].astype(str)))
    required_control_mechanisms = {("controlled", mechanism) for mechanism in REQUIRED_MECHANISMS}
    rows = [
        {
            "claim_id": "G34_C1_truth_contract_ready",
            "status": "supported" if truth_contract["truth_contract_pass"].astype(bool).all() else "partial",
            "statement": "Controlled and real-background formal benchmarks contain positive truth edges, weak positives, and high-expression negative controls that remain ground-truth negative.",
            "evidence": "goal34_truth_contract_summary.csv",
            "claim_level": "benchmark_quality_realism_summary_not_final_external_realism_test",
        },
        {
            "claim_id": "G34_C2_controlled_mechanism_coverage",
            "status": "supported" if required_control_mechanisms.issubset(mechanism_pairs) else "partial",
            "statement": "Controlled simulator v2 covers contact, short/medium paracrine, long-range niche, and high-expression negative-control mechanisms.",
            "evidence": "goal34_mechanism_coverage_matrix.csv",
            "claim_level": "benchmark_quality_realism_summary_not_final_external_realism_test",
        },
        {
            "claim_id": "G34_C3_real_background_triplet_quality",
            "status": "supported"
            if required_real_datasets.issubset(real_datasets) and real_quality["benchmark_quality_pass"].astype(bool).all()
            else "partial",
            "statement": "Real-background formal benchmarks cover lymph node, tonsil, and melanoma with preserved measured ST properties and injected truth controls.",
            "evidence": "goal34_benchmark_quality_summary.csv;goal34_real_background_property_summary.csv",
            "claim_level": "benchmark_quality_realism_summary_not_final_external_realism_test",
        },
        {
            "claim_id": "G34_C4_heuristic_not_decisive",
            "status": "supported" if quality["heuristic_not_trivial"].astype(bool).all() else "partial",
            "statement": "Simple coexpression-distance heuristic performance is nontrivial but not saturating, supporting use of the benchmark as a stress test rather than a solved toy.",
            "evidence": "goal34_benchmark_quality_summary.csv",
            "claim_level": "benchmark_quality_realism_summary_not_final_external_realism_test",
        },
    ]
    return pd.DataFrame(rows)


def _has_required_real_background_mechanisms(observed: str) -> bool:
    mechanisms = set(str(observed).split("|"))
    return {"short_medium_paracrine", "long_range_chemokine_niche", "false_positive_high_expression"}.issubset(mechanisms)


def _sum_first_available(table: pd.DataFrame, columns: list[str]) -> float:
    for column in columns:
        if column in table.columns:
            return float(pd.to_numeric(table[column], errors="coerce").fillna(0).sum())
    return 0.0


if __name__ == "__main__":
    main()
