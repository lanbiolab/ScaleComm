#!/usr/bin/env python
"""Validate the shared Goal 3/4 benchmark truth-table contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_FILES = (
    "cells.csv",
    "expression.csv",
    "lr_pairs.csv",
    "candidate_interactions.csv",
    "ground_truth_table.csv",
    "truth_lr_pairs.csv",
    "truth_cell_type_pairs.csv",
    "truth_spatial_niches.csv",
    "metadata.json",
)

REQUIRED_CONTROLLED_MECHANISMS = {
    "juxtacrine_contact",
    "short_medium_paracrine",
    "long_range_chemokine_niche",
    "false_positive_high_expression",
}

REQUIRED_REAL_BACKGROUND_MECHANISMS = {
    "short_medium_paracrine",
    "long_range_chemokine_niche",
    "false_positive_high_expression",
}

REQUIRED_LR_COLUMNS_COMMON = {
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "weak_signal",
    "false_positive_high_expression",
    "true_lr",
}

REQUIRED_LR_COLUMNS_CONTROLLED = {
    "sender_cell_type",
    "receiver_cell_type",
    "spatial_range",
    "expression_strength",
    "background_expression",
    "active_niche",
}

REQUIRED_LR_COLUMNS_REAL_BACKGROUND = {
    "sender_classes",
    "receiver_classes",
    "active_regions",
    "range_multiplier",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "sender_id",
    "receiver_id",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "distance",
    "sender_cell_type",
    "receiver_cell_type",
    "sender_niche",
    "receiver_niche",
    "ground_truth",
    "weak_signal",
    "false_positive_high_expression",
    "type_rule_satisfied",
    "range_rule_satisfied",
    "niche_rule_satisfied",
    "ligand_expr",
    "receptor_expr",
}

REQUIRED_GROUND_TRUTH_COLUMNS = {
    "candidate_id",
    "sender_id",
    "receiver_id",
    "lr_id",
    "mechanism_type",
    "sender_cell_type",
    "receiver_cell_type",
    "sender_spatial_niche",
    "receiver_spatial_niche",
    "active_niche",
    "distance",
    "spatial_range",
    "weak_signal",
    "weak_or_strong",
    "false_positive_high_expression",
    "true_lr_pair",
    "true_cell_type_pair",
    "true_spatial_niche",
    "true_edge",
}

REQUIRED_TRUTH_LR_COLUMNS = {
    "lr_id",
    "ligand",
    "receptor",
    "mechanism_type",
    "weak_signal",
    "weak_or_strong",
    "false_positive_high_expression",
    "candidate_edges",
    "true_edges",
    "true_lr_pair",
}

REQUIRED_TRUTH_CELL_TYPE_COLUMNS = {
    "lr_id",
    "mechanism_type",
    "weak_or_strong",
    "sender_cell_type",
    "receiver_cell_type",
    "candidate_edges",
    "true_edges",
    "true_cell_type_pair",
}

REQUIRED_TRUTH_NICHE_COLUMNS = {
    "lr_id",
    "mechanism_type",
    "weak_or_strong",
    "sender_spatial_niche",
    "receiver_spatial_niche",
    "active_niche",
    "candidate_edges",
    "true_edges",
    "true_spatial_niche",
}

REQUIRED_REAL_PROPERTIES = {
    "spatial_coordinates",
    "library_size",
    "observed_expression_sparsity",
    "cell_type_composition",
    "local_density",
    "tissue_domain_structure",
}


@dataclass
class ContractRow:
    benchmark: str
    dataset_key: str
    replicate: str
    status: str
    cells: int
    lr_pairs: int
    candidate_interactions: int
    true_edges: int
    weak_true_edges: int
    false_positive_true_edges: int
    mechanisms_observed: str
    required_mechanisms_present: bool
    truth_rows_match_candidates: bool
    truth_summary_has_positives: bool
    weak_labels_present: bool
    strong_labels_present: bool
    high_expression_controls_present: bool
    high_expression_controls_negative: bool
    real_properties_present: bool
    missing_files: str
    missing_columns: str
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-root",
        default="outputs/phase2_revised/goal3_simulator_v2/formal_datasets",
        help="Root containing controlled simulator replicate_* directories.",
    )
    parser.add_argument(
        "--real-background-root",
        default="outputs/phase2_revised/goal4_real_background_v2/replicated_tuned_datasets",
        help="Root containing real-background *_replicate_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2_revised/goal34_truth_contract",
        help="Output directory for contract summaries.",
    )
    args = parser.parse_args()

    controlled_root = PROJECT_ROOT / args.controlled_root
    real_root = PROJECT_ROOT / args.real_background_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ContractRow] = []
    controlled_dirs = sorted(path for path in controlled_root.glob("replicate_*") if path.is_dir())
    real_dirs = sorted(path for path in real_root.glob("*_replicate_*") if path.is_dir())

    for rep_dir in controlled_dirs:
        rows.append(_validate_replicate(rep_dir, benchmark="controlled_simulator_v2", dataset_key="controlled"))
    for rep_dir in real_dirs:
        rows.append(
            _validate_replicate(
                rep_dir,
                benchmark="real_background_semisynthetic_v2",
                dataset_key=_dataset_key_from_real_replicate(rep_dir.name),
            )
        )

    if not rows:
        raise SystemExit("No benchmark replicate directories found to validate.")

    table = pd.DataFrame([asdict(row) for row in rows])
    dataset_summary = _dataset_summary(table)
    claim_summary = _claim_summary(table, dataset_summary)

    contract_path = output_dir / "goal34_truth_contract_replicate_summary.csv"
    dataset_path = output_dir / "goal34_truth_contract_dataset_summary.csv"
    claims_path = output_dir / "goal34_truth_contract_claims.csv"
    manifest_path = output_dir / "goal34_truth_contract_manifest.json"
    table.to_csv(contract_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    claim_summary.to_csv(claims_path, index=False)
    manifest = {
        "analysis_label": "phase2_goal34_truth_contract",
        "controlled_root": _rel(controlled_root),
        "real_background_root": _rel(real_root),
        "output_dir": _rel(output_dir),
        "replicates_validated": int(len(table)),
        "complete_replicates": int((table["status"] == "complete").sum()),
        "outputs": {
            "replicate_summary": _rel(contract_path),
            "dataset_summary": _rel(dataset_path),
            "claims": _rel(claims_path),
        },
        "contract_scope": (
            "Shared truth-table schema and invariants for controlled simulator v2 "
            "and real-background semi-synthetic benchmarks."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(dataset_summary.to_string(index=False))
    print(claim_summary.to_string(index=False))
    print(f"Wrote Goal3/4 truth contract validation to {output_dir}")

    if not (table["status"] == "complete").all():
        raise SystemExit("At least one benchmark replicate failed the truth contract.")


def _validate_replicate(rep_dir: Path, *, benchmark: str, dataset_key: str) -> ContractRow:
    missing_files = [name for name in REQUIRED_FILES if not (rep_dir / name).exists()]
    if missing_files:
        return _failed_row(
            benchmark=benchmark,
            dataset_key=dataset_key,
            replicate=rep_dir.name,
            missing_files=missing_files,
            detail="missing required files",
        )

    try:
        metadata = json.loads((rep_dir / "metadata.json").read_text(encoding="utf-8"))
        cells = pd.read_csv(rep_dir / "cells.csv")
        expression = pd.read_csv(rep_dir / "expression.csv")
        lr_pairs = pd.read_csv(rep_dir / "lr_pairs.csv")
        candidates = pd.read_csv(rep_dir / "candidate_interactions.csv")
        truth = pd.read_csv(rep_dir / "ground_truth_table.csv")
        truth_lr = pd.read_csv(rep_dir / "truth_lr_pairs.csv")
        truth_ct = pd.read_csv(rep_dir / "truth_cell_type_pairs.csv")
        truth_niche = pd.read_csv(rep_dir / "truth_spatial_niches.csv")
    except Exception as exc:  # pragma: no cover - explicit audit output is more useful than a traceback.
        return _failed_row(
            benchmark=benchmark,
            dataset_key=dataset_key,
            replicate=rep_dir.name,
            detail=f"failed to read tables: {exc}",
        )

    lr_required = set(REQUIRED_LR_COLUMNS_COMMON)
    if benchmark == "controlled_simulator_v2":
        lr_required.update(REQUIRED_LR_COLUMNS_CONTROLLED)
        required_mechanisms = REQUIRED_CONTROLLED_MECHANISMS
    else:
        lr_required.update(REQUIRED_LR_COLUMNS_REAL_BACKGROUND)
        required_mechanisms = REQUIRED_REAL_BACKGROUND_MECHANISMS

    missing_columns: dict[str, list[str]] = {
        "lr_pairs": sorted(lr_required - set(lr_pairs.columns)),
        "candidate_interactions": sorted(REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)),
        "ground_truth_table": sorted(REQUIRED_GROUND_TRUTH_COLUMNS - set(truth.columns)),
        "truth_lr_pairs": sorted(REQUIRED_TRUTH_LR_COLUMNS - set(truth_lr.columns)),
        "truth_cell_type_pairs": sorted(REQUIRED_TRUTH_CELL_TYPE_COLUMNS - set(truth_ct.columns)),
        "truth_spatial_niches": sorted(REQUIRED_TRUTH_NICHE_COLUMNS - set(truth_niche.columns)),
    }
    missing_columns = {key: value for key, value in missing_columns.items() if value}

    mechanisms = set(lr_pairs["mechanism"].astype(str)) if "mechanism" in lr_pairs else set()
    true_edges = _sum_int(truth, "true_edge")
    weak_true_edges = _sum_int(truth[truth["weak_signal"].astype(bool)] if "weak_signal" in truth else truth.iloc[0:0], "true_edge")
    fp_truth = truth[truth["false_positive_high_expression"].astype(bool)] if "false_positive_high_expression" in truth else truth.iloc[0:0]
    false_positive_true_edges = _sum_int(fp_truth, "true_edge")
    high_expression_controls_present = bool(
        "false_positive_high_expression" in lr_pairs
        and lr_pairs["false_positive_high_expression"].astype(bool).any()
    )
    weak_labels_present = bool("weak_or_strong" in truth and (truth["weak_or_strong"].astype(str) == "weak").any())
    strong_labels_present = bool("weak_or_strong" in truth and (truth["weak_or_strong"].astype(str) == "strong").any())
    truth_rows_match_candidates = bool(
        len(truth) == len(candidates)
        and "ground_truth" in candidates
        and _sum_int(truth, "true_edge") == int(candidates["ground_truth"].astype(int).sum())
    )
    truth_summary_has_positives = bool(
        _sum_int(truth_lr, "true_lr_pair") > 0
        and _sum_int(truth_ct, "true_cell_type_pair") > 0
        and _sum_int(truth_niche, "true_spatial_niche") > 0
    )
    real_properties_present = True
    real_property_detail = ""
    if benchmark == "real_background_semisynthetic_v2":
        preserved = set(metadata.get("preserved_real_properties", []))
        real_properties_present = bool(REQUIRED_REAL_PROPERTIES.issubset(preserved))
        real_property_detail = f"missing_real_properties={sorted(REQUIRED_REAL_PROPERTIES - preserved)}"
        real_properties_present = real_properties_present and _real_cell_fields_non_degenerate(cells)

    expression_rows_match = len(expression) == len(cells)
    metadata_count_ok = int(metadata.get("counts", {}).get("candidate_interactions", len(candidates))) == len(candidates)
    required_mechanisms_present = required_mechanisms.issubset(mechanisms)
    high_expression_controls_negative = high_expression_controls_present and false_positive_true_edges == 0
    complete = all(
        [
            not missing_columns,
            len(cells) > 0,
            len(lr_pairs) > 0,
            len(candidates) > 0,
            expression_rows_match,
            metadata_count_ok,
            required_mechanisms_present,
            true_edges > 0,
            weak_true_edges > 0,
            truth_rows_match_candidates,
            truth_summary_has_positives,
            weak_labels_present,
            strong_labels_present,
            high_expression_controls_present,
            high_expression_controls_negative,
            real_properties_present,
        ]
    )

    detail_parts = [
        f"expression_rows_match={expression_rows_match}",
        f"metadata_count_ok={metadata_count_ok}",
    ]
    if real_property_detail:
        detail_parts.append(real_property_detail)
    return ContractRow(
        benchmark=benchmark,
        dataset_key=dataset_key,
        replicate=rep_dir.name,
        status="complete" if complete else "partial",
        cells=int(len(cells)),
        lr_pairs=int(len(lr_pairs)),
        candidate_interactions=int(len(candidates)),
        true_edges=int(true_edges),
        weak_true_edges=int(weak_true_edges),
        false_positive_true_edges=int(false_positive_true_edges),
        mechanisms_observed="|".join(sorted(mechanisms)),
        required_mechanisms_present=bool(required_mechanisms_present),
        truth_rows_match_candidates=bool(truth_rows_match_candidates),
        truth_summary_has_positives=bool(truth_summary_has_positives),
        weak_labels_present=bool(weak_labels_present),
        strong_labels_present=bool(strong_labels_present),
        high_expression_controls_present=bool(high_expression_controls_present),
        high_expression_controls_negative=bool(high_expression_controls_negative),
        real_properties_present=bool(real_properties_present),
        missing_files="",
        missing_columns=json.dumps(missing_columns, sort_keys=True),
        detail="; ".join(detail_parts),
    )


def _failed_row(
    *,
    benchmark: str,
    dataset_key: str,
    replicate: str,
    missing_files: list[str] | None = None,
    detail: str = "",
) -> ContractRow:
    return ContractRow(
        benchmark=benchmark,
        dataset_key=dataset_key,
        replicate=replicate,
        status="missing",
        cells=0,
        lr_pairs=0,
        candidate_interactions=0,
        true_edges=0,
        weak_true_edges=0,
        false_positive_true_edges=0,
        mechanisms_observed="",
        required_mechanisms_present=False,
        truth_rows_match_candidates=False,
        truth_summary_has_positives=False,
        weak_labels_present=False,
        strong_labels_present=False,
        high_expression_controls_present=False,
        high_expression_controls_negative=False,
        real_properties_present=False,
        missing_files="|".join(missing_files or []),
        missing_columns="",
        detail=detail,
    )


def _dataset_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, dataset_key), group in table.groupby(["benchmark", "dataset_key"], sort=False):
        mechanisms: set[str] = set()
        for observed in group["mechanisms_observed"].astype(str):
            mechanisms.update(item for item in observed.split("|") if item)
        rows.append(
            {
                "benchmark": benchmark,
                "dataset_key": dataset_key,
                "replicates": int(len(group)),
                "complete_replicates": int((group["status"] == "complete").sum()),
                "all_replicates_complete": bool((group["status"] == "complete").all()),
                "min_cells": int(group["cells"].min()),
                "median_candidate_interactions": float(group["candidate_interactions"].median()),
                "truth_edges": int(group["true_edges"].sum()),
                "weak_truth_edges": int(group["weak_true_edges"].sum()),
                "false_positive_true_edges": int(group["false_positive_true_edges"].sum()),
                "mechanisms_observed": "|".join(sorted(mechanisms)),
                "required_mechanisms_present_all_replicates": bool(group["required_mechanisms_present"].astype(bool).all()),
                "truth_rows_match_all_replicates": bool(group["truth_rows_match_candidates"].astype(bool).all()),
                "truth_summary_positive_all_replicates": bool(group["truth_summary_has_positives"].astype(bool).all()),
                "high_expression_controls_negative_all_replicates": bool(group["high_expression_controls_negative"].astype(bool).all()),
                "real_properties_present_all_replicates": bool(group["real_properties_present"].astype(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def _claim_summary(table: pd.DataFrame, dataset_summary: pd.DataFrame) -> pd.DataFrame:
    controlled = dataset_summary[dataset_summary["benchmark"] == "controlled_simulator_v2"]
    real = dataset_summary[dataset_summary["benchmark"] == "real_background_semisynthetic_v2"]
    real_datasets = set(real["dataset_key"].astype(str))
    rows = [
        {
            "claim_id": "G34_CONTRACT_C1_SHARED_SCHEMA",
            "status": "supported" if (table["status"] == "complete").all() else "partial",
            "statement": "Controlled simulator v2 and real-background v2 outputs satisfy the shared Phase 2 truth-table contract.",
            "evidence": "goal34_truth_contract_replicate_summary.csv",
        },
        {
            "claim_id": "G34_CONTRACT_C2_CONTROLLED_MECHANISMS",
            "status": "supported"
            if not controlled.empty and controlled["required_mechanisms_present_all_replicates"].astype(bool).all()
            else "partial",
            "statement": "Controlled simulator v2 covers contact, short/medium paracrine, longer-range niche, and high-expression negative-control mechanisms.",
            "evidence": "goal34_truth_contract_dataset_summary.csv",
        },
        {
            "claim_id": "G34_CONTRACT_C3_REAL_BACKGROUND_TRIPLET",
            "status": "supported"
            if {"lymph_node", "tonsil", "melanoma"}.issubset(real_datasets)
            and real["all_replicates_complete"].astype(bool).all()
            else "partial",
            "statement": "Real-background v2 has validated lymph node, tonsil, and melanoma replicate sets with preserved measured ST properties.",
            "evidence": "goal34_truth_contract_dataset_summary.csv",
        },
        {
            "claim_id": "G34_CONTRACT_C4_FALSE_POSITIVE_CONTROLS",
            "status": "supported"
            if dataset_summary["high_expression_controls_negative_all_replicates"].astype(bool).all()
            and int(dataset_summary["false_positive_true_edges"].sum()) == 0
            else "partial",
            "statement": "High-expression false-positive controls are present and remain ground-truth negative across validated benchmark datasets.",
            "evidence": "goal34_truth_contract_dataset_summary.csv",
        },
    ]
    return pd.DataFrame(rows)


def _real_cell_fields_non_degenerate(cells: pd.DataFrame) -> bool:
    required = {"x", "y", "cell_type", "region", "library_size", "observed_sparsity"}
    if not required.issubset(cells.columns):
        return False
    library_size = pd.to_numeric(cells["library_size"], errors="coerce")
    sparsity = pd.to_numeric(cells["observed_sparsity"], errors="coerce")
    return bool(
        cells["x"].nunique() > 1
        and cells["y"].nunique() > 1
        and cells["cell_type"].nunique() >= 2
        and cells["region"].nunique() >= 2
        and not library_size.isna().any()
        and float(library_size.median()) > 0
        and not sparsity.isna().any()
        and 0.0 < float(sparsity.mean()) < 1.0
    )


def _dataset_key_from_real_replicate(name: str) -> str:
    if "lymph_node" in name:
        return "lymph_node"
    if "tonsil" in name:
        return "tonsil"
    if "melanoma" in name:
        return "melanoma"
    return name.rsplit("_replicate_", maxsplit=1)[0]


def _sum_int(table: pd.DataFrame, column: str) -> int:
    if column not in table.columns:
        return 0
    return int(pd.to_numeric(table[column], errors="coerce").fillna(0).astype(int).sum())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
