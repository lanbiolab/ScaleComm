#!/usr/bin/env python
"""Validate semi-synthetic real-background CCC benchmark outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_ground_truth_predictions


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


def validate_replicate(path: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise AssertionError(f"{path}: missing files {missing}")

    cells = pd.read_csv(path / "cells.csv")
    expression = pd.read_csv(path / "expression.csv")
    lr_pairs = pd.read_csv(path / "lr_pairs.csv")
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    truth = pd.read_csv(path / "ground_truth_table.csv")
    truth_lr = pd.read_csv(path / "truth_lr_pairs.csv")
    truth_ct = pd.read_csv(path / "truth_cell_type_pairs.csv")
    truth_niche = pd.read_csv(path / "truth_spatial_niches.csv")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    expression_genes = len([col for col in expression.columns if col != "cell_id"])

    if len(expression) != len(cells):
        raise AssertionError(f"{path}: expression row count does not match cells")
    for col in ("x", "y", "cell_type", "region", "library_size", "observed_sparsity"):
        if col not in cells.columns:
            raise AssertionError(f"{path}: cells.csv missing {col}")
    if cells["region"].nunique() < 2:
        raise AssertionError(f"{path}: less than two real regions represented")
    if cells["cell_type"].nunique() < 2:
        raise AssertionError(f"{path}: less than two broad cell types represented")

    positives = int(candidates["ground_truth"].sum())
    weak_positives = int(candidates.loc[candidates["weak_signal"], "ground_truth"].sum())
    false_positive_labels = int(
        candidates.loc[candidates["false_positive_high_expression"], "ground_truth"].sum()
    )
    if positives <= 0:
        raise AssertionError(f"{path}: no positive interactions")
    if weak_positives <= 0:
        raise AssertionError(f"{path}: no weak positive interactions")
    if false_positive_labels != 0:
        raise AssertionError(f"{path}: false-positive LR pairs have positive labels")

    required_truth_columns = {
        "candidate_id",
        "true_edge",
        "true_lr_pair",
        "true_cell_type_pair",
        "true_spatial_niche",
        "weak_or_strong",
        "mechanism_type",
    }
    missing_truth_columns = required_truth_columns - set(truth.columns)
    if missing_truth_columns:
        raise AssertionError(f"{path}: ground_truth_table missing {missing_truth_columns}")
    if len(truth) != len(candidates):
        raise AssertionError(f"{path}: ground_truth_table row count does not match candidates")
    if int(truth["true_edge"].sum()) != positives:
        raise AssertionError(f"{path}: ground_truth_table true_edge count mismatch")
    if int(truth_lr["true_lr_pair"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_lr_pairs has no positive LR-pair truths")
    if int(truth_ct["true_cell_type_pair"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_cell_type_pairs has no positive cell-type-pair truths")
    if int(truth_niche["true_spatial_niche"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_spatial_niches has no positive niche truths")

    by_lr = candidates.groupby("lr_id")["ground_truth"].sum()
    true_lr_ids = lr_pairs.loc[~lr_pairs["false_positive_high_expression"], "lr_id"]
    missing_positive_lr = [lr_id for lr_id in true_lr_ids if int(by_lr.get(lr_id, 0)) == 0]
    if missing_positive_lr:
        raise AssertionError(f"{path}: true LR pair(s) with no positives: {missing_positive_lr}")

    expected_count = int(metadata["counts"]["candidate_interactions"])
    if expected_count != len(candidates):
        raise AssertionError(f"{path}: metadata candidate count mismatch")
    if int(metadata.get("expression_genes", expression_genes)) != expression_genes:
        raise AssertionError(f"{path}: metadata expression gene count mismatch")
    coord_span_x = float(cells["x"].max() - cells["x"].min())
    coord_span_y = float(cells["y"].max() - cells["y"].min())
    if coord_span_x <= 0 or coord_span_y <= 0:
        raise AssertionError(f"{path}: spatial coordinates have zero span")

    library_size = pd.to_numeric(cells["library_size"], errors="coerce")
    if library_size.isna().any() or float(library_size.median()) <= 0:
        raise AssertionError(f"{path}: invalid library size values")

    local_density = _local_density(cells[["x", "y"]].to_numpy(dtype=float), k=6)

    return {
        "replicate": path.name,
        "cells": len(cells),
        "expression_genes": expression_genes,
        "mean_observed_sparsity": float(cells["observed_sparsity"].mean()),
        "median_library_size": float(library_size.median()),
        "library_size_iqr": float(library_size.quantile(0.75) - library_size.quantile(0.25)),
        "coordinate_span_x": coord_span_x,
        "coordinate_span_y": coord_span_y,
        "median_nn_distance": float(local_density["median_nn_distance"]),
        "median_local_density": float(local_density["median_local_density"]),
        "dominant_region_fraction": float(cells["region"].value_counts(normalize=True).max()),
        "dominant_cell_type_fraction": float(cells["cell_type"].value_counts(normalize=True).max()),
        "regions": int(cells["region"].nunique()),
        "cell_types": int(cells["cell_type"].nunique()),
        "lr_pairs": len(lr_pairs),
        "candidate_interactions": len(candidates),
        "positive_interactions": positives,
        "weak_positive_interactions": weak_positives,
        "false_positive_labels": false_positive_labels,
        "truth_lr_positive_pairs": int(truth_lr["true_lr_pair"].sum()),
        "truth_cell_type_pairs": int(truth_ct["true_cell_type_pair"].sum()),
        "truth_spatial_niches": int(truth_niche["true_spatial_niche"].sum()),
        "heuristic_coexpr_distance_auroc": _heuristic_metric(candidates, "auroc"),
        "heuristic_coexpr_distance_aupr": _heuristic_metric(candidates, "aupr"),
    }


def summarize_injections(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    rows = []
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        first = group.iloc[0]
        positives = group[group["ground_truth"].astype(int) == 1]
        negatives = group[group["ground_truth"].astype(int) == 0]
        rows.append(
            {
                "replicate": path.name,
                "lr_id": lr_id,
                "ligand": first["ligand"],
                "receptor": first["receptor"],
                "mechanism": first["mechanism"],
                "weak_signal": bool(first["weak_signal"]),
                "false_positive_high_expression": bool(first["false_positive_high_expression"]),
                "candidates": int(len(group)),
                "positives": int(len(positives)),
                "positive_rate": float(len(positives) / len(group)) if len(group) else float("nan"),
                "positive_median_distance": float(positives["distance"].median())
                if len(positives)
                else float("nan"),
                "negative_median_distance": float(negatives["distance"].median())
                if len(negatives)
                else float("nan"),
                "positive_median_ligand_expr": float(positives["ligand_expr"].median())
                if len(positives)
                else float("nan"),
                "negative_median_ligand_expr": float(negatives["ligand_expr"].median())
                if len(negatives)
                else float("nan"),
                "positive_median_receptor_expr": float(positives["receptor_expr"].median())
                if len(positives)
                else float("nan"),
                "negative_median_receptor_expr": float(negatives["receptor_expr"].median())
                if len(negatives)
                else float("nan"),
                "type_rule_positive_fraction": _positive_rule_fraction(group, "type_rule_satisfied"),
                "range_rule_positive_fraction": _positive_rule_fraction(group, "range_rule_satisfied"),
                "niche_rule_positive_fraction": _positive_rule_fraction(group, "niche_rule_satisfied"),
                "heuristic_coexpr_distance_auroc": _heuristic_metric(group, "auroc"),
                "heuristic_coexpr_distance_aupr": _heuristic_metric(group, "aupr"),
            }
        )
    return pd.DataFrame(rows)


def summarize_mechanisms(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    rows = []
    for mechanism, group in candidates.groupby("mechanism", sort=False):
        positives = group[group["ground_truth"].astype(int) == 1]
        rows.append(
            {
                "replicate": path.name,
                "mechanism": mechanism,
                "candidates": int(len(group)),
                "positives": int(len(positives)),
                "weak_positives": int(
                    ((group["weak_signal"].astype(bool)) & (group["ground_truth"].astype(int) == 1)).sum()
                ),
                "false_positive_candidates": int(group["false_positive_high_expression"].astype(bool).sum()),
                "false_positive_labels": int(
                    group.loc[group["false_positive_high_expression"].astype(bool), "ground_truth"].sum()
                ),
                "positive_rate": float(len(positives) / len(group)) if len(group) else float("nan"),
                "positive_median_distance": float(positives["distance"].median())
                if len(positives)
                else float("nan"),
                "type_rule_positive_fraction": _positive_rule_fraction(group, "type_rule_satisfied"),
                "range_rule_positive_fraction": _positive_rule_fraction(group, "range_rule_satisfied"),
                "niche_rule_positive_fraction": _positive_rule_fraction(group, "niche_rule_satisfied"),
                "heuristic_coexpr_distance_auroc": _heuristic_metric(group, "auroc"),
                "heuristic_coexpr_distance_aupr": _heuristic_metric(group, "aupr"),
            }
        )
    return pd.DataFrame(rows)


def _local_density(coords: np.ndarray, *, k: int) -> dict[str, float]:
    if len(coords) < 2:
        return {"median_nn_distance": float("nan"), "median_local_density": float("nan")}
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    kth = min(int(k), len(coords) - 1)
    sorted_distances = np.sort(distances, axis=1)
    nn = sorted_distances[:, 0]
    radius = sorted_distances[:, kth - 1]
    density = kth / (np.pi * np.maximum(radius, 1e-12) ** 2)
    return {
        "median_nn_distance": float(np.median(nn)),
        "median_local_density": float(np.median(density)),
    }


def _heuristic_metric(candidates: pd.DataFrame, metric: str) -> float:
    if candidates.empty or candidates["ground_truth"].nunique() < 2:
        return float("nan")
    predictions = candidates[["candidate_id", "coexpression_distance_score"]].rename(
        columns={"coexpression_distance_score": "score"}
    )
    predictions["method"] = "coexpression_distance_heuristic"
    return float(evaluate_ground_truth_predictions(candidates, predictions)[metric])


def _positive_rule_fraction(candidates: pd.DataFrame, column: str) -> float:
    positives = candidates[candidates["ground_truth"].astype(int) == 1]
    if positives.empty or column not in positives.columns:
        return float("nan")
    return float(positives[column].astype(bool).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    replicate_dirs = sorted(path for path in root.glob("*_replicate_*") if path.is_dir())
    if not replicate_dirs:
        raise SystemExit(f"No *_replicate_* directories found under {root}")
    summaries = [validate_replicate(path) for path in replicate_dirs]
    for summary in summaries:
        print(summary)
    pd.DataFrame(summaries).to_csv(root / "validation_summary.csv", index=False)
    injection_summary = pd.concat([summarize_injections(path) for path in replicate_dirs], ignore_index=True)
    mechanism_summary = pd.concat([summarize_mechanisms(path) for path in replicate_dirs], ignore_index=True)
    injection_summary.to_csv(root / "semisynthetic_injection_summary.csv", index=False)
    mechanism_summary.to_csv(root / "semisynthetic_mechanism_summary.csv", index=False)
    print(f"Validated {len(summaries)} semi-synthetic replicate(s) under {root}")


if __name__ == "__main__":
    main()
