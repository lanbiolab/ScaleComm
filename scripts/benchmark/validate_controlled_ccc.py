#!/usr/bin/env python
"""Validate the controlled CCC benchmark schema and core label invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

REQUIRED_MECHANISMS = {
    "juxtacrine_contact",
    "short_medium_paracrine",
    "long_range_chemokine_niche",
    "false_positive_high_expression",
}


def validate_replicate(path: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise AssertionError(f"{path}: missing required files: {', '.join(missing)}")

    cells = pd.read_csv(path / "cells.csv")
    expression = pd.read_csv(path / "expression.csv")
    lr_pairs = pd.read_csv(path / "lr_pairs.csv")
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    truth = pd.read_csv(path / "ground_truth_table.csv")
    truth_lr = pd.read_csv(path / "truth_lr_pairs.csv")
    truth_ct = pd.read_csv(path / "truth_cell_type_pairs.csv")
    truth_niche = pd.read_csv(path / "truth_spatial_niches.csv")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    if len(cells) == 0 or len(lr_pairs) == 0 or len(candidates) == 0:
        raise AssertionError(f"{path}: empty cells/lr_pairs/candidate_interactions")
    if len(expression) != len(cells):
        raise AssertionError(f"{path}: expression row count does not match cells")

    mechanisms = set(lr_pairs["mechanism"])
    if not REQUIRED_MECHANISMS.issubset(mechanisms):
        raise AssertionError(f"{path}: missing mechanisms {REQUIRED_MECHANISMS - mechanisms}")

    positives = int(candidates["ground_truth"].sum())
    weak_positives = int(candidates.loc[candidates["weak_signal"], "ground_truth"].sum())
    false_positive_labels = int(
        candidates.loc[candidates["false_positive_high_expression"], "ground_truth"].sum()
    )
    if positives <= 0:
        raise AssertionError(f"{path}: no positive ground-truth interactions")
    if weak_positives <= 0:
        raise AssertionError(f"{path}: no weak-signal positives")
    if false_positive_labels != 0:
        raise AssertionError(f"{path}: false-positive high-expression pairs have positive labels")

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
    if set(truth["weak_or_strong"]) < {"weak", "strong", "false_positive_high_expression"}:
        raise AssertionError(f"{path}: ground_truth_table lacks weak/strong/false-positive labels")
    if int(truth_lr["true_lr_pair"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_lr_pairs has no positive LR-pair truths")
    if int(truth_ct["true_cell_type_pair"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_cell_type_pairs has no positive cell-type-pair truths")
    if int(truth_niche["true_spatial_niche"].sum()) <= 0:
        raise AssertionError(f"{path}: truth_spatial_niches has no positive niche truths")

    mechanism_positive_counts = candidates.groupby("mechanism")["ground_truth"].sum().to_dict()
    for mechanism in (
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
    ):
        if int(mechanism_positive_counts.get(mechanism, 0)) <= 0:
            raise AssertionError(f"{path}: no positives for mechanism {mechanism}")

    expected_count = metadata["counts"]["candidate_interactions"]
    if int(expected_count) != len(candidates):
        raise AssertionError(f"{path}: metadata candidate count mismatch")

    return {
        "replicate": path.name,
        "cells": len(cells),
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
        "positive_median_distance": _positive_median_distance(candidates),
        "long_range_positive_median_distance": _positive_median_distance(
            candidates[candidates["mechanism"] == "long_range_chemokine_niche"]
        ),
        "contact_positive_max_distance": _positive_max_distance(
            candidates[candidates["mechanism"] == "juxtacrine_contact"]
        ),
    }


def summarize_mechanisms(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    rows = []
    for mechanism, group in candidates.groupby("mechanism", sort=False):
        positives = group[group["ground_truth"].astype(int) == 1]
        negatives = group[group["ground_truth"].astype(int) == 0]
        rows.append(
            {
                "replicate": path.name,
                "mechanism": mechanism,
                "candidates": int(len(group)),
                "positives": int(len(positives)),
                "weak_positives": int(
                    ((group["weak_signal"].astype(bool)) & (group["ground_truth"].astype(int) == 1)).sum()
                ),
                "false_positive_labels": int(
                    group.loc[group["false_positive_high_expression"].astype(bool), "ground_truth"].sum()
                ),
                "positive_rate": float(positives.shape[0] / len(group)) if len(group) else float("nan"),
                "positive_median_distance": _positive_median_distance(group),
                "positive_max_distance": _positive_max_distance(group),
                "negative_median_distance": float(negatives["distance"].median()) if len(negatives) else float("nan"),
                "positive_median_coexpr_distance_score": float(
                    positives["coexpression_distance_score"].median()
                )
                if len(positives)
                else float("nan"),
                "negative_median_coexpr_distance_score": float(
                    negatives["coexpression_distance_score"].median()
                )
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


def summarize_lr_pairs(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path / "candidate_interactions.csv")
    rows = []
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        first = group.iloc[0]
        positives = group[group["ground_truth"].astype(int) == 1]
        rows.append(
            {
                "replicate": path.name,
                "lr_id": lr_id,
                "mechanism": first["mechanism"],
                "weak_signal": bool(first["weak_signal"]),
                "false_positive_high_expression": bool(first["false_positive_high_expression"]),
                "candidates": int(len(group)),
                "positives": int(len(positives)),
                "positive_median_distance": _positive_median_distance(group),
                "positive_median_coexpr_distance_score": float(
                    positives["coexpression_distance_score"].median()
                )
                if len(positives)
                else float("nan"),
                "heuristic_coexpr_distance_auroc": _heuristic_metric(group, "auroc"),
                "heuristic_coexpr_distance_aupr": _heuristic_metric(group, "aupr"),
            }
        )
    return pd.DataFrame(rows)


def _heuristic_metric(candidates: pd.DataFrame, metric: str) -> float:
    if candidates.empty or candidates["ground_truth"].nunique() < 2:
        return float("nan")
    predictions = candidates[["candidate_id", "coexpression_distance_score"]].rename(
        columns={"coexpression_distance_score": "score"}
    )
    predictions["method"] = "coexpression_distance_heuristic"
    return float(evaluate_ground_truth_predictions(candidates, predictions)[metric])


def _positive_median_distance(candidates: pd.DataFrame) -> float:
    positives = candidates[candidates["ground_truth"].astype(int) == 1]
    return float(positives["distance"].median()) if len(positives) else float("nan")


def _positive_max_distance(candidates: pd.DataFrame) -> float:
    positives = candidates[candidates["ground_truth"].astype(int) == 1]
    return float(positives["distance"].max()) if len(positives) else float("nan")


def _positive_rule_fraction(candidates: pd.DataFrame, column: str) -> float:
    positives = candidates[candidates["ground_truth"].astype(int) == 1]
    if positives.empty or column not in positives.columns:
        return float("nan")
    return float(positives[column].astype(bool).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate_* directories.")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    replicate_dirs = sorted(path for path in root.glob("replicate_*") if path.is_dir())
    if not replicate_dirs:
        raise SystemExit(f"No replicate_* directories found under {root}")

    summaries = [validate_replicate(path) for path in replicate_dirs]
    for summary in summaries:
        print(summary)
    mechanism_summary = pd.concat([summarize_mechanisms(path) for path in replicate_dirs], ignore_index=True)
    lr_summary = pd.concat([summarize_lr_pairs(path) for path in replicate_dirs], ignore_index=True)
    mechanism_summary.to_csv(root / "controlled_mechanism_summary.csv", index=False)
    lr_summary.to_csv(root / "controlled_lr_summary.csv", index=False)
    print(f"Validated {len(summaries)} controlled CCC replicate(s) under {root}")


if __name__ == "__main__":
    main()
