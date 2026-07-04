#!/usr/bin/env python
"""Diagnose pseudo-label composition for SpatialSelf benchmark candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import _pseudo_label_indices, _pseudo_label_score


def diagnose_pseudo_labels(
    candidate_path: Path,
    output_dir: Path,
    *,
    strategies: tuple[str, ...],
    positive_quantile: float,
    negative_quantile: float,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lr_rows = []
    for strategy in strategies:
        scores = _pseudo_label_score(candidates, strategy=strategy)
        pos_idx, neg_idx = _pseudo_label_indices(
            candidates,
            scores,
            strategy=strategy,
            positive_quantile=positive_quantile,
            negative_quantile=negative_quantile,
        )
        pos = candidates.iloc[pos_idx]
        neg = candidates.iloc[neg_idx]
        rows.append(_summarize_subset(candidate_path.parent.name, strategy, "pseudo_positive", pos))
        rows.append(_summarize_subset(candidate_path.parent.name, strategy, "pseudo_negative", neg))
        for lr_id, group in pos.groupby("lr_id", sort=False):
            lr_rows.append(
                {
                    "replicate": candidate_path.parent.name,
                    "strategy": strategy,
                    "lr_id": lr_id,
                    "mechanism": group["mechanism"].iloc[0],
                    "weak_signal": bool(group["weak_signal"].astype(bool).any()),
                    "false_positive_high_expression": bool(
                        group["false_positive_high_expression"].astype(bool).any()
                    ),
                    "pseudo_positive_count": int(len(group)),
                    "truth_count": int(group["ground_truth"].astype(int).sum()),
                    "weak_truth_count": int(
                        ((group["ground_truth"].astype(int) == 1) & group["weak_signal"].astype(bool)).sum()
                    ),
                    "truth_rate": float(group["ground_truth"].astype(int).mean()),
                }
            )

    summary = pd.DataFrame(rows)
    lr_summary = pd.DataFrame(lr_rows)
    summary.to_csv(output_dir / f"{candidate_path.parent.name}__pseudo_label_summary.csv", index=False)
    lr_summary.to_csv(output_dir / f"{candidate_path.parent.name}__pseudo_label_lr_summary.csv", index=False)
    return summary


def _summarize_subset(replicate: str, strategy: str, subset_name: str, subset: pd.DataFrame) -> dict:
    truth = subset["ground_truth"].astype(int)
    weak_truth = (truth == 1) & subset["weak_signal"].astype(bool)
    false_positive_high = subset["false_positive_high_expression"].astype(bool)
    return {
        "replicate": replicate,
        "strategy": strategy,
        "subset": subset_name,
        "n": int(len(subset)),
        "truth_count": int(truth.sum()),
        "truth_rate": float(truth.mean()) if len(subset) else float("nan"),
        "weak_truth_count": int(weak_truth.sum()),
        "weak_truth_rate": float(weak_truth.mean()) if len(subset) else float("nan"),
        "false_positive_high_expression_count": int(false_positive_high.sum()),
        "false_positive_high_expression_rate": float(false_positive_high.mean())
        if len(subset)
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("--output-dir", default="outputs/phase1/exp04_model_ablation/pseudo_label_diagnostics")
    parser.add_argument("--strategy", action="append", dest="strategies")
    parser.add_argument("--positive-quantile", type=float, default=0.9)
    parser.add_argument("--negative-quantile", type=float, default=0.2)
    args = parser.parse_args()

    strategies = tuple(
        args.strategies
        or (
            "expression_distance",
            "lr_balanced_expression_distance",
            "lr_capped_expression_distance",
            "expression_plus_distance",
            "lr_balanced_expression_plus_distance",
            "lr_capped_expression_plus_distance",
        )
    )
    summary = diagnose_pseudo_labels(
        Path(args.candidates),
        Path(args.output_dir),
        strategies=strategies,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
    )
    print(summary.to_string(index=False))
    print(f"Wrote pseudo-label diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
