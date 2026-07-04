#!/usr/bin/env python
"""Run method-agnostic permutation p-value and BH-FDR analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import DEFAULT_NULL_MODELS, PermutationFDRConfig
from src.spitialself.fdr import permutation_fdr, summarize_fdr, summarize_fdr_with_truth, summarize_group_fdr


def run_permutation_fdr(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    null_models: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    score_column: str,
    alpha: float,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    config = PermutationFDRConfig(
        n_permutations=n_permutations,
        random_seed=random_seed,
        score_column=score_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for null_model in null_models:
        table = permutation_fdr(candidates, predictions, null_model=null_model, config=config)
        output_path = output_dir / f"{prediction_path.stem}__{null_model}__fdr.csv"
        table.to_csv(output_path, index=False)
        lr_summary_path = output_dir / f"{prediction_path.stem}__{null_model}__lr_pair_fdr.csv"
        lr_summary = summarize_group_fdr(
            candidates,
            table,
            group_columns=["lr_id", "ligand", "receptor", "mechanism"],
            alpha=alpha,
        )
        lr_summary.to_csv(lr_summary_path, index=False)

        cell_type_columns = [
            "sender_cell_type",
            "receiver_cell_type",
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
        ]
        cell_type_summary_path = (
            output_dir / f"{prediction_path.stem}__{null_model}__cell_type_pair_fdr.csv"
        )
        cell_type_summary = summarize_group_fdr(
            candidates,
            table,
            group_columns=cell_type_columns,
            alpha=alpha,
        )
        cell_type_summary.to_csv(cell_type_summary_path, index=False)

        summary = summarize_fdr(table, alpha=alpha)
        truth_columns = {"ground_truth", "weak_signal", "false_positive_high_expression"}
        if truth_columns.issubset(candidates.columns):
            summary.update(summarize_fdr_with_truth(candidates, table, alpha=alpha))
        else:
            summary["truth_diagnostics_available"] = False
        summary.update(
            {
                "candidate_path": str(candidate_path),
                "prediction_path": str(prediction_path),
                "fdr_path": str(output_path),
                "lr_pair_fdr_path": str(lr_summary_path),
                "cell_type_pair_fdr_path": str(cell_type_summary_path),
                "n_significant_group_lr_pairs": int(lr_summary["group_significant"].sum()),
                "n_significant_cell_type_pair_interactions": int(
                    cell_type_summary["group_significant"].sum()
                ),
                "n_permutations": int(n_permutations),
            }
        )
        summaries.append(summary)

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output_dir / "permutation_fdr_summary.csv", index=False)
    with (output_dir / "permutation_fdr_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=True)
    return summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV with candidate_id and score.")
    parser.add_argument("--output-dir", required=True, help="Directory for FDR outputs.")
    parser.add_argument(
        "--null-model",
        action="append",
        dest="null_models",
        choices=DEFAULT_NULL_MODELS,
        help="Null model to run. Repeat to run multiple. Defaults to all configured nulls.",
    )
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    null_models = tuple(args.null_models) if args.null_models else DEFAULT_NULL_MODELS
    summary = run_permutation_fdr(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.output_dir),
        null_models=null_models,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        score_column=args.score_column,
        alpha=args.alpha,
    )
    print(summary.to_string(index=False))
    print(f"Wrote permutation/FDR outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
