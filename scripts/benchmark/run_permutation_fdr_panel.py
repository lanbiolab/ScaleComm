#!/usr/bin/env python
"""Run permutation/FDR analysis across methods and replicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_permutation_fdr import run_permutation_fdr
from src.spitialself.fdr import DEFAULT_NULL_MODELS


def run_permutation_fdr_panel(
    dataset_root: Path,
    method_prediction_roots: dict[str, Path],
    output_dir: Path,
    *,
    null_models: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    score_column: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run FDR for every method/replicate and write combined summaries."""

    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for method, prediction_root in method_prediction_roots.items():
        for replicate_dir in replicate_dirs:
            replicate = replicate_dir.name
            candidate_path = replicate_dir / "candidate_interactions.csv"
            prediction_path = _find_prediction_path(prediction_root, replicate, method)
            run_dir = output_dir / method / replicate
            row: dict[str, object] = {
                "method": method,
                "replicate": replicate,
                "candidate_path": str(candidate_path),
                "prediction_path": str(prediction_path),
                "output_dir": str(run_dir),
                "n_permutations": int(n_permutations),
            }
            try:
                if not candidate_path.exists():
                    raise FileNotFoundError(candidate_path)
                if prediction_path is None or not prediction_path.exists():
                    raise FileNotFoundError(prediction_path)
                summary = run_permutation_fdr(
                    candidate_path,
                    prediction_path,
                    run_dir,
                    null_models=null_models,
                    n_permutations=n_permutations,
                    random_seed=random_seed,
                    score_column=score_column,
                    alpha=alpha,
                )
                summary.insert(0, "replicate", replicate)
                summary.insert(0, "method_panel", method)
                combined_rows.append(summary)
                row.update({"status": "completed", "detail": ""})
            except Exception as exc:  # noqa: BLE001 - status table should capture all method failures.
                row.update({"status": "failed", "detail": repr(exc)})
            status_rows.append(row)

    combined = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    method_summary = _summarize_panel(combined)

    combined.to_csv(output_dir / "permutation_fdr_panel_summary.csv", index=False)
    status.to_csv(output_dir / "permutation_fdr_panel_status.csv", index=False)
    method_summary.to_csv(output_dir / "permutation_fdr_panel_method_summary.csv", index=False)
    with (output_dir / "permutation_fdr_panel_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "method_prediction_roots": {
                    method: str(path) for method, path in method_prediction_roots.items()
                },
                "output_dir": str(output_dir),
                "null_models": list(null_models),
                "n_permutations": int(n_permutations),
                "random_seed": int(random_seed),
                "score_column": score_column,
                "alpha": float(alpha),
                "n_replicates": len(replicate_dirs),
                "n_completed_runs": int((status["status"] == "completed").sum()),
                "n_failed_runs": int((status["status"] == "failed").sum()),
            },
            handle,
            indent=2,
        )
    return combined, status, method_summary


def _summarize_panel(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_columns = ["method_panel", "null_model"]
    metric_columns = [
        "n_candidates",
        "n_significant_edges",
        "min_p_value",
        "min_bh_fdr",
        "n_significant_lr_pairs",
        "realized_fdr",
        "realized_precision",
        "realized_recall",
        "realized_weak_signal_recall",
        "realized_false_positive_high_expression_call_rate",
        "p_value_ks_uniform_stat",
        "n_significant_group_lr_pairs",
        "n_significant_cell_type_pair_interactions",
    ]
    rows = []
    for key, group in summary.groupby(group_columns, dropna=False, sort=True):
        row = {column: value for column, value in zip(group_columns, key)}
        row["n_replicates"] = int(group["replicate"].nunique())
        row["n_rows"] = int(len(group))
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{column}_min"] = float(values.min()) if values.notna().any() else float("nan")
            row[f"{column}_max"] = float(values.max()) if values.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _find_prediction_path(prediction_root: Path, replicate: str, method: str) -> Path | None:
    exact = prediction_root / f"{replicate}__{method}_predictions.csv"
    if exact.exists():
        return exact

    candidates = sorted(prediction_root.glob(f"{replicate}__*predictions.csv"))
    if not candidates:
        return exact
    method_matches = [path for path in candidates if method in path.stem]
    if method_matches:
        return method_matches[0]
    return candidates[0]


def _parse_method_root(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=PATH, got: {value}")
        method, path = value.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Empty method name in: {value}")
        result[method] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--method-root",
        action="append",
        required=True,
        help="Prediction root as METHOD=PATH. Repeat for multiple methods.",
    )
    parser.add_argument("--output-dir", required=True)
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
    combined, status, method_summary = run_permutation_fdr_panel(
        Path(args.dataset_root),
        _parse_method_root(args.method_root),
        Path(args.output_dir),
        null_models=null_models,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        score_column=args.score_column,
        alpha=args.alpha,
    )
    print(f"Wrote panel FDR outputs to {args.output_dir}")
    print(f"Completed runs: {(status['status'] == 'completed').sum()} / {len(status)}")
    if not combined.empty:
        print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
