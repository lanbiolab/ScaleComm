#!/usr/bin/env python
"""Run a focused direct group-level permutation grid for Phase 2 Goal 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import DEFAULT_NULL_MODELS, PermutationFDRConfig, permutation_group_fdr


DEFAULT_NULLS = ("score_permutation", "lr_identity_permutation", "receiver_permutation")
DEFAULT_GROUP_LEVELS = ("lr_pair", "cell_type_pair", "pathway_module")


def run_direct_group_grid(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    null_models: tuple[str, ...],
    group_levels: tuple[str, ...],
    top_fractions: tuple[float, ...],
    n_permutations: int,
    random_seed: int,
    score_column: str,
    alpha: float,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PermutationFDRConfig(
        n_permutations=n_permutations,
        random_seed=random_seed,
        score_column=score_column,
    )

    rows: list[dict[str, object]] = []
    for null_model in null_models:
        for group_level in group_levels:
            group_columns = _group_columns(group_level)
            missing = set(group_columns) - set(candidates.columns)
            if missing:
                rows.append(
                    {
                        "status": "skipped_missing_columns",
                        "null_model": null_model,
                        "group_level": group_level,
                        "top_fraction": float("nan"),
                        "missing_columns": ",".join(sorted(missing)),
                    }
                )
                continue
            for top_fraction in top_fractions:
                table = permutation_group_fdr(
                    candidates,
                    predictions,
                    null_model=null_model,
                    group_columns=group_columns,
                    config=config,
                    top_fraction=top_fraction,
                )
                table["group_significant"] = table["group_bh_fdr"] <= alpha
                output_path = (
                    output_dir
                    / f"{prediction_path.stem}__{null_model}__{group_level}__top{_fraction_label(top_fraction)}__direct_fdr.csv"
                )
                table.to_csv(output_path, index=False)
                rows.append(
                    _summarize_direct_group_table(
                        candidates,
                        table,
                        group_columns=group_columns,
                        null_model=null_model,
                        group_level=group_level,
                        top_fraction=top_fraction,
                        output_path=output_path,
                        alpha=alpha,
                    )
                    | {
                        "status": "completed",
                        "n_permutations": int(n_permutations),
                        "random_seed": int(random_seed),
                        "score_column": score_column,
                    }
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "direct_group_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "candidate_path": str(candidate_path),
        "prediction_path": str(prediction_path),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "null_models": list(null_models),
        "group_levels": list(group_levels),
        "top_fractions": list(top_fractions),
        "n_permutations": int(n_permutations),
        "random_seed": int(random_seed),
        "score_column": score_column,
        "alpha": float(alpha),
    }
    (output_dir / "direct_group_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_markdown_summary(summary, output_dir / "direct_group_grid_summary.md")
    return summary


def _summarize_direct_group_table(
    candidates: pd.DataFrame,
    table: pd.DataFrame,
    *,
    group_columns: list[str],
    null_model: str,
    group_level: str,
    top_fraction: float,
    output_path: Path,
    alpha: float,
) -> dict[str, object]:
    flags = _group_control_flags(candidates, group_columns)
    merged = table.merge(flags, on=group_columns, how="left", validate="one_to_one")
    significant = merged["group_bh_fdr"].astype(float) <= alpha
    positive = merged["group_has_positive"].fillna(False).astype(bool)
    negative = merged["group_has_high_expression_negative"].fillna(False).astype(bool) & ~positive
    positive_fdr = pd.to_numeric(merged.loc[positive, "group_bh_fdr"], errors="coerce")
    negative_fdr = pd.to_numeric(merged.loc[negative, "group_bh_fdr"], errors="coerce")
    top = merged.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False]).iloc[0]
    row = {
        "null_model": null_model,
        "group_level": group_level,
        "top_fraction": float(top_fraction),
        "alpha": float(alpha),
        "n_groups": int(len(merged)),
        "n_significant_groups": int(significant.sum()),
        "min_group_p_value": float(merged["group_p_value"].min()),
        "min_group_bh_fdr": float(merged["group_bh_fdr"].min()),
        "positive_control_groups": int(positive.sum()),
        "positive_control_significant_groups": int((positive & significant).sum()),
        "positive_control_min_group_bh_fdr": float(positive_fdr.min()) if positive_fdr.notna().any() else float("nan"),
        "negative_control_groups": int(negative.sum()),
        "negative_control_significant_groups": int((negative & significant).sum()),
        "negative_control_min_group_bh_fdr": float(negative_fdr.min()) if negative_fdr.notna().any() else float("nan"),
        "top_group": ";".join(f"{column}={top.get(column, '')}" for column in group_columns),
        "output_path": str(output_path),
    }
    row["phase2_goal2_pass"] = bool(
        row["positive_control_significant_groups"] > 0
        and row["negative_control_significant_groups"] == 0
    )
    return row


def _group_control_flags(candidates: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    required = {"ground_truth", "false_positive_high_expression"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing control columns: {sorted(missing)}")
    flags = candidates[[*group_columns, "ground_truth", "false_positive_high_expression"]].copy()
    flags["ground_truth"] = flags["ground_truth"].astype(int)
    flags["false_positive_high_expression"] = flags["false_positive_high_expression"].astype(bool)
    return (
        flags.groupby(group_columns, dropna=False, sort=False)
        .agg(
            group_has_positive=("ground_truth", lambda values: bool((values.astype(int) == 1).any())),
            group_has_high_expression_negative=(
                "false_positive_high_expression",
                lambda values: bool(values.astype(bool).any()),
            ),
        )
        .reset_index()
    )


def _group_columns(group_level: str) -> list[str]:
    if group_level == "lr_pair":
        return ["lr_id", "ligand", "receptor", "mechanism"]
    if group_level == "cell_type_pair":
        return ["sender_cell_type", "receiver_cell_type", "lr_id", "ligand", "receptor", "mechanism"]
    if group_level == "pathway_module":
        return ["mechanism"]
    raise ValueError(f"Unsupported group level: {group_level}")


def _fraction_label(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        output_path.write_text("# Direct Group FDR Grid\n\nNo rows.\n", encoding="utf-8")
        return
    completed = summary[summary["status"] == "completed"].copy()
    lines = ["# Direct Group FDR Grid", ""]
    if completed.empty:
        lines.append("No completed rows.")
    else:
        best = completed.sort_values(
            ["positive_control_min_group_bh_fdr", "negative_control_significant_groups", "min_group_bh_fdr"],
            ascending=[True, True, True],
        ).head(12)
        fields = [
            "null_model",
            "group_level",
            "top_fraction",
            "positive_control_min_group_bh_fdr",
            "positive_control_significant_groups",
            "negative_control_significant_groups",
            "phase2_goal2_pass",
        ]
        lines.extend(_markdown_table(best[fields]))
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> list[str]:
    fields = list(table.columns)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[field]) for field in fields) + " |")
    return lines


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
        return "NA"
    return str(value)


def _parse_tuple(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(values) if values else default


def _parse_float_tuple(values: list[float] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(values) if values else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--null-model", action="append", choices=DEFAULT_NULL_MODELS)
    parser.add_argument("--group-level", action="append", choices=DEFAULT_GROUP_LEVELS)
    parser.add_argument("--top-fraction", action="append", type=float)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    summary = run_direct_group_grid(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.output_dir),
        null_models=_parse_tuple(args.null_model, DEFAULT_NULLS),
        group_levels=_parse_tuple(args.group_level, DEFAULT_GROUP_LEVELS),
        top_fractions=_parse_float_tuple(args.top_fraction, (0.01, 0.03, 0.05, 0.10)),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        score_column=args.score_column,
        alpha=args.alpha,
    )
    print(summary.to_string(index=False))
    print(f"Wrote direct group FDR grid to {args.output_dir}")


if __name__ == "__main__":
    main()
