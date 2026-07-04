#!/usr/bin/env python
"""Run Phase 2 Goal 2 FDR calibration diagnostics for one prediction table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import DEFAULT_NULL_MODELS, PermutationFDRConfig, bh_adjust
from src.spitialself.fdr import permutation_fdr, permutation_group_fdr, summarize_fdr
from src.spitialself.fdr import summarize_fdr_with_truth, summarize_group_fdr


PHASE2_NULL_MODELS = (
    "score_permutation",
    "spatial_permutation",
    "receiver_permutation",
    "lr_identity_permutation",
    "within_region_permutation",
    "cell_type_preserving_permutation",
)


def run_goal2_fdr_calibration(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    null_models: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    score_column: str,
    alpha: float,
    operating_quantile: float,
    direct_group_top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    config = PermutationFDRConfig(
        n_permutations=n_permutations,
        random_seed=random_seed,
        score_column=score_column,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    group_summaries: list[dict[str, object]] = []
    fdr_paths: dict[str, str] = {}

    for null_model in null_models:
        table = permutation_fdr(candidates, predictions, null_model=null_model, config=config)
        table = _add_empirical_null_diagnostics(table)
        fdr_path = output_dir / f"{prediction_path.stem}__{null_model}__candidate_edge_fdr.csv"
        table.to_csv(fdr_path, index=False)
        fdr_paths[null_model] = str(fdr_path)

        summary = summarize_fdr(table, alpha=alpha)
        if _has_truth(candidates):
            summary.update(summarize_fdr_with_truth(candidates, table, alpha=alpha))
        else:
            summary["truth_diagnostics_available"] = False
        summary.update(
            _operating_point_summary(
                candidates,
                predictions,
                score_column=score_column,
                operating_quantile=operating_quantile,
            )
        )
        summary.update(
            {
                "min_gaussian_empirical_p_value": float(table["gaussian_empirical_p_value"].min()),
                "min_gaussian_bh_fdr": float(table["gaussian_bh_fdr"].min()),
                "n_gaussian_bh_significant_edges": int((table["gaussian_bh_fdr"] <= alpha).sum()),
                "n_local_fdr_edges": int((table["local_fdr"] <= alpha).sum()),
                "median_null_z": float(table["null_z"].median()),
                "max_null_z": float(table["null_z"].max()),
                "candidate_path": str(candidate_path),
                "prediction_path": str(prediction_path),
                "fdr_path": str(fdr_path),
                "n_permutations": int(n_permutations),
                "random_seed": int(random_seed),
            }
        )
        summaries.append(summary)

        group_specs = _available_group_specs(candidates)
        for group_level, group_columns in group_specs:
            group_table = summarize_group_fdr(
                candidates,
                table,
                group_columns=group_columns,
                alpha=alpha,
            )
            group_path = output_dir / f"{prediction_path.stem}__{null_model}__{group_level}_simes_fdr.csv"
            group_table.to_csv(group_path, index=False)
            group_summaries.append(
                _summarize_group_table(
                    group_table,
                    candidates=candidates,
                    group_level=group_level,
                    group_columns=group_columns,
                    group_path=group_path,
                    alpha=alpha,
                )
                | {
                    "group_test": "simes_edge_p",
                    "null_model": null_model,
                    "candidate_path": str(candidate_path),
                    "prediction_path": str(prediction_path),
                }
            )
            direct_group_table = permutation_group_fdr(
                candidates,
                predictions,
                null_model=null_model,
                group_columns=group_columns,
                config=config,
                top_fraction=direct_group_top_fraction,
            )
            direct_group_table["group_significant"] = direct_group_table["group_bh_fdr"] <= alpha
            direct_group_path = output_dir / f"{prediction_path.stem}__{null_model}__{group_level}_direct_fdr.csv"
            direct_group_table.to_csv(direct_group_path, index=False)
            group_summaries.append(
                _summarize_group_table(
                    direct_group_table,
                    candidates=candidates,
                    group_level=group_level,
                    group_columns=group_columns,
                    group_path=direct_group_path,
                    alpha=alpha,
                )
                | {
                    "group_test": "direct_top_tail_permutation",
                    "null_model": null_model,
                    "candidate_path": str(candidate_path),
                    "prediction_path": str(prediction_path),
                }
            )

    summary_table = pd.DataFrame(summaries)
    group_summary_table = pd.DataFrame(group_summaries)
    summary_table.to_csv(output_dir / "fdr_calibration_summary.csv", index=False)
    group_summary_table.to_csv(output_dir / "fdr_calibration_group_summary.csv", index=False)
    manifest = {
        "candidate_path": str(candidate_path),
        "prediction_path": str(prediction_path),
        "output_dir": str(output_dir),
        "null_models": list(null_models),
        "n_permutations": int(n_permutations),
        "random_seed": int(random_seed),
        "score_column": score_column,
        "alpha": float(alpha),
        "operating_quantile": float(operating_quantile),
        "direct_group_top_fraction": float(direct_group_top_fraction),
        "fdr_paths": fdr_paths,
        "summary_path": str(output_dir / "fdr_calibration_summary.csv"),
        "group_summary_path": str(output_dir / "fdr_calibration_group_summary.csv"),
    }
    with (output_dir / "fdr_calibration_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=True)
    return summary_table, group_summary_table


def _add_empirical_null_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    """Add Gaussian empirical-null and histogram local-FDR diagnostics.

    These columns are diagnostic rather than final claims: they let Phase 2
    separate permutation granularity problems from genuinely non-separated score
    distributions before committing to a manuscript-level FDR procedure.
    """

    result = table.copy()
    null_std = pd.to_numeric(result["null_std"], errors="coerce").to_numpy(dtype=float)
    null_std = np.where(np.isfinite(null_std) & (null_std > 1e-8), null_std, np.nan)
    score = pd.to_numeric(result["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    null_mean = pd.to_numeric(result["null_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    z = np.divide(score - null_mean, null_std, out=np.zeros_like(score), where=np.isfinite(null_std))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    gaussian_p = _normal_survival(z)
    result["null_z"] = z
    result["gaussian_empirical_p_value"] = gaussian_p
    result["gaussian_bh_fdr"] = bh_adjust(gaussian_p)
    result["local_fdr"] = _local_fdr_from_z(z)
    return result


def _normal_survival(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.array([0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in z], dtype=float)


def _local_fdr_from_z(z: np.ndarray, *, n_bins: int = 120) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    out = np.ones(len(z), dtype=float)
    if finite.sum() < 100:
        return out
    values = z[finite]
    center_mask = np.abs(values - np.median(values)) <= np.quantile(np.abs(values - np.median(values)), 0.5)
    center = values[center_mask]
    if len(center) < 30:
        center = values
    mu = float(np.median(center))
    sigma = float(np.std(center))
    if not math.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0

    low, high = np.quantile(values, [0.001, 0.999])
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        return out
    counts, edges = np.histogram(values, bins=n_bins, range=(float(low), float(high)), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    empirical_density = np.interp(values, centers, counts, left=counts[0], right=counts[-1])
    null_density = _normal_density(values, mu, sigma)
    lfdr = np.divide(null_density, empirical_density, out=np.ones_like(values), where=empirical_density > 1e-12)
    out[finite] = np.clip(lfdr, 0.0, 1.0)
    return out


def _normal_density(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    scaled = (values - mu) / sigma
    return np.exp(-0.5 * scaled * scaled) / (sigma * math.sqrt(2.0 * math.pi))


def _has_truth(candidates: pd.DataFrame) -> bool:
    return {
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
    }.issubset(candidates.columns)


def _operating_point_summary(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    score_column: str,
    operating_quantile: float,
) -> dict[str, object]:
    merged = candidates.merge(
        predictions[["candidate_id", score_column]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    scores = pd.to_numeric(merged[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    threshold = float(np.quantile(scores, operating_quantile)) if len(scores) else float("nan")
    called = scores >= threshold if math.isfinite(threshold) else np.zeros(len(merged), dtype=bool)
    result: dict[str, object] = {
        "operating_quantile": float(operating_quantile),
        "operating_score_threshold": threshold,
        "operating_candidate_calls": int(called.sum()),
    }
    if not _has_truth(merged):
        return result

    truth = merged["ground_truth"].astype(int).to_numpy()
    weak = merged["weak_signal"].astype(bool).to_numpy()
    fp_high = merged["false_positive_high_expression"].astype(bool).to_numpy()
    result.update(
        {
            "operating_realized_fdr": _safe_divide(int((called & (truth == 0)).sum()), int(called.sum())),
            "operating_recall": _safe_divide(int((called & (truth == 1)).sum()), int((truth == 1).sum())),
            "operating_weak_recall": _safe_divide(
                int((called & (truth == 1) & weak).sum()),
                int(((truth == 1) & weak).sum()),
            ),
            "operating_fp_high_expression_rate": _safe_divide(
                int((called & fp_high).sum()),
                int(fp_high.sum()),
            ),
        }
    )
    return result


def _available_group_specs(candidates: pd.DataFrame) -> list[tuple[str, list[str]]]:
    specs = [
        ("lr_pair", ["lr_id", "ligand", "receptor", "mechanism"]),
        (
            "cell_type_pair",
            ["sender_cell_type", "receiver_cell_type", "lr_id", "ligand", "receptor", "mechanism"],
        ),
        ("pathway_module", ["mechanism"]),
    ]
    available: list[tuple[str, list[str]]] = []
    for group_level, columns in specs:
        if set(columns).issubset(candidates.columns):
            available.append((group_level, columns))
    return available


def _summarize_group_table(
    group_table: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    group_level: str,
    group_columns: list[str],
    group_path: Path,
    alpha: float,
) -> dict[str, object]:
    control_summary = _group_control_summary(
        candidates,
        group_table,
        group_columns=group_columns,
        alpha=alpha,
    )
    if group_table.empty:
        return control_summary | {
            "group_level": group_level,
            "group_columns": ",".join(group_columns),
            "alpha": float(alpha),
            "n_groups": 0,
            "n_significant_groups": 0,
            "min_group_p_value": float("nan"),
            "min_group_bh_fdr": float("nan"),
            "top_group": "",
            "group_path": str(group_path),
        }
    score_column = "max_score" if "max_score" in group_table.columns else "group_statistic"
    sort_columns = ["group_bh_fdr", "group_p_value"]
    ascending = [True, True]
    if score_column in group_table.columns:
        sort_columns.append(score_column)
        ascending.append(False)
    top = group_table.sort_values(sort_columns, ascending=ascending).iloc[0]
    return control_summary | {
        "group_level": group_level,
        "group_columns": ",".join(group_columns),
        "alpha": float(alpha),
        "n_groups": int(len(group_table)),
        "n_significant_groups": int(group_table["group_significant"].sum()),
        "min_group_p_value": float(group_table["group_p_value"].min()),
        "min_group_bh_fdr": float(group_table["group_bh_fdr"].min()),
        "top_group": _format_group_key(top, group_columns),
        "group_path": str(group_path),
    }


def _group_control_summary(
    candidates: pd.DataFrame,
    group_table: pd.DataFrame,
    *,
    group_columns: list[str],
    alpha: float,
) -> dict[str, object]:
    default = {
        "positive_control_groups": 0,
        "positive_control_significant_groups": 0,
        "positive_control_min_group_bh_fdr": float("nan"),
        "negative_control_groups": 0,
        "negative_control_significant_groups": 0,
        "negative_control_min_group_bh_fdr": float("nan"),
    }
    if group_table.empty or not {"ground_truth", "false_positive_high_expression"}.issubset(candidates.columns):
        return default

    flags = candidates[["candidate_id", *group_columns, "ground_truth", "false_positive_high_expression"]].copy()
    flags["ground_truth"] = flags["ground_truth"].astype(int)
    flags["false_positive_high_expression"] = flags["false_positive_high_expression"].astype(bool)
    group_flags = (
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
    merged = group_table.merge(group_flags, on=group_columns, how="left", validate="one_to_one")
    significant = merged["group_bh_fdr"].astype(float) <= alpha
    positive = merged["group_has_positive"].fillna(False).astype(bool)
    negative = merged["group_has_high_expression_negative"].fillna(False).astype(bool) & ~positive
    positive_fdr = pd.to_numeric(merged.loc[positive, "group_bh_fdr"], errors="coerce")
    negative_fdr = pd.to_numeric(merged.loc[negative, "group_bh_fdr"], errors="coerce")
    return {
        "positive_control_groups": int(positive.sum()),
        "positive_control_significant_groups": int((positive & significant).sum()),
        "positive_control_min_group_bh_fdr": float(positive_fdr.min()) if positive_fdr.notna().any() else float("nan"),
        "negative_control_groups": int(negative.sum()),
        "negative_control_significant_groups": int((negative & significant).sum()),
        "negative_control_min_group_bh_fdr": float(negative_fdr.min()) if negative_fdr.notna().any() else float("nan"),
    }


def _format_group_key(row: pd.Series, group_columns: list[str]) -> str:
    return ";".join(f"{column}={row.get(column, '')}" for column in group_columns)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _parse_null_models(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return PHASE2_NULL_MODELS
    unsupported = sorted(set(values) - set(DEFAULT_NULL_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported null model(s): {unsupported}")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--null-model",
        action="append",
        dest="null_models",
        choices=DEFAULT_NULL_MODELS,
        help="Null model to run. Repeat to run multiple. Defaults to Phase 2 calibration nulls.",
    )
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--operating-quantile", type=float, default=0.94)
    parser.add_argument("--direct-group-top-fraction", type=float, default=0.05)
    args = parser.parse_args()

    summary, group_summary = run_goal2_fdr_calibration(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.output_dir),
        null_models=_parse_null_models(args.null_models),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        score_column=args.score_column,
        alpha=args.alpha,
        operating_quantile=args.operating_quantile,
        direct_group_top_fraction=args.direct_group_top_fraction,
    )
    print(summary.to_string(index=False))
    print("\nGroup-level summary:")
    print(group_summary.to_string(index=False))
    print(f"\nWrote Phase 2 Goal 2 FDR calibration outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
