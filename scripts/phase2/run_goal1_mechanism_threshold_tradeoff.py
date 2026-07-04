#!/usr/bin/env python
"""Evaluate mechanism-specific operating thresholds for Goal 1."""

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


DEFAULT_QUANTILES = (0.99, 0.975, 0.95, 0.945, 0.94, 0.93, 0.92, 0.90, 0.875, 0.85, 0.80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_threshold_tradeoff",
    )
    parser.add_argument("--quantile", action="append", type=float, dest="quantiles")
    parser.add_argument("--reference-quantile", type=float, default=0.95)
    parser.add_argument(
        "--default-mechanism-quantile",
        type=float,
        default=0.95,
        help="Retained for manifest compatibility; non-target mechanisms use the global reference call mask.",
    )
    parser.add_argument(
        "--target-mechanism",
        action="append",
        dest="target_mechanisms",
        default=None,
        help="Mechanism(s) allowed to choose a relaxed threshold. Defaults to all non-negative mechanisms.",
    )
    parser.add_argument("--weak-recall-reference", type=float, default=0.040)
    parser.add_argument("--fdr-increase-max", type=float, default=0.02)
    parser.add_argument("--fp-rate-max", type=float, default=0.005)
    args = parser.parse_args()

    candidates = pd.read_csv(PROJECT_ROOT / args.candidates if not Path(args.candidates).is_absolute() else args.candidates)
    predictions = pd.read_csv(PROJECT_ROOT / args.prediction if not Path(args.prediction).is_absolute() else args.prediction)
    quantiles = tuple(args.quantiles) if args.quantiles else DEFAULT_QUANTILES
    label = args.label or Path(args.prediction).parent.name

    merged = _merged_table(candidates, predictions)
    baseline = _metrics_for_call_mask(
        merged,
        _global_call_mask(merged, float(args.reference_quantile)),
        label=label,
        strategy="global_reference",
        mechanism="all",
        threshold_quantile=float(args.reference_quantile),
        baseline_fdr=np.nan,
        weak_recall_reference=float(args.weak_recall_reference),
        fdr_increase_max=float(args.fdr_increase_max),
        fp_rate_max=float(args.fp_rate_max),
    )
    baseline_fdr = float(baseline["empirical_fdr_at_threshold"])

    global_rows = []
    for quantile in quantiles:
        metrics = evaluate_ground_truth_predictions(candidates, predictions, threshold_quantile=float(quantile))
        global_rows.append(
            {
                "label": label,
                "strategy": "global_quantile",
                "mechanism": "all",
                "threshold_quantile": float(quantile),
                "weak_signal_recall_at_threshold": metrics["weak_signal_recall_at_threshold"],
                "recall_at_threshold": metrics["recall_at_threshold"],
                "empirical_fdr_at_threshold": metrics["empirical_fdr_at_threshold"],
                "false_positive_high_expression_rate_at_threshold": metrics[
                    "false_positive_high_expression_rate_at_threshold"
                ],
                "n_predicted_positive": int((merged["score_norm"] >= np.quantile(merged["score_norm"], float(quantile))).sum()),
                "baseline_reference_quantile": float(args.reference_quantile),
                "baseline_reference_fdr": baseline_fdr,
                "fdr_delta_vs_reference": metrics["empirical_fdr_at_threshold"] - baseline_fdr,
                "weak_recall_reference": float(args.weak_recall_reference),
                "candidate_pass": (
                    metrics["weak_signal_recall_at_threshold"] > float(args.weak_recall_reference)
                    and metrics["empirical_fdr_at_threshold"] - baseline_fdr <= float(args.fdr_increase_max)
                    and metrics["false_positive_high_expression_rate_at_threshold"] <= float(args.fp_rate_max)
                ),
            }
        )

    mechanism_rows = []
    mechanisms = sorted(merged["mechanism"].dropna().astype(str).unique())
    target_mechanisms = set(args.target_mechanisms or [m for m in mechanisms if m != "false_positive_high_expression"])
    for mechanism in mechanisms:
        for quantile in quantiles:
            call_mask = _mechanism_specific_call_mask(
                merged,
                target_mechanism=mechanism,
                target_quantile=float(quantile),
                reference_quantile=float(args.reference_quantile),
            )
            row = _metrics_for_call_mask(
                merged,
                call_mask,
                label=label,
                strategy="mechanism_specific_quantile",
                mechanism=mechanism,
                threshold_quantile=float(quantile),
                baseline_fdr=baseline_fdr,
                weak_recall_reference=float(args.weak_recall_reference),
                fdr_increase_max=float(args.fdr_increase_max),
                fp_rate_max=float(args.fp_rate_max),
            )
            row["target_mechanism_allowed"] = mechanism in target_mechanisms
            row["default_mechanism_quantile"] = float(args.default_mechanism_quantile)
            mechanism_rows.append(row)

    mechanism_table = pd.DataFrame(mechanism_rows)
    allowed = mechanism_table[mechanism_table["target_mechanism_allowed"].astype(bool)].copy()
    best_by_mechanism = _best_rows(allowed, group_columns=["mechanism"])
    combined_call = _combined_best_call_mask(
        merged,
        best_by_mechanism,
        reference_quantile=float(args.reference_quantile),
    )
    combined = _metrics_for_call_mask(
        merged,
        combined_call,
        label=label,
        strategy="combined_mechanism_specific",
        mechanism="allowed_targets",
        threshold_quantile=np.nan,
        baseline_fdr=baseline_fdr,
        weak_recall_reference=float(args.weak_recall_reference),
        fdr_increase_max=float(args.fdr_increase_max),
        fp_rate_max=float(args.fp_rate_max),
    )
    combined["selected_mechanism_quantiles"] = json.dumps(
        {
            str(row.mechanism): float(row.threshold_quantile)
            for row in best_by_mechanism.itertuples(index=False)
        },
        sort_keys=True,
    )
    combined["default_mechanism_quantile"] = float(args.default_mechanism_quantile)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    global_table = pd.DataFrame(global_rows)
    summary = pd.concat(
        [
            pd.DataFrame([baseline]),
            _best_rows(global_table, group_columns=["strategy"]),
            best_by_mechanism,
            pd.DataFrame([combined]),
        ],
        ignore_index=True,
        sort=False,
    )

    global_table.to_csv(output_dir / "global_threshold_tradeoff.csv", index=False)
    mechanism_table.to_csv(output_dir / "mechanism_threshold_tradeoff.csv", index=False)
    summary.to_csv(output_dir / "mechanism_threshold_tradeoff_summary.csv", index=False)
    manifest = {
        "candidates": args.candidates,
        "prediction": args.prediction,
        "label": label,
        "quantiles": list(quantiles),
        "reference_quantile": args.reference_quantile,
        "default_mechanism_quantile": args.default_mechanism_quantile,
        "target_mechanisms": sorted(target_mechanisms),
        "weak_recall_reference": args.weak_recall_reference,
        "fdr_increase_max": args.fdr_increase_max,
        "fp_rate_max": args.fp_rate_max,
        "analysis_scope": (
            "Post-hoc mechanism-specific operating-point diagnostic from an existing prediction table; "
            "not a newly trained model and not a formal FDR procedure."
        ),
    }
    (output_dir / "mechanism_threshold_tradeoff_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote mechanism threshold tradeoff outputs to {output_dir}")


def _merged_table(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "ground_truth", "weak_signal", "false_positive_high_expression", "mechanism"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    if not {"candidate_id", "score"}.issubset(predictions.columns):
        raise ValueError("Prediction table must contain candidate_id and score columns.")
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions.")
    merged["score_norm"] = _minmax(merged["score"].to_numpy(dtype=float))
    return merged


def _global_call_mask(merged: pd.DataFrame, quantile: float) -> np.ndarray:
    return merged["score_norm"].to_numpy(dtype=float) >= float(np.quantile(merged["score_norm"], quantile))


def _mechanism_specific_call_mask(
    merged: pd.DataFrame,
    *,
    target_mechanism: str,
    target_quantile: float,
    reference_quantile: float,
) -> np.ndarray:
    scores = merged["score_norm"].to_numpy(dtype=float)
    mechanisms = merged["mechanism"].astype(str).to_numpy()
    call = _global_call_mask(merged, float(reference_quantile))
    mask = mechanisms == target_mechanism
    if mask.any():
        threshold = float(np.quantile(scores[mask], float(target_quantile)))
        call[mask] = scores[mask] >= threshold
    return call


def _combined_best_call_mask(merged: pd.DataFrame, best: pd.DataFrame, *, reference_quantile: float) -> np.ndarray:
    quantiles = {
        str(row.mechanism): float(row.threshold_quantile)
        for row in best.itertuples(index=False)
        if pd.notna(row.threshold_quantile)
    }
    scores = merged["score_norm"].to_numpy(dtype=float)
    mechanisms = merged["mechanism"].astype(str).to_numpy()
    call = _global_call_mask(merged, float(reference_quantile))
    for mechanism, quantile in quantiles.items():
        mask = mechanisms == mechanism
        if not mask.any():
            continue
        threshold = float(np.quantile(scores[mask], quantile))
        call[mask] = scores[mask] >= threshold
    return call


def _metrics_for_call_mask(
    merged: pd.DataFrame,
    call_mask: np.ndarray,
    *,
    label: str,
    strategy: str,
    mechanism: str,
    threshold_quantile: float,
    baseline_fdr: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> dict:
    truth = merged["ground_truth"].astype(bool).to_numpy()
    weak = merged["weak_signal"].astype(bool).to_numpy()
    false_positive = merged["false_positive_high_expression"].astype(bool).to_numpy()
    called = np.asarray(call_mask, dtype=bool)
    fdr = _safe_divide(int((called & ~truth).sum()), int(called.sum()))
    fp_rate = _safe_divide(int((called & false_positive).sum()), int(false_positive.sum()))
    weak_recall = _safe_divide(int((called & truth & weak).sum()), int((truth & weak).sum()))
    recall = _safe_divide(int((called & truth).sum()), int(truth.sum()))
    fdr_delta = fdr - baseline_fdr if np.isfinite(baseline_fdr) else np.nan
    return {
        "label": label,
        "strategy": strategy,
        "mechanism": mechanism,
        "threshold_quantile": float(threshold_quantile) if np.isfinite(threshold_quantile) else np.nan,
        "n_candidates": int(len(merged)),
        "n_predicted_positive": int(called.sum()),
        "weak_signal_recall_at_threshold": weak_recall,
        "recall_at_threshold": recall,
        "empirical_fdr_at_threshold": fdr,
        "false_positive_high_expression_rate_at_threshold": fp_rate,
        "baseline_reference_fdr": baseline_fdr,
        "fdr_delta_vs_reference": fdr_delta,
        "weak_recall_reference": float(weak_recall_reference),
        "candidate_pass": bool(
            weak_recall > float(weak_recall_reference)
            and (not np.isfinite(fdr_delta) or fdr_delta <= float(fdr_increase_max))
            and fp_rate <= float(fp_rate_max)
        ),
    }


def _best_rows(table: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    if table.empty:
        return table.copy()
    for _, group in table.groupby(group_columns, sort=False):
        passing = group[group["candidate_pass"].astype(bool)]
        source = passing if not passing.empty else group
        best = source.sort_values(
            ["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold", "false_positive_high_expression_rate_at_threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        best = best.copy()
        best["selection_status"] = "passing_operating_point" if not passing.empty else "no_passing_operating_point"
        best["n_passing_quantiles"] = int(len(passing))
        rows.append(best)
    return pd.DataFrame(rows)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


if __name__ == "__main__":
    main()
