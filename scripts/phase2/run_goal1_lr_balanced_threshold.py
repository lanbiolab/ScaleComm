#!/usr/bin/env python
"""Evaluate LR-pair-balanced operating rules for Phase 2 Goal 1."""

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


DEFAULT_GLOBAL_QUANTILES = (0.95, 0.94, 0.93, 0.92, 0.90, 0.85)
DEFAULT_LR_QUANTILES = (0.995, 0.99, 0.975, 0.95, 0.925, 0.90, 0.85, 0.80)
DEFAULT_GLOBAL_FLOORS = (0.95, 0.94, 0.93, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70)
DEFAULT_TOP_K_PER_LR = (1, 2, 5, 10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--reference-quantile", type=float, default=0.95)
    parser.add_argument("--weak-recall-reference", type=float, default=0.040)
    parser.add_argument("--fdr-increase-max", type=float, default=0.02)
    parser.add_argument("--fp-rate-max", type=float, default=0.005)
    args = parser.parse_args()

    label = args.label or Path(args.prediction).parent.name
    candidates = pd.read_csv(_resolve_path(args.candidates))
    predictions = pd.read_csv(_resolve_path(args.prediction))
    table, summary = evaluate_lr_balanced_threshold(
        candidates,
        predictions,
        label=label,
        reference_quantile=args.reference_quantile,
        weak_recall_reference=args.weak_recall_reference,
        fdr_increase_max=args.fdr_increase_max,
        fp_rate_max=args.fp_rate_max,
    )

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "lr_balanced_threshold_metrics.csv", index=False)
    summary.to_csv(output_dir / "lr_balanced_threshold_summary.csv", index=False)
    manifest = {
        "candidates": args.candidates,
        "prediction": args.prediction,
        "label": label,
        "reference_quantile": args.reference_quantile,
        "global_quantiles": list(DEFAULT_GLOBAL_QUANTILES),
        "lr_quantiles": list(DEFAULT_LR_QUANTILES),
        "global_floor_quantiles": list(DEFAULT_GLOBAL_FLOORS),
        "top_k_per_lr": list(DEFAULT_TOP_K_PER_LR),
        "weak_recall_reference": args.weak_recall_reference,
        "fdr_increase_max": args.fdr_increase_max,
        "fp_rate_max": args.fp_rate_max,
        "analysis_scope": (
            "Post-hoc LR-pair-balanced operating-rule diagnostic from an existing prediction table. "
            "This evaluates thresholding behavior and should not be treated as a newly trained model "
            "or as a formal FDR procedure."
        ),
    }
    (output_dir / "lr_balanced_threshold_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote LR-balanced threshold outputs to {output_dir}")


def evaluate_lr_balanced_threshold(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    label: str,
    reference_quantile: float = 0.95,
    weak_recall_reference: float = 0.040,
    fdr_increase_max: float = 0.02,
    fp_rate_max: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-setting metrics and per-strategy summary for LR-balanced calls."""

    merged = _merged_table(candidates, predictions)
    reference_call = _global_call_mask(merged, reference_quantile)
    reference = _metrics_for_call_mask(
        merged,
        reference_call,
        label=label,
        strategy="global_reference",
        lr_quantile=np.nan,
        global_floor_quantile=np.nan,
        top_k_per_lr=np.nan,
        threshold_quantile=reference_quantile,
        baseline_fdr=np.nan,
        weak_recall_reference=weak_recall_reference,
        fdr_increase_max=fdr_increase_max,
        fp_rate_max=fp_rate_max,
    )
    baseline_fdr = float(reference["empirical_fdr_at_threshold"])

    rows: list[dict[str, object]] = [reference]
    rows.extend(
        _global_rows(
            merged,
            label=label,
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    rows.extend(
        _lr_quantile_rows(
            merged,
            label=label,
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    rows.extend(
        _lr_top_k_rows(
            merged,
            label=label,
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    table = pd.DataFrame(rows)
    return table, _summarize(table)


def _global_rows(
    merged: pd.DataFrame,
    *,
    label: str,
    baseline_fdr: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> list[dict[str, object]]:
    rows = []
    for quantile in DEFAULT_GLOBAL_QUANTILES:
        rows.append(
            _metrics_for_call_mask(
                merged,
                _global_call_mask(merged, quantile),
                label=label,
                strategy="global_quantile",
                lr_quantile=np.nan,
                global_floor_quantile=np.nan,
                top_k_per_lr=np.nan,
                threshold_quantile=quantile,
                baseline_fdr=baseline_fdr,
                weak_recall_reference=weak_recall_reference,
                fdr_increase_max=fdr_increase_max,
                fp_rate_max=fp_rate_max,
            )
        )
    return rows


def _lr_quantile_rows(
    merged: pd.DataFrame,
    *,
    label: str,
    baseline_fdr: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> list[dict[str, object]]:
    rows = []
    for lr_quantile in DEFAULT_LR_QUANTILES:
        rows.append(
            _metrics_for_call_mask(
                merged,
                _lr_quantile_call_mask(merged, lr_quantile),
                label=label,
                strategy="lrwise_quantile",
                lr_quantile=lr_quantile,
                global_floor_quantile=np.nan,
                top_k_per_lr=np.nan,
                threshold_quantile=np.nan,
                baseline_fdr=baseline_fdr,
                weak_recall_reference=weak_recall_reference,
                fdr_increase_max=fdr_increase_max,
                fp_rate_max=fp_rate_max,
            )
        )
        for floor_quantile in DEFAULT_GLOBAL_FLOORS:
            lr_call = _lr_quantile_call_mask(merged, lr_quantile)
            floor_call = _global_call_mask(merged, floor_quantile)
            rows.append(
                _metrics_for_call_mask(
                    merged,
                    lr_call & floor_call,
                    label=label,
                    strategy="lrwise_quantile_with_global_floor",
                    lr_quantile=lr_quantile,
                    global_floor_quantile=floor_quantile,
                    top_k_per_lr=np.nan,
                    threshold_quantile=np.nan,
                    baseline_fdr=baseline_fdr,
                    weak_recall_reference=weak_recall_reference,
                    fdr_increase_max=fdr_increase_max,
                    fp_rate_max=fp_rate_max,
                )
            )
            rows.append(
                _metrics_for_call_mask(
                    merged,
                    _global_call_mask(merged, 0.95) | (lr_call & floor_call),
                    label=label,
                    strategy="global_reference_plus_lrwise_floor",
                    lr_quantile=lr_quantile,
                    global_floor_quantile=floor_quantile,
                    top_k_per_lr=np.nan,
                    threshold_quantile=0.95,
                    baseline_fdr=baseline_fdr,
                    weak_recall_reference=weak_recall_reference,
                    fdr_increase_max=fdr_increase_max,
                    fp_rate_max=fp_rate_max,
                )
            )
    return rows


def _lr_top_k_rows(
    merged: pd.DataFrame,
    *,
    label: str,
    baseline_fdr: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> list[dict[str, object]]:
    rows = []
    for top_k in DEFAULT_TOP_K_PER_LR:
        rows.append(
            _metrics_for_call_mask(
                merged,
                _lr_top_k_call_mask(merged, top_k),
                label=label,
                strategy="lrwise_top_k",
                lr_quantile=np.nan,
                global_floor_quantile=np.nan,
                top_k_per_lr=float(top_k),
                threshold_quantile=np.nan,
                baseline_fdr=baseline_fdr,
                weak_recall_reference=weak_recall_reference,
                fdr_increase_max=fdr_increase_max,
                fp_rate_max=fp_rate_max,
            )
        )
        for floor_quantile in DEFAULT_GLOBAL_FLOORS:
            top_call = _lr_top_k_call_mask(merged, top_k)
            floor_call = _global_call_mask(merged, floor_quantile)
            rows.append(
                _metrics_for_call_mask(
                    merged,
                    _global_call_mask(merged, 0.95) | (top_call & floor_call),
                    label=label,
                    strategy="global_reference_plus_lr_top_k_floor",
                    lr_quantile=np.nan,
                    global_floor_quantile=floor_quantile,
                    top_k_per_lr=float(top_k),
                    threshold_quantile=0.95,
                    baseline_fdr=baseline_fdr,
                    weak_recall_reference=weak_recall_reference,
                    fdr_increase_max=fdr_increase_max,
                    fp_rate_max=fp_rate_max,
                )
            )
    return rows


def _merged_table(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "lr_id",
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
        "mechanism",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    if not {"candidate_id", "score"}.issubset(predictions.columns):
        raise ValueError("Prediction table must contain candidate_id and score columns.")
    merged = candidates.merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions.")
    merged["score_norm"] = _minmax(merged["score"].to_numpy(dtype=float))
    merged["lr_id"] = merged["lr_id"].astype(str)
    return merged


def _global_call_mask(merged: pd.DataFrame, quantile: float) -> np.ndarray:
    scores = merged["score_norm"].to_numpy(dtype=float)
    return scores >= float(np.quantile(scores, quantile))


def _lr_quantile_call_mask(merged: pd.DataFrame, lr_quantile: float) -> np.ndarray:
    scores = merged["score_norm"].to_numpy(dtype=float)
    call = np.zeros(len(merged), dtype=bool)
    for indices in _lr_indices(merged).values():
        group_scores = scores[indices]
        threshold = float(np.quantile(group_scores, lr_quantile))
        call[indices] = group_scores >= threshold
    return call


def _lr_top_k_call_mask(merged: pd.DataFrame, top_k: int) -> np.ndarray:
    scores = merged["score_norm"].to_numpy(dtype=float)
    call = np.zeros(len(merged), dtype=bool)
    top_k = int(top_k)
    for indices in _lr_indices(merged).values():
        if len(indices) == 0:
            continue
        keep = min(top_k, len(indices))
        local_order = np.argsort(-scores[indices])[:keep]
        call[indices[local_order]] = True
    return call


def _lr_indices(merged: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(lr_id): group.index.to_numpy(dtype=int)
        for lr_id, group in merged.groupby("lr_id", sort=False)
    }


def _metrics_for_call_mask(
    merged: pd.DataFrame,
    call_mask: np.ndarray,
    *,
    label: str,
    strategy: str,
    lr_quantile: float,
    global_floor_quantile: float,
    top_k_per_lr: float,
    threshold_quantile: float,
    baseline_fdr: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> dict[str, object]:
    y_true = merged["ground_truth"].astype(int).to_numpy()
    weak = merged["weak_signal"].astype(bool).to_numpy()
    high_fp = merged["false_positive_high_expression"].astype(bool).to_numpy()
    call = np.asarray(call_mask, dtype=bool)
    n_pred = int(call.sum())
    tp = int((call & (y_true == 1)).sum())
    fp = int((call & (y_true == 0)).sum())
    weak_tp = int((call & (y_true == 1) & weak).sum())
    n_pos = int(y_true.sum())
    n_weak_pos = int(((y_true == 1) & weak).sum())
    n_high_fp = int(high_fp.sum())
    empirical_fdr = _safe_divide(fp, n_pred)
    fdr_delta = empirical_fdr - baseline_fdr if np.isfinite(baseline_fdr) else np.nan
    fp_rate = _safe_divide(int((call & high_fp).sum()), n_high_fp)
    row = {
        "label": label,
        "strategy": strategy,
        "threshold_quantile": threshold_quantile,
        "lr_quantile": lr_quantile,
        "global_floor_quantile": global_floor_quantile,
        "top_k_per_lr": top_k_per_lr,
        "n_candidates": int(len(merged)),
        "n_lr_pairs": int(merged["lr_id"].nunique()),
        "n_predicted_positive": n_pred,
        "recall_at_threshold": _safe_divide(tp, n_pos),
        "weak_signal_recall_at_threshold": _safe_divide(weak_tp, n_weak_pos),
        "empirical_fdr_at_threshold": empirical_fdr,
        "false_positive_high_expression_rate_at_threshold": fp_rate,
        "baseline_reference_fdr": baseline_fdr,
        "fdr_delta_vs_reference": fdr_delta,
        "weak_recall_reference": weak_recall_reference,
        "candidate_pass": (
            _safe_divide(weak_tp, n_weak_pos) > weak_recall_reference
            and (np.isnan(fdr_delta) or fdr_delta <= fdr_increase_max)
            and fp_rate <= fp_rate_max
        ),
    }
    for mechanism, subset in merged.groupby("mechanism", sort=False):
        idx = subset.index.to_numpy(dtype=int)
        mech_true = y_true[idx]
        mech_weak = weak[idx]
        mech_call = call[idx]
        row[f"weak_signal_recall__{mechanism}"] = _safe_divide(
            int((mech_call & (mech_true == 1) & mech_weak).sum()),
            int(((mech_true == 1) & mech_weak).sum()),
        )
        row[f"recall__{mechanism}"] = _safe_divide(
            int((mech_call & (mech_true == 1)).sum()),
            int(mech_true.sum()),
        )
    return row


def _summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in table.groupby("strategy", sort=False):
        passing = group[group["candidate_pass"].astype(bool)]
        if passing.empty:
            best = group.sort_values(
                ["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold"],
                ascending=[False, True],
            ).iloc[0]
            status = "no_passing_operating_point"
        else:
            best = passing.sort_values(
                ["weak_signal_recall_at_threshold", "empirical_fdr_at_threshold"],
                ascending=[False, True],
            ).iloc[0]
            status = "passing_operating_point"
        rows.append(
            {
                "strategy": strategy,
                "status": status,
                "best_threshold_quantile": best.get("threshold_quantile", np.nan),
                "best_lr_quantile": best.get("lr_quantile", np.nan),
                "best_global_floor_quantile": best.get("global_floor_quantile", np.nan),
                "best_top_k_per_lr": best.get("top_k_per_lr", np.nan),
                "best_n_predicted_positive": int(best["n_predicted_positive"]),
                "best_weak_signal_recall": best["weak_signal_recall_at_threshold"],
                "best_recall": best["recall_at_threshold"],
                "best_fdr": best["empirical_fdr_at_threshold"],
                "best_fdr_delta_vs_reference": best["fdr_delta_vs_reference"],
                "best_false_positive_high_expression_rate": best[
                    "false_positive_high_expression_rate_at_threshold"
                ],
                "n_passing_settings": int(len(passing)),
            }
        )
    return pd.DataFrame(rows)


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
