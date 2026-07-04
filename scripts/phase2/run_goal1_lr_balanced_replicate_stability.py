#!/usr/bin/env python
"""Evaluate LR-balanced operating-rule stability across controlled replicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2 import run_goal1_lr_balanced_threshold as lr_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/phase1/exp01_controlled_benchmark/datasets",
    )
    parser.add_argument(
        "--prediction-root",
        default=(
            "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/"
            "wavelet_mechanism_niche_edge_prior_ranking"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal1_weak_signal_recovery/controlled_lr_balanced_threshold_stability",
    )
    parser.add_argument("--reference-quantile", type=float, default=0.95)
    parser.add_argument("--weak-recall-reference", type=float, default=0.155)
    parser.add_argument("--fdr-increase-max", type=float, default=0.02)
    parser.add_argument("--fp-rate-max", type=float, default=0.005)
    args = parser.parse_args()

    dataset_root = _resolve(args.dataset_root)
    prediction_root = _resolve(args.prediction_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for candidate_path in sorted(dataset_root.glob("replicate_*/candidate_interactions.csv")):
        replicate = candidate_path.parent.name
        prediction_path = prediction_root / f"{replicate}__spatialself_wavelet_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction for {replicate}: {prediction_path}")
        metrics, summary = evaluate_focused_lr_balanced_threshold(
            candidates=pd.read_csv(candidate_path),
            predictions=pd.read_csv(prediction_path),
            label="edge_prior_ranking",
            reference_quantile=args.reference_quantile,
            weak_recall_reference=args.weak_recall_reference,
            fdr_increase_max=args.fdr_increase_max,
            fp_rate_max=args.fp_rate_max,
        )
        metrics.insert(0, "replicate", replicate)
        summary.insert(0, "replicate", replicate)
        metrics_tables.append(metrics)
        summary_tables.append(summary)

    if not metrics_tables:
        raise SystemExit(f"No controlled replicate candidate tables found under {dataset_root}")

    metrics_table = pd.concat(metrics_tables, ignore_index=True)
    replicate_summary = pd.concat(summary_tables, ignore_index=True)
    aggregate = _aggregate_summary(replicate_summary)

    metrics_table.to_csv(output_dir / "lr_balanced_replicate_metrics.csv", index=False)
    replicate_summary.to_csv(output_dir / "lr_balanced_replicate_summary.csv", index=False)
    aggregate.to_csv(output_dir / "lr_balanced_replicate_aggregate.csv", index=False)
    manifest = {
        "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT)),
        "prediction_root": str(prediction_root.relative_to(PROJECT_ROOT)),
        "reference_quantile": args.reference_quantile,
        "weak_recall_reference": args.weak_recall_reference,
        "fdr_increase_max": args.fdr_increase_max,
        "fp_rate_max": args.fp_rate_max,
        "n_replicates": len(summary_tables),
        "analysis_scope": (
            "Controlled benchmark cross-replicate stability diagnostic for a post-hoc LR-balanced "
            "operating rule. This does not replace full-slide realistic-background replicate validation."
        ),
    }
    (output_dir / "lr_balanced_replicate_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Wrote controlled LR-balanced stability outputs to {output_dir}")


def _aggregate_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "best_n_predicted_positive",
        "best_weak_signal_recall",
        "best_recall",
        "best_fdr",
        "best_fdr_delta_vs_reference",
        "best_false_positive_high_expression_rate",
        "n_passing_settings",
    ]
    rows = []
    for strategy, group in replicate_summary.groupby("strategy", sort=False):
        row: dict[str, object] = {
            "strategy": strategy,
            "n_replicates": int(group["replicate"].nunique()),
            "n_passing_replicates": int((group["status"] == "passing_operating_point").sum()),
            "passing_replicate_fraction": float((group["status"] == "passing_operating_point").mean()),
            "all_replicates_pass": bool((group["status"] == "passing_operating_point").all()),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        passing = group[group["status"] == "passing_operating_point"]
        if passing.empty:
            best = group.sort_values(
                ["best_weak_signal_recall", "best_fdr"],
                ascending=[False, True],
            ).iloc[0]
            row["aggregate_status"] = "no_all_replicate_operating_point"
        else:
            best = passing.sort_values(
                ["best_weak_signal_recall", "best_fdr"],
                ascending=[False, True],
            ).iloc[0]
            row["aggregate_status"] = (
                "all_replicates_pass" if bool(row["all_replicates_pass"]) else "partial_replicate_pass"
            )
        row["representative_lr_quantile"] = best.get("best_lr_quantile")
        row["representative_global_floor_quantile"] = best.get("best_global_floor_quantile")
        row["representative_threshold_quantile"] = best.get("best_threshold_quantile")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_focused_lr_balanced_threshold(
    *,
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    label: str,
    reference_quantile: float,
    weak_recall_reference: float,
    fdr_increase_max: float,
    fp_rate_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected operating rules for replicate-stability, not a full threshold search."""

    merged = lr_threshold._merged_table(candidates, predictions)
    reference_call = lr_threshold._global_call_mask(merged, reference_quantile)
    reference = lr_threshold._metrics_for_call_mask(
        merged,
        reference_call,
        label=label,
        strategy="global_reference",
        lr_quantile=float("nan"),
        global_floor_quantile=float("nan"),
        top_k_per_lr=float("nan"),
        threshold_quantile=reference_quantile,
        baseline_fdr=float("nan"),
        weak_recall_reference=weak_recall_reference,
        fdr_increase_max=fdr_increase_max,
        fp_rate_max=fp_rate_max,
    )
    baseline_fdr = float(reference["empirical_fdr_at_threshold"])
    rows = [reference]
    for quantile in (0.94, 0.90):
        rows.append(
            lr_threshold._metrics_for_call_mask(
                merged,
                lr_threshold._global_call_mask(merged, quantile),
                label=label,
                strategy="global_quantile",
                lr_quantile=float("nan"),
                global_floor_quantile=float("nan"),
                top_k_per_lr=float("nan"),
                threshold_quantile=quantile,
                baseline_fdr=baseline_fdr,
                weak_recall_reference=weak_recall_reference,
                fdr_increase_max=fdr_increase_max,
                fp_rate_max=fp_rate_max,
            )
        )
    lr_call = lr_threshold._lr_quantile_call_mask(merged, 0.90)
    floor_call = lr_threshold._global_call_mask(merged, 0.90)
    rows.append(
        lr_threshold._metrics_for_call_mask(
            merged,
            lr_call,
            label=label,
            strategy="lrwise_quantile",
            lr_quantile=0.90,
            global_floor_quantile=float("nan"),
            top_k_per_lr=float("nan"),
            threshold_quantile=float("nan"),
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    rows.append(
        lr_threshold._metrics_for_call_mask(
            merged,
            lr_call & floor_call,
            label=label,
            strategy="lrwise_quantile_with_global_floor",
            lr_quantile=0.90,
            global_floor_quantile=0.90,
            top_k_per_lr=float("nan"),
            threshold_quantile=float("nan"),
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    rows.append(
        lr_threshold._metrics_for_call_mask(
            merged,
            reference_call | (lr_call & floor_call),
            label=label,
            strategy="global_reference_plus_lrwise_floor",
            lr_quantile=0.90,
            global_floor_quantile=0.90,
            top_k_per_lr=float("nan"),
            threshold_quantile=reference_quantile,
            baseline_fdr=baseline_fdr,
            weak_recall_reference=weak_recall_reference,
            fdr_increase_max=fdr_increase_max,
            fp_rate_max=fp_rate_max,
        )
    )
    metrics = pd.DataFrame(rows)
    return metrics, lr_threshold._summarize(metrics)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


if __name__ == "__main__":
    main()
