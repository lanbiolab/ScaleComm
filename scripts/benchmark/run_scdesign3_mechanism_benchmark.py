#!/usr/bin/env python
"""Assemble the scDesign3-style mechanism-aware CCC benchmark.

This runner keeps the evaluation contract explicit:

1. reference/scDesign3-like expression backgrounds are evaluated through one
   shared candidate-edge universe;
2. injected LR mechanisms are contact, short/medium paracrine, and long-range
   niche-restricted communication;
3. every method is scored against the same candidate_id table, including
   wrong-range and wrong-niche decoys.

The script can reuse existing prediction files from longer method runs. It does
not tune any method while assembling the benchmark.
"""

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

from scripts.benchmark.evaluate_range_mechanism_benchmark import evaluate_range_mechanism
from src.spitialself.evaluation import evaluate_predictions_auto


DATASETS = {
    "lymph_scdesign3": {
        "tissue": "lymph_node",
        "replicate_dir": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/lymph/reference_lymph_node_scdesign3_replicate_00",
        "predictions": {
            "SpatialSelf": "outputs/reference_based_benchmark/spatialself_mechanism_tuning/runs/range_context_guard/lymph_scdesign3/reference_lymph_node_scdesign3_replicate_00__spatialself_range_context_predictions.csv",
            "COMMOT": "outputs/reference_based_benchmark/native_rerun/commot_native/lymph/reference_lymph_node_scdesign3_replicate_00__commot_predictions.csv",
            "CellChat": "outputs/reference_based_benchmark/full_scdesign3_strongest/lymph/cellchat/reference_lymph_node_scdesign3_replicate_00__cellchat_predictions.csv",
            "SpaTalk": "outputs/reference_based_benchmark/full_scdesign3_strongest/lymph/spatalk/reference_lymph_node_scdesign3_replicate_00__spatalk_predictions.csv",
            "CellNEST": "outputs/reference_based_benchmark/native_rerun/cellnest_native_official_like/lymph_th60_k50_e500/reference_lymph_node_scdesign3_replicate_00__cellnest_predictions.csv",
            "DeepTalk": "outputs/reference_based_benchmark/full_scdesign3_strongest/lymph/deeptalk/reference_lymph_node_scdesign3_replicate_00__deeptalk_predictions.csv",
            "CellPhoneDB": "outputs/reference_based_benchmark/native_rerun/extended_baselines/lymph/cellphonedb/reference_lymph_node_scdesign3_replicate_00__cellphonedb_predictions.csv",
            "NICHES": "outputs/reference_based_benchmark/native_rerun/extended_baselines/lymph/niches/reference_lymph_node_scdesign3_replicate_00__niches_predictions.csv",
            "CytoSignal": "outputs/reference_based_benchmark/native_rerun/extended_baselines/lymph/cytosignal/reference_lymph_node_scdesign3_replicate_00__cytosignal_predictions.csv",
            "SpaOTsc": "outputs/reference_based_benchmark/native_rerun/extended_baselines/lymph/spaotsc/reference_lymph_node_scdesign3_replicate_00__spaotsc_predictions.csv",
        },
    },
    "melanoma_scdesign3": {
        "tissue": "melanoma",
        "replicate_dir": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/melanoma/reference_melanoma_scdesign3_replicate_00",
        "predictions": {
            "SpatialSelf": "outputs/reference_based_benchmark/spatialself_mechanism_tuning/runs/range_context_guard/melanoma_scdesign3/reference_melanoma_scdesign3_replicate_00__spatialself_range_context_predictions.csv",
            "COMMOT": "outputs/reference_based_benchmark/native_rerun/commot_native/melanoma/reference_melanoma_scdesign3_replicate_00__commot_predictions.csv",
            "CellChat": "outputs/reference_based_benchmark/full_scdesign3_strongest/melanoma/cellchat/reference_melanoma_scdesign3_replicate_00__cellchat_predictions.csv",
            "SpaTalk": "outputs/reference_based_benchmark/full_scdesign3_strongest/melanoma/spatalk/reference_melanoma_scdesign3_replicate_00__spatalk_predictions.csv",
            "CellNEST": "outputs/reference_based_benchmark/native_rerun/cellnest_native_official_like/melanoma_th60_k50_e500/reference_melanoma_scdesign3_replicate_00__cellnest_predictions.csv",
            "DeepTalk": "outputs/reference_based_benchmark/native_rerun/deeptalk_official/melanoma/reference_melanoma_scdesign3_replicate_00__deeptalk_predictions.csv",
            "CellPhoneDB": "outputs/reference_based_benchmark/native_rerun/extended_baselines/melanoma/cellphonedb/reference_melanoma_scdesign3_replicate_00__cellphonedb_predictions.csv",
            "NICHES": "outputs/reference_based_benchmark/native_rerun/extended_baselines/melanoma/niches/reference_melanoma_scdesign3_replicate_00__niches_predictions.csv",
            "CytoSignal": "outputs/reference_based_benchmark/native_rerun/extended_baselines/melanoma/cytosignal/reference_melanoma_scdesign3_replicate_00__cytosignal_predictions.csv",
            "SpaOTsc": "outputs/reference_based_benchmark/native_rerun/extended_baselines/melanoma/spaotsc/reference_melanoma_scdesign3_replicate_00__spaotsc_predictions.csv",
        },
    },
}

METHOD_ORDER = {
    "SpatialSelf": 0,
    "COMMOT": 1,
    "CellChat": 2,
    "CellNEST": 3,
    "DeepTalk": 4,
    "SpaTalk": 5,
    "CellPhoneDB": 6,
    "NICHES": 7,
    "CytoSignal": 8,
    "SpaOTsc": 9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_benchmark",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    status_rows: list[dict] = []
    challenge_rows: list[dict] = []
    truth_rows: list[dict] = []
    range_tables: list[pd.DataFrame] = []

    for dataset_id, dataset_cfg in DATASETS.items():
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]
        if not replicate_dir.exists():
            raise FileNotFoundError(f"Missing benchmark replicate: {replicate_dir}")

        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        truth_rows.extend(_truth_contract_rows(dataset_id, dataset_cfg["tissue"], candidates))

        prediction_paths: dict[str, Path] = {}
        for method, rel_path in dataset_cfg["predictions"].items():
            prediction_path = PROJECT_ROOT / rel_path
            if not prediction_path.exists():
                status_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "status": "missing_prediction",
                        "detail": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )
                continue
            prediction = _read_prediction(prediction_path, method)
            metrics = evaluate_predictions_auto(
                candidates,
                prediction,
                top_k=args.top_k,
                threshold_quantile=args.threshold_quantile,
            )
            metrics.update(
                {
                    "dataset_id": dataset_id,
                    "tissue": dataset_cfg["tissue"],
                    "method": method,
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                }
            )
            metric_rows.append(metrics)
            challenge_rows.append(
                {
                    "dataset_id": dataset_id,
                    "tissue": dataset_cfg["tissue"],
                    "method": method,
                    **_challenge_metrics(
                        candidates,
                        prediction,
                        threshold_quantile=args.threshold_quantile,
                    ),
                }
            )
            status_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "status": "completed",
                    "detail": str(prediction_path.relative_to(PROJECT_ROOT)),
                }
            )
            local_prediction_dir = output_dir / "normalized_predictions" / dataset_id
            local_prediction_dir.mkdir(parents=True, exist_ok=True)
            local_prediction_path = local_prediction_dir / f"{method.lower().replace(' ', '_')}__predictions.csv"
            prediction.to_csv(local_prediction_path, index=False)
            prediction_paths[method] = local_prediction_path

        if prediction_paths:
            range_out = output_dir / "range_mechanism" / dataset_id
            tables = evaluate_range_mechanism(
                replicate_dir,
                range_out,
                predictions=prediction_paths,
                top_k=args.top_k,
                threshold_quantile=args.threshold_quantile,
            )
            mechanism = tables["mechanism_range_summary"].copy()
            mechanism["dataset_id"] = dataset_id
            mechanism["tissue"] = dataset_cfg["tissue"]
            range_tables.append(mechanism)

    detail = _sort_methods(pd.DataFrame(metric_rows))
    status = _sort_methods(pd.DataFrame(status_rows))
    challenge = _sort_methods(pd.DataFrame(challenge_rows))
    truth_contract = pd.DataFrame(truth_rows)
    mechanism = _sort_methods(pd.concat(range_tables, ignore_index=True)) if range_tables else pd.DataFrame()
    range_class_summary = _range_class_summary(mechanism)
    summary = _summary(detail, challenge)

    detail.to_csv(output_dir / "mechanism_benchmark_detail.csv", index=False)
    summary.to_csv(output_dir / "mechanism_benchmark_summary.csv", index=False)
    status.to_csv(output_dir / "mechanism_benchmark_status.csv", index=False)
    challenge.to_csv(output_dir / "mechanism_benchmark_challenge_metrics.csv", index=False)
    truth_contract.to_csv(output_dir / "mechanism_benchmark_truth_contract.csv", index=False)
    mechanism.to_csv(output_dir / "mechanism_benchmark_by_mechanism.csv", index=False)
    range_class_summary.to_csv(output_dir / "mechanism_benchmark_range_class_summary.csv", index=False)

    manifest = {
        "benchmark": "scDesign3/reference-style mechanism-aware CCC benchmark",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": {
            key: {
                "tissue": value["tissue"],
                "replicate_dir": value["replicate_dir"],
                "methods": list(value["predictions"]),
            }
            for key, value in DATASETS.items()
        },
        "evaluation_contract": {
            "shared_candidate_universe": True,
            "truth_level": "candidate sender-receiver-LR edge",
            "mechanisms": [
                "juxtacrine_contact",
                "short_range_paracrine",
                "medium_range_paracrine",
                "long_range_chemokine_niche",
            ],
            "stress_decoys": [
                "wrong_range_false_positive",
                "wrong_niche_decoy",
                "too_close_for_long_range_lr",
            ],
            "threshold_quantile": float(args.threshold_quantile),
            "top_k": int(args.top_k),
        },
        "outputs": {
            "detail": "mechanism_benchmark_detail.csv",
            "summary": "mechanism_benchmark_summary.csv",
            "status": "mechanism_benchmark_status.csv",
            "challenge_metrics": "mechanism_benchmark_challenge_metrics.csv",
            "truth_contract": "mechanism_benchmark_truth_contract.csv",
            "by_mechanism": "mechanism_benchmark_by_mechanism.csv",
            "range_class_summary": "mechanism_benchmark_range_class_summary.csv",
        },
    }
    (output_dir / "mechanism_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nWrote mechanism benchmark to {output_dir}")


def _read_prediction(path: Path, method: str) -> pd.DataFrame:
    prediction = pd.read_csv(path)
    if "candidate_id" not in prediction.columns or "score" not in prediction.columns:
        raise ValueError(f"Prediction table missing candidate_id/score: {path}")
    out = prediction[["candidate_id", "score"]].copy()
    out["method"] = method
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out[["candidate_id", "method", "score"]]


def _challenge_metrics(
    candidates: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    threshold_quantile: float,
) -> dict:
    merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner")
    score = _minmax(merged["score"].to_numpy(dtype=float))
    selected = score >= float(np.quantile(score, threshold_quantile))
    truth = merged["ground_truth"].astype(int).to_numpy() == 1
    wrong_range = merged.get("wrong_range_false_positive", False)
    wrong_range = np.asarray(wrong_range, dtype=bool)
    niche_ok = merged.get("niche_rule_satisfied", True)
    niche_ok = np.asarray(niche_ok, dtype=bool)
    range_ok = merged.get("range_rule_satisfied", True)
    range_ok = np.asarray(range_ok, dtype=bool)
    false_high = merged.get("false_positive_high_expression", False)
    false_high = np.asarray(false_high, dtype=bool)
    too_close_long = (
        merged.get("range_violation_type", "").astype(str).eq("too_close_for_long_range_lr").to_numpy()
        if "range_violation_type" in merged
        else np.zeros(len(merged), dtype=bool)
    )
    wrong_niche = range_ok & (~niche_ok) & (~truth)
    selected_count = int(selected.sum())
    return {
        "n_scored_candidates": int(len(merged)),
        "n_selected_top_quantile": selected_count,
        "selected_truth_rate": _safe_divide(int((selected & truth).sum()), selected_count),
        "selected_empirical_fdr": _safe_divide(int((selected & (~truth)).sum()), selected_count),
        "wrong_range_call_rate": _safe_divide(int((selected & wrong_range).sum()), int(wrong_range.sum())),
        "wrong_range_fraction_among_calls": _safe_divide(int((selected & wrong_range).sum()), selected_count),
        "wrong_niche_call_rate": _safe_divide(int((selected & wrong_niche).sum()), int(wrong_niche.sum())),
        "wrong_niche_fraction_among_calls": _safe_divide(int((selected & wrong_niche).sum()), selected_count),
        "too_close_long_range_call_rate": _safe_divide(
            int((selected & too_close_long).sum()),
            int(too_close_long.sum()),
        ),
        "high_expression_decoy_call_rate": _safe_divide(
            int((selected & false_high).sum()),
            int(false_high.sum()),
        ),
    }


def _truth_contract_rows(dataset_id: str, tissue: str, candidates: pd.DataFrame) -> list[dict]:
    rows = []
    for mechanism, group in candidates.groupby("mechanism", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "tissue": tissue,
                "mechanism": mechanism,
                "n_candidates": int(len(group)),
                "n_positive": int(group["ground_truth"].astype(int).sum()),
                "positive_rate": float(group["ground_truth"].astype(int).mean()),
                "n_wrong_range_decoys": int(group["wrong_range_false_positive"].astype(bool).sum())
                if "wrong_range_false_positive" in group
                else 0,
                "n_wrong_niche_decoys": int(
                    (
                        group["range_rule_satisfied"].astype(bool)
                        & (~group["niche_rule_satisfied"].astype(bool))
                        & (group["ground_truth"].astype(int) == 0)
                    ).sum()
                )
                if {"range_rule_satisfied", "niche_rule_satisfied"}.issubset(group.columns)
                else 0,
                "median_positive_distance": _median_or_nan(
                    group.loc[group["ground_truth"].astype(int) == 1, "distance"]
                ),
                "median_negative_distance": _median_or_nan(
                    group.loc[group["ground_truth"].astype(int) == 0, "distance"]
                ),
            }
        )
    return rows


def _summary(detail: pd.DataFrame, challenge: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metrics = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
    ]
    available = [column for column in metrics if column in detail.columns]
    summary = (
        detail.groupby("method", sort=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    if not challenge.empty:
        challenge_metrics = [
            "wrong_range_call_rate",
            "wrong_range_fraction_among_calls",
            "wrong_niche_call_rate",
            "wrong_niche_fraction_among_calls",
            "too_close_long_range_call_rate",
        ]
        challenge_summary = (
            challenge.groupby("method", sort=False)[challenge_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary = summary.merge(challenge_summary, on="method", how="left")
    summary["n_datasets"] = detail.groupby("method")["dataset_id"].nunique().reindex(summary["method"]).to_numpy()
    summary["aupr_rank"] = summary["aupr"].rank(ascending=False, method="min").astype("Int64")
    summary["fdr_rank"] = summary["empirical_fdr_at_threshold"].rank(ascending=True, method="min").astype("Int64")
    return _sort_methods(summary)


def _range_class_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()
    metric_columns = [
        "auroc",
        "aupr",
        "top100_precision",
        "recall_top5pct",
        "fdr_top5pct",
        "wrong_range_rate_top5pct",
        "too_close_long_range_rate_top5pct",
        "distance_median_error",
        "distance_profile_emd",
    ]
    available = [column for column in metric_columns if column in mechanism.columns]
    summary = (
        mechanism.groupby(["method", "range_class"], sort=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    return _sort_methods(summary)


def _sort_methods(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "method" not in table.columns:
        return table
    out = table.copy()
    out["_method_order"] = out["method"].map(METHOD_ORDER).fillna(999)
    sort_cols = [column for column in ("dataset_id", "_method_order", "method") if column in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").drop(columns="_method_order")
    return out.reset_index(drop=True)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _median_or_nan(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").median())


if __name__ == "__main__":
    main()
