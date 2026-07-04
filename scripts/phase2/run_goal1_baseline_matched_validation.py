#!/usr/bin/env python
"""Run revised Phase 2 baseline-matched injected-control validation."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal1_baseline_matched_validation.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prediction-root", default="", help="Override prediction root from config.")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset key from config. Repeatable.")
    parser.add_argument("--method", action="append", dest="methods", help="Method to validate. Repeatable.")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal1_baseline_matched_validation"))
    prediction_root = PROJECT_ROOT / (args.prediction_root or config["prediction_root"])
    method_prediction_roots = {
        str(method): PROJECT_ROOT / str(path)
        for method, path in (config.get("method_prediction_roots") or {}).items()
    }
    methods = tuple(args.methods or config["methods"])
    datasets = {key: value for key, value in config["datasets"].items() if not args.datasets or key in set(args.datasets)}

    method_rows: list[dict] = []
    lr_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    for dataset, dataset_cfg in datasets.items():
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]
        try:
            candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
            lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
            expression = pd.read_csv(replicate_dir / "expression.csv")
        except Exception as exc:  # noqa: BLE001
            status_rows.append({"dataset": dataset, "method": "dataset", "status": "failed", "detail": str(exc)})
            continue

        for method in methods:
            method_root = method_prediction_roots.get(method, prediction_root)
            prediction_dataset = str(dataset_cfg.get("prediction_dataset", dataset))
            prediction_path = _prediction_path(method_root, prediction_dataset, method, replicate_dir.name)
            if not prediction_path.exists():
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "missing_prediction",
                        "detail": str(prediction_path),
                    }
                )
                continue
            try:
                prediction = pd.read_csv(prediction_path)
                annotated = _merge_prediction(candidates, prediction)
                lr_summary = _lr_control_summary(
                    annotated,
                    lr_pairs,
                    expression,
                    dataset=dataset,
                    method=method,
                    prediction_path=prediction_path,
                )
                method_summary = _method_summary(lr_summary, annotated, dataset=dataset, method=method, prediction_path=prediction_path)
                method_rows.append(method_summary)
                lr_rows.append(lr_summary)
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "completed",
                        "detail": "",
                        "prediction_path": str(prediction_path),
                        "n_prediction_rows": int(len(prediction)),
                        "n_unique_scores": int(prediction["score"].nunique()),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "prediction_path": str(prediction_path),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    method_table = pd.DataFrame(method_rows)
    lr_table = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    status_table = pd.DataFrame(status_rows)
    claim_table = _claim_table(method_table, status_table)

    method_table.to_csv(output_dir / "goal1_baseline_matched_method_summary.csv", index=False)
    lr_table.to_csv(output_dir / "goal1_baseline_matched_lr_control_summary.csv", index=False)
    status_table.to_csv(output_dir / "goal1_baseline_matched_status.csv", index=False)
    claim_table.to_csv(output_dir / "goal1_baseline_matched_claims.csv", index=False)
    manifest = {
        "config": args.config,
        "output_dir": str(output_dir),
        "datasets": list(datasets.keys()),
        "methods": list(methods),
        "prediction_root": str(prediction_root),
        "method_prediction_roots": {method: str(path) for method, path in method_prediction_roots.items()},
        "prediction_dataset_overrides": {
            key: str(value.get("prediction_dataset", key)) for key, value in datasets.items()
        },
        "claim_level": config.get("claim_level", "injected_control_receiver_response_proxy_not_manuscript_ready"),
        "method_summary": str(output_dir / "goal1_baseline_matched_method_summary.csv"),
        "lr_control_summary": str(output_dir / "goal1_baseline_matched_lr_control_summary.csv"),
        "status": str(output_dir / "goal1_baseline_matched_status.csv"),
        "claims": str(output_dir / "goal1_baseline_matched_claims.csv"),
        "limitations": [
            "Validation uses injected real-background controls, not unlabeled real-data ground truth.",
            "Receiver-response Spearman is a receptor/receiver-expression proxy until curated LR-target maps are added.",
            "Adapter-light baseline projections are useful for matched screening but final manuscript claims still need tuned baseline panels.",
        ],
    }
    (output_dir / "goal1_baseline_matched_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if method_table.empty:
        print("No Goal 1 validation metrics generated.")
    else:
        cols = [
            "dataset",
            "method",
            "mean_positive_rank",
            "mean_negative_rank",
            "negative_top3_count",
            "mean_positive_receiver_response_proxy_spearman",
            "empirical_fdr_at_q95",
            "high_expression_false_positive_rate_at_q95",
            "rank_validation_pass",
        ]
        print(method_table[cols].to_string(index=False))
    print(f"Wrote revised Goal 1 baseline-matched validation outputs to {output_dir}")


def _prediction_path(prediction_root: Path, dataset: str, method: str, replicate_name: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        filename = f"{replicate_name}__spatialself_wavelet_predictions.csv"
    else:
        filename = f"{replicate_name}__{method}_predictions.csv"
    return prediction_root / dataset / method / filename


def _merge_prediction(candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "score"}
    missing = required - set(prediction.columns)
    if missing:
        raise ValueError(f"Prediction missing columns: {sorted(missing)}")
    merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    if len(merged) != len(candidates):
        raise ValueError(f"Prediction/candidate row mismatch: {len(merged)} != {len(candidates)}")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _lr_control_summary(
    annotated: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    expression: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    prediction_path: Path,
) -> pd.DataFrame:
    lr_meta = lr_pairs.set_index("lr_id")
    rows = []
    for lr_id, group in annotated.groupby("lr_id", sort=False):
        lr = lr_meta.loc[lr_id]
        response = _receiver_response_proxy(group, expression, str(lr["receptor"]))
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "lr_id": lr_id,
                "ligand": lr["ligand"],
                "receptor": lr["receptor"],
                "mechanism": lr["mechanism"],
                "true_lr": bool(lr.get("true_lr", True)) and not bool(lr.get("false_positive_high_expression", False)),
                "weak_signal": bool(lr.get("weak_signal", False)),
                "false_positive_high_expression": bool(lr.get("false_positive_high_expression", False)),
                "n_candidates": int(len(group)),
                "n_true_edges": int(group["ground_truth"].astype(int).sum()) if "ground_truth" in group else 0,
                "mean_score": float(group["score"].mean()),
                "q95_score": float(group["score"].quantile(0.95)),
                "max_score": float(group["score"].max()),
                "receiver_response_proxy_spearman": response["spearman"],
                "receiver_response_proxy_status": response["status"],
                "prediction_path": str(prediction_path),
            }
        )
    table = pd.DataFrame(rows).sort_values(["q95_score", "mean_score"], ascending=False).reset_index(drop=True)
    table["global_rank"] = np.arange(1, len(table) + 1)
    table["positive_control"] = table["true_lr"].astype(bool)
    table["negative_control"] = table["false_positive_high_expression"].astype(bool)
    return table


def _receiver_response_proxy(group: pd.DataFrame, expression: pd.DataFrame, receptor: str) -> dict:
    if receptor not in expression.columns:
        return {"spearman": float("nan"), "status": "receptor_not_observed"}
    expr = expression[["cell_id", receptor]].copy()
    expr[receptor] = pd.to_numeric(expr[receptor], errors="coerce").fillna(0.0)
    comm = group.groupby("receiver_id", sort=False)["score"].mean()
    merged = expr.set_index("cell_id").join(comm.rename("communication"), how="inner")
    if len(merged) < 4:
        return {"spearman": float("nan"), "status": "too_few_receiver_cells"}
    x = merged["communication"].to_numpy(dtype=float)
    y = np.log1p(merged[receptor].to_numpy(dtype=float))
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return {"spearman": float("nan"), "status": "constant_vector"}
    return {"spearman": _spearman(x, y), "status": "receiver_receptor_expression_proxy"}


def _method_summary(lr_summary: pd.DataFrame, annotated: pd.DataFrame, *, dataset: str, method: str, prediction_path: Path) -> dict:
    positives = lr_summary[lr_summary["positive_control"].astype(bool)]
    negatives = lr_summary[lr_summary["negative_control"].astype(bool)]
    threshold = float(annotated["score"].quantile(0.95))
    called = annotated["score"].to_numpy(dtype=float) >= threshold
    truth = annotated["ground_truth"].astype(int).to_numpy()
    fp_high = annotated["false_positive_high_expression"].astype(bool).to_numpy()
    positive_proxy = positives["receiver_response_proxy_spearman"].dropna()
    negative_proxy = negatives["receiver_response_proxy_spearman"].dropna()
    mean_positive_rank = float(positives["global_rank"].mean()) if len(positives) else float("nan")
    mean_negative_rank = float(negatives["global_rank"].mean()) if len(negatives) else float("nan")
    negative_top3 = int((negatives["global_rank"] <= 3).sum()) if len(negatives) else 0
    positive_top3 = int((positives["global_rank"] <= 3).sum()) if len(positives) else 0
    return {
        "dataset": dataset,
        "method": method,
        "prediction_path": str(prediction_path),
        "n_lr_pairs": int(len(lr_summary)),
        "n_positive_controls": int(len(positives)),
        "n_negative_controls": int(len(negatives)),
        "mean_positive_rank": mean_positive_rank,
        "mean_negative_rank": mean_negative_rank,
        "best_positive_rank": float(positives["global_rank"].min()) if len(positives) else float("nan"),
        "best_negative_rank": float(negatives["global_rank"].min()) if len(negatives) else float("nan"),
        "positive_top3_count": positive_top3,
        "negative_top3_count": negative_top3,
        "rank_validation_pass": bool(len(positives) and len(negatives) and mean_positive_rank < mean_negative_rank and negative_top3 == 0),
        "mean_positive_receiver_response_proxy_spearman": float(positive_proxy.mean()) if len(positive_proxy) else float("nan"),
        "mean_negative_receiver_response_proxy_spearman": float(negative_proxy.mean()) if len(negative_proxy) else float("nan"),
        "receiver_response_proxy_level": "receiver_receptor_expression_proxy_not_curated_downstream_targets",
        "empirical_fdr_at_q95": _safe_divide(int((called & (truth == 0)).sum()), int(called.sum())),
        "high_expression_false_positive_rate_at_q95": _safe_divide(int((called & fp_high).sum()), int(fp_high.sum())),
        "q95_threshold": threshold,
    }


def _claim_table(method_table: pd.DataFrame, status_table: pd.DataFrame) -> pd.DataFrame:
    if method_table.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in method_table.groupby("dataset", sort=False):
        best_rank = group.sort_values(["rank_validation_pass", "mean_positive_rank"], ascending=[False, True]).iloc[0]
        best_fp = group.sort_values("high_expression_false_positive_rate_at_q95", ascending=True).iloc[0]
        spatialself = group[group["method"].astype(str).str.startswith("spatialself")]
        commot = group[group["method"].astype(str) == "commot"]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "injected_control_receiver_response_proxy_not_manuscript_ready",
                "best_rank_method": best_rank["method"],
                "best_rank_validation_pass": bool(best_rank["rank_validation_pass"]),
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["high_expression_false_positive_rate_at_q95"],
                "spatialself_beats_commot_on_fp": _spatialself_beats_commot(spatialself, commot, "high_expression_false_positive_rate_at_q95", lower_is_better=True),
                "spatialself_beats_commot_on_rank": _spatialself_beats_commot(spatialself, commot, "mean_positive_rank", lower_is_better=True),
                "completed_methods": int(((status_table["dataset"] == dataset) & (status_table["status"] == "completed")).sum()) if not status_table.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _spatialself_beats_commot(spatialself: pd.DataFrame, commot: pd.DataFrame, column: str, *, lower_is_better: bool) -> bool:
    if spatialself.empty or commot.empty:
        return False
    spatial_value = float(spatialself[column].min() if lower_is_better else spatialself[column].max())
    commot_value = float(commot[column].min() if lower_is_better else commot[column].max())
    return spatial_value <= commot_value if lower_is_better else spatial_value >= commot_value


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
