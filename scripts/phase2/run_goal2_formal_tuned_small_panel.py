#!/usr/bin/env python
"""Run a small tuned baseline panel on formal real-background v2 datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_baseline_comparison import PENDING_METHODS, run_baselines_on_dataset


DEFAULT_CONFIG = "configs/experiments/phase2_revised_goal2_formal_tuned_small_panel.yaml"


@dataclass
class ParameterRow:
    method: str
    parameter: str
    value: str
    evidence_role: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--dataset-root", default="", help="Override dataset root or a single replicate directory.")
    parser.add_argument("--dataset-name", default="", help="Override dataset label written to output tables.")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal2_formal_tuned_small_panel"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config["dataset"]
    dataset_name = str(args.dataset_name or dataset_cfg["name"])
    dataset_root = PROJECT_ROOT / (args.dataset_root or dataset_cfg["dataset_root"])
    configured_metric_methods = tuple(config["methods"]["metric_panel"])
    configured_status_methods = tuple(config["methods"].get("status_only", []))
    if args.methods:
        requested_methods = tuple(args.methods)
        metric_methods = tuple(method for method in requested_methods if method in configured_metric_methods)
        status_methods = tuple(method for method in requested_methods if method in configured_status_methods)
        unknown_methods = tuple(method for method in requested_methods if method not in configured_metric_methods and method not in configured_status_methods)
        methods = metric_methods + status_methods + unknown_methods
    else:
        metric_methods = configured_metric_methods
        status_methods = tuple(method for method in configured_status_methods if method not in metric_methods)
        methods = metric_methods + status_methods

    kwargs = _runner_kwargs(config)
    metrics, status = _run_methods_isolated(dataset_root, output_dir, methods, kwargs)
    if not metrics.empty and status_methods:
        metrics = metrics[~metrics["method"].astype(str).isin(set(status_methods))].copy()
    if not metrics.empty:
        metrics = metrics.copy()
        metrics["dataset"] = dataset_name
        metrics["baseline_evidence_level"] = metrics["method"].map(_evidence_level).fillna("candidate_edge_metric")
        metrics["interpretation_note"] = metrics["method"].map(_interpretation_note).fillna("")
        metrics.to_csv(output_dir / "goal2_formal_tuned_small_metrics.csv", index=False)
        _summary(metrics).to_csv(output_dir / "goal2_formal_tuned_small_summary.csv", index=False)
        _claims(metrics, status, config).to_csv(output_dir / "goal2_formal_tuned_small_claims.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "goal2_formal_tuned_small_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "goal2_formal_tuned_small_summary.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "goal2_formal_tuned_small_claims.csv", index=False)

    if not status.empty:
        status = status.copy()
        status["dataset"] = dataset_name
        status["baseline_evidence_level"] = status["method"].map(_evidence_level).fillna("status_only")
        status["interpretation_note"] = status["method"].map(_interpretation_note).fillna("")
    status.to_csv(output_dir / "goal2_formal_tuned_small_status.csv", index=False)

    parameters = _parameter_table(config)
    parameters.to_csv(output_dir / "goal2_formal_tuned_small_parameters.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_formal_tuned_small_panel",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "dataset": dataset_name,
        "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT)),
        "methods": list(methods),
        "methods_with_metrics": sorted(metrics["method"].astype(str).unique()) if not metrics.empty else [],
        "methods_with_status": sorted(status["method"].astype(str).unique()) if not status.empty else [],
        "evidence_status": "formal_tuned_small_panel_not_final_replicated_baseline",
        "status_scope": "tuned small real-background panel; not final replicated manuscript baseline",
        "pending_methods": PENDING_METHODS,
        "outputs": {
            "metrics": str((output_dir / "goal2_formal_tuned_small_metrics.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_formal_tuned_small_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_formal_tuned_small_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_formal_tuned_small_status.csv").relative_to(PROJECT_ROOT)),
            "parameters": str((output_dir / "goal2_formal_tuned_small_parameters.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Small formal real-background panel only: lymph node, tonsil, and melanoma one replicate each.",
            "CellChat and SpaTalk are cell-type/LR scores projected to candidate edges.",
            "CellNEST is run with a short tuned-small setting, not a large seed sweep.",
            "DeepTalk remains model-required unless LR-specific predict_ccc outputs are supplied.",
            "NICHES remains dependency-blocked in the local R environment.",
        ],
    }
    (output_dir / "goal2_formal_tuned_small_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if metrics.empty:
        print("No tuned-small metric rows were generated.")
    else:
        cols = [
            "replicate",
            "method",
            "aupr",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
            "false_positive_high_expression_rate_at_threshold",
        ]
        print(metrics[[column for column in cols if column in metrics]].to_string(index=False))
    print(f"Wrote formal tuned-small baseline panel to {output_dir}")


def _run_methods_isolated(
    dataset_root: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    kwargs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_tables: list[pd.DataFrame] = []
    status_tables: list[pd.DataFrame] = []
    for method in methods:
        method_out = output_dir / "method_runs" / method
        try:
            metrics = run_baselines_on_dataset(dataset_root, method_out, methods=(method,), **kwargs)
            if not metrics.empty:
                metric_tables.append(metrics)
            status_path = method_out / "baseline_status.csv"
            if status_path.exists():
                status_tables.append(pd.read_csv(status_path))
            else:
                status_tables.append(
                    pd.DataFrame(
                        [
                            {
                                "method": method,
                                "status": "missing_status",
                                "detail": "run_baselines_on_dataset returned without baseline_status.csv",
                            }
                        ]
                    )
                )
        except Exception as exc:  # noqa: BLE001 - per-method failure is benchmark evidence.
            status_tables.append(
                pd.DataFrame(
                    [
                        {
                            "method": method,
                            "status": "failed",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    ]
                )
            )
    metrics = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    status = pd.concat(status_tables, ignore_index=True) if status_tables else pd.DataFrame()
    return metrics, status


def _runner_kwargs(config: dict) -> dict:
    params = config["tuned_small_parameters"]
    spatialself = params["spatialself"]
    return {
        "cot_nitermax": int(params["commot"]["cot_nitermax"]),
        "cellchat_nboot": int(params["cellchat"]["nboot"]),
        "spatialself_epochs": int(spatialself["epochs"]),
        "spatialself_k_nn": int(spatialself["k_nn"]),
        "spatialself_max_train_samples": int(spatialself["max_train_samples"]),
        "spatialself_pseudo_label_strategy": str(spatialself["pseudo_label_strategy"]),
        "spatialself_weak_positive_fraction": float(spatialself["weak_positive_fraction"]),
        "spatialself_ranking_loss_weight": float(spatialself["ranking_loss_weight"]),
        "spatialself_ranking_margin": float(spatialself["ranking_margin"]),
        "spatialself_lr_cap_positive": int(spatialself["lr_cap_positive"]),
        "spatialself_lr_cap_negative": int(spatialself["lr_cap_negative"]),
        "spatialself_no_lr_embedding": False,
        "spatialself_no_distance": False,
        "spatialself_no_spatial_graph": False,
        "spatialself_no_expression_features": False,
        "cellnest_num_epoch": int(params["cellnest"]["num_epoch"]),
        "cellnest_hidden": int(params["cellnest"]["hidden"]),
        "deeptalk_input_root": None,
        "spatalk_per_num": int(params["spatalk"]["per_num"]),
        "spatalk_min_pairs": int(params["spatalk"]["min_pairs"]),
        "spatalk_use_synthetic_pathways": bool(params["spatalk"]["use_synthetic_pathways"]),
    }


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in columns if column in metrics]
    summary = metrics.groupby(["dataset", "method"], sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _claims(metrics: pd.DataFrame, status: pd.DataFrame, config: dict) -> pd.DataFrame:
    guardrail = config.get("tradeoff_guardrail", {})
    max_fp = float(guardrail.get("max_high_expression_fp_rate", 0.05))
    max_fdr = float(guardrail.get("max_empirical_fdr", 0.65))
    rows = []
    rows.append(_claim_for_group("aggregate_real_background_v2_formal", metrics, status, max_fp, max_fdr))
    for replicate, group in metrics.groupby("replicate", sort=False):
        rows.append(_claim_for_group(str(replicate), group, status, max_fp, max_fdr))
    return pd.DataFrame(rows)


def _claim_for_group(scope: str, metrics: pd.DataFrame, status: pd.DataFrame, max_fp: float, max_fdr: float) -> dict:
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metric_columns if column in metrics]
    scored = metrics.groupby("method", as_index=False)[available].mean(numeric_only=True)
    scored["unconstrained_tradeoff_score"] = (
        scored["weak_signal_recall_at_threshold"].fillna(0.0)
        + 0.25 * scored["aupr"].fillna(0.0)
        - 0.5 * scored["empirical_fdr_at_threshold"].fillna(1.0)
        - 0.5 * scored["false_positive_high_expression_rate_at_threshold"].fillna(1.0)
    )
    constrained = scored[
        (scored["false_positive_high_expression_rate_at_threshold"].fillna(1.0) <= max_fp)
        & (scored["empirical_fdr_at_threshold"].fillna(1.0) <= max_fdr)
    ].copy()
    if constrained.empty:
        constrained = scored.copy()
        constrained["best_tradeoff_status"] = "no_method_met_fp_fdr_guardrail"
    else:
        constrained["best_tradeoff_status"] = "fp_fdr_guardrail_met"
    constrained["constrained_tradeoff_score"] = (
        constrained["weak_signal_recall_at_threshold"].fillna(0.0)
        + 0.25 * constrained["aupr"].fillna(0.0)
        - 0.25 * constrained["empirical_fdr_at_threshold"].fillna(1.0)
        - constrained["false_positive_high_expression_rate_at_threshold"].fillna(1.0)
    )
    best_tradeoff = constrained.sort_values("constrained_tradeoff_score", ascending=False).iloc[0]
    best_unconstrained = scored.sort_values("unconstrained_tradeoff_score", ascending=False).iloc[0]
    best_fp = scored.sort_values("false_positive_high_expression_rate_at_threshold", ascending=True).iloc[0]
    best_weak = scored.sort_values("weak_signal_recall_at_threshold", ascending=False).iloc[0]
    return {
        "scope": scope,
        "claim_level": "formal_tuned_small_panel_not_final_replicated_baseline",
        "max_high_expression_fp_guardrail": max_fp,
        "max_empirical_fdr_guardrail": max_fdr,
        "best_tradeoff_method": best_tradeoff["method"],
        "best_tradeoff_score": best_tradeoff["constrained_tradeoff_score"],
        "best_tradeoff_status": best_tradeoff["best_tradeoff_status"],
        "best_unconstrained_tradeoff_method": best_unconstrained["method"],
        "best_unconstrained_tradeoff_score": best_unconstrained["unconstrained_tradeoff_score"],
        "lowest_high_expression_fp_method": best_fp["method"],
        "lowest_high_expression_fp_rate": best_fp["false_positive_high_expression_rate_at_threshold"],
        "highest_weak_recall_method": best_weak["method"],
        "highest_weak_recall": best_weak["weak_signal_recall_at_threshold"],
        "metric_methods": int(scored["method"].nunique()),
        "metric_rows": int(len(metrics)),
        "status_methods": int(status["method"].nunique()) if not status.empty and "method" in status else 0,
        "spatialself_best_tradeoff": str(best_tradeoff["method"]).startswith("spatialself"),
        "spatialself_lowest_fp": str(best_fp["method"]).startswith("spatialself"),
        "spatialself_highest_weak_recall": str(best_weak["method"]).startswith("spatialself"),
    }


def _parameter_table(config: dict) -> pd.DataFrame:
    rows: list[ParameterRow] = []
    for method, params in config["tuned_small_parameters"].items():
        for name, value in params.items():
            rows.append(
                ParameterRow(
                    method=method,
                    parameter=str(name),
                    value=str(value),
                    evidence_role="tuned_small_not_final_replicated",
                )
            )
    return pd.DataFrame([asdict(row) for row in rows])


def _evidence_level(method: str) -> str:
    mapping = {
        "spatialself_wavelet_edge_prior": "candidate_edge_metric_tuned_small_panel",
        "spatialself_wavelet": "candidate_edge_metric_tuned_small_panel",
        "commot": "candidate_edge_metric_tuned_small_panel",
        "cellchat": "cell_type_pair_projected_tuned_small_panel",
        "spatalk": "cell_type_pair_projected_tuned_small_panel",
        "cellnest": "candidate_edge_tuned_small_panel",
        "deeptalk": "projection_only_model_required_unless_predict_ccc_exists",
        "niches": "optional_blocked_dependency_unreproducible",
    }
    return mapping.get(str(method), "status_only")


def _interpretation_note(method: str) -> str:
    notes = {
        "cellchat": "CellChat nboot=20 tuned-small run; cell-type-pair scores are projected to candidate edges.",
        "spatalk": "SpaTalk per_num=100 tuned-small run; LR scores are projected to candidate edges.",
        "cellnest": "CellNEST epoch=2 hidden=32 tuned-small run; not a final seed sweep.",
        "deeptalk": "DeepTalk requires LR-specific predict_ccc outputs; status evidence until supplied.",
        "niches": "NICHES remains dependency-blocked in the local R chain.",
    }
    return notes.get(str(method), "")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
