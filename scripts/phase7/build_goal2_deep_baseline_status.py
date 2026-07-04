#!/usr/bin/env python
"""Build Phase 7 Goal 2 deep baseline closure status table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    goal_dir = PROJECT_ROOT / config["goals"]["goal2_deep_baseline_closure"]["output_dir"]
    goal_dir.mkdir(parents=True, exist_ok=True)

    deeptalk_summary = _summarize_deeptalk()
    deeptalk_summary.to_csv(goal_dir / "phase7_deeptalk_formal_run_summary.csv", index=False)

    status = _build_status_table(deeptalk_summary)
    status_path = goal_dir / "phase7_deep_baseline_status_table.csv"
    status.to_csv(status_path, index=False)

    logs_dir = goal_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _write_failure_logs(logs_dir, status)

    manifest = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "status_table": str(status_path.relative_to(PROJECT_ROOT)),
        "deeptalk_summary": str((goal_dir / "phase7_deeptalk_formal_run_summary.csv").relative_to(PROJECT_ROOT)),
        "logs_dir": str(logs_dir.relative_to(PROJECT_ROOT)),
        "claim": "Deep spatial CCC baselines were audited; DeepTalk has reproducible input/status but lacks required trained predict_ccc model outputs in the current local environment.",
        "limitation": "Status/failure records do not imply biological inferiority of incompatible baselines.",
    }
    (goal_dir / "phase7_goal2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(status.to_string(index=False))


def _summarize_deeptalk() -> pd.DataFrame:
    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel/real_background_v2_formal/deeptalk"
    rows = []
    for status_path in sorted(root.glob("*__deeptalk_status.csv")):
        dataset = status_path.name.replace("__deeptalk_status.csv", "")
        status = pd.read_csv(status_path)
        pred_path = root / f"{dataset}__deeptalk_predictions.csv"
        predictions = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
        n_nonzero = int((pd.to_numeric(predictions.get("score", pd.Series(dtype=float)), errors="coerce").fillna(0.0) != 0.0).sum()) if not predictions.empty else 0
        rows.append(
            {
                "dataset": dataset,
                "n_lr_pairs": int(status["lr_id"].nunique()) if "lr_id" in status else 0,
                "n_candidate_edges": int(status["n_candidates"].sum()) if "n_candidates" in status else int(len(predictions)),
                "status_values": ";".join(sorted(status["status"].astype(str).unique())) if "status" in status else "",
                "prediction_path": str(pred_path.relative_to(PROJECT_ROOT)) if pred_path.exists() else "",
                "n_prediction_rows": int(len(predictions)),
                "n_nonzero_prediction_scores": n_nonzero,
                "usable_for_metric_panel": bool(n_nonzero > 0 and "completed" in set(status.get("status", []))),
            }
        )
    return pd.DataFrame(rows)


def _build_status_table(deeptalk_summary: pd.DataFrame) -> pd.DataFrame:
    deeptalk_datasets = ";".join(deeptalk_summary["dataset"].astype(str).tolist()) if not deeptalk_summary.empty else ""
    deeptalk_status_values = ";".join(sorted(set(";".join(deeptalk_summary["status_values"].astype(str)).split(";")))) if not deeptalk_summary.empty else "not_run"
    deeptalk_outputs = ";".join(deeptalk_summary["prediction_path"].astype(str).tolist()) if not deeptalk_summary.empty else ""
    rows = [
        {
            "method": "DeepTalk",
            "code_availability": "adapter_available_locally",
            "version_or_source": "local benchmark adapter using scripts/benchmark/prepare_deeptalk_inputs.py and run_deeptalk.py",
            "dependency_reproducibility": "input_preparation_reproducible;predict_ccc_model_required_missing",
            "input_compatibility": "candidate_interactions/expression/coordinates converted to DeepTalk per-LR input directories",
            "output_level": "candidate_edge_when_predict_ccc_exists;current_outputs_are_zero_placeholders",
            "lr_pair_ranking_support": "not_usable_currently",
            "cell_or_spot_level_support": "candidate_edge_projection_prepared",
            "runtime_memory_status": "input_preparation_completed_for_lymph_node_tonsil_melanoma",
            "final_status": "status_only_model_required_missing_predict_ccc",
            "exact_failure_reason": f"DeepTalk per-LR statuses: {deeptalk_status_values}; datasets: {deeptalk_datasets}; nonzero prediction scores absent.",
            "output_path": deeptalk_outputs,
        },
        {
            "method": "HoloNet",
            "code_availability": "not_installed_in_current_benchmark_env",
            "version_or_source": "not pinned",
            "dependency_reproducibility": "not_audited",
            "input_compatibility": "requires separate audit of higher-order/network inputs and output projection",
            "output_level": "unknown_or_not_formal_candidate_edge",
            "lr_pair_ranking_support": "not_audited",
            "cell_or_spot_level_support": "not_audited",
            "runtime_memory_status": "not_run",
            "final_status": "status_only_not_audited",
            "exact_failure_reason": "No local formal installation/run record in current Phase 7 workspace.",
            "output_path": "",
        },
        {
            "method": "stMLnet",
            "code_availability": "not_installed_in_current_benchmark_env",
            "version_or_source": "not pinned",
            "dependency_reproducibility": "not_audited",
            "input_compatibility": "requires separate audit of multi-layer network inputs and LR-pair output",
            "output_level": "unknown_or_not_formal_candidate_edge",
            "lr_pair_ranking_support": "not_audited",
            "cell_or_spot_level_support": "not_audited",
            "runtime_memory_status": "not_run",
            "final_status": "status_only_not_audited",
            "exact_failure_reason": "No local formal installation/run record in current Phase 7 workspace.",
            "output_path": "",
        },
        {
            "method": "CytoSignal",
            "code_availability": "not_installed_in_current_benchmark_env",
            "version_or_source": "not pinned",
            "dependency_reproducibility": "not_audited",
            "input_compatibility": "requires separate audit of expected expression/LR output level",
            "output_level": "unknown_or_not_formal_candidate_edge",
            "lr_pair_ranking_support": "not_audited",
            "cell_or_spot_level_support": "not_audited",
            "runtime_memory_status": "not_run",
            "final_status": "status_only_not_audited",
            "exact_failure_reason": "No local formal installation/run record in current Phase 7 workspace.",
            "output_path": "",
        },
        {
            "method": "NICHES",
            "code_availability": "status_from_phase5",
            "version_or_source": "configs/baselines/phase1_baselines.yaml and Phase 5 methods support",
            "dependency_reproducibility": "blocked_or_not_formal_candidate_edge",
            "input_compatibility": "not formalized for candidate-edge metric panel in current retained baseline set",
            "output_level": "blocked_or_not_formal_candidate_edge",
            "lr_pair_ranking_support": "not_usable_currently",
            "cell_or_spot_level_support": "not_usable_currently",
            "runtime_memory_status": "not_run_in_formal_phase7",
            "final_status": "status_only_dependency_or_output_incompatibility",
            "exact_failure_reason": "Phase 5 support table records NICHES as status-only because local R dependency chain was not reproducible.",
            "output_path": "outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_baseline_status_table.csv",
        },
    ]
    return pd.DataFrame(rows)


def _write_failure_logs(logs_dir: Path, status: pd.DataFrame) -> None:
    for row in status.itertuples(index=False):
        method = str(row.method).lower().replace("+", "plus")
        path = logs_dir / f"{method}_status.txt"
        path.write_text(
            "\n".join(
                [
                    f"method: {row.method}",
                    f"final_status: {row.final_status}",
                    f"input_compatibility: {row.input_compatibility}",
                    f"output_level: {row.output_level}",
                    f"exact_failure_reason: {row.exact_failure_reason}",
                    f"output_path: {row.output_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
