#!/usr/bin/env python
"""Create an execution matrix for the replicated tuned baseline panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/experiments/phase2_revised_goal2_replicated_tuned_baseline_jobs.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal2_replicated_tuned_baseline_jobs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    power = pd.read_csv(PROJECT_ROOT / config["inputs"]["power_plan_overall"])
    target_replicates = int(power.iloc[0]["target_min_replicates"])
    current_replicates = int(power.iloc[0]["current_replicates"])
    additional_replicates = int(power.iloc[0]["additional_replicates_needed"])

    dataset_command = _dataset_command(config)
    replicate_table = _replicate_table(config)
    method_jobs = _method_jobs(config, replicate_table)
    execution_plan = _execution_plan(config, target_replicates, current_replicates, additional_replicates, dataset_command, replicate_table)
    status = _status_table(method_jobs)
    claims = _claims(config, target_replicates, additional_replicates, method_jobs)

    execution_plan.to_csv(output_dir / "goal2_replicated_tuned_execution_plan.csv", index=False)
    replicate_table.to_csv(output_dir / "goal2_replicated_tuned_replicate_manifest.csv", index=False)
    method_jobs.to_csv(output_dir / "goal2_replicated_tuned_method_jobs.csv", index=False)
    status.to_csv(output_dir / "goal2_replicated_tuned_job_status_template.csv", index=False)
    claims.to_csv(output_dir / "goal2_replicated_tuned_job_claims.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_replicated_tuned_job_matrix",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_status": _evidence_status(replicate_table),
        "target_matched_replicates": target_replicates,
        "current_matched_replicates": current_replicates,
        "additional_replicates_needed": additional_replicates,
        "generated_matched_replicates": int((replicate_table["planned_status"].astype(str) == "generated").sum()),
        "method_jobs": int(len(method_jobs)),
        "outputs": {
            "execution_plan": str((output_dir / "goal2_replicated_tuned_execution_plan.csv").relative_to(PROJECT_ROOT)),
            "replicate_manifest": str((output_dir / "goal2_replicated_tuned_replicate_manifest.csv").relative_to(PROJECT_ROOT)),
            "method_jobs": str((output_dir / "goal2_replicated_tuned_method_jobs.csv").relative_to(PROJECT_ROOT)),
            "status_template": str((output_dir / "goal2_replicated_tuned_job_status_template.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_replicated_tuned_job_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This creates an execution matrix only; it does not generate datasets or run baselines.",
            "Final claims require completed matched replicate outputs and rerun endpoint/paired analyses.",
        ],
    }
    (output_dir / "goal2_replicated_tuned_job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(execution_plan.to_string(index=False))
    print(method_jobs.head(20).to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote replicated tuned baseline job matrix to {output_dir}")


def _dataset_command(config: dict) -> str:
    dataset = config["dataset_generation"]
    datasets = " ".join(dataset["datasets"])
    return (
        f"conda run -n Benchmark python {dataset['script']} "
        f"--config {config['inputs']['real_background_config']} "
        f"--output-dir {dataset['output_dir']} "
        f"--replicates {int(dataset['replicates_per_dataset'])} "
        f"--n-spots {int(dataset['n_spots'])} "
        f"--seed {int(dataset['seed'])} "
        f"--datasets {datasets} "
        f"--clean"
    )


def _replicate_table(config: dict) -> pd.DataFrame:
    dataset = config["dataset_generation"]
    status_lookup = _dataset_status_lookup(dataset["output_dir"])
    rows = []
    for dataset_key in dataset["datasets"]:
        dataset_name = _dataset_name(dataset_key)
        for rep in range(int(dataset["replicates_per_dataset"])):
            replicate_name = f"{dataset_name}_replicate_{rep:02d}"
            expected_path = f"{dataset['output_dir']}/{replicate_name}"
            path = PROJECT_ROOT / expected_path
            status = "generated" if path.exists() and status_lookup.get((dataset_key, rep)) == "generated" else "pending_generation"
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "replicate_id": rep,
                    "replicate_name": replicate_name,
                    "expected_path": expected_path,
                    "planned_status": status,
                }
            )
    return pd.DataFrame(rows)


def _method_jobs(config: dict, replicate_table: pd.DataFrame) -> pd.DataFrame:
    baseline = config["baseline_panel"]
    methods = list(baseline["metric_methods"]) + list(baseline["status_methods"])
    rows = []
    for row in replicate_table.itertuples(index=False):
        for method in methods:
            method_type = "metric" if method in baseline["metric_methods"] else "status_only"
            rows.append(
                {
                    "job_id": f"{row.replicate_name}__{method}",
                    "dataset_key": row.dataset_key,
                    "replicate_name": row.replicate_name,
                    "replicate_path": row.expected_path,
                    "method": method,
                    "method_type": method_type,
                    "command": _baseline_command(config, row.expected_path, row.replicate_name, method),
                    "expected_output_dir": f"{baseline['output_dir']}/method_batches/{row.replicate_name}/{method}",
                    "planned_status": "pending_baseline_run" if method_type == "metric" else "pending_status_record",
                }
            )
    return pd.DataFrame(rows)


def _baseline_command(config: dict, replicate_path: str, replicate_name: str, method: str) -> str:
    baseline = config["baseline_panel"]
    return (
        f"conda run -n Benchmark python {baseline['script']} "
        f"--config {config['inputs']['tuned_baseline_config']} "
        f"--output-dir {baseline['output_dir']}/method_batches/{Path(replicate_path).name}/{method} "
        f"--dataset-root {replicate_path} "
        f"--dataset-name {replicate_name} "
        f"--method {method}"
    )


def _execution_plan(
    config: dict,
    target_replicates: int,
    current_replicates: int,
    additional_replicates: int,
    dataset_command: str,
    replicate_table: pd.DataFrame,
) -> pd.DataFrame:
    baseline = config["baseline_panel"]
    generated = int((replicate_table["planned_status"].astype(str) == "generated").sum())
    dataset_status = "completed" if generated >= target_replicates else "planned_not_run"
    return pd.DataFrame(
        [
            {
                "step": "generate_replicated_real_background_datasets",
                "status": dataset_status,
                "command": dataset_command,
                "expected_output": config["dataset_generation"]["output_dir"],
                "claim_level": "execution_plan_not_run_results",
            },
            {
                "step": "run_metric_baselines_per_replicate",
                "status": "planned_not_run",
                "command": "see goal2_replicated_tuned_method_jobs.csv",
                "expected_output": baseline["output_dir"],
                "claim_level": "execution_plan_not_run_results",
            },
            {
                "step": "rerun_frozen_endpoint_policy_and_paired_diagnostics",
                "status": "planned_not_run",
                "command": "rerun endpoint, guardrail sensitivity, paired diagnostics, and power/audit scripts on completed replicated panel",
                "expected_output": "updated Goal2/Goal6 source tables",
                "claim_level": "execution_plan_not_run_results",
            },
            {
                "step": "replicate_target",
                "status": "planned_not_run",
                "command": f"target={target_replicates}; current={current_replicates}; additional_needed={additional_replicates}",
                "expected_output": "matched replicate-complete baseline panel",
                "claim_level": "execution_plan_not_run_results",
            },
        ]
    )


def _status_table(method_jobs: pd.DataFrame) -> pd.DataFrame:
    return method_jobs[
        [
            "job_id",
            "dataset_key",
            "replicate_name",
            "method",
            "method_type",
            "planned_status",
            "expected_output_dir",
        ]
    ].rename(columns={"planned_status": "status"})


def _claims(config: dict, target_replicates: int, additional_replicates: int, method_jobs: pd.DataFrame) -> pd.DataFrame:
    replicate_table = _replicate_table(config)
    generated = int((replicate_table["planned_status"].astype(str) == "generated").sum())
    return pd.DataFrame(
        [
            {
                "claim_id": "replicated_tuned_job_matrix_created",
                "claim_level": "execution_plan_not_run_results",
                "supported": True,
                "summary": f"Created job matrix for {target_replicates} matched replicates and {len(method_jobs)} method/status jobs.",
            },
            {
                "claim_id": "replicated_tuned_additional_replicates_planned",
                "claim_level": "execution_plan_not_run_results",
                "supported": True,
                "summary": f"Plan requires {additional_replicates} additional matched replicates beyond the current diagnostic panel.",
            },
            {
                "claim_id": "replicated_tuned_datasets_generated",
                "claim_level": "execution_plan_dataset_generated_no_baseline_results",
                "supported": bool(generated >= target_replicates),
                "summary": f"Generated {generated}/{target_replicates} matched replicated real-background datasets for the tuned baseline panel.",
            },
            {
                "claim_id": "replicated_tuned_no_results_yet",
                "claim_level": "execution_plan_not_run_results",
                "supported": True,
                "summary": "This is an execution plan only; no replicated tuned baseline result is claimed.",
            },
        ]
    )


def _dataset_status_lookup(output_dir: str) -> dict[tuple[str, int], str]:
    status_path = PROJECT_ROOT / output_dir / "formal_dataset_status.csv"
    if not status_path.exists():
        return {}
    table = pd.read_csv(status_path)
    lookup: dict[tuple[str, int], str] = {}
    for row in table.itertuples(index=False):
        try:
            replicate = int(getattr(row, "replicate"))
        except (TypeError, ValueError):
            continue
        lookup[(str(getattr(row, "dataset_key")), replicate)] = str(getattr(row, "status"))
    return lookup


def _evidence_status(replicate_table: pd.DataFrame) -> str:
    if (replicate_table["planned_status"].astype(str) == "generated").all():
        return "datasets_generated_baselines_not_run"
    return "execution_plan_not_run_results"


def _dataset_name(dataset_key: str) -> str:
    mapping = {
        "lymph_node": "human_lymph_node_visium",
        "tonsil": "human_tonsil_slide_tags",
        "melanoma": "human_melanoma_multiome",
    }
    return mapping.get(dataset_key, dataset_key)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
