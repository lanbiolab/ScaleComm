#!/usr/bin/env python
"""Run replicated tuned baseline jobs from the Goal 2 job matrix.

The runner is intentionally resumable: each job writes a log row and the
combined status table is refreshed after every job. Failed jobs are recorded and
do not erase completed work from earlier runs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JOBS = "outputs/phase2_revised/goal2_replicated_tuned_baseline_jobs/goal2_replicated_tuned_method_jobs.csv"
DEFAULT_OUTPUT = "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", default=DEFAULT_JOBS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-key", action="append", dest="dataset_keys")
    parser.add_argument("--replicate-name", action="append", dest="replicate_names")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--method-type", choices=["metric", "status_only"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun jobs even when completed outputs exist.")
    args = parser.parse_args()

    jobs_path = PROJECT_ROOT / args.jobs_csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(jobs_path)
    selected = _select_jobs(jobs, args)
    if args.limit > 0:
        selected = selected.head(args.limit).copy()

    status_path = output_dir / "goal2_replicated_tuned_job_status.csv"
    log_path = output_dir / "goal2_replicated_tuned_job_run_log.jsonl"
    manifest_path = output_dir / "goal2_replicated_tuned_run_manifest.json"

    status = _initial_status(jobs, status_path)
    run_rows = []
    for job in selected.itertuples(index=False):
        job_id = str(job.job_id)
        expected_output_dir = PROJECT_ROOT / str(job.expected_output_dir)
        if not args.force and _job_completed(expected_output_dir, str(job.method_type)):
            row = _row(job, "completed_existing", 0, 0.0, "completed output already exists")
            status = _update_status(status, row)
            _append_jsonl(log_path, row)
            run_rows.append(row)
            continue

        if args.dry_run:
            row = _row(job, "dry_run", 0, 0.0, str(job.command))
            status = _update_status(status, row)
            _append_jsonl(log_path, row)
            run_rows.append(row)
            continue

        expected_output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        completed = subprocess.run(
            shlex.split(str(job.command)),
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.monotonic() - started
        (expected_output_dir / "job_stdout.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode == 0 and _job_completed(expected_output_dir, str(job.method_type)):
            state = "completed"
            detail = "completed and expected output detected"
        elif completed.returncode == 0:
            state = "completed_missing_expected_output"
            detail = "command returned zero but expected status/metrics output was not found"
        else:
            state = "failed"
            detail = _last_lines(completed.stdout)
        row = _row(job, state, completed.returncode, elapsed, detail)
        status = _update_status(status, row)
        status.to_csv(status_path, index=False)
        _append_jsonl(log_path, row)
        run_rows.append(row)

    status.to_csv(status_path, index=False)
    summary = _summary(status)
    summary.to_csv(output_dir / "goal2_replicated_tuned_job_status_summary.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_replicated_tuned_baseline_job_runner",
        "jobs_csv": str(jobs_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "selected_jobs": int(len(selected)),
        "run_or_recorded_jobs": int(len(run_rows)),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "status_counts": summary.to_dict(orient="records"),
        "outputs": {
            "status": str(status_path.relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_replicated_tuned_job_status_summary.csv").relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote replicated tuned job status to {status_path}")


def _select_jobs(jobs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = jobs.copy()
    if args.dataset_keys:
        selected = selected[selected["dataset_key"].astype(str).isin(set(args.dataset_keys))]
    if args.replicate_names:
        selected = selected[selected["replicate_name"].astype(str).isin(set(args.replicate_names))]
    if args.methods:
        selected = selected[selected["method"].astype(str).isin(set(args.methods))]
    if args.method_type:
        selected = selected[selected["method_type"].astype(str) == args.method_type]
    return selected.reset_index(drop=True)


def _initial_status(jobs: pd.DataFrame, status_path: Path) -> pd.DataFrame:
    if status_path.exists():
        return pd.read_csv(status_path)
    status = jobs[
        [
            "job_id",
            "dataset_key",
            "replicate_name",
            "replicate_path",
            "method",
            "method_type",
            "expected_output_dir",
        ]
    ].copy()
    status["status"] = jobs["planned_status"]
    status["returncode"] = pd.NA
    status["elapsed_seconds"] = pd.NA
    status["last_updated_utc"] = ""
    status["detail"] = ""
    return status


def _job_completed(expected_output_dir: Path, method_type: str = "metric") -> bool:
    metrics = expected_output_dir / "goal2_formal_tuned_small_metrics.csv"
    status = expected_output_dir / "goal2_formal_tuned_small_status.csv"
    if not status.exists():
        return False
    if method_type == "status_only":
        try:
            status_table = pd.read_csv(status)
        except pd.errors.EmptyDataError:
            return False
        return bool(len(status_table) >= 1 and status_table["status"].astype(str).str.len().gt(0).all())
    if metrics.exists():
        try:
            return metrics.stat().st_size > 0 and pd.read_csv(metrics).shape[0] >= 1
        except pd.errors.EmptyDataError:
            return False
    try:
        status_table = pd.read_csv(status)
    except pd.errors.EmptyDataError:
        return False
    return bool(len(status_table) >= 1 and status_table["status"].astype(str).str.len().gt(0).all())


def _row(job: object, state: str, returncode: int, elapsed: float, detail: str) -> dict:
    return {
        "job_id": str(job.job_id),
        "dataset_key": str(job.dataset_key),
        "replicate_name": str(job.replicate_name),
        "replicate_path": str(job.replicate_path),
        "method": str(job.method),
        "method_type": str(job.method_type),
        "expected_output_dir": str(job.expected_output_dir),
        "status": state,
        "returncode": int(returncode),
        "elapsed_seconds": round(float(elapsed), 3),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "detail": str(detail).replace("\n", " ")[:2000],
    }


def _update_status(status: pd.DataFrame, row: dict) -> pd.DataFrame:
    mask = status["job_id"].astype(str) == str(row["job_id"])
    if mask.any():
        for key, value in row.items():
            if key in status.columns:
                status.loc[mask, key] = value
    else:
        status = pd.concat([status, pd.DataFrame([row])], ignore_index=True)
    return status


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _last_lines(text: str, n: int = 12) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _summary(status: pd.DataFrame) -> pd.DataFrame:
    return (
        status.groupby(["method_type", "method", "status"], dropna=False)
        .size()
        .reset_index(name="n_jobs")
        .sort_values(["method_type", "method", "status"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
