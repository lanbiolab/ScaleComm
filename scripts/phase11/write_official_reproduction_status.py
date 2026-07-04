#!/usr/bin/env python
"""Write status tables for official COMMOT/CellNEST reproduction runs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pid_running(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    return Path(f"/proc/{pid}").exists()


def latest_epoch(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    epoch = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if line.startswith("Epoch:"):
            token = line.split(",", 1)[0].replace("Epoch:", "").strip()
            try:
                epoch = int(token)
            except ValueError:
                pass
    return epoch


def main() -> None:
    out = PROJECT_ROOT / "outputs/phase11/official_method_reproduction"
    cellnest = out / "cellnest_human_lymph_node"
    commot = out / "commot_human_lymph_node"
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "method": "COMMOT",
            "workflow": "Official README API: read_visium, normalize_total, log1p, CellChat human LR, spatial_communication(dis_thr=200, heteromeric=True)",
            "sample": "10x V1_Human_Lymph_Node filtered Visium spots",
            "status": "running" if any("run_official_commot_human_lymph_node.py" in p for p in process_lines()) else ("complete" if (commot / "commot_official_summary.json").exists() else "not_running_or_failed"),
            "n_spots": 4035,
            "result_file": str(commot / "commot_official_summary.json"),
            "notes": "Full CellChat human database official run; not the previous standardized adapter.",
        },
        {
            "method": "CellNEST",
            "workflow": "Official vignette: preprocess filter_min_cell=1 data_type=visium; run 5 seeds x 80000 epochs; postprocess total_runs=5; visualize top3000",
            "sample": "10x V1_Human_Lymph_Node filtered Visium spots",
            "status": "training_running" if any(pid_running(cellnest / f"train_run{i}.pid") for i in range(1, 6)) else "training_not_running",
            "n_spots": 4035,
            "result_file": str(cellnest / "cellnest_official_summary.json"),
            "notes": "Preprocess complete: 4035 nodes and 308378 input graph edges. 5 CUDA seed runs started from official run_CellNEST.py.",
        },
    ]
    pd.DataFrame(rows).to_csv(out / "official_reproduction_status.csv", index=False)

    seed_rows = []
    for i in range(1, 6):
        seed_rows.append(
            {
                "run_id": i,
                "pid_file": str(cellnest / f"train_run{i}.pid"),
                "running": pid_running(cellnest / f"train_run{i}.pid"),
                "latest_logged_epoch": latest_epoch(cellnest / "logs" / f"train_run{i}.stdout.log"),
                "stdout_log": str(cellnest / "logs" / f"train_run{i}.stdout.log"),
                "stderr_log": str(cellnest / "logs" / f"train_run{i}.stderr.log"),
            }
        )
    pd.DataFrame(seed_rows).to_csv(cellnest / "cellnest_training_seed_status.csv", index=False)
    print(json.dumps({"status_csv": str(out / "official_reproduction_status.csv")}, indent=2))


def process_lines() -> list[str]:
    try:
        proc = subprocess.run(["ps", "-eo", "pid,cmd"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return proc.stdout.splitlines()


if __name__ == "__main__":
    main()
