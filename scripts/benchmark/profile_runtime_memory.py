#!/usr/bin/env python
"""Profile wall time and peak memory for benchmark methods."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUNNABLE_METHODS = (
    "spatialself_wavelet",
    "spatialself_wavelet_edge_prior",
    "spatialself_gat",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
    "deeptalk",
)
PENDING_METHODS = {
    "spatialself": "training/inference runner is not wired to benchmark candidate schema yet",
    "niches": "optional blocked; package install failed in local R env",
    "nichenet": "downstream target validation prior only; not a candidate-edge scoring baseline",
}


def profile_methods(
    dataset_root: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    conda_env: str,
    cot_nitermax: int,
    cellchat_nboot: int,
    spatialself_epochs: int,
    spatialself_k_nn: int,
    spatialself_max_train_samples: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_weak_positive_fraction: float,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    spatialself_no_lr_embedding: bool,
    spatialself_no_distance: bool,
    spatialself_no_spatial_graph: bool,
    spatialself_no_expression_features: bool,
    cellnest_num_epoch: int,
    cellnest_hidden: int,
    deeptalk_input_root: str,
    spatalk_per_num: int,
    spatalk_min_pairs: int,
    spatalk_use_synthetic_pathways: bool,
) -> pd.DataFrame:
    dataset_profile = _profile_dataset(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    rows = []
    for method in methods:
        row = {
            **dataset_profile,
            "method": method,
            "status": "optional_blocked" if method == "niches" else "pending_adapter",
            "detail": PENDING_METHODS.get(method, "method is not registered"),
            "wall_clock_seconds": float("nan"),
            "peak_cpu_memory_mb": float("nan"),
            "peak_gpu_memory_mb": float("nan"),
            "gpu_monitor_status": _gpu_monitor_status(),
            "runtime_per_epoch_seconds": float("nan"),
            "total_runtime_seconds": float("nan"),
            "returncode": None,
            "command": "",
            "stdout_log": "",
            "stderr_log": "",
        }
        if method in RUNNABLE_METHODS:
            command = _method_command(
                method,
                dataset_root,
                output_dir / method,
                conda_env=conda_env,
                cot_nitermax=cot_nitermax,
                cellchat_nboot=cellchat_nboot,
                spatialself_epochs=spatialself_epochs,
                spatialself_k_nn=spatialself_k_nn,
                spatialself_max_train_samples=spatialself_max_train_samples,
                spatialself_pseudo_label_strategy=spatialself_pseudo_label_strategy,
                spatialself_weak_positive_fraction=spatialself_weak_positive_fraction,
                spatialself_ranking_loss_weight=spatialself_ranking_loss_weight,
                spatialself_ranking_margin=spatialself_ranking_margin,
                spatialself_no_lr_embedding=spatialself_no_lr_embedding,
                spatialself_no_distance=spatialself_no_distance,
                spatialself_no_spatial_graph=spatialself_no_spatial_graph,
                spatialself_no_expression_features=spatialself_no_expression_features,
                cellnest_num_epoch=cellnest_num_epoch,
                cellnest_hidden=cellnest_hidden,
                deeptalk_input_root=deeptalk_input_root,
                spatalk_per_num=spatalk_per_num,
                spatalk_min_pairs=spatalk_min_pairs,
                spatalk_use_synthetic_pathways=spatalk_use_synthetic_pathways,
            )
            result = _run_profiled(command, cwd=PROJECT_ROOT)
            stdout_log = log_dir / f"{method}.stdout.log"
            stderr_log = log_dir / f"{method}.stderr.log"
            stdout_log.write_text(result["stdout"], encoding="utf-8")
            stderr_log.write_text(result["stderr"], encoding="utf-8")
            row.update(
                {
                    "status": "completed" if result["returncode"] == 0 else "failed",
                    "detail": "profiled runnable adapter"
                    if result["returncode"] == 0
                    else "adapter command failed; inspect logs",
                    "wall_clock_seconds": result["wall_clock_seconds"],
                    "peak_cpu_memory_mb": result["peak_cpu_memory_mb"],
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    "gpu_monitor_status": result["gpu_monitor_status"],
                    "runtime_per_epoch_seconds": _runtime_per_epoch(
                        method,
                        result["wall_clock_seconds"],
                        spatialself_epochs=spatialself_epochs,
                        cellnest_num_epoch=cellnest_num_epoch,
                    ),
                    "total_runtime_seconds": result["wall_clock_seconds"],
                    "returncode": result["returncode"],
                    "command": " ".join(command),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                }
            )
        rows.append(row)

    profile_table = pd.DataFrame(rows)
    profile_table.to_csv(output_dir / "runtime_memory_profile.csv", index=False)
    with (output_dir / "runtime_memory_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    return profile_table


def _method_command(
    method: str,
    dataset_root: Path,
    output_dir: Path,
    *,
    conda_env: str,
    cot_nitermax: int,
    cellchat_nboot: int,
    spatialself_epochs: int,
    spatialself_k_nn: int,
    spatialself_max_train_samples: int,
    spatialself_pseudo_label_strategy: str,
    spatialself_weak_positive_fraction: float,
    spatialself_ranking_loss_weight: float,
    spatialself_ranking_margin: float,
    spatialself_no_lr_embedding: bool,
    spatialself_no_distance: bool,
    spatialself_no_spatial_graph: bool,
    spatialself_no_expression_features: bool,
    cellnest_num_epoch: int,
    cellnest_hidden: int,
    deeptalk_input_root: str,
    spatalk_per_num: int,
    spatalk_min_pairs: int,
    spatalk_use_synthetic_pathways: bool,
) -> list[str]:
    script_name = f"scripts/benchmark/run_{method}.py"
    if method in {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}:
        script_name = "scripts/benchmark/run_spatialself.py"
    if method == "cellnest":
        script_name = "scripts/benchmark/run_cellnest.py"
    if method == "deeptalk":
        script_name = "scripts/benchmark/run_deeptalk.py"

    command = ["conda", "run", "-n", conda_env, "python", script_name, str(dataset_root), "--output-dir", str(output_dir)]
    if method == "commot":
        command.extend(["--cot-nitermax", str(int(cot_nitermax))])
    if method == "cellchat":
        command.extend(["--nboot", str(int(cellchat_nboot))])
    if method == "cellnest":
        command.extend(["--num-epoch", str(int(cellnest_num_epoch)), "--hidden", str(int(cellnest_hidden))])
    if method == "deeptalk" and deeptalk_input_root:
        command.extend(["--deeptalk-input-root", deeptalk_input_root])
    if method == "spatalk":
        command.extend(["--per-num", str(int(spatalk_per_num)), "--min-pairs", str(int(spatalk_min_pairs))])
        if spatalk_use_synthetic_pathways:
            command.append("--use-synthetic-pathways")
    if method in {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}:
        model_name = "spatialself_wavelet" if method == "spatialself_wavelet_edge_prior" else method
        command.extend(
            [
                "--model-name",
                model_name,
                "--epochs",
                str(int(spatialself_epochs)),
                "--k-nn",
                str(int(spatialself_k_nn)),
                "--max-train-samples",
                str(int(spatialself_max_train_samples)),
                "--pseudo-label-strategy",
                spatialself_pseudo_label_strategy,
                "--weak-positive-fraction",
                str(float(spatialself_weak_positive_fraction)),
                "--ranking-loss-weight",
                str(float(spatialself_ranking_loss_weight)),
                "--ranking-margin",
                str(float(spatialself_ranking_margin)),
            ]
        )
        if spatialself_no_lr_embedding:
            command.append("--no-lr-embedding")
        if spatialself_no_distance:
            command.append("--no-distance")
        if spatialself_no_spatial_graph:
            command.append("--no-spatial-graph")
        if spatialself_no_expression_features:
            command.append("--no-expression-features")
        if method == "spatialself_wavelet_edge_prior":
            command.append("--use-edge-prior-features")
    return command


def _run_profiled(command: list[str], *, cwd: Path) -> dict:
    time_bin = shutil.which("/usr/bin/time") or shutil.which("time")
    profiled_command = command
    if time_bin:
        profiled_command = [time_bin, "-v", *command]

    start = time.perf_counter()
    with tempfile.TemporaryFile("w+", encoding="utf-8") as stdout_handle, tempfile.TemporaryFile(
        "w+",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            profiled_command,
            cwd=cwd,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        peak_gpu_memory_mb = float("nan")
        gpu_monitor_status = _gpu_monitor_status()
        while process.poll() is None:
            current_gpu_memory_mb = _query_gpu_memory_mb(process.pid)
            if current_gpu_memory_mb == current_gpu_memory_mb:
                if peak_gpu_memory_mb != peak_gpu_memory_mb:
                    peak_gpu_memory_mb = current_gpu_memory_mb
                else:
                    peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_gpu_memory_mb)
                gpu_monitor_status = "nvidia_smi_process_tree_memory"
            time.sleep(0.25)
        stdout_handle.seek(0)
        stderr_handle.seek(0)
        stdout = stdout_handle.read()
        stderr = stderr_handle.read()
    elapsed = time.perf_counter() - start
    peak_memory_mb = _parse_peak_memory_mb(stderr)
    return {
        "returncode": process.returncode,
        "wall_clock_seconds": float(elapsed),
        "peak_cpu_memory_mb": peak_memory_mb,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "gpu_monitor_status": gpu_monitor_status,
        "stdout": stdout,
        "stderr": stderr,
    }


def _parse_peak_memory_mb(stderr: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if not match:
        return float("nan")
    return int(match.group(1)) / 1024.0


def _gpu_monitor_status() -> str:
    if not shutil.which("nvidia-smi"):
        return "nvidia_smi_unavailable"
    return "nvidia_smi_available_not_sampled"


def _query_gpu_memory_mb(root_pid: int) -> float:
    if not shutil.which("nvidia-smi"):
        return float("nan")
    profile_pids = _process_tree_pids(root_pid)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return float("nan")
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            memory_mb = float(parts[1].split()[0])
        except ValueError:
            continue
        if pid in profile_pids:
            values.append(memory_mb)
    if not values:
        return float("nan")
    return sum(values)


def _process_tree_pids(root_pid: int) -> set[int]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {int(root_pid)}
    children_by_parent: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)
    seen = {int(root_pid)}
    stack = [int(root_pid)]
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _runtime_per_epoch(
    method: str,
    wall_clock_seconds: float,
    *,
    spatialself_epochs: int,
    cellnest_num_epoch: int,
) -> float:
    if wall_clock_seconds != wall_clock_seconds:
        return float("nan")
    if method in {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"} and spatialself_epochs > 0:
        return float(wall_clock_seconds) / float(spatialself_epochs)
    if method == "cellnest" and cellnest_num_epoch > 0:
        return float(wall_clock_seconds) / float(cellnest_num_epoch)
    return float("nan")


def _profile_dataset(dataset_root: Path) -> dict:
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise ValueError(f"No replicate directories found under {dataset_root}")

    n_cells = 0
    n_candidates = 0
    n_lr_pairs = 0
    for replicate_dir in replicate_dirs:
        n_cells += _row_count(replicate_dir / "cells.csv")
        n_candidates += _row_count(replicate_dir / "candidate_interactions.csv")
        n_lr_pairs += _row_count(replicate_dir / "lr_pairs.csv")
    return {
        "dataset_root": str(dataset_root),
        "n_replicates": len(replicate_dirs),
        "n_cells_or_spots": n_cells,
        "n_candidate_edges": n_candidates,
        "n_lr_pairs": n_lr_pairs,
    }


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp08_runtime_memory",
        help="Directory for runtime and memory outputs.",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method to profile or record. Repeatable. Defaults to runnable adapters.",
    )
    parser.add_argument("--conda-env", default="Benchmark")
    parser.add_argument("--cot-nitermax", type=int, default=500)
    parser.add_argument("--cellchat-nboot", type=int, default=20)
    parser.add_argument("--spatialself-epochs", type=int, default=60)
    parser.add_argument("--spatialself-k-nn", type=int, default=12)
    parser.add_argument("--spatialself-max-train-samples", type=int, default=20000)
    parser.add_argument("--spatialself-pseudo-label-strategy", default="expression_distance")
    parser.add_argument("--spatialself-weak-positive-fraction", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatialself-ranking-margin", type=float, default=1.0)
    parser.add_argument("--spatialself-no-lr-embedding", action="store_true")
    parser.add_argument("--spatialself-no-distance", action="store_true")
    parser.add_argument("--spatialself-no-spatial-graph", action="store_true")
    parser.add_argument("--spatialself-no-expression-features", action="store_true")
    parser.add_argument("--cellnest-num-epoch", type=int, default=2)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    parser.add_argument("--deeptalk-input-root", default="")
    parser.add_argument("--spatalk-per-num", type=int, default=100)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    args = parser.parse_args()

    methods = tuple(args.methods) if args.methods else RUNNABLE_METHODS
    profile_table = profile_methods(
        Path(args.dataset_root),
        Path(args.output_dir),
        methods=methods,
        conda_env=args.conda_env,
        cot_nitermax=args.cot_nitermax,
        cellchat_nboot=args.cellchat_nboot,
        spatialself_epochs=args.spatialself_epochs,
        spatialself_k_nn=args.spatialself_k_nn,
        spatialself_max_train_samples=args.spatialself_max_train_samples,
        spatialself_pseudo_label_strategy=args.spatialself_pseudo_label_strategy,
        spatialself_weak_positive_fraction=args.spatialself_weak_positive_fraction,
        spatialself_ranking_loss_weight=args.spatialself_ranking_loss_weight,
        spatialself_ranking_margin=args.spatialself_ranking_margin,
        spatialself_no_lr_embedding=args.spatialself_no_lr_embedding,
        spatialself_no_distance=args.spatialself_no_distance,
        spatialself_no_spatial_graph=args.spatialself_no_spatial_graph,
        spatialself_no_expression_features=args.spatialself_no_expression_features,
        cellnest_num_epoch=args.cellnest_num_epoch,
        cellnest_hidden=args.cellnest_hidden,
        deeptalk_input_root=args.deeptalk_input_root,
        spatalk_per_num=args.spatalk_per_num,
        spatalk_min_pairs=args.spatalk_min_pairs,
        spatalk_use_synthetic_pathways=args.spatalk_use_synthetic_pathways,
    )
    columns = [
        "method",
        "status",
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "peak_gpu_memory_mb",
        "gpu_monitor_status",
        "runtime_per_epoch_seconds",
    ]
    print(profile_table[columns].to_string(index=False))
    print(f"Wrote runtime/memory profile to {args.output_dir}")


if __name__ == "__main__":
    main()
