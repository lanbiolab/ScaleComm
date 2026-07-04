#!/usr/bin/env python
"""Run matched 9,000-cell CosMx baselines for the Fig. 5 method-support panel.

The goal is not to create a new benchmark with ground truth. It is to generate
auditable, same-subset receiver-response support for SpatialSelf, COMMOT,
CellNEST, CellChat, and SpaTalk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellchat import run_cellchat_on_replicate
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.benchmark.run_commot import run_commot_on_replicate
from scripts.benchmark.run_spatalk import run_spatalk_on_replicate


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs/current_fig5_cosmx_nsclc_scale_context/standardized_inputs/cosmx_nsclc_he_subset_9000"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/baselines_9000"
DEFAULT_SPATIALSELF = (
    PROJECT_ROOT
    / "outputs/current_fig5_cosmx_nsclc_scale_context/spatialself_scale_9000_e120/"
    "cosmx_nsclc_he_subset_9000__spatialself_wavelet_scale_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--methods", nargs="+", default=["commot", "cellnest", "cellchat", "spatalk"])
    parser.add_argument("--cellchat-nboot", type=int, default=20)
    parser.add_argument("--cellnest-epochs", type=int, default=60)
    parser.add_argument("--cellnest-threshold-gene-exp", type=float, default=60.0)
    parser.add_argument("--cellnest-k", type=int, default=50)
    parser.add_argument("--cellnest-hidden", type=int, default=32)
    parser.add_argument("--spatalk-per-num", type=int, default=100)
    parser.add_argument("--spatalk-min-pairs", type=int, default=1)
    parser.add_argument("--spatalk-use-synthetic-pathways", action="store_true")
    parser.add_argument("--commot-cot-eps-p", type=float, default=0.1)
    parser.add_argument("--commot-cot-nitermax", type=int, default=2000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    status_rows: list[dict] = []
    metrics_rows: list[dict] = []

    add_spatialself_status(dataset_dir, output_dir, status_rows)

    jobs: dict[str, Callable[[], dict]] = {
        "commot": lambda: run_commot_in_env(
            dataset_dir,
            output_dir / "commot",
            cot_eps_p=float(args.commot_cot_eps_p),
            cot_nitermax=int(args.commot_cot_nitermax),
        ),
        "cellnest": lambda: run_cellnest_on_replicate(
            dataset_dir,
            output_dir / "cellnest",
            conda_env="Benchmark",
            num_epoch=int(args.cellnest_epochs),
            hidden=int(args.cellnest_hidden),
            seed=1,
            top_percent=100,
            threshold_gene_exp=float(args.cellnest_threshold_gene_exp),
            k=int(args.cellnest_k),
        ),
        "cellchat": lambda: run_cellchat_on_replicate(
            dataset_dir,
            output_dir / "cellchat",
            conda_env="cellchat_env",
            nboot=int(args.cellchat_nboot),
        ),
        "spatalk": lambda: run_spatalk_on_replicate(
            dataset_dir,
            output_dir / "spatalk",
            conda_env="spatalk_env",
            per_num=int(args.spatalk_per_num),
            min_pairs=int(args.spatalk_min_pairs),
            use_synthetic_pathways=bool(args.spatalk_use_synthetic_pathways),
        ),
    }

    method_notes = {
        "commot": "COMMOT adapter using commot.tl.spatial_communication and candidate-level edge projection.",
        "cellnest": (
            "Official CellNEST preprocess/train/postprocess scripts; "
            f"k={args.cellnest_k}, threshold_gene_exp={args.cellnest_threshold_gene_exp}, "
            f"epochs={args.cellnest_epochs}, hidden={args.cellnest_hidden}, top_percent=100."
        ),
        "cellchat": (
            "CellChat run on cell-type groups; cell-type-pair LR probabilities are projected "
            "to candidate cell-cell edges for receiver-support scoring."
        ),
        "spatalk": (
            "SpaTalk run on cell-type groups; LR/cell-type-pair scores are projected to "
            "candidate cell-cell edges for receiver-support scoring."
        ),
    }

    for method in args.methods:
        method = method.lower()
        if method not in jobs:
            status_rows.append({"method": method, "status": "skipped_unknown_method", "detail": ""})
            continue
        started = time.perf_counter()
        try:
            metrics = jobs[method]()
            runtime = time.perf_counter() - started
            metrics["runtime_seconds"] = float(runtime)
            metrics_rows.append(metrics)
            status_rows.append(
                {
                    "method": method,
                    "status": "completed",
                    "detail": method_notes[method],
                    "runtime_seconds": float(runtime),
                }
            )
        except Exception as exc:  # noqa: BLE001 - status table must capture external tool failures.
            runtime = time.perf_counter() - started
            log_path = output_dir / "logs" / f"{method}_failure.txt"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            status_rows.append(
                {
                    "method": method,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "runtime_seconds": float(runtime),
                    "failure_log": str(log_path),
                }
            )

    status = pd.DataFrame(status_rows)
    status.to_csv(output_dir / "baseline_status.csv", index=False)
    (output_dir / "baseline_status.json").write_text(
        json.dumps(status_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "n_cells": int(pd.read_csv(dataset_dir / "cells.csv", usecols=["cell_id"]).shape[0]),
        "n_candidates": int(pd.read_csv(dataset_dir / "candidate_interactions.csv", usecols=["candidate_id"]).shape[0]),
        "n_lr_pairs": int(pd.read_csv(dataset_dir / "lr_pairs.csv", usecols=["lr_id"]).shape[0]),
        "spatialself_prediction": str(DEFAULT_SPATIALSELF),
        "method_notes": method_notes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(status.to_string(index=False))
    print(f"Wrote matched 9,000-cell baseline outputs to {output_dir}")


def add_spatialself_status(dataset_dir: Path, output_dir: Path, rows: list[dict]) -> None:
    expected = DEFAULT_SPATIALSELF
    status = "completed" if expected.exists() else "missing_prediction"
    rows.append(
        {
            "method": "spatialself",
            "status": status,
            "detail": "SpatialSelf scale-aware 9,000-cell run reused for matched comparison.",
            "prediction_path": str(expected),
        }
    )


def run_commot_in_env(
    dataset_dir: Path,
    output_dir: Path,
    *,
    cot_eps_p: float,
    cot_nitermax: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        "commot_env",
        "python",
        str(PROJECT_ROOT / "scripts/current/run_cosmx_commot_single.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--cot-eps-p",
        str(float(cot_eps_p)),
        "--cot-nitermax",
        str(int(cot_nitermax)),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    log_path = log_dir / f"{dataset_dir.name}__commot_env.log"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"COMMOT conda run failed with exit code {completed.returncode}. Log: {log_path}")
    metrics_path = output_dir / f"{dataset_dir.name}__commot_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "replicate": dataset_dir.name,
        "method": "commot",
        "prediction_path": str(output_dir / f"{dataset_dir.name}__commot_predictions.csv"),
    }


if __name__ == "__main__":
    main()
