#!/usr/bin/env python
"""Run fresh tonsil high-resolution subset scaling experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated")
    parser.add_argument("--n-cells", action="append", type=int, dest="n_cells")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--n-background-lr", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reuse-existing-720", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cells_values = tuple(args.n_cells) if args.n_cells else (720, 1200, 1800)

    run_rows = []
    started = time.perf_counter()
    for n_cells in n_cells_values:
        run_dir = output_dir / f"n{n_cells}"
        if n_cells == 720 and args.reuse_existing_720:
            existing = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset"
            if (existing / "tonsil_fresh_dataset_profile.csv").exists():
                _copy_summary_inputs(existing, run_dir)
                run_rows.append(_run_row_from_outputs(run_dir, n_cells=n_cells, reused_from=str(existing.relative_to(PROJECT_ROOT))))
                continue
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/phase2/run_goal4_fresh_tonsil_subset.py"),
            "--output-dir",
            str(run_dir.relative_to(PROJECT_ROOT)),
            "--n-cells",
            str(n_cells),
            "--n-background-lr",
            str(args.n_background_lr),
            "--epochs",
            str(args.epochs),
            "--max-train-samples",
            str(args.max_train_samples),
            "--seed",
            str(args.seed),
        ]
        run_started = time.perf_counter()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        row = _run_row_from_outputs(run_dir, n_cells=n_cells, reused_from="")
        row["runner_wall_clock_seconds"] = time.perf_counter() - run_started
        row["stdout_tail"] = "\n".join(result.stdout.strip().splitlines()[-4:])
        row["stderr_tail"] = "\n".join(result.stderr.strip().splitlines()[-4:])
        run_rows.append(row)

    run_summary = pd.DataFrame(run_rows).sort_values("n_cells").reset_index(drop=True)
    scaling_summary = _scaling_summary(run_summary, output_dir=output_dir, elapsed=time.perf_counter() - started)
    run_summary.to_csv(output_dir / "tonsil_scaling_run_summary.csv", index=False)
    pd.DataFrame([scaling_summary]).to_csv(output_dir / "tonsil_scaling_summary.csv", index=False)
    (output_dir / "tonsil_scaling_manifest.json").write_text(
        json.dumps(
            {
                "analysis_label": "tonsil_scaling_wavelet_calibrated",
                "evidence_status": "fresh_scaling_initial",
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "n_cells": list(n_cells_values),
                "epochs": args.epochs,
                "max_train_samples": args.max_train_samples,
                "n_background_lr": args.n_background_lr,
                "seed": args.seed,
                "run_summary": str((output_dir / "tonsil_scaling_run_summary.csv").relative_to(PROJECT_ROOT)),
                "scaling_summary": str((output_dir / "tonsil_scaling_summary.csv").relative_to(PROJECT_ROOT)),
                "limitations": [
                    "Initial subset scaling, not a full Slide-tags tonsil run.",
                    "Runs reuse the same model settings with fewer epochs for larger subsets.",
                    "Biological validation remains control-rank based until downstream validation is expanded.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(run_summary.to_string(index=False))
    print(pd.DataFrame([scaling_summary]).to_string(index=False))


def _copy_summary_inputs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "tonsil_fresh_dataset_profile.csv",
        "tonsil_fresh_goal4_summary.csv",
        "tonsil_fresh_control_summary.csv",
        "tonsil_fresh_lr_summary.csv",
        "tonsil_fresh_cell_type_pair_summary.csv",
        "tonsil_fresh_manifest.json",
    ]:
        src_path = src / name
        if src_path.exists():
            (dst / name).write_bytes(src_path.read_bytes())


def _run_row_from_outputs(run_dir: Path, *, n_cells: int, reused_from: str) -> dict[str, object]:
    profile = pd.read_csv(run_dir / "tonsil_fresh_dataset_profile.csv").iloc[0]
    summary = pd.read_csv(run_dir / "tonsil_fresh_goal4_summary.csv").iloc[0]
    controls = pd.read_csv(run_dir / "tonsil_fresh_control_summary.csv")
    positives = controls[controls["control_class"] == "tonsil_positive_prior"]
    negatives = controls[controls["control_class"] != "tonsil_positive_prior"]
    return {
        "analysis_label": f"tonsil_scaling_n{n_cells}",
        "evidence_status": "fresh_scaling_initial",
        "n_cells": int(profile["n_cells"]),
        "n_cell_types": int(profile["n_cell_types"]),
        "n_lr_pairs": int(profile["n_lr_pairs"]),
        "n_candidate_edges": int(profile["n_candidate_edges"]),
        "epochs": int(profile["epochs"]),
        "max_train_samples": int(profile["max_train_samples"]),
        "wall_clock_seconds": float(profile["wall_clock_seconds"]),
        "train_and_inference_seconds": float(profile["train_and_inference_seconds"]),
        "peak_rss_mb": float(profile["peak_rss_mb"]),
        "n_positive_controls_top_decile": int(summary["n_positive_controls_top_decile"]),
        "n_negative_controls_top_decile": int(summary["n_negative_controls_top_decile"]),
        "best_positive_control": summary["best_positive_control"],
        "best_positive_control_rank": int(summary["best_positive_control_rank"]),
        "best_negative_control": summary["best_negative_control"],
        "best_negative_control_rank": int(summary["best_negative_control_rank"]),
        "median_positive_rank_percentile": float(summary["median_positive_rank_percentile"]),
        "median_negative_rank_percentile": float(summary["median_negative_rank_percentile"]),
        "best_positive_rank_percentile": float(positives["rank_percentile_by_score_mean"].min()),
        "best_negative_rank_percentile": float(negatives["rank_percentile_by_score_mean"].min()),
        "reused_from": reused_from,
        "output_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def _scaling_summary(run_summary: pd.DataFrame, *, output_dir: Path, elapsed: float) -> dict[str, object]:
    largest = run_summary.sort_values("n_cells").iloc[-1]
    edge_slope = _slope(run_summary["n_cells"], run_summary["n_candidate_edges"])
    runtime_slope = _slope(run_summary["n_candidate_edges"], run_summary["wall_clock_seconds"])
    return {
        "goal_id": "4",
        "analysis_label": "tonsil_scaling_wavelet_calibrated",
        "evidence_status": "fresh_scaling_initial",
        "n_runs": int(len(run_summary)),
        "n_cells_min": int(run_summary["n_cells"].min()),
        "n_cells_max": int(run_summary["n_cells"].max()),
        "n_candidate_edges_max": int(run_summary["n_candidate_edges"].max()),
        "largest_run_wall_clock_seconds": float(largest["wall_clock_seconds"]),
        "largest_run_peak_rss_mb": float(largest["peak_rss_mb"]),
        "largest_run_positive_controls_top_decile": int(largest["n_positive_controls_top_decile"]),
        "largest_run_negative_controls_top_decile": int(largest["n_negative_controls_top_decile"]),
        "best_positive_rank_percentile_min": float(run_summary["best_positive_rank_percentile"].min()),
        "best_negative_rank_percentile_min": float(run_summary["best_negative_rank_percentile"].min()),
        "edge_count_slope_per_cell": edge_slope,
        "runtime_seconds_per_candidate_edge": runtime_slope,
        "total_runner_wall_clock_seconds": float(elapsed),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "interpretation": (
            "Initial high-resolution subset scaling; supports scalability wiring but does not replace "
            "full-data or baseline-matched high-resolution claims."
        ),
    }


def _slope(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    dx = float(x.iloc[-1] - x.iloc[0])
    dy = float(y.iloc[-1] - y.iloc[0])
    return dy / dx if dx else float("nan")


if __name__ == "__main__":
    main()
