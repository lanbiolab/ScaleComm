#!/usr/bin/env python
"""Run CellNEST priority-axis baseline for mouse colon MERFISH multislice data."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.current.run_mouse_colon_merfish_commot_priority_baseline import PRIORITY_AXES, find_replicates
from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import STAGE_ORDER, infer_stage, read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-root", default="outputs/current_mouse_colon_ibd_merfish/multislice_replicates")
    parser.add_argument("--output-dir", default="outputs/current_mouse_colon_ibd_merfish/cellnest_priority_baseline")
    parser.add_argument("--conda-env", default="Benchmark")
    parser.add_argument("--num-epoch", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-percent", type=int, default=100)
    parser.add_argument("--threshold-gene-exp", type=float, default=60.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--max-replicates", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    filtered_root = output_dir / "filtered_priority_replicates"
    filtered_root.mkdir(parents=True, exist_ok=True)

    status_rows = []
    replicate_dirs = find_replicates(Path(args.replicate_root))[: int(args.max_replicates)]
    for replicate_dir in replicate_dirs:
        metadata = read_metadata(replicate_dir)
        stage = metadata.get("stage", infer_stage(replicate_dir.name))
        filtered_dir = make_filtered_replicate(replicate_dir, filtered_root / replicate_dir.name)
        pred_path = output_dir / f"{filtered_dir.name}__cellnest_predictions.csv"
        started = time.perf_counter()
        try:
            if args.skip_existing and pred_path.exists():
                status = "skipped_existing"
                metrics = {"prediction_path": str(pred_path)}
            else:
                metrics = run_cellnest_on_replicate(
                    filtered_dir,
                    output_dir,
                    conda_env=str(args.conda_env),
                    num_epoch=int(args.num_epoch),
                    hidden=int(args.hidden),
                    seed=int(args.seed),
                    top_percent=int(args.top_percent),
                    threshold_gene_exp=float(args.threshold_gene_exp),
                    k=int(args.k),
                )
                status = "completed"
            status_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "filtered_replicate": filtered_dir.name,
                    "stage": stage,
                    "stage_day": STAGE_ORDER.get(stage, np.nan),
                    "source_slice_file": metadata.get("source_slice_file", ""),
                    "mouse_id": metadata.get("mouse_id", ""),
                    "status": status,
                    "prediction_path": metrics.get("prediction_path", ""),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "filtered_replicate": filtered_dir.name,
                    "stage": stage,
                    "stage_day": STAGE_ORDER.get(stage, np.nan),
                    "source_slice_file": metadata.get("source_slice_file", ""),
                    "mouse_id": metadata.get("mouse_id", ""),
                    "status": "failed",
                    "prediction_path": "",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": repr(exc),
                }
            )
    status = pd.DataFrame(status_rows)
    status.to_csv(output_dir / "cellnest_priority_status.csv", index=False)
    summary = summarize(filtered_root, status)
    summary.to_csv(output_dir / "cellnest_priority_stage_trajectory_summary.csv", index=False)
    write_report(status, summary, output_dir / "cellnest_priority_report.md")
    print(status[["replicate", "stage", "status", "runtime_seconds", "error"]].to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Wrote CellNEST priority baseline to {output_dir}")


def make_filtered_replicate(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["cells.csv", "expression.csv", "metadata.json"]:
        shutil.copy2(src / name, dst / name)
    lr = pd.read_csv(src / "lr_pairs.csv")
    lr["lr_pair"] = lr["ligand"].astype(str) + "-" + lr["receptor"].astype(str)
    lr = lr[lr["lr_pair"].isin(PRIORITY_AXES)].drop(columns=["lr_pair"])
    lr.to_csv(dst / "lr_pairs.csv", index=False)
    cand = pd.read_csv(src / "candidate_interactions.csv")
    cand["lr_pair"] = cand["ligand"].astype(str) + "-" + cand["receptor"].astype(str)
    cand = cand[cand["lr_pair"].isin(PRIORITY_AXES)].drop(columns=["lr_pair"])
    cand.to_csv(dst / "candidate_interactions.csv", index=False)
    return dst


def summarize(filtered_root: Path, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in status.itertuples(index=False):
        if not str(row.prediction_path):
            continue
        pred_path = Path(row.prediction_path)
        if not pred_path.exists():
            continue
        filtered_dir = filtered_root / row.filtered_replicate
        candidates = pd.read_csv(filtered_dir / "candidate_interactions.csv")
        pred = pd.read_csv(pred_path)
        merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        merged["lr_pair"] = merged["ligand"].astype(str) + "-" + merged["receptor"].astype(str)
        for lr_pair, group in merged.groupby("lr_pair"):
            rows.append(
                {
                    "method": "CellNEST",
                    "replicate": row.replicate,
                    "stage": row.stage,
                    "stage_day": row.stage_day,
                    "lr_pair": lr_pair,
                    "n_edges": int(len(group)),
                    "score_mean": float(group["score"].mean()),
                    "score_q95": float(group["score"].quantile(0.95)),
                    "score_max": float(group["score"].max()),
                    "nonzero_fraction": float((group["score"] > 0).mean()),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    return (
        long.groupby(["method", "lr_pair", "stage"], as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            score_q95_mean=("score_q95", "mean"),
            score_q95_std=("score_q95", "std"),
            score_mean_mean=("score_mean", "mean"),
            nonzero_fraction_mean=("nonzero_fraction", "mean"),
        )
        .sort_values(["lr_pair", "stage"])
    )


def write_report(status: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# CellNEST Priority Baseline on Mouse Colon MERFISH",
        "",
        f"- Completed: {int(status['status'].eq('completed').sum())}/{len(status)}",
        f"- Failed: {int(status['status'].eq('failed').sum())}/{len(status)}",
        "",
        "This uses CellNEST official preprocessing/training/postprocessing through the local CellNEST checkout, with the LR database filtered to the priority axes.",
        "",
    ]
    if status["status"].eq("failed").any():
        lines += ["## Failures", ""]
        for row in status[status["status"].eq("failed")].itertuples(index=False):
            lines.append(f"- `{row.replicate}`: {row.error}")
    if not summary.empty:
        lines += ["", "## Stage Summary", ""]
        for lr_pair, group in summary.groupby("lr_pair"):
            vals = ", ".join(f"{r.stage}={r.score_q95_mean:.3g}" for r in group.itertuples(index=False))
            lines.append(f"- `{lr_pair}`: {vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
