#!/usr/bin/env python
"""Run COMMOT baseline on priority LR axes for mouse colon MERFISH multislice data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import STAGE_ORDER, infer_stage, read_metadata
from src.spitialself.baselines import score_commot


PRIORITY_AXES = [
    "Cxcl10-Cxcr3",
    "Ccl5-Ccr5",
    "Cxcl9-Cxcr3",
    "Cxcl12-Cxcr4",
    "Ccl2-Ccr2",
    "Tnf-Tnfrsf1a",
    "Tnf-Tnfrsf1b",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-root", default="outputs/current_mouse_colon_ibd_merfish/multislice_replicates")
    parser.add_argument("--output-dir", default="outputs/current_mouse_colon_ibd_merfish/commot_priority_baseline")
    parser.add_argument("--dis-thr", type=float, default=420.0)
    parser.add_argument("--cot-eps-p", type=float, default=0.1)
    parser.add_argument("--cot-nitermax", type=int, default=2000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate_dir in find_replicates(Path(args.replicate_root)):
        metadata = read_metadata(replicate_dir)
        stage = metadata.get("stage", infer_stage(replicate_dir.name))
        pred_path = pred_dir / f"{replicate_dir.name}__commot_priority_predictions.csv"
        started = time.perf_counter()
        try:
            if args.skip_existing and pred_path.exists():
                status = "skipped_existing"
            else:
                run_one(
                    replicate_dir,
                    pred_path,
                    dis_thr=float(args.dis_thr),
                    cot_eps_p=float(args.cot_eps_p),
                    cot_nitermax=int(args.cot_nitermax),
                )
                status = "completed"
            rows.append(
                {
                    "replicate": replicate_dir.name,
                    "stage": stage,
                    "stage_day": STAGE_ORDER.get(stage, np.nan),
                    "source_slice_file": metadata.get("source_slice_file", ""),
                    "mouse_id": metadata.get("mouse_id", ""),
                    "status": status,
                    "prediction_path": str(pred_path),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "replicate": replicate_dir.name,
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
    status = pd.DataFrame(rows)
    status.to_csv(output_dir / "commot_priority_status.csv", index=False)
    summary = summarize(Path(args.replicate_root), status)
    summary.to_csv(output_dir / "commot_priority_stage_trajectory_summary.csv", index=False)
    write_report(status, summary, output_dir / "commot_priority_report.md")
    print(status[["replicate", "stage", "status", "runtime_seconds", "error"]].to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Wrote COMMOT priority baseline to {output_dir}")


def find_replicates(root: Path) -> list[Path]:
    required = {"cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"}
    reps = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and required.issubset({file.name for file in path.iterdir() if file.is_file()}):
            reps.append(path)
    return sorted(reps, key=lambda p: (STAGE_ORDER.get(infer_stage(p.name), 999), p.name))


def run_one(replicate_dir: Path, pred_path: Path, *, dis_thr: float, cot_eps_p: float, cot_nitermax: int) -> None:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs["lr_pair"] = lr_pairs["ligand"].astype(str) + "-" + lr_pairs["receptor"].astype(str)
    candidates["lr_pair"] = candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)
    lr_pairs = lr_pairs[lr_pairs["lr_pair"].isin(PRIORITY_AXES)].drop(columns=["lr_pair"])
    candidates = candidates[candidates["lr_pair"].isin(PRIORITY_AXES)].drop(columns=["lr_pair"])
    if lr_pairs.empty or candidates.empty:
        raise ValueError("No priority LR candidates available for COMMOT")
    predictions = score_commot(
        cells,
        expression,
        lr_pairs,
        candidates,
        dis_thr=dis_thr,
        cot_eps_p=cot_eps_p,
        cot_nitermax=cot_nitermax,
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)


def summarize(replicate_root: Path, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in status.itertuples(index=False):
        if not str(row.prediction_path):
            continue
        pred_path = Path(row.prediction_path)
        if not pred_path.exists():
            continue
        replicate_dir = replicate_root / row.replicate
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        pred = pd.read_csv(pred_path)
        merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        merged["lr_pair"] = merged["ligand"].astype(str) + "-" + merged["receptor"].astype(str)
        for lr_pair, group in merged.groupby("lr_pair"):
            rows.append(
                {
                    "method": "COMMOT",
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
    out = (
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
    return out


def write_report(status: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# COMMOT Priority Baseline on Mouse Colon MERFISH",
        "",
        f"- Completed: {int(status['status'].eq('completed').sum())}/{len(status)}",
        f"- Failed: {int(status['status'].eq('failed').sum())}/{len(status)}",
        "",
        "## Notes",
        "",
        "This is a targeted priority-axis baseline, not a full LR-atlas COMMOT run.",
        "",
    ]
    if not summary.empty:
        lines += ["## Stage Summary", ""]
        for lr_pair, group in summary.groupby("lr_pair"):
            vals = ", ".join(f"{r.stage}={r.score_q95_mean:.3g}" for r in group.itertuples(index=False))
            lines.append(f"- `{lr_pair}`: {vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
