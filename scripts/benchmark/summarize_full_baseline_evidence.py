#!/usr/bin/env python
"""Build a unified Phase 1 full controlled baseline evidence table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_METHOD_PRIORITY = {
    "spatialself_wavelet": 0,
    "spatialself_wavelet_edge_prior": 1,
    "spatialself_gat": 2,
    "commot": 3,
    "deeptalk": 4,
    "nichenet": 5,
    "niches": 6,
}
ADAPTER_METHOD_PRIORITY = {
    "cellchat": 10,
    "spatalk": 11,
    "cellnest": 12,
}


def summarize_full_baseline_evidence(
    output_dir: Path,
    *,
    edge_prior_core_dir: Path,
    adapter_light_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _combine_summaries(edge_prior_core_dir, adapter_light_dir)
    status = _combine_status(edge_prior_core_dir, adapter_light_dir)

    summary_path = output_dir / "baseline_metrics_summary.csv"
    status_path = output_dir / "baseline_status.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    status.to_csv(status_path, index=False)
    manifest = {
        "edge_prior_core_dir": str(edge_prior_core_dir),
        "adapter_light_dir": str(adapter_light_dir),
        "summary_path": str(summary_path),
        "status_path": str(status_path),
        "n_methods_with_metrics": int(len(summary)),
        "n_status_rows": int(len(status)),
        "note": (
            "Unified evidence table built from existing full controlled runs. "
            "CellChat/SpaTalk/CellNEST adapter-light rows are reproducibility "
            "checks and should not be over-interpreted as fully tuned baselines."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, status


def _combine_summaries(edge_prior_core_dir: Path, adapter_light_dir: Path) -> pd.DataFrame:
    frames = []
    for source_name, source_dir in (
        ("full_controlled_edge_prior_core", edge_prior_core_dir),
        ("full_controlled_retained_adapter_light", adapter_light_dir),
    ):
        path = source_dir / "baseline_metrics_summary.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "source_run", source_name)
        table["baseline_evidence_level"] = (
            "full_controlled_core"
            if source_name == "full_controlled_edge_prior_core"
            else "full_controlled_adapter_light"
        )
        table["interpretation_note"] = [
            _interpretation_note(method, source_name) for method in table["method"].astype(str)
        ]
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_priority"] = combined["method"].map(
        {**CORE_METHOD_PRIORITY, **ADAPTER_METHOD_PRIORITY}
    ).fillna(999)
    combined = combined.sort_values(["_priority", "method"]).drop(columns=["_priority"])
    return combined


def _combine_status(edge_prior_core_dir: Path, adapter_light_dir: Path) -> pd.DataFrame:
    frames = []
    for source_name, source_dir in (
        ("full_controlled_edge_prior_core", edge_prior_core_dir),
        ("full_controlled_retained_adapter_light", adapter_light_dir),
    ):
        path = source_dir / "baseline_status.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "source_run", source_name)
        table["baseline_evidence_level"] = (
            "full_controlled_core"
            if source_name == "full_controlled_edge_prior_core"
            else "full_controlled_adapter_light"
        )
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_priority"] = combined["method"].map(
        {**CORE_METHOD_PRIORITY, **ADAPTER_METHOD_PRIORITY}
    ).fillna(999)
    combined = combined.sort_values(["_priority", "method"]).drop(columns=["_priority"])
    return combined


def _interpretation_note(method: str, source_name: str) -> str:
    if method == "deeptalk":
        return "projection/status only unless LR-specific trained model files are supplied"
    if method == "nichenet":
        return "downstream validation only, not candidate-edge AUROC baseline"
    if method == "niches":
        return "optional blocked by local R dependency installation"
    if source_name == "full_controlled_retained_adapter_light":
        return "adapter-light reproducibility check; not final tuned baseline"
    if method == "spatialself_wavelet_edge_prior":
        return "candidate model variant, compared side by side with current default"
    return "full controlled candidate-edge benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified",
    )
    parser.add_argument(
        "--edge-prior-core-dir",
        default="outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core",
    )
    parser.add_argument(
        "--adapter-light-dir",
        default="outputs/phase1/exp03_baseline_comparison/full_controlled_retained_adapter_light",
    )
    args = parser.parse_args()

    summary, status = summarize_full_baseline_evidence(
        Path(args.output_dir),
        edge_prior_core_dir=Path(args.edge_prior_core_dir),
        adapter_light_dir=Path(args.adapter_light_dir),
    )
    print(summary[["method", "source_run", "auroc_mean", "aupr_mean", "weak_signal_recall_at_threshold_mean"]].to_string(index=False))
    print(status[["method", "status", "baseline_evidence_level"]].to_string(index=False))
    print(f"Wrote unified full baseline evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
