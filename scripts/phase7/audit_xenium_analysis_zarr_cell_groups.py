#!/usr/bin/env python
"""Audit 10x Xenium analysis.zarr cell groups for the Phase 7 lymph-node subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/optional/xenium_human_lymph_node/raw"
RAW_PREFIX = "Xenium_V1_hLymphNode_nondiseased_section"
SUBSET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    "standardized_inputs/xenium_human_lymph_node_subset"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--subset-dir", default=str(SUBSET_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_raw = pd.read_parquet(raw_dir / f"{RAW_PREFIX}_cells.parquet")
    subset_cells = pd.read_csv(subset_dir / "cells.csv")
    zarr_path = raw_dir / f"{RAW_PREFIX}_analysis.zarr.zip"
    root = zarr.open(str(zarr_path), mode="r")
    attrs = dict(root["cell_groups"].attrs)
    grouping_names = list(attrs.get("grouping_names", []))
    group_names = list(attrs.get("group_names", []))

    raw_index = pd.Series(np.arange(len(cells_raw), dtype=int), index=cells_raw["cell_id"].astype(str))
    rows = []
    assignment_frames = []
    for grouping_idx, grouping_name in enumerate(grouping_names):
        assignments = _assignment_for_grouping(root, grouping_idx, group_names[grouping_idx])
        assignments["cell_index"] = assignments["cell_index"].astype(int)
        assignments = assignments.merge(
            cells_raw[["cell_id"]].reset_index(names="cell_index"),
            on="cell_index",
            how="left",
            validate="one_to_one",
        )
        subset = subset_cells[["cell_id", "cell_type", "niche"]].merge(
            assignments[["cell_id", "cluster_name", "cluster_id"]],
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
        subset["grouping_name"] = grouping_name
        assignment_frames.append(subset)
        rows.append(
            {
                "grouping_index": grouping_idx,
                "grouping_name": grouping_name,
                "n_groups": int(len(group_names[grouping_idx])),
                "n_raw_assigned": int(assignments["cell_id"].notna().sum()),
                "n_subset_cells": int(len(subset)),
                "n_subset_assigned": int(subset["cluster_name"].notna().sum()),
                "n_subset_clusters": int(subset["cluster_name"].nunique(dropna=True)),
                "top_subset_clusters": _top_counts(subset["cluster_name"], 10),
                "cluster_by_marker_proxy": _cluster_marker_cross_tab(subset),
            }
        )

    summary = pd.DataFrame(rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    summary_path = output_dir / "xenium_analysis_zarr_cell_group_summary.csv"
    assignments_path = output_dir / "xenium_analysis_zarr_subset_cell_group_assignments.csv"
    manifest_path = output_dir / "xenium_analysis_zarr_cell_group_manifest.json"
    summary.to_csv(summary_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    manifest = {
        "raw_zarr": _rel(zarr_path),
        "subset_dir": _rel(subset_dir),
        "output_dir": _rel(output_dir),
        "n_groupings": int(len(grouping_names)),
        "summary": _rel(summary_path),
        "assignments": _rel(assignments_path),
        "claim_boundary": (
            "10x analysis.zarr cell groups are unsupervised expression clusters, not curated cell-type/domain annotations. "
            "They may support sensitivity analyses or within-cluster nulls but do not provide native CCC ground truth."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _assignment_for_grouping(root: zarr.Group, grouping_idx: int, names: list[str]) -> pd.DataFrame:
    group = root["cell_groups"][str(grouping_idx)]
    indices = np.asarray(group["indices"][:], dtype=int)
    indptr = np.asarray(group["indptr"][:], dtype=int)
    rows = []
    start = 0
    for cluster_idx, end in enumerate(indptr):
        for cell_index in indices[start:end]:
            rows.append(
                {
                    "cell_index": int(cell_index),
                    "cluster_id": int(cluster_idx + 1),
                    "cluster_name": names[cluster_idx] if cluster_idx < len(names) else f"Cluster {cluster_idx + 1}",
                }
            )
        start = int(end)
    return pd.DataFrame(rows)


def _top_counts(series: pd.Series, n: int) -> str:
    counts = series.value_counts(dropna=True).head(n)
    return ";".join(f"{idx}:{int(value)}" for idx, value in counts.items())


def _cluster_marker_cross_tab(frame: pd.DataFrame) -> str:
    if frame.empty or "cluster_name" not in frame:
        return ""
    rows = []
    for cluster, group in frame.dropna(subset=["cluster_name"]).groupby("cluster_name"):
        top_type = group["cell_type"].value_counts().head(1)
        if top_type.empty:
            continue
        rows.append(f"{cluster}->{top_type.index[0]}:{int(top_type.iloc[0])}")
    return ";".join(rows[:20])


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
