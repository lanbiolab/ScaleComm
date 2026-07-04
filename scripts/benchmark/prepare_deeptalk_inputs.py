#!/usr/bin/env python
"""Prepare DeepTalk input directories from generated benchmark replicates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.baselines import prepare_deeptalk_inputs


def prepare_replicate(replicate_dir: Path, output_dir: Path) -> pd.DataFrame:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    return prepare_deeptalk_inputs(cells, expression, candidates, output_dir / replicate_dir.name)


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/deeptalk_inputs",
        help="Output root for DeepTalk-formatted inputs.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    manifests = []
    for replicate_dir in replicate_dirs:
        manifest = prepare_replicate(replicate_dir, output_dir)
        manifest.insert(0, "replicate", replicate_dir.name)
        manifests.append(manifest)

    summary = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "deeptalk_input_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote DeepTalk inputs to {output_dir}")


if __name__ == "__main__":
    main()
