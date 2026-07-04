#!/usr/bin/env python
"""Prepare CellNEST h5ad/database inputs from generated benchmark replicates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.baselines import prepare_cellnest_inputs


def prepare_replicate(replicate_dir: Path, output_dir: Path) -> dict:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    dataset_name = f"phase1_{replicate_dir.name}".replace("-", "_")
    return prepare_cellnest_inputs(
        cells,
        expression,
        lr_pairs,
        output_dir / replicate_dir.name,
        dataset_name=dataset_name,
    )


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/cellnest_inputs",
        help="Output root for CellNEST-formatted inputs.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    rows = [prepare_replicate(replicate_dir, output_dir) for replicate_dir in replicate_dirs]
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "cellnest_input_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote CellNEST inputs to {output_dir}")


if __name__ == "__main__":
    main()
