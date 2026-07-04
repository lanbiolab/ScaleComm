#!/usr/bin/env python
"""Build multi-slice SpatialSelf inputs for Cadinu 2024 mouse colon MERFISH.

This extends the single representative-slice pilot into a small replication
experiment: two slices per disease stage when available locally.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from build_mouse_colon_ibd_merfish_replicates import (
    STAGE_ORDER,
    build_candidates,
    build_stage_cells_expression,
    load_lr_pairs,
    select_stage_lr_pairs,
)


MULTISLICE_STAGE_FILES = {
    "Healthy_D0": [
        "082421_D0_m6_1_slice_3.csv",
        "082421_D0_m7_1_slice_1.csv",
    ],
    "DSS3": [
        "092421_D3_m3_1_slice_4.csv",
        "092421_D3_m1_1_slice_1.csv",
    ],
    "DSS9": [
        "100221_D9_m5_2_slice_3.csv",
        "100221_D9_m3_2_slice_2.csv",
    ],
    "DSS21": [
        "082421_D21_m2_1_slice_2.csv",
        "082421_D21_m2_1_slice_1.csv",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/optional/mouse_colon_ibd_merfish/raw")
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mouse_colon_ibd_merfish/multislice_replicates",
    )
    parser.add_argument("--max-cells-per-replicate", type=int, default=900)
    parser.add_argument("--max-lr-pairs", type=int, default=260)
    parser.add_argument("--max-candidates-per-lr", type=int, default=2400)
    parser.add_argument("--max-candidates-per-sender-lr", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ["cell_properties.csv", "ligand_receptor_pair_masterlist.csv"]
    required += [filename for filenames in MULTISLICE_STAGE_FILES.values() for filename in filenames]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required raw MERFISH files. Download them first with "
            f"scripts/current/download_mouse_colon_ibd_merfish_s3_ranges.py: {missing}"
        )

    lr_pairs_all = load_lr_pairs(raw_dir, int(args.max_lr_pairs))
    rows = []
    replicate_index = 0
    for stage, filenames in MULTISLICE_STAGE_FILES.items():
        for stage_rep_idx, filename in enumerate(filenames, start=1):
            slice_id = Path(filename).stem
            mouse_id = parse_mouse_id(slice_id)
            replicate_name = f"mouse_colon_ibd_merfish_{stage}_rep{stage_rep_idx}_{slice_id}"
            replicate_dir = output_dir / replicate_name
            replicate_dir.mkdir(parents=True, exist_ok=True)
            print(f"building {stage} rep{stage_rep_idx} from {filename}", flush=True)
            cells, expression = build_stage_cells_expression(
                raw_dir=raw_dir,
                stage=stage,
                filename=filename,
                max_cells=int(args.max_cells_per_replicate),
                seed=int(args.seed) + replicate_index * 17 + STAGE_ORDER[stage],
            )
            lr_pairs = select_stage_lr_pairs(lr_pairs_all, expression, int(args.max_lr_pairs))
            candidates = build_candidates(
                cells,
                expression,
                lr_pairs,
                dataset_key=replicate_name,
                max_candidates_per_lr=int(args.max_candidates_per_lr),
                max_candidates_per_sender_lr=int(args.max_candidates_per_sender_lr),
                seed=int(args.seed) + 1000 + replicate_index * 17,
            )
            cells.to_csv(replicate_dir / "cells.csv", index=False)
            expression.to_csv(replicate_dir / "expression.csv", index=False)
            lr_pairs.to_csv(replicate_dir / "lr_pairs.csv", index=False)
            candidates.to_csv(replicate_dir / "candidate_interactions.csv", index=False)
            metadata = {
                "dataset": "Cadinu 2024 mouse colon IBD MERFISH",
                "doi": "10.5061/dryad.rjdfn2zh3",
                "primary_article": "10.1016/j.cell.2024.03.013",
                "experiment": "multislice_stagewise_replication",
                "stage": stage,
                "stage_day": STAGE_ORDER[stage],
                "stage_replicate": stage_rep_idx,
                "source_slice_file": filename,
                "slice_id": slice_id,
                "mouse_id": mouse_id,
                "builder": "scripts/current/build_mouse_colon_ibd_merfish_multislice_replicates.py",
                "claim_boundary": "Two-slice-per-stage replication screen; not a full 1.35M-cell atlas.",
                "n_cells": int(len(cells)),
                "n_expression_genes": int(expression.shape[1] - 1),
                "n_lr_pairs": int(len(lr_pairs)),
                "n_candidates": int(len(candidates)),
            }
            (replicate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rows.append({"replicate_dir": str(replicate_dir), **metadata})
            replicate_index += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "multislice_replicate_manifest.csv", index=False)
    print(manifest[["stage", "stage_replicate", "mouse_id", "n_cells", "n_lr_pairs", "n_candidates"]].to_string(index=False))


def parse_mouse_id(slice_id: str) -> str:
    match = re.search(r"_D(?:0|3|9|21)_(m[^_]+)", slice_id)
    return match.group(1) if match else "unknown"


if __name__ == "__main__":
    main()
