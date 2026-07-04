#!/usr/bin/env python
"""Extract full-transcriptome counts for the Visium HD CRC P2CRC subset."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_H5 = PROJECT_ROOT / "data/optional/visium_hd_crc/raw/GSM8594568_P2CRC/GSM8594568_P2CRC_filtered_feature_bc_matrix.h5"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    / "standardized_inputs/visium_hd_crc_p2crc_subset"
)
OUT_DIR = (
    PROJECT_ROOT
    / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    wanted = cells["cell_id"].astype(str).tolist()

    with h5py.File(RAW_H5, "r") as handle:
        matrix = handle["matrix"]
        all_barcodes = [x.decode("utf-8") for x in matrix["barcodes"][:]]
        barcode_to_col = {barcode: idx for idx, barcode in enumerate(all_barcodes)}
        missing = [barcode for barcode in wanted if barcode not in barcode_to_col]
        if missing:
            raise ValueError(f"{len(missing)} subset barcodes are absent from raw h5; first={missing[:5]}")
        selected_cols = np.asarray([barcode_to_col[barcode] for barcode in wanted], dtype=np.int64)
        raw_csc = sparse.csc_matrix(
            (
                matrix["data"][:],
                matrix["indices"][:],
                matrix["indptr"][:],
            ),
            shape=tuple(matrix["shape"][:]),
        )
        subset = raw_csc[:, selected_cols].T.tocsr()
        genes = pd.DataFrame(
            {
                "gene_id": [x.decode("utf-8") for x in matrix["features"]["id"][:]],
                "gene": [x.decode("utf-8") for x in matrix["features"]["name"][:]],
                "feature_type": [x.decode("utf-8") for x in matrix["features"]["feature_type"][:]],
            }
        )

    sparse.save_npz(OUT_DIR / "expression_full_transcriptome_counts.npz", subset)
    pd.DataFrame({"cell_id": wanted}).to_csv(OUT_DIR / "barcodes.csv", index=False)
    genes.to_csv(OUT_DIR / "genes.csv", index=False)

    detected = np.asarray((subset > 0).sum(axis=0)).ravel()
    library_size = np.asarray(subset.sum(axis=1)).ravel()
    summary = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "raw_h5": str(RAW_H5.relative_to(PROJECT_ROOT)),
        "n_spots": int(subset.shape[0]),
        "n_genes": int(subset.shape[1]),
        "nnz": int(subset.nnz),
        "density": float(subset.nnz / (subset.shape[0] * subset.shape[1])),
        "genes_detected_ge_10_spots": int(np.sum(detected >= 10)),
        "genes_detected_ge_25_spots": int(np.sum(detected >= 25)),
        "mean_library_size": float(np.mean(library_size)),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "full_transcriptome_summary.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
