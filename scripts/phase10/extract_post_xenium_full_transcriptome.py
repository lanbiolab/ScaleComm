#!/usr/bin/env python
"""Extract full-transcriptome counts for the post-Xenium colon subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    / "standardized_inputs/post_xenium_human_colon_cancer_subset"
)
MATRIX_H5 = (
    PROJECT_ROOT
    / "data/optional/post_xenium_human_colon_cancer/raw"
    / "CytAssist_FFPE_Human_Colon_Post_Xenium_Rep1_filtered_feature_bc_matrix.h5"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome",
    )
    args = parser.parse_args()

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    selected_barcodes = cells["cell_id"].astype(str).tolist()

    with h5py.File(MATRIX_H5, "r") as handle:
        matrix = handle["matrix"]
        all_barcodes = [x.decode() for x in matrix["barcodes"][:]]
        genes = [x.decode() for x in matrix["features"]["name"][:]]
        gene_ids = [x.decode() for x in matrix["features"]["id"][:]]
        feature_types = [x.decode() for x in matrix["features"]["feature_type"][:]]
        shape = tuple(int(x) for x in matrix["shape"][:])
        csc = sparse.csc_matrix(
            (
                matrix["data"][:],
                matrix["indices"][:],
                matrix["indptr"][:],
            ),
            shape=shape,
        )

    barcode_to_index = {barcode: idx for idx, barcode in enumerate(all_barcodes)}
    missing = [barcode for barcode in selected_barcodes if barcode not in barcode_to_index]
    if missing:
        raise ValueError(f"{len(missing)} selected barcodes missing from raw h5. First missing: {missing[:5]}")

    selected_indices = np.array([barcode_to_index[barcode] for barcode in selected_barcodes], dtype=int)
    subset = csc[:, selected_indices].T.tocsr()

    sparse.save_npz(out / "expression_full_transcriptome_counts.npz", subset)
    pd.DataFrame({"cell_id": selected_barcodes}).to_csv(out / "barcodes.csv", index=False)
    pd.DataFrame(
        {
            "gene": genes,
            "gene_id": gene_ids,
            "feature_type": feature_types,
        }
    ).to_csv(out / "genes.csv", index=False)

    nonzero_by_gene = np.asarray((subset > 0).sum(axis=0)).ravel()
    mean_counts = np.asarray(subset.mean(axis=0)).ravel()
    pd.DataFrame(
        {
            "n_cells": [subset.shape[0]],
            "n_genes": [subset.shape[1]],
            "nnz": [int(subset.nnz)],
            "density": [float(subset.nnz / (subset.shape[0] * subset.shape[1]))],
            "n_genes_detected_ge_10_cells": [int((nonzero_by_gene >= 10).sum())],
            "n_genes_detected_ge_25_cells": [int((nonzero_by_gene >= 25).sum())],
            "mean_library_size": [float(np.asarray(subset.sum(axis=1)).ravel().mean())],
        }
    ).to_csv(out / "full_transcriptome_summary.csv", index=False)

    manifest = {
        "source_h5": str(MATRIX_H5.relative_to(PROJECT_ROOT)),
        "source_dataset": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "counts_npz": str((out / "expression_full_transcriptome_counts.npz").relative_to(PROJECT_ROOT)),
        "barcodes": str((out / "barcodes.csv").relative_to(PROJECT_ROOT)),
        "genes": str((out / "genes.csv").relative_to(PROJECT_ROOT)),
        "summary": str((out / "full_transcriptome_summary.csv").relative_to(PROJECT_ROOT)),
        "orientation": "cells_by_genes",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
