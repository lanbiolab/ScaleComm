#!/usr/bin/env python
"""Prepare lymph-node 10x H5 full-transcriptome inputs for Phase 12 pilots."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
H5_PATH = PROJECT_ROOT / "data/optional/human_lymph_node_visium_full/raw/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_PATH, "r") as h5:
        matrix = h5["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][:])
        features_by_barcodes = sparse.csc_matrix(
            (
                matrix["data"][:],
                matrix["indices"][:],
                matrix["indptr"][:],
            ),
            shape=shape,
        )
        counts = features_by_barcodes.T.tocsr()
        barcodes = decode_array(matrix["barcodes"][:])
        genes = decode_array(matrix["features/name"][:])
        gene_ids = decode_array(matrix["features/id"][:])
        feature_types = decode_array(matrix["features/feature_type"][:])

    sparse.save_npz(OUT_DIR / "expression_full_transcriptome_counts.npz", counts)
    pd.DataFrame({"cell_id": barcodes}).to_csv(OUT_DIR / "barcodes.csv", index=False)
    pd.DataFrame({"gene": [g.upper() for g in genes], "gene_id": gene_ids, "feature_type": feature_types}).to_csv(
        OUT_DIR / "genes.csv",
        index=False,
    )
    manifest = {
        "task": "prepare_phase12_lymph_node_full_tx",
        "source_h5": rel(H5_PATH),
        "matrix_orientation": "rows are spots/barcodes, columns are genes",
        "n_barcodes": len(barcodes),
        "n_genes": len(genes),
        "outputs": [
            rel(OUT_DIR / "expression_full_transcriptome_counts.npz"),
            rel(OUT_DIR / "barcodes.csv"),
            rel(OUT_DIR / "genes.csv"),
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def decode_array(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
