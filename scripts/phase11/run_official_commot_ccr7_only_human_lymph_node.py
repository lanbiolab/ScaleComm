#!/usr/bin/env python
"""Run a CCR7-axis-only COMMOT official API sanity check.

This is not a full-database ranking run. It uses COMMOT's public API with the
two CCR7 chemokine pairs to inspect spatial localization while the full
CellChat run is pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import commot as ct
import pandas as pd
import scanpy as sc

from run_official_commot_human_lymph_node import PROJECT_ROOT, load_tzone_labels, summarize_commot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/optional/human_lymph_node_visium_full/raw")
    parser.add_argument("--dataset-dir", default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_all_spots/dataset")
    parser.add_argument("--output-dir", default="outputs/phase11/official_method_reproduction/commot_human_lymph_node_ccr7_only")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_visium(path=str(raw_dir), count_file="V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    labels = load_tzone_labels(dataset_dir, adata.obs_names)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    df_ligrec = pd.DataFrame(
        [["CCL19", "CCR7", "CCL", "Secreted Signaling"], ["CCL21", "CCR7", "CCL", "Secreted Signaling"]]
    )
    ct.tl.spatial_communication(
        adata,
        database_name="CCR7_axis",
        df_ligrec=df_ligrec,
        dis_thr=200,
        heteromeric=True,
    )
    result = summarize_commot(adata, "CCR7_axis", labels, output_dir)
    result["ranking_scope"] = "CCR7-axis only; do not compare as full LR rank."
    (output_dir / "commot_ccr7_only_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
