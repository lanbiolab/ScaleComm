#!/usr/bin/env python
"""Reproduce CellNEST T-cell-zone LR abundance tables using official annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = [
        "from_cell",
        "to_cell",
        "ligand",
        "receptor",
        "edge_rank",
        "component",
        "from_id",
        "to_id",
        "attention_score",
    ]
    if list(df.columns[: len(expected)]) != expected:
        df = pd.read_csv(path, header=None, names=expected)
        df = df[df["from_cell"].ne("from_cell")]
    df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
    return df


def load_annotation(path: Path) -> dict[str, str]:
    ann = pd.read_csv(path)
    if ann.iloc[0, 0] == "Barcode":
        ann = ann.iloc[1:].copy()
    ann.columns = ["Barcode", "Type"]
    return dict(zip(ann["Barcode"].astype(str), ann["Type"].astype(str)))


def summarize(path: Path, annotation: dict[str, str], output_dir: Path, label: str) -> pd.DataFrame:
    edges = load_edges(path)
    edges["from_type"] = edges["from_cell"].map(annotation).fillna("")
    edges["to_type"] = edges["to_cell"].map(annotation).fillna("")
    filtered = edges[(edges["from_type"].eq("T-cell")) | (edges["to_type"].eq("T-cell"))].copy()
    hist = (
        filtered.groupby("lr_pair", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "lr_pair"], ascending=[False, True])
        .reset_index(drop=True)
    )
    hist["rank"] = range(1, len(hist) + 1)
    hist.to_csv(output_dir / f"cellnest_official_tcell_histogram_{label}.csv", index=False)
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top20",
        default="/home/xd/y/BenchmarkBaselines/CellNEST/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_top20percent.csv",
    )
    parser.add_argument(
        "--allccc",
        default="/home/xd/y/BenchmarkBaselines/CellNEST/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_allCCC.csv",
    )
    parser.add_argument(
        "--annotation",
        default="/home/xd/y/BenchmarkBaselines/CellNEST/data/V1_Human_Lymph_Node_spatial_annotation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase11/official_method_reproduction/cellnest_human_lymph_node",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = load_annotation(Path(args.annotation))

    top20_hist = summarize(Path(args.top20), annotation, output_dir, "top20percent")
    allccc_hist = summarize(Path(args.allccc), annotation, output_dir, "allCCC")

    focus = {}
    for label, hist in [("top20percent", top20_hist), ("allCCC", allccc_hist)]:
        sub = hist[hist["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CCL21-CXCR4", "CCL19-CXCR4", "MIF-CXCR4"])]
        focus[label] = sub.to_dict(orient="records")

    print({
        "top20percent_focus": focus["top20percent"],
        "allCCC_focus": focus["allCCC"],
        "top20percent_path": str(output_dir / "cellnest_official_tcell_histogram_top20percent.csv"),
        "allCCC_path": str(output_dir / "cellnest_official_tcell_histogram_allCCC.csv"),
    })


if __name__ == "__main__":
    main()
