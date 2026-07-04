#!/usr/bin/env python
"""Extract CCR7-axis ranks from official CellNEST output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["from_cell", "to_cell", "ligand", "receptor", "edge_rank", "component", "from_id", "to_id", "attention_score"]
    if list(df.columns[: len(expected)]) != expected:
        df = pd.read_csv(path, header=None, names=expected)
        df = df[df["from_cell"].ne("from_cell")]
    for col in ["edge_rank", "attention_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["ligand", "receptor"])


def summarize(df: pd.DataFrame, output_dir: Path) -> dict:
    df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
    lr_summary = (
        df.groupby("lr_pair", as_index=False)
        .agg(
            n_edges=("lr_pair", "size"),
            min_edge_rank=("edge_rank", "min"),
            median_edge_rank=("edge_rank", "median"),
            mean_attention=("attention_score", "mean"),
            q95_attention=("attention_score", lambda x: float(np.quantile(pd.to_numeric(x, errors="coerce").fillna(0.0), 0.95))),
        )
    )
    lr_summary["min_edge_rank_lr_rank"] = lr_summary["min_edge_rank"].rank(ascending=True, method="min").astype(int)
    lr_summary["mean_attention_lr_rank"] = lr_summary["mean_attention"].rank(ascending=False, method="min").astype(int)
    lr_summary = lr_summary.sort_values(["min_edge_rank_lr_rank", "lr_pair"])
    focus = lr_summary[lr_summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CCR7-CCL19", "CCR7-CCL21"])]
    lr_summary.to_csv(output_dir / "cellnest_official_lr_summary.csv", index=False)
    focus.to_csv(output_dir / "cellnest_official_ccr7_focus.csv", index=False)

    result = {
        "n_edges": int(len(df)),
        "n_lr_pairs": int(lr_summary["lr_pair"].nunique()),
        "rank_definition": "LR pair rank by minimum official CellNEST edge_rank across edges; lower is stronger. Also reports mean attention rank.",
    }
    for pair in ["CCL19-CCR7", "CCL21-CCR7", "CCR7-CCL19", "CCR7-CCL21"]:
        row = focus[focus["lr_pair"].eq(pair)]
        key = pair.lower().replace("-", "_")
        if len(row):
            item = row.iloc[0].to_dict()
            result[f"{key}_min_edge_rank_lr_rank"] = int(item["min_edge_rank_lr_rank"])
            result[f"{key}_mean_attention_lr_rank"] = int(item["mean_attention_lr_rank"])
            result[f"{key}_n_edges"] = int(item["n_edges"])
            result[f"{key}_min_edge_rank"] = float(item["min_edge_rank"])
        else:
            result[f"{key}_min_edge_rank_lr_rank"] = None
    (output_dir / "cellnest_official_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cellnest-output", required=True)
    parser.add_argument("--output-dir", default="outputs/phase11/official_method_reproduction/cellnest_human_lymph_node")
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(load_table(Path(args.cellnest_output)), output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
