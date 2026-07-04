#!/usr/bin/env python
"""Compute all-LR region enrichment from candidate-level predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import bh_adjust


def run_region_enrichment(
    candidate_path: Path,
    prediction_path: Path,
    cells_path: Path,
    output_dir: Path,
    *,
    region_column: str = "region",
    receiver_region: bool = True,
    score_column: str = "score",
    n_permutations: int = 100,
    random_seed: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    cells = pd.read_csv(cells_path)
    table = _merge_inputs(
        candidates,
        predictions,
        cells,
        region_column=region_column,
        receiver_region=receiver_region,
        score_column=score_column,
    )

    enrichment = _region_enrichment_table(
        table,
        score_column=score_column,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    ranking = _lr_ranking_table(table, score_column=score_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    enrichment_path = output_dir / f"{prediction_path.stem}__region_enrichment.csv"
    ranking_path = output_dir / f"{prediction_path.stem}__lr_global_ranking.csv"
    summary_path = output_dir / f"{prediction_path.stem}__region_enrichment_summary.json"
    enrichment.to_csv(enrichment_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    summary = {
        "candidate_path": str(candidate_path),
        "prediction_path": str(prediction_path),
        "cells_path": str(cells_path),
        "region_column": region_column,
        "region_on": "receiver" if receiver_region else "sender",
        "n_edges": int(len(table)),
        "n_lr_pairs": int(table["lr_id"].nunique()),
        "n_regions": int(table["region"].nunique()),
        "enrichment_path": str(enrichment_path),
        "ranking_path": str(ranking_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return enrichment, ranking


def _merge_inputs(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    region_column: str,
    receiver_region: bool,
    score_column: str,
) -> pd.DataFrame:
    required_candidates = {"candidate_id", "sender_id", "receiver_id", "lr_id", "ligand", "receptor", "mechanism"}
    missing_candidates = required_candidates - set(candidates.columns)
    if missing_candidates:
        raise ValueError(f"Candidate table missing columns: {sorted(missing_candidates)}")
    required_predictions = {"candidate_id", score_column}
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"Prediction table missing columns: {sorted(missing_predictions)}")
    required_cells = {"cell_id", region_column}
    missing_cells = required_cells - set(cells.columns)
    if missing_cells:
        raise ValueError(f"Cells table missing columns: {sorted(missing_cells)}")

    region_cell_id = "receiver_id" if receiver_region else "sender_id"
    region = cells[["cell_id", region_column]].rename(
        columns={"cell_id": region_cell_id, region_column: "region"}
    )
    prediction_columns = ["candidate_id", score_column]
    if "method" in predictions.columns:
        prediction_columns.append("method")
    table = candidates.merge(
        predictions[prediction_columns],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    ).merge(region, on=region_cell_id, how="left", validate="many_to_one")
    table["region"] = table["region"].fillna("Unknown").astype(str)
    table[score_column] = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0)
    return table


def _region_enrichment_table(
    table: pd.DataFrame,
    *,
    score_column: str,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    observed = _observed_region_stats(table, score_column=score_column)
    rng = np.random.default_rng(random_seed)
    exceed = np.zeros(len(observed), dtype=int)
    key_columns = ["lr_id", "region"]
    observed_lookup = {
        tuple(row[key] for key in key_columns): idx for idx, row in observed.iterrows()
    }

    shuffled = table.copy()
    for _ in range(int(n_permutations)):
        shuffled["region"] = rng.permutation(table["region"].to_numpy())
        null_stats = _observed_region_stats(shuffled, score_column=score_column)
        for _, row in null_stats.iterrows():
            idx = observed_lookup.get((row["lr_id"], row["region"]))
            if idx is None:
                continue
            exceed[idx] += float(row["region_enrichment_score"]) >= float(
                observed.iloc[idx]["region_enrichment_score"]
            )

    observed["empirical_p_value"] = (exceed + 1.0) / (int(n_permutations) + 1.0)
    observed["bh_fdr"] = bh_adjust(observed["empirical_p_value"].to_numpy(dtype=float))
    observed["n_permutations"] = int(n_permutations)
    return observed.sort_values(["bh_fdr", "region_enrichment_score"], ascending=[True, False])


def _observed_region_stats(table: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    group_columns = ["lr_id", "ligand", "receptor", "mechanism"]
    if "method" in table.columns:
        group_columns.append("method")
    global_lr = table.groupby(group_columns, sort=False)[score_column].mean()
    rows = []
    for group_key, group in table.groupby(group_columns + ["region"], sort=False):
        lr_key = group_key[: len(group_columns)]
        region = group_key[len(group_columns)]
        global_mean = float(global_lr.loc[lr_key])
        region_mean = float(group[score_column].mean())
        enrichment = region_mean - global_mean
        row = {
            "lr_id": lr_key[0],
            "ligand": lr_key[1],
            "receptor": lr_key[2],
            "mechanism": lr_key[3],
            "region": region,
            "n_edges_in_region": int(len(group)),
            "region_mean_score": region_mean,
            "lr_global_mean_score": global_mean,
            "region_enrichment_score": enrichment,
        }
        if "method" in group_columns:
            row["method"] = lr_key[group_columns.index("method")]
        rows.append(row)
    return pd.DataFrame(rows)


def _lr_ranking_table(table: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    rows = []
    for group_key, group in table.groupby(["lr_id", "ligand", "receptor", "mechanism"], sort=False):
        top_region = (
            group.groupby("region", sort=False)[score_column]
            .mean()
            .sort_values(ascending=False)
        )
        rows.append(
            {
                "lr_id": group_key[0],
                "ligand": group_key[1],
                "receptor": group_key[2],
                "mechanism": group_key[3],
                "method": str(group["method"].iloc[0]) if "method" in group.columns else "unknown",
                "n_edges": int(len(group)),
                "mean_score": float(group[score_column].mean()),
                "max_score": float(group[score_column].max()),
                "top_region": str(top_region.index[0]) if len(top_region) else "Unknown",
                "top_region_mean_score": float(top_region.iloc[0]) if len(top_region) else float("nan"),
            }
        )
    result = pd.DataFrame(rows)
    result["global_rank"] = result["mean_score"].rank(ascending=False, method="min").astype(int)
    return result.sort_values(["global_rank", "mean_score"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV path.")
    parser.add_argument("--cells", required=True, help="cells.csv path with region annotations.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--sender-region", action="store_true")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    args = parser.parse_args()

    enrichment, ranking = run_region_enrichment(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.cells),
        Path(args.output_dir),
        region_column=args.region_column,
        receiver_region=not args.sender_region,
        score_column=args.score_column,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
    )
    print(enrichment.head(10).to_string(index=False))
    print(f"Wrote region enrichment outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
