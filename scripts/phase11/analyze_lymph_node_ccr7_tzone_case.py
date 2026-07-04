#!/usr/bin/env python
"""Analyze CCL19/CCL21-CCR7 T-zone communication in human lymph node predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def summarize_case(dataset_dir: Path, prediction_path: Path, output_dir: Path, method: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(dataset_dir / "cells.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    predictions = pd.read_csv(prediction_path)

    lr_lookup = lr_pairs.set_index("lr_id")["control_name"].to_dict()
    ann = lr_pairs.set_index("lr_id").to_dict(orient="index")
    cell_niche = cells.set_index("cell_id")["niche"].astype(str).to_dict()
    cell_region = cells.set_index("cell_id")["region"].astype(str).to_dict()

    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="left")
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    table["lr_pair"] = table["lr_id"].map(lr_lookup).fillna(table["ligand"].astype(str) + "-" + table["receptor"].astype(str))
    table["sender_niche"] = table["sender_id"].map(cell_niche).fillna(table["sender_niche"].astype(str))
    table["receiver_niche"] = table["receiver_id"].map(cell_niche).fillna(table["receiver_niche"].astype(str))
    table["receiver_region"] = table["receiver_id"].map(cell_region).fillna("")

    lr_summary = (
        table.groupby(["lr_id", "lr_pair"], as_index=False)
        .agg(
            n_edges=("candidate_id", "count"),
            score_sum=("score", "sum"),
            score_mean=("score", "mean"),
            score_q90=("score", lambda x: float(np.quantile(x, 0.9))),
            score_q95=("score", lambda x: float(np.quantile(x, 0.95))),
            tzone_receiver_score_sum=("score", lambda x: float(x[table.loc[x.index, "receiver_niche"].eq("T_cell_zone_proxy")].sum())),
            tzone_receiver_edge_fraction=("score", lambda x: float(table.loc[x.index, "receiver_niche"].eq("T_cell_zone_proxy").mean())),
            tzone_receiver_top10_fraction=(
                "score",
                lambda x: top_fraction(table.loc[x.index], "receiver_niche", "T_cell_zone_proxy", 0.9),
            ),
        )
    )
    lr_summary["score_sum_rank"] = lr_summary["score_sum"].rank(ascending=False, method="min").astype(int)
    lr_summary["score_q95_rank"] = lr_summary["score_q95"].rank(ascending=False, method="min").astype(int)
    lr_summary["tzone_score_fraction"] = lr_summary["tzone_receiver_score_sum"] / lr_summary["score_sum"].replace(0, np.nan)
    lr_summary["false_positive_high_expression"] = lr_summary["lr_id"].map(
        {lr_id: bool(meta.get("false_positive_high_expression", False)) for lr_id, meta in ann.items()}
    )
    lr_summary["control_class"] = lr_summary["lr_id"].map({lr_id: meta.get("control_class", "") for lr_id, meta in ann.items()})
    lr_summary["method"] = method

    flow = (
        table.groupby(["lr_pair", "sender_niche", "receiver_niche"], as_index=False)
        .agg(total_score=("score", "sum"), mean_score=("score", "mean"), n_edges=("candidate_id", "count"), median_distance=("distance", "median"))
    )
    flow["rank_within_lr"] = flow.groupby("lr_pair")["total_score"].rank(ascending=False, method="dense").astype(int)
    flow["method"] = method

    focus_pairs = ["CCL19-CCR7", "CCL21-CCR7", "EEF1A1-CCR7", "RPLP0-CXCR4", "ACTB-CD44"]
    focus_lr = lr_summary[lr_summary["lr_pair"].isin(focus_pairs)].sort_values(["score_sum_rank", "lr_pair"])
    focus_flow = flow[flow["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "EEF1A1-CCR7"])].sort_values(
        ["lr_pair", "rank_within_lr", "total_score"],
        ascending=[True, True, False],
    )

    lr_summary.to_csv(output_dir / f"{method}__lymph_node_lr_summary.csv", index=False)
    flow.to_csv(output_dir / f"{method}__lymph_node_niche_flow.csv", index=False)
    focus_lr.to_csv(output_dir / f"{method}__ccr7_focus_lr_summary.csv", index=False)
    focus_flow.to_csv(output_dir / f"{method}__ccr7_focus_niche_flow.csv", index=False)

    ccl19 = row_or_empty(focus_lr, "CCL19-CCR7")
    ccl21 = row_or_empty(focus_lr, "CCL21-CCR7")
    eef = row_or_empty(focus_lr, "EEF1A1-CCR7")
    result = {
        "method": method,
        "prediction_path": str(prediction_path),
        "n_lr_pairs": int(lr_summary["lr_id"].nunique()),
        "ccl19_ccr7_score_sum_rank": safe_int(ccl19.get("score_sum_rank")),
        "ccl19_ccr7_score_q95_rank": safe_int(ccl19.get("score_q95_rank")),
        "ccl19_ccr7_tzone_score_fraction": safe_float(ccl19.get("tzone_score_fraction")),
        "ccl19_ccr7_tzone_top10_fraction": safe_float(ccl19.get("tzone_receiver_top10_fraction")),
        "ccl21_ccr7_score_sum_rank": safe_int(ccl21.get("score_sum_rank")),
        "ccl21_ccr7_tzone_score_fraction": safe_float(ccl21.get("tzone_score_fraction")),
        "negative_eef1a1_ccr7_score_sum_rank": safe_int(eef.get("score_sum_rank")),
        "negative_eef1a1_ccr7_tzone_score_fraction": safe_float(eef.get("tzone_score_fraction")),
        "claim_boundary": (
            "Native human lymph-node Visium has no CCC ground truth. This summary tests whether known CCR7-axis "
            "chemokine priors are prioritized and localized to marker-rule T-zone receiver spots while receptor-shared "
            "high-expression controls remain lower or less specific."
        ),
    }
    (output_dir / f"{method}__ccr7_tzone_case_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def top_fraction(table: pd.DataFrame, column: str, value: str, quantile: float) -> float:
    if table.empty:
        return np.nan
    threshold = float(np.quantile(table["score"].to_numpy(dtype=float), quantile))
    top = table[table["score"].ge(threshold)]
    if top.empty:
        return np.nan
    return float(top[column].astype(str).eq(value).mean())


def row_or_empty(table: pd.DataFrame, lr_pair: str) -> dict:
    sub = table[table["lr_pair"].eq(lr_pair)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset",
    )
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/results/ccr7_tzone_case",
    )
    parser.add_argument("--method", default="spatialself_range_context")
    args = parser.parse_args()

    result = summarize_case(
        PROJECT_ROOT / args.dataset_dir,
        PROJECT_ROOT / args.prediction_path,
        PROJECT_ROOT / args.output_dir,
        args.method,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
