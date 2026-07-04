#!/usr/bin/env python
"""Scan current lymph-node predictions for receiver-region-specific LR axes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_METHODS = ("commot", "cellchat", "cellnest", "spatalk")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="outputs/phase12/atlas_model_expanded/replicates/human_lymph_node_visium_raw_full",
    )
    parser.add_argument(
        "--spatialself-prediction",
        default=(
            "outputs/current_lymph_node_region_discovery/current_model_scores/"
            "spatialself_range_context/"
            "human_lymph_node_visium_raw_full__spatialself_range_context_predictions.csv"
        ),
    )
    parser.add_argument("--spatialself-method", default="spatialself_range_context")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/phase12/atlas_model_expanded/official_adapter_baselines",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/current_lymph_node_region_discovery/region_scan",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-region-edges", type=int, default=20)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = candidates.merge(
        lr_pairs[
            [
                "lr_id",
                "control_name",
                "biological_category",
                "spatial_range",
                "active_niche",
                "source_count",
                "selected_reason",
            ]
        ],
        on="lr_id",
        how="left",
        suffixes=("", "_lrdb"),
    )
    candidates["lr_pair"] = candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)
    candidates["expr_product"] = np.log1p(candidates["ligand_expr"].astype(float)) * np.log1p(
        candidates["receptor_expr"].astype(float)
    )
    candidates["receiver_niche"] = candidates["receiver_niche"].fillna("unknown")

    method_order = (args.spatialself_method, *BASELINE_METHODS)
    prediction_paths = _prediction_paths(
        Path(args.baseline_dir),
        Path(args.spatialself_prediction),
        spatialself_method=args.spatialself_method,
    )
    long_table_parts = []
    method_status = []
    for method, path in prediction_paths.items():
        if not path.exists():
            method_status.append({"method": method, "status": "missing", "path": str(path), "n_rows": 0})
            continue
        pred = pd.read_csv(path, usecols=lambda col: col in {"candidate_id", "score", "method", "spatalk_status"})
        if "score" not in pred:
            method_status.append({"method": method, "status": "missing_score", "path": str(path), "n_rows": len(pred)})
            continue
        pred = pred[["candidate_id", "score"]].copy()
        pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0.0)
        merged = candidates.merge(pred, on="candidate_id", how="left")
        merged["score"] = merged["score"].fillna(0.0)
        merged["method"] = method
        long_table_parts.append(merged)
        nonzero = int((merged["score"] > 0).sum())
        method_status.append(
            {
                "method": method,
                "status": "loaded",
                "path": str(path),
                "n_rows": int(len(pred)),
                "n_candidates_scored": int(len(merged)),
                "n_nonzero_scores": nonzero,
                "nonzero_fraction": nonzero / max(len(merged), 1),
            }
        )

    if not long_table_parts:
        raise RuntimeError("No prediction tables could be loaded.")
    long_table = pd.concat(long_table_parts, ignore_index=True)
    pd.DataFrame(method_status).to_csv(output_dir / "method_score_status.csv", index=False)

    region_lr = _region_lr_summary(long_table, min_region_edges=args.min_region_edges)
    region_lr.to_csv(output_dir / "region_lr_method_summary_long.csv", index=False)

    rank_wide = _rank_wide(region_lr, spatialself_method=args.spatialself_method, method_order=method_order)
    rank_wide.to_csv(output_dir / "region_lr_rank_comparison_wide.csv", index=False)

    candidates_table = _spatialself_specific_candidates(
        rank_wide,
        top_n=args.top_n,
        spatialself_method=args.spatialself_method,
    )
    candidates_table.to_csv(output_dir / "spatialself_specific_region_candidates.csv", index=False)

    long_range_candidates = candidates_table[
        candidates_table["mechanism"].fillna("").str.contains("long", case=False)
        | candidates_table["biological_category"].fillna("").str.contains("chemokine", case=False)
    ].copy()
    long_range_candidates.to_csv(output_dir / "spatialself_specific_long_or_chemokine_candidates.csv", index=False)

    top_by_region = rank_wide.sort_values(["receiver_niche", f"{args.spatialself_method}_rank_q95"]).groupby(
        "receiver_niche", as_index=False
    ).head(args.top_n)
    top_by_region.to_csv(output_dir / "spatialself_top_region_lr_by_receiver_niche.csv", index=False)

    print(f"Wrote region scan outputs to {output_dir}")
    print("Top SpatialSelf-specific region candidates:")
    cols = [
        "receiver_niche",
        "lr_pair",
        "mechanism",
        f"{args.spatialself_method}_rank_q95",
        "commot_rank_q95",
        "cellchat_rank_q95",
        "cellnest_rank_q95",
        "spatalk_rank_q95",
        f"{args.spatialself_method}_score_q95",
        "spatialself_region_lift",
        "expr_product_mean",
    ]
    if candidates_table.empty:
        print("  none under current filters")
    else:
        print(candidates_table[cols].head(30).to_string(index=False))


def _prediction_paths(baseline_dir: Path, spatialself_prediction: Path, *, spatialself_method: str) -> dict[str, Path]:
    paths = {spatialself_method: spatialself_prediction}
    for method in BASELINE_METHODS:
        paths[method] = baseline_dir / method / f"human_lymph_node_visium_raw_full__{method}_predictions.csv"
    return paths


def _region_lr_summary(long_table: pd.DataFrame, *, min_region_edges: int) -> pd.DataFrame:
    group_cols = [
        "method",
        "receiver_niche",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "mechanism",
        "biological_category",
        "spatial_range",
        "active_niche",
        "true_lr",
        "false_positive_high_expression",
    ]
    summary = (
        long_table.groupby(group_cols, dropna=False)
        .agg(
            n_edges=("candidate_id", "size"),
            score_sum=("score", "sum"),
            score_mean=("score", "mean"),
            score_q90=("score", lambda x: float(np.quantile(x, 0.90))),
            score_q95=("score", lambda x: float(np.quantile(x, 0.95))),
            score_max=("score", "max"),
            expr_product_mean=("expr_product", "mean"),
            ligand_expr_mean=("ligand_expr", "mean"),
            receptor_expr_mean=("receptor_expr", "mean"),
            expressed_edge_fraction=("expr_product", lambda x: float((x > 0).mean())),
            median_distance=("distance", "median"),
        )
        .reset_index()
    )
    summary = summary[summary["n_edges"] >= int(min_region_edges)].copy()

    total = (
        summary.groupby(["method", "lr_id"], dropna=False)["score_sum"]
        .sum()
        .rename("lr_total_score_sum")
        .reset_index()
    )
    region_total = (
        summary.groupby(["method", "receiver_niche"], dropna=False)["score_sum"]
        .sum()
        .rename("region_total_score_sum")
        .reset_index()
    )
    summary = summary.merge(total, on=["method", "lr_id"], how="left")
    summary = summary.merge(region_total, on=["method", "receiver_niche"], how="left")
    summary["region_score_fraction_within_lr"] = summary["score_sum"] / summary["lr_total_score_sum"].replace(0, np.nan)
    summary["lr_score_fraction_within_region"] = summary["score_sum"] / summary["region_total_score_sum"].replace(0, np.nan)

    all_region_fraction = (
        summary.groupby(["method", "receiver_niche"], dropna=False)["score_sum"].sum()
        / summary.groupby("method", dropna=False)["score_sum"].sum()
    ).rename("region_background_fraction").reset_index()
    summary = summary.merge(all_region_fraction, on=["method", "receiver_niche"], how="left")
    summary["region_lift"] = summary["region_score_fraction_within_lr"] / summary["region_background_fraction"].replace(
        0, np.nan
    )

    rank_metrics = ["score_q95", "score_sum", "score_mean", "score_max"]
    for metric in rank_metrics:
        summary[f"rank_{metric.removeprefix('score_')}"] = summary.groupby(
            ["method", "receiver_niche"], dropna=False
        )[metric].rank(method="min", ascending=False)
    return summary


def _rank_wide(region_lr: pd.DataFrame, *, spatialself_method: str, method_order: tuple[str, ...]) -> pd.DataFrame:
    keep_meta = [
        "receiver_niche",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "mechanism",
        "biological_category",
        "spatial_range",
        "active_niche",
        "true_lr",
        "false_positive_high_expression",
    ]
    meta = region_lr[keep_meta].drop_duplicates()
    value_cols = [
        "n_edges",
        "score_sum",
        "score_mean",
        "score_q90",
        "score_q95",
        "score_max",
        "expr_product_mean",
        "ligand_expr_mean",
        "receptor_expr_mean",
        "expressed_edge_fraction",
        "median_distance",
        "region_score_fraction_within_lr",
        "lr_score_fraction_within_region",
        "region_lift",
        "rank_q95",
        "rank_sum",
        "rank_mean",
        "rank_max",
    ]
    pieces = [meta]
    for method in method_order:
        sub = region_lr[region_lr["method"] == method][["receiver_niche", "lr_id", *value_cols]].copy()
        sub = sub.rename(columns={col: f"{method}_{col}" for col in value_cols})
        pieces.append(sub)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=["receiver_niche", "lr_id"], how="left")

    # Shared biology/expression fields are identical across methods after candidate merging; keep one readable copy.
    for col in ["expr_product_mean", "ligand_expr_mean", "receptor_expr_mean", "expressed_edge_fraction", "median_distance"]:
        src = f"{spatialself_method}_{col}"
        if src in out:
            out[col] = out[src]
    out["spatialself_region_lift"] = out[f"{spatialself_method}_region_lift"]
    baseline_rank_cols = [f"{m}_rank_q95" for m in BASELINE_METHODS if f"{m}_rank_q95" in out]
    out["best_baseline_rank_q95"] = out[baseline_rank_cols].min(axis=1, skipna=True)
    out["second_best_baseline_rank_q95"] = out[baseline_rank_cols].apply(
        lambda row: _nth_smallest(row.dropna().to_numpy(), 2), axis=1
    )
    out["rank_gap_vs_best_baseline"] = out["best_baseline_rank_q95"] - out[f"{spatialself_method}_rank_q95"]
    return out


def _nth_smallest(values: np.ndarray, n: int) -> float:
    if len(values) < n:
        return np.nan
    return float(np.sort(values)[n - 1])


def _spatialself_specific_candidates(rank_wide: pd.DataFrame, *, top_n: int, spatialself_method: str) -> pd.DataFrame:
    out = rank_wide.copy()
    baseline_cols = [f"{m}_rank_q95" for m in ("commot", "cellchat", "cellnest") if f"{m}_rank_q95" in out]
    for col in baseline_cols:
        out[col] = out[col].fillna(np.inf)
    mask = (
        (out[f"{spatialself_method}_rank_q95"] <= int(top_n))
        & (out["expr_product_mean"] > 0)
        & (out["expressed_edge_fraction"] >= 0.05)
        & (out[f"{spatialself_method}_score_q95"] > 0)
    )
    if baseline_cols:
        mask &= (out[baseline_cols] > int(top_n)).all(axis=1)
    out = out[mask].copy()
    out = out.sort_values(
        [
            "receiver_niche",
            f"{spatialself_method}_rank_q95",
            "rank_gap_vs_best_baseline",
            f"{spatialself_method}_score_q95",
        ],
        ascending=[True, True, False, False],
    )
    return out


if __name__ == "__main__":
    main()
