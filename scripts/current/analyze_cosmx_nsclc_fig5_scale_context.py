#!/usr/bin/env python
"""Analyze CosMx NSCLC SpatialSelf-scale results for the new Fig. 5 story.

This script is intentionally analysis-only. It does not train the model; it
combines standardized CosMx inputs, SpatialSelf predictions, learned LR-scale
profiles, and curated target signatures into source tables that can drive the
standalone Fig. 5 mechanism story.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs/current_fig5_cosmx_nsclc_scale_context/standardized_inputs/cosmx_nsclc_he_subset_9000"
)
DEFAULT_RUN = PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/spatialself_scale_9000_e120"
DEFAULT_OUT = PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables"
DEFAULT_BASELINE_DATASET = (
    PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/standardized_inputs/cosmx_nsclc_he_subset_9000"
)
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context"
DEFAULT_BASELINE_OUTPUT_DIR = DEFAULT_BASELINE_ROOT / "baselines_9000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--prediction-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--baseline-dataset-dir", default=str(DEFAULT_BASELINE_DATASET))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--baseline-output-dir", default=str(DEFAULT_BASELINE_OUTPUT_DIR))
    parser.add_argument("--method-support-label", default="9000")
    parser.add_argument("--top-fraction", type=float, default=0.02)
    parser.add_argument("--top-n-receivers", type=int, default=300)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    target_signatures = pd.read_csv(dataset_dir / "target_signatures.csv")

    predictions = load_predictions(dataset_dir.name, prediction_dir)
    scale = load_scale_distribution(dataset_dir.name, prediction_dir)
    scored = (
        candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", validate="one_to_one")
        .merge(
            lr_pairs[
                [
                    "lr_id",
                    "control_name",
                    "mechanism",
                    "spatial_range",
                    "active_niche",
                    "biological_category",
                    "selected_reason",
                ]
            ],
            on="lr_id",
            how="left",
            suffixes=("", "_lr"),
        )
    )

    target_scores = build_target_score_table(expression, target_signatures)
    lr_summary = summarize_lr(scored, scale, top_fraction=float(args.top_fraction))
    receiver_support = summarize_receiver_support(
        scored=scored,
        cells=cells,
        target_scores=target_scores,
        top_n=int(args.top_n_receivers),
    )
    edge_distance = summarize_edge_distance(scored)
    hotspot_cells = build_hotspot_cell_table(
        scored=scored,
        cells=cells,
        target_scores=target_scores,
        top_n=int(args.top_n_receivers),
    )
    story_candidates = rank_story_candidates(lr_summary, receiver_support)
    method_support = build_method_support_table(
        dataset_dir=Path(args.baseline_dataset_dir).resolve(),
        baseline_root=Path(args.baseline_root).resolve(),
        baseline_output_dir=Path(args.baseline_output_dir).resolve(),
        top_n=int(args.top_n_receivers),
    )

    cells.to_csv(output_dir / "cosmx_cells_for_context_map.csv", index=False)
    scored.to_csv(output_dir / "cosmx_spatialself_scored_candidates.csv", index=False)
    scale.to_csv(output_dir / "cosmx_lr_scale_distribution.csv", index=False)
    lr_summary.to_csv(output_dir / "cosmx_lr_scale_context_summary.csv", index=False)
    receiver_support.to_csv(output_dir / "cosmx_receiver_target_support.csv", index=False)
    edge_distance.to_csv(output_dir / "cosmx_edge_distance_summary.csv", index=False)
    hotspot_cells.to_csv(output_dir / "cosmx_hotspot_receiver_cells.csv", index=False)
    story_candidates.to_csv(output_dir / "cosmx_fig5_story_candidate_ranking.csv", index=False)
    method_support_name = f"cosmx_{args.method_support_label}_method_receiver_support.csv"
    method_support.to_csv(output_dir / method_support_name, index=False)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "n_cells": int(len(cells)),
        "n_candidates": int(len(scored)),
        "n_lr_pairs": int(lr_pairs.shape[0]),
        "top_fraction": float(args.top_fraction),
        "top_n_receivers": int(args.top_n_receivers),
        "claim_boundary": (
            "CosMx cell states are coarse proxy annotations. Receiver target signatures "
            "are curated marker/response proxies from genes present in the CosMx panel, "
            "not full NicheNet/Reactome signatures."
        ),
        "primary_tables": [
            "cosmx_lr_scale_context_summary.csv",
            "cosmx_receiver_target_support.csv",
            "cosmx_fig5_story_candidate_ranking.csv",
            "cosmx_hotspot_receiver_cells.csv",
            method_support_name,
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nTop Fig5 story candidates")
    cols = [
        "LR",
        "mechanism",
        "mean_score",
        "expected_scale",
        "scale_class",
        "top_receiver_target_lift",
        "target_spearman_r",
        "top_niche_match_rate",
        "story_score",
    ]
    print(story_candidates[cols].round(4).to_string(index=False))
    if not method_support.empty:
        print(f"\n{args.method_support_label}-cell baseline support snapshot")
        keep = [
            "method",
            "LR",
            "rank_by_mean_score",
            "mean_score",
            "top_receiver_target_lift",
            "target_spearman_r",
            "top_receiver_niche_mode",
        ]
        print(
            method_support[method_support["LR"].isin(["CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2", "SPP1-CD44"])]
            [keep]
            .round(4)
            .to_string(index=False)
        )
    print(f"\nWrote source tables to {output_dir}")


def load_predictions(dataset_name: str, prediction_dir: Path) -> pd.DataFrame:
    path = prediction_dir / f"{dataset_name}__spatialself_wavelet_scale_predictions.csv"
    if not path.exists():
        matches = sorted(prediction_dir.glob("*__spatialself_wavelet_scale_predictions.csv"))
        if not matches:
            raise FileNotFoundError(f"No SpatialSelf prediction file found under {prediction_dir}")
        path = matches[0]
    pred = pd.read_csv(path)
    if {"candidate_id", "score"} - set(pred.columns):
        raise ValueError(f"Prediction file missing required columns: {path}")
    return pred


def load_scale_distribution(dataset_name: str, prediction_dir: Path) -> pd.DataFrame:
    path = prediction_dir / f"{dataset_name}__spatialself_wavelet_scale_lr_scale_distribution.csv"
    if not path.exists():
        matches = sorted(prediction_dir.glob("*__spatialself_wavelet_scale_lr_scale_distribution.csv"))
        if not matches:
            raise FileNotFoundError(f"No LR scale distribution file found under {prediction_dir}")
        path = matches[0]
    return pd.read_csv(path)


def build_target_score_table(expression: pd.DataFrame, target_signatures: pd.DataFrame) -> pd.DataFrame:
    expr = expression.set_index("cell_id")
    rows = []
    for row in target_signatures.to_dict("records"):
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        lr_pair = f"{ligand}-{receptor}"
        raw_targets = str(row.get("targets", row.get("target_genes", ""))).replace(";", ",").split(",")
        targets = [gene.strip() for gene in raw_targets if gene.strip() in expr.columns]
        if not targets:
            continue
        values = np.log1p(expr[targets].to_numpy(dtype=float)).mean(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": expr.index.to_numpy(),
                    "ligand": ligand,
                    "receptor": receptor,
                    "LR": lr_pair,
                    "target_score": values,
                    "n_targets_present": len(targets),
                    "targets_present": ",".join(targets),
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["cell_id", "ligand", "receptor", "LR", "target_score", "n_targets_present", "targets_present"]
        )
    return pd.concat(rows, ignore_index=True)


def summarize_lr(scored: pd.DataFrame, scale: pd.DataFrame, *, top_fraction: float) -> pd.DataFrame:
    rows = []
    for lr_id, group in scored.groupby("lr_id", sort=False):
        top_n = max(100, int(np.ceil(len(group) * top_fraction)))
        top = group.nlargest(top_n, "score")
        lr_name = str(group["control_name"].iloc[0])
        rows.append(
            {
                "lr_id": lr_id,
                "LR": lr_name,
                "ligand": str(group["ligand"].iloc[0]),
                "receptor": str(group["receptor"].iloc[0]),
                "mechanism": str(group["mechanism_lr"].iloc[0])
                if "mechanism_lr" in group
                else str(group["mechanism"].iloc[0]),
                "biological_category": str(group["biological_category"].iloc[0]),
                "active_niche": str(group["active_niche"].iloc[0]),
                "n_candidates": int(len(group)),
                "mean_score": float(group["score"].mean()),
                "q90_score": float(group["score"].quantile(0.90)),
                "q95_score": float(group["score"].quantile(0.95)),
                "max_score": float(group["score"].max()),
                "top_n_edges": int(top_n),
                "top_median_distance": float(top["distance"].median()),
                "top_q25_distance": float(top["distance"].quantile(0.25)),
                "top_q75_distance": float(top["distance"].quantile(0.75)),
                "top_niche_match_rate": safe_mean(top.get("niche_rule_satisfied")),
                "top_type_match_rate": safe_mean(top.get("type_rule_satisfied")),
                "top_range_match_rate": safe_mean(top.get("range_rule_satisfied")),
                "top_sender_niche_mode": mode_label(top["sender_niche"]),
                "top_receiver_niche_mode": mode_label(top["receiver_niche"]),
                "top_sender_cell_type_mode": mode_label(top["sender_cell_type"]),
                "top_receiver_cell_type_mode": mode_label(top["receiver_cell_type"]),
            }
        )
    summary = pd.DataFrame(rows)
    if "lr_pair" in scale.columns:
        scale_use = scale.rename(columns={"lr_pair": "LR"}).copy()
    else:
        scale_use = scale.copy()
        scale_use["LR"] = scale_use["ligand"].astype(str) + "-" + scale_use["receptor"].astype(str)
    scale_cols = [
        "LR",
        "expected_scale",
        "local_weight",
        "broad_weight",
        "target_expected_scale",
        "target_local_weight",
        "target_broad_weight",
        *[col for col in scale_use.columns if col.startswith("scale_")],
    ]
    scale_cols = [col for col in scale_cols if col in scale_use.columns]
    summary = summary.merge(scale_use[scale_cols], on="LR", how="left")
    summary["scale_class"] = summary["expected_scale"].map(scale_class)
    return summary.sort_values(["mean_score", "max_score"], ascending=False).reset_index(drop=True)


def summarize_receiver_support(
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    target_scores: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    cell_meta = cells.set_index("cell_id")[["cell_type", "niche"]]
    for lr_id, group in scored.groupby("lr_id", sort=False):
        lr_name = str(group["control_name"].iloc[0])
        receiver_activity = (
            group.groupby("receiver_id", as_index=False)
            .agg(
                receiver_activity=("score", "max"),
                receiver_mean_activity=("score", "mean"),
                n_incoming=("candidate_id", "size"),
            )
            .rename(columns={"receiver_id": "cell_id"})
        )
        target = target_scores[target_scores["LR"].eq(lr_name)].copy()
        if target.empty:
            continue
        merged = receiver_activity.merge(target[["cell_id", "target_score"]], on="cell_id", how="left")
        merged["target_score"] = merged["target_score"].fillna(0.0)
        merged = merged.join(cell_meta, on="cell_id")
        top = merged.nlargest(min(top_n, len(merged)), "receiver_activity")
        rest = merged.drop(index=top.index)
        if len(rest) == 0:
            rest = merged
        rho, pval = safe_spearman(merged["receiver_activity"], merged["target_score"])
        lift = float(top["target_score"].mean() - rest["target_score"].mean())
        fold = safe_div(top["target_score"].mean(), rest["target_score"].mean())
        mw_p = safe_mannwhitney(top["target_score"], rest["target_score"])
        rows.append(
            {
                "lr_id": lr_id,
                "LR": lr_name,
                "ligand": str(group["ligand"].iloc[0]),
                "receptor": str(group["receptor"].iloc[0]),
                "top_n_receivers": int(len(top)),
                "n_receivers_scored": int(len(merged)),
                "target_spearman_r": rho,
                "target_spearman_p": pval,
                "top_receiver_target_mean": float(top["target_score"].mean()),
                "background_target_mean": float(rest["target_score"].mean()),
                "top_receiver_target_lift": lift,
                "top_receiver_target_fold": fold,
                "top_vs_rest_mannwhitney_p": mw_p,
                "top_receiver_cell_type_mode": mode_label(top["cell_type"]),
                "top_receiver_niche_mode": mode_label(top["niche"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_receiver_target_lift", "target_spearman_r"],
        ascending=False,
    )


def summarize_edge_distance(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, float(scored["distance"].quantile(0.995)), 13)
    if len(np.unique(bins)) < 3:
        bins = np.linspace(0.0, float(scored["distance"].max()), 13)
    for lr_id, group in scored.groupby("lr_id", sort=False):
        lr_name = str(group["control_name"].iloc[0])
        tmp = group.copy()
        tmp["distance_bin"] = pd.cut(tmp["distance"], bins=bins, include_lowest=True, duplicates="drop")
        for dist_bin, b in tmp.groupby("distance_bin", observed=True):
            rows.append(
                {
                    "lr_id": lr_id,
                    "LR": lr_name,
                    "mechanism": str(group["mechanism_lr"].iloc[0])
                    if "mechanism_lr" in group
                    else str(group["mechanism"].iloc[0]),
                    "distance_bin": str(dist_bin),
                    "bin_mid": float((dist_bin.left + dist_bin.right) / 2.0),
                    "mean_score": float(b["score"].mean()),
                    "q90_score": float(b["score"].quantile(0.90)),
                    "n_edges": int(len(b)),
                }
            )
    out = pd.DataFrame(rows)
    out["mean_score_norm_by_lr"] = out.groupby("LR")["mean_score"].transform(minmax_series)
    return out


def build_hotspot_cell_table(
    scored: pd.DataFrame,
    cells: pd.DataFrame,
    target_scores: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    meta = cells[["cell_id", "x", "y", "cell_type", "niche"]].copy()
    rows = []
    for _, lr_row in scored[["lr_id", "control_name", "ligand", "receptor"]].drop_duplicates().iterrows():
        lr_id = str(lr_row["lr_id"])
        lr_name = str(lr_row["control_name"])
        group = scored[scored["lr_id"].eq(lr_id)]
        receiver_activity = (
            group.groupby("receiver_id", as_index=False)
            .agg(receiver_activity=("score", "max"), receiver_mean_activity=("score", "mean"))
            .rename(columns={"receiver_id": "cell_id"})
        )
        dat = meta.merge(receiver_activity, on="cell_id", how="left")
        dat["receiver_activity"] = dat["receiver_activity"].fillna(0.0)
        dat["receiver_mean_activity"] = dat["receiver_mean_activity"].fillna(0.0)
        target = target_scores[target_scores["LR"].eq(lr_name)][["cell_id", "target_score"]]
        dat = dat.merge(target, on="cell_id", how="left")
        dat["target_score"] = dat["target_score"].fillna(0.0)
        cutoff = dat["receiver_activity"].nlargest(min(top_n, len(dat))).min()
        dat["is_top_receiver"] = dat["receiver_activity"] >= cutoff
        dat["lr_id"] = lr_id
        dat["LR"] = lr_name
        rows.append(dat)
    return pd.concat(rows, ignore_index=True)


def rank_story_candidates(lr_summary: pd.DataFrame, receiver_support: pd.DataFrame) -> pd.DataFrame:
    dat = lr_summary.merge(
        receiver_support[
            [
                "LR",
                "target_spearman_r",
                "top_receiver_target_lift",
                "top_receiver_target_fold",
                "top_vs_rest_mannwhitney_p",
            ]
        ],
        on="LR",
        how="left",
    )
    for col in ["mean_score", "top_niche_match_rate", "top_receiver_target_lift"]:
        dat[f"{col}_rank01"] = rank01(dat[col].fillna(0.0).to_numpy(dtype=float))
    dat["scale_separation_strength"] = (
        dat["broad_weight"].fillna(0.0).clip(0, 1) + dat["local_weight"].fillna(0.0).clip(0, 1)
    )
    dat["target_support"] = rank01(dat["top_receiver_target_lift"].fillna(0.0).to_numpy(dtype=float)) + 0.35 * np.maximum(
        dat["target_spearman_r"].fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    dat["decoy_penalty"] = dat["mechanism"].astype(str).str.contains("negative_control", regex=False).astype(float)
    dat["story_score"] = (
        0.30 * dat["mean_score_rank01"]
        + 0.25 * dat["top_niche_match_rate_rank01"]
        + 0.25 * rank01(dat["scale_separation_strength"].to_numpy(dtype=float))
        + 0.20 * rank01(dat["target_support"].to_numpy(dtype=float))
        - 0.60 * dat["decoy_penalty"]
    )
    return dat.sort_values("story_score", ascending=False).reset_index(drop=True)


def build_method_support_table(
    dataset_dir: Path,
    baseline_root: Path,
    baseline_output_dir: Path,
    *,
    top_n: int,
) -> pd.DataFrame:
    required = ["cells.csv", "expression.csv", "candidate_interactions.csv", "target_signatures.csv"]
    if not all((dataset_dir / name).exists() for name in required):
        return pd.DataFrame()

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    target_signatures = pd.read_csv(dataset_dir / "target_signatures.csv")
    target_scores = build_target_score_table(expression, target_signatures)
    prediction_paths = {
        "SpatialSelf": baseline_root
        / "spatialself_scale_9000_e120"
        / f"{dataset_dir.name}__spatialself_wavelet_scale_predictions.csv",
        "COMMOT": baseline_output_dir / "commot" / f"{dataset_dir.name}__commot_predictions.csv",
        "CellChat": baseline_output_dir / "cellchat" / f"{dataset_dir.name}__cellchat_predictions.csv",
        "SpaTalk": baseline_output_dir / "spatalk" / f"{dataset_dir.name}__spatalk_predictions.csv",
        "CellNEST": baseline_output_dir / "cellnest" / f"{dataset_dir.name}__cellnest_predictions.csv",
    }
    rows = []
    for method, path in prediction_paths.items():
        if not path.exists():
            rows.append({"method": method, "status": "missing_prediction", "prediction_path": str(path)})
            continue
        pred = pd.read_csv(path)
        scored = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
        if scored.empty:
            rows.append({"method": method, "status": "empty_projection", "prediction_path": str(path)})
            continue
        lr_mean = (
            scored.groupby(["lr_id", "ligand", "receptor"], as_index=False)
            .agg(mean_score=("score", "mean"), max_score=("score", "max"))
            .sort_values(["mean_score", "max_score"], ascending=False)
            .reset_index(drop=True)
        )
        lr_mean["rank_by_mean_score"] = np.arange(1, len(lr_mean) + 1)
        for lr_row in lr_mean.to_dict("records"):
            lr_id = str(lr_row["lr_id"])
            ligand = str(lr_row["ligand"])
            receptor = str(lr_row["receptor"])
            lr_pair = f"{ligand}-{receptor}"
            subset = scored[scored["lr_id"].eq(lr_id)]
            receiver_activity = (
                subset.groupby("receiver_id", as_index=False)
                .agg(receiver_activity=("score", "max"), n_incoming=("candidate_id", "size"))
                .rename(columns={"receiver_id": "cell_id"})
            )
            target = target_scores[target_scores["LR"].eq(lr_pair)][["cell_id", "target_score"]]
            support = receiver_activity.merge(target, on="cell_id", how="left")
            support["target_score"] = support["target_score"].fillna(0.0)
            support = support.merge(cells[["cell_id", "cell_type", "niche"]], on="cell_id", how="left")
            top = support.nlargest(min(top_n, len(support)), "receiver_activity")
            rest = support.drop(index=top.index)
            if rest.empty:
                rest = support
            rho, pval = safe_spearman(support["receiver_activity"], support["target_score"])
            rows.append(
                {
                    "method": method,
                    "status": "completed",
                    "prediction_path": str(path),
                    "lr_id": lr_id,
                    "LR": lr_pair,
                    "ligand": ligand,
                    "receptor": receptor,
                    "rank_by_mean_score": int(lr_row["rank_by_mean_score"]),
                    "mean_score": float(lr_row["mean_score"]),
                    "max_score": float(lr_row["max_score"]),
                    "target_spearman_r": rho,
                    "target_spearman_p": pval,
                    "top_receiver_target_lift": float(top["target_score"].mean() - rest["target_score"].mean()),
                    "top_receiver_target_fold": safe_div(top["target_score"].mean(), rest["target_score"].mean()),
                    "top_receiver_niche_mode": mode_label(top["niche"]),
                    "top_receiver_cell_type_mode": mode_label(top["cell_type"]),
                }
            )
    return pd.DataFrame(rows)


def mode_label(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.astype(str).value_counts()
    if counts.empty:
        return ""
    return str(counts.index[0])


def safe_mean(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    values = pd.Series(series).astype(float)
    if values.empty:
        return float("nan")
    return float(values.mean())


def minmax_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - low) / (high - low)


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def scale_class(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 1.25:
        return "local"
    if value < 3.25:
        return "intermediate"
    return "broad"


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan")
    rho, pval = spearmanr(x[mask], y[mask])
    return float(rho), float(pval)


def safe_mannwhitney(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    try:
        return float(mannwhitneyu(x, y, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return float("nan")
    return float(num) / den


if __name__ == "__main__":
    main()
