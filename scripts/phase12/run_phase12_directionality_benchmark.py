#!/usr/bin/env python
"""Quantitative directionality benchmark for Phase 12 using Phase 11 trajectory outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_TRAJ_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/trajectory_summary"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/directionality_benchmark"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(PHASE11_TRAJ_DIR / "phase11_signal_trajectory_summary.csv")
    flow = pd.read_csv(PHASE11_TRAJ_DIR / "phase11_signal_trajectory_niche_flow_all.csv")
    metrics = build_lr_metrics(trajectory, flow)
    method_summary = summarize_methods(metrics)
    comparison = compare_methods(metrics)
    metrics.to_csv(OUT_DIR / "phase12_directionality_lr_metrics.csv", index=False)
    method_summary.to_csv(OUT_DIR / "phase12_directionality_method_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_directionality_method_comparison.csv", index=False)
    write_status(method_summary, comparison)
    write_manifest()
    print(method_summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


def build_lr_metrics(trajectory: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in trajectory.itertuples(index=False):
        dataset_key = row.dataset_key
        method = row.method
        lr_pair = row.lr_pair
        flow_sub = flow[
            flow["dataset_key"].eq(dataset_key)
            & flow["method"].eq(method)
            & flow["lr_pair"].eq(lr_pair)
        ].copy()
        flow_stats = expected_flow_stats(flow_sub)
        median_distance = to_float(getattr(row, "median_top_edge_distance", np.nan))
        trajectory_length = to_float(getattr(row, "trajectory_length", np.nan))
        vector_coherence = trajectory_length / median_distance if np.isfinite(trajectory_length) and median_distance > 0 else np.nan
        centroid_same_niche = str(getattr(row, "dominant_sender_niche", "")) == str(getattr(row, "dominant_receiver_niche", ""))
        boundary_crossing = not centroid_same_niche
        expected_fraction = to_float(getattr(row, "expected_flow_score_fraction", np.nan))
        expected_p = flow_stats["expected_flow_empirical_p"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": row.dataset,
                "method": method,
                "lr_pair": lr_pair,
                "status": getattr(row, "status", ""),
                "expected_flow_score_fraction": expected_fraction,
                "expected_flow_empirical_p": expected_p,
                "expected_flow_bh_fdr": np.nan,
                "top_niche_flow_expected": bool(getattr(row, "top_niche_flow_expected", False)),
                "best_flow_is_expected": bool(getattr(row, "best_flow_is_expected", False)),
                "expected_flow_rank_best": flow_stats["expected_flow_best_rank"],
                "n_expected_niche_flows": flow_stats["n_expected_niche_flows"],
                "top_type_rule_fraction": to_float(getattr(row, "top_type_rule_fraction", np.nan)),
                "top_niche_rule_fraction": to_float(getattr(row, "top_niche_rule_fraction", np.nan)),
                "trajectory_length": trajectory_length,
                "median_top_edge_distance": median_distance,
                "vector_coherence": vector_coherence,
                "centroid_flow_label": getattr(row, "centroid_flow_label", ""),
                "top_niche_flow_label": getattr(row, "top_niche_flow_label", ""),
                "boundary_crossing_centroid": bool(boundary_crossing),
                "local_retained_centroid": bool(centroid_same_niche),
                "long_range_like": bool(boundary_crossing and median_distance >= dataset_distance_threshold(trajectory, dataset_key)),
                "directionality_support_class": support_class(expected_fraction, expected_p, bool(getattr(row, "top_niche_flow_expected", False))),
            }
        )
    out = pd.DataFrame(rows)
    out["expected_flow_bh_fdr"] = bh_adjust_by_group(out, ["dataset_key", "method"], "expected_flow_empirical_p")
    out["directionality_support_class"] = [
        support_class(row.expected_flow_score_fraction, row.expected_flow_bh_fdr, row.top_niche_flow_expected)
        for row in out.itertuples(index=False)
    ]
    return out


def expected_flow_stats(flow: pd.DataFrame) -> dict[str, float | int]:
    if flow.empty or "expected_flow" not in flow.columns:
        return {
            "expected_flow_empirical_p": np.nan,
            "expected_flow_best_rank": -1,
            "n_expected_niche_flows": 0,
        }
    total = float(pd.to_numeric(flow["total_score"], errors="coerce").fillna(0).sum())
    if total <= 0:
        return {
            "expected_flow_empirical_p": np.nan,
            "expected_flow_best_rank": -1,
            "n_expected_niche_flows": int(flow["expected_flow"].astype(bool).sum()),
        }
    flow = flow.copy()
    flow["score_fraction"] = pd.to_numeric(flow["total_score"], errors="coerce").fillna(0) / total
    expected = flow[flow["expected_flow"].astype(bool)]
    non_expected = flow[~flow["expected_flow"].astype(bool)]
    if expected.empty:
        return {
            "expected_flow_empirical_p": np.nan,
            "expected_flow_best_rank": -1,
            "n_expected_niche_flows": 0,
        }
    observed = float(expected["score_fraction"].max())
    null = non_expected["score_fraction"].to_numpy(dtype=float)
    if len(null):
        p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    else:
        p = np.nan
    ranks = flow["score_fraction"].rank(ascending=False, method="min")
    best_rank = int(ranks.loc[expected.index].min())
    return {
        "expected_flow_empirical_p": p,
        "expected_flow_best_rank": best_rank,
        "n_expected_niche_flows": int(len(expected)),
    }


def summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in metrics.groupby(["dataset_key", "method"], sort=False):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "mean_expected_flow_fraction": float(sub["expected_flow_score_fraction"].mean()),
                "median_expected_flow_fraction": float(sub["expected_flow_score_fraction"].median()),
                "n_top_flow_expected": int(sub["top_niche_flow_expected"].sum()),
                "top_flow_expected_fraction": safe_divide(int(sub["top_niche_flow_expected"].sum()), len(sub)),
                "n_expected_fdr_0_1": int(sub["expected_flow_bh_fdr"].fillna(1.0).le(0.1).sum()),
                "mean_top_niche_rule_fraction": float(sub["top_niche_rule_fraction"].mean()),
                "mean_vector_coherence": float(sub["vector_coherence"].mean()),
                "n_local_retained": int(sub["local_retained_centroid"].sum()),
                "n_boundary_crossing": int(sub["boundary_crossing_centroid"].sum()),
                "n_strong_directionality_support": int(sub["directionality_support_class"].eq("strong_expected_flow_support").sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_cols = [
        "expected_flow_score_fraction",
        "expected_flow_bh_fdr",
        "top_niche_flow_expected",
        "top_niche_rule_fraction",
        "vector_coherence",
    ]
    for (dataset_key, lr_pair), sub in metrics.groupby(["dataset_key", "lr_pair"], sort=False):
        by_method = {row.method: row for row in sub.itertuples(index=False)}
        if "spatialself" not in by_method or "commot" not in by_method:
            continue
        ss = by_method["spatialself"]
        cm = by_method["commot"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": ss.dataset,
                "lr_pair": lr_pair,
                "spatialself_expected_flow_fraction": to_float(ss.expected_flow_score_fraction),
                "commot_expected_flow_fraction": to_float(cm.expected_flow_score_fraction),
                "spatialself_minus_commot_expected_flow_fraction": to_float(ss.expected_flow_score_fraction)
                - to_float(cm.expected_flow_score_fraction),
                "spatialself_fdr": to_float(ss.expected_flow_bh_fdr),
                "commot_fdr": to_float(cm.expected_flow_bh_fdr),
                "spatialself_top_flow_expected": bool(ss.top_niche_flow_expected),
                "commot_top_flow_expected": bool(cm.top_niche_flow_expected),
                "spatialself_support_class": ss.directionality_support_class,
                "commot_support_class": cm.directionality_support_class,
                "comparison_call": comparison_call(ss, cm),
            }
        )
    return pd.DataFrame(rows)


def support_class(expected_fraction: float, fdr_or_p: float, top_expected: bool) -> str:
    expected_fraction = to_float(expected_fraction)
    fdr_or_p = to_float(fdr_or_p)
    if np.isfinite(expected_fraction) and expected_fraction >= 0.5 and (top_expected or (np.isfinite(fdr_or_p) and fdr_or_p <= 0.1)):
        return "strong_expected_flow_support"
    if np.isfinite(expected_fraction) and expected_fraction >= 0.2:
        return "partial_expected_flow_support"
    if top_expected:
        return "top_flow_expected_but_low_fraction"
    return "limited_expected_flow_support"


def comparison_call(ss: object, cm: object) -> str:
    delta = to_float(ss.expected_flow_score_fraction) - to_float(cm.expected_flow_score_fraction)
    ss_strong = str(ss.directionality_support_class) == "strong_expected_flow_support"
    cm_strong = str(cm.directionality_support_class) == "strong_expected_flow_support"
    if ss_strong and not cm_strong:
        return "spatialself_stronger_expected_flow"
    if cm_strong and not ss_strong:
        return "commot_stronger_expected_flow"
    if np.isfinite(delta) and delta >= 0.1:
        return "spatialself_higher_expected_fraction"
    if np.isfinite(delta) and delta <= -0.1:
        return "commot_higher_expected_fraction"
    return "similar_or_mixed_directionality"


def dataset_distance_threshold(metrics: pd.DataFrame, dataset_key: str) -> float:
    sub = metrics[metrics["dataset_key"].eq(dataset_key)]
    distance = pd.to_numeric(sub["median_top_edge_distance"], errors="coerce").dropna()
    if distance.empty:
        return np.inf
    return float(distance.median())


def bh_adjust_by_group(table: pd.DataFrame, group_cols: list[str], p_col: str) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=float)
    for _, idx in table.groupby(group_cols).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        out[idx] = bh_adjust(table.iloc[idx][p_col].to_numpy(dtype=float))
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def write_status(method_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Benchmark Status",
        "",
        "## Scope",
        "",
        "This benchmark converts Phase 11 COMMOT-style trajectories into quantitative directionality metrics: expected-flow fraction, empirical expected-flow FDR, top-flow correctness, vector coherence, and local-retained versus boundary-crossing flow labels.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}`: LR={int(row.n_lr)}, mean expected-flow={float(row.mean_expected_flow_fraction):.3f}, top-flow expected={int(row.n_top_flow_expected)}/{int(row.n_lr)}, FDR<=0.1={int(row.n_expected_fdr_0_1)}, strong={int(row.n_strong_directionality_support)}"
        )
    if not comparison.empty:
        lines += ["", "## SpatialSelf vs COMMOT", ""]
        calls = comparison["comparison_call"].value_counts()
        for call, count in calls.items():
            lines.append(f"- {call}: {int(count)} LR cases")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "These are descriptive source-to-sink consistency metrics from scored spatial edges. They do not prove physical ligand transport, temporal causality, or perturbational directionality.",
    ]
    (OUT_DIR / "phase12_directionality_benchmark_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_directionality_benchmark",
        "source": "outputs/phase11/commot_style_spatial_story/trajectory_summary",
        "outputs": [
            "phase12_directionality_lr_metrics.csv",
            "phase12_directionality_method_summary.csv",
            "phase12_directionality_method_comparison.csv",
            "phase12_directionality_benchmark_status.md",
        ],
        "claim_boundary": "Descriptive directionality consistency benchmark, not causal transport validation.",
    }
    (OUT_DIR / "phase12_directionality_benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def safe_divide(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


if __name__ == "__main__":
    main()
