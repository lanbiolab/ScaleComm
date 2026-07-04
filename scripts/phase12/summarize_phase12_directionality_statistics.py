#!/usr/bin/env python
"""Audit Phase 12 directionality statistics and permutation readiness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
DIRECTION_DIR = PHASE12_DIR / "directionality_benchmark"
OUT_DIR = PHASE12_DIR / "directionality_statistics"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(DIRECTION_DIR / "phase12_directionality_lr_metrics.csv")
    comparison = pd.read_csv(DIRECTION_DIR / "phase12_directionality_method_comparison.csv")
    method_summary = pd.read_csv(DIRECTION_DIR / "phase12_directionality_method_summary.csv")

    lr_stats = annotate_lr_stats(metrics)
    dataset_stats = summarize_dataset_stats(lr_stats, comparison)
    comparison_stats = summarize_comparison_stats(comparison)
    permutation = permutation_readiness(lr_stats)

    lr_stats.to_csv(OUT_DIR / "phase12_directionality_statistical_lr_summary.csv", index=False)
    dataset_stats.to_csv(OUT_DIR / "phase12_directionality_statistical_dataset_summary.csv", index=False)
    comparison_stats.to_csv(OUT_DIR / "phase12_directionality_statistical_method_comparison.csv", index=False)
    permutation.to_csv(OUT_DIR / "phase12_directionality_permutation_readiness.csv", index=False)
    write_status(dataset_stats, comparison_stats, permutation)
    write_manifest()

    print(dataset_stats.to_string(index=False))
    print()
    print(comparison_stats.to_string(index=False))
    print()
    print(permutation.to_string(index=False))


def annotate_lr_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["expected_flow_score_fraction"] = pd.to_numeric(out["expected_flow_score_fraction"], errors="coerce")
    out["expected_flow_empirical_p"] = pd.to_numeric(out["expected_flow_empirical_p"], errors="coerce")
    out["expected_flow_bh_fdr"] = pd.to_numeric(out["expected_flow_bh_fdr"], errors="coerce")
    out["vector_coherence"] = pd.to_numeric(out["vector_coherence"], errors="coerce")
    out["top_niche_rule_fraction"] = pd.to_numeric(out["top_niche_rule_fraction"], errors="coerce")
    out["expected_flow_fdr_0_1"] = out["expected_flow_bh_fdr"].le(0.1)
    out["expected_flow_fdr_0_25"] = out["expected_flow_bh_fdr"].le(0.25)
    out["directionality_statistical_class"] = [
        statistical_class(row.expected_flow_score_fraction, row.expected_flow_bh_fdr, row.top_niche_flow_expected)
        for row in out.itertuples(index=False)
    ]
    return out


def summarize_dataset_stats(metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in metrics.groupby(["dataset_key", "method"], sort=False):
        comp = comparison[comparison["dataset_key"].eq(dataset_key)] if not comparison.empty else pd.DataFrame()
        if method == "spatialself":
            wins = int(comp["comparison_call"].astype(str).str.startswith("spatialself").sum()) if len(comp) else 0
            losses = int(comp["comparison_call"].astype(str).str.startswith("commot").sum()) if len(comp) else 0
        elif method == "commot":
            wins = int(comp["comparison_call"].astype(str).str.startswith("commot").sum()) if len(comp) else 0
            losses = int(comp["comparison_call"].astype(str).str.startswith("spatialself").sum()) if len(comp) else 0
        else:
            wins = losses = 0
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "n_expected_flow_fdr_0_1": int(sub["expected_flow_fdr_0_1"].sum()),
                "n_expected_flow_fdr_0_25": int(sub["expected_flow_fdr_0_25"].sum()),
                "median_expected_flow_fdr": float(sub["expected_flow_bh_fdr"].median()),
                "mean_expected_flow_fraction": float(sub["expected_flow_score_fraction"].mean()),
                "n_top_flow_expected": int(sub["top_niche_flow_expected"].sum()),
                "mean_vector_coherence": float(sub["vector_coherence"].mean()),
                "n_boundary_crossing": int(sub["boundary_crossing_centroid"].astype(bool).sum()),
                "n_local_retained": int(sub["local_retained_centroid"].astype(bool).sum()),
                "n_strong_statistical_support": int(sub["directionality_statistical_class"].eq("fdr_supported_expected_flow").sum()),
                "n_partial_statistical_support": int(sub["directionality_statistical_class"].eq("partial_empirical_support").sum()),
                "method_comparison_wins": wins,
                "method_comparison_losses": losses,
                "statistical_call": dataset_statistical_call(sub, wins, losses),
            }
        )
    return pd.DataFrame(rows)


def summarize_comparison_stats(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if comparison.empty:
        return pd.DataFrame()
    for dataset_key, sub in comparison.groupby("dataset_key"):
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr_compared": int(len(sub)),
                "n_spatialself_stronger_or_higher": int(sub["comparison_call"].astype(str).str.startswith("spatialself").sum()),
                "n_commot_stronger_or_higher": int(sub["comparison_call"].astype(str).str.startswith("commot").sum()),
                "n_similar_or_mixed": int(sub["comparison_call"].astype(str).eq("similar_or_mixed_directionality").sum()),
                "mean_spatialself_minus_commot_expected_fraction": float(
                    pd.to_numeric(sub["spatialself_minus_commot_expected_flow_fraction"], errors="coerce").mean()
                ),
                "n_spatialself_fdr_0_1": int(pd.to_numeric(sub["spatialself_fdr"], errors="coerce").le(0.1).sum()),
                "n_commot_fdr_0_1": int(pd.to_numeric(sub["commot_fdr"], errors="coerce").le(0.1).sum()),
                "comparison_interpretation": comparison_interpretation(sub),
            }
        )
    return pd.DataFrame(rows)


def permutation_readiness(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = [
        "source_sink_alignment_score",
        "vector_coherence",
        "expected_niche_flow_enrichment",
        "boundary_crossing_flow",
        "local_retained_vs_long_range",
        "permutation_p_value",
        "commot_comparison",
    ]
    for dataset_key, sub in metrics.groupby("dataset_key"):
        available = {
            "source_sink_alignment_score": bool(sub["expected_flow_score_fraction"].notna().any()),
            "vector_coherence": bool(sub["vector_coherence"].notna().any()),
            "expected_niche_flow_enrichment": bool(sub["expected_flow_empirical_p"].notna().any()),
            "boundary_crossing_flow": "boundary_crossing_centroid" in sub.columns,
            "local_retained_vs_long_range": {"local_retained_centroid", "long_range_like"}.issubset(sub.columns),
            "permutation_p_value": False,
            "commot_comparison": bool(set(sub["method"].astype(str)) >= {"spatialself", "commot"}),
        }
        missing = [name for name in required if not available[name]]
        rows.append(
            {
                "dataset_key": dataset_key,
                "available_metrics": ";".join(name for name in required if available[name]),
                "missing_metrics": ";".join(missing),
                "has_empirical_expected_flow_fdr": bool(sub["expected_flow_bh_fdr"].notna().any()),
                "has_permutation_p_value": False,
                "readiness_call": "needs_permutation_null" if "permutation_p_value" in missing else "ready",
                "claim_boundary": "Current expected-flow p/FDR uses non-expected niche flows as the empirical null; coordinate/LR-label permutation p-values are not yet available.",
            }
        )
    return pd.DataFrame(rows)


def statistical_class(expected_fraction: float, fdr: float, top_expected: bool) -> str:
    expected_fraction = to_float(expected_fraction)
    fdr = to_float(fdr)
    if np.isfinite(fdr) and fdr <= 0.1 and expected_fraction >= 0.5:
        return "fdr_supported_expected_flow"
    if np.isfinite(fdr) and fdr <= 0.25 and expected_fraction >= 0.2:
        return "partial_empirical_support"
    if top_expected:
        return "top_flow_expected_without_fdr"
    return "limited_or_no_statistical_support"


def dataset_statistical_call(sub: pd.DataFrame, wins: int, losses: int) -> str:
    strong = int(sub["directionality_statistical_class"].eq("fdr_supported_expected_flow").sum())
    partial = int(sub["directionality_statistical_class"].eq("partial_empirical_support").sum())
    if strong >= 2 and wins >= losses:
        return "dataset_directionality_support"
    if strong + partial >= 2:
        return "partial_directionality_support"
    if losses > wins:
        return "weaker_than_commot_or_mixed"
    return "limited_directionality_support"


def comparison_interpretation(sub: pd.DataFrame) -> str:
    ss = int(sub["comparison_call"].astype(str).str.startswith("spatialself").sum())
    cm = int(sub["comparison_call"].astype(str).str.startswith("commot").sum())
    if ss > cm:
        return "spatialself_higher_in_this_dataset"
    if cm > ss:
        return "commot_higher_in_this_dataset"
    return "mixed_or_similar"


def write_status(dataset_stats: pd.DataFrame, comparison_stats: pd.DataFrame, permutation: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Statistical Audit",
        "",
        "## Scope",
        "",
        "This audit separates existing empirical expected-flow FDR from missing permutation-calibrated directionality p-values.",
        "",
        "## Dataset/Method Statistics",
        "",
    ]
    for row in dataset_stats.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}` `{row.statistical_call}`: "
            f"FDR<=0.1={int(row.n_expected_flow_fdr_0_1)}/{int(row.n_lr)}, "
            f"FDR<=0.25={int(row.n_expected_flow_fdr_0_25)}/{int(row.n_lr)}, "
            f"mean expected-flow={fmt(row.mean_expected_flow_fraction)}, "
            f"top-flow expected={int(row.n_top_flow_expected)}/{int(row.n_lr)}, "
            f"wins/losses={int(row.method_comparison_wins)}/{int(row.method_comparison_losses)}"
        )
    if not comparison_stats.empty:
        lines += ["", "## SpatialSelf vs COMMOT", ""]
        for row in comparison_stats.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}`: SpatialSelf higher={int(row.n_spatialself_stronger_or_higher)}, "
                f"COMMOT higher={int(row.n_commot_stronger_or_higher)}, mixed={int(row.n_similar_or_mixed)}, "
                f"delta mean={fmt(row.mean_spatialself_minus_commot_expected_fraction)}, "
                f"call={row.comparison_interpretation}"
            )
    lines += ["", "## Permutation Readiness", ""]
    for row in permutation.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.readiness_call}`: missing={row.missing_metrics or 'none'}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Current FDR is empirical expected-flow FDR against non-expected niche flows, not a coordinate/LR-label permutation p-value.",
        "- Do not claim global directionality superiority over COMMOT; current comparisons are dataset- and LR-dependent.",
        "- Directionality remains source-sink score consistency, not physical ligand transport or causal temporal direction.",
    ]
    (OUT_DIR / "phase12_directionality_statistical_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_directionality_statistical_audit",
        "inputs": [
            "outputs/phase12/directionality_benchmark/phase12_directionality_lr_metrics.csv",
            "outputs/phase12/directionality_benchmark/phase12_directionality_method_summary.csv",
            "outputs/phase12/directionality_benchmark/phase12_directionality_method_comparison.csv",
        ],
        "outputs": [
            "phase12_directionality_statistical_lr_summary.csv",
            "phase12_directionality_statistical_dataset_summary.csv",
            "phase12_directionality_statistical_method_comparison.csv",
            "phase12_directionality_permutation_readiness.csv",
            "phase12_directionality_statistical_audit_status.md",
        ],
        "claim_boundary": "Existing empirical expected-flow FDR is available; formal permutation p-values remain missing.",
    }
    (OUT_DIR / "phase12_directionality_statistical_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


if __name__ == "__main__":
    main()
