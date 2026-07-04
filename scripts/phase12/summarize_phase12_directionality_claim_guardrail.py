#!/usr/bin/env python
"""Conservative directionality claim guardrail for Phase 12."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "directionality_claim_guardrail"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = read_csv(PHASE12_DIR / "directionality_statistics/phase12_directionality_statistical_lr_summary.csv")
    label_perm = read_csv(PHASE12_DIR / "directionality_permutation/phase12_directionality_permutation_lr_summary.csv")
    edge_perm = read_csv(PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_lr_summary.csv")
    comparison = read_csv(PHASE12_DIR / "directionality_benchmark/phase12_directionality_method_comparison.csv")
    edge_method = read_csv(PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_method_comparison.csv")

    lr_guardrail = build_lr_guardrail(stats, label_perm, edge_perm, comparison)
    dataset_guardrail = summarize_dataset(lr_guardrail, edge_method)
    method_guardrail = summarize_method(lr_guardrail)

    lr_guardrail.to_csv(OUT_DIR / "phase12_directionality_claim_guardrail_lr.csv", index=False)
    dataset_guardrail.to_csv(OUT_DIR / "phase12_directionality_claim_guardrail_dataset.csv", index=False)
    method_guardrail.to_csv(OUT_DIR / "phase12_directionality_claim_guardrail_method.csv", index=False)
    write_status(lr_guardrail, dataset_guardrail)
    write_manifest()
    print(dataset_guardrail.to_string(index=False))
    print()
    print(lr_guardrail.sort_values(["claim_strength_score", "dataset_key", "method"], ascending=[False, True, True]).head(40).to_string(index=False))


def build_lr_guardrail(
    stats: pd.DataFrame,
    label_perm: pd.DataFrame,
    edge_perm: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    rows = []
    for row in stats.itertuples(index=False):
        dataset_key = str(row.dataset_key)
        method = str(row.method)
        lr_pair = str(row.lr_pair)
        label = pick(label_perm, dataset_key=dataset_key, method=method, lr_pair=lr_pair)
        edge = pick(edge_perm, dataset_key=dataset_key, method=method, lr_pair=lr_pair)
        comp = pick(comparison, dataset_key=dataset_key, lr_pair=lr_pair)
        if method == "spatialself":
            commot_delta = get(comp, "spatialself_minus_commot_expected_flow_fraction")
            comparison_call = perspective_comparison_call(method, get(comp, "comparison_call", ""))
        elif method == "commot":
            commot_delta = -safe_float(get(comp, "spatialself_minus_commot_expected_flow_fraction"))
            comparison_call = perspective_comparison_call(method, get(comp, "comparison_call", ""))
        else:
            commot_delta = np.nan
            comparison_call = ""

        receiver_q = get(edge, "receiver_pairing_bh_fdr")
        vector_q = get(edge, "edge_vector_bh_fdr")
        label_lr_q = first_non_null(get(label, "lr_label_bh_fdr"), get(label, "lr_identity_label_bh_fdr"))
        label_niche_q = first_non_null(get(label, "niche_label_bh_fdr"), get(label, "niche_label_permutation_bh_fdr"))
        expected_q = get(row, "expected_flow_bh_fdr")
        claim_call, score = classify_claim(
            expected_fraction=get(row, "expected_flow_score_fraction"),
            expected_q=expected_q,
            label_lr_q=label_lr_q,
            label_niche_q=label_niche_q,
            receiver_q=receiver_q,
            vector_q=vector_q,
            high_expression_negative=get(edge, "high_expression_negative", False),
            comparison_call=comparison_call,
        )
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": get(row, "dataset"),
                "method": method,
                "lr_pair": lr_pair,
                "expected_flow_score_fraction": safe_float(get(row, "expected_flow_score_fraction")),
                "expected_flow_bh_fdr": safe_float(expected_q),
                "lr_label_bh_fdr": safe_float(label_lr_q),
                "niche_label_bh_fdr": safe_float(label_niche_q),
                "receiver_pairing_bh_fdr": safe_float(receiver_q),
                "edge_vector_bh_fdr": safe_float(vector_q),
                "vector_coherence": safe_float(get(row, "vector_coherence")),
                "trajectory_length": safe_float(get(row, "trajectory_length")),
                "median_top_edge_distance": safe_float(get(row, "median_top_edge_distance")),
                "source_sink_alignment_score": safe_float(get(row, "top_niche_rule_fraction")),
                "top_type_rule_fraction": safe_float(get(row, "top_type_rule_fraction")),
                "boundary_crossing_centroid": bool(get(row, "boundary_crossing_centroid", False)),
                "local_retained_centroid": bool(get(row, "local_retained_centroid", False)),
                "long_range_like": bool(get(row, "long_range_like", False)),
                "flow_range_class": flow_range_class(row),
                "high_expression_negative": bool(get(edge, "high_expression_negative", False)),
                "spatialself_minus_commot_expected_flow_fraction": safe_float(commot_delta),
                "method_comparison_call": comparison_call,
                "claim_strength_score": score,
                "directionality_claim_call": claim_call,
                "claim_boundary": claim_boundary(claim_call, method),
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(lr_guardrail: pd.DataFrame, edge_method: pd.DataFrame) -> pd.DataFrame:
    if lr_guardrail.empty:
        return pd.DataFrame()
    rows = []
    for (dataset_key, method), sub in lr_guardrail.groupby(["dataset_key", "method"], sort=False):
        negatives = sub[sub["high_expression_negative"].astype(bool)]
        positives = sub[~sub["high_expression_negative"].astype(bool)]
        supported = sub[sub["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"])]
        vector_supported = sub[sub["edge_vector_bh_fdr"].le(0.1)]
        best = sub.sort_values(["claim_strength_score", "receiver_pairing_bh_fdr"], ascending=[False, True]).head(1)
        edge_row = pick(edge_method, dataset_key=dataset_key, method=method)
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "n_supported_directionality_claims": int(len(supported)),
                "n_vector_supported": int(len(vector_supported)),
                "n_local_retained": int(sub["local_retained_centroid"].sum()),
                "n_boundary_crossing": int(sub["boundary_crossing_centroid"].sum()),
                "n_long_range_like": int(sub["long_range_like"].sum()),
                "n_high_expression_negative_directionality_supported": int(
                    negatives["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"]).sum()
                ),
                "best_lr": best.iloc[0]["lr_pair"] if len(best) else "",
                "best_lr_claim": best.iloc[0]["directionality_claim_call"] if len(best) else "",
                "best_lr_score": int(best.iloc[0]["claim_strength_score"]) if len(best) else 0,
                "edge_method_comparison_note": get(edge_row, "comparison_call", ""),
                "dataset_claim_call": dataset_claim(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(lr_guardrail: pd.DataFrame) -> pd.DataFrame:
    if lr_guardrail.empty:
        return pd.DataFrame()
    rows = []
    for method, sub in lr_guardrail.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_lr": int(len(sub)),
                "n_supported_directionality_claims": int(sub["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"]).sum()),
                "n_vector_supported": int(sub["edge_vector_bh_fdr"].le(0.1).sum()),
                "mean_expected_flow_fraction": float(sub["expected_flow_score_fraction"].mean()),
                "mean_source_sink_alignment_score": float(sub["source_sink_alignment_score"].mean()),
                "mean_vector_coherence": float(sub["vector_coherence"].mean()),
                "n_high_expression_negative_supported": int(
                    sub[sub["high_expression_negative"].astype(bool)]["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_claim(
    expected_fraction: Any,
    expected_q: Any,
    label_lr_q: Any,
    label_niche_q: Any,
    receiver_q: Any,
    vector_q: Any,
    high_expression_negative: Any,
    comparison_call: str,
) -> tuple[str, int]:
    score = 0
    expected_fraction = safe_float(expected_fraction)
    if expected_fraction >= 0.5:
        score += 1
    if safe_float(expected_q) <= 0.1:
        score += 1
    if safe_float(label_lr_q) <= 0.1:
        score += 1
    if safe_float(label_niche_q) <= 0.1:
        score += 1
    if safe_float(receiver_q) <= 0.1:
        score += 1
    if safe_float(vector_q) <= 0.1:
        score += 1
    if bool(high_expression_negative):
        score -= 2
    if str(comparison_call).startswith("spatialself_stronger") or str(comparison_call).startswith("commot_stronger"):
        score += 0

    if bool(high_expression_negative) and score > 0:
        return "negative_directionality_leakage", int(score)
    if safe_float(receiver_q) <= 0.1 and safe_float(vector_q) <= 0.1:
        return "multi_null_directionality_supported", int(score)
    if safe_float(receiver_q) <= 0.1:
        return "receiver_pairing_directionality_supported", int(score)
    if safe_float(expected_q) <= 0.1 and safe_float(label_niche_q) <= 0.1:
        return "aggregate_niche_flow_supported", int(score)
    if expected_fraction >= 0.5:
        return "descriptive_high_expected_flow_only", int(score)
    return "not_supported_for_directionality_claim", int(score)


def claim_boundary(call: str, method: str) -> str:
    if call == "multi_null_directionality_supported":
        return "Strongest fixed-score directionality support, but still not causal ligand transport or model-rerun vector FDR."
    if call == "receiver_pairing_directionality_supported":
        return "Can claim source-sink/receiver-pairing support, not vector-coherent causal transport."
    if call == "aggregate_niche_flow_supported":
        return "Can claim aggregate niche-flow consistency, not edge-level vector support."
    if call == "descriptive_high_expected_flow_only":
        return "Descriptive expected-flow enrichment only; do not call statistically supported directionality."
    if call == "negative_directionality_leakage":
        return "Hard negative shows directionality-like behavior; use as false-positive leakage, not biological evidence."
    return "Do not use for directionality claim."


def dataset_claim(sub: pd.DataFrame) -> str:
    n_supported = int(sub["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"]).sum())
    n_vector = int(sub["edge_vector_bh_fdr"].le(0.1).sum())
    n_negative = int(
        sub[sub["high_expression_negative"].astype(bool)]["directionality_claim_call"].isin(["receiver_pairing_directionality_supported", "multi_null_directionality_supported"]).sum()
    )
    if n_supported >= 2 and n_vector == 0 and n_negative == 0:
        return "source_sink_supported_vector_not_supported_negatives_clean"
    if n_supported >= 2 and n_negative > 0:
        return "source_sink_supported_with_negative_leakage"
    if n_supported >= 1:
        return "limited_source_sink_support"
    return "directionality_claim_not_supported"


def flow_range_class(row: pd.Series | Any) -> str:
    if bool(get(row, "long_range_like", False)):
        return "long_range_boundary_crossing"
    if bool(get(row, "boundary_crossing_centroid", False)):
        return "boundary_crossing"
    if bool(get(row, "local_retained_centroid", False)):
        return "local_retained"
    return "unclassified"


def pick(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    sub = df
    for key, value in kwargs.items():
        if key not in sub.columns:
            return pd.Series(dtype=object)
        sub = sub[sub[key].astype(str).eq(str(value))]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def get(row: pd.Series | Any, key: str, default: Any = np.nan) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return getattr(row, key, default)


def first_non_null(*values: Any) -> Any:
    for value in values:
        v = safe_float(value)
        if np.isfinite(v):
            return v
    return np.nan


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def perspective_comparison_call(method: str, call: Any) -> str:
    text = str(call)
    if text.startswith(f"{method}_"):
        return text.replace(f"{method}_", "current_method_", 1)
    other = "commot" if method == "spatialself" else "spatialself"
    if text.startswith(f"{other}_"):
        return text.replace(f"{other}_", "other_method_", 1)
    return text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_status(lr_guardrail: pd.DataFrame, dataset_guardrail: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Claim Guardrail",
        "",
        "## Scope",
        "",
        "This table consolidates source-sink alignment, vector coherence, local-retained versus long-range flow classes, permutation support, hard-negative leakage and COMMOT comparison into one conservative directionality audit.",
        "",
        "## Dataset Calls",
        "",
    ]
    for row in dataset_guardrail.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}`: {row.dataset_claim_call}; supported={int(row.n_supported_directionality_claims)}, vector={int(row.n_vector_supported)}, negative-supported={int(row.n_high_expression_negative_directionality_supported)}, best={row.best_lr} ({row.best_lr_claim})"
        )
    lines += ["", "## Claim Boundary", ""]
    lines.append(
        "Current evidence supports source-sink or receiver-pairing directionality for selected LR, especially lymph-node candidates. It does not support global SpatialSelf directionality superiority over COMMOT, and vector-null support is limited."
    )
    (OUT_DIR / "phase12_directionality_claim_guardrail_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_directionality_claim_guardrail",
        "outputs": [
            rel(OUT_DIR / "phase12_directionality_claim_guardrail_lr.csv"),
            rel(OUT_DIR / "phase12_directionality_claim_guardrail_dataset.csv"),
            rel(OUT_DIR / "phase12_directionality_claim_guardrail_method.csv"),
            rel(OUT_DIR / "phase12_directionality_claim_guardrail_status.md"),
        ],
        "claim_boundary": "Directionality guardrail only; not causal ligand transport and not global method superiority.",
    }
    (OUT_DIR / "phase12_directionality_claim_guardrail_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
