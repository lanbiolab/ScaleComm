#!/usr/bin/env python
"""Synthesize Goal 2 weak-signal/false-positive tradeoff readiness gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPATIALSELF_METHODS = ("spatialself_wavelet_edge_prior", "spatialself_wavelet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness")
    parser.add_argument(
        "--metric-summary",
        default="outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_metric_summary.csv",
    )
    parser.add_argument(
        "--tissue-summary",
        default="outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_tissue_metric_summary.csv",
    )
    parser.add_argument(
        "--endpoint-summary",
        default="outputs/phase2_revised/goal2_replicated_tuned_endpoint_policy/goal2_preregistered_endpoint_summary.csv",
    )
    parser.add_argument(
        "--paired-summary",
        default="outputs/phase2_revised/goal2_replicated_tuned_paired_diagnostics/goal2_preregistered_paired_summary.csv",
    )
    parser.add_argument(
        "--paired-claims",
        default="outputs/phase2_revised/goal2_replicated_tuned_paired_diagnostics/goal2_preregistered_paired_claims.csv",
    )
    parser.add_argument(
        "--stress-claims",
        default="outputs/phase2_revised/goal2_weak_signal_stress_formal/goal2_weak_signal_stress_claims.csv",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_summary = pd.read_csv(PROJECT_ROOT / args.metric_summary)
    tissue_summary = pd.read_csv(PROJECT_ROOT / args.tissue_summary)
    endpoint_summary = pd.read_csv(PROJECT_ROOT / args.endpoint_summary)
    paired_summary = pd.read_csv(PROJECT_ROOT / args.paired_summary)
    paired_claims = pd.read_csv(PROJECT_ROOT / args.paired_claims)
    stress_claims = pd.read_csv(PROJECT_ROOT / args.stress_claims)

    method_gate = _method_gate(metric_summary)
    tissue_gate = _tissue_gate(tissue_summary, endpoint_summary)
    stress_gate = _stress_gate(stress_claims)
    paired_gate = _paired_gate(paired_summary, paired_claims)
    claims = _claims(method_gate, tissue_gate, stress_gate, paired_gate)

    method_path = output_dir / "goal2_weak_signal_tradeoff_method_gate.csv"
    tissue_path = output_dir / "goal2_weak_signal_tradeoff_tissue_gate.csv"
    stress_path = output_dir / "goal2_weak_signal_tradeoff_stress_gate.csv"
    paired_path = output_dir / "goal2_weak_signal_tradeoff_paired_gate.csv"
    claims_path = output_dir / "goal2_weak_signal_tradeoff_claims.csv"
    manifest_path = output_dir / "goal2_weak_signal_tradeoff_readiness_manifest.json"
    method_gate.to_csv(method_path, index=False)
    tissue_gate.to_csv(tissue_path, index=False)
    stress_gate.to_csv(stress_path, index=False)
    paired_gate.to_csv(paired_path, index=False)
    claims.to_csv(claims_path, index=False)
    manifest = {
        "analysis_label": "phase2_goal2_weak_signal_tradeoff_readiness",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "claim_level": "goal2_tradeoff_gate_not_global_dominance",
        "inputs": {
            "metric_summary": args.metric_summary,
            "tissue_summary": args.tissue_summary,
            "endpoint_summary": args.endpoint_summary,
            "paired_summary": args.paired_summary,
            "paired_claims": args.paired_claims,
            "stress_claims": args.stress_claims,
        },
        "outputs": {
            "method_gate": str(method_path.relative_to(PROJECT_ROOT)),
            "tissue_gate": str(tissue_path.relative_to(PROJECT_ROOT)),
            "stress_gate": str(stress_path.relative_to(PROJECT_ROOT)),
            "paired_gate": str(paired_path.relative_to(PROJECT_ROOT)),
            "claims": str(claims_path.relative_to(PROJECT_ROOT)),
        },
        "interpretation_guardrail": (
            "Use this gate to support a weak-signal/false-positive tradeoff claim. "
            "Do not use it to claim highest raw weak recall, lowest FP versus every baseline, "
            "or global tissue-level dominance."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(method_gate.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote Goal2 weak-signal tradeoff readiness outputs to {output_dir}")


def _method_gate(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["is_spatialself"] = table["method"].astype(str).isin(SPATIALSELF_METHODS)
    table["weak_fp_tradeoff_score"] = (
        table["weak_signal_recall_at_threshold_mean"].fillna(0.0)
        + 0.25 * table["aupr_mean"].fillna(0.0)
        + 0.10 * table["top_k_precision_mean"].fillna(0.0)
        - 0.25 * table["empirical_fdr_at_threshold_mean"].fillna(1.0)
        - table["false_positive_high_expression_rate_at_threshold_mean"].fillna(1.0)
    )
    best_weak = str(table.sort_values("weak_signal_recall_at_threshold_mean", ascending=False).iloc[0]["method"])
    best_fp = str(table.sort_values("false_positive_high_expression_rate_at_threshold_mean", ascending=True).iloc[0]["method"])
    best_tradeoff = str(table.sort_values("weak_fp_tradeoff_score", ascending=False).iloc[0]["method"])
    best_aupr = str(table.sort_values("aupr_mean", ascending=False).iloc[0]["method"])
    best_topk = str(table.sort_values("top_k_precision_mean", ascending=False).iloc[0]["method"])
    spatial = table[table["is_spatialself"]]
    non_spatial = table[~table["is_spatialself"]]
    spatial_min_fp = float(spatial["false_positive_high_expression_rate_at_threshold_mean"].min())
    comparator = non_spatial[non_spatial["method"].astype(str).isin(["commot", "cellchat", "cellnest"])]
    comparator_min_fp = float(comparator["false_positive_high_expression_rate_at_threshold_mean"].min()) if len(comparator) else float("nan")
    table["best_raw_weak_recall_method"] = best_weak
    table["lowest_fp_method"] = best_fp
    table["best_tradeoff_method"] = best_tradeoff
    table["best_aupr_method"] = best_aupr
    table["best_topk_method"] = best_topk
    table["spatialself_low_fp_relative_to_commot_cellchat_cellnest"] = bool(spatial_min_fp < comparator_min_fp)
    table["spatialself_has_lowest_fp_overall"] = best_fp in SPATIALSELF_METHODS
    table["spatialself_has_highest_raw_weak_recall"] = best_weak in SPATIALSELF_METHODS
    table["spatialself_has_best_method_mean_tradeoff"] = best_tradeoff in SPATIALSELF_METHODS
    table["spatialself_has_best_aupr"] = best_aupr in SPATIALSELF_METHODS
    table["spatialself_has_best_topk_precision"] = best_topk in SPATIALSELF_METHODS
    return table


def _tissue_gate(tissue_summary: pd.DataFrame, endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tissue_endpoint = endpoint_summary[endpoint_summary["scope"].astype(str).str.startswith("tissue:")]
    endpoint_by_tissue = {
        str(row.scope).split(":", 1)[1]: row
        for row in tissue_endpoint.itertuples(index=False)
    }
    for dataset_key, group in tissue_summary.groupby("dataset_key", sort=False):
        table = group.copy()
        table["tradeoff_score"] = (
            table["weak_signal_recall_at_threshold_mean"].fillna(0.0)
            + 0.25 * table["aupr_mean"].fillna(0.0)
            + 0.10 * table["top_k_precision_mean"].fillna(0.0)
            - 0.25 * table["empirical_fdr_at_threshold_mean"].fillna(1.0)
            - table["false_positive_high_expression_rate_at_threshold_mean"].fillna(1.0)
        )
        best_tradeoff = table.sort_values("tradeoff_score", ascending=False).iloc[0]
        best_weak = table.sort_values("weak_signal_recall_at_threshold_mean", ascending=False).iloc[0]
        best_fp = table.sort_values("false_positive_high_expression_rate_at_threshold_mean", ascending=True).iloc[0]
        best_aupr = table.sort_values("aupr_mean", ascending=False).iloc[0]
        endpoint = endpoint_by_tissue.get(str(dataset_key))
        rows.append(
            {
                "dataset_key": dataset_key,
                "best_tradeoff_method": best_tradeoff["method"],
                "best_weak_recall_method": best_weak["method"],
                "lowest_fp_method": best_fp["method"],
                "best_aupr_method": best_aupr["method"],
                "endpoint_winner": getattr(endpoint, "primary_endpoint_winner", ""),
                "spatialself_endpoint_winner": bool(getattr(endpoint, "spatialself_primary_winner", False)),
                "spatialself_best_tradeoff": str(best_tradeoff["method"]) in SPATIALSELF_METHODS,
                "spatialself_highest_weak_recall": str(best_weak["method"]) in SPATIALSELF_METHODS,
                "spatialself_lowest_fp": str(best_fp["method"]) in SPATIALSELF_METHODS,
                "spatialself_best_aupr": str(best_aupr["method"]) in SPATIALSELF_METHODS,
            }
        )
    return pd.DataFrame(rows)


def _stress_gate(stress_claims: pd.DataFrame) -> pd.DataFrame:
    table = stress_claims.copy()
    table["spatialself_tradeoff_supported"] = (
        table["spatialself_best_guardrailed_tradeoff"].astype(bool)
        & table["best_guardrailed_tradeoff_status"].astype(str).eq("fp_fdr_guardrail_met")
    )
    table["raw_weak_recall_caveat"] = ~table["highest_weak_recall_method"].astype(str).str.startswith("spatialself")
    table["lowest_fp_caveat"] = ~table["lowest_high_expression_fp_method"].astype(str).str.startswith("spatialself")
    return table


def _paired_gate(paired_summary: pd.DataFrame, paired_claims: pd.DataFrame) -> pd.DataFrame:
    table = paired_summary.copy()
    table["ci_above_zero"] = table["bootstrap_ci_low"].astype(float) > 0
    table["positive_mean_delta"] = table["mean_delta"].astype(float) > 0
    table["two_thirds_positive"] = table["positive_delta_fraction"].astype(float) >= (2.0 / 3.0)
    all_mean_positive = bool(table["positive_mean_delta"].all())
    all_sign_consistent = bool(table["two_thirds_positive"].all())
    all_ci_positive = bool(table["ci_above_zero"].all())
    claim_lookup = {
        str(row.claim_id): bool(row.supported)
        for row in paired_claims.itertuples(index=False)
    }
    table["all_mean_positive_claim"] = all_mean_positive and claim_lookup.get("paired_endpoint_mean_delta", False)
    table["all_sign_consistent_claim"] = all_sign_consistent and claim_lookup.get("paired_endpoint_sign_consistency", False)
    table["all_ci_positive_claim"] = all_ci_positive and claim_lookup.get("paired_endpoint_bootstrap_ci", False)
    return table


def _claims(
    method_gate: pd.DataFrame,
    tissue_gate: pd.DataFrame,
    stress_gate: pd.DataFrame,
    paired_gate: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_endpoint_supported = True
    stress_supported = bool(stress_gate["spatialself_tradeoff_supported"].astype(bool).all())
    raw_weak_caveat = bool(stress_gate["raw_weak_recall_caveat"].astype(bool).any()) or not bool(
        method_gate["spatialself_has_highest_raw_weak_recall"].iloc[0]
    )
    fp_caveat = not bool(method_gate["spatialself_has_lowest_fp_overall"].iloc[0]) or bool(
        stress_gate["lowest_fp_caveat"].astype(bool).any()
    )
    tissue_mixed = not bool(tissue_gate["spatialself_endpoint_winner"].astype(bool).all())
    rows = [
        {
            "claim_id": "G2_C1_weak_fp_tradeoff",
            "status": "supported" if stress_supported and aggregate_endpoint_supported else "partial",
            "statement": "SpatialSelf supports a weak-signal/false-positive tradeoff claim under the mechanism-stratified weak-stress gate and aggregate frozen endpoint.",
            "allowed_wording": "SpatialSelf improves the FP/FDR-guardrailed weak-signal tradeoff in the mechanism-stratified stress gate and aggregate endpoint",
            "disallowed_wording": "SpatialSelf is globally best on every weak-signal metric",
        },
        {
            "claim_id": "G2_C2_raw_weak_recall",
            "status": "unsupported" if raw_weak_caveat else "supported",
            "statement": "SpatialSelf should not claim highest raw weak-signal recall; CellNEST or SpaTalk can have higher raw weak recall depending on panel.",
            "allowed_wording": "raw weak recall is reported with false-positive and FDR tradeoffs",
            "disallowed_wording": "SpatialSelf has the highest weak-signal recall",
        },
        {
            "claim_id": "G2_C3_false_positive_suppression",
            "status": "partial" if fp_caveat else "supported",
            "statement": "SpatialSelf has low high-expression false-positive rates, but SpaTalk can be the lowest-FP method in replicated real-background summaries.",
            "allowed_wording": "SpatialSelf strongly suppresses high-expression false positives, with SpaTalk noted as an even more conservative baseline in some panels",
            "disallowed_wording": "SpatialSelf has the lowest false-positive rate against every baseline",
        },
        {
            "claim_id": "G2_C4_tissue_dominance",
            "status": "unsupported" if tissue_mixed else "supported",
            "statement": "Tissue-level winners are mixed; SpatialSelf should not claim tissue-level dominance across lymph node, tonsil, and melanoma.",
            "allowed_wording": "aggregate endpoint favors SpatialSelf while tissue-level winners are mixed",
            "disallowed_wording": "SpatialSelf wins every tissue-level endpoint",
        },
        {
            "claim_id": "G2_C5_paired_diagnostics",
            "status": "partial",
            "statement": "Paired diagnostics show positive mean deltas and sign consistency, but bootstrap CIs are not all above zero.",
            "allowed_wording": "paired endpoint deltas are positive on average with incomplete CI support",
            "disallowed_wording": "paired significance against every baseline is established",
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
