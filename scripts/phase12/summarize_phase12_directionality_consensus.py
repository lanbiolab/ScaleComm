#!/usr/bin/env python
"""Consensus directionality audit for Phase 12.

This script merges the Phase 12 directionality layers into one conservative
table:

- empirical expected-flow FDR;
- LR-label and niche-label permutation FDR;
- edge-level receiver-pairing and vector permutation FDR;
- SpatialSelf-vs-COMMOT directionality comparisons;
- high-expression negative leakage flags.

It does not create new model scores. Its purpose is to make the directionality
claim boundary explicit for manuscript writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "directionality_consensus"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guardrail = read_csv(PHASE12_DIR / "directionality_claim_guardrail/phase12_directionality_claim_guardrail_lr.csv")
    edge_method = read_csv(PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_method_comparison.csv")
    aggregate_method = read_csv(PHASE12_DIR / "directionality_benchmark/phase12_directionality_method_comparison.csv")

    lr_consensus = build_lr_consensus(guardrail, edge_method, aggregate_method)
    method_pair = build_method_pair_consensus(lr_consensus)
    dataset_summary = summarize_dataset(lr_consensus)
    method_summary = summarize_method(lr_consensus)

    lr_consensus.to_csv(OUT_DIR / "phase12_directionality_consensus_lr.csv", index=False)
    method_pair.to_csv(OUT_DIR / "phase12_directionality_consensus_method_pair.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_directionality_consensus_dataset_summary.csv", index=False)
    method_summary.to_csv(OUT_DIR / "phase12_directionality_consensus_method_summary.csv", index=False)
    write_status(dataset_summary, method_pair, lr_consensus)
    write_manifest(lr_consensus, method_pair, dataset_summary, method_summary)
    print(dataset_summary.to_string(index=False))
    print()
    print(method_pair.to_string(index=False))


def build_lr_consensus(
    guardrail: pd.DataFrame,
    edge_method: pd.DataFrame,
    aggregate_method: pd.DataFrame,
) -> pd.DataFrame:
    if guardrail.empty:
        return pd.DataFrame()
    rows = []
    for row in guardrail.itertuples(index=False):
        dataset_key = str(row.dataset_key)
        method = str(row.method)
        lr_pair = str(row.lr_pair)
        edge = pick(edge_method, dataset_key=dataset_key, lr_pair=lr_pair)
        agg = pick(aggregate_method, dataset_key=dataset_key, lr_pair=lr_pair)
        method_edge_call = method_specific_edge_call(method, edge)
        method_aggregate_call = method_specific_aggregate_call(method, agg)
        evidence = evidence_flags(row)
        consensus_call = classify_consensus(row, evidence, method_edge_call)
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": row.dataset,
                "method": method,
                "lr_pair": lr_pair,
                "expected_flow_score_fraction": safe_float(row.expected_flow_score_fraction),
                "expected_flow_bh_fdr": safe_float(row.expected_flow_bh_fdr),
                "lr_label_bh_fdr": safe_float(row.lr_label_bh_fdr),
                "niche_label_bh_fdr": safe_float(row.niche_label_bh_fdr),
                "receiver_pairing_bh_fdr": safe_float(row.receiver_pairing_bh_fdr),
                "edge_vector_bh_fdr": safe_float(row.edge_vector_bh_fdr),
                "source_sink_alignment_score": safe_float(row.source_sink_alignment_score),
                "vector_coherence": safe_float(row.vector_coherence),
                "flow_range_class": row.flow_range_class,
                "high_expression_negative": bool(row.high_expression_negative),
                "directionality_claim_call": row.directionality_claim_call,
                "method_edge_comparison_call": method_edge_call,
                "method_aggregate_comparison_call": method_aggregate_call,
                "n_consensus_evidence_flags": int(sum(evidence.values())),
                "consensus_support_level": int(consensus_level(consensus_call)),
                "evidence_flags": ";".join(k for k, v in evidence.items() if v),
                "directionality_consensus_call": consensus_call,
                "manuscript_use": manuscript_use(consensus_call),
            }
        )
    return pd.DataFrame(rows)


def evidence_flags(row: object) -> dict[str, bool]:
    return {
        "expected_flow_fdr_0_1": safe_float(row.expected_flow_bh_fdr) <= 0.1,
        "lr_label_fdr_0_1": safe_float(row.lr_label_bh_fdr) <= 0.1,
        "niche_label_fdr_0_1": safe_float(row.niche_label_bh_fdr) <= 0.1,
        "receiver_pairing_fdr_0_1": safe_float(row.receiver_pairing_bh_fdr) <= 0.1,
        "edge_vector_fdr_0_1": safe_float(row.edge_vector_bh_fdr) <= 0.1,
        "expected_flow_fraction_ge_0_5": safe_float(row.expected_flow_score_fraction) >= 0.5,
        "source_sink_alignment_ge_0_5": safe_float(row.source_sink_alignment_score) >= 0.5,
        "not_high_expression_negative": not bool(row.high_expression_negative),
    }


def classify_consensus(row: object, evidence: dict[str, bool], method_edge_call: str) -> str:
    if bool(row.high_expression_negative) and (
        evidence["receiver_pairing_fdr_0_1"] or evidence["expected_flow_fdr_0_1"]
    ):
        return "negative_directionality_leakage"
    if (
        evidence["receiver_pairing_fdr_0_1"]
        and evidence["edge_vector_fdr_0_1"]
        and evidence["niche_label_fdr_0_1"]
    ):
        return "strict_edge_vector_consensus"
    if evidence["receiver_pairing_fdr_0_1"] and evidence["niche_label_fdr_0_1"]:
        if method_edge_call.endswith("_supported_other_not"):
            return "method_specific_source_sink_consensus"
        return "shared_source_sink_consensus"
    if evidence["receiver_pairing_fdr_0_1"]:
        if method_edge_call.endswith("_supported_other_not"):
            return "method_specific_receiver_pairing_only"
        return "receiver_pairing_only"
    if evidence["expected_flow_fdr_0_1"] and evidence["niche_label_fdr_0_1"]:
        return "aggregate_niche_flow_only"
    if evidence["expected_flow_fraction_ge_0_5"]:
        return "descriptive_expected_flow_only"
    return "not_supported"


def build_method_pair_consensus(lr_consensus: pd.DataFrame) -> pd.DataFrame:
    if lr_consensus.empty:
        return pd.DataFrame()
    rows = []
    for (dataset_key, lr_pair), sub in lr_consensus.groupby(["dataset_key", "lr_pair"], sort=False):
        methods = {row.method: row for row in sub.itertuples(index=False)}
        if not {"spatialself", "commot"}.issubset(methods):
            continue
        ss = methods["spatialself"]
        cm = methods["commot"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": ss.dataset,
                "lr_pair": lr_pair,
                "spatialself_consensus_call": ss.directionality_consensus_call,
                "commot_consensus_call": cm.directionality_consensus_call,
                "spatialself_evidence_flags": ss.evidence_flags,
                "commot_evidence_flags": cm.evidence_flags,
                "spatialself_receiver_pairing_fdr": safe_float(ss.receiver_pairing_bh_fdr),
                "commot_receiver_pairing_fdr": safe_float(cm.receiver_pairing_bh_fdr),
                "spatialself_edge_vector_fdr": safe_float(ss.edge_vector_bh_fdr),
                "commot_edge_vector_fdr": safe_float(cm.edge_vector_bh_fdr),
                "spatialself_expected_flow_fraction": safe_float(ss.expected_flow_score_fraction),
                "commot_expected_flow_fraction": safe_float(cm.expected_flow_score_fraction),
                "consensus_method_comparison_call": compare_consensus(ss, cm),
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(lr_consensus: pd.DataFrame) -> pd.DataFrame:
    if lr_consensus.empty:
        return pd.DataFrame()
    rows = []
    supported_calls = {
        "strict_edge_vector_consensus",
        "method_specific_source_sink_consensus",
        "shared_source_sink_consensus",
        "method_specific_receiver_pairing_only",
        "receiver_pairing_only",
    }
    for (dataset_key, method), sub in lr_consensus.groupby(["dataset_key", "method"], sort=False):
        supported = sub[sub["directionality_consensus_call"].isin(supported_calls)]
        strict = sub[sub["directionality_consensus_call"].eq("strict_edge_vector_consensus")]
        method_specific = sub[sub["directionality_consensus_call"].astype(str).str.startswith("method_specific")]
        leakage = sub[sub["directionality_consensus_call"].eq("negative_directionality_leakage")]
        best = sub.sort_values(
            ["consensus_support_level", "n_consensus_evidence_flags", "receiver_pairing_bh_fdr", "expected_flow_score_fraction"],
            ascending=[False, False, True, False],
        ).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "n_consensus_supported": int(len(supported)),
                "n_strict_edge_vector_consensus": int(len(strict)),
                "n_method_specific_supported": int(len(method_specific)),
                "n_receiver_pairing_only": int(sub["directionality_consensus_call"].eq("receiver_pairing_only").sum()),
                "n_negative_directionality_leakage": int(len(leakage)),
                "best_lr": best.iloc[0]["lr_pair"] if len(best) else "",
                "best_lr_consensus_call": best.iloc[0]["directionality_consensus_call"] if len(best) else "",
                "best_lr_evidence_flags": best.iloc[0]["evidence_flags"] if len(best) else "",
                "dataset_consensus_call": dataset_call(supported, strict, leakage),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(lr_consensus: pd.DataFrame) -> pd.DataFrame:
    if lr_consensus.empty:
        return pd.DataFrame()
    rows = []
    for method, sub in lr_consensus.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_lr": int(len(sub)),
                "n_consensus_supported": int(
                    sub["directionality_consensus_call"].isin(
                        [
                            "strict_edge_vector_consensus",
                            "method_specific_source_sink_consensus",
                            "shared_source_sink_consensus",
                            "method_specific_receiver_pairing_only",
                            "receiver_pairing_only",
                        ]
                    ).sum()
                ),
                "n_strict_edge_vector_consensus": int(sub["directionality_consensus_call"].eq("strict_edge_vector_consensus").sum()),
                "n_method_specific_supported": int(sub["directionality_consensus_call"].astype(str).str.startswith("method_specific").sum()),
                "n_negative_directionality_leakage": int(sub["directionality_consensus_call"].eq("negative_directionality_leakage").sum()),
                "mean_consensus_evidence_flags": float(sub["n_consensus_evidence_flags"].mean()),
            }
        )
    return pd.DataFrame(rows)


def dataset_call(supported: pd.DataFrame, strict: pd.DataFrame, leakage: pd.DataFrame) -> str:
    if len(strict) > 0 and len(leakage) == 0:
        return "strict_vector_directionality_supported"
    if len(supported) >= 2 and len(leakage) == 0:
        return "source_sink_consensus_supported_vector_caveat"
    if len(supported) >= 1 and len(leakage) == 0:
        return "limited_source_sink_consensus"
    if len(leakage) > 0:
        return "directionality_confounded_by_negative_leakage"
    return "directionality_not_supported"


def compare_consensus(ss: object, cm: object) -> str:
    ss_level = consensus_level(ss.directionality_consensus_call)
    cm_level = consensus_level(cm.directionality_consensus_call)
    if ss_level > cm_level:
        return "spatialself_stronger_consensus"
    if cm_level > ss_level:
        return "commot_stronger_consensus"
    if (
        safe_float(ss.expected_flow_score_fraction) - safe_float(cm.expected_flow_score_fraction) > 0.1
        and ss_level > 0
    ):
        return "same_class_spatialself_higher_expected_flow"
    if (
        safe_float(cm.expected_flow_score_fraction) - safe_float(ss.expected_flow_score_fraction) > 0.1
        and cm_level > 0
    ):
        return "same_class_commot_higher_expected_flow"
    return "similar_or_no_consensus_difference"


def consensus_level(call: str) -> int:
    order = {
        "strict_edge_vector_consensus": 5,
        "method_specific_source_sink_consensus": 4,
        "shared_source_sink_consensus": 3,
        "method_specific_receiver_pairing_only": 2,
        "receiver_pairing_only": 1,
        "aggregate_niche_flow_only": 1,
        "descriptive_expected_flow_only": 0,
        "not_supported": 0,
        "negative_directionality_leakage": -1,
    }
    return order.get(str(call), 0)


def method_specific_edge_call(method: str, edge_row: pd.Series) -> str:
    call = str(get(edge_row, "edge_permutation_comparison_call", ""))
    if method == "spatialself" and call == "spatialself_supported_commot_not":
        return "spatialself_supported_other_not"
    if method == "commot" and call == "commot_supported_spatialself_not":
        return "commot_supported_other_not"
    if call == "both_supported":
        return "both_supported"
    if call == "similar_or_mixed":
        return "similar_or_mixed"
    return call


def method_specific_aggregate_call(method: str, agg_row: pd.Series) -> str:
    call = str(get(agg_row, "comparison_call", ""))
    if method == "spatialself":
        return call
    if method == "commot":
        if call.startswith("commot_"):
            return call
        if call.startswith("spatialself_"):
            return "other_method_" + call
    return call


def manuscript_use(call: str) -> str:
    if call == "strict_edge_vector_consensus":
        return "Strongest directionality evidence; still phrase as spatial source-sink consistency, not causality."
    if call in {"method_specific_source_sink_consensus", "shared_source_sink_consensus"}:
        return "Use as source-sink/receiver-pairing directionality support with vector caveat."
    if call in {"method_specific_receiver_pairing_only", "receiver_pairing_only"}:
        return "Use only as receiver-pairing support; do not claim vector-coherent transport."
    if call == "aggregate_niche_flow_only":
        return "Use as aggregate niche-flow consistency only."
    if call == "negative_directionality_leakage":
        return "Use as hard-negative leakage evidence."
    return "Do not use as a directionality claim."


def write_status(dataset: pd.DataFrame, method_pair: pd.DataFrame, lr_consensus: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Consensus Audit",
        "",
        "## Scope",
        "",
        "This table merges empirical expected-flow FDR, label permutation, edge receiver-pairing/vector permutation, high-expression negative flags and SpatialSelf-vs-COMMOT comparisons. It is a claim guardrail, not a causal transport proof.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}` `{row.dataset_consensus_call}`: "
            f"supported={int(row.n_consensus_supported)}/{int(row.n_lr)}, strict-vector={int(row.n_strict_edge_vector_consensus)}, "
            f"method-specific={int(row.n_method_specific_supported)}, negative-leakage={int(row.n_negative_directionality_leakage)}, "
            f"best={row.best_lr} ({row.best_lr_consensus_call})"
        )
    lines += ["", "## SpatialSelf vs COMMOT", ""]
    if not method_pair.empty:
        counts = method_pair["consensus_method_comparison_call"].value_counts()
        lines.append("; ".join(f"{idx}={int(val)}" for idx, val in counts.items()))
        for row in method_pair[
            method_pair["consensus_method_comparison_call"].astype(str).str.contains("spatialself|commot", na=False)
        ].sort_values(["dataset_key", "lr_pair"]).itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.lr_pair}`: SpatialSelf={row.spatialself_consensus_call}, "
                f"COMMOT={row.commot_consensus_call}, call={row.consensus_method_comparison_call}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Current consensus supports selected lymph-node source-sink/receiver-pairing claims. It does not support global SpatialSelf directionality superiority over COMMOT, and no dataset has strict edge-vector consensus under the current fixed-score nulls.",
    ]
    (OUT_DIR / "phase12_directionality_consensus_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr: pd.DataFrame, pair: pd.DataFrame, dataset: pd.DataFrame, method: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_directionality_consensus",
        "outputs": [
            rel(OUT_DIR / "phase12_directionality_consensus_lr.csv"),
            rel(OUT_DIR / "phase12_directionality_consensus_method_pair.csv"),
            rel(OUT_DIR / "phase12_directionality_consensus_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_directionality_consensus_method_summary.csv"),
            rel(OUT_DIR / "phase12_directionality_consensus_status.md"),
        ],
        "n_lr_rows": int(len(lr)),
        "n_method_pair_rows": int(len(pair)),
        "n_dataset_rows": int(len(dataset)),
        "n_method_rows": int(len(method)),
        "claim_boundary": "Consensus directionality claim guardrail; not causal transport proof.",
    }
    (OUT_DIR / "phase12_directionality_consensus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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


def get(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if row is None or len(row) == 0:
        return default
    return row[name] if name in row.index else default


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
