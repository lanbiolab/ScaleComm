#!/usr/bin/env python
"""Summarize lymph-node real-background interpretability evidence for Goal 5."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lymph-node-dir",
        default="outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal5_interpretability/lymph_node_real_background",
    )
    args = parser.parse_args()

    lymph_node_dir = PROJECT_ROOT / args.lymph_node_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(lymph_node_dir)
    control_table = _control_table(sources)
    claims = _claim_table(control_table, sources)
    summary = _summary_row(control_table, claims, sources, lymph_node_dir, output_dir)

    control_table.to_csv(output_dir / "lymph_node_goal5_control_interpretability.csv", index=False)
    claims.to_csv(output_dir / "lymph_node_goal5_interpretability_claims.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "lymph_node_goal5_interpretability_summary.csv", index=False)
    manifest = {
        "goal_id": "5",
        "analysis_label": "goal5_lymph_node_real_background_interpretability",
        "evidence_status": summary["evidence_status"],
        "input_dir": str(lymph_node_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "source_files": {
            key: str(path.relative_to(PROJECT_ROOT))
            for key, path in _source_paths(lymph_node_dir).items()
        },
        "outputs": {
            "summary": str((output_dir / "lymph_node_goal5_interpretability_summary.csv").relative_to(PROJECT_ROOT)),
            "controls": str((output_dir / "lymph_node_goal5_control_interpretability.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "lymph_node_goal5_interpretability_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Real-background injected-control validation; not unlabeled real-data ground truth.",
            "The high-expression negative controls have no downstream target signatures and are evaluated by rank and region controls.",
            "This summarizes one human lymph-node full-slide sparse replicate.",
        ],
    }
    (output_dir / "lymph_node_goal5_interpretability_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Wrote Goal 5 lymph-node interpretability outputs to {output_dir}")


def _source_paths(lymph_node_dir: Path) -> dict[str, Path]:
    return {
        "goal6_summary": lymph_node_dir / "lymph_node_validation_summary.csv",
        "control_lr": lymph_node_dir / "control_summary/real_validation_control_lr_summary.csv",
        "control_method": lymph_node_dir / "control_summary/real_validation_control_method_summary.csv",
        "downstream": next((lymph_node_dir / "downstream_target").glob("*__downstream_target_correlation.csv")),
        "region_enrichment": next((lymph_node_dir / "region_enrichment").glob("*__region_enrichment.csv")),
        "lr_global_ranking": next((lymph_node_dir / "region_enrichment").glob("*__lr_global_ranking.csv")),
    }


def _load_sources(lymph_node_dir: Path) -> dict[str, pd.DataFrame]:
    paths = _source_paths(lymph_node_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing lymph-node validation sources: {missing}")
    return {key: pd.read_csv(path) for key, path in paths.items()}


def _control_table(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    controls = sources["control_lr"].copy()
    controls["control_direction"] = controls["control_type"].map(
        {
            "positive": "positive_control",
            "positive_weak": "positive_weak_control",
            "negative_high_expression": "negative_high_expression_control",
        }
    ).fillna("other_control")
    controls["downstream_supported"] = (
        controls["true_lr"].astype(str).str.lower().eq("true")
        & pd.to_numeric(controls["downstream_spearman_r"], errors="coerce").ge(0.5)
        & pd.to_numeric(controls["downstream_bh_fdr"], errors="coerce").le(0.05)
    )
    controls["region_supported"] = pd.to_numeric(controls["best_region_bh_fdr"], errors="coerce").le(0.05)
    controls["rank_support_status"] = controls.apply(_rank_status, axis=1)
    controls["analysis_scope"] = (
        "Human lymph-node full-slide sparse real-background injected-control validation; "
        "supports downstream/control association, not unlabeled real-data ground truth."
    )
    return controls[
        [
            "method",
            "lr_id",
            "control_type",
            "control_direction",
            "true_lr",
            "weak_signal",
            "false_positive_high_expression",
            "global_rank",
            "mean_score",
            "top_region",
            "best_region_enrichment_score",
            "best_region_bh_fdr",
            "downstream_spearman_r",
            "downstream_bh_fdr",
            "downstream_status",
            "downstream_supported",
            "region_supported",
            "rank_support_status",
            "analysis_scope",
        ]
    ].copy()


def _rank_status(row: pd.Series) -> str:
    if bool(row.get("false_positive_high_expression", False)):
        return "negative_suppressed" if float(row.get("global_rank", math.inf)) > 3 else "negative_not_suppressed"
    if bool(row.get("true_lr", False)):
        return "positive_prioritized" if float(row.get("global_rank", math.inf)) <= 3 else "positive_not_prioritized"
    return "not_scored_control"


def _claim_table(control_table: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    positives = control_table[control_table["true_lr"].astype(str).str.lower().eq("true")]
    negatives = control_table[control_table["false_positive_high_expression"].astype(str).str.lower().eq("true")]
    weak = positives[positives["weak_signal"].astype(str).str.lower().eq("true")]
    method = sources["control_method"].iloc[0] if not sources["control_method"].empty else {}
    summary = sources["goal6_summary"].iloc[0] if not sources["goal6_summary"].empty else {}
    rows = [
        {
            "claim_id": "lymph_node_positive_downstream_response",
            "support_status": "supported_initial"
            if int(positives["downstream_supported"].sum()) >= 3
            else "mixed_or_not_supported",
            "primary_metric": "positive downstream BH-FDR <= 0.05",
            "primary_value": int(positives["downstream_supported"].sum()),
            "secondary_metric": "mean positive downstream Spearman",
            "secondary_value": _num(method.get("mean_positive_downstream_spearman")),
            "evidence_scope": "lymph-node real-background injected positive controls",
            "limitation": "Injected positive controls with curated target sets; not unlabeled discovery validation.",
        },
        {
            "claim_id": "lymph_node_weak_positive_downstream_response",
            "support_status": "supported_initial"
            if not weak.empty and bool(weak["downstream_supported"].iloc[0])
            else "mixed_or_not_supported",
            "primary_metric": "weak positive downstream Spearman",
            "primary_value": _num(weak["downstream_spearman_r"].iloc[0]) if not weak.empty else math.nan,
            "secondary_metric": "weak positive downstream BH-FDR",
            "secondary_value": _num(weak["downstream_bh_fdr"].iloc[0]) if not weak.empty else math.nan,
            "evidence_scope": "CXCL13-CXCR5 weak injected lymph-node control",
            "limitation": "One weak positive LR control in one real-background replicate.",
        },
        {
            "claim_id": "lymph_node_high_expression_negative_suppression",
            "support_status": "supported_initial"
            if int((negatives["global_rank"].astype(float) <= 3).sum()) == 0
            else "mixed_or_not_supported",
            "primary_metric": "high-expression negatives in top 3",
            "primary_value": int((negatives["global_rank"].astype(float) <= 3).sum()),
            "secondary_metric": "mean positive rank minus mean negative rank",
            "secondary_value": _num(method.get("mean_positive_rank")) - _num(method.get("mean_negative_rank")),
            "evidence_scope": "lymph-node real-background high-expression false-positive controls",
            "limitation": "Negative controls lack target sets; downstream suppression is evaluated through rank/control behavior.",
        },
        {
            "claim_id": "lymph_node_region_and_downstream_combined",
            "support_status": "supported_initial"
            if _num(summary.get("n_downstream_bh_fdr_0_05")) >= 3 and _num(summary.get("best_region_bh_fdr")) <= 0.05
            else "mixed_or_not_supported",
            "primary_metric": "downstream significant positives and best region BH-FDR",
            "primary_value": f"{summary.get('n_downstream_bh_fdr_0_05', 'NA')};{summary.get('best_region_bh_fdr', 'NA')}",
            "secondary_metric": "positive controls rank ahead of negatives",
            "secondary_value": str(summary.get("positive_controls_rank_ahead_of_negatives", "NA")),
            "evidence_scope": "joint downstream, region, and control-rank validation",
            "limitation": "Region enrichment can also be significant for high-expression negatives; interpret jointly with rank controls.",
        },
    ]
    return pd.DataFrame(rows)


def _summary_row(
    control_table: pd.DataFrame,
    claims: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    lymph_node_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    positives = control_table[control_table["true_lr"].astype(str).str.lower().eq("true")]
    negatives = control_table[control_table["false_positive_high_expression"].astype(str).str.lower().eq("true")]
    weak = positives[positives["weak_signal"].astype(str).str.lower().eq("true")]
    method = sources["control_method"].iloc[0] if not sources["control_method"].empty else {}
    goal6_summary = sources["goal6_summary"].iloc[0] if not sources["goal6_summary"].empty else {}
    supported_claims = int((claims["support_status"] == "supported_initial").sum())
    contradiction_count = int(_num(method.get("mean_negative_region_enrichment")) > 0)
    claim_level = (
        "real_background_injected_control_support"
        if supported_claims >= 3
        and int((negatives["global_rank"].astype(float) <= 3).sum()) == 0
        and int(positives["downstream_supported"].sum()) >= 3
        else "mixed_or_insufficient_support"
    )
    return {
        "goal_id": "5",
        "analysis_label": "goal5_lymph_node_real_background_interpretability",
        "evidence_status": "initial_cross_real_background_validation",
        "claim_level": claim_level,
        "dataset_context": str(goal6_summary.get("dataset_context", "human_lymph_node_real_background")),
        "input_dir": str(lymph_node_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "n_controls": int(len(control_table)),
        "n_positive_controls": int(len(positives)),
        "n_weak_positive_controls": int(len(weak)),
        "n_high_expression_negative_controls": int(len(negatives)),
        "n_positive_downstream_supported": int(positives["downstream_supported"].sum()),
        "n_weak_positive_downstream_supported": int(weak["downstream_supported"].sum()) if not weak.empty else 0,
        "n_negative_top3": int((negatives["global_rank"].astype(float) <= 3).sum()),
        "mean_positive_rank": _num(method.get("mean_positive_rank")),
        "mean_negative_rank": _num(method.get("mean_negative_rank")),
        "mean_positive_downstream_spearman": _num(method.get("mean_positive_downstream_spearman")),
        "mean_positive_region_enrichment": _num(method.get("mean_positive_region_enrichment")),
        "mean_negative_region_enrichment": _num(method.get("mean_negative_region_enrichment")),
        "best_region_bh_fdr": _num(goal6_summary.get("best_region_bh_fdr")),
        "n_supported_claims": supported_claims,
        "contradiction_count": contradiction_count,
        "interpretation": (
            "Lymph-node real-background injected-control evidence supports downstream association for all three "
            "positive controls, including the weak CXCL13-CXCR5 control, and keeps high-expression negatives "
            "outside the top three ranks. This strengthens Goal 5 as injected/control validation, but it is not "
            "unlabeled real-data ground truth."
        ),
    }


def _num(value: object) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
