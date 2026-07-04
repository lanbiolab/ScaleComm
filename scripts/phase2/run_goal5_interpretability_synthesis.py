#!/usr/bin/env python
"""Synthesize Phase 2 Goal 5 interpretability evidence across diagnostics."""

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
    parser.add_argument("--output-dir", default="outputs/phase2/goal5_interpretability/synthesis")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources()
    summary = _summary_row(sources)
    claims = _claim_rows(sources, summary)
    metrics = _metric_rows(sources)

    pd.DataFrame([summary]).to_csv(output_dir / "interpretability_synthesis_summary.csv", index=False)
    pd.DataFrame(claims).to_csv(output_dir / "interpretability_synthesis_claims.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "interpretability_synthesis_metrics.csv", index=False)
    manifest = {
        "goal_id": "5",
        "analysis_label": "goal5_interpretability_synthesis",
        "evidence_status": summary["evidence_status"],
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "source_files": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in _source_paths().items()},
        "limitations": [
            "Synthesis of existing Phase 2 outputs; no new biological validation is introduced here.",
            "720-cell tonsil and 1,800-cell stress-test evidence are not interchangeable.",
            "Interpretability claims are constrained to injected/control and association evidence, not real ground truth.",
        ],
    }
    (output_dir / "interpretability_synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Wrote Goal 5 synthesis outputs to {output_dir}")


def _load_sources() -> dict[str, pd.DataFrame]:
    return {key: pd.read_csv(path) for key, path in _source_paths().items()}


def _source_paths() -> dict[str, Path]:
    return {
        "fresh_summary": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/goal5_interpretability_summary.csv",
        "fresh_controls": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_interpretability_control_summary.csv",
        "n1800_summary": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/goal5_interpretability_summary.csv",
        "n1800_controls": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_interpretability_control_summary.csv",
        "embedding_summary": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/goal5_lr_embedding_summary.csv",
        "mechanism_summary": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/goal5_mechanism_category_summary.csv",
        "lymph_node_summary": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_summary.csv",
        "lymph_node_claims": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_claims.csv",
        "lymph_node_controls": PROJECT_ROOT / "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_control_interpretability.csv",
        "module720_overview": PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/tonsil_ccc_module_overview.csv",
        "module720_enrichment": PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/tonsil_ccc_module_control_enrichment.csv",
        "module1800_overview": PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_overview.csv",
        "module1800_enrichment": PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_control_enrichment.csv",
    }


def _summary_row(sources: dict[str, pd.DataFrame]) -> dict[str, object]:
    fresh = sources["fresh_summary"].iloc[0]
    n1800 = sources["n1800_summary"].iloc[0]
    embedding = sources["embedding_summary"].iloc[0]
    mechanism = sources["mechanism_summary"].iloc[0]
    lymph_node = sources["lymph_node_summary"].iloc[0]
    module720_pos_fdr = _control_fdr(sources["module720_enrichment"], "tonsil_positive_prior")
    module720_neg_fdr = _control_fdr(sources["module720_enrichment"], "housekeeping_negative_control")
    module1800_pos_fdr = _control_fdr(sources["module1800_enrichment"], "tonsil_positive_prior")
    module1800_neg_fdr = _control_fdr(sources["module1800_enrichment"], "housekeeping_negative_control")

    fresh_positive_minus_negative = _num(fresh["positive_minus_negative_mean_residual_spearman"])
    n1800_positive_minus_negative = _num(n1800["positive_minus_negative_mean_residual_spearman"])
    fresh_background_gap = _num(fresh["positive_control_mean_residual_spearman"]) - _num(
        fresh["background_lr_mean_residual_spearman"]
    )
    n1800_background_gap = _num(n1800["positive_control_mean_residual_spearman"]) - _num(
        n1800["background_lr_mean_residual_spearman"]
    )
    fresh_support = (
        fresh_positive_minus_negative > 0
        and _num(fresh["significant_negative_controls_residual_fdr_0_10"]) == 0
        and _num(fresh["significant_positive_controls_residual_fdr_0_10"]) > 0
    )
    stress_support = n1800_positive_minus_negative > 0 and _num(n1800["significant_positive_controls_residual_fdr_0_10"]) > 0
    module_support = module720_pos_fdr <= 0.10 and module720_neg_fdr <= 0.10
    module_stress_support = module1800_pos_fdr <= 0.10 and module1800_neg_fdr <= 0.10
    embedding_support = (
        _num(embedding["silhouette_mechanism"]) > 0
        or _num(embedding["silhouette_control_class"]) > 0
        or _num(embedding["n_significant_cluster_enrichments_fdr_0_10"]) > 0
    )
    mechanism_support = (
        _num(mechanism["positive_minus_housekeeping_residual_spearman"]) > 0
        and _num(mechanism["n_housekeeping_negatives_residual_bh_fdr_0_10"]) == 0
    )
    lymph_node_support = (
        _num(lymph_node["n_positive_downstream_supported"]) >= 3
        and _num(lymph_node["n_weak_positive_downstream_supported"]) >= 1
        and _num(lymph_node["n_negative_top3"]) == 0
    )
    support_score = sum(
        [
            fresh_support,
            stress_support,
            module_support,
            module_stress_support,
            embedding_support,
            mechanism_support,
            lymph_node_support,
        ]
    )
    contradiction_score = sum(
        [
            fresh_background_gap < 0,
            n1800_positive_minus_negative <= 0,
            n1800_background_gap < 0,
            not module_stress_support,
            not embedding_support,
        ]
    )
    if fresh_support and mechanism_support and module_support and contradiction_score >= 3:
        claim_level = "mixed_initial_support_not_manuscript_ready"
    elif support_score >= 5 and contradiction_score <= 1:
        claim_level = "broad_initial_support"
    else:
        claim_level = "insufficient_or_mixed_support"
    return {
        "goal_id": "5",
        "analysis_label": "goal5_interpretability_synthesis",
        "evidence_status": "synthesis_completed",
        "claim_level": claim_level,
        "support_score": int(support_score),
        "contradiction_score": int(contradiction_score),
        "fresh_positive_minus_housekeeping_residual": fresh_positive_minus_negative,
        "fresh_positive_minus_background_residual": fresh_background_gap,
        "fresh_positive_residual_sig_controls": int(_num(fresh["significant_positive_controls_residual_fdr_0_10"])),
        "fresh_negative_residual_sig_controls": int(_num(fresh["significant_negative_controls_residual_fdr_0_10"])),
        "n1800_positive_minus_housekeeping_residual": n1800_positive_minus_negative,
        "n1800_positive_minus_background_residual": n1800_background_gap,
        "n1800_positive_residual_sig_controls": int(_num(n1800["significant_positive_controls_residual_fdr_0_10"])),
        "n1800_negative_residual_sig_controls": int(_num(n1800["significant_negative_controls_residual_fdr_0_10"])),
        "module720_positive_bh_fdr": module720_pos_fdr,
        "module720_housekeeping_bh_fdr": module720_neg_fdr,
        "module1800_positive_bh_fdr": module1800_pos_fdr,
        "module1800_housekeeping_bh_fdr": module1800_neg_fdr,
        "embedding_silhouette_mechanism": _num(embedding["silhouette_mechanism"]),
        "embedding_silhouette_control_class": _num(embedding["silhouette_control_class"]),
        "embedding_significant_cluster_enrichments": int(_num(embedding["n_significant_cluster_enrichments_fdr_0_10"])),
        "mechanism_positive_minus_housekeeping_residual": _num(
            mechanism["positive_minus_housekeeping_residual_spearman"]
        ),
        "mechanism_significant_group_enrichments": int(_num(mechanism["n_significant_group_enrichments_fdr_0_10"])),
        "lymph_node_positive_downstream_supported": int(_num(lymph_node["n_positive_downstream_supported"])),
        "lymph_node_weak_positive_downstream_supported": int(
            _num(lymph_node["n_weak_positive_downstream_supported"])
        ),
        "lymph_node_negative_top3": int(_num(lymph_node["n_negative_top3"])),
        "lymph_node_mean_positive_downstream_spearman": _num(
            lymph_node["mean_positive_downstream_spearman"]
        ),
        "lymph_node_claim_level": str(lymph_node["claim_level"]),
        "interpretation": (
            "Goal 5 currently supports only cautious, mixed interpretability claims: the 720-cell tonsil subset "
            "shows positive-vs-housekeeping residual and module/control signals, and lymph-node real-background "
            "injected controls support downstream/rank validation. However, broad background LR remain strong, "
            "LR embedding clustering is weak, and the 1,800-cell tonsil stress test does not preserve the "
            "positive-control downstream advantage."
        ),
    }


def _claim_rows(sources: dict[str, pd.DataFrame], summary: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "claim_id": "downstream_720_positive_vs_housekeeping",
            "support_status": "supported_initial" if summary["fresh_positive_minus_housekeeping_residual"] > 0 else "not_supported",
            "primary_metric": "positive minus housekeeping residual Spearman",
            "primary_value": summary["fresh_positive_minus_housekeeping_residual"],
            "contradiction": "background LR residual mean is higher than positive controls",
            "evidence_scope": "720-cell tonsil subset only",
        },
        {
            "claim_id": "downstream_n1800_stress_test",
            "support_status": "not_supported"
            if summary["n1800_positive_minus_housekeeping_residual"] <= 0
            else "supported_initial",
            "primary_metric": "positive minus housekeeping residual Spearman",
            "primary_value": summary["n1800_positive_minus_housekeeping_residual"],
            "contradiction": "larger subset does not preserve positive-control residual advantage",
            "evidence_scope": "1,800-cell tonsil stress test",
        },
        {
            "claim_id": "ccc_module_control_separation",
            "support_status": "mixed_support"
            if summary["module720_positive_bh_fdr"] <= 0.10 and summary["module1800_positive_bh_fdr"] > 0.10
            else "supported_initial",
            "primary_metric": "positive-control module BH-FDR 720 / 1800",
            "primary_value": f"{summary['module720_positive_bh_fdr']:.4g}/{summary['module1800_positive_bh_fdr']:.4g}",
            "contradiction": "positive module enrichment weakens at 1,800 cells",
            "evidence_scope": "tonsil module diagnostics",
        },
        {
            "claim_id": "lr_embedding_category_structure",
            "support_status": "not_supported"
            if summary["embedding_significant_cluster_enrichments"] == 0
            else "supported_initial",
            "primary_metric": "cluster enrichments and silhouette",
            "primary_value": (
                f"enrich={summary['embedding_significant_cluster_enrichments']}; "
                f"sil_mech={summary['embedding_silhouette_mechanism']:.3g}"
            ),
            "contradiction": "silhouette values are negative and no BH-significant enrichments remain",
            "evidence_scope": "720-cell tonsil LR embedding diagnostic",
        },
        {
            "claim_id": "mechanism_category_association",
            "support_status": "supported_initial"
            if summary["mechanism_positive_minus_housekeeping_residual"] > 0
            else "not_supported",
            "primary_metric": "mechanism positive minus housekeeping residual Spearman",
            "primary_value": summary["mechanism_positive_minus_housekeeping_residual"],
            "contradiction": "small positive-control category counts and strong background LR",
            "evidence_scope": "720-cell tonsil mechanism/category diagnostic",
        },
        {
            "claim_id": "lymph_node_real_background_injected_controls",
            "support_status": "supported_initial"
            if summary["lymph_node_positive_downstream_supported"] >= 3
            and summary["lymph_node_weak_positive_downstream_supported"] >= 1
            and summary["lymph_node_negative_top3"] == 0
            else "mixed_support",
            "primary_metric": "positive downstream support / weak support / negative top3",
            "primary_value": (
                f"{summary['lymph_node_positive_downstream_supported']}/"
                f"{summary['lymph_node_weak_positive_downstream_supported']}/"
                f"{summary['lymph_node_negative_top3']}"
            ),
            "contradiction": "real-background injected controls are not unlabeled real-data ground truth",
            "evidence_scope": "human lymph-node full-slide sparse real-background injected-control validation",
        },
    ]


def _metric_rows(sources: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows = []
    for dataset, key in [("tonsil_720", "fresh_summary"), ("tonsil_1800", "n1800_summary")]:
        row = sources[key].iloc[0]
        for metric in [
            "positive_control_mean_residual_spearman",
            "housekeeping_negative_mean_residual_spearman",
            "background_lr_mean_residual_spearman",
            "positive_minus_negative_mean_residual_spearman",
            "significant_positive_controls_residual_fdr_0_10",
            "significant_negative_controls_residual_fdr_0_10",
        ]:
            rows.append({"source": dataset, "metric": metric, "value": row[metric]})
    lymph = sources["lymph_node_summary"].iloc[0]
    for metric in [
        "n_positive_downstream_supported",
        "n_weak_positive_downstream_supported",
        "n_negative_top3",
        "mean_positive_downstream_spearman",
        "mean_positive_region_enrichment",
        "mean_negative_region_enrichment",
        "n_supported_claims",
        "contradiction_count",
    ]:
        rows.append({"source": "lymph_node_real_background", "metric": metric, "value": lymph[metric]})
    return rows


def _control_fdr(enrichment: pd.DataFrame, control_class: str) -> float:
    subset = enrichment[enrichment["control_class"] == control_class].copy()
    if subset.empty:
        return math.nan
    return float(pd.to_numeric(subset["bh_fdr"], errors="coerce").min())


def _num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
