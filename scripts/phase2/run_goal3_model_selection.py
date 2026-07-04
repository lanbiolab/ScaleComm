#!/usr/bin/env python
"""Assemble Phase 2 Goal 3 model-selection evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CANDIDATES = {
    "wavelet_default": {
        "label": "Wavelet default",
        "ablation": "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
        "method": "spatialself_wavelet",
        "real_method": "spatialself_wavelet",
    },
    "wavelet_edge_prior": {
        "label": "Wavelet + edge prior",
        "ablation": "wavelet_mechanism_niche_edge_prior_ranking",
        "method": "spatialself_wavelet_edge_prior",
        "real_method": "spatialself_wavelet_edge_prior",
    },
    "wavelet_weak_mining": {
        "label": "Wavelet + weak mining",
        "ablation": "phase2_weak_mining_edge_prior_wavelet",
        "method": "phase2_weak_mining_edge_prior_wavelet",
        "real_method": "",
    },
    "wavelet_focal_weak_mining": {
        "label": "Wavelet + focal weak mining",
        "ablation": "phase2_focal_weak_mining_edge_prior_wavelet",
        "method": "phase2_focal_weak_mining_edge_prior_wavelet",
        "real_method": "",
    },
    "wavelet_calibrated_fdr": {
        "label": "Wavelet + edge prior + calibrated FDR",
        "ablation": "wavelet_mechanism_niche_edge_prior_ranking",
        "method": "spatialself_wavelet_edge_prior",
        "real_method": "spatialself_wavelet_edge_prior",
        "inherits_edge_prior": True,
    },
    "gat_reference": {
        "label": "GAT reference",
        "ablation": "gat_hard_expression_distance",
        "method": "spatialself_gat",
        "real_method": "",
    },
    "no_lr_identity": {
        "label": "No LR identity",
        "ablation": "wavelet_no_lr_embedding",
        "method": "",
        "real_method": "",
    },
    "no_distance": {
        "label": "No distance",
        "ablation": "wavelet_no_distance",
        "method": "",
        "real_method": "",
    },
    "no_expression": {
        "label": "No expression",
        "ablation": "wavelet_no_expression_features",
        "method": "",
        "real_method": "",
    },
    "hybrid_wavelet_gat": {
        "label": "Hybrid wavelet-GAT",
        "ablation": "",
        "method": "",
        "real_method": "",
    },
}


def run_model_selection(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _load_sources()
    rows = [_candidate_row(candidate_id, spec, sources) for candidate_id, spec in CANDIDATES.items()]
    evidence = pd.DataFrame(rows)
    evidence = _score_candidates(evidence)
    criteria = _criteria_table(evidence)
    recommendation = _recommendation(evidence)

    evidence_path = output_dir / "model_selection_evidence.csv"
    criteria_path = output_dir / "model_selection_criteria.csv"
    recommendation_path = output_dir / "model_selection_recommendation.json"
    md_path = output_dir / "model_selection_summary.md"
    evidence.to_csv(evidence_path, index=False)
    criteria.to_csv(criteria_path, index=False)
    recommendation_path.write_text(json.dumps(recommendation, indent=2, allow_nan=True), encoding="utf-8")
    md_path.write_text(_markdown_summary(evidence, criteria, recommendation), encoding="utf-8")
    manifest = {
        "output_dir": str(output_dir),
        "evidence_path": str(evidence_path),
        "criteria_path": str(criteria_path),
        "recommendation_path": str(recommendation_path),
        "summary_path": str(md_path),
        "sources": {name: str(path) for name, path in _source_paths().items()},
    }
    (output_dir / "model_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return evidence, criteria, recommendation


def _load_sources() -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for name, path in _source_paths().items():
        sources[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return sources


def _source_paths() -> dict[str, Path]:
    return {
        "controlled_goal1": PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv",
        "full_slide_goal1": PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/weak_signal_recovery_summary.csv",
        "threshold_tradeoff": PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv",
        "lr_balanced_threshold": PROJECT_ROOT / "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv",
        "direct_group_fdr_500": PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv",
        "direct_group_seed_stability": PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/direct_group_seed_stability_summary.csv",
        "direct_group_replicate_stability": PROJECT_ROOT / "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_summary.csv",
        "component_ablation": PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv",
        "controlled_baseline": PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core/baseline_metrics_summary.csv",
        "full_slide_baseline": PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_metrics_summary.csv",
        "lymph_node_control": PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_control_summary/real_validation_control_method_summary.csv",
    }


def _candidate_row(candidate_id: str, spec: dict[str, object], sources: dict[str, pd.DataFrame]) -> dict[str, object]:
    ablation = str(spec.get("ablation", ""))
    method = str(spec.get("method", ""))
    real_method = str(spec.get("real_method", ""))
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "label": spec["label"],
        "ablation": ablation,
        "method": method,
        "real_method": real_method,
        "evidence_status": "missing",
    }

    controlled = _lookup(sources["controlled_goal1"], "ablation", ablation)
    if controlled.empty:
        controlled = _lookup(sources["component_ablation"], "ablation", ablation)
    if controlled.empty and method:
        controlled = _lookup(sources["controlled_baseline"], "method", method)
    full_slide = _lookup(sources["full_slide_goal1"], "ablation", ablation)
    if full_slide.empty and method:
        full_slide = _lookup(sources["full_slide_baseline"], "method", method)

    _copy_metrics(row, controlled, "controlled")
    _copy_metrics(row, full_slide, "full_slide")
    _copy_threshold(row, candidate_id, sources["threshold_tradeoff"])
    _copy_lr_balanced_threshold(row, candidate_id, sources["lr_balanced_threshold"])
    _copy_fdr(row, candidate_id, sources["direct_group_fdr_500"])
    _copy_fdr_seed_stability(row, candidate_id, sources["direct_group_seed_stability"])
    _copy_fdr_replicate_stability(row, candidate_id, sources["direct_group_replicate_stability"])
    _copy_lymph_node(row, real_method, sources["lymph_node_control"])
    _copy_evidence_flags(row)
    return row


def _copy_metrics(row: dict[str, object], record: pd.DataFrame, prefix: str) -> None:
    columns = {
        "auroc_mean": f"{prefix}_auroc",
        "aupr_mean": f"{prefix}_aupr",
        "weak_signal_recall_at_threshold_mean": f"{prefix}_weak_recall",
        "empirical_fdr_at_threshold_mean": f"{prefix}_fdr",
        "false_positive_high_expression_rate_at_threshold_mean": f"{prefix}_fp_high_expr_rate",
        "calibration_error_10bin_mean": f"{prefix}_calibration_error",
    }
    for source, target in columns.items():
        row[target] = _value(record, source)
    row[f"{prefix}_evidence"] = "available" if not record.empty else "missing"


def _copy_threshold(row: dict[str, object], candidate_id: str, threshold: pd.DataFrame) -> None:
    if candidate_id not in {"wavelet_edge_prior", "wavelet_calibrated_fdr"} or threshold.empty:
        row["operating_point_status"] = "not_applicable"
        return
    record = _lookup(threshold, "label", "edge_prior_ranking")
    row["operating_point_status"] = str(_value(record, "status", "missing"))
    row["operating_point_source"] = "global_quantile"
    row["operating_quantile"] = _value(record, "best_quantile")
    row["operating_lr_quantile"] = float("nan")
    row["operating_global_floor_quantile"] = float("nan")
    row["operating_weak_recall"] = _value(record, "best_weak_signal_recall")
    row["operating_fdr"] = _value(record, "best_fdr")
    row["operating_fdr_delta_vs_95"] = _value(record, "best_fdr_delta_vs_95")
    row["operating_fp_high_expr_rate"] = _value(record, "best_false_positive_high_expression_rate")


def _copy_lr_balanced_threshold(row: dict[str, object], candidate_id: str, threshold: pd.DataFrame) -> None:
    if candidate_id not in {"wavelet_edge_prior", "wavelet_calibrated_fdr"} or threshold.empty:
        row["lr_balanced_operating_point_status"] = "not_applicable"
        return
    record = _lookup(threshold, "strategy", "lrwise_quantile_with_global_floor")
    row["lr_balanced_operating_point_status"] = str(_value(record, "status", "missing"))
    row["lr_balanced_strategy"] = _value(record, "strategy", "")
    row["lr_balanced_lr_quantile"] = _value(record, "best_lr_quantile")
    row["lr_balanced_global_floor_quantile"] = _value(record, "best_global_floor_quantile")
    row["lr_balanced_weak_recall"] = _value(record, "best_weak_signal_recall")
    row["lr_balanced_fdr"] = _value(record, "best_fdr")
    row["lr_balanced_fdr_delta_vs_reference"] = _value(record, "best_fdr_delta_vs_reference")
    row["lr_balanced_fp_high_expr_rate"] = _value(record, "best_false_positive_high_expression_rate")
    row["lr_balanced_claim_scope"] = (
        "full-slide single-replicate post-hoc diagnostic; controlled replicate stability failed"
    )


def _copy_fdr(row: dict[str, object], candidate_id: str, grid: pd.DataFrame) -> None:
    if candidate_id != "wavelet_calibrated_fdr" or grid.empty:
        row["group_fdr_status"] = "not_applicable"
        return
    candidates = grid[
        (grid["status"] == "completed")
        & (grid["group_level"] == "lr_pair")
        & (grid["null_model"].isin(["score_permutation", "lr_identity_permutation"]))
        & (pd.to_numeric(grid["negative_control_significant_groups"], errors="coerce") == 0)
    ].copy()
    if candidates.empty:
        row["group_fdr_status"] = "missing"
        return
    candidates["positive_control_min_group_bh_fdr"] = pd.to_numeric(
        candidates["positive_control_min_group_bh_fdr"], errors="coerce"
    )
    best = candidates.sort_values(
        ["positive_control_min_group_bh_fdr", "positive_control_significant_groups"],
        ascending=[True, False],
    ).iloc[0]
    row["group_fdr_status"] = "available"
    row["group_fdr_null_model"] = best["null_model"]
    row["group_fdr_level"] = best["group_level"]
    row["group_fdr_top_fraction"] = float(best["top_fraction"])
    row["group_fdr_positive_bh_fdr"] = float(best["positive_control_min_group_bh_fdr"])
    row["group_fdr_positive_sig_groups"] = int(best["positive_control_significant_groups"])
    row["group_fdr_negative_sig_groups"] = int(best["negative_control_significant_groups"])


def _copy_fdr_seed_stability(row: dict[str, object], candidate_id: str, table: pd.DataFrame) -> None:
    if candidate_id != "wavelet_calibrated_fdr" or table.empty:
        row["group_fdr_seed_stability_status"] = "not_applicable"
        return
    record = table.iloc[0]
    row["group_fdr_seed_stability_status"] = str(record.get("status", "missing"))
    row["group_fdr_seed_stability_claim_level"] = str(record.get("claim_level", ""))
    row["group_fdr_seed_stability_n_seeds"] = _num(record.get("n_seeds"))
    row["group_fdr_seed_stability_n_completed_seeds"] = _num(record.get("n_completed_seeds"))
    row["group_fdr_seed_stability_passing_fraction"] = _num(record.get("passing_seed_fraction"))
    row["group_fdr_seed_stability_min_positive_sig_groups"] = _num(
        record.get("min_positive_control_significant_groups")
    )
    row["group_fdr_seed_stability_max_negative_sig_groups"] = _num(
        record.get("max_negative_control_significant_groups")
    )
    row["group_fdr_seed_stability_n_permutations"] = _num(record.get("n_permutations"))


def _copy_fdr_replicate_stability(row: dict[str, object], candidate_id: str, table: pd.DataFrame) -> None:
    if candidate_id != "wavelet_calibrated_fdr" or table.empty:
        row["group_fdr_replicate_stability_status"] = "not_applicable"
        return
    record = table.iloc[0]
    row["group_fdr_replicate_stability_status"] = str(record.get("status", "missing"))
    row["group_fdr_replicate_stability_claim_level"] = str(record.get("claim_level", ""))
    row["group_fdr_replicate_stability_n_replicates"] = _num(record.get("n_replicates"))
    row["group_fdr_replicate_stability_n_completed_replicates"] = _num(record.get("n_completed_replicates"))
    row["group_fdr_replicate_stability_passing_fraction"] = _num(record.get("passing_replicate_fraction"))
    row["group_fdr_replicate_stability_min_positive_sig_groups"] = _num(
        record.get("min_positive_control_significant_groups")
    )
    row["group_fdr_replicate_stability_max_negative_sig_groups"] = _num(
        record.get("max_negative_control_significant_groups")
    )
    row["group_fdr_replicate_stability_n_permutations"] = _num(record.get("n_permutations"))


def _copy_lymph_node(row: dict[str, object], real_method: str, table: pd.DataFrame) -> None:
    record = _lookup(table, "method", real_method) if real_method else pd.DataFrame()
    row["lymph_node_evidence"] = "available" if not record.empty else "missing"
    row["lymph_node_mean_positive_rank"] = _value(record, "mean_positive_rank")
    row["lymph_node_mean_negative_rank"] = _value(record, "mean_negative_rank")
    row["lymph_node_negative_top3_count"] = _value(record, "negative_top3_count")
    row["lymph_node_mean_positive_downstream_spearman"] = _value(record, "mean_positive_downstream_spearman")
    row["lymph_node_mean_positive_region_enrichment"] = _value(record, "mean_positive_region_enrichment")
    row["lymph_node_mean_negative_region_enrichment"] = _value(record, "mean_negative_region_enrichment")


def _copy_evidence_flags(row: dict[str, object]) -> None:
    available = [
        row.get("controlled_evidence") == "available",
        row.get("full_slide_evidence") == "available",
        row.get("operating_point_status") == "passing_operating_point",
        row.get("group_fdr_status") == "available",
        row.get("group_fdr_seed_stability_status") == "seed_stability_completed",
        row.get("group_fdr_replicate_stability_status") == "replicate_stability_completed",
        row.get("lymph_node_evidence") == "available",
    ]
    row["n_available_evidence_blocks"] = int(sum(available))
    if row["n_available_evidence_blocks"] >= 4:
        row["evidence_status"] = "selection_ready"
    elif row["n_available_evidence_blocks"] >= 2:
        row["evidence_status"] = "partial"
    else:
        row["evidence_status"] = "insufficient"


def _score_candidates(evidence: pd.DataFrame) -> pd.DataFrame:
    result = evidence.copy()
    scores = []
    reasons = []
    for _, row in result.iterrows():
        score = 0
        reason: list[str] = []
        if _num(row.get("controlled_weak_recall")) > 0.155:
            score += 1
            reason.append("controlled weak recall exceeds 0.155")
        if _num(row.get("full_slide_weak_recall")) >= 0.039:
            score += 1
            reason.append("full-slide weak recall reaches Phase 1 boundary")
        if _num(row.get("operating_weak_recall")) > 0.04 and _num(row.get("operating_fdr_delta_vs_95")) <= 0.02:
            score += 2
            reason.append("calibrated operating point improves full-slide weak recall")
        if _num(row.get("full_slide_fp_high_expr_rate")) <= 0.005 or _num(row.get("operating_fp_high_expr_rate")) <= 0.005:
            score += 1
            reason.append("high-expression FP rate remains low")
        if (
            _num(row.get("group_fdr_positive_bh_fdr")) <= 0.05
            and _num(row.get("group_fdr_negative_sig_groups")) == 0
        ):
            score += 2
            reason.append("direct group FDR finds positives without high-expression negatives")
        if (
            row.get("group_fdr_seed_stability_status") == "seed_stability_completed"
            and _num(row.get("group_fdr_seed_stability_passing_fraction")) == 1.0
            and _num(row.get("group_fdr_seed_stability_max_negative_sig_groups")) == 0
        ):
            score += 1
            reason.append("selected direct group FDR setting is stable across random seeds")
        if (
            row.get("group_fdr_replicate_stability_status") == "replicate_stability_completed"
            and _num(row.get("group_fdr_replicate_stability_passing_fraction")) == 1.0
            and _num(row.get("group_fdr_replicate_stability_max_negative_sig_groups")) == 0
        ):
            score += 1
            reason.append("selected direct group FDR setting is stable across controlled replicates")
        if _num(row.get("lymph_node_negative_top3_count")) == 0 and _num(row.get("lymph_node_mean_positive_downstream_spearman")) > 0.7:
            score += 1
            reason.append("lymph-node controls/downstream validation pass")
        if row.get("candidate_id") in {"wavelet_weak_mining", "wavelet_focal_weak_mining"} and _num(row.get("full_slide_fdr")) > 0.55:
            score -= 2
            reason.append("weak-mining variant fails full-slide FDR transfer")
        if row.get("candidate_id") in {"hybrid_wavelet_gat", "no_lr_identity", "no_distance", "no_expression"} and row.get("evidence_status") == "insufficient":
            score -= 1
            reason.append("insufficient Phase 2 evidence")
        scores.append(score)
        reasons.append("; ".join(reason) if reason else "No positive selection evidence.")
    result["selection_score"] = scores
    result["selection_rationale"] = reasons
    result["recommended_default"] = False
    if not result.empty:
        ready = result[result["evidence_status"] == "selection_ready"]
        pool = ready if not ready.empty else result
        best_index = pool.sort_values(["selection_score", "n_available_evidence_blocks"], ascending=[False, False]).index[0]
        result.loc[best_index, "recommended_default"] = True
    return result.sort_values(["recommended_default", "selection_score", "n_available_evidence_blocks"], ascending=[False, False, False])


def _criteria_table(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = [
        ("controlled_weak_recall_gt_0p155", lambda row: _num(row.get("controlled_weak_recall")) > 0.155),
        ("full_slide_weak_recall_ge_0p039", lambda row: _num(row.get("full_slide_weak_recall")) >= 0.039),
        ("operating_point_pass", lambda row: row.get("operating_point_status") == "passing_operating_point"),
        ("group_fdr_pass", lambda row: _num(row.get("group_fdr_positive_bh_fdr")) <= 0.05 and _num(row.get("group_fdr_negative_sig_groups")) == 0),
        (
            "group_fdr_seed_stability_pass",
            lambda row: row.get("group_fdr_seed_stability_status") == "seed_stability_completed"
            and _num(row.get("group_fdr_seed_stability_passing_fraction")) == 1.0
            and _num(row.get("group_fdr_seed_stability_max_negative_sig_groups")) == 0,
        ),
        (
            "group_fdr_replicate_stability_pass",
            lambda row: row.get("group_fdr_replicate_stability_status") == "replicate_stability_completed"
            and _num(row.get("group_fdr_replicate_stability_passing_fraction")) == 1.0
            and _num(row.get("group_fdr_replicate_stability_max_negative_sig_groups")) == 0,
        ),
        ("lymph_node_validation_pass", lambda row: _num(row.get("lymph_node_negative_top3_count")) == 0 and _num(row.get("lymph_node_mean_positive_downstream_spearman")) > 0.7),
    ]
    for _, row in evidence.iterrows():
        for criterion, fn in checks:
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "criterion": criterion,
                    "status": "pass" if fn(row) else "fail_or_missing",
                }
            )
    return pd.DataFrame(rows)


def _recommendation(evidence: pd.DataFrame) -> dict[str, object]:
    recommended = evidence[evidence["recommended_default"]]
    if recommended.empty:
        return {"status": "no_default_selected"}
    row = recommended.iloc[0].to_dict()
    return {
        "status": "default_selected",
        "phase2_default_model": row["candidate_id"],
        "label": row["label"],
        "selection_score": int(row["selection_score"]),
        "evidence_status": row["evidence_status"],
        "rationale": row["selection_rationale"],
        "caveat": (
            "This is a Phase 2 default candidate, not a final manuscript claim. "
            "It should be rerun across additional semi-synthetic/real datasets in Goals 4-5."
        ),
    }


def _markdown_summary(evidence: pd.DataFrame, criteria: pd.DataFrame, recommendation: dict[str, object]) -> str:
    fields = [
        "candidate_id",
        "selection_score",
        "evidence_status",
        "controlled_weak_recall",
        "full_slide_weak_recall",
        "operating_weak_recall",
        "lr_balanced_weak_recall",
        "group_fdr_positive_bh_fdr",
        "group_fdr_seed_stability_passing_fraction",
        "group_fdr_replicate_stability_passing_fraction",
        "lymph_node_mean_positive_downstream_spearman",
        "recommended_default",
    ]
    lines = [
        "# Phase 2 Goal 3 Model Selection",
        "",
        f"Recommendation: `{recommendation.get('phase2_default_model', 'none')}`",
        "",
        _markdown_table(evidence[fields]),
        "",
        "## Criteria",
        "",
        _markdown_table(criteria.pivot(index="candidate_id", columns="criterion", values="status").reset_index()),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(table: pd.DataFrame) -> str:
    fields = list(table.columns)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_cell(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
        return "NA"
    return str(value)


def _lookup(table: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if table.empty or not value or column not in table.columns:
        return pd.DataFrame()
    return table[table[column].astype(str) == value].head(1)


def _value(record: pd.DataFrame, column: str, default: object = float("nan")) -> object:
    if record.empty or column not in record.columns:
        return default
    value = record.iloc[0][column]
    if pd.isna(value):
        return default
    return value


def _num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2/goal3_model_selection")
    args = parser.parse_args()
    evidence, criteria, recommendation = run_model_selection(PROJECT_ROOT / args.output_dir)
    print(evidence.to_string(index=False))
    print("\nRecommendation:")
    print(json.dumps(recommendation, indent=2, allow_nan=True))
    print(f"Wrote Goal 3 model-selection outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
