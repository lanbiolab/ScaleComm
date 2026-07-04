#!/usr/bin/env python
"""Audit Goal 5 cross-platform functional/FDR success criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
ALPHA = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL5_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    goal_dir = PROJECT_ROOT / args.goal_dir
    rows = [
        _slide_tags_tonsil(goal_dir),
        _xenium(goal_dir),
        _post_xenium_colon(goal_dir),
        _visium_hd_crc(goal_dir),
        _spamosaic_tonsil(goal_dir),
        _spamosaic_tonsil_section(goal_dir, "spamosaic_tonsil_D1"),
        _lung(goal_dir),
        _cosmx(goal_dir),
    ]
    audit = pd.DataFrame(rows)
    audit_path = goal_dir / "phase7_goal5_functional_success_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = {
        "audit_path": _rel(audit_path),
        "n_datasets": int(len(audit)),
        "status_counts": audit["goal5_functional_status"].value_counts().to_dict(),
        "n_strict_success": int((audit["goal5_functional_status"] == "strict_success").sum()),
        "best_current_evidence": _best_current_evidence(audit),
        "claim_boundary": (
            "Strict Goal 5 success requires positive/control ranking or context support, downstream target-response "
            "support, and FDR evidence without high-expression negative-control leakage. Conservative no-recovery "
            "FDR is not counted as positive recovery."
        ),
    }
    summary_path = goal_dir / "phase7_goal5_functional_success_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2))


def _slide_tags_tonsil(goal_dir: Path) -> dict[str, Any]:
    control = _csv(PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/tonsil_fresh_control_summary.csv")
    downstream = _csv(PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_downstream_target_correlation.csv")
    targeted = _csv(goal_dir / "slide_tags_tonsil_targeted_region_fdr_100perm/slide_tags_tonsil_targeted_region_fdr_audit.csv")
    targeted_control = _csv(
        goal_dir / "slide_tags_tonsil_targeted_region_fdr_100perm/slide_tags_tonsil_targeted_control_fdr_summary.csv"
    )
    positive = control[control.get("control_class", pd.Series(dtype=str)).astype(str).eq("tonsil_positive_prior")]
    negative = control[
        control.get("control_class", pd.Series(dtype=str)).astype(str).eq("housekeeping_negative_control")
    ]
    ds_positive = downstream[downstream.get("control_class", pd.Series(dtype=str)).astype(str).eq("tonsil_positive_prior")]
    ds_negative = downstream[
        downstream.get("control_class", pd.Series(dtype=str)).astype(str).eq("housekeeping_negative_control")
    ]
    best_positive = _best_ranked(positive, "rank_by_score_mean")
    best_negative = _best_ranked(negative, "rank_by_score_mean")
    best_ds_positive = _best_downstream_positive(ds_positive)
    positive_ranking = _safe_float(best_positive.get("rank_by_score_mean", 999)) <= 3
    negative_leakage = _safe_float(best_negative.get("rank_by_score_mean", 999)) <= 3
    downstream_positive = (
        _safe_float(best_ds_positive.get("spearman_r", 0)) > 0
        and _safe_float(best_ds_positive.get("bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = bool(
        (pd.to_numeric(ds_negative.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).any()
    )
    targeted_strict = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str)
        == "targeted_positive_significant_negative_suppressed").any()
    )
    fdr_negative_leakage = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=targeted_strict,
        fdr_negative_leakage=fdr_negative_leakage,
        data_blocked=False,
    )
    return {
        "dataset": "slide_tags_tonsil_subset",
        "platform": "Slide-tags / high-resolution tonsil",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(targeted_strict),
        "prediction_fdr_negative_leakage": bool(fdr_negative_leakage),
        "targeted_region_fdr_strict_success": bool(targeted_strict),
        "targeted_region_lr_identity_status": _targeted_control_status(targeted_control, "lr_identity_permutation"),
        "best_positive": best_positive.get("control_name", ""),
        "best_positive_rank": best_positive.get("rank_by_score_mean", ""),
        "best_negative": best_negative.get("control_name", ""),
        "best_negative_rank": best_negative.get("rank_by_score_mean", ""),
        "best_positive_downstream": best_ds_positive.get("canonical_pair", ""),
        "best_positive_downstream_r": best_ds_positive.get("spearman_r", ""),
        "best_positive_downstream_fdr": best_ds_positive.get("bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": int(
            (pd.to_numeric(ds_negative.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()
        ),
        "best_positive_prediction_fdr": _targeted_best_positive_fdr(targeted_control),
        "best_negative_prediction_fdr": _targeted_best_negative_fdr(targeted_control),
        "evidence_summary": (
            "High-resolution tonsil has clean control ranking and CCL21-CCR7 downstream support without housekeeping "
            "negative downstream leakage. Targeted FDR suppresses negatives and recovers positives under LR-identity "
            "only; spatial/receiver nulls remain conservative."
        ),
        "next_action": "Use as supportive high-resolution evidence; strict Goal 5 still needs spatial/receiver FDR recovery or another platform.",
    }


def _xenium(goal_dir: Path) -> dict[str, Any]:
    control = _csv(goal_dir / "xenium_human_lymph_node_method_control_summary.csv")
    downstream = _csv(goal_dir / "xenium_downstream_target_validation/xenium_downstream_method_summary.csv")
    fdr = _csv(goal_dir / "xenium_prediction_fdr_50perm/xenium_prediction_control_fdr_summary.csv")
    graphclust_audit = _csv(goal_dir / "xenium_graphclust_validation_200perm/xenium_graphclust_validation_audit.csv")
    graphclust_downstream = _csv(
        goal_dir
        / "xenium_graphclust_residual_downstream_200perm/"
        "xenium_graphclust_residual_downstream_method_summary.csv"
    )
    graphclust_claims = _csv(
        goal_dir
        / "xenium_graphclust_residual_downstream_200perm/"
        "xenium_graphclust_residual_downstream_claims.csv"
    )
    spatialself = _match(control, "method", "spatialself_wavelet_edge_prior")
    ds = _match(downstream, "method", "spatialself_wavelet_edge_prior")
    fdr_rows = fdr[fdr.get("method", pd.Series(dtype=str)).astype(str).eq("spatialself_wavelet_edge_prior")]
    graph_ss = _match(graphclust_audit, "method", "spatialself_wavelet_edge_prior")
    graph_ds = _match(graphclust_downstream, "method", "spatialself_wavelet_edge_prior")
    graph_claim = _match(graphclust_claims, "method", "spatialself_wavelet_edge_prior")
    positive_ranking = _safe_float(spatialself.get("best_positive_rank", 999)) <= 3
    negative_leakage = _safe_float(spatialself.get("best_negative_rank", 999)) <= 3
    downstream_positive = (
        _safe_float(ds.get("best_positive_spearman", 0)) > 0
        and _safe_float(ds.get("best_positive_bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = int(_safe_float(ds.get("n_negative_fdr_0_1", 999))) > 0
    fdr_positive_recovery = bool((fdr_rows.get("n_positive_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    fdr_negative_leakage = bool((fdr_rows.get("n_negative_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    graphclust_strict = graph_ss.get("graphclust_validation_status", "") == "strict_graphclust_success"
    graphclust_downstream_negative = int(_safe_float(graph_ds.get("n_negative_fdr_0_1", 999))) > 0
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=fdr_positive_recovery,
        fdr_negative_leakage=fdr_negative_leakage,
        data_blocked=False,
    )
    return {
        "dataset": "xenium_human_lymph_node_subset",
        "platform": "10x Xenium human lymph node",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(fdr_positive_recovery),
        "prediction_fdr_negative_leakage": bool(fdr_negative_leakage),
        "graphclust_fdr_strict_success": bool(graphclust_strict),
        "graphclust_residual_downstream_negative_leakage": bool(graphclust_downstream_negative),
        "best_positive": spatialself.get("best_positive_control", ""),
        "best_positive_rank": spatialself.get("best_positive_rank", ""),
        "best_negative": spatialself.get("best_negative_control", ""),
        "best_negative_rank": spatialself.get("best_negative_rank", ""),
        "best_positive_downstream_r": ds.get("best_positive_spearman", ""),
        "best_positive_downstream_fdr": ds.get("best_positive_bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
        "best_graphclust_positive": graph_ss.get("best_graphclust_positive", ""),
        "best_graphclust_positive_rank": graph_ss.get("best_graphclust_positive_rank", ""),
        "best_graphclust_negative": graph_ss.get("best_graphclust_negative", ""),
        "best_graphclust_negative_rank": graph_ss.get("best_graphclust_negative_rank", ""),
        "best_graphclust_positive_fdr": graph_ss.get("best_graphclust_positive_fdr", ""),
        "best_graphclust_negative_fdr": graph_ss.get("best_graphclust_negative_fdr", ""),
        "graphclust_residual_downstream_claim": graph_claim.get("claim_status", ""),
        "best_positive_prediction_fdr": _min_or_blank(fdr_rows, "best_positive_group_bh_fdr"),
        "best_negative_prediction_fdr": _min_or_blank(fdr_rows, "best_negative_group_bh_fdr"),
        "evidence_summary": (
            "CCL19-CCR7 raw target response is strong, but high-expression negative controls also rank/correlate; "
            "prediction-level FDR is conservative with no positive recovery. Supplemental 10x graphclust-aware "
            f"validation is stronger: SpatialSelf status={graph_ss.get('graphclust_validation_status', '')}, "
            f"best graphclust positive={graph_ss.get('best_graphclust_positive', '')} rank="
            f"{graph_ss.get('best_graphclust_positive_rank', '')}, best negative="
            f"{graph_ss.get('best_graphclust_negative', '')} rank={graph_ss.get('best_graphclust_negative_rank', '')}, "
            f"positive/negative graphclust FDR={graph_ss.get('best_graphclust_positive_fdr', '')}/"
            f"{graph_ss.get('best_graphclust_negative_fdr', '')}. However receiver graphclust-residual downstream "
            f"remains {graph_claim.get('claim_status', '')}, so Xenium is not strict functional closure."
        ),
        "next_action": "Use Xenium graphclust-aware FDR as supportive evidence; still need clean downstream functional support without negative-control leakage or a richer annotated platform.",
    }


def _post_xenium_colon(goal_dir: Path) -> dict[str, Any]:
    control = _csv(goal_dir / "post_xenium_human_colon_cancer_method_control_summary.csv")
    downstream = _csv(
        goal_dir / "post_xenium_colon_downstream_target_validation_50perm/post_xenium_colon_downstream_method_summary.csv"
    )
    claims = _csv(goal_dir / "post_xenium_colon_downstream_target_validation_50perm/post_xenium_colon_downstream_claims.csv")
    fdr = _csv(goal_dir / "post_xenium_colon_prediction_fdr_50perm/post_xenium_colon_prediction_control_fdr_summary.csv")
    targeted = _csv(
        goal_dir / "post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_region_fdr_audit.csv"
    )
    targeted_control = _csv(
        goal_dir / "post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_control_fdr_summary.csv"
    )
    region_aware_control = _csv(
        goal_dir
        / "post_xenium_colon_region_aware_targeted_fdr_200perm/"
        "post_xenium_colon_region_aware_targeted_control_fdr_summary.csv"
    )
    calibrated = _match(control, "method", "spatialself_calibrated")
    ds = _match(downstream, "method", "spatialself_calibrated")
    claim = _match(claims, "method", "spatialself_calibrated")
    fdr_rows = fdr[fdr.get("method", pd.Series(dtype=str)).astype(str).eq("spatialself_calibrated")]
    positive_ranking = int(_safe_float(calibrated.get("positive_top3", 0))) > 0 and int(
        _safe_float(calibrated.get("negative_top3", 999))
    ) == 0
    negative_leakage = int(_safe_float(calibrated.get("negative_top3", 999))) > 0 or int(
        _safe_float(calibrated.get("negative_top100_edges", 999))
    ) > 0
    downstream_positive = (
        _safe_float(ds.get("best_positive_spearman", 0)) > 0
        and _safe_float(ds.get("best_positive_bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = int(_safe_float(ds.get("n_negative_fdr_0_1", 999))) > 0
    fdr_positive_recovery = bool((fdr_rows.get("n_positive_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    fdr_negative_leakage = bool((fdr_rows.get("n_negative_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    targeted_strict = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str)
        == "targeted_positive_significant_negative_suppressed").any()
    )
    targeted_lr_identity = _targeted_control_status(targeted_control, "lr_identity_permutation")
    region_aware_within = _targeted_control_status(region_aware_control, "within_region_permutation")
    region_aware_cell_type = _targeted_control_status(region_aware_control, "cell_type_preserving_permutation")
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=fdr_positive_recovery,
        fdr_negative_leakage=fdr_negative_leakage,
        data_blocked=False,
    )
    return {
        "dataset": "post_xenium_human_colon_cancer_subset",
        "platform": "10x Visium CytAssist post-Xenium colon cancer",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(fdr_positive_recovery),
        "prediction_fdr_negative_leakage": bool(fdr_negative_leakage),
        "targeted_region_fdr_strict_success": bool(targeted_strict),
        "targeted_region_lr_identity_status": targeted_lr_identity,
        "best_positive": calibrated.get("best_positive_control", ""),
        "best_positive_rank": calibrated.get("best_positive_rank", ""),
        "best_negative": calibrated.get("best_negative_control", ""),
        "best_negative_rank": calibrated.get("best_negative_rank", ""),
        "best_positive_downstream_r": ds.get("best_positive_spearman", ""),
        "best_positive_downstream_fdr": ds.get("best_positive_bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
        "best_positive_prediction_fdr": _min_or_blank(fdr_rows, "best_positive_group_bh_fdr"),
        "best_negative_prediction_fdr": _min_or_blank(fdr_rows, "best_negative_group_bh_fdr"),
        "evidence_summary": (
            f"Calibrated SpatialSelf has claim={claim.get('claim_status', '')}, positive_top3="
            f"{calibrated.get('positive_top3', '')}, negative_top3={calibrated.get('negative_top3', '')}, "
            "and no downstream negative FDR signal, but prediction-level FDR has no positive recovery. "
            f"Supplemental targeted region-aware FDR strict_success={targeted_strict}; "
            f"LR-identity targeted control status={targeted_lr_identity}. A stricter within-region/cell-type-preserving "
            f"targeted FDR audit remains conservative ({region_aware_within}/{region_aware_cell_type}), so this is still "
            "not strict Goal 5 closure."
        ),
        "next_action": "Run a stronger targeted/region-aware FDR or find a dataset where calibrated positives recover under FDR.",
    }


def _visium_hd_crc(goal_dir: Path) -> dict[str, Any]:
    control = _csv(goal_dir / "visium_hd_crc_p2crc_method_control_summary.csv")
    downstream = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_downstream_target_validation/"
        "visium_hd_crc_p2crc_downstream_method_summary.csv"
    )
    claims = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_downstream_target_validation/"
        "visium_hd_crc_p2crc_downstream_claims.csv"
    )
    competitive_claims = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_competitive_downstream_200perm/"
        "visium_hd_crc_competitive_downstream_claims.csv"
    )
    competitive_spatialself = _match(competitive_claims, "method", "spatialself_calibrated")
    fdr = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_prediction_fdr_50perm/"
        "visium_hd_crc_p2crc_prediction_control_fdr_summary.csv"
    )
    targeted = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/"
        "visium_hd_crc_p2crc_region_aware_targeted_fdr_audit.csv"
    )
    targeted_control = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/"
        "visium_hd_crc_p2crc_region_aware_targeted_control_fdr_summary.csv"
    )
    calibrated = _match(control, "method", "spatialself_calibrated")
    default = _match(control, "method", "spatialself_wavelet_edge_prior")
    ds = _match(downstream, "method", "spatialself_calibrated")
    claim = _match(claims, "method", "spatialself_calibrated")
    fdr_rows = fdr[fdr.get("method", pd.Series(dtype=str)).astype(str).eq("spatialself_calibrated")]
    positive_ranking = int(_safe_float(calibrated.get("positive_top3", 0))) > 0 and int(
        _safe_float(calibrated.get("negative_top3", 999))
    ) == 0
    negative_leakage = int(_safe_float(calibrated.get("negative_top3", 999))) > 0 or int(
        _safe_float(calibrated.get("negative_top100_edges", 999))
    ) > 0
    downstream_positive = (
        _safe_float(ds.get("best_positive_spearman", 0)) > 0
        and _safe_float(ds.get("best_positive_bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = int(_safe_float(ds.get("n_negative_fdr_0_1", 999))) > 0
    prediction_fdr_positive = bool((fdr_rows.get("n_positive_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    prediction_fdr_negative = bool((fdr_rows.get("n_negative_significant", pd.Series(dtype=float)).fillna(0) > 0).any())
    targeted_negative_leakage = bool(
        (targeted.get("region_aware_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    targeted_positive = bool(
        targeted.get("region_aware_fdr_status", pd.Series(dtype=str))
        .astype(str)
        .isin(["region_aware_positive_significant_negative_suppressed", "region_aware_cell_type_positive_significant"])
        .any()
    )
    targeted_strict = bool(
        (targeted.get("region_aware_fdr_status", pd.Series(dtype=str)).astype(str)
        == "region_aware_positive_significant_negative_suppressed").any()
    ) and not targeted_negative_leakage
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=targeted_strict,
        fdr_negative_leakage=targeted_negative_leakage or prediction_fdr_negative,
        data_blocked=False,
    )
    return {
        "dataset": "visium_hd_crc_p2crc_subset",
        "platform": "10x Visium HD human colon cancer P2CRC 8um bins",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(prediction_fdr_positive or targeted_positive),
        "prediction_fdr_negative_leakage": bool(prediction_fdr_negative or targeted_negative_leakage),
        "targeted_region_fdr_strict_success": bool(targeted_strict),
        "targeted_region_lr_identity_status": _targeted_control_status(targeted_control, "lr_identity_permutation"),
        "best_positive": calibrated.get("best_positive_control", ""),
        "best_positive_rank": calibrated.get("best_positive_rank", ""),
        "best_negative": calibrated.get("best_negative_control", ""),
        "best_negative_rank": calibrated.get("best_negative_rank", ""),
        "best_default_negative": default.get("best_negative_control", ""),
        "best_default_negative_rank": default.get("best_negative_rank", ""),
        "best_positive_downstream": ds.get("best_positive_lr", ""),
        "best_positive_downstream_r": ds.get("best_positive_spearman", ""),
        "best_positive_downstream_fdr": ds.get("best_positive_bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
        "best_positive_prediction_fdr": _targeted_best_positive_fdr(targeted_control),
        "best_negative_prediction_fdr": _targeted_best_negative_fdr(targeted_control),
        "evidence_summary": (
            "Visium HD CRC P2CRC adds a real 8um high-resolution platform with GEO deconvolution labels. "
            "Calibrated SpatialSelf improves ranking versus default/baselines: CD274-PDCD1 rank=1, ACTB-KRT8 "
            "negative rank=6, negative_top3=0, whereas default SpatialSelf/COMMOT/CellChat/CellNEST rank ACTB-KRT8 "
            "first. However downstream target proxies still show negative-control correlation "
            f"(claim={claim.get('claim_status', '')}), and region-aware targeted FDR has ACTB-KRT8 negative leakage. "
            f"A competitive downstream diagnostic that tests all LR pairs against the same positive response programs "
            f"also remains {competitive_spatialself.get('claim_status', 'not_run')}. Therefore this is a useful "
            "cross-platform stress test and calibration-support result, not strict Goal 5 closure."
        ),
        "next_action": "Use as mixed Visium HD evidence; Goal 5 still needs a platform/case with clean downstream and no negative-control FDR leakage.",
    }


def _spamosaic_tonsil(goal_dir: Path) -> dict[str, Any]:
    control = _csv(goal_dir / "spamosaic_tonsil_A1_method_control_summary.csv")
    downstream = _csv(
        goal_dir / "spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_method_summary.csv"
    )
    claims = _csv(goal_dir / "spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_claims.csv")
    restricted = _csv(
        goal_dir
        / "spamosaic_tonsil_A1_restricted_downstream_validation/spamosaic_tonsil_restricted_downstream_method_summary.csv"
    )
    restricted_claims = _csv(
        goal_dir
        / "spamosaic_tonsil_A1_restricted_downstream_validation/spamosaic_tonsil_restricted_downstream_claims.csv"
    )
    targeted = _csv(
        goal_dir / "spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_region_fdr_audit.csv"
    )
    targeted_control = _csv(
        goal_dir / "spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_control_fdr_summary.csv"
    )
    spatialself = _match(control, "method", "spatialself_wavelet_edge_prior")
    ds = _match(downstream, "method", "spatialself_wavelet_edge_prior")
    claim = _match(claims, "method", "spatialself_wavelet_edge_prior")
    restricted_ds = _match(restricted, "method", "spatialself_wavelet_edge_prior")
    restricted_claim = _match(restricted_claims, "method", "spatialself_wavelet_edge_prior")
    positive_ranking = int(_safe_float(spatialself.get("positive_top3", 0))) > 0 and int(
        _safe_float(spatialself.get("negative_top3", 999))
    ) == 0
    negative_leakage = int(_safe_float(spatialself.get("negative_top3", 999))) > 0 or int(
        _safe_float(spatialself.get("negative_top100_edges", 999))
    ) > 0
    downstream_positive = (
        _safe_float(ds.get("best_positive_spearman", 0)) > 0
        and _safe_float(ds.get("best_positive_bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = int(_safe_float(ds.get("n_negative_fdr_0_1", 999))) > 0
    targeted_lr_identity = _targeted_control_status(targeted_control, "lr_identity_permutation")
    targeted_lr_identity_recovery = targeted_lr_identity == "positive_significant_negative_suppressed"
    targeted_negative_leakage = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    targeted_strict = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str)
        == "targeted_positive_significant_negative_suppressed").any()
    ) and not targeted_negative_leakage
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=targeted_strict,
        fdr_negative_leakage=targeted_negative_leakage,
        data_blocked=False,
    )
    return {
        "dataset": "spamosaic_tonsil_A1_subset",
        "platform": "SPAMOSAIC tonsil RNA+ADT",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(targeted_lr_identity_recovery or targeted_strict),
        "prediction_fdr_negative_leakage": bool(targeted_negative_leakage),
        "targeted_region_fdr_strict_success": bool(targeted_strict),
        "targeted_region_lr_identity_status": targeted_lr_identity,
        "best_positive": spatialself.get("best_positive_control", ""),
        "best_positive_rank": spatialself.get("best_positive_rank", ""),
        "best_negative": spatialself.get("best_negative_control", ""),
        "best_negative_rank": spatialself.get("best_negative_rank", ""),
        "best_positive_downstream": ds.get("best_positive_lr", ""),
        "best_positive_downstream_r": ds.get("best_positive_spearman", ""),
        "best_positive_downstream_fdr": ds.get("best_positive_bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
        "best_positive_prediction_fdr": _targeted_best_positive_fdr(targeted_control),
        "best_negative_prediction_fdr": _targeted_best_negative_fdr(targeted_control),
        "evidence_summary": (
            "SPAMOSAIC tonsil has strong five-method ranking evidence: SpatialSelf ranks positive controls above "
            "high-expression negatives and IL21-IL21R rank=2, while COMMOT/CellChat/CellNEST show negative-control "
            "top-rank risk. It is not strict functional/FDR success because downstream target proxies also correlate "
            f"for negatives (claim={claim.get('claim_status', '')}); receiver-restricted residual downstream remains "
            f"mixed (claim={restricted_claim.get('claim_status', '')}, best_negative_r="
            f"{restricted_ds.get('best_negative_spearman', '')}, best_negative_fdr="
            f"{restricted_ds.get('best_negative_bh_fdr', '')}); and targeted cell-type FDR has negative-control leakage."
        ),
        "next_action": "Use as cross-platform ranking evidence; do receiver-restricted/residual downstream validation or another platform for strict Goal 5 success.",
    }


def _spamosaic_tonsil_section(goal_dir: Path, section_id: str) -> dict[str, Any]:
    control = _csv(goal_dir / f"{section_id}_method_control_summary.csv")
    downstream = _csv(goal_dir / f"{section_id}_downstream_target_validation/{section_id}_downstream_method_summary.csv")
    claims = _csv(goal_dir / f"{section_id}_downstream_target_validation/{section_id}_downstream_claims.csv")
    restricted = _csv(
        goal_dir / f"{section_id}_restricted_downstream_validation/{section_id}_restricted_downstream_method_summary.csv"
    )
    restricted_claims = _csv(
        goal_dir / f"{section_id}_restricted_downstream_validation/{section_id}_restricted_downstream_claims.csv"
    )
    targeted = _csv(goal_dir / f"{section_id}_targeted_region_fdr_200perm/{section_id}_targeted_region_fdr_audit.csv")
    targeted_control = _csv(
        goal_dir / f"{section_id}_targeted_region_fdr_200perm/{section_id}_targeted_control_fdr_summary.csv"
    )
    spatialself = _match(control, "method", "spatialself_wavelet_edge_prior")
    ds = _match(downstream, "method", "spatialself_wavelet_edge_prior")
    claim = _match(claims, "method", "spatialself_wavelet_edge_prior")
    restricted_ds = _match(restricted, "method", "spatialself_wavelet_edge_prior")
    restricted_claim = _match(restricted_claims, "method", "spatialself_wavelet_edge_prior")
    positive_ranking = int(_safe_float(spatialself.get("positive_top3", 0))) > 0 and int(
        _safe_float(spatialself.get("negative_top3", 999))
    ) == 0
    negative_leakage = int(_safe_float(spatialself.get("negative_top3", 999))) > 0 or int(
        _safe_float(spatialself.get("negative_top100_edges", 999))
    ) > 0
    downstream_positive = (
        _safe_float(ds.get("best_positive_spearman", 0)) > 0
        and _safe_float(ds.get("best_positive_bh_fdr", 1)) <= 0.1
    )
    downstream_negative_leakage = int(_safe_float(ds.get("n_negative_fdr_0_1", 999))) > 0
    targeted_lr_identity = _targeted_control_status(targeted_control, "lr_identity_permutation")
    targeted_lr_identity_recovery = targeted_lr_identity == "positive_significant_negative_suppressed"
    targeted_negative_leakage = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    targeted_strict = bool(
        (targeted.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str)
        == "targeted_positive_significant_negative_suppressed").any()
    ) and not targeted_negative_leakage
    status = _status(
        positive_ranking=positive_ranking,
        negative_leakage=negative_leakage,
        downstream_positive=downstream_positive,
        downstream_negative_leakage=downstream_negative_leakage,
        fdr_positive_recovery=targeted_strict,
        fdr_negative_leakage=targeted_negative_leakage,
        data_blocked=False,
    )
    return {
        "dataset": f"{section_id}_subset",
        "platform": "SPAMOSAIC tonsil RNA+ADT independent section",
        "goal5_functional_status": status,
        "positive_ranking_or_context_pass": bool(positive_ranking),
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": bool(downstream_positive),
        "downstream_negative_leakage": bool(downstream_negative_leakage),
        "prediction_fdr_positive_recovery": bool(targeted_lr_identity_recovery or targeted_strict),
        "prediction_fdr_negative_leakage": bool(targeted_negative_leakage),
        "targeted_region_fdr_strict_success": bool(targeted_strict),
        "targeted_region_lr_identity_status": targeted_lr_identity,
        "best_positive": spatialself.get("best_positive_control", ""),
        "best_positive_rank": spatialself.get("best_positive_rank", ""),
        "best_negative": spatialself.get("best_negative_control", ""),
        "best_negative_rank": spatialself.get("best_negative_rank", ""),
        "best_positive_downstream": ds.get("best_positive_lr", ""),
        "best_positive_downstream_r": ds.get("best_positive_spearman", ""),
        "best_positive_downstream_fdr": ds.get("best_positive_bh_fdr", ""),
        "n_negative_downstream_fdr_0_1": ds.get("n_negative_fdr_0_1", ""),
        "best_positive_prediction_fdr": _targeted_best_positive_fdr(targeted_control),
        "best_negative_prediction_fdr": _targeted_best_negative_fdr(targeted_control),
        "evidence_summary": (
            f"{section_id} independently replicates SPAMOSAIC ranking evidence: SpatialSelf positive controls rank ahead "
            "of high-expression negatives, while COMMOT/CellChat/CellNEST show negative-control top-rank risk. It is "
            f"not strict functional/FDR success because downstream is {claim.get('claim_status', '')}; receiver-restricted "
            f"residual downstream remains {restricted_claim.get('claim_status', 'not_available')} "
            f"(best_negative_r={restricted_ds.get('best_negative_spearman', '')}, best_negative_fdr="
            f"{restricted_ds.get('best_negative_bh_fdr', '')}); and targeted cell-type FDR has negative-control leakage."
        ),
        "next_action": "Treat as section-level replication of ranking/negative-control stress behavior, not strict Goal 5 closure.",
    }


def _lung(goal_dir: Path) -> dict[str, Any]:
    control = _csv(goal_dir / "lung_cancer_visium_method_control_summary.csv")
    spatialself = _match(control, "method", "spatialself_wavelet_edge_prior")
    negative_leakage = _safe_float(spatialself.get("best_negative_rank", 999)) <= 3
    return {
        "dataset": "lung_cancer_visium_subset",
        "platform": "10x Visium lung cancer supplemental",
        "goal5_functional_status": "negative_control_risk_no_downstream_fdr",
        "positive_ranking_or_context_pass": _safe_float(spatialself.get("best_positive_rank", 999)) <= 3,
        "negative_ranking_leakage": bool(negative_leakage),
        "downstream_positive_pass": False,
        "downstream_negative_leakage": False,
        "prediction_fdr_positive_recovery": False,
        "prediction_fdr_negative_leakage": False,
        "targeted_region_fdr_strict_success": False,
        "targeted_region_lr_identity_status": "",
        "best_positive": spatialself.get("best_positive_control", ""),
        "best_positive_rank": spatialself.get("best_positive_rank", ""),
        "best_negative": spatialself.get("best_negative_control", ""),
        "best_negative_rank": spatialself.get("best_negative_rank", ""),
        "best_positive_downstream_r": "",
        "best_positive_downstream_fdr": "",
        "n_negative_downstream_fdr_0_1": "",
        "best_positive_prediction_fdr": "",
        "best_negative_prediction_fdr": "",
        "evidence_summary": "Workflow runs, but high-expression negative controls rank near the top and downstream/FDR are not run.",
        "next_action": "Use as risk evidence only unless downstream and FDR panels are added and negatives are controlled.",
    }


def _cosmx(goal_dir: Path) -> dict[str, Any]:
    readiness = _csv(goal_dir / "cosmx_melanoma__standardization_readiness.csv")
    row = readiness.iloc[0].to_dict() if not readiness.empty else {}
    return {
        "dataset": "cosmx_melanoma",
        "platform": "CosMx RNA-SMI melanoma",
        "goal5_functional_status": "blocked_invalid_or_missing_data",
        "positive_ranking_or_context_pass": False,
        "negative_ranking_leakage": False,
        "downstream_positive_pass": False,
        "downstream_negative_leakage": False,
        "prediction_fdr_positive_recovery": False,
        "prediction_fdr_negative_leakage": False,
        "targeted_region_fdr_strict_success": False,
        "targeted_region_lr_identity_status": "",
        "best_positive": "",
        "best_positive_rank": "",
        "best_negative": "",
        "best_negative_rank": "",
        "best_positive_downstream_r": "",
        "best_positive_downstream_fdr": "",
        "n_negative_downstream_fdr_0_1": "",
        "best_positive_prediction_fdr": "",
        "best_negative_prediction_fdr": "",
        "evidence_summary": (
            f"Readiness status={row.get('readiness_status', 'missing')}; invalid_raw_files="
            f"{row.get('invalid_raw_files', '')}."
        ),
        "next_action": "Redownload valid expression, metadata, and coordinate files or provide standardized CSV exports.",
    }


def _status(
    *,
    positive_ranking: bool,
    negative_leakage: bool,
    downstream_positive: bool,
    downstream_negative_leakage: bool,
    fdr_positive_recovery: bool,
    fdr_negative_leakage: bool,
    data_blocked: bool,
) -> str:
    if data_blocked:
        return "blocked_invalid_or_missing_data"
    if (
        positive_ranking
        and not negative_leakage
        and downstream_positive
        and not downstream_negative_leakage
        and fdr_positive_recovery
        and not fdr_negative_leakage
    ):
        return "strict_success"
    if positive_ranking and not negative_leakage and downstream_positive and not downstream_negative_leakage:
        return "functional_support_negative_suppressed_but_fdr_no_recovery"
    if downstream_positive and downstream_negative_leakage:
        return "mixed_functional_support_negative_controls_also_correlate"
    if negative_leakage:
        return "negative_control_risk"
    return "incomplete_or_conservative"


def _targeted_control_status(frame: pd.DataFrame, null_model: str) -> str:
    if frame.empty or "null_model" not in frame:
        return ""
    subset = frame[frame["null_model"].astype(str).eq(null_model)]
    if subset.empty:
        return ""
    return str(subset.iloc[0].get("control_status", ""))


def _targeted_best_positive_fdr(frame: pd.DataFrame) -> float | str:
    if frame.empty or "best_positive_group_bh_fdr" not in frame:
        return ""
    return _min_or_blank(frame, "best_positive_group_bh_fdr")


def _targeted_best_negative_fdr(frame: pd.DataFrame) -> float | str:
    if frame.empty or "best_negative_group_bh_fdr" not in frame:
        return ""
    return _min_or_blank(frame, "best_negative_group_bh_fdr")


def _best_ranked(frame: pd.DataFrame, rank_column: str) -> dict[str, Any]:
    if frame.empty or rank_column not in frame:
        return {}
    return frame.sort_values(rank_column).iloc[0].to_dict()


def _best_downstream_positive(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "spearman_r" not in frame:
        return {}
    candidates = frame[pd.to_numeric(frame.get("spearman_r", pd.Series(dtype=float)), errors="coerce") > 0]
    if candidates.empty:
        return {}
    return candidates.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0].to_dict()


def _best_current_evidence(audit: pd.DataFrame) -> str:
    if audit.empty:
        return ""
    priority = [
        "strict_success",
        "functional_support_negative_suppressed_but_fdr_no_recovery",
        "mixed_functional_support_negative_controls_also_correlate",
        "negative_control_risk",
        "incomplete_or_conservative",
        "blocked_invalid_or_missing_data",
    ]
    for status in priority:
        subset = audit[audit["goal5_functional_status"].astype(str).eq(status)]
        if not subset.empty:
            row = subset.iloc[0]
            return f"{row['dataset']}: {row['goal5_functional_status']} ({row['evidence_summary']})"
    return ""


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _match(frame: pd.DataFrame, column: str, value: Any) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    subset = frame[frame[column].astype(str).eq(str(value))]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _min_or_blank(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return float(values.min())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
