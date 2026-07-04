#!/usr/bin/env python
"""Build a tiered functional/FDR closure audit for Phase 7 Goal 5.

This table intentionally separates strict all-null success from weaker
supporting tiers. LR-identity and competitive LR-pair FDR tiers are useful
manuscript evidence only when ranking and downstream negative-control checks
are clean, and they should not be reported as strict all-null closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL5_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    goal_dir = PROJECT_ROOT / args.goal_dir
    rows = [
        _slide_tags_tonsil(goal_dir),
        _post_xenium_colon(goal_dir),
        _xenium_graphclust(goal_dir),
        _visium_hd_crc(goal_dir),
        _spamosaic_tonsil(goal_dir, "spamosaic_tonsil_A1"),
        _spamosaic_tonsil(goal_dir, "spamosaic_tonsil_D1"),
        _melanoma_targeted_real_background(goal_dir),
        _native_human_melanoma_multiome(goal_dir),
    ]
    audit = pd.DataFrame(rows)
    bool_columns = [
        "strict_all_null_success",
        "model_rerun_all_null_success",
        "local_contrast_all_null_success",
        "functional_lr_identity_fdr_success",
        "functional_competitive_lr_fdr_success",
        "targeted_real_background_strict_support",
    ]
    for column in bool_columns:
        if column in audit:
            audit[column] = audit[column].map(_bool)
    audit_path = goal_dir / "phase7_goal5_functional_fdr_closure_audit.csv"
    audit.to_csv(audit_path, index=False)

    summary = {
        "audit_path": _rel(audit_path),
        "n_datasets": int(len(audit)),
        "n_strict_all_null_success": int(audit["strict_all_null_success"].fillna(False).astype(bool).sum()),
        "n_model_rerun_all_null_success": int(
            audit["model_rerun_all_null_success"].fillna(False).astype(bool).sum()
        ),
        "n_local_contrast_all_null_success": int(
            audit.get("local_contrast_all_null_success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "n_functional_lr_identity_success": int(
            audit["functional_lr_identity_fdr_success"].fillna(False).astype(bool).sum()
        ),
        "n_functional_competitive_success": int(
            audit["functional_competitive_lr_fdr_success"].fillna(False).astype(bool).sum()
        ),
        "n_targeted_real_background_strict_support": int(
            audit["targeted_real_background_strict_support"].fillna(False).astype(bool).sum()
        ),
        "best_functional_lr_identity_evidence": _best_evidence(audit, "functional_lr_identity_fdr_success"),
        "best_functional_competitive_evidence": _best_evidence(audit, "functional_competitive_lr_fdr_success"),
        "best_local_contrast_evidence": _best_evidence(audit, "local_contrast_all_null_success"),
        "best_targeted_real_background_evidence": _best_evidence(
            audit, "targeted_real_background_strict_support"
        ),
        "claim_boundary": (
            "strict_all_null_success requires clean ranking, clean downstream, positive FDR recovery, and no negative "
            "leakage under spatial/receiver/LR-identity or equivalent all-null criteria. "
            "model_rerun_all_null_success is stricter evidence when available because SpatialSelf is retrained on "
            "perturbed inputs; failed model-rerun nulls are retained as diagnostic evidence, not reclassified as closure. "
            "functional_lr_identity_fdr_success is a weaker tier: it requires clean ranking, clean downstream, and "
            "LR-identity FDR positive recovery without high-expression negative-control leakage. "
            "functional_competitive_lr_fdr_success is another weaker tier using a non-degenerate candidate-pool "
            "competitive LR null after documenting that LR-level spatial/receiver nulls are degenerate. Neither "
            "weaker tier should be reported as strict all-null FDR closure. "
            "local_contrast_all_null_success is a new diagnostic closure tier: it requires clean ranking/downstream "
            "and all-null positive recovery with no high-expression negative leakage after subtracting matched local "
            "negative-control background. It addresses score saturation in real-platform targeted tests, but should "
            "be described as local-contrast FDR evidence rather than raw-score all-null closure. "
            "targeted_real_background_strict_support records pre-specified real-background targeted screens that pass "
            "ranking, downstream, context, raw LR-identity FDR, and targeted spatial/receiver/LR-identity FDR; this is "
            "strong support for a context-specific case, but it is not native real-sample ground truth and does not by "
            "itself close Goal 5."
        ),
    }
    summary_path = goal_dir / "phase7_goal5_functional_fdr_closure_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2))


def _slide_tags_tonsil(goal_dir: Path) -> dict[str, Any]:
    functional = _match(_csv(goal_dir / "phase7_goal5_functional_success_audit.csv"), "dataset", "slide_tags_tonsil_subset")
    control = _csv(
        goal_dir
        / "slide_tags_tonsil_targeted_region_fdr_200perm/"
        "slide_tags_tonsil_targeted_control_fdr_summary.csv"
    )
    if control.empty:
        control = _csv(
            goal_dir
            / "slide_tags_tonsil_targeted_region_fdr_100perm/"
            "slide_tags_tonsil_targeted_control_fdr_summary.csv"
        )
    identity = _match(control, "null_model", "lr_identity_permutation")
    spatial = _match(control, "null_model", "spatial_permutation")
    receiver = _match(control, "null_model", "receiver_permutation")
    competitive = _first_control(
        goal_dir
        / "slide_tags_tonsil_competitive_lr_fdr_1000perm/"
        "slide_tags_tonsil_competitive_control_fdr_summary.csv"
    )
    model_rerun = _model_rerun_summary(
        goal_dir
        / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm/"
        "slide_tags_tonsil_n1200_model_rerun_control_fdr_summary.csv",
        goal_dir
        / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm/"
        "slide_tags_tonsil_n1200_model_rerun_fdr_audit.csv",
    )
    context_rerun = _context_restricted_model_rerun_summary(
        goal_dir
        / "slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw/"
        "slide_tags_tonsil_n1200_context_restricted_control_fdr_summary.csv",
        goal_dir
        / "slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw/"
        "slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_audit.csv",
    )
    matched_context = _matched_context_enrichment_summary(
        goal_dir
        / "slide_tags_tonsil_n1200_matched_context_enrichment_fdr/"
        "slide_tags_tonsil_n1200_matched_context_control_fdr_summary.csv",
        goal_dir
        / "slide_tags_tonsil_n1200_matched_context_enrichment_fdr/"
        "slide_tags_tonsil_n1200_matched_context_fdr_audit.csv",
    )
    context_rerun.update(matched_context)
    return _row(
        dataset="slide_tags_tonsil_subset",
        platform="Slide-tags / high-resolution tonsil",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive=competitive,
        model_rerun=model_rerun,
        all_null_strict=False,
        evidence_note=(
            "Clean ranking and downstream negative-control suppression. LR-identity targeted FDR recovers positives "
            "without high-expression negative controls; non-degenerate competitive LR FDR also recovers named "
            "positive controls without high-expression negatives. LR-level spatial/receiver nulls are degenerate or conservative. "
            "A 20-per-null model-rerun diagnostic on the n=1200 Slide-tags subset completed without failures, but "
            "did not pass strict all-null closure because spatial-coordinate reruns also called high-expression "
            "negative controls while LR-identity/receiver nulls were clean but just above FDR 0.05. "
            f"Context-restricted model-rerun diagnostic={context_rerun.get('context_context_restricted_model_rerun_status', 'not_run')} "
            f"({context_rerun.get('context_context_restricted_control_status', '')}). "
            f"Matched-context enrichment diagnostic={context_rerun.get('matched_context_fdr_status', 'not_run')} "
            f"({context_rerun.get('matched_context_control_status', '')})."
        ),
        extra=context_rerun,
    )


def _post_xenium_colon(goal_dir: Path) -> dict[str, Any]:
    functional = _match(
        _csv(goal_dir / "phase7_goal5_functional_success_audit.csv"),
        "dataset",
        "post_xenium_human_colon_cancer_subset",
    )
    control = _csv(
        goal_dir
        / "post_xenium_colon_region_aware_targeted_fdr_200perm/"
        "post_xenium_colon_region_aware_targeted_control_fdr_summary.csv"
    )
    if control.empty:
        control = _csv(
            goal_dir
            / "post_xenium_colon_targeted_region_fdr_100perm/"
            "post_xenium_colon_targeted_control_fdr_summary.csv"
        )
    identity = _match(control, "null_model", "lr_identity_permutation")
    spatial = _match_any(control, "null_model", ["spatial_permutation", "within_region_permutation"])
    receiver = _match_any(control, "null_model", ["receiver_permutation", "cell_type_preserving_permutation"])
    competitive = _first_control(
        goal_dir
        / "post_xenium_colon_competitive_lr_fdr_1000perm/"
        "post_xenium_colon_competitive_control_fdr_summary.csv"
    )
    model_rerun = _model_rerun_summary(
        goal_dir
        / "post_xenium_colon_model_rerun_fdr_50perm/"
        "post_xenium_colon_model_rerun_control_fdr_summary.csv",
        goal_dir
        / "post_xenium_colon_model_rerun_fdr_50perm/"
        "post_xenium_colon_model_rerun_fdr_audit.csv",
    )
    context_rerun = _context_restricted_model_rerun_summary(
        goal_dir
        / "post_xenium_colon_context_restricted_model_rerun_fdr_raw/"
        "post_xenium_colon_context_restricted_control_fdr_summary.csv",
        goal_dir
        / "post_xenium_colon_context_restricted_model_rerun_fdr_raw/"
        "post_xenium_colon_context_restricted_model_rerun_fdr_audit.csv",
    )
    context_fp_suppressed = _context_restricted_model_rerun_summary(
        goal_dir
        / "post_xenium_colon_context_restricted_model_rerun_fdr_fp_suppressed_diagnostic/"
        "post_xenium_colon_context_restricted_control_fdr_summary.csv",
        goal_dir
        / "post_xenium_colon_context_restricted_model_rerun_fdr_fp_suppressed_diagnostic/"
        "post_xenium_colon_context_restricted_model_rerun_fdr_audit.csv",
        prefix="fp_suppressed_context_",
    )
    context_rerun.update(context_fp_suppressed)
    matched_context = _matched_context_enrichment_summary(
        goal_dir
        / "post_xenium_colon_matched_context_enrichment_fdr/"
        "post_xenium_colon_matched_context_control_fdr_summary.csv",
        goal_dir
        / "post_xenium_colon_matched_context_enrichment_fdr/"
        "post_xenium_colon_matched_context_fdr_audit.csv",
    )
    context_rerun.update(matched_context)
    local_contrast = _local_contrast_summary(
        goal_dir
        / "post_xenium_colon_local_contrast_fdr_200perm/"
        "post_xenium_colon_local_contrast_control_summary.csv",
        goal_dir
        / "post_xenium_colon_local_contrast_fdr_200perm/"
        "post_xenium_colon_local_contrast_audit.csv",
    )
    context_rerun.update(local_contrast)
    return _row(
        dataset="post_xenium_human_colon_cancer_subset",
        platform="10x Visium CytAssist post-Xenium colon cancer",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive=competitive,
        model_rerun=model_rerun,
        local_contrast=local_contrast,
        all_null_strict=False,
        evidence_note=(
            "Clean calibrated ranking and downstream negative-control suppression. LR-identity targeted FDR recovers "
            "positive controls with negative suppression; non-degenerate competitive LR FDR recovers positive controls "
            "with negative suppression. A 50-per-null model-rerun FDR completed without run failures but did not pass "
            "strict all-null closure because spatial/receiver reruns also called high-expression negative controls. "
            f"Context-restricted model-rerun diagnostic={context_rerun.get('context_context_restricted_model_rerun_status', 'not_run')} "
            f"({context_rerun.get('context_context_restricted_control_status', '')}); fp-suppressed diagnostic="
            f"{context_rerun.get('fp_suppressed_context_context_restricted_model_rerun_status', 'not_run')} "
            f"({context_rerun.get('fp_suppressed_context_context_restricted_control_status', '')}). "
            f"Matched-context enrichment diagnostic={context_rerun.get('matched_context_fdr_status', 'not_run')} "
            f"({context_rerun.get('matched_context_control_status', '')}). "
            f"Local-contrast targeted FDR={local_contrast.get('local_contrast_all_null_status', 'not_run')} "
            f"({local_contrast.get('local_contrast_control_status', '')})."
        ),
        extra=context_rerun,
    )


def _xenium_graphclust(goal_dir: Path) -> dict[str, Any]:
    functional = _match(
        _csv(goal_dir / "phase7_goal5_functional_success_audit.csv"),
        "dataset",
        "xenium_human_lymph_node_subset",
    )
    graph = _match(
        _csv(goal_dir / "xenium_graphclust_validation_200perm/xenium_graphclust_validation_audit.csv"),
        "method",
        "spatialself_wavelet_edge_prior",
    )
    graph_strict = str(graph.get("graphclust_validation_status", "")) == "strict_graphclust_success"
    downstream_clean = _bool(functional.get("downstream_positive_pass")) and not _bool(
        functional.get("downstream_negative_leakage")
    )
    return {
        "dataset": "xenium_human_lymph_node_subset",
        "platform": "10x Xenium human lymph node",
        "ranking_clean": _bool(functional.get("positive_ranking_or_context_pass")) and not _bool(functional.get("negative_ranking_leakage")),
        "downstream_clean": downstream_clean,
        "lr_identity_positive_recovery": graph_strict,
        "lr_identity_negative_suppressed": graph_strict,
        "spatial_or_structure_positive_recovery": graph_strict,
        "spatial_or_structure_negative_suppressed": graph_strict,
        "competitive_lr_positive_recovery": False,
        "competitive_lr_negative_suppressed": False,
        "model_rerun_all_null_success": False,
        "model_rerun_completed_reruns": 0,
        "model_rerun_failed_reruns": 0,
        "model_rerun_status": "",
        "model_rerun_nulls_passing": "",
        "strict_all_null_success": bool(graph_strict and downstream_clean),
        "functional_lr_identity_fdr_success": bool(graph_strict and downstream_clean),
        "functional_competitive_lr_fdr_success": False,
        "best_positive": functional.get("best_positive", ""),
        "best_positive_rank": functional.get("best_positive_rank", ""),
        "best_negative": functional.get("best_negative", ""),
        "best_negative_rank": functional.get("best_negative_rank", ""),
        "best_positive_lr_identity_fdr": graph.get("best_graphclust_positive_fdr", ""),
        "best_negative_lr_identity_fdr": graph.get("best_graphclust_negative_fdr", ""),
        "best_positive_model_rerun_fdr": "",
        "best_negative_model_rerun_fdr": "",
        "evidence_note": (
            "Graphclust-aware FDR is strong, but receiver graphclust-residual downstream has negative-control leakage; "
            "therefore Xenium remains mixed for functional closure."
        ),
        "claim_tier": "mixed_graphclust_fdr_downstream_leakage",
    }


def _visium_hd_crc(goal_dir: Path) -> dict[str, Any]:
    functional = _match(_csv(goal_dir / "phase7_goal5_functional_success_audit.csv"), "dataset", "visium_hd_crc_p2crc_subset")
    control = _csv(
        goal_dir
        / "visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/"
        "visium_hd_crc_p2crc_region_aware_targeted_control_fdr_summary.csv"
    )
    identity = _match(control, "null_model", "lr_identity_permutation")
    spatial = _match(control, "null_model", "within_region_permutation")
    receiver = _match(control, "null_model", "cell_type_preserving_permutation")
    competitive = {}
    return _row(
        dataset="visium_hd_crc_p2crc_subset",
        platform="10x Visium HD human colon cancer P2CRC",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive=competitive,
        model_rerun={},
        all_null_strict=False,
        evidence_note=(
            "Calibrated ranking improves, but downstream and targeted FDR have high-expression negative-control leakage."
        ),
    )


def _spamosaic_tonsil(goal_dir: Path, section_id: str) -> dict[str, Any]:
    dataset = f"{section_id}_subset"
    functional = _match(_csv(goal_dir / "phase7_goal5_functional_success_audit.csv"), "dataset", dataset)
    control = _csv(
        goal_dir
        / f"{section_id}_targeted_region_fdr_200perm/"
        f"{section_id}_targeted_control_fdr_summary.csv"
    )
    identity = _match(control, "null_model", "lr_identity_permutation")
    spatial = _match(control, "null_model", "spatial_permutation")
    receiver = _match(control, "null_model", "receiver_permutation")
    competitive = {}
    local_contrast = _local_contrast_summary(
        goal_dir
        / f"{section_id}_local_contrast_fdr_200perm/"
        f"{section_id}_local_contrast_control_summary.csv",
        goal_dir
        / f"{section_id}_local_contrast_fdr_200perm/"
        f"{section_id}_local_contrast_audit.csv",
    )
    return _row(
        dataset=dataset,
        platform="SPAMOSAIC tonsil RNA+ADT",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive=competitive,
        model_rerun={},
        local_contrast=local_contrast,
        all_null_strict=False,
        evidence_note=(
            "Ranking is clean and LR-identity FDR is supportive, but raw-score downstream/targeted cell-type FDR "
            "can have negative-control leakage. Local-contrast targeted FDR tests whether positive LR scores exceed "
            f"matched high-expression negative-control background: {local_contrast.get('local_contrast_all_null_status', 'not_run')} "
            f"({local_contrast.get('local_contrast_control_status', '')})."
        ),
    )


def _melanoma_targeted_real_background(goal_dir: Path) -> dict[str, Any]:
    phase7_dir = goal_dir.parent
    melanoma_dir = phase7_dir / "goal4_expanded_melanoma_subset_screen"
    audit = _csv(melanoma_dir / "phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv")
    manifest = _json(melanoma_dir / "phase7_goal4_expanded_melanoma_manifest.json")
    strict_hits = audit[audit.get("strict_goal4_success", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    if strict_hits.empty:
        functional = {}
        identity = {}
        spatial = {}
        receiver = {}
        targeted_support = False
        best_positive = ""
        best_rank = ""
        best_negative = ""
        source_boundary = "real-background screen"
    else:
        best = strict_hits.sort_values("raw_spatialself_rank").iloc[0].to_dict()
        functional = {
            "positive_ranking_or_context_pass": True,
            "negative_ranking_leakage": False,
            "downstream_positive_pass": True,
            "downstream_negative_leakage": False,
            "best_positive": best.get("canonical_pair", ""),
            "best_positive_rank": best.get("raw_spatialself_rank", ""),
            "best_negative": "pre-specified high-expression/control panel",
            "best_negative_rank": "",
        }
        identity = {
            "n_positive_significant": int((strict_hits["targeted_lr_identity_bh_fdr"].astype(float) <= 0.05).sum()),
            "n_negative_significant": 0,
            "best_positive_group_bh_fdr": best.get("targeted_lr_identity_bh_fdr", ""),
            "best_negative_group_bh_fdr": 1.0,
        }
        spatial = {
            "n_positive_significant": int((strict_hits["targeted_spatial_bh_fdr"].astype(float) <= 0.05).sum()),
            "n_negative_significant": 0,
            "best_positive_group_bh_fdr": best.get("targeted_spatial_bh_fdr", ""),
            "best_negative_group_bh_fdr": 1.0,
        }
        receiver = {
            "n_positive_significant": int((strict_hits["targeted_receiver_bh_fdr"].astype(float) <= 0.05).sum()),
            "n_negative_significant": 0,
            "best_positive_group_bh_fdr": best.get("targeted_receiver_bh_fdr", ""),
            "best_negative_group_bh_fdr": 1.0,
        }
        targeted_support = True
        best_positive = best.get("canonical_pair", "")
        best_rank = best.get("raw_spatialself_rank", "")
        best_negative = "pre-specified controls suppressed"
        source_boundary = (
            str(manifest.get("limitations", ["", "real-background screen"])[1]).removeprefix("This is a ")
            if manifest
            else "real-background screen"
        )
    model_rerun = _model_rerun_summary(
        GOAL5_DIR
        / "melanoma_real_background_model_rerun_fdr_20perm/"
        "post_xenium_colon_model_rerun_control_fdr_summary.csv",
        GOAL5_DIR
        / "melanoma_real_background_model_rerun_fdr_20perm/"
        "post_xenium_colon_model_rerun_fdr_audit.csv",
    )
    matched_context = _matched_context_enrichment_summary(
        GOAL5_DIR
        / "melanoma_real_background_matched_context_enrichment_fdr/"
        "melanoma_real_background_matched_context_control_fdr_summary.csv",
        GOAL5_DIR
        / "melanoma_real_background_matched_context_enrichment_fdr/"
        "melanoma_real_background_matched_context_fdr_audit.csv",
    )
    hardneg_smoke = _diagnostic_model_rerun_summary(
        GOAL5_DIR
        / "melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/"
        "post_xenium_colon_model_rerun_control_fdr_summary.csv",
        GOAL5_DIR
        / "melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/"
        "post_xenium_colon_model_rerun_fdr_audit.csv",
        prefix="hardneg_feasgate_",
    )
    matched_context.update(hardneg_smoke)

    row = _row(
        dataset="melanoma_real_background_targeted_context_screen",
        platform="Human melanoma real-background targeted context screen",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive={},
        model_rerun=model_rerun,
        all_null_strict=False,
        evidence_note=(
            "Goal 4 melanoma real-background screen has targeted strict successes for "
            f"{', '.join(strict_hits['canonical_pair'].astype(str).tolist()) if not strict_hits.empty else 'no LR pairs'}: "
            "SpatialSelf ranks them in the top 10 while COMMOT/CellChat/CellNEST are outside the top 10, downstream "
            "target correlation is significant, context localizes to Macrophage->T, and targeted spatial/receiver/"
            "LR-identity FDR all pass. This is strong targeted real-background support, but the source is a "
            f"{source_boundary.rstrip('.')}. A 20-per-null model-rerun diagnostic was also run and did not pass strict retraining "
            f"null closure: status={model_rerun.get('model_rerun_all_null_status', 'not_run')}; "
            f"null statuses={model_rerun.get('model_rerun_control_status', '')}. "
            f"A Macrophage->T CXCR3-axis matched-context rerun diagnostic also remained non-strict: "
            f"status={matched_context.get('matched_context_fdr_status', 'not_run')}; "
            f"null statuses={matched_context.get('matched_context_control_status', '')}. "
            f"A hard-negative/feasibility-gate model-rerun smoke diagnostic also remained non-strict: "
            f"status={matched_context.get('hardneg_feasgate_model_rerun_status', 'not_run')}; "
            f"null statuses={matched_context.get('hardneg_feasgate_control_status', '')}."
        ),
        extra=matched_context,
    )
    row["targeted_real_background_strict_support"] = targeted_support
    row["best_positive"] = best_positive
    row["best_positive_rank"] = best_rank
    row["best_negative"] = best_negative
    row["claim_tier"] = (
        "targeted_real_background_strict_support_not_goal5_all_null_closure"
        if targeted_support
        else "mixed_or_negative_control_risk"
    )
    return row


def _native_human_melanoma_multiome(goal_dir: Path) -> dict[str, Any]:
    native_dir = goal_dir / "native_human_melanoma_multiome"
    claim = _match(
        _csv(native_dir / "native_melanoma_claim_audit.csv"),
        "method",
        "spatialself_wavelet_edge_prior",
    )
    ranking = _match(
        _csv(native_dir / "native_melanoma_control_ranking.csv"),
        "method",
        "spatialself_wavelet_edge_prior",
    )
    fdr = _csv(native_dir / "native_melanoma_targeted_fdr_audit.csv")
    fdr = fdr[fdr.get("method", pd.Series(dtype=str)).astype(str).eq("spatialself_wavelet_edge_prior")]
    positive = fdr[~fdr.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    negative = fdr[fdr.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    identity_pos = int((pd.to_numeric(positive.get("targeted_lr_identity_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    identity_neg = int((pd.to_numeric(negative.get("targeted_lr_identity_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    spatial_pos = int((pd.to_numeric(positive.get("targeted_spatial_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    spatial_neg = int((pd.to_numeric(negative.get("targeted_spatial_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    receiver_pos = int((pd.to_numeric(positive.get("targeted_receiver_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    receiver_neg = int((pd.to_numeric(negative.get("targeted_receiver_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.05).sum())
    functional = {
        "positive_ranking_or_context_pass": _bool(claim.get("positive_ranking_pass")),
        "negative_ranking_leakage": _safe_float(ranking.get("best_negative_rank", 999)) <= 3,
        "downstream_positive_pass": _bool(claim.get("competitive_downstream_pass")),
        "downstream_negative_leakage": _safe_int(claim.get("n_negative_downstream_fdr_0_1", 999)) > 0,
        "best_positive": ranking.get("best_positive", claim.get("best_positive", "")),
        "best_positive_rank": ranking.get("best_positive_rank", claim.get("best_positive_rank", "")),
        "best_negative": ranking.get("best_negative", claim.get("best_negative", "")),
        "best_negative_rank": ranking.get("best_negative_rank", claim.get("best_negative_rank", "")),
    }
    identity = {
        "n_positive_significant": identity_pos,
        "n_negative_significant": identity_neg,
        "best_positive_group_bh_fdr": _min_numeric(positive, "targeted_lr_identity_bh_fdr"),
        "best_negative_group_bh_fdr": _min_numeric(negative, "targeted_lr_identity_bh_fdr"),
    }
    spatial = {
        "n_positive_significant": spatial_pos,
        "n_negative_significant": spatial_neg,
        "best_positive_group_bh_fdr": _min_numeric(positive, "targeted_spatial_bh_fdr"),
        "best_negative_group_bh_fdr": _min_numeric(negative, "targeted_spatial_bh_fdr"),
    }
    receiver = {
        "n_positive_significant": receiver_pos,
        "n_negative_significant": receiver_neg,
        "best_positive_group_bh_fdr": _min_numeric(positive, "targeted_receiver_bh_fdr"),
        "best_negative_group_bh_fdr": _min_numeric(negative, "targeted_receiver_bh_fdr"),
    }
    row = _row(
        dataset="native_human_melanoma_multiome_subset",
        platform="Slide-tags / 10x Multiome human melanoma lymph-node metastasis",
        functional=functional,
        identity=identity,
        spatial=spatial,
        receiver=receiver,
        competitive={},
        model_rerun={},
        all_null_strict=False,
        evidence_note=(
            "Native real-sample melanoma workflow completed on 2,535 real cells and 60,000 candidate edges. "
            "SpatialSelf ranks PDCD1-CD274 first, but ACTB-EGFR is also ranked second and competitive downstream "
            "validation has high-expression negative-control leakage. Therefore this is a useful cross-platform "
            "diagnostic run, not Goal 5 closure."
        ),
    )
    row["claim_tier"] = "native_real_sample_diagnostic_negative_control_leakage"
    return row


def _row(
    *,
    dataset: str,
    platform: str,
    functional: dict[str, Any],
    identity: dict[str, Any],
    spatial: dict[str, Any],
    receiver: dict[str, Any],
    competitive: dict[str, Any],
    model_rerun: dict[str, Any],
    all_null_strict: bool,
    evidence_note: str,
    local_contrast: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranking_clean = _bool(functional.get("positive_ranking_or_context_pass")) and not _bool(
        functional.get("negative_ranking_leakage")
    )
    downstream_clean = _bool(functional.get("downstream_positive_pass")) and not _bool(
        functional.get("downstream_negative_leakage")
    )
    identity_pos = _safe_int(identity.get("n_positive_significant", 0)) > 0
    identity_neg_suppressed = _safe_int(identity.get("n_negative_significant", 999)) == 0
    spatial_pos = _safe_int(spatial.get("n_positive_significant", 0)) > 0
    spatial_neg_suppressed = _safe_int(spatial.get("n_negative_significant", 999)) == 0
    receiver_pos = _safe_int(receiver.get("n_positive_significant", 0)) > 0
    receiver_neg_suppressed = _safe_int(receiver.get("n_negative_significant", 999)) == 0
    competitive_pos = _safe_int(competitive.get("n_positive_significant", 0)) > 0
    competitive_neg_suppressed = _safe_int(competitive.get("n_negative_significant", 999)) == 0
    model_rerun_success = str(model_rerun.get("model_rerun_all_null_status", "")) == (
        "positive_significant_negative_suppressed_all_nulls"
    )
    model_rerun_completed = _safe_int(model_rerun.get("n_completed_reruns", 0))
    model_rerun_failed = _safe_int(model_rerun.get("n_failed_reruns", 0))
    local_contrast = local_contrast or {}
    local_contrast_success = str(local_contrast.get("local_contrast_all_null_status", "")) == (
        "positive_significant_negative_suppressed_all_nulls"
    )
    strict = bool(
        ranking_clean
        and downstream_clean
        and (all_null_strict or model_rerun_success)
        and identity_pos
        and identity_neg_suppressed
        and spatial_pos
        and spatial_neg_suppressed
        and receiver_pos
        and receiver_neg_suppressed
    )
    lr_identity_success = bool(ranking_clean and downstream_clean and identity_pos and identity_neg_suppressed)
    competitive_success = bool(ranking_clean and downstream_clean and competitive_pos and competitive_neg_suppressed)
    local_contrast_success = bool(ranking_clean and downstream_clean and local_contrast_success)
    row = {
        "dataset": dataset,
        "platform": platform,
        "ranking_clean": ranking_clean,
        "downstream_clean": downstream_clean,
        "lr_identity_positive_recovery": identity_pos,
        "lr_identity_negative_suppressed": identity_neg_suppressed,
        "spatial_or_structure_positive_recovery": spatial_pos,
        "spatial_or_structure_negative_suppressed": spatial_neg_suppressed,
        "receiver_or_celltype_positive_recovery": receiver_pos,
        "receiver_or_celltype_negative_suppressed": receiver_neg_suppressed,
        "competitive_lr_positive_recovery": competitive_pos,
        "competitive_lr_negative_suppressed": competitive_neg_suppressed,
        "model_rerun_all_null_success": bool(model_rerun_success),
        "model_rerun_completed_reruns": model_rerun_completed,
        "model_rerun_failed_reruns": model_rerun_failed,
        "model_rerun_status": model_rerun.get("model_rerun_all_null_status", ""),
        "model_rerun_nulls_passing": model_rerun.get("null_models_passing", ""),
        "local_contrast_all_null_success": local_contrast_success,
        "local_contrast_status": local_contrast.get("local_contrast_all_null_status", ""),
        "local_contrast_nulls_passing": local_contrast.get("null_models_passing", ""),
        "local_contrast_best_positive_fdr": local_contrast.get("best_positive_local_contrast_fdr", ""),
        "local_contrast_best_negative_fdr": local_contrast.get("best_negative_local_contrast_fdr", ""),
        "strict_all_null_success": strict,
        "functional_lr_identity_fdr_success": lr_identity_success,
        "functional_competitive_lr_fdr_success": competitive_success,
        "targeted_real_background_strict_support": False,
        "best_positive": functional.get("best_positive", ""),
        "best_positive_rank": functional.get("best_positive_rank", ""),
        "best_negative": functional.get("best_negative", ""),
        "best_negative_rank": functional.get("best_negative_rank", ""),
        "best_positive_lr_identity_fdr": identity.get("best_positive_group_bh_fdr", ""),
        "best_negative_lr_identity_fdr": identity.get("best_negative_group_bh_fdr", ""),
        "best_positive_competitive_fdr": competitive.get("best_positive_group_bh_fdr", ""),
        "best_negative_competitive_fdr": competitive.get("best_negative_group_bh_fdr", ""),
        "best_positive_model_rerun_fdr": model_rerun.get("best_positive_model_rerun_fdr", ""),
        "best_negative_model_rerun_fdr": model_rerun.get("best_negative_model_rerun_fdr", ""),
        "evidence_note": evidence_note,
        "claim_tier": _claim_tier(
            strict,
            lr_identity_success,
            competitive_success,
            local_contrast_success,
            ranking_clean,
            downstream_clean,
        ),
    }
    if extra:
        row.update(extra)
    return row


def _claim_tier(
    strict: bool,
    lr_identity: bool,
    competitive: bool,
    local_contrast: bool,
    ranking_clean: bool,
    downstream_clean: bool,
) -> str:
    if strict:
        return "strict_all_null_functional_fdr_success"
    if local_contrast:
        return "local_contrast_all_null_functional_fdr_success_not_raw_score_closure"
    if lr_identity and competitive:
        return "functional_lr_identity_and_competitive_fdr_support_not_strict_all_null"
    if lr_identity:
        return "functional_lr_identity_fdr_support_not_strict_all_null"
    if competitive:
        return "functional_competitive_fdr_support_not_strict_all_null"
    if ranking_clean and downstream_clean:
        return "ranking_downstream_clean_but_fdr_no_recovery"
    return "mixed_or_negative_control_risk"


def _best_evidence(audit: pd.DataFrame, column: str) -> str:
    subset = audit[audit[column].fillna(False).astype(bool)]
    if subset.empty:
        return ""
    row = subset.iloc[0]
    return f"{row['dataset']}: {row['claim_tier']} ({row['evidence_note']})"


def _local_contrast_summary(control_path: Path, audit_path: Path) -> dict[str, Any]:
    control = _csv(control_path)
    audit = _csv(audit_path)
    if control.empty and audit.empty:
        return {
            "local_contrast_all_null_status": "",
            "null_models_passing": "",
            "best_positive_local_contrast_fdr": "",
            "best_negative_local_contrast_fdr": "",
            "local_contrast_control_status": "",
        }
    status = audit.iloc[0].to_dict() if not audit.empty else {}
    return {
        "local_contrast_all_null_status": status.get("local_contrast_all_null_status", ""),
        "null_models_passing": status.get("null_models_passing", ""),
        "best_positive_local_contrast_fdr": _min_numeric(control, "best_positive_local_contrast_fdr"),
        "best_negative_local_contrast_fdr": _min_numeric(control, "best_negative_local_contrast_fdr"),
        "local_contrast_control_status": ";".join(control.get("control_status", pd.Series(dtype=str)).astype(str).tolist()),
    }


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _match(frame: pd.DataFrame, column: str, value: Any) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    subset = frame[frame[column].astype(str).eq(str(value))]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _match_any(frame: pd.DataFrame, column: str, values: list[str]) -> dict[str, Any]:
    for value in values:
        row = _match(frame, column, value)
        if row:
            return row
    return {}


def _first_control(path: Path) -> dict[str, Any]:
    frame = _csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _model_rerun_summary(control_path: Path, audit_path: Path) -> dict[str, Any]:
    audit = _csv(audit_path)
    if audit.empty:
        return {}
    row = audit.iloc[0].to_dict()
    control = _csv(control_path)
    if not control.empty:
        row["best_positive_model_rerun_fdr"] = _min_numeric(control, "best_positive_group_bh_fdr")
        row["best_negative_model_rerun_fdr"] = _min_numeric(control, "best_negative_group_bh_fdr")
        if "control_status" in control:
            if "null_model" in control:
                row["model_rerun_control_status"] = ";".join(
                    f"{item.null_model}:{item.control_status}" for item in control.itertuples(index=False)
                )
            else:
                row["model_rerun_control_status"] = ";".join(control["control_status"].astype(str).tolist())
    return row


def _context_restricted_model_rerun_summary(
    control_path: Path,
    audit_path: Path,
    *,
    prefix: str = "context_",
) -> dict[str, Any]:
    audit = _csv(audit_path)
    control = _csv(control_path)
    if audit.empty:
        return {}
    audit_row = audit.iloc[0].to_dict()
    statuses = (
        ";".join(f"{row.null_model}:{row.control_status}" for row in control.itertuples(index=False))
        if not control.empty and {"null_model", "control_status"}.issubset(control.columns)
        else ""
    )
    positive_min = _min_numeric(control, "best_positive_group_bh_fdr")
    negative_min = _min_numeric(control, "best_negative_group_bh_fdr")
    return {
        f"{prefix}context_restricted_model_rerun_status": audit_row.get(
            "context_restricted_model_rerun_status", ""
        ),
        f"{prefix}context_restricted_nulls_passing": audit_row.get("null_models_passing", ""),
        f"{prefix}context_restricted_completed_reruns": audit_row.get("n_completed_null_predictions", ""),
        f"{prefix}context_restricted_failed_reruns": audit_row.get("n_failed_null_predictions", ""),
        f"{prefix}context_restricted_best_positive_fdr": positive_min,
        f"{prefix}context_restricted_best_negative_fdr": negative_min,
        f"{prefix}context_restricted_control_status": statuses,
    }


def _matched_context_enrichment_summary(control_path: Path, audit_path: Path) -> dict[str, Any]:
    audit = _csv(audit_path)
    control = _csv(control_path)
    if audit.empty:
        return {}
    audit_row = audit.iloc[0].to_dict()
    statuses = (
        ";".join(f"{row.null_model}:{row.control_status}" for row in control.itertuples(index=False))
        if not control.empty and {"null_model", "control_status"}.issubset(control.columns)
        else ""
    )
    return {
        "matched_context_fdr_status": audit_row.get("matched_context_fdr_status", ""),
        "matched_context_nulls_passing": audit_row.get("null_models_passing", ""),
        "matched_context_completed_reruns": audit_row.get("n_completed_null_predictions", ""),
        "matched_context_failed_reruns": audit_row.get("n_failed_null_predictions", ""),
        "matched_context_best_positive_fdr": _min_numeric(control, "best_positive_group_bh_fdr"),
        "matched_context_best_negative_fdr": _min_numeric(control, "best_negative_group_bh_fdr"),
        "matched_context_control_status": statuses,
    }


def _diagnostic_model_rerun_summary(control_path: Path, audit_path: Path, *, prefix: str) -> dict[str, Any]:
    audit = _csv(audit_path)
    control = _csv(control_path)
    if audit.empty:
        return {}
    audit_row = audit.iloc[0].to_dict()
    statuses = (
        ";".join(f"{row.null_model}:{row.control_status}" for row in control.itertuples(index=False))
        if not control.empty and {"null_model", "control_status"}.issubset(control.columns)
        else ""
    )
    return {
        f"{prefix}model_rerun_status": audit_row.get("model_rerun_all_null_status", ""),
        f"{prefix}nulls_passing": audit_row.get("null_models_passing", ""),
        f"{prefix}completed_reruns": audit_row.get("n_completed_reruns", ""),
        f"{prefix}failed_reruns": audit_row.get("n_failed_reruns", ""),
        f"{prefix}best_positive_fdr": _min_numeric(control, "best_positive_group_bh_fdr"),
        f"{prefix}best_negative_fdr": _min_numeric(control, "best_negative_group_bh_fdr"),
        f"{prefix}control_status": statuses,
    }


def _min_numeric(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return float(values.min())


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
