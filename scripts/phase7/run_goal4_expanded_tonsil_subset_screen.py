#!/usr/bin/env python
"""Screen the expanded high-resolution tonsil subset for Phase 7 Goal 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset"
CELLNEST_PATCH_DIR = PROJECT_ROOT / "outputs/phase3/end_to_end/high_resolution_tonsil_cellnest_patch"
DATASET_DIR = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/constructed_replicate"
INTERPRETABILITY_DIR = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset"
FDR_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_tonsil_subset_screen/fdr_screening_20perm"
)
FORMAL_FDR_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_tonsil_subset_screen/fdr_formal_100perm"
)

METHOD_FILES = {
    "spatialself_wavelet_calibrated": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}

ACTIVE_BASELINES = ("commot", "cellchat", "cellnest")
STATUS_BASELINES = ("spatalk",)
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_tonsil_subset_screen"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    predictions = pd.read_csv(
        SOURCE_DIR
        / "predictions/spatialself_wavelet_calibrated/constructed_replicate__spatialself_wavelet_calibrated_predictions.csv"
    )

    rank_table = _build_rank_table(lr_pairs)
    method_status = _build_method_status()
    candidate_table = _screen_candidates(rank_table, method_status)
    validation_table = _validate_candidates(candidate_table, candidates, predictions)
    fdr_table = _collect_fdr(candidate_table)
    audit_table = _build_audit(candidate_table, validation_table, fdr_table, method_status, config)

    rank_table.to_csv(output_dir / "phase7_goal4_expanded_tonsil_method_lr_rank_table.csv", index=False)
    method_status.to_csv(output_dir / "phase7_goal4_expanded_tonsil_method_status.csv", index=False)
    candidate_table.to_csv(
        output_dir / "phase7_goal4_expanded_tonsil_spatialself_specific_candidates.csv",
        index=False,
    )
    validation_table.to_csv(output_dir / "phase7_goal4_expanded_tonsil_candidate_validation.csv", index=False)
    fdr_table.to_csv(output_dir / "phase7_goal4_expanded_tonsil_fdr_status.csv", index=False)
    audit_table.to_csv(output_dir / "phase7_goal4_expanded_tonsil_audit.csv", index=False)

    manifest = {
        "config": _rel(config_path),
        "output_dir": _rel(output_dir),
        "source_subset": _rel(SOURCE_DIR),
        "cellnest_patch_subset": _rel(CELLNEST_PATCH_DIR),
        "dataset_dir": _rel(DATASET_DIR),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_edges": int(len(candidates)),
        "strict_success": bool((audit_table["audit_status"] == "strict_success").any()),
        "available_baseline_candidate_found": bool(
            (candidate_table["candidate_status"] == "available_baseline_candidate").any()
        ),
        "screening_fdr_dir": _rel(FDR_DIR) if FDR_DIR.exists() else "",
        "formal_fdr_dir": _rel(FORMAL_FDR_DIR) if FORMAL_FDR_DIR.exists() else "",
        "interpretation": (
            "The expanded 45-LR high-resolution tonsil subset enables rank >10 screening. "
            "CXCL13-CXCR5 emerges as a SpatialSelf-prioritized candidate missed by COMMOT/CellChat, "
            "but full Phase 7 Goal 4 success still requires stronger downstream support and spatial/receiver-null FDR. "
            "CellNEST is included from the Phase 3 high-resolution patch. SpaTalk is status-only because its "
            "scores are constant zero for this subset."
        ),
        "outputs": {
            "rank_table": _rel(output_dir / "phase7_goal4_expanded_tonsil_method_lr_rank_table.csv"),
            "method_status": _rel(output_dir / "phase7_goal4_expanded_tonsil_method_status.csv"),
            "candidates": _rel(output_dir / "phase7_goal4_expanded_tonsil_spatialself_specific_candidates.csv"),
            "validation": _rel(output_dir / "phase7_goal4_expanded_tonsil_candidate_validation.csv"),
            "fdr": _rel(output_dir / "phase7_goal4_expanded_tonsil_fdr_status.csv"),
            "audit": _rel(output_dir / "phase7_goal4_expanded_tonsil_audit.csv"),
        },
    }
    (output_dir / "phase7_goal4_expanded_tonsil_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(audit_table.to_string(index=False))


def _build_rank_table(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    control = lr_pairs.set_index("lr_id").to_dict("index")
    summary_dir = SOURCE_DIR / "control_summaries"
    for method, label in METHOD_FILES.items():
        if method == "cellnest":
            path = CELLNEST_PATCH_DIR / "control_summaries/cellnest__lr_summary.csv"
        else:
            path = summary_dir / f"{method}__lr_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            meta = control.get(row.lr_id, {})
            rows.append(
                {
                    "dataset": "tonsil_highres_45lr_subset",
                    "method": method,
                    "method_label": label,
                    "lr_id": row.lr_id,
                    "canonical_pair": row.canonical_pair,
                    "ligand": row.ligand,
                    "receptor": row.receptor,
                    "mechanism": row.mechanism,
                    "control_class": row.control_class,
                    "active_niche": meta.get("active_niche", ""),
                    "true_lr": bool(meta.get("true_lr", False)),
                    "false_positive_high_expression": bool(
                        meta.get("false_positive_high_expression", False)
                    ),
                    "score_mean": float(row.score_mean),
                    "score_max": float(row.score_max),
                    "top_edge_count": int(row.top_edge_count),
                    "rank_by_score_mean": int(row.rank_by_score_mean),
                    "rank_by_top_edge_count": int(row.rank_by_top_edge_count),
                    "dominant_cell_type_pair": row.dominant_cell_type_pair,
                    "receiver_spatial_concentration": float(row.receiver_spatial_concentration),
                    "summary_path": _rel(path),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "rank_by_score_mean", "lr_id"])


def _build_method_status() -> pd.DataFrame:
    method_summary = pd.read_csv(SOURCE_DIR / "tonsil_baseline_matched_method_summary.csv")
    rows = []
    for method, label in METHOD_FILES.items():
        if method == "cellnest":
            cellnest_summary = CELLNEST_PATCH_DIR / "tonsil_baseline_matched_method_summary.csv"
            row = pd.read_csv(cellnest_summary) if cellnest_summary.exists() else pd.DataFrame()
        else:
            row = method_summary[method_summary["method"] == method]
        if row.empty:
            rows.append(
                {
                    "method": method,
                    "method_label": label,
                    "status_for_goal4_screen": "missing",
                    "n_unique_scores": "",
                    "n_nonzero_scores": "",
                    "reason": "method_summary_row_missing",
                }
            )
            continue
        item = row.iloc[0]
        unique = int(item.get("n_unique_scores", 0))
        nonzero = int(item.get("n_nonzero_scores", 0))
        if method == "spatalk" and unique <= 1 and nonzero == 0:
            status = "status_only_constant_zero_no_lr_resolution"
            reason = "SpaTalk adapter returned constant zero scores for this subset."
        else:
            status = "rank_usable"
            reason = ""
        rows.append(
            {
                "method": method,
                "method_label": label,
                "status_for_goal4_screen": status,
                "n_unique_scores": unique,
                "n_nonzero_scores": nonzero,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _screen_candidates(rank_table: pd.DataFrame, method_status: pd.DataFrame) -> pd.DataFrame:
    pivot = rank_table.pivot_table(
        index=[
            "lr_id",
            "canonical_pair",
            "ligand",
            "receptor",
            "mechanism",
            "control_class",
            "active_niche",
            "true_lr",
            "false_positive_high_expression",
        ],
        columns="method",
        values="rank_by_score_mean",
        aggfunc="first",
    ).reset_index()
    for method in METHOD_FILES:
        if method not in pivot.columns:
            pivot[method] = np.nan
    status = method_status.set_index("method")["status_for_goal4_screen"].to_dict()
    rows = []
    for row in pivot.itertuples(index=False):
        spatial_rank = int(getattr(row, "spatialself_wavelet_calibrated"))
        commot_rank = _maybe_int(getattr(row, "commot"))
        cellchat_rank = _maybe_int(getattr(row, "cellchat"))
        spatalk_rank = _maybe_int(getattr(row, "spatalk"))
        weak_or_context = (
            str(row.control_class).endswith("positive_prior")
            or "niche" in str(row.mechanism)
            or "contact" in str(row.mechanism)
            or str(row.active_niche) not in {"", "any", "nan"}
        )
        cellnest_rank = _maybe_int(getattr(row, "cellnest"))
        available_baseline_pass = (
            spatial_rank <= 10
            and commot_rank != ""
            and int(commot_rank) > 10
            and cellchat_rank != ""
            and int(cellchat_rank) > 10
            and cellnest_rank != ""
            and int(cellnest_rank) > 10
            and not bool(row.false_positive_high_expression)
            and weak_or_context
        )
        full_baseline_status_pass = (
            available_baseline_pass
            and status.get("spatalk") == "status_only_constant_zero_no_lr_resolution"
            and status.get("cellnest") == "rank_usable"
        )
        reasons = []
        if spatial_rank > 10:
            reasons.append("spatialself_rank_gt10")
        if commot_rank == "" or int(commot_rank) <= 10:
            reasons.append("commot_not_gt10")
        if cellchat_rank == "" or int(cellchat_rank) <= 10:
            reasons.append("cellchat_not_gt10")
        if cellnest_rank == "" or int(cellnest_rank) <= 10:
            reasons.append("cellnest_not_gt10")
        if bool(row.false_positive_high_expression):
            reasons.append("high_expression_negative_control")
        if not weak_or_context:
            reasons.append("not_weak_or_context_specific")
        candidate_status = (
            "available_baseline_candidate" if available_baseline_pass else "not_candidate"
        )
        rows.append(
            {
                "dataset": "tonsil_highres_45lr_subset",
                "lr_id": row.lr_id,
                "canonical_pair": row.canonical_pair,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": row.mechanism,
                "control_class": row.control_class,
                "active_niche": row.active_niche,
                "true_lr": bool(row.true_lr),
                "false_positive_high_expression": bool(row.false_positive_high_expression),
                "spatialself_rank": spatial_rank,
                "commot_rank": commot_rank,
                "cellchat_rank": cellchat_rank,
                "spatalk_rank": spatalk_rank,
                "cellnest_rank": cellnest_rank,
                "spatalk_status": status.get("spatalk", ""),
                "cellnest_status": status.get("cellnest", ""),
                "passes_available_baselines_gt10": available_baseline_pass,
                "passes_full_goal4_baselines_gt10_or_status_absent": full_baseline_status_pass,
                "candidate_status": candidate_status,
                "failure_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["spatialself_rank", "lr_id"])


def _validate_candidates(
    candidate_table: pd.DataFrame,
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    scored = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="left")
    downstream = _load_downstream()
    rows = []
    for row in candidate_table.itertuples(index=False):
        edges = scored[scored["lr_id"] == row.lr_id].copy()
        top = edges.nlargest(max(1, int(np.ceil(len(edges) * 0.1))), "score")
        receiver_niche, receiver_niche_fc = _top_enrichment(top, edges, "receiver_niche")
        sender_niche, sender_niche_fc = _top_enrichment(top, edges, "sender_niche")
        context, context_fc = _top_enrichment(top, edges, ["sender_cell_type", "receiver_cell_type"])
        range_rule_rate = float(top["range_rule_satisfied"].mean()) if len(top) else np.nan
        niche_rule_rate = float(top["niche_rule_satisfied"].mean()) if len(top) else np.nan
        ds = downstream.get(row.lr_id, {})
        ds_status = ds.get("status", "not_run_for_this_lr")
        ds_r = ds.get("spearman_r", "")
        ds_fdr = ds.get("bh_fdr", "")
        residual_r = ds.get("residual_spearman_r", "")
        residual_fdr = ds.get("residual_bh_fdr", "")
        has_downstream_support = (
            ds_status == "tested"
            and _to_float(ds_fdr) <= 0.1
            and (_to_float(ds_r) > 0 or _to_float(residual_r) > 0)
        )
        rows.append(
            {
                "dataset": row.dataset,
                "lr_id": row.lr_id,
                "canonical_pair": row.canonical_pair,
                "candidate_status": row.candidate_status,
                "receiver_niche_enrichment": receiver_niche,
                "receiver_niche_fold_change": receiver_niche_fc,
                "sender_niche_enrichment": sender_niche,
                "sender_niche_fold_change": sender_niche_fc,
                "dominant_sender_receiver_context": context,
                "dominant_context_fold_change": context_fc,
                "top_decile_range_rule_satisfied_rate": range_rule_rate,
                "top_decile_niche_rule_satisfied_rate": niche_rule_rate,
                "downstream_status": ds_status,
                "downstream_target_signatures": ds.get("target_signatures", ""),
                "downstream_n_target_genes": ds.get("n_target_genes", ""),
                "downstream_spearman_r": ds_r,
                "downstream_bh_fdr": ds_fdr,
                "downstream_residual_spearman_r": residual_r,
                "downstream_residual_bh_fdr": residual_fdr,
                "has_downstream_support_fdr10": has_downstream_support,
                "validation_status": (
                    "rank_niche_downstream_supported"
                    if row.candidate_status == "available_baseline_candidate"
                    and receiver_niche_fc > 1.0
                    and context_fc > 1.0
                    and has_downstream_support
                    else "rank_niche_only_or_not_candidate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _collect_fdr(candidate_table: pd.DataFrame) -> pd.DataFrame:
    prediction_stem = "constructed_replicate__spatialself_wavelet_calibrated_predictions"
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    formal_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for null_model in NULL_MODELS:
        path = FDR_DIR / f"{prediction_stem}__{null_model}__lr_pair_direct_fdr.csv"
        if path.exists():
            maps[null_model] = pd.read_csv(path).set_index("lr_id").to_dict("index")
        else:
            maps[null_model] = {}
        formal_path = FORMAL_FDR_DIR / f"{prediction_stem}__{null_model}__lr_pair_direct_fdr.csv"
        if formal_path.exists():
            formal_maps[null_model] = pd.read_csv(formal_path).set_index("lr_id").to_dict("index")
        else:
            formal_maps[null_model] = {}
    rows = []
    for row in candidate_table.itertuples(index=False):
        out: dict[str, Any] = {
            "dataset": row.dataset,
            "lr_id": row.lr_id,
            "canonical_pair": row.canonical_pair,
            "candidate_status": row.candidate_status,
            "fdr_depth": "20_permutation_screening" if FDR_DIR.exists() else "not_run",
            "fdr_claim_level": "screening_not_final_manuscript_fdr" if FDR_DIR.exists() else "not_run",
        }
        any_sig = False
        any_formal_sig = False
        all_formal_present = True
        all_present = True
        for null_model in NULL_MODELS:
            item = maps.get(null_model, {}).get(row.lr_id, {})
            if not item:
                all_present = False
            out[f"{null_model}_p_value"] = item.get("group_p_value", "")
            out[f"{null_model}_bh_fdr"] = item.get("group_bh_fdr", "")
            if _to_float(item.get("group_bh_fdr", np.nan)) <= 0.05:
                any_sig = True
            formal_item = formal_maps.get(null_model, {}).get(row.lr_id, {})
            if not formal_item:
                all_formal_present = False
            out[f"formal_100perm_{null_model}_p_value"] = formal_item.get("group_p_value", "")
            out[f"formal_100perm_{null_model}_bh_fdr"] = formal_item.get("group_bh_fdr", "")
            if _to_float(formal_item.get("group_bh_fdr", np.nan)) <= 0.05:
                any_formal_sig = True
        out["all_null_models_present"] = all_present
        out["any_screening_fdr05"] = any_sig
        out["formal_100perm_all_null_models_present"] = all_formal_present
        out["formal_100perm_any_fdr05"] = any_formal_sig
        out["formal_100perm_all_three_nulls_fdr05"] = all(
            _to_float(formal_maps.get(null_model, {}).get(row.lr_id, {}).get("group_bh_fdr", np.nan)) <= 0.05
            for null_model in NULL_MODELS
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _build_audit(
    candidate_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    fdr_table: pd.DataFrame,
    method_status: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    merged = candidate_table.merge(
        validation_table[["lr_id", "validation_status", "has_downstream_support_fdr10"]],
        on="lr_id",
        how="left",
    ).merge(
        fdr_table[
            [
                "lr_id",
                "any_screening_fdr05",
                "all_null_models_present",
                "formal_100perm_any_fdr05",
                "formal_100perm_all_three_nulls_fdr05",
            ]
        ],
        on="lr_id",
        how="left",
    )
    available = merged[merged["candidate_status"] == "available_baseline_candidate"].copy()
    strict = available[
        available["validation_status"].eq("rank_niche_downstream_supported")
        & available["formal_100perm_all_three_nulls_fdr05"].fillna(False).astype(bool)
    ].copy()
    if len(strict):
        status = "strict_success"
        reason = "At least one expanded tonsil candidate has ranking, validation, and screening FDR support."
    elif len(available):
        status = "candidate_pending_functional_or_final_fdr"
        reason = (
            "Expanded tonsil candidate(s) satisfy SpatialSelf-vs-COMMOT/CellChat/CellNEST ranking, "
            "but full Goal 4 still needs stronger downstream support and spatial/receiver-null FDR."
        )
    else:
        status = "no_available_baseline_candidate"
        reason = "No LR passed the SpatialSelf top10 and COMMOT/CellChat rank >10 screen."
    top_candidates = available.sort_values("spatialself_rank")["canonical_pair"].head(5).tolist()
    return pd.DataFrame(
        [
            {
                "goal_id": "goal4_expanded_tonsil_subset_screen",
                "audit_status": status,
                "audit_reason": reason,
                "n_lr_pairs": int(candidate_table["lr_id"].nunique()),
                "n_available_baseline_candidates": int(len(available)),
                "n_strict_screening_candidates": int(len(strict)),
                "top_available_candidates": ";".join(top_candidates),
                "spatalk_status": _status(method_status, "spatalk"),
                "cellnest_status": _status(method_status, "cellnest"),
                "phase7_goal4_completion_gate": config["goals"][
                    "goal4_cross_background_spatialself_specific_discovery"
                ]["strict_screen"],
            }
        ]
    )


def _load_downstream() -> dict[str, dict[str, Any]]:
    path = INTERPRETABILITY_DIR / "tonsil_downstream_target_correlation.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.set_index("lr_id").to_dict("index")


def _top_enrichment(top: pd.DataFrame, all_edges: pd.DataFrame, columns: str | list[str]) -> tuple[str, float]:
    if len(top) == 0 or len(all_edges) == 0:
        return "", np.nan
    if isinstance(columns, str):
        top_key = top[columns].astype(str)
        all_key = all_edges[columns].astype(str)
    else:
        top_key = top[columns].astype(str).agg("->".join, axis=1)
        all_key = all_edges[columns].astype(str).agg("->".join, axis=1)
    best = top_key.value_counts().index[0]
    top_rate = float((top_key == best).mean())
    all_rate = float((all_key == best).mean())
    return str(best), top_rate / max(all_rate, 1e-12)


def _status(method_status: pd.DataFrame, method: str) -> str:
    row = method_status[method_status["method"] == method]
    if row.empty:
        return ""
    return str(row.iloc[0]["status_for_goal4_screen"])


def _maybe_int(value: Any) -> int | str:
    if pd.isna(value):
        return ""
    return int(value)


def _to_float(value: Any) -> float:
    try:
        if value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
