#!/usr/bin/env python
"""Summarize raw and calibrated melanoma Goal 4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen"
KEY_LR = (
    "melanoma_lr_001_CXCL10_CXCR3",
    "melanoma_lr_002_CXCL11_CXCR3",
    "melanoma_lr_000_CXCR3_CXCL9",
    "melanoma_lr_004_PDCD1_CD274",
    "melanoma_lr_003_SPP1_CD44",
)
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    goal_dir = PROJECT_ROOT / args.goal_dir
    dataset_dir = goal_dir / "constructed_replicate"
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    rank_table = pd.read_csv(goal_dir / "phase7_goal4_expanded_melanoma_method_lr_rank_table.csv")
    validation = pd.read_csv(goal_dir / "phase7_goal4_expanded_melanoma_candidate_validation.csv")
    raw_fdr = pd.read_csv(goal_dir / "phase7_goal4_expanded_melanoma_fdr_status.csv")
    raw_pred = pd.read_csv(
        goal_dir
        / "baseline_panel/spatialself_wavelet_edge_prior/constructed_replicate__spatialself_wavelet_predictions.csv"
    )
    calibrated_pred_path = (
        goal_dir
        / "calibrated_spatialself/"
        "constructed_replicate__spatialself_wavelet_predictions__lr_mechanism_prior_rank_predictions.csv"
    )
    if calibrated_pred_path.exists():
        calibrated_pred = pd.read_csv(calibrated_pred_path)
        calibrated_rank = _rank_predictions(candidates, calibrated_pred)
        calibrated_context = _context_summary(candidates, calibrated_pred)
        calibrated_fdr = _collect_calibrated_fdr(goal_dir / "calibrated_spatialself_fdr_50perm")
    else:
        calibrated_rank = pd.DataFrame()
        calibrated_context = pd.DataFrame()
        calibrated_fdr = pd.DataFrame()

    raw_context = _context_summary(candidates, raw_pred)
    rows = []
    for lr_id in KEY_LR:
        lr_rows = candidates[candidates["lr_id"] == lr_id]
        if lr_rows.empty:
            continue
        meta = lr_rows.iloc[0].to_dict()
        method_ranks = _method_ranks(rank_table, lr_id)
        val = _first(validation, "lr_id", lr_id)
        raw_fdr_row = _first(raw_fdr, "lr_id", lr_id)
        raw_ctx = _first(raw_context, "lr_id", lr_id)
        cal_rank = _first(calibrated_rank, "lr_id", lr_id)
        cal_ctx = _first(calibrated_context, "lr_id", lr_id)
        cal_fdr_rows = calibrated_fdr[calibrated_fdr["lr_id"] == lr_id] if not calibrated_fdr.empty else pd.DataFrame()
        rows.append(
            {
                "lr_id": lr_id,
                "canonical_pair": _canonical_pair(meta["ligand"], meta["receptor"]),
                "ligand": meta["ligand"],
                "receptor": meta["receptor"],
                "mechanism": meta["mechanism"],
                "weak_signal": bool(meta.get("weak_signal", False)),
                "raw_spatialself_rank": method_ranks.get("spatialself_wavelet_edge_prior", ""),
                "commot_rank": method_ranks.get("commot", ""),
                "cellchat_rank": method_ranks.get("cellchat", ""),
                "spatalk_rank": method_ranks.get("spatalk", ""),
                "cellnest_rank": method_ranks.get("cellnest", ""),
                "raw_dominant_context": raw_ctx.get("dominant_context", ""),
                "raw_top10_type_rule_rate": raw_ctx.get("top10_type_rule_rate", ""),
                "raw_top10_niche_rule_rate": raw_ctx.get("top10_niche_rule_rate", ""),
                "raw_top10_both_rule_rate": raw_ctx.get("top10_both_rule_rate", ""),
                "downstream_spearman_r": val.get("downstream_spearman_r", ""),
                "downstream_bh_fdr": val.get("downstream_bh_fdr", ""),
                "raw_lr_identity_bh_fdr": raw_fdr_row.get("lr_identity_permutation_bh_fdr", ""),
                "raw_spatial_bh_fdr": raw_fdr_row.get("spatial_permutation_bh_fdr", ""),
                "raw_receiver_bh_fdr": raw_fdr_row.get("receiver_permutation_bh_fdr", ""),
                "calibrated_spatialself_rank": cal_rank.get("rank_by_score_mean", ""),
                "calibrated_dominant_context": cal_ctx.get("dominant_context", ""),
                "calibrated_top10_type_rule_rate": cal_ctx.get("top10_type_rule_rate", ""),
                "calibrated_top10_niche_rule_rate": cal_ctx.get("top10_niche_rule_rate", ""),
                "calibrated_top10_both_rule_rate": cal_ctx.get("top10_both_rule_rate", ""),
                "calibrated_macrophage_to_t_min_bh_fdr": _targeted_min_fdr(
                    cal_fdr_rows, sender="Macrophage", receiver="T"
                ),
                "calibrated_all_three_nulls_fdr05": _all_three_targeted_fdr05(
                    cal_fdr_rows, sender="Macrophage", receiver="T"
                ),
                "interpretation_status": _interpretation_status(
                    method_ranks,
                    val,
                    raw_fdr_row,
                    cal_ctx,
                    cal_fdr_rows,
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = goal_dir / "phase7_goal4_expanded_melanoma_calibrated_evidence.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "summary_path": _rel(summary_path),
        "calibrated_prediction_path": _rel(calibrated_pred_path) if calibrated_pred_path.exists() else "",
        "calibrated_fdr_dir": _rel(goal_dir / "calibrated_spatialself_fdr_50perm"),
        "n_rows": int(len(summary)),
        "strict_success": bool(summary.get("calibrated_all_three_nulls_fdr05", pd.Series(dtype=bool)).fillna(False).any()),
        "claim_scope": (
            "Calibrated SpatialSelf improves context localization for CXCR3-axis candidates, "
            "but current 50-permutation screening does not establish strict all-three-null FDR."
        ),
    }
    manifest_path = goal_dir / "phase7_goal4_expanded_melanoma_calibrated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def _rank_predictions(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    rows = []
    for lr_id, group in merged.groupby("lr_id", sort=False):
        rows.append(
            {
                "lr_id": lr_id,
                "score_mean": float(group["score"].mean()),
                "score_max": float(group["score"].max()),
            }
        )
    table = pd.DataFrame(rows)
    table["rank_by_score_mean"] = table["score_mean"].rank(method="min", ascending=False).astype(int)
    return table


def _context_summary(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    rows = []
    for lr_id, group in merged.groupby("lr_id", sort=False):
        top = group.nlargest(max(1, int(np.ceil(len(group) * 0.1))), "score")
        context = top["sender_cell_type"].astype(str) + "->" + top["receiver_cell_type"].astype(str)
        rows.append(
            {
                "lr_id": lr_id,
                "dominant_context": context.value_counts().index[0] if len(context) else "",
                "top10_type_rule_rate": float(top["type_rule_satisfied"].mean()) if len(top) else np.nan,
                "top10_niche_rule_rate": float(top["niche_rule_satisfied"].mean()) if len(top) else np.nan,
                "top10_both_rule_rate": float((top["type_rule_satisfied"] & top["niche_rule_satisfied"]).mean())
                if len(top)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _collect_calibrated_fdr(fdr_dir: Path) -> pd.DataFrame:
    rows = []
    prediction_stem = "constructed_replicate__spatialself_wavelet_predictions__lr_mechanism_prior_rank_predictions"
    for null_model in NULL_MODELS:
        path = fdr_dir / f"{prediction_stem}__{null_model}__cell_type_pair_direct_fdr.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["null_model"] = null_model
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _method_ranks(rank_table: pd.DataFrame, lr_id: str) -> dict[str, int]:
    rows = rank_table[rank_table["lr_id"] == lr_id]
    return {
        str(row.method): int(row.rank_by_score_mean)
        for row in rows.itertuples(index=False)
        if not pd.isna(row.rank_by_score_mean)
    }


def _targeted_min_fdr(frame: pd.DataFrame, *, sender: str, receiver: str) -> float | str:
    if frame.empty:
        return ""
    target = frame[
        frame["sender_cell_type"].astype(str).eq(sender)
        & frame["receiver_cell_type"].astype(str).eq(receiver)
    ]
    if target.empty:
        return ""
    return float(pd.to_numeric(target["group_bh_fdr"], errors="coerce").min())


def _all_three_targeted_fdr05(frame: pd.DataFrame, *, sender: str, receiver: str) -> bool:
    if frame.empty:
        return False
    target = frame[
        frame["sender_cell_type"].astype(str).eq(sender)
        & frame["receiver_cell_type"].astype(str).eq(receiver)
    ].copy()
    if target.empty or set(target["null_model"]) < set(NULL_MODELS):
        return False
    by_null = target.groupby("null_model")["group_bh_fdr"].min()
    return bool((pd.to_numeric(by_null, errors="coerce") <= 0.05).all())


def _interpretation_status(
    method_ranks: dict[str, int],
    validation: dict[str, Any],
    raw_fdr: dict[str, Any],
    calibrated_context: dict[str, Any],
    calibrated_fdr: pd.DataFrame,
) -> str:
    spatial_rank = method_ranks.get("spatialself_wavelet_edge_prior", 999)
    baseline_gt10 = all(method_ranks.get(method, 0) > 10 for method in ("commot", "cellchat", "cellnest"))
    downstream = _to_float(validation.get("downstream_bh_fdr", np.nan)) <= 0.1 and _to_float(
        validation.get("downstream_spearman_r", np.nan)
    ) > 0
    raw_lr_identity = _to_float(raw_fdr.get("lr_identity_permutation_bh_fdr", np.nan)) <= 0.05
    context_clean = _to_float(calibrated_context.get("top10_both_rule_rate", np.nan)) >= 0.9
    targeted_fdr = _all_three_targeted_fdr05(calibrated_fdr, sender="Macrophage", receiver="T")
    if spatial_rank <= 10 and baseline_gt10 and downstream and raw_lr_identity and context_clean and targeted_fdr:
        return "strict_success"
    if spatial_rank <= 10 and baseline_gt10 and downstream and raw_lr_identity and context_clean:
        return "strong_mixed_candidate_pending_stricter_fdr"
    if spatial_rank <= 10 and downstream:
        return "rank_downstream_candidate"
    return "not_candidate_or_background"


def _first(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    rows = frame[frame[column] == value]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _to_float(value: Any) -> float:
    try:
        if value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
