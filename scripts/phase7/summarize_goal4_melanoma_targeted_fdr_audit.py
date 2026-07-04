#!/usr/bin/env python
"""Build a targeted FDR audit for melanoma CXCR3-axis Goal 4 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOAL_DIR = (
    PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen"
)
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")
TARGET_CONTEXT = ("Macrophage", "T")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(DEFAULT_GOAL_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument(
        "--evidence-table",
        default="phase7_goal4_expanded_melanoma_calibrated_evidence.csv",
    )
    parser.add_argument(
        "--fdr-dir",
        default="calibrated_spatialself_fdr_50perm",
        help="FDR output directory under --goal-dir.",
    )
    parser.add_argument(
        "--output-stem",
        default="phase7_goal4_melanoma_targeted_fdr_audit",
        help="Output file stem written under --goal-dir.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    goal_dir = PROJECT_ROOT / args.goal_dir
    evidence = pd.read_csv(goal_dir / args.evidence_table)
    fdr_dir = goal_dir / args.fdr_dir
    targeted = _collect_targeted_fdr(fdr_dir)
    audit = _build_audit(evidence, targeted, alpha=float(args.alpha))

    audit_path = goal_dir / f"{args.output_stem}.csv"
    audit.to_csv(audit_path, index=False)
    manifest = {
        "audit_path": _rel(audit_path),
        "input_evidence_table": _rel(goal_dir / args.evidence_table),
        "calibrated_fdr_dir": _rel(fdr_dir),
        "target_context": {"sender_cell_type": TARGET_CONTEXT[0], "receiver_cell_type": TARGET_CONTEXT[1]},
        "alpha": float(args.alpha),
        "n_candidates": int(len(audit)),
        "n_strict_success": int((audit["strict_goal4_success"] == True).sum()),
        "n_strong_mixed": int((audit["audit_status"] == "strong_mixed_pending_stricter_fdr").sum()),
        "claim_boundary": (
            "This table audits a pre-specified Macrophage-to-T CXCR3-axis context. It does not relax the main "
            "three-null FDR success criterion; candidates that fail any null remain exploratory/mixed evidence."
        ),
    }
    manifest_path = goal_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))


def _collect_targeted_fdr(fdr_dir: Path) -> pd.DataFrame:
    rows = []
    prediction_stem = "constructed_replicate__spatialself_wavelet_predictions__lr_mechanism_prior_rank_predictions"
    for null_model in NULL_MODELS:
        path = fdr_dir / f"{prediction_stem}__{null_model}__cell_type_pair_direct_fdr.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["null_model"] = null_model
        frame["source_path"] = _rel(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    target = table[
        table["sender_cell_type"].astype(str).eq(TARGET_CONTEXT[0])
        & table["receiver_cell_type"].astype(str).eq(TARGET_CONTEXT[1])
    ].copy()
    return target


def _build_audit(evidence: pd.DataFrame, targeted: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for row in evidence.itertuples(index=False):
        lr_id = str(row.lr_id)
        fdr_by_null = _fdr_by_null(targeted, lr_id)
        ranking_pass = (
            _safe_float(getattr(row, "raw_spatialself_rank", 999)) <= 10
            and _safe_float(getattr(row, "commot_rank", 0)) > 10
            and _safe_float(getattr(row, "cellchat_rank", 0)) > 10
            and _safe_float(getattr(row, "cellnest_rank", 0)) > 10
        )
        downstream_pass = (
            _safe_float(getattr(row, "downstream_spearman_r", 0)) > 0
            and _safe_float(getattr(row, "downstream_bh_fdr", 1)) <= 0.1
        )
        context_pass = _safe_float(getattr(row, "calibrated_top10_both_rule_rate", 0)) >= 0.9
        lr_identity_pass = _safe_float(getattr(row, "raw_lr_identity_bh_fdr", 1)) <= alpha
        fdr_pass_by_null = {
            null_model: fdr_by_null.get(null_model, 1.0) <= alpha for null_model in NULL_MODELS
        }
        all_three_targeted_fdr_pass = all(fdr_pass_by_null.values()) and set(fdr_by_null) >= set(NULL_MODELS)
        strict_success = ranking_pass and downstream_pass and context_pass and lr_identity_pass and all_three_targeted_fdr_pass
        if strict_success:
            status = "strict_success"
        elif ranking_pass and downstream_pass and context_pass and lr_identity_pass:
            status = "strong_mixed_pending_stricter_fdr"
        elif ranking_pass and downstream_pass:
            status = "rank_downstream_candidate"
        else:
            status = "not_strict_candidate"
        rows.append(
            {
                "lr_id": lr_id,
                "canonical_pair": getattr(row, "canonical_pair", ""),
                "raw_spatialself_rank": getattr(row, "raw_spatialself_rank", ""),
                "commot_rank": getattr(row, "commot_rank", ""),
                "cellchat_rank": getattr(row, "cellchat_rank", ""),
                "cellnest_rank": getattr(row, "cellnest_rank", ""),
                "downstream_spearman_r": getattr(row, "downstream_spearman_r", ""),
                "downstream_bh_fdr": getattr(row, "downstream_bh_fdr", ""),
                "raw_lr_identity_bh_fdr": getattr(row, "raw_lr_identity_bh_fdr", ""),
                "calibrated_dominant_context": getattr(row, "calibrated_dominant_context", ""),
                "calibrated_top10_both_rule_rate": getattr(row, "calibrated_top10_both_rule_rate", ""),
                "target_context": f"{TARGET_CONTEXT[0]}->{TARGET_CONTEXT[1]}",
                "targeted_spatial_bh_fdr": fdr_by_null.get("spatial_permutation", ""),
                "targeted_receiver_bh_fdr": fdr_by_null.get("receiver_permutation", ""),
                "targeted_lr_identity_bh_fdr": fdr_by_null.get("lr_identity_permutation", ""),
                "ranking_specific_pass": bool(ranking_pass),
                "downstream_pass": bool(downstream_pass),
                "context_pass": bool(context_pass),
                "raw_lr_identity_pass": bool(lr_identity_pass),
                "targeted_spatial_pass": bool(fdr_pass_by_null["spatial_permutation"]),
                "targeted_receiver_pass": bool(fdr_pass_by_null["receiver_permutation"]),
                "targeted_lr_identity_pass": bool(fdr_pass_by_null["lr_identity_permutation"]),
                "all_three_targeted_fdr_pass": bool(all_three_targeted_fdr_pass),
                "strict_goal4_success": bool(strict_success),
                "audit_status": status,
                "claim_boundary": (
                    "Exploratory/mixed unless ranking, downstream, context, raw LR-identity FDR, and all three "
                    "targeted null FDR checks pass."
                ),
            }
        )
    return pd.DataFrame(rows)


def _fdr_by_null(targeted: pd.DataFrame, lr_id: str) -> dict[str, float]:
    if targeted.empty:
        return {}
    subset = targeted[targeted["lr_id"].astype(str).eq(lr_id)].copy()
    if subset.empty:
        return {}
    values: dict[str, float] = {}
    for null_model, group in subset.groupby("null_model", sort=False):
        values[str(null_model)] = float(pd.to_numeric(group["group_bh_fdr"], errors="coerce").min())
    return values


def _safe_float(value: Any) -> float:
    try:
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
