#!/usr/bin/env python
"""Run targeted region-aware FDR for the Slide-tags tonsil Goal 5 subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import PermutationFDRConfig, permutation_group_fdr


GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/constructed_replicate"
PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/predictions/"
    "spatialself_wavelet_calibrated/constructed_replicate__spatialself_wavelet_calibrated_predictions.csv"
)
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL_DIR))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--prediction-path", default=str(PREDICTION_PATH))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=202609)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    prediction_path = Path(args.prediction_path)
    output_dir = goal_dir / f"slide_tags_tonsil_targeted_region_fdr_{int(args.n_permutations)}perm"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    predictions = pd.read_csv(prediction_path)

    targeted = _targeted_candidates(candidates)
    targeted_path = output_dir / "slide_tags_tonsil_targeted_region_candidates.csv"
    targeted.to_csv(targeted_path, index=False)

    config = PermutationFDRConfig(
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
        score_column="score",
    )
    lr_rows = []
    cell_type_rows = []
    for null_model in NULL_MODELS:
        lr_table = permutation_group_fdr(
            targeted,
            predictions,
            null_model=null_model,
            group_columns=["lr_id", "ligand", "receptor", "mechanism"],
            config=config,
            top_fraction=float(args.top_fraction),
        ).merge(lr_pairs, on=["lr_id", "ligand", "receptor", "mechanism"], how="left")
        lr_table["null_model"] = null_model
        lr_path = output_dir / f"spatialself_calibrated__{null_model}__targeted_lr_pair_fdr.csv"
        lr_table.to_csv(lr_path, index=False)
        lr_rows.append(lr_table.assign(fdr_path=_rel(lr_path)))

        ct_table = permutation_group_fdr(
            targeted,
            predictions,
            null_model=null_model,
            group_columns=["sender_cell_type", "receiver_cell_type", "lr_id", "ligand", "receptor", "mechanism"],
            config=config,
            top_fraction=float(args.top_fraction),
        ).merge(lr_pairs, on=["lr_id", "ligand", "receptor", "mechanism"], how="left")
        ct_table["null_model"] = null_model
        ct_path = output_dir / f"spatialself_calibrated__{null_model}__targeted_cell_type_pair_fdr.csv"
        ct_table.to_csv(ct_path, index=False)
        cell_type_rows.append(ct_table.assign(fdr_path=_rel(ct_path)))

    lr_all = pd.concat(lr_rows, ignore_index=True)
    ct_all = pd.concat(cell_type_rows, ignore_index=True)
    control_summary = _control_summary(lr_all, alpha=float(args.alpha))
    candidate_summary = _candidate_summary(targeted, lr_pairs)
    target_audit = _target_audit(lr_all, ct_all, alpha=float(args.alpha))

    lr_all.to_csv(output_dir / "slide_tags_tonsil_targeted_lr_pair_fdr_all.csv", index=False)
    ct_all.to_csv(output_dir / "slide_tags_tonsil_targeted_cell_type_pair_fdr_all.csv", index=False)
    control_summary.to_csv(output_dir / "slide_tags_tonsil_targeted_control_fdr_summary.csv", index=False)
    candidate_summary.to_csv(output_dir / "slide_tags_tonsil_targeted_candidate_summary.csv", index=False)
    target_audit.to_csv(output_dir / "slide_tags_tonsil_targeted_region_fdr_audit.csv", index=False)

    manifest = {
        "dataset": "slide_tags_tonsil_subset",
        "dataset_dir": _rel(dataset_dir),
        "prediction_path": _rel(prediction_path),
        "output_dir": _rel(output_dir),
        "targeted_candidate_path": _rel(targeted_path),
        "null_models": list(NULL_MODELS),
        "n_permutations": int(args.n_permutations),
        "alpha": float(args.alpha),
        "top_fraction": float(args.top_fraction),
        "target_definition": (
            "positive/control-prior LR pairs use type_rule_satisfied & range_rule_satisfied & niche_rule_satisfied; "
            "high-expression negative controls retain all candidate edges."
        ),
        "strict_success": bool(
            (target_audit["targeted_region_fdr_status"] == "targeted_positive_significant_negative_suppressed").any()
        ),
        "claim_boundary": (
            "Supplemental region-aware FDR audit on a high-resolution tonsil subset. It supports platform validation "
            "only when paired with downstream target-response evidence and negative-control suppression."
        ),
        "outputs": {
            "candidate_summary": _rel(output_dir / "slide_tags_tonsil_targeted_candidate_summary.csv"),
            "control_summary": _rel(output_dir / "slide_tags_tonsil_targeted_control_fdr_summary.csv"),
            "target_audit": _rel(output_dir / "slide_tags_tonsil_targeted_region_fdr_audit.csv"),
            "lr_pair_fdr_all": _rel(output_dir / "slide_tags_tonsil_targeted_lr_pair_fdr_all.csv"),
            "cell_type_pair_fdr_all": _rel(output_dir / "slide_tags_tonsil_targeted_cell_type_pair_fdr_all.csv"),
        },
    }
    manifest_path = output_dir / "slide_tags_tonsil_targeted_region_fdr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target_audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _targeted_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = ["type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied", "false_positive_high_expression"]
    missing = [column for column in required if column not in candidates.columns]
    if missing:
        raise ValueError(f"candidate table missing required targeted columns: {missing}")
    positive_rule = (
        candidates["type_rule_satisfied"].fillna(False).astype(bool)
        & candidates["range_rule_satisfied"].fillna(False).astype(bool)
        & candidates["niche_rule_satisfied"].fillna(False).astype(bool)
        & ~candidates["false_positive_high_expression"].fillna(False).astype(bool)
    )
    negative_control = candidates["false_positive_high_expression"].fillna(False).astype(bool)
    targeted = candidates[positive_rule | negative_control].copy()
    targeted["targeted_region_reason"] = "positive_rule_satisfied"
    targeted.loc[negative_control.loc[targeted.index].to_numpy(dtype=bool), "targeted_region_reason"] = (
        "negative_control_all_edges"
    )
    return targeted


def _candidate_summary(targeted: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        targeted.groupby(["lr_id", "targeted_region_reason"], dropna=False)
        .size()
        .reset_index(name="n_targeted_edges")
    )
    totals = targeted.groupby("lr_id").size().reset_index(name="n_total_targeted_edges")
    return (
        summary.merge(totals, on="lr_id", how="left")
        .merge(lr_pairs, on="lr_id", how="left")
        .sort_values(["false_positive_high_expression", "n_total_targeted_edges", "lr_id"], ascending=[True, False, True])
    )


def _control_summary(lr_fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for null_model, group in lr_fdr.groupby("null_model", sort=False):
        positives = group[group["true_lr"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "group_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "group_bh_fdr"),
                "n_positive_significant": int((pd.to_numeric(positives["group_bh_fdr"], errors="coerce") <= alpha).sum()),
                "n_negative_significant": int((pd.to_numeric(negatives["group_bh_fdr"], errors="coerce") <= alpha).sum()),
                "control_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _target_audit(lr_fdr: pd.DataFrame, cell_type_fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for lr_id, group in lr_fdr.groupby("lr_id", sort=False):
        meta = group.iloc[0].to_dict()
        by_null = group.groupby("null_model")["group_bh_fdr"].min()
        all_three_lr = set(by_null.index) >= set(NULL_MODELS) and bool((pd.to_numeric(by_null, errors="coerce") <= alpha).all())
        ct_group = cell_type_fdr[cell_type_fdr["lr_id"].astype(str).eq(str(lr_id))]
        ct_min = ct_group.groupby("null_model")["group_bh_fdr"].min() if not ct_group.empty else pd.Series(dtype=float)
        any_cell_type_all_three = _any_cell_type_all_three(ct_group, alpha=alpha)
        is_negative = bool(meta.get("false_positive_high_expression", False))
        if is_negative and (all_three_lr or any_cell_type_all_three):
            status = "negative_control_leakage"
        elif (not is_negative) and all_three_lr:
            status = "targeted_positive_significant_negative_suppressed"
        elif (not is_negative) and any_cell_type_all_three:
            status = "targeted_cell_type_positive_significant"
        else:
            status = "not_targeted_fdr_significant"
        rows.append(
            {
                "lr_id": lr_id,
                "control_name": meta.get("control_name", ""),
                "control_class": meta.get("control_class", ""),
                "false_positive_high_expression": is_negative,
                "targeted_lr_spatial_bh_fdr": by_null.get("spatial_permutation", ""),
                "targeted_lr_receiver_bh_fdr": by_null.get("receiver_permutation", ""),
                "targeted_lr_identity_bh_fdr": by_null.get("lr_identity_permutation", ""),
                "targeted_lr_all_three_fdr05": bool(all_three_lr),
                "best_cell_type_spatial_bh_fdr": ct_min.get("spatial_permutation", ""),
                "best_cell_type_receiver_bh_fdr": ct_min.get("receiver_permutation", ""),
                "best_cell_type_identity_bh_fdr": ct_min.get("lr_identity_permutation", ""),
                "targeted_cell_type_any_all_three_fdr05": bool(any_cell_type_all_three),
                "targeted_region_fdr_status": status,
            }
        )
    return pd.DataFrame(rows)


def _any_cell_type_all_three(cell_type_fdr: pd.DataFrame, *, alpha: float) -> bool:
    if cell_type_fdr.empty:
        return False
    for _, group in cell_type_fdr.groupby(["sender_cell_type", "receiver_cell_type", "lr_id"], dropna=False):
        by_null = group.groupby("null_model")["group_bh_fdr"].min()
        if set(by_null.index) >= set(NULL_MODELS) and bool((pd.to_numeric(by_null, errors="coerce") <= alpha).all()):
            return True
    return False


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False]).iloc[0]
    return str(row.get("control_name", f"{row.get('ligand', '')}-{row.get('receptor', '')}"))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    if positives.empty or negatives.empty:
        return "missing_positive_or_negative_controls"
    n_pos = int((pd.to_numeric(positives["group_bh_fdr"], errors="coerce") <= alpha).sum())
    n_neg = int((pd.to_numeric(negatives["group_bh_fdr"], errors="coerce") <= alpha).sum())
    if n_pos > 0 and n_neg == 0:
        return "positive_significant_negative_suppressed"
    if n_pos > 0 and n_neg > 0:
        return "positive_and_negative_controls_significant"
    if n_pos == 0 and n_neg > 0:
        return "negative_controls_significant_without_positive"
    return "no_control_significant"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
