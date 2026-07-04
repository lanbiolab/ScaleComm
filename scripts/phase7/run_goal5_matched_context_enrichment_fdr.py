#!/usr/bin/env python
"""Matched-context enrichment FDR using existing model-rerun null predictions.

This diagnostic asks a narrower question than global LR-pair scoring:
does a candidate LR become specifically enriched in pre-defined biological
contexts compared with its own off-context background, and does that remain
significant under model-rerun nulls without high-expression negative leakage?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

DATASET_SPECS = {
    "post_xenium_colon": {
        "dataset_dir": GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset",
        "observed_prediction": GOAL5_DIR
        / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "model_rerun_dir": GOAL5_DIR / "post_xenium_colon_model_rerun_fdr_50perm",
        "output_dir": GOAL5_DIR / "post_xenium_colon_matched_context_enrichment_fdr",
    },
    "slide_tags_tonsil_n1200": {
        "dataset_dir": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/constructed_replicate",
        "observed_prediction": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/"
        "tonsil_fresh_spatialself_wavelet_predictions.csv",
        "model_rerun_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm",
        "output_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_matched_context_enrichment_fdr",
    },
    "melanoma_real_background": {
        "dataset_dir": PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/constructed_replicate",
        "observed_prediction": PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/calibrated_spatialself/"
        "constructed_replicate__spatialself_wavelet_predictions__lr_mechanism_prior_rank_predictions.csv",
        "model_rerun_dir": GOAL5_DIR / "melanoma_real_background_model_rerun_fdr_20perm",
        "output_dir": GOAL5_DIR / "melanoma_real_background_matched_context_enrichment_fdr",
    },
}

NULL_MODELS = ("spatial_coordinate_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="post_xenium_colon")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--min-target-edges", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-sender-cell-type", default="")
    parser.add_argument("--target-receiver-cell-type", default="")
    parser.add_argument("--positive-control-regex", default="")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    dataset_dir = Path(spec["dataset_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    observed = pd.read_csv(spec["observed_prediction"])

    target_masks = _build_matched_context_masks(
        candidates,
        lr_pairs,
        min_target_edges=int(args.min_target_edges),
        target_sender_cell_type=str(args.target_sender_cell_type),
        target_receiver_cell_type=str(args.target_receiver_cell_type),
        positive_control_regex=str(args.positive_control_regex),
    )
    observed_stats = _lr_context_stats(
        candidates,
        observed,
        lr_pairs,
        target_masks,
        top_fraction=float(args.top_fraction),
    )

    null_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for null_model in NULL_MODELS:
        pred_paths = sorted((Path(spec["model_rerun_dir"]) / "null_predictions" / null_model).glob("perm_*/*_predictions.csv"))
        for pred_path in pred_paths:
            perm = pred_path.parent.name
            try:
                pred = pd.read_csv(pred_path)
                stats = _lr_context_stats(
                    candidates,
                    pred,
                    lr_pairs,
                    target_masks,
                    top_fraction=float(args.top_fraction),
                )
                for row in stats.to_dict("records"):
                    row["null_model"] = null_model
                    row["permutation"] = perm
                    row["prediction_path"] = _rel(pred_path)
                    null_rows.append(row)
                status_rows.append({"null_model": null_model, "permutation": perm, "status": "completed", "detail": ""})
            except Exception as exc:  # noqa: BLE001
                status_rows.append({"null_model": null_model, "permutation": perm, "status": "failed", "detail": repr(exc)})

    null_stats = pd.DataFrame(null_rows)
    status = pd.DataFrame(status_rows)
    fdr = _fdr(observed_stats, null_stats, lr_pairs, null_models=NULL_MODELS)
    control = _control_summary(fdr, alpha=float(args.alpha))
    audit = _audit(control, status, null_models=NULL_MODELS)

    observed_path = output_dir / f"{args.dataset}_matched_context_observed_lr_stats.csv"
    null_path = output_dir / f"{args.dataset}_matched_context_null_lr_stats.csv"
    fdr_path = output_dir / f"{args.dataset}_matched_context_lr_pair_fdr.csv"
    control_path = output_dir / f"{args.dataset}_matched_context_control_fdr_summary.csv"
    audit_path = output_dir / f"{args.dataset}_matched_context_fdr_audit.csv"
    status_path = output_dir / f"{args.dataset}_matched_context_status.csv"
    target_path = output_dir / f"{args.dataset}_matched_context_target_masks.csv"
    observed_stats.to_csv(observed_path, index=False)
    null_stats.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    audit.to_csv(audit_path, index=False)
    status.to_csv(status_path, index=False)
    _target_mask_summary(candidates, lr_pairs, target_masks).to_csv(target_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "dataset_dir": _rel(dataset_dir),
        "observed_prediction": _rel(Path(spec["observed_prediction"])),
        "model_rerun_dir": _rel(Path(spec["model_rerun_dir"])),
        "output_dir": _rel(output_dir),
        "null_models": list(NULL_MODELS),
        "top_fraction": float(args.top_fraction),
        "min_target_edges": int(args.min_target_edges),
        "alpha": float(args.alpha),
        "target_sender_cell_type": str(args.target_sender_cell_type),
        "target_receiver_cell_type": str(args.target_receiver_cell_type),
        "positive_control_regex": str(args.positive_control_regex),
        "n_completed_null_predictions": int(status["status"].eq("completed").sum()) if not status.empty else 0,
        "strict_matched_context_success": bool(
            audit.get("matched_context_fdr_status", pd.Series(dtype=str)).eq(
                "positive_significant_negative_suppressed_all_nulls"
            ).any()
        ),
        "claim_boundary": (
            "Post hoc diagnostic using existing model-rerun predictions. It tests target-context enrichment "
            "relative to each LR's off-context background, with high-expression negatives projected into the same "
            "positive-control context masks. It should be reported as a candidate FDR statistic unless prospectively "
            "adopted and rerun."
        ),
        "outputs": {
            "target_masks": _rel(target_path),
            "observed_lr_stats": _rel(observed_path),
            "null_lr_stats": _rel(null_path),
            "lr_pair_fdr": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "audit": _rel(audit_path),
            "status": _rel(status_path),
        },
    }
    (output_dir / f"{args.dataset}_matched_context_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _build_matched_context_masks(
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    min_target_edges: int,
    target_sender_cell_type: str = "",
    target_receiver_cell_type: str = "",
    positive_control_regex: str = "",
) -> dict[str, set[str]]:
    positive_pairs = lr_pairs[_positive_control_mask(lr_pairs)].copy()
    if positive_control_regex:
        names = positive_pairs.get("control_name", pd.Series("", index=positive_pairs.index)).fillna("").astype(str)
        ids = positive_pairs.get("lr_id", pd.Series("", index=positive_pairs.index)).fillna("").astype(str)
        positive_pairs = positive_pairs[names.str.contains(positive_control_regex, regex=True) | ids.str.contains(positive_control_regex, regex=True)]
    negative_pairs = lr_pairs[_negative_control_mask(lr_pairs)].copy()
    positive_contexts: set[tuple[str, str]] = set()
    masks: dict[str, set[str]] = {}

    for lr_id in positive_pairs["lr_id"].astype(str):
        subset = candidates[candidates["lr_id"].astype(str).eq(lr_id)]
        target = subset[
            _bool_col(subset, "type_rule_satisfied")
            & _bool_col(subset, "range_rule_satisfied")
            & _bool_col(subset, "niche_rule_satisfied")
        ]
        target = _restrict_context(
            target,
            sender_cell_type=target_sender_cell_type,
            receiver_cell_type=target_receiver_cell_type,
        )
        if len(target) < min_target_edges:
            continue
        masks[lr_id] = set(target["candidate_id"].astype(str))
        positive_contexts.update(zip(target["sender_niche"].astype(str), target["receiver_niche"].astype(str)))

    for lr_id in negative_pairs["lr_id"].astype(str):
        subset = candidates[candidates["lr_id"].astype(str).eq(lr_id)]
        context_mask = [
            (sender, receiver) in positive_contexts
            for sender, receiver in zip(subset["sender_niche"].astype(str), subset["receiver_niche"].astype(str))
        ]
        target = subset[pd.Series(context_mask, index=subset.index)]
        target = _restrict_context(
            target,
            sender_cell_type=target_sender_cell_type,
            receiver_cell_type=target_receiver_cell_type,
        )
        if len(target) < min_target_edges:
            continue
        masks[lr_id] = set(target["candidate_id"].astype(str))

    return masks


def _restrict_context(frame: pd.DataFrame, *, sender_cell_type: str, receiver_cell_type: str) -> pd.DataFrame:
    out = frame
    if sender_cell_type:
        out = out[out.get("sender_cell_type", pd.Series("", index=out.index)).astype(str).eq(sender_cell_type)]
    if receiver_cell_type:
        out = out[out.get("receiver_cell_type", pd.Series("", index=out.index)).astype(str).eq(receiver_cell_type)]
    return out


def _lr_context_stats(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    target_masks: dict[str, set[str]],
    *,
    top_fraction: float,
) -> pd.DataFrame:
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    merged["candidate_id"] = merged["candidate_id"].astype(str)
    merged["lr_id"] = merged["lr_id"].astype(str)
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    rows = []
    for lr_id, target_ids in target_masks.items():
        group = merged[merged["lr_id"].eq(str(lr_id))]
        if group.empty:
            continue
        target = group[group["candidate_id"].isin(target_ids)]["score"].to_numpy(dtype=float)
        background = group[~group["candidate_id"].isin(target_ids)]["score"].to_numpy(dtype=float)
        target_stat = _top_tail_mean(target, top_fraction=top_fraction)
        background_stat = _top_tail_mean(background, top_fraction=top_fraction)
        rows.append(
            {
                "lr_id": lr_id,
                "matched_context_enrichment": target_stat - background_stat,
                "target_top_tail_mean": target_stat,
                "off_context_top_tail_mean": background_stat,
                "n_target_edges": int(len(target)),
                "n_off_context_edges": int(len(background)),
                "top_fraction": float(top_fraction),
            }
        )
    return pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")


def _fdr(
    observed: pd.DataFrame,
    null_stats: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    null_models: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for null_model in null_models:
        null = null_stats[null_stats.get("null_model", pd.Series(dtype=str)).astype(str).eq(null_model)]
        for _, obs in observed.iterrows():
            values = pd.to_numeric(
                null[null["lr_id"].astype(str).eq(str(obs["lr_id"]))]["matched_context_enrichment"],
                errors="coerce",
            ).dropna()
            observed_value = float(obs["matched_context_enrichment"])
            if values.empty:
                p_value = np.nan
                null_mean = np.nan
                null_q95 = np.nan
            else:
                arr = values.to_numpy(dtype=float)
                p_value = (float((arr >= observed_value).sum()) + 1.0) / (float(len(arr)) + 1.0)
                null_mean = float(arr.mean())
                null_q95 = float(np.quantile(arr, 0.95))
            rows.append(
                {
                    "null_model": null_model,
                    "lr_id": obs["lr_id"],
                    "matched_context_enrichment": observed_value,
                    "matched_context_p_value": p_value,
                    "n_completed_permutations": int(len(values)),
                    "null_mean": null_mean,
                    "null_q95": null_q95,
                }
            )
    out = pd.DataFrame(rows)
    out["matched_context_bh_fdr"] = np.nan
    for _, idx in out.groupby("null_model").groups.items():
        out.loc[idx, "matched_context_bh_fdr"] = _bh(out.loc[idx, "matched_context_p_value"].to_numpy(dtype=float))
    return out.merge(lr_pairs, on="lr_id", how="left")


def _control_summary(fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    for null_model, group in fdr.groupby("null_model", sort=False):
        positives = group[_positive_control_mask(group)]
        negatives = group[_negative_control_mask(group)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "matched_context_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "matched_context_bh_fdr"),
                "n_positive_significant": int((pd.to_numeric(positives.get("matched_context_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_negative_significant": int((pd.to_numeric(negatives.get("matched_context_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()),
                "n_completed_permutations_min": int(group["n_completed_permutations"].min()) if not group.empty else 0,
                "control_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _audit(control: pd.DataFrame, status: pd.DataFrame, *, null_models: tuple[str, ...]) -> pd.DataFrame:
    statuses = dict(zip(control["null_model"].astype(str), control["control_status"].astype(str))) if not control.empty else {}
    passing = [null_model for null_model in null_models if statuses.get(null_model) == "positive_significant_negative_suppressed"]
    final = (
        "positive_significant_negative_suppressed_all_nulls"
        if set(passing) == set(null_models)
        else "not_strict_matched_context_success"
    )
    return pd.DataFrame(
        [
            {
                "matched_context_fdr_status": final,
                "null_models_required": ";".join(null_models),
                "null_models_passing": ";".join(passing),
                "n_completed_null_predictions": int(status["status"].eq("completed").sum()) if not status.empty else 0,
                "n_failed_null_predictions": int(status["status"].eq("failed").sum()) if not status.empty else 0,
            }
        ]
    )


def _target_mask_summary(
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    target_masks: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for lr_id, target_ids in target_masks.items():
        group = candidates[candidates["lr_id"].astype(str).eq(str(lr_id))]
        rows.append(
            {
                "lr_id": lr_id,
                "n_target_edges": int(len(target_ids)),
                "n_total_edges": int(len(group)),
                "target_fraction": float(len(target_ids) / len(group)) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")


def _positive_control_mask(frame: pd.DataFrame) -> pd.Series:
    control_class = frame.get("control_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    fp = frame.get("false_positive_high_expression", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return control_class.str.contains("positive_prior|context_specific_prior", regex=True) & ~fp


def _negative_control_mask(frame: pd.DataFrame) -> pd.Series:
    fp = frame.get("false_positive_high_expression", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    control_class = frame.get("control_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    return fp | control_class.str.contains("negative", case=False, regex=False)


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    pos = int((pd.to_numeric(positives.get("matched_context_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg = int((pd.to_numeric(negatives.get("matched_context_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos > 0 and neg == 0:
        return "positive_significant_negative_suppressed"
    if pos > 0 and neg > 0:
        return "positive_significant_but_negative_leakage"
    return "no_positive_recovery_negative_suppressed" if neg == 0 else "negative_leakage_no_positive_recovery"


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values(["matched_context_bh_fdr", "matched_context_enrichment"], ascending=[True, False]).iloc[0].get("control_name", ""))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series(False, index=frame.index)).fillna(False).astype(bool)


def _top_tail_mean(scores: np.ndarray, *, top_fraction: float) -> float:
    if len(scores) == 0:
        return 0.0
    n_top = max(int(np.ceil(len(scores) * top_fraction)), 1)
    return float(np.sort(scores)[-n_top:].mean())


def _bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
