#!/usr/bin/env python
"""Summarize Phase 7 Goal 5 model-rerun calibration diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

RUNS = [
    {
        "diagnostic_id": "current_calibrated_raw_50perm",
        "description": "Current calibrated SpatialSelf, raw sigmoid scores, 50 model reruns per null.",
        "path": GOAL5_DIR / "post_xenium_colon_model_rerun_fdr_50perm",
    },
    {
        "diagnostic_id": "hardneg_v1_raw_20perm",
        "description": "Hard-negative calibration v1, raw sigmoid scores, 20 model reruns per null.",
        "path": GOAL5_DIR / "post_xenium_hardneg_v1_model_rerun_fdr_20perm",
    },
    {
        "diagnostic_id": "hardneg_v1_priorrank_20perm",
        "description": "Hard-negative calibration v1 plus LR mechanism prior-rank score transform, 20 model reruns per null.",
        "path": GOAL5_DIR / "post_xenium_hardneg_v1_priorrank_model_rerun_fdr_20perm",
    },
    {
        "diagnostic_id": "hardneg_v1_feasgate_20perm",
        "description": "Hard-negative calibration v1 plus LR feasibility-gated rank score transform, 20 model reruns per null.",
        "path": GOAL5_DIR / "post_xenium_hardneg_v1_feasgate_model_rerun_fdr_20perm",
    },
    {
        "diagnostic_id": "feasibility_loss_v1_raw_20perm",
        "description": (
            "Feasibility-aware training loss v1 with hard-negative penalty, raw sigmoid scores, "
            "20 model reruns per null."
        ),
        "path": GOAL5_DIR / "post_xenium_feasibility_loss_v1_model_rerun_fdr_20perm",
    },
    {
        "diagnostic_id": "feas020_hn005_raw_5perm_smoke",
        "description": (
            "Lower feasibility/hard-negative weights, raw sigmoid scores, 5 model reruns per null. "
            "Low-resolution smoke diagnostic only; not enough permutations for final BH-FDR closure."
        ),
        "path": GOAL5_DIR / "post_xenium_loss_sweep_feas020_hn005_model_rerun_fdr_5perm",
    },
    {
        "diagnostic_id": "context_contrastive_v1_raw_5perm_smoke",
        "description": (
            "Receiver-context contrastive loss v1 plus lower feasibility/hard-negative weights, raw sigmoid scores, "
            "5 model reruns per null. Low-resolution smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "post_xenium_context_contrastive_v1_model_rerun_fdr_5perm",
    },
    {
        "diagnostic_id": "context_contrastive_hard_v1_raw_5perm_smoke",
        "description": (
            "Hard receiver-context contrastive loss v1 plus lower feasibility/hard-negative weights, raw sigmoid "
            "scores, 5 model reruns per null. Low-resolution smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "post_xenium_context_contrastive_hard_v1_model_rerun_fdr_5perm",
    },
    {
        "diagnostic_id": "context_contrastive_refeature_v1_raw_5perm_smoke",
        "description": (
            "Hard receiver-context contrastive loss v1 with corrupted receiver edge-prior features recomputed, raw "
            "sigmoid scores, 5 model reruns per null. Low-resolution smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "post_xenium_context_contrastive_refeature_v1_model_rerun_fdr_5perm",
    },
    {
        "diagnostic_id": "null_candidate_v1_raw_5perm_smoke",
        "description": (
            "Null-like candidate softplus penalty v1 using hard receiver corrupted negatives, raw sigmoid scores, "
            "5 model reruns per null. Low-resolution smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "post_xenium_null_candidate_v1_model_rerun_fdr_5perm",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1800_raw_1perm_smoke",
        "description": (
            "Slide-tags tonsil n=1800 high-resolution subset, raw scores, 1 model rerun per null. "
            "Smoke only: useful for direction/negative leakage, impossible for final BH-FDR closure."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1800_model_rerun_fdr_smoke",
        "prefix": "slide_tags_tonsil_n1800",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_raw_5perm_smoke",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset, raw scores, 5 model reruns per null. "
            "Smoke only: positives exceed null q95 and high-expression negatives are suppressed, but "
            "minimum attainable p-value remains too coarse for final BH-FDR closure."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr_5perm_smoke",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_raw_20perm",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset, raw scores, 20 model reruns per null. "
            "This is a pre-final strict-rerun diagnostic: LR-identity/receiver nulls are clean but just above "
            "FDR 0.05, while spatial-coordinate null recovers positives and also calls high-expression negatives."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_feasgate_20perm",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset, LR feasibility-gated rank score transform, "
            "20 model reruns per null. This tests whether a shared post-hoc feasibility gate can suppress "
            "high-expression false positives under strict rerun nulls."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_feasgate_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_fp_suppressed_20perm",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset, false-positive/high-expression suppressed rank "
            "score transform, 20 model reruns per null. Diagnostic only: it tests whether explicit control-label "
            "suppression can resolve the housekeeping negative leakage observed under strict rerun nulls."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_fp_suppressed_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_hardtrain_5perm_smoke",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset with hard-negative, feasibility, and hard-receiver "
            "context contrastive training, raw scores, 5 model reruns per null. Smoke only."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_hardtrain_model_rerun_fdr_5perm_smoke",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_hardtrain_20perm",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset with hard-negative, feasibility, and hard-receiver "
            "context contrastive training, raw scores, 20 model reruns per null. This tests whether training-side "
            "negative suppression separates housekeeping false positives from true tonsil positives."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_hardtrain_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_lr_identity_contrastive_1perm_smoke",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset with training-side LR-identity contrastive loss, "
            "4 epochs, 1 model rerun per null. Smoke only: tests whether same-edge LR identity contrast can "
            "suppress high-expression false positives while retaining positive controls."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_lr_identity_contrastive_model_rerun_fdr_1perm_smoke",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_control_contrastive_w010_1perm_smoke",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset with narrow control-aware contrastive loss "
            "(positive-control LR versus explicit high-expression negative-control LR on the same edge), "
            "8 epochs, 1 model rerun per null. Historical smoke only: null reruns used control-aware training "
            "but the observed prediction came from the previous calibrated model, so this is not a matched "
            "model-version closure diagnostic."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_control_contrastive_w010_model_rerun_fdr_1perm_smoke",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_control_contrastive_w010_matched_1perm_smoke",
        "description": (
            "Slide-tags tonsil n=1200 high-resolution subset with matched observed/null control-aware contrastive "
            "training, 8 epochs, 1 model rerun per null. Smoke only: high-expression negative controls are "
            "suppressed, but positive controls are not recovered under retrained nulls, so this direction is "
            "too conservative for strict closure."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_control_contrastive_w010_matched_model_rerun_fdr_1perm_smoke",
        "prefix": "slide_tags_tonsil_n1200",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_control_separation_20perm",
        "description": (
            "Slide-tags tonsil n=1200 raw model-rerun outputs tested with control-separation FDR: "
            "best positive top-tail statistic minus best high-expression negative top-tail statistic. "
            "This supports false-positive suppression only if positive-vs-negative separation exceeds rerun nulls."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_control_separation_fdr",
        "prefix": "slide_tags_tonsil_n1200_control_separation",
        "kind": "control_separation",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_hardtrain_control_separation_20perm",
        "description": (
            "Slide-tags tonsil n=1200 hardtrain model-rerun outputs tested with control-separation FDR."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_hardtrain_control_separation_fdr",
        "prefix": "slide_tags_tonsil_n1200_hardtrain_control_separation",
        "kind": "control_separation",
    },
    {
        "diagnostic_id": "post_xenium_colon_control_separation_50perm",
        "description": (
            "Post-Xenium colon model-rerun outputs tested with control-separation FDR."
        ),
        "path": GOAL5_DIR / "post_xenium_colon_control_separation_fdr",
        "prefix": "post_xenium_colon_control_separation",
        "kind": "control_separation",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_edge_background_residual_rank_20perm",
        "description": (
            "Slide-tags tonsil n=1200 existing model-rerun predictions re-scored with edge-background residual "
            "rank calibration, 20 permutations per null. Diagnostic only: this tests whether post-hoc subtraction "
            "of generic sender-receiver edge background can suppress high-expression leakage without retraining."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_spatial_background_residual_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200_spatial_background_calibrated",
        "kind": "spatial_background_calibrated",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_edge_background_residual_w025_5perm",
        "description": (
            "Slide-tags tonsil n=1200 existing model-rerun predictions re-scored with mild edge-background "
            "residual calibration, weight 0.25, 5 permutations per null. Smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_edge_background_residual_w0.25_fdr_5perm",
        "prefix": "slide_tags_tonsil_n1200_spatial_background_calibrated",
        "kind": "spatial_background_calibrated",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_edge_background_residual_w050_5perm",
        "description": (
            "Slide-tags tonsil n=1200 existing model-rerun predictions re-scored with mild edge-background "
            "residual calibration, weight 0.50, 5 permutations per null. Smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_edge_background_residual_w0.5_fdr_5perm",
        "prefix": "slide_tags_tonsil_n1200_spatial_background_calibrated",
        "kind": "spatial_background_calibrated",
    },
    {
        "diagnostic_id": "slide_tags_tonsil_n1200_edge_background_product_w025_5perm",
        "description": (
            "Slide-tags tonsil n=1200 existing model-rerun predictions re-scored with mild multiplicative "
            "edge-background suppression, weight 0.25, 5 permutations per null. Smoke diagnostic only."
        ),
        "path": GOAL5_DIR / "slide_tags_tonsil_n1200_edge_background_product_w0.25_fdr_5perm",
        "prefix": "slide_tags_tonsil_n1200_spatial_background_calibrated",
        "kind": "spatial_background_calibrated",
    },
]


def main() -> None:
    rows = []
    for run in RUNS:
        rows.append(_summarize_run(run))
    summary = pd.DataFrame(rows)
    out_path = GOAL5_DIR / "post_xenium_model_calibration_diagnostic_summary.csv"
    summary.to_csv(out_path, index=False)

    manifest = {
        "summary": _rel(out_path),
        "n_diagnostics": int(len(summary)),
        "best_current_direction": (
            "Hard-negative and feasibility-aware training suppress high-expression negatives but become conservative; "
            "mechanism/feasibility post-processing improves edge biological plausibility but does not close strict "
            "model-rerun FDR. The next model iteration should tune feasibility constraints, receptor/ligand "
            "presence, and hard-negative suppression so positive controls remain recoverable under retrained nulls. "
            "Receiver-context contrastive training improves the direction of the null comparison but is still too "
            "weak to separate observed positives from spatial/receiver rerun nulls; hard receiver corruption is "
            "cleaner for negative controls but more conservative for positive recovery, and recomputing corrupted "
            "edge-prior features does not restore positive recovery. A direct null-like candidate penalty further "
            "suppresses negatives but also depresses observed positive controls and does not separate receiver/spatial "
            "rerun nulls. Narrow control-aware contrastive training suppresses high-expression negative controls "
            "in matched and unmatched 1-per-null smoke runs, but the matched diagnostic remains conservative and "
            "does not recover positive controls under retrained nulls. "
            "Smoke runs with very few permutations are useful only for ranking/debugging because their minimum "
            "attainable p-value is too coarse for final BH-FDR closure."
        ),
        "claim_boundary": (
            "These are diagnostic runs. They should not be counted as Phase 7 strict success unless "
            "model_rerun_all_null_success is true and high-expression negative controls are suppressed under all "
            "requested nulls."
        ),
        "runs": [
            {
                "diagnostic_id": row["diagnostic_id"],
                "path": _rel(row["path"]),
                "description": row["description"],
            }
            for row in RUNS
        ],
    }
    manifest_path = GOAL5_DIR / "post_xenium_model_calibration_diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _summarize_run(run: dict) -> dict:
    path = Path(run["path"])
    prefix = str(run.get("prefix", "post_xenium_colon"))
    if run.get("kind") == "spatial_background_calibrated":
        control = _csv(path / f"{prefix}_control_fdr_summary.csv")
        audit = _csv(path / f"{prefix}_fdr_audit.csv")
        manifest = _json(path / f"{prefix}_manifest.json")
    elif run.get("kind") == "control_separation":
        return _summarize_control_separation_run(run)
    else:
        control = _csv(path / f"{prefix}_model_rerun_control_fdr_summary.csv")
        audit = _csv(path / f"{prefix}_model_rerun_fdr_audit.csv")
        manifest = _json(path / f"{prefix}_model_rerun_fdr_manifest.json")
    status = audit.iloc[0].to_dict() if not audit.empty else {}
    row = {
        "diagnostic_id": run["diagnostic_id"],
        "description": run["description"],
        "path": _rel(path),
        "score_transform": manifest.get("score_transform", manifest.get("calibration", "raw")),
        "background_weight": manifest.get("background_weight", ""),
        "n_completed_reruns": status.get("n_completed_reruns", 0),
        "n_failed_reruns": status.get("n_failed_reruns", 0),
            "model_rerun_all_null_status": status.get("model_rerun_all_null_status", "missing"),
            "null_models_passing": status.get("null_models_passing", ""),
        }
    if control.empty:
        row.update(
            {
                "best_positive_fdr_min": "",
                "best_negative_fdr_min": "",
                "n_nulls_positive_recovered": 0,
                "n_nulls_negative_suppressed": 0,
                "n_nulls_positive_recovered_negative_suppressed": 0,
                "diagnostic_interpretation": "missing_control_summary",
            }
        )
        return row
    positive_sig = pd.to_numeric(control["n_positive_significant"], errors="coerce").fillna(0) > 0
    negative_sig = pd.to_numeric(control["n_negative_significant"], errors="coerce").fillna(999) > 0
    row.update(
        {
            "best_positive_fdr_min": _min_numeric(control, "best_positive_group_bh_fdr"),
            "best_negative_fdr_min": _min_numeric(control, "best_negative_group_bh_fdr"),
            "min_completed_per_null": _min_numeric(control, "n_completed_permutations_min"),
            "min_possible_raw_p": _min_possible_raw_p(control),
            "n_nulls_positive_recovered": int(positive_sig.sum()),
            "n_nulls_negative_suppressed": int((~negative_sig).sum()),
            "n_nulls_positive_recovered_negative_suppressed": int((positive_sig & ~negative_sig).sum()),
            "control_statuses": ";".join(control["control_status"].astype(str).tolist()),
            "diagnostic_interpretation": _interpret(positive_sig, negative_sig),
        }
    )
    return row


def _summarize_control_separation_run(run: dict) -> dict:
    path = Path(run["path"])
    prefix = str(run["prefix"])
    fdr = _csv(path / f"{prefix}_fdr.csv")
    observed = _csv(path / f"{prefix}_observed.csv")
    manifest = _json(path / f"{prefix}_manifest.json")
    if fdr.empty:
        return {
            "diagnostic_id": run["diagnostic_id"],
            "description": run["description"],
            "path": _rel(path),
            "score_transform": "control_separation",
            "background_weight": "",
            "n_completed_reruns": "",
            "n_failed_reruns": "",
            "model_rerun_all_null_status": "missing",
            "null_models_passing": "",
            "best_positive_fdr_min": "",
            "best_negative_fdr_min": "",
            "n_nulls_positive_recovered": 0,
            "n_nulls_negative_suppressed": 0,
            "n_nulls_positive_recovered_negative_suppressed": 0,
            "diagnostic_interpretation": "missing_control_separation_fdr",
        }
    passing = fdr["separation_status"].astype(str).eq("positive_separated_from_negative")
    observed_delta = (
        float(observed["positive_minus_negative"].iloc[0])
        if not observed.empty and "positive_minus_negative" in observed
        else ""
    )
    return {
        "diagnostic_id": run["diagnostic_id"],
        "description": run["description"],
        "path": _rel(path),
        "score_transform": "control_separation",
        "background_weight": "",
        "n_completed_reruns": manifest.get("n_completed_reruns", ""),
        "n_failed_reruns": "",
        "model_rerun_all_null_status": (
            "control_separation_success" if bool(passing.all()) else "not_strict_all_null_success"
        ),
        "null_models_passing": ";".join(fdr.loc[passing, "null_model"].astype(str).tolist()),
        "best_positive_fdr_min": _min_numeric(fdr, "bh_fdr"),
        "best_negative_fdr_min": "",
        "min_completed_per_null": _min_numeric(fdr, "n_completed_permutations"),
        "min_possible_raw_p": _min_possible_raw_p_from_n(_min_numeric(fdr, "n_completed_permutations")),
        "n_nulls_positive_recovered": int(passing.sum()),
        "n_nulls_negative_suppressed": "",
        "n_nulls_positive_recovered_negative_suppressed": int(passing.sum()),
        "control_statuses": ";".join(fdr["separation_status"].astype(str).tolist()),
        "observed_positive_minus_negative": observed_delta,
        "diagnostic_interpretation": (
            "positive_negative_separation_under_all_nulls"
            if bool(passing.all())
            else "positive_negative_separation_not_significant"
        ),
    }


def _interpret(positive_sig: pd.Series, negative_sig: pd.Series) -> str:
    if bool((positive_sig & ~negative_sig).all()):
        return "strict_candidate"
    if bool((~positive_sig & ~negative_sig).all()):
        return "conservative_negative_suppressed_but_no_positive_recovery"
    if bool((positive_sig & negative_sig).any()):
        return "positive_recovery_with_negative_control_leakage"
    return "mixed_no_strict_closure"


def _min_possible_raw_p(frame: pd.DataFrame) -> float | str:
    if frame.empty or "n_completed_permutations_min" not in frame:
        return ""
    values = pd.to_numeric(frame["n_completed_permutations_min"], errors="coerce").dropna()
    if values.empty:
        return ""
    n_perm = float(values.min())
    return 1.0 / (n_perm + 1.0)


def _min_possible_raw_p_from_n(n_perm: float | str) -> float | str:
    try:
        value = float(n_perm)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""
    return 1.0 / (value + 1.0)


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _min_numeric(frame: pd.DataFrame, column: str) -> float | str:
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
