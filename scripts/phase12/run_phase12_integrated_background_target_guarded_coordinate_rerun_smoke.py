#!/usr/bin/env python
"""Coordinate-rerun smoke for the target-aware background-guard candidate.

This extends the integrated coordinate-null smoke to the target-aware model path
that consumes the broad/background target panel during training. Coordinates are
permuted, candidate distances are refreshed, SpatialSelf is retrained, and the
integrated LR ranking is rebuilt for each null run.

This is a low-permutation execution and diagnostics check, not formal
whole-atlas FDR closure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
from scripts.phase7 import run_goal5_model_rerun_fdr as rerun  # noqa: E402
import scripts.phase12.run_phase12_integrated_guarded_spatial_training_atlas as integrated  # noqa: E402
from scripts.phase12.run_phase12_integrated_guarded_spatial_coordinate_rerun_smoke import (  # noqa: E402
    build_lr_fdr,
    load_resume_state,
    normalize_candidate_schema,
    rel,
    remove_empty_dir,
    summarize_smoke,
    write_status_md,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
INTEGRATED_DIR = PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training"
TARGET_TRAINING_DIR = PHASE12_DIR / "integrated_background_guard_training_targets"
TARGET_AWARE_DIR = PHASE12_DIR / "integrated_background_target_guarded_training_smoke"
OUT_DIR = PHASE12_DIR / "integrated_background_target_guarded_coordinate_rerun_smoke"
DATASET_KEY = "post_xenium_colon_full"
NULL_MODEL = "spatial_coordinate_permutation"
TARGET_PANEL = TARGET_TRAINING_DIR / "phase12_integrated_background_guard_target_panel.csv"
OBSERVED_RANKING = TARGET_AWARE_DIR / "phase12_integrated_background_target_guarded_lr_ranking.csv"
OBSERVED_PREDICTION = (
    TARGET_AWARE_DIR
    / "predictions/post_xenium_colon_full/spatialself_wavelet_integrated_background_target_guarded/"
    / "post_xenium_colon_full__spatialself_wavelet_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--k-nn", type=int, default=18)
    parser.add_argument("--seed", type=int, default=121617)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--keep-null-inputs", action="store_true")
    parser.add_argument("--drop-null-predictions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.75,
        help="Stop before starting a new permutation if the output filesystem has less free space.",
    )
    args = parser.parse_args()

    replicate_dir = INTEGRATED_DIR / "replicates" / DATASET_KEY
    output_dir = Path(args.output_dir) if args.output_dir else OUT_DIR / DATASET_KEY
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_path = output_dir / f"{DATASET_KEY}_target_guarded_coordinate_observed_lr_ranking.csv"
    null_path = output_dir / f"{DATASET_KEY}_target_guarded_coordinate_null_lr_ranking.csv"
    fdr_path = output_dir / f"{DATASET_KEY}_target_guarded_coordinate_lr_fdr.csv"
    summary_path = output_dir / f"{DATASET_KEY}_target_guarded_coordinate_summary.csv"
    status_path = output_dir / f"{DATASET_KEY}_target_guarded_coordinate_status.csv"

    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    observed_ranking = load_observed_target_ranking(replicate_dir)

    existing_null, existing_status, completed_permutations = load_resume_state(
        null_path,
        status_path,
        resume=bool(args.resume),
    )
    completed_before = len(completed_permutations)
    skipped_permutations = sorted(int(x) for x in completed_permutations if int(x) < int(args.n_permutations))
    null_rankings: list[pd.DataFrame] = [existing_null] if not existing_null.empty else []
    status_rows: list[dict[str, Any]] = existing_status.to_dict("records") if not existing_status.empty else []
    rng = np.random.default_rng(int(args.seed))
    permutation_seeds = rng.integers(1, np.iinfo(np.int32).max, size=int(args.n_permutations))

    def flush_checkpoint() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        null_ranking = pd.concat(null_rankings, ignore_index=True) if null_rankings else pd.DataFrame()
        status = pd.DataFrame(status_rows)
        lr_fdr = build_lr_fdr(observed_ranking, null_ranking)
        summary = summarize_smoke(DATASET_KEY, "TGFB1-TGFBR2", lr_fdr, status)
        observed_ranking.to_csv(observed_path, index=False)
        null_ranking.to_csv(null_path, index=False)
        lr_fdr.to_csv(fdr_path, index=False)
        summary.to_csv(summary_path, index=False)
        status.to_csv(status_path, index=False)
        return null_ranking, status, lr_fdr, summary

    for perm_idx in range(int(args.n_permutations)):
        if perm_idx in completed_permutations:
            continue
        perm_seed = int(permutation_seeds[perm_idx])
        null_input_dir = output_dir / "null_inputs" / NULL_MODEL / f"perm_{perm_idx:04d}"
        null_pred_dir = output_dir / "null_predictions" / NULL_MODEL / f"perm_{perm_idx:04d}"
        free_gb = shutil.disk_usage(output_dir).free / (1024**3)
        if free_gb < float(args.min_free_gb):
            status_rows.append(
                {
                    "dataset_key": DATASET_KEY,
                    "null_model": NULL_MODEL,
                    "permutation": int(perm_idx),
                    "seed": int(perm_seed),
                    "status": "skipped_low_disk",
                    "prediction_path": "",
                    "n_candidates": 0,
                    "n_background_guard_target_train": 0,
                    "detail": f"free_gb={free_gb:.3f} below min_free_gb={float(args.min_free_gb):.3f}",
                }
            )
            flush_checkpoint()
            break
        try:
            rerun._write_null_replicate(
                null_input_dir,
                cells,
                expression,
                lr_pairs,
                candidates,
                null_model=NULL_MODEL,
                seed=perm_seed,
            )
            metrics = run_spatialself_on_replicate(
                null_input_dir,
                null_pred_dir,
                model_name="spatialself_wavelet",
                k_nn=int(args.k_nn),
                epochs=int(args.epochs),
                positive_quantile=0.9,
                negative_quantile=0.2,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.20,
                weak_positive_label=0.82,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.24,
                ranking_margin=1.10,
                feasibility_loss_weight=0.14,
                hard_negative_loss_weight=0.20,
                context_contrastive_loss_weight=0.18,
                context_contrastive_margin=0.62,
                context_contrastive_negative_mode="hard_receiver",
                lr_identity_contrastive_loss_weight=0.10,
                lr_identity_contrastive_margin=0.50,
                control_contrastive_loss_weight=0.18,
                control_contrastive_margin=0.62,
                null_candidate_loss_weight=0.10,
                spatial_null_contrastive_loss_weight=0.16,
                spatial_null_contrastive_margin=0.62,
                spatial_null_negative_mode="range_violation_receiver",
                background_guard_loss_weight=0.36,
                background_guard_target_path=TARGET_PANEL,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=int(args.max_train_samples),
                seed=perm_seed,
            )
            pred_path = Path(metrics["prediction_path"])
            normalize_candidate_schema(null_input_dir / "candidate_interactions.csv")
            ranking = integrated.build_integrated_ranking(DATASET_KEY, null_input_dir, pred_path)
            ranking["null_model"] = NULL_MODEL
            ranking["permutation"] = int(perm_idx)
            ranking["seed"] = int(perm_seed)
            ranking["prediction_path"] = "" if args.drop_null_predictions else rel(pred_path)
            null_rankings.append(ranking)
            status_rows.append(
                {
                    "dataset_key": DATASET_KEY,
                    "null_model": NULL_MODEL,
                    "permutation": int(perm_idx),
                    "seed": int(perm_seed),
                    "status": "completed",
                    "prediction_path": "" if args.drop_null_predictions else rel(pred_path),
                    "n_candidates": int(metrics.get("n_candidates", len(candidates))),
                    "n_background_guard_target_train": int(metrics.get("n_background_guard_target_train", 0)),
                    "detail": "null prediction directory dropped after ranking summary"
                    if args.drop_null_predictions
                    else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "dataset_key": DATASET_KEY,
                    "null_model": NULL_MODEL,
                    "permutation": int(perm_idx),
                    "seed": int(perm_seed),
                    "status": "failed",
                    "prediction_path": "",
                    "n_candidates": 0,
                    "n_background_guard_target_train": 0,
                    "detail": repr(exc),
                }
            )
        finally:
            if not args.keep_null_inputs and null_input_dir.exists():
                shutil.rmtree(null_input_dir)
            if args.drop_null_predictions and null_pred_dir.exists():
                shutil.rmtree(null_pred_dir)
            flush_checkpoint()

    null_ranking, status, lr_fdr, summary = flush_checkpoint()
    completed_after = int(status["status"].eq("completed").sum()) if not status.empty else 0

    if not args.keep_null_inputs:
        remove_empty_dir(output_dir / "null_inputs" / NULL_MODEL)
        remove_empty_dir(output_dir / "null_inputs")
    if args.drop_null_predictions:
        remove_empty_dir(output_dir / "null_predictions" / NULL_MODEL)
        remove_empty_dir(output_dir / "null_predictions")

    manifest = {
        "task": "phase12_integrated_background_target_guarded_coordinate_rerun_smoke",
        "dataset": DATASET_KEY,
        "target_lr": "TGFB1-TGFBR2",
        "replicate_dir": rel(replicate_dir),
        "target_panel": rel(TARGET_PANEL),
        "observed_ranking": rel(OBSERVED_RANKING),
        "observed_prediction": rel(OBSERVED_PREDICTION),
        "null_model": NULL_MODEL,
        "n_permutations_requested": int(args.n_permutations),
        "n_permutations_completed": completed_after,
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "min_free_gb": float(args.min_free_gb),
        "drop_null_predictions": bool(args.drop_null_predictions),
        "resume": bool(args.resume),
        "n_completed_before_resume": int(completed_before),
        "skipped_completed_permutations": skipped_permutations,
        "n_completed_after_run": completed_after,
        "n_new_completed_this_run": int(max(completed_after - completed_before, 0)),
        "claim_boundary": (
            "End-to-end coordinate-rerun smoke for the target-aware background-guard candidate. "
            "Low permutation count gives coarse p-values; this is not formal whole-atlas FDR."
        ),
        "outputs": {
            "observed_lr_ranking": rel(observed_path),
            "null_lr_ranking": rel(null_path),
            "lr_fdr": rel(fdr_path),
            "summary": rel(summary_path),
            "status": rel(status_path),
        },
    }
    (output_dir / f"{DATASET_KEY}_target_guarded_coordinate_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_status_md(output_dir, DATASET_KEY, summary, manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def load_observed_target_ranking(replicate_dir: Path) -> pd.DataFrame:
    if OBSERVED_RANKING.exists():
        ranking = pd.read_csv(OBSERVED_RANKING)
        ranking = ranking[ranking["dataset_key"].astype(str).eq(DATASET_KEY)].copy()
        if not ranking.empty:
            return ranking.reset_index(drop=True)
    if OBSERVED_PREDICTION.exists():
        return integrated.build_integrated_ranking(DATASET_KEY, replicate_dir, OBSERVED_PREDICTION)
    raise FileNotFoundError(
        "Target-aware observed ranking/prediction is missing. Run "
        "scripts/phase12/run_phase12_integrated_background_target_guarded_smoke.py first."
    )


if __name__ == "__main__":
    main()
