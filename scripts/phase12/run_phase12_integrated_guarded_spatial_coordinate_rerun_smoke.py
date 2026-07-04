#!/usr/bin/env python
"""End-to-end coordinate-rerun smoke for the integrated guarded spatial candidate.

This diagnostic shuffles coordinates, refreshes candidate metadata, retrains the
same integrated guarded SpatialSelf candidate with a small epoch budget, reruns
the guarded spatial LR calibration, and compares observed LR scores with the
retrained coordinate-null scores.

The output is intentionally labelled smoke: low permutation counts cannot close
whole-atlas FDR.
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


MODEL_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded_integrated_guarded_spatial_training"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/integrated_guarded_spatial_coordinate_rerun_smoke"
NULL_MODEL = "spatial_coordinate_permutation"
OBSERVED_RANKING_PATH = MODEL_DIR / "phase12_integrated_guarded_spatial_training_lr_ranking.csv"

DATASET_SPECS = {
    "post_xenium_colon_full": {
        "replicate_dir": MODEL_DIR / "replicates/post_xenium_colon_full",
        "observed_prediction": MODEL_DIR
        / "predictions/post_xenium_colon_full/spatialself_wavelet_edge_prior_integrated_guarded_spatial/"
        "post_xenium_colon_full__spatialself_wavelet_predictions.csv",
        "target_lr": "TGFB1-TGFBR2",
    },
    "human_lymph_node_visium_raw_full": {
        "replicate_dir": MODEL_DIR / "replicates/human_lymph_node_visium_raw_full",
        "observed_prediction": MODEL_DIR
        / "predictions/human_lymph_node_visium_raw_full/spatialself_wavelet_edge_prior_integrated_guarded_spatial/"
        "human_lymph_node_visium_raw_full__spatialself_wavelet_predictions.csv",
        "target_lr": "CCL19-CCR7",
    },
    "visium_hd_crc_full": {
        "replicate_dir": MODEL_DIR / "replicates/visium_hd_crc_full",
        "observed_prediction": MODEL_DIR
        / "predictions/visium_hd_crc_full/spatialself_wavelet_edge_prior_integrated_guarded_spatial/"
        "visium_hd_crc_full__spatialself_wavelet_predictions.csv",
        "target_lr": "SPP1-CD44",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="post_xenium_colon_full")
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--k-nn", type=int, default=18)
    parser.add_argument("--seed", type=int, default=121260)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--keep-null-inputs", action="store_true")
    parser.add_argument(
        "--drop-null-predictions",
        action="store_true",
        help="Delete each null prediction directory after its LR ranking has been summarized.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed permutations already present in the output directory and run only missing indices.",
    )
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    replicate_dir = Path(spec["replicate_dir"])
    observed_prediction = Path(spec["observed_prediction"])
    output_dir = Path(args.output_dir) if args.output_dir else OUT_DIR / args.dataset
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_path = output_dir / f"{args.dataset}_integrated_coordinate_observed_lr_ranking.csv"
    null_path = output_dir / f"{args.dataset}_integrated_coordinate_null_lr_ranking.csv"
    fdr_path = output_dir / f"{args.dataset}_integrated_coordinate_lr_fdr.csv"
    summary_path = output_dir / f"{args.dataset}_integrated_coordinate_summary.csv"
    status_path = output_dir / f"{args.dataset}_integrated_coordinate_status.csv"

    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")

    observed_ranking = load_observed_ranking(args.dataset, replicate_dir, observed_prediction)

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
    for perm_idx in range(int(args.n_permutations)):
        if perm_idx in completed_permutations:
            continue
        perm_seed = int(permutation_seeds[perm_idx])
        null_input_dir = output_dir / "null_inputs" / NULL_MODEL / f"perm_{perm_idx:04d}"
        null_pred_dir = output_dir / "null_predictions" / NULL_MODEL / f"perm_{perm_idx:04d}"
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
                background_guard_loss_weight=0.28,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=int(args.max_train_samples),
                seed=perm_seed,
            )
            pred_path = Path(metrics["prediction_path"])
            normalize_candidate_schema(null_input_dir / "candidate_interactions.csv")
            ranking = integrated.build_integrated_ranking(args.dataset, null_input_dir, pred_path)
            ranking["null_model"] = NULL_MODEL
            ranking["permutation"] = int(perm_idx)
            ranking["seed"] = int(perm_seed)
            ranking["prediction_path"] = "" if args.drop_null_predictions else rel(pred_path)
            null_rankings.append(ranking)
            status_rows.append(
                {
                    "dataset_key": args.dataset,
                    "null_model": NULL_MODEL,
                    "permutation": int(perm_idx),
                    "seed": int(perm_seed),
                    "status": "completed",
                    "prediction_path": "" if args.drop_null_predictions else rel(pred_path),
                    "n_candidates": int(metrics.get("n_candidates", len(candidates))),
                    "detail": "null prediction directory dropped after ranking summary"
                    if args.drop_null_predictions
                    else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "dataset_key": args.dataset,
                    "null_model": NULL_MODEL,
                    "permutation": int(perm_idx),
                    "seed": int(perm_seed),
                    "status": "failed",
                    "prediction_path": "",
                    "n_candidates": 0,
                    "detail": repr(exc),
                }
            )
        finally:
            if not args.keep_null_inputs and null_input_dir.exists():
                shutil.rmtree(null_input_dir)
            if args.drop_null_predictions and null_pred_dir.exists():
                shutil.rmtree(null_pred_dir)

    null_ranking = pd.concat(null_rankings, ignore_index=True) if null_rankings else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    lr_fdr = build_lr_fdr(observed_ranking, null_ranking)
    summary = summarize_smoke(args.dataset, str(spec["target_lr"]), lr_fdr, status)
    completed_after = int(status["status"].eq("completed").sum()) if not status.empty else 0
    if not args.keep_null_inputs:
        remove_empty_dir(output_dir / "null_inputs" / NULL_MODEL)
        remove_empty_dir(output_dir / "null_inputs")
    if args.drop_null_predictions:
        remove_empty_dir(output_dir / "null_predictions" / NULL_MODEL)
        remove_empty_dir(output_dir / "null_predictions")

    observed_ranking.to_csv(observed_path, index=False)
    null_ranking.to_csv(null_path, index=False)
    lr_fdr.to_csv(fdr_path, index=False)
    summary.to_csv(summary_path, index=False)
    status.to_csv(status_path, index=False)

    manifest = {
        "task": "phase12_integrated_guarded_spatial_coordinate_rerun_smoke",
        "dataset": args.dataset,
        "target_lr": str(spec["target_lr"]),
        "replicate_dir": rel(replicate_dir),
        "observed_prediction": rel(observed_prediction),
        "null_model": NULL_MODEL,
        "n_permutations_requested": int(args.n_permutations),
        "n_permutations_completed": int(status["status"].eq("completed").sum()) if not status.empty else 0,
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "drop_null_predictions": bool(args.drop_null_predictions),
        "resume": bool(args.resume),
        "n_completed_before_resume": int(completed_before),
        "skipped_completed_permutations": skipped_permutations,
        "n_completed_after_run": int(completed_after),
        "n_new_completed_this_run": int(max(completed_after - completed_before, 0)),
        "claim_boundary": (
            "End-to-end coordinate-rerun smoke for the integrated guarded spatial candidate. "
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
    (output_dir / f"{args.dataset}_integrated_coordinate_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_status_md(output_dir, args.dataset, summary, manifest)

    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def load_observed_ranking(dataset_key: str, replicate_dir: Path, observed_prediction: Path) -> pd.DataFrame:
    if observed_prediction.exists():
        return integrated.build_integrated_ranking(dataset_key, replicate_dir, observed_prediction)
    if not OBSERVED_RANKING_PATH.exists():
        raise FileNotFoundError(
            f"Observed prediction missing ({observed_prediction}) and observed ranking file missing "
            f"({OBSERVED_RANKING_PATH})."
        )
    ranking = pd.read_csv(OBSERVED_RANKING_PATH)
    ranking = ranking[ranking["dataset_key"].astype(str).eq(str(dataset_key))].copy()
    if ranking.empty:
        raise ValueError(f"No observed ranking rows found for dataset {dataset_key} in {OBSERVED_RANKING_PATH}.")
    return ranking.reset_index(drop=True)


def load_resume_state(
    null_path: Path,
    status_path: Path,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    if not resume or not status_path.exists():
        return pd.DataFrame(), pd.DataFrame(), set()
    status = pd.read_csv(status_path)
    if status.empty or "permutation" not in status.columns:
        return pd.DataFrame(), pd.DataFrame(), set()
    status["permutation"] = pd.to_numeric(status["permutation"], errors="coerce")
    completed_status = status[
        status["status"].astype(str).eq("completed") & status["permutation"].notna()
    ].copy()
    completed_status["permutation"] = completed_status["permutation"].astype(int)
    completed = set(int(x) for x in completed_status["permutation"].tolist())
    if not completed or not null_path.exists():
        return pd.DataFrame(), completed_status, completed
    null_ranking = pd.read_csv(null_path)
    if "permutation" in null_ranking.columns:
        null_ranking["permutation"] = pd.to_numeric(null_ranking["permutation"], errors="coerce")
        null_ranking = null_ranking[null_ranking["permutation"].isin(completed)].copy()
        null_ranking["permutation"] = null_ranking["permutation"].astype(int)
    return null_ranking.reset_index(drop=True), completed_status.reset_index(drop=True), completed


def remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def build_lr_fdr(observed: pd.DataFrame, null_ranking: pd.DataFrame) -> pd.DataFrame:
    obs = observed.copy()
    obs["lr_pair"] = obs["ligand"].astype(str) + "-" + obs["receptor"].astype(str)
    obs["coordinate_rerun_p_value"] = np.nan
    obs["strict_coordinate_rerun_p_value"] = np.nan
    obs["null_mean_calibrated_score"] = np.nan
    obs["null_q95_calibrated_score"] = np.nan
    obs["null_mean_strict_score"] = np.nan
    obs["null_q95_strict_score"] = np.nan
    obs["n_completed_permutations"] = 0

    if null_ranking.empty:
        obs["coordinate_rerun_bh_fdr"] = np.nan
        obs["strict_coordinate_rerun_bh_fdr"] = np.nan
        return obs

    null = null_ranking.copy()
    for idx, row in obs.iterrows():
        sub = null[null["lr_id"].astype(str).eq(str(row["lr_id"]))]
        scores = pd.to_numeric(sub["calibrated_score"], errors="coerce").dropna().to_numpy(dtype=float)
        strict_scores = pd.to_numeric(sub["strict_discovery_score"], errors="coerce").dropna().to_numpy(dtype=float)
        obs.at[idx, "n_completed_permutations"] = int(len(scores))
        if len(scores):
            score = float(row["calibrated_score"])
            obs.at[idx, "coordinate_rerun_p_value"] = (float((scores >= score).sum()) + 1.0) / (len(scores) + 1.0)
            obs.at[idx, "null_mean_calibrated_score"] = float(np.mean(scores))
            obs.at[idx, "null_q95_calibrated_score"] = float(np.quantile(scores, 0.95))
        if len(strict_scores):
            score = float(row["strict_discovery_score"])
            obs.at[idx, "strict_coordinate_rerun_p_value"] = (
                float((strict_scores >= score).sum()) + 1.0
            ) / (len(strict_scores) + 1.0)
            obs.at[idx, "null_mean_strict_score"] = float(np.mean(strict_scores))
            obs.at[idx, "null_q95_strict_score"] = float(np.quantile(strict_scores, 0.95))
    obs["coordinate_rerun_bh_fdr"] = bh_adjust(
        pd.to_numeric(obs["coordinate_rerun_p_value"], errors="coerce").to_numpy(dtype=float)
    )
    obs["strict_coordinate_rerun_bh_fdr"] = bh_adjust(
        pd.to_numeric(obs["strict_coordinate_rerun_p_value"], errors="coerce").to_numpy(dtype=float)
    )
    return obs.sort_values(
        ["coordinate_rerun_bh_fdr", "coordinate_rerun_p_value", "calibrated_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def normalize_candidate_schema(candidate_path: Path) -> None:
    """Remove LR-pair metadata columns that are merged again during calibration."""
    candidates = pd.read_csv(candidate_path)
    drop_cols = [
        "spatial_range",
        "active_niche",
        "control_name",
        "control_class",
        "biological_category",
        "source_count",
        "selected_reason",
    ]
    existing = [col for col in drop_cols if col in candidates.columns]
    if existing:
        candidates = candidates.drop(columns=existing)
        candidates.to_csv(candidate_path, index=False)


def summarize_smoke(dataset_key: str, target_lr: str, fdr: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if fdr.empty:
        return pd.DataFrame(
            [
                {
                    "dataset_key": dataset_key,
                    "status": "no_fdr_rows",
                    "n_completed_permutations": int(status["status"].eq("completed").sum()) if not status.empty else 0,
                }
            ]
        )
    positive = fdr["true_lr"].fillna(False).astype(bool)
    negative = fdr["false_positive_high_expression"].fillna(False).astype(bool)
    target = fdr[fdr["lr_pair"].astype(str).eq(target_lr)]
    best_positive = fdr[positive].sort_values(["coordinate_rerun_p_value", "calibrated_score"], ascending=[True, False])
    best_negative = fdr[negative].sort_values(["coordinate_rerun_p_value", "calibrated_score"], ascending=[True, False])
    completed = int(status["status"].eq("completed").sum()) if not status.empty else 0
    rows = [
        {
            "dataset_key": dataset_key,
            "status": "completed_smoke" if completed else "no_completed_permutations",
            "n_completed_permutations": completed,
            "target_lr": target_lr,
            "target_rank": value(target, "rank"),
            "target_score": value(target, "calibrated_score"),
            "target_null_q95": value(target, "null_q95_calibrated_score"),
            "target_p": value(target, "coordinate_rerun_p_value"),
            "target_bh_fdr": value(target, "coordinate_rerun_bh_fdr"),
            "target_strict_p": value(target, "strict_coordinate_rerun_p_value"),
            "target_strict_bh_fdr": value(target, "strict_coordinate_rerun_bh_fdr"),
            "best_positive_lr": lr_name(best_positive),
            "best_positive_p": value(best_positive, "coordinate_rerun_p_value"),
            "best_positive_bh_fdr": value(best_positive, "coordinate_rerun_bh_fdr"),
            "best_negative_lr": lr_name(best_negative),
            "best_negative_p": value(best_negative, "coordinate_rerun_p_value"),
            "best_negative_bh_fdr": value(best_negative, "coordinate_rerun_bh_fdr"),
            "n_positive_p_min": int((pd.to_numeric(fdr.loc[positive, "coordinate_rerun_p_value"], errors="coerce") <= (1.0 / (completed + 1.0))).sum()) if completed else 0,
            "n_negative_p_min": int((pd.to_numeric(fdr.loc[negative, "coordinate_rerun_p_value"], errors="coerce") <= (1.0 / (completed + 1.0))).sum()) if completed else 0,
            "claim_boundary": "smoke only; p-value resolution is limited by completed coordinate reruns",
        }
    ]
    return pd.DataFrame(rows)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / float(i + 1))
        adjusted[i] = running
    out[idx[order]] = np.clip(adjusted, 0.0, 1.0)
    return out


def lr_name(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    return str(table.iloc[0]["lr_pair"])


def value(table: pd.DataFrame, column: str) -> float | int | str:
    if table.empty or column not in table:
        return np.nan
    return table.iloc[0][column]


def write_status_md(output_dir: Path, dataset_key: str, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Phase 12 Integrated Guarded Spatial Coordinate-Rerun Smoke",
        "",
        "## Scope",
        "",
        "Coordinates are permuted, candidate metadata are refreshed, SpatialSelf is retrained, and the integrated guarded spatial LR calibration is rerun.",
        "",
        "## Result",
        "",
        f"- Dataset: `{dataset_key}`",
        f"- Completed coordinate reruns: {manifest['n_permutations_completed']}/{manifest['n_permutations_requested']}",
        f"- Completed before resume: {manifest.get('n_completed_before_resume', 0)}",
        f"- New completed this run: {manifest.get('n_new_completed_this_run', manifest['n_permutations_completed'])}",
        f"- Target LR: `{row.get('target_lr', '')}` rank={row.get('target_rank', 'NA')} p={row.get('target_p', 'NA')} BH={row.get('target_bh_fdr', 'NA')}",
        f"- Best positive: `{row.get('best_positive_lr', '')}` p={row.get('best_positive_p', 'NA')} BH={row.get('best_positive_bh_fdr', 'NA')}",
        f"- Best high-expression negative: `{row.get('best_negative_lr', '')}` p={row.get('best_negative_p', 'NA')} BH={row.get('best_negative_bh_fdr', 'NA')}",
        "",
        "## Claim Boundary",
        "",
        "This is a low-permutation end-to-end smoke, not manuscript-level whole-atlas FDR.",
    ]
    (output_dir / f"{dataset_key}_integrated_coordinate_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
