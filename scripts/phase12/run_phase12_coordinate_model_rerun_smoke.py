#!/usr/bin/env python
"""Small end-to-end coordinate-permutation model-rerun smoke for Phase 12.

This is a feasibility/diagnostic layer: it perturbs cell coordinates, refreshes
candidate distances, retrains SpatialSelf, and compares LR top-tail statistics
against the observed expanded-atlas run. Low-permutation smoke output must not
be used as final whole-atlas FDR closure.
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

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
from scripts.phase7 import run_goal5_model_rerun_fdr as rerun


ATLAS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/coordinate_model_rerun_smoke"

DATASET_SPECS = {
    "post_xenium_colon_full": {
        "replicate_dir": ATLAS_DIR / "replicates/post_xenium_colon_full",
        "observed_prediction": ATLAS_DIR
        / "predictions/post_xenium_colon_full/spatialself_wavelet_edge_prior/"
        "post_xenium_colon_full__spatialself_wavelet_predictions.csv",
    },
    "human_lymph_node_visium_raw_full": {
        "replicate_dir": ATLAS_DIR / "replicates/human_lymph_node_visium_raw_full",
        "observed_prediction": ATLAS_DIR
        / "predictions/human_lymph_node_visium_raw_full/spatialself_wavelet_edge_prior/"
        "human_lymph_node_visium_raw_full__spatialself_wavelet_predictions.csv",
    },
}

NULL_MODEL = "spatial_coordinate_permutation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="human_lymph_node_visium_raw_full")
    parser.add_argument("--n-permutations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--k-nn", type=int, default=18)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--score-transform",
        choices=(
            "raw",
            "lr_mechanism_prior_rank",
            "lr_feasibility_gate_rank",
            "false_positive_suppressed_rank",
            "false_positive_feasgate_rank",
        ),
        default="false_positive_feasgate_rank",
    )
    parser.add_argument("--seed", type=int, default=121200)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--keep-null-inputs", action="store_true")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    replicate_dir = Path(spec["replicate_dir"])
    observed_prediction = Path(spec["observed_prediction"])
    output_dir = Path(args.output_dir) if args.output_dir else OUT_DIR / args.dataset
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    observed = pd.read_csv(observed_prediction)

    observed_stats = rerun._lr_stats(
        rerun._merge_scores(candidates, observed),
        lr_pairs,
        top_fraction=float(args.top_fraction),
        score_transform=str(args.score_transform),
    )

    null_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    for perm_idx in range(int(args.n_permutations)):
        perm_seed = int(rng.integers(1, np.iinfo(np.int32).max))
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
                weak_positive_fraction=0.15,
                weak_positive_label=0.75,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.18,
                feasibility_loss_weight=0.12,
                hard_negative_loss_weight=0.12,
                context_contrastive_loss_weight=0.12,
                context_contrastive_margin=0.5,
                context_contrastive_negative_mode="random_receiver",
                lr_identity_contrastive_loss_weight=0.06,
                lr_identity_contrastive_margin=0.35,
                control_contrastive_loss_weight=0.12,
                control_contrastive_margin=0.5,
                null_candidate_loss_weight=0.06,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=int(args.max_train_samples),
                seed=perm_seed,
            )
            pred_path = Path(metrics["prediction_path"])
            null_candidates = pd.read_csv(null_input_dir / "candidate_interactions.csv")
            null_pred = pd.read_csv(pred_path)
            null_stats = rerun._lr_stats(
                rerun._merge_scores(null_candidates, null_pred),
                lr_pairs,
                top_fraction=float(args.top_fraction),
                score_transform=str(args.score_transform),
            )
            for row in null_stats.to_dict("records"):
                row.update(
                    {
                        "null_model": NULL_MODEL,
                        "permutation": perm_idx,
                        "seed": perm_seed,
                        "prediction_path": rel(pred_path),
                    }
                )
                null_rows.append(row)
            status_rows.append(
                {
                    "dataset_key": args.dataset,
                    "null_model": NULL_MODEL,
                    "permutation": perm_idx,
                    "seed": perm_seed,
                    "status": "completed",
                    "prediction_path": rel(pred_path),
                    "n_candidates": int(metrics.get("n_candidates", len(candidates))),
                    "detail": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "dataset_key": args.dataset,
                    "null_model": NULL_MODEL,
                    "permutation": perm_idx,
                    "seed": perm_seed,
                    "status": "failed",
                    "prediction_path": "",
                    "n_candidates": 0,
                    "detail": repr(exc),
                }
            )
        finally:
            if not args.keep_null_inputs and null_input_dir.exists():
                shutil.rmtree(null_input_dir)

    null_stats = pd.DataFrame(null_rows)
    status = pd.DataFrame(status_rows)
    fdr = rerun._model_rerun_fdr(
        observed_stats,
        null_stats,
        lr_pairs,
        null_models=(NULL_MODEL,),
        top_fraction=float(args.top_fraction),
    )
    summary = summarize_smoke(args.dataset, fdr, status)

    observed_path = output_dir / f"{args.dataset}_coordinate_model_rerun_observed_lr_stats.csv"
    null_path = output_dir / f"{args.dataset}_coordinate_model_rerun_null_lr_stats.csv"
    fdr_path = output_dir / f"{args.dataset}_coordinate_model_rerun_lr_fdr.csv"
    status_path = output_dir / f"{args.dataset}_coordinate_model_rerun_status.csv"
    summary_path = output_dir / f"{args.dataset}_coordinate_model_rerun_summary.csv"
    observed_stats.to_csv(observed_path, index=False)
    null_stats.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    status.to_csv(status_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "task": "phase12_coordinate_model_rerun_smoke",
        "dataset": args.dataset,
        "replicate_dir": rel(replicate_dir),
        "observed_prediction": rel(observed_prediction),
        "null_model": NULL_MODEL,
        "n_permutations_requested": int(args.n_permutations),
        "n_permutations_completed": int(status["status"].eq("completed").sum()) if not status.empty else 0,
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "score_transform": str(args.score_transform),
        "top_fraction": float(args.top_fraction),
        "claim_boundary": (
            "End-to-end coordinate-permutation smoke only. The model is retrained on coordinate-shuffled inputs, "
            "but low permutation count gives coarse p-values and does not solve Phase 12 whole-atlas FDR."
        ),
        "outputs": {
            "observed_lr_stats": rel(observed_path),
            "null_lr_stats": rel(null_path),
            "lr_fdr": rel(fdr_path),
            "status": rel(status_path),
            "summary": rel(summary_path),
        },
    }
    (output_dir / f"{args.dataset}_coordinate_model_rerun_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_status_md(output_dir, args.dataset, summary, manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def summarize_smoke(dataset_key: str, fdr: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
    fdr = fdr.copy()
    positive = positive_mask(fdr)
    negative = fdr.get("false_positive_high_expression", pd.Series(False, index=fdr.index)).fillna(False).astype(bool)
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    rows.append(
        {
            "dataset_key": dataset_key,
            "status": "completed_smoke" if status["status"].eq("completed").any() else "no_completed_permutations",
            "n_completed_permutations": int(status["status"].eq("completed").sum()) if not status.empty else 0,
            "best_positive_lr": best_lr(fdr[positive]),
            "best_positive_p": best_value(fdr[positive], "model_rerun_p_value"),
            "best_positive_bh_fdr": best_value(fdr[positive], "model_rerun_bh_fdr"),
            "best_negative_lr": best_lr(fdr[negative]),
            "best_negative_p": best_value(fdr[negative], "model_rerun_p_value"),
            "best_negative_bh_fdr": best_value(fdr[negative], "model_rerun_bh_fdr"),
            "n_positive_p_le_0_1": int((pd.to_numeric(fdr.loc[positive, "model_rerun_p_value"], errors="coerce") <= 0.1).sum()),
            "n_negative_p_le_0_1": int((pd.to_numeric(fdr.loc[negative, "model_rerun_p_value"], errors="coerce") <= 0.1).sum()),
            "claim_boundary": "smoke only; p-value resolution is limited by the number of completed coordinate reruns",
        }
    )
    return pd.DataFrame(rows)


def positive_mask(table: pd.DataFrame) -> pd.Series:
    if "true_lr" in table:
        return table["true_lr"].fillna(False).astype(bool)
    if "control_class" in table:
        return table["control_class"].astype(str).str.contains("positive|anchor", case=False, na=False)
    return pd.Series(False, index=table.index)


def best_lr(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    ordered = table.sort_values(["model_rerun_p_value", "model_rerun_group_statistic"], ascending=[True, False])
    return str(ordered.iloc[0]["lr_pair"])


def best_value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return float("nan")
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def write_status_md(output_dir: Path, dataset_key: str, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Phase 12 Coordinate Model-Rerun Smoke",
        "",
        "## Scope",
        "",
        "This smoke test shuffles cell coordinates, refreshes candidate distances, retrains SpatialSelf, and compares LR top-tail statistics with the observed expanded-atlas run.",
        "",
        "## Result",
        "",
        f"- Dataset: `{dataset_key}`",
        f"- Completed coordinate reruns: {manifest['n_permutations_completed']}/{manifest['n_permutations_requested']}",
        f"- Best positive: `{row.get('best_positive_lr', '')}` p={row.get('best_positive_p', 'NA')} BH={row.get('best_positive_bh_fdr', 'NA')}",
        f"- Best high-expression negative: `{row.get('best_negative_lr', '')}` p={row.get('best_negative_p', 'NA')} BH={row.get('best_negative_bh_fdr', 'NA')}",
        "",
        "## Claim Boundary",
        "",
        "This is an end-to-end rerun feasibility smoke, not final Phase 12 FDR. With only a few permutations, p-values are coarse and should not be used as manuscript-level discovery significance.",
    ]
    (output_dir / f"{dataset_key}_coordinate_model_rerun_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
