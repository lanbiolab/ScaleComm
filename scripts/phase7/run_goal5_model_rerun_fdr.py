#!/usr/bin/env python
"""Run model-rerun null FDR for Phase 7 Goal 5 real-platform validation.

This is stricter than score shuffling: each null replicate perturbs the input
tables, retrains SpatialSelf with the same calibrated settings, and compares
observed LR-pair top-tail scores against the retrained null distribution.
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


GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

DATASET_SPECS = {
    "post_xenium_colon": {
        "dataset_id": "post_xenium_human_colon_cancer_subset",
        "dataset_dir": GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset",
        "observed_prediction": GOAL5_DIR
        / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        "output_dir": GOAL5_DIR / "post_xenium_colon_model_rerun_fdr",
    },
    "slide_tags_tonsil_n1800": {
        "dataset_id": "slide_tags_tonsil_n1800_subset",
        "dataset_dir": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/constructed_replicate",
        "observed_prediction": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/"
        "tonsil_fresh_spatialself_wavelet_predictions.csv",
        "output_dir": GOAL5_DIR / "slide_tags_tonsil_n1800_model_rerun_fdr",
    },
    "slide_tags_tonsil_n1200": {
        "dataset_id": "slide_tags_tonsil_n1200_subset",
        "dataset_dir": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/constructed_replicate",
        "observed_prediction": PROJECT_ROOT
        / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1200/"
        "tonsil_fresh_spatialself_wavelet_predictions.csv",
        "output_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr",
    },
}

NULL_MODELS = ("spatial_coordinate_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="post_xenium_colon")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--observed-prediction", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--null-model", action="append", choices=NULL_MODELS)
    parser.add_argument("--n-permutations", type=int, default=20)
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
        default="raw",
        help="Optional candidate-level score calibration applied to observed and every model-rerun null.",
    )
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--weak-positive-fraction", type=float, default=0.25)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.2)
    parser.add_argument("--feasibility-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--context-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--context-contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--context-contrastive-negative-mode",
        choices=("random_receiver", "hard_receiver"),
        default="random_receiver",
    )
    parser.add_argument("--lr-identity-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr-identity-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--control-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--control-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--null-candidate-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr-cap-positive", type=int, default=80)
    parser.add_argument("--lr-cap-negative", type=int, default=260)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument(
        "--keep-null-inputs",
        action="store_true",
        help="Keep per-permutation null input directories for debugging.",
    )
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(spec["dataset_dir"])
    observed_prediction = (
        Path(args.observed_prediction) if args.observed_prediction else Path(spec["observed_prediction"])
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(spec["output_dir"])
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    null_models = tuple(args.null_model) if args.null_model else NULL_MODELS
    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    observed = pd.read_csv(observed_prediction)

    observed_stats = _lr_stats(
        _merge_scores(candidates, observed),
        lr_pairs,
        top_fraction=float(args.top_fraction),
        score_transform=str(args.score_transform),
    )

    null_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    for null_model in null_models:
        for perm_idx in range(int(args.n_permutations)):
            perm_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            null_input_dir = output_dir / "null_inputs" / null_model / f"perm_{perm_idx:04d}"
            null_pred_dir = output_dir / "null_predictions" / null_model / f"perm_{perm_idx:04d}"
            try:
                _write_null_replicate(
                    null_input_dir,
                    cells,
                    expression,
                    lr_pairs,
                    candidates,
                    null_model=null_model,
                    seed=perm_seed,
                )
                metrics = run_spatialself_on_replicate(
                    null_input_dir,
                    null_pred_dir,
                    model_name="spatialself_wavelet",
                    k_nn=int(args.k_nn),
                    epochs=int(args.epochs),
                    max_train_samples=int(args.max_train_samples),
                    pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                    weak_positive_fraction=float(args.weak_positive_fraction),
                    ranking_loss_weight=float(args.ranking_loss_weight),
                    feasibility_loss_weight=float(args.feasibility_loss_weight),
                    hard_negative_loss_weight=float(args.hard_negative_loss_weight),
                    context_contrastive_loss_weight=float(args.context_contrastive_loss_weight),
                    context_contrastive_margin=float(args.context_contrastive_margin),
                    context_contrastive_negative_mode=str(args.context_contrastive_negative_mode),
                    lr_identity_contrastive_loss_weight=float(args.lr_identity_contrastive_loss_weight),
                    lr_identity_contrastive_margin=float(args.lr_identity_contrastive_margin),
                    control_contrastive_loss_weight=float(args.control_contrastive_loss_weight),
                    control_contrastive_margin=float(args.control_contrastive_margin),
                    null_candidate_loss_weight=float(args.null_candidate_loss_weight),
                    lr_cap_positive=int(args.lr_cap_positive),
                    lr_cap_negative=int(args.lr_cap_negative),
                    use_edge_prior_features=True,
                    seed=perm_seed,
                )
                pred_path = Path(metrics["prediction_path"])
                null_candidates = pd.read_csv(null_input_dir / "candidate_interactions.csv")
                null_pred = pd.read_csv(pred_path)
                stats = _lr_stats(
                    _merge_scores(null_candidates, null_pred),
                    lr_pairs,
                    top_fraction=float(args.top_fraction),
                    score_transform=str(args.score_transform),
                )
                for row in stats.to_dict("records"):
                    row.update(
                        {
                            "null_model": null_model,
                            "permutation": perm_idx,
                            "seed": perm_seed,
                            "prediction_path": _rel(pred_path),
                        }
                    )
                    null_rows.append(row)
                status_rows.append(
                    {
                        "null_model": null_model,
                        "permutation": perm_idx,
                        "seed": perm_seed,
                        "status": "completed",
                        "detail": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - status capture for long jobs.
                status_rows.append(
                    {
                        "null_model": null_model,
                        "permutation": perm_idx,
                        "seed": perm_seed,
                        "status": "failed",
                        "detail": repr(exc),
                    }
                )
            finally:
                if not args.keep_null_inputs and null_input_dir.exists():
                    shutil.rmtree(null_input_dir)

    null_stats = pd.DataFrame(null_rows)
    status = pd.DataFrame(status_rows)
    fdr = _model_rerun_fdr(
        observed_stats,
        null_stats,
        lr_pairs,
        null_models=null_models,
        top_fraction=float(args.top_fraction),
    )
    control = _control_summary(fdr, alpha=float(args.alpha))
    audit = _audit(control, status, null_models=null_models, alpha=float(args.alpha))

    observed_path = output_dir / f"{args.dataset}_model_rerun_observed_lr_stats.csv"
    null_path = output_dir / f"{args.dataset}_model_rerun_null_lr_stats.csv"
    fdr_path = output_dir / f"{args.dataset}_model_rerun_lr_pair_fdr.csv"
    control_path = output_dir / f"{args.dataset}_model_rerun_control_fdr_summary.csv"
    status_path = output_dir / f"{args.dataset}_model_rerun_status.csv"
    audit_path = output_dir / f"{args.dataset}_model_rerun_fdr_audit.csv"
    observed_stats.to_csv(observed_path, index=False)
    null_stats.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    control.to_csv(control_path, index=False)
    status.to_csv(status_path, index=False)
    audit.to_csv(audit_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "dataset_dir": _rel(dataset_dir),
        "observed_prediction": _rel(observed_prediction),
        "output_dir": _rel(output_dir),
        "null_models": list(null_models),
        "n_permutations_requested_per_null": int(args.n_permutations),
        "n_permutations_completed": int(status["status"].eq("completed").sum()) if not status.empty else 0,
        "top_fraction": float(args.top_fraction),
        "score_transform": str(args.score_transform),
        "alpha": float(args.alpha),
        "spatialself_settings": {
            "model_name": "spatialself_wavelet",
            "use_edge_prior_features": True,
            "pseudo_label_strategy": "lr_prior_lr_capped_mechanism_niche_expression_distance",
            "epochs": int(args.epochs),
            "max_train_samples": int(args.max_train_samples),
            "k_nn": int(args.k_nn),
            "weak_positive_fraction": float(args.weak_positive_fraction),
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "feasibility_loss_weight": float(args.feasibility_loss_weight),
            "hard_negative_loss_weight": float(args.hard_negative_loss_weight),
            "context_contrastive_loss_weight": float(args.context_contrastive_loss_weight),
            "context_contrastive_margin": float(args.context_contrastive_margin),
            "context_contrastive_negative_mode": str(args.context_contrastive_negative_mode),
            "lr_identity_contrastive_loss_weight": float(args.lr_identity_contrastive_loss_weight),
            "lr_identity_contrastive_margin": float(args.lr_identity_contrastive_margin),
            "control_contrastive_loss_weight": float(args.control_contrastive_loss_weight),
            "control_contrastive_margin": float(args.control_contrastive_margin),
            "null_candidate_loss_weight": float(args.null_candidate_loss_weight),
            "lr_cap_positive": int(args.lr_cap_positive),
            "lr_cap_negative": int(args.lr_cap_negative),
        },
        "strict_model_rerun_success": bool(
            audit.get("model_rerun_all_null_status", pd.Series(dtype=str)).eq(
                "positive_significant_negative_suppressed_all_nulls"
            ).any()
        ),
        "claim_boundary": (
            "Model-rerun FDR retrains SpatialSelf on perturbed real-platform inputs. It is closer to a strict "
            "spatial/receiver/LR-identity null than score shuffling, but p-value resolution depends on the number "
            "of completed reruns; low-permutation smoke runs should not be reported as final strict FDR closure."
        ),
        "outputs": {
            "observed_lr_stats": _rel(observed_path),
            "null_lr_stats": _rel(null_path),
            "lr_pair_fdr": _rel(fdr_path),
            "control_summary": _rel(control_path),
            "status": _rel(status_path),
            "audit": _rel(audit_path),
        },
    }
    manifest_path = output_dir / f"{args.dataset}_model_rerun_fdr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(control.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _write_null_replicate(
    output_dir: Path,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    null_model: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    null_cells = cells.copy()
    null_candidates = candidates.copy()

    if null_model == "spatial_coordinate_permutation":
        perm = rng.permutation(len(null_cells))
        null_cells[["x", "y"]] = null_cells[["x", "y"]].to_numpy(dtype=float)[perm]
    elif null_model == "receiver_permutation":
        null_candidates["receiver_id"] = rng.permutation(null_candidates["receiver_id"].to_numpy())
    elif null_model == "lr_identity_permutation":
        shuffled_lr = rng.permutation(null_candidates["lr_id"].to_numpy())
        null_candidates["lr_id"] = shuffled_lr
    else:
        raise ValueError(f"Unsupported null model: {null_model}")

    null_candidates = _refresh_candidate_metadata(null_candidates, null_cells, expression, lr_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    null_cells.to_csv(output_dir / "cells.csv", index=False)
    expression.to_csv(output_dir / "expression.csv", index=False)
    lr_pairs.to_csv(output_dir / "lr_pairs.csv", index=False)
    null_candidates.to_csv(output_dir / "candidate_interactions.csv", index=False)


def _refresh_candidate_metadata(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.copy()
    lr_meta_cols = [
        column
        for column in [
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
            "spatial_range",
            "active_niche",
            "false_positive_high_expression",
            "true_lr",
        ]
        if column in lr_pairs.columns
    ]
    keep = [column for column in result.columns if column not in set(lr_meta_cols) - {"lr_id"}]
    result = result[keep].merge(lr_pairs[lr_meta_cols], on="lr_id", how="left", validate="many_to_one")

    cell_index = cells.set_index("cell_id")
    expr_index = expression.set_index("cell_id")
    sender = cell_index.loc[result["sender_id"]]
    receiver = cell_index.loc[result["receiver_id"]]
    result["sender_cell_type"] = sender["cell_type"].to_numpy()
    result["receiver_cell_type"] = receiver["cell_type"].to_numpy()
    result["sender_niche"] = sender["niche"].to_numpy()
    result["receiver_niche"] = receiver["niche"].to_numpy()
    sx = sender["x"].to_numpy(dtype=float)
    sy = sender["y"].to_numpy(dtype=float)
    rx = receiver["x"].to_numpy(dtype=float)
    ry = receiver["y"].to_numpy(dtype=float)
    result["distance"] = np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2)

    result["ligand_expr"] = _lookup_expression(expr_index, result["sender_id"], result["ligand"])
    result["receptor_expr"] = _lookup_expression(expr_index, result["receiver_id"], result["receptor"])
    if "type_rule_satisfied" not in result:
        result["type_rule_satisfied"] = True
    result["range_rule_satisfied"] = result["distance"].to_numpy(dtype=float) <= pd.to_numeric(
        result["spatial_range"], errors="coerce"
    ).fillna(np.inf).to_numpy(dtype=float)
    result["niche_rule_satisfied"] = _niche_rule(result)
    expression_score = np.log1p(result["ligand_expr"].to_numpy(dtype=float)) * np.log1p(
        result["receptor_expr"].to_numpy(dtype=float)
    )
    bandwidth = max(float(np.quantile(result["distance"].to_numpy(dtype=float), 0.5)), 1e-6)
    result["coexpression_distance_score"] = expression_score * np.exp(
        -result["distance"].to_numpy(dtype=float) / bandwidth
    )
    if "weak_signal" not in result:
        result["weak_signal"] = False
    return result


def _lookup_expression(expr_index: pd.DataFrame, cell_ids: pd.Series, genes: pd.Series) -> np.ndarray:
    values = np.zeros(len(cell_ids), dtype=float)
    for gene in sorted(set(genes.astype(str))):
        mask = genes.astype(str).eq(gene).to_numpy()
        if gene in expr_index.columns:
            values[mask] = expr_index.loc[cell_ids[mask], gene].to_numpy(dtype=float)
    return values


def _niche_rule(candidates: pd.DataFrame) -> np.ndarray:
    active = candidates.get("active_niche", pd.Series(["any"] * len(candidates))).astype(str).to_numpy()
    sender = candidates["sender_niche"].astype(str).to_numpy()
    receiver = candidates["receiver_niche"].astype(str).to_numpy()
    unrestricted = (active == "any") | (active == "") | (active == "nan")
    return unrestricted | ((sender == active) & (receiver == active))


def _merge_scores(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions.")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _lr_stats(
    merged: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    top_fraction: float,
    score_transform: str,
) -> pd.DataFrame:
    merged = _apply_score_transform(merged, transform=score_transform)
    rows = []
    for lr_id, group in merged.groupby("lr_id", sort=False):
        scores = group["score"].to_numpy(dtype=float)
        rows.append(
            {
                "lr_id": lr_id,
                "observed_or_null_statistic": _top_tail_mean(scores, top_fraction=top_fraction),
                "mean_score": float(np.mean(scores)) if len(scores) else np.nan,
                "max_score": float(np.max(scores)) if len(scores) else np.nan,
                "n_candidate_edges": int(len(scores)),
                "top_fraction": float(top_fraction),
            }
        )
    stats = pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")
    return stats.sort_values("observed_or_null_statistic", ascending=False)


def _top_tail_mean(scores: np.ndarray, *, top_fraction: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float("nan")
    n_top = max(int(np.ceil(len(scores) * float(top_fraction))), 1)
    return float(np.sort(scores)[-n_top:].mean())


def _apply_score_transform(merged: pd.DataFrame, *, transform: str) -> pd.DataFrame:
    if transform == "raw":
        return merged
    if transform not in {
        "lr_mechanism_prior_rank",
        "lr_feasibility_gate_rank",
        "false_positive_suppressed_rank",
        "false_positive_feasgate_rank",
    }:
        raise ValueError(f"Unsupported score transform: {transform}")
    out = merged.copy()
    calibrated = np.zeros(len(out), dtype=float)
    score = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for _, idx_values in out.groupby("lr_id", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        rank = _percentile_rank(score[idx])
        prior = _mechanism_distance_prior(out.iloc[idx], use_oracle_rules=True)
        if transform in {"lr_feasibility_gate_rank", "false_positive_feasgate_rank"}:
            prior = prior * _expression_feasibility_prior(out.iloc[idx])
        if transform in {"false_positive_suppressed_rank", "false_positive_feasgate_rank"}:
            prior = prior * _false_positive_high_expression_prior(out.iloc[idx])
        calibrated[idx] = rank * prior
    out["score"] = calibrated
    return out


def _mechanism_distance_prior(table: pd.DataFrame, *, use_oracle_rules: bool) -> np.ndarray:
    prior = np.ones(len(table), dtype=float)
    if use_oracle_rules:
        for column in ("type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied"):
            if column in table.columns:
                prior *= np.where(table[column].astype(bool).to_numpy(), 1.0, 0.25)
    if "mechanism" in table.columns and "distance" in table.columns:
        distance = table["distance"].to_numpy(dtype=float)
        mechanism = table["mechanism"].astype(str).to_numpy()
        for value in np.unique(mechanism):
            mask = mechanism == value
            if not mask.any():
                continue
            scale = max(float(np.quantile(distance[mask], 0.75)), 1e-6)
            if value == "long_range_chemokine_niche":
                ranked_distance = _percentile_rank(distance[mask])
                prior[mask] *= 0.45 + 0.55 * ranked_distance
            else:
                prior[mask] *= np.exp(-distance[mask] / scale)
    return np.clip(prior, 0.0, 1.0)


def _expression_feasibility_prior(table: pd.DataFrame) -> np.ndarray:
    prior = np.ones(len(table), dtype=float)
    if {"ligand_expr", "receptor_expr"}.issubset(table.columns):
        ligand = table["ligand_expr"].to_numpy(dtype=float)
        receptor = table["receptor_expr"].to_numpy(dtype=float)
        both_present = (ligand > 0) & (receptor > 0)
        one_missing = ((ligand > 0) ^ (receptor > 0))
        prior *= np.where(both_present, 1.0, np.where(one_missing, 0.10, 0.02))
        expression_strength = np.log1p(ligand) * np.log1p(receptor)
        if np.any(expression_strength > 0):
            prior *= 0.25 + 0.75 * _percentile_rank(expression_strength)
    return np.clip(prior, 0.0, 1.0)


def _false_positive_high_expression_prior(table: pd.DataFrame) -> np.ndarray:
    prior = np.ones(len(table), dtype=float)
    if "false_positive_high_expression" in table.columns:
        false_positive = table["false_positive_high_expression"].fillna(False).astype(bool).to_numpy()
        prior *= np.where(false_positive, 0.02, 1.0)
    if "control_class" in table.columns:
        housekeeping = table["control_class"].astype(str).str.contains(
            "housekeeping|negative",
            case=False,
            na=False,
        ).to_numpy()
        prior *= np.where(housekeeping, 0.02, 1.0)
    return np.clip(prior, 0.0, 1.0)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks / max(len(values) - 1, 1)


def _model_rerun_fdr(
    observed: pd.DataFrame,
    null_stats: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    null_models: tuple[str, ...],
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    for null_model in null_models:
        null_subset = null_stats[null_stats.get("null_model", pd.Series(dtype=str)).astype(str).eq(null_model)]
        for _, obs in observed.iterrows():
            lr_id = obs["lr_id"]
            null_values = pd.to_numeric(
                null_subset[null_subset["lr_id"].astype(str).eq(str(lr_id))]["observed_or_null_statistic"],
                errors="coerce",
            ).dropna()
            observed_value = float(obs["observed_or_null_statistic"])
            n_perm = int(len(null_values))
            if n_perm == 0:
                p_value = np.nan
                null_mean = np.nan
                null_q95 = np.nan
            else:
                p_value = (float((null_values.to_numpy(dtype=float) >= observed_value).sum()) + 1.0) / (
                    float(n_perm) + 1.0
                )
                null_mean = float(null_values.mean())
                null_q95 = float(null_values.quantile(0.95))
            rows.append(
                {
                    "null_model": null_model,
                    "lr_id": lr_id,
                    "model_rerun_group_statistic": observed_value,
                    "model_rerun_p_value": p_value,
                    "n_completed_permutations": n_perm,
                    "null_mean": null_mean,
                    "null_q95": null_q95,
                    "top_fraction": float(top_fraction),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["model_rerun_bh_fdr"] = np.nan
    for null_model, idx in result.groupby("null_model").groups.items():
        p = pd.to_numeric(result.loc[idx, "model_rerun_p_value"], errors="coerce").to_numpy(dtype=float)
        result.loc[idx, "model_rerun_bh_fdr"] = _bh_adjust(p)
    return result.merge(lr_pairs, on="lr_id", how="left").sort_values(
        ["null_model", "model_rerun_bh_fdr", "model_rerun_p_value", "model_rerun_group_statistic"],
        ascending=[True, True, True, False],
    )


def _control_summary(fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    if fdr.empty:
        return pd.DataFrame()
    for null_model, group in fdr.groupby("null_model", sort=False):
        positives = group[_positive_control_mask(group)]
        negatives = group[group.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        rows.append(
            {
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "model_rerun_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "model_rerun_bh_fdr"),
                "n_positive_significant": int(
                    (pd.to_numeric(positives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()
                ),
                "n_negative_significant": int(
                    (pd.to_numeric(negatives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum()
                ),
                "n_completed_permutations_min": int(group["n_completed_permutations"].min()),
                "control_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _audit(
    control: pd.DataFrame,
    status: pd.DataFrame,
    *,
    null_models: tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    completed = int(status["status"].eq("completed").sum()) if not status.empty else 0
    failed = int(status["status"].eq("failed").sum()) if not status.empty else 0
    passing = []
    for null_model in null_models:
        row = _match(control, "null_model", null_model)
        passing.append(
            bool(
                row
                and int(row.get("n_positive_significant", 0)) > 0
                and int(row.get("n_negative_significant", 999)) == 0
            )
        )
    all_pass = bool(passing and all(passing))
    return pd.DataFrame(
        [
            {
                "model_rerun_all_null_status": (
                    "positive_significant_negative_suppressed_all_nulls"
                    if all_pass
                    else "not_strict_all_null_success"
                ),
                "alpha": float(alpha),
                "null_models_required": ";".join(null_models),
                "null_models_passing": ";".join(
                    null_model for null_model, ok in zip(null_models, passing, strict=False) if ok
                ),
                "n_completed_reruns": completed,
                "n_failed_reruns": failed,
                "claim_boundary": (
                    "Strict model-rerun success requires every requested null model to recover at least one positive "
                    "control at BH FDR <= alpha and no high-expression negative controls at BH FDR <= alpha."
                ),
            }
        ]
    )


def _positive_control_mask(frame: pd.DataFrame) -> pd.Series:
    if "true_lr" in frame:
        return frame["true_lr"].fillna(False).astype(bool) & ~frame.get(
            "false_positive_high_expression",
            pd.Series(False, index=frame.index),
        ).fillna(False).astype(bool)
    if "control_class" in frame:
        return ~frame["control_class"].astype(str).str.contains("negative", case=False, na=False)
    return pd.Series(False, index=frame.index)


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    pos_sig = int((pd.to_numeric(positives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg_sig = int((pd.to_numeric(negatives.get("model_rerun_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos_sig > 0 and neg_sig == 0:
        return "positive_significant_negative_suppressed"
    if pos_sig > 0 and neg_sig > 0:
        return "positive_significant_but_negative_leakage"
    if pos_sig == 0 and neg_sig == 0:
        return "no_positive_recovery_negative_suppressed"
    return "negative_control_leakage"


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    ordered = frame.sort_values(["model_rerun_bh_fdr", "model_rerun_p_value", "model_rerun_group_statistic"], ascending=[True, True, False])
    row = ordered.iloc[0]
    return str(row.get("control_name", row.get("lr_id", "")))


def _best_value(frame: pd.DataFrame, column: str) -> float | str:
    if frame.empty or column not in frame:
        return ""
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _match(frame: pd.DataFrame, column: str, value: Any) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    subset = frame[frame[column].astype(str).eq(str(value))]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    out = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return out
    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[valid] = tmp
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
