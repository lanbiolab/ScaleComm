#!/usr/bin/env python
"""Run an integrated guarded spatial-specificity SpatialSelf candidate.

This Phase 12 Goal 1 variant combines the strongest existing training-side
guards with the guarded spatial-window specificity ranking layer. It is a
candidate model experiment, not the reporting baseline unless exact-score null
audits close the whole-atlas FDR gap.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
from scripts.phase12.calibrate_phase12_guarded_spatial_specificity import (  # noqa: E402
    add_guarded_spatial_specificity_calibration,
    select_columns,
    summarize_dataset,
    summarize_story,
    summarize_topk,
)
from scripts.phase12.calibrate_phase12_spatial_specificity import summarize_spatial_specificity  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_expanded as expanded  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_pilot as pilot  # noqa: E402
import scripts.phase12.calibrate_phase12_expanded_atlas_ranking as calibrator  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded_integrated_guarded_spatial_training"
REPLICATE_ROOT = OUT_DIR / "replicates"
PRED_ROOT = OUT_DIR / "predictions"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLICATE_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    status_rows: list[dict[str, Any]] = []
    ranking_rows: list[pd.DataFrame] = []
    hard_rows: list[pd.DataFrame] = []

    for spec in expanded.DATASETS:
        dataset_key = str(spec["dataset_key"])
        started = time.perf_counter()
        try:
            replicate_dir = copy_replicate(dataset_key)
            pred_dir = PRED_ROOT / dataset_key / "spatialself_wavelet_edge_prior_integrated_guarded_spatial"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=9,
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
                max_train_samples=24000,
                seed=2212,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = build_integrated_ranking(dataset_key, replicate_dir, pred_path)
            hard = pilot.summarize_hard_negative(replicate_dir, ranking, dataset_key)
            ranking_rows.append(ranking)
            hard_rows.append(hard)
            status_rows.append(status_row(dataset_key, "completed", started, replicate_dir, pred_path, metrics, ""))
        except Exception as exc:  # noqa: BLE001
            status_rows.append(status_row(dataset_key, "failed", started, Path(""), Path(""), {}, repr(exc)))

    ranking_table = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()
    hard_table = pd.concat(hard_rows, ignore_index=True) if hard_rows else pd.DataFrame()
    status_table = pd.DataFrame(status_rows)
    topk = summarize_topk(ranking_table)
    dataset = summarize_dataset(ranking_table, topk)
    dataset = add_hard_and_runtime(dataset, hard_table, status_table)
    story = summarize_story(dataset)
    comparison = compare_to_references(dataset)

    status_table.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_status.csv", index=False)
    ranking_table.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_hard_negative_leakage.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_dataset_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_training_story.csv", index=False)
    write_status(status_table, dataset, comparison, story)
    write_manifest()

    print(status_table.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def copy_replicate(dataset_key: str) -> Path:
    src = expanded.OUT_DIR / "replicates" / dataset_key
    dst = REPLICATE_ROOT / dataset_key
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    metadata_path = dst / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "dataset_key": dataset_key,
            "purpose": "Phase 12 integrated guarded spatial-specificity training candidate",
            "claim_boundary": "Candidate Goal 1 model direction; exact-score null audit required before FDR claims.",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dst


def build_integrated_ranking(dataset_key: str, replicate_dir: Path, pred_path: Path) -> pd.DataFrame:
    base = calibrator.calibrate_dataset(dataset_key, replicate_dir, pred_path)
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    features = summarize_spatial_specificity(candidates, lr_pairs)
    table = base.merge(features, on="lr_id", how="left", suffixes=("", "_spatial"))
    table = add_guarded_spatial_specificity_calibration(table)
    table["dataset_key"] = dataset_key
    table["method"] = "spatialself_wavelet_edge_prior_integrated_guarded_spatial_training"
    table["mean_score"] = table["calibrated_score"]
    table["q95_score"] = table["calibrated_q95_score"]
    table["rank"] = table["calibrated_score"].rank(ascending=False, method="min").astype(int)
    table["strict_rank"] = table["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    table["rank_q95"] = table["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    return select_columns(table)


def add_hard_and_runtime(dataset: pd.DataFrame, hard: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    if not hard.empty:
        rows = []
        for dataset_key, sub in hard.groupby("dataset_key", sort=False):
            col = "best_negative_beats_positive"
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "hard_negative_best_comparison_beats": int(sub[col].astype(bool).sum()) if col in sub else 0,
                    "hard_negative_comparisons": int(len(sub)),
                }
            )
        out = out.merge(pd.DataFrame(rows), on="dataset_key", how="left")
    cols = [
        "dataset_key",
        "runtime_seconds",
        "n_candidates",
        "n_spatial_null_contrastive_train",
        "n_background_guard_train",
        "n_control_contrastive_train",
        "n_hard_negative_train",
    ]
    existing = [col for col in cols if col in status.columns]
    if existing:
        out = out.merge(status[existing], on="dataset_key", how="left")
    out["integrated_guarded_spatial_training_call"] = [dataset_call(row) for row in out.itertuples(index=False)]
    return out


def compare_to_references(dataset: pd.DataFrame) -> pd.DataFrame:
    references = [
        (
            "main",
            PROJECT_ROOT / "outputs/phase12/atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv",
        ),
        (
            "guarded_spatial_posthoc",
            PROJECT_ROOT
            / "outputs/phase12/atlas_model_expanded_guarded_spatial_specificity/"
            / "phase12_guarded_spatial_specificity_dataset_summary.csv",
        ),
        (
            "anchor_background_range_guarded_training",
            PROJECT_ROOT
            / "outputs/phase12/atlas_model_expanded_anchor_background_range_guarded_training/"
            / "phase12_anchor_background_range_guarded_training_dataset_summary.csv",
        ),
    ]
    cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk",
        "top10_control_panel_empirical_fdr",
        "top50_true_lr",
    ]
    out = dataset[[col for col in cols if col in dataset.columns]].copy()
    for label, path in references:
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        ref = normalize_reference(ref)
        ref_cols = [col for col in cols if col in ref.columns]
        out = out.merge(ref[ref_cols], on="dataset_key", how="left", suffixes=("", f"_{label}"))
        for metric in ["top10_true_lr", "top10_negative_or_risk", "top50_true_lr"]:
            ref_metric = f"{metric}_{label}"
            if metric in out and ref_metric in out:
                out[f"delta_{metric}_vs_{label}"] = out[metric] - out[ref_metric]
    out["comparison_call"] = [comparison_call(row) for row in out.itertuples(index=False)]
    return out


def normalize_reference(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "top10_negative_or_risk" not in out and "top10_negative_or_risk_fraction" in out:
        out["top10_negative_or_risk"] = (
            pd.to_numeric(out["top10_negative_or_risk_fraction"], errors="coerce").fillna(0.0) * 10
        ).round().astype(int)
    return out


def comparison_call(row: object) -> str:
    delta_true = float(getattr(row, "delta_top10_true_lr_vs_main", 0.0))
    delta_risk = float(getattr(row, "delta_top10_negative_or_risk_vs_main", 0.0))
    delta_posthoc_risk = float(getattr(row, "delta_top10_negative_or_risk_vs_guarded_spatial_posthoc", 0.0))
    if delta_true > 0 and delta_risk <= 0:
        return "integrated_variant_improves_main_top10_without_extra_risk"
    if delta_risk < 0 and delta_true >= 0:
        return "integrated_variant_reduces_main_risk"
    if delta_posthoc_risk <= 0 and delta_true >= -1:
        return "integrated_variant_preserves_posthoc_guardrail"
    return "integrated_variant_tradeoff_or_worse"


def dataset_call(row: object) -> str:
    true10 = int(getattr(row, "top10_true_lr", 0))
    risk10 = int(getattr(row, "top10_negative_or_risk", 999))
    fdr = float(getattr(row, "top10_control_panel_empirical_fdr", np.nan))
    if true10 >= 5 and risk10 == 0:
        return "strong_clean_top10_candidate"
    if true10 >= 3 and np.isfinite(fdr) and fdr <= 0.1:
        return "moderate_clean_control_candidate"
    if true10 >= 3:
        return "moderate_candidate_with_risk"
    return "mixed_candidate"


def status_row(
    dataset_key: str,
    status: str,
    started: float,
    replicate_dir: Path,
    pred_path: Path,
    metrics: dict[str, object],
    error: str,
) -> dict[str, object]:
    return {
        "dataset_key": dataset_key,
        "status": status,
        "runtime_seconds": float(time.perf_counter() - started),
        "replicate_dir": rel(replicate_dir) if str(replicate_dir) else "",
        "prediction_path": rel(pred_path) if str(pred_path) else "",
        "n_candidates": int(metrics.get("n_candidates", 0)),
        "n_hard_negative_train": int(metrics.get("n_hard_negative_train", 0)),
        "n_context_contrastive_train": int(metrics.get("n_context_contrastive_train", 0)),
        "n_lr_identity_contrastive_train": int(metrics.get("n_lr_identity_contrastive_train", 0)),
        "n_control_contrastive_train": int(metrics.get("n_control_contrastive_train", 0)),
        "n_spatial_null_contrastive_train": int(metrics.get("n_spatial_null_contrastive_train", 0)),
        "n_background_guard_train": int(metrics.get("n_background_guard_train", 0)),
        "spatial_null_negative_mode": str(metrics.get("spatial_null_negative_mode", "")),
        "error": error,
    }


def write_status(
    status: pd.DataFrame,
    dataset: pd.DataFrame,
    comparison: pd.DataFrame,
    story: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 12 Integrated Guarded Spatial Training Atlas",
        "",
        "## Scope",
        "",
        "This candidate combines null-margin/range-violation training pressure with the guarded spatial-specificity ranking layer.",
        "It is designed to test the next-model direction exposed by the post-hoc guarded spatial-specificity audit.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; spatial-null={int(row.n_spatial_null_contrastive_train)}, "
            f"background={int(row.n_background_guard_train)}, runtime={float(row.runtime_seconds):.1f}s."
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    lines += ["", "## Ranking Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.integrated_guarded_spatial_training_call}`: "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
            f"top50 true={int(row.top50_true_lr)}, best true={row.best_true_lr}."
        )
    if not comparison.empty:
        lines += ["", "## Comparison", ""]
        for row in comparison.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.comparison_call}`: "
                f"delta top10 true vs main={fmt(getattr(row, 'delta_top10_true_lr_vs_main', 'NA'))}, "
                f"delta top10 risk vs main={fmt(getattr(row, 'delta_top10_negative_or_risk_vs_main', 'NA'))}."
            )
    if not story.empty:
        lines += ["", "## Story", ""]
        first = story.iloc[0]
        lines.append(
            f"- Overall `{first.get('overall_call', 'NA')}`; "
            f"mean top10 true={fmt(first.get('mean_top10_true_lr', 'NA'))}; "
            f"clean top10 datasets={fmt(first.get('n_clean_top10_risk', 'NA'))}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Safe wording: candidate model/ranking integration for the spatial-null bottleneck.",
        "- Unsafe wording before permutation audit: whole-atlas FDR is controlled.",
    ]
    (OUT_DIR / "phase12_integrated_guarded_spatial_training_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_integrated_guarded_spatial_training_atlas",
        "inputs": ["outputs/phase12/atlas_model_expanded/replicates/*"],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_status.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_hard_negative_leakage.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_topk_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_comparison.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_story.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_status.md"),
        ],
        "claim_boundary": "Candidate integrated training/ranking variant; exact null audit required.",
    }
    (OUT_DIR / "phase12_integrated_guarded_spatial_training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
