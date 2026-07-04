#!/usr/bin/env python
"""Run an anchor-preserving plus background-guarded SpatialSelf training variant."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
import scripts.phase12.calibrate_phase12_expanded_atlas_background_suppressed as bg  # noqa: E402
import scripts.phase12.calibrate_phase12_expanded_atlas_ranking as calibrator  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_expanded as expanded  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_pilot as pilot  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded_anchor_background_range_guarded_training"
REPLICATE_ROOT = OUT_DIR / "replicates"
PRED_ROOT = OUT_DIR / "predictions"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLICATE_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    all_status: list[dict[str, Any]] = []
    all_ranking: list[pd.DataFrame] = []
    all_hard: list[pd.DataFrame] = []

    for spec in expanded.DATASETS:
        started = time.perf_counter()
        dataset_key = str(spec["dataset_key"])
        try:
            replicate_dir = copy_replicate(dataset_key)
            pred_dir = PRED_ROOT / dataset_key / "spatialself_wavelet_edge_prior_anchor_background_range_guarded"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=7,
                positive_quantile=0.9,
                negative_quantile=0.2,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.20,
                weak_positive_label=0.82,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.22,
                feasibility_loss_weight=0.12,
                hard_negative_loss_weight=0.16,
                context_contrastive_loss_weight=0.12,
                context_contrastive_margin=0.5,
                context_contrastive_negative_mode="hard_receiver",
                lr_identity_contrastive_loss_weight=0.06,
                lr_identity_contrastive_margin=0.35,
                control_contrastive_loss_weight=0.12,
                control_contrastive_margin=0.5,
                null_candidate_loss_weight=0.06,
                spatial_null_contrastive_loss_weight=0.11,
                spatial_null_contrastive_margin=0.52,
                spatial_null_negative_mode="range_violation_receiver",
                background_guard_loss_weight=0.22,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=18000,
                seed=1812,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = calibrator.calibrate_dataset(dataset_key, replicate_dir, pred_path)
            ranking["method"] = "spatialself_wavelet_edge_prior_anchor_background_range_guarded_training_calibrated"
            ranking = bg.apply_background_suppression(dataset_key, ranking)
            ranking["method"] = "spatialself_wavelet_edge_prior_anchor_background_range_guarded_training_background_suppressed"
            hard = pilot.summarize_hard_negative(replicate_dir, ranking, dataset_key)
            all_ranking.append(ranking)
            all_hard.append(hard)
            all_status.append(status_row(dataset_key, "completed", started, replicate_dir, pred_path, metrics, ""))
        except Exception as exc:  # noqa: BLE001
            all_status.append(status_row(dataset_key, "failed", started, Path(""), Path(""), {}, repr(exc)))

    ranking_table = pd.concat(all_ranking, ignore_index=True) if all_ranking else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    status_table = pd.DataFrame(all_status)
    topk = bg.summarize_topk(ranking_table)
    dataset = summarize_dataset(ranking_table, topk, hard_table, status_table)
    comparison = compare_to_references(dataset)

    status_table.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_status.csv", index=False)
    ranking_table.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_hard_negative_leakage.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_dataset_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_comparison.csv", index=False)
    write_status(status_table, dataset, comparison)
    write_manifest()

    print(status_table.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


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
            "purpose": "Phase 12 anchor-preserving plus background-guarded training variant",
            "claim_boundary": "Training variant for Goal 1 FDR closure attempts; exact null audit required before FDR claims.",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dst


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
        "n_control_contrastive_train": int(metrics.get("n_control_contrastive_train", 0)),
        "n_spatial_null_contrastive_train": int(metrics.get("n_spatial_null_contrastive_train", 0)),
        "n_background_guard_train": int(metrics.get("n_background_guard_train", 0)),
        "spatial_null_negative_mode": str(metrics.get("spatial_null_negative_mode", "")),
        "error": error,
    }


def summarize_dataset(
    ranking: pd.DataFrame,
    topk: pd.DataFrame,
    hard: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    base = bg.summarize_dataset(ranking, topk)
    hard_summary = summarize_hard(hard)
    if not hard_summary.empty:
        base = base.merge(hard_summary, on="dataset_key", how="left")
    status_cols = [
        "dataset_key",
        "runtime_seconds",
        "n_candidates",
        "n_hard_negative_train",
        "n_control_contrastive_train",
        "n_spatial_null_contrastive_train",
        "n_background_guard_train",
    ]
    existing = [col for col in status_cols if col in status.columns]
    if existing:
        base = base.merge(status[existing], on="dataset_key", how="left")
    base["anchor_background_range_guarded_training_call"] = [dataset_call(row) for row in base.itertuples(index=False)]
    return base


def summarize_hard(hard: pd.DataFrame) -> pd.DataFrame:
    if hard.empty:
        return pd.DataFrame()
    rows = []
    for dataset_key, sub in hard.groupby("dataset_key", sort=False):
        beat_col = "best_negative_beats_positive"
        rows.append(
            {
                "dataset_key": dataset_key,
                "hard_negative_best_comparison_beats": int(sub[beat_col].astype(bool).sum()) if beat_col in sub else 0,
                "hard_negative_comparisons": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def compare_to_references(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk",
        "top10_control_panel_empirical_fdr",
        "top20_broad_other_lr",
        "top100_true_lr",
        "hard_negative_best_comparison_beats",
    ]
    out = dataset[[col for col in cols if col in dataset.columns]].copy()
    references = [
        ("main", PROJECT_ROOT / "outputs/phase12/atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv"),
        (
            "anchor_bg",
            PROJECT_ROOT
            / "outputs/phase12/atlas_model_expanded_anchor_background_suppressed/"
            / "phase12_anchor_background_suppressed_dataset_summary.csv",
        ),
        (
            "background_guarded",
            PROJECT_ROOT
            / "outputs/phase12/atlas_model_expanded_background_guarded_training/"
            / "phase12_background_guarded_training_dataset_summary.csv",
        ),
    ]
    for label, path in references:
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        ref = normalize_reference(ref)
        ref_cols = [col for col in cols if col in ref.columns]
        out = out.merge(ref[ref_cols], on="dataset_key", how="left", suffixes=("", f"_{label}"))
        for metric in ["top10_true_lr", "top10_negative_or_risk", "top20_broad_other_lr", "top100_true_lr"]:
            if metric in out.columns and f"{metric}_{label}" in out.columns:
                out[f"delta_{metric}_vs_{label}"] = out[metric] - out[f"{metric}_{label}"]
    out["comparison_call"] = [comparison_call(row) for row in out.itertuples(index=False)]
    return out


def normalize_reference(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "top10_negative_or_risk" not in out and "top10_negative_or_risk_fraction" in out:
        out["top10_negative_or_risk"] = (pd.to_numeric(out["top10_negative_or_risk_fraction"], errors="coerce") * 10).round()
    return out


def comparison_call(row: object) -> str:
    delta_true = float(getattr(row, "delta_top10_true_lr_vs_main", 0.0))
    delta_risk = float(getattr(row, "delta_top10_negative_or_risk_vs_main", 0.0))
    delta_broad = float(getattr(row, "delta_top20_broad_other_lr_vs_main", 0.0))
    if delta_true > 0 and delta_risk <= 0 and delta_broad <= 0:
        return "training_variant_improves_main_without_extra_top_risk"
    if delta_true > 0 and delta_risk <= 0:
        return "training_variant_improves_top10_with_broad_caveat"
    if delta_risk < 0 and delta_true >= 0:
        return "training_variant_reduces_risk"
    if delta_true < 0 and delta_risk <= 0:
        return "training_variant_conservative_tradeoff"
    return "training_variant_tradeoff_or_worse"


def dataset_call(row: object) -> str:
    true10 = int(getattr(row, "top10_true_lr", 0))
    risk10 = int(getattr(row, "top10_negative_or_risk", 0))
    control = float(getattr(row, "top10_control_panel_empirical_fdr", 1.0))
    if true10 >= 5 and risk10 == 0:
        return "strong_top10_recovery_clean_top10"
    if true10 >= 3 and control <= 0.1:
        return "moderate_top10_recovery_clean_control"
    if true10 >= 3:
        return "moderate_top10_recovery_with_risk"
    return "weak_or_mixed_top10_recovery"


def write_status(status: pd.DataFrame, dataset: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor-Background Range-Guarded Training Atlas",
        "",
        "## Scope",
        "",
        "This variant combines anchor-preserving spatial-null contrastive pressure with background/high-expression guards inside training, then applies the same background-suppressed ranking layer used in the prior post-hoc candidate.",
        "It is a candidate model direction, not the reporting baseline unless exact-score null audits support it.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; spatial-null pairs={int(row.n_spatial_null_contrastive_train)}, "
            f"background guard pairs={int(row.n_background_guard_train)}, runtime={float(row.runtime_seconds):.1f}s."
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    lines += ["", "## Ranking Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.anchor_background_range_guarded_training_call}`: "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={int(row.top10_negative_or_risk)}, "
            f"top20 broad-other={int(row.top20_broad_other_lr)}, "
            f"hard-neg beats={int(getattr(row, 'hard_negative_best_comparison_beats', 0))}."
        )
    if not comparison.empty:
        lines += ["", "## Comparison To References", ""]
        for row in comparison.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.comparison_call}`: "
                f"delta top10 true vs main={int(getattr(row, 'delta_top10_true_lr_vs_main', 0))}, "
                f"delta top10 risk vs main={int(getattr(row, 'delta_top10_negative_or_risk_vs_main', 0))}, "
                f"delta top20 broad-other vs main={int(getattr(row, 'delta_top20_broad_other_lr_vs_main', 0))}."
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Safe wording: candidate training integration with range-violation spatial null of the best Phase 12 calibration direction.",
        "- Unsafe wording before permutation audit: whole-atlas FDR is controlled.",
    ]
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_anchor_background_range_guarded_training_atlas",
        "inputs": [
            "outputs/phase12/atlas_model_expanded/replicates/*",
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_status.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_hard_negative_leakage.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_topk_summary.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_comparison.csv"),
            rel(OUT_DIR / "phase12_anchor_background_range_guarded_training_status.md"),
        ],
        "claim_boundary": "Training-integrated anchor/background candidate; exact-score or end-to-end null audit required before broad FDR claims.",
    }
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
