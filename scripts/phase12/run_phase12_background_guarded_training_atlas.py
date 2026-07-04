#!/usr/bin/env python
"""Run a background-guarded SpatialSelf training variant on Phase 12 atlases."""

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
import scripts.phase12.calibrate_phase12_expanded_atlas_ranking as calibrator  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_expanded as expanded  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_pilot as pilot  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded_background_guarded_training"
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
            replicate_dir = copy_replicate(dataset_key, spec)
            pred_dir = PRED_ROOT / dataset_key / "spatialself_wavelet_edge_prior_background_guarded"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=6,
                positive_quantile=0.9,
                negative_quantile=0.2,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.18,
                weak_positive_label=0.80,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.20,
                feasibility_loss_weight=0.12,
                hard_negative_loss_weight=0.14,
                context_contrastive_loss_weight=0.12,
                context_contrastive_margin=0.5,
                context_contrastive_negative_mode="hard_receiver",
                lr_identity_contrastive_loss_weight=0.06,
                lr_identity_contrastive_margin=0.35,
                control_contrastive_loss_weight=0.10,
                control_contrastive_margin=0.5,
                null_candidate_loss_weight=0.06,
                spatial_null_contrastive_loss_weight=0.06,
                spatial_null_contrastive_margin=0.45,
                background_guard_loss_weight=0.18,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=16000,
                seed=1512,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = calibrator.calibrate_dataset(dataset_key, replicate_dir, pred_path)
            ranking["method"] = "spatialself_wavelet_edge_prior_background_guarded_training_calibrated"
            hard = pilot.summarize_hard_negative(replicate_dir, ranking, dataset_key)
            all_ranking.append(ranking)
            all_hard.append(hard)
            all_status.append(status_row(dataset_key, "completed", started, replicate_dir, pred_path, metrics, ""))
        except Exception as exc:  # noqa: BLE001
            all_status.append(status_row(dataset_key, "failed", started, Path(""), Path(""), {}, repr(exc)))

    ranking_table = pd.concat(all_ranking, ignore_index=True) if all_ranking else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    status_table = pd.DataFrame(all_status)
    topk = calibrator.summarize_topk(ranking_table)
    dataset = calibrator.summarize_dataset(ranking_table, topk, hard_table, status_table)
    comparison = compare_to_main_and_anchor(dataset)

    status_table.to_csv(OUT_DIR / "phase12_background_guarded_training_status.csv", index=False)
    ranking_table.to_csv(OUT_DIR / "phase12_background_guarded_training_calibrated_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_background_guarded_training_hard_negative_leakage.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_background_guarded_training_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_background_guarded_training_dataset_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_background_guarded_training_comparison.csv", index=False)
    write_status(status_table, dataset, comparison)
    write_manifest()

    print(status_table.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


def copy_replicate(dataset_key: str, spec: dict[str, Any]) -> Path:
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
            "purpose": "Phase 12 background-guarded training variant",
            "claim_boundary": "Training variant for broad-background and high-expression risk suppression; requires exact null audit before FDR claims.",
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
        "error": error,
    }


def compare_to_main_and_anchor(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_broad_other_lr",
        "top10_negative_or_risk_fraction",
        "top10_control_panel_empirical_fdr",
        "top50_true_lr",
        "top50_negative_or_risk_fraction",
        "hard_negative_best_comparison_beats",
    ]
    out = dataset[cols].copy()
    references = [
        ("main", PROJECT_ROOT / "outputs/phase12/atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv"),
        (
            "anchor",
            PROJECT_ROOT
            / "outputs/phase12/atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_dataset_summary.csv",
        ),
    ]
    for label, path in references:
        if not path.exists():
            continue
        ref = pd.read_csv(path)[cols]
        out = out.merge(ref, on="dataset_key", how="left", suffixes=("", f"_{label}"))
        out[f"delta_top10_true_lr_vs_{label}"] = out["top10_true_lr"] - out[f"top10_true_lr_{label}"]
        out[f"delta_top10_risk_vs_{label}"] = (
            out["top10_negative_or_risk_fraction"] - out[f"top10_negative_or_risk_fraction_{label}"]
        )
        out[f"delta_top50_true_lr_vs_{label}"] = out["top50_true_lr"] - out[f"top50_true_lr_{label}"]
        out[f"delta_hard_negative_beats_vs_{label}"] = (
            out["hard_negative_best_comparison_beats"] - out[f"hard_negative_best_comparison_beats_{label}"]
        )
    out["comparison_call"] = [comparison_call(row) for row in out.itertuples(index=False)]
    return out


def comparison_call(row: object) -> str:
    delta_true = float(getattr(row, "delta_top10_true_lr_vs_main", 0.0))
    delta_risk = float(getattr(row, "delta_top10_risk_vs_main", 0.0))
    if delta_true > 0 and delta_risk <= 0:
        return "guarded_training_improves_main_top10"
    if delta_true >= 0 and delta_risk < 0:
        return "guarded_training_reduces_main_risk"
    if delta_true < 0 and delta_risk <= 0:
        return "guarded_training_conservative_tradeoff"
    if delta_risk > 0:
        return "guarded_training_adds_risk"
    return "guarded_training_similar_to_main"


def write_status(status: pd.DataFrame, dataset: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Background-Guarded Training Atlas",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; background guard pairs={int(row.n_background_guard_train)}; "
            f"spatial-null pairs={int(row.n_spatial_null_contrastive_train)}; runtime={float(row.runtime_seconds):.1f}s"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not dataset.empty:
        lines += ["", "## Calibrated Ranking Summary", ""]
        for row in dataset.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}`: top10 true={int(row.top10_true_lr)}, "
                f"top10 risk={float(row.top10_negative_or_risk_fraction):.3g}, "
                f"top50 true={int(row.top50_true_lr)}, hard-neg beats={int(row.hard_negative_best_comparison_beats)}."
            )
    if not comparison.empty:
        lines += ["", "## Comparison", ""]
        for row in comparison.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.comparison_call}`: "
                f"delta top10 true vs main={int(getattr(row, 'delta_top10_true_lr_vs_main', 0))}, "
                f"delta top10 risk vs main={float(getattr(row, 'delta_top10_risk_vs_main', 0.0)):.3g}."
            )
    lines += [
        "",
        "## Boundary",
        "",
        "- This variant moves broad-background and high-expression risk suppression into training.",
        "- It is not a reporting baseline unless exact-score or end-to-end null audits support discovery FDR.",
    ]
    (OUT_DIR / "phase12_background_guarded_training_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_background_guarded_training_atlas",
        "outputs": [
            rel(OUT_DIR / "phase12_background_guarded_training_status.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_calibrated_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_hard_negative_leakage.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_topk_summary.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_comparison.csv"),
            rel(OUT_DIR / "phase12_background_guarded_training_status.md"),
        ],
        "claim_boundary": "Background-guarded full expanded-atlas training variant; exact null audit required before FDR claims.",
    }
    (OUT_DIR / "phase12_background_guarded_training_manifest.json").write_text(
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
