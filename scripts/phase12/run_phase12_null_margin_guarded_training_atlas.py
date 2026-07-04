#!/usr/bin/env python
"""Run a stronger null-margin guarded SpatialSelf training variant."""

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
import scripts.phase12.run_phase12_anchor_background_guarded_training_atlas as abg  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_expanded as expanded  # noqa: E402
import scripts.phase12.run_phase12_atlas_model_pilot as pilot  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded_null_margin_guarded_training"
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
            pred_dir = PRED_ROOT / dataset_key / "spatialself_wavelet_edge_prior_null_margin_guarded"
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
                ranking_loss_weight=0.26,
                ranking_margin=1.15,
                feasibility_loss_weight=0.14,
                hard_negative_loss_weight=0.20,
                context_contrastive_loss_weight=0.20,
                context_contrastive_margin=0.65,
                context_contrastive_negative_mode="hard_receiver",
                lr_identity_contrastive_loss_weight=0.12,
                lr_identity_contrastive_margin=0.55,
                control_contrastive_loss_weight=0.20,
                control_contrastive_margin=0.65,
                null_candidate_loss_weight=0.12,
                spatial_null_contrastive_loss_weight=0.20,
                spatial_null_contrastive_margin=0.68,
                spatial_null_negative_mode="range_violation_receiver",
                background_guard_loss_weight=0.26,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=24000,
                seed=1912,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = calibrator.calibrate_dataset(dataset_key, replicate_dir, pred_path)
            ranking["method"] = "spatialself_wavelet_edge_prior_null_margin_guarded_training_calibrated"
            ranking = bg.apply_background_suppression(dataset_key, ranking)
            ranking["method"] = "spatialself_wavelet_edge_prior_null_margin_guarded_training_background_suppressed"
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
    dataset = abg.summarize_dataset(ranking_table, topk, hard_table, status_table)
    if "anchor_background_guarded_training_call" in dataset.columns:
        dataset = dataset.rename(columns={"anchor_background_guarded_training_call": "null_margin_guarded_training_call"})
    comparison = abg.compare_to_references(dataset)

    status_table.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_status.csv", index=False)
    ranking_table.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_hard_negative_leakage.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_dataset_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_comparison.csv", index=False)
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
            "purpose": "Phase 12 stronger null-margin guarded training variant",
            "claim_boundary": "Candidate model experiment for Goal 1 FDR closure; exact null audit required.",
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
        "n_context_contrastive_train": int(metrics.get("n_context_contrastive_train", 0)),
        "n_lr_identity_contrastive_train": int(metrics.get("n_lr_identity_contrastive_train", 0)),
        "n_spatial_null_contrastive_train": int(metrics.get("n_spatial_null_contrastive_train", 0)),
        "n_background_guard_train": int(metrics.get("n_background_guard_train", 0)),
        "spatial_null_negative_mode": str(metrics.get("spatial_null_negative_mode", "")),
        "error": error,
    }


def write_status(status: pd.DataFrame, dataset: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Null-Margin Guarded Training Atlas",
        "",
        "## Scope",
        "",
        "This variant increases context, LR-identity, control and spatial-null contrastive margins/weights inside training while retaining anchor/background guards.",
        "It tests whether stronger explicit null separation improves whole-atlas FDR candidates.",
        "It is not a reporting baseline unless exact-score/null audits are clean.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; context={int(row.n_context_contrastive_train)}, "
            f"LR-id={int(row.n_lr_identity_contrastive_train)}, spatial-null={int(row.n_spatial_null_contrastive_train)}, "
            f"background={int(row.n_background_guard_train)}, runtime={float(row.runtime_seconds):.1f}s."
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    lines += ["", "## Ranking Summary", ""]
    for row in dataset.itertuples(index=False):
        call = getattr(row, "null_margin_guarded_training_call", "")
        lines.append(
            f"- `{row.dataset_key}` `{call}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 risk={int(row.top10_negative_or_risk)}, top20 broad-other={int(row.top20_broad_other_lr)}, "
            f"top100 true={int(row.top100_true_lr)}."
        )
    lines += ["", "## Comparison Summary", ""]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.comparison_call}`: "
            f"delta top10 true vs main={fmt(getattr(row, 'delta_top10_true_lr_vs_main', 'NA'))}, "
            f"delta top10 risk vs main={fmt(getattr(row, 'delta_top10_negative_or_risk_vs_main', 'NA'))}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is a stronger null-margin candidate model experiment.",
        "- It still needs exact-score and end-to-end null audits before any whole-atlas FDR claim.",
    ]
    (OUT_DIR / "phase12_null_margin_guarded_training_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_null_margin_guarded_training_atlas",
        "inputs": [
            "outputs/phase12/atlas_model_expanded/replicates/*",
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_null_margin_guarded_training_status.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_lr_ranking.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_hard_negative_leakage.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_topk_summary.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_comparison.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_status.md"),
        ],
        "claim_boundary": "Candidate stronger null-margin training variant; exact null audit required.",
    }
    (OUT_DIR / "phase12_null_margin_guarded_training_manifest.json").write_text(
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
