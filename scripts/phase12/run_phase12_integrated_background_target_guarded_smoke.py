#!/usr/bin/env python
"""Run a low-budget target-aware integrated background-guard smoke.

This is the first model path that consumes the formal20 broad/background target
panel during training. It is deliberately scoped to post-Xenium colon and a
small epoch/sample budget to validate the model direction before expensive
atlas and null reruns.

This is not a reporting baseline and not FDR evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402
import scripts.phase12.run_phase12_integrated_guarded_spatial_training_atlas as integrated  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
INTEGRATED_DIR = PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training"
TARGET_DIR = PHASE12_DIR / "integrated_background_guard_training_targets"
OUT_DIR = PHASE12_DIR / "integrated_background_target_guarded_training_smoke"
DATASET_KEY = "post_xenium_colon_full"
TARGET_PANEL = TARGET_DIR / "phase12_integrated_background_guard_target_panel.csv"
REFERENCE_RANKING = INTEGRATED_DIR / "phase12_integrated_guarded_spatial_training_lr_ranking.csv"
MAGNITUDE_GATE = TARGET_DIR / "phase12_integrated_background_guard_training_spec.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=121314)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT_DIR / "predictions" / DATASET_KEY / "spatialself_wavelet_integrated_background_target_guarded"
    if args.clean_output and pred_dir.exists():
        import shutil

        shutil.rmtree(pred_dir)
    replicate_dir = INTEGRATED_DIR / "replicates" / DATASET_KEY
    started = time.perf_counter()
    metrics = run_spatialself_on_replicate(
        replicate_dir,
        pred_dir,
        model_name="spatialself_wavelet",
        k_nn=18,
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
        seed=int(args.seed),
    )
    runtime = time.perf_counter() - started
    pred_path = Path(metrics["prediction_path"])
    ranking = integrated.build_integrated_ranking(DATASET_KEY, replicate_dir, pred_path)
    ranking["method"] = "spatialself_wavelet_integrated_background_target_guarded_smoke"
    ranking.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_lr_ranking.csv", index=False)

    target_panel = pd.read_csv(TARGET_PANEL)
    reference = pd.read_csv(REFERENCE_RANKING)
    reference = reference[reference["dataset_key"].astype(str).eq(DATASET_KEY)].copy()
    gate = pd.read_csv(MAGNITUDE_GATE)
    threshold = float(gate.iloc[0]["magnitude_gate_threshold"]) if len(gate) else 0.03
    target_summary = summarize_targets(ranking, reference, target_panel, threshold)
    anchor_summary = summarize_anchors(ranking, reference)
    dataset_summary = summarize_dataset(ranking, reference, target_summary, anchor_summary, metrics, runtime)

    target_summary.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_target_summary.csv", index=False)
    anchor_summary.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_anchor_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_dataset_summary.csv", index=False)
    write_status(dataset_summary, target_summary, anchor_summary)
    write_manifest(args, metrics, pred_path)

    print(dataset_summary.to_string(index=False))
    print()
    print(target_summary.to_string(index=False))
    print()
    print(anchor_summary.to_string(index=False))


def summarize_targets(
    ranking: pd.DataFrame,
    reference: pd.DataFrame,
    target_panel: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    target_lr = set(target_panel["lr_id"].astype(str))
    for priority, group in target_panel.groupby("target_priority", dropna=False):
        lr_ids = set(group["lr_id"].astype(str))
        sub = ranking[ranking["lr_id"].astype(str).isin(lr_ids)].copy()
        ref = reference[reference["lr_id"].astype(str).isin(lr_ids)].copy()
        rows.append(
            {
                "dataset_key": DATASET_KEY,
                "target_priority": priority,
                "n_target_lr": int(len(lr_ids)),
                "n_scored_target_lr": int(len(sub)),
                "median_score": median(sub, "calibrated_score"),
                "reference_median_score": median(ref, "calibrated_score"),
                "delta_median_score": median(sub, "calibrated_score") - median(ref, "calibrated_score"),
                "max_score": max_value(sub, "calibrated_score"),
                "reference_max_score": max_value(ref, "calibrated_score"),
                "n_above_magnitude_gate": int(pd.to_numeric(sub["calibrated_score"], errors="coerce").ge(threshold).sum())
                if len(sub)
                else 0,
                "n_top10": int(pd.to_numeric(sub["rank"], errors="coerce").le(10).sum()) if len(sub) else 0,
                "reference_n_top10": int(pd.to_numeric(ref["rank"], errors="coerce").le(10).sum()) if len(ref) else 0,
            }
        )
    all_target = ranking[ranking["lr_id"].astype(str).isin(target_lr)].copy()
    all_ref = reference[reference["lr_id"].astype(str).isin(target_lr)].copy()
    rows.append(
        {
            "dataset_key": DATASET_KEY,
            "target_priority": "all_background_targets",
            "n_target_lr": int(len(target_lr)),
            "n_scored_target_lr": int(len(all_target)),
            "median_score": median(all_target, "calibrated_score"),
            "reference_median_score": median(all_ref, "calibrated_score"),
            "delta_median_score": median(all_target, "calibrated_score") - median(all_ref, "calibrated_score"),
            "max_score": max_value(all_target, "calibrated_score"),
            "reference_max_score": max_value(all_ref, "calibrated_score"),
            "n_above_magnitude_gate": int(
                pd.to_numeric(all_target["calibrated_score"], errors="coerce").ge(threshold).sum()
            )
            if len(all_target)
            else 0,
            "n_top10": int(pd.to_numeric(all_target["rank"], errors="coerce").le(10).sum()) if len(all_target) else 0,
            "reference_n_top10": int(pd.to_numeric(all_ref["rank"], errors="coerce").le(10).sum()) if len(all_ref) else 0,
        }
    )
    return pd.DataFrame(rows)


def summarize_anchors(ranking: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    anchors = ["TGFB1-TGFBR2", "CXCL12-CXCR4", "SPP1-CD44"]
    rows = []
    for lr_pair in anchors:
        sub = ranking[ranking["lr_pair"].astype(str).eq(lr_pair)]
        ref = reference[reference["lr_pair"].astype(str).eq(lr_pair)]
        rows.append(
            {
                "dataset_key": DATASET_KEY,
                "lr_pair": lr_pair,
                "rank": value(sub, "rank"),
                "reference_rank": value(ref, "rank"),
                "calibrated_score": value(sub, "calibrated_score"),
                "reference_calibrated_score": value(ref, "calibrated_score"),
                "delta_score": value(sub, "calibrated_score") - value(ref, "calibrated_score"),
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(
    ranking: pd.DataFrame,
    reference: pd.DataFrame,
    target_summary: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    metrics: dict[str, object],
    runtime: float,
) -> pd.DataFrame:
    true = ranking["true_lr"].fillna(False).astype(bool)
    ref_true = reference["true_lr"].fillna(False).astype(bool)
    top10 = ranking[pd.to_numeric(ranking["rank"], errors="coerce").le(10)]
    ref_top10 = reference[pd.to_numeric(reference["rank"], errors="coerce").le(10)]
    all_targets = target_summary[target_summary["target_priority"].eq("all_background_targets")].iloc[0]
    anchor_min_delta = float(pd.to_numeric(anchor_summary["delta_score"], errors="coerce").min())
    target_top10_delta = int(all_targets["n_top10"]) - int(all_targets["reference_n_top10"])
    if target_top10_delta < 0 and anchor_min_delta > -0.05:
        call = "target_guard_smoke_suppresses_targets_without_large_anchor_loss"
    elif target_top10_delta < 0:
        call = "target_guard_smoke_suppresses_targets_with_anchor_tradeoff"
    else:
        call = "target_guard_smoke_no_target_suppression"
    return pd.DataFrame(
        [
            {
                "dataset_key": DATASET_KEY,
                "status": "completed_smoke",
                "runtime_seconds": float(runtime),
                "epochs": int(metrics.get("epochs", 0)),
                "max_train_samples": int(metrics.get("max_train_samples", 0)),
                "n_background_guard_train": int(metrics.get("n_background_guard_train", 0)),
                "n_background_guard_target_train": int(metrics.get("n_background_guard_target_train", 0)),
                "top10_true_lr": int(top10["true_lr"].fillna(False).astype(bool).sum()),
                "reference_top10_true_lr": int(ref_top10["true_lr"].fillna(False).astype(bool).sum()),
                "top10_background_targets": int(all_targets["n_top10"]),
                "reference_top10_background_targets": int(all_targets["reference_n_top10"]),
                "all_target_median_delta_score": float(all_targets["delta_median_score"]),
                "all_target_max_score": float(all_targets["max_score"]),
                "n_targets_above_magnitude_gate": int(all_targets["n_above_magnitude_gate"]),
                "anchor_min_delta_score": anchor_min_delta,
                "target_guard_smoke_call": call,
                "claim_boundary": "Low-budget target-aware training smoke; not a reporting baseline or FDR result.",
            }
        ]
    )


def write_status(dataset: pd.DataFrame, target: pd.DataFrame, anchor: pd.DataFrame) -> None:
    row = dataset.iloc[0]
    lines = [
        "# Integrated Background-Target Guarded Training Smoke",
        "",
        "## Result",
        "",
        f"- Dataset: `{row.dataset_key}`",
        f"- Status: `{row.status}`",
        f"- Call: `{row.target_guard_smoke_call}`",
        f"- Background guard target train: {int(row.n_background_guard_target_train)}/{int(row.n_background_guard_train)}",
        f"- Top10 true LR: {int(row.top10_true_lr)} vs reference {int(row.reference_top10_true_lr)}",
        f"- Top10 background targets: {int(row.top10_background_targets)} vs reference {int(row.reference_top10_background_targets)}",
        f"- Targets above magnitude gate: {int(row.n_targets_above_magnitude_gate)}",
        f"- Anchor minimum delta score: {float(row.anchor_min_delta_score):.4g}",
        "",
        "## Claim Boundary",
        "",
        str(row.claim_boundary),
    ]
    (OUT_DIR / "phase12_integrated_background_target_guarded_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(args: argparse.Namespace, metrics: dict[str, object], pred_path: Path) -> None:
    manifest = {
        "task": "phase12_integrated_background_target_guarded_training_smoke",
        "dataset": DATASET_KEY,
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "seed": int(args.seed),
        "target_panel": rel(TARGET_PANEL),
        "reference_ranking": rel(REFERENCE_RANKING),
        "prediction_path": rel(pred_path),
        "metrics": {
            "n_background_guard_train": int(metrics.get("n_background_guard_train", 0)),
            "n_background_guard_target_train": int(metrics.get("n_background_guard_target_train", 0)),
        },
        "outputs": {
            "ranking": rel(OUT_DIR / "phase12_integrated_background_target_guarded_lr_ranking.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_integrated_background_target_guarded_dataset_summary.csv"),
            "target_summary": rel(OUT_DIR / "phase12_integrated_background_target_guarded_target_summary.csv"),
            "anchor_summary": rel(OUT_DIR / "phase12_integrated_background_target_guarded_anchor_summary.csv"),
            "status": rel(OUT_DIR / "phase12_integrated_background_target_guarded_status.md"),
        },
        "claim_boundary": "Low-budget target-aware training smoke only.",
    }
    (OUT_DIR / "phase12_integrated_background_target_guarded_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return float("nan")
    return float(pd.to_numeric(table[column], errors="coerce").iloc[0])


def median(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return float("nan")
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def max_value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return float("nan")
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.max()) if len(values) else float("nan")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
