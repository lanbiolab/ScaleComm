#!/usr/bin/env python
"""Smoke test a SpatialSelf training objective with spatial-null contrastive loss."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs/phase12/spatial_null_training_smoke"
DATASET_KEY = "human_lymph_node_visium_raw_full"
REPLICATE_DIR = PROJECT_ROOT / f"outputs/phase12/atlas_model_expanded/replicates/{DATASET_KEY}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / DATASET_KEY
    if run_dir.exists():
        shutil.rmtree(run_dir)
    started = time.perf_counter()
    metrics = run_spatialself_on_replicate(
        REPLICATE_DIR,
        run_dir,
        model_name="spatialself_wavelet",
        k_nn=18,
        epochs=1,
        positive_quantile=0.9,
        negative_quantile=0.2,
        pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
        train_sampling_strategy="balanced_per_lr",
        ranking_loss_weight=0.02,
        feasibility_loss_weight=0.02,
        hard_negative_loss_weight=0.02,
        context_contrastive_loss_weight=0.02,
        context_contrastive_negative_mode="hard_receiver",
        lr_identity_contrastive_loss_weight=0.02,
        control_contrastive_loss_weight=0.02,
        null_candidate_loss_weight=0.02,
        spatial_null_contrastive_loss_weight=0.05,
        spatial_null_contrastive_margin=0.5,
        lr_cap_positive=8,
        lr_cap_negative=20,
        use_edge_prior_features=True,
        max_train_samples=4000,
        seed=121212,
    )
    row = {
        "dataset_key": DATASET_KEY,
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "replicate_dir": rel(REPLICATE_DIR),
        "prediction_path": rel(Path(metrics["prediction_path"])),
        "n_candidates": int(metrics.get("n_candidates", 0)),
        "n_spatial_null_contrastive_train": int(metrics.get("n_spatial_null_contrastive_train", 0)),
        "n_context_contrastive_train": int(metrics.get("n_context_contrastive_train", 0)),
        "n_hard_negative_train": int(metrics.get("n_hard_negative_train", 0)),
        "n_lr_identity_contrastive_train": int(metrics.get("n_lr_identity_contrastive_train", 0)),
        "n_control_contrastive_train": int(metrics.get("n_control_contrastive_train", 0)),
        "spatial_null_contrastive_loss_weight": float(metrics.get("spatial_null_contrastive_loss_weight", 0.0)),
        "claim_boundary": "Training smoke only; proves the new loss path runs, not that whole-atlas FDR is solved.",
    }
    table = pd.DataFrame([row])
    table.to_csv(OUT_DIR / "phase12_spatial_null_training_smoke_summary.csv", index=False)
    write_status(row)
    write_manifest(row)
    print(table.to_string(index=False))


def write_status(row: dict[str, object]) -> None:
    lines = [
        "# Phase 12 Spatial-Null Training Smoke",
        "",
        "## Result",
        "",
        f"- Dataset: `{row['dataset_key']}`.",
        f"- Status: `{row['status']}`.",
        f"- Spatial-null contrastive train pairs: {int(row['n_spatial_null_contrastive_train'])}.",
        f"- Context contrastive train pairs: {int(row['n_context_contrastive_train'])}.",
        f"- Hard-negative train pairs: {int(row['n_hard_negative_train'])}.",
        f"- Runtime: {float(row['runtime_seconds']):.1f}s.",
        "",
        "## Boundary",
        "",
        "- This proves the next-model training path is executable on a full-LR atlas replicate.",
        "- It does not prove improved FDR; that requires full atlas training plus exact-score or end-to-end null audit.",
    ]
    (OUT_DIR / "phase12_spatial_null_training_smoke_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(row: dict[str, object]) -> None:
    manifest = {
        "task": "phase12_spatial_null_training_smoke",
        "dataset_key": row["dataset_key"],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_null_training_smoke_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_null_training_smoke_status.md"),
            row["prediction_path"],
        ],
        "claim_boundary": row["claim_boundary"],
    }
    (OUT_DIR / "phase12_spatial_null_training_smoke_manifest.json").write_text(
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
