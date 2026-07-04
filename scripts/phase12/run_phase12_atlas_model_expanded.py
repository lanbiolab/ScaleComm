#!/usr/bin/env python
"""Run expanded Phase 12 atlas scoring with the SpatialSelf wavelet model.

This extends the previous 55-LR pilot to the covered LR universe for full
transcriptome backgrounds. It intentionally writes to a separate output
directory so the pilot remains reproducible.
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

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
import scripts.phase12.run_phase12_atlas_model_pilot as pilot


READINESS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_readiness"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
REPLICATE_ROOT = OUT_DIR / "replicates"
PRED_ROOT = OUT_DIR / "predictions"


DATASETS = [
    {
        "dataset_key": "visium_hd_crc_full",
        "label": "Visium HD CRC P2CRC full-transcriptome expanded atlas",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 9999,
    },
    {
        "dataset_key": "post_xenium_colon_full",
        "label": "Post-Xenium/CytAssist colon cancer full-transcriptome expanded atlas",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/post_xenium_human_colon_cancer_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 9999,
    },
    {
        "dataset_key": "human_lymph_node_visium_raw_full",
        "label": "Human lymph node Visium raw full-transcriptome expanded atlas",
        "cells": "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset/cells.csv",
        "genes": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 9999,
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLICATE_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    # Reuse the pilot builder/summarizers, but redirect their globals so all
    # intermediate replicate and prediction files stay under atlas_model_expanded.
    pilot.OUT_DIR = OUT_DIR
    pilot.REPLICATE_ROOT = REPLICATE_ROOT
    pilot.PRED_ROOT = PRED_ROOT

    lr_universe = pd.read_csv(READINESS_DIR / "phase12_lr_universe.csv")
    hard_templates = pd.read_csv(READINESS_DIR / "phase12_hard_negative_templates.csv")

    all_rankings: list[pd.DataFrame] = []
    all_hard: list[pd.DataFrame] = []
    all_status: list[dict[str, Any]] = []
    all_lr_manifest: list[pd.DataFrame] = []

    for spec in DATASETS:
        started = time.perf_counter()
        try:
            replicate_dir, build_summary, lr_pairs = pilot.build_replicate(spec, lr_universe, hard_templates)
            rewrite_metadata(replicate_dir, spec, build_summary)
            pred_dir = PRED_ROOT / spec["dataset_key"] / "spatialself_wavelet_edge_prior"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=8,
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
                max_train_samples=16000,
                seed=1212,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = pilot.summarize_lr_ranking(replicate_dir, pred_path, lr_pairs, spec["dataset_key"])
            hard = pilot.summarize_hard_negative(replicate_dir, ranking, spec["dataset_key"])
            all_rankings.append(ranking)
            all_hard.append(hard)
            all_lr_manifest.append(lr_pairs.assign(dataset_key=spec["dataset_key"]))
            all_status.append(
                {
                    "dataset_key": spec["dataset_key"],
                    "status": "completed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "replicate_dir": rel(replicate_dir),
                    "prediction_path": rel(pred_path),
                    **build_summary,
                    "n_candidates_scored": int(metrics.get("n_candidates", 0)),
                    "score_min": float(metrics.get("score_min", np.nan)),
                    "score_max": float(metrics.get("score_max", np.nan)),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            all_status.append(
                {
                    "dataset_key": spec["dataset_key"],
                    "status": "failed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "replicate_dir": "",
                    "prediction_path": "",
                    "error": repr(exc),
                }
            )

    ranking_table = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    status_table = pd.DataFrame(all_status)
    lr_manifest = pd.concat(all_lr_manifest, ignore_index=True) if all_lr_manifest else pd.DataFrame()

    ranking_table.to_csv(OUT_DIR / "phase12_atlas_model_expanded_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_atlas_model_expanded_hard_negative_leakage.csv", index=False)
    status_table.to_csv(OUT_DIR / "phase12_atlas_model_expanded_status.csv", index=False)
    lr_manifest.to_csv(OUT_DIR / "phase12_atlas_model_expanded_lr_manifest.csv", index=False)
    write_status(ranking_table, hard_table, status_table)
    write_manifest(status_table)

    print(status_table.to_string(index=False))
    if not ranking_table.empty:
        print()
        print(ranking_table.groupby("dataset_key").head(20).to_string(index=False))


def rewrite_metadata(replicate_dir: Path, spec: dict[str, Any], build_summary: dict[str, Any]) -> None:
    (replicate_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_key": spec["dataset_key"],
                "label": spec["label"],
                "purpose": "Phase 12 expanded whole-LR atlas model run",
                "candidate_edges": "multi-scale nearest, medium-range and longer-range sender-receiver candidates",
                **build_summary,
                "claim_boundary": "Expanded covered-LR model run; still requires formal permutation-calibrated global FDR before whole-atlas superiority claims.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_status(ranking: pd.DataFrame, hard: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Atlas Model Status",
        "",
        "## Scope",
        "",
        "This run expands the SpatialSelf atlas model from the previous 55-LR pilot to all LR pairs covered by each full-transcriptome background.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; cells={getattr(row, 'n_cells', 'NA')}; "
            f"LR={getattr(row, 'n_lr_pairs', 'NA')}; candidates={getattr(row, 'n_candidates', 'NA')}; "
            f"runtime={float(row.runtime_seconds):.1f}s"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not ranking.empty:
        lines += ["", "## Top SpatialSelf LR", ""]
        for dataset_key, sub in ranking.groupby("dataset_key"):
            top = sub.sort_values("rank").head(15)
            top_text = "; ".join(
                f"{r.lr_pair}#{int(r.rank)}{' NEG' if bool(r.false_positive_high_expression) else ''}"
                for r in top.itertuples(index=False)
            )
            lines.append(f"- `{dataset_key}`: {top_text}")
    if not hard.empty:
        lines += ["", "## Hard-Negative Leakage", ""]
        grouped = hard.groupby("dataset_key")["negative_beats_positive"].agg(["sum", "count"])
        for dataset_key, row in grouped.iterrows():
            lines.append(
                f"- `{dataset_key}`: {int(row['sum'])}/{int(row['count'])} best hard-negative comparisons beat their positive anchor."
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This satisfies the expanded covered-LR scoring layer for at least two full-transcriptome datasets. It does not by itself establish final global FDR, whole-atlas superiority over baselines, or causal downstream validation.",
    ]
    (OUT_DIR / "phase12_atlas_model_expanded_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_atlas_model_expanded",
        "model": "spatialself_wavelet_edge_prior",
        "datasets": status["dataset_key"].tolist() if "dataset_key" in status else [],
        "outputs": [
            "phase12_atlas_model_expanded_lr_ranking.csv",
            "phase12_atlas_model_expanded_hard_negative_leakage.csv",
            "phase12_atlas_model_expanded_lr_manifest.csv",
            "phase12_atlas_model_expanded_status.csv",
            "phase12_atlas_model_expanded_status.md",
        ],
        "claim_boundary": "Expanded covered-LR run; formal permutation-calibrated FDR and baseline whole-atlas comparison remain separate Phase 12 requirements.",
    }
    (OUT_DIR / "phase12_atlas_model_expanded_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
