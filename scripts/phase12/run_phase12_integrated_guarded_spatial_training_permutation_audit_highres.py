#!/usr/bin/env python
"""Higher-permutation exact-score audit for the integrated candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_integrated_guarded_spatial_training_permutation_audit as base  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_training_permutation_audit_highres"
N_PERMUTATIONS = 240
RNG_SEED = 12122026 + 909


def main() -> None:
    base.OUT_DIR = OUT_DIR
    base.N_PERMUTATIONS = N_PERMUTATIONS
    base.RNG_SEED = RNG_SEED
    base.main()
    rename_highres_outputs()
    write_manifest()


def rename_highres_outputs() -> None:
    mapping = {
        "phase12_integrated_guarded_spatial_training_permutation_lr_fdr.csv": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_lr_fdr.csv"
        ),
        "phase12_integrated_guarded_spatial_training_permutation_topk_summary.csv": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_topk_summary.csv"
        ),
        "phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_dataset_summary.csv"
        ),
        "phase12_integrated_guarded_spatial_training_permutation_null_summary.csv": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_null_summary.csv"
        ),
        "phase12_integrated_guarded_spatial_training_permutation_story.csv": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_story.csv"
        ),
        "phase12_integrated_guarded_spatial_training_permutation_status.md": (
            "phase12_integrated_guarded_spatial_training_highres_permutation_status.md"
        ),
    }
    for src_name, dst_name in mapping.items():
        src = OUT_DIR / src_name
        dst = OUT_DIR / dst_name
        if src.exists():
            src.replace(dst)
    dataset_path = OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_dataset_summary.csv"
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        for col in [
            "spatial_specificity_permutation_call",
            "integrated_guarded_spatial_training_permutation_call",
        ]:
            if col in dataset.columns:
                dataset["integrated_guarded_spatial_training_highres_permutation_call"] = dataset[col].astype(str).str.replace(
                    "integrated_guarded_spatial_training",
                    "integrated_guarded_spatial_training_highres",
                    regex=False,
                )
                break
        dataset.to_csv(dataset_path, index=False)
    story_path = OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_story.csv"
    if story_path.exists():
        story = pd.read_csv(story_path)
        story["topic"] = "integrated_guarded_spatial_training_highres_permutation"
        story["claim_boundary"] = (
            "Higher-permutation exact-score audit for candidate integrated guarded spatial training; "
            "not end-to-end retraining FDR."
        )
        story.to_csv(story_path, index=False)


def write_manifest() -> None:
    manifest = {
        "task": "phase12_integrated_guarded_spatial_training_highres_permutation_audit",
        "n_permutations_per_null": N_PERMUTATIONS,
        "null_types": list(base.spatial_perm.NULL_TYPES),
        "inputs": [
            rel(base.MODEL_DIR / "phase12_integrated_guarded_spatial_training_lr_ranking.csv"),
            rel(base.MODEL_DIR / "replicates/*/candidate_interactions.csv"),
            rel(base.MODEL_DIR / "replicates/*/lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_lr_fdr.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_topk_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_null_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_story.csv"),
        ],
        "claim_boundary": "Higher-permutation exact-score candidate audit; not end-to-end FDR.",
    }
    (OUT_DIR / "phase12_integrated_guarded_spatial_training_highres_permutation_manifest.json").write_text(
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
