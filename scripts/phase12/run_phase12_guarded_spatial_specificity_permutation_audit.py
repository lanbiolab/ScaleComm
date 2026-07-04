#!/usr/bin/env python
"""Exact-score audit for guarded spatial-specificity calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_spatial_specificity_permutation_audit as spatial_perm  # noqa: E402
from scripts.phase12.calibrate_phase12_guarded_spatial_specificity import (  # noqa: E402
    add_guarded_spatial_specificity_calibration,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = PHASE12_DIR / "guarded_spatial_specificity_permutation_audit"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_perm.OUT_DIR = OUT_DIR
    spatial_perm.N_PERMUTATIONS = 80
    spatial_perm.RNG_SEED = 12122026 + 404
    spatial_perm.add_spatial_specificity_calibration = add_guarded_spatial_specificity_calibration
    spatial_perm.score_table = guarded_score_table
    spatial_perm.main()
    rename_outputs()
    write_manifest()


def guarded_score_table(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for col in spatial_perm.FEATURE_COLS:
        out[col] = features[col].to_numpy()
    out = add_guarded_spatial_specificity_calibration(out)
    out["method"] = "spatialself_wavelet_edge_prior_guarded_spatial_specificity_permutation_audit"
    out["rank"] = out["calibrated_score"].rank(ascending=False, method="min").astype(int)
    out["strict_rank"] = out["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    out["rank_q95"] = out["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    return out


def rename_outputs() -> None:
    mapping = {
        "phase12_spatial_specificity_permutation_lr_fdr.csv": "phase12_guarded_spatial_specificity_permutation_lr_fdr.csv",
        "phase12_spatial_specificity_permutation_topk_summary.csv": "phase12_guarded_spatial_specificity_permutation_topk_summary.csv",
        "phase12_spatial_specificity_permutation_dataset_summary.csv": "phase12_guarded_spatial_specificity_permutation_dataset_summary.csv",
        "phase12_spatial_specificity_permutation_null_summary.csv": "phase12_guarded_spatial_specificity_permutation_null_summary.csv",
        "phase12_spatial_specificity_permutation_story.csv": "phase12_guarded_spatial_specificity_permutation_story.csv",
        "phase12_spatial_specificity_permutation_status.md": "phase12_guarded_spatial_specificity_permutation_status.md",
    }
    for src_name, dst_name in mapping.items():
        src = OUT_DIR / src_name
        dst = OUT_DIR / dst_name
        if src.exists():
            src.replace(dst)
    dataset_path = OUT_DIR / "phase12_guarded_spatial_specificity_permutation_dataset_summary.csv"
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path)
        if "spatial_specificity_permutation_call" in dataset.columns:
            dataset["guarded_spatial_specificity_permutation_call"] = dataset[
                "spatial_specificity_permutation_call"
            ].astype(str).str.replace("spatial_specificity", "guarded_spatial_specificity", regex=False)
            dataset.to_csv(dataset_path, index=False)
    story_path = OUT_DIR / "phase12_guarded_spatial_specificity_permutation_story.csv"
    if story_path.exists():
        story = pd.read_csv(story_path)
        story["topic"] = "guarded_spatial_specificity_permutation"
        story["claim_boundary"] = (
            "Exact-score post-hoc guarded spatial-specificity audit; not an end-to-end discovery-FDR procedure."
        )
        story.to_csv(story_path, index=False)


def write_manifest() -> None:
    manifest = {
        "task": "phase12_guarded_spatial_specificity_permutation_audit",
        "n_permutations_per_null": spatial_perm.N_PERMUTATIONS,
        "null_types": list(spatial_perm.NULL_TYPES),
        "inputs": [
            rel(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv"),
            rel(EXPANDED_DIR / "replicates/*/candidate_interactions.csv"),
            rel(EXPANDED_DIR / "replicates/*/lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_permutation_lr_fdr.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_permutation_topk_summary.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_permutation_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_permutation_null_summary.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_permutation_story.csv"),
        ],
        "claim_boundary": "Post-hoc exact-score guarded spatial-specificity audit, not end-to-end FDR.",
    }
    (OUT_DIR / "phase12_guarded_spatial_specificity_permutation_manifest.json").write_text(
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
