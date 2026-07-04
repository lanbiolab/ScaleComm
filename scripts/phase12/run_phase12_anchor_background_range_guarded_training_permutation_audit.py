#!/usr/bin/env python
"""Exact-score permutation audit for anchor-background range-guarded training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm  # noqa: E402
from scripts.phase12.calibrate_phase12_expanded_atlas_background_suppressed import (  # noqa: E402
    add_background_suppression_calibration,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
MODEL_DIR = PHASE12_DIR / "atlas_model_expanded_anchor_background_range_guarded_training"
OUT_DIR = PHASE12_DIR / "anchor_background_range_guarded_training_permutation_audit"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perm.add_calibration = add_background_suppression_calibration
    perm.OUT_DIR = OUT_DIR

    ranking = pd.read_csv(MODEL_DIR / "phase12_anchor_background_range_guarded_training_lr_ranking.csv")
    status = pd.read_csv(MODEL_DIR / "phase12_anchor_background_range_guarded_training_status.csv")
    lr_rows = []
    null_summary_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        dataset_ranking = ranking[ranking["dataset_key"].eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        replicate_dir = MODEL_DIR / "replicates" / dataset_key
        result = perm.run_dataset(dataset_key, dataset_ranking, replicate_dir)
        lr_rows.append(result["lr_fdr"])
        null_summary_rows.append(result["null_summary"])

    lr_fdr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null_summary = pd.concat(null_summary_rows, ignore_index=True) if null_summary_rows else pd.DataFrame()
    if not lr_fdr.empty:
        lr_fdr["method"] = "spatialself_wavelet_edge_prior_anchor_background_range_guarded_training_permutation_audit"
    topk = perm.summarize_topk(lr_fdr)
    dataset = perm.summarize_dataset(lr_fdr, topk, null_summary, status)
    if not dataset.empty:
        dataset["anchor_background_range_guarded_permutation_call"] = dataset["permutation_audit_call"].map(
            lambda x: str(x).replace("expanded_permutation", "anchor_background_range_guarded_permutation")
        )

    lr_fdr.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_null_summary.csv", index=False)
    write_status(dataset, topk, null_summary)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr_fdr.groupby("dataset_key").head(15).to_string(index=False))


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, null_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor-Background Range-Guarded Training Permutation Audit",
        "",
        "## Scope",
        "",
        "This is an exact-score post-hoc permutation audit for the training-integrated anchor/background range-guarded range-guarded variant.",
        "It recomputes the background-suppressed calibration layer under receiver, spatial-distance and LR-identity nulls.",
        "It does not retrain SpatialSelf under each permutation.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        call = getattr(row, "anchor_background_range_guarded_permutation_call", row.permutation_audit_call)
        lines.append(
            f"- `{row.dataset_key}` `{call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 q<=0.1={int(row.top10_conservative_fdr_0_1)}, "
            f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
            f"best true LR={row.best_true_lr} q={perm.fmt(row.best_true_lr_fdr)}, "
            f"top10 risk={perm.fmt(row.top10_negative_or_risk_fraction)}"
        )
    lines += ["", "## Top-K Summary", ""]
    for row in topk[topk["top_k"].isin([10, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"risk={perm.fmt(row.negative_or_risk_fraction)}, q<=0.1={int(row.n_conservative_fdr_0_1)}, "
            f"median q={perm.fmt(row.median_conservative_fdr)}"
        )
    lines += ["", "## Null Summary", ""]
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.null_type}`: null q99={perm.fmt(row.null_score_q99)}, "
            f"null max={perm.fmt(row.null_score_max)}, observed top={perm.fmt(row.observed_top_score)}, "
            f"top>q99={bool(row.observed_top_exceeds_null_q99)}, top>max={bool(row.observed_top_exceeds_null_max)}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is a candidate model audit for Goal 1, not an end-to-end retraining FDR.",
        "- Do not claim broad whole-atlas FDR unless q<=0.1 discoveries and hard-control diagnostics are clean.",
    ]
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_anchor_background_range_guarded_training_permutation_audit",
        "n_permutations_per_null": perm.N_PERMUTATIONS,
        "null_types": list(perm.NULL_TYPES),
        "inputs": [
            "outputs/phase12/atlas_model_expanded_anchor_background_range_guarded_training/phase12_anchor_background_range_guarded_training_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded_anchor_background_range_guarded_training/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded_anchor_background_range_guarded_training/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            "phase12_anchor_background_range_guarded_training_permutation_lr_fdr.csv",
            "phase12_anchor_background_range_guarded_training_permutation_topk_summary.csv",
            "phase12_anchor_background_range_guarded_training_permutation_dataset_summary.csv",
            "phase12_anchor_background_range_guarded_training_permutation_null_summary.csv",
            "phase12_anchor_background_range_guarded_training_permutation_status.md",
        ],
        "claim_boundary": "Exact-score post-hoc audit for training-integrated anchor/background range-guarded variant; not end-to-end FDR.",
    }
    (OUT_DIR / "phase12_anchor_background_range_guarded_training_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
