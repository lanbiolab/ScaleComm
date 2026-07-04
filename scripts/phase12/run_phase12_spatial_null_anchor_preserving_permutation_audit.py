#!/usr/bin/env python
"""Permutation audit for the anchor-preserving spatial-null training variant.

This reuses the expanded-atlas post-hoc null machinery, but points it at the
new anchor-preserving model predictions and replicate directories. It recomputes
the standard atlas calibration layer under receiver, spatial-distance and
LR-identity nulls. It does not retrain SpatialSelf under each permutation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
ANCHOR_DIR = PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving"
OUT_DIR = PHASE12_DIR / "spatial_null_anchor_preserving_permutation_audit"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perm.OUT_DIR = OUT_DIR

    ranking = pd.read_csv(ANCHOR_DIR / "phase12_spatial_null_anchor_preserving_calibrated_lr_ranking.csv")
    status = pd.read_csv(ANCHOR_DIR / "phase12_spatial_null_anchor_preserving_status.csv")
    lr_rows = []
    null_summary_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        dataset_ranking = ranking[ranking["dataset_key"].eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        replicate_dir = ANCHOR_DIR / "replicates" / dataset_key
        result = perm.run_dataset(dataset_key, dataset_ranking, replicate_dir)
        lr_rows.append(result["lr_fdr"])
        null_summary_rows.append(result["null_summary"])

    lr_fdr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null_summary = pd.concat(null_summary_rows, ignore_index=True) if null_summary_rows else pd.DataFrame()
    if not lr_fdr.empty:
        lr_fdr["method"] = "spatialself_wavelet_edge_prior_spatial_null_anchor_preserving_permutation_audit"
    topk = perm.summarize_topk(lr_fdr)
    dataset = perm.summarize_dataset(lr_fdr, topk, null_summary, status)
    if not dataset.empty:
        dataset["anchor_preserving_permutation_call"] = dataset["permutation_audit_call"].map(
            lambda x: str(x).replace("expanded_permutation", "anchor_preserving_permutation")
        )

    lr_fdr.to_csv(OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_null_summary.csv", index=False)
    write_status(dataset, topk, null_summary)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr_fdr.groupby("dataset_key").head(15).to_string(index=False))


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, null_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor-Preserving Spatial-Null Permutation Audit",
        "",
        "## Scope",
        "",
        "This is an exact-score post-hoc permutation audit for the anchor-preserving spatial-null training variant.",
        "It recomputes the standard expanded-atlas calibration layer under receiver, spatial-distance and LR-identity nulls.",
        "It does not retrain SpatialSelf under each permutation.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        call = getattr(row, "anchor_preserving_permutation_call", row.permutation_audit_call)
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
        "- This tests whether the anchor-preserving spatial-null trained model improves exact-score null support.",
        "- It is still post-hoc and not an end-to-end model-rerun permutation FDR.",
        "- Do not adopt this variant as the reporting baseline unless exact-score/control diagnostics and CRC recovery are acceptable.",
    ]
    (OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_spatial_null_anchor_preserving_permutation_audit",
        "n_permutations_per_null": perm.N_PERMUTATIONS,
        "null_types": list(perm.NULL_TYPES),
        "inputs": [
            "outputs/phase12/atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded_spatial_null_anchor_preserving/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded_spatial_null_anchor_preserving/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            "phase12_spatial_null_anchor_preserving_permutation_lr_fdr.csv",
            "phase12_spatial_null_anchor_preserving_permutation_topk_summary.csv",
            "phase12_spatial_null_anchor_preserving_permutation_dataset_summary.csv",
            "phase12_spatial_null_anchor_preserving_permutation_null_summary.csv",
            "phase12_spatial_null_anchor_preserving_permutation_status.md",
        ],
        "claim_boundary": "Exact-score post-hoc null audit for anchor-preserving spatial-null variant; not end-to-end retraining FDR.",
    }
    (OUT_DIR / "phase12_spatial_null_anchor_preserving_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
