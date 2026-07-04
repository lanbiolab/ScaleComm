#!/usr/bin/env python
"""Background-suppressed calibration for the anchor-preserving variant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.calibrate_phase12_expanded_atlas_background_suppressed as bg  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
ANCHOR_DIR = PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving"
OUT_DIR = PHASE12_DIR / "atlas_model_expanded_anchor_background_suppressed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ANCHOR_DIR / "phase12_spatial_null_anchor_preserving_calibrated_lr_ranking.csv")
    tables = []
    for dataset_key, sub in source.groupby("dataset_key", sort=False):
        table = bg.apply_background_suppression(str(dataset_key), sub.copy())
        table["method"] = "spatialself_wavelet_edge_prior_anchor_background_suppressed"
        tables.append(table)
    ranking = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    topk = bg.summarize_topk(ranking)
    dataset = bg.summarize_dataset(ranking, topk)

    ranking.to_csv(OUT_DIR / "phase12_anchor_background_suppressed_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_anchor_background_suppressed_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_anchor_background_suppressed_dataset_summary.csv", index=False)
    write_status(dataset, topk)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(ranking.groupby("dataset_key").head(15).to_string(index=False))


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Anchor Background-Suppressed Atlas Calibration",
        "",
        "## Scope",
        "",
        "This applies the existing background-suppression calibration to the anchor-preserving spatial-null trained scores.",
        "It is a post-hoc calibration candidate, not a new end-to-end model.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.background_suppressed_call}`: "
            f"top10 true={int(row.top10_true_lr)}, "
            f"top10 broad-other={int(row.top10_broad_other_lr)}, "
            f"top10 control FDR={bg.fmt(row.top10_control_panel_empirical_fdr)}, "
            f"top100 true={int(row.top100_true_lr)}."
        )
    lines += ["", "## Top-K", ""]
    for row in topk[topk["top_k"].isin([10, 20, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"broad-other={int(row.n_broad_other_lr)}, "
            f"high-expression negatives={int(row.n_high_expression_negative)}, "
            f"control FDR={bg.fmt(row.control_panel_empirical_fdr)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this only as a model/calibration diagnostic unless exact-score or end-to-end null audits support it.",
        "- Do not claim broad whole-atlas FDR from top-k improvements alone.",
    ]
    (OUT_DIR / "phase12_anchor_background_suppressed_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_anchor_background_suppressed_calibration",
        "inputs": [
            "outputs/phase12/atlas_model_expanded_spatial_null_anchor_preserving/phase12_spatial_null_anchor_preserving_calibrated_lr_ranking.csv",
        ],
        "outputs": [
            "phase12_anchor_background_suppressed_lr_ranking.csv",
            "phase12_anchor_background_suppressed_topk_summary.csv",
            "phase12_anchor_background_suppressed_dataset_summary.csv",
            "phase12_anchor_background_suppressed_status.md",
        ],
        "claim_boundary": "Post-hoc background suppression on anchor-preserving trained scores; not FDR closure.",
    }
    (OUT_DIR / "phase12_anchor_background_suppressed_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
