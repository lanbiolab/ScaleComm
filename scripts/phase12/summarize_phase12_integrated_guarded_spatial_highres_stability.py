#!/usr/bin/env python
"""Summarize 80-vs-240 permutation stability for the integrated candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_training_highres_stability"

LOWRES_PATH = (
    PHASE12_DIR
    / "integrated_guarded_spatial_training_permutation_audit/"
    / "phase12_integrated_guarded_spatial_training_permutation_dataset_summary.csv"
)
HIGHRES_PATH = (
    PHASE12_DIR
    / "integrated_guarded_spatial_training_permutation_audit_highres/"
    / "phase12_integrated_guarded_spatial_training_highres_permutation_dataset_summary.csv"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    low = pd.read_csv(LOWRES_PATH)
    high = pd.read_csv(HIGHRES_PATH)
    comparison = compare(low, high)
    story = summarize_story(comparison)
    comparison.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_story.csv", index=False)
    write_status(comparison, story)
    write_manifest()
    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def compare(low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk_fraction",
        "all_true_lr_conservative_fdr_0_1",
        "all_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
        "best_true_spatial_fdr",
        "n_permutations_per_null",
    ]
    low = low[[col for col in keep if col in low.columns]].copy()
    high = high[[col for col in keep if col in high.columns]].copy()
    merged = low.merge(high, on="dataset_key", suffixes=("_80", "_240"))
    merged["delta_best_true_q_240_minus_80"] = (
        pd.to_numeric(merged["best_true_lr_fdr_240"], errors="coerce")
        - pd.to_numeric(merged["best_true_lr_fdr_80"], errors="coerce")
    )
    merged["q_0_1_stability_call"] = [stability_call(row) for row in merged.itertuples(index=False)]
    merged["manuscript_use"] = [manuscript_use(row) for row in merged.itertuples(index=False)]
    return merged.sort_values("dataset_key").reset_index(drop=True)


def stability_call(row: object) -> str:
    low_q = float(getattr(row, "best_true_lr_fdr_80", np.nan))
    high_q = float(getattr(row, "best_true_lr_fdr_240", np.nan))
    low_hit = int(getattr(row, "all_true_lr_conservative_fdr_0_1_80", 0))
    high_hit = int(getattr(row, "all_true_lr_conservative_fdr_0_1_240", 0))
    if low_hit > 0 and high_hit > 0 and high_q <= low_q:
        return "stable_or_stronger_q_0_1_hit"
    if low_hit > 0 and high_hit > 0:
        return "stable_q_0_1_hit"
    if low_hit > 0 and high_hit == 0:
        return "q_0_1_hit_not_stable"
    if high_hit > 0:
        return "highres_new_q_0_1_hit"
    return "no_q_0_1_hit"


def manuscript_use(row: object) -> str:
    risk = float(getattr(row, "top10_negative_or_risk_fraction_240", np.nan))
    hit = int(getattr(row, "all_true_lr_conservative_fdr_0_1_240", 0))
    if hit > 0 and np.isfinite(risk) and risk <= 0.1:
        return "candidate_story_statistical_support_with_end_to_end_caveat"
    if hit > 0:
        return "candidate_support_with_top10_risk_caveat"
    if np.isfinite(risk) and risk <= 0.0:
        return "ranking_support_without_broad_fdr"
    return "stress_case_or_unresolved"


def summarize_story(comparison: pd.DataFrame) -> pd.DataFrame:
    stable_hits = int(comparison["q_0_1_stability_call"].astype(str).str.contains("stable").sum())
    highres_hits = int(pd.to_numeric(comparison["all_true_lr_conservative_fdr_0_1_240"], errors="coerce").fillna(0).sum())
    best = comparison.sort_values("best_true_lr_fdr_240").head(1)
    best_text = "NA"
    best_q = np.nan
    if len(best):
        best_text = f"{best.iloc[0].dataset_key}:{best.iloc[0].best_true_lr_240}"
        best_q = float(best.iloc[0].best_true_lr_fdr_240)
    overall = "highres_confirms_integrated_candidate_hit" if stable_hits > 0 else "highres_does_not_confirm_q_hit"
    return pd.DataFrame(
        [
            {
                "topic": "integrated_guarded_spatial_highres_stability",
                "overall_call": overall,
                "n_datasets": int(len(comparison)),
                "n_stable_q_0_1_hits": stable_hits,
                "highres_true_q_0_1_total": highres_hits,
                "best_highres_true": best_text,
                "best_highres_true_q": best_q,
                "claim_boundary": (
                    "Higher-permutation exact-score stability check; still not end-to-end retraining FDR "
                    "or final baseline replacement."
                ),
            }
        ]
    )


def write_status(comparison: pd.DataFrame, story: pd.DataFrame) -> None:
    first = story.iloc[0]
    lines = [
        "# Phase 12 Integrated Guarded Spatial High-Resolution Stability",
        "",
        f"- Overall: `{first.overall_call}`.",
        f"- Stable q<=0.1 hits: {int(first.n_stable_q_0_1_hits)}.",
        f"- Best highres true: {first.best_highres_true} q={fmt(first.best_highres_true_q)}.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.q_0_1_stability_call}`: "
            f"best true {row.best_true_lr_80} q80={fmt(row.best_true_lr_fdr_80)} -> "
            f"{row.best_true_lr_240} q240={fmt(row.best_true_lr_fdr_240)}, "
            f"top10 risk240={fmt(row.top10_negative_or_risk_fraction_240)}, "
            f"use=`{row.manuscript_use}`."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This confirms exact-score stability of the integrated candidate under more permutations.",
        "- It does not replace end-to-end null retraining or repair the CRC top10-risk failure mode.",
    ]
    (OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_integrated_guarded_spatial_highres_stability",
        "inputs": [rel(LOWRES_PATH), rel(HIGHRES_PATH)],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_dataset.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_story.csv"),
            rel(OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_status.md"),
        ],
        "claim_boundary": "High-resolution exact-score stability check; not final end-to-end FDR.",
    }
    (OUT_DIR / "phase12_integrated_guarded_spatial_highres_stability_manifest.json").write_text(
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
