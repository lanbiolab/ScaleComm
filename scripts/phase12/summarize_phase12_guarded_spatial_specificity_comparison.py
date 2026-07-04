#!/usr/bin/env python
"""Compare guarded spatial-specificity calibration with main and unguarded spatial-specificity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
GUARDED_DIR = PHASE12_DIR / "guarded_spatial_specificity_permutation_audit"
SPATIAL_DIR = PHASE12_DIR / "spatial_specificity_permutation_audit"
MAIN_DIR = PHASE12_DIR / "atlas_model_expanded/permutation_audit"
OUT_DIR = PHASE12_DIR / "guarded_spatial_specificity_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guarded = pd.read_csv(GUARDED_DIR / "phase12_guarded_spatial_specificity_permutation_dataset_summary.csv")
    spatial = pd.read_csv(SPATIAL_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv")
    main_perm = pd.read_csv(MAIN_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv")
    dataset = build_dataset(guarded, spatial, main_perm)
    story = build_story(dataset)
    dataset.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_guarded_spatial_specificity_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_dataset(guarded: pd.DataFrame, spatial: pd.DataFrame, main_perm: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk_fraction",
        "all_true_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
    ]
    out = guarded[cols].merge(
        spatial[cols],
        on="dataset_key",
        how="left",
        suffixes=("_guarded", "_spatial"),
    )
    out = out.merge(
        main_perm[cols],
        on="dataset_key",
        how="left",
    ).rename(
        columns={
            "top10_true_lr": "top10_true_lr_main",
            "top10_negative_or_risk_fraction": "top10_negative_or_risk_fraction_main",
            "all_true_lr_conservative_fdr_0_1": "all_true_lr_conservative_fdr_0_1_main",
            "best_true_lr": "best_true_lr_main",
            "best_true_lr_fdr": "best_true_lr_fdr_main",
        }
    )
    out["delta_top10_true_vs_main"] = num(out["top10_true_lr_guarded"]) - num(out["top10_true_lr_main"])
    out["delta_top10_risk_vs_main"] = num(out["top10_negative_or_risk_fraction_guarded"]) - num(
        out["top10_negative_or_risk_fraction_main"]
    )
    out["delta_best_true_q_vs_main"] = num(out["best_true_lr_fdr_guarded"]) - num(out["best_true_lr_fdr_main"])
    out["delta_top10_true_vs_spatial"] = num(out["top10_true_lr_guarded"]) - num(out["top10_true_lr_spatial"])
    out["delta_top10_risk_vs_spatial"] = num(out["top10_negative_or_risk_fraction_guarded"]) - num(
        out["top10_negative_or_risk_fraction_spatial"]
    )
    out["delta_best_true_q_vs_spatial"] = num(out["best_true_lr_fdr_guarded"]) - num(out["best_true_lr_fdr_spatial"])
    out["best_true_q_gap_to_0_1"] = num(out["best_true_lr_fdr_guarded"]) / 0.1
    out["dataset_decision"] = [dataset_decision(row) for row in out.itertuples(index=False)]
    return out.sort_values("dataset_key")


def build_story(dataset: pd.DataFrame) -> pd.DataFrame:
    true_q = int(num(dataset["all_true_lr_conservative_fdr_0_1_guarded"]).sum())
    q_improved_vs_main = int(num(dataset["delta_best_true_q_vs_main"]).lt(0).sum())
    q_improved_vs_spatial = int(num(dataset["delta_best_true_q_vs_spatial"]).lt(0).sum())
    risk_improved_vs_spatial = int(num(dataset["delta_top10_risk_vs_spatial"]).lt(0).sum())
    no_added_risk_vs_main = int(num(dataset["delta_top10_risk_vs_main"]).le(0).sum())
    best_q = float(num(dataset["best_true_lr_fdr_guarded"]).min())
    if true_q > 0:
        call = "guarded_spatial_specificity_fdr_candidate"
    elif best_q <= 0.12 and risk_improved_vs_spatial > 0:
        call = "guarded_spatial_specificity_near_fdr_not_closed"
    else:
        call = "guarded_spatial_specificity_diagnostic_not_closed"
    return pd.DataFrame(
        [
            {
                "topic": "guarded_spatial_specificity_comparison",
                "overall_call": call,
                "n_datasets": int(len(dataset)),
                "n_true_q_0_1": true_q,
                "n_best_true_q_improved_vs_main": q_improved_vs_main,
                "n_best_true_q_improved_vs_spatial_specificity": q_improved_vs_spatial,
                "n_top10_risk_improved_vs_spatial_specificity": risk_improved_vs_spatial,
                "n_no_added_top10_risk_vs_main": no_added_risk_vs_main,
                "best_true_q_min": best_q,
                "best_true_q_gap_to_0_1": best_q / 0.1,
                "claim_boundary": (
                    "Guarded spatial-specificity improves risk over the unguarded spatial-specificity score "
                    "and nearly closes post-Xenium, but broad whole-atlas FDR remains unclosed."
                ),
            }
        ]
    )


def dataset_decision(row: object) -> str:
    true_q = int(getattr(row, "all_true_lr_conservative_fdr_0_1_guarded", 0))
    risk_guarded = float(getattr(row, "top10_negative_or_risk_fraction_guarded", 1.0))
    delta_q_main = float(getattr(row, "delta_best_true_q_vs_main", 0.0))
    delta_q_spatial = float(getattr(row, "delta_best_true_q_vs_spatial", 0.0))
    delta_risk_spatial = float(getattr(row, "delta_top10_risk_vs_spatial", 0.0))
    if true_q > 0 and risk_guarded <= 0.1:
        return "candidate_fdr_support_clean_top10"
    if delta_q_main < 0 and delta_q_spatial < 0 and delta_risk_spatial < 0:
        return "q_gap_and_risk_improved_not_closed"
    if delta_q_main < 0 and risk_guarded <= 0.1:
        return "q_gap_improved_clean_top10_not_closed"
    if delta_q_main < 0:
        return "q_gap_improved_with_caveat"
    return "no_q_gap_improvement"


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    first = story.iloc[0]
    lines = [
        "# Phase 12 Guarded Spatial-Specificity Comparison",
        "",
        f"- Overall: `{first.overall_call}`.",
        f"- True q<=0.1 discoveries: {int(first.n_true_q_0_1)}.",
        f"- Best true q: {fmt(first.best_true_q_min)} ({fmt(first.best_true_q_gap_to_0_1)}x over q<=0.1).",
        "",
        "## Dataset Decisions",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_decision}`: "
            f"main q={fmt(row.best_true_lr_fdr_main)}, spatial q={fmt(row.best_true_lr_fdr_spatial)}, "
            f"guarded q={fmt(row.best_true_lr_fdr_guarded)}; "
            f"risk main/spatial/guarded={fmt(row.top10_negative_or_risk_fraction_main)}/"
            f"{fmt(row.top10_negative_or_risk_fraction_spatial)}/{fmt(row.top10_negative_or_risk_fraction_guarded)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This is a post-hoc exact-score diagnostic, not a replacement baseline.",
        "- Goal 1 remains partial unless broad true q<=0.1 discoveries are obtained under conservative nulls.",
    ]
    (OUT_DIR / "phase12_guarded_spatial_specificity_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_guarded_spatial_specificity_comparison",
        "inputs": [
            rel(GUARDED_DIR / "phase12_guarded_spatial_specificity_permutation_dataset_summary.csv"),
            rel(SPATIAL_DIR / "phase12_spatial_specificity_permutation_dataset_summary.csv"),
            rel(MAIN_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_dataset_comparison.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_story.csv"),
            rel(OUT_DIR / "phase12_guarded_spatial_specificity_comparison_status.md"),
        ],
        "claim_boundary": "Diagnostic comparison only; no broad FDR closure.",
    }
    (OUT_DIR / "phase12_guarded_spatial_specificity_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def num(series: object) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
