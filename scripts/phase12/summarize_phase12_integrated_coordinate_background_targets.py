#!/usr/bin/env python
"""Summarize broad/background LR leakage targets for the integrated coordinate rerun."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
IN_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_leakage"
GATE_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_magnitude_gate"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_background_targets"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leakage_path = IN_DIR / "phase12_integrated_coordinate_leakage_lr.csv"
    gate_path = GATE_DIR / "phase12_integrated_coordinate_magnitude_gate_best.csv"
    gate_lr_path = GATE_DIR / "phase12_integrated_coordinate_magnitude_gate_lr.csv"
    leakage = pd.read_csv(leakage_path)
    gate = pd.read_csv(gate_path) if gate_path.exists() else pd.DataFrame()
    gate_lr = pd.read_csv(gate_lr_path) if gate_lr_path.exists() else pd.DataFrame()

    leakage["p_value"] = pd.to_numeric(leakage["p_value"], errors="coerce")
    leakage["calibrated_score"] = pd.to_numeric(leakage["calibrated_score"], errors="coerce")
    leakage["rank"] = pd.to_numeric(leakage["rank"], errors="coerce")
    leakage["true_lr"] = leakage["true_lr"].astype(bool)
    leakage["broad_background_lr"] = leakage["broad_background_lr"].astype(bool)
    leakage["high_expression_negative"] = leakage["high_expression_negative"].astype(bool)

    min_p = float(leakage["p_value"].min())
    gate_threshold = float(gate.iloc[0]["score_threshold"]) if len(gate) else np.nan
    if len(gate_lr) and "true_lr" in gate_lr.columns:
        gate_lr["true_lr"] = gate_lr["true_lr"].astype(bool)
        gate_lr["calibrated_score"] = pd.to_numeric(gate_lr["calibrated_score"], errors="coerce")
        reference_true_score = float(gate_lr.loc[gate_lr["true_lr"], "calibrated_score"].min())
    else:
        reference_true_score = float(leakage.loc[leakage["true_lr"], "calibrated_score"].max())

    broad = leakage[leakage["broad_background_lr"]].copy()
    broad["is_min_p"] = np.isclose(broad["p_value"], min_p)
    broad["score_ratio_to_retained_true_floor"] = (
        broad["calibrated_score"] / reference_true_score if reference_true_score else np.nan
    )
    broad["score_ratio_to_gate_threshold"] = (
        broad["calibrated_score"] / gate_threshold if gate_threshold and not np.isnan(gate_threshold) else np.nan
    )
    broad["recommended_guard"] = np.where(
        broad["calibrated_score"] < gate_threshold,
        "magnitude_guard_suppresses",
        "needs_additional_background_guard",
    )
    broad["next_training_use"] = np.where(
        broad["is_min_p"],
        "priority_broad_background_hard_negative",
        "background_pool_negative",
    )

    target_cols = [
        "dataset_key",
        "rank",
        "lr_pair",
        "biological_category",
        "mechanism",
        "selected_reason",
        "p_value",
        "bh_fdr",
        "calibrated_score",
        "null_q95_calibrated_score",
        "score_ratio_to_retained_true_floor",
        "score_ratio_to_gate_threshold",
        "recommended_guard",
        "next_training_use",
    ]
    target_cols = [c for c in target_cols if c in broad.columns]
    broad_targets = broad.sort_values(["is_min_p", "rank"], ascending=[False, True]).head(50)
    broad_targets[target_cols].to_csv(
        OUT_DIR / "phase12_integrated_coordinate_background_target_lr.csv",
        index=False,
    )

    category = (
        broad.assign(min_p_hit=broad["is_min_p"].astype(int))
        .groupby(["dataset_key", "biological_category"], dropna=False)
        .agg(
            n_broad_background=("lr_pair", "count"),
            n_min_p_broad_background=("min_p_hit", "sum"),
            min_rank=("rank", "min"),
            max_score=("calibrated_score", "max"),
            median_score=("calibrated_score", "median"),
        )
        .reset_index()
        .sort_values(["n_min_p_broad_background", "max_score"], ascending=[False, False])
    )
    category.to_csv(OUT_DIR / "phase12_integrated_coordinate_background_category_summary.csv", index=False)

    summary_rows = []
    for dataset_key, group in leakage.groupby("dataset_key"):
        broad_group = group[group["broad_background_lr"]]
        high_group = group[group["high_expression_negative"]]
        min_p_group = group[np.isclose(group["p_value"], min_p)]
        max_broad_score = float(broad_group["calibrated_score"].max()) if len(broad_group) else np.nan
        score_gap = (
            reference_true_score / max_broad_score
            if max_broad_score and not np.isnan(max_broad_score) and reference_true_score
            else np.nan
        )
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(group)),
                "minimum_p_value": min_p,
                "n_min_p_lr": int(len(min_p_group)),
                "n_min_p_true_lr": int(min_p_group["true_lr"].sum()),
                "n_min_p_high_expression_negative": int(min_p_group["high_expression_negative"].sum()),
                "n_min_p_broad_background": int(min_p_group["broad_background_lr"].sum()),
                "n_broad_background_top10": int((broad_group["rank"] <= 10).sum()),
                "n_broad_background_top50": int((broad_group["rank"] <= 50).sum()),
                "max_broad_background_score": max_broad_score,
                "retained_true_score_floor": reference_true_score,
                "retained_true_to_max_broad_score_gap": score_gap,
                "magnitude_gate_threshold": gate_threshold,
                "n_broad_background_above_gate": int((broad_group["calibrated_score"] >= gate_threshold).sum())
                if not np.isnan(gate_threshold)
                else -1,
                "n_high_expression_negative_above_gate": int((high_group["calibrated_score"] >= gate_threshold).sum())
                if not np.isnan(gate_threshold)
                else -1,
                "target_call": (
                    "magnitude_separates_true_from_broad_background"
                    if not np.isnan(score_gap) and score_gap > 10 and not np.isnan(gate_threshold)
                    else "needs_background_guard_design"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "phase12_integrated_coordinate_background_target_summary.csv", index=False)

    status_lines = [
        "# Integrated Coordinate Background Target Audit",
        "",
        "This is a Goal 1 diagnostic for the integrated guarded spatial coordinate rerun.",
        "It identifies broad/background LR that tie true LR under the low-permutation coordinate p-value.",
        "",
    ]
    if len(summary):
        first = summary.iloc[0]
        status_lines.extend(
            [
                f"- Dataset: `{first['dataset_key']}`",
                f"- Minimum-p LR: {int(first['n_min_p_lr'])}",
                f"- Minimum-p true LR: {int(first['n_min_p_true_lr'])}",
                f"- Minimum-p high-expression negatives: {int(first['n_min_p_high_expression_negative'])}",
                f"- Minimum-p broad/background LR: {int(first['n_min_p_broad_background'])}",
                f"- Broad/background top10 LR: {int(first['n_broad_background_top10'])}",
                f"- Broad/background top50 LR: {int(first['n_broad_background_top50'])}",
                f"- Retained true score floor / max broad-background score: {float(first['retained_true_to_max_broad_score_gap']):.2f}x",
                f"- Broad/background LR above magnitude gate: {int(first['n_broad_background_above_gate'])}",
                "",
                "Claim boundary: this supports a next-model guardrail target. It does not close whole-atlas FDR.",
            ]
        )
    (OUT_DIR / "phase12_integrated_coordinate_background_target_status.md").write_text(
        "\n".join(status_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "task": "phase12_integrated_coordinate_background_targets",
        "inputs": [rel(leakage_path), rel(gate_path), rel(gate_lr_path)],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_coordinate_background_target_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_coordinate_background_category_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_coordinate_background_target_lr.csv"),
            rel(OUT_DIR / "phase12_integrated_coordinate_background_target_status.md"),
        ],
        "claim_boundary": "Diagnostic target list for suppressing broad/background leakage; not a final FDR procedure.",
    }
    (OUT_DIR / "phase12_integrated_coordinate_background_target_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(category.head(10).to_string(index=False))


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
