#!/usr/bin/env python
"""Summarize broad/background leakage in integrated coordinate-rerun smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_leakage"
SMOKE_ROOT = PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdr_path = best_fdr_path()
    if fdr_path is None:
        raise FileNotFoundError(f"No integrated coordinate FDR files found under {SMOKE_ROOT}")
    fdr = pd.read_csv(fdr_path)
    lr = classify_lr(fdr)
    summary = summarize(lr, fdr_path)
    low_p = low_p_table(lr)

    lr.to_csv(OUT_DIR / "phase12_integrated_coordinate_leakage_lr.csv", index=False)
    summary.to_csv(OUT_DIR / "phase12_integrated_coordinate_leakage_summary.csv", index=False)
    low_p.to_csv(OUT_DIR / "phase12_integrated_coordinate_leakage_low_p_lr.csv", index=False)
    write_status(summary, low_p)
    write_manifest(fdr_path)

    print(summary.to_string(index=False))
    print()
    print(low_p.head(40).to_string(index=False))


def best_fdr_path() -> Path | None:
    paths = sorted(SMOKE_ROOT.glob("*/*_integrated_coordinate_lr_fdr.csv"))
    if not paths:
        return None
    rows = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["n_completed_permutations"])
        n_perm = int(pd.to_numeric(frame["n_completed_permutations"], errors="coerce").fillna(0).max())
        rows.append((n_perm, path))
    rows.sort(key=lambda item: (item[0], str(item[1])))
    return rows[-1][1]


def classify_lr(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["p_value"] = pd.to_numeric(out["coordinate_rerun_p_value"], errors="coerce")
    out["bh_fdr"] = pd.to_numeric(out["coordinate_rerun_bh_fdr"], errors="coerce")
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    selected = out["selected_reason"].astype(str)
    category = out["biological_category"].astype(str)
    out["broad_background_lr"] = (
        ~out["true_lr"]
        & ~out["false_positive_high_expression"]
        & (
            selected.eq("category_balanced")
            | category.eq("other")
            | out["guarded_spatial_specificity_note"].astype(str).str.contains("broad|category", case=False, na=False)
        )
    )
    out["priority_true_lr"] = out["true_lr"] & selected.eq("phase12_priority_anchor")
    out["high_expression_negative"] = out["false_positive_high_expression"]
    return out


def summarize(lr: pd.DataFrame, fdr_path: Path) -> pd.DataFrame:
    p_min = float(pd.to_numeric(lr["p_value"], errors="coerce").dropna().min())
    low = lr[lr["p_value"].le(p_min + 1e-12)].copy()
    true_low = int(low["true_lr"].sum())
    high_neg_low = int(low["high_expression_negative"].sum())
    broad_low = int(low["broad_background_lr"].sum())
    other_low = int(len(low) - true_low - high_neg_low - broad_low)
    rows = [
        {
            "dataset_key": str(lr["dataset_key"].iloc[0]) if len(lr) else "",
            "source_fdr_file": rel(fdr_path),
            "n_lr": int(len(lr)),
            "n_completed_permutations": int(pd.to_numeric(lr["n_completed_permutations"], errors="coerce").max()),
            "minimum_p_value": p_min,
            "n_min_p_lr": int(len(low)),
            "n_min_p_true_lr": true_low,
            "n_min_p_high_expression_negative": high_neg_low,
            "n_min_p_broad_background": broad_low,
            "n_min_p_other": other_low,
            "min_p_true_fraction": safe_div(true_low, len(low)),
            "min_p_broad_background_fraction": safe_div(broad_low, len(low)),
            "best_true_lr": best_lr(lr[lr["true_lr"]]),
            "best_true_p": best_value(lr[lr["true_lr"]], "p_value"),
            "best_true_bh_fdr": best_value(lr[lr["true_lr"]], "bh_fdr"),
            "best_high_expression_negative_lr": best_lr(lr[lr["high_expression_negative"]]),
            "best_high_expression_negative_p": best_value(lr[lr["high_expression_negative"]], "p_value"),
            "best_broad_background_lr": best_lr(lr[lr["broad_background_lr"]]),
            "best_broad_background_p": best_value(lr[lr["broad_background_lr"]], "p_value"),
            "leakage_call": leakage_call(true_low, high_neg_low, broad_low, len(low)),
        }
    ]
    return pd.DataFrame(rows)


def low_p_table(lr: pd.DataFrame) -> pd.DataFrame:
    p_min = float(pd.to_numeric(lr["p_value"], errors="coerce").dropna().min())
    cols = [
        "dataset_key",
        "rank",
        "lr_pair",
        "true_lr",
        "high_expression_negative",
        "broad_background_lr",
        "biological_category",
        "selected_reason",
        "p_value",
        "bh_fdr",
        "calibrated_score",
        "null_q95_calibrated_score",
    ]
    return lr[lr["p_value"].le(p_min + 1e-12)][cols].sort_values(["rank", "lr_pair"]).reset_index(drop=True)


def leakage_call(true_low: int, high_neg_low: int, broad_low: int, n_low: int) -> str:
    if true_low > 0 and high_neg_low == 0 and broad_low == 0:
        return "true_min_p_clean"
    if true_low > 0 and high_neg_low == 0 and broad_low > 0:
        return "true_min_p_high_expression_clean_broad_background_leakage"
    if true_low > 0:
        return "true_min_p_with_negative_leakage"
    if n_low > 0:
        return "no_true_min_p"
    return "no_low_p_rows"


def best_lr(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    return str(table.sort_values(["p_value", "rank"]).iloc[0]["lr_pair"])


def best_value(table: pd.DataFrame, column: str) -> float:
    if table.empty:
        return float("nan")
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def safe_div(num: int | float, den: int | float) -> float:
    den = float(den)
    return float(num) / den if den else float("nan")


def write_status(summary: pd.DataFrame, low_p: pd.DataFrame) -> None:
    row = summary.iloc[0]
    lines = [
        "# Phase 12 Integrated Coordinate Leakage Audit",
        "",
        "## Result",
        "",
        f"- Dataset: `{row.dataset_key}`",
        f"- Coordinate reruns: {int(row.n_completed_permutations)}",
        f"- Minimum p-value: {float(row.minimum_p_value):.4g}",
        f"- Minimum-p LR: {int(row.n_min_p_lr)} total; true={int(row.n_min_p_true_lr)}, high-expression negative={int(row.n_min_p_high_expression_negative)}, broad/background={int(row.n_min_p_broad_background)}",
        f"- Best true: `{row.best_true_lr}` p={float(row.best_true_p):.4g}, BH={float(row.best_true_bh_fdr):.4g}",
        f"- Best high-expression negative: `{row.best_high_expression_negative_lr}` p={float(row.best_high_expression_negative_p):.4g}",
        f"- Best broad/background: `{row.best_broad_background_lr}` p={float(row.best_broad_background_p):.4g}",
        f"- Call: `{row.leakage_call}`",
        "",
        "## Claim Boundary",
        "",
        "High-expression negatives are suppressed relative to the best true LR, but broad/background LR still share the minimum p-value. This supports candidate development, not formal whole-atlas FDR closure.",
    ]
    (OUT_DIR / "phase12_integrated_coordinate_leakage_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(fdr_path: Path) -> None:
    manifest = {
        "task": "phase12_integrated_coordinate_leakage_audit",
        "input": rel(fdr_path),
        "outputs": {
            "lr": rel(OUT_DIR / "phase12_integrated_coordinate_leakage_lr.csv"),
            "summary": rel(OUT_DIR / "phase12_integrated_coordinate_leakage_summary.csv"),
            "low_p_lr": rel(OUT_DIR / "phase12_integrated_coordinate_leakage_low_p_lr.csv"),
            "status": rel(OUT_DIR / "phase12_integrated_coordinate_leakage_status.md"),
        },
        "claim_boundary": "Leakage audit for integrated coordinate-rerun smoke; not a discovery FDR procedure.",
    }
    (OUT_DIR / "phase12_integrated_coordinate_leakage_manifest.json").write_text(
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
