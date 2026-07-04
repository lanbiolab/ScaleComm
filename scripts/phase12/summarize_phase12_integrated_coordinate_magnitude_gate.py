#!/usr/bin/env python
"""Magnitude-gated diagnostic for integrated coordinate-rerun FDR.

The integrated coordinate smoke showed many broad/background LR with the same
minimum p-value as true LR but with much smaller absolute calibrated scores. This
script scans label-independent calibrated-score floors to test whether absolute
signal magnitude can separate strong true LR from weak broad/background p-value
ties.

This is a diagnostic calibration target, not a final discovery rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
SMOKE_ROOT = PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_magnitude_gate"
Q_THRESHOLD = 0.10
SCORE_THRESHOLDS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdr_path = best_fdr_path()
    if fdr_path is None:
        raise FileNotFoundError(f"No integrated coordinate FDR files found under {SMOKE_ROOT}")
    table = classify(pd.read_csv(fdr_path))
    scan = scan_thresholds(table)
    best = select_best_gate(scan)
    gated = gated_lr_table(table, float(best["score_threshold"])) if best else pd.DataFrame()

    scan.to_csv(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_scan.csv", index=False)
    pd.DataFrame([best]).to_csv(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_best.csv", index=False)
    gated.to_csv(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_lr.csv", index=False)
    write_status(best, gated)
    write_manifest(fdr_path)

    print(scan.to_string(index=False))
    print()
    print(pd.DataFrame([best]).to_string(index=False))


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


def classify(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
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
    out["coordinate_p"] = pd.to_numeric(out["coordinate_rerun_p_value"], errors="coerce")
    out["calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce").fillna(0.0)
    return out


def scan_thresholds(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in SCORE_THRESHOLDS:
        sub = table[table["calibrated_score"].ge(float(threshold))].copy()
        if sub.empty:
            continue
        sub["magnitude_gate_bh_fdr"] = bh_adjust(sub["coordinate_p"].to_numpy(dtype=float))
        hits = sub[sub["magnitude_gate_bh_fdr"].le(Q_THRESHOLD)]
        true = sub[sub["true_lr"]]
        rows.append(
            {
                "dataset_key": str(sub["dataset_key"].iloc[0]),
                "score_threshold": float(threshold),
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
                "n_broad_background": int(sub["broad_background_lr"].sum()),
                "n_q_0_1": int(len(hits)),
                "n_true_q_0_1": int(hits["true_lr"].sum()) if len(hits) else 0,
                "n_high_expression_negative_q_0_1": int(hits["false_positive_high_expression"].sum()) if len(hits) else 0,
                "n_broad_background_q_0_1": int(hits["broad_background_lr"].sum()) if len(hits) else 0,
                "best_true_q": float(true["magnitude_gate_bh_fdr"].min()) if len(true) else np.nan,
                "top_lr_pairs": ";".join(sub.sort_values(["magnitude_gate_bh_fdr", "rank"])["lr_pair"].head(8)),
                "gate_call": gate_call(sub, hits),
            }
        )
    return pd.DataFrame(rows)


def select_best_gate(scan: pd.DataFrame) -> dict[str, object]:
    if scan.empty:
        return {}
    candidates = scan[
        scan["n_true_q_0_1"].gt(0)
        & scan["n_high_expression_negative_q_0_1"].eq(0)
        & scan["n_broad_background_q_0_1"].eq(0)
    ].copy()
    if candidates.empty:
        row = scan.sort_values(["best_true_q", "score_threshold"], ascending=[True, True]).head(1).iloc[0]
    else:
        row = candidates.sort_values(
            ["n_true_q_0_1", "score_threshold"],
            ascending=[False, True],
        ).head(1).iloc[0]
    out = row.to_dict()
    out["selection_note"] = (
        "diagnostic_clean_true_gate_not_final_rule"
        if int(out.get("n_true_q_0_1", 0)) > 0
        and int(out.get("n_high_expression_negative_q_0_1", 0)) == 0
        and int(out.get("n_broad_background_q_0_1", 0)) == 0
        else "no_clean_gate_found"
    )
    return out


def gated_lr_table(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sub = table[table["calibrated_score"].ge(threshold)].copy()
    sub["magnitude_gate_bh_fdr"] = bh_adjust(sub["coordinate_p"].to_numpy(dtype=float))
    cols = [
        "dataset_key",
        "rank",
        "lr_pair",
        "true_lr",
        "false_positive_high_expression",
        "broad_background_lr",
        "biological_category",
        "selected_reason",
        "calibrated_score",
        "coordinate_p",
        "magnitude_gate_bh_fdr",
    ]
    return sub[cols].sort_values(["magnitude_gate_bh_fdr", "rank", "lr_pair"]).reset_index(drop=True)


def gate_call(sub: pd.DataFrame, hits: pd.DataFrame) -> str:
    if hits.empty:
        return "no_q_0_1_hits"
    true_hits = int(hits["true_lr"].sum())
    high_hits = int(hits["false_positive_high_expression"].sum())
    broad_hits = int(hits["broad_background_lr"].sum())
    if true_hits > 0 and high_hits == 0 and broad_hits == 0:
        return "clean_true_q_0_1_diagnostic_gate"
    if true_hits > 0:
        return "true_q_0_1_with_leakage"
    return "non_true_q_0_1_only"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / float(i + 1))
        adjusted[i] = running
    out[idx[order]] = np.clip(adjusted, 0.0, 1.0)
    return out


def write_status(best: dict[str, object], gated: pd.DataFrame) -> None:
    if not best:
        text = "# Phase 12 Integrated Coordinate Magnitude Gate\n\nNo scan rows available.\n"
    else:
        text = "\n".join(
            [
                "# Phase 12 Integrated Coordinate Magnitude Gate",
                "",
                "## Best Diagnostic Gate",
                "",
                f"- Score threshold: {float(best['score_threshold']):.4g}",
                f"- LR retained: {int(best['n_lr'])}",
                f"- True LR retained: {int(best['n_true_lr'])}",
                f"- Broad/background retained: {int(best['n_broad_background'])}",
                f"- q<=0.1 hits: total={int(best['n_q_0_1'])}, true={int(best['n_true_q_0_1'])}, highNeg={int(best['n_high_expression_negative_q_0_1'])}, broad/background={int(best['n_broad_background_q_0_1'])}",
                f"- Best true q: {float(best['best_true_q']):.4g}",
                f"- Call: `{best['gate_call']}`",
                f"- Selection note: `{best['selection_note']}`",
                "",
                "## Claim Boundary",
                "",
                "This is a diagnostic magnitude gate scan. It suggests absolute score magnitude can suppress weak broad/background p-value ties, but it is not a pre-registered final FDR rule.",
            ]
        )
        text += "\n"
    (OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_status.md").write_text(text, encoding="utf-8")


def write_manifest(fdr_path: Path) -> None:
    manifest = {
        "task": "phase12_integrated_coordinate_magnitude_gate",
        "input": rel(fdr_path),
        "score_thresholds": list(SCORE_THRESHOLDS),
        "outputs": {
            "scan": rel(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_scan.csv"),
            "best": rel(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_best.csv"),
            "lr": rel(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_lr.csv"),
            "status": rel(OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_status.md"),
        },
        "claim_boundary": "Diagnostic magnitude-gate scan; not final broad whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_integrated_coordinate_magnitude_gate_manifest.json").write_text(
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
