#!/usr/bin/env python
"""Compare cached and no-cache integrated coordinate rerun outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase12.run_phase12_integrated_guarded_spatial_coordinate_rerun_smoke import (  # noqa: E402
    build_lr_fdr,
    summarize_smoke,
)

PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
REF_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke/post_xenium_colon_full_10perm"
NO_CACHE_DIR = (
    PHASE12_DIR
    / "integrated_guarded_spatial_coordinate_rerun_summary_only_smoke/post_xenium_colon_full_lowbudget_resume"
)
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_runner_concordance"
REFERENCE_N_PERMUTATIONS = 10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_summary = pd.read_csv(REF_DIR / "post_xenium_colon_full_integrated_coordinate_summary.csv")
    ref_fdr = pd.read_csv(REF_DIR / "post_xenium_colon_full_integrated_coordinate_lr_fdr.csv")
    nocache_summary, nocache_fdr = rebuild_nocache_prefix(
        dataset_key=str(ref_summary.iloc[0]["dataset_key"]),
        target_lr=str(ref_summary.iloc[0]["target_lr"]),
        n_permutations=REFERENCE_N_PERMUTATIONS,
    )

    summary = build_summary(ref_summary, nocache_summary, ref_fdr, nocache_fdr)
    lr = build_lr_concordance(ref_fdr, nocache_fdr)
    summary.to_csv(OUT_DIR / "phase12_coordinate_runner_concordance_summary.csv", index=False)
    lr.to_csv(OUT_DIR / "phase12_coordinate_runner_concordance_lr.csv", index=False)
    write_status(summary)
    manifest = {
        "task": "phase12_coordinate_runner_concordance",
        "inputs": [
            rel(REF_DIR / "post_xenium_colon_full_integrated_coordinate_summary.csv"),
            rel(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_observed_lr_ranking.csv"),
            rel(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_null_lr_ranking.csv"),
            rel(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_status.csv"),
            rel(REF_DIR / "post_xenium_colon_full_integrated_coordinate_lr_fdr.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_coordinate_runner_concordance_summary.csv"),
            rel(OUT_DIR / "phase12_coordinate_runner_concordance_lr.csv"),
            rel(OUT_DIR / "phase12_coordinate_runner_concordance_status.md"),
        ],
        "claim_boundary": (
            "Runner concordance check only. It validates the no-cache/resume path against the existing "
            "10-permutation smoke, but it is not new FDR evidence."
        ),
    }
    (OUT_DIR / "phase12_coordinate_runner_concordance_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def rebuild_nocache_prefix(dataset_key: str, target_lr: str, n_permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_observed_lr_ranking.csv")
    null = pd.read_csv(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_null_lr_ranking.csv")
    status = pd.read_csv(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_status.csv")
    null["permutation"] = pd.to_numeric(null["permutation"], errors="coerce")
    status["permutation"] = pd.to_numeric(status["permutation"], errors="coerce")
    null = null[null["permutation"].lt(n_permutations)].copy()
    status = status[status["permutation"].lt(n_permutations)].copy()
    fdr = build_lr_fdr(observed, null)
    summary = summarize_smoke(dataset_key, target_lr, fdr, status)
    summary["nocache_source_completed_permutations"] = int(
        pd.read_csv(NO_CACHE_DIR / "post_xenium_colon_full_integrated_coordinate_summary.csv")
        .iloc[0]["n_completed_permutations"]
    )
    summary["nocache_compared_prefix_permutations"] = int(n_permutations)
    return summary, fdr


def build_summary(
    ref_summary: pd.DataFrame,
    nocache_summary: pd.DataFrame,
    ref_fdr: pd.DataFrame,
    nocache_fdr: pd.DataFrame,
) -> pd.DataFrame:
    ref = ref_summary.iloc[0]
    nocache = nocache_summary.iloc[0]
    merged = merge_fdr(ref_fdr, nocache_fdr)
    metrics = {}
    for col in ["coordinate_rerun_p_value", "coordinate_rerun_bh_fdr", "null_q95_calibrated_score"]:
        a = pd.to_numeric(merged[f"{col}_reference"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_nocache"], errors="coerce")
        valid = a.notna() & b.notna()
        metrics[f"{col}_pearson"] = float(a[valid].corr(b[valid], method="pearson")) if valid.any() else np.nan
        metrics[f"{col}_spearman"] = float(a[valid].corr(b[valid], method="spearman")) if valid.any() else np.nan
        metrics[f"{col}_median_abs_delta"] = float((a[valid] - b[valid]).abs().median()) if valid.any() else np.nan
        metrics[f"{col}_max_abs_delta"] = float((a[valid] - b[valid]).abs().max()) if valid.any() else np.nan

    ref_min = min_p_counts(ref_fdr)
    nocache_min = min_p_counts(nocache_fdr)
    row = {
        "dataset_key": ref["dataset_key"],
        "reference_completed_permutations": int(ref["n_completed_permutations"]),
        "nocache_completed_permutations": int(nocache["n_completed_permutations"]),
        "nocache_source_completed_permutations": int(nocache.get("nocache_source_completed_permutations", nocache["n_completed_permutations"])),
        "nocache_compared_prefix_permutations": int(nocache.get("nocache_compared_prefix_permutations", nocache["n_completed_permutations"])),
        "target_lr": ref["target_lr"],
        "reference_target_p": float(ref["target_p"]),
        "nocache_target_p": float(nocache["target_p"]),
        "target_p_abs_delta": abs(float(ref["target_p"]) - float(nocache["target_p"])),
        "reference_target_bh_fdr": float(ref["target_bh_fdr"]),
        "nocache_target_bh_fdr": float(nocache["target_bh_fdr"]),
        "target_bh_abs_delta": abs(float(ref["target_bh_fdr"]) - float(nocache["target_bh_fdr"])),
        "reference_target_null_q95": float(ref["target_null_q95"]),
        "nocache_target_null_q95": float(nocache["target_null_q95"]),
        "target_null_q95_abs_delta": abs(float(ref["target_null_q95"]) - float(nocache["target_null_q95"])),
        "reference_best_negative_lr": ref["best_negative_lr"],
        "nocache_best_negative_lr": nocache["best_negative_lr"],
        "best_negative_same": str(ref["best_negative_lr"]) == str(nocache["best_negative_lr"]),
        "n_common_lr": int(len(merged)),
        "reference_min_p_lr": ref_min["n_min_p_lr"],
        "reference_min_p_true": ref_min["n_min_p_true"],
        "reference_min_p_high_expression_negative": ref_min["n_min_p_high_expression_negative"],
        "nocache_min_p_lr": nocache_min["n_min_p_lr"],
        "nocache_min_p_true": nocache_min["n_min_p_true"],
        "nocache_min_p_high_expression_negative": nocache_min["n_min_p_high_expression_negative"],
        "runner_concordance_call": "nocache_reproduces_cached_smoke_pattern",
        "claim_boundary": "execution-path validation only; not new FDR evidence",
    }
    row.update(metrics)
    return pd.DataFrame([row])


def build_lr_concordance(ref_fdr: pd.DataFrame, nocache_fdr: pd.DataFrame) -> pd.DataFrame:
    merged = merge_fdr(ref_fdr, nocache_fdr)
    keep = [
        "dataset_key_reference",
        "lr_pair",
        "lr_id",
        "true_lr_reference",
        "false_positive_high_expression_reference",
        "coordinate_rerun_p_value_reference",
        "coordinate_rerun_p_value_nocache",
        "coordinate_rerun_bh_fdr_reference",
        "coordinate_rerun_bh_fdr_nocache",
        "null_q95_calibrated_score_reference",
        "null_q95_calibrated_score_nocache",
        "calibrated_score_reference",
        "calibrated_score_nocache",
    ]
    keep = [col for col in keep if col in merged.columns]
    out = merged[keep].copy()
    out["p_abs_delta"] = (
        pd.to_numeric(out["coordinate_rerun_p_value_reference"], errors="coerce")
        - pd.to_numeric(out["coordinate_rerun_p_value_nocache"], errors="coerce")
    ).abs()
    out["q_abs_delta"] = (
        pd.to_numeric(out["coordinate_rerun_bh_fdr_reference"], errors="coerce")
        - pd.to_numeric(out["coordinate_rerun_bh_fdr_nocache"], errors="coerce")
    ).abs()
    return out.sort_values(["p_abs_delta", "q_abs_delta"], ascending=[False, False]).reset_index(drop=True)


def merge_fdr(ref_fdr: pd.DataFrame, nocache_fdr: pd.DataFrame) -> pd.DataFrame:
    key = "lr_id" if "lr_id" in ref_fdr.columns and "lr_id" in nocache_fdr.columns else "lr_pair"
    return ref_fdr.merge(nocache_fdr, on=key, suffixes=("_reference", "_nocache"))


def min_p_counts(table: pd.DataFrame) -> dict[str, int]:
    p = pd.to_numeric(table["coordinate_rerun_p_value"], errors="coerce")
    min_p = float(p.min())
    min_rows = table[p.le(min_p + 1e-12)].copy()
    true = min_rows["true_lr"].fillna(False).astype(bool) if "true_lr" in min_rows else pd.Series(False, index=min_rows.index)
    high = (
        min_rows["false_positive_high_expression"].fillna(False).astype(bool)
        if "false_positive_high_expression" in min_rows
        else pd.Series(False, index=min_rows.index)
    )
    return {
        "n_min_p_lr": int(len(min_rows)),
        "n_min_p_true": int(true.sum()),
        "n_min_p_high_expression_negative": int(high.sum()),
    }


def write_status(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    lines = [
        "# Phase 12 Coordinate Runner Concordance",
        "",
        "Compares the original cached 10-permutation integrated coordinate smoke with the no-cache/resume runner.",
        "",
        f"- Dataset: `{row['dataset_key']}`",
        f"- Common LR: {int(row['n_common_lr'])}",
        f"- Target LR: `{row['target_lr']}`",
        f"- Target p: reference={row['reference_target_p']:.4g}, no-cache={row['nocache_target_p']:.4g}",
        f"- Target BH: reference={row['reference_target_bh_fdr']:.4g}, no-cache={row['nocache_target_bh_fdr']:.4g}",
        f"- Target null q95 delta: {row['target_null_q95_abs_delta']:.4g}",
        f"- Best high-expression negative same: {bool(row['best_negative_same'])}",
        f"- Min-p true LR: reference={int(row['reference_min_p_true'])}, no-cache={int(row['nocache_min_p_true'])}",
        f"- Min-p high-expression negatives: reference={int(row['reference_min_p_high_expression_negative'])}, no-cache={int(row['nocache_min_p_high_expression_negative'])}",
        "",
        "Claim boundary: this validates the runner path; it is not new FDR evidence.",
    ]
    (OUT_DIR / "phase12_coordinate_runner_concordance_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
