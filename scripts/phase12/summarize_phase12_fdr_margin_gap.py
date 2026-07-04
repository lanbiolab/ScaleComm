#!/usr/bin/env python
"""Quantify how far Phase 12 model variants are from broad all-null FDR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "fdr_margin_gap"

VARIANTS = {
    "main_calibrated": PHASE12_DIR
    / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
    "anchor_background_guarded_training": PHASE12_DIR
    / "anchor_background_guarded_training_permutation_audit/"
    / "phase12_anchor_background_guarded_training_permutation_lr_fdr.csv",
    "anchor_background_range_guarded_training": PHASE12_DIR
    / "anchor_background_range_guarded_training_permutation_audit/"
    / "phase12_anchor_background_range_guarded_training_permutation_lr_fdr.csv",
    "null_margin_guarded_training": PHASE12_DIR
    / "null_margin_guarded_training_permutation_audit/"
    / "phase12_null_margin_guarded_training_permutation_lr_fdr.csv",
}

NULLS = ("receiver_permutation", "spatial_distance_permutation", "lr_identity_permutation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = load_variants()
    dataset = summarize_dataset(lr)
    method = summarize_method(dataset)
    story = summarize_story(method, dataset)
    lr.to_csv(OUT_DIR / "phase12_fdr_margin_gap_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_fdr_margin_gap_dataset.csv", index=False)
    method.to_csv(OUT_DIR / "phase12_fdr_margin_gap_method.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_fdr_margin_gap_story.csv", index=False)
    write_status(method, dataset, story)
    write_manifest()
    print(method.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def load_variants() -> pd.DataFrame:
    rows = []
    for variant, path in VARIANTS.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table["variant"] = variant
        table = annotate_gap(table)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def annotate_gap(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in [
        "conservative_global_empirical_p",
        "conservative_global_bh_fdr",
        "strict_conservative_global_empirical_p",
        "strict_conservative_global_bh_fdr",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for null_type in NULLS:
        out[f"{null_type}_global_empirical_p"] = pd.to_numeric(
            out[f"{null_type}_global_empirical_p"], errors="coerce"
        )
        out[f"{null_type}_global_bh_fdr"] = pd.to_numeric(out[f"{null_type}_global_bh_fdr"], errors="coerce")
        out[f"strict_{null_type}_global_empirical_p"] = pd.to_numeric(
            out[f"strict_{null_type}_global_empirical_p"], errors="coerce"
        )
        out[f"strict_{null_type}_global_bh_fdr"] = pd.to_numeric(
            out[f"strict_{null_type}_global_bh_fdr"], errors="coerce"
        )
    out["true_lr"] = out["true_lr"].astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].astype(bool)
    out["q_gap_to_0_1"] = out["conservative_global_bh_fdr"] / 0.1
    out["strict_q_gap_to_0_1"] = out["strict_conservative_global_bh_fdr"] / 0.1
    out["p_gap_to_nominal_bh_0_1"] = np.nan
    out["strict_p_gap_to_nominal_bh_0_1"] = np.nan
    pieces = []
    for (_, dataset_key), sub in out.groupby(["variant", "dataset_key"], sort=False):
        sub = sub.copy()
        n_lr = len(sub)
        sub["p_gap_to_nominal_bh_0_1"] = [
            p / (0.1 * rank / n_lr)
            for p, rank in zip(sub["conservative_global_empirical_p"], sub["conservative_global_empirical_p"].rank(method="min"))
        ]
        sub["strict_p_gap_to_nominal_bh_0_1"] = [
            p / (0.1 * rank / n_lr)
            for p, rank in zip(
                sub["strict_conservative_global_empirical_p"],
                sub["strict_conservative_global_empirical_p"].rank(method="min"),
            )
        ]
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True) if pieces else out
    out["dominant_p_null"] = [dominant_null(row, strict=False) for row in out.itertuples(index=False)]
    out["strict_dominant_p_null"] = [dominant_null(row, strict=True) for row in out.itertuples(index=False)]
    return out


def summarize_dataset(lr: pd.DataFrame) -> pd.DataFrame:
    if lr.empty:
        return pd.DataFrame()
    rows = []
    for (variant, dataset_key), sub in lr.groupby(["variant", "dataset_key"], sort=False):
        true = sub[sub["true_lr"]].copy()
        top10 = sub.sort_values("rank").head(10)
        if true.empty:
            continue
        best_q = true.sort_values(["conservative_global_bh_fdr", "rank"]).iloc[0]
        best_strict_q = true.sort_values(["strict_conservative_global_bh_fdr", "strict_rank"]).iloc[0]
        best_p = true.sort_values(["p_gap_to_nominal_bh_0_1", "rank"]).iloc[0]
        rows.append(
            {
                "variant": variant,
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(len(true)),
                "top10_true_lr": int(top10["true_lr"].sum()),
                "top10_high_expression_negative": int(top10["false_positive_high_expression"].sum()),
                "best_true_by_q": best_q["lr_pair"],
                "best_true_q": float(best_q["conservative_global_bh_fdr"]),
                "best_true_q_gap_to_0_1": float(best_q["q_gap_to_0_1"]),
                "best_true_p_gap_to_nominal_bh_0_1": float(best_p["p_gap_to_nominal_bh_0_1"]),
                "best_true_dominant_p_null": best_p["dominant_p_null"],
                "best_true_receiver_p": float(best_p["receiver_permutation_global_empirical_p"]),
                "best_true_spatial_p": float(best_p["spatial_distance_permutation_global_empirical_p"]),
                "best_true_identity_p": float(best_p["lr_identity_permutation_global_empirical_p"]),
                "best_strict_true_by_q": best_strict_q["lr_pair"],
                "best_strict_true_q": float(best_strict_q["strict_conservative_global_bh_fdr"]),
                "best_strict_true_q_gap_to_0_1": float(best_strict_q["strict_q_gap_to_0_1"]),
                "best_strict_true_dominant_p_null": best_strict_q["strict_dominant_p_null"],
                "dataset_gap_call": dataset_gap_call(best_q, best_p, top10),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows = []
    for variant, sub in dataset.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "n_datasets": int(len(sub)),
                "mean_top10_true_lr": float(pd.to_numeric(sub["top10_true_lr"], errors="coerce").mean()),
                "min_best_true_q": float(pd.to_numeric(sub["best_true_q"], errors="coerce").min()),
                "median_best_true_q": float(pd.to_numeric(sub["best_true_q"], errors="coerce").median()),
                "min_q_gap_to_0_1": float(pd.to_numeric(sub["best_true_q_gap_to_0_1"], errors="coerce").min()),
                "median_q_gap_to_0_1": float(
                    pd.to_numeric(sub["best_true_q_gap_to_0_1"], errors="coerce").median()
                ),
                "min_p_gap_to_nominal_bh_0_1": float(
                    pd.to_numeric(sub["best_true_p_gap_to_nominal_bh_0_1"], errors="coerce").min()
                ),
                "dominant_null_counts": ";".join(
                    f"{k}:{v}" for k, v in sub["best_true_dominant_p_null"].astype(str).value_counts().items()
                ),
                "method_gap_call": method_gap_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["min_q_gap_to_0_1", "median_q_gap_to_0_1", "mean_top10_true_lr"],
        ascending=[True, True, False],
    )


def summarize_story(method: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if method.empty:
        return pd.DataFrame()
    best = method.iloc[0]
    return pd.DataFrame(
        [
            {
                "topic": "fdr_margin_gap",
                "overall_call": "broad_fdr_gap_quantified_not_closed",
                "n_variants": int(len(method)),
                "best_variant_by_min_q_gap": best["variant"],
                "best_variant_min_q_gap_to_0_1": float(best["min_q_gap_to_0_1"]),
                "best_variant_median_q_gap_to_0_1": float(best["median_q_gap_to_0_1"]),
                "best_variant_dominant_null_counts": best["dominant_null_counts"],
                "n_dataset_variant_rows": int(len(dataset)),
                "claim_boundary": (
                    "This quantifies distance to broad all-null FDR. It does not close FDR; "
                    "use it to set next-model optimization targets."
                ),
            }
        ]
    )


def dominant_null(row: object, *, strict: bool) -> str:
    prefix = "strict_" if strict else ""
    values = {
        "receiver": float(getattr(row, f"{prefix}receiver_permutation_global_empirical_p", np.nan)),
        "spatial": float(getattr(row, f"{prefix}spatial_distance_permutation_global_empirical_p", np.nan)),
        "identity": float(getattr(row, f"{prefix}lr_identity_permutation_global_empirical_p", np.nan)),
    }
    values = {key: value for key, value in values.items() if np.isfinite(value)}
    return max(values, key=values.get) if values else ""


def dataset_gap_call(best_q: pd.Series, best_p: pd.Series, top10: pd.DataFrame) -> str:
    q_gap = float(best_q["q_gap_to_0_1"])
    p_gap = float(best_p["p_gap_to_nominal_bh_0_1"])
    top10_true = int(top10["true_lr"].sum())
    top10_high_neg = int(top10["false_positive_high_expression"].sum())
    if q_gap <= 1:
        return "fdr_closed"
    if q_gap <= 3 and top10_high_neg == 0:
        return "near_fdr_with_clean_top10"
    if top10_true >= 5 and top10_high_neg == 0:
        return "ranking_strong_but_fdr_far"
    if p_gap <= 10:
        return "nominal_p_gap_moderate"
    return "fdr_gap_large"


def method_gap_call(sub: pd.DataFrame) -> str:
    min_gap = float(pd.to_numeric(sub["best_true_q_gap_to_0_1"], errors="coerce").min())
    top10 = float(pd.to_numeric(sub["top10_true_lr"], errors="coerce").mean())
    if min_gap <= 1:
        return "has_fdr_closed_dataset"
    if min_gap <= 3 and top10 >= 5:
        return "near_fdr_with_ranking_support"
    if top10 >= 5:
        return "ranking_support_fdr_gap_remaining"
    return "fdr_gap_remaining"


def write_status(method: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 FDR Margin Gap",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.variant}` `{row.method_gap_call}`: mean top10 true={row.mean_top10_true_lr:.2f}, "
            f"min q-gap={row.min_q_gap_to_0_1:.2f}x, median q-gap={row.median_q_gap_to_0_1:.2f}x, "
            f"dominant nulls={row.dominant_null_counts}."
        )
    lines += ["", "## Dataset Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.variant}` / `{row.dataset_key}` `{row.dataset_gap_call}`: "
            f"best={row.best_true_by_q} q={fmt(row.best_true_q)} ({fmt(row.best_true_q_gap_to_0_1)}x over 0.1), "
            f"dominant null={row.best_true_dominant_p_null}, top10 true={int(row.top10_true_lr)}."
        )
    if not story.empty:
        first = story.iloc[0]
        lines += [
            "",
            "## Decision",
            "",
            f"- Overall: `{first['overall_call']}`.",
            f"- Best variant by min q-gap: `{first['best_variant_by_min_q_gap']}`.",
            "- Manuscript boundary: diagnostic target setting only, not FDR closure.",
        ]
    (OUT_DIR / "phase12_fdr_margin_gap_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_fdr_margin_gap",
        "variants": list(VARIANTS),
        "outputs": [
            "phase12_fdr_margin_gap_lr.csv",
            "phase12_fdr_margin_gap_dataset.csv",
            "phase12_fdr_margin_gap_method.csv",
            "phase12_fdr_margin_gap_story.csv",
            "phase12_fdr_margin_gap_status.md",
        ],
        "claim_boundary": "Quantifies distance to broad FDR; not a discovery result.",
    }
    (OUT_DIR / "phase12_fdr_margin_gap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
