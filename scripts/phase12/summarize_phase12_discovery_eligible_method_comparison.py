#!/usr/bin/env python
"""Method comparison on the Phase 12 discovery-eligible LR universe.

This audit compares SpatialSelf with adapter-light COMMOT, CellChat, SpaTalk and
CellNEST on the same pre-defined discovery-eligible LR pairs used by
summarize_phase12_discovery_eligible_atlas_fdr.py. It is a ranking/contamination
comparison, not a fully tuned official end-to-end benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
ELIGIBLE_DIR = PHASE12_DIR / "discovery_eligible_atlas_fdr"
OFFICIAL_DIR = PHASE12_DIR / "atlas_model_expanded/official_adapter_baselines"
OUT_DIR = PHASE12_DIR / "discovery_eligible_method_comparison"
TOP_K_VALUES = (5, 10, 20)
METHOD_LABELS = {
    "spatialself_calibrated": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_csv(ELIGIBLE_DIR / "phase12_discovery_eligible_atlas_lr.csv")
    official = pd.read_csv(OFFICIAL_DIR / "phase12_expanded_official_adapter_lr_ranking.csv")

    ranking = build_method_ranking(eligible, official)
    topk = summarize_topk(ranking)
    dataset = summarize_dataset(ranking, topk)
    method = summarize_method(dataset)
    comparison = compare_to_spatialself(dataset)
    story = summarize_story(method, comparison)

    ranking.to_csv(OUT_DIR / "phase12_discovery_eligible_method_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_discovery_eligible_method_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_discovery_eligible_method_dataset.csv", index=False)
    method.to_csv(OUT_DIR / "phase12_discovery_eligible_method_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_discovery_eligible_method_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_discovery_eligible_method_story.csv", index=False)
    write_status(method, dataset, comparison, story)
    write_manifest()

    print(method.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_method_ranking(eligible: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "dataset_key",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "mechanism",
        "biological_category",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "primary_control_decoy",
        "decoy_class",
        "eligible_matched_all_null_bh_fdr",
        "eligible_strict_matched_all_null_bh_fdr",
        "rank",
        "calibrated_score",
    ]
    meta = eligible[meta_cols].copy()
    rows = []
    ss = meta.copy()
    ss["method"] = "spatialself_calibrated"
    ss["method_label"] = METHOD_LABELS["spatialself_calibrated"]
    ss["score"] = pd.to_numeric(ss["calibrated_score"], errors="coerce")
    ss["full_rank"] = pd.to_numeric(ss["rank"], errors="coerce")
    rows.append(ss)

    official_keep = official[
        [
            "dataset_key",
            "method",
            "method_label",
            "lr_id",
            "score",
            "rank",
            "negative_or_risk",
        ]
    ].copy()
    for method, sub in official_keep.groupby("method", sort=False):
        merged = meta.merge(sub, on=["dataset_key", "lr_id"], how="inner", suffixes=("", "_official"))
        merged["method"] = method
        merged["method_label"] = METHOD_LABELS.get(method, str(method))
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
        merged["full_rank"] = pd.to_numeric(merged["rank_official"], errors="coerce")
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranked = []
    for (dataset_key, method), sub in out.groupby(["dataset_key", "method"], sort=False):
        sub = sub.copy()
        sub["eligible_method_rank"] = (
            sub["score"].rank(ascending=False, method="min", na_option="bottom").astype(int)
        )
        sub["dataset_key"] = dataset_key
        sub["method"] = method
        ranked.append(sub)
    out = pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()
    keep = [
        "dataset_key",
        "method",
        "method_label",
        "eligible_method_rank",
        "full_rank",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "mechanism",
        "biological_category",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "primary_control_decoy",
        "decoy_class",
        "score",
        "eligible_matched_all_null_bh_fdr",
        "eligible_strict_matched_all_null_bh_fdr",
    ]
    return out[keep].sort_values(["dataset_key", "method", "eligible_method_rank", "lr_pair"]).reset_index(drop=True)


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        sub = sub.sort_values("eligible_method_rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method": method,
                    "method_label": str(top["method_label"].iloc[0]) if len(top) else METHOD_LABELS.get(method, method),
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_primary_control_decoy": int(top["primary_control_decoy"].astype(bool).sum()),
                    "n_high_expression_control": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_unlabeled_background": int(top["decoy_class"].astype(str).eq("unlabeled_background_lr").sum()),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(ranking: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        top10_df = topk[
            topk["dataset_key"].eq(dataset_key) & topk["method"].eq(method) & topk["top_k"].eq(10)
        ].head(1)
        t10 = top10_df.iloc[0].to_dict() if len(top10_df) else {}
        true = sub[sub["true_lr"].astype(bool)].copy()
        best_true = true.sort_values("eligible_method_rank").head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_eligible_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "n_primary_control_decoy": int(sub["primary_control_decoy"].astype(bool).sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_primary_control_decoy": int(t10.get("n_primary_control_decoy", 0)),
                "top10_high_expression_control": int(t10.get("n_high_expression_control", 0)),
                "top10_unlabeled_background": int(t10.get("n_unlabeled_background", 0)),
                "best_true_lr": str(best_true.iloc[0]["lr_pair"]) if len(best_true) else "",
                "best_true_eligible_rank": int(best_true.iloc[0]["eligible_method_rank"]) if len(best_true) else 0,
                "median_true_eligible_rank": float(true["eligible_method_rank"].median()) if len(true) else np.nan,
                "dataset_call": classify_dataset(pd.Series(t10) if t10 else pd.Series(dtype=object)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "method"])


def summarize_method(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in dataset.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_datasets": int(len(sub)),
                "mean_top10_true_lr": float(sub["top10_true_lr"].mean()),
                "mean_top10_primary_control_decoy": float(sub["top10_primary_control_decoy"].mean()),
                "mean_top10_high_expression_control": float(sub["top10_high_expression_control"].mean()),
                "mean_top10_unlabeled_background": float(sub["top10_unlabeled_background"].mean()),
                "median_true_eligible_rank": float(sub["median_true_eligible_rank"].median()),
                "n_clean_top10_datasets": int(
                    (sub["top10_primary_control_decoy"].eq(0) & sub["top10_true_lr"].ge(2)).sum()
                ),
                "method_call": classify_method(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_clean_top10_datasets", "mean_top10_true_lr", "mean_top10_primary_control_decoy"],
        ascending=[False, False, True],
    )


def compare_to_spatialself(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in dataset.groupby("dataset_key", sort=False):
        ss = sub[sub["method"].eq("spatialself_calibrated")]
        if ss.empty:
            continue
        ssr = ss.iloc[0]
        for row in sub[~sub["method"].eq("spatialself_calibrated")].itertuples(index=False):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "baseline_method": row.method,
                    "baseline_label": row.method_label,
                    "spatialself_top10_true_lr": int(ssr.top10_true_lr),
                    "baseline_top10_true_lr": int(row.top10_true_lr),
                    "spatialself_top10_controls": int(ssr.top10_primary_control_decoy),
                    "baseline_top10_controls": int(row.top10_primary_control_decoy),
                    "spatialself_median_true_rank": float(ssr.median_true_eligible_rank),
                    "baseline_median_true_rank": float(row.median_true_eligible_rank),
                    "comparison_call": comparison_call(ssr, row),
                }
            )
    return pd.DataFrame(rows)


def summarize_story(method: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    ss = method[method["method"].eq("spatialself_calibrated")]
    ss_clean = int(ss["n_clean_top10_datasets"].iloc[0]) if len(ss) else 0
    wins = int(comparison["comparison_call"].astype(str).str.startswith("spatialself").sum()) if len(comparison) else 0
    losses = int(comparison["comparison_call"].astype(str).str.startswith("baseline").sum()) if len(comparison) else 0
    if ss_clean >= 1 and wins > losses:
        call = "spatialself_eligible_universe_topk_advantage"
    elif wins > 0:
        call = "mixed_eligible_universe_advantage"
    else:
        call = "eligible_universe_baseline_advantage_not_supported"
    return pd.DataFrame(
        [
            {
                "n_methods": int(method["method"].nunique()) if len(method) else 0,
                "spatialself_clean_top10_datasets": ss_clean,
                "spatialself_pairwise_wins": wins,
                "baseline_pairwise_wins": losses,
                "overall_eligible_method_call": call,
                "claim_boundary": "Same discovery-eligible LR universe adapter-light comparison. It supports ranking/contamination differences, not fully tuned official end-to-end superiority.",
            }
        ]
    )


def classify_dataset(top10: pd.Series) -> str:
    if top10.empty:
        return "missing_top10"
    true_lr = int(top10.get("n_true_lr", 0))
    controls = int(top10.get("n_primary_control_decoy", 0))
    high = int(top10.get("n_high_expression_control", 0))
    if true_lr >= 3 and controls == 0:
        return "top10_true_enriched_controls_clean"
    if true_lr >= 3 and high == 0:
        return "top10_true_enriched_control_caveat"
    if controls > true_lr:
        return "top10_control_contaminated"
    return "top10_mixed_or_low_recall"


def classify_method(sub: pd.DataFrame) -> str:
    clean = int((sub["top10_primary_control_decoy"].eq(0) & sub["top10_true_lr"].ge(2)).sum())
    contaminated = bool((sub["top10_primary_control_decoy"] > sub["top10_true_lr"]).any())
    if clean >= max(1, len(sub) // 2) and not contaminated:
        return "eligible_topk_support_controls_clean"
    if contaminated:
        return "eligible_topk_contaminated_or_risk_prone"
    return "eligible_topk_mixed_or_low_recall"


def comparison_call(ss: pd.Series, baseline: object) -> str:
    ss_true = int(ss.top10_true_lr)
    ss_controls = int(ss.top10_primary_control_decoy)
    b_true = int(baseline.top10_true_lr)
    b_controls = int(baseline.top10_primary_control_decoy)
    if ss_true > b_true and ss_controls <= b_controls:
        return "spatialself_higher_recall_no_more_controls"
    if ss_controls < b_controls and ss_true >= b_true:
        return "spatialself_cleaner_no_lower_recall"
    if b_true > ss_true and b_controls <= ss_controls:
        return "baseline_higher_recall_no_more_controls"
    if b_controls < ss_controls and b_true >= ss_true:
        return "baseline_cleaner_no_lower_recall"
    return "mixed_tradeoff"


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def write_status(method: pd.DataFrame, dataset: pd.DataFrame, comparison: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Discovery-Eligible Method Comparison",
        "",
        "## Scope",
        "",
        "This compares SpatialSelf and adapter-light baselines on the same discovery-eligible LR universe. It is not a fully tuned official end-to-end benchmark.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.method_label}` `{row.method_call}`: mean top10 true={fmt(row.mean_top10_true_lr)}, "
            f"controls={fmt(row.mean_top10_primary_control_decoy)}, high-expression={fmt(row.mean_top10_high_expression_control)}, "
            f"clean datasets={int(row.n_clean_top10_datasets)}/{int(row.n_datasets)}."
        )
    lines += ["", "## Dataset Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method_label}` `{row.dataset_call}`: top10 true={int(row.top10_true_lr)}, "
            f"controls={int(row.top10_primary_control_decoy)}, high-expression={int(row.top10_high_expression_control)}, "
            f"best true={row.best_true_lr} rank={int(row.best_true_eligible_rank)}."
        )
    lines += ["", "## Pairwise SpatialSelf Comparisons", ""]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` vs `{row.baseline_label}`: {row.comparison_call}; "
            f"SS true/control={int(row.spatialself_top10_true_lr)}/{int(row.spatialself_top10_controls)}, "
            f"baseline true/control={int(row.baseline_top10_true_lr)}/{int(row.baseline_top10_controls)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_eligible_method_call}`: SS wins={int(s.spatialself_pairwise_wins)}, "
            f"baseline wins={int(s.baseline_pairwise_wins)}, clean SS datasets={int(s.spatialself_clean_top10_datasets)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this as an eligible-universe ranking/contamination comparison.",
        "- Do not describe adapter-light projected results as fully tuned official end-to-end superiority.",
    ]
    (OUT_DIR / "phase12_discovery_eligible_method_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_discovery_eligible_method_comparison",
        "inputs": [
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_lr.csv",
            "outputs/phase12/atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_lr_ranking.csv",
        ],
        "outputs": [
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_lr_ranking.csv",
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_topk.csv",
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_dataset.csv",
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_summary.csv",
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_comparison.csv",
            "outputs/phase12/discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv",
        ],
        "claim_boundary": "Same discovery-eligible LR universe adapter-light comparison; not fully tuned official end-to-end superiority.",
    }
    (OUT_DIR / "phase12_discovery_eligible_method_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
