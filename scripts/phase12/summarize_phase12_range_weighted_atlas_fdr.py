#!/usr/bin/env python
"""Range-matched weighted FDR diagnostic for Phase 12 expanded atlases.

The conservative pooled whole-atlas audit is intentionally broad and remains
the primary guardrail. This diagnostic asks whether a label-independent
weighted BH procedure, applied to the range-matched all-null p-values, can
recover spatially plausible anchors while keeping hard controls low.

Weights use only observed covariates that are available for every LR:
expression support, database/source support, broad category support and explicit
hard-negative penalties. They do not use true_lr labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
MATCHED_DIR = PHASE12_DIR / "matched_spatial_atlas_fdr"
OUT_DIR = PHASE12_DIR / "range_weighted_atlas_fdr"
TOP_K_VALUES = (5, 10, 20, 50, 100)
Q_THRESHOLD = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    matched = pd.read_csv(MATCHED_DIR / "phase12_matched_spatial_atlas_fdr_lr.csv")
    table = build_table(ranking, matched)
    topk = summarize_topk(table)
    dataset = summarize_dataset(table, topk)
    story = summarize_story(dataset)

    table.to_csv(OUT_DIR / "phase12_range_weighted_atlas_fdr_lr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_range_weighted_atlas_fdr_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_range_weighted_atlas_fdr_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_range_weighted_atlas_fdr_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(table.groupby("dataset_key").head(20).to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_table(ranking: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    keep_rank = [
        "dataset_key",
        "lr_id",
        "raw_rank",
        "raw_rank_q95",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
        "expr_gate",
        "source_support",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "calibration_note",
    ]
    keep_matched = [
        "dataset_key",
        "rank",
        "strict_rank",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "calibrated_score",
        "strict_discovery_score",
        "matched_null_scope",
        "matched_spatial_empirical_p",
        "matched_all_null_empirical_p",
        "strict_matched_all_null_empirical_p",
        "receiver_global_empirical_p",
        "lr_identity_global_empirical_p",
        "pooled_spatial_global_empirical_p",
        "original_conservative_bh_fdr",
        "strict_original_conservative_bh_fdr",
    ]
    out = matched[keep_matched].merge(ranking[keep_rank], on=["dataset_key", "lr_id"], how="left")
    out = annotate_controls(out)
    out["range_weight_prior"] = compute_prior(out)
    rows = []
    for _, sub in out.groupby("dataset_key", sort=False):
        sub = sub.copy()
        weights = make_weights(sub["range_weight_prior"].to_numpy(dtype=float))
        sub["range_weight"] = weights
        for p_col, q_col in [
            ("matched_spatial_empirical_p", "weighted_matched_spatial_bh_fdr"),
            ("matched_all_null_empirical_p", "weighted_matched_all_null_bh_fdr"),
            ("strict_matched_all_null_empirical_p", "weighted_strict_matched_all_null_bh_fdr"),
        ]:
            p = pd.to_numeric(sub[p_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            sub[q_col] = weighted_bh(p, weights)
        q_for_rank = pd.to_numeric(sub["weighted_matched_all_null_bh_fdr"], errors="coerce").fillna(1.0)
        sub["range_weighted_score"] = (
            pd.to_numeric(sub["calibrated_score"], errors="coerce").fillna(0.0)
            * sub["range_weight"]
            * (-np.log10(q_for_rank.clip(lower=1e-12)))
        )
        sub["range_weighted_fdr_call"] = [classify_lr(row) for row in sub.itertuples(index=False)]
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["range_weighted_rank"] = out.groupby("dataset_key")["range_weighted_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    return out.sort_values(["dataset_key", "range_weighted_rank", "rank", "lr_pair"]).reset_index(drop=True)


def annotate_controls(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    out["receptor_shared_decoy"] = False
    out["ligand_shared_decoy"] = False
    for dataset_key, sub in out.groupby("dataset_key", sort=False):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_decoy"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_decoy"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["primary_control_decoy"] = (
        out["false_positive_high_expression"] | out["receptor_shared_decoy"] | out["ligand_shared_decoy"]
    )
    out["decoy_class"] = np.select(
        [
            out["true_lr"],
            out["false_positive_high_expression"],
            out["receptor_shared_decoy"],
            out["ligand_shared_decoy"],
        ],
        [
            "prespecified_positive_anchor",
            "high_expression_control",
            "receptor_shared_control",
            "ligand_shared_control",
        ],
        default="unlabeled_background_lr",
    )
    return out


def compute_prior(table: pd.DataFrame) -> pd.Series:
    expr = pd.to_numeric(table["expr_gate"], errors="coerce").fillna(0.0).clip(0, 1)
    source = pd.to_numeric(table["source_support"], errors="coerce").fillna(0.0).clip(0, 1)
    annotated = table["biological_category"].astype(str).ne("other").astype(float)
    support = 0.42 * source + 0.38 * expr + 0.20 * annotated
    penalties = (
        pd.to_numeric(table["hard_negative_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(table["zero_expression_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(table["low_support_weight"], errors="coerce").fillna(1.0).clip(0, 1)
        * pd.to_numeric(table["category_weight"], errors="coerce").fillna(1.0).clip(0, 1)
    )
    prior = (support * penalties).clip(lower=0)
    return prior


def make_weights(prior: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    prior[~np.isfinite(prior)] = 0.0
    lo = float(np.nanmin(prior)) if len(prior) else 0.0
    hi = float(np.nanmax(prior)) if len(prior) else 0.0
    scaled = (prior - lo) / (hi - lo + 1e-12)
    weights = 0.20 + 3.80 * scaled
    mean = float(np.nanmean(weights)) if len(weights) else 1.0
    if not np.isfinite(mean) or mean <= 0:
        mean = 1.0
    return np.clip(weights / mean, 0.05, 20.0)


def weighted_bh(p_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p[~np.isfinite(p)] = 1.0
    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
    weights = weights / np.nanmean(weights)
    weighted_p = np.minimum(p / weights, 1.0)
    return bh_adjust(weighted_p)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p[~np.isfinite(p)] = 1.0
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / float(rank + 1)
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    return q


def classify_lr(row: object) -> str:
    if bool(getattr(row, "primary_control_decoy")) and getattr(row, "weighted_matched_all_null_bh_fdr") <= Q_THRESHOLD:
        return "control_leakage_weighted_fdr"
    if bool(getattr(row, "true_lr")) and getattr(row, "weighted_strict_matched_all_null_bh_fdr") <= Q_THRESHOLD:
        return "strict_range_weighted_true_supported"
    if bool(getattr(row, "true_lr")) and getattr(row, "weighted_matched_all_null_bh_fdr") <= Q_THRESHOLD:
        return "range_weighted_true_supported"
    if getattr(row, "weighted_matched_all_null_bh_fdr") <= Q_THRESHOLD:
        return "range_weighted_nonanchor_supported"
    return "not_range_weighted_supported"


def summarize_topk(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values("range_weighted_rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_primary_control_decoy": int(top["primary_control_decoy"].sum()),
                    "n_high_expression_control": int(top["false_positive_high_expression"].sum()),
                    "n_weighted_q_0_1": int(top["weighted_matched_all_null_bh_fdr"].le(Q_THRESHOLD).sum()),
                    "n_weighted_true_q_0_1": int(
                        (top["true_lr"] & top["weighted_matched_all_null_bh_fdr"].le(Q_THRESHOLD)).sum()
                    ),
                    "n_weighted_control_q_0_1": int(
                        (top["primary_control_decoy"] & top["weighted_matched_all_null_bh_fdr"].le(Q_THRESHOLD)).sum()
                    ),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(table: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        top10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))]
        t10 = top10.iloc[0].to_dict() if len(top10) else {}
        weighted_hits = sub[sub["weighted_matched_all_null_bh_fdr"].le(Q_THRESHOLD)]
        strict_hits = sub[sub["weighted_strict_matched_all_null_bh_fdr"].le(Q_THRESHOLD)]
        best_true = sub[sub["true_lr"]].sort_values(["weighted_matched_all_null_bh_fdr", "rank"]).head(1)
        best_strict_true = sub[sub["true_lr"]].sort_values(
            ["weighted_strict_matched_all_null_bh_fdr", "strict_rank", "rank"]
        ).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_primary_control_decoy": int(t10.get("n_primary_control_decoy", 0)),
                "top10_high_expression_control": int(t10.get("n_high_expression_control", 0)),
                "weighted_q_0_1": int(len(weighted_hits)),
                "weighted_true_q_0_1": int(weighted_hits["true_lr"].sum()) if len(weighted_hits) else 0,
                "weighted_control_q_0_1": int(weighted_hits["primary_control_decoy"].sum()) if len(weighted_hits) else 0,
                "weighted_background_q_0_1": int(weighted_hits["decoy_class"].eq("unlabeled_background_lr").sum())
                if len(weighted_hits)
                else 0,
                "strict_weighted_q_0_1": int(len(strict_hits)),
                "strict_weighted_true_q_0_1": int(strict_hits["true_lr"].sum()) if len(strict_hits) else 0,
                "strict_weighted_control_q_0_1": int(strict_hits["primary_control_decoy"].sum())
                if len(strict_hits)
                else 0,
                "best_true_lr": best_true.iloc[0]["lr_pair"] if len(best_true) else "",
                "best_true_weighted_q": float(best_true.iloc[0]["weighted_matched_all_null_bh_fdr"])
                if len(best_true)
                else float("nan"),
                "best_strict_true_lr": best_strict_true.iloc[0]["lr_pair"] if len(best_strict_true) else "",
                "best_strict_true_weighted_q": float(
                    best_strict_true.iloc[0]["weighted_strict_matched_all_null_bh_fdr"]
                )
                if len(best_strict_true)
                else float("nan"),
                "range_weighted_call": classify_dataset(weighted_hits, strict_hits),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_dataset(weighted_hits: pd.DataFrame, strict_hits: pd.DataFrame) -> str:
    true_hits = int(weighted_hits["true_lr"].sum()) if len(weighted_hits) else 0
    controls = int(weighted_hits["primary_control_decoy"].sum()) if len(weighted_hits) else 0
    strict_true = int(strict_hits["true_lr"].sum()) if len(strict_hits) else 0
    strict_controls = int(strict_hits["primary_control_decoy"].sum()) if len(strict_hits) else 0
    if strict_true > 0 and strict_controls == 0:
        return "strict_range_weighted_support_controls_clean"
    if true_hits > 0 and controls == 0:
        return "range_weighted_support_controls_clean"
    if true_hits > 0:
        return "range_weighted_support_with_control_caveat"
    return "range_weighted_not_supportive"


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    supportive = dataset["range_weighted_call"].astype(str).str.contains("support").sum() if len(dataset) else 0
    strict_clean = dataset["range_weighted_call"].astype(str).eq("strict_range_weighted_support_controls_clean").sum() if len(dataset) else 0
    true_hits = int(dataset["weighted_true_q_0_1"].sum()) if len(dataset) else 0
    control_hits = int(dataset["weighted_control_q_0_1"].sum()) if len(dataset) else 0
    overall = "range_weighted_diagnostic_supportive_with_caveat" if true_hits > 0 else "range_weighted_not_supportive"
    return pd.DataFrame(
        [
            {
                "topic": "range_weighted_atlas_fdr",
                "overall_range_weighted_call": overall,
                "n_datasets": int(len(dataset)),
                "n_supportive_datasets": int(supportive),
                "n_strict_clean_supportive_datasets": int(strict_clean),
                "n_weighted_true_q_0_1": true_hits,
                "n_weighted_control_q_0_1": control_hits,
                "claim_boundary": (
                    "Label-independent weighted BH over range-matched all-null p-values. "
                    "This is a diagnostic calibration experiment and does not replace the "
                    "original conservative pooled whole-atlas FDR guardrail."
                ),
            }
        ]
    )


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Range-Weighted Atlas FDR Diagnostic",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.range_weighted_call}`: weighted true q<=0.1="
            f"{int(row.weighted_true_q_0_1)}, weighted controls={int(row.weighted_control_q_0_1)}, "
            f"strict true={int(row.strict_weighted_true_q_0_1)}, strict controls="
            f"{int(row.strict_weighted_control_q_0_1)}, best={row.best_true_lr} q={fmt(row.best_true_weighted_q)}, "
            f"best strict={row.best_strict_true_lr} q={fmt(row.best_strict_true_weighted_q)}."
        )
    if not story.empty:
        row = story.iloc[0]
        lines += [
            "",
            "## Claim Boundary",
            "",
            f"- Overall: `{row.overall_range_weighted_call}`.",
            f"- {row.claim_boundary}",
        ]
    (OUT_DIR / "phase12_range_weighted_atlas_fdr_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_range_weighted_atlas_fdr_diagnostic",
        "inputs": [
            "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv",
        ],
        "outputs": [
            "phase12_range_weighted_atlas_fdr_lr.csv",
            "phase12_range_weighted_atlas_fdr_topk.csv",
            "phase12_range_weighted_atlas_fdr_dataset.csv",
            "phase12_range_weighted_atlas_fdr_story.csv",
            "phase12_range_weighted_atlas_fdr_status.md",
        ],
        "claim_boundary": "Diagnostic range-matched weighted BH; not a replacement for conservative pooled whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_range_weighted_atlas_fdr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(val):
        return "NA"
    return f"{val:.3g}"


if __name__ == "__main__":
    main()
