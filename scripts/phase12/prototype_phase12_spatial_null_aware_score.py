#!/usr/bin/env python
"""Prototype a spatial-null-aware discovery score for the next model iteration.

This is a score-level prototype, not a replacement model. It uses the existing
expanded-atlas permutation audit and asks whether a score that explicitly
penalizes pooled spatial-null risk and broad/background LR can close broad
whole-atlas FDR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "spatial_null_aware_score_prototype"

FDR_PATH = PHASE12_DIR / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv"
RANK_PATH = PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv"
ELIGIBLE_PATH = PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_all_lr.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_table = build_lr_table()
    dataset = summarize_dataset(lr_table)
    story = summarize_story(dataset)

    lr_table.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_null_aware_score_story.csv", index=False)
    write_status(dataset, story)
    manifest = {
        "task": "phase12_spatial_null_aware_score_prototype",
        "inputs": [rel(FDR_PATH), rel(RANK_PATH), rel(ELIGIBLE_PATH)],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_null_aware_score_lr.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_dataset.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_story.csv"),
            rel(OUT_DIR / "phase12_spatial_null_aware_score_status.md"),
        ],
        "claim_boundary": (
            "Exploratory score-level next-model prototype. It reuses existing null p-values "
            "and is not an end-to-end retrained model or final discovery-FDR procedure."
        ),
    }
    (OUT_DIR / "phase12_spatial_null_aware_score_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(dataset.to_string(index=False))
    print(story.to_string(index=False))


def build_lr_table() -> pd.DataFrame:
    fdr = pd.read_csv(FDR_PATH)
    rank_cols = [
        "dataset_key",
        "lr_id",
        "raw_rank",
        "raw_rank_q95",
        "raw_mean_score",
        "raw_q95_score",
        "n_nonzero_expr",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "source_support",
        "category_weight",
        "hard_negative_weight",
        "zero_expression_weight",
        "low_support_weight",
        "calibration_note",
    ]
    ranking = pd.read_csv(RANK_PATH, usecols=rank_cols)
    flags = pd.read_csv(
        ELIGIBLE_PATH,
        usecols=[
            "dataset_key",
            "lr_id",
            "primary_control_decoy",
            "receptor_shared_decoy",
            "ligand_shared_decoy",
            "decoy_class",
            "discovery_eligible",
        ],
    )
    table = fdr.merge(ranking, on=["dataset_key", "lr_id"], how="left").merge(
        flags,
        on=["dataset_key", "lr_id"],
        how="left",
    )
    for col in ["primary_control_decoy", "receptor_shared_decoy", "ligand_shared_decoy", "discovery_eligible"]:
        table[col] = table[col].fillna(False).astype(bool)
    table["decoy_class"] = table["decoy_class"].fillna("unlabeled_background")
    table = add_scores(table)
    return table.sort_values(["dataset_key", "spatial_null_aware_rank", "lr_pair"]).reset_index(drop=True)


def add_scores(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in [
        "calibrated_score",
        "strict_discovery_score",
        "raw_mean_score",
        "raw_q95_score",
        "paired_expr_fraction",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "source_support",
        "spatial_distance_permutation_global_empirical_p",
        "receiver_permutation_global_empirical_p",
        "lr_identity_permutation_global_empirical_p",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    expr_support = np.sqrt(
        out["ligand_positive_fraction"].clip(0, 1)
        * out["receptor_positive_fraction"].clip(0, 1)
    )
    pair_support = (0.55 * out["paired_expr_fraction"].clip(0, 1) + 0.45 * expr_support).clip(0, 1)
    receiver_guard = (-np.log10(out["receiver_permutation_global_empirical_p"].clip(lower=1e-6))).clip(0, 6) / 6
    identity_guard = (-np.log10(out["lr_identity_permutation_global_empirical_p"].clip(lower=1e-6))).clip(0, 6) / 6
    spatial_guard = (-np.log10(out["spatial_distance_permutation_global_empirical_p"].clip(lower=1e-6))).clip(0, 6) / 6
    broad_other = (
        out["selected_reason"].astype(str).eq("category_balanced")
        & out["biological_category"].astype(str).eq("other")
    )
    category_balanced = out["selected_reason"].astype(str).eq("category_balanced")
    source = out["source_support"].clip(0, 1)
    hard_negative = out["false_positive_high_expression"].astype(bool)

    broad_penalty = np.where(broad_other, 0.04, np.where(category_balanced, 0.30, 1.0))
    decoy_penalty = np.where(
        out["primary_control_decoy"] | out["receptor_shared_decoy"] | out["ligand_shared_decoy"],
        0.18,
        1.0,
    )
    hard_penalty = np.where(hard_negative, 0.03, 1.0)
    source_weight = (0.12 + 0.88 * source).pow(1.6)
    expression_weight = (0.08 + 0.92 * pair_support).pow(1.25)

    signal = (
        0.45 * pct_by_dataset(out, "strict_discovery_score")
        + 0.25 * pct_by_dataset(out, "calibrated_score")
        + 0.12 * pct_by_dataset(out, "raw_mean_score")
        + 0.08 * pct_by_dataset(out, "raw_q95_score")
        + 0.10 * spatial_guard
    )
    null_guard = (0.35 * receiver_guard + 0.35 * identity_guard + 0.30 * spatial_guard).clip(0, 1).pow(1.4)
    out["spatial_null_aware_score"] = (
        signal
        * null_guard
        * source_weight
        * expression_weight
        * broad_penalty
        * decoy_penalty
        * hard_penalty
    )
    out["spatial_null_aware_p"] = score_to_empirical_p(out, "spatial_null_aware_score")
    out["spatial_null_aware_bh_q"] = np.nan
    out["spatial_null_aware_rank"] = np.nan
    for _, idx in out.groupby("dataset_key").groups.items():
        out.loc[idx, "spatial_null_aware_bh_q"] = bh(out.loc[idx, "spatial_null_aware_p"].to_numpy())
        out.loc[idx, "spatial_null_aware_rank"] = (
            out.loc[idx, "spatial_null_aware_score"].rank(ascending=False, method="min").astype(int)
        )
    out["spatial_null_aware_supported_q_0_1"] = out["spatial_null_aware_bh_q"] <= 0.1
    out["spatial_null_aware_note"] = np.select(
        [
            hard_negative,
            out["primary_control_decoy"] | out["receptor_shared_decoy"] | out["ligand_shared_decoy"],
            broad_other,
            category_balanced,
            out["spatial_null_aware_supported_q_0_1"],
        ],
        [
            "hard_negative_penalized",
            "decoy_penalized",
            "broad_other_penalized",
            "category_balanced_penalized",
            "score_supported",
        ],
        default="not_supported",
    )
    return out


def score_to_empirical_p(table: pd.DataFrame, score_col: str) -> pd.Series:
    out = pd.Series(index=table.index, dtype=float)
    for _, idx in table.groupby("dataset_key").groups.items():
        scores = table.loc[idx, score_col].to_numpy(dtype=float)
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # Higher score is better; empirical rank p over the full LR universe.
        out.loc[idx] = (len(scores) - ranks + 1.0) / len(scores)
    return out


def pct_by_dataset(table: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(index=table.index, dtype=float)
    for _, idx in table.groupby("dataset_key").groups.items():
        values = pd.to_numeric(table.loc[idx, col], errors="coerce").fillna(0.0)
        ranks = values.rank(method="average", pct=True)
        out.loc[idx] = ranks
    return out.fillna(0.0)


def summarize_dataset(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, group in table.groupby("dataset_key"):
        ranked = group.sort_values("spatial_null_aware_rank")
        top10 = ranked.head(10)
        supported = group[group["spatial_null_aware_supported_q_0_1"]]
        true_supported = supported[supported["true_lr"].astype(bool)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": len(group),
                "top10_true_lr": int(top10["true_lr"].astype(bool).sum()),
                "top10_primary_control_decoy": int(top10["primary_control_decoy"].sum()),
                "top10_receptor_shared_decoy": int(top10["receptor_shared_decoy"].sum()),
                "top10_ligand_shared_decoy": int(top10["ligand_shared_decoy"].sum()),
                "top10_broad_other_lr": int(
                    (
                        top10["selected_reason"].astype(str).eq("category_balanced")
                        & top10["biological_category"].astype(str).eq("other")
                    ).sum()
                ),
                "n_supported_q_0_1": len(supported),
                "n_true_supported_q_0_1": int(true_supported.shape[0]),
                "n_primary_control_supported_q_0_1": int(supported["primary_control_decoy"].sum()),
                "n_receptor_shared_supported_q_0_1": int(supported["receptor_shared_decoy"].sum()),
                "n_background_supported_q_0_1": int(
                    ((~supported["true_lr"].astype(bool)) & (~supported["primary_control_decoy"])).sum()
                ),
                "best_true_lr": best_true_lr(ranked),
                "best_true_q": best_true_q(ranked),
                "dataset_call": dataset_call(supported),
                "top_lr_pairs": ";".join(ranked["lr_pair"].astype(str).head(12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(summary: pd.DataFrame) -> pd.DataFrame:
    total_true = int(summary["n_true_supported_q_0_1"].sum()) if not summary.empty else 0
    total_control = int(summary["n_primary_control_supported_q_0_1"].sum()) if not summary.empty else 0
    total_receptor = int(summary["n_receptor_shared_supported_q_0_1"].sum()) if not summary.empty else 0
    supported_datasets = int((summary["n_true_supported_q_0_1"] > 0).sum()) if not summary.empty else 0
    if total_true > 0 and total_control == 0 and total_receptor == 0:
        call = "prototype_improves_selected_support_but_not_valid_fdr"
    else:
        call = "prototype_not_adopted"
    return pd.DataFrame(
        [
            {
                "topic": "spatial_null_aware_score_prototype",
                "overall_call": call,
                "n_datasets": len(summary),
                "n_supported_datasets": supported_datasets,
                "total_true_supported_q_0_1": total_true,
                "total_primary_control_supported_q_0_1": total_control,
                "total_receptor_shared_supported_q_0_1": total_receptor,
                "claim_boundary": (
                    "Score-level next-model prototype based on observed LR ranking and existing null p-values. "
                    "It is useful for design direction but cannot be used as final FDR because the score was not "
                    "recomputed under null permutations."
                ),
            }
        ]
    )


def dataset_call(supported: pd.DataFrame) -> str:
    if supported.empty:
        return "no_score_level_support"
    if supported["primary_control_decoy"].sum() == 0 and supported["receptor_shared_decoy"].sum() == 0:
        if supported["true_lr"].astype(bool).sum() > 0:
            return "score_level_true_support_without_control_contamination"
    return "score_level_support_contaminated_or_background"


def best_true_lr(ranked: pd.DataFrame) -> str:
    true_rows = ranked[ranked["true_lr"].astype(bool)]
    if true_rows.empty:
        return "none"
    return str(true_rows.iloc[0]["lr_pair"])


def best_true_q(ranked: pd.DataFrame) -> float:
    true_rows = ranked[ranked["true_lr"].astype(bool)]
    if true_rows.empty:
        return np.nan
    return float(true_rows.iloc[0]["spatial_null_aware_bh_q"])


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = p[order[i]] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q


def write_status(summary: pd.DataFrame, story: pd.DataFrame) -> None:
    overall = story.iloc[0]
    lines = [
        "# Phase 12 Spatial-Null-Aware Score Prototype",
        "",
        "## Overall",
        "",
        f"- Overall call: `{overall.overall_call}`.",
        f"- Supported datasets: {int(overall.n_supported_datasets)}/{int(overall.n_datasets)}.",
        f"- True LR supported at prototype q<=0.1: {int(overall.total_true_supported_q_0_1)}.",
        f"- Primary-control supported: {int(overall.total_primary_control_supported_q_0_1)}.",
        f"- Receptor-shared supported: {int(overall.total_receptor_shared_supported_q_0_1)}.",
        "",
        "## Dataset Calls",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_call}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 controls={int(row.top10_primary_control_decoy)}, "
            f"top10 receptor-shared={int(row.top10_receptor_shared_decoy)}, "
            f"supported true={int(row.n_true_supported_q_0_1)}, "
            f"supported controls={int(row.n_primary_control_supported_q_0_1)}, "
            f"best true={row.best_true_lr} q={fmt_num(row.best_true_q)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: spatial-null-aware scoring is a next-model design prototype and suggests which signals can survive stronger guards.",
        "- Unsafe wording: this prototype controls whole-atlas FDR. The score must be recomputed under null permutations or trained end-to-end before use as discovery FDR.",
    ]
    (OUT_DIR / "phase12_spatial_null_aware_score_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def fmt_num(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
