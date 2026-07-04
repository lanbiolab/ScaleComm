#!/usr/bin/env python
"""Targeted matched-null audit for Phase 12 expanded atlas anchors.

The expanded whole-atlas permutation audit asks a discovery-level question over
886/957 LR pairs. This script asks a narrower validation question for the
pre-specified Phase 12 anchor LR panel:

1. Does each anchor survive permutation p-value adjustment within the anchor
   panel?
2. Does each anchor score above matched non-anchor controls with similar
   mechanism and expression support?
3. Do high-expression hard negatives stay below the anchor?

This is not a replacement for whole-atlas FDR. It is a targeted biological-story
validation layer with explicit claim boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PERM_DIR = EXPANDED_DIR / "permutation_audit"
OFFICIAL_DIR = EXPANDED_DIR / "official_adapter_baselines"
OUT_DIR = PHASE12_DIR / "targeted_matched_null"
N_MATCHED_CONTROLS = 80
TARGETED_Q = 0.10
MATCHED_Q = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    permutation = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    official = load_official_ranks()

    audit = build_audit(ranking, permutation, official)
    dataset = summarize_dataset(audit)
    method_comp = summarize_method_comparison(audit)

    audit.to_csv(OUT_DIR / "phase12_targeted_matched_null_lr_audit.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_targeted_matched_null_dataset_summary.csv", index=False)
    method_comp.to_csv(OUT_DIR / "phase12_targeted_matched_null_method_comparison.csv", index=False)
    write_status(dataset, audit, method_comp)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(audit.sort_values(["dataset_key", "spatialself_rank"]).to_string(index=False))


def load_official_ranks() -> pd.DataFrame:
    path = OFFICIAL_DIR / "phase12_expanded_official_adapter_lr_ranking.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    keep = table[
        [
            "dataset_key",
            "method",
            "lr_id",
            "rank",
            "score",
            "negative_or_risk",
        ]
    ].copy()
    wide = keep.pivot_table(
        index=["dataset_key", "lr_id"],
        columns="method",
        values=["rank", "score", "negative_or_risk"],
        aggfunc="first",
    )
    wide.columns = [f"official_{method}_{metric}" for metric, method in wide.columns]
    return wide.reset_index()


def build_audit(ranking: pd.DataFrame, permutation: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key", sort=False):
        sub = sub.copy().sort_values("rank").reset_index(drop=True)
        perm_sub = permutation[permutation["dataset_key"].eq(dataset_key)].copy()
        anchors = sub[sub["true_lr"].astype(bool)].copy().sort_values("rank")
        if anchors.empty:
            continue
        panel_p = anchors["lr_id"].map(
            perm_sub.set_index("lr_id")["conservative_global_empirical_p"].to_dict()
        ).to_numpy(dtype=float)
        panel_q = bh_adjust(panel_p)
        matched_p_values = []
        hard_p_values = []
        anchor_rows = []
        for idx, anchor in enumerate(anchors.itertuples(index=False)):
            matched = matched_control_pool(sub, anchor)
            matched_p = empirical_upper_p(float(anchor.calibrated_score), matched["calibrated_score"].to_numpy(dtype=float))
            hard = hard_negative_pool(sub, anchor)
            hard_p = empirical_upper_p(float(anchor.calibrated_score), hard["calibrated_score"].to_numpy(dtype=float))
            matched_p_values.append(matched_p)
            hard_p_values.append(hard_p)
            perm_row = perm_sub[perm_sub["lr_id"].eq(anchor.lr_id)].head(1)
            anchor_rows.append(
                {
                    "dataset_key": dataset_key,
                    "lr_id": anchor.lr_id,
                    "lr_pair": anchor.lr_pair,
                    "ligand": anchor.ligand,
                    "receptor": anchor.receptor,
                    "biological_category": anchor.biological_category,
                    "mechanism": anchor.mechanism,
                    "selected_reason": anchor.selected_reason,
                    "spatialself_rank": int(anchor.rank),
                    "spatialself_score": float(anchor.calibrated_score),
                    "whole_atlas_bh_fdr": float(perm_row["conservative_global_bh_fdr"].iloc[0]) if len(perm_row) else np.nan,
                    "targeted_panel_empirical_p": float(panel_p[idx]) if np.isfinite(panel_p[idx]) else np.nan,
                    "targeted_panel_bh_q": float(panel_q[idx]) if np.isfinite(panel_q[idx]) else np.nan,
                    "within_lr_empirical_p": float(perm_row["conservative_within_lr_empirical_p"].iloc[0]) if len(perm_row) else np.nan,
                    "receiver_global_p": float(perm_row["receiver_permutation_global_empirical_p"].iloc[0]) if len(perm_row) else np.nan,
                    "spatial_distance_global_p": float(perm_row["spatial_distance_permutation_global_empirical_p"].iloc[0]) if len(perm_row) else np.nan,
                    "lr_identity_global_p": float(perm_row["lr_identity_permutation_global_empirical_p"].iloc[0]) if len(perm_row) else np.nan,
                    "matched_control_empirical_p": float(matched_p),
                    "matched_control_n": int(len(matched)),
                    "matched_control_n_ge_anchor": int((matched["calibrated_score"].to_numpy(dtype=float) >= float(anchor.calibrated_score)).sum())
                    if len(matched)
                    else 0,
                    "matched_control_best_pair": best_pair(matched),
                    "matched_control_best_rank": best_rank(matched),
                    "matched_control_best_score": best_score(matched),
                    "matched_control_scope": matched.attrs.get("scope", "none"),
                    "hard_negative_empirical_p": float(hard_p),
                    "hard_negative_n": int(len(hard)),
                    "hard_negative_n_ge_anchor": int((hard["calibrated_score"].to_numpy(dtype=float) >= float(anchor.calibrated_score)).sum())
                    if len(hard)
                    else 0,
                    "hard_negative_best_pair": best_pair(hard),
                    "hard_negative_best_rank": best_rank(hard),
                    "hard_negative_best_score": best_score(hard),
                    "hard_negative_scope": hard.attrs.get("scope", "none"),
                }
            )

        matched_q = bh_adjust(np.asarray(matched_p_values, dtype=float))
        hard_q = bh_adjust(np.asarray(hard_p_values, dtype=float))
        for row, m_q, h_q in zip(anchor_rows, matched_q, hard_q):
            row["matched_control_bh_q"] = float(m_q) if np.isfinite(m_q) else np.nan
            row["hard_negative_bh_q"] = float(h_q) if np.isfinite(h_q) else np.nan
            row["targeted_validation_call"] = classify_anchor(row)
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and not official.empty:
        out = out.merge(official, on=["dataset_key", "lr_id"], how="left")
    return out


def matched_control_pool(dataset: pd.DataFrame, anchor: object) -> pd.DataFrame:
    controls = dataset[
        ~dataset["true_lr"].astype(bool) & ~dataset["false_positive_high_expression"].astype(bool)
    ].copy()
    if controls.empty:
        controls.attrs["scope"] = "empty"
        return controls

    pools = [
        ("same_mechanism_same_category", controls[
            controls["mechanism"].astype(str).eq(str(anchor.mechanism))
            & controls["biological_category"].astype(str).eq(str(anchor.biological_category))
        ]),
        ("same_mechanism", controls[controls["mechanism"].astype(str).eq(str(anchor.mechanism))]),
        ("same_category", controls[controls["biological_category"].astype(str).eq(str(anchor.biological_category))]),
        ("all_non_anchor_non_hard_negative", controls),
    ]
    selected_scope = "all_non_anchor_non_hard_negative"
    selected = controls
    for scope, pool in pools:
        if len(pool) >= min(20, N_MATCHED_CONTROLS // 2):
            selected_scope = scope
            selected = pool.copy()
            break

    features = [
        "n_nonzero_expr",
        "ligand_positive_fraction",
        "receptor_positive_fraction",
        "mean_coexpression_distance",
        "q95_coexpression_distance",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
    ]
    selected = selected.copy()
    for col in features:
        selected[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0)
    anchor_vec = np.asarray([float(getattr(anchor, col)) for col in features], dtype=float)
    selected_matrix = selected[features].to_numpy(dtype=float)
    scale = np.nanstd(np.vstack([selected_matrix, anchor_vec]), axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    distance = np.sqrt(np.sum(((selected_matrix - anchor_vec) / scale) ** 2, axis=1))
    selected["_match_distance"] = distance
    selected = selected.sort_values(["_match_distance", "rank"]).head(N_MATCHED_CONTROLS).copy()
    selected.attrs["scope"] = selected_scope
    return selected


def hard_negative_pool(dataset: pd.DataFrame, anchor: object) -> pd.DataFrame:
    high = dataset[dataset["false_positive_high_expression"].astype(bool)].copy()
    shared = dataset[
        ~dataset["true_lr"].astype(bool)
        & (
            dataset["ligand"].astype(str).eq(str(anchor.ligand))
            | dataset["receptor"].astype(str).eq(str(anchor.receptor))
        )
    ].copy()
    hard = pd.concat([high, shared], ignore_index=True).drop_duplicates("lr_id")
    hard = hard[~hard["lr_id"].astype(str).eq(str(anchor.lr_id))].copy()
    hard.attrs["scope"] = "high_expression_or_shared_ligand_receptor"
    return hard


def summarize_dataset(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if audit.empty:
        return pd.DataFrame()
    for dataset_key, sub in audit.groupby("dataset_key", sort=False):
        strong = sub["targeted_validation_call"].isin(["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"])
        strict = sub["targeted_validation_call"].eq("strong_targeted_matched_support")
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_anchor_lr": int(len(sub)),
                "n_rank_top10": int(sub["spatialself_rank"].le(10).sum()),
                "n_whole_atlas_fdr_0_1": int(sub["whole_atlas_bh_fdr"].le(TARGETED_Q).sum()),
                "n_targeted_panel_q_0_1": int(sub["targeted_panel_bh_q"].le(TARGETED_Q).sum()),
                "n_matched_control_q_0_1": int(sub["matched_control_bh_q"].le(MATCHED_Q).sum()),
                "n_hard_negative_q_0_1": int(sub["hard_negative_bh_q"].le(MATCHED_Q).sum()),
                "n_strong_targeted_matched_support": int(strict.sum()),
                "n_targeted_or_matched_support": int(strong.sum()),
                "best_supported_lr": best_supported_lr(sub),
                "best_supported_rank": best_supported_rank(sub),
                "dataset_call": classify_dataset(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_method_comparison(audit: pd.DataFrame) -> pd.DataFrame:
    methods = ["commot", "cellchat", "spatalk", "cellnest"]
    rows = []
    if audit.empty:
        return pd.DataFrame()
    for method in methods:
        rank_col = f"official_{method}_rank"
        if rank_col not in audit.columns:
            continue
        sub = audit[pd.notna(audit[rank_col])].copy()
        if sub.empty:
            continue
        supported = sub[sub["targeted_validation_call"].isin(["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"])]
        rows.append(
            {
                "method": method,
                "n_anchor_with_rank": int(len(sub)),
                "n_supported_anchor": int(len(supported)),
                "n_supported_spatialself_top10_baseline_after10": int(
                    ((supported["spatialself_rank"] <= 10) & (supported[rank_col] > 10)).sum()
                ),
                "median_rank_on_supported_anchor": float(supported[rank_col].median()) if len(supported) else np.nan,
                "best_rank_on_supported_anchor": float(supported[rank_col].min()) if len(supported) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def classify_anchor(row: dict[str, object]) -> str:
    rank = to_float(row.get("spatialself_rank"))
    panel_q = to_float(row.get("targeted_panel_bh_q"))
    matched_q = to_float(row.get("matched_control_bh_q"))
    hard_q = to_float(row.get("hard_negative_bh_q"))
    whole_q = to_float(row.get("whole_atlas_bh_fdr"))
    if rank <= 10 and whole_q <= TARGETED_Q:
        return "whole_atlas_fdr_supported"
    if rank <= 10 and panel_q <= TARGETED_Q and matched_q <= MATCHED_Q and hard_q <= MATCHED_Q:
        return "strong_targeted_matched_support"
    if rank <= 10 and panel_q <= TARGETED_Q and matched_q <= 0.25:
        return "targeted_permutation_with_matched_caveat"
    if rank <= 10 and matched_q <= MATCHED_Q and hard_q <= MATCHED_Q:
        return "matched_controls_supported_without_permutation_q"
    if rank <= 10:
        return "rank_support_only"
    return "not_targeted_supported"


def classify_dataset(sub: pd.DataFrame) -> str:
    strong = int(sub["targeted_validation_call"].eq("strong_targeted_matched_support").sum())
    target = int(sub["targeted_validation_call"].isin(["strong_targeted_matched_support", "targeted_permutation_with_matched_caveat"]).sum())
    whole = int(sub["whole_atlas_bh_fdr"].le(TARGETED_Q).sum())
    if whole:
        return "has_whole_atlas_fdr_anchor"
    if strong >= 2:
        return "targeted_matched_supportive"
    if target >= 2:
        return "targeted_supportive_with_matched_caveat"
    if target:
        return "one_targeted_anchor_supportive"
    return "rank_or_biology_only"


def best_supported_lr(sub: pd.DataFrame) -> str:
    order = {
        "whole_atlas_fdr_supported": 0,
        "strong_targeted_matched_support": 1,
        "targeted_permutation_with_matched_caveat": 2,
        "matched_controls_supported_without_permutation_q": 3,
        "rank_support_only": 4,
        "not_targeted_supported": 5,
    }
    ranked = sub.copy()
    ranked["_call_order"] = ranked["targeted_validation_call"].map(order).fillna(99)
    ranked = ranked.sort_values(["_call_order", "spatialself_rank"])
    return str(ranked.iloc[0]["lr_pair"]) if len(ranked) else ""


def best_supported_rank(sub: pd.DataFrame) -> int:
    lr = best_supported_lr(sub)
    if not lr:
        return -1
    row = sub[sub["lr_pair"].eq(lr)].head(1)
    return int(row["spatialself_rank"].iloc[0]) if len(row) else -1


def empirical_upper_p(observed: float, null_scores: np.ndarray) -> float:
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if len(null_scores) == 0:
        return float("nan")
    return float((1.0 + np.sum(null_scores >= observed)) / (1.0 + len(null_scores)))


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def best_pair(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    return str(table.sort_values(["rank", "calibrated_score"], ascending=[True, False]).iloc[0]["lr_pair"])


def best_rank(table: pd.DataFrame) -> int:
    if table.empty:
        return -1
    return int(table.sort_values(["rank", "calibrated_score"], ascending=[True, False]).iloc[0]["rank"])


def best_score(table: pd.DataFrame) -> float:
    if table.empty:
        return np.nan
    return float(table.sort_values(["rank", "calibrated_score"], ascending=[True, False]).iloc[0]["calibrated_score"])


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.3g}"


def write_status(dataset: pd.DataFrame, audit: pd.DataFrame, method_comp: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Targeted Matched-Null Audit",
        "",
        "## Scope",
        "",
        "This audit validates pre-specified Phase 12 anchor LR pairs against anchor-panel permutation p-values, matched non-anchor controls and hard-negative controls.",
        "It complements, but does not replace, the whole-atlas 886/957-LR BH FDR audit.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_call}`: anchors={int(row.n_anchor_lr)}, "
            f"top10={int(row.n_rank_top10)}, whole-atlas q<=0.1={int(row.n_whole_atlas_fdr_0_1)}, "
            f"targeted panel q<=0.1={int(row.n_targeted_panel_q_0_1)}, "
            f"matched q<=0.1={int(row.n_matched_control_q_0_1)}, "
            f"hard-neg q<=0.1={int(row.n_hard_negative_q_0_1)}, "
            f"best={row.best_supported_lr}#{int(row.best_supported_rank)}"
        )
    lines += ["", "## Anchor Detail", ""]
    for row in audit.sort_values(["dataset_key", "spatialself_rank"]).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.lr_pair}` `{row.targeted_validation_call}`: "
            f"rank={int(row.spatialself_rank)}, score={fmt(row.spatialself_score)}, "
            f"whole_q={fmt(row.whole_atlas_bh_fdr)}, panel_q={fmt(row.targeted_panel_bh_q)}, "
            f"matched_q={fmt(row.matched_control_bh_q)} ({int(row.matched_control_n)} controls, best={row.matched_control_best_pair}#{int(row.matched_control_best_rank)}), "
            f"hard_q={fmt(row.hard_negative_bh_q)} (best={row.hard_negative_best_pair}#{int(row.hard_negative_best_rank)})"
        )
    if not method_comp.empty:
        lines += ["", "## Adapter-Light Baseline Rank Comparison", ""]
        for row in method_comp.itertuples(index=False):
            lines.append(
                f"- `{row.method}`: supported anchors={int(row.n_supported_anchor)}, "
                f"SpatialSelf top10 but baseline rank>10={int(row.n_supported_spatialself_top10_baseline_after10)}, "
                f"median baseline rank={fmt(row.median_rank_on_supported_anchor)}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this for pre-specified biological-story validation, not unbiased whole-atlas discovery.",
        "- `targeted_panel_bh_q` adjusts permutation p-values within the anchor panel only.",
        "- `matched_control_bh_q` uses observed matched non-anchor controls; it is not a coordinate-level permutation p-value.",
        "- Whole-atlas BH FDR remains the stricter discovery-level criterion.",
    ]
    (OUT_DIR / "phase12_targeted_matched_null_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_targeted_matched_null_audit",
        "n_matched_controls": N_MATCHED_CONTROLS,
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "outputs/phase12/atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_lr_ranking.csv",
        ],
        "outputs": [
            "phase12_targeted_matched_null_lr_audit.csv",
            "phase12_targeted_matched_null_dataset_summary.csv",
            "phase12_targeted_matched_null_method_comparison.csv",
            "phase12_targeted_matched_null_status.md",
        ],
        "claim_boundary": "Targeted anchor-panel and matched-control audit; not whole-atlas discovery FDR.",
    }
    (OUT_DIR / "phase12_targeted_matched_null_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
